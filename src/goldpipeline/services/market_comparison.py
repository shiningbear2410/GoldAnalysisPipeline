"""Comparing two market-data sources, without believing either one.

Two feeds for the same instrument will not agree exactly, and that is not a
defect. Different venues have different spreads, different last ticks inside a
minute, different price precision, and - on higher timeframes - different
session anchors. So this module does not judge. It measures, and it is careful
about the one thing that would make the measurements lies: comparing bars that
are not the same bar.

**Provider-neutral by construction.** Both inputs are
:class:`~goldpipeline.schemas.market.MarketDataInput`, the model every provider
already produces. Nothing here imports a websocket, a terminal, or either
adapter, and no metric assumes which side is which: ``a`` and ``b`` are
symmetric, and a caller may pass them in any order.

**Why the un-normalized model.** :class:`MarketDataSnapshot` guarantees sorted,
unique, UTC bars - which would make three of the completeness metrics
unmeasurable, because a snapshot with a duplicate timestamp cannot be
constructed. Completeness is exactly what a comparison should be able to
report, so the input is the model that can still be wrong.

**Anchors, not assumptions.** Before any price is compared, each series' bars
are checked against the timeframe grid. If the two sources place their bar
openings at different offsets within the period - a 09:00 H4 against a 12:00
H4 - they are not describing the same intervals, and comparing them bar by bar
would produce a number that looks like a price difference and is really a
four-hour time shift. That case is reported as
:attr:`AlignmentKind.DIFFERENT_SESSION_ANCHOR` and the per-bar price metrics
are withheld rather than fabricated. Window-level coherence is still computed,
because it does not require pairing candles.

**Gaps are described, not condemned.** Gold does not trade through the
weekend, and a broker legitimately omits a minute in which nothing traded. A
gap is therefore classified by shape - long enough to be a session break, or
short enough to be a quiet interval - and never called corruption. There is no
holiday calendar here and there should not be one.

**No thresholds.** Nothing in this module decides whether a difference is
acceptable. It has no notion of a tolerable spread, because the pipeline has
no such notion, and inventing one here would turn an observation into a policy
nobody agreed to.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import Field

from goldpipeline.domain.errors import IncomparableSourcesError
from goldpipeline.schemas.common import StrictModel, Timeframe, UtcDatetime
from goldpipeline.schemas.market import MarketDataInput, OHLCBar

MARKET_COMPARISON_VERSION = "1"

SESSION_BREAK_MINIMUM = timedelta(hours=6)
"""A gap at least this long is read as a session break, not a missing bar.

Chosen to sit above any plausible quiet interval on a market that trades
roughly twenty-three hours a day, and below the weekend. It is a description,
not a verdict: a gap on either side of this line is reported either way, and
neither classification is a failure.
"""

MAX_REPORTED_TIMESTAMPS = 6
"""How many example timestamps a difference list carries.

A report is for a person. Two hundred timestamps proving the same point is not
more evidence, it is less readable.
"""


class AlignmentKind(StrEnum):
    """Whether the two series describe the same intervals."""

    ALIGNED = "ALIGNED"
    """Same grid offset. Bars may be paired by timestamp."""

    DIFFERENT_SESSION_ANCHOR = "DIFFERENT_SESSION_ANCHOR"
    """Same timeframe, different bar openings.

    Expected on higher timeframes, where a broker's session boundary decides
    where a period starts. Not an error, and not a reason to pair candles that
    cover different minutes.
    """

    NO_OVERLAP = "NO_OVERLAP"
    """The two series cover disjoint time windows. Nothing to compare per bar."""


class GapKind(StrEnum):
    """What a break in a series' timestamp grid looks like."""

    SESSION_BREAK = "SESSION_BREAK"
    """Long enough to be a weekend or daily close. Expected for this instrument."""

    QUIET_INTERVAL = "QUIET_INTERVAL"
    """Short. Often a period in which nothing traded, which providers may omit.

    Reported because an unexpected hole would look the same, and a person
    reading the report should get to decide - but never called corruption.
    """


class VolumeShape(StrEnum):
    """Whether a series carries volume at all. Never compared across providers.

    Tick counts from a broker and a venue's own activity are different
    quantities that happen to share a field name, so this module records
    presence and nothing else. Volume cannot inform an authority decision.
    """

    ABSENT = "ABSENT"
    PARTIAL = "PARTIAL"
    PRESENT = "PRESENT"


class SeriesGap(StrictModel):
    """One break in a series' grid."""

    after: UtcDatetime
    before: UtcDatetime
    missing_periods: int = Field(ge=1)
    kind: GapKind


class SeriesQuality(StrictModel):
    """What one source's series looks like on its own terms.

    Computed per source, so a problem is attributed to a provider rather than
    to "the comparison". Nothing in here needs the other side to exist.
    """

    provider: str
    provider_symbol: str | None
    bar_count: int = Field(ge=0)

    unique_timestamps: int = Field(ge=0)
    duplicate_timestamps: int = Field(ge=0)
    ascending: bool

    oldest: UtcDatetime | None
    newest: UtcDatetime | None

    newest_is_closed: bool
    """Proved by arithmetic on the bar's own open time, never a provider label."""

    newest_age_seconds: int | None
    expected_latest_closed_open: UtcDatetime | None
    newest_is_expected_latest: bool
    """Whether the newest bar is the most recent interval that could have closed.

    False is not automatically a fault: on a grid whose anchor is offset from
    the epoch, or right after a session break, the expected interval may not
    exist. It is a flag to read alongside the anchor finding.
    """

    grid_offset_seconds: int | None
    """Where this source opens its bars within the period, modulo the duration."""

    off_grid_bars: int = Field(ge=0)
    """Bars that do not sit on this source's own modal grid offset."""

    session_breaks: int = Field(ge=0)
    quiet_intervals: int = Field(ge=0)
    gaps: tuple[SeriesGap, ...] = ()

    volume_shape: VolumeShape
    price_decimals_max: int = Field(ge=0)


class Dispersion(StrictModel):
    """Absolute differences, described without being averaged away.

    Median, p95 and maximum rather than a mean: a mean over a series with one
    bad bar hides the bad bar, and the maximum is the number worth knowing.
    Values are :class:`~decimal.Decimal` at full precision - nothing here
    rounds a difference out of existence.
    """

    sample_count: int = Field(ge=0)
    median_abs: Decimal
    p95_abs: Decimal
    max_abs: Decimal
    max_at: UtcDatetime | None


class DirectionalAgreement(StrictModel):
    """How often the two sources moved the same way, bar to bar."""

    comparable_intervals: int = Field(ge=0)
    """Consecutive aligned pairs where neither source was flat."""

    agreed: int = Field(ge=0)
    flat_ties: int = Field(ge=0)
    """Intervals where at least one source did not move. Excluded, not counted
    as agreement - a flat close agrees with nothing."""

    @property
    def ratio(self) -> Decimal | None:
        """Agreement as a proportion of comparable intervals."""
        if not self.comparable_intervals:
            return None
        return Decimal(self.agreed) / Decimal(self.comparable_intervals)


class WindowCoherence(StrictModel):
    """Price agreement over a shared time window, without pairing candles.

    This is what remains measurable when the two sources use different bar
    boundaries: over the same stretch of clock time, both feeds saw the same
    market, so their extremes and their last traded level should be close even
    though no two candles line up.
    """

    window_start: UtcDatetime
    window_end: UtcDatetime
    bars_a: int = Field(ge=0)
    bars_b: int = Field(ge=0)
    high_difference: Decimal
    low_difference: Decimal
    last_close_difference: Decimal
    last_close_gap_seconds: int
    """How far apart the two 'last closes' are in time. A large value makes the
    price difference partly a time difference, so it is reported beside it."""


class MarketSourceComparison(StrictModel):
    """Everything measurable about two sources of one instrument."""

    schema_version: Literal["1"] = "1"

    canonical_symbol: str
    timeframe: Timeframe
    observed_at: UtcDatetime

    a: SeriesQuality
    b: SeriesQuality

    alignment: AlignmentKind
    anchor_offset_seconds: int | None
    """Signed difference between the two grid offsets, when both are known."""

    intersection_count: int = Field(ge=0)
    comparable_window_bars: int = Field(ge=0)
    """Smaller of the two sources' bar counts inside the shared window.

    The denominator for :attr:`intersection_ratio`. Using the full series
    length instead would report a low ratio simply because one source was
    asked for more bars than the other.
    """

    only_in_a: int = Field(ge=0)
    only_in_b: int = Field(ge=0)
    only_in_a_samples: tuple[UtcDatetime, ...] = ()
    only_in_b_samples: tuple[UtcDatetime, ...] = ()

    close: Dispersion | None = None
    open: Dispersion | None = None
    high: Dispersion | None = None
    low: Dispersion | None = None
    bar_range: Dispersion | None = None
    directional: DirectionalAgreement | None = None
    window: WindowCoherence | None = None

    notes: tuple[str, ...] = ()
    """Plain-language observations. Never a verdict."""

    @property
    def intersection_ratio(self) -> Decimal | None:
        if not self.comparable_window_bars:
            return None
        return Decimal(self.intersection_count) / Decimal(self.comparable_window_bars)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def compare_market_sources(
    source_a: MarketDataInput,
    source_b: MarketDataInput,
    *,
    observed_at: datetime,
    timeframe: Timeframe | None = None,
) -> MarketSourceComparison:
    """Measure two sources of the same instrument against each other.

    Args:
        source_a: One provider's payload.
        source_b: The other's. Symmetric - order carries no meaning.
        observed_at: The shared moment both series are judged fresh against.
        timeframe: Optional cross-check. When given it must match both
            payloads; when omitted the payloads must agree with each other.

    Raises:
        IncomparableSourcesError: The two payloads describe different instruments,
            or disagree about the timeframe. Refused rather than compared:
            silently measuring gold against silver would produce numbers that
            look like a data-quality finding and mean nothing at all.
    """
    canonical = _require_same_symbol(source_a, source_b)
    resolved = _require_same_timeframe(source_a, source_b, timeframe)
    moment = _as_utc(observed_at, "observed_at")

    quality_a = describe_series(source_a, timeframe=resolved, observed_at=moment)
    quality_b = describe_series(source_b, timeframe=resolved, observed_at=moment)

    bars_a = _by_timestamp(source_a.bars)
    bars_b = _by_timestamp(source_b.bars)
    notes: list[str] = []

    alignment, anchor_offset = _classify_alignment(quality_a, quality_b, bars_a, bars_b, notes)
    window = _window_coherence(bars_a, bars_b, notes)
    shared = _shared_window(bars_a, bars_b)

    intersection = sorted(set(bars_a) & set(bars_b) & shared)
    only_a = sorted((set(bars_a) & shared) - set(bars_b))
    only_b = sorted((set(bars_b) & shared) - set(bars_a))
    comparable = min(len(set(bars_a) & shared), len(set(bars_b) & shared))

    close = open_ = high = low = bar_range = None
    directional = None
    if alignment is AlignmentKind.ALIGNED and intersection:
        close = _dispersion(intersection, bars_a, bars_b, lambda bar: bar.close)
        open_ = _dispersion(intersection, bars_a, bars_b, lambda bar: bar.open)
        high = _dispersion(intersection, bars_a, bars_b, lambda bar: bar.high)
        low = _dispersion(intersection, bars_a, bars_b, lambda bar: bar.low)
        bar_range = _dispersion(intersection, bars_a, bars_b, lambda bar: bar.high - bar.low)
        directional = _directional(intersection, bars_a, bars_b, resolved)
    elif alignment is AlignmentKind.DIFFERENT_SESSION_ANCHOR:
        notes.append(
            "per-bar price metrics withheld: the two sources open their bars at "
            "different offsets, so no two candles cover the same interval"
        )

    if quality_a.duplicate_timestamps or quality_b.duplicate_timestamps:
        notes.append("a source repeated a timestamp; the later bar was used for comparison")
    if not quality_a.newest_is_closed or not quality_b.newest_is_closed:
        notes.append("a source's newest bar had not closed at the observation time")

    return MarketSourceComparison(
        canonical_symbol=canonical,
        timeframe=resolved,
        observed_at=moment,
        a=quality_a,
        b=quality_b,
        alignment=alignment,
        anchor_offset_seconds=anchor_offset,
        intersection_count=len(intersection),
        comparable_window_bars=comparable,
        only_in_a=len(only_a),
        only_in_b=len(only_b),
        only_in_a_samples=tuple(only_a[:MAX_REPORTED_TIMESTAMPS]),
        only_in_b_samples=tuple(only_b[:MAX_REPORTED_TIMESTAMPS]),
        close=close,
        open=open_,
        high=high,
        low=low,
        bar_range=bar_range,
        directional=directional,
        window=window,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# one series, on its own terms
# --------------------------------------------------------------------------


def describe_series(
    source: MarketDataInput, *, timeframe: Timeframe, observed_at: datetime
) -> SeriesQuality:
    """Completeness, freshness and grid shape for one source."""
    duration = _duration(timeframe)
    stamps = [_as_utc(bar.timestamp, "bar timestamp") for bar in source.bars]
    unique = sorted(set(stamps))

    offset, off_grid = _grid_offset(unique, duration)
    gaps = _gaps(unique, duration)
    newest = unique[-1] if unique else None

    closed = newest is not None and newest + duration <= observed_at
    age = int((observed_at - (newest + duration)).total_seconds()) if closed and newest else None
    expected = _expected_latest_closed(observed_at, duration, offset)

    return SeriesQuality(
        provider=source.provider,
        provider_symbol=source.provider_symbol,
        bar_count=len(source.bars),
        unique_timestamps=len(unique),
        duplicate_timestamps=len(stamps) - len(unique),
        ascending=stamps == sorted(stamps),
        oldest=unique[0] if unique else None,
        newest=newest,
        newest_is_closed=closed,
        newest_age_seconds=age,
        expected_latest_closed_open=expected,
        newest_is_expected_latest=newest is not None and newest == expected,
        grid_offset_seconds=offset,
        off_grid_bars=off_grid,
        session_breaks=sum(1 for gap in gaps if gap.kind is GapKind.SESSION_BREAK),
        quiet_intervals=sum(1 for gap in gaps if gap.kind is GapKind.QUIET_INTERVAL),
        gaps=gaps,
        volume_shape=_volume_shape(source.bars),
        price_decimals_max=_max_decimals(source.bars),
    )


def _grid_offset(stamps: list[datetime], duration: timedelta) -> tuple[int | None, int]:
    """The modal position of this source's bar openings within the period.

    Modal rather than "the first bar's", so one stray bar does not redefine the
    grid and then make every other bar look off it.
    """
    if not stamps:
        return None, 0
    period = int(duration.total_seconds())
    offsets = [int(stamp.timestamp()) % period for stamp in stamps]
    modal = statistics.mode(offsets)
    return modal, sum(1 for value in offsets if value != modal)


def _gaps(stamps: list[datetime], duration: timedelta) -> tuple[SeriesGap, ...]:
    """Breaks in the grid, classified by shape and never by blame."""
    found: list[SeriesGap] = []
    for earlier, later in zip(stamps, stamps[1:], strict=False):
        span = later - earlier
        if span <= duration:
            continue
        missing = int(span / duration) - 1
        if missing < 1:
            continue
        found.append(
            SeriesGap(
                after=earlier,
                before=later,
                missing_periods=missing,
                kind=(
                    GapKind.SESSION_BREAK
                    if span >= SESSION_BREAK_MINIMUM
                    else GapKind.QUIET_INTERVAL
                ),
            )
        )
    return tuple(found)


def _expected_latest_closed(
    observed_at: datetime, duration: timedelta, offset: int | None
) -> datetime | None:
    """The most recent interval that could have closed by *observed_at*.

    Computed on the source's *own* grid offset, so a feed whose sessions start
    at 01:00 is not judged against a feed whose sessions start at 00:00.
    """
    if offset is None:
        return None
    period = int(duration.total_seconds())
    seconds = int(observed_at.timestamp())
    current_open = seconds - ((seconds - offset) % period)
    return datetime.fromtimestamp(current_open - period, tz=UTC)


def _volume_shape(bars: list[OHLCBar]) -> VolumeShape:
    if not bars:
        return VolumeShape.ABSENT
    present = sum(1 for bar in bars if bar.volume is not None)
    if present == 0:
        return VolumeShape.ABSENT
    return VolumeShape.PRESENT if present == len(bars) else VolumeShape.PARTIAL


def _max_decimals(bars: list[OHLCBar]) -> int:
    """Most decimal places any price in the series carries.

    Recorded because two feeds quoting the same market to different precision
    will differ in the last digit by construction, and a reader of the price
    metrics should know that before interpreting them.
    """
    most = 0
    for bar in bars:
        for value in (bar.open, bar.high, bar.low, bar.close):
            exponent = value.as_tuple().exponent
            if isinstance(exponent, int):
                most = max(most, -exponent)
    return most


# --------------------------------------------------------------------------
# the two together
# --------------------------------------------------------------------------


def _classify_alignment(
    a: SeriesQuality,
    b: SeriesQuality,
    bars_a: dict[datetime, OHLCBar],
    bars_b: dict[datetime, OHLCBar],
    notes: list[str],
) -> tuple[AlignmentKind, int | None]:
    if not bars_a or not bars_b:
        notes.append("a source returned no bars")
        return AlignmentKind.NO_OVERLAP, None
    if not _shared_window(bars_a, bars_b):
        notes.append("the two series cover disjoint time windows")
        return AlignmentKind.NO_OVERLAP, None

    offset_a, offset_b = a.grid_offset_seconds, b.grid_offset_seconds
    if offset_a is None or offset_b is None:  # pragma: no cover - bars exist, so do offsets
        return AlignmentKind.NO_OVERLAP, None
    if offset_a != offset_b:
        notes.append(
            f"grid offsets differ by {offset_b - offset_a}s: the sources anchor "
            f"their {a.provider}/{b.provider} periods at different session boundaries"
        )
        return AlignmentKind.DIFFERENT_SESSION_ANCHOR, offset_b - offset_a
    return AlignmentKind.ALIGNED, 0


def _shared_window(
    bars_a: dict[datetime, OHLCBar], bars_b: dict[datetime, OHLCBar]
) -> frozenset[datetime]:
    """Timestamps from either source inside the window both cover.

    Restricting to the overlap is what keeps "only in A" honest: a source asked
    for three hundred bars against one asked for two hundred is not missing a
    hundred candles, it simply reaches further back.
    """
    if not bars_a or not bars_b:
        return frozenset()
    start = max(min(bars_a), min(bars_b))
    end = min(max(bars_a), max(bars_b))
    if start > end:
        return frozenset()
    return frozenset(stamp for stamp in (*bars_a, *bars_b) if start <= stamp <= end)


def _dispersion(
    stamps: list[datetime],
    bars_a: dict[datetime, OHLCBar],
    bars_b: dict[datetime, OHLCBar],
    read: Callable[[OHLCBar], Decimal],
) -> Dispersion:
    pairs = [(stamp, abs(read(bars_a[stamp]) - read(bars_b[stamp]))) for stamp in stamps]
    values = sorted(value for _, value in pairs)
    worst = max(pairs, key=lambda item: item[1])
    return Dispersion(
        sample_count=len(values),
        median_abs=_quantile(values, Decimal("0.5")),
        p95_abs=_quantile(values, Decimal("0.95")),
        max_abs=worst[1],
        max_at=worst[0],
    )


def _quantile(sorted_values: list[Decimal], fraction: Decimal) -> Decimal:
    """Nearest-rank quantile over exact decimals.

    Nearest-rank rather than interpolated, so every reported figure is a
    difference that actually occurred on some bar rather than an average of two
    that did not. No statistics dependency, and none is warranted for this.
    """
    if not sorted_values:  # pragma: no cover - callers check first
        return Decimal(0)
    index = int((len(sorted_values) - 1) * fraction)
    return sorted_values[index]


def _directional(
    stamps: list[datetime],
    bars_a: dict[datetime, OHLCBar],
    bars_b: dict[datetime, OHLCBar],
    timeframe: Timeframe,
) -> DirectionalAgreement:
    """Agreement on the sign of each close-to-close move.

    Only over pairs that are genuinely consecutive on the grid: comparing the
    move across a weekend to the move across a minute would count a gap as a
    disagreement.
    """
    duration = _duration(timeframe)
    agreed = 0
    comparable = 0
    ties = 0
    for earlier, later in zip(stamps, stamps[1:], strict=False):
        if later - earlier != duration:
            continue
        delta_a = bars_a[later].close - bars_a[earlier].close
        delta_b = bars_b[later].close - bars_b[earlier].close
        if not delta_a or not delta_b:
            ties += 1
            continue
        comparable += 1
        if (delta_a > 0) == (delta_b > 0):
            agreed += 1
    return DirectionalAgreement(comparable_intervals=comparable, agreed=agreed, flat_ties=ties)


def _window_coherence(
    bars_a: dict[datetime, OHLCBar], bars_b: dict[datetime, OHLCBar], notes: list[str]
) -> WindowCoherence | None:
    """Price agreement over shared clock time, pairing nothing.

    Valid whether or not the grids align, which is the point: when anchors
    differ this is the only honest price comparison left.
    """
    if not bars_a or not bars_b:
        return None
    start = max(min(bars_a), min(bars_b))
    end = min(max(bars_a), max(bars_b))
    if start > end:
        return None

    inside_a = [bar for stamp, bar in bars_a.items() if start <= stamp <= end]
    inside_b = [bar for stamp, bar in bars_b.items() if start <= stamp <= end]
    if not inside_a or not inside_b:  # pragma: no cover - the window is built from them
        return None

    last_a = max(inside_a, key=lambda bar: bar.timestamp)
    last_b = max(inside_b, key=lambda bar: bar.timestamp)
    gap = abs(int((last_a.timestamp - last_b.timestamp).total_seconds()))
    if gap:
        notes.append(
            f"the two 'last closes' in the shared window are {gap}s apart, so their "
            f"difference is partly a difference in time"
        )

    return WindowCoherence(
        window_start=start,
        window_end=end,
        bars_a=len(inside_a),
        bars_b=len(inside_b),
        high_difference=abs(max(bar.high for bar in inside_a) - max(bar.high for bar in inside_b)),
        low_difference=abs(min(bar.low for bar in inside_a) - min(bar.low for bar in inside_b)),
        last_close_difference=abs(last_a.close - last_b.close),
        last_close_gap_seconds=gap,
    )


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------


def _require_same_symbol(a: MarketDataInput, b: MarketDataInput) -> str:
    if a.symbol != b.symbol:
        raise IncomparableSourcesError(
            "refusing to compare two different instruments",
            symbol_a=a.symbol,
            symbol_b=b.symbol,
        )
    return a.symbol


def _require_same_timeframe(
    a: MarketDataInput, b: MarketDataInput, requested: Timeframe | None
) -> Timeframe:
    if a.timeframe != b.timeframe:
        raise IncomparableSourcesError(
            "refusing to compare two different timeframes",
            timeframe_a=str(a.timeframe),
            timeframe_b=str(b.timeframe),
        )
    if requested is not None and requested != a.timeframe:
        raise IncomparableSourcesError(
            "the requested timeframe is not the one these payloads carry",
            requested=str(requested),
            payload=str(a.timeframe),
        )
    return Timeframe(a.timeframe)


def _duration(timeframe: Timeframe) -> timedelta:
    duration = timeframe.duration
    if duration is None:
        raise IncomparableSourcesError(
            "this timeframe has no fixed duration, so a bar grid is undefined",
            timeframe=str(timeframe),
        )
    return duration


def _by_timestamp(bars: list[OHLCBar]) -> dict[datetime, OHLCBar]:
    """Bars keyed by timestamp, later occurrences winning.

    Order and duplication are reported by :func:`describe_series`; this mapping
    exists so the comparison has one bar per instant regardless.
    """
    return {_as_utc(bar.timestamp, "bar timestamp"): bar for bar in bars}


def _as_utc(moment: datetime, label: str) -> datetime:
    if moment.tzinfo is None:
        raise IncomparableSourcesError(
            f"{label} is naive; a comparison cannot guess which zone it means",
            label=label,
        )
    return moment.astimezone(UTC)


__all__ = [
    "MARKET_COMPARISON_VERSION",
    "SESSION_BREAK_MINIMUM",
    "AlignmentKind",
    "DirectionalAgreement",
    "Dispersion",
    "GapKind",
    "MarketSourceComparison",
    "SeriesGap",
    "SeriesQuality",
    "VolumeShape",
    "WindowCoherence",
    "compare_market_sources",
    "describe_series",
]
