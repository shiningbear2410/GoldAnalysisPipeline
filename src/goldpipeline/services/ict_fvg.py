"""What later candles did to a fair value gap. Nothing about what it means.

Round 6.6c.2. Round 6.6a found three-candle imbalances and stopped there,
which was right: a gap is a place price moved without transacting, and whether
anyone should care depends on things that round could not compute. This module
adds the one fact that needs no interpretation - **did later observed trading
enter this band, and did it cover the whole of it?**

**Observed range only. No inferred path.** The engine sees a candle's
``[low, high]`` and nothing else. It does not know the order prices were
visited in, so it never reasons about a path it cannot see. That restraint is
the whole point of the fill rule below: a candle sitting entirely under a
bullish gap has a low beneath the gap's lower edge, and a naive
``low <= lower`` test would call the gap filled by price that demonstrably
never traded inside it.

**Direction is about formation, not lifecycle.** Bullish and bearish describe
how the imbalance was made. The lifecycle question - did trading enter this
price band - is the same question either way, so there is one geometric rule
rather than two mirrored ones.

**Status is observation, never a recommendation.** An ``OPEN`` bullish gap is
not a buy zone, support, or BAI; an ``OPEN`` bearish gap is not resistance or
SEO. Those readings need structure, liquidity and a policy about freshness,
none of which is here, and `test_ict_fvg_guards` pins their absence.

**One detection authority.** The three-candle test lives in
:func:`~goldpipeline.services.ict_primitives.fair_value_gaps` and is not
reimplemented here. This module either receives gaps already detected or calls
that function exactly once, so a future composite ICT stage can detect gaps a
single time and hand them to several consumers.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import (
    IctMarketSnapshot,
    IctTimeframeSnapshot,
    build_timeframe_snapshot,
)
from goldpipeline.schemas.market import OHLCBar
from goldpipeline.services.ict_primitives import FairValueGap, fair_value_gaps

logger = logging.getLogger(__name__)

FVG_LIFECYCLE_METHOD_VERSION = "1.0.0"
"""Stamped on every state and analysis, and part of every gap identity.

Bumped whenever a definition below changes meaning, so a state recorded by one
version can never be read as though another produced it.
"""


def fvg_id(gap: FairValueGap, *, symbol: str = "") -> str:
    """Deterministic identity for one fair value gap.

    Sixteen hex characters of SHA-256 over exactly these fields, joined by
    ``|``, in this order: symbol, timeframe, direction, the method version, the
    three formation bar open times, and the band's two edges. Everything in it
    is fixed when the gap forms, so an identity never changes and replaying
    identical candles reproduces it exactly.

    Provider and provider symbol are deliberately absent: the same candles from
    TradingView, MetaTrader or a fixture describe the same gap, and an identity
    that disagreed would make provenance look like meaning.

    Written as a function rather than a field on ``FairValueGap`` for the same
    reason :func:`~goldpipeline.services.ict_structure.swing_id` is - the 6.6a
    detector stays the single authority for what a gap *is*, untouched by
    rounds that come later and want to name one.
    """
    preimage = "|".join(
        (
            symbol,
            gap.timeframe.value,
            gap.direction.value,
            FVG_LIFECYCLE_METHOD_VERSION,
            gap.first_time.isoformat(),
            gap.middle_time.isoformat(),
            gap.last_time.isoformat(),
            _canonical(gap.lower),
            _canonical(gap.upper),
        )
    )
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]


def _canonical(price: Decimal) -> str:
    """A price as it appears in an identity, trailing zeros removed.

    ``Decimal("4000")`` and ``Decimal("4000.00")`` are the same price written
    with different precision, and providers disagree about that routinely - one
    feed reports two decimals on a round number, another does not. Without this
    the same candle from two sources would produce two gap identities, which is
    exactly the provider-neutrality the preimage above is careful to preserve.
    """
    return format(price.normalize(), "f")


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------


class FvgStatus(StrEnum):
    """How far a gap has got. Three values, and no fourth.

    ``MITIGATED``, ``INVERTED``, ``BREAKER`` and ``REBALANCED`` are absent
    because each names a reading rather than an observation - mitigated by
    which order, inverted for whose entry - and a status carrying one would be
    a promise this code cannot keep.
    """

    OPEN = "OPEN"
    TOUCHED = "TOUCHED"
    FILLED = "FILLED"


@dataclass(frozen=True)
class BarWitness:
    """The candle that supplied the evidence for a transition.

    Recorded rather than merely timestamped, so a later reader can check the
    verdict without replaying the analysis: the range that touched or covered
    the band is right here.
    """

    bar_open_time: datetime
    bar_close_time: datetime
    low: Decimal
    high: Decimal


@dataclass(frozen=True)
class FvgState:
    """One gap, and what has happened to it as of the analysis instant."""

    fvg_id: str
    method_version: str
    gap: FairValueGap
    status: FvgStatus

    first_touched_at: datetime | None
    first_touch: BarWitness | None
    """The **first** qualifying interaction, never overwritten by a later one.

    Repeated touches are deliberately not counted. A reaction count is a
    quality signal, and quality signals belong to candidate scoring where the
    weighting can be argued about, not to a primitive that has to stay true.
    """

    filled_at: datetime | None
    fill: BarWitness | None

    @property
    def is_terminal(self) -> bool:
        return self.status is FvgStatus.FILLED


def touchable(status: FvgStatus, formed_at: datetime, close_time: datetime) -> bool:
    """Whether a bar closing at *close_time* may interact with a gap in this state.

    Two conditions:

    * the gap is not already ``FILLED`` - a filled gap is finished, and twenty
      later candles through the same band change nothing;
    * it was formed **strictly before** this close. Candle C is what makes the
      gap knowable, and it may not also be credited with entering it. Seeing a
      zone at T and interacting with it at T are separate permissions, exactly
      as in Rounds 6.6b and 6.6c.1.

    Reported honestly: the strictness cannot bind under the 6.6a definition.
    The only bar closing at ``formed_at`` is C itself, and C's range can never
    enter its own band - a bullish gap's upper edge *is* ``C.low``, so C has
    nothing strictly inside, and the bearish case mirrors it. The rule is
    enforced regardless, because that argument rests on a gap definition a
    later round may change.
    """
    return status is not FvgStatus.FILLED and formed_at < close_time


def covers(gap: FairValueGap, low: Decimal, high: Decimal) -> bool:
    """Whether an observed range covers the gap's whole band.

    ``low <= gap.lower and high >= gap.upper``. Both halves are required, and
    that is the guard against a false fill.

    A bullish gap at ``[4000, 4010]`` and a later candle with ``low = 3980``,
    ``high = 3995`` satisfies ``low <= lower`` while having demonstrably never
    traded inside the band - it gapped clean underneath. Treating that as a
    fill would retire a zone the market has not been to, and every later stage
    reading "filled" would inherit the mistake.
    """
    return low <= gap.lower and high >= gap.upper


def enters(gap: FairValueGap, low: Decimal, high: Decimal) -> bool:
    """Whether an observed range intersects the gap's **interior**, ``(lower, upper)``.

    ``high > gap.lower and low < gap.upper``. Strict on both sides, so contact
    with a single outer edge is not an entry: a candle whose low is exactly the
    upper edge sat on the band without going into it, and calling that a touch
    would make every zone in a trending market instantly touched.

    Direction-neutral by design - see the module docstring.
    """
    return high > gap.lower and low < gap.upper


@dataclass(frozen=True)
class FvgLifecycleAnalysis:
    """One timeframe's gaps and their states, as of one instant.

    The smallest thing later rounds need. No scoring, no confluence with
    liquidity or structure, no candidate zones, and nothing merged across
    timeframes.
    """

    method_version: str
    timeframe: Timeframe
    symbol: str
    observed_at: datetime

    states: tuple[FvgState, ...]
    open_ids: tuple[str, ...]
    touched_ids: tuple[str, ...]
    filled_ids: tuple[str, ...]

    bars_considered: int

    def state(self, identity: str) -> FvgState | None:
        return next((entry for entry in self.states if entry.fvg_id == identity), None)

    def with_status(self, status: FvgStatus) -> tuple[FvgState, ...]:
        return tuple(entry for entry in self.states if entry.status is status)


@dataclass
class _Working:
    """A gap's state while the replay is still running."""

    identity: str
    gap: FairValueGap
    status: FvgStatus = FvgStatus.OPEN
    first_touched_at: datetime | None = None
    first_touch: BarWitness | None = None
    filled_at: datetime | None = None
    fill: BarWitness | None = None

    def freeze(self) -> FvgState:
        return FvgState(
            fvg_id=self.identity,
            method_version=FVG_LIFECYCLE_METHOD_VERSION,
            gap=self.gap,
            status=self.status,
            first_touched_at=self.first_touched_at,
            first_touch=self.first_touch,
            filled_at=self.filled_at,
            fill=self.fill,
        )


def _ordering(gap: FairValueGap, identity: str) -> tuple[datetime, str, Decimal, Decimal, str]:
    """A total order over gaps, derived only from formation facts.

    Formation time first, which is the order a reader expects. The rest breaks
    ties that the 6.6a detector cannot actually produce - one candle C yields at
    most one gap - but which a caller supplying its own list could, and no
    ordering here may depend on how a set or dict happened to iterate.
    """
    return (gap.formed_at, gap.direction.value, gap.lower, gap.upper, identity)


def analyse_fvg_lifecycle(
    series: IctTimeframeSnapshot,
    *,
    gaps: Sequence[FairValueGap] | None = None,
    symbol: str = "",
    as_of: datetime | None = None,
) -> FvgLifecycleAnalysis:
    """Replay *series* bar by bar and report what happened to each gap.

    The exact order of operations at each closed bar:

    1. Skip gaps already ``FILLED``, and gaps formed at or after this close.
    2. If the bar's range **covers** the whole band, the gap is ``FILLED`` and
       the fill provenance is recorded.
    3. Otherwise, if the range **enters** the interior and the gap is still
       ``OPEN``, it becomes ``TOUCHED`` and the first-touch provenance is
       recorded. An already-``TOUCHED`` gap keeps its original first touch.
    4. Otherwise nothing changes.

    Fill is tested before touch because a covering candle both covers and
    enters, and the stronger fact is the verdict. On a direct ``OPEN → FILLED``
    the first touch is set to the fill bar, so "when did price first enter this
    gap?" always has an answer.

    Each gap evolves independently; one candle may fill several and touch
    several more, and overlapping gaps are never merged.

    Args:
        series: Closed candles for one timeframe.
        gaps: Already-detected gaps, for a caller that has them. Filtered to
            those formed by the analysis instant, so passing a list computed
            over a longer series cannot leak a future gap into an earlier
            answer. Omit to detect them here, which calls the one authority
            exactly once.
        symbol: Instrument name. Provenance, and part of every gap identity.
        as_of: Reconstruct the state as it stood at this instant.

    Raises:
        ValueError: *as_of* precedes every bar's close, so there is no series to
            analyse. Refused rather than answered with an empty result, which
            would be indistinguishable from "no gaps".
    """
    working_series = series
    if as_of is not None:
        working_series = build_timeframe_snapshot(
            timeframe=series.timeframe, bars=series.bars, observed_at=as_of
        )
    observed_at = as_of if as_of is not None else series.latest_closed_at

    duration = working_series.timeframe.duration
    assert duration is not None  # IctTimeframeSnapshot refuses calendar timeframes

    detected = fair_value_gaps(working_series) if gaps is None else list(gaps)
    # Belt and braces against a caller handing over gaps from a longer series:
    # a gap whose third candle had not closed by the analysis instant did not
    # exist yet, and must not appear in an answer dated then.
    known = [gap for gap in detected if gap.formed_at <= observed_at]

    tracked = [_Working(identity=fvg_id(gap, symbol=symbol), gap=gap) for gap in known]
    tracked.sort(key=lambda entry: _ordering(entry.gap, entry.identity))

    for bar in working_series.bars:
        close_time = bar.timestamp + duration
        _apply_bar(tracked, bar, close_time)

    states = tuple(entry.freeze() for entry in tracked)
    return FvgLifecycleAnalysis(
        method_version=FVG_LIFECYCLE_METHOD_VERSION,
        timeframe=working_series.timeframe,
        symbol=symbol,
        observed_at=observed_at,
        states=states,
        open_ids=tuple(s.fvg_id for s in states if s.status is FvgStatus.OPEN),
        touched_ids=tuple(s.fvg_id for s in states if s.status is FvgStatus.TOUCHED),
        filled_ids=tuple(s.fvg_id for s in states if s.status is FvgStatus.FILLED),
        bars_considered=working_series.bar_count,
    )


def analyse_snapshot_fvg_lifecycle(
    snapshot: IctMarketSnapshot,
    timeframe: Timeframe,
    *,
    gaps: Sequence[FairValueGap] | None = None,
    as_of: datetime | None = None,
) -> FvgLifecycleAnalysis:
    """:func:`analyse_fvg_lifecycle` for one timeframe of a multi-timeframe snapshot.

    One timeframe, named explicitly. There is deliberately no call that pools an
    H4 gap with an H1 gap at the same price: that is a confluence rule with real
    content, and it belongs to candidate scoring where it can be argued with.
    """
    return analyse_fvg_lifecycle(
        snapshot.require(timeframe),
        gaps=gaps,
        symbol=snapshot.symbol,
        as_of=as_of,
    )


def _apply_bar(tracked: Sequence[_Working], bar: OHLCBar, close_time: datetime) -> None:
    """Update every eligible gap against one closed candle.

    No one-transition-per-bar rule. A single wide candle can genuinely fill one
    gap while touching another, and can act on bullish and bearish gaps at once;
    imposing the structure engine's single-event rule here would silently drop
    real observations.
    """
    for entry in tracked:
        if not touchable(entry.status, entry.gap.formed_at, close_time):
            continue

        witness = BarWitness(
            bar_open_time=bar.timestamp,
            bar_close_time=close_time,
            low=bar.low,
            high=bar.high,
        )

        if covers(entry.gap, bar.low, bar.high):
            entry.status = FvgStatus.FILLED
            entry.filled_at = close_time
            entry.fill = witness
            if entry.first_touched_at is None:
                # A direct OPEN -> FILLED is still the first time price entered
                # this band, and recording it keeps "when was this gap first
                # reached?" answerable for every filled gap rather than only the
                # ones that happened to be touched first.
                entry.first_touched_at = close_time
                entry.first_touch = witness
            continue

        if enters(entry.gap, bar.low, bar.high) and entry.status is FvgStatus.OPEN:
            entry.status = FvgStatus.TOUCHED
            entry.first_touched_at = close_time
            entry.first_touch = witness


__all__ = [
    "FVG_LIFECYCLE_METHOD_VERSION",
    "BarWitness",
    "FvgLifecycleAnalysis",
    "FvgState",
    "FvgStatus",
    "analyse_fvg_lifecycle",
    "analyse_snapshot_fvg_lifecycle",
    "covers",
    "enters",
    "fvg_id",
    "touchable",
]
