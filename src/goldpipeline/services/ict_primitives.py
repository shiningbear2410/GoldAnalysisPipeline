"""Three deterministic observations over closed candles. Nothing more.

Round 6.6a. These are the primitives every later ICT round inherits, which is
the whole argument for making them boring first: a vague swing definition
becomes a wrong break of structure, a wrong order block, and a wrong published
level, and by then nobody can find the original mistake.

**Observations, not readings.** A confirmed swing high is a shape three or more
candles make. It is *not* buy-side liquidity, resistance, or a place to sell -
those are interpretations, they need structure and context, and they belong to
later rounds. The names here stay geometric on purpose, because a type called
``LiquidityHigh`` would be asserting something this code has not established.

**Two ATRs live in this project, deliberately.**
:func:`~goldpipeline.services.levels.average_true_range` is a simple mean, and
its docstring explains why: an analysis claim citing it must be recomputable
from the bars in one ``context.json``, and Wilder's recursion reaches back past
whatever history that file holds. The ICT branch has the opposite requirement -
it wants the smoothing traders actually use, over a series it controls the depth
of - so it computes Wilder here rather than changing a number seven published
Runs already cite. Same input, two named definitions, neither pretending to be
the other.

**No lookahead, ever.** A swing is confirmed only once its right-hand bars have
closed, and the confirmation instant is recorded separately from the pivot's own
timestamp. A later stage asking "what did we know at time T" filters on
``confirmed_at``, and gets an answer that does not depend on how much future the
snapshot happens to contain.

**Provider-neutral.** These functions read normalized candles and nothing else.
The same bars from TradingView, MetaTrader or a fixture produce identical
numbers; the provider is recorded on the snapshot as provenance and is never
consulted for meaning.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import IctTimeframeSnapshot
from goldpipeline.schemas.market import OHLCBar

logger = logging.getLogger(__name__)

DEFAULT_ATR_PERIOD = 14
"""Wilder's own period, and the one every charting platform defaults to."""

DEFAULT_LEFT_BARS = 2
DEFAULT_RIGHT_BARS = 2
"""Pivot window either side of a candidate swing.

Two is the smallest window that rejects ordinary noise while still confirming
within a few bars. Both sides are parameters rather than one shared ``window``
because they answer different questions - how much history a pivot dominates,
and how long the market must fail to exceed it - and a later round may well want
them asymmetric.
"""


# --------------------------------------------------------------------------
# ATR
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AtrPoint:
    """Wilder's ATR as of one closed bar."""

    timeframe: Timeframe
    bar_open_time: datetime
    bar_close_time: datetime
    period: int
    true_range: Decimal
    """This bar's own true range, before smoothing."""

    atr: Decimal
    """The smoothed value this bar produces."""


def true_range(previous: OHLCBar, current: OHLCBar) -> Decimal:
    """``max(high-low, |high-prev_close|, |low-prev_close|)``.

    Takes the predecessor explicitly because a true range is a two-bar
    measurement. The first bar of any finite series therefore has none - see
    :func:`atr_series` for what that means for the caller, and note that
    substituting ``high - low`` there would silently understate an opening gap.
    """
    prev_close = previous.close
    return max(
        current.high - current.low,
        abs(current.high - prev_close),
        abs(current.low - prev_close),
    )


def true_range_series(bars: Sequence[OHLCBar]) -> list[tuple[OHLCBar, Decimal]]:
    """Each bar that has a predecessor, with its true range."""
    return [
        (current, true_range(previous, current))
        for previous, current in zip(bars, bars[1:], strict=False)
    ]


def atr_series(series: IctTimeframeSnapshot, period: int = DEFAULT_ATR_PERIOD) -> list[AtrPoint]:
    """Wilder's ATR for every bar that has enough history behind it.

    Seeded and recursed the standard way, pinned here so nothing later has to
    guess which variant produced a number:

    * the first value is the **simple average of the first ``period`` true
      ranges**;
    * every later value is ``(previous * (period - 1) + TR) / period``.

    The first bar of the series contributes no true range - it has no previous
    close - so the first ATR belongs to bar ``period + 1``, and one warm-up bar
    is consumed. A snapshot of exactly ``period`` bars yields nothing rather
    than a shorter average wearing the wrong name.

    Raises:
        ValueError: ``period`` is not positive. A zero or negative period is a
            caller mistake, not a market condition, and returning ``[]`` would
            hide it among the honest empty answers.
    """
    if period < 1:
        raise ValueError(f"ATR period must be positive, got {period}")

    ranges = true_range_series(series.bars)
    if len(ranges) < period:
        # Not "average what we have": a six-bar mean labelled a 14-period ATR is
        # a wrong number wearing a right name.
        logger.debug(
            "ict.atr insufficient timeframe=%s bars=%d period=%d",
            series.timeframe,
            len(series.bars),
            period,
        )
        return []

    duration = series.timeframe.duration
    assert duration is not None  # the snapshot validator guarantees this

    seed_bars = ranges[:period]
    current = sum((value for _, value in seed_bars), Decimal(0)) / Decimal(period)

    points = [
        AtrPoint(
            timeframe=series.timeframe,
            bar_open_time=seed_bars[-1][0].timestamp,
            bar_close_time=seed_bars[-1][0].timestamp + duration,
            period=period,
            true_range=seed_bars[-1][1],
            atr=current,
        )
    ]

    for bar, value in ranges[period:]:
        current = (current * Decimal(period - 1) + value) / Decimal(period)
        points.append(
            AtrPoint(
                timeframe=series.timeframe,
                bar_open_time=bar.timestamp,
                bar_close_time=bar.timestamp + duration,
                period=period,
                true_range=value,
                atr=current,
            )
        )
    return points


def latest_atr(series: IctTimeframeSnapshot, period: int = DEFAULT_ATR_PERIOD) -> AtrPoint | None:
    """The most recent ATR, or ``None`` when there is not enough history."""
    points = atr_series(series, period)
    return points[-1] if points else None


# --------------------------------------------------------------------------
# confirmed swings
# --------------------------------------------------------------------------


class SwingType(StrEnum):
    """What kind of pivot a bar formed. A shape, not a role."""

    SWING_HIGH = "SWING_HIGH"
    SWING_LOW = "SWING_LOW"


@dataclass(frozen=True)
class SwingPoint:
    """One confirmed pivot, and the moment it became knowable."""

    timeframe: Timeframe
    swing_type: SwingType
    pivot_time: datetime
    """Open time of the bar that made the extreme."""

    confirmed_at: datetime
    """Close time of the last right-hand bar required to confirm it.

    Separate from :attr:`pivot_time` and this is the field that matters. A
    pivot at 10:00 with two right bars was not knowable at 10:00; a stage
    reasoning "as of T" must filter on this, or it will use a level the market
    had not yet finished forming.
    """

    price: Decimal
    left_bars: int
    right_bars: int


def confirmed_swings(
    series: IctTimeframeSnapshot,
    *,
    left_bars: int = DEFAULT_LEFT_BARS,
    right_bars: int = DEFAULT_RIGHT_BARS,
) -> list[SwingPoint]:
    """Every confirmed swing high and low, oldest first.

    A bar is a swing high when its high is **strictly greater** than every high
    in the ``left_bars`` before it *and* every high in the ``right_bars`` after
    it. Swing lows mirror it with strict ``<``.

    **Strict on both sides, and equal highs therefore make no pivot.** That is
    the intended answer, not a gap. Two bars sharing a high are a shape with its
    own meaning - a pool of resting orders - and calling one of them a pivot by
    a first-wins or last-wins tie-break would be inventing a distinction the
    market did not make. Equal highs and lows become a liquidity concept in a
    later round, where they can be named for what they are.

    Note this differs deliberately from
    :func:`~goldpipeline.services.levels.swing_highs`, which resolves ties with
    a strict-left / non-strict-right rule so that ANALYSIS always gets exactly
    one pivot from a tie. That module serves a published claim path and its
    behaviour is fixed; this one serves a structure engine that needs ties left
    visible.

    Raises:
        ValueError: Either window is not positive.
    """
    if left_bars < 1 or right_bars < 1:
        raise ValueError(f"pivot windows must be positive, got left={left_bars} right={right_bars}")

    bars = series.bars
    duration = series.timeframe.duration
    assert duration is not None

    found: list[SwingPoint] = []
    for index in range(left_bars, len(bars) - right_bars):
        bar = bars[index]
        left = bars[index - left_bars : index]
        right = bars[index + 1 : index + 1 + right_bars]
        # The last right-hand bar's close is when this pivot became knowable.
        confirmed_at = right[-1].timestamp + duration

        if all(bar.high > other.high for other in left) and all(
            bar.high > other.high for other in right
        ):
            found.append(
                SwingPoint(
                    timeframe=series.timeframe,
                    swing_type=SwingType.SWING_HIGH,
                    pivot_time=bar.timestamp,
                    confirmed_at=confirmed_at,
                    price=bar.high,
                    left_bars=left_bars,
                    right_bars=right_bars,
                )
            )

        if all(bar.low < other.low for other in left) and all(
            bar.low < other.low for other in right
        ):
            found.append(
                SwingPoint(
                    timeframe=series.timeframe,
                    swing_type=SwingType.SWING_LOW,
                    pivot_time=bar.timestamp,
                    confirmed_at=confirmed_at,
                    price=bar.low,
                    left_bars=left_bars,
                    right_bars=right_bars,
                )
            )

    found.sort(key=lambda point: (point.pivot_time, point.swing_type))
    return found


def swings_known_at(swings: Sequence[SwingPoint], moment: datetime) -> list[SwingPoint]:
    """The subset a stage reasoning "as of *moment*" is allowed to see.

    The whole point of recording ``confirmed_at``. Inclusive of the instant
    itself: a pivot confirmed by a bar closing exactly at *moment* was knowable
    then, in the same way the snapshot's own closed-bar rule admits a candle
    closing exactly at the observation.
    """
    return [swing for swing in swings if swing.confirmed_at <= moment]


# --------------------------------------------------------------------------
# fair value gaps
# --------------------------------------------------------------------------


class GapDirection(StrEnum):
    """Which way the three-candle displacement ran. Not a trade direction."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


@dataclass(frozen=True)
class FairValueGap:
    """A three-candle imbalance, as it stood when its third candle closed.

    ``lower`` and ``upper`` are the untraded band itself - the region the middle
    candle skipped. Nothing here says what to do about it: a bullish gap is not
    a buy zone, it is a place where price moved without transacting, and whether
    that matters depends on structure this round does not compute.
    """

    timeframe: Timeframe
    direction: GapDirection

    formed_at: datetime
    """Close time of candle C. The gap exists once that candle has finished.

    Not B's open, not B's close, not C's open. A three-candle pattern is not a
    fact until its third candle is closed, and a stage that treated C's open as
    the formation time would be using a shape the market had not yet made.
    """

    first_time: datetime
    """Open time of candle A, whose extreme forms one edge."""

    middle_time: datetime
    """Open time of candle B, the displacement that leaves the gap."""

    last_time: datetime
    """Open time of candle C, whose extreme forms the other edge."""

    lower: Decimal
    upper: Decimal

    @property
    def size(self) -> Decimal:
        """Height of the untraded band. Exact; no tick or pip rounding."""
        return self.upper - self.lower

    @property
    def midpoint(self) -> Decimal:
        """The band's centre. Named without judgement - this round attaches no
        meaning to the halfway point of a gap."""
        return (self.lower + self.upper) / Decimal(2)


def fair_value_gaps(series: IctTimeframeSnapshot) -> list[FairValueGap]:
    """Every three-candle fair value gap in the series, oldest first.

    For candles ``A = i-2``, ``B = i-1``, ``C = i``:

    * **bullish** when ``C.low > A.high`` - the band is ``[A.high, C.low]``;
    * **bearish** when ``C.high < A.low`` - the band is ``[C.high, A.low]``.

    **Strict inequality, so a touch is not a gap.** ``C.low == A.high`` means
    the two candles met exactly: every price in between traded, there is no
    untraded band, and a zone of zero height would be a level pretending to be
    a region.
    """
    bars = series.bars
    duration = series.timeframe.duration
    assert duration is not None

    found: list[FairValueGap] = []
    for index in range(2, len(bars)):
        first, middle, last = bars[index - 2], bars[index - 1], bars[index]

        if last.low > first.high:
            lower, upper, direction = first.high, last.low, GapDirection.BULLISH
        elif last.high < first.low:
            lower, upper, direction = last.high, first.low, GapDirection.BEARISH
        else:
            continue

        found.append(
            FairValueGap(
                timeframe=series.timeframe,
                direction=direction,
                formed_at=last.timestamp + duration,
                first_time=first.timestamp,
                middle_time=middle.timestamp,
                last_time=last.timestamp,
                lower=lower,
                upper=upper,
            )
        )
    return found


__all__ = [
    "DEFAULT_ATR_PERIOD",
    "DEFAULT_LEFT_BARS",
    "DEFAULT_RIGHT_BARS",
    "AtrPoint",
    "FairValueGap",
    "GapDirection",
    "SwingPoint",
    "SwingType",
    "atr_series",
    "confirmed_swings",
    "fair_value_gaps",
    "latest_atr",
    "swings_known_at",
    "true_range",
    "true_range_series",
]
