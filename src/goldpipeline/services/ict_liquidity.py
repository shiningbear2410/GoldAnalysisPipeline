"""Repeated swings, the band they make, and the candle that took it.

Round 6.6c.1, and the first liquidity layer. Round 6.6a said a swing high is a
shape and refused to call it buy-side liquidity; this is the module allowed to
make that reading, and only for the specific case it can actually establish:
**the market printed two or more swings at effectively the same price.**

**Why repetition is the whole claim.** One swing high is a place price turned
once. Two at the same level is a place price turned, was remembered, and turned
again - and the orders that accumulate above such a level are the thing an
eventual trade plan cares about. This module asserts nothing beyond that
geometry. It does not know whose stops are there, or whether anyone intended
them to be taken.

**Tolerance is the caller's policy, never this module's.** ``LiquidityConfig``
has no default. There is no ``0.5``, no ``1.0``, and no ``ATR * 0.1`` anywhere
below. A production tolerance for XAUUSD is a product parameter that deserves
its own argument in its own round; baking one in here would make a tuning
choice look like a definition, which is how a number nobody can defend ends up
underneath a published price level.

**Equality here is not Round 6.6b's equality.**
:class:`~goldpipeline.services.ict_structure.SwingRelation` ``EQUAL`` means
exactly equal Decimals, and structure needs it that way. Liquidity equality is
a band. A swing labelled ``HIGHER_HIGH`` by the structure engine can sit in the
same buy-side pool as the high before it, and that is correct rather than a
contradiction: they answer different questions. Nothing in this module reads a
swing relation, and ``test_a_relation_label_is_not_a_prerequisite`` pins it.

**A sweep is geometry, not intent.** The event is called ``WICK_SWEEP``, never
``STOP_HUNT`` and never ``MANIPULATION``. Price traded strictly beyond a band
and closed back inside it. Who was on the other side, and why, is not
observable from candles and this engine does not guess.

**One swing authority.** Pivots come from
:func:`~goldpipeline.services.ict_primitives.confirmed_swings` and identities
from :func:`~goldpipeline.services.ict_structure.swing_id`. No pivot is derived
here, so structure and liquidity can never disagree about what a swing is.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
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
from goldpipeline.services.ict_primitives import (
    DEFAULT_LEFT_BARS,
    DEFAULT_RIGHT_BARS,
    SwingPoint,
    SwingType,
    confirmed_swings,
)
from goldpipeline.services.ict_structure import swing_id

logger = logging.getLogger(__name__)

LIQUIDITY_METHOD_VERSION = "1.0.0"
"""Stamped on every pool, event and analysis.

Bumped whenever a definition below changes meaning. Pool and event identities
include it, so a pool recorded by one version can never be mistaken for the
same pool under another.
"""


# --------------------------------------------------------------------------
# side
# --------------------------------------------------------------------------


class LiquiditySide(StrEnum):
    """Which side of price the resting orders sit on.

    A swing high has buy-side liquidity above it - stops from shorts, and
    breakout buys. A swing low has sell-side liquidity below it. This is the
    ICT convention and it is deliberately *not* a trade instruction: buy-side
    liquidity is a place to sell into as often as not. Nothing here emits SEO,
    BAI, BUY or SELL.
    """

    BUY_SIDE = "BUY_SIDE"
    SELL_SIDE = "SELL_SIDE"


def side_of(swing_type: SwingType) -> LiquiditySide:
    """The liquidity side a confirmed swing of *swing_type* creates."""
    return LiquiditySide.BUY_SIDE if swing_type is SwingType.SWING_HIGH else LiquiditySide.SELL_SIDE


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LiquidityConfig:
    """How close two swings must be to count as the same liquidity.

    One required field and no default, on purpose. The engine cannot pick this
    number: it depends on the instrument, the timeframe, the venue's spread and
    what the eventual product intends to do with a pool, none of which is
    visible from a candle series. Supplying it is the caller's decision and is
    recorded in every pool identity, so two runs under different policies are
    never confused for each other.

    ``price_tolerance = 0`` is valid and useful - it means exact Decimal
    equality only, which is the strictest possible reading and a good baseline
    for regression work.
    """

    price_tolerance: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.price_tolerance, Decimal):
            raise TypeError(
                "price_tolerance must be a Decimal; binary floats cannot express "
                "a price band exactly and would make pool membership depend on "
                "representation error"
            )
        if not self.price_tolerance.is_finite():
            raise ValueError(f"price_tolerance must be finite, got {self.price_tolerance}")
        if self.price_tolerance < 0:
            raise ValueError(f"price_tolerance must not be negative, got {self.price_tolerance}")

    @property
    def canonical_tolerance(self) -> str:
        """The tolerance as it appears in pool and event identities.

        Normalised so that ``Decimal("0.50")`` and ``Decimal("0.5")`` - the same
        policy written two ways - produce the same identity rather than two
        pools that differ only in trailing zeros.
        """
        normalised = self.price_tolerance.normalize()
        return format(normalised, "f")


# --------------------------------------------------------------------------
# pools
# --------------------------------------------------------------------------


class PoolStatus(StrEnum):
    """Where a pool is in its life.

    Three values, and the two terminal ones are named for what was *observed*.
    ``INVALID``, ``MITIGATED`` and ``BROKEN_STRUCTURE`` are absent because each
    imports a concept this round does not implement - validity against what,
    mitigated by which order, broken relative to which structural swing.
    """

    ACTIVE = "ACTIVE"
    SWEPT = "SWEPT"
    CLOSED_THROUGH = "CLOSED_THROUGH"


class LiquidityEventType(StrEnum):
    """What a closed candle did to a pool."""

    WICK_SWEEP = "WICK_SWEEP"
    CLOSE_THROUGH = "CLOSE_THROUGH"


@dataclass(frozen=True)
class LiquidityPool:
    """Two or more confirmed swings at effectively one price, and their band.

    A **band**, not a level. ``lower`` and ``upper`` are observed swing prices,
    never padded or rounded, and ``midpoint`` is offered for reference only -
    collapsing the pool to its average and treating that as the authority would
    invent a price no swing ever printed. Sweep detection uses the boundary the
    liquidity actually sits behind: ``upper`` for buy-side, ``lower`` for
    sell-side.
    """

    pool_id: str
    method_version: str

    timeframe: Timeframe
    symbol: str
    side: LiquiditySide

    formed_at: datetime
    """When the pool became knowable - the later confirmation of its two
    founding swings. Not a pivot time."""

    founding_swing_ids: tuple[str, str]
    member_swing_ids: tuple[str, ...]

    lower: Decimal
    upper: Decimal
    midpoint: Decimal
    tolerance: Decimal

    status: PoolStatus
    terminal_event_id: str | None

    @property
    def boundary(self) -> Decimal:
        """The edge liquidity rests behind, and the one events are tested against."""
        return self.upper if self.side is LiquiditySide.BUY_SIDE else self.lower

    @property
    def span(self) -> Decimal:
        return self.upper - self.lower

    @property
    def is_terminal(self) -> bool:
        return self.status is not PoolStatus.ACTIVE

    def is_touchable_at(self, close_time: datetime) -> bool:
        """Whether a bar closing at *close_time* may take this pool."""
        return touchable(self.status, self.formed_at, close_time)


def touchable(status: PoolStatus, formed_at: datetime, close_time: datetime) -> bool:
    """Whether a bar closing at *close_time* may take a pool in this state.

    Two conditions, and the second is the temporal one that matters:

    * the pool is still ``ACTIVE`` - a terminal pool is finished, and twenty
      later candles beyond the same band produce no further events;
    * it was formed **strictly before** this close. A bar may not confirm the
      second founding swing, create the pool, and sweep it, all at one close.

    As in Round 6.6b, seeing something at T and being allowed to act on it at T
    are separate permissions. Reported honestly: with strict-both-sides pivots
    the second condition cannot bind, because the bar closing at a pool's
    formation instant is always inside the founding swing's right-hand window,
    and a wick beyond the pool's boundary there would have destroyed that swing
    before it could found anything. It is enforced regardless, since that
    argument rests on a pivot definition a later round may change.
    """
    return status is PoolStatus.ACTIVE and formed_at < close_time


@dataclass(frozen=True)
class LiquidityEvent:
    """One closed candle, one pool, and exactly what was observed.

    Over-specified deliberately, like the structure events: the pool's band and
    membership are recorded *as they stood at the event*, so a later reader does
    not have to replay the analysis to know what was actually taken.
    """

    event_id: str
    method_version: str

    timeframe: Timeframe
    symbol: str
    pool_id: str
    side: LiquiditySide
    event_type: LiquidityEventType

    pool_lower: Decimal
    pool_upper: Decimal
    boundary: Decimal
    member_swing_ids_at_event: tuple[str, ...]

    event_bar_open_time: datetime
    event_bar_close_time: datetime
    wick_extreme: Decimal
    """The bar's high for a buy-side pool, its low for a sell-side one."""

    close: Decimal
    observed_at: datetime


def _digest(*parts: str) -> str:
    """Sixteen hex characters over the given fields joined by ``|``.

    No clock, no counter, no random source, and provider metadata never
    participates: the same candles from any source describe the same pool.
    """
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _pool_id(
    *,
    symbol: str,
    timeframe: Timeframe,
    side: LiquiditySide,
    tolerance: str,
    founding: tuple[str, str],
) -> str:
    """Identity from the founding observations and the policy that paired them.

    Founding swing ids are fixed at formation, so a pool keeps its identity when
    a third member joins later. Two swings at the same price but different times
    are different founders, so a fresh pool forming at the level of a terminal
    one is a genuinely different pool with a different id.
    """
    return _digest(
        symbol,
        timeframe.value,
        side.value,
        LIQUIDITY_METHOD_VERSION,
        tolerance,
        founding[0],
        founding[1],
    )


def _event_id(
    *,
    symbol: str,
    timeframe: Timeframe,
    pool_id: str,
    event_type: LiquidityEventType,
    close_time: datetime,
) -> str:
    return _digest(
        symbol,
        timeframe.value,
        LIQUIDITY_METHOD_VERSION,
        pool_id,
        event_type.value,
        close_time.isoformat(),
    )


@dataclass
class _WorkingPool:
    """A pool while the replay is still running. Frozen into a
    :class:`LiquidityPool` at the end."""

    pool_id: str
    side: LiquiditySide
    formed_at: datetime
    founding: tuple[str, str]
    members: list[tuple[str, Decimal]] = field(default_factory=list)
    status: PoolStatus = PoolStatus.ACTIVE
    terminal_event_id: str | None = None

    @property
    def lower(self) -> Decimal:
        return min(price for _, price in self.members)

    @property
    def upper(self) -> Decimal:
        return max(price for _, price in self.members)

    @property
    def boundary(self) -> Decimal:
        return self.upper if self.side is LiquiditySide.BUY_SIDE else self.lower

    def span_with(self, price: Decimal) -> Decimal:
        """The full span this pool would have if *price* joined it."""
        prices = [existing for _, existing in self.members] + [price]
        return max(prices) - min(prices)

    def freeze(self, *, timeframe: Timeframe, symbol: str, tolerance: Decimal) -> LiquidityPool:
        lower, upper = self.lower, self.upper
        return LiquidityPool(
            pool_id=self.pool_id,
            method_version=LIQUIDITY_METHOD_VERSION,
            timeframe=timeframe,
            symbol=symbol,
            side=self.side,
            formed_at=self.formed_at,
            founding_swing_ids=self.founding,
            member_swing_ids=tuple(identity for identity, _ in self.members),
            lower=lower,
            upper=upper,
            # Exact halving of a Decimal sum. No float, and no rounding: with
            # two-decimal prices the midpoint may legitimately carry a third
            # decimal, and truncating it would move a reported number.
            midpoint=(lower + upper) / Decimal(2),
            tolerance=tolerance,
            status=self.status,
            terminal_event_id=self.terminal_event_id,
        )


# --------------------------------------------------------------------------
# the analysis
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LiquidityAnalysis:
    """One timeframe's liquidity, as of one instant.

    The smallest thing later rounds need. No candidate zones, no ranking, no
    direction, and no combination with any other timeframe.
    """

    method_version: str
    timeframe: Timeframe
    symbol: str
    observed_at: datetime
    tolerance: Decimal

    pools: tuple[LiquidityPool, ...]
    events: tuple[LiquidityEvent, ...]
    active_pool_ids: tuple[str, ...]
    terminal_pool_ids: tuple[str, ...]
    unpaired_swing_ids: tuple[str, ...]

    left_bars: int
    right_bars: int
    bars_considered: int

    def pool(self, pool_id: str) -> LiquidityPool | None:
        return next((entry for entry in self.pools if entry.pool_id == pool_id), None)

    def pools_on(self, side: LiquiditySide) -> tuple[LiquidityPool, ...]:
        return tuple(entry for entry in self.pools if entry.side is side)

    def events_of(self, event_type: LiquidityEventType) -> tuple[LiquidityEvent, ...]:
        return tuple(entry for entry in self.events if entry.event_type is event_type)


def analyse_liquidity(
    series: IctTimeframeSnapshot,
    *,
    config: LiquidityConfig,
    symbol: str = "",
    as_of: datetime | None = None,
    left_bars: int = DEFAULT_LEFT_BARS,
    right_bars: int = DEFAULT_RIGHT_BARS,
) -> LiquidityAnalysis:
    """Replay *series* bar by bar and report the liquidity it built and lost.

    The exact order of operations at each closed bar, because the answers depend
    on it:

    1. Test the bar against every pool that was ``ACTIVE`` and formed
       **strictly before** this close. A close beyond the boundary is a
       ``CLOSE_THROUGH``; otherwise a wick strictly beyond it that closed back
       inside is a ``WICK_SWEEP``; otherwise nothing.
    2. Terminalise every pool that took an event. Each pool may take at most
       one, ever.
    3. Only now, take the swings this bar's close confirmed.
    4. Let each of them join an active pool, or pair with a waiting unpaired
       swing to form a new one. Pools created here are dated at this close and
       so cannot be touched until the next bar.
    5. Report what is knowable at this instant.

    Steps 1 and 3 are in that order for a specific reason: otherwise a swing
    confirmed by this very bar could widen a pool's band and the same bar would
    then be judged against the band it had just moved.

    Args:
        series: Closed candles for one timeframe.
        config: The tolerance policy. Required - see :class:`LiquidityConfig`.
        symbol: Instrument name. Provenance, and part of every identity.
        as_of: Reconstruct the state as it stood at this instant.
        left_bars: Pivot window, passed through to the one swing authority.
        right_bars: The same on the right, and the confirmation delay.

    Raises:
        ValueError: *as_of* precedes every bar's close, so there is no series to
            analyse.
    """
    working = series
    if as_of is not None:
        working = build_timeframe_snapshot(
            timeframe=series.timeframe, bars=series.bars, observed_at=as_of
        )
    observed_at = as_of if as_of is not None else series.latest_closed_at

    duration = working.timeframe.duration
    assert duration is not None  # IctTimeframeSnapshot refuses calendar timeframes

    swings = confirmed_swings(working, left_bars=left_bars, right_bars=right_bars)
    by_confirmation: dict[datetime, list[SwingPoint]] = {}
    for swing in swings:
        by_confirmation.setdefault(swing.confirmed_at, []).append(swing)

    pools: list[_WorkingPool] = []
    unpaired: list[SwingPoint] = []
    events: list[LiquidityEvent] = []
    tolerance = config.price_tolerance

    for bar in working.bars:
        close_time = bar.timestamp + duration

        # (1) and (2): pools known strictly before this close.
        for event in _events_for_bar(
            bar,
            close_time,
            pools,
            timeframe=working.timeframe,
            symbol=symbol,
            observed_at=observed_at,
        ):
            events.append(event)
            struck = next(entry for entry in pools if entry.pool_id == event.pool_id)
            struck.status = (
                PoolStatus.CLOSED_THROUGH
                if event.event_type is LiquidityEventType.CLOSE_THROUGH
                else PoolStatus.SWEPT
            )
            struck.terminal_event_id = event.event_id

        # (3) and (4): swings this close confirmed, in chronological order.
        for swing in by_confirmation.get(close_time, ()):
            _admit(
                swing,
                pools=pools,
                unpaired=unpaired,
                tolerance=tolerance,
                timeframe=working.timeframe,
                symbol=symbol,
                canonical_tolerance=config.canonical_tolerance,
                confirmed_at=close_time,
            )

    frozen = tuple(
        entry.freeze(timeframe=working.timeframe, symbol=symbol, tolerance=tolerance)
        for entry in pools
    )
    return LiquidityAnalysis(
        method_version=LIQUIDITY_METHOD_VERSION,
        timeframe=working.timeframe,
        symbol=symbol,
        observed_at=observed_at,
        tolerance=tolerance,
        pools=frozen,
        events=tuple(events),
        active_pool_ids=tuple(entry.pool_id for entry in frozen if not entry.is_terminal),
        terminal_pool_ids=tuple(entry.pool_id for entry in frozen if entry.is_terminal),
        unpaired_swing_ids=tuple(swing_id(swing) for swing in unpaired),
        left_bars=left_bars,
        right_bars=right_bars,
        bars_considered=working.bar_count,
    )


def analyse_snapshot_liquidity(
    snapshot: IctMarketSnapshot,
    timeframe: Timeframe,
    *,
    config: LiquidityConfig,
    as_of: datetime | None = None,
    left_bars: int = DEFAULT_LEFT_BARS,
    right_bars: int = DEFAULT_RIGHT_BARS,
) -> LiquidityAnalysis:
    """:func:`analyse_liquidity` for one timeframe of a multi-timeframe snapshot.

    One timeframe, named explicitly. There is deliberately no call that merges
    an H4 pool with an H1 pool into a "stronger" zone: confluence is a scoring
    rule with real content, and it belongs to the candidate engine where it can
    be argued with rather than to a geometry module where it would be assumed.
    """
    return analyse_liquidity(
        snapshot.require(timeframe),
        config=config,
        symbol=snapshot.symbol,
        as_of=as_of,
        left_bars=left_bars,
        right_bars=right_bars,
    )


def _events_for_bar(
    bar: OHLCBar,
    close_time: datetime,
    pools: Sequence[_WorkingPool],
    *,
    timeframe: Timeframe,
    symbol: str,
    observed_at: datetime,
) -> list[LiquidityEvent]:
    """Every pool this bar terminated, in a stable order.

    Unlike the structure engine, one bar may clear several pools. That rule is
    right there and wrong here: structure asks "what did this close say about
    the trend", which has one answer, while liquidity asks "what did this candle
    take", and a single wide bar can genuinely run several bands.

    Events are ordered by ``(side, boundary, formed_at, pool_id)`` - buy-side
    before sell-side, then upward through the bands. The specific order matters
    less than that it is total and derived only from the data, so no set or dict
    iteration can reach it.
    """
    struck: list[tuple[tuple[str, Decimal, datetime, str], LiquidityEvent]] = []

    for pool in pools:
        if not touchable(pool.status, pool.formed_at, close_time):
            continue

        boundary = pool.boundary
        if pool.side is LiquiditySide.BUY_SIDE:
            wick_extreme = bar.high
            if bar.close > boundary:
                event_type = LiquidityEventType.CLOSE_THROUGH
            elif bar.high > boundary:
                event_type = LiquidityEventType.WICK_SWEEP
            else:
                continue
        else:
            wick_extreme = bar.low
            if bar.close < boundary:
                event_type = LiquidityEventType.CLOSE_THROUGH
            elif bar.low < boundary:
                event_type = LiquidityEventType.WICK_SWEEP
            else:
                continue

        event = LiquidityEvent(
            event_id=_event_id(
                symbol=symbol,
                timeframe=timeframe,
                pool_id=pool.pool_id,
                event_type=event_type,
                close_time=close_time,
            ),
            method_version=LIQUIDITY_METHOD_VERSION,
            timeframe=timeframe,
            symbol=symbol,
            pool_id=pool.pool_id,
            side=pool.side,
            event_type=event_type,
            pool_lower=pool.lower,
            pool_upper=pool.upper,
            boundary=boundary,
            member_swing_ids_at_event=tuple(identity for identity, _ in pool.members),
            event_bar_open_time=bar.timestamp,
            event_bar_close_time=close_time,
            wick_extreme=wick_extreme,
            close=bar.close,
            observed_at=observed_at,
        )
        struck.append(((pool.side.value, boundary, pool.formed_at, pool.pool_id), event))

    struck.sort(key=lambda item: item[0])
    return [event for _, event in struck]


def _admit(
    swing: SwingPoint,
    *,
    pools: list[_WorkingPool],
    unpaired: list[SwingPoint],
    tolerance: Decimal,
    timeframe: Timeframe,
    symbol: str,
    canonical_tolerance: str,
    confirmed_at: datetime,
) -> None:
    """Place one newly confirmed swing: join a pool, found one, or wait.

    Tried in that order, and a swing that joins anything is removed from the
    waiting list, which is what makes "at most one pool per swing" true by
    construction rather than by a later check.
    """
    side = side_of(swing.swing_type)
    identity = swing_id(swing)

    joined = _best_pool_for(swing, side, pools, tolerance)
    if joined is not None:
        joined.members.append((identity, swing.price))
        return

    partner = _best_partner_for(swing, side, unpaired, tolerance)
    if partner is None:
        unpaired.append(swing)
        return

    unpaired.remove(partner)
    founding = (swing_id(partner), identity)
    pool = _WorkingPool(
        pool_id=_pool_id(
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            tolerance=canonical_tolerance,
            founding=founding,
        ),
        side=side,
        formed_at=confirmed_at,
        founding=founding,
        members=[(founding[0], partner.price), (identity, swing.price)],
    )

    # Anything else still waiting on this side that fits inside the *full* span
    # was always part of this liquidity; it was simply waiting for a partner.
    # Chronological, so the result never depends on list order beyond time.
    for candidate in list(unpaired):
        if side_of(candidate.swing_type) is not side:
            continue
        if pool.span_with(candidate.price) <= tolerance:
            unpaired.remove(candidate)
            pool.members.append((swing_id(candidate), candidate.price))

    pool.members.sort(key=lambda member: member[0])
    pools.append(pool)


def _eligible_pools(
    price: Decimal,
    side: LiquiditySide,
    pools: Iterable[_WorkingPool],
    tolerance: Decimal,
) -> list[_WorkingPool]:
    """Active same-side pools that could take a swing at *price*.

    **This list can never hold more than one entry**, and the proof is worth
    writing down because it is what makes the tie-break below dead code rather
    than a coin toss nobody can audit. For buy-side pools A and B, both active
    when a swing at price P is admitted, both accepting it:

    * P must not exceed either upper bound, or the bar printing that high would
      already have swept that pool - a swing high's price *is* its bar's high;
    * so with ``upperA <= upperB``, we have ``upperB - upperA <= upperB - P <=
      tolerance``, meaning the swing that set ``upperA`` was itself within
      tolerance of B's span and would have joined B rather than founding or
      joining A - unless B did not yet exist;
    * if B formed later, its founder at ``upperB > upperA`` printed a high above
      A's boundary and swept A, so A is not active;
    * and if ``upperB == upperA``, that founder was within A's span and would
      have joined A instead of founding B.

    Every branch contradicts the premise. Sell-side mirrors it. The ordering in
    :func:`_best_pool_for` is therefore defensive: it exists so that a later
    round loosening any of those conditions gets a defined answer rather than a
    silent dependency on list order.
    """
    return [
        pool
        for pool in pools
        if pool.side is side
        and pool.status is PoolStatus.ACTIVE
        and pool.span_with(price) <= tolerance
    ]


def _best_pool_for(
    swing: SwingPoint,
    side: LiquiditySide,
    pools: Iterable[_WorkingPool],
    tolerance: Decimal,
) -> _WorkingPool | None:
    """The active pool this swing should join, or ``None``.

    A terminal pool never accepts a member: once a band has been swept or closed
    through, later swings at that price are new liquidity, not a continuation of
    what was already taken.

    When more than one pool would accept the swing, the tie is broken by
    smallest resulting full span, then by distance to the pool's current
    midpoint, then by the most recently formed pool, then by pool id. Every term
    is derived from the data, so the answer cannot depend on the order pools
    happen to sit in the list.
    """
    eligible = _eligible_pools(swing.price, side, pools, tolerance)
    if not eligible:
        return None

    def closeness(pool: _WorkingPool) -> tuple[Decimal, Decimal]:
        midpoint = (pool.lower + pool.upper) / Decimal(2)
        return (pool.span_with(swing.price), abs(swing.price - midpoint))

    # Least significant key first, relying on a stable sort. Written this way
    # rather than as one tuple because the formation key runs *descending* and
    # inverting a datetime would mean turning it into a number - which on this
    # branch means a float, and there are no floats in this module.
    eligible.sort(key=lambda pool: pool.pool_id)
    eligible.sort(key=lambda pool: pool.formed_at, reverse=True)
    eligible.sort(key=closeness)
    return eligible[0]


def _best_partner_for(
    swing: SwingPoint,
    side: LiquiditySide,
    unpaired: Sequence[SwingPoint],
    tolerance: Decimal,
) -> SwingPoint | None:
    """The waiting swing this one should pair with, or ``None``.

    Partners need not be *consecutive* same-side swings. A high at 4000, an
    unrelated high at 4050, then a high at 4000.20 forms a pool from the first
    and third: the 4050 in between says nothing about whether orders are resting
    at 4000, and requiring adjacency would discard the pool for no reason.

    Closest price wins, then the earliest waiting swing - older untouched levels
    pair first - then the swing id.
    """
    candidates = [
        other
        for other in unpaired
        if side_of(other.swing_type) is side and abs(other.price - swing.price) <= tolerance
    ]
    if not candidates:
        return None

    return min(
        candidates,
        key=lambda other: (
            abs(other.price - swing.price),
            other.confirmed_at,
            swing_id(other),
        ),
    )


__all__ = [
    "LIQUIDITY_METHOD_VERSION",
    "LiquidityAnalysis",
    "LiquidityConfig",
    "LiquidityEvent",
    "LiquidityEventType",
    "LiquidityPool",
    "LiquiditySide",
    "PoolStatus",
    "analyse_liquidity",
    "analyse_snapshot_liquidity",
    "side_of",
    "touchable",
]
