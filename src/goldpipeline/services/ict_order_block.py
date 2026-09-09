"""The candle a structural event left behind it, and the zone that candle draws.

Round 6.6d.1. Formation only - what an order block *is*, and where it came
from. What later happens to one is Round 6.6d.2's question, and nothing here
knows the words for it.

**An order block is event-scoped, not "the last opposite candle somewhere".**
The common phrasing - "the last down candle before the move up" - has no
boundaries, so two people applying it to the same chart pick different candles
and both are right. Here the boundaries come from structure that already exists:
a close-confirmed break, the swing it protected, and the leg it drew. The source
candle is the most recent opposite-bodied closed candle *inside that leg*. If
there is no leg there is no order block, and if the leg holds no opposite candle
there is no order block. Neither absence is repaired.

**Body direction is arithmetic, not appearance.** ``close < open`` is bearish,
``close > open`` is bullish, ``close == open`` is neither. No colour metadata,
no provider flag, no comparison against the previous close, and no tolerance
band that would make a one-cent body "basically a doji". A doji is not an
opposite candle and cannot be a source; the search simply steps back past it.

**Recency inside the leg is the whole selection rule.** No score, no ATR
threshold, no "strongest candle", no preference for the one nearest some price.
The leg has already answered "which move are we talking about?", so the only
remaining question is which of its opposite candles came last - and that has
one answer for everybody.

**Zone basis is a policy, not a fact.** ICT material disagrees about whether an
order block runs wick-to-wick or body-only, and both readings are defensible.
Pretending one is universal would bury a choice inside geometry, so
:class:`OrderBlockZoneBasis` makes the caller state it and
:class:`OrderBlockConfig` has no default. The two policies produce different
zones and therefore different identities, which is correct: they are different
claims about the same candle.

**Formation is dated by the event, not by the candle.** The source candle may
be twenty bars old. It was not an order block then - it was a candle. It becomes
one when the break confirms, so ``formed_at`` is the break bar's close, and a
historical as-of query before that break reports nothing.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import (
    IctMarketSnapshot,
    IctTimeframeSnapshot,
    build_timeframe_snapshot,
)
from goldpipeline.schemas.market import OHLCBar
from goldpipeline.services.ict_primitives import DEFAULT_LEFT_BARS, DEFAULT_RIGHT_BARS
from goldpipeline.services.ict_protected import (
    ProtectedStructureAnalysis,
    ProtectedSwingAssignment,
    StructuralLeg,
    analyse_protected_structure,
)
from goldpipeline.services.ict_structure import (
    BreakClassification,
    BreakDirection,
    StructureAnalysis,
    StructureBreak,
    analyse_structure,
)

logger = logging.getLogger(__name__)

ORDER_BLOCK_METHOD_VERSION = "1.0.0"
"""Stamped on every order block and analysis, and part of every identity."""


class BodyDirection(StrEnum):
    """Which way a candle's body points.

    Three values, because a candle really can point neither way. Collapsing
    ``NEUTRAL`` into one of the other two would make an arbitrary tie-break look
    like an observation.
    """

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class OrderBlockZoneBasis(StrEnum):
    """How a source candle's zone is drawn.

    Exactly two, both traditional and both defensible. ``OPEN_TO_LOW``,
    ``FIFTY_PERCENT``, ``REFINED`` and the rest are absent because each is a
    product decision about entry precision, and this module reports what the
    candle was rather than how tightly somebody wants to trade it.
    """

    FULL_CANDLE = "FULL_CANDLE"
    """Wick to wick: ``low``..``high``, exactly as the candle printed."""

    BODY = "BODY"
    """Body only: ``min(open, close)``..``max(open, close)``."""


@dataclass(frozen=True)
class OrderBlockConfig:
    """The policy the caller chose.

    ``zone_basis`` has no default on purpose. A default here would silently make
    one tradition the project's answer, and the round that later wanted the
    other would find the choice already embedded in stored identities.
    """

    zone_basis: OrderBlockZoneBasis


class OrderBlockError(ValueError):
    """The engine could not honestly answer, so it refused to guess."""


def body_direction(bar: OHLCBar) -> BodyDirection:
    """Which way *bar*'s body points, by exact Decimal comparison.

    The only definition used anywhere in this module. Not the wick, not the
    colour a provider rendered, and not the close against the previous close -
    that last one is a different measurement entirely and would make the
    direction of a candle depend on its neighbour.
    """
    if bar.close < bar.open:
        return BodyDirection.BEARISH
    if bar.close > bar.open:
        return BodyDirection.BULLISH
    return BodyDirection.NEUTRAL


def source_body_for(direction: BreakDirection) -> BodyDirection:
    """The body a leg in *direction* is looking for.

    Opposite by definition: a bullish leg expanded away from a down candle, a
    bearish leg away from an up one. For an MSS *direction* is already the new
    direction, so a bearish MSS out of a bullish state looks for a bullish
    candle - the search is never run against the prior bias.
    """
    return BodyDirection.BEARISH if direction is BreakDirection.BULLISH else BodyDirection.BULLISH


def zone_of(bar: OHLCBar, basis: OrderBlockZoneBasis) -> tuple[Decimal, Decimal]:
    """The ``(lower, upper)`` *basis* draws on *bar*.

    No padding, no tolerance, no ATR and no rounding. Both readings are exact
    OHLC arithmetic, which is what lets the same candle from any provider draw
    the same zone.
    """
    if basis is OrderBlockZoneBasis.FULL_CANDLE:
        return bar.low, bar.high
    return min(bar.open, bar.close), max(bar.open, bar.close)


@dataclass(frozen=True)
class OrderBlock:
    """One structural event's source candle, and the zone one policy draws on it.

    No lifecycle. There is no ``status``, no ``mitigated``, no ``touch_count``
    and no ``is_valid``, because every one of those needs a rule this round has
    not written. What is here is a formation record: which event established it,
    which candle it is, and what that candle's geometry is under the chosen
    basis.
    """

    order_block_id: str
    method_version: str

    timeframe: Timeframe
    symbol: str

    direction: BreakDirection
    """The structural direction, reused. Not BUY, SELL, SEO or BAI."""

    zone_basis: OrderBlockZoneBasis

    structure_event_id: str
    event_classification: BreakClassification

    protected_assignment_id: str
    structural_leg_id: str

    formed_at: datetime
    """The break bar's close. Before this instant the candle was just a candle."""

    source_bar_open_time: datetime
    source_bar_close_time: datetime

    source_open: Decimal
    source_high: Decimal
    source_low: Decimal
    source_close: Decimal
    """The candle exactly as it printed.

    All four are kept so a later round can inspect the body range and the full
    candle range without going back to provider data - but only one pair is
    canonical, and it is ``lower``/``upper`` below.
    """

    source_body_direction: BodyDirection

    lower: Decimal
    upper: Decimal
    midpoint: Decimal
    """``(lower + upper) / 2``, exact. Geometric metadata and nothing more.

    Deliberately not called consequent encroachment or CE: those name a way of
    *using* the level, and this round defines no use.
    """

    width: Decimal

    @property
    def body_bounds(self) -> tuple[Decimal, Decimal]:
        """The body reading of the source candle, whatever the chosen basis."""
        return min(self.source_open, self.source_close), max(self.source_open, self.source_close)

    @property
    def full_candle_bounds(self) -> tuple[Decimal, Decimal]:
        """The wick-to-wick reading of the source candle."""
        return self.source_low, self.source_high


@dataclass(frozen=True)
class OrderBlockAnalysis:
    """One timeframe's order blocks under one policy, as of one instant."""

    method_version: str
    timeframe: Timeframe
    symbol: str
    observed_at: datetime
    zone_basis: OrderBlockZoneBasis

    order_blocks: tuple[OrderBlock, ...]

    def order_block(self, identity: str) -> OrderBlock | None:
        return next(
            (entry for entry in self.order_blocks if entry.order_block_id == identity), None
        )

    def for_event(self, event_id: str) -> OrderBlock | None:
        """The order block one structural event established, if it established one."""
        return next(
            (entry for entry in self.order_blocks if entry.structure_event_id == event_id), None
        )


def _order_block_id(
    *,
    symbol: str,
    timeframe: Timeframe,
    event_id: str,
    leg_id: str,
    source_open_time: datetime,
    basis: OrderBlockZoneBasis,
) -> str:
    """Identity from immutable facts, none of which is a price.

    The event and leg identities already encode the symbol, timeframe, anchor
    and classification; the source open time names the candle; the basis names
    the policy, so ``FULL_CANDLE`` and ``BODY`` readings of one candle are two
    identities rather than one that quietly changed shape. No price enters the
    preimage, which is why this round needs no fourth copy of the Decimal
    canonicalisation the liquidity and gap identities require.
    """
    preimage = "|".join(
        (
            symbol,
            timeframe.value,
            ORDER_BLOCK_METHOD_VERSION,
            event_id,
            leg_id,
            source_open_time.isoformat(),
            basis.value,
        )
    )
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]


def analyse_order_blocks(
    series: IctTimeframeSnapshot,
    *,
    config: OrderBlockConfig,
    structure: StructureAnalysis | None = None,
    protected: ProtectedStructureAnalysis | None = None,
    symbol: str = "",
    as_of: datetime | None = None,
    left_bars: int = DEFAULT_LEFT_BARS,
    right_bars: int = DEFAULT_RIGHT_BARS,
) -> OrderBlockAnalysis:
    """Walk the structural legs and report the source candle each one contains.

    Per leg, in this order:

    1. Resolve the event and assignment the leg belongs to, refusing any
       mismatch rather than pairing an event with somebody else's leg.
    2. Resolve the leg window: from the protected swing's own pivot candle up to
       but not including the break bar.
    3. Take the latest closed candle in that window whose body points opposite
       the leg. Dojis are skipped, not counted.
    4. If there is none, this event has no order block. That is a result.

    Events with no assignment and no leg never reach step 2 - there is nothing
    to scan - so an anchorless break produces nothing, with no fallback to the
    break bar, the event-bar extreme or an N-bar candle.

    Args:
        series: Closed candles for one timeframe.
        config: The zone-basis policy. Required; there is no default.
        structure: A structure analysis for this same series and instant.
        protected: A protected-structure analysis for the same. Both are
            optional; supply them when a composite stage already has them, and
            their instant, timeframe and symbol must match or they are refused.
        symbol: Instrument name. Provenance, and part of every identity.
        as_of: Reconstruct the state as it stood at this instant.
        left_bars: Pivot window, passed through to the structure engine.
        right_bars: The same on the right.

    Raises:
        OrderBlockError: A supplied analysis describes a different instant,
            symbol or timeframe; a leg names an event that is not present, or
            an event/assignment pair that disagrees; a leg's origin pivot or
            break bar is missing from the series; or a source candle would draw
            a zone with no width.
        ValueError: *as_of* precedes every bar's close.
    """
    working = series
    if as_of is not None:
        working = build_timeframe_snapshot(
            timeframe=series.timeframe, bars=series.bars, observed_at=as_of
        )
    observed_at = as_of if as_of is not None else series.latest_closed_at
    timeframe = working.timeframe

    duration = timeframe.duration
    assert duration is not None  # IctTimeframeSnapshot refuses calendar timeframes

    if structure is None:
        structure = analyse_structure(
            series, symbol=symbol, as_of=as_of, left_bars=left_bars, right_bars=right_bars
        )
    _require_same_moment(
        "structure",
        structure.observed_at,
        structure.timeframe,
        structure.symbol,
        observed_at,
        timeframe,
        symbol,
    )

    if protected is None:
        protected = analyse_protected_structure(
            series,
            structure=structure,
            symbol=symbol,
            as_of=as_of,
            left_bars=left_bars,
            right_bars=right_bars,
        )
    _require_same_moment(
        "protected structure",
        protected.observed_at,
        protected.timeframe,
        protected.symbol,
        observed_at,
        timeframe,
        symbol,
    )

    events_by_id = {event.event_id: event for event in structure.breaks}

    found: list[OrderBlock] = []
    for leg in protected.legs:
        event = events_by_id.get(leg.event_id)
        if event is None:
            raise OrderBlockError(
                f"leg {leg.leg_id} names event {leg.event_id}, which this structure does not hold"
            )
        assignment = protected.assignment(leg.protected_assignment_id)
        if assignment is None:
            raise OrderBlockError(
                f"leg {leg.leg_id} names assignment {leg.protected_assignment_id}, "
                "which this analysis does not hold"
            )
        _require_consistent(event, assignment, leg)

        block = _order_block_for(
            leg,
            event=event,
            bars=working.bars,
            duration=duration,
            basis=config.zone_basis,
            symbol=symbol,
            timeframe=timeframe,
        )
        if block is not None:
            found.append(block)

    # Already chronological, because legs follow events. Sorted anyway so the
    # total order is a property of the output rather than of the loop that
    # happened to build it.
    found.sort(key=lambda entry: (entry.formed_at, entry.structure_event_id, entry.order_block_id))

    return OrderBlockAnalysis(
        method_version=ORDER_BLOCK_METHOD_VERSION,
        timeframe=timeframe,
        symbol=symbol,
        observed_at=observed_at,
        zone_basis=config.zone_basis,
        order_blocks=tuple(found),
    )


def analyse_snapshot_order_blocks(
    snapshot: IctMarketSnapshot,
    timeframe: Timeframe,
    *,
    config: OrderBlockConfig,
    structure: StructureAnalysis | None = None,
    protected: ProtectedStructureAnalysis | None = None,
    as_of: datetime | None = None,
    left_bars: int = DEFAULT_LEFT_BARS,
    right_bars: int = DEFAULT_RIGHT_BARS,
) -> OrderBlockAnalysis:
    """:func:`analyse_order_blocks` for one timeframe of a snapshot.

    One timeframe, named explicitly. An H4 order block does not replace an M15
    one: which timeframe's zone a plan should use is a synthesis rule with real
    content, and it is not decided by a geometry module.
    """
    return analyse_order_blocks(
        snapshot.require(timeframe),
        config=config,
        structure=structure,
        protected=protected,
        symbol=snapshot.symbol,
        as_of=as_of,
        left_bars=left_bars,
        right_bars=right_bars,
    )


def _require_same_moment(
    what: str,
    given_at: datetime,
    given_timeframe: Timeframe,
    given_symbol: str,
    observed_at: datetime,
    timeframe: Timeframe,
    symbol: str,
) -> None:
    """A supplied analysis must describe this exact request.

    Refused rather than trimmed. An analysis carries current references as well
    as history, and quietly cutting its event list would leave those describing
    another moment - a subtler leak than the one being prevented. Same reasoning
    as Rounds 6.6c.3a and 6.6c.3b.
    """
    if given_at != observed_at:
        raise OrderBlockError(
            f"{what} describes {given_at.isoformat()} but this analysis is as of "
            f"{observed_at.isoformat()}; supply one computed for the same instant"
        )
    if given_timeframe is not timeframe:
        raise OrderBlockError(f"{what} is for {given_timeframe.value}, not {timeframe.value}")
    if given_symbol != symbol:
        raise OrderBlockError(f"{what} is for {given_symbol!r}, not {symbol!r}")


def _require_consistent(
    event: StructureBreak, assignment: ProtectedSwingAssignment, leg: StructuralLeg
) -> None:
    """Event, assignment and leg must be three views of one thing.

    Checked rather than assumed, because the analyses are accepted from callers.
    Pairing event E1 with leg L2 would produce an order block that looks entirely
    normal and describes a move that never happened.
    """
    if assignment.established_by_event_id != event.event_id:
        raise OrderBlockError(
            f"assignment {assignment.assignment_id} was established by "
            f"{assignment.established_by_event_id}, not by {event.event_id}"
        )
    if leg.direction is not event.direction:
        raise OrderBlockError(
            f"leg {leg.leg_id} is {leg.direction.value} but event {event.event_id} is "
            f"{event.direction.value}"
        )
    if leg.origin_swing_id != assignment.swing_id:
        raise OrderBlockError(
            f"leg {leg.leg_id} originates at {leg.origin_swing_id} but its assignment names "
            f"{assignment.swing_id}"
        )
    if leg.terminal_bar_close_time != event.break_bar_close_time:
        raise OrderBlockError(
            f"leg {leg.leg_id} ends at {leg.terminal_bar_close_time.isoformat()} but event "
            f"{event.event_id} closed at {event.break_bar_close_time.isoformat()}"
        )


def _order_block_for(
    leg: StructuralLeg,
    *,
    event: StructureBreak,
    bars: tuple[OHLCBar, ...],
    duration: timedelta,
    basis: OrderBlockZoneBasis,
    symbol: str,
    timeframe: Timeframe,
) -> OrderBlock | None:
    """The order block this leg established, or ``None`` if it holds no candle."""
    source = _source_candle(leg, bars=bars)
    if source is None:
        return None

    lower, upper = zone_of(source, basis)
    if upper <= lower:
        # Unreachable as written - a candle with no body is NEUTRAL and never
        # eligible, and a candle with no range has no body either - but a zone
        # of zero width would be a level pretending to be a zone, so it is
        # refused rather than published.
        raise OrderBlockError(
            f"source candle at {source.timestamp.isoformat()} draws no {basis.value} zone: "
            f"{lower}..{upper}"
        )

    return OrderBlock(
        order_block_id=_order_block_id(
            symbol=symbol,
            timeframe=timeframe,
            event_id=leg.event_id,
            leg_id=leg.leg_id,
            source_open_time=source.timestamp,
            basis=basis,
        ),
        method_version=ORDER_BLOCK_METHOD_VERSION,
        timeframe=timeframe,
        symbol=symbol,
        direction=leg.direction,
        zone_basis=basis,
        structure_event_id=leg.event_id,
        event_classification=event.classification,
        protected_assignment_id=leg.protected_assignment_id,
        structural_leg_id=leg.leg_id,
        formed_at=leg.formed_at,
        source_bar_open_time=source.timestamp,
        source_bar_close_time=source.timestamp + duration,
        source_open=source.open,
        source_high=source.high,
        source_low=source.low,
        source_close=source.close,
        source_body_direction=body_direction(source),
        lower=lower,
        upper=upper,
        midpoint=(lower + upper) / Decimal(2),
        width=upper - lower,
    )


def _source_candle(leg: StructuralLeg, *, bars: tuple[OHLCBar, ...]) -> OHLCBar | None:
    """The latest opposite-bodied closed candle inside *leg*'s window.

    The window runs from the protected swing's own pivot candle - which is
    inside the leg and therefore eligible - up to but not including the break
    bar. The break bar is the event, not its origin, and including it would make
    every order block the candle that broke the level.

    Both boundary bars are resolved by exact timestamp. A missing one is refused
    rather than approximated: the nearest-time fallback would silently widen the
    scan and change which candle is selected, and it would do so invisibly.
    """
    by_open = {bar.timestamp: bar for bar in bars}
    if leg.origin_pivot_time not in by_open:
        raise OrderBlockError(
            f"leg {leg.leg_id} originates at a pivot opening "
            f"{leg.origin_pivot_time.isoformat()}, which is not in this series"
        )
    if leg.terminal_bar_open_time not in by_open:
        raise OrderBlockError(
            f"leg {leg.leg_id} names a break bar opening "
            f"{leg.terminal_bar_open_time.isoformat()}, which is not in this series"
        )

    wanted = source_body_for(leg.direction)
    found: OHLCBar | None = None
    for bar in bars:
        if bar.timestamp < leg.origin_pivot_time:
            continue
        if bar.timestamp >= leg.terminal_bar_open_time:
            break
        if body_direction(bar) is wanted:
            found = bar
    return found


__all__ = [
    "ORDER_BLOCK_METHOD_VERSION",
    "BodyDirection",
    "OrderBlock",
    "OrderBlockAnalysis",
    "OrderBlockConfig",
    "OrderBlockError",
    "OrderBlockZoneBasis",
    "analyse_order_blocks",
    "analyse_snapshot_order_blocks",
    "body_direction",
    "source_body_for",
    "zone_of",
]
