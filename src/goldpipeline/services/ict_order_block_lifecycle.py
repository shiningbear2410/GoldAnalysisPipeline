"""What later price did to an order block, and which parts of that are policy.

Round 6.6d.2. Formation is Round 6.6d.1's and is not revisited here: this module
consumes finished :class:`~goldpipeline.services.ict_order_block.OrderBlock`
objects, never chooses a source candle, and never moves, refines, widens,
narrows or merges a zone.

**Three questions, and they are not the same kind of question.**

*Touch* is a geometric observation. Two intervals either intersect or they do
not, and no convention is involved.

*Invalidation* is a rule this round locks: a close strictly beyond the far edge.
It is a decision, but it is the branch's existing decision - every structural
break on this branch is close-confirmed, and letting a wick retire an order
block would leave the two halves of the same engine disagreeing about what
counts as evidence.

*Mitigation* is neither. Real ICT practice uses the word for at least three
different things - first touch, the midpoint being traded, the whole zone being
covered - and all three are defensible. So :class:`OrderBlockMitigationRule`
makes the caller name one and :class:`OrderBlockLifecycleConfig` has no default.
A module that quietly picked one would answer a live disagreement by omission,
and every consumer downstream would inherit the answer as though it were
geometry.

**The zone is a closed interval, and that differs from a fair value gap on
purpose.** :func:`~goldpipeline.services.ict_fvg.enters` is strict on both
sides, because a gap's edges are the boundary of an *absence* - a candle resting
exactly on the edge never entered the empty space. An order block's edges are
the source candle's own printed prices; a later candle reaching exactly the
upper edge has traded at a price that candle traded at. That is contact, so
``[lower, upper]`` is closed here and ``(lower, upper)`` is open there. The two
engines are not inconsistent - they are describing different objects.

**Status is derived, never assembled.** Every state carries the evidence -
first touch, first mitigation, first invalidation, each with its own witness
candle - and :func:`status_of` reads that evidence. There is no transition graph
to get out of step with the record, and a state can therefore never claim
something its witnesses do not support.

**Facts, not usage.** Nothing here decides whether a touched block is still
worth trading, whether a mitigated one should be discarded, or which of two
overlapping blocks is fresher. Those are candidate-engine questions, several
rounds away, and answering them here would bury a trading policy inside a
geometry module.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
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
from goldpipeline.services.ict_order_block import (
    OrderBlock,
    OrderBlockAnalysis,
    OrderBlockConfig,
    analyse_order_blocks,
)
from goldpipeline.services.ict_primitives import DEFAULT_LEFT_BARS, DEFAULT_RIGHT_BARS
from goldpipeline.services.ict_structure import BreakDirection

logger = logging.getLogger(__name__)

LIFECYCLE_METHOD_VERSION = "1.0.0"
"""Stamped on every state and analysis."""


class OrderBlockStatus(StrEnum):
    """Where an order block stands against later observed price.

    Four values, and each is derivable from the evidence fields rather than from
    a remembered transition. ``BREAKER``, ``MITIGATION_BLOCK``, ``REJECTED`` and
    ``FRESH`` are absent: the first two name a conversion this round refuses to
    perform, and the last two name a judgement about usefulness that belongs to
    whatever eventually picks candidates.
    """

    ACTIVE = "ACTIVE"
    """Never touched, never mitigated, never invalidated."""

    TOUCHED = "TOUCHED"
    """Price reached the zone, but the chosen mitigation rule is not satisfied."""

    MITIGATED = "MITIGATED"
    """The chosen mitigation rule is satisfied. Not terminal - see the module docstring."""

    INVALIDATED = "INVALIDATED"
    """A close went strictly beyond the far edge. Terminal."""


class OrderBlockMitigationRule(StrEnum):
    """Which reading of "mitigated" the caller means.

    Three values, all in real use, none of them wrong. The engine applies
    whichever is named, deterministically; it does not have an opinion about
    which one a product should choose.
    """

    TOUCH = "TOUCH"
    """Any contact with the zone mitigates it."""

    MIDPOINT = "MIDPOINT"
    """The observed range must contain the zone's midpoint."""

    FULL_ZONE = "FULL_ZONE"
    """The observed range must cover the whole zone, edge to edge."""


@dataclass(frozen=True)
class OrderBlockLifecycleConfig:
    """The mitigation convention the caller chose.

    No default, deliberately. Formation already requires an explicit
    ``zone_basis``; requiring an explicit ``mitigation_rule`` here means a future
    production caller has to make both choices consciously rather than inherit
    either of them from a module that never argued for it.
    """

    mitigation_rule: OrderBlockMitigationRule


class OrderBlockLifecycleError(ValueError):
    """The engine could not honestly answer, so it refused to guess."""


@dataclass(frozen=True)
class BarWitness:
    """The candle that established one lifecycle fact, exactly as it printed.

    Kept per fact rather than as a bar index so a reader never has to hold a
    series to interpret a state, and so a witness cannot drift if the series it
    came from is re-sliced.
    """

    bar_open_time: datetime
    bar_close_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal

    @classmethod
    def of(cls, bar: OHLCBar, close_time: datetime) -> BarWitness:
        return cls(
            bar_open_time=bar.timestamp,
            bar_close_time=close_time,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
        )


def touches(block: OrderBlock, low: Decimal, high: Decimal) -> bool:
    """Whether an observed range intersects the block's **closed** zone.

    ``high >= block.lower and low <= block.upper``. Exact boundary contact
    counts, which is the deliberate difference from
    :func:`~goldpipeline.services.ict_fvg.enters` - see the module docstring for
    why the two engines differ.

    Direction-neutral: a bullish and a bearish block occupying the same interval
    are touched by exactly the same candles.
    """
    return high >= block.lower and low <= block.upper


def contains_midpoint(block: OrderBlock, low: Decimal, high: Decimal) -> bool:
    """Whether the observed range actually traded the zone's midpoint.

    ``low <= block.midpoint <= high``. Both halves are required. A candle whose
    low is merely *below* the midpoint may never have been near it - a bar
    running 3990 to 3995 beneath a zone of ``[4000, 4010]`` satisfies
    ``low < midpoint`` while having demonstrably never traded there - and
    accepting that would infer an intrabar path nobody observed.
    """
    return low <= block.midpoint <= high


def covers(block: OrderBlock, low: Decimal, high: Decimal) -> bool:
    """Whether the observed range covers the whole zone, edge to edge.

    ``low <= block.lower and high >= block.upper``. Both halves are required,
    and the rule is direction-neutral: a bullish block is not fully mitigated by
    a low beneath its lower edge alone, because the bar may have gapped clean
    underneath without ever trading the band. Same guard, same reasoning, as the
    fair-value-gap fill rule.
    """
    return low <= block.lower and high >= block.upper


def satisfies(
    block: OrderBlock, rule: OrderBlockMitigationRule, low: Decimal, high: Decimal
) -> bool:
    """Whether *rule* considers this observed range to have mitigated *block*.

    Purely geometric under all three rules. The block's bullish or bearish
    direction describes the structure it came from; it does not change which
    part of the zone has to be observed to trade.
    """
    if rule is OrderBlockMitigationRule.TOUCH:
        return touches(block, low, high)
    if rule is OrderBlockMitigationRule.MIDPOINT:
        return contains_midpoint(block, low, high)
    return covers(block, low, high)


def far_edge(block: OrderBlock) -> Decimal:
    """The edge a close beyond which retires the block.

    A bullish block was left behind by a move up, so price closing back under
    its lower edge says that move failed. A bearish block mirrors it.
    """
    return block.lower if block.direction is BreakDirection.BULLISH else block.upper


def invalidates(block: OrderBlock, close: Decimal) -> bool:
    """Whether a closing price retires *block*.

    Strictly beyond the far edge. A close exactly *on* the far edge does not
    invalidate - the level held, by the same close-confirmed standard Round 6.6b
    uses for a structural break, where touching a level is not breaking it.

    A wick beyond the edge is not consulted at all. That is this round's locked
    rule, and a wick-based alternative would be a second explicit policy rather
    than a tweak to this one.
    """
    if block.direction is BreakDirection.BULLISH:
        return close < block.lower
    return close > block.upper


@dataclass(frozen=True)
class OrderBlockState:
    """One order block, and everything later price is known to have done to it.

    The three evidence pairs are independent and each records only its **first**
    occurrence. A later touch does not replace the first, a second qualifying
    bar does not re-mitigate, and there is no count of either: how many times
    price returned is a usefulness question, not a lifecycle fact.
    """

    order_block_id: str
    order_block: OrderBlock
    status: OrderBlockStatus

    mitigation_rule: OrderBlockMitigationRule
    """Which convention produced this state. Changing it may change the status
    and can never change the underlying block."""

    first_touched_at: datetime | None
    first_touch_witness: BarWitness | None

    mitigated_at: datetime | None
    mitigation_witness: BarWitness | None

    invalidated_at: datetime | None
    invalidation_witness: BarWitness | None

    @property
    def is_terminal(self) -> bool:
        return self.status is OrderBlockStatus.INVALIDATED


def status_of(
    *,
    first_touched_at: datetime | None,
    mitigated_at: datetime | None,
    invalidated_at: datetime | None,
) -> OrderBlockStatus:
    """Derive the status from the evidence, in precedence order.

    Invalidation first, then mitigation, then touch. This is *final-status*
    precedence, not a claim about intrabar ordering: when one candle supplies
    several facts at once, all of them are recorded, and the status simply says
    the strongest thing now true.
    """
    if invalidated_at is not None:
        return OrderBlockStatus.INVALIDATED
    if mitigated_at is not None:
        return OrderBlockStatus.MITIGATED
    if first_touched_at is not None:
        return OrderBlockStatus.TOUCHED
    return OrderBlockStatus.ACTIVE


@dataclass(frozen=True)
class OrderBlockLifecycleAnalysis:
    """One timeframe's order-block states under one mitigation rule, at one instant."""

    method_version: str
    timeframe: Timeframe
    symbol: str
    observed_at: datetime
    mitigation_rule: OrderBlockMitigationRule

    states: tuple[OrderBlockState, ...]
    """In formation order, which lifecycle never reorders."""

    active_ids: tuple[str, ...]
    touched_ids: tuple[str, ...]
    mitigated_ids: tuple[str, ...]
    invalidated_ids: tuple[str, ...]

    def state(self, identity: str) -> OrderBlockState | None:
        return next((entry for entry in self.states if entry.order_block_id == identity), None)


def analyse_order_block_lifecycle(
    series: IctTimeframeSnapshot,
    *,
    config: OrderBlockLifecycleConfig,
    formation: OrderBlockConfig | None = None,
    order_blocks: OrderBlockAnalysis | None = None,
    symbol: str = "",
    as_of: datetime | None = None,
    left_bars: int = DEFAULT_LEFT_BARS,
    right_bars: int = DEFAULT_RIGHT_BARS,
) -> OrderBlockLifecycleAnalysis:
    """Replay the closed candles after each block formed and record what they did.

    Per block, over every candle whose close is strictly after ``formed_at``:

    1. If the observed range meets the zone, record the first touch.
    2. If it satisfies the chosen mitigation rule, record the first mitigation.
    3. If the close is strictly beyond the far edge, record the invalidation and
       stop - invalidation is terminal.

    All three are read from the same candle before moving on, so one bar may
    supply several facts at once; none of them is discarded to make a tidier
    story out of an intrabar path nobody observed.

    Exactly one of *formation* and *order_blocks* must be given, which is what
    keeps both policy choices conscious: either you name the zone basis to
    compute blocks with, or you hand in an analysis that already recorded one.

    A supplied ``OrderBlockAnalysis`` is required to describe this same instant,
    timeframe and symbol, and is refused otherwise. Accepting a bare list and
    filtering it by ``formed_at`` was the other option and was rejected: a list
    carries no record of the moment it was computed for, so nothing in it could
    prove it had not been derived with lookahead.

    Args:
        series: Closed candles for one timeframe.
        config: The mitigation rule. Required; there is no default.
        formation: The zone-basis policy to form blocks with, when none are
            supplied.
        order_blocks: Blocks a composite stage has already computed.
        symbol: Instrument name. Provenance.
        as_of: Reconstruct the state as it stood at this instant.
        left_bars: Pivot window, passed through to the formation engine.
        right_bars: The same on the right.

    Raises:
        OrderBlockLifecycleError: Neither or both of *formation* and
            *order_blocks* were given, or a supplied analysis describes a
            different instant, timeframe or symbol.
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

    if (formation is None) == (order_blocks is None):
        raise OrderBlockLifecycleError(
            "supply exactly one of formation (a zone basis to form blocks with) or "
            "order_blocks (an analysis that already recorded one)"
        )

    if order_blocks is None:
        assert formation is not None
        order_blocks = analyse_order_blocks(
            series,
            config=formation,
            symbol=symbol,
            as_of=as_of,
            left_bars=left_bars,
            right_bars=right_bars,
        )
    else:
        _require_same_moment(order_blocks, observed_at, timeframe, symbol)

    states = tuple(
        _state_for(block, bars=working.bars, duration=duration, rule=config.mitigation_rule)
        for block in order_blocks.order_blocks
    )

    return OrderBlockLifecycleAnalysis(
        method_version=LIFECYCLE_METHOD_VERSION,
        timeframe=timeframe,
        symbol=symbol,
        observed_at=observed_at,
        mitigation_rule=config.mitigation_rule,
        states=states,
        active_ids=_ids(states, OrderBlockStatus.ACTIVE),
        touched_ids=_ids(states, OrderBlockStatus.TOUCHED),
        mitigated_ids=_ids(states, OrderBlockStatus.MITIGATED),
        invalidated_ids=_ids(states, OrderBlockStatus.INVALIDATED),
    )


def analyse_snapshot_order_block_lifecycle(
    snapshot: IctMarketSnapshot,
    timeframe: Timeframe,
    *,
    config: OrderBlockLifecycleConfig,
    formation: OrderBlockConfig | None = None,
    order_blocks: OrderBlockAnalysis | None = None,
    as_of: datetime | None = None,
    left_bars: int = DEFAULT_LEFT_BARS,
    right_bars: int = DEFAULT_RIGHT_BARS,
) -> OrderBlockLifecycleAnalysis:
    """:func:`analyse_order_block_lifecycle` for one timeframe of a snapshot.

    One timeframe, named explicitly. An H4 block's mitigation says nothing about
    an M15 block: whether a higher timeframe's state governs a lower one is a
    synthesis rule with real content, and it is not decided by a geometry module.
    """
    return analyse_order_block_lifecycle(
        snapshot.require(timeframe),
        config=config,
        formation=formation,
        order_blocks=order_blocks,
        symbol=snapshot.symbol,
        as_of=as_of,
        left_bars=left_bars,
        right_bars=right_bars,
    )


def _require_same_moment(
    analysis: OrderBlockAnalysis, observed_at: datetime, timeframe: Timeframe, symbol: str
) -> None:
    """A supplied formation analysis must describe this exact request.

    Refused rather than trimmed. Filtering its blocks by ``formed_at`` would
    leave the analysis' own ``observed_at`` describing another moment, and the
    caller would have no way to notice - the same reasoning Rounds 6.6c.3a,
    6.6c.3b and 6.6d.1 use.
    """
    if analysis.observed_at != observed_at:
        raise OrderBlockLifecycleError(
            f"order blocks describe {analysis.observed_at.isoformat()} but this analysis is "
            f"as of {observed_at.isoformat()}; supply one computed for the same instant"
        )
    if analysis.timeframe is not timeframe:
        raise OrderBlockLifecycleError(
            f"order blocks are for {analysis.timeframe.value}, not {timeframe.value}"
        )
    if analysis.symbol != symbol:
        raise OrderBlockLifecycleError(f"order blocks are for {analysis.symbol!r}, not {symbol!r}")


def _state_for(
    block: OrderBlock,
    *,
    bars: tuple[OHLCBar, ...],
    duration: timedelta,
    rule: OrderBlockMitigationRule,
) -> OrderBlockState:
    """Walk the candles that closed after *block* formed and gather the evidence."""
    first_touched_at: datetime | None = None
    first_touch_witness: BarWitness | None = None
    mitigated_at: datetime | None = None
    mitigation_witness: BarWitness | None = None
    invalidated_at: datetime | None = None
    invalidation_witness: BarWitness | None = None

    for bar in bars:
        close_time = bar.timestamp + duration
        # Strictly after. The break bar closes exactly at formed_at, and it made
        # the block rather than interacted with one: before that instant the
        # source candle was just a candle, so there was nothing there to touch.
        if close_time <= block.formed_at:
            continue

        if first_touched_at is None and touches(block, bar.low, bar.high):
            first_touched_at = close_time
            first_touch_witness = BarWitness.of(bar, close_time)

        if mitigated_at is None and satisfies(block, rule, bar.low, bar.high):
            mitigated_at = close_time
            mitigation_witness = BarWitness.of(bar, close_time)

        if invalidates(block, bar.close):
            invalidated_at = close_time
            invalidation_witness = BarWitness.of(bar, close_time)
            # Terminal. Later bars cannot add a touch, a mitigation or a second
            # invalidation, so the walk stops rather than recording history the
            # block was no longer alive for.
            break

    return OrderBlockState(
        order_block_id=block.order_block_id,
        order_block=block,
        status=status_of(
            first_touched_at=first_touched_at,
            mitigated_at=mitigated_at,
            invalidated_at=invalidated_at,
        ),
        mitigation_rule=rule,
        first_touched_at=first_touched_at,
        first_touch_witness=first_touch_witness,
        mitigated_at=mitigated_at,
        mitigation_witness=mitigation_witness,
        invalidated_at=invalidated_at,
        invalidation_witness=invalidation_witness,
    )


def _ids(states: Sequence[OrderBlockState], status: OrderBlockStatus) -> tuple[str, ...]:
    """The identities at *status*, in formation order.

    Built by filtering the ordered states rather than by grouping into a mapping,
    so no set or dict iteration order can reach the output.
    """
    return tuple(entry.order_block_id for entry in states if entry.status is status)


__all__ = [
    "LIFECYCLE_METHOD_VERSION",
    "BarWitness",
    "OrderBlockLifecycleAnalysis",
    "OrderBlockLifecycleConfig",
    "OrderBlockLifecycleError",
    "OrderBlockMitigationRule",
    "OrderBlockState",
    "OrderBlockStatus",
    "analyse_order_block_lifecycle",
    "analyse_snapshot_order_block_lifecycle",
    "contains_midpoint",
    "covers",
    "far_edge",
    "invalidates",
    "satisfies",
    "status_of",
    "touches",
]
