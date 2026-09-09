"""The range a structural leg opens, how far it has run, and where a price sits in it.

Round 6.6c.3b, and the payoff for having split the anchor question off into
6.6c.3a. A dealing range is what most ICT material means by "the range" - the
thing premium and discount are measured against - and it is only definable once
"which swing anchors this leg?" has a deterministic answer.

**A leg is a record; a range is a state.** ``StructuralLeg`` is immutable: one
protected origin, one break bar's extreme, fixed forever. A ``DealingRange``
*starts* from a leg and then its directional edge extends as later closed
candles print new extremes. The leg is never mutated and never redefined
because the range grew - they answer different questions, and conflating them
would lose the record of what the breaking candle itself did.

**The fixed edge never moves.** A bullish range's lower edge is the protected
low, exactly, for the whole life of the range. Not a later higher low, not a
changed ``active_low``, not a wick that dipped beneath it. Price may trade
outside a range that is still current - that is why
:class:`PriceLocation` has ``BELOW_RANGE`` and ``ABOVE_RANGE`` - and widening
the anchor to swallow the excursion would erase the very thing the range is
anchored to. Only a new structural event may create a different range.

**Wicks, not closes.** Round 6.6c.3a records a leg's terminal as the break
bar's own high (bullish) or low (bearish). The range continues that convention:
its extending edge is the extreme of closed candles. Switching to closes at the
extension step would leave the initial leg and its own range following two
incompatible price definitions.

**Premium and discount are halves of a range, not instructions.** ``PREMIUM``
is the upper half and ``DISCOUNT`` the lower half, for bullish and bearish
ranges alike. The geometry does not flip with direction, and nothing here says
discount means buy. What to do about a location is the candidate engine's
question, several rounds away.

**One protected-anchor authority.** Assignments and legs come from
:mod:`goldpipeline.services.ict_protected`, and structure events from
:mod:`goldpipeline.services.ict_structure`. No swing is chosen here, no break
is detected here, and no anchor is inferred from raw candles.
"""

from __future__ import annotations

import hashlib
import logging
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
from goldpipeline.services.ict_primitives import DEFAULT_LEFT_BARS, DEFAULT_RIGHT_BARS
from goldpipeline.services.ict_protected import (
    ProtectedStructureAnalysis,
    StructuralLeg,
    analyse_protected_structure,
)
from goldpipeline.services.ict_structure import (
    BreakClassification,
    BreakDirection,
    StructureAnalysis,
    StructureBias,
    analyse_structure,
)

logger = logging.getLogger(__name__)

RANGE_METHOD_VERSION = "1.0.0"
"""Stamped on every range and analysis, and part of every range identity."""


class RangeStatus(StrEnum):
    """Whether a range still speaks for the market.

    Two values. ``INVALID``, ``MITIGATED``, ``BROKEN`` and ``FILLED`` are absent
    because each belongs to a different concept - a range is not invalidated by
    price leaving it, and it is not filled by anything. It is replaced, by a
    later structural event, and until then it is current.
    """

    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"


class PriceLocation(StrEnum):
    """Where a price sits relative to a range. Five values, all geometric.

    The two outside values exist because a still-current range genuinely can be
    left behind: the fixed edge is a protected swing, and price is free to trade
    through it without any structural event having replaced the range yet.
    Reporting that honestly is better than widening the range to hide it.
    """

    BELOW_RANGE = "BELOW_RANGE"
    DISCOUNT = "DISCOUNT"
    EQUILIBRIUM = "EQUILIBRIUM"
    PREMIUM = "PREMIUM"
    ABOVE_RANGE = "ABOVE_RANGE"


class DealingRangeError(ValueError):
    """The engine could not honestly answer, so it refused to guess."""


def locate(price: Decimal, *, lower: Decimal, upper: Decimal) -> PriceLocation:
    """Classify *price* against a range's two edges.

    The boundaries are decided, not left to chance:

    * exactly ``lower`` is ``DISCOUNT`` - it is in the range, in its lower half;
    * exactly ``upper`` is ``PREMIUM``;
    * exactly the equilibrium is ``EQUILIBRIUM``, with no tolerance around it.
      A band of "near enough" would be a parameter nobody chose.

    Identical for bullish and bearish ranges. Premium is the upper half of a
    price range whichever way the market got there.
    """
    if upper <= lower:
        raise DealingRangeError(f"a range needs upper > lower, got {lower}..{upper}")

    if price < lower:
        return PriceLocation.BELOW_RANGE
    if price > upper:
        return PriceLocation.ABOVE_RANGE

    middle = (lower + upper) / Decimal(2)
    if price < middle:
        return PriceLocation.DISCOUNT
    if price > middle:
        return PriceLocation.PREMIUM
    return PriceLocation.EQUILIBRIUM


@dataclass(frozen=True)
class DealingRange:
    """One structural leg's range, as it stood at the analysis instant.

    ``origin_price`` is the fixed edge and never changes. ``terminal_price`` is
    the extending edge, and ``lower``/``upper`` are the two of them sorted, so a
    reader never has to remember which is which for a bearish range.
    """

    range_id: str
    method_version: str
    timeframe: Timeframe
    symbol: str
    direction: BreakDirection

    protected_assignment_id: str
    structural_leg_id: str
    establishing_event_id: str

    formed_at: datetime

    origin_price: Decimal
    initial_terminal_price: Decimal
    """The leg's own terminal - the break bar's extreme - kept so a reader can
    see how far the range has run since it opened."""

    lower: Decimal
    upper: Decimal
    equilibrium: Decimal
    width: Decimal

    terminal_price: Decimal
    terminal_bar_open_time: datetime
    terminal_bar_close_time: datetime
    """The candle currently supplying the extending edge.

    The **first** candle to reach that extreme. A later bar printing exactly the
    same high does not take the provenance from it: price revisiting a level is
    not price establishing it.
    """

    status: RangeStatus
    superseded_at: datetime | None
    superseded_by_event_id: str | None
    superseded_by_range_id: str | None

    def locate(self, price: Decimal) -> PriceLocation:
        """Where *price* sits in this range."""
        return locate(price, lower=self.lower, upper=self.upper)

    @property
    def is_active(self) -> bool:
        return self.status is RangeStatus.ACTIVE


@dataclass(frozen=True)
class DealingRangeAnalysis:
    """One timeframe's dealing ranges, as of one instant."""

    method_version: str
    timeframe: Timeframe
    symbol: str
    observed_at: datetime
    structure_bias: StructureBias

    ranges: tuple[DealingRange, ...]
    active_range_id: str | None

    def range_of(self, identity: str) -> DealingRange | None:
        return next((entry for entry in self.ranges if entry.range_id == identity), None)

    @property
    def active_range(self) -> DealingRange | None:
        if self.active_range_id is None:
            return None
        return self.range_of(self.active_range_id)

    def locate(self, price: Decimal) -> PriceLocation | None:
        """Where *price* sits in the current range, or ``None`` if there is none."""
        current = self.active_range
        return None if current is None else current.locate(price)


def _range_id(
    *,
    symbol: str,
    timeframe: Timeframe,
    assignment_id: str,
    leg_id: str,
) -> str:
    """Identity from the formation facts, and nothing that moves.

    The extending terminal is deliberately absent: the same range observed an
    hour later, having run further, must still be the same range. Assignment and
    leg identities already encode the symbol, timeframe, anchor and event, so no
    price enters this preimage - which is why this round needed no fourth copy
    of the Decimal canonicalisation the liquidity and gap identities require.
    """
    preimage = "|".join((symbol, timeframe.value, RANGE_METHOD_VERSION, assignment_id, leg_id))
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]


@dataclass
class _Working:
    """A range while the replay is still running."""

    range_id: str
    direction: BreakDirection
    leg: StructuralLeg
    origin_price: Decimal
    initial_terminal_price: Decimal
    terminal_price: Decimal
    terminal_bar_open_time: datetime
    terminal_bar_close_time: datetime
    status: RangeStatus = RangeStatus.ACTIVE
    superseded_at: datetime | None = None
    superseded_by_event_id: str | None = None
    superseded_by_range_id: str | None = None

    @property
    def bounds(self) -> tuple[Decimal, Decimal]:
        if self.direction is BreakDirection.BULLISH:
            return self.origin_price, self.terminal_price
        return self.terminal_price, self.origin_price

    def extend(self, bar: OHLCBar, close_time: datetime) -> None:
        """Push the directional edge out if this candle reached further.

        Strictly further. A candle matching the current extreme leaves the
        witness alone, because it revisited a level rather than establishing
        one.
        """
        if self.direction is BreakDirection.BULLISH:
            if bar.high > self.terminal_price:
                self.terminal_price = bar.high
                self.terminal_bar_open_time = bar.timestamp
                self.terminal_bar_close_time = close_time
        elif bar.low < self.terminal_price:
            self.terminal_price = bar.low
            self.terminal_bar_open_time = bar.timestamp
            self.terminal_bar_close_time = close_time

    def freeze(self, *, timeframe: Timeframe, symbol: str) -> DealingRange:
        lower, upper = self.bounds
        if upper <= lower:
            raise DealingRangeError(f"range {self.range_id} has no width: {lower}..{upper}")
        return DealingRange(
            range_id=self.range_id,
            method_version=RANGE_METHOD_VERSION,
            timeframe=timeframe,
            symbol=symbol,
            direction=self.direction,
            protected_assignment_id=self.leg.protected_assignment_id,
            structural_leg_id=self.leg.leg_id,
            establishing_event_id=self.leg.event_id,
            formed_at=self.leg.formed_at,
            origin_price=self.origin_price,
            initial_terminal_price=self.initial_terminal_price,
            lower=lower,
            upper=upper,
            equilibrium=(lower + upper) / Decimal(2),
            width=upper - lower,
            terminal_price=self.terminal_price,
            terminal_bar_open_time=self.terminal_bar_open_time,
            terminal_bar_close_time=self.terminal_bar_close_time,
            status=self.status,
            superseded_at=self.superseded_at,
            superseded_by_event_id=self.superseded_by_event_id,
            superseded_by_range_id=self.superseded_by_range_id,
        )


def analyse_dealing_ranges(
    series: IctTimeframeSnapshot,
    *,
    structure: StructureAnalysis | None = None,
    protected: ProtectedStructureAnalysis | None = None,
    symbol: str = "",
    as_of: datetime | None = None,
    left_bars: int = DEFAULT_LEFT_BARS,
    right_bars: int = DEFAULT_RIGHT_BARS,
) -> DealingRangeAnalysis:
    """Replay the series and report the ranges each structural leg opened.

    The exact order of operations at each closed bar, because the answers depend
    on it:

    1. Extend the active range, if any, with **this** bar's directional extreme.
    2. Look for a structure event closing at this instant.
    3. If a protected assignment and leg were established here, supersede the
       active range and open a new one from that leg.
    4. Otherwise, if the event was an MSS - a direction flip - with no anchor,
       supersede the active range and open nothing.
    5. Otherwise leave the range alone.

    Step 1 comes before steps 3 and 4 on purpose. A structural event is only
    known when its bar closes, so that bar is a fully observed candle during the
    old range's lifetime; it may legitimately be the old range's final terminal
    witness *and* the new range's first. Doing it the other way round would
    discard an extreme the market actually printed.

    Args:
        series: Closed candles for one timeframe.
        structure: A structure analysis for this same series and instant.
        protected: A protected-structure analysis for the same. Both are
            optional; supply them when a composite stage already has them, and
            their ``observed_at`` must match or they are refused.
        symbol: Instrument name. Provenance, and part of every identity.
        as_of: Reconstruct the state as it stood at this instant.
        left_bars: Pivot window, passed through to the structure engine.
        right_bars: The same on the right.

    Raises:
        DealingRangeError: A supplied analysis describes a different instant,
            symbol or timeframe; a leg's terminal bar is missing or disagrees
            with the leg; a range would have no width; or the surviving range
            disagrees with the structure bias.
        ValueError: *as_of* precedes every bar's close.
    """
    working_series = series
    if as_of is not None:
        working_series = build_timeframe_snapshot(
            timeframe=series.timeframe, bars=series.bars, observed_at=as_of
        )
    observed_at = as_of if as_of is not None else series.latest_closed_at
    timeframe = working_series.timeframe

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

    events_by_close = {event.break_bar_close_time: event for event in structure.breaks}
    legs_by_event = {leg.event_id: leg for leg in protected.legs}
    bars_by_open = {bar.timestamp: bar for bar in working_series.bars}

    ranges: list[_Working] = []
    active: _Working | None = None

    for bar in working_series.bars:
        close_time = bar.timestamp + duration

        # (1) The active range sees this candle before anything else happens.
        if active is not None and close_time > active.leg.formed_at:
            active.extend(bar, close_time)

        event = events_by_close.get(close_time)
        if event is None:
            continue

        leg = legs_by_event.get(event.event_id)
        if leg is not None:
            fresh = _open_range(leg, bars_by_open=bars_by_open, symbol=symbol, timeframe=timeframe)
            if active is not None:
                _supersede(
                    active, at=close_time, event_id=event.event_id, replacement=fresh.range_id
                )
            ranges.append(fresh)
            active = fresh
        elif event.classification is BreakClassification.MSS and active is not None:
            # (4) The direction flipped and nothing anchored the new one. The old
            # range cannot speak for a market that has turned, so it ends here
            # and nothing takes its place until an event establishes an anchor.
            _supersede(active, at=close_time, event_id=event.event_id, replacement=None)
            active = None
        # (5) Anything else - a same-direction event that established no anchor -
        # leaves the range exactly as it is. See `_SAME_DIRECTION_NOTE`.

    frozen = tuple(entry.freeze(timeframe=timeframe, symbol=symbol) for entry in ranges)
    current = next((entry for entry in frozen if entry.is_active), None)

    if current is not None and not _agrees(current.direction, structure.current_bias):
        raise DealingRangeError(
            f"range {current.range_id} is {current.direction.value} but structure bias is "
            f"{structure.current_bias.value}; this is an invariant failure, not something "
            "to repair silently"
        )

    return DealingRangeAnalysis(
        method_version=RANGE_METHOD_VERSION,
        timeframe=timeframe,
        symbol=symbol,
        observed_at=observed_at,
        structure_bias=structure.current_bias,
        ranges=frozen,
        active_range_id=None if current is None else current.range_id,
    )


_SAME_DIRECTION_NOTE = """A same-direction structural event that establishes no
anchor leaves the current range untouched.

Reachable: Round 6.6b's own journey contains a bearish BOS with no eligible
opposite high (its bar 31). But it is not reachable *while a range is active* -
for a bullish range's protected low to have been consumed, a bearish close must
have taken it out, and from a bullish state that close is an MSS, which
supersedes the range and flips the bias. So by the time a same-direction event
can find itself without an anchor, there is no active range for this branch to
protect. The behaviour is defined anyway, because the reasoning depends on
Round 6.6b's classification rules and those are free to change.
"""


def analyse_snapshot_dealing_ranges(
    snapshot: IctMarketSnapshot,
    timeframe: Timeframe,
    *,
    structure: StructureAnalysis | None = None,
    protected: ProtectedStructureAnalysis | None = None,
    as_of: datetime | None = None,
    left_bars: int = DEFAULT_LEFT_BARS,
    right_bars: int = DEFAULT_RIGHT_BARS,
) -> DealingRangeAnalysis:
    """:func:`analyse_dealing_ranges` for one timeframe of a snapshot.

    One timeframe, named explicitly. An H4 range does not govern an M15 one:
    which timeframe's range premium is measured against is a synthesis rule with
    real content, and it is not decided by a geometry module.
    """
    return analyse_dealing_ranges(
        snapshot.require(timeframe),
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

    Refused rather than trimmed. An analysis carries current references - a
    bias, a current anchor - as well as history, and quietly cutting its event
    list would leave those describing another moment: a subtler leak than the
    one being prevented. Same reasoning as Round 6.6c.3a.
    """
    if given_at != observed_at:
        raise DealingRangeError(
            f"{what} describes {given_at.isoformat()} but this analysis is as of "
            f"{observed_at.isoformat()}; supply one computed for the same instant"
        )
    if given_timeframe is not timeframe:
        raise DealingRangeError(f"{what} is for {given_timeframe.value}, not {timeframe.value}")
    if given_symbol != symbol:
        raise DealingRangeError(f"{what} is for {given_symbol!r}, not {symbol!r}")


def _open_range(
    leg: StructuralLeg,
    *,
    bars_by_open: dict[datetime, OHLCBar],
    symbol: str,
    timeframe: Timeframe,
) -> _Working:
    """Create a range from *leg*, verifying its terminal against the real candle."""
    bar = bars_by_open.get(leg.terminal_bar_open_time)
    if bar is None:
        raise DealingRangeError(
            f"leg {leg.leg_id} names a terminal bar opening at "
            f"{leg.terminal_bar_open_time.isoformat()}, which is not in this series"
        )

    extreme = bar.high if leg.direction is BreakDirection.BULLISH else bar.low
    if extreme != leg.terminal_price:
        # Verified rather than assumed. If these ever disagree the two engines
        # are reading different candles, and guessing which is right would bury
        # the fault in a price nobody could trace.
        raise DealingRangeError(
            f"leg {leg.leg_id} claims terminal {leg.terminal_price} but its bar's "
            f"{'high' if leg.direction is BreakDirection.BULLISH else 'low'} is {extreme}"
        )

    return _Working(
        range_id=_range_id(
            symbol=symbol,
            timeframe=timeframe,
            assignment_id=leg.protected_assignment_id,
            leg_id=leg.leg_id,
        ),
        direction=leg.direction,
        leg=leg,
        origin_price=leg.origin_price,
        initial_terminal_price=leg.terminal_price,
        terminal_price=leg.terminal_price,
        terminal_bar_open_time=leg.terminal_bar_open_time,
        terminal_bar_close_time=leg.terminal_bar_close_time,
    )


def _supersede(entry: _Working, *, at: datetime, event_id: str, replacement: str | None) -> None:
    entry.status = RangeStatus.SUPERSEDED
    entry.superseded_at = at
    entry.superseded_by_event_id = event_id
    entry.superseded_by_range_id = replacement


def _agrees(direction: BreakDirection, bias: StructureBias) -> bool:
    return (direction is BreakDirection.BULLISH and bias is StructureBias.BULLISH) or (
        direction is BreakDirection.BEARISH and bias is StructureBias.BEARISH
    )


__all__ = [
    "RANGE_METHOD_VERSION",
    "DealingRange",
    "DealingRangeAnalysis",
    "DealingRangeError",
    "PriceLocation",
    "RangeStatus",
    "analyse_dealing_ranges",
    "analyse_snapshot_dealing_ranges",
    "locate",
]
