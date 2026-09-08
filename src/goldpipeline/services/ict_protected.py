"""Which swing a directional break left behind it, and the leg that break drew.

Round 6.6c.3a. Two ideas, and the first one is the whole point of splitting this
round off from the dealing range.

**A protected swing is established by an event, not by being recent.** The
tempting definition - "the protected low is the latest confirmed swing low" -
is just a second name for
:attr:`~goldpipeline.services.ict_structure.StructureAnalysis.active_low`, and
renaming a thing is not defining it. Here a swing becomes protected only when a
confirmed structural break establishes it as the *opposite-side origin* of that
break: the low a bullish move set off from, the high a bearish move fell away
from. Before that break it was a confirmed swing like any other.

**Active and protected are related but different.** After a bullish BOS a newer
higher low may confirm and become the active low - the swing a later bearish
close would break, and therefore the one that can trigger an MSS. It is *not*
protected by having confirmed. It becomes protected only if a later bullish
break establishes it. So the swing capable of triggering an MSS and the swing
currently protected are not guaranteed to be the same one, and this module does
not pretend otherwise. Round 6.6b's rules are untouched.

**No anchor, no fabrication.** A structure break can happen with no eligible
opposite-side swing at all - the whole other side may have been consumed. That
break stays valid and is still an event; it simply establishes nothing and
draws no leg. Substituting the break bar's own low, or the lowest low in N
bars, would be inventing an origin the market never printed.

**A structural leg is not a dealing range.** It is one event's observed
expansion: from the protected swing's price to the break bar's own extreme, and
then it stops. It does not grow with later highs, it has no equilibrium, and
nothing here computes a midpoint. Extending a leg into a live range is Round
6.6c.3b's job, and doing it here would settle that question by accident.
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
from goldpipeline.services.ict_primitives import DEFAULT_LEFT_BARS, DEFAULT_RIGHT_BARS
from goldpipeline.services.ict_structure import (
    BreakClassification,
    BreakDirection,
    StructureAnalysis,
    StructureBias,
    StructureBreak,
    analyse_structure,
)

logger = logging.getLogger(__name__)

PROTECTED_METHOD_VERSION = "1.0.0"
"""Stamped on every assignment, leg and analysis, and part of both identities."""


class ProtectedSwingType(StrEnum):
    """Which side of the leg the anchor sits on.

    ``LOW`` and ``HIGH``, named for the geometry. Not support, resistance, BUY,
    SELL, SEO or BAI - a protected low is where a bullish leg started, which is
    a fact about the past, not an instruction about the future.
    """

    LOW = "LOW"
    HIGH = "HIGH"


def protected_type_for(direction: BreakDirection) -> ProtectedSwingType:
    """The anchor side a break in *direction* establishes.

    Bullish leg, low anchor; bearish leg, high anchor. Keyed on the break's own
    direction, which for an MSS is already the **new** direction - so a bearish
    MSS out of a bullish state establishes a protected *high*, never the low it
    just broke.
    """
    return (
        ProtectedSwingType.LOW if direction is BreakDirection.BULLISH else ProtectedSwingType.HIGH
    )


def _digest(*parts: str) -> str:
    """Sixteen hex characters over the given fields joined by ``|``.

    No clock, no counter, no random source, and no provider metadata: the same
    candles from any source describe the same assignment and the same leg.

    Note that no price participates in either identity below. Both are built
    from a swing identity and an event identity, neither of which carries a
    number, so the Decimal-canonicalisation care needed by the liquidity and
    gap identities has nothing to do here.
    """
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ProtectedSwingAssignment:
    """One structural break naming one swing as the origin of its leg.

    A permanent record of a causal fact - "event E established swing S" - and
    deliberately not a lifecycle. A later break may take S out, and that does
    not make this assignment false; it makes it history. Whether an assignment
    is still the usable anchor is a question for :func:`current_assignment` and,
    later, for whatever builds ranges out of these.
    """

    assignment_id: str
    method_version: str
    timeframe: Timeframe
    symbol: str

    protected_type: ProtectedSwingType
    swing_id: str
    swing_price: Decimal
    swing_pivot_time: datetime
    swing_confirmed_at: datetime

    established_by_event_id: str
    established_at: datetime
    """The break bar's close time.

    The swing itself existed well before this, and was confirmed before this.
    What begins here is its *protected status*, which is a property of the event
    rather than of the swing.
    """

    event_direction: BreakDirection
    event_classification: BreakClassification


@dataclass(frozen=True)
class StructuralLeg:
    """One break's observed expansion, from its protected origin to its extreme.

    Immutable, and immutable in a specific way worth stating: the terminal is
    the break bar's own extreme and it never grows. A later, higher high after a
    bullish break does not extend this leg. That restraint is what keeps a leg a
    record of one event rather than a slowly drifting object nobody can date.
    """

    leg_id: str
    method_version: str
    timeframe: Timeframe
    symbol: str

    event_id: str
    event_classification: BreakClassification
    direction: BreakDirection

    protected_assignment_id: str
    origin_swing_id: str
    origin_price: Decimal
    origin_pivot_time: datetime
    origin_confirmed_at: datetime

    terminal_bar_open_time: datetime
    terminal_bar_close_time: datetime
    terminal_price: Decimal
    """The break bar's high for a bullish leg, its low for a bearish one.

    Not the close, and not the level that was broken. By the time the event
    exists the whole bar has closed, so its extreme is already observed - and it
    is the furthest point that candle can be said to have reached. Using the
    close would understate an expansion the market actually printed.
    """

    formed_at: datetime

    @property
    def span(self) -> Decimal:
        """Distance from origin to terminal. A magnitude, not a range."""
        return abs(self.terminal_price - self.origin_price)


@dataclass(frozen=True)
class ProtectedStructureAnalysis:
    """One timeframe's protected anchors and legs, as of one instant."""

    method_version: str
    timeframe: Timeframe
    symbol: str
    observed_at: datetime

    assignments: tuple[ProtectedSwingAssignment, ...]
    legs: tuple[StructuralLeg, ...]

    current_protected_assignment_id: str | None
    current_leg_id: str | None

    def assignment(self, identity: str) -> ProtectedSwingAssignment | None:
        return next((a for a in self.assignments if a.assignment_id == identity), None)

    def leg(self, identity: str) -> StructuralLeg | None:
        return next((entry for entry in self.legs if entry.leg_id == identity), None)

    @property
    def current_assignment(self) -> ProtectedSwingAssignment | None:
        if self.current_protected_assignment_id is None:
            return None
        return self.assignment(self.current_protected_assignment_id)

    @property
    def current_leg(self) -> StructuralLeg | None:
        if self.current_leg_id is None:
            return None
        return self.leg(self.current_leg_id)


class ProtectedStructureError(ValueError):
    """The engine could not honestly answer, so it refused to guess."""


def analyse_protected_structure(
    series: IctTimeframeSnapshot,
    *,
    structure: StructureAnalysis | None = None,
    symbol: str = "",
    as_of: datetime | None = None,
    left_bars: int = DEFAULT_LEFT_BARS,
    right_bars: int = DEFAULT_RIGHT_BARS,
) -> ProtectedStructureAnalysis:
    """Walk the structure events and record what each one anchored.

    Per event, in this order:

    1. Read the opposite-side active swing the structure engine recorded *at*
       the event. That is event-time state, not something re-derived afterwards
       - see ``StructureBreak.opposite_active_swing_id``.
    2. If there was none, the event establishes nothing and draws no leg.
    3. Otherwise create one assignment, dated at the break bar's close.
    4. Resolve the actual break bar and take its directional extreme.
    5. Create one leg from the anchor's price to that extreme.

    Assignments and legs are appended in event order and never revised, so a
    later break cannot alter an earlier one.

    Args:
        series: Closed candles for one timeframe.
        structure: An analysis already computed over this same series and
            instant, for a caller that has one. Its ``observed_at`` must match,
            so a full-history analysis cannot be handed to an as-of query.
        symbol: Instrument name. Provenance, and part of both identities.
        as_of: Reconstruct the state as it stood at this instant.
        left_bars: Pivot window, passed through to the structure engine.
        right_bars: The same on the right.

    Raises:
        ProtectedStructureError: A supplied structure describes a different
            instant, a break bar cannot be found in the series, or a leg would
            run the wrong way.
        ValueError: *as_of* precedes every bar's close.
    """
    working = series
    if as_of is not None:
        working = build_timeframe_snapshot(
            timeframe=series.timeframe, bars=series.bars, observed_at=as_of
        )
    observed_at = as_of if as_of is not None else series.latest_closed_at

    if structure is None:
        structure = analyse_structure(
            series, symbol=symbol, as_of=as_of, left_bars=left_bars, right_bars=right_bars
        )
    elif structure.observed_at != observed_at:
        # Fail closed rather than filter. A structure analysis carries a bias
        # and a set of active levels as well as events, and silently trimming
        # its event list would leave those describing a different moment - a
        # subtler leak than the one being prevented.
        raise ProtectedStructureError(
            f"structure describes {structure.observed_at.isoformat()} but this analysis is "
            f"as of {observed_at.isoformat()}; supply one computed for the same instant"
        )

    bars_by_open = {bar.timestamp: bar for bar in working.bars}

    assignments: list[ProtectedSwingAssignment] = []
    legs: list[StructuralLeg] = []

    for event in structure.breaks:
        assignment = _assignment_for(event, symbol=symbol, timeframe=working.timeframe)
        if assignment is None:
            continue
        assignments.append(assignment)
        legs.append(
            _leg_for(
                event,
                assignment=assignment,
                bars_by_open=bars_by_open,
                symbol=symbol,
                timeframe=working.timeframe,
            )
        )

    current = current_assignment(assignments, structure.current_bias)
    current_leg = None
    if current is not None:
        current_leg = next(
            (leg for leg in legs if leg.protected_assignment_id == current.assignment_id), None
        )

    return ProtectedStructureAnalysis(
        method_version=PROTECTED_METHOD_VERSION,
        timeframe=working.timeframe,
        symbol=symbol,
        observed_at=observed_at,
        assignments=tuple(assignments),
        legs=tuple(legs),
        current_protected_assignment_id=None if current is None else current.assignment_id,
        current_leg_id=None if current_leg is None else current_leg.leg_id,
    )


def analyse_snapshot_protected_structure(
    snapshot: IctMarketSnapshot,
    timeframe: Timeframe,
    *,
    structure: StructureAnalysis | None = None,
    as_of: datetime | None = None,
    left_bars: int = DEFAULT_LEFT_BARS,
    right_bars: int = DEFAULT_RIGHT_BARS,
) -> ProtectedStructureAnalysis:
    """:func:`analyse_protected_structure` for one timeframe of a snapshot.

    One timeframe, named explicitly. Nothing here lets an H4 protected low
    override an M15 one: which timeframe's anchor governs is a synthesis rule
    with real content, and it is not decided by a geometry module.
    """
    return analyse_protected_structure(
        snapshot.require(timeframe),
        structure=structure,
        symbol=snapshot.symbol,
        as_of=as_of,
        left_bars=left_bars,
        right_bars=right_bars,
    )


def current_assignment(
    assignments: Sequence[ProtectedSwingAssignment], bias: StructureBias
) -> ProtectedSwingAssignment | None:
    """The assignment that still speaks for the state the market is in.

    The latest one whose event direction matches the current structure bias. A
    ``NEUTRAL`` bias has no current anchor, because nothing has established a
    direction to anchor.

    Deliberately simple, and it holds up on the case that looks dangerous:
    "what if the current anchor's swing has since been broken?" Breaking a
    protected low downward requires a bearish close through it, which from a
    bullish state is an MSS - and that MSS establishes a bearish assignment and
    flips the bias, so this function moves on by itself. From a bearish state
    the bias already disagreed with the bullish assignment. Either way the
    stale one cannot be returned.

    What this never does is quietly promote a newer active swing that no event
    has established. That swing may well be the one an MSS would break next; it
    is still not protected, and saying so would collapse the distinction this
    module exists to draw.
    """
    if bias is StructureBias.NEUTRAL:
        return None

    wanted = BreakDirection.BULLISH if bias is StructureBias.BULLISH else BreakDirection.BEARISH
    found: ProtectedSwingAssignment | None = None
    for entry in assignments:
        if entry.event_direction is wanted:
            found = entry
    return found


def _assignment_for(
    event: StructureBreak, *, symbol: str, timeframe: Timeframe
) -> ProtectedSwingAssignment | None:
    """The anchor this event established, or ``None`` if it had none."""
    if (
        event.opposite_active_swing_id is None
        or event.opposite_active_price is None
        or event.opposite_active_pivot_time is None
        or event.opposite_active_confirmed_at is None
    ):
        return None

    # The structure engine resolves the opposite side from swings confirmed
    # strictly before the break bar closed. Asserted rather than re-checked:
    # duplicating the filter would create a second rule that could drift from
    # the first, and this is the invariant that matters.
    if not event.opposite_active_confirmed_at < event.break_bar_close_time:
        raise ProtectedStructureError(
            f"event {event.event_id} names an opposite swing confirmed at "
            f"{event.opposite_active_confirmed_at.isoformat()}, which is not strictly before "
            f"its close at {event.break_bar_close_time.isoformat()}"
        )

    return ProtectedSwingAssignment(
        assignment_id=_digest(
            symbol,
            timeframe.value,
            PROTECTED_METHOD_VERSION,
            protected_type_for(event.direction).value,
            event.opposite_active_swing_id,
            event.event_id,
        ),
        method_version=PROTECTED_METHOD_VERSION,
        timeframe=timeframe,
        symbol=symbol,
        protected_type=protected_type_for(event.direction),
        swing_id=event.opposite_active_swing_id,
        swing_price=event.opposite_active_price,
        swing_pivot_time=event.opposite_active_pivot_time,
        swing_confirmed_at=event.opposite_active_confirmed_at,
        established_by_event_id=event.event_id,
        established_at=event.break_bar_close_time,
        event_direction=event.direction,
        event_classification=event.classification,
    )


def _leg_for(
    event: StructureBreak,
    *,
    assignment: ProtectedSwingAssignment,
    bars_by_open: dict[datetime, OHLCBar],
    symbol: str,
    timeframe: Timeframe,
) -> StructuralLeg:
    """The leg this event drew, from its anchor to the break bar's extreme."""
    bar = bars_by_open.get(event.break_bar_open_time)
    if bar is None:
        # Fail closed. The alternative - assuming high == low == close - would
        # silently understate every leg, and it would do so invisibly.
        raise ProtectedStructureError(
            f"event {event.event_id} names a break bar opening at "
            f"{event.break_bar_open_time.isoformat()}, which is not in this series"
        )

    terminal = bar.high if event.direction is BreakDirection.BULLISH else bar.low
    origin = assignment.swing_price

    valid = terminal > origin if event.direction is BreakDirection.BULLISH else terminal < origin
    if not valid:
        # Swapping them to make a tidy interval would turn a contradiction into
        # a plausible-looking number, which is the worse outcome.
        raise ProtectedStructureError(
            f"event {event.event_id} is {event.direction.value} but its terminal {terminal} "
            f"does not lie beyond its origin {origin}"
        )

    return StructuralLeg(
        leg_id=_digest(
            symbol,
            timeframe.value,
            PROTECTED_METHOD_VERSION,
            event.event_id,
            assignment.assignment_id,
            event.direction.value,
            event.break_bar_close_time.isoformat(),
        ),
        method_version=PROTECTED_METHOD_VERSION,
        timeframe=timeframe,
        symbol=symbol,
        event_id=event.event_id,
        event_classification=event.classification,
        direction=event.direction,
        protected_assignment_id=assignment.assignment_id,
        origin_swing_id=assignment.swing_id,
        origin_price=origin,
        origin_pivot_time=assignment.swing_pivot_time,
        origin_confirmed_at=assignment.swing_confirmed_at,
        terminal_bar_open_time=bar.timestamp,
        terminal_bar_close_time=event.break_bar_close_time,
        terminal_price=terminal,
        formed_at=event.break_bar_close_time,
    )


__all__ = [
    "PROTECTED_METHOD_VERSION",
    "ProtectedStructureAnalysis",
    "ProtectedStructureError",
    "ProtectedSwingAssignment",
    "ProtectedSwingType",
    "StructuralLeg",
    "analyse_protected_structure",
    "analyse_snapshot_protected_structure",
    "current_assignment",
    "protected_type_for",
]
