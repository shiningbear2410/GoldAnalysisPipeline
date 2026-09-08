"""Market structure: what the swings say, and when a close said otherwise.

Round 6.6b. This is the first ICT module that draws a conclusion. The
primitives in :mod:`goldpipeline.services.ict_primitives` report shapes - a
pivot is a pivot, a gap is a gap - and stop there. Structure is where those
shapes become a *state*: bullish, bearish, or not yet either, plus the specific
closes that moved it.

**Why this had to come before order blocks and zones.** An order block is
defined relative to an impulse that broke structure, a liquidity sweep is
defined against a level structure says matters, and a candidate entry is
defined by which side of the range structure puts price on. Every one of those
is downstream of "was that a break, and in which direction". Building them on a
vague break definition means the wrongness arrives already laundered into a
published price level, where nobody can trace it back. So the break definition
is written out here in full, in code, with no thresholds hidden in it.

**A close, or nothing.** A structure break happens when a *closed* candle
*closes* strictly beyond a confirmed swing. Not a wick through it, not a touch
of it, not "close enough". A wick beyond a level and back is a real and
interesting event - it is where liquidity gets taken - but it is a different
event with a different name, and it belongs to the liquidity round. Here it
leaves no trace at all, and the level stays live.

**Known before broken.** The swing being broken must have been confirmed
*strictly before* the breaking bar closed. Without the strictness a single
candle could both complete the pivot that makes a level knowable and be
credited with breaking it, which is a level that never existed unbroken. That
one character is most of this module's no-lookahead guarantee.

**Structure state is not a trade bias.** :class:`StructureBias` says what the
geometry has established. It does not say SEO, BAI, buy or sell, and no future
stage should read it as though it did: an analyst with macro context is
entitled to disagree with market structure, and the naming keeps that
disagreement expressible rather than hidden behind a shared word.

**Nothing here consults an ATR, a gap, a volume, a headline, or another
timeframe.** Each of those is a real refinement and each is deliberately absent.
A displacement filter or an ATR-fraction minimum would be a subjective
confirmation threshold buried in the definition of truth; if a later round wants
to rank breaks by strength it can, with the raw event in hand and its own filter
named as its own. See ``tests/test_ict_structure_guards.py``, which pins every
one of those absences.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Sequence
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
from goldpipeline.services.ict_primitives import (
    DEFAULT_LEFT_BARS,
    DEFAULT_RIGHT_BARS,
    SwingPoint,
    SwingType,
    confirmed_swings,
)

logger = logging.getLogger(__name__)

STRUCTURE_METHOD_VERSION = "1.0.0"
"""Recorded on every analysis.

Bumped whenever a definition below changes meaning, so an event stored by one
version is never read as though a later one produced it.
"""


# --------------------------------------------------------------------------
# swing relations
# --------------------------------------------------------------------------


class SwingRelation(StrEnum):
    """How a confirmed swing compares with the previous confirmed swing of its own type.

    Four values rather than eight. "Higher high" is exactly the pair
    ``(SwingType.SWING_HIGH, SwingRelation.HIGHER)``, and the swing already
    carries its type, so spelling the side twice would only create a way for the
    two copies to disagree. :attr:`AnnotatedSwing.label` renders the familiar
    eight-way name when one is wanted for reading.

    ``EQUAL`` is a legitimate outcome, not a failure to classify. Two swing
    highs at the same price, far enough apart that each dominates its own pivot
    window, are each a real confirmed swing - unlike two *adjacent* equal highs,
    which make no pivot at all under the strict-both-sides rule in
    :func:`~goldpipeline.services.ict_primitives.confirmed_swings`. What an
    equal pair means - resting orders above a double top - is a liquidity
    reading, and this module does not make it.
    """

    FIRST = "FIRST"
    HIGHER = "HIGHER"
    LOWER = "LOWER"
    EQUAL = "EQUAL"


_HIGH_LABELS = {
    SwingRelation.FIRST: "FIRST_HIGH",
    SwingRelation.HIGHER: "HIGHER_HIGH",
    SwingRelation.LOWER: "LOWER_HIGH",
    SwingRelation.EQUAL: "EQUAL_HIGH",
}
_LOW_LABELS = {
    SwingRelation.FIRST: "FIRST_LOW",
    SwingRelation.HIGHER: "HIGHER_LOW",
    SwingRelation.LOWER: "LOWER_LOW",
    SwingRelation.EQUAL: "EQUAL_LOW",
}


def swing_id(swing: SwingPoint) -> str:
    """Deterministic identity for one confirmed pivot.

    Timeframe, side and pivot open time. Not price: two swings at the same
    Decimal are two different observations of the market, and conflating them
    would let a break of the first mark the second as already broken. The
    triple is unique because a given bar can be at most one swing high and at
    most one swing low.

    Readable on purpose - these appear in event records and in test failures,
    and a hash there would cost an hour every time something needed debugging.
    """
    return f"{swing.timeframe.value}:{swing.swing_type.value}:{swing.pivot_time.isoformat()}"


@dataclass(frozen=True)
class AnnotatedSwing:
    """A confirmed swing, plus how it sits against the previous one of its type."""

    swing: SwingPoint
    relation: SwingRelation
    previous_pivot_time: datetime | None
    """Open time of the swing this was compared against, or ``None`` for the first."""

    @property
    def swing_id(self) -> str:
        return swing_id(self.swing)

    @property
    def label(self) -> str:
        """The conventional eight-way name, e.g. ``"HIGHER_HIGH"``."""
        table = _HIGH_LABELS if self.swing.swing_type is SwingType.SWING_HIGH else _LOW_LABELS
        return table[self.relation]


def annotate_swings(swings: Sequence[SwingPoint]) -> tuple[AnnotatedSwing, ...]:
    """Label each swing against the previous confirmed swing of the same type.

    Chronological, and each side tracked independently: a swing high is
    compared with the swing high before it, never with an intervening low.

    Prices are compared as exact ``Decimal``. There is no tolerance here and
    there should not be - ``4001.00`` against ``4001.00`` is ``EQUAL`` and
    ``4001.01`` against ``4001.00`` is ``HIGHER``, full stop. Tolerance is what
    turns two near-equal highs into one liquidity pool, and that is a different
    engine with a different parameter that deserves to be argued about in its
    own module rather than smuggled in here.
    """
    latest: dict[SwingType, SwingPoint] = {}
    annotated: list[AnnotatedSwing] = []

    for swing in swings:
        previous = latest.get(swing.swing_type)
        if previous is None:
            relation = SwingRelation.FIRST
        elif swing.price > previous.price:
            relation = SwingRelation.HIGHER
        elif swing.price < previous.price:
            relation = SwingRelation.LOWER
        else:
            relation = SwingRelation.EQUAL

        annotated.append(
            AnnotatedSwing(
                swing=swing,
                relation=relation,
                previous_pivot_time=None if previous is None else previous.pivot_time,
            )
        )
        latest[swing.swing_type] = swing

    return tuple(annotated)


# --------------------------------------------------------------------------
# structure state
# --------------------------------------------------------------------------


class StructureBias(StrEnum):
    """What the geometry has established. Three values, and no fourth.

    ``NEUTRAL`` means "structure has not established a direction", which covers
    both too little history and genuinely mixed geometry. It deliberately does
    *not* say ``RANGE``: claiming the market is ranging is a positive assertion
    about market condition, and this engine has not earned it - a broadening
    formation and a quiet consolidation both land here and they are not the
    same thing.

    Note the difference from
    :class:`~goldpipeline.schemas.context.MarketStructure`, which serves the
    ANALYSIS product and has four values including ``RANGE`` and
    ``INSUFFICIENT_DATA``. That one is a label recomputed from the last two
    pivots each time it is asked. This one is a *state* that persists until a
    close moves it, which is why the two cannot be merged even though their
    bootstrap arithmetic agrees.
    """

    NEUTRAL = "NEUTRAL"
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


def bootstrap_bias(annotated: Sequence[AnnotatedSwing]) -> StructureBias:
    """The state implied by swing geometry alone, before any break.

    Both sides must agree, and both must be strict:

    * latest high ``HIGHER`` **and** latest low ``HIGHER`` - ``BULLISH``
    * latest high ``LOWER`` **and** latest low ``LOWER`` - ``BEARISH``
    * anything else, including ``FIRST`` and ``EQUAL`` - ``NEUTRAL``

    One swing establishes nothing, which is why a lone ``FIRST`` on either side
    falls through to ``NEUTRAL`` rather than seeding a direction. A market
    making higher highs while making lower lows is broadening, and calling that
    bullish because half the pattern fits is exactly the confident wrongness
    this layer exists to avoid.

    Worth recording that this rule was not invented here: the ANALYSIS product's
    :func:`~goldpipeline.services.levels.classify_structure` has used the same
    conjunction since long before the ICT branch existed, arrived at
    independently. Two products reaching the same test is corroboration, and the
    audit for this round found no reason to prefer a different convention. What
    is *not* shared is anything else about it - see :class:`StructureBias`.
    """
    latest_high = _latest_of(annotated, SwingType.SWING_HIGH)
    latest_low = _latest_of(annotated, SwingType.SWING_LOW)

    if latest_high is None or latest_low is None:
        return StructureBias.NEUTRAL

    if latest_high.relation is SwingRelation.HIGHER and latest_low.relation is SwingRelation.HIGHER:
        return StructureBias.BULLISH
    if latest_high.relation is SwingRelation.LOWER and latest_low.relation is SwingRelation.LOWER:
        return StructureBias.BEARISH
    return StructureBias.NEUTRAL


def _latest_of(
    annotated: Iterable[AnnotatedSwing],
    swing_type: SwingType,
    consumed: frozenset[str] = frozenset(),
) -> AnnotatedSwing | None:
    """The most recent swing of *swing_type* that no break has consumed."""
    found: AnnotatedSwing | None = None
    for entry in annotated:
        if entry.swing.swing_type is swing_type and entry.swing_id not in consumed:
            found = entry
    return found


def eligible_before(
    annotated: Sequence[AnnotatedSwing], close_time: datetime
) -> tuple[AnnotatedSwing, ...]:
    """The swings a bar closing at *close_time* is allowed to break.

    **Strictly** before, and the strictness is the whole point. With ``<=`` a
    single candle could complete the pivot that makes a level knowable *and* be
    credited with breaking it - a level that was never once observed unbroken.
    An engine that does that reports structure the market never showed.

    Note this is deliberately stricter than
    :func:`~goldpipeline.services.ict_primitives.swings_known_at`, which is
    inclusive because it answers a different question: what a stage reasoning
    "as of T" may *see*. Seeing a level and being allowed to break it are not
    the same permission, and the two rules are separate so neither has to be
    bent to serve the other.

    Given the current strict-both-sides pivot definition this rule is
    belt-and-braces: a bar closing beyond a pivot necessarily has a high above
    it (or a low below it), which would have destroyed that pivot had the bar
    been inside its right-hand window. It is written and enforced anyway,
    because that reasoning depends on a pivot definition a later round is free
    to change, and the guarantee should not quietly depend on it.
    """
    return tuple(entry for entry in annotated if entry.swing.confirmed_at < close_time)


# --------------------------------------------------------------------------
# break events
# --------------------------------------------------------------------------


class BreakDirection(StrEnum):
    """Which way the close went through the level.

    Not ``BUY``/``SELL``, and not ``SEO``/``BAI``. A direction describes what
    price did to a level. What to do about it is a decision nothing in this
    module is entitled to make.
    """

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


class BreakClassification(StrEnum):
    """What the break meant, given the state it happened in.

    ``INITIAL_BREAK`` exists so the first directional event in a neutral market
    is not mislabelled. Calling it a BOS would claim a structure was continuing
    when nothing had established one; calling it an MSS would claim a reversal
    of a direction that was never held.
    """

    INITIAL_BREAK = "INITIAL_BREAK"
    BOS = "BOS"
    MSS = "MSS"


MARKET_STRUCTURE_SHIFT = BreakClassification.MSS
"""The canonical long name, recorded once.

**CHOCH is deliberately not a separate event type.** Across ICT material the
word is used for the first counter-trend break, sometimes for any counter-trend
break, and sometimes as an exact synonym for MSS - three incompatible
definitions sharing one name. Introducing it here would mean picking one
silently and having every later stage inherit the ambiguity. There is one
reversal concept in this engine, it is ``MSS``, and if product copy eventually
wants the word CHOCH it can be derived from these events by a rule written down
at that point.
"""


@dataclass(frozen=True)
class StructureBreak:
    """One close, through one level, and everything needed to check it later.

    Deliberately over-specified. Every field here is something a future
    reviewer, a candidate-zone engine, or a person arguing with the output would
    otherwise have to recover by re-running the analysis: the level's own
    identity and timing, the breaking bar's open and close, and the state on
    both sides of the transition. No prose explains this event, because prose is
    not checkable.
    """

    event_id: str
    method_version: str

    timeframe: Timeframe
    symbol: str

    direction: BreakDirection
    classification: BreakClassification

    broken_swing_id: str
    broken_level: Decimal
    broken_swing_pivot_time: datetime
    broken_swing_confirmed_at: datetime

    break_bar_open_time: datetime
    break_bar_close_time: datetime
    break_close: Decimal

    prior_bias: StructureBias
    resulting_bias: StructureBias

    also_consumed_swing_ids: tuple[str, ...] = ()
    """Older levels on the same side that this close also cleared.

    A close that takes out the most recent swing high, and three lower ones
    behind it, has structurally taken out all four. Only the most recent one is
    news, so only it gets an event - but the others must not be left live, or a
    later bar drifting sideways would break them one at a time and emit a
    parade of stale BOS events. They are recorded here rather than only in the
    analysis-level set so that every consumption can be attributed to the close
    that caused it.
    """


def _event_id(
    *,
    symbol: str,
    timeframe: Timeframe,
    direction: BreakDirection,
    classification: BreakClassification,
    broken: str,
    break_bar_close_time: datetime,
) -> str:
    """A stable id, derived from the event rather than assigned to it.

    The preimage is exactly these six fields joined by ``|``, in this order,
    hashed with SHA-256 and truncated to 16 hex characters. No clock, no
    counter, no random source: replaying identical bars produces identical ids,
    which is what lets two runs of this engine be diffed at all.

    The preimage excludes provider and provider symbol. The same candles from
    TradingView, MetaTrader or a fixture describe the same event, and an id that
    disagreed about that would make provenance look like meaning.
    """
    preimage = "|".join(
        (
            symbol,
            timeframe.value,
            direction.value,
            classification.value,
            broken,
            break_bar_close_time.isoformat(),
        )
    )
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]


_TRANSITIONS: dict[tuple[StructureBias, BreakDirection], BreakClassification] = {
    (StructureBias.NEUTRAL, BreakDirection.BULLISH): BreakClassification.INITIAL_BREAK,
    (StructureBias.NEUTRAL, BreakDirection.BEARISH): BreakClassification.INITIAL_BREAK,
    (StructureBias.BULLISH, BreakDirection.BULLISH): BreakClassification.BOS,
    (StructureBias.BULLISH, BreakDirection.BEARISH): BreakClassification.MSS,
    (StructureBias.BEARISH, BreakDirection.BEARISH): BreakClassification.BOS,
    (StructureBias.BEARISH, BreakDirection.BULLISH): BreakClassification.MSS,
}


def classify_break(
    prior: StructureBias, direction: BreakDirection
) -> tuple[BreakClassification, StructureBias]:
    """What this break is called, and the state it leaves behind.

    The resulting state is always the break's own direction. A confirmed close
    beyond structure is the market's statement, and making the engine wait for
    two more pivots to agree would leave it reporting ``BULLISH`` well after a
    close had said otherwise - stale in exactly the situation where staleness
    costs most.
    """
    resulting = (
        StructureBias.BULLISH if direction is BreakDirection.BULLISH else StructureBias.BEARISH
    )
    return _TRANSITIONS[(prior, direction)], resulting


# --------------------------------------------------------------------------
# the analysis
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StructureAnalysis:
    """One timeframe's structure, as of one instant.

    The smallest thing later rounds need, and nothing more: no candidate zones,
    no ranking, no narrative. A liquidity engine reads the annotated swings; an
    order-block engine reads the breaks and their bars; a plan renderer reads
    the active levels. None of them needs an opinion from here.
    """

    timeframe: Timeframe
    symbol: str
    method_version: str
    observed_at: datetime
    """The instant this analysis is current to - the ``as_of`` asked for, or the
    series' own latest close."""

    initial_bias: StructureBias
    """What geometry alone said, immediately before the first break event.

    Equal to :attr:`current_bias` when no break ever occurred. This is kept
    separate because it is the only record of the bootstrap once the state
    machine has taken over.
    """

    current_bias: StructureBias
    annotated_swings: tuple[AnnotatedSwing, ...]
    breaks: tuple[StructureBreak, ...]
    active_high: AnnotatedSwing | None
    active_low: AnnotatedSwing | None
    consumed_swing_ids: frozenset[str]

    left_bars: int
    right_bars: int
    bars_considered: int

    @property
    def latest_break(self) -> StructureBreak | None:
        return self.breaks[-1] if self.breaks else None

    def breaks_by(self, classification: BreakClassification) -> tuple[StructureBreak, ...]:
        return tuple(event for event in self.breaks if event.classification is classification)


def analyse_structure(
    series: IctTimeframeSnapshot,
    *,
    symbol: str = "",
    as_of: datetime | None = None,
    left_bars: int = DEFAULT_LEFT_BARS,
    right_bars: int = DEFAULT_RIGHT_BARS,
) -> StructureAnalysis:
    """Replay *series* bar by bar and report the structure it built.

    The exact order of operations at each closed bar, because the answers depend
    on it and a reader should not have to infer it:

    1. Collect the swings confirmed **strictly before** this bar's close. This
       is the known-before-broken rule; a swing confirmed *at* this close was
       not available to be broken by it.
    2. Take the most recent unconsumed swing on each side as the active levels.
       Both sides, always - a bullish state still needs its low, because that is
       what a reversal would break.
    3. While no break has yet occurred, recompute the bootstrap state from those
       known swings. After the first break, the state machine owns the state and
       geometry no longer overrides it.
    4. Test the close against the active levels, strictly beyond, and emit at
       most one event.
    5. Mark the broken level - and any older same-side levels the close also
       cleared - consumed, then move the state.

    Afterwards, and only if no break ever fired, the bootstrap is taken once
    more over every swing in hand. Step 1 deliberately hides a swing confirmed
    *by* the final bar, because that swing was not breakable by it; but it is
    knowable now, so it belongs in the state being reported.

    Nothing in this loop iterates a set or a dict in a way that reaches the
    output: swings arrive in chronological order and stay in it, and
    ``consumed`` is only ever asked whether it contains something.

    Args:
        series: Closed candles for one timeframe.
        symbol: Instrument name, e.g. ``"XAUUSD"``. Provenance, and part of the
            event id preimage. Not the provider.
        as_of: Reconstruct the state as it stood at this instant. Bars are
            filtered to those closed by then and swings recomputed on that
            shorter series, so a pivot needing a later bar to confirm is absent
            rather than borrowed from the future.
        left_bars: Bars required strictly below/above a pivot on its left.
        right_bars: The same on its right. Also the confirmation delay.

    Raises:
        ValueError: *as_of* precedes every bar's close, so there is no series to
            analyse. Refused rather than answered with an empty ``NEUTRAL``,
            which would be indistinguishable from a real reading.
    """
    working = series
    if as_of is not None:
        working = build_timeframe_snapshot(
            timeframe=series.timeframe, bars=series.bars, observed_at=as_of
        )
    observed_at = as_of if as_of is not None else series.latest_closed_at

    duration = working.timeframe.duration
    assert duration is not None  # IctTimeframeSnapshot refuses calendar timeframes

    annotated = annotate_swings(
        confirmed_swings(working, left_bars=left_bars, right_bars=right_bars)
    )

    bias = StructureBias.NEUTRAL
    initial_bias: StructureBias | None = None
    consumed: set[str] = set()
    events: list[StructureBreak] = []

    for bar in working.bars:
        close_time = bar.timestamp + duration

        # (1) strictly before: a swing this very bar confirmed is not eligible.
        known = eligible_before(annotated, close_time)
        frozen_consumed = frozenset(consumed)

        # (2) both sides, regardless of state.
        active_high = _latest_of(known, SwingType.SWING_HIGH, frozen_consumed)
        active_low = _latest_of(known, SwingType.SWING_LOW, frozen_consumed)

        # (3) geometry leads until the first close overrules it.
        if not events:
            bias = bootstrap_bias(known)

        # (4) strictly beyond, or nothing.
        candidates: list[tuple[BreakDirection, AnnotatedSwing]] = []
        if active_high is not None and bar.close > active_high.swing.price:
            candidates.append((BreakDirection.BULLISH, active_high))
        if active_low is not None and bar.close < active_low.swing.price:
            candidates.append((BreakDirection.BEARISH, active_low))

        chosen = _resolve_candidates(candidates)
        if chosen is None:
            continue
        direction, broken = chosen

        if initial_bias is None:
            initial_bias = bias
        classification, resulting = classify_break(bias, direction)

        swept = _also_cleared(known, direction, broken, bar.close, frozen_consumed)
        events.append(
            StructureBreak(
                event_id=_event_id(
                    symbol=symbol,
                    timeframe=working.timeframe,
                    direction=direction,
                    classification=classification,
                    broken=broken.swing_id,
                    break_bar_close_time=close_time,
                ),
                method_version=STRUCTURE_METHOD_VERSION,
                timeframe=working.timeframe,
                symbol=symbol,
                direction=direction,
                classification=classification,
                broken_swing_id=broken.swing_id,
                broken_level=broken.swing.price,
                broken_swing_pivot_time=broken.swing.pivot_time,
                broken_swing_confirmed_at=broken.swing.confirmed_at,
                break_bar_open_time=bar.timestamp,
                break_bar_close_time=close_time,
                break_close=bar.close,
                prior_bias=bias,
                resulting_bias=resulting,
                also_consumed_swing_ids=swept,
            )
        )

        # (5)
        consumed.add(broken.swing_id)
        consumed.update(swept)
        bias = resulting

    if not events:
        # The loop's last pass could only see swings confirmed *strictly* before
        # the final close, because that is the break-eligibility rule. What is
        # reported as current state is a different question - what is knowable
        # now - and a swing confirmed by the closing bar is knowable now. So the
        # bootstrap is taken once more over everything in hand, matching the
        # active levels below, which have always been reported that way.
        #
        # Only while no break has fired. After one, the state machine owns the
        # state, and re-deriving it from geometry would let a stale pivot pair
        # overrule a close that already said otherwise.
        bias = bootstrap_bias(annotated)

    frozen_consumed = frozenset(consumed)
    return StructureAnalysis(
        timeframe=working.timeframe,
        symbol=symbol,
        method_version=STRUCTURE_METHOD_VERSION,
        observed_at=observed_at,
        initial_bias=bias if initial_bias is None else initial_bias,
        current_bias=bias,
        annotated_swings=annotated,
        breaks=tuple(events),
        # Reported against everything knowable *now*, not against the last bar's
        # strictly-before set: a swing confirmed by the final close is live for
        # whatever bar comes next, and this is the answer to "what is in play".
        active_high=_latest_of(annotated, SwingType.SWING_HIGH, frozen_consumed),
        active_low=_latest_of(annotated, SwingType.SWING_LOW, frozen_consumed),
        consumed_swing_ids=frozen_consumed,
        left_bars=left_bars,
        right_bars=right_bars,
        bars_considered=working.bar_count,
    )


def analyse_snapshot_structure(
    snapshot: IctMarketSnapshot,
    timeframe: Timeframe,
    *,
    as_of: datetime | None = None,
    left_bars: int = DEFAULT_LEFT_BARS,
    right_bars: int = DEFAULT_RIGHT_BARS,
) -> StructureAnalysis:
    """:func:`analyse_structure` for one timeframe of a multi-timeframe snapshot.

    Each timeframe is analysed entirely on its own. There is deliberately no
    function here that combines them: "H4 bullish and H1 bullish therefore
    bullish" is a synthesis rule with real content - which timeframe wins on
    disagreement, whether H4 gates H1 at all - and inventing one as a convenience
    would ship an unargued trading opinion inside a geometry module.
    """
    return analyse_structure(
        snapshot.require(timeframe),
        symbol=snapshot.symbol,
        as_of=as_of,
        left_bars=left_bars,
        right_bars=right_bars,
    )


def _resolve_candidates(
    candidates: Sequence[tuple[BreakDirection, AnnotatedSwing]],
) -> tuple[BreakDirection, AnnotatedSwing] | None:
    """Pick the single event a bar may emit.

    Almost always there is nothing to pick: with the active low below the active
    high, one close cannot be both above the high and below the low. Both sides
    can only qualify in an inverted arrangement - an unconsumed swing high
    sitting *below* the active swing low - and no valid candle series producing
    one was found while writing this round. The rule is defined anyway, because
    the pivot windows are caller-supplied and "we could not construct it" is not
    "it cannot happen"; an undefined answer would surface as a silent ordering
    dependency rather than as a bug.

    The more recently confirmed level wins. It is the market's more current
    statement about structure, and a level that a newer opposite-side swing has
    already overtaken is not news when price finally crosses it. Ties fall to
    the later pivot, then to the swing id, so the answer never depends on
    argument order.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return max(
        candidates,
        key=lambda item: (
            item[1].swing.confirmed_at,
            item[1].swing.pivot_time,
            item[1].swing_id,
        ),
    )


def _also_cleared(
    known: Sequence[AnnotatedSwing],
    direction: BreakDirection,
    broken: AnnotatedSwing,
    close: Decimal,
    consumed: frozenset[str],
) -> tuple[str, ...]:
    """Older same-side levels this close also went strictly beyond.

    See :attr:`StructureBreak.also_consumed_swing_ids` for why they must not be
    left live. Returned in chronological order, which is the order *known*
    already holds them in.
    """
    side = SwingType.SWING_HIGH if direction is BreakDirection.BULLISH else SwingType.SWING_LOW
    cleared: list[str] = []
    for entry in known:
        if entry.swing.swing_type is not side:
            continue
        if entry.swing_id in consumed or entry.swing_id == broken.swing_id:
            continue
        beyond = (
            close > entry.swing.price
            if direction is BreakDirection.BULLISH
            else close < entry.swing.price
        )
        if beyond:
            cleared.append(entry.swing_id)
    return tuple(cleared)


__all__ = [
    "MARKET_STRUCTURE_SHIFT",
    "STRUCTURE_METHOD_VERSION",
    "AnnotatedSwing",
    "BreakClassification",
    "BreakDirection",
    "StructureAnalysis",
    "StructureBias",
    "StructureBreak",
    "SwingRelation",
    "analyse_snapshot_structure",
    "analyse_structure",
    "annotate_swings",
    "bootstrap_bias",
    "classify_break",
    "eligible_before",
    "swing_id",
]
