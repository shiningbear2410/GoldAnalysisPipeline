"""Which swing a break anchored on, and the leg it drew.

Round 6.6c.3a §42-§43, §45. Small hand-checkable fixtures; the long realistic
path is in ``test_ict_protected_fixture.py``.

The bar vocabulary is Round 6.6b's, because these fixtures have to produce real
structure events before there is anything to anchor. Prices are gold-shaped and
mean nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.services.ict_protected import (
    PROTECTED_METHOD_VERSION,
    ProtectedStructureAnalysis,
    ProtectedStructureError,
    ProtectedSwingType,
    analyse_protected_structure,
    current_assignment,
    protected_type_for,
)
from goldpipeline.services.ict_structure import (
    BreakClassification,
    BreakDirection,
    StructureBias,
    StructureBreak,
    _event_id,
    analyse_structure,
)
from tests.test_ict_structure import (
    FLAT,
    START,
    Row,
    bar_index,
    peak,
    series,
    trough,
)

HOUR = timedelta(hours=1)


def closing_bar(moment: datetime) -> int:
    """Index of the bar that *closed* at *moment*.

    ``bar_index`` maps an open time, which is what Round 6.6b's own assertions
    needed. Most timestamps in this round are closes - a confirmation, an
    establishment, a leg's formation - so they get their own name rather than a
    ``- 1`` scattered through the file.
    """
    return bar_index(moment) - 1


def wicked(close: str, high: str, low: str) -> Row:
    """A break bar whose extreme runs well past its close.

    The whole point of §43.3: a leg's terminal is the bar's extreme, so a
    fixture whose break bars close at their own high could never tell the two
    apart.
    """
    return (high, low, close)


def protected(rows: Sequence[Row], **kwargs: object) -> ProtectedStructureAnalysis:
    return analyse_protected_structure(series(rows), symbol="XAUUSD", **kwargs)  # type: ignore[arg-type]


def legs_of(rows: Sequence[Row]) -> list[tuple[str, str, str, str]]:
    return [
        (
            leg.direction.value,
            leg.event_classification.value,
            str(leg.origin_price),
            str(leg.terminal_price),
        )
        for leg in protected(rows).legs
    ]


# The smallest series with one bullish break and one opposite low to anchor on.
BULL_ONE: list[Row] = [
    FLAT, FLAT, peak("4030"), FLAT, FLAT, trough("3970"), FLAT, FLAT,
    wicked("4035", "4055", "3990"), wicked("4035", "4055", "3990"), FLAT, FLAT,
]  # fmt: skip

# The mirror: one bearish break with an opposite high to anchor on.
BEAR_ONE: list[Row] = [
    FLAT, FLAT, trough("3970"), FLAT, FLAT, peak("4030"), FLAT, FLAT,
    wicked("3965", "4010", "3945"), wicked("3965", "4010", "3945"), FLAT, FLAT,
]  # fmt: skip


# --------------------------------------------------------------------------
# §45: the structure field this round added
# --------------------------------------------------------------------------


def test_the_opposite_active_field_matches_the_event_time_active_swing() -> None:
    """The field records event-time state, not a later re-derivation.

    A bullish break's opposite is the active swing low, and a bearish break's is
    the active swing high. Checked here against the swings the structure engine
    itself reports, filtered the way the engine filters them.
    """
    result = analyse_structure(series(BULL_ONE), symbol="XAUUSD")
    (event,) = result.breaks

    assert event.direction is BreakDirection.BULLISH
    assert event.opposite_active_swing_id is not None
    assert event.opposite_active_swing_id.endswith("T05:00:00+00:00")
    assert "SWING_LOW" in event.opposite_active_swing_id
    assert event.opposite_active_price == Decimal("3970")


def test_the_opposite_swing_was_confirmed_strictly_before_the_break() -> None:
    """§33. The same known-before rule the broken swing obeys."""
    for rows in (BULL_ONE, BEAR_ONE):
        for event in analyse_structure(series(rows), symbol="XAUUSD").breaks:
            if event.opposite_active_confirmed_at is None:
                continue
            assert event.opposite_active_confirmed_at < event.break_bar_close_time


def test_the_opposite_swing_was_unconsumed_at_the_event() -> None:
    """It cannot be a level an earlier break already took out."""
    result = analyse_structure(series(BULL_ONE), symbol="XAUUSD")
    (event,) = result.breaks

    assert event.opposite_active_swing_id not in {
        earlier.broken_swing_id for earlier in result.breaks
    }


def test_the_new_field_takes_no_part_in_the_event_id() -> None:
    """§2, §45. Adding it changed no Round 6.6b identity.

    Proved by rebuilding each id from only the six fields the preimage has
    always used. If the opposite-swing fields had crept in, these would differ.
    """
    result = analyse_structure(series(BULL_ONE), symbol="XAUUSD")

    for event in result.breaks:
        assert event.event_id == _event_id(
            symbol=event.symbol,
            timeframe=event.timeframe,
            direction=event.direction,
            classification=event.classification,
            broken=event.broken_swing_id,
            break_bar_close_time=event.break_bar_close_time,
        )


def test_the_new_field_changed_no_classification_bias_or_consumption() -> None:
    """The 6.6b semantics this round promised not to touch, restated here.

    ``test_ict_structure_fixture`` already pins the journey's classifications,
    levels and biases and still passes unchanged; this asserts the same
    properties on the fixtures that exercise the new field directly.
    """
    result = analyse_structure(series(BULL_ONE), symbol="XAUUSD")
    (event,) = result.breaks

    assert event.classification is BreakClassification.INITIAL_BREAK
    assert event.prior_bias is StructureBias.NEUTRAL
    assert event.resulting_bias is StructureBias.BULLISH
    assert event.broken_level == Decimal("4030")
    assert result.consumed_swing_ids == {event.broken_swing_id}


# --------------------------------------------------------------------------
# §3-§6: what a protected swing is
# --------------------------------------------------------------------------


def test_the_protected_type_enum_is_geometric() -> None:
    assert {member.value for member in ProtectedSwingType} == {"LOW", "HIGH"}


def test_a_bullish_break_protects_a_low_and_a_bearish_one_a_high() -> None:
    assert protected_type_for(BreakDirection.BULLISH) is ProtectedSwingType.LOW
    assert protected_type_for(BreakDirection.BEARISH) is ProtectedSwingType.HIGH


def test_a_bullish_initial_break_establishes_the_opposite_low() -> None:
    """§42.1."""
    (assignment,) = protected(BULL_ONE).assignments

    assert assignment.protected_type is ProtectedSwingType.LOW
    assert assignment.swing_price == Decimal("3970")
    assert assignment.event_classification is BreakClassification.INITIAL_BREAK
    assert assignment.event_direction is BreakDirection.BULLISH


def test_a_bearish_initial_break_establishes_the_opposite_high() -> None:
    """§42.2."""
    (assignment,) = protected(BEAR_ONE).assignments

    assert assignment.protected_type is ProtectedSwingType.HIGH
    assert assignment.swing_price == Decimal("4030")
    assert assignment.event_direction is BreakDirection.BEARISH


def test_protection_begins_at_the_break_not_at_confirmation() -> None:
    """§9, §44 B. The swing existed and was confirmed long before it was protected."""
    (assignment,) = protected(BULL_ONE).assignments
    result = analyse_structure(series(BULL_ONE), symbol="XAUUSD")
    (event,) = result.breaks

    assert assignment.swing_confirmed_at < assignment.established_at
    assert assignment.established_at == event.break_bar_close_time
    assert closing_bar(assignment.swing_confirmed_at) == 7
    assert closing_bar(assignment.established_at) == 8


# --------------------------------------------------------------------------
# §7: no anchor, no fabrication
# --------------------------------------------------------------------------

NO_OPPOSITE_LOW: list[Row] = [
    FLAT, FLAT, peak("4030"), FLAT, FLAT, wicked("4035", "4055", "3990"),
    wicked("4035", "4055", "3990"), FLAT, FLAT,
]  # fmt: skip

NO_OPPOSITE_HIGH: list[Row] = [
    FLAT, FLAT, trough("3970"), FLAT, FLAT, wicked("3965", "4010", "3945"),
    wicked("3965", "4010", "3945"), FLAT, FLAT,
]  # fmt: skip


def test_a_bullish_break_with_no_opposite_low_establishes_nothing() -> None:
    """§42.3. The event stays valid; it simply anchors nothing."""
    structure = analyse_structure(series(NO_OPPOSITE_LOW), symbol="XAUUSD")
    result = protected(NO_OPPOSITE_LOW)

    assert len(structure.breaks) == 1, "the break itself is unaffected"
    assert structure.breaks[0].opposite_active_swing_id is None
    assert result.assignments == ()
    assert result.legs == ()
    assert result.current_protected_assignment_id is None
    assert result.current_leg_id is None


def test_a_bearish_break_with_no_opposite_high_establishes_nothing() -> None:
    """§42.4."""
    result = protected(NO_OPPOSITE_HIGH)

    assert analyse_structure(series(NO_OPPOSITE_HIGH), symbol="XAUUSD").breaks
    assert result.assignments == ()
    assert result.legs == ()


def test_nothing_is_substituted_for_a_missing_anchor() -> None:
    """No break-bar low, no previous candle, no lowest-of-N.

    The bar that broke structure had a low of 3990 and there is a candle low of
    3990 all over this fixture. Neither becomes an origin, because neither is a
    confirmed swing the market turned at.
    """
    result = protected(NO_OPPOSITE_LOW)

    assert result.assignments == ()
    assert not any(leg.origin_price == Decimal("3990") for leg in result.legs)


# --------------------------------------------------------------------------
# §10-§11: reaffirmation, and a newer swing taking over
# --------------------------------------------------------------------------

REAFFIRM: list[Row] = [
    FLAT, FLAT, peak("4030"), FLAT, FLAT, trough("3970"), FLAT, FLAT,
    wicked("4035", "4055", "3990"), wicked("4035", "4055", "3990"), FLAT, FLAT,
    peak("4045"), FLAT, FLAT,
    wicked("4050", "4070", "3990"), wicked("4050", "4070", "3990"), FLAT, FLAT,
]  # fmt: skip


def test_two_bullish_events_may_reaffirm_the_same_low() -> None:
    """§10, §42.5. One assignment per event, both naming one swing.

    Kept as two records rather than one mutable state because the causal fact -
    "this event anchored here" - is what a later reviewer needs, and it is
    immutable. A single state with a reaffirmation list would lose which event
    drew which leg.
    """
    result = protected(REAFFIRM)
    first, second = result.assignments

    assert first.swing_id == second.swing_id
    assert first.swing_price == second.swing_price == Decimal("3970")
    assert first.assignment_id != second.assignment_id
    assert first.established_by_event_id != second.established_by_event_id
    assert closing_bar(first.established_at) == 8
    assert closing_bar(second.established_at) == 15


def test_a_reaffirmed_anchor_still_draws_two_different_legs() -> None:
    """§24. Same origin, different events, different observed expansions."""
    result = protected(REAFFIRM)
    first, second = result.legs

    assert first.origin_swing_id == second.origin_swing_id
    assert first.origin_price == second.origin_price
    assert first.terminal_price != second.terminal_price
    assert first.leg_id != second.leg_id


def test_a_newer_confirmed_low_does_not_become_protected_by_itself() -> None:
    """§11, §42.6, §44 B. Active is not protected.

    The 3980 low confirms and becomes the structure engine's ``active_low`` -
    the swing a bearish close would break, and therefore the one that could
    trigger an MSS. It is not protected, because no event has established it.
    """
    rows = [*REAFFIRM, FLAT, FLAT, trough("3980"), FLAT, FLAT, FLAT]
    structure = analyse_structure(series(rows), symbol="XAUUSD")
    result = protected(rows)

    assert structure.active_low is not None
    assert structure.active_low.swing.price == Decimal("3980"), "active moved"

    current = result.current_assignment
    assert current is not None
    assert current.swing_price == Decimal("3970"), "protected did not"
    assert all(a.swing_price == Decimal("3970") for a in result.assignments)


def test_the_next_bullish_break_establishes_the_newer_low() -> None:
    """§11, §42.7. Protection starts at that break, not at the low's confirmation."""
    rows = [
        *REAFFIRM, FLAT, FLAT, trough("3980"), FLAT, FLAT, FLAT, FLAT,
        peak("4060"), FLAT, FLAT,
        wicked("4065", "4085", "3990"), wicked("4065", "4085", "3990"), FLAT, FLAT,
    ]  # fmt: skip
    result = protected(rows)

    assert [str(a.swing_price) for a in result.assignments] == ["3970", "3970", "3980"]
    newest = result.assignments[-1]
    assert closing_bar(newest.established_at) == 29
    assert newest.swing_confirmed_at < newest.established_at


# --------------------------------------------------------------------------
# §12, §14: active versus protected, and the current anchor
# --------------------------------------------------------------------------


def test_the_protected_swing_and_the_mss_trigger_may_differ() -> None:
    """§12, stated as a test so nobody later assumes they are the same.

    Round 6.6b's MSS rule is untouched: an opposite close through the *current
    active* swing is an MSS. After a newer low confirms, that swing is no longer
    the protected one, and this module does not pretend otherwise.
    """
    rows = [*REAFFIRM, FLAT, FLAT, trough("3980"), FLAT, FLAT, FLAT]
    structure = analyse_structure(series(rows), symbol="XAUUSD")
    result = protected(rows)
    current = result.current_assignment

    assert structure.active_low is not None
    assert current is not None
    assert structure.active_low.swing.price != current.swing_price
    assert structure.active_low.swing_id != current.swing_id


def test_a_neutral_market_has_no_current_anchor() -> None:
    rows = [FLAT, FLAT, peak("4030"), FLAT, FLAT, trough("3970"), FLAT, FLAT]
    result = protected(rows)

    assert analyse_structure(series(rows), symbol="XAUUSD").current_bias is StructureBias.NEUTRAL
    assert result.assignments == ()
    assert result.current_protected_assignment_id is None


def test_the_current_anchor_is_the_latest_matching_the_bias() -> None:
    result = protected(REAFFIRM)
    current = result.current_assignment

    assert current is not None
    assert current is result.assignments[-1]
    assert current.event_direction is BreakDirection.BULLISH


def test_current_derivation_ignores_assignments_of_the_wrong_direction() -> None:
    """Exercised directly, so the rule is legible without a fixture."""
    result = protected(REAFFIRM)
    bullish = list(result.assignments)

    assert current_assignment(bullish, StructureBias.BULLISH) is bullish[-1]
    assert current_assignment(bullish, StructureBias.BEARISH) is None
    assert current_assignment(bullish, StructureBias.NEUTRAL) is None
    assert current_assignment([], StructureBias.BULLISH) is None


def test_the_current_leg_belongs_to_the_current_assignment() -> None:
    result = protected(REAFFIRM)

    assert result.current_leg is not None
    assert result.current_leg.protected_assignment_id == result.current_protected_assignment_id


# --------------------------------------------------------------------------
# §15-§20: the structural leg
# --------------------------------------------------------------------------


def test_a_bullish_leg_runs_from_the_protected_low_to_the_break_bar_high() -> None:
    """§17-§18, §43.1."""
    (leg,) = protected(BULL_ONE).legs

    assert leg.direction is BreakDirection.BULLISH
    assert leg.origin_price == Decimal("3970")
    assert leg.terminal_price == Decimal("4055")
    assert leg.span == Decimal("85")


def test_a_bearish_leg_runs_from_the_protected_high_to_the_break_bar_low() -> None:
    """§43.2."""
    (leg,) = protected(BEAR_ONE).legs

    assert leg.direction is BreakDirection.BEARISH
    assert leg.origin_price == Decimal("4030")
    assert leg.terminal_price == Decimal("3945")


def test_the_terminal_is_the_bar_extreme_and_not_its_close() -> None:
    """§18, §43.3. The distinction the ``wicked`` fixtures exist to prove."""
    structure = analyse_structure(series(BULL_ONE), symbol="XAUUSD")
    (event,) = structure.breaks
    (leg,) = protected(BULL_ONE).legs

    assert event.break_close == Decimal("4035")
    assert leg.terminal_price == Decimal("4055")
    assert leg.terminal_price > event.break_close


def test_the_terminal_is_not_the_broken_level() -> None:
    structure = analyse_structure(series(BULL_ONE), symbol="XAUUSD")
    (leg,) = protected(BULL_ONE).legs

    assert structure.breaks[0].broken_level == Decimal("4030")
    assert leg.terminal_price != structure.breaks[0].broken_level


def test_a_leg_carries_its_terminal_bar_and_its_origin_swing() -> None:
    (leg,) = protected(BULL_ONE).legs
    (assignment,) = protected(BULL_ONE).assignments

    assert bar_index(leg.terminal_bar_open_time) == 8
    assert leg.terminal_bar_close_time == leg.formed_at
    assert leg.origin_swing_id == assignment.swing_id
    assert leg.origin_pivot_time == assignment.swing_pivot_time
    assert leg.origin_confirmed_at == assignment.swing_confirmed_at
    assert leg.protected_assignment_id == assignment.assignment_id


def test_a_leg_is_formed_at_the_break_close() -> None:
    """§22. Before that close there is no confirmed leg."""
    structure = analyse_structure(series(BULL_ONE), symbol="XAUUSD")
    (leg,) = protected(BULL_ONE).legs

    assert leg.formed_at == structure.breaks[0].break_bar_close_time


def test_an_initial_break_leg_is_still_called_an_initial_break() -> None:
    """§26. Having a leg does not promote the event to a BOS."""
    (leg,) = protected(BULL_ONE).legs

    assert leg.event_classification is BreakClassification.INITIAL_BREAK


def test_no_leg_exists_without_a_protected_origin() -> None:
    """§43.9."""
    assert protected(NO_OPPOSITE_LOW).legs == ()


# --------------------------------------------------------------------------
# §19-§20: failing closed
# --------------------------------------------------------------------------


def fake_break(
    *,
    direction: BreakDirection = BreakDirection.BULLISH,
    break_bar_open_time: datetime | None = None,
    opposite_price: str = "3970",
) -> StructureBreak:
    """A hand-built event, for the paths a valid candle series cannot reach."""
    opened = START + HOUR * 8 if break_bar_open_time is None else break_bar_open_time
    return StructureBreak(
        event_id="event",
        method_version="1.0.0",
        timeframe=Timeframe.H1,
        symbol="XAUUSD",
        direction=direction,
        classification=BreakClassification.INITIAL_BREAK,
        broken_swing_id="H1:SWING_HIGH:x",
        broken_level=Decimal("4030"),
        broken_swing_pivot_time=START + HOUR * 2,
        broken_swing_confirmed_at=START + HOUR * 5,
        break_bar_open_time=opened,
        break_bar_close_time=opened + HOUR,
        break_close=Decimal("4035"),
        prior_bias=StructureBias.NEUTRAL,
        resulting_bias=StructureBias.BULLISH,
        opposite_active_swing_id="H1:SWING_LOW:y",
        opposite_active_price=Decimal(opposite_price),
        opposite_active_pivot_time=START + HOUR * 5,
        opposite_active_confirmed_at=START + HOUR * 7,
    )


def analysis_with(event: StructureBreak, rows: Sequence[Row]) -> ProtectedStructureAnalysis:
    """Run the protected engine over *rows* with one substituted event."""
    snapshot = series(rows)
    real = analyse_structure(snapshot, symbol="XAUUSD")
    from dataclasses import replace

    doctored = replace(real, breaks=(event,))
    return analyse_protected_structure(snapshot, structure=doctored, symbol="XAUUSD")


def test_a_break_bar_that_is_not_in_the_series_fails_closed() -> None:
    """§19, §43.13. Assuming high == low == close would understate every leg."""
    with pytest.raises(ProtectedStructureError, match="not in this series"):
        analysis_with(fake_break(break_bar_open_time=START + HOUR * 500), BULL_ONE)


def test_a_bullish_leg_whose_terminal_is_not_above_its_origin_fails() -> None:
    """§20, §43.15. Swapping them would turn a contradiction into a tidy number."""
    with pytest.raises(ProtectedStructureError, match="does not lie beyond its origin"):
        analysis_with(fake_break(opposite_price="9999"), BULL_ONE)


def test_a_bearish_leg_whose_terminal_is_not_below_its_origin_fails() -> None:
    """§43.16."""
    with pytest.raises(ProtectedStructureError, match="does not lie beyond its origin"):
        analysis_with(fake_break(direction=BreakDirection.BEARISH, opposite_price="1"), BEAR_ONE)


def test_an_opposite_swing_confirmed_at_the_break_close_is_refused() -> None:
    """§34. Event-time origin and end-of-bar reported state are different questions.

    Enforced as an invariant rather than a second filter: the structure engine
    already resolves the opposite side from swings confirmed strictly before the
    close, so a violation would mean that guarantee had broken, and this refuses
    rather than quietly anchoring on it.
    """
    from dataclasses import replace

    event = fake_break()
    same_close = replace(event, opposite_active_confirmed_at=event.break_bar_close_time)

    with pytest.raises(ProtectedStructureError, match="not strictly before"):
        analysis_with(same_close, BULL_ONE)


def test_a_structure_analysis_for_another_instant_is_refused() -> None:
    """Fail closed rather than trim: a supplied analysis carries a bias too."""
    snapshot = series(REAFFIRM)
    full = analyse_structure(snapshot, symbol="XAUUSD")

    with pytest.raises(ProtectedStructureError, match="supply one computed for the same instant"):
        analyse_protected_structure(
            snapshot, structure=full, symbol="XAUUSD", as_of=START + HOUR * 10
        )


def test_a_matching_structure_analysis_is_accepted() -> None:
    """§49-style reuse: a composite stage computes structure once."""
    snapshot = series(REAFFIRM)
    computed = analyse_structure(snapshot, symbol="XAUUSD")

    supplied = analyse_protected_structure(snapshot, structure=computed, symbol="XAUUSD")
    derived = analyse_protected_structure(snapshot, symbol="XAUUSD")

    assert supplied == derived


# --------------------------------------------------------------------------
# §23, §35: identity
# --------------------------------------------------------------------------


def test_identities_are_derived_and_stable() -> None:
    one = protected(REAFFIRM)
    two = protected(REAFFIRM)

    assert [a.assignment_id for a in one.assignments] == [a.assignment_id for a in two.assignments]
    assert [leg.leg_id for leg in one.legs] == [leg.leg_id for leg in two.legs]
    assert all(len(a.assignment_id) == 16 for a in one.assignments)
    assert all(len(leg.leg_id) == 16 for leg in one.legs)


def test_a_different_symbol_gives_different_identities() -> None:
    gold = analyse_protected_structure(series(REAFFIRM), symbol="XAUUSD")
    other = analyse_protected_structure(series(REAFFIRM), symbol="XAGUSD")

    assert [a.assignment_id for a in gold.assignments] != [
        a.assignment_id for a in other.assignments
    ]
    assert [str(a.swing_price) for a in gold.assignments] == [
        str(a.swing_price) for a in other.assignments
    ]


def test_no_price_participates_in_either_identity() -> None:
    """§49. Both are built from a swing id and an event id, neither numeric.

    So the Decimal-canonicalisation care the liquidity and gap identities need
    has nothing to do here, and no fourth copy of that logic was introduced.
    """
    import inspect

    from goldpipeline.services import ict_protected

    source = inspect.getsource(ict_protected._digest)
    for caller in (ict_protected._assignment_for, ict_protected._leg_for):
        body = inspect.getsource(caller)
        assert "_digest(" in body
    assert "normalize" not in source
    for assignment in protected(REAFFIRM).assignments:
        assert str(assignment.swing_price) not in assignment.assignment_id


def test_the_method_version_is_stamped_everywhere() -> None:
    result = protected(REAFFIRM)

    assert result.method_version == PROTECTED_METHOD_VERSION
    assert all(a.method_version == PROTECTED_METHOD_VERSION for a in result.assignments)
    assert all(leg.method_version == PROTECTED_METHOD_VERSION for leg in result.legs)


def test_prices_are_exact_decimals() -> None:
    """§42.14."""
    for assignment in protected(REAFFIRM).assignments:
        assert isinstance(assignment.swing_price, Decimal)
    for leg in protected(REAFFIRM).legs:
        assert isinstance(leg.origin_price, Decimal)
        assert isinstance(leg.terminal_price, Decimal)
        assert isinstance(leg.span, Decimal)
