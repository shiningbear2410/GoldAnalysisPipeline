"""What price did to an order block, one rule at a time.

Round 6.6d.2 §44-§48, §51. The geometry matrices call the predicates directly on
zones written out in full, because §44 and §45 are claims about intervals rather
than about any particular market path - wrestling a 13-bar fixture into printing
exactly ``[4000, 4010]`` would obscure the very numbers being pinned. The
progression and same-bar matrices then run end to end over real snapshots, where
formation, ordering and the known-before rule all have to hold together.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.services import ict_order_block_lifecycle
from goldpipeline.services.ict_order_block import (
    BodyDirection,
    OrderBlock,
    OrderBlockZoneBasis,
    analyse_order_blocks,
)
from goldpipeline.services.ict_order_block_lifecycle import (
    OrderBlockLifecycleAnalysis,
    OrderBlockLifecycleConfig,
    OrderBlockLifecycleError,
    OrderBlockMitigationRule,
    OrderBlockStatus,
    analyse_order_block_lifecycle,
    contains_midpoint,
    covers,
    far_edge,
    invalidates,
    satisfies,
    status_of,
    touches,
)
from goldpipeline.services.ict_structure import (
    BreakClassification,
    BreakDirection,
    analyse_structure,
)
from tests.test_ict_order_block import BODY, BULL_LEG, FULL, bodied
from tests.test_ict_protected import closing_bar
from tests.test_ict_structure import FLAT, START, Row

HOUR = timedelta(hours=1)

TOUCH = OrderBlockLifecycleConfig(mitigation_rule=OrderBlockMitigationRule.TOUCH)
MIDPOINT = OrderBlockLifecycleConfig(mitigation_rule=OrderBlockMitigationRule.MIDPOINT)
FULL_ZONE = OrderBlockLifecycleConfig(mitigation_rule=OrderBlockMitigationRule.FULL_ZONE)

RULES = (
    OrderBlockMitigationRule.TOUCH,
    OrderBlockMitigationRule.MIDPOINT,
    OrderBlockMitigationRule.FULL_ZONE,
)


def zone_block(
    lower: str, upper: str, direction: BreakDirection = BreakDirection.BULLISH
) -> OrderBlock:
    """An order block occupying exactly ``[lower, upper]``.

    Built directly rather than grown from candles, because the geometry matrices
    are about the interval and nothing else. Everything the lifecycle actually
    reads - ``lower``, ``upper``, ``midpoint``, ``direction``, ``formed_at`` - is
    real; the provenance fields are placeholders and are never consulted here.
    """
    low, high = Decimal(lower), Decimal(upper)
    return OrderBlock(
        order_block_id=f"zone:{lower}-{upper}:{direction.value}",
        method_version="1.0.0",
        timeframe=Timeframe.H1,
        symbol="XAUUSD",
        direction=direction,
        zone_basis=OrderBlockZoneBasis.BODY,
        structure_event_id="event",
        event_classification=BreakClassification.BOS,
        protected_assignment_id="assignment",
        structural_leg_id="leg",
        formed_at=START,
        source_bar_open_time=START - HOUR,
        source_bar_close_time=START,
        source_open=high,
        source_high=high,
        source_low=low,
        source_close=low,
        source_body_direction=BodyDirection.BEARISH,
        lower=low,
        upper=high,
        midpoint=(low + high) / Decimal(2),
        width=high - low,
    )


ZONE = zone_block("4000", "4010")
BEAR_ZONE = zone_block("4000", "4010", BreakDirection.BEARISH)


def lifecycle(
    rows: Sequence[Row],
    opens: Mapping[int, str],
    *,
    config: OrderBlockLifecycleConfig,
    basis: OrderBlockZoneBasis = OrderBlockZoneBasis.FULL_CANDLE,
) -> OrderBlockLifecycleAnalysis:
    formation = FULL if basis is OrderBlockZoneBasis.FULL_CANDLE else BODY
    return analyse_order_block_lifecycle(
        bodied(rows, opens), config=config, formation=formation, symbol="XAUUSD"
    )


# A bullish leg whose source candle is bars 6/7 written as an identical pair, so
# the taller high makes no strict pivot. Its BODY zone is exactly [4000, 4010]
# and its FULL_CANDLE zone [3995, 4015].
#
# Bars 11 and 12 are rewritten to sit at 4030-4040, clear above both zones. The
# scaffold's own filler ran 3990-4010, which touched the block one bar after it
# formed and left every appended tail testing a block that had already moved on.
# Parking price above the zone means the first interaction is whichever bar a
# test appends, and nothing else.
SPEC_LEG: list[Row] = list(BULL_LEG)
SPEC_LEG[6] = SPEC_LEG[7] = ("4015", "3995", "4000")
SPEC_LEG[10] = ("4040", "3990", "4040")
SPEC_LEG[11] = SPEC_LEG[12] = ("4040", "4030", "4035")
SPEC_OPENS = {7: "4010"}

FIRST_TAIL_BAR = len(SPEC_LEG)
"""Index of the first bar a test appends, and so of the first possible touch."""

TOUCH_ONLY_BODY: Row = ("4012", "4008", "4010")
"""Meets [4000, 4010] without reaching its 4005 midpoint or covering it."""

AWAY: Row = ("4040", "4030", "4035")
"""Well clear of both zones, so it changes nothing."""


def after(*tail: Row, rows: Sequence[Row] = tuple(SPEC_LEG)) -> list[Row]:
    """The scaffold with *tail* appended after its last filler bar."""
    return [*rows, *tail]


def test_the_scaffold_forms_the_single_block_this_file_assumes() -> None:
    """Read once here so no other test has to re-derive it."""
    snapshot = bodied(SPEC_LEG, SPEC_OPENS)
    structure = analyse_structure(snapshot, symbol="XAUUSD")
    body = analyse_order_blocks(snapshot, config=BODY, symbol="XAUUSD").order_blocks
    full = analyse_order_blocks(snapshot, config=FULL, symbol="XAUUSD").order_blocks

    assert len(structure.breaks) == 1
    assert len(body) == len(full) == 1
    assert (body[0].lower, body[0].upper, body[0].midpoint) == (
        Decimal("4000"),
        Decimal("4010"),
        Decimal("4005"),
    )
    assert (full[0].lower, full[0].upper) == (Decimal("3995"), Decimal("4015"))
    assert closing_bar(body[0].formed_at) == 10


# --------------------------------------------------------------------------
# §5, §32: the rule is the caller's, and there is no default
# --------------------------------------------------------------------------


def test_the_config_has_no_default_mitigation_rule() -> None:
    """§5. Constructing one without naming a convention is an error."""
    with pytest.raises(TypeError):
        OrderBlockLifecycleConfig()  # type: ignore[call-arg]


def test_the_entry_points_demand_a_config() -> None:
    for function in (
        ict_order_block_lifecycle.analyse_order_block_lifecycle,
        ict_order_block_lifecycle.analyse_snapshot_order_block_lifecycle,
    ):
        parameter = inspect.signature(function).parameters["config"]
        assert parameter.default is inspect.Parameter.empty
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_the_config_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        TOUCH.mitigation_rule = OrderBlockMitigationRule.MIDPOINT  # type: ignore[misc]


def test_there_are_exactly_three_rules_and_four_statuses() -> None:
    assert [member.value for member in OrderBlockMitigationRule] == [
        "TOUCH",
        "MIDPOINT",
        "FULL_ZONE",
    ]
    assert [member.value for member in OrderBlockStatus] == [
        "ACTIVE",
        "TOUCHED",
        "MITIGATED",
        "INVALIDATED",
    ]


def test_both_policies_must_be_chosen_consciously() -> None:
    """§32. Neither the zone basis nor the mitigation rule has a default."""
    snapshot = bodied(SPEC_LEG, SPEC_OPENS)

    with pytest.raises(OrderBlockLifecycleError, match="exactly one"):
        analyse_order_block_lifecycle(snapshot, config=TOUCH, symbol="XAUUSD")


def test_supplying_both_sources_of_blocks_is_refused() -> None:
    snapshot = bodied(SPEC_LEG, SPEC_OPENS)
    formed = analyse_order_blocks(snapshot, config=BODY, symbol="XAUUSD")

    with pytest.raises(OrderBlockLifecycleError, match="exactly one"):
        analyse_order_block_lifecycle(
            snapshot, config=TOUCH, formation=BODY, order_blocks=formed, symbol="XAUUSD"
        )


# --------------------------------------------------------------------------
# §44: touch geometry, on a closed interval
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("low", "high", "expected", "why"),
    [
        ("4011", "4020", False, "A: entirely above"),
        ("4010", "4020", True, "B: exact upper boundary"),
        ("3990", "4000", True, "C: exact lower boundary"),
        ("3990", "3999", False, "D: entirely below"),
        ("4001", "4009", True, "E: wholly inside"),
        ("3990", "4020", True, "F: engulfing"),
    ],
)
def test_the_touch_matrix(low: str, high: str, expected: bool, why: str) -> None:
    """§44, on ``[4000, 4010]``. Exact boundary contact counts."""
    assert touches(ZONE, Decimal(low), Decimal(high)) is expected, why


@pytest.mark.parametrize(
    ("low", "high"),
    [("4011", "4020"), ("4010", "4020"), ("3990", "4000"), ("3990", "3999"), ("4001", "4009")],
)
def test_touch_geometry_is_the_same_for_both_directions(low: str, high: str) -> None:
    """§44. A bullish and a bearish block on one interval are touched alike."""
    assert touches(ZONE, Decimal(low), Decimal(high)) is touches(
        BEAR_ZONE, Decimal(low), Decimal(high)
    )


def test_a_single_price_resting_on_an_edge_is_a_touch() -> None:
    """The clearest statement of the closed interval: a doji on the boundary."""
    assert touches(ZONE, Decimal("4010"), Decimal("4010")) is True
    assert touches(ZONE, Decimal("4000"), Decimal("4000")) is True


def test_a_hundredth_beyond_the_edge_is_not_a_touch() -> None:
    """No tolerance in either direction."""
    assert touches(ZONE, Decimal("4010.01"), Decimal("4010.01")) is False
    assert touches(ZONE, Decimal("3999.98"), Decimal("3999.99")) is False


def test_the_order_block_boundary_differs_from_the_gap_boundary_on_purpose() -> None:
    """§7, §F. The two engines describe different objects.

    A gap's edges bound an absence, so :func:`ict_fvg.enters` is strict and a
    candle resting on the edge never entered it. An order block's edges are
    prices its source candle actually printed, so contact is contact.
    """
    from goldpipeline.services.ict_fvg import enters

    assert "high > gap.lower and low < gap.upper" in inspect.getsource(enters), "open there"
    assert "high >= block.lower and low <= block.upper" in inspect.getsource(touches), "closed here"

    resting = (Decimal("4010"), Decimal("4010"))
    assert touches(ZONE, *resting) is True


# --------------------------------------------------------------------------
# §45: the mitigation policy matrix
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("low", "high", "expected", "why"),
    [
        ("4010", "4020", (True, False, False), "1: rests on the upper edge"),
        ("4004", "4006", (True, True, False), "2: straddles the midpoint"),
        ("3995", "4015", (True, True, True), "3: engulfs the zone"),
        ("3990", "3999", (False, False, False), "4: never reached it"),
        ("4005", "4005", (True, True, False), "5: exactly the midpoint"),
    ],
)
def test_the_mitigation_matrix(
    low: str, high: str, expected: tuple[bool, bool, bool], why: str
) -> None:
    """§45, on ``[4000, 4010]`` with midpoint 4005."""
    got = tuple(satisfies(ZONE, rule, Decimal(low), Decimal(high)) for rule in RULES)

    assert got == expected, why


def test_midpoint_requires_the_range_to_contain_it_not_merely_pass_below() -> None:
    """§13. A low beneath the midpoint proves nothing about where price went."""
    assert contains_midpoint(ZONE, Decimal("3990"), Decimal("3995")) is False
    assert contains_midpoint(ZONE, Decimal("4004"), Decimal("4006")) is True


def test_full_zone_requires_both_halves() -> None:
    """§14. Neither edge alone is coverage, in either direction."""
    assert covers(ZONE, Decimal("3990"), Decimal("4005")) is False, "low alone"
    assert covers(ZONE, Decimal("4005"), Decimal("4020")) is False, "high alone"
    assert covers(ZONE, Decimal("4000"), Decimal("4010")) is True, "exactly, edge to edge"


def test_a_candle_that_gapped_clean_under_the_zone_covers_nothing() -> None:
    """§14, and the same false-fill guard the gap engine needed."""
    assert covers(ZONE, Decimal("3980"), Decimal("3990")) is False


@pytest.mark.parametrize("rule", RULES)
def test_no_mitigation_rule_reads_the_block_direction(rule: OrderBlockMitigationRule) -> None:
    """§15. Bullish and bearish blocks on one interval mitigate identically."""
    for low, high in (("4010", "4020"), ("4004", "4006"), ("3995", "4015"), ("3990", "3999")):
        assert satisfies(ZONE, rule, Decimal(low), Decimal(high)) is satisfies(
            BEAR_ZONE, rule, Decimal(low), Decimal(high)
        )


# --------------------------------------------------------------------------
# §46: the invalidation matrix
# --------------------------------------------------------------------------


def test_the_far_edge_of_each_direction() -> None:
    assert far_edge(ZONE) == Decimal("4000")
    assert far_edge(BEAR_ZONE) == Decimal("4010")


@pytest.mark.parametrize(
    ("close", "expected"),
    [("4000", False), ("3999.99", True), ("4001", False), ("4010", False), ("3985", True)],
)
def test_the_bullish_invalidation_matrix(close: str, expected: bool) -> None:
    """§17, §20, §46. Strictly below the lower edge, and nothing else."""
    assert invalidates(ZONE, Decimal(close)) is expected


@pytest.mark.parametrize(
    ("close", "expected"),
    [("4010", False), ("4010.01", True), ("4009", False), ("4000", False), ("4025", True)],
)
def test_the_bearish_invalidation_matrix(close: str, expected: bool) -> None:
    assert invalidates(BEAR_ZONE, Decimal(close)) is expected


def test_a_wick_through_the_far_edge_does_not_invalidate() -> None:
    """§18, §19. The branch is close-confirmed everywhere, and stays so here.

    The brief's own candle: low 3990, high 4006, close 4004 against ``[4000,
    4010]``. It touched, it traded the midpoint, and it closed back inside.
    """
    low, high, close = Decimal("3990"), Decimal("4006"), Decimal("4004")

    assert touches(ZONE, low, high) is True
    assert contains_midpoint(ZONE, low, high) is True
    assert invalidates(ZONE, close) is False


def test_the_bearish_wick_mirror() -> None:
    low, high, close = Decimal("4004"), Decimal("4020"), Decimal("4009")

    assert touches(BEAR_ZONE, low, high) is True
    assert invalidates(BEAR_ZONE, close) is False


# --------------------------------------------------------------------------
# §24: status is derived from the evidence
# --------------------------------------------------------------------------


def test_status_derivation_follows_precedence() -> None:
    moment = START + HOUR

    assert status_of(first_touched_at=None, mitigated_at=None, invalidated_at=None) is (
        OrderBlockStatus.ACTIVE
    )
    assert status_of(first_touched_at=moment, mitigated_at=None, invalidated_at=None) is (
        OrderBlockStatus.TOUCHED
    )
    assert status_of(first_touched_at=moment, mitigated_at=moment, invalidated_at=None) is (
        OrderBlockStatus.MITIGATED
    )
    assert status_of(first_touched_at=moment, mitigated_at=moment, invalidated_at=moment) is (
        OrderBlockStatus.INVALIDATED
    )


def test_invalidation_outranks_everything_even_with_no_touch() -> None:
    """§21. A block can be retired without ever having been reached."""
    assert (
        status_of(first_touched_at=None, mitigated_at=None, invalidated_at=START)
        is OrderBlockStatus.INVALIDATED
    )


# --------------------------------------------------------------------------
# §9-§10: the formation bar cannot interact
# --------------------------------------------------------------------------


def test_the_break_bar_does_not_touch_the_block_it_created() -> None:
    """§9, §10. Known-before, and it is not a hypothetical.

    Bar 10 is the break bar and its range runs 3990-4040, which overlaps the
    3995-4015 zone outright. It is still not the first touch: until that bar
    closed there was no order block there to touch.
    """
    snapshot = bodied(SPEC_LEG, SPEC_OPENS)
    breaking = snapshot.bars[10]
    formed = analyse_order_blocks(snapshot, config=FULL, symbol="XAUUSD").order_blocks[0]

    assert touches(formed, breaking.low, breaking.high) is True, "geometrically it does overlap"

    at_formation = analyse_order_block_lifecycle(
        snapshot, config=TOUCH, formation=FULL, symbol="XAUUSD", as_of=START + HOUR * 11
    )
    state = at_formation.states[0]

    assert state.status is OrderBlockStatus.ACTIVE
    assert state.first_touched_at is None
    assert state.mitigated_at is None


@pytest.mark.parametrize("rule", RULES)
def test_no_rule_lets_the_formation_bar_mitigate(rule: OrderBlockMitigationRule) -> None:
    snapshot = bodied(SPEC_LEG, SPEC_OPENS)
    at_formation = analyse_order_block_lifecycle(
        snapshot,
        config=OrderBlockLifecycleConfig(mitigation_rule=rule),
        formation=FULL,
        symbol="XAUUSD",
        as_of=START + HOUR * 11,
    )

    assert at_formation.states[0].status is OrderBlockStatus.ACTIVE


def test_interaction_begins_strictly_after_formation() -> None:
    """§9. The predicate is ``formed_at < close_time``, not ``<=``."""
    source = inspect.getsource(ict_order_block_lifecycle._state_for)

    assert "close_time <= block.formed_at" in source, "skips anything up to and including it"


# --------------------------------------------------------------------------
# §47: several facts from one candle
# --------------------------------------------------------------------------


def one_bar_state(tail: Row, *, config: OrderBlockLifecycleConfig) -> object:
    """The state after exactly one interaction bar following the break."""
    analysis = lifecycle(after(tail, tail), SPEC_OPENS, config=config)
    assert len(analysis.states) == 1, "the tail must not create a second block"
    return analysis.states[0]


@pytest.mark.parametrize("rule", RULES)
def test_a_bar_that_only_touches(rule: OrderBlockMitigationRule) -> None:
    """§47. Resting on the upper edge: TOUCH mitigates, the other two do not."""
    state = one_bar_state(("4020", "4015", "4018"), config=OrderBlockLifecycleConfig(rule))

    assert state.first_touched_at is not None  # type: ignore[attr-defined]
    assert state.invalidated_at is None  # type: ignore[attr-defined]
    if rule is OrderBlockMitigationRule.TOUCH:
        assert state.status is OrderBlockStatus.MITIGATED  # type: ignore[attr-defined]
    else:
        assert state.status is OrderBlockStatus.TOUCHED  # type: ignore[attr-defined]


def test_a_bar_that_touches_and_mitigates_under_midpoint() -> None:
    """§47. 4004-4006 straddles the 4005 midpoint of the BODY zone."""
    analysis = lifecycle(
        after(("4006", "4004", "4005"), ("4006", "4004", "4005")),
        SPEC_OPENS,
        config=MIDPOINT,
        basis=OrderBlockZoneBasis.BODY,
    )
    state = analysis.states[0]

    assert state.status is OrderBlockStatus.MITIGATED
    assert state.first_touched_at == state.mitigated_at
    assert state.invalidated_at is None


def test_a_bar_that_touches_and_invalidates_without_mitigating_under_full_zone() -> None:
    """§47, §23. Three independent facts, read off one candle.

    The bar runs 3990-4006 against the BODY zone [4000, 4010]: it touched, it
    did not cover, and it closed at 3995 - below the far edge.
    """
    analysis = lifecycle(
        after(("4006", "3990", "3995"), ("4006", "3990", "3995")),
        SPEC_OPENS,
        config=FULL_ZONE,
        basis=OrderBlockZoneBasis.BODY,
    )
    state = analysis.states[0]

    assert state.first_touched_at is not None
    assert state.mitigated_at is None, "it never covered the whole band"
    assert state.invalidated_at == state.first_touched_at, "the same candle"
    assert state.status is OrderBlockStatus.INVALIDATED


def test_a_bar_that_touches_mitigates_and_invalidates_at_once() -> None:
    """§23, the brief's own candle: low 3990, high 4012, close 3995.

    All three witnesses are recorded and all three name the same bar. The status
    says INVALIDATED because that is the strongest thing now true - not because
    the engine claims to know the intrabar order.
    """
    analysis = lifecycle(
        after(("4012", "3990", "3995"), ("4012", "3990", "3995")),
        SPEC_OPENS,
        config=FULL_ZONE,
        basis=OrderBlockZoneBasis.BODY,
    )
    state = analysis.states[0]

    assert state.first_touch_witness is not None
    assert state.mitigation_witness is not None
    assert state.invalidation_witness is not None
    assert (
        state.first_touch_witness.bar_open_time
        == state.mitigation_witness.bar_open_time
        == state.invalidation_witness.bar_open_time
    )
    assert state.status is OrderBlockStatus.INVALIDATED


def test_a_bar_that_invalidates_without_touching() -> None:
    """§21, §47. A gap clean beneath the zone, closing below it.

    The candle never traded in [4000, 4010], so there is no touch witness and no
    mitigation witness - and the block is retired all the same.
    """
    for config in (TOUCH, MIDPOINT, FULL_ZONE):
        analysis = lifecycle(
            after(("3990", "3980", "3985"), ("3990", "3980", "3985")),
            SPEC_OPENS,
            config=config,
            basis=OrderBlockZoneBasis.BODY,
        )
        state = analysis.states[0]

        assert state.first_touched_at is None, config.mitigation_rule
        assert state.mitigated_at is None, config.mitigation_rule
        assert state.invalidated_at is not None
        assert state.status is OrderBlockStatus.INVALIDATED


# --------------------------------------------------------------------------
# §11, §16, §22, §26: first witnesses, and only first witnesses
# --------------------------------------------------------------------------


def test_a_later_touch_does_not_replace_the_first() -> None:
    """§11, §26. No count, no most-recent, no reaction tally."""
    analysis = lifecycle(
        after(
            TOUCH_ONLY_BODY,
            TOUCH_ONLY_BODY,
            AWAY,
            AWAY,
            TOUCH_ONLY_BODY,
            TOUCH_ONLY_BODY,
        ),
        SPEC_OPENS,
        config=MIDPOINT,
        basis=OrderBlockZoneBasis.BODY,
    )
    state = analysis.states[0]

    assert state.first_touch_witness is not None
    assert closing_bar(state.first_touched_at) == FIRST_TAIL_BAR, "the first of four"  # type: ignore[arg-type]
    assert state.status is OrderBlockStatus.TOUCHED, "none of them reached the midpoint"


def test_a_mitigated_block_is_not_mitigated_twice() -> None:
    """§26. The first witness stays fixed however often price returns."""
    analysis = lifecycle(
        after(
            ("4006", "4004", "4005"),
            ("4006", "4004", "4005"),
            AWAY,
            AWAY,
            ("4006", "4004", "4005"),
            ("4006", "4004", "4005"),
        ),
        SPEC_OPENS,
        config=MIDPOINT,
        basis=OrderBlockZoneBasis.BODY,
    )
    state = analysis.states[0]

    assert closing_bar(state.mitigated_at) == FIRST_TAIL_BAR  # type: ignore[arg-type]
    assert state.status is OrderBlockStatus.MITIGATED


def test_the_state_records_which_rule_produced_it() -> None:
    """§16. The policy is on the record, not implied by the numbers."""
    for config, rule in ((TOUCH, "TOUCH"), (MIDPOINT, "MIDPOINT"), (FULL_ZONE, "FULL_ZONE")):
        analysis = lifecycle(after(FLAT), SPEC_OPENS, config=config)

        assert analysis.mitigation_rule.value == rule
        assert analysis.states[0].mitigation_rule.value == rule


def test_changing_the_rule_never_changes_the_block() -> None:
    """§16. Policy decides the state; it cannot reach the formation record."""
    blocks = [
        lifecycle(after(("4006", "4004", "4005"), ("4006", "4004", "4005")), SPEC_OPENS, config=c)
        .states[0]
        .order_block
        for c in (TOUCH, MIDPOINT, FULL_ZONE)
    ]

    assert blocks[0] == blocks[1] == blocks[2]


# --------------------------------------------------------------------------
# §48: status progression
# --------------------------------------------------------------------------


def progression(
    tail: Sequence[Row], *, config: OrderBlockLifecycleConfig, basis: OrderBlockZoneBasis
) -> list[str]:
    """The status after each successive appended bar."""
    rows = after(*tail)
    formation = FULL if basis is OrderBlockZoneBasis.FULL_CANDLE else BODY
    seen: list[str] = []
    for kept in range(len(SPEC_LEG), len(rows) + 1):
        analysis = analyse_order_block_lifecycle(
            bodied(rows, SPEC_OPENS, bars_kept=kept),
            config=config,
            formation=formation,
            symbol="XAUUSD",
        )
        assert len(analysis.states) == 1, "the tail must not create a second block"
        seen.append(analysis.states[0].status.value)
    return seen


def test_active_then_touched_then_mitigated_then_invalidated() -> None:
    """§48. The long path, one transition per bar."""
    tail = (
        TOUCH_ONLY_BODY,
        TOUCH_ONLY_BODY,
        ("4006", "4004", "4005"),
        ("4006", "4004", "4005"),
        ("4006", "3990", "3995"),
        ("4006", "3990", "3995"),
    )
    seen = progression(tail, config=MIDPOINT, basis=OrderBlockZoneBasis.BODY)

    assert seen[0] == "ACTIVE"
    assert "TOUCHED" in seen
    assert seen.index("TOUCHED") < seen.index("MITIGATED") < seen.index("INVALIDATED")
    assert seen[-1] == "INVALIDATED"


def test_active_straight_to_invalidated() -> None:
    """§48. A gap under the zone, with no touch on the way."""
    seen = progression(
        (("3990", "3980", "3985"), ("3990", "3980", "3985")),
        config=TOUCH,
        basis=OrderBlockZoneBasis.BODY,
    )

    assert seen[0] == "ACTIVE"
    assert seen[-1] == "INVALIDATED"
    assert "TOUCHED" not in seen and "MITIGATED" not in seen


def test_active_straight_to_mitigated_under_touch() -> None:
    """§48. Under TOUCH there is no intermediate TOUCHED state to pass through."""
    seen = progression(
        (TOUCH_ONLY_BODY, TOUCH_ONLY_BODY), config=TOUCH, basis=OrderBlockZoneBasis.BODY
    )

    assert seen == ["ACTIVE", "MITIGATED", "MITIGATED"]


def test_touched_then_invalidated_without_ever_mitigating() -> None:
    """§48, under FULL_ZONE, which the touching candles never satisfy."""
    seen = progression(
        (
            TOUCH_ONLY_BODY,
            TOUCH_ONLY_BODY,
            ("4006", "3990", "3995"),
            ("4006", "3990", "3995"),
        ),
        config=FULL_ZONE,
        basis=OrderBlockZoneBasis.BODY,
    )

    assert seen[0] == "ACTIVE"
    assert "TOUCHED" in seen
    assert "MITIGATED" not in seen
    assert seen[-1] == "INVALIDATED"


def test_invalidated_is_terminal_over_many_later_bars() -> None:
    """§22, §25. Later price cannot add a touch or a second invalidation."""
    tail = (
        ("3990", "3980", "3985"),
        ("3990", "3980", "3985"),
        *([AWAY] * 6),
        ("4006", "4004", "4005"),
        ("4006", "4004", "4005"),
    )
    analysis = lifecycle(after(*tail), SPEC_OPENS, config=MIDPOINT, basis=OrderBlockZoneBasis.BODY)
    state = analysis.states[0]

    assert state.status is OrderBlockStatus.INVALIDATED
    assert state.is_terminal
    assert state.first_touched_at is None, "the later return is not recorded"
    assert state.mitigated_at is None
    assert closing_bar(state.invalidated_at) == FIRST_TAIL_BAR  # type: ignore[arg-type]


def test_a_status_never_moves_backwards() -> None:
    """§25, stated as an ordering over the whole progression."""
    rank = {"ACTIVE": 0, "TOUCHED": 1, "MITIGATED": 2, "INVALIDATED": 3}
    tail = (
        TOUCH_ONLY_BODY,
        TOUCH_ONLY_BODY,
        ("4006", "4004", "4005"),
        ("4006", "4004", "4005"),
        AWAY,
        AWAY,
        ("4006", "3990", "3995"),
        ("4006", "3990", "3995"),
    )

    for config in (TOUCH, MIDPOINT, FULL_ZONE):
        seen = progression(tail, config=config, basis=OrderBlockZoneBasis.BODY)
        assert [rank[value] for value in seen] == sorted(rank[value] for value in seen), config


# --------------------------------------------------------------------------
# §27, §28, §54, §55: what does not retire a block
# --------------------------------------------------------------------------


def test_a_later_structure_event_does_not_retire_an_untouched_block() -> None:
    """§27. Lifecycle is price interaction, and nothing else.

    The realistic fixture's own case: the bullish block formed at bar 11 is not
    retired by the bearish MSS at bar 33 as an event - it is retired at bar 40,
    by a close beneath its lower edge, which is a different claim entirely.
    """
    from tests.test_ict_order_block_lifecycle_fixture import journey

    analysis = analyse_order_block_lifecycle(
        journey(), config=TOUCH, formation=FULL, symbol="XAUUSD"
    )
    structure = analyse_structure(journey(), symbol="XAUUSD")
    first = analysis.states[0]

    assert closing_bar(first.invalidated_at) == 40  # type: ignore[arg-type]
    assert any(
        closing_bar(e.break_bar_close_time) == 33 and e.classification is BreakClassification.MSS
        for e in structure.breaks
    ), "an opposite MSS did occur earlier, and did not retire it"


def test_an_invalidated_block_stays_an_order_block() -> None:
    """§28, §57. No conversion, no breaker, no new object."""
    analysis = lifecycle(
        after(("3990", "3980", "3985"), ("3990", "3980", "3985")),
        SPEC_OPENS,
        config=TOUCH,
        basis=OrderBlockZoneBasis.BODY,
    )
    state = analysis.states[0]
    formed = analyse_order_blocks(
        bodied(after(("3990", "3980", "3985"), ("3990", "3980", "3985")), SPEC_OPENS),
        config=BODY,
        symbol="XAUUSD",
    ).order_blocks[0]

    assert state.status is OrderBlockStatus.INVALIDATED
    assert state.order_block == formed, "the formation record is untouched"
    assert state.order_block_id == formed.order_block_id


# --------------------------------------------------------------------------
# §36: supplied blocks
# --------------------------------------------------------------------------


def test_supplied_blocks_give_the_same_answer_as_computing_them() -> None:
    """§36. What a composite stage saves is a recomputation, not a difference."""
    snapshot = bodied(after(("4006", "4004", "4005"), ("4006", "4004", "4005")), SPEC_OPENS)
    formed = analyse_order_blocks(snapshot, config=BODY, symbol="XAUUSD")

    assert analyse_order_block_lifecycle(
        snapshot, config=MIDPOINT, order_blocks=formed, symbol="XAUUSD"
    ) == analyse_order_block_lifecycle(snapshot, config=MIDPOINT, formation=BODY, symbol="XAUUSD")


def test_blocks_from_another_instant_are_refused_not_trimmed() -> None:
    """§36. A final analysis handed to a historical query would leak blocks."""
    snapshot = bodied(SPEC_LEG, SPEC_OPENS)
    final = analyse_order_blocks(snapshot, config=BODY, symbol="XAUUSD")

    with pytest.raises(OrderBlockLifecycleError, match="same instant"):
        analyse_order_block_lifecycle(
            snapshot,
            config=TOUCH,
            order_blocks=final,
            symbol="XAUUSD",
            as_of=START + HOUR * 11,
        )


def test_blocks_for_another_timeframe_are_refused() -> None:
    snapshot = bodied(SPEC_LEG, SPEC_OPENS)
    other = replace(
        analyse_order_blocks(snapshot, config=BODY, symbol="XAUUSD"), timeframe=Timeframe.M15
    )

    with pytest.raises(OrderBlockLifecycleError, match="are for M15"):
        analyse_order_block_lifecycle(snapshot, config=TOUCH, order_blocks=other, symbol="XAUUSD")


def test_blocks_for_another_symbol_are_refused() -> None:
    snapshot = bodied(SPEC_LEG, SPEC_OPENS)
    other = analyse_order_blocks(snapshot, config=BODY, symbol="EURUSD")

    with pytest.raises(OrderBlockLifecycleError, match="are for 'EURUSD'"):
        analyse_order_block_lifecycle(snapshot, config=TOUCH, order_blocks=other, symbol="XAUUSD")


def test_an_as_of_before_any_bar_closed_is_refused() -> None:
    with pytest.raises(ValueError, match="no bar closed"):
        analyse_order_block_lifecycle(
            bodied(SPEC_LEG, SPEC_OPENS),
            config=TOUCH,
            formation=FULL,
            symbol="XAUUSD",
            as_of=START,
        )


def test_a_series_with_no_blocks_gives_an_empty_analysis() -> None:
    """Absence stays absence; it does not become an error or a placeholder."""
    analysis = analyse_order_block_lifecycle(
        bodied(BULL_LEG, {}), config=TOUCH, formation=FULL, symbol="XAUUSD"
    )

    assert analysis.states == ()
    assert analysis.active_ids == analysis.touched_ids == ()
    assert analysis.mitigated_ids == analysis.invalidated_ids == ()
    assert analysis.state("nope") is None
