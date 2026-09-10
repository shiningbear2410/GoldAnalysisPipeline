"""The five-timeframe snapshot consolidated, pinned exactly.

Round 6.6e.2c1 §32-§33, §41-§48. Every number below was read out of the engine
before it was written down.

The snapshot is Round 6.6e.2b's staggered one. Two of this round's cases turn up
in it naturally and are not manufactured: H1's two overlapping gaps land on the
identical band 3960-4050 and consolidate, and under a policy that admits
mitigated blocks H1's two same-source order blocks at 3990-4085 - the pair Round
6.6d.1 created from one candle and two structure events - become eligible
together and consolidate too.

What the realistic path does *not* contain is an eligible liquidity reference
(every pool in it is terminal) or a cross-source exact match. Those are proved
on constructed readings in ``test_ict_candidate_consolidation.py`` rather than
forced into this one.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.services.ict_candidate_consolidation import (
    CandidateConsolidationAnalysis,
    consolidate_candidates,
)
from goldpipeline.services.ict_candidate_eligibility import (
    CandidateRole,
    EntrySide,
)
from goldpipeline.services.ict_candidate_source import CandidateSourceKind
from goldpipeline.services.ict_fvg import FvgStatus
from goldpipeline.services.ict_order_block import OrderBlockZoneBasis
from goldpipeline.services.ict_order_block_lifecycle import (
    OrderBlockMitigationRule,
    OrderBlockStatus,
)
from tests.test_ict_candidate_eligibility_fixture import analysis
from tests.test_ict_candidate_source_fixture import block_of, gap_of

WIDE_BLOCKS = frozenset(
    {OrderBlockStatus.ACTIVE, OrderBlockStatus.TOUCHED, OrderBlockStatus.MITIGATED}
)


def consolidated(**kwargs: object) -> CandidateConsolidationAnalysis:
    return consolidate_candidates(analysis(**kwargs))  # type: ignore[arg-type]


def shape(result: CandidateConsolidationAnalysis) -> list[tuple[object, ...]]:
    return [
        (
            candidate.role.value,
            candidate.entry_side.value if candidate.entry_side is not None else None,
            None if candidate.lower is None else str(candidate.lower),
            None if candidate.upper is None else str(candidate.upper),
            None if candidate.reference_level is None else str(candidate.reference_level),
            candidate.market_relation.value,
            str(candidate.distance_to_reference),
            candidate.support_count,
            tuple(tf.value for tf in candidate.timeframes),
            tuple(kind.value for kind in candidate.source_kinds),
        )
        for candidate in result.candidates
    ]


# --------------------------------------------------------------------------
# §47: the whole consolidation, pinned
# --------------------------------------------------------------------------


def test_the_default_policy_consolidation_is_pinned() -> None:
    assert shape(consolidated()) == [
        (
            "ENTRY_ZONE",
            "BAI",
            "3960",
            "4050",
            None,
            "OVERLAPS",
            "0",
            2,
            ("H1",),
            ("FAIR_VALUE_GAP",),
        ),
        ("ENTRY_ZONE", "BAI", "4012", "4025", None, "BELOW", "18", 1, ("M1",), ("FAIR_VALUE_GAP",)),
        (
            "ENTRY_ZONE",
            "BAI",
            "4030",
            "4052",
            None,
            "OVERLAPS",
            "0",
            1,
            ("M1",),
            ("FAIR_VALUE_GAP",),
        ),
        (
            "ENTRY_ZONE",
            "BAI",
            "4025",
            "4050",
            None,
            "OVERLAPS",
            "0",
            1,
            ("M1",),
            ("FAIR_VALUE_GAP",),
        ),
        ("ENTRY_ZONE", "SEO", "4046", "4048", None, "ABOVE", "3", 1, ("M1",), ("FAIR_VALUE_GAP",)),
        ("ENTRY_ZONE", "SEO", "4045", "4052", None, "ABOVE", "2", 1, ("M1",), ("FAIR_VALUE_GAP",)),
    ]


def test_the_wide_block_policy_adds_the_same_source_order_block_pair() -> None:
    """§17, §48. Two blocks from one candle and two events, now both eligible."""
    result = consolidated(blocks=WIDE_BLOCKS)
    blocks = [
        candidate
        for candidate in result.candidates
        if candidate.source_kinds == (CandidateSourceKind.ORDER_BLOCK,)
    ]

    assert len(result.candidates) == 7
    assert len(blocks) == 1
    pair = blocks[0]
    assert (pair.lower, pair.upper) == (Decimal("3990"), Decimal("4085"))
    assert pair.entry_side is EntrySide.SEO
    assert pair.support_count == 2
    assert pair.timeframes == (Timeframe.H1,)
    assert len(set(pair.supporting_source_ids)) == 2, "two order blocks, not one counted twice"


def test_the_same_source_blocks_share_a_candle_and_differ_by_event() -> None:
    """§17. The provenance both members keep."""
    pair = next(
        candidate
        for candidate in consolidated(blocks=WIDE_BLOCKS).candidates
        if candidate.source_kinds == (CandidateSourceKind.ORDER_BLOCK,)
    )
    first, second = (
        block_of(decision.source).order_block for decision in pair.supporting_decisions
    )

    assert first.source_bar_open_time == second.source_bar_open_time
    assert first.structure_event_id != second.structure_event_id
    assert first.order_block_id != second.order_block_id


def test_the_overlapping_h1_gaps_consolidate_because_they_are_identical() -> None:
    """§4, §48. Round 6.6e.2a's overlapping pair share an exact band."""
    pair = consolidated().candidates[0]

    assert (pair.lower, pair.upper) == (Decimal("3960"), Decimal("4050"))
    assert pair.support_count == 2
    assert len(set(pair.supporting_source_ids)) == 2
    formed = {gap_of(decision.source).gap.formed_at for decision in pair.supporting_decisions}
    assert len(formed) == 2, "two gaps, formed at different instants"


def test_the_heavily_overlapping_m1_gaps_stay_separate() -> None:
    """§21, in the realistic path rather than a constructed one.

    4025-4050 and 4030-4052 share most of their span and are still two
    candidates, because they are not the same zone.
    """
    result = consolidated()
    m1_bai = [
        candidate
        for candidate in result.candidates
        if candidate.timeframes == (Timeframe.M1,) and candidate.entry_side is EntrySide.BAI
    ]

    bounds = {(candidate.lower, candidate.upper) for candidate in m1_bai}
    assert (Decimal("4025"), Decimal("4050")) in bounds
    assert (Decimal("4030"), Decimal("4052")) in bounds
    assert all(candidate.support_count == 1 for candidate in m1_bai)


def test_an_overlapping_pair_on_opposite_sides_stays_separate() -> None:
    """§19, in the realistic path. 4030-4052 is BAI; 4045-4052 is SEO."""
    result = consolidated()
    sharing_an_edge = [
        candidate for candidate in result.candidates if candidate.upper == Decimal("4052")
    ]

    assert len(sharing_an_edge) == 2
    assert {candidate.entry_side for candidate in sharing_an_edge} == {
        EntrySide.BAI,
        EntrySide.SEO,
    }


def test_no_liquidity_reference_is_eligible_in_this_fixture() -> None:
    """§47. Said out loud rather than papered over."""
    result = consolidated()

    assert result.of_role(CandidateRole.UPPER_REFERENCE) == ()
    assert result.of_role(CandidateRole.LOWER_REFERENCE) == ()
    assert len(result.eligibility.require(Timeframe.M15).decisions) == 5


# --------------------------------------------------------------------------
# §29-§30: the counting invariants, over the realistic path
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"blocks": WIDE_BLOCKS},
        {"gaps": frozenset({FvgStatus.OPEN})},
        {"gaps": frozenset({FvgStatus.TOUCHED})},
        {"gaps": frozenset()},
    ],
)
def test_support_totals_equal_the_eligible_count(kwargs: dict[str, object]) -> None:
    """§29. Under five policies, nothing is lost and nothing is double-counted."""
    result = consolidated(**kwargs)
    eligible = [
        decision
        for entry in result.eligibility.timeframes
        for decision in entry.decisions
        if decision.eligible
    ]

    placed = [
        member.candidate_id
        for candidate in result.candidates
        for member in candidate.supporting_decisions
    ]
    assert sorted(placed) == sorted(decision.candidate_id for decision in eligible)
    assert len(placed) == len(set(placed))
    assert sum(candidate.support_count for candidate in result.candidates) == len(eligible)


def test_the_default_and_wide_policies_pin_their_totals() -> None:
    assert (
        len(consolidated().candidates),
        sum(c.support_count for c in consolidated().candidates),
    ) == (
        6,
        7,
    )
    wide = consolidated(blocks=WIDE_BLOCKS)
    assert (len(wide.candidates), sum(c.support_count for c in wide.candidates)) == (7, 9)


def test_an_empty_policy_produces_no_candidates() -> None:
    """§28. Every supporter excluded means the view is empty, not broken."""
    result = consolidated(blocks=frozenset(), gaps=frozenset())

    assert result.candidates == ()
    assert sum(len(entry.decisions) for entry in result.eligibility.timeframes) == 27


# --------------------------------------------------------------------------
# §32-§33: ordering
# --------------------------------------------------------------------------


def test_groups_are_ordered_by_role_then_side_then_identity() -> None:
    """§33. A total order that expresses no preference."""
    role_index = {role: index for index, role in enumerate(CandidateRole)}
    side_index = {None: 0, EntrySide.BAI: 1, EntrySide.SEO: 2}

    for kwargs in ({}, {"blocks": WIDE_BLOCKS}):
        keys = [
            (
                role_index[candidate.role],
                side_index[candidate.entry_side],
                candidate.consolidated_candidate_id,
            )
            for candidate in consolidated(**kwargs).candidates
        ]
        assert keys == sorted(keys)


def test_groups_are_not_ordered_by_distance_or_support() -> None:
    """§33. Neither is a ranking key, and the fixture would show it if they were."""
    result = consolidated(blocks=WIDE_BLOCKS)
    distances = [candidate.distance_to_reference for candidate in result.candidates]
    supports = [candidate.support_count for candidate in result.candidates]

    assert distances != sorted(distances)
    assert supports != sorted(supports, reverse=True)


def test_supporters_keep_the_eligibility_order() -> None:
    """§32. Branch timeframe order, then decision order."""
    result = consolidated(blocks=WIDE_BLOCKS)
    order = [
        decision.candidate_id
        for entry in result.eligibility.timeframes
        for decision in entry.decisions
        if decision.eligible
    ]

    for candidate in result.candidates:
        positions = [order.index(member.candidate_id) for member in candidate.supporting_decisions]
        assert positions == sorted(positions), candidate.consolidated_candidate_id


# --------------------------------------------------------------------------
# §41-§46: policy blast radius
# --------------------------------------------------------------------------


def identities(result: CandidateConsolidationAnalysis) -> set[str]:
    return {candidate.consolidated_candidate_id for candidate in result.candidates}


def by_id(result: CandidateConsolidationAnalysis) -> dict[str, tuple[object, ...]]:
    return {
        candidate.consolidated_candidate_id: (
            candidate.role,
            candidate.entry_side,
            candidate.lower,
            candidate.upper,
            candidate.reference_level,
        )
        for candidate in result.candidates
    }


def test_a_status_policy_change_never_moves_a_surviving_candidate() -> None:
    """§41. Membership may change; the candidate may not."""
    narrow = consolidated()
    wide = consolidated(blocks=WIDE_BLOCKS)

    assert identities(narrow) < identities(wide), "the wider policy adds one"
    shared = identities(narrow) & identities(wide)
    for identity in shared:
        assert by_id(narrow)[identity] == by_id(wide)[identity]


def test_a_gap_policy_change_never_moves_a_surviving_candidate() -> None:
    """§41."""
    everything = consolidated()
    open_only = consolidated(gaps=frozenset({FvgStatus.OPEN}))

    shared = identities(everything) & identities(open_only)
    assert shared
    for identity in shared:
        assert by_id(everything)[identity] == by_id(open_only)[identity]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (OrderBlockMitigationRule.TOUCH, OrderBlockMitigationRule.MIDPOINT),
        (OrderBlockMitigationRule.TOUCH, OrderBlockMitigationRule.FULL_ZONE),
        (OrderBlockMitigationRule.MIDPOINT, OrderBlockMitigationRule.FULL_ZONE),
    ],
)
def test_the_mitigation_rule_leaves_gap_backed_candidates_alone(
    first: OrderBlockMitigationRule, second: OrderBlockMitigationRule
) -> None:
    """§42."""
    left = consolidated(rule=first, blocks=WIDE_BLOCKS)
    right = consolidated(rule=second, blocks=WIDE_BLOCKS)

    def gap_backed(result: CandidateConsolidationAnalysis) -> list[object]:
        return [
            (candidate.consolidated_candidate_id, candidate.support_count)
            for candidate in result.candidates
            if CandidateSourceKind.ORDER_BLOCK not in candidate.source_kinds
        ]

    assert gap_backed(left) == gap_backed(right)
    for identity in identities(left) & identities(right):
        assert by_id(left)[identity] == by_id(right)[identity]


def test_the_zone_basis_may_move_block_candidates_and_leaves_gaps_alone() -> None:
    """§43."""
    full = consolidated(basis=OrderBlockZoneBasis.FULL_CANDLE, blocks=WIDE_BLOCKS)
    body = consolidated(basis=OrderBlockZoneBasis.BODY, blocks=WIDE_BLOCKS)

    def gap_backed(result: CandidateConsolidationAnalysis) -> list[object]:
        return [
            (candidate.consolidated_candidate_id, candidate.support_count)
            for candidate in result.candidates
            if CandidateSourceKind.ORDER_BLOCK not in candidate.source_kinds
        ]

    assert gap_backed(full) == gap_backed(body)
    block_ids = {
        result: {
            candidate.consolidated_candidate_id
            for candidate in result.candidates
            if CandidateSourceKind.ORDER_BLOCK in candidate.source_kinds
        }
        for result in (full, body)
    }
    assert block_ids[full] != block_ids[body]


def test_the_liquidity_tolerance_leaves_entry_candidates_alone() -> None:
    """§44."""
    narrow = consolidated(tolerance="0.50", blocks=WIDE_BLOCKS)
    wide = consolidated(tolerance="25.00", blocks=WIDE_BLOCKS)

    assert shape(narrow) == shape(wide), "every candidate here is an entry zone"
    assert narrow.reference_price == wide.reference_price


def test_the_atr_period_changes_nothing_but_the_carried_config() -> None:
    """§45. Documented rather than asserted away, as in the previous round."""
    short = consolidated(atr=5)
    long = consolidated(atr=20)

    assert short != long, "the retained composite config differs"
    assert short.composite_config != long.composite_config
    assert short.candidates == long.candidates
    assert short.reference_price == long.reference_price


def test_a_reference_price_move_keeps_surviving_candidate_identities() -> None:
    """§46. The global price moves, and a candidate that survives is the same one."""
    from goldpipeline.services.ict_candidate_eligibility import analyse_candidate_eligibility
    from goldpipeline.services.ict_composite import analyse_ict_composite
    from tests.test_ict_candidate_eligibility_fixture import (
        eligibility_config,
        staggered_snapshot,
    )
    from tests.test_ict_composite_fixture import PATHS, config
    from tests.test_ict_range_fixture import QUIET

    changed = dict(PATHS)
    changed[Timeframe.M1] = [*QUIET, *QUIET, *QUIET, *QUIET]
    moved = consolidate_candidates(
        analyse_candidate_eligibility(
            analyse_ict_composite(staggered_snapshot(changed), config=config()),
            config=eligibility_config(),
        )
    )
    before = consolidated()

    assert before.reference_price.price != moved.reference_price.price
    shared = identities(before) & identities(moved)
    assert shared, "at least one candidate survived the move"
    for identity in shared:
        assert by_id(before)[identity] == by_id(moved)[identity]
