"""The five-timeframe feature graph, pinned exactly.

Round 6.6f §33-§37, §44. Every number below was read out of the engine before it
was written down.

The staggered Round 6.6e.2b snapshot turns out to contain most of this round's
matrices without any construction: a candidate whose two supporters are an hour
apart in age, a pair that touches at exactly one price, three pairs that are
genuinely disjoint, zones that sit wholly in discount, wholly in premium, across
equilibrium and entirely outside the active range, and two timeframes that have
no active range at all.

What it does not contain is an eligible liquidity reference - every M15 pool in
it is terminal - so §34's reference features are pinned on the constructed
readings instead, and this file says so rather than implying the coverage.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.services.ict_candidate_consolidation import consolidate_candidates
from goldpipeline.services.ict_candidate_eligibility import CandidateRole, EntrySide
from goldpipeline.services.ict_candidate_features import (
    CandidateFeatureAnalysis,
    ZoneRelation,
    build_candidate_features,
)
from goldpipeline.services.ict_candidate_source import CandidateSourceKind
from goldpipeline.services.ict_fvg import FvgStatus
from goldpipeline.services.ict_order_block import OrderBlockZoneBasis
from goldpipeline.services.ict_order_block_lifecycle import (
    OrderBlockMitigationRule,
    OrderBlockStatus,
)
from goldpipeline.services.ict_range import PriceLocation
from goldpipeline.services.ict_structure import BreakDirection, StructureBias
from tests.test_ict_candidate_eligibility_fixture import analysis

WIDE_BLOCKS = frozenset(
    {OrderBlockStatus.ACTIVE, OrderBlockStatus.TOUCHED, OrderBlockStatus.MITIGATED}
)


def graph(**kwargs: object) -> CandidateFeatureAnalysis:
    return build_candidate_features(consolidate_candidates(analysis(**kwargs)))  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# §33: the whole feature graph
# --------------------------------------------------------------------------


def test_the_six_feature_records_are_pinned() -> None:
    """§33. Identity, side, geometry, the three counts and both ages."""
    got = [
        (
            feature.candidate_id,
            feature.entry_side.value if feature.entry_side else None,
            str(feature.lower),
            str(feature.upper),
            feature.support_count,
            feature.timeframe_count,
            feature.source_kind_count,
            feature.oldest_source_age_seconds,
            feature.newest_source_age_seconds,
        )
        for feature in graph().candidates
    ]

    assert got == [
        ("097b4590617fa647", "BAI", "3960", "4050", 2, 1, 1, 3780, 180),
        ("a5cc2e8b2526914a", "BAI", "4012", "4025", 1, 1, 1, 360, 360),
        ("b0ac49bdde522f4a", "BAI", "4030", "4052", 1, 1, 1, 240, 240),
        ("edf4afe7ca543f85", "BAI", "4025", "4050", 1, 1, 1, 300, 300),
        ("17d67732ea0614e7", "SEO", "4046", "4048", 1, 1, 1, 0, 0),
        ("be94c8741bcb02a3", "SEO", "4045", "4052", 1, 1, 1, 60, 60),
    ]


def test_the_feature_ids_are_the_consolidated_ids() -> None:
    """§33. The feature layer renames nothing."""
    result = graph()

    assert [feature.candidate_id for feature in result.candidates] == [
        candidate.consolidated_candidate_id for candidate in result.consolidation.candidates
    ]


def test_the_reference_price_is_the_readings_own() -> None:
    """§33."""
    result = graph()

    assert result.reference_price.price == Decimal("4043")
    assert result.reference_price is result.consolidation.reference_price
    assert [witness.timeframe.value for witness in result.reference_price.witnesses] == ["M1"]


# --------------------------------------------------------------------------
# §37: ages, read from the real snapshot
# --------------------------------------------------------------------------


def test_the_two_supporter_candidate_has_two_genuinely_different_ages() -> None:
    """§37. An hour apart, which is what makes the bracket meaningful."""
    pair = graph().candidates[0]

    assert pair.support_count == 2
    assert [fact.age_seconds for fact in pair.support_facts] == [3780, 180]
    assert pair.oldest_source_age_seconds == 3780
    assert pair.newest_source_age_seconds == 180
    assert pair.oldest_source_formed_at < pair.newest_source_formed_at


def test_every_age_in_the_fixture_is_non_negative() -> None:
    """§5, over the whole realistic reading."""
    for feature in graph(blocks=WIDE_BLOCKS).candidates:
        for fact in feature.support_facts:
            assert fact.age_seconds >= 0, (feature.candidate_id, fact.source_id)
            assert fact.formed_at <= graph().observed_at


def test_the_oldest_and_newest_are_the_edges_of_the_support_facts() -> None:
    """§5. Not a separate computation that could disagree."""
    for feature in graph(blocks=WIDE_BLOCKS).candidates:
        ages = [fact.age_seconds for fact in feature.support_facts]
        assert feature.oldest_source_age_seconds == max(ages)
        assert feature.newest_source_age_seconds == min(ages)


# --------------------------------------------------------------------------
# §35: pair relations, read from the real snapshot
# --------------------------------------------------------------------------


def test_the_fifteen_pair_relations_are_pinned() -> None:
    """§35. Fifteen pairs for six candidates - every pair, including disjoint ones."""
    got = [
        (
            relation.first_candidate_id[:8],
            relation.second_candidate_id[:8],
            relation.relation.value,
            None if relation.intersection_lower is None else str(relation.intersection_lower),
            None if relation.intersection_upper is None else str(relation.intersection_upper),
            None if relation.intersection_width is None else str(relation.intersection_width),
            None if relation.gap_distance is None else str(relation.gap_distance),
        )
        for relation in graph().pair_relations
    ]

    assert got == [
        ("097b4590", "17d67732", "OVERLAPPING", "4046", "4048", "2", None),
        ("097b4590", "a5cc2e8b", "OVERLAPPING", "4012", "4025", "13", None),
        ("097b4590", "b0ac49bd", "OVERLAPPING", "4030", "4050", "20", None),
        ("097b4590", "be94c874", "OVERLAPPING", "4045", "4050", "5", None),
        ("097b4590", "edf4afe7", "OVERLAPPING", "4025", "4050", "25", None),
        ("17d67732", "a5cc2e8b", "DISJOINT", None, None, None, "21"),
        ("17d67732", "b0ac49bd", "OVERLAPPING", "4046", "4048", "2", None),
        ("17d67732", "be94c874", "OVERLAPPING", "4046", "4048", "2", None),
        ("17d67732", "edf4afe7", "OVERLAPPING", "4046", "4048", "2", None),
        ("a5cc2e8b", "b0ac49bd", "DISJOINT", None, None, None, "5"),
        ("a5cc2e8b", "be94c874", "DISJOINT", None, None, None, "20"),
        ("a5cc2e8b", "edf4afe7", "TOUCHING", "4025", "4025", "0", None),
        ("b0ac49bd", "be94c874", "OVERLAPPING", "4045", "4052", "7", None),
        ("b0ac49bd", "edf4afe7", "OVERLAPPING", "4030", "4050", "20", None),
        ("be94c874", "edf4afe7", "OVERLAPPING", "4045", "4050", "5", None),
    ]


def test_the_pair_count_is_every_entry_zone_pair_exactly_once() -> None:
    """§8. Six zones give fifteen pairs; nothing is skipped for being boring."""
    result = graph()
    zones = [c for c in result.candidates if c.role is CandidateRole.ENTRY_ZONE]

    assert len(zones) == 6
    assert len(result.pair_relations) == 15 == len(zones) * (len(zones) - 1) // 2
    keys = {
        (relation.first_candidate_id, relation.second_candidate_id)
        for relation in result.pair_relations
    }
    assert len(keys) == 15


def test_the_realistic_snapshot_contains_all_three_geometric_answers() -> None:
    """§35. Found, not constructed."""
    seen = {relation.relation for relation in graph().pair_relations}

    assert seen == {ZoneRelation.OVERLAPPING, ZoneRelation.TOUCHING, ZoneRelation.DISJOINT}


def test_the_touching_pair_shares_exactly_one_price() -> None:
    """§9. 4012-4025 and 4025-4050 meet at 4025 and nowhere else."""
    relation = graph().relation("a5cc2e8b2526914a", "edf4afe7ca543f85")

    assert relation is not None
    assert relation.relation is ZoneRelation.TOUCHING
    assert relation.intersection_lower == relation.intersection_upper == Decimal("4025")
    assert relation.intersection_width == Decimal("0")
    assert relation.gap_distance is None


def test_a_disjoint_pair_reports_its_exact_gap_and_no_intersection() -> None:
    """§9. 4012-4025 and 4030-4052 are five apart."""
    relation = graph().relation("a5cc2e8b2526914a", "b0ac49bdde522f4a")

    assert relation is not None
    assert relation.relation is ZoneRelation.DISJOINT
    assert relation.gap_distance == Decimal("5")
    assert relation.intersection_lower is None
    assert relation.intersection_upper is None
    assert relation.intersection_width is None


def test_a_fully_contained_zone_on_the_opposite_side_stays_its_own_candidate() -> None:
    """§10, §26. SEO 4046-4048 sits inside BAI 4030-4052 and is not absorbed."""
    result = graph()
    relation = result.relation("17d67732ea0614e7", "b0ac49bdde522f4a")
    inner = result.feature("17d67732ea0614e7")
    outer = result.feature("b0ac49bdde522f4a")

    assert relation is not None and inner is not None and outer is not None
    assert relation.relation is ZoneRelation.OVERLAPPING
    assert (relation.intersection_lower, relation.intersection_upper) == (
        inner.lower,
        inner.upper,
    ), "the intersection is the whole inner zone"
    assert inner.entry_side is EntrySide.SEO
    assert outer.entry_side is EntrySide.BAI


def test_no_relation_invented_a_candidate() -> None:
    """§10. Fifteen relations, six candidates, still exactly the same six.

    Two intersections do coincide with a candidate's own bounds - 4012-4025 and
    4046-4048 - and that is containment being reported, not a zone being minted:
    each is a candidate that already existed, wholly inside a larger one. What
    would be alarming is a *union* appearing in the candidate set, so that is
    what is checked.
    """
    result = graph()
    bounds: list[tuple[Decimal, Decimal]] = []
    for feature in result.candidates:
        assert feature.lower is not None and feature.upper is not None
        bounds.append((feature.lower, feature.upper))

    assert bounds == [
        (candidate.lower, candidate.upper) for candidate in result.consolidation.candidates
    ]

    unions = {
        (min(first[0], second[0]), max(first[1], second[1]))
        for first in bounds
        for second in bounds
        if first != second
    } - set(bounds)
    assert unions, "the fixture must contain unions that could have been invented"
    assert unions.isdisjoint(bounds)


# --------------------------------------------------------------------------
# §36: range context, read from the real snapshot
# --------------------------------------------------------------------------


def test_the_five_timeframe_contexts_are_pinned() -> None:
    """§7, §36. Two timeframes have an active range and three do not."""
    got = [
        (
            context.timeframe.value,
            context.structure_bias.value,
            context.active_range_id,
            None if context.range_lower is None else str(context.range_lower),
            None if context.range_upper is None else str(context.range_upper),
            None if context.range_equilibrium is None else str(context.range_equilibrium),
            context.decision_count,
            context.eligible_count,
        )
        for context in graph().timeframe_contexts
    ]

    assert got == [
        ("H4", "BEARISH", "110d6bbebcbb891a", "3945", "4030", "3987.5", 1, 0),
        ("H1", "BEARISH", "c2dc66dc6e3d3e7b", "3900", "4200", "4050", 8, 2),
        ("M15", "BULLISH", None, None, None, None, 5, 0),
        ("M5", "NEUTRAL", None, None, None, None, 0, 0),
        ("M1", "BEARISH", None, None, None, None, 13, 5),
    ]


@pytest.mark.parametrize(
    ("candidate_id", "timeframe", "locations"),
    [
        # whole zone in discount
        ("a5cc2e8b2526914a", Timeframe.H1, ("DISCOUNT", "DISCOUNT", "DISCOUNT")),
        # whole zone in premium
        ("a5cc2e8b2526914a", Timeframe.H4, ("PREMIUM", "PREMIUM", "PREMIUM")),
        # crosses equilibrium
        ("b0ac49bdde522f4a", Timeframe.H1, ("DISCOUNT", "DISCOUNT", "PREMIUM")),
        # wholly outside the active range
        ("17d67732ea0614e7", Timeframe.H4, ("ABOVE_RANGE", "ABOVE_RANGE", "ABOVE_RANGE")),
        # lands exactly on equilibrium at its upper edge
        ("097b4590617fa647", Timeframe.H1, ("DISCOUNT", "DISCOUNT", "EQUILIBRIUM")),
        # crosses out of the range entirely
        ("097b4590617fa647", Timeframe.H4, ("DISCOUNT", "PREMIUM", "ABOVE_RANGE")),
    ],
)
def test_the_range_context_matrix(
    candidate_id: str, timeframe: Timeframe, locations: tuple[str, str, str]
) -> None:
    """§36. Five distinct location shapes, all present in the real snapshot."""
    feature = graph().feature(candidate_id)

    assert feature is not None
    context = feature.range_context(timeframe)
    assert context is not None
    assert (
        context.lower_location,
        context.midpoint_location,
        context.upper_location,
    ) == tuple(PriceLocation(name) for name in locations)
    assert context.reference_location is None


@pytest.mark.parametrize("timeframe", [Timeframe.M15, Timeframe.M5, Timeframe.M1])
def test_a_timeframe_without_an_active_range_reports_nothing(timeframe: Timeframe) -> None:
    """§36. Absence is a state, not a gap to fill."""
    for feature in graph().candidates:
        context = feature.range_context(timeframe)
        assert context is not None
        assert context.active_range_id is None
        assert context.lower_location is None
        assert context.upper_location is None
        assert context.reference_location is None


def test_no_candidate_is_rejected_for_where_it_sits() -> None:
    """§36, §6. Range context is metadata, still not a filter."""
    result = graph()
    outside = result.feature("17d67732ea0614e7")

    assert outside is not None
    h4 = outside.range_context(Timeframe.H4)
    assert h4 is not None and h4.lower_location is PriceLocation.ABOVE_RANGE
    assert len(result.candidates) == 6, "the outside-range candidate is still one of six"


def test_every_candidate_gets_a_context_for_every_timeframe() -> None:
    """§6. One record per supplied timeframe, always."""
    result = graph()

    for feature in result.candidates:
        assert [context.timeframe for context in feature.range_contexts] == [
            context.timeframe for context in result.timeframe_contexts
        ]


def test_the_range_direction_and_edges_travel_with_the_location() -> None:
    """§6. A location is only meaningful beside the range it was measured in."""
    feature = graph().feature("a5cc2e8b2526914a")

    assert feature is not None
    context = feature.range_context(Timeframe.H4)
    assert context is not None
    assert context.active_range_id == "110d6bbebcbb891a"
    assert (context.range_lower, context.range_upper) == (Decimal("3945"), Decimal("4030"))
    assert context.range_equilibrium == Decimal("3987.5")
    assert context.range_direction is BreakDirection.BEARISH


# --------------------------------------------------------------------------
# §11, §4: support facts on the real snapshot
# --------------------------------------------------------------------------


def test_the_order_block_pair_carries_both_policies_on_both_supporters() -> None:
    """§4, §17. Under the wide policy the same-source block pair is eligible."""
    result = graph(blocks=WIDE_BLOCKS)
    pair = next(
        feature
        for feature in result.candidates
        if feature.source_kinds == (CandidateSourceKind.ORDER_BLOCK,)
    )

    assert pair.support_count == 2
    assert pair.source_kind_count == 1
    for fact in pair.support_facts:
        assert fact.order_block_status in WIDE_BLOCKS
        assert fact.order_block_zone_basis is OrderBlockZoneBasis.FULL_CANDLE
        assert fact.order_block_mitigation_rule is OrderBlockMitigationRule.TOUCH
        assert fact.fvg_status is None
    assert len({fact.source_id for fact in pair.support_facts}) == 2


def test_every_gap_backed_feature_carries_only_a_gap_status() -> None:
    """§4."""
    for feature in graph().candidates:
        for fact in feature.support_facts:
            assert fact.source_kind is CandidateSourceKind.FAIR_VALUE_GAP
            assert fact.fvg_status in {FvgStatus.OPEN, FvgStatus.TOUCHED}
            assert fact.order_block_status is None
            assert fact.liquidity_pool_status is None


def test_the_fixture_has_no_eligible_reference_and_says_so() -> None:
    """§34. Recorded rather than papered over; references are pinned elsewhere."""
    result = graph(blocks=WIDE_BLOCKS)

    assert result.of_role(CandidateRole.UPPER_REFERENCE) == ()
    assert result.of_role(CandidateRole.LOWER_REFERENCE) == ()
    m15 = result.consolidation.eligibility.require(Timeframe.M15)
    assert len(m15.decisions) == 5
    assert all(not decision.eligible for decision in m15.decisions)


def test_support_totals_match_the_consolidation() -> None:
    """§11."""
    for kwargs in ({}, {"blocks": WIDE_BLOCKS}, {"gaps": frozenset({FvgStatus.OPEN})}):
        result = graph(**kwargs)
        assert sum(feature.support_count for feature in result.candidates) == sum(
            candidate.support_count for candidate in result.consolidation.candidates
        )
        assert len(result.candidates) == len(result.consolidation.candidates)


def test_the_structure_bias_vocabulary_is_untranslated() -> None:
    """§7. Three values, no score, no alignment number."""
    seen = {context.structure_bias for context in graph().timeframe_contexts}

    assert seen <= {StructureBias.NEUTRAL, StructureBias.BULLISH, StructureBias.BEARISH}
    assert seen == {StructureBias.BEARISH, StructureBias.BULLISH, StructureBias.NEUTRAL}
