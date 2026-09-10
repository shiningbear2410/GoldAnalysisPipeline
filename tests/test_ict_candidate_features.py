"""What the feature graph says about candidates, one constructed case at a time.

Round 6.6f §3-§14. Everything here is built rather than found, so each case can
say exactly one thing. The realistic five-timeframe reading is pinned separately.

The helpers below reuse Round 6.6e.2c1's constructed-reading machinery, so a
"candidate" in this file is a genuine consolidation of genuine eligible
decisions - not a dataclass filled in by hand that happens to have the right
field names.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.services.ict_candidate_consolidation import consolidate_candidates
from goldpipeline.services.ict_candidate_eligibility import (
    CandidateEligibilityAnalysis,
    CandidateEligibilityDecision,
    CandidateEligibilityTimeframe,
    CandidateRole,
    EntrySide,
    MarketRelation,
)
from goldpipeline.services.ict_candidate_features import (
    CANDIDATE_FEATURE_METHOD_VERSION,
    FEATURE_TIMEFRAME_ORDER,
    CandidateFeature,
    CandidateFeatureAnalysis,
    CandidateFeatureError,
    ZoneRelation,
    build_candidate_features,
    relate_zones,
)
from goldpipeline.services.ict_candidate_source import CandidateSource, CandidateSourceKind
from goldpipeline.services.ict_fvg import FvgStatus
from goldpipeline.services.ict_liquidity import LiquiditySide, PoolStatus
from goldpipeline.services.ict_order_block import OrderBlockZoneBasis
from goldpipeline.services.ict_order_block_lifecycle import (
    OrderBlockMitigationRule,
    OrderBlockStatus,
)
from goldpipeline.services.ict_primitives import GapDirection
from goldpipeline.services.ict_range import DealingRange, PriceLocation, RangeStatus
from goldpipeline.services.ict_structure import BreakDirection, StructureBias
from tests.test_ict_candidate_consolidation import MOMENT, decide, eligibility
from tests.test_ict_candidate_eligibility import block_source, gap_source, pool_source

HOUR = timedelta(hours=1)
MINUTE = timedelta(minutes=1)


def features(
    *decisions: CandidateEligibilityDecision, price: str = "4000"
) -> CandidateFeatureAnalysis:
    """A feature analysis over a constructed reading, through the real chain."""
    return build_candidate_features(consolidate_candidates(eligibility(*decisions, price=price)))


def aged(source: CandidateSource, *, seconds: int) -> CandidateSource:
    """The same source, formed *seconds* before the observation instant."""
    return replace(source, formed_at=MOMENT - timedelta(seconds=seconds))


def only(analysis: CandidateFeatureAnalysis) -> CandidateFeature:
    assert len(analysis.candidates) == 1, [c.candidate_id for c in analysis.candidates]
    return analysis.candidates[0]


def with_range(
    analysis: CandidateEligibilityAnalysis,
    *,
    timeframe: Timeframe = Timeframe.H1,
    lower: str = "3900",
    upper: str = "4100",
    bias: StructureBias = StructureBias.BULLISH,
) -> CandidateEligibilityAnalysis:
    """The same reading with one timeframe carrying an active dealing range.

    Constructed rather than driven from bars, because what is under test here is
    the classification of a candidate against two edges - the range engine's own
    tests already prove those edges are found correctly.
    """
    low, high = Decimal(lower), Decimal(upper)
    found = DealingRange(
        range_id=f"range:{lower}-{upper}",
        method_version="1.0.0",
        timeframe=timeframe,
        symbol="XAUUSD",
        direction=BreakDirection.BULLISH,
        protected_assignment_id="pa",
        structural_leg_id="leg",
        establishing_event_id="event",
        formed_at=MOMENT - HOUR,
        origin_price=low,
        initial_terminal_price=high,
        lower=low,
        upper=high,
        equilibrium=(low + high) / Decimal(2),
        width=high - low,
        terminal_price=high,
        terminal_bar_open_time=MOMENT - HOUR,
        terminal_bar_close_time=MOMENT,
        status=RangeStatus.ACTIVE,
        superseded_at=None,
        superseded_by_event_id=None,
        superseded_by_range_id=None,
    )
    return replace(
        analysis,
        timeframes=tuple(
            replace(entry, structure_bias=bias, active_dealing_range=found)
            if entry.timeframe is timeframe
            else entry
            for entry in analysis.timeframes
        ),
    )


# --------------------------------------------------------------------------
# §3: the feature model
# --------------------------------------------------------------------------


def test_a_feature_copies_the_candidate_geometry_exactly() -> None:
    """§3. Nothing is recomputed, so nothing can drift."""
    analysis = features(decide(gap_source("3990", "4010")))
    feature = only(analysis)
    candidate = analysis.candidate(feature.candidate_id)

    assert candidate is not None
    assert (feature.lower, feature.upper, feature.midpoint, feature.width) == (
        candidate.lower,
        candidate.upper,
        candidate.midpoint,
        candidate.width,
    )
    assert feature.reference_level == candidate.reference_level
    assert feature.role is candidate.role
    assert feature.entry_side is candidate.entry_side
    assert feature.market_relation is candidate.market_relation
    assert feature.distance_to_reference == candidate.distance_to_reference


def test_a_feature_carries_no_score_of_any_kind() -> None:
    """§3. Enumerated as a field set, so a new field has to be argued for."""
    fields = set(CandidateFeature.__dataclass_fields__)

    assert fields == {
        "candidate_id",
        "method_version",
        "symbol",
        "role",
        "entry_side",
        "lower",
        "upper",
        "midpoint",
        "width",
        "reference_level",
        "market_relation",
        "distance_to_reference",
        "support_count",
        "timeframe_count",
        "source_kind_count",
        "source_kinds",
        "timeframes",
        "oldest_source_formed_at",
        "newest_source_formed_at",
        "oldest_source_age_seconds",
        "newest_source_age_seconds",
        "support_facts",
        "range_contexts",
    }


def test_the_method_version_is_stamped() -> None:
    analysis = features(decide(gap_source("3990", "4010")))

    assert analysis.method_version == CANDIDATE_FEATURE_METHOD_VERSION
    assert only(analysis).method_version == CANDIDATE_FEATURE_METHOD_VERSION


def test_every_model_is_frozen() -> None:
    analysis = features(decide(gap_source("3990", "4010")))
    feature = only(analysis)

    for target, field in (
        (analysis, "candidates"),
        (feature, "support_count"),
        (feature.support_facts[0], "age_seconds"),
        (feature.range_contexts[0], "timeframe"),
        (analysis.timeframe_contexts[0], "structure_bias"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(target, field, None)


# --------------------------------------------------------------------------
# §4: support facts, in three vocabularies
# --------------------------------------------------------------------------


def test_an_order_block_keeps_its_own_policy_metadata() -> None:
    """§4. Zone basis and mitigation rule are facts about how it was measured."""
    fact = only(features(decide(block_source("3990", "4010")))).support_facts[0]

    assert fact.source_kind is CandidateSourceKind.ORDER_BLOCK
    assert fact.order_block_status is OrderBlockStatus.ACTIVE
    assert fact.order_block_zone_basis is OrderBlockZoneBasis.BODY
    assert fact.order_block_mitigation_rule is OrderBlockMitigationRule.TOUCH
    assert fact.fvg_status is None
    assert fact.liquidity_side is None
    assert fact.liquidity_pool_status is None


def test_a_gap_keeps_only_its_own_status() -> None:
    """§4."""
    fact = only(
        features(decide(gap_source("3990", "4010", GapDirection.BULLISH, FvgStatus.TOUCHED)))
    ).support_facts[0]

    assert fact.source_kind is CandidateSourceKind.FAIR_VALUE_GAP
    assert fact.fvg_status is FvgStatus.TOUCHED
    assert fact.order_block_status is None
    assert fact.order_block_zone_basis is None
    assert fact.liquidity_pool_status is None


def test_a_pool_keeps_its_side_status_and_tolerance() -> None:
    """§4."""
    fact = only(
        features(decide(pool_source("4050", "4050", LiquiditySide.BUY_SIDE)))
    ).support_facts[0]

    assert fact.source_kind is CandidateSourceKind.LIQUIDITY_POOL
    assert fact.liquidity_side is LiquiditySide.BUY_SIDE
    assert fact.liquidity_pool_status is PoolStatus.ACTIVE
    assert fact.liquidity_tolerance is not None
    assert fact.order_block_status is None
    assert fact.fvg_status is None


def test_the_three_status_vocabularies_are_three_fields() -> None:
    """§4. No shared ``status`` key exists to flatten them into.

    A block's TOUCHED and a gap's TOUCHED are different events measured by
    different engines, and one field would say they are the same.
    """
    from goldpipeline.services.ict_candidate_features import SupportFact

    fields = set(SupportFact.__dataclass_fields__)

    assert {"order_block_status", "fvg_status", "liquidity_pool_status"} <= fields
    assert "status" not in fields
    assert "lifecycle_status" not in fields


def test_support_facts_keep_the_supporting_decision_order() -> None:
    """§14."""
    analysis = features(
        decide(gap_source("3990", "4010"), tag="-open"),
        decide(gap_source("3990", "4010", GapDirection.BULLISH, FvgStatus.TOUCHED), tag="-touched"),
    )
    feature = only(analysis)
    candidate = analysis.candidate(feature.candidate_id)

    assert candidate is not None
    assert [fact.candidate_source_id for fact in feature.support_facts] == list(
        candidate.supporting_candidate_source_ids
    )


def test_a_supporter_can_be_walked_back_to_its_source() -> None:
    """§4. Ids are preserved, not summarised."""
    analysis = features(decide(block_source("3990", "4010")))
    fact = only(analysis).support_facts[0]
    candidate = analysis.candidate(only(analysis).candidate_id)

    assert candidate is not None
    decision = candidate.supporting_decisions[0]
    assert fact.candidate_id == decision.candidate_id
    assert fact.source_id == decision.source_id
    assert fact.candidate_source_id == decision.candidate_source_id


# --------------------------------------------------------------------------
# §5: age is a fact
# --------------------------------------------------------------------------


def test_age_is_the_difference_between_observation_and_formation() -> None:
    """§5. Subtraction, in whole seconds."""
    feature = only(features(decide(aged(gap_source("3990", "4010"), seconds=3600))))

    assert feature.oldest_source_age_seconds == 3600
    assert feature.newest_source_age_seconds == 3600
    assert feature.oldest_source_formed_at == MOMENT - HOUR
    assert feature.newest_source_formed_at == MOMENT - HOUR


def test_one_supporter_makes_oldest_and_newest_the_same() -> None:
    """§37."""
    feature = only(features(decide(aged(gap_source("3990", "4010"), seconds=600))))

    assert feature.support_count == 1
    assert feature.oldest_source_age_seconds == feature.newest_source_age_seconds == 600
    assert feature.oldest_source_formed_at == feature.newest_source_formed_at


def test_two_supporters_of_different_ages_bracket_the_candidate() -> None:
    """§37. Oldest is the largest age, newest the smallest - and they are edges."""
    feature = only(
        features(
            decide(aged(gap_source("3990", "4010"), seconds=7200), tag="-old"),
            decide(
                aged(
                    gap_source("3990", "4010", GapDirection.BULLISH, FvgStatus.TOUCHED),
                    seconds=60,
                ),
                tag="-new",
            ),
        )
    )

    assert feature.support_count == 2
    assert feature.oldest_source_age_seconds == 7200
    assert feature.newest_source_age_seconds == 60
    assert feature.oldest_source_formed_at < feature.newest_source_formed_at


def test_same_age_supporters_collapse_to_one_instant() -> None:
    """§37."""
    feature = only(
        features(
            decide(aged(gap_source("3990", "4010"), seconds=900), tag="-a"),
            decide(
                aged(
                    gap_source("3990", "4010", GapDirection.BULLISH, FvgStatus.TOUCHED),
                    seconds=900,
                ),
                tag="-b",
            ),
        )
    )

    assert feature.support_count == 2
    assert feature.oldest_source_age_seconds == feature.newest_source_age_seconds == 900
    assert feature.oldest_source_formed_at == feature.newest_source_formed_at


def test_a_candidate_gaining_a_newer_supporter_keeps_its_identity() -> None:
    """§37. Age facts move; the candidate does not."""
    before = only(features(decide(aged(gap_source("3990", "4010"), seconds=7200), tag="-old")))
    after = only(
        features(
            decide(aged(gap_source("3990", "4010"), seconds=7200), tag="-old"),
            decide(
                aged(
                    gap_source("3990", "4010", GapDirection.BULLISH, FvgStatus.TOUCHED),
                    seconds=30,
                ),
                tag="-new",
            ),
        )
    )

    assert before.candidate_id == after.candidate_id
    assert (before.lower, before.upper) == (after.lower, after.upper)
    assert before.newest_source_age_seconds == 7200
    assert after.newest_source_age_seconds == 30
    assert after.support_count == before.support_count + 1


def test_a_source_formed_after_the_observation_is_refused() -> None:
    """§5. A negative age means a leak or a broken snapshot. Neither is clamped."""
    future = replace(gap_source("3990", "4010"), formed_at=MOMENT + MINUTE)

    with pytest.raises(CandidateFeatureError, match="is after the observation"):
        features(decide(future))


def test_a_zero_age_is_allowed() -> None:
    """§5. Formed at the observing instant is legal, and common on M1."""
    feature = only(features(decide(gap_source("3990", "4010"))))

    assert feature.newest_source_age_seconds == 0


def test_no_freshness_notion_exists() -> None:
    """§5. Age is reported; nothing is rejected for having it."""
    old = only(features(decide(aged(gap_source("3990", "4010"), seconds=60 * 60 * 24 * 30))))

    assert old.oldest_source_age_seconds == 2592000
    assert old.support_count == 1, "a month-old source is still a candidate"


# --------------------------------------------------------------------------
# §6: range context
# --------------------------------------------------------------------------


def test_a_timeframe_with_no_active_range_reports_nothing_rather_than_guessing() -> None:
    """§6."""
    context = only(features(decide(gap_source("3990", "4010")))).range_context(Timeframe.H1)

    assert context is not None
    assert context.active_range_id is None
    assert context.range_lower is None
    assert context.lower_location is None
    assert context.midpoint_location is None
    assert context.upper_location is None
    assert context.reference_location is None


def ranged(*decisions: CandidateEligibilityDecision, **kwargs: object) -> CandidateFeatureAnalysis:
    return build_candidate_features(
        consolidate_candidates(with_range(eligibility(*decisions), **kwargs))  # type: ignore[arg-type]
    )


def test_an_entry_zone_gets_all_three_edges_classified() -> None:
    """§6. Lower, midpoint and upper, each exactly."""
    context = only(ranged(decide(gap_source("3920", "3960")))).range_context(Timeframe.H1)

    assert context is not None
    assert context.active_range_id == "range:3900-4100"
    assert context.range_equilibrium == Decimal("4000")
    assert context.lower_location is PriceLocation.DISCOUNT
    assert context.midpoint_location is PriceLocation.DISCOUNT
    assert context.upper_location is PriceLocation.DISCOUNT
    assert context.reference_location is None


def test_a_zone_crossing_equilibrium_says_so_edge_by_edge() -> None:
    """§36. No single verdict for a zone that is in two halves at once."""
    context = only(ranged(decide(gap_source("3960", "4040")))).range_context(Timeframe.H1)

    assert context is not None
    assert context.lower_location is PriceLocation.DISCOUNT
    assert context.midpoint_location is PriceLocation.EQUILIBRIUM
    assert context.upper_location is PriceLocation.PREMIUM


def test_a_zone_outside_the_active_range_is_located_outside_it() -> None:
    """§36. Outside is a location, not an error and not a rejection."""
    analysis = ranged(decide(gap_source("3800", "3850")), lower="3900", upper="4100")
    context = only(analysis).range_context(Timeframe.H1)

    assert context is not None
    assert context.lower_location is PriceLocation.BELOW_RANGE
    assert context.upper_location is PriceLocation.BELOW_RANGE
    assert len(analysis.candidates) == 1, "still a candidate"


def test_a_reference_gets_its_level_classified_and_no_zone() -> None:
    """§6, §12. A level has one location, and no invented edges."""
    context = only(
        ranged(decide(pool_source("4050", "4050", LiquiditySide.BUY_SIDE)))
    ).range_context(Timeframe.H1)

    assert context is not None
    assert context.reference_location is PriceLocation.PREMIUM
    assert context.lower_location is None
    assert context.midpoint_location is None
    assert context.upper_location is None


def test_the_range_edges_are_preserved_beside_the_locations() -> None:
    """§6."""
    context = only(ranged(decide(gap_source("3920", "3960")))).range_context(Timeframe.H1)

    assert context is not None
    assert (context.range_lower, context.range_upper) == (Decimal("3900"), Decimal("4100"))
    assert context.range_direction is BreakDirection.BULLISH


def test_range_contexts_follow_the_timeframe_order() -> None:
    """§14. Slowest to fastest, and one per timeframe present."""
    analysis = features(
        decide(gap_source("3990", "4010")),
        decide(gap_source("3990", "4010"), timeframe=Timeframe.M15),
    )
    feature = only(analysis)

    assert [context.timeframe for context in feature.range_contexts] == [
        Timeframe.H1,
        Timeframe.M15,
    ]
    assert [context.timeframe for context in analysis.timeframe_contexts] == [
        Timeframe.H1,
        Timeframe.M15,
    ]


def test_the_feature_timeframe_order_is_slowest_to_fastest() -> None:
    """§14."""
    assert FEATURE_TIMEFRAME_ORDER == (
        Timeframe.H4,
        Timeframe.H1,
        Timeframe.M15,
        Timeframe.M5,
        Timeframe.M1,
    )


# --------------------------------------------------------------------------
# §7: structure context
# --------------------------------------------------------------------------


def test_structure_bias_is_carried_per_timeframe_untranslated() -> None:
    """§7. Three values, no alignment score."""
    analysis = ranged(decide(gap_source("3920", "3960")), bias=StructureBias.BEARISH)
    context = analysis.timeframe_context(Timeframe.H1)

    assert context is not None
    assert context.structure_bias is StructureBias.BEARISH


def test_a_timeframe_context_counts_its_decisions() -> None:
    """§7. Two counts, both lengths."""
    analysis = features(
        decide(gap_source("3990", "4010")),
        decide(gap_source("4000", "4020")),
    )
    context = analysis.timeframe_context(Timeframe.H1)

    assert context is not None
    assert context.decision_count == 2
    assert context.eligible_count == 2


# --------------------------------------------------------------------------
# §8-§9: pair relations
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second", "expected", "lower", "upper", "width", "gap"),
    [
        (("3990", "4010"), ("4000", "4020"), ZoneRelation.OVERLAPPING, "4000", "4010", "10", None),
        (("3990", "4010"), ("4010", "4030"), ZoneRelation.TOUCHING, "4010", "4010", "0", None),
        (("3990", "4010"), ("4010.01", "4030"), ZoneRelation.DISJOINT, None, None, None, "0.01"),
        (("3990", "4010"), ("3995", "4005"), ZoneRelation.OVERLAPPING, "3995", "4005", "10", None),
        (("3990", "4010"), ("3990", "4010"), ZoneRelation.EQUAL, "3990", "4010", "20", None),
        (("3990", "4010"), ("4100", "4120"), ZoneRelation.DISJOINT, None, None, None, "90"),
    ],
)
def test_the_overlap_matrix(
    first: tuple[str, str],
    second: tuple[str, str],
    expected: ZoneRelation,
    lower: str | None,
    upper: str | None,
    width: str | None,
    gap: str | None,
) -> None:
    """§35, §9. Exact classification, exact intersection, exact gap. No tolerance."""
    relation, got_lower, got_upper, got_gap = relate_zones(
        first_lower=Decimal(first[0]),
        first_upper=Decimal(first[1]),
        second_lower=Decimal(second[0]),
        second_upper=Decimal(second[1]),
    )

    assert relation is expected
    assert got_lower == (None if lower is None else Decimal(lower))
    assert got_upper == (None if upper is None else Decimal(upper))
    assert got_gap == (None if gap is None else Decimal(gap))
    if width is not None:
        assert got_upper is not None and got_lower is not None
        assert got_upper - got_lower == Decimal(width)


def test_relate_zones_is_symmetric() -> None:
    """§8. Undirected, so the answer cannot depend on which came first."""
    for first, second in (
        (("3990", "4010"), ("4000", "4020")),
        (("3990", "4010"), ("4010", "4030")),
        (("3990", "4010"), ("4020", "4030")),
    ):
        forward = relate_zones(
            first_lower=Decimal(first[0]),
            first_upper=Decimal(first[1]),
            second_lower=Decimal(second[0]),
            second_upper=Decimal(second[1]),
        )
        backward = relate_zones(
            first_lower=Decimal(second[0]),
            first_upper=Decimal(second[1]),
            second_lower=Decimal(first[0]),
            second_upper=Decimal(first[1]),
        )
        assert forward == backward, (first, second)


def test_a_pair_relation_is_stored_once_under_sorted_ids() -> None:
    """§8, §14."""
    analysis = features(
        decide(gap_source("3990", "4010"), price="4100"),
        decide(gap_source("4000", "4020"), price="4100"),
        price="4100",
    )
    first, second = (candidate.candidate_id for candidate in analysis.candidates)
    relations = analysis.pair_relations

    assert len(relations) == 1
    assert relations[0].first_candidate_id < relations[0].second_candidate_id
    assert analysis.relation(first, second) is analysis.relation(second, first)


def test_every_entry_zone_pair_gets_a_relation_including_disjoint_ones() -> None:
    """§8. Absence would be ambiguous: "no overlap" or "nobody looked"?"""
    analysis = features(
        decide(gap_source("3990", "4010"), price="4200"),
        decide(gap_source("4020", "4040"), price="4200"),
        decide(gap_source("4100", "4120"), price="4200"),
        price="4200",
    )

    assert len(analysis.candidates) == 3
    assert len(analysis.pair_relations) == 3
    assert all(relation.relation is ZoneRelation.DISJOINT for relation in analysis.pair_relations)


def test_equal_geometry_on_opposite_sides_is_equal_and_still_two_candidates() -> None:
    """§8. Geometry matching is not semantics matching."""
    analysis = features(
        decide(gap_source("3990", "4010", GapDirection.BULLISH)),
        decide(gap_source("3990", "4010", GapDirection.BEARISH)),
    )

    assert len(analysis.candidates) == 2
    assert {candidate.entry_side for candidate in analysis.candidates} == {
        EntrySide.BAI,
        EntrySide.SEO,
    }
    assert len(analysis.pair_relations) == 1
    assert analysis.pair_relations[0].relation is ZoneRelation.EQUAL
    assert analysis.pair_relations[0].intersection_width == Decimal("20")


def test_a_reference_candidate_takes_part_in_no_pair_relation() -> None:
    """§12. A level is not a zone, so it has no overlap to report."""
    analysis = features(
        decide(gap_source("3990", "4010")),
        decide(pool_source("4050", "4050", LiquiditySide.BUY_SIDE)),
    )

    assert len(analysis.candidates) == 2
    assert analysis.pair_relations == ()


# --------------------------------------------------------------------------
# §10: no transitive merge
# --------------------------------------------------------------------------


def test_a_chain_of_overlaps_creates_relations_and_no_new_candidate() -> None:
    """§10, and the same trap Round 6.6e.2c1 refused at the consolidation layer."""
    analysis = features(
        decide(gap_source("3990", "4010"), price="4200"),
        decide(gap_source("4000", "4020"), price="4200"),
        decide(gap_source("4010", "4030"), price="4200"),
        price="4200",
    )

    assert len(analysis.candidates) == 3
    assert len(analysis.pair_relations) == 3
    bounds = {(candidate.lower, candidate.upper) for candidate in analysis.candidates}
    assert (Decimal("3990"), Decimal("4030")) not in bounds, "no union zone was invented"
    assert all(candidate.support_count == 1 for candidate in analysis.candidates)


def test_no_intersection_becomes_a_candidate() -> None:
    """§10. The intersection is recorded on the relation and nowhere else."""
    analysis = features(
        decide(gap_source("3990", "4010"), price="4200"),
        decide(gap_source("4000", "4020"), price="4200"),
        price="4200",
    )
    relation = analysis.pair_relations[0]

    assert relation.intersection_lower == Decimal("4000")
    assert relation.intersection_upper == Decimal("4010")
    assert len(analysis.candidates) == 2
    assert analysis.feature("4000") is None


# --------------------------------------------------------------------------
# §11: cross-source and multi-timeframe counts
# --------------------------------------------------------------------------


def test_the_three_counts_are_counts() -> None:
    """§11. Support, timeframes, source kinds - three lengths, no fourth number."""
    analysis = features(
        decide(gap_source("3990", "4010"), tag="-h1"),
        decide(gap_source("3990", "4010"), timeframe=Timeframe.M15, tag="-m15"),
        decide(block_source("3990", "4010"), tag="-ob"),
    )
    feature = only(analysis)

    assert feature.support_count == 3
    assert feature.timeframe_count == 2
    assert feature.source_kind_count == 2
    assert set(feature.source_kinds) == {
        CandidateSourceKind.ORDER_BLOCK,
        CandidateSourceKind.FAIR_VALUE_GAP,
    }
    assert set(feature.timeframes) == {Timeframe.H1, Timeframe.M15}


def test_the_counts_match_the_consolidated_candidate() -> None:
    """§11. Copied, not recounted from a different collection."""
    analysis = features(
        decide(gap_source("3990", "4010"), tag="-a"),
        decide(gap_source("3990", "4010"), timeframe=Timeframe.M15, tag="-b"),
    )
    feature = only(analysis)
    candidate = analysis.candidate(feature.candidate_id)

    assert candidate is not None
    assert feature.support_count == candidate.support_count
    assert feature.timeframe_count == candidate.timeframe_count
    assert feature.source_kind_count == len(candidate.source_kinds)


def test_no_count_is_combined_into_a_confluence_number() -> None:
    """§11. Three fields, and no product, sum or ratio of them anywhere."""
    feature = only(
        features(
            decide(gap_source("3990", "4010"), tag="-a"),
            decide(gap_source("3990", "4010"), timeframe=Timeframe.M15, tag="-b"),
        )
    )

    numeric = {
        name
        for name, value in vars(feature).items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    assert numeric == {
        "support_count",
        "timeframe_count",
        "source_kind_count",
        "oldest_source_age_seconds",
        "newest_source_age_seconds",
    }


# --------------------------------------------------------------------------
# §12: reference candidates
# --------------------------------------------------------------------------


def test_a_reference_keeps_its_level_and_gains_no_zone() -> None:
    """§12. No fake zone, in either direction."""
    feature = only(features(decide(pool_source("4050", "4050", LiquiditySide.BUY_SIDE))))

    assert feature.role is CandidateRole.UPPER_REFERENCE
    assert feature.reference_level == Decimal("4050")
    assert feature.lower is None
    assert feature.upper is None
    assert feature.midpoint is None
    assert feature.width is None
    assert feature.entry_side is None


def test_a_reference_still_gets_support_age_and_distance_facts() -> None:
    """§12. Refusing it a zone is not refusing it features."""
    feature = only(
        features(decide(aged(pool_source("4050", "4050", LiquiditySide.BUY_SIDE), seconds=1800)))
    )

    assert feature.support_count == 1
    assert feature.oldest_source_age_seconds == 1800
    assert feature.market_relation is MarketRelation.ABOVE
    assert feature.distance_to_reference == Decimal("50")
    assert len(feature.support_facts) == 1
    assert feature.support_facts[0].liquidity_side is LiquiditySide.BUY_SIDE


def test_a_lower_reference_is_a_lower_reference() -> None:
    """§12, §34."""
    feature = only(
        features(
            decide(pool_source("3950", "3950", LiquiditySide.SELL_SIDE)),
        )
    )

    assert feature.role is CandidateRole.LOWER_REFERENCE
    assert feature.reference_level == Decimal("3950")
    assert feature.market_relation is MarketRelation.BELOW
    assert feature.entry_side is None


# --------------------------------------------------------------------------
# §13-§14: the analysis and its ordering
# --------------------------------------------------------------------------


def test_the_analysis_carries_its_provenance_and_the_consolidation() -> None:
    """§13."""
    reading = eligibility(decide(gap_source("3990", "4010")))
    consolidation = consolidate_candidates(reading)
    analysis = build_candidate_features(consolidation)

    assert analysis.observed_at == reading.observed_at
    assert analysis.symbol == reading.symbol
    assert analysis.provider == reading.provider
    assert analysis.provider_symbol == reading.provider_symbol
    assert analysis.composite_config is reading.composite_config
    assert analysis.eligibility_config is reading.eligibility_config
    assert analysis.reference_price is reading.reference_price
    assert analysis.consolidation is consolidation


def test_the_analysis_holds_no_selection() -> None:
    """§13, §46."""
    fields = set(CandidateFeatureAnalysis.__dataclass_fields__)

    assert fields == {
        "method_version",
        "observed_at",
        "symbol",
        "provider",
        "provider_symbol",
        "composite_config",
        "eligibility_config",
        "reference_price",
        "candidates",
        "pair_relations",
        "timeframe_contexts",
        "consolidation",
    }


def test_features_keep_consolidation_order() -> None:
    """§14. Not re-sorted by distance, support, age or width."""
    reading = eligibility(
        decide(gap_source("3990", "4010"), price="4200"),
        decide(gap_source("4100", "4120"), price="4200"),
        decide(gap_source("4020", "4040"), price="4200"),
        price="4200",
    )
    consolidation = consolidate_candidates(reading)
    analysis = build_candidate_features(consolidation)

    assert [feature.candidate_id for feature in analysis.candidates] == [
        candidate.consolidated_candidate_id for candidate in consolidation.candidates
    ]


def test_pair_relations_are_ordered_by_their_pair_ids() -> None:
    """§14."""
    analysis = features(
        decide(gap_source("3990", "4010"), price="4300"),
        decide(gap_source("4020", "4040"), price="4300"),
        decide(gap_source("4100", "4120"), price="4300"),
        price="4300",
    )
    keys = [
        (relation.first_candidate_id, relation.second_candidate_id)
        for relation in analysis.pair_relations
    ]

    assert keys == sorted(keys)


def test_lookups_find_what_they_are_asked_for() -> None:
    analysis = features(
        decide(gap_source("3990", "4010")),
        decide(pool_source("4050", "4050", LiquiditySide.BUY_SIDE)),
    )
    zone = analysis.entry_zones(EntrySide.BAI)[0]

    assert analysis.feature(zone.candidate_id) is zone
    assert analysis.feature("nope") is None
    assert analysis.of_role(CandidateRole.UPPER_REFERENCE)[0].reference_level == Decimal("4050")
    assert analysis.entry_zones(EntrySide.SEO) == ()
    assert analysis.timeframe_context(Timeframe.M5) is None
    assert analysis.candidate(zone.candidate_id) is not None


def test_an_empty_candidate_set_produces_an_empty_analysis() -> None:
    """§13. Nothing to arrange is a valid answer, not an error."""
    from goldpipeline.services.ict_candidate_eligibility import CandidateEligibilityConfig
    from tests.test_ict_candidate_eligibility import PERMISSIVE

    empty = replace(
        eligibility(decide(gap_source("3990", "4010"))),
        timeframes=(
            CandidateEligibilityTimeframe(
                method_version="1.0.0",
                timeframe=Timeframe.H1,
                symbol="XAUUSD",
                observed_at=MOMENT,
                decisions=(),
                structure_bias=StructureBias.NEUTRAL,
                active_dealing_range=None,
            ),
        ),
    )
    analysis = build_candidate_features(consolidate_candidates(empty))

    assert isinstance(PERMISSIVE, CandidateEligibilityConfig)
    assert analysis.candidates == ()
    assert analysis.pair_relations == ()
    assert len(analysis.timeframe_contexts) == 1
    assert analysis.timeframe_contexts[0].decision_count == 0


def test_a_timeframe_outside_the_known_order_is_refused() -> None:
    """§14. Silently dropping it would lose a whole timeframe's context."""
    reading = eligibility(decide(gap_source("3990", "4010")))
    strange = replace(
        reading,
        timeframes=(replace(reading.timeframes[0], timeframe=Timeframe.D1),),
    )

    with pytest.raises(CandidateFeatureError, match="outside the feature order"):
        build_candidate_features(consolidate_candidates(strange))


def test_nothing_upstream_is_mutated() -> None:
    consolidation = consolidate_candidates(eligibility(decide(gap_source("3990", "4010"))))
    before = repr(consolidation)

    build_candidate_features(consolidation)

    assert repr(consolidation) == before


def test_repeated_construction_is_identical() -> None:
    consolidation = consolidate_candidates(eligibility(decide(gap_source("3990", "4010"))))

    assert build_candidate_features(consolidation) == build_candidate_features(consolidation)


def test_the_observation_instant_is_the_readings_own() -> None:
    """§45. No clock of this layer's own."""
    consolidation = consolidate_candidates(eligibility(decide(gap_source("3990", "4010"))))
    analysis = build_candidate_features(consolidation)

    assert analysis.observed_at == consolidation.observed_at == MOMENT
    assert analysis.observed_at != datetime.now(UTC)
