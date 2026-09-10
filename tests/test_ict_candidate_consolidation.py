"""What counts as the same candidate, and everything that does not.

Round 6.6e.2c1 §4-§9, §17-§31. The grouping matrices run over eligibility
analyses assembled here from real decisions, because that is exactly the shape
this layer consumes - it never sees a candle, so it can be tested honestly
without one.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.services.ict_candidate_consolidation import (
    CANDIDATE_CONSOLIDATION_METHOD_VERSION,
    CandidateConsolidationError,
    ConsolidatedCandidate,
    canonical_price,
    consolidate_candidates,
)
from goldpipeline.services.ict_candidate_eligibility import (
    CandidateEligibilityAnalysis,
    CandidateEligibilityConfig,
    CandidateEligibilityDecision,
    CandidateEligibilityTimeframe,
    CandidateRole,
    EntrySide,
    MarketRelation,
    ReferencePrice,
    ReferencePriceWitness,
)
from goldpipeline.services.ict_candidate_source import (
    CandidateSource,
    CandidateSourceKind,
)
from goldpipeline.services.ict_fvg import FvgStatus
from goldpipeline.services.ict_liquidity import LiquiditySide
from goldpipeline.services.ict_order_block_lifecycle import OrderBlockStatus
from goldpipeline.services.ict_primitives import GapDirection
from goldpipeline.services.ict_structure import BreakDirection, StructureBias
from tests.test_ict_candidate_eligibility import (
    PERMISSIVE,
    block_source,
    gap_source,
    pool_source,
)
from tests.test_ict_candidate_source_fixture import block_of
from tests.test_ict_composite_fixture import config as composite_config

MOMENT = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
MINUTE = timedelta(minutes=1)
PRICE = Decimal("4000")


# --------------------------------------------------------------------------
# assembling honest eligibility readings, which is all this layer consumes
# --------------------------------------------------------------------------


def reference(price: str = "4000") -> ReferencePrice:
    return ReferencePrice(
        symbol="XAUUSD",
        observed_at=MOMENT,
        bar_close_time=MOMENT,
        price=Decimal(price),
        witnesses=(
            ReferencePriceWitness(
                timeframe=Timeframe.M1,
                bar_open_time=MOMENT - MINUTE,
                bar_close_time=MOMENT,
                close=Decimal(price),
            ),
        ),
    )


def decide(
    source: CandidateSource,
    *,
    price: str = "4000",
    config: CandidateEligibilityConfig = PERMISSIVE,
    timeframe: Timeframe = Timeframe.H1,
    tag: str = "",
) -> CandidateEligibilityDecision:
    """One eligible decision over *source*, scoped to a timeframe.

    The source identities are re-scoped because the factories in
    ``test_ict_candidate_eligibility`` key their ids on geometry alone. A real
    projection id already carries the timeframe, and a real ``fvg_id`` carries
    the candle times, so two same-shaped sources are never actually the same
    object - *tag* is how a test says "these two differ for a reason the
    factory does not model".
    """
    from goldpipeline.services.ict_candidate_eligibility import _decide

    suffix = f"{timeframe.value}{tag}"
    scoped = replace(
        source,
        timeframe=timeframe,
        candidate_source_id=f"{source.candidate_source_id}@{suffix}",
        source_id=f"{source.source_id}@{suffix}",
    )
    made = _decide(scoped, price=Decimal(price), config=config)
    assert made.eligible, f"the matrices need eligible decisions: {made.reasons}"
    return made


def eligibility(
    *decisions: CandidateEligibilityDecision,
    price: str = "4000",
    extra: tuple[CandidateEligibilityDecision, ...] = (),
) -> CandidateEligibilityAnalysis:
    """One eligibility reading, its decisions grouped into their timeframes."""
    everything = [*decisions, *extra]
    by_timeframe: dict[Timeframe, list[CandidateEligibilityDecision]] = {}
    for decision in everything:
        by_timeframe.setdefault(decision.timeframe, []).append(decision)

    from goldpipeline.schemas.ict import ICT_TIMEFRAMES

    return CandidateEligibilityAnalysis(
        method_version="1.0.0",
        observed_at=MOMENT,
        symbol="XAUUSD",
        provider="fixture",
        provider_symbol=None,
        composite_config=composite_config(),
        eligibility_config=PERMISSIVE,
        reference_price=reference(price),
        timeframes=tuple(
            CandidateEligibilityTimeframe(
                method_version="1.0.0",
                timeframe=tf,
                symbol="XAUUSD",
                observed_at=MOMENT,
                decisions=tuple(by_timeframe[tf]),
                structure_bias=StructureBias.NEUTRAL,
                active_dealing_range=None,
            )
            for tf in ICT_TIMEFRAMES
            if tf in by_timeframe
        ),
    )


def groups(
    *decisions: CandidateEligibilityDecision, price: str = "4000"
) -> tuple[ConsolidatedCandidate, ...]:
    """Consolidate a reading. *price* must match the one the decisions were taken at."""
    for decision in decisions:
        assert decision.reference_price == Decimal(price), (
            "the reading's reference price must be the one the decisions used"
        )
    return consolidate_candidates(eligibility(*decisions, price=price)).candidates


# --------------------------------------------------------------------------
# §24: Decimal canonicalisation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("4000", "4000"),
        ("4000.0", "4000"),
        ("4000.00", "4000"),
        ("4000.0001", "4000.0001"),
        ("4050.30", "4050.3"),
        ("1E+3", "1000"),
        ("0.1", "0.1"),
        ("123.4500", "123.45"),
    ],
)
def test_one_price_written_three_ways_canonicalises_once(written: str, expected: str) -> None:
    """§9, §24."""
    assert canonical_price(Decimal(written)) == expected


def test_the_new_helper_agrees_with_the_gap_engines_across_a_wide_range() -> None:
    """§24. Identical behaviour, deliberately not an extraction.

    Editing the module that mints every historical fair-value-gap identity is
    not worth the tidiness, so the two are kept separate and pinned equal here.
    Unifying them is a backlog item conditioned on byte-identical ids.
    """
    from goldpipeline.services.ict_fvg import _canonical

    values = [
        "0",
        "0.00",
        "1",
        "1.0",
        "4000",
        "4000.0",
        "4000.00",
        "4000.0001",
        "4050.30",
        "3999.999999",
        "1E+3",
        "1e-7",
        "123.4500",
        "99999999.99",
    ]
    for written in values:
        price = Decimal(written)
        assert canonical_price(price) == _canonical(price), written


def test_equal_prices_at_different_precision_reach_one_identity() -> None:
    """§9, §24. The identity requirement the canonicaliser exists for."""
    first = groups(decide(gap_source("4000", "4010")))
    second = groups(decide(gap_source("4000.00", "4010.0")))

    assert len(first) == len(second) == 1
    assert first[0].consolidated_candidate_id == second[0].consolidated_candidate_id


# --------------------------------------------------------------------------
# §4-§5, §18: exact entry-zone grouping
# --------------------------------------------------------------------------


def test_two_identical_zones_become_one_candidate() -> None:
    """§4."""
    found = groups(
        decide(gap_source("3990", "4010"), tag="-open"),
        decide(gap_source("3990", "4010", GapDirection.BULLISH, FvgStatus.TOUCHED), tag="-touched"),
    )

    assert len(found) == 1
    assert found[0].support_count == 2
    assert found[0].lower == Decimal("3990")
    assert found[0].upper == Decimal("4010")


def test_an_order_block_and_a_gap_on_one_zone_consolidate() -> None:
    """§5, §18, §49. Cross-source, and neither becomes the winner."""
    found = groups(
        decide(block_source("3990", "4010", BreakDirection.BULLISH)),
        decide(gap_source("3990", "4010", GapDirection.BULLISH)),
    )

    assert len(found) == 1
    candidate = found[0]
    assert candidate.support_count == 2
    assert candidate.source_kinds == (
        CandidateSourceKind.ORDER_BLOCK,
        CandidateSourceKind.FAIR_VALUE_GAP,
    )
    assert candidate.entry_side is EntrySide.BAI


def test_the_consolidated_zone_is_exactly_the_members_zone() -> None:
    """§11. Nothing is unioned, averaged or widened."""
    members = [decide(gap_source("3990", "4010")), decide(block_source("3990", "4010"))]
    candidate = groups(*members)[0]

    for member in members:
        assert (candidate.lower, candidate.upper) == (member.lower, member.upper)
        assert (candidate.midpoint, candidate.width) == (member.midpoint, member.width)
    assert candidate.reference_level is None


# --------------------------------------------------------------------------
# §19-§22: everything that must not consolidate
# --------------------------------------------------------------------------


def test_the_same_zone_on_opposite_sides_stays_two_candidates() -> None:
    """§19. Geometry equality does not erase direction."""
    found = groups(
        decide(gap_source("3990", "4010", GapDirection.BULLISH)),
        decide(gap_source("3990", "4010", GapDirection.BEARISH)),
    )

    assert len(found) == 2
    assert {candidate.entry_side for candidate in found} == {
        EntrySide.BAI,
        EntrySide.SEO,
    }
    assert found[0].consolidated_candidate_id != found[1].consolidated_candidate_id


@pytest.mark.parametrize(
    ("lower", "upper"), [("3990", "4010.01"), ("3990.01", "4010"), ("3989.99", "4010")]
)
def test_a_hundredth_of_a_difference_is_a_different_candidate(lower: str, upper: str) -> None:
    """§20. No tolerance anywhere."""
    found = groups(decide(gap_source("3990", "4010")), decide(gap_source(lower, upper)))

    assert len(found) == 2


def test_overlapping_zones_stay_separate() -> None:
    """§21, and one of the most important guards in this round."""
    found = groups(decide(gap_source("3990", "4010")), decide(gap_source("4000", "4020")))

    assert len(found) == 2
    assert {(c.lower, c.upper) for c in found} == {
        (Decimal("3990"), Decimal("4010")),
        (Decimal("4000"), Decimal("4020")),
    }


def test_a_chain_of_overlaps_never_collapses() -> None:
    """§22. A, B and C overlap pairwise and remain three candidates.

    Transitive merging would produce 3990-4030 - a band none of them drew, and
    wider than all three.
    """
    found = groups(
        decide(gap_source("3990", "4010"), price="4100"),
        decide(gap_source("4000", "4020"), price="4100"),
        decide(gap_source("4010", "4030"), price="4100"),
        price="4100",
    )

    assert len(found) == 3
    assert all(candidate.support_count == 1 for candidate in found)
    assert not any(
        candidate.lower == Decimal("3990") and candidate.upper == Decimal("4030")
        for candidate in found
    )


def test_zones_touching_at_one_boundary_stay_separate() -> None:
    """§22. Sharing an edge is not being the same zone."""
    found = groups(
        decide(gap_source("3990", "4010"), price="4100"),
        decide(gap_source("4010", "4030"), price="4100"),
        price="4100",
    )

    assert len(found) == 2


# --------------------------------------------------------------------------
# §6-§8, §50: reference-level grouping
# --------------------------------------------------------------------------


def test_two_pools_at_one_level_become_one_reference_candidate() -> None:
    """§6, §50. Different timeframes, different pools, one level."""
    found = groups(
        decide(pool_source("4050", "4050", LiquiditySide.BUY_SIDE), timeframe=Timeframe.H1),
        decide(pool_source("4040", "4050", LiquiditySide.BUY_SIDE), timeframe=Timeframe.M15),
    )

    assert len(found) == 1
    candidate = found[0]
    assert candidate.role is CandidateRole.UPPER_REFERENCE
    assert candidate.reference_level == Decimal("4050")
    assert candidate.support_count == 2
    assert candidate.timeframes == (Timeframe.H1, Timeframe.M15)


def test_an_upper_and_a_lower_reference_at_one_price_stay_separate() -> None:
    """§6, §50. Role is part of identity, so one price is two candidates.

    They cannot be eligible simultaneously: an upper reference must sit above
    the price and a lower one below it, and 4050 cannot do both at once. So the
    two readings are taken at the prices that make each one eligible, and what
    is compared is their identities.
    """
    upper = groups(
        decide(pool_source("4050", "4050", LiquiditySide.BUY_SIDE), price="4000"), price="4000"
    )[0]
    lower = groups(
        decide(pool_source("4050", "4050", LiquiditySide.SELL_SIDE), price="4100"), price="4100"
    )[0]

    assert upper.role is CandidateRole.UPPER_REFERENCE
    assert lower.role is CandidateRole.LOWER_REFERENCE
    assert upper.reference_level == lower.reference_level == Decimal("4050")
    assert upper.consolidated_candidate_id != lower.consolidated_candidate_id


def test_an_entry_boundary_equal_to_a_reference_level_does_not_consolidate() -> None:
    """§7. Different kinds of thing that happen to share a number."""
    found = groups(
        decide(gap_source("4030", "4050", GapDirection.BEARISH)),
        decide(pool_source("4050", "4050", LiquiditySide.BUY_SIDE)),
    )

    assert len(found) == 2
    assert {candidate.role for candidate in found} == {
        CandidateRole.ENTRY_ZONE,
        CandidateRole.UPPER_REFERENCE,
    }


@pytest.mark.parametrize("other", ["4050.01", "4049.99"])
def test_reference_levels_a_hundredth_apart_stay_separate(other: str) -> None:
    """§20."""
    found = groups(
        decide(pool_source("4050", "4050", LiquiditySide.BUY_SIDE)),
        decide(pool_source(other, other, LiquiditySide.BUY_SIDE)),
    )

    assert len(found) == 2


def test_a_zero_width_pool_consolidates_at_its_exact_price() -> None:
    """§8. No fake zone, no widening, no division by source width."""
    candidate = groups(decide(pool_source("4060", "4060", LiquiditySide.BUY_SIDE)))[0]

    assert candidate.reference_level == Decimal("4060")
    assert (candidate.lower, candidate.upper, candidate.midpoint, candidate.width) == (
        None,
        None,
        None,
        None,
    )
    assert candidate.supporting_decisions[0].source.width == 0


def test_a_reference_candidate_carries_no_zone() -> None:
    """§12. Its band stays reachable through the supporting decision."""
    candidate = groups(decide(pool_source("4040", "4060", LiquiditySide.BUY_SIDE)))[0]

    assert candidate.reference_level == Decimal("4060")
    assert candidate.lower is None
    assert candidate.supporting_decisions[0].source.lower == Decimal("4040")


# --------------------------------------------------------------------------
# §38-§40: multi-timeframe exact consolidation
# --------------------------------------------------------------------------


def test_one_zone_on_two_timeframes_becomes_one_candidate() -> None:
    """§38. The same price is the same candidate, whatever timeframe saw it."""
    candidate = groups(
        decide(gap_source("3990", "4010"), timeframe=Timeframe.H1),
        decide(gap_source("3990", "4010"), timeframe=Timeframe.M15),
    )[0]

    assert candidate.support_count == 2
    assert candidate.timeframes == (Timeframe.H1, Timeframe.M15)
    assert candidate.timeframe_count == 2


def test_timeframes_are_recorded_in_branch_order_without_weighting() -> None:
    """§39. H4 is not worth more than M1; it is simply listed first."""
    candidate = groups(
        decide(gap_source("3990", "4010"), timeframe=Timeframe.M1),
        decide(gap_source("3990", "4010"), timeframe=Timeframe.H4),
    )[0]

    assert candidate.timeframes == (Timeframe.H4, Timeframe.M1)
    assert candidate.timeframe_count == 2


def test_one_timeframe_supporting_twice_counts_once_in_timeframes() -> None:
    """§16. ``support_count`` and ``timeframe_count`` measure different things."""
    candidate = groups(
        decide(gap_source("3990", "4010"), timeframe=Timeframe.H1),
        decide(block_source("3990", "4010"), timeframe=Timeframe.H1),
    )[0]

    assert candidate.support_count == 2
    assert candidate.timeframe_count == 1


# --------------------------------------------------------------------------
# §23, §25-§26: group identity
# --------------------------------------------------------------------------


def test_a_new_supporter_does_not_rename_an_existing_candidate() -> None:
    """§23. Identity is what a candidate is, not who agrees with it."""
    alone = groups(decide(gap_source("3990", "4010")))[0]
    joined = groups(decide(gap_source("3990", "4010")), decide(block_source("3990", "4010")))[0]

    assert alone.consolidated_candidate_id == joined.consolidated_candidate_id
    assert alone.support_count == 1
    assert joined.support_count == 2


def test_the_identity_survives_a_reference_price_move() -> None:
    """§25. A zone below price and the same zone overlapping it are one candidate."""
    below = groups(decide(gap_source("3990", "4010"), price="4100"), price="4100")[0]
    overlapping = groups(decide(gap_source("3990", "4010"), price="4000"), price="4000")[0]

    assert below.consolidated_candidate_id == overlapping.consolidated_candidate_id
    assert below.market_relation is MarketRelation.BELOW
    assert overlapping.market_relation is MarketRelation.OVERLAPS


def test_the_identity_survives_a_policy_change() -> None:
    """§26. A narrower policy loses a supporter and keeps the candidate."""
    strict = CandidateEligibilityConfig(
        allowed_order_block_statuses=frozenset({OrderBlockStatus.ACTIVE}),
        allowed_fvg_statuses=frozenset({FvgStatus.OPEN}),
    )
    wide = groups(
        decide(gap_source("3990", "4010", GapDirection.BULLISH, FvgStatus.OPEN), tag="-open"),
        decide(gap_source("3990", "4010", GapDirection.BULLISH, FvgStatus.TOUCHED), tag="-touched"),
    )[0]
    narrow = groups(
        decide(gap_source("3990", "4010", GapDirection.BULLISH, FvgStatus.OPEN), config=strict)
    )[0]

    assert wide.consolidated_candidate_id == narrow.consolidated_candidate_id
    assert (wide.support_count, narrow.support_count) == (2, 1)


def test_no_price_verdict_or_supporter_enters_the_identity() -> None:
    """§23, §25. Read from the preimage rather than inferred from behaviour."""
    import inspect

    from goldpipeline.services.ict_candidate_consolidation import _consolidated_id

    source = inspect.getsource(_consolidated_id)
    joined = source.split('"|".join(')[1]

    for forbidden in (
        "support",
        "decision",
        "reference_price",
        "relation",
        "distance",
        "status",
        "timeframe",
        "kind",
    ):
        assert forbidden not in joined, forbidden


def test_the_two_roles_of_one_price_do_not_collide() -> None:
    """§6, §23."""
    upper = groups(
        decide(pool_source("4050", "4050", LiquiditySide.BUY_SIDE), price="4000"), price="4000"
    )[0]
    lower = groups(
        decide(pool_source("4050", "4050", LiquiditySide.SELL_SIDE), price="4100"), price="4100"
    )[0]

    assert upper.consolidated_candidate_id != lower.consolidated_candidate_id


# --------------------------------------------------------------------------
# §29-§31: the counting invariants
# --------------------------------------------------------------------------


def test_every_eligible_decision_lands_in_exactly_one_group() -> None:
    """§29."""
    decisions = [
        decide(gap_source("3990", "4010")),
        decide(gap_source("3990", "4010"), timeframe=Timeframe.M15),
        decide(gap_source("4000", "4020")),
        decide(pool_source("4050", "4050", LiquiditySide.BUY_SIDE)),
    ]
    found = consolidate_candidates(eligibility(*decisions)).candidates

    placed = [
        member.candidate_id for candidate in found for member in candidate.supporting_decisions
    ]
    assert sorted(placed) == sorted(decision.candidate_id for decision in decisions)
    assert len(placed) == len(set(placed)), "no decision supports two groups"
    assert sum(candidate.support_count for candidate in found) == len(decisions)


def test_an_ineligible_decision_supports_nothing() -> None:
    """§3, §30. It is not deleted either - it stays in the eligibility reading."""
    from goldpipeline.services.ict_candidate_eligibility import _decide

    rejected = _decide(
        gap_source("3900", "3950", GapDirection.BEARISH, FvgStatus.FILLED),
        price=Decimal("4000"),
        config=PERMISSIVE,
    )
    assert rejected.eligible is False

    reading = eligibility(decide(gap_source("3990", "4010")), extra=(rejected,))
    result = consolidate_candidates(reading)

    assert sum(candidate.support_count for candidate in result.candidates) == 1
    supported = {
        member.candidate_id
        for candidate in result.candidates
        for member in candidate.supporting_decisions
    }
    assert rejected.candidate_id not in supported
    assert any(
        decision.candidate_id == rejected.candidate_id
        for entry in result.eligibility.timeframes
        for decision in entry.decisions
    ), "still reachable, with its reasons"


def test_a_reading_with_no_eligible_decision_produces_no_candidate() -> None:
    """§28."""
    from goldpipeline.services.ict_candidate_eligibility import _decide

    rejected = _decide(
        gap_source("3900", "3950", GapDirection.BEARISH, FvgStatus.FILLED),
        price=Decimal("4000"),
        config=PERMISSIVE,
    )
    result = consolidate_candidates(eligibility(extra=(rejected,)))

    assert result.candidates == ()
    assert result.eligibility.timeframes[0].decisions == (rejected,)


def test_supporting_identities_are_unique_within_a_group() -> None:
    """§31."""
    candidate = groups(
        decide(gap_source("3990", "4010")),
        decide(block_source("3990", "4010")),
        decide(gap_source("3990", "4010"), timeframe=Timeframe.M15),
    )[0]

    assert len(set(candidate.supporting_candidate_ids)) == 3
    assert len(set(candidate.supporting_candidate_source_ids)) == 3
    assert len(set(candidate.supporting_source_ids)) == 3


# --------------------------------------------------------------------------
# §13-§14: consistency, checked rather than assumed
# --------------------------------------------------------------------------


def test_members_disagreeing_about_relation_fail_closed() -> None:
    """§13. Two identical zones measured against different prices is a contradiction."""
    first = decide(gap_source("3990", "4010"))
    drifted = replace(first, market_relation=MarketRelation.ABOVE, candidate_id="other")

    with pytest.raises(CandidateConsolidationError, match="market relation"):
        consolidate_candidates(eligibility(first, drifted))


def test_members_disagreeing_about_distance_fail_closed() -> None:
    """§13."""
    first = decide(gap_source("3990", "4010"))
    drifted = replace(first, distance_to_reference=Decimal("7"), candidate_id="other")

    with pytest.raises(CandidateConsolidationError, match="distance"):
        consolidate_candidates(eligibility(first, drifted))


def test_members_disagreeing_about_midpoint_fail_closed() -> None:
    """§13. Same edges, different midpoint would mean two readings of one zone."""
    first = decide(gap_source("3990", "4010"))
    drifted = replace(first, midpoint=Decimal("4001"), candidate_id="other")

    with pytest.raises(CandidateConsolidationError, match="zone geometry"):
        consolidate_candidates(eligibility(first, drifted))


def test_an_entry_zone_with_no_side_fails_closed() -> None:
    """§14."""
    broken = replace(decide(gap_source("3990", "4010")), entry_side=None)

    with pytest.raises(CandidateConsolidationError, match="missing a side or an edge"):
        consolidate_candidates(eligibility(broken))


def test_a_reference_carrying_an_entry_side_fails_closed() -> None:
    """§14."""
    broken = replace(
        decide(pool_source("4050", "4050", LiquiditySide.BUY_SIDE)), entry_side=EntrySide.BAI
    )

    with pytest.raises(CandidateConsolidationError, match="carries entry side"):
        consolidate_candidates(eligibility(broken))


def test_a_reference_with_no_level_fails_closed() -> None:
    broken = replace(
        decide(pool_source("4050", "4050", LiquiditySide.BUY_SIDE)), reference_level=None
    )

    with pytest.raises(CandidateConsolidationError, match="no reference level"):
        consolidate_candidates(eligibility(broken))


# --------------------------------------------------------------------------
# §10, §15-§16, §54: the model
# --------------------------------------------------------------------------


def test_a_candidate_can_be_walked_back_to_its_original_market_object() -> None:
    """§15. Candidate → decision → source → block or gap, by identity throughout."""
    candidate = groups(decide(block_source("3990", "4010")))[0]
    decision = candidate.supporting_decisions[0]

    assert decision.source.source_id == decision.source_id
    assert candidate.supporting_source_ids == (decision.source_id,)
    # The original object is reachable without any geometry lookup. Its own id
    # is the unscoped one, because decide re-scopes only the projection-level
    # identities that the geometry-keyed factory would otherwise collide on.
    block = block_of(decision.source).order_block
    assert block.lower == decision.lower
    assert block.upper == decision.upper


def test_the_analysis_carries_its_provenance_and_its_inputs() -> None:
    """§34."""
    reading = eligibility(decide(gap_source("3990", "4010")))
    result = consolidate_candidates(reading)

    assert result.method_version == CANDIDATE_CONSOLIDATION_METHOD_VERSION
    assert result.observed_at == reading.observed_at
    assert (result.symbol, result.provider, result.provider_symbol) == (
        reading.symbol,
        reading.provider,
        reading.provider_symbol,
    )
    assert result.composite_config is reading.composite_config
    assert result.eligibility_config is reading.eligibility_config
    assert result.reference_price is reading.reference_price
    assert result.eligibility is reading


def test_the_lookups_are_by_identity() -> None:
    result = consolidate_candidates(
        eligibility(
            decide(gap_source("3990", "4010")),
            decide(pool_source("4050", "4050", LiquiditySide.BUY_SIDE)),
        )
    )

    for candidate in result.candidates:
        assert result.candidate(candidate.consolidated_candidate_id) is candidate
    assert result.candidate("nope") is None
    assert len(result.of_role(CandidateRole.ENTRY_ZONE)) == 1
    assert len(result.of_role(CandidateRole.UPPER_REFERENCE)) == 1
    assert result.of_role(CandidateRole.LOWER_REFERENCE) == ()


def test_every_model_is_frozen() -> None:
    """§54."""
    result = consolidate_candidates(eligibility(decide(gap_source("3990", "4010"))))

    with pytest.raises(FrozenInstanceError):
        result.observed_at = MOMENT  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.candidates[0].lower = Decimal("1")  # type: ignore[misc]


def test_the_eligibility_reading_is_not_mutated() -> None:
    """§54."""
    reading = eligibility(decide(gap_source("3990", "4010")), decide(block_source("3990", "4010")))
    before = repr(reading)

    consolidate_candidates(reading)

    assert repr(reading) == before
