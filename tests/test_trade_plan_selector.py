"""Which ranked candidates reach the page, one constructed case at a time.

Round 6.6g §2, §4-§15, §26-§30, §32. Everything here is built rather than found,
so each case says exactly one thing; the realistic reading is pinned separately.

Rankings are real: they are written as JSON, run through the analyst's own
``parse_ranking`` validator, and only then handed to the selector. A test that
constructed a ``TradeAnalystRanking`` directly would be able to express rankings
no model could ever produce.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.services.ict_candidate_consolidation import consolidate_candidates
from goldpipeline.services.ict_candidate_eligibility import (
    CandidateEligibilityDecision,
    CandidateEligibilityTimeframe,
    CandidateRole,
    EntrySide,
)
from goldpipeline.services.ict_candidate_features import (
    CandidateFeatureAnalysis,
    build_candidate_features,
)
from goldpipeline.services.ict_liquidity import LiquiditySide
from goldpipeline.services.ict_primitives import GapDirection
from goldpipeline.services.ict_structure import StructureBias
from goldpipeline.services.trade_analyst import (
    BAI_KEY,
    RANKING_KEYS,
    UPPER_KEY,
    TradeAnalystRanking,
    expected_buckets,
    parse_ranking,
)
from goldpipeline.services.trade_plan_selector import (
    MAX_ENTRY_ZONES_PER_SIDE,
    TRADE_PLAN_SELECTION_METHOD_VERSION,
    SelectionOutcome,
    TradePlanSelection,
    TradePlanSelectionError,
    ZoneLabel,
    contains,
    select_trade_plan,
)
from tests.test_ict_candidate_consolidation import MOMENT, decide, eligibility
from tests.test_ict_candidate_eligibility import gap_source, pool_source


def features_of(
    *decisions: CandidateEligibilityDecision, price: str = "4000"
) -> CandidateFeatureAnalysis:
    return build_candidate_features(consolidate_candidates(eligibility(*decisions, price=price)))


def bai_zones(*bounds: tuple[str, str], price: str = "4200") -> CandidateFeatureAnalysis:
    """A reading whose only candidates are BAI entry zones, in the given shapes."""
    return features_of(
        *(
            decide(gap_source(lower, upper, GapDirection.BULLISH), price=price)
            for lower, upper in bounds
        ),
        price=price,
    )


def seo_zones(*bounds: tuple[str, str], price: str = "3800") -> CandidateFeatureAnalysis:
    return features_of(
        *(
            decide(gap_source(lower, upper, GapDirection.BEARISH), price=price)
            for lower, upper in bounds
        ),
        price=price,
    )


def ranked(features: CandidateFeatureAnalysis, **order: list[str]) -> TradeAnalystRanking:
    """A validated ranking. Buckets not named keep their natural order."""
    natural = expected_buckets(features)
    body = {key: list(order.get(key, natural[key])) for key in RANKING_KEYS}
    return parse_ranking(json.dumps(body), features=features)


def natural(features: CandidateFeatureAnalysis) -> TradeAnalystRanking:
    return ranked(features)


def by_bounds(features: CandidateFeatureAnalysis, lower: str, upper: str) -> str:
    """The candidate id of the zone with those exact bounds."""
    found = [
        feature.candidate_id
        for feature in features.candidates
        if feature.lower == Decimal(lower) and feature.upper == Decimal(upper)
    ]
    assert len(found) == 1, (lower, upper, found)
    return found[0]


def selected_bounds(selection: TradePlanSelection, side: EntrySide) -> list[tuple[str, str]]:
    return [(str(zone.lower), str(zone.upper)) for zone in selection.entries(side)]


def outcomes(selection: TradePlanSelection) -> dict[str, SelectionOutcome]:
    return {decision.candidate_id: decision.outcome for decision in selection.decisions}


# --------------------------------------------------------------------------
# §2: the ranking must belong to these features
# --------------------------------------------------------------------------


def test_a_matching_ranking_is_accepted() -> None:
    features = bai_zones(("3900", "3910"))

    selection = select_trade_plan(features, natural(features))

    assert selection.method_version == TRADE_PLAN_SELECTION_METHOD_VERSION
    assert selection.observed_at == features.observed_at
    assert selection.symbol == features.symbol
    assert selection.reference_price is features.reference_price
    assert selection.features is features


def test_a_ranking_for_another_symbol_is_refused() -> None:
    """§2. A ranking is a list of opaque ids; provenance is all that guards it."""
    features = bai_zones(("3900", "3910"))
    foreign = replace(natural(features), symbol="XAGUSD")

    with pytest.raises(TradePlanSelectionError, match="ranking is for XAGUSD"):
        select_trade_plan(features, foreign)


def test_a_ranking_from_another_instant_is_refused() -> None:
    """§2."""
    from datetime import timedelta

    features = bai_zones(("3900", "3910"))
    stale = replace(natural(features), observed_at=features.observed_at - timedelta(hours=1))

    with pytest.raises(TradePlanSelectionError, match="ranking observed"):
        select_trade_plan(features, stale)


def test_a_ranking_from_another_feature_version_is_refused() -> None:
    """§2."""
    features = bai_zones(("3900", "3910"))
    older = replace(natural(features), candidate_feature_method_version="0.9.0")

    with pytest.raises(TradePlanSelectionError, match="feature method version"):
        select_trade_plan(features, older)


def test_a_ranking_over_a_different_candidate_set_is_refused() -> None:
    """§2. The sharpest case: same symbol, same instant, different candidates."""
    features = bai_zones(("3900", "3910"), ("3920", "3930"))
    other = bai_zones(("3800", "3810"))
    borrowed = replace(
        natural(other),
        observed_at=features.observed_at,
        symbol=features.symbol,
        candidate_feature_method_version=features.method_version,
    )

    with pytest.raises(TradePlanSelectionError, match="does not describe this candidate set"):
        select_trade_plan(features, borrowed)


def test_a_partial_ranking_is_refused() -> None:
    """§2, §3. The analyst is not allowed to decide omission, so nor is a caller."""
    features = bai_zones(("3900", "3910"), ("3920", "3930"))
    short = replace(
        natural(features),
        bai_entry_candidate_ids=natural(features).bai_entry_candidate_ids[:1],
    )

    with pytest.raises(TradePlanSelectionError, match="does not describe this candidate set"):
        select_trade_plan(features, short)


def test_the_selector_takes_only_features_and_a_ranking() -> None:
    """§2, §33. No clock, no snapshot, no config."""
    import inspect

    parameters = inspect.signature(select_trade_plan).parameters

    assert list(parameters) == ["features", "ranking"]
    for forbidden in ("as_of", "observed_at", "now", "snapshot", "config", "limit"):
        assert forbidden not in parameters


# --------------------------------------------------------------------------
# §5-§6, §26: containment is the only redundancy
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("outer", "inner", "expected"),
    [
        (("3990", "4030"), ("4000", "4020"), True),
        (("3990", "4030"), ("3990", "4030"), True),
        (("3990", "4030"), ("3990", "4020"), True),
        (("3990", "4030"), ("4000", "4030"), True),
        (("3990", "4010"), ("4000", "4020"), False),
        (("3990", "4010"), ("4010", "4030"), False),
        (("3990", "4010"), ("4020", "4030"), False),
        (("4000", "4020"), ("3990", "4030"), False),
    ],
)
def test_containment_is_exact_and_closed(
    outer: tuple[str, str], inner: tuple[str, str], expected: bool
) -> None:
    """§5. No tolerance anywhere - equal edges count, one tick apart does not."""
    assert (
        contains(
            outer_lower=Decimal(outer[0]),
            outer_upper=Decimal(outer[1]),
            inner_lower=Decimal(inner[0]),
            inner_upper=Decimal(inner[1]),
        )
        is expected
    )


def test_26_a_contained_zone_is_suppressed_and_the_ranking_decides_which() -> None:
    """§26. Same two zones, two rankings, two different survivors."""
    features = bai_zones(("3990", "4030"), ("4000", "4020"))
    outer = by_bounds(features, "3990", "4030")
    inner = by_bounds(features, "4000", "4020")

    first = select_trade_plan(features, ranked(features, bai_entry_candidate_ids=[outer, inner]))
    second = select_trade_plan(features, ranked(features, bai_entry_candidate_ids=[inner, outer]))

    assert selected_bounds(first, EntrySide.BAI) == [("3990", "4030")]
    assert outcomes(first)[inner] is SelectionOutcome.SUPPRESSED_CONTAINMENT

    assert selected_bounds(second, EntrySide.BAI) == [("4000", "4020")]
    assert outcomes(second)[outer] is SelectionOutcome.SUPPRESSED_CONTAINMENT


def test_the_suppression_record_names_the_zone_that_did_it() -> None:
    """§8. The audit trail answers "why is that zone not on the page"."""
    features = bai_zones(("3990", "4030"), ("4000", "4020"))
    outer = by_bounds(features, "3990", "4030")
    inner = by_bounds(features, "4000", "4020")

    selection = select_trade_plan(
        features, ranked(features, bai_entry_candidate_ids=[outer, inner])
    )
    record = selection.decision(inner)

    assert record is not None
    assert record.outcome is SelectionOutcome.SUPPRESSED_CONTAINMENT
    assert record.suppressed_by_candidate_id == outer
    assert record.ai_rank == 2
    assert record.bucket == BAI_KEY


def test_26_partial_overlap_keeps_both() -> None:
    """§6. Two different propositions about two different bands."""
    features = bai_zones(("3990", "4010"), ("4000", "4020"))

    selection = select_trade_plan(features, natural(features))

    assert len(selection.bai_entries) == 2
    assert set(selected_bounds(selection, EntrySide.BAI)) == {
        ("3990", "4010"),
        ("4000", "4020"),
    }


def test_26_touching_keeps_both() -> None:
    """§6. Sharing one boundary price is contact, not containment."""
    features = bai_zones(("3990", "4010"), ("4010", "4030"))

    selection = select_trade_plan(features, natural(features))

    assert len(selection.bai_entries) == 2


@pytest.mark.parametrize(
    "second", [("3991", "4011"), ("3989", "4009"), ("4009", "4029"), ("3990.01", "4010.01")]
)
def test_near_misses_are_not_suppressed(second: tuple[str, str]) -> None:
    """§6. No tolerance: one tick of difference in either edge is a different zone.

    Each shape here shifts *both* edges the same way, which is what keeps it out
    of containment. Widening only one edge - 3990-4011 against 3990-4010 - is
    containment by the exact rule, and the case below says so rather than
    pretending a hundredth of a point is a rounding error.
    """
    features = bai_zones(("3990", "4010"), second)

    selection = select_trade_plan(features, natural(features))

    assert len(selection.bai_entries) == 2, selected_bounds(selection, EntrySide.BAI)


@pytest.mark.parametrize("second", [("3990", "4011"), ("3989.99", "4010")])
def test_widening_one_edge_is_containment(second: tuple[str, str]) -> None:
    """§5. The exact rule, including the case a tolerance would have blurred.

    An exactly identical second zone is absent from this list on purpose: two
    equal geometries are one consolidated candidate, so the case cannot be built
    through the honest pipeline. §7 covers what happens if it ever appears.
    """
    features = bai_zones(("3990", "4010"), second)
    first_id = by_bounds(features, "3990", "4010")
    second_id = by_bounds(features, *second)

    selection = select_trade_plan(
        features, ranked(features, bai_entry_candidate_ids=[first_id, second_id])
    )

    assert len(selection.bai_entries) == 1
    assert outcomes(selection)[second_id] is SelectionOutcome.SUPPRESSED_CONTAINMENT


def test_a_chain_of_overlaps_creates_no_merge() -> None:
    """§6, §26. Three partial overlaps stay three zones and no union appears."""
    features = bai_zones(("3990", "4010"), ("4000", "4020"), ("4010", "4030"))

    selection = select_trade_plan(features, natural(features))
    bounds = selected_bounds(selection, EntrySide.BAI)

    assert len(bounds) == 3
    assert ("3990", "4030") not in bounds


def test_a_suppressed_zone_is_never_backfilled() -> None:
    """§4, §8. Reaching three is not a reason to republish a redundant zone.

    The ranking is named explicitly, because feature order is consolidation
    order - keyed on candidate identity - and is not the order these zones were
    written in. Relying on it would make the test say something it does not mean.
    """
    features = bai_zones(("3990", "4030"), ("4000", "4020"), ("4005", "4015"))
    order = [
        by_bounds(features, "3990", "4030"),
        by_bounds(features, "4000", "4020"),
        by_bounds(features, "4005", "4015"),
    ]

    selection = select_trade_plan(features, ranked(features, bai_entry_candidate_ids=order))

    assert len(selection.bai_entries) == 1
    assert selected_bounds(selection, EntrySide.BAI) == [("3990", "4030")]
    suppressed = [
        decision
        for decision in selection.decisions
        if decision.outcome is SelectionOutcome.SUPPRESSED_CONTAINMENT
    ]
    assert len(suppressed) == 2


def test_suppression_compares_against_selected_zones_only() -> None:
    """§8. A zone suppressed at rank 2 cannot suppress anything at rank 3.

    3990-4030 takes rank 1 and swallows 4000-4020. The third zone, 4001-4019,
    is inside the suppressed one too - but it is suppressed by the *selected*
    outer zone, which is the only thing on the page to be redundant with.
    """
    features = bai_zones(("3990", "4030"), ("4000", "4020"), ("4001", "4019"))
    order = [
        by_bounds(features, "3990", "4030"),
        by_bounds(features, "4000", "4020"),
        by_bounds(features, "4001", "4019"),
    ]

    selection = select_trade_plan(features, ranked(features, bai_entry_candidate_ids=order))
    record = selection.decision(by_bounds(features, "4001", "4019"))

    assert record is not None
    assert record.suppressed_by_candidate_id == by_bounds(features, "3990", "4030")


# --------------------------------------------------------------------------
# §4, §27: the cap
# --------------------------------------------------------------------------


SEVEN_BAI = (
    ("3900", "3910"),
    ("3920", "3930"),
    ("3940", "3950"),
    ("3960", "3970"),
    ("3980", "3990"),
    ("4000", "4010"),
    ("4020", "4030"),
)

SEVEN_SEO = (
    ("3900", "3910"),
    ("3920", "3930"),
    ("3940", "3950"),
    ("3960", "3970"),
    ("3980", "3990"),
    ("4000", "4010"),
    ("4020", "4030"),
)


def test_27_only_six_bai_zones_survive_and_the_ranking_picks_which() -> None:
    """§4, §27; Round 6.7 §R. Seven non-contained zones; the last one is recorded."""
    features = bai_zones(*SEVEN_BAI)
    order = list(expected_buckets(features)[BAI_KEY])

    forward = select_trade_plan(features, ranked(features, bai_entry_candidate_ids=order))
    backward = select_trade_plan(
        features, ranked(features, bai_entry_candidate_ids=list(reversed(order)))
    )

    assert len(forward.bai_entries) == MAX_ENTRY_ZONES_PER_SIDE == 6
    assert len(backward.bai_entries) == 6

    assert {zone.candidate_id for zone in forward.bai_entries} == set(order[:6])
    assert {zone.candidate_id for zone in backward.bai_entries} == set(order[-6:])
    assert {zone.candidate_id for zone in forward.bai_entries} != {
        zone.candidate_id for zone in backward.bai_entries
    }


def test_27_the_one_that_misses_the_cut_is_recorded_as_below_max_count() -> None:
    """§8, §27."""
    features = bai_zones(*SEVEN_BAI)
    order = list(expected_buckets(features)[BAI_KEY])

    selection = select_trade_plan(features, ranked(features, bai_entry_candidate_ids=order))
    missed = [
        decision.candidate_id
        for decision in selection.decisions
        if decision.outcome is SelectionOutcome.BELOW_MAX_COUNT
    ]

    assert missed == order[6:]
    assert len(missed) == 1


def test_27_geometry_never_changes_with_the_cut() -> None:
    """§27, §32. The same candidate renders identically whichever ranking wins."""
    features = bai_zones(*SEVEN_BAI)
    order = list(expected_buckets(features)[BAI_KEY])

    forward = select_trade_plan(features, ranked(features, bai_entry_candidate_ids=order))
    backward = select_trade_plan(
        features, ranked(features, bai_entry_candidate_ids=list(reversed(order)))
    )

    shared = {zone.candidate_id for zone in forward.bai_entries} & {
        zone.candidate_id for zone in backward.bai_entries
    }
    assert shared
    for identity in shared:
        first = next(z for z in forward.bai_entries if z.candidate_id == identity)
        second = next(z for z in backward.bai_entries if z.candidate_id == identity)
        assert (first.lower, first.upper, first.midpoint) == (
            second.lower,
            second.upper,
            second.midpoint,
        )


def test_27_the_seo_side_caps_independently() -> None:
    """§4. Six per side, not six in total."""
    features = seo_zones(*SEVEN_SEO)

    selection = select_trade_plan(features, natural(features))

    assert len(selection.seo_entries) == 6
    assert len(selection.bai_entries) == 0


def test_both_sides_can_be_full_at_once() -> None:
    """§4. Twelve published zones is legal; six is a per-side cap."""
    features = features_of(
        *(
            decide(gap_source(lower, upper, GapDirection.BULLISH), price="4200", tag="-bai")
            for lower, upper in SEVEN_BAI
        ),
        *(
            decide(gap_source(lower, upper, GapDirection.BEARISH), price="4200", tag="-seo")
            for lower, upper in (
                ("4300", "4310"),
                ("4320", "4330"),
                ("4340", "4350"),
                ("4360", "4370"),
                ("4380", "4390"),
                ("4400", "4410"),
            )
        ),
        price="4200",
    )

    selection = select_trade_plan(features, natural(features))

    assert len(selection.bai_entries) == 6
    assert len(selection.seo_entries) == 6


# --------------------------------------------------------------------------
# §28: fewer than three
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ((), 0),
        ((("3900", "3910"),), 1),
        ((("3900", "3910"), ("3920", "3930")), 2),
        ((("3900", "3910"), ("3920", "3930"), ("3940", "3950")), 3),
    ],
)
def test_28_no_padding_to_reach_three(bounds: tuple[tuple[str, str], ...], expected: int) -> None:
    """§4, §28. Two honest zones publish two."""
    features = bai_zones(*bounds) if bounds else empty_features()

    selection = select_trade_plan(features, natural(features))

    assert len(selection.bai_entries) == expected
    assert len(selection.seo_entries) == 0


def empty_features() -> CandidateFeatureAnalysis:
    """A reading with no candidates at all. A real state, not an error."""
    reading = eligibility(decide(gap_source("3990", "4010")))
    empty = replace(
        reading,
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
    return build_candidate_features(consolidate_candidates(empty))


def test_28_an_empty_reading_selects_nothing_and_does_not_fail() -> None:
    """§28."""
    features = empty_features()

    selection = select_trade_plan(features, natural(features))

    assert selection.seo_entries == ()
    assert selection.bai_entries == ()
    assert selection.seo_reference is None
    assert selection.bai_reference is None
    assert selection.decisions == ()


# --------------------------------------------------------------------------
# §9, §29: the main zone
# --------------------------------------------------------------------------


def test_29_the_highest_ranked_survivor_is_the_main_zone() -> None:
    """§9."""
    features = bai_zones(("3900", "3910"), ("3920", "3930"), ("3940", "3950"))
    order = list(expected_buckets(features)[BAI_KEY])

    selection = select_trade_plan(features, ranked(features, bai_entry_candidate_ids=order))
    main = selection.main_zone(EntrySide.BAI)

    assert main is not None
    assert main.candidate_id == order[0]
    assert main.ai_rank == 1
    assert main.label is ZoneLabel.VUNG_CHINH


def test_29_the_main_zone_follows_the_ranking_and_not_the_price() -> None:
    """§9, §32. Reordering the ranking moves the label and nothing else."""
    features = bai_zones(("3900", "3910"), ("3920", "3930"), ("3940", "3950"))
    order = list(expected_buckets(features)[BAI_KEY])

    first = select_trade_plan(features, ranked(features, bai_entry_candidate_ids=order))
    second = select_trade_plan(
        features, ranked(features, bai_entry_candidate_ids=list(reversed(order)))
    )

    assert first.main_zone(EntrySide.BAI) is not None
    assert second.main_zone(EntrySide.BAI) is not None
    assert first.main_zone(EntrySide.BAI).candidate_id != (  # type: ignore[union-attr]
        second.main_zone(EntrySide.BAI).candidate_id  # type: ignore[union-attr]
    )
    assert selected_bounds(first, EntrySide.BAI) == selected_bounds(second, EntrySide.BAI)


def test_29_when_rank_two_is_contained_by_rank_one_rank_one_is_main() -> None:
    """§29."""
    features = bai_zones(("3990", "4030"), ("4000", "4020"), ("3900", "3910"))
    order = [
        by_bounds(features, "3990", "4030"),
        by_bounds(features, "4000", "4020"),
        by_bounds(features, "3900", "3910"),
    ]

    selection = select_trade_plan(features, ranked(features, bai_entry_candidate_ids=order))
    main = selection.main_zone(EntrySide.BAI)

    assert main is not None
    assert (main.lower, main.upper) == (Decimal("3990"), Decimal("4030"))
    assert len(selection.bai_entries) == 2


def test_29_exactly_one_main_zone_per_side() -> None:
    """§9."""
    features = features_of(
        *(
            decide(gap_source(lower, upper, GapDirection.BULLISH), price="4200", tag="-bai")
            for lower, upper in SEVEN_BAI[:3]
        ),
        *(
            decide(gap_source(lower, upper, GapDirection.BEARISH), price="4200", tag="-seo")
            for lower, upper in (("4300", "4310"), ("4320", "4330"))
        ),
        price="4200",
    )

    selection = select_trade_plan(features, natural(features))

    for side in (EntrySide.SEO, EntrySide.BAI):
        labelled = [zone for zone in selection.entries(side) if zone.label is ZoneLabel.VUNG_CHINH]
        assert len(labelled) == 1, side


def test_29_no_selected_zone_means_no_main_zone() -> None:
    """§9."""
    selection = select_trade_plan(empty_features(), natural(empty_features()))

    assert selection.main_zone(EntrySide.BAI) is None
    assert selection.main_zone(EntrySide.SEO) is None


def test_only_two_labels_exist() -> None:
    """§10. SCALP, CANH_CA_TUAN and HOLD are absent, deliberately."""
    assert {member.value for member in ZoneLabel} == {"VUNG_CHINH", "SAU_HON"}


# --------------------------------------------------------------------------
# §11-§12, §30: the farther reference
# --------------------------------------------------------------------------


def with_references(
    *, upper: tuple[str, ...] = (), lower: tuple[str, ...] = (), price: str = "4000"
) -> CandidateFeatureAnalysis:
    """A reading with SEO zones 4020-4050, BAI zones 3950-3980 and pools."""
    return features_of(
        decide(gap_source("4020", "4050", GapDirection.BEARISH), price=price, tag="-seo"),
        decide(gap_source("3950", "3980", GapDirection.BULLISH), price=price, tag="-bai"),
        *(
            decide(pool_source(level, level, LiquiditySide.BUY_SIDE), price=price, tag=f"-u{level}")
            for level in upper
        ),
        *(
            decide(
                pool_source(level, level, LiquiditySide.SELL_SIDE), price=price, tag=f"-l{level}"
            )
            for level in lower
        ),
        price=price,
    )


def test_30_the_first_strictly_farther_upper_reference_is_selected() -> None:
    """§11, §30. 4048 and 4050 are not farther than a zone ending at 4050."""
    features = with_references(upper=("4048", "4050", "4060", "4070"))
    order = [
        feature.candidate_id
        for level in ("4048", "4050", "4060", "4070")
        for feature in features.of_role(CandidateRole.UPPER_REFERENCE)
        if feature.reference_level == Decimal(level)
    ]

    selection = select_trade_plan(features, ranked(features, upper_reference_candidate_ids=order))

    assert selection.seo_reference is not None
    assert selection.seo_reference.level == Decimal("4060")
    assert selection.seo_reference.label is ZoneLabel.SAU_HON
    assert selection.seo_reference.role is CandidateRole.UPPER_REFERENCE


def test_30_the_first_strictly_farther_lower_reference_is_selected() -> None:
    """§11, §30. The mirror: 3960 and 3950 are not below a zone starting at 3950."""
    features = with_references(lower=("3960", "3950", "3940", "3920"))
    order = [
        feature.candidate_id
        for level in ("3960", "3950", "3940", "3920")
        for feature in features.of_role(CandidateRole.LOWER_REFERENCE)
        if feature.reference_level == Decimal(level)
    ]

    selection = select_trade_plan(features, ranked(features, lower_reference_candidate_ids=order))

    assert selection.bai_reference is not None
    assert selection.bai_reference.level == Decimal("3940")


def test_30_only_one_reference_per_side() -> None:
    """§11."""
    features = with_references(upper=("4060", "4070", "4080"))

    selection = select_trade_plan(features, natural(features))

    assert selection.seo_reference is not None
    published = [
        decision
        for decision in selection.decisions
        if decision.bucket == UPPER_KEY and decision.outcome is SelectionOutcome.SELECTED
    ]
    assert len(published) == 1
    later = [
        decision
        for decision in selection.decisions
        if decision.outcome is SelectionOutcome.REFERENCE_ALREADY_SELECTED
    ]
    assert len(later) == 2


def test_a_reference_equal_to_the_boundary_is_not_farther() -> None:
    """§11. Strictly beyond, so an edge-touching level is refused."""
    features = with_references(upper=("4050",))

    selection = select_trade_plan(features, natural(features))

    assert selection.seo_reference is None
    record = selection.decision(features.of_role(CandidateRole.UPPER_REFERENCE)[0].candidate_id)
    assert record is not None
    assert record.outcome is SelectionOutcome.NOT_FARTHER


def test_no_reference_is_published_when_the_side_has_no_zone() -> None:
    """§11. A farther level with nothing to be farther than is not a plan."""
    features = features_of(
        decide(gap_source("3950", "3980", GapDirection.BULLISH), tag="-bai"),
        decide(pool_source("4060", "4060", LiquiditySide.BUY_SIDE), tag="-u"),
    )

    selection = select_trade_plan(features, natural(features))

    assert selection.seo_entries == ()
    assert selection.seo_reference is None
    record = selection.decision(features.of_role(CandidateRole.UPPER_REFERENCE)[0].candidate_id)
    assert record is not None
    assert record.outcome is SelectionOutcome.NO_ENTRY_ZONE


def test_a_reference_stays_a_single_price() -> None:
    """§12. Never widened into a zone, in either direction."""
    features = with_references(upper=("4060",))

    selection = select_trade_plan(features, natural(features))
    reference = selection.seo_reference

    assert reference is not None
    assert reference.level == Decimal("4060")
    fields = set(type(reference).__dataclass_fields__)
    for forbidden in ("lower", "upper", "midpoint", "width", "side"):
        assert forbidden not in fields, forbidden


def test_the_reference_ranking_decides_which_farther_level_wins() -> None:
    """§11, §32."""
    features = with_references(upper=("4060", "4070"))
    high = next(
        f.candidate_id
        for f in features.of_role(CandidateRole.UPPER_REFERENCE)
        if f.reference_level == Decimal("4070")
    )
    low = next(
        f.candidate_id
        for f in features.of_role(CandidateRole.UPPER_REFERENCE)
        if f.reference_level == Decimal("4060")
    )

    first = select_trade_plan(features, ranked(features, upper_reference_candidate_ids=[low, high]))
    second = select_trade_plan(
        features, ranked(features, upper_reference_candidate_ids=[high, low])
    )

    assert first.seo_reference is not None and second.seo_reference is not None
    assert first.seo_reference.level == Decimal("4060")
    assert second.seo_reference.level == Decimal("4070")


# --------------------------------------------------------------------------
# §7: fail closed on an impossible input
# --------------------------------------------------------------------------


def test_7_identical_geometry_on_one_side_fails_closed() -> None:
    """§7. Consolidation should have merged these; papering over it would hide a bug.

    Constructed by duplicating a candidate inside a finished feature analysis,
    because the honest pipeline cannot produce this state - which is the point.
    """
    features = bai_zones(("3900", "3910"))
    twin = replace(features.candidates[0], candidate_id="ffffffffffffffff")
    broken = replace(features, candidates=(*features.candidates, twin))
    ranking = replace(
        natural(features),
        bai_entry_candidate_ids=(*natural(features).bai_entry_candidate_ids, twin.candidate_id),
    )

    with pytest.raises(TradePlanSelectionError, match="identical geometry"):
        select_trade_plan(broken, ranking)


def test_a_reference_in_an_entry_bucket_is_refused() -> None:
    """§8. Refused - by the provenance check, which fires first.

    Worth recording which check catches it. ``expected_buckets`` is derived from
    role and side, so a candidate in the wrong bucket always makes the bucket
    contents disagree before the per-candidate role check is reached. The role
    check is a backstop, and the next test exercises it as one.
    """
    features = with_references(upper=("4060",))
    reference_id = features.of_role(CandidateRole.UPPER_REFERENCE)[0].candidate_id
    ranking = replace(
        natural(features),
        bai_entry_candidate_ids=(*natural(features).bai_entry_candidate_ids, reference_id),
        upper_reference_candidate_ids=(),
    )

    with pytest.raises(TradePlanSelectionError, match="does not describe this candidate set"):
        select_trade_plan(features, ranking)


def test_a_seo_zone_in_the_bai_bucket_is_refused() -> None:
    """§8. Same route, same refusal."""
    features = with_references()
    seo_id = features.entry_zones(EntrySide.SEO)[0].candidate_id
    ranking = replace(
        natural(features),
        bai_entry_candidate_ids=(*natural(features).bai_entry_candidate_ids, seo_id),
        seo_entry_candidate_ids=(),
    )

    with pytest.raises(TradePlanSelectionError, match="does not describe this candidate set"):
        select_trade_plan(features, ranking)


def test_the_role_backstop_refuses_a_reference_offered_as_a_zone() -> None:
    """§8. The backstop itself, reached directly because nothing else can reach it."""
    from goldpipeline.services.trade_plan_selector import _require_role

    features = with_references(upper=("4060",))
    reference = features.of_role(CandidateRole.UPPER_REFERENCE)[0]

    with pytest.raises(TradePlanSelectionError, match="which is a UPPER_REFERENCE"):
        _require_role(reference, side=EntrySide.BAI, bucket=BAI_KEY)


def test_the_role_backstop_refuses_a_zone_from_the_other_side() -> None:
    """§8."""
    from goldpipeline.services.trade_plan_selector import _require_role

    features = with_references()
    zone = features.entry_zones(EntrySide.SEO)[0]

    with pytest.raises(TradePlanSelectionError, match="whose side is SEO"):
        _require_role(zone, side=EntrySide.BAI, bucket=BAI_KEY)


# --------------------------------------------------------------------------
# §14-§17: geometry and display order
# --------------------------------------------------------------------------


def test_14_geometry_is_copied_from_the_consolidated_candidate() -> None:
    """§14. Exactly - no round, offset, padding or adjustment of any kind."""
    features = bai_zones(("3900", "3910"), ("3920", "3930"))

    selection = select_trade_plan(features, natural(features))

    for zone in selection.bai_entries:
        candidate = features.candidate(zone.candidate_id)
        assert candidate is not None
        assert zone.lower == candidate.lower
        assert zone.upper == candidate.upper
        assert zone.midpoint == candidate.midpoint
        assert zone.candidate is candidate


def test_14_a_reference_level_is_copied_exactly() -> None:
    """§14."""
    features = with_references(upper=("4060.50",))

    selection = select_trade_plan(features, natural(features))

    assert selection.seo_reference is not None
    candidate = features.candidate(selection.seo_reference.candidate_id)
    assert candidate is not None
    assert selection.seo_reference.level == candidate.reference_level == Decimal("4060.50")


def test_16_seo_renders_nearest_first_then_upward() -> None:
    """§16. Distance ascending, then geometry ascending."""
    features = seo_zones(("4000", "4010"), ("4020", "4030"), ("4040", "4050"), price="3900")

    selection = select_trade_plan(features, natural(features))

    assert selected_bounds(selection, EntrySide.SEO) == [
        ("4000", "4010"),
        ("4020", "4030"),
        ("4040", "4050"),
    ]


def test_17_bai_renders_nearest_first_then_downward() -> None:
    """§17. Distance ascending, then geometry descending."""
    features = bai_zones(("3900", "3910"), ("3920", "3930"), ("3940", "3950"), price="4000")

    selection = select_trade_plan(features, natural(features))

    assert selected_bounds(selection, EntrySide.BAI) == [
        ("3940", "3950"),
        ("3920", "3930"),
        ("3900", "3910"),
    ]


def test_15_display_order_ignores_the_ranking() -> None:
    """§15. Price decides what a reader sees; rank decides what survives."""
    features = bai_zones(("3900", "3910"), ("3920", "3930"), ("3940", "3950"), price="4000")
    order = list(expected_buckets(features)[BAI_KEY])

    forward = select_trade_plan(features, ranked(features, bai_entry_candidate_ids=order))
    backward = select_trade_plan(
        features, ranked(features, bai_entry_candidate_ids=list(reversed(order)))
    )

    assert selected_bounds(forward, EntrySide.BAI) == selected_bounds(backward, EntrySide.BAI)
    assert [zone.ai_rank for zone in forward.bai_entries] != [
        zone.ai_rank for zone in backward.bai_entries
    ]


def test_16_zones_at_zero_distance_order_by_geometry() -> None:
    """§16. Overlapping the market gives distance 0, so price breaks the tie."""
    features = seo_zones(("3990", "4010"), ("3995", "4020"), price="4000")

    selection = select_trade_plan(features, natural(features))

    assert all(
        zone.candidate.distance_to_reference == Decimal("0") for zone in selection.seo_entries
    )
    assert selected_bounds(selection, EntrySide.SEO) == [("3990", "4010"), ("3995", "4020")]


# --------------------------------------------------------------------------
# §13: the model
# --------------------------------------------------------------------------


def test_the_selection_carries_no_prose() -> None:
    """§13."""
    fields = set(TradePlanSelection.__dataclass_fields__)

    assert fields == {
        "method_version",
        "observed_at",
        "symbol",
        "reference_price",
        "seo_entries",
        "bai_entries",
        "seo_reference",
        "bai_reference",
        "decisions",
        "features",
        "ranking",
    }
    for forbidden in ("text", "prose", "body", "summary", "note", "comment", "headline"):
        assert forbidden not in fields, forbidden


def test_every_ranked_candidate_gets_exactly_one_decision() -> None:
    """§8. The audit trail is complete, so absence is never unexplained."""
    features = with_references(upper=("4048", "4060", "4070"), lower=("3940",))

    selection = select_trade_plan(features, natural(features))
    recorded = [decision.candidate_id for decision in selection.decisions]

    assert sorted(recorded) == sorted(selection.ranking.ranked_candidate_ids)
    assert len(recorded) == len(set(recorded))


def test_every_model_is_frozen() -> None:
    features = with_references(upper=("4060",))
    selection = select_trade_plan(features, natural(features))

    assert selection.seo_reference is not None
    for target, field in (
        (selection, "seo_entries"),
        (selection.seo_entries[0], "lower"),
        (selection.seo_reference, "level"),
        (selection.decisions[0], "outcome"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(target, field, None)


def test_nothing_upstream_is_mutated() -> None:
    features = with_references(upper=("4060",))
    ranking = natural(features)
    before = (repr(features), repr(ranking))

    select_trade_plan(features, ranking)

    assert (repr(features), repr(ranking)) == before


def test_repeated_selection_is_identical() -> None:
    features = with_references(upper=("4060",))
    ranking = natural(features)

    assert select_trade_plan(features, ranking) == select_trade_plan(features, ranking)
