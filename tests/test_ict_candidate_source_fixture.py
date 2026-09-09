"""Five timeframes projected, pinned exactly, with every awkward case kept.

Round 6.6e.2a §20, §26-§30, §39-§45. The counts and identities below were read
out of the engine before they were written down.

The snapshot is Round 6.6e.1's, whose sparse shape is the point: only H1 has
both blocks and gaps, only M15 has pools, and M5 has nothing at all. A fixture
where every source type appeared on every timeframe would hide exactly the
cases this layer has to get right.
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import ICT_TIMEFRAMES, IctMarketSnapshot
from goldpipeline.services.ict_candidate_source import (
    CandidateGeometryKind,
    CandidateSource,
    CandidateSourceKind,
    CandidateSourceProjection,
    CandidateSourceTimeframe,
    FairValueGapEvidence,
    LiquidityPoolEvidence,
    OrderBlockEvidence,
    project_candidate_sources,
)
from goldpipeline.services.ict_composite import analyse_ict_composite
from goldpipeline.services.ict_order_block import OrderBlockZoneBasis
from goldpipeline.services.ict_order_block_lifecycle import OrderBlockMitigationRule
from tests.test_ict_composite_fixture import (
    OBSERVED_AT,
    PATHS,
    TF_OPENS,
    config,
    divergent_snapshot,
)
from tests.test_ict_order_block import reopen
from tests.test_ict_range_fixture import QUIET, ending_at

BASES = tuple(OrderBlockZoneBasis)
RULES = tuple(OrderBlockMitigationRule)


def block_of(source: CandidateSource) -> OrderBlockEvidence:
    """Narrow the evidence union, asserting the kind rather than assuming it.

    The union is typed precisely so a reader cannot reach for the wrong payload;
    these helpers are what honouring that looks like in a test.
    """
    assert isinstance(source.evidence, OrderBlockEvidence), source.kind
    return source.evidence


def gap_of(source: CandidateSource) -> FairValueGapEvidence:
    assert isinstance(source.evidence, FairValueGapEvidence), source.kind
    return source.evidence


def pool_of(source: CandidateSource) -> LiquidityPoolEvidence:
    assert isinstance(source.evidence, LiquidityPoolEvidence), source.kind
    return source.evidence


def direction_of(source: CandidateSource) -> str:
    """A source's own direction or side, whichever it has."""
    if source.kind is CandidateSourceKind.ORDER_BLOCK:
        return block_of(source).order_block.direction.value
    if source.kind is CandidateSourceKind.FAIR_VALUE_GAP:
        return gap_of(source).gap.direction.value
    return pool_of(source).pool.side.value


def projection(**kwargs: object) -> CandidateSourceProjection:
    return project_candidate_sources(
        analyse_ict_composite(divergent_snapshot(), config=config(**kwargs))  # type: ignore[arg-type]
    )


def kinds(entry: CandidateSourceTimeframe) -> dict[str, int]:
    return dict(Counter(source.kind.value for source in entry.sources))


def statuses(entry: CandidateSourceTimeframe) -> dict[str, int]:
    return dict(
        Counter(f"{source.kind.value}:{source.evidence.status.value}" for source in entry.sources)
    )


# --------------------------------------------------------------------------
# §39: the whole projection, pinned
# --------------------------------------------------------------------------


def test_the_projection_pins_every_source_count() -> None:
    assert [
        (entry.timeframe.value, len(entry.sources), kinds(entry))
        for entry in projection().timeframes
    ] == [
        ("H4", 1, {"ORDER_BLOCK": 1}),
        ("H1", 8, {"ORDER_BLOCK": 5, "FAIR_VALUE_GAP": 3}),
        ("M15", 5, {"LIQUIDITY_POOL": 5}),
        ("M5", 0, {}),
        ("M1", 13, {"FAIR_VALUE_GAP": 13}),
    ]


def test_the_projection_pins_every_status() -> None:
    assert [(entry.timeframe.value, statuses(entry)) for entry in projection().timeframes] == [
        ("H4", {"ORDER_BLOCK:MITIGATED": 1}),
        (
            "H1",
            {"ORDER_BLOCK:INVALIDATED": 3, "ORDER_BLOCK:MITIGATED": 2, "FAIR_VALUE_GAP:OPEN": 3},
        ),
        ("M15", {"LIQUIDITY_POOL:SWEPT": 4, "LIQUIDITY_POOL:CLOSED_THROUGH": 1}),
        ("M5", {}),
        ("M1", {"FAIR_VALUE_GAP:FILLED": 6, "FAIR_VALUE_GAP:TOUCHED": 4, "FAIR_VALUE_GAP:OPEN": 3}),
    ]


def test_the_h1_sources_are_pinned_in_full() -> None:
    """The richest timeframe, geometry and identity together."""
    assert [
        (
            source.kind.value,
            str(source.lower),
            str(source.upper),
            str(source.midpoint),
            str(source.width),
            source.evidence.status.value,
            source.source_id,
        )
        for source in projection().require(Timeframe.H1).sources
    ] == [
        ("ORDER_BLOCK", "3970", "4010", "3990", "40", "INVALIDATED", "17ef00f0e5fd28eb"),
        ("ORDER_BLOCK", "3990", "4010", "4000", "20", "INVALIDATED", "b920461298688cb1"),
        ("ORDER_BLOCK", "3990", "4085", "4037.5", "95", "MITIGATED", "708facc159013afe"),
        ("ORDER_BLOCK", "3990", "4085", "4037.5", "95", "MITIGATED", "dbc28651faf1ae5c"),
        ("ORDER_BLOCK", "3990", "4045", "4017.5", "55", "INVALIDATED", "6110ecd2f9a7cbb3"),
        ("FAIR_VALUE_GAP", "3960", "3990", "3975", "30", "OPEN", "9a9f9271f1939479"),
        ("FAIR_VALUE_GAP", "3960", "4050", "4005", "90", "OPEN", "ee7df4d95d7ef376"),
        ("FAIR_VALUE_GAP", "3960", "4050", "4005", "90", "OPEN", "8875c1e95356e648"),
    ]


def test_the_m15_liquidity_bands_are_pinned_in_full() -> None:
    assert [
        (
            str(source.lower),
            str(source.upper),
            str(source.width),
            pool_of(source).pool.side.value,
            source.evidence.status.value,
        )
        for source in projection().require(Timeframe.M15).sources
    ] == [
        ("4060", "4060", "0", "BUY_SIDE", "SWEPT"),
        ("4030.00", "4030.30", "0.30", "BUY_SIDE", "CLOSED_THROUGH"),
        ("4030.00", "4030.00", "0.00", "BUY_SIDE", "SWEPT"),
        ("3950", "3950", "0", "SELL_SIDE", "SWEPT"),
        ("3974.70", "3975.00", "0.30", "SELL_SIDE", "SWEPT"),
    ]


def test_a_timeframe_with_nothing_to_say_projects_nothing() -> None:
    """§39. Absence is a legitimate projection, not a missing one."""
    entry = projection().require(Timeframe.M5)

    assert entry.sources == ()
    assert entry.structure_bias.value == "NEUTRAL"
    assert entry.active_dealing_range is None


# --------------------------------------------------------------------------
# §40: terminal sources are preserved
# --------------------------------------------------------------------------


def test_every_terminal_source_type_survives_the_projection() -> None:
    """§9, §40. The boundary the next round needs in order to show its working."""
    result = projection()
    seen = {
        f"{source.kind.value}:{source.evidence.status.value}"
        for entry in result.timeframes
        for source in entry.sources
    }

    assert "ORDER_BLOCK:INVALIDATED" in seen
    assert "FAIR_VALUE_GAP:FILLED" in seen
    assert {"LIQUIDITY_POOL:SWEPT", "LIQUIDITY_POOL:CLOSED_THROUGH"} <= seen


def test_terminal_sources_carry_their_terminal_evidence() -> None:
    """§40. Not merely present - the witness that retired them is reachable."""
    result = projection()

    retired = [
        s
        for s in result.require(Timeframe.H1).sources
        if s.kind is CandidateSourceKind.ORDER_BLOCK and s.evidence.status.value == "INVALIDATED"
    ]
    assert len(retired) == 3
    for source in retired:
        assert block_of(source).state.invalidated_at is not None
        assert block_of(source).state.invalidation_witness is not None

    filled = [
        s for s in result.require(Timeframe.M1).sources if s.evidence.status.value == "FILLED"
    ]
    assert len(filled) == 6
    for source in filled:
        assert gap_of(source).state.filled_at is not None

    swept = [s for s in result.require(Timeframe.M15).sources if pool_of(s).pool.is_terminal]
    assert len(swept) == 5
    for source in swept:
        assert pool_of(source).terminal_event_id is not None


# --------------------------------------------------------------------------
# §41: duplicate geometry survives
# --------------------------------------------------------------------------


def test_two_order_blocks_with_one_zone_stay_two_facts() -> None:
    """§15, §41. Round 6.6d.1's same-source pair, projected independently."""
    blocks = [
        s
        for s in projection().require(Timeframe.H1).sources
        if s.kind is CandidateSourceKind.ORDER_BLOCK
        and (s.lower, s.upper) == (Decimal("3990"), Decimal("4085"))
    ]

    assert len(blocks) == 2
    first, second = blocks
    assert first.lower == second.lower and first.upper == second.upper
    assert first.midpoint == second.midpoint
    assert (
        block_of(first).order_block.source_bar_open_time
        == block_of(second).order_block.source_bar_open_time
    ), "the same source candle"
    assert first.source_id != second.source_id
    assert first.candidate_source_id != second.candidate_source_id


def test_two_gaps_with_one_band_stay_two_facts() -> None:
    """§15, §41. Overlapping gaps are not merged either."""
    gaps = [
        s
        for s in projection().require(Timeframe.H1).sources
        if s.kind is CandidateSourceKind.FAIR_VALUE_GAP
        and (s.lower, s.upper) == (Decimal("3960"), Decimal("4050"))
    ]

    assert len(gaps) == 2
    assert gaps[0].source_id != gaps[1].source_id
    assert gaps[0].candidate_source_id != gaps[1].candidate_source_id
    assert gaps[0].formed_at != gaps[1].formed_at


def test_nothing_anywhere_is_deduplicated_by_geometry() -> None:
    """§15, stated across the whole projection rather than case by case."""
    result = projection()
    for entry in result.timeframes:
        by_bounds: dict[tuple[str, str], list[str]] = {}
        for source in entry.sources:
            by_bounds.setdefault((str(source.lower), str(source.upper)), []).append(
                source.candidate_source_id
            )
        shared = {bounds: ids for bounds, ids in by_bounds.items() if len(ids) > 1}
        for bounds, ids in shared.items():
            assert len(set(ids)) == len(ids), bounds

    assert any(
        len([s for s in result.require(Timeframe.H1).sources if (s.lower, s.upper) == pair]) == 2
        for pair in ((Decimal("3990"), Decimal("4085")), (Decimal("3960"), Decimal("4050")))
    ), "the fixture really does contain shared geometry"


# --------------------------------------------------------------------------
# §42: zero-width liquidity
# --------------------------------------------------------------------------


def test_an_exact_equality_pool_keeps_its_single_price() -> None:
    """§42. Not widened, not given an invented tick."""
    flat = [s for s in projection().require(Timeframe.M15).sources if s.lower == s.upper]

    assert len(flat) == 3
    for source in flat:
        assert source.geometry is CandidateGeometryKind.BAND
        assert source.width == 0
        assert source.lower == source.midpoint == source.upper
        assert pool_of(source).pool.lower == source.lower, "the pool's own price, unchanged"


def test_the_zero_width_prices_are_pinned_exactly() -> None:
    """§42, including the Decimal exponents the liquidity engine produced."""
    flat = [s for s in projection().require(Timeframe.M15).sources if s.width == 0]

    assert [str(source.lower) for source in flat] == ["4060", "4030.00", "3950"]


# --------------------------------------------------------------------------
# §19-§20: range context, which never filters
# --------------------------------------------------------------------------


def test_the_active_range_context_is_the_composite_s_own() -> None:
    """§19. Attached, never reconstructed."""
    full = analyse_ict_composite(divergent_snapshot(), config=config())
    result = project_candidate_sources(full)

    for entry, source_entry in zip(full.timeframes, result.timeframes, strict=True):
        assert source_entry.active_dealing_range is entry.dealing_ranges.active_range
        assert source_entry.structure_bias is entry.dealing_ranges.structure_bias


def test_a_timeframe_with_a_bias_and_no_active_range_still_projects_everything() -> None:
    """§20. Round 6.6c.3b's legitimate state, and it filters nothing."""
    entry = projection().require(Timeframe.M15)

    assert entry.structure_bias.value == "BULLISH"
    assert entry.active_dealing_range is None
    assert len(entry.sources) == 5


def test_a_bearish_gap_in_the_discount_half_still_projects() -> None:
    """§20's own named case, present in the fixture for real.

    H1 carries a BEARISH fair value gap whose midpoint sits in the DISCOUNT half
    of its range - the "wrong" side, and exactly what a premium/discount filter
    would have removed. It projects like anything else. Three BEARISH order
    blocks sit in discount beside it for the same reason.

    The mirror case, a bullish source in premium, is not in this fixture and is
    not manufactured for the sake of symmetry; the location distribution pinned
    below is what stands in for it.
    """
    from goldpipeline.services.ict_range import PriceLocation
    from goldpipeline.services.ict_structure import BreakDirection

    entry = projection().require(Timeframe.H1)
    current = entry.active_dealing_range
    assert current is not None

    wrong_side = [
        source
        for source in entry.sources
        if current.locate(source.midpoint) is PriceLocation.DISCOUNT
        and direction_of(source) == "BEARISH"
    ]

    assert len(wrong_side) == 4
    assert [s.kind.value for s in wrong_side].count("FAIR_VALUE_GAP") == 1
    assert current.direction is BreakDirection.BEARISH
    assert len(entry.sources) == 8, "and every one of them is still here"


def test_every_source_has_a_location_and_none_of_them_filters() -> None:
    """§20. The location is computable for every source and used for nothing.

    Pinned as a distribution rather than an inequality, so a later filter shows
    up as a changed count instead of quietly passing. No source in this fixture
    lies outside its own active range - that case is not constructible here and
    is not manufactured; what is proved instead is that the projected count
    equals the composite's regardless of where anything sits.
    """
    from collections import Counter as _Counter

    result = projection()
    seen: dict[str, dict[str, int]] = {}
    for entry in result.timeframes:
        current = entry.active_dealing_range
        if current is None:
            continue
        seen[entry.timeframe.value] = dict(
            _Counter(current.locate(source.midpoint).value for source in entry.sources)
        )

    assert seen == {"H4": {"PREMIUM": 1}, "H1": {"DISCOUNT": 8}}


def test_a_timeframe_whose_sources_have_no_range_at_all_projects_them_anyway() -> None:
    """§20. Two timeframes carry eighteen sources between them and no range."""
    result = projection()
    rangeless = [entry for entry in result.timeframes if entry.active_dealing_range is None]

    assert [entry.timeframe.value for entry in rangeless] == ["M15", "M5", "M1"]
    assert sum(len(entry.sources) for entry in rangeless) == 18


# --------------------------------------------------------------------------
# §26-§30: policy blast radius
# --------------------------------------------------------------------------


def facts(result: CandidateSourceProjection, kind: CandidateSourceKind) -> list[tuple[object, ...]]:
    return [
        (
            entry.timeframe,
            source.candidate_source_id,
            source.source_id,
            source.lower,
            source.upper,
            source.midpoint,
            source.formed_at,
        )
        for entry in result.timeframes
        for source in entry.sources
        if source.kind is kind
    ]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (OrderBlockMitigationRule.TOUCH, OrderBlockMitigationRule.MIDPOINT),
        (OrderBlockMitigationRule.TOUCH, OrderBlockMitigationRule.FULL_ZONE),
        (OrderBlockMitigationRule.MIDPOINT, OrderBlockMitigationRule.FULL_ZONE),
    ],
)
def test_the_mitigation_rule_moves_only_order_block_evidence(
    first: OrderBlockMitigationRule, second: OrderBlockMitigationRule
) -> None:
    """§26. Same ids, same geometry, possibly different lifecycle evidence."""
    left = projection(rule=first)
    right = projection(rule=second)

    assert facts(left, CandidateSourceKind.ORDER_BLOCK) == facts(
        right, CandidateSourceKind.ORDER_BLOCK
    )
    assert facts(left, CandidateSourceKind.FAIR_VALUE_GAP) == facts(
        right, CandidateSourceKind.FAIR_VALUE_GAP
    )
    assert facts(left, CandidateSourceKind.LIQUIDITY_POOL) == facts(
        right, CandidateSourceKind.LIQUIDITY_POOL
    )
    for entry, other in zip(left.timeframes, right.timeframes, strict=True):
        assert entry.of_kind(CandidateSourceKind.FAIR_VALUE_GAP) == other.of_kind(
            CandidateSourceKind.FAIR_VALUE_GAP
        )
        assert entry.of_kind(CandidateSourceKind.LIQUIDITY_POOL) == other.of_kind(
            CandidateSourceKind.LIQUIDITY_POOL
        )


def test_the_full_zone_rule_really_does_move_the_evidence() -> None:
    """The other half of §26, so the pairwise test cannot pass vacuously."""
    touch = projection(rule=OrderBlockMitigationRule.TOUCH).require(Timeframe.H1)
    full_zone = projection(rule=OrderBlockMitigationRule.FULL_ZONE).require(Timeframe.H1)

    assert statuses(touch) != statuses(full_zone)
    assert statuses(full_zone)["ORDER_BLOCK:TOUCHED"] == 2


def test_the_zone_basis_moves_order_blocks_and_nothing_else() -> None:
    """§27. No cross-contamination into gaps or pools."""
    full = projection(basis=OrderBlockZoneBasis.FULL_CANDLE)
    body = projection(basis=OrderBlockZoneBasis.BODY)

    assert facts(full, CandidateSourceKind.ORDER_BLOCK) != facts(
        body, CandidateSourceKind.ORDER_BLOCK
    )
    assert facts(full, CandidateSourceKind.FAIR_VALUE_GAP) == facts(
        body, CandidateSourceKind.FAIR_VALUE_GAP
    )
    assert facts(full, CandidateSourceKind.LIQUIDITY_POOL) == facts(
        body, CandidateSourceKind.LIQUIDITY_POOL
    )


def test_the_liquidity_tolerance_moves_pools_and_nothing_else() -> None:
    """§28."""
    narrow = projection(tolerance="0.50")
    wide = projection(tolerance="25.00")

    assert facts(narrow, CandidateSourceKind.LIQUIDITY_POOL) != facts(
        wide, CandidateSourceKind.LIQUIDITY_POOL
    )
    assert facts(narrow, CandidateSourceKind.ORDER_BLOCK) == facts(
        wide, CandidateSourceKind.ORDER_BLOCK
    )
    assert facts(narrow, CandidateSourceKind.FAIR_VALUE_GAP) == facts(
        wide, CandidateSourceKind.FAIR_VALUE_GAP
    )


def test_the_atr_period_moves_no_candidate_source_at_all() -> None:
    """§29, and the strongest statement in this matrix.

    ATR is not projected as a source and no source formation depends on it, so a
    different period must leave the entire projection - context included -
    byte-identical.
    """
    short = projection(atr=5)
    long = projection(atr=20)

    assert short.timeframes == long.timeframes
    assert short.config != long.config, "the policy really did change"


def test_the_swing_window_moves_the_swing_branch_but_not_gaps() -> None:
    """§30. The dependency boundary: gaps never looked at a swing."""
    narrow = projection(left=2, right=2)
    wide = projection(left=4, right=4)

    assert facts(narrow, CandidateSourceKind.FAIR_VALUE_GAP) == facts(
        wide, CandidateSourceKind.FAIR_VALUE_GAP
    )
    moved = facts(narrow, CandidateSourceKind.ORDER_BLOCK) != facts(
        wide, CandidateSourceKind.ORDER_BLOCK
    ) or facts(narrow, CandidateSourceKind.LIQUIDITY_POOL) != facts(
        wide, CandidateSourceKind.LIQUIDITY_POOL
    )
    assert moved, "a window change that moved nothing would prove nothing"


# --------------------------------------------------------------------------
# §43: the six-policy grid
# --------------------------------------------------------------------------


def test_gaps_and_pools_are_identical_across_all_six_policies() -> None:
    """§43."""
    grid = [projection(basis=basis, rule=rule) for basis in BASES for rule in RULES]

    assert len(grid) == 6
    reference = grid[0]
    for other in grid[1:]:
        assert facts(other, CandidateSourceKind.FAIR_VALUE_GAP) == facts(
            reference, CandidateSourceKind.FAIR_VALUE_GAP
        )
        assert facts(other, CandidateSourceKind.LIQUIDITY_POOL) == facts(
            reference, CandidateSourceKind.LIQUIDITY_POOL
        )


def test_order_block_facts_are_shared_within_a_basis_and_differ_across_them() -> None:
    """§43. Six policies, two families of order-block facts."""
    families = {
        basis.value: {
            tuple(str(item) for item in fact)
            for fact in facts(projection(basis=basis, rule=rule), CandidateSourceKind.ORDER_BLOCK)
        }
        for basis in BASES
        for rule in RULES
    }

    assert len(families) == 2
    full, body = families["FULL_CANDLE"], families["BODY"]
    assert full != body
    ids = {fact[1] for fact in full}, {fact[1] for fact in body}
    assert ids[0].isdisjoint(ids[1])


def test_the_source_counts_never_change_across_the_grid() -> None:
    """§43. Policy changes what a source *is*, never how many there are."""
    for basis in BASES:
        for rule in RULES:
            result = projection(basis=basis, rule=rule)
            assert [len(entry.sources) for entry in result.timeframes] == [1, 8, 5, 0, 13]


# --------------------------------------------------------------------------
# §44-§45: cross-timeframe isolation
# --------------------------------------------------------------------------


def altered(timeframe: Timeframe) -> IctMarketSnapshot:
    changed = dict(PATHS)
    changed[timeframe] = [*QUIET, *QUIET]
    return IctMarketSnapshot(
        observed_at=OBSERVED_AT,
        symbol="XAUUSD",
        provider="fixture",
        provider_symbol=None,
        timeframes=tuple(
            reopen(ending_at(tf, changed[tf], end=OBSERVED_AT), TF_OPENS.get(tf, {}))
            for tf in ICT_TIMEFRAMES
        ),
    )


@pytest.mark.parametrize("changed", [Timeframe.M1, Timeframe.H4, Timeframe.M15])
def test_changing_one_timeframe_leaves_the_other_projections_untouched(
    changed: Timeframe,
) -> None:
    """§44. No shared mutable candidate collection."""
    reference = projection()
    after = project_candidate_sources(analyse_ict_composite(altered(changed), config=config()))

    for tf in ICT_TIMEFRAMES:
        left = reference.require(tf)
        right = after.require(tf)
        if tf is changed:
            assert left != right, f"{tf.value} was supposed to change"
        else:
            assert left == right, f"{tf.value} moved when only {changed.value} did"


def test_no_source_is_merged_across_timeframes() -> None:
    """§45. Every fact carries its own timeframe and stays there."""
    result = projection()

    for entry in result.timeframes:
        for source in entry.sources:
            assert source.timeframe is entry.timeframe
    everywhere = [s.candidate_source_id for e in result.timeframes for s in e.sources]
    assert len(everywhere) == len(set(everywhere))
