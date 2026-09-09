"""Reference price, roles, relations and reasons, one rule at a time.

Round 6.6e.2b §2-§5, §12-§28, §42-§47. The geometry matrices call the pure
predicates on numbers written out in full, because §43-§45 are claims about
intervals against a price rather than about any particular market. The reason
and role matrices then run end to end over constructed sources, where a decision
has to hold together.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.services.ict_candidate_eligibility import (
    CANDIDATE_ELIGIBILITY_METHOD_VERSION,
    REASON_ORDER,
    TERMINAL_FVG_STATUSES,
    TERMINAL_ORDER_BLOCK_STATUSES,
    TERMINAL_POOL_STATUSES,
    CandidateEligibilityConfig,
    CandidateEligibilityError,
    CandidateRole,
    EligibilityReason,
    EntrySide,
    MarketRelation,
    ReferencePrice,
    analyse_candidate_eligibility,
    distance_of_zone,
    entry_side_for,
    reference_level_for,
    relation_of_level,
    relation_of_zone,
    resolve_reference_price,
    role_for,
)
from goldpipeline.services.ict_candidate_source import (
    CandidateGeometryKind,
    CandidateSource,
    CandidateSourceKind,
    FairValueGapEvidence,
    LiquidityPoolEvidence,
    OrderBlockEvidence,
)
from goldpipeline.services.ict_composite import analyse_ict_composite
from goldpipeline.services.ict_fvg import BarWitness, FvgState, FvgStatus
from goldpipeline.services.ict_liquidity import LiquidityPool, LiquiditySide, PoolStatus
from goldpipeline.services.ict_order_block import (
    BodyDirection,
    OrderBlock,
    OrderBlockZoneBasis,
)
from goldpipeline.services.ict_order_block_lifecycle import (
    OrderBlockMitigationRule,
    OrderBlockState,
    OrderBlockStatus,
)
from goldpipeline.services.ict_primitives import FairValueGap, GapDirection
from goldpipeline.services.ict_structure import BreakClassification, BreakDirection
from tests.test_ict_composite_fixture import config as composite_config
from tests.test_ict_composite_fixture import divergent_snapshot

MOMENT = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
HOUR = timedelta(hours=1)
PRICE = Decimal("4000")

PERMISSIVE = CandidateEligibilityConfig(
    allowed_order_block_statuses=frozenset(
        {OrderBlockStatus.ACTIVE, OrderBlockStatus.TOUCHED, OrderBlockStatus.MITIGATED}
    ),
    allowed_fvg_statuses=frozenset({FvgStatus.OPEN, FvgStatus.TOUCHED}),
)


# --------------------------------------------------------------------------
# constructed sources, so the matrices can say exactly what they mean
# --------------------------------------------------------------------------


def _order_block(lower: str, upper: str, direction: BreakDirection) -> OrderBlock:
    low, high = Decimal(lower), Decimal(upper)
    return OrderBlock(
        order_block_id=f"ob:{lower}-{upper}:{direction.value}",
        method_version="1.0.0",
        timeframe=Timeframe.H1,
        symbol="XAUUSD",
        direction=direction,
        zone_basis=OrderBlockZoneBasis.BODY,
        structure_event_id="event",
        event_classification=BreakClassification.BOS,
        protected_assignment_id="assignment",
        structural_leg_id="leg",
        formed_at=MOMENT,
        source_bar_open_time=MOMENT - HOUR,
        source_bar_close_time=MOMENT,
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


def block_source(
    lower: str,
    upper: str,
    direction: BreakDirection = BreakDirection.BULLISH,
    status: OrderBlockStatus = OrderBlockStatus.ACTIVE,
) -> CandidateSource:
    block = _order_block(lower, upper, direction)
    state = OrderBlockState(
        order_block_id=block.order_block_id,
        order_block=block,
        status=status,
        mitigation_rule=OrderBlockMitigationRule.TOUCH,
        first_touched_at=None,
        first_touch_witness=None,
        mitigated_at=None,
        mitigation_witness=None,
        invalidated_at=MOMENT if status is OrderBlockStatus.INVALIDATED else None,
        invalidation_witness=None,
    )
    return _source(
        CandidateSourceKind.ORDER_BLOCK,
        CandidateGeometryKind.ZONE,
        block.order_block_id,
        block.lower,
        block.upper,
        OrderBlockEvidence(
            order_block=block,
            state=state,
            status=status,
            mitigation_rule=OrderBlockMitigationRule.TOUCH,
        ),
    )


def gap_source(
    lower: str,
    upper: str,
    direction: GapDirection = GapDirection.BULLISH,
    status: FvgStatus = FvgStatus.OPEN,
) -> CandidateSource:
    low, high = Decimal(lower), Decimal(upper)
    gap = FairValueGap(
        timeframe=Timeframe.H1,
        direction=direction,
        formed_at=MOMENT,
        first_time=MOMENT - HOUR * 3,
        middle_time=MOMENT - HOUR * 2,
        last_time=MOMENT - HOUR,
        lower=low,
        upper=high,
    )
    witness = BarWitness(bar_open_time=MOMENT - HOUR, bar_close_time=MOMENT, low=low, high=high)
    identity = f"fvg:{lower}-{upper}:{direction.value}"
    state = FvgState(
        fvg_id=identity,
        method_version="1.0.0",
        gap=gap,
        status=status,
        first_touched_at=MOMENT if status is not FvgStatus.OPEN else None,
        first_touch=witness if status is not FvgStatus.OPEN else None,
        filled_at=MOMENT if status is FvgStatus.FILLED else None,
        fill=witness if status is FvgStatus.FILLED else None,
    )
    return _source(
        CandidateSourceKind.FAIR_VALUE_GAP,
        CandidateGeometryKind.ZONE,
        identity,
        low,
        high,
        FairValueGapEvidence(gap=gap, state=state, status=status),
    )


def pool_source(
    lower: str,
    upper: str,
    side: LiquiditySide = LiquiditySide.BUY_SIDE,
    status: PoolStatus = PoolStatus.ACTIVE,
) -> CandidateSource:
    low, high = Decimal(lower), Decimal(upper)
    identity = f"pool:{lower}-{upper}:{side.value}"
    pool = LiquidityPool(
        pool_id=identity,
        method_version="1.0.0",
        timeframe=Timeframe.H1,
        symbol="XAUUSD",
        side=side,
        formed_at=MOMENT,
        founding_swing_ids=("a", "b"),
        member_swing_ids=("a", "b"),
        lower=low,
        upper=high,
        midpoint=(low + high) / Decimal(2),
        tolerance=Decimal("0.5"),
        status=status,
        terminal_event_id=None if status is PoolStatus.ACTIVE else "event",
    )
    return _source(
        CandidateSourceKind.LIQUIDITY_POOL,
        CandidateGeometryKind.BAND,
        identity,
        low,
        high,
        LiquidityPoolEvidence(
            pool=pool,
            status=status,
            tolerance=pool.tolerance,
            terminal_event_id=pool.terminal_event_id,
        ),
    )


def _source(
    kind: CandidateSourceKind,
    geometry: CandidateGeometryKind,
    source_id: str,
    lower: Decimal,
    upper: Decimal,
    evidence: OrderBlockEvidence | FairValueGapEvidence | LiquidityPoolEvidence,
) -> CandidateSource:
    return CandidateSource(
        candidate_source_id=f"cs:{source_id}",
        method_version="1.0.0",
        timeframe=Timeframe.H1,
        symbol="XAUUSD",
        kind=kind,
        geometry=geometry,
        source_id=source_id,
        formed_at=MOMENT,
        lower=lower,
        upper=upper,
        midpoint=(lower + upper) / Decimal(2),
        width=upper - lower,
        evidence=evidence,
    )


def decide(
    source: CandidateSource, *, price: str = "4000", config: CandidateEligibilityConfig = PERMISSIVE
) -> object:
    from goldpipeline.services.ict_candidate_eligibility import _decide

    return _decide(source, price=Decimal(price), config=config)


# --------------------------------------------------------------------------
# §3-§5: the global reference price
# --------------------------------------------------------------------------


def test_the_reference_price_is_the_latest_close_across_every_timeframe() -> None:
    """§3. One authority, not five."""
    from tests.test_ict_candidate_eligibility_fixture import staggered_snapshot

    composite = analyse_ict_composite(staggered_snapshot(), config=composite_config())
    reference = resolve_reference_price(composite)

    latest = max(entry.series.latest_closed_at for entry in composite.timeframes)
    assert reference.bar_close_time == latest
    assert reference.price == Decimal("4043")
    assert reference.symbol == "XAUUSD"
    assert reference.observed_at == composite.observed_at


def test_the_reference_price_names_every_witness_at_that_instant() -> None:
    """§5, §39. Provenance is kept, not thrown away because sources agreed."""
    from tests.test_ict_candidate_eligibility_fixture import staggered_snapshot

    composite = analyse_ict_composite(staggered_snapshot(), config=composite_config())
    reference = resolve_reference_price(composite)

    assert [witness.timeframe for witness in reference.witnesses] == [Timeframe.M1]
    witness = reference.witnesses[0]
    assert witness.close == reference.price
    assert witness.bar_close_time == reference.bar_close_time
    assert witness.bar_open_time == witness.bar_close_time - timedelta(minutes=1)


def test_timeframes_disagreeing_at_the_selected_instant_fail_closed() -> None:
    """§4, §38, and the case is not manufactured.

    Round 6.6e.1's own five-timeframe snapshot has every timeframe ending at
    exactly the same instant with three different closes, because each path was
    written independently for a different engine. That is precisely the shape
    §38 asks for, and resolution refuses it rather than preferring the finest
    timeframe.
    """
    composite = analyse_ict_composite(divergent_snapshot(), config=composite_config())
    closes = {entry.timeframe.value: entry.series.bars[-1].close for entry in composite.timeframes}

    assert len({entry.series.latest_closed_at for entry in composite.timeframes}) == 1
    assert len(set(closes.values())) == 3, closes
    with pytest.raises(CandidateEligibilityError, match="cannot have two closing prices"):
        resolve_reference_price(composite)


def test_agreeing_timeframes_at_one_instant_are_all_kept_as_witnesses() -> None:
    """§39. Two timeframes, one instant, one price - and both recorded."""
    from goldpipeline.schemas.ict import IctMarketSnapshot

    shot = divergent_snapshot()
    agreeing = IctMarketSnapshot(
        observed_at=shot.observed_at,
        symbol=shot.symbol,
        provider=shot.provider,
        provider_symbol=shot.provider_symbol,
        timeframes=(shot.require(Timeframe.H4), shot.require(Timeframe.M5)),
    )
    composite = analyse_ict_composite(agreeing, config=composite_config())
    reference = resolve_reference_price(composite)

    assert [witness.timeframe for witness in reference.witnesses] == [
        Timeframe.H4,
        Timeframe.M5,
    ]
    assert {witness.close for witness in reference.witnesses} == {Decimal("4000")}
    assert reference.price == Decimal("4000")
    assert len(reference.witnesses) == len({w.timeframe for w in reference.witnesses})


def test_a_stale_latest_close_is_still_the_reference() -> None:
    """§6. No maximum-age policy, no fabricated newer price."""
    from tests.test_ict_candidate_eligibility_fixture import staggered_snapshot

    composite = analyse_ict_composite(staggered_snapshot(), config=composite_config())
    reference = resolve_reference_price(composite)
    h4 = composite.require(Timeframe.H4)

    assert h4.series.latest_closed_at < reference.bar_close_time
    assert reference.bar_close_time <= composite.observed_at


def test_the_reference_price_reads_no_clock_and_no_forming_candle() -> None:
    """§3. Derived from closed series alone.

    Read as identifiers, not as text: the docstring is free to say the function
    makes no provider call, which is exactly what it must not be punished for.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(resolve_reference_price))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert "latest_closed_at" in names
    for forbidden in ("now", "utcnow", "time", "fetch", "provider", "monotonic"):
        assert forbidden not in names, forbidden


def test_a_reference_price_is_frozen() -> None:
    from tests.test_ict_candidate_eligibility_fixture import staggered_snapshot

    composite = analyse_ict_composite(staggered_snapshot(), config=composite_config())
    reference = resolve_reference_price(composite)

    with pytest.raises(FrozenInstanceError):
        reference.price = Decimal("1")  # type: ignore[misc]
    assert isinstance(reference, ReferencePrice)


# --------------------------------------------------------------------------
# §8-§11, §42: the eligibility config
# --------------------------------------------------------------------------


def test_the_config_has_no_defaults() -> None:
    """§8."""
    with pytest.raises(TypeError):
        CandidateEligibilityConfig()  # type: ignore[call-arg]


def test_an_invalidated_order_block_cannot_be_configured_back_in() -> None:
    """§9, §42. Policy does not outrank deterministic terminal truth."""
    with pytest.raises(CandidateEligibilityError, match="terminal order-block statuses"):
        CandidateEligibilityConfig(
            allowed_order_block_statuses=frozenset(
                {OrderBlockStatus.ACTIVE, OrderBlockStatus.INVALIDATED}
            ),
            allowed_fvg_statuses=frozenset({FvgStatus.OPEN}),
        )


def test_a_filled_gap_cannot_be_configured_back_in() -> None:
    """§9, §42."""
    with pytest.raises(CandidateEligibilityError, match="terminal gap statuses"):
        CandidateEligibilityConfig(
            allowed_order_block_statuses=frozenset({OrderBlockStatus.ACTIVE}),
            allowed_fvg_statuses=frozenset({FvgStatus.OPEN, FvgStatus.FILLED}),
        )


@pytest.mark.parametrize(
    "allowed",
    [
        frozenset({OrderBlockStatus.ACTIVE}),
        frozenset({OrderBlockStatus.ACTIVE, OrderBlockStatus.TOUCHED}),
        frozenset({OrderBlockStatus.ACTIVE, OrderBlockStatus.TOUCHED, OrderBlockStatus.MITIGATED}),
        frozenset({OrderBlockStatus.MITIGATED}),
        frozenset(),
    ],
)
def test_any_non_terminal_order_block_subset_is_a_valid_policy(
    allowed: frozenset[OrderBlockStatus],
) -> None:
    """§10. Including the empty one, which simply yields no order-block entries."""
    config = CandidateEligibilityConfig(
        allowed_order_block_statuses=allowed, allowed_fvg_statuses=frozenset()
    )

    assert config.allowed_order_block_statuses == allowed


@pytest.mark.parametrize(
    "allowed",
    [
        frozenset({FvgStatus.OPEN}),
        frozenset({FvgStatus.OPEN, FvgStatus.TOUCHED}),
        frozenset({FvgStatus.TOUCHED}),
        frozenset(),
    ],
)
def test_any_non_terminal_gap_subset_is_a_valid_policy(allowed: frozenset[FvgStatus]) -> None:
    """§11."""
    config = CandidateEligibilityConfig(
        allowed_order_block_statuses=frozenset(), allowed_fvg_statuses=allowed
    )

    assert config.allowed_fvg_statuses == allowed


def test_the_terminal_sets_are_exactly_the_lifecycle_terminal_states() -> None:
    """§9. Named once, from the engines' own definitions."""
    assert {OrderBlockStatus.INVALIDATED} == TERMINAL_ORDER_BLOCK_STATUSES
    assert {FvgStatus.FILLED} == TERMINAL_FVG_STATUSES
    assert {PoolStatus.SWEPT, PoolStatus.CLOSED_THROUGH} == TERMINAL_POOL_STATUSES


def test_liquidity_has_no_configurable_status() -> None:
    """§8. Only ACTIVE can be a reference, so there is nothing to choose."""
    fields = set(CandidateEligibilityConfig.__dataclass_fields__)

    assert fields == {"allowed_order_block_statuses", "allowed_fvg_statuses"}


# --------------------------------------------------------------------------
# §12-§14: role and side
# --------------------------------------------------------------------------


def test_the_role_mapping() -> None:
    """§12. Liquidity is never an entry zone."""
    assert role_for(block_source("3990", "4010")) is CandidateRole.ENTRY_ZONE
    assert role_for(gap_source("3990", "4010")) is CandidateRole.ENTRY_ZONE
    assert (
        role_for(pool_source("4050", "4050", LiquiditySide.BUY_SIDE))
        is CandidateRole.UPPER_REFERENCE
    )
    assert (
        role_for(pool_source("3950", "3950", LiquiditySide.SELL_SIDE))
        is CandidateRole.LOWER_REFERENCE
    )


def test_the_entry_side_mapping() -> None:
    """§13."""
    assert entry_side_for(BreakDirection.BULLISH) is EntrySide.BAI
    assert entry_side_for(BreakDirection.BEARISH) is EntrySide.SEO
    assert entry_side_for(GapDirection.BULLISH) is EntrySide.BAI
    assert entry_side_for(GapDirection.BEARISH) is EntrySide.SEO


def test_liquidity_never_receives_an_entry_side() -> None:
    """§14. Buy-side liquidity is not a sell entry, and is not called SEO."""
    for side in LiquiditySide:
        decision = decide(pool_source("4050", "4050", side))

        assert decision.entry_side is None  # type: ignore[attr-defined]
        assert decision.role is not CandidateRole.ENTRY_ZONE  # type: ignore[attr-defined]


def test_there_are_exactly_three_roles_two_sides_and_three_relations() -> None:
    assert [member.value for member in CandidateRole] == [
        "ENTRY_ZONE",
        "UPPER_REFERENCE",
        "LOWER_REFERENCE",
    ]
    assert [member.value for member in EntrySide] == ["BAI", "SEO"]
    assert [member.value for member in MarketRelation] == ["BELOW", "OVERLAPS", "ABOVE"]


# --------------------------------------------------------------------------
# §17-§18, §44: market relation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        ("3900", "3950", MarketRelation.BELOW),
        ("3950", "4050", MarketRelation.OVERLAPS),
        ("4050", "4100", MarketRelation.ABOVE),
        ("3900", "4000", MarketRelation.OVERLAPS),
        ("4000", "4100", MarketRelation.OVERLAPS),
        ("4000", "4000", MarketRelation.OVERLAPS),
        ("3999.99", "3999.99", MarketRelation.BELOW),
        ("4000.01", "4000.01", MarketRelation.ABOVE),
    ],
)
def test_the_zone_relation_matrix(lower: str, upper: str, expected: MarketRelation) -> None:
    """§17, §44. Exact boundaries, no epsilon."""
    assert relation_of_zone(Decimal(lower), Decimal(upper), PRICE) is expected


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("3950", MarketRelation.BELOW),
        ("4000", MarketRelation.OVERLAPS),
        ("4050", MarketRelation.ABOVE),
        ("3999.99", MarketRelation.BELOW),
        ("4000.01", MarketRelation.ABOVE),
    ],
)
def test_the_level_relation_matrix(level: str, expected: MarketRelation) -> None:
    """§18."""
    assert relation_of_level(Decimal(level), PRICE) is expected


# --------------------------------------------------------------------------
# §24: distance
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        ("3900", "3950", "50"),
        ("3950", "4050", "0"),
        ("4050", "4100", "50"),
        ("3900", "4000", "0"),
        ("4000", "4100", "0"),
        ("3999.90", "3999.95", "0.05"),
    ],
)
def test_the_zone_distance_matrix(lower: str, upper: str, expected: str) -> None:
    """§24. Non-negative, exact, and zero whenever price is inside."""
    got = distance_of_zone(Decimal(lower), Decimal(upper), PRICE)

    assert got == Decimal(expected)
    assert got >= 0
    assert isinstance(got, Decimal)


def test_the_reference_level_distance_is_absolute() -> None:
    """§24."""
    above = decide(pool_source("4050", "4050", LiquiditySide.BUY_SIDE))
    below = decide(pool_source("3950", "3950", LiquiditySide.SELL_SIDE))

    assert above.distance_to_reference == Decimal("50")  # type: ignore[attr-defined]
    assert below.distance_to_reference == Decimal("50")  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# §19-§21, §43: entry-zone location
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lower", "upper", "direction", "relation", "eligible"),
    [
        ("3900", "3950", BreakDirection.BULLISH, MarketRelation.BELOW, True),
        ("3950", "4050", BreakDirection.BULLISH, MarketRelation.OVERLAPS, True),
        ("4050", "4100", BreakDirection.BULLISH, MarketRelation.ABOVE, False),
        ("4050", "4100", BreakDirection.BEARISH, MarketRelation.ABOVE, True),
        ("3950", "4050", BreakDirection.BEARISH, MarketRelation.OVERLAPS, True),
        ("3900", "3950", BreakDirection.BEARISH, MarketRelation.BELOW, False),
    ],
)
def test_the_entry_location_matrix(
    lower: str,
    upper: str,
    direction: BreakDirection,
    relation: MarketRelation,
    eligible: bool,
) -> None:
    """§43. Reference 4000, both directions, all three relations."""
    decision = decide(block_source(lower, upper, direction))

    assert decision.market_relation is relation  # type: ignore[attr-defined]
    assert decision.eligible is eligible  # type: ignore[attr-defined]
    assert decision.entry_side is entry_side_for(direction)  # type: ignore[attr-defined]
    if not eligible:
        assert decision.reasons == (EligibilityReason.ENTRY_ZONE_WRONG_SIDE,)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("lower", "upper"),
    [("3900", "4000"), ("4000", "4100"), ("4000", "4000")],
)
def test_a_zone_touching_the_reference_exactly_overlaps_and_survives(
    lower: str, upper: str
) -> None:
    """§21, §44. No epsilon, and an already-entered zone is not pretended away."""
    for direction in BreakDirection:
        decision = decide(block_source(lower, upper, direction))

        assert decision.market_relation is MarketRelation.OVERLAPS  # type: ignore[attr-defined]
        assert decision.distance_to_reference == 0  # type: ignore[attr-defined]
        assert decision.eligible is True  # type: ignore[attr-defined]


def test_a_wrong_side_zone_keeps_its_own_direction() -> None:
    """§19, §20. It is not re-read as the other side."""
    bullish_above = decide(block_source("4050", "4100", BreakDirection.BULLISH))
    bearish_below = decide(block_source("3900", "3950", BreakDirection.BEARISH))

    assert bullish_above.entry_side is EntrySide.BAI  # type: ignore[attr-defined]
    assert bearish_below.entry_side is EntrySide.SEO  # type: ignore[attr-defined]
    assert EligibilityReason.ENTRY_ZONE_WRONG_SIDE in bullish_above.reasons  # type: ignore[attr-defined]
    assert EligibilityReason.ENTRY_ZONE_WRONG_SIDE in bearish_below.reasons  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# §16, §22, §45-§46: liquidity references
# --------------------------------------------------------------------------


def test_the_reference_level_is_the_far_edge_not_the_midpoint() -> None:
    """§16."""
    buy = pool_source("4040", "4060", LiquiditySide.BUY_SIDE)
    sell = pool_source("3940", "3960", LiquiditySide.SELL_SIDE)

    assert reference_level_for(buy) == Decimal("4060")
    assert reference_level_for(sell) == Decimal("3940")
    assert reference_level_for(buy) != buy.midpoint


@pytest.mark.parametrize(
    ("upper", "eligible", "distance"),
    [("4050", True, "50"), ("4000", False, "0"), ("3950", False, "50")],
)
def test_the_buy_side_reference_matrix(upper: str, eligible: bool, distance: str) -> None:
    """§22, §45. Only a level ahead of price is a reference; equality is not."""
    decision = decide(pool_source(upper, upper, LiquiditySide.BUY_SIDE))

    assert decision.role is CandidateRole.UPPER_REFERENCE  # type: ignore[attr-defined]
    assert decision.reference_level == Decimal(upper)  # type: ignore[attr-defined]
    assert decision.eligible is eligible  # type: ignore[attr-defined]
    assert decision.distance_to_reference == Decimal(distance)  # type: ignore[attr-defined]
    if not eligible:
        assert decision.reasons == (EligibilityReason.REFERENCE_NOT_AHEAD,)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("lower", "eligible", "distance"),
    [("3950", True, "50"), ("4000", False, "0"), ("4050", False, "50")],
)
def test_the_sell_side_reference_matrix(lower: str, eligible: bool, distance: str) -> None:
    """§22, §45."""
    decision = decide(pool_source(lower, lower, LiquiditySide.SELL_SIDE))

    assert decision.role is CandidateRole.LOWER_REFERENCE  # type: ignore[attr-defined]
    assert decision.reference_level == Decimal(lower)  # type: ignore[attr-defined]
    assert decision.eligible is eligible  # type: ignore[attr-defined]
    if not eligible:
        assert decision.reasons == (EligibilityReason.REFERENCE_NOT_AHEAD,)  # type: ignore[attr-defined]


@pytest.mark.parametrize("status", [PoolStatus.SWEPT, PoolStatus.CLOSED_THROUGH])
def test_a_terminal_pool_is_always_ineligible(status: PoolStatus) -> None:
    """§23. Regardless of where it sits."""
    ahead = decide(pool_source("4050", "4050", LiquiditySide.BUY_SIDE, status))
    behind = decide(pool_source("3950", "3950", LiquiditySide.BUY_SIDE, status))

    assert ahead.eligible is False  # type: ignore[attr-defined]
    assert ahead.reasons == (EligibilityReason.LIQUIDITY_TERMINAL,)  # type: ignore[attr-defined]
    assert behind.reasons == (  # type: ignore[attr-defined]
        EligibilityReason.LIQUIDITY_TERMINAL,
        EligibilityReason.REFERENCE_NOT_AHEAD,
    )


def test_a_zero_width_pool_keeps_its_single_price_as_the_reference() -> None:
    """§46. No division by width, no widening."""
    buy = decide(pool_source("4050", "4050", LiquiditySide.BUY_SIDE))
    sell = decide(pool_source("3950", "3950", LiquiditySide.SELL_SIDE))

    assert buy.reference_level == Decimal("4050")  # type: ignore[attr-defined]
    assert buy.eligible is True  # type: ignore[attr-defined]
    assert sell.reference_level == Decimal("3950")  # type: ignore[attr-defined]
    assert sell.eligible is True  # type: ignore[attr-defined]
    assert buy.source.width == 0  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# §25-§26, §47: reasons
# --------------------------------------------------------------------------


def test_every_reason_code_is_distinct_and_ordered() -> None:
    """§25. Seven reasons, never collapsed into one word."""
    assert [member.value for member in EligibilityReason] == [
        "ORDER_BLOCK_INVALIDATED",
        "ORDER_BLOCK_STATUS_NOT_ALLOWED",
        "FVG_FILLED",
        "FVG_STATUS_NOT_ALLOWED",
        "LIQUIDITY_TERMINAL",
        "ENTRY_ZONE_WRONG_SIDE",
        "REFERENCE_NOT_AHEAD",
    ]
    assert tuple(EligibilityReason) == REASON_ORDER


def test_a_terminal_block_and_a_policy_exclusion_are_different_reasons() -> None:
    """§25. "Invalidated by price" and "your policy excludes it" are not the same."""
    strict = CandidateEligibilityConfig(
        allowed_order_block_statuses=frozenset({OrderBlockStatus.ACTIVE}),
        allowed_fvg_statuses=frozenset({FvgStatus.OPEN}),
    )
    invalidated = decide(
        block_source("3900", "3950", status=OrderBlockStatus.INVALIDATED), config=strict
    )
    mitigated = decide(
        block_source("3900", "3950", status=OrderBlockStatus.MITIGATED), config=strict
    )

    assert invalidated.reasons == (EligibilityReason.ORDER_BLOCK_INVALIDATED,)  # type: ignore[attr-defined]
    assert mitigated.reasons == (EligibilityReason.ORDER_BLOCK_STATUS_NOT_ALLOWED,)  # type: ignore[attr-defined]


def test_a_filled_gap_and_a_policy_exclusion_are_different_reasons() -> None:
    strict = CandidateEligibilityConfig(
        allowed_order_block_statuses=frozenset(),
        allowed_fvg_statuses=frozenset({FvgStatus.OPEN}),
    )
    filled = decide(gap_source("3900", "3950", status=FvgStatus.FILLED), config=strict)
    touched = decide(gap_source("3900", "3950", status=FvgStatus.TOUCHED), config=strict)

    assert filled.reasons == (EligibilityReason.FVG_FILLED,)  # type: ignore[attr-defined]
    assert touched.reasons == (EligibilityReason.FVG_STATUS_NOT_ALLOWED,)  # type: ignore[attr-defined]


def test_a_terminal_block_on_the_wrong_side_carries_both_reasons() -> None:
    """§26, §47. No short-circuit after the first failure."""
    decision = decide(
        block_source("4050", "4100", BreakDirection.BULLISH, OrderBlockStatus.INVALIDATED)
    )

    assert decision.reasons == (  # type: ignore[attr-defined]
        EligibilityReason.ORDER_BLOCK_INVALIDATED,
        EligibilityReason.ENTRY_ZONE_WRONG_SIDE,
    )
    assert decision.eligible is False  # type: ignore[attr-defined]


def test_a_filled_gap_on_the_wrong_side_carries_both_reasons() -> None:
    """§26, §47."""
    decision = decide(gap_source("3900", "3950", GapDirection.BEARISH, FvgStatus.FILLED))

    assert decision.reasons == (  # type: ignore[attr-defined]
        EligibilityReason.FVG_FILLED,
        EligibilityReason.ENTRY_ZONE_WRONG_SIDE,
    )


def test_a_terminal_pool_behind_price_carries_both_reasons() -> None:
    """§26, §47."""
    decision = decide(pool_source("3950", "3950", LiquiditySide.BUY_SIDE, PoolStatus.SWEPT))

    assert decision.reasons == (  # type: ignore[attr-defined]
        EligibilityReason.LIQUIDITY_TERMINAL,
        EligibilityReason.REFERENCE_NOT_AHEAD,
    )


def test_reasons_always_follow_the_declared_order() -> None:
    """§26. One fixed order, whatever sequence the checks happened to run in."""
    for decision in (
        decide(block_source("4050", "4100", BreakDirection.BULLISH, OrderBlockStatus.INVALIDATED)),
        decide(gap_source("3900", "3950", GapDirection.BEARISH, FvgStatus.FILLED)),
        decide(pool_source("3950", "3950", LiquiditySide.BUY_SIDE, PoolStatus.SWEPT)),
    ):
        indexes = [REASON_ORDER.index(reason) for reason in decision.reasons]  # type: ignore[attr-defined]
        assert indexes == sorted(indexes)


def test_eligible_is_exactly_the_absence_of_reasons() -> None:
    """§26."""
    for source in (
        block_source("3900", "3950"),
        block_source("4050", "4100"),
        gap_source("3900", "3950", GapDirection.BEARISH, FvgStatus.FILLED),
        pool_source("4050", "4050"),
        pool_source("3950", "3950"),
    ):
        decision = decide(source)
        assert decision.eligible is (decision.reasons == ())  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# §27-§29: the decision model
# --------------------------------------------------------------------------


def test_an_entry_zone_carries_the_source_geometry_and_no_level() -> None:
    """§15, §29."""
    source = block_source("3900", "3950")
    decision = decide(source)

    assert (decision.lower, decision.upper) == (source.lower, source.upper)  # type: ignore[attr-defined]
    assert (decision.midpoint, decision.width) == (source.midpoint, source.width)  # type: ignore[attr-defined]
    assert decision.reference_level is None  # type: ignore[attr-defined]


def test_a_reference_carries_the_level_and_no_zone() -> None:
    """§29. Its band stays reachable through the source, but is not candidate geometry."""
    source = pool_source("4040", "4060", LiquiditySide.BUY_SIDE)
    decision = decide(source)

    assert decision.reference_level == Decimal("4060")  # type: ignore[attr-defined]
    assert (decision.lower, decision.upper, decision.midpoint, decision.width) == (  # type: ignore[attr-defined]
        None,
        None,
        None,
        None,
    )
    assert decision.source.lower == Decimal("4040")  # type: ignore[attr-defined]


def test_the_decision_keeps_the_projected_source() -> None:
    source = block_source("3900", "3950")
    decision = decide(source)

    assert decision.source is source  # type: ignore[attr-defined]
    assert decision.candidate_source_id == source.candidate_source_id  # type: ignore[attr-defined]
    assert decision.source_id == source.source_id  # type: ignore[attr-defined]
    assert decision.method_version == CANDIDATE_ELIGIBILITY_METHOD_VERSION  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# §28: candidate identity
# --------------------------------------------------------------------------


def test_the_candidate_identity_survives_a_policy_change() -> None:
    """§28. The decision changes; the candidate does not."""
    source = block_source("3900", "3950", status=OrderBlockStatus.MITIGATED)
    permissive = decide(source)
    strict = decide(
        source,
        config=CandidateEligibilityConfig(
            allowed_order_block_statuses=frozenset({OrderBlockStatus.ACTIVE}),
            allowed_fvg_statuses=frozenset({FvgStatus.OPEN}),
        ),
    )

    assert permissive.candidate_id == strict.candidate_id  # type: ignore[attr-defined]
    assert permissive.eligible is not strict.eligible  # type: ignore[attr-defined]


def test_the_candidate_identity_survives_a_price_change() -> None:
    """§28. No price, no distance and no relation enter the preimage."""
    source = block_source("3900", "3950")

    assert decide(source, price="4000").candidate_id == (  # type: ignore[attr-defined]
        decide(source, price="5000").candidate_id  # type: ignore[attr-defined]
    )


def test_no_price_or_verdict_enters_the_candidate_identity() -> None:
    """§28."""
    import inspect

    from goldpipeline.services.ict_candidate_eligibility import _candidate_id

    source = inspect.getsource(_candidate_id)
    joined = source.split('"|".join(')[1]

    for forbidden in ("price", "status", "distance", "reason", "eligible", "relation"):
        assert forbidden not in joined


def test_two_roles_of_one_source_would_not_collide() -> None:
    """§28. Role is in the preimage."""
    from goldpipeline.services.ict_candidate_eligibility import _candidate_id

    identities = {
        _candidate_id(candidate_source_id="same", role=role, entry_side=None)
        for role in CandidateRole
    }

    assert len(identities) == len(CandidateRole)


def test_two_entry_sides_of_one_source_would_not_collide() -> None:
    from goldpipeline.services.ict_candidate_eligibility import _candidate_id

    identities = {
        _candidate_id(candidate_source_id="same", role=CandidateRole.ENTRY_ZONE, entry_side=side)
        for side in (None, EntrySide.BAI, EntrySide.SEO)
    }

    assert len(identities) == 3


# --------------------------------------------------------------------------
# §57: the precomputed projection seam
# --------------------------------------------------------------------------


def test_a_supplied_projection_gives_the_same_answer() -> None:
    """§57."""
    from goldpipeline.services.ict_candidate_source import project_candidate_sources
    from tests.test_ict_candidate_eligibility_fixture import staggered_snapshot

    composite = analyse_ict_composite(staggered_snapshot(), config=composite_config())
    supplied = project_candidate_sources(composite)

    assert analyse_candidate_eligibility(
        composite, config=PERMISSIVE, projection=supplied
    ) == analyse_candidate_eligibility(composite, config=PERMISSIVE)


def test_a_projection_from_another_instant_is_refused() -> None:
    """§57. Refused, never trimmed."""
    from goldpipeline.services.ict_candidate_source import project_candidate_sources
    from tests.test_ict_candidate_eligibility_fixture import staggered_snapshot

    composite = analyse_ict_composite(staggered_snapshot(), config=composite_config())
    foreign = project_candidate_sources(composite)
    drifted = replace(foreign, observed_at=foreign.observed_at + HOUR)

    with pytest.raises(CandidateEligibilityError, match="supply one built from it"):
        analyse_candidate_eligibility(composite, config=PERMISSIVE, projection=drifted)


def test_a_projection_for_another_symbol_or_provider_is_refused() -> None:
    from goldpipeline.services.ict_candidate_source import project_candidate_sources
    from tests.test_ict_candidate_eligibility_fixture import staggered_snapshot

    composite = analyse_ict_composite(staggered_snapshot(), config=composite_config())
    projected = project_candidate_sources(composite)

    with pytest.raises(CandidateEligibilityError, match="is for 'EURUSD'"):
        analyse_candidate_eligibility(
            composite, config=PERMISSIVE, projection=replace(projected, symbol="EURUSD")
        )
    with pytest.raises(CandidateEligibilityError, match="is from 'metatrader'"):
        analyse_candidate_eligibility(
            composite, config=PERMISSIVE, projection=replace(projected, provider="metatrader")
        )


def test_a_projection_built_under_another_policy_is_refused() -> None:
    """§57. A projection carries the composite policy it was built from."""
    from goldpipeline.services.ict_candidate_source import project_candidate_sources
    from tests.test_ict_candidate_eligibility_fixture import staggered_snapshot

    shot = staggered_snapshot()
    composite = analyse_ict_composite(shot, config=composite_config())
    other = project_candidate_sources(analyse_ict_composite(shot, config=composite_config(atr=21)))

    with pytest.raises(CandidateEligibilityError, match="different composite policy"):
        analyse_candidate_eligibility(composite, config=PERMISSIVE, projection=other)


def test_a_composite_with_no_timeframe_has_no_reference_price() -> None:
    """A snapshot cannot be empty, so this is reachable only by construction."""
    from tests.test_ict_candidate_eligibility_fixture import staggered_snapshot

    composite = analyse_ict_composite(staggered_snapshot(), config=composite_config())
    empty = replace(composite, timeframes=())

    with pytest.raises(CandidateEligibilityError, match="no reference price"):
        resolve_reference_price(empty)
