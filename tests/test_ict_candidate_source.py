"""The projection's own contract: one fact per object, joined by identity.

Round 6.6e.2a §11-§14, §17, §31-§33. The invariants here are counting and
joining rather than geometry, because the failure this round has to prevent is
not a wrong number - it is a source quietly disappearing, which would erase the
boundary between "never existed" and "existed and was ruled out".
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import ICT_TIMEFRAMES
from goldpipeline.services.ict_candidate_source import (
    CANDIDATE_SOURCE_METHOD_VERSION,
    GEOMETRY_OF,
    CandidateGeometryKind,
    CandidateSource,
    CandidateSourceError,
    CandidateSourceKind,
    CandidateSourceProjection,
    FairValueGapEvidence,
    OrderBlockEvidence,
    project_candidate_sources,
    project_timeframe_sources,
)
from goldpipeline.services.ict_composite import IctCompositeAnalysis, analyse_ict_composite
from tests.test_ict_composite_fixture import config, divergent_snapshot

KINDS = tuple(CandidateSourceKind)


def projection(**kwargs: object) -> CandidateSourceProjection:
    return project_candidate_sources(composite(**kwargs))


def composite(**kwargs: object) -> IctCompositeAnalysis:
    return analyse_ict_composite(divergent_snapshot(), config=config(**kwargs))  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# §31: one fact per object, no filtering
# --------------------------------------------------------------------------


@pytest.mark.parametrize("timeframe", ICT_TIMEFRAMES)
def test_every_lifecycle_state_and_pool_becomes_exactly_one_fact(timeframe: Timeframe) -> None:
    """§31, and the most important test in this round.

    No missing, no extras, no filtering - on every timeframe, including the one
    with nothing at all.
    """
    entry = composite().require(timeframe)
    projected = project_timeframe_sources(entry)

    assert len(projected.of_kind(CandidateSourceKind.ORDER_BLOCK)) == len(
        entry.order_block_lifecycle.states
    )
    assert len(projected.of_kind(CandidateSourceKind.FAIR_VALUE_GAP)) == len(
        entry.fvg_lifecycle.states
    )
    assert len(projected.of_kind(CandidateSourceKind.LIQUIDITY_POOL)) == len(entry.liquidity.pools)
    assert len(projected.sources) == (
        len(entry.order_block_lifecycle.states)
        + len(entry.fvg_lifecycle.states)
        + len(entry.liquidity.pools)
    )


def test_the_counts_hold_across_the_whole_snapshot() -> None:
    """§31, summed - so a timeframe cannot lose a source another one gains."""
    full = composite()
    projected = projection()

    expected = sum(
        len(entry.order_block_lifecycle.states)
        + len(entry.fvg_lifecycle.states)
        + len(entry.liquidity.pools)
        for entry in full.timeframes
    )
    got = sum(len(entry.sources) for entry in projected.timeframes)

    assert got == expected == 27


@pytest.mark.parametrize("rule", ["TOUCH", "MIDPOINT", "FULL_ZONE"])
def test_no_lifecycle_status_is_ever_filtered_out(rule: str) -> None:
    """§9, §31. Terminal sources are facts, whatever the policy says about them."""
    from goldpipeline.services.ict_order_block_lifecycle import OrderBlockMitigationRule

    full = composite(rule=OrderBlockMitigationRule(rule))
    projected = project_candidate_sources(full)

    for entry, source_entry in zip(full.timeframes, projected.timeframes, strict=True):
        assert {s.source_id for s in source_entry.of_kind(CandidateSourceKind.ORDER_BLOCK)} == {
            state.order_block_id for state in entry.order_block_lifecycle.states
        }


# --------------------------------------------------------------------------
# §32: joined by identity only
# --------------------------------------------------------------------------


def test_every_source_id_resolves_exactly_once_to_its_original() -> None:
    """§32. Never by price, bounds, timestamp or source candle."""
    full = composite()
    projected = projection()

    for entry, source_entry in zip(full.timeframes, projected.timeframes, strict=True):
        for source in source_entry.sources:
            matches: list[object]
            if source.kind is CandidateSourceKind.ORDER_BLOCK:
                matches = [
                    b
                    for b in entry.order_blocks.order_blocks
                    if b.order_block_id == source.source_id
                ]
            elif source.kind is CandidateSourceKind.FAIR_VALUE_GAP:
                matches = [s for s in entry.fvg_lifecycle.states if s.fvg_id == source.source_id]
            else:
                matches = [p for p in entry.liquidity.pools if p.pool_id == source.source_id]

            assert len(matches) == 1, f"{source.kind.value} {source.source_id}"


def test_the_evidence_holds_the_original_object_itself() -> None:
    """§14. An audit walks from a projected fact to the exact market object."""
    full = composite()
    projected = project_candidate_sources(full)

    for entry, source_entry in zip(full.timeframes, projected.timeframes, strict=True):
        for source in source_entry.sources:
            evidence = source.evidence
            if isinstance(evidence, OrderBlockEvidence):
                assert evidence.order_block is entry.order_blocks.order_block(source.source_id)
                assert evidence.state.order_block_id == source.source_id
                assert evidence.status is evidence.state.status
            elif isinstance(evidence, FairValueGapEvidence):
                assert evidence.state is entry.fvg_lifecycle.state(source.source_id)
                assert evidence.gap is evidence.state.gap
                assert evidence.status is evidence.state.status
            else:
                assert evidence.pool is entry.liquidity.pool(source.source_id)
                assert evidence.status is evidence.pool.status
                assert evidence.tolerance == evidence.pool.tolerance


def test_a_lifecycle_state_with_no_block_fails_closed() -> None:
    """§32. A disagreement between two analyses is a contradiction, not a gap."""
    from dataclasses import replace

    entry = composite().require(Timeframe.H1)
    broken = replace(
        entry,
        order_blocks=replace(entry.order_blocks, order_blocks=entry.order_blocks.order_blocks[1:]),
    )

    with pytest.raises(CandidateSourceError, match="does not hold"):
        project_timeframe_sources(broken)


# --------------------------------------------------------------------------
# §5, §11-§12: geometry
# --------------------------------------------------------------------------


def test_the_geometry_kind_mapping_is_total_and_agrees_with_every_fact() -> None:
    """§5. One mapping, in one place."""
    assert set(GEOMETRY_OF) == set(KINDS)
    assert GEOMETRY_OF[CandidateSourceKind.ORDER_BLOCK] is CandidateGeometryKind.ZONE
    assert GEOMETRY_OF[CandidateSourceKind.FAIR_VALUE_GAP] is CandidateGeometryKind.ZONE
    assert GEOMETRY_OF[CandidateSourceKind.LIQUIDITY_POOL] is CandidateGeometryKind.BAND

    for entry in projection().timeframes:
        for source in entry.sources:
            assert source.geometry is GEOMETRY_OF[source.kind]


def test_every_fact_satisfies_the_common_geometry_contract() -> None:
    """§11. ``lower <= midpoint <= upper``, everywhere."""
    for entry in projection().timeframes:
        for source in entry.sources:
            assert source.lower <= source.midpoint <= source.upper
            assert source.width == source.upper - source.lower
            assert isinstance(source.width, Decimal)


def test_a_zone_always_has_width_and_a_band_may_not() -> None:
    """§11, §12. The invariant is applied per geometry kind, not globally."""
    zones: list[CandidateSource] = []
    bands: list[CandidateSource] = []
    for entry in projection().timeframes:
        for source in entry.sources:
            (zones if source.geometry is CandidateGeometryKind.ZONE else bands).append(source)

    assert zones and bands
    for source in zones:
        assert source.width > 0, source.source_id
    assert any(source.width == 0 for source in bands), "the fixture has an exact-equality pool"


def test_the_projected_geometry_is_the_source_geometry() -> None:
    """§6-§8. Copied, never recomputed - and cross-checked against each engine's
    own width property, which are three differently-named fields."""
    full = composite()
    projected = projection()

    for entry, source_entry in zip(full.timeframes, projected.timeframes, strict=True):
        for source in source_entry.sources:
            evidence = source.evidence
            if isinstance(evidence, OrderBlockEvidence):
                bounds = (
                    evidence.order_block.lower,
                    evidence.order_block.upper,
                    evidence.order_block.midpoint,
                )
                assert source.width == evidence.order_block.width
            elif isinstance(evidence, FairValueGapEvidence):
                bounds = (evidence.gap.lower, evidence.gap.upper, evidence.gap.midpoint)
                assert source.width == evidence.gap.size
            else:
                bounds = (
                    evidence.pool.lower,
                    evidence.pool.upper,
                    evidence.pool.midpoint,
                )
                assert source.width == evidence.pool.span
            assert (source.lower, source.upper, source.midpoint) == bounds
        assert entry.symbol == source_entry.symbol


def test_a_contradictory_geometry_is_refused() -> None:
    """§11. Checked rather than assumed, on both halves."""
    from goldpipeline.services.ict_candidate_source import _check_geometry

    _check_geometry(
        CandidateSourceKind.LIQUIDITY_POOL, Decimal("4000"), Decimal("4000"), Decimal("4000"), "ok"
    )
    with pytest.raises(CandidateSourceError, match="has no width"):
        _check_geometry(
            CandidateSourceKind.ORDER_BLOCK, Decimal("4000"), Decimal("4000"), Decimal("4000"), "z"
        )
    with pytest.raises(CandidateSourceError, match="below lower"):
        _check_geometry(
            CandidateSourceKind.ORDER_BLOCK, Decimal("4010"), Decimal("4000"), Decimal("4005"), "z"
        )
    with pytest.raises(CandidateSourceError, match="outside"):
        _check_geometry(
            CandidateSourceKind.FAIR_VALUE_GAP,
            Decimal("4000"),
            Decimal("4010"),
            Decimal("4020"),
            "z",
        )


# --------------------------------------------------------------------------
# §13-§14: identity
# --------------------------------------------------------------------------


def test_every_projected_identity_is_distinct() -> None:
    """§13. Across the whole snapshot, not merely within a timeframe."""
    identities = [
        source.candidate_source_id for entry in projection().timeframes for source in entry.sources
    ]

    assert len(identities) == len(set(identities)) == 27


def test_the_projected_identity_is_stable_and_deterministic() -> None:
    first = projection()
    second = projection()

    assert [[s.candidate_source_id for s in entry.sources] for entry in first.timeframes] == [
        [s.candidate_source_id for s in entry.sources] for entry in second.timeframes
    ]


def test_two_kinds_sharing_a_source_id_would_not_collide() -> None:
    """§13. ``kind`` is in the preimage, so a coincidence cannot merge two facts."""
    from goldpipeline.services.ict_candidate_source import _candidate_source_id

    same = "identical-source-id"
    identities = {
        _candidate_source_id(timeframe=Timeframe.H1, kind=kind, source_id=same) for kind in KINDS
    }

    assert len(identities) == len(KINDS)


def test_the_same_source_on_two_timeframes_would_not_collide() -> None:
    from goldpipeline.services.ict_candidate_source import _candidate_source_id

    identities = {
        _candidate_source_id(timeframe=tf, kind=CandidateSourceKind.ORDER_BLOCK, source_id="shared")
        for tf in ICT_TIMEFRAMES
    }

    assert len(identities) == len(ICT_TIMEFRAMES)


def test_no_price_enters_the_projected_identity() -> None:
    """§13. So this round needs no Decimal canonicalisation of its own."""
    import inspect

    from goldpipeline.services.ict_candidate_source import _candidate_source_id

    source = inspect.getsource(_candidate_source_id)
    joined = source.split('"|".join(')[1]

    assert "normalize" not in source
    for forbidden in ("lower", "upper", "midpoint", "width", "price"):
        assert forbidden not in joined


def test_the_source_id_is_carried_beside_the_projected_one() -> None:
    """§14. The projection id never replaces the market object's own."""
    for entry in projection().timeframes:
        for source in entry.sources:
            assert source.source_id
            assert source.source_id != source.candidate_source_id
            assert entry.source(source.candidate_source_id) is source


# --------------------------------------------------------------------------
# §17-§18: ordering
# --------------------------------------------------------------------------


def test_sources_are_ordered_by_formation_then_kind_then_identity() -> None:
    """§17. A total order that needs no reference price."""
    for entry in projection().timeframes:
        keys = [(s.formed_at, s.kind.value, s.source_id) for s in entry.sources]
        assert keys == sorted(keys), entry.timeframe


def test_sources_are_not_ordered_by_price() -> None:
    """§17. Price ordering needs a reference price, and there is none this round."""
    entry = projection().require(Timeframe.M15)
    lows = [source.lower for source in entry.sources]

    assert lows != sorted(lows), "the fixture would not detect price ordering otherwise"


def test_timeframes_keep_the_branch_order() -> None:
    """§18. Ordering only - nothing here makes H4 more important than M1."""
    assert [entry.timeframe for entry in projection().timeframes] == list(ICT_TIMEFRAMES)


# --------------------------------------------------------------------------
# §25, §33: provenance and immutability
# --------------------------------------------------------------------------


def test_the_composite_config_is_carried_not_rebuilt() -> None:
    """§25. No seventh policy - the six travel through as they are."""
    full = composite(left=3, right=3, atr=21)
    projected = project_candidate_sources(full)

    assert projected.config is full.config
    assert projected.config.swing_left_bars == 3
    assert projected.config.atr_period == 21


def test_provenance_is_carried_not_invented() -> None:
    full = composite()
    projected = project_candidate_sources(full)

    assert (projected.symbol, projected.provider, projected.provider_symbol) == (
        full.symbol,
        full.provider,
        full.provider_symbol,
    )
    assert projected.observed_at == full.observed_at
    for entry in projected.timeframes:
        assert entry.observed_at == full.observed_at
        assert entry.method_version == CANDIDATE_SOURCE_METHOD_VERSION


def test_the_projection_does_not_mutate_the_composite() -> None:
    """§33. Captured before and after."""
    full = composite()
    before = repr(full)

    project_candidate_sources(full)

    assert repr(full) == before


def test_every_projected_object_is_frozen() -> None:
    """§33."""
    projected = projection()
    entry = projected.require(Timeframe.H1)
    source = entry.sources[0]

    for target, field, value in (
        (projected, "observed_at", None),
        (entry, "structure_bias", None),
        (source, "lower", Decimal("1")),
        (source.evidence, "status", None),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(target, field, value)


def test_the_collections_are_immutable_tuples() -> None:
    projected = projection()

    assert isinstance(projected.timeframes, tuple)
    for entry in projected.timeframes:
        assert isinstance(entry.sources, tuple)


def test_lookups_are_by_identity_and_a_missing_timeframe_fails_closed() -> None:
    projected = projection()

    for entry in projected.timeframes:
        assert projected.timeframe(entry.timeframe) is entry
        assert projected.require(entry.timeframe) is entry
    assert projected.require(Timeframe.M5).sources == ()


def test_a_projection_of_a_subset_composite_holds_only_that_subset() -> None:
    from goldpipeline.schemas.ict import IctMarketSnapshot

    shot = divergent_snapshot()
    subset = IctMarketSnapshot(
        observed_at=shot.observed_at,
        symbol=shot.symbol,
        provider=shot.provider,
        provider_symbol=shot.provider_symbol,
        timeframes=(shot.require(Timeframe.M15),),
    )
    projected = project_candidate_sources(analyse_ict_composite(subset, config=config()))

    assert [entry.timeframe for entry in projected.timeframes] == [Timeframe.M15]
    assert projected.timeframe(Timeframe.H1) is None
    with pytest.raises(CandidateSourceError, match="holds no H1 sources"):
        projected.require(Timeframe.H1)
