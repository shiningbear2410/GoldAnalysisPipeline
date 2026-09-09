"""Five timeframes, one price, twenty-seven decisions, pinned exactly.

Round 6.6e.2b §33-§34, §37, §40-§41, §48-§56, §63. Every number below was read
out of the engine before it was written down.

The snapshot is Round 6.6e.1's five paths with **M1 running three minutes
past the rest**. That stagger is the point: with every timeframe ending at one
instant the five closes disagree - 4000, 4060 and 4043 - and reference-price
resolution correctly refuses the whole thing. Staggering gives the global latest
close a single honest witness, and simultaneously satisfies §37's requirement
that the five last closes not all share a timestamp.

The unstaggered snapshot is not discarded; it is Round 6.6e.2b's own §38
conflict fixture, tested in ``test_ict_candidate_eligibility.py``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import ICT_TIMEFRAMES, IctMarketSnapshot
from goldpipeline.services.ict_candidate_eligibility import (
    CandidateEligibilityAnalysis,
    CandidateEligibilityConfig,
    CandidateEligibilityTimeframe,
    EligibilityReason,
    analyse_candidate_eligibility,
)
from goldpipeline.services.ict_candidate_source import CandidateSourceKind
from goldpipeline.services.ict_composite import analyse_ict_composite
from goldpipeline.services.ict_fvg import FvgStatus
from goldpipeline.services.ict_order_block import OrderBlockZoneBasis
from goldpipeline.services.ict_order_block_lifecycle import (
    OrderBlockMitigationRule,
    OrderBlockStatus,
)
from tests.test_ict_composite_fixture import OBSERVED_AT, PATHS, TF_OPENS, config
from tests.test_ict_order_block import reopen
from tests.test_ict_range_fixture import QUIET, ending_at
from tests.test_ict_structure import Row

MINUTE = timedelta(minutes=1)
SNAP_AT = OBSERVED_AT + MINUTE * 3

ENDS: dict[Timeframe, object] = dict.fromkeys(ICT_TIMEFRAMES, OBSERVED_AT)
ENDS[Timeframe.M1] = SNAP_AT
"""M1 runs three minutes past everything else - see the module docstring."""


def staggered_snapshot(paths: Mapping[Timeframe, Sequence[Row]] | None = None) -> IctMarketSnapshot:
    chosen = PATHS if paths is None else paths
    return IctMarketSnapshot(
        observed_at=SNAP_AT,
        symbol="XAUUSD",
        provider="fixture",
        provider_symbol=None,
        timeframes=tuple(
            reopen(
                ending_at(tf, chosen[tf], end=ENDS[tf]),  # type: ignore[arg-type]
                TF_OPENS.get(tf, {}),
            )
            for tf in ICT_TIMEFRAMES
        ),
    )


def eligibility_config(
    *,
    blocks: frozenset[OrderBlockStatus] | None = None,
    gaps: frozenset[FvgStatus] | None = None,
) -> CandidateEligibilityConfig:
    """A fully-stated policy. The defaults live here, in a test, never in ``src``."""
    return CandidateEligibilityConfig(
        allowed_order_block_statuses=(
            frozenset({OrderBlockStatus.ACTIVE, OrderBlockStatus.TOUCHED})
            if blocks is None
            else blocks
        ),
        allowed_fvg_statuses=(
            frozenset({FvgStatus.OPEN, FvgStatus.TOUCHED}) if gaps is None else gaps
        ),
    )


def analysis(
    *,
    blocks: frozenset[OrderBlockStatus] | None = None,
    gaps: frozenset[FvgStatus] | None = None,
    **composite_kwargs: object,
) -> CandidateEligibilityAnalysis:
    return analyse_candidate_eligibility(
        analyse_ict_composite(staggered_snapshot(), config=config(**composite_kwargs)),  # type: ignore[arg-type]
        config=eligibility_config(blocks=blocks, gaps=gaps),
    )


def reasons_of(entry: CandidateEligibilityTimeframe) -> dict[str, int]:
    return dict(Counter(reason.value for d in entry.decisions for reason in d.reasons))


# --------------------------------------------------------------------------
# §37: one global reference price
# --------------------------------------------------------------------------


def test_the_five_timeframes_do_not_all_close_at_one_instant() -> None:
    """§37. Which is what makes a *global* reference price a real choice."""
    composite = analyse_ict_composite(staggered_snapshot(), config=config())
    closes = {entry.timeframe: entry.series.latest_closed_at for entry in composite.timeframes}

    assert len(set(closes.values())) == 2
    assert closes[Timeframe.M1] == SNAP_AT
    assert all(value == OBSERVED_AT for tf, value in closes.items() if tf is not Timeframe.M1)


def test_the_reference_price_is_pinned() -> None:
    reference = analysis().reference_price

    assert reference.price == Decimal("4043")
    assert reference.bar_close_time == SNAP_AT
    assert reference.observed_at == SNAP_AT
    assert [witness.timeframe.value for witness in reference.witnesses] == ["M1"]


def test_every_decision_on_every_timeframe_carries_the_same_reference() -> None:
    """§37. No per-timeframe current price anywhere."""
    result = analysis()

    prices = {
        decision.reference_price for entry in result.timeframes for decision in entry.decisions
    }
    assert prices == {Decimal("4043")}
    assert result.reference_price.price == Decimal("4043")


def test_the_h4_decisions_are_measured_against_a_newer_price_than_h4_has() -> None:
    """§7. The global price may be newer than a timeframe's own last close.

    That decides location; it does not reach back and rewrite H4's lifecycle.
    """
    composite = analyse_ict_composite(staggered_snapshot(), config=config())
    result = analyse_candidate_eligibility(composite, config=eligibility_config())
    h4 = composite.require(Timeframe.H4)

    assert h4.series.latest_closed_at < result.reference_price.bar_close_time
    assert result.require(Timeframe.H4).decisions[0].reference_price == Decimal("4043")
    assert h4.order_block_lifecycle.states[0].status is OrderBlockStatus.MITIGATED


# --------------------------------------------------------------------------
# §48: the whole decision table
# --------------------------------------------------------------------------


def test_the_decision_counts_are_pinned() -> None:
    assert [
        (entry.timeframe.value, len(entry.decisions), len(entry.eligible))
        for entry in analysis().timeframes
    ] == [("H4", 1, 0), ("H1", 8, 2), ("M15", 5, 0), ("M5", 0, 0), ("M1", 13, 5)]


def test_the_roles_are_pinned() -> None:
    assert [
        (entry.timeframe.value, dict(Counter(d.role.value for d in entry.decisions)))
        for entry in analysis().timeframes
    ] == [
        ("H4", {"ENTRY_ZONE": 1}),
        ("H1", {"ENTRY_ZONE": 8}),
        ("M15", {"UPPER_REFERENCE": 3, "LOWER_REFERENCE": 2}),
        ("M5", {}),
        ("M1", {"ENTRY_ZONE": 13}),
    ]


def test_the_entry_sides_are_pinned() -> None:
    assert [
        (
            entry.timeframe.value,
            dict(Counter(d.entry_side.value for d in entry.decisions if d.entry_side)),
        )
        for entry in analysis().timeframes
    ] == [
        ("H4", {"SEO": 1}),
        ("H1", {"BAI": 4, "SEO": 4}),
        ("M15", {}),
        ("M5", {}),
        ("M1", {"BAI": 8, "SEO": 5}),
    ]


def test_the_market_relations_are_pinned() -> None:
    assert [
        (entry.timeframe.value, dict(Counter(d.market_relation.value for d in entry.decisions)))
        for entry in analysis().timeframes
    ] == [
        ("H4", {"BELOW": 1}),
        ("H1", {"BELOW": 3, "OVERLAPS": 5}),
        ("M15", {"ABOVE": 1, "BELOW": 4}),
        ("M5", {}),
        ("M1", {"BELOW": 9, "OVERLAPS": 2, "ABOVE": 2}),
    ]


def test_the_reason_distribution_is_pinned() -> None:
    assert [(entry.timeframe.value, reasons_of(entry)) for entry in analysis().timeframes] == [
        ("H4", {"ORDER_BLOCK_STATUS_NOT_ALLOWED": 1, "ENTRY_ZONE_WRONG_SIDE": 1}),
        (
            "H1",
            {
                "ORDER_BLOCK_INVALIDATED": 3,
                "ORDER_BLOCK_STATUS_NOT_ALLOWED": 2,
                "ENTRY_ZONE_WRONG_SIDE": 1,
            },
        ),
        ("M15", {"LIQUIDITY_TERMINAL": 5, "REFERENCE_NOT_AHEAD": 2}),
        ("M5", {}),
        ("M1", {"FVG_FILLED": 6, "ENTRY_ZONE_WRONG_SIDE": 3}),
    ]


def test_the_h1_decisions_are_pinned_in_full() -> None:
    assert [
        (
            d.role.value,
            d.entry_side.value if d.entry_side is not None else None,
            d.market_relation.value,
            str(d.distance_to_reference),
            d.eligible,
            tuple(reason.value for reason in d.reasons),
        )
        for d in analysis().require(Timeframe.H1).decisions
    ] == [
        ("ENTRY_ZONE", "BAI", "BELOW", "33", False, ("ORDER_BLOCK_INVALIDATED",)),
        ("ENTRY_ZONE", "BAI", "BELOW", "33", False, ("ORDER_BLOCK_INVALIDATED",)),
        ("ENTRY_ZONE", "SEO", "OVERLAPS", "0", False, ("ORDER_BLOCK_STATUS_NOT_ALLOWED",)),
        ("ENTRY_ZONE", "SEO", "OVERLAPS", "0", False, ("ORDER_BLOCK_STATUS_NOT_ALLOWED",)),
        ("ENTRY_ZONE", "SEO", "OVERLAPS", "0", False, ("ORDER_BLOCK_INVALIDATED",)),
        ("ENTRY_ZONE", "SEO", "BELOW", "53", False, ("ENTRY_ZONE_WRONG_SIDE",)),
        ("ENTRY_ZONE", "BAI", "OVERLAPS", "0", True, ()),
        ("ENTRY_ZONE", "BAI", "OVERLAPS", "0", True, ()),
    ]


def test_the_m15_reference_decisions_are_pinned_in_full() -> None:
    assert [
        (
            d.role.value,
            str(d.reference_level),
            d.market_relation.value,
            str(d.distance_to_reference),
            tuple(reason.value for reason in d.reasons),
        )
        for d in analysis().require(Timeframe.M15).decisions
    ] == [
        ("UPPER_REFERENCE", "4060", "ABOVE", "17", ("LIQUIDITY_TERMINAL",)),
        (
            "UPPER_REFERENCE",
            "4030.30",
            "BELOW",
            "12.70",
            ("LIQUIDITY_TERMINAL", "REFERENCE_NOT_AHEAD"),
        ),
        (
            "UPPER_REFERENCE",
            "4030.00",
            "BELOW",
            "13.00",
            ("LIQUIDITY_TERMINAL", "REFERENCE_NOT_AHEAD"),
        ),
        ("LOWER_REFERENCE", "3950", "BELOW", "93", ("LIQUIDITY_TERMINAL",)),
        ("LOWER_REFERENCE", "3974.70", "BELOW", "68.30", ("LIQUIDITY_TERMINAL",)),
    ]


def test_the_m1_eligible_candidates_are_pinned() -> None:
    """§48. Both sides, three relations, and a wrong-side pair beside them."""
    entry = analysis().require(Timeframe.M1)

    assert [
        (
            None if d.entry_side is None else d.entry_side.value,
            d.market_relation.value,
            str(d.distance_to_reference),
        )
        for d in entry.eligible
    ] == [
        ("BAI", "BELOW", "18"),
        ("BAI", "OVERLAPS", "0"),
        ("BAI", "OVERLAPS", "0"),
        ("SEO", "ABOVE", "2"),
        ("SEO", "ABOVE", "3"),
    ]


def test_a_timeframe_with_no_sources_produces_no_decisions() -> None:
    """§48. Still a legitimate reading, and still carries its context."""
    entry = analysis().require(Timeframe.M5)

    assert entry.decisions == ()
    assert entry.eligible == ()
    assert entry.structure_bias.value == "NEUTRAL"


def test_no_liquidity_reference_is_eligible_in_this_fixture() -> None:
    """§48 allows this, and it is worth saying out loud.

    Every M15 pool is terminal, so the eligible UPPER/LOWER_REFERENCE cases are
    proved on constructed pools in ``test_ict_candidate_eligibility.py`` rather
    than pretended into this path.
    """
    entry = analysis().require(Timeframe.M15)

    assert len(entry.decisions) == 5
    assert entry.eligible == ()
    assert all(EligibilityReason.LIQUIDITY_TERMINAL in d.reasons for d in entry.decisions)


# --------------------------------------------------------------------------
# §30-§31, §49-§50: nothing lost, nothing merged
# --------------------------------------------------------------------------


def test_there_is_exactly_one_decision_per_projected_source() -> None:
    """§30, and the most important invariant in this round."""
    from goldpipeline.services.ict_candidate_source import project_candidate_sources

    composite = analyse_ict_composite(staggered_snapshot(), config=config())
    projection = project_candidate_sources(composite)
    result = analyse_candidate_eligibility(composite, config=eligibility_config())

    for projected, decided in zip(projection.timeframes, result.timeframes, strict=True):
        assert len(decided.decisions) == len(projected.sources)
        assert [d.candidate_source_id for d in decided.decisions] == [
            s.candidate_source_id for s in projected.sources
        ]
    assert sum(len(e.decisions) for e in result.timeframes) == 27


def test_the_eligible_subset_is_a_view_of_the_decisions() -> None:
    """§31. Never a second, independently computed collection."""
    for entry in analysis().timeframes:
        assert list(entry.eligible) == [d for d in entry.decisions if d.eligible]
        for decision in entry.eligible:
            assert decision in entry.decisions


def test_every_terminal_source_still_has_a_decision() -> None:
    """§49. Present, ineligible, and naming its terminal reason."""
    result = analysis()
    terminal = [
        (entry.timeframe.value, decision)
        for entry in result.timeframes
        for decision in entry.decisions
        if {
            EligibilityReason.ORDER_BLOCK_INVALIDATED,
            EligibilityReason.FVG_FILLED,
            EligibilityReason.LIQUIDITY_TERMINAL,
        }
        & set(decision.reasons)
    ]

    assert len(terminal) == 14
    assert {name for name, _ in terminal} == {"H1", "M15", "M1"}
    for _, decision in terminal:
        assert decision.eligible is False
        assert decision.source_id
        assert decision.candidate_id


def test_two_order_blocks_with_one_zone_keep_two_decisions() -> None:
    """§35, §50. Same role, same side, same relation, same distance - still two."""
    entry = analysis().require(Timeframe.H1)
    shared = [
        d
        for d in entry.decisions
        if d.source.kind is CandidateSourceKind.ORDER_BLOCK
        and (d.lower, d.upper) == (Decimal("3990"), Decimal("4085"))
    ]

    assert len(shared) == 2
    first, second = shared
    assert (first.role, first.entry_side) == (second.role, second.entry_side)
    assert first.market_relation is second.market_relation
    assert first.distance_to_reference == second.distance_to_reference
    assert first.candidate_id != second.candidate_id
    assert first.source_id != second.source_id


def test_two_gaps_with_one_band_keep_two_decisions() -> None:
    """§35, §50."""
    entry = analysis().require(Timeframe.H1)
    shared = [
        d
        for d in entry.decisions
        if d.source.kind is CandidateSourceKind.FAIR_VALUE_GAP
        and (d.lower, d.upper) == (Decimal("3960"), Decimal("4050"))
    ]

    assert len(shared) == 2
    assert shared[0].candidate_id != shared[1].candidate_id
    assert shared[0].eligible is shared[1].eligible is True


def test_every_candidate_identity_is_distinct() -> None:
    identities = [d.candidate_id for e in analysis().timeframes for d in e.decisions]

    assert len(identities) == len(set(identities)) == 27


# --------------------------------------------------------------------------
# §33-§34: range context is metadata
# --------------------------------------------------------------------------


def test_the_range_context_is_carried_from_the_projection() -> None:
    """§33."""
    from goldpipeline.services.ict_candidate_source import project_candidate_sources

    composite = analyse_ict_composite(staggered_snapshot(), config=config())
    projection = project_candidate_sources(composite)
    result = analyse_candidate_eligibility(composite, config=eligibility_config())

    for projected, decided in zip(projection.timeframes, result.timeframes, strict=True):
        assert decided.active_dealing_range is projected.active_dealing_range
        assert decided.structure_bias is projected.structure_bias


def test_no_reason_code_mentions_the_range() -> None:
    """§33, §34, §64. Premium, discount and outside-range are not reasons."""
    values = {member.value for member in EligibilityReason}

    assert not any("RANGE" in value for value in values)
    assert not any("PREMIUM" in value or "DISCOUNT" in value for value in values)


def test_a_bearish_source_in_discount_is_still_evaluated_normally() -> None:
    """§33. Its verdict comes from status and side, never from where the range puts it."""
    from goldpipeline.services.ict_range import PriceLocation

    entry = analysis().require(Timeframe.H1)
    current = entry.active_dealing_range
    assert current is not None

    in_discount = [
        d
        for d in entry.decisions
        if current.locate(d.midpoint or Decimal(0)) is PriceLocation.DISCOUNT
    ]
    assert len(in_discount) == 8, "every H1 source sits in the discount half"
    assert any(d.eligible for d in in_discount), "and two of them are eligible anyway"


def test_a_timeframe_with_no_active_range_still_produces_decisions() -> None:
    """§33."""
    result = analysis()
    rangeless = [entry for entry in result.timeframes if entry.active_dealing_range is None]

    assert [entry.timeframe.value for entry in rangeless] == ["M15", "M5", "M1"]
    assert sum(len(entry.decisions) for entry in rangeless) == 18


def test_the_reference_price_may_sit_outside_an_active_range_without_consequence() -> None:
    """§34. No PRICE_OUTSIDE_RANGE reason exists, and none is needed."""
    from goldpipeline.services.ict_range import PriceLocation

    result = analysis()
    h4 = result.require(Timeframe.H4)
    current = h4.active_dealing_range
    assert current is not None

    assert current.locate(result.reference_price.price) is PriceLocation.ABOVE_RANGE
    assert len(h4.decisions) == 1, "and the timeframe still decides normally"


# --------------------------------------------------------------------------
# §40-§41, §51: status policy matrices
# --------------------------------------------------------------------------


def stable(result: CandidateEligibilityAnalysis) -> list[tuple[object, ...]]:
    """Everything a status-policy change is being asked not to touch."""
    return [
        (
            entry.timeframe,
            d.candidate_id,
            d.candidate_source_id,
            d.source_id,
            d.role,
            d.entry_side,
            d.reference_price,
            d.market_relation,
            d.distance_to_reference,
            d.lower,
            d.upper,
            d.reference_level,
        )
        for entry in result.timeframes
        for d in entry.decisions
    ]


@pytest.mark.parametrize(
    ("allowed", "h4_eligible", "h1_eligible"),
    [
        (frozenset({OrderBlockStatus.ACTIVE}), 0, 2),
        (frozenset({OrderBlockStatus.ACTIVE, OrderBlockStatus.TOUCHED}), 0, 2),
        (
            frozenset(
                {OrderBlockStatus.ACTIVE, OrderBlockStatus.TOUCHED, OrderBlockStatus.MITIGATED}
            ),
            0,
            4,
        ),
        (frozenset({OrderBlockStatus.MITIGATED}), 0, 4),
        (frozenset(), 0, 2),
    ],
)
def test_the_order_block_status_matrix(
    allowed: frozenset[OrderBlockStatus], h4_eligible: int, h1_eligible: int
) -> None:
    """§40. Five policies over one market.

    H4 stays at zero under every one of them, which is the point rather than a
    disappointment: its only block is *also* on the wrong side of price, so
    widening the status policy removes one reason and leaves the other standing.
    """
    result = analysis(blocks=allowed)

    assert len(result.require(Timeframe.H4).eligible) == h4_eligible
    assert len(result.require(Timeframe.H1).eligible) == h1_eligible


def test_allowing_mitigated_blocks_removes_one_h4_reason_but_not_the_other() -> None:
    """§26, §40. The multi-reason case, seen through a policy change."""
    strict = analysis(blocks=frozenset({OrderBlockStatus.ACTIVE}))
    wide = analysis(
        blocks=frozenset(
            {OrderBlockStatus.ACTIVE, OrderBlockStatus.TOUCHED, OrderBlockStatus.MITIGATED}
        )
    )

    assert strict.require(Timeframe.H4).decisions[0].reasons == (
        EligibilityReason.ORDER_BLOCK_STATUS_NOT_ALLOWED,
        EligibilityReason.ENTRY_ZONE_WRONG_SIDE,
    )
    assert wide.require(Timeframe.H4).decisions[0].reasons == (
        EligibilityReason.ENTRY_ZONE_WRONG_SIDE,
    )
    assert wide.require(Timeframe.H4).decisions[0].eligible is False


@pytest.mark.parametrize(
    ("allowed", "m1_eligible"),
    [
        (frozenset({FvgStatus.OPEN}), 2),
        (frozenset({FvgStatus.OPEN, FvgStatus.TOUCHED}), 5),
        (frozenset({FvgStatus.TOUCHED}), 3),
        (frozenset(), 0),
    ],
)
def test_the_fvg_status_matrix(allowed: frozenset[FvgStatus], m1_eligible: int) -> None:
    """§41. Four policies, including the empty one."""
    assert len(analysis(gaps=allowed).require(Timeframe.M1).eligible) == m1_eligible


def test_a_restrictive_gap_policy_produces_the_status_reason() -> None:
    """§41. ``FVG_STATUS_NOT_ALLOWED``, which the permissive policy never shows."""
    strict = analysis(gaps=frozenset({FvgStatus.OPEN}))
    entry = strict.require(Timeframe.M1)

    assert reasons_of(entry)["FVG_STATUS_NOT_ALLOWED"] == 4
    assert EligibilityReason.FVG_STATUS_NOT_ALLOWED not in {
        r for e in analysis().timeframes for d in e.decisions for r in d.reasons
    }


def test_changing_only_the_status_policy_moves_only_the_verdicts() -> None:
    """§51. Identities, geometry, relation, distance and reference all hold still."""
    permissive = analysis(
        blocks=frozenset(
            {OrderBlockStatus.ACTIVE, OrderBlockStatus.TOUCHED, OrderBlockStatus.MITIGATED}
        ),
        gaps=frozenset({FvgStatus.OPEN, FvgStatus.TOUCHED}),
    )
    strict = analysis(blocks=frozenset({OrderBlockStatus.ACTIVE}), gaps=frozenset({FvgStatus.OPEN}))

    assert stable(permissive) == stable(strict)
    assert permissive.reference_price == strict.reference_price
    moved = [
        (e.timeframe.value, d.candidate_id)
        for e, o in zip(permissive.timeframes, strict.timeframes, strict=True)
        for d, other in zip(e.decisions, o.decisions, strict=True)
        if d.eligible != other.eligible
    ]
    assert moved, "a policy change that moved nothing would prove nothing"


# --------------------------------------------------------------------------
# §52-§56: composite policy blast radius
# --------------------------------------------------------------------------


def test_the_atr_period_changes_nothing_but_the_carried_config() -> None:
    """§52. Documented rather than asserted away.

    Top-level equality *does* differ, because the analysis deliberately retains
    the composite config as provenance and that config changed. Everything the
    analysis computes - the reference price and every decision - is identical.
    """
    short = analysis(atr=5)
    long = analysis(atr=20)

    assert short != long, "the retained composite config differs"
    assert short.composite_config != long.composite_config
    assert short.reference_price == long.reference_price
    assert short.timeframes == long.timeframes


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (OrderBlockMitigationRule.TOUCH, OrderBlockMitigationRule.MIDPOINT),
        (OrderBlockMitigationRule.TOUCH, OrderBlockMitigationRule.FULL_ZONE),
        (OrderBlockMitigationRule.MIDPOINT, OrderBlockMitigationRule.FULL_ZONE),
    ],
)
def test_the_mitigation_rule_leaves_identity_geometry_and_other_kinds_alone(
    first: OrderBlockMitigationRule, second: OrderBlockMitigationRule
) -> None:
    """§53. Statuses may move; nothing else about an order block may."""
    left = analysis(rule=first)
    right = analysis(rule=second)

    assert stable(left) == stable(right)
    assert left.reference_price == right.reference_price
    for entry, other in zip(left.timeframes, right.timeframes, strict=True):
        non_blocks = [
            d for d in entry.decisions if d.source.kind is not CandidateSourceKind.ORDER_BLOCK
        ]
        other_non_blocks = [
            d for d in other.decisions if d.source.kind is not CandidateSourceKind.ORDER_BLOCK
        ]
        assert non_blocks == other_non_blocks


def kind_facts(result: CandidateEligibilityAnalysis, kind: CandidateSourceKind) -> list[object]:
    return [d for entry in result.timeframes for d in entry.decisions if d.source.kind is kind]


def test_the_zone_basis_moves_order_blocks_and_nothing_else() -> None:
    """§54."""
    full = analysis(basis=OrderBlockZoneBasis.FULL_CANDLE)
    body = analysis(basis=OrderBlockZoneBasis.BODY)

    assert kind_facts(full, CandidateSourceKind.ORDER_BLOCK) != kind_facts(
        body, CandidateSourceKind.ORDER_BLOCK
    )
    assert kind_facts(full, CandidateSourceKind.FAIR_VALUE_GAP) == kind_facts(
        body, CandidateSourceKind.FAIR_VALUE_GAP
    )
    assert kind_facts(full, CandidateSourceKind.LIQUIDITY_POOL) == kind_facts(
        body, CandidateSourceKind.LIQUIDITY_POOL
    )
    assert full.reference_price == body.reference_price


def test_the_liquidity_tolerance_moves_pools_and_nothing_else() -> None:
    """§55."""
    narrow = analysis(tolerance="0.50")
    wide = analysis(tolerance="25.00")

    assert kind_facts(narrow, CandidateSourceKind.LIQUIDITY_POOL) != kind_facts(
        wide, CandidateSourceKind.LIQUIDITY_POOL
    )
    assert kind_facts(narrow, CandidateSourceKind.ORDER_BLOCK) == kind_facts(
        wide, CandidateSourceKind.ORDER_BLOCK
    )
    assert kind_facts(narrow, CandidateSourceKind.FAIR_VALUE_GAP) == kind_facts(
        wide, CandidateSourceKind.FAIR_VALUE_GAP
    )
    assert narrow.reference_price == wide.reference_price


def test_the_swing_window_leaves_gaps_and_the_reference_price_alone() -> None:
    """§56."""
    narrow = analysis(left=2, right=2)
    wide = analysis(left=4, right=4)

    assert kind_facts(narrow, CandidateSourceKind.FAIR_VALUE_GAP) == kind_facts(
        wide, CandidateSourceKind.FAIR_VALUE_GAP
    )
    assert narrow.reference_price == wide.reference_price
    moved = kind_facts(narrow, CandidateSourceKind.ORDER_BLOCK) != kind_facts(
        wide, CandidateSourceKind.ORDER_BLOCK
    ) or kind_facts(narrow, CandidateSourceKind.LIQUIDITY_POOL) != kind_facts(
        wide, CandidateSourceKind.LIQUIDITY_POOL
    )
    assert moved


# --------------------------------------------------------------------------
# §63: cross-timeframe isolation, and the one case that is not isolated
# --------------------------------------------------------------------------


def test_changing_a_timeframe_that_is_not_the_reference_leaves_the_others_alone() -> None:
    """§63 case A. H4 is not the reference witness, so nothing else moves."""
    changed = dict(PATHS)
    changed[Timeframe.H4] = [*QUIET, *QUIET]
    after = analyse_candidate_eligibility(
        analyse_ict_composite(staggered_snapshot(changed), config=config()),
        config=eligibility_config(),
    )
    before = analysis()

    assert before.reference_price == after.reference_price
    for tf in ICT_TIMEFRAMES:
        left = before.require(tf)
        right = after.require(tf)
        if tf is Timeframe.H4:
            assert left != right
        else:
            assert left == right, tf


def test_changing_the_reference_timeframe_may_move_every_decision() -> None:
    """§63 case B, and the distinction that makes a global price a real design.

    M1 owns the latest close, so replacing its path changes the reference price -
    and location, distance and therefore eligibility may legitimately move on
    every other timeframe, which have not themselves changed at all.
    """
    changed = dict(PATHS)
    changed[Timeframe.M1] = [*QUIET, *QUIET, *QUIET, *QUIET]
    after = analyse_candidate_eligibility(
        analyse_ict_composite(staggered_snapshot(changed), config=config()),
        config=eligibility_config(),
    )
    before = analysis()

    assert before.reference_price.price != after.reference_price.price
    h1_before = before.require(Timeframe.H1)
    h1_after = after.require(Timeframe.H1)
    assert [d.candidate_id for d in h1_before.decisions] == [
        d.candidate_id for d in h1_after.decisions
    ], "the H1 sources themselves did not change"
    assert [d.market_relation for d in h1_before.decisions] != [
        d.market_relation for d in h1_after.decisions
    ], "but where they sit relative to price did"


def test_no_decision_is_merged_across_timeframes() -> None:
    """§45 of the previous round, still true here."""
    result = analysis()

    for entry in result.timeframes:
        for decision in entry.decisions:
            assert decision.timeframe is entry.timeframe


# --------------------------------------------------------------------------
# §36, §67: nothing is ordered by distance or selected
# --------------------------------------------------------------------------


def test_decisions_keep_the_projection_order_and_are_not_sorted_by_distance() -> None:
    """§36, §67."""
    entry = analysis().require(Timeframe.M1)
    distances = [d.distance_to_reference for d in entry.decisions]

    assert distances != sorted(distances)
    assert distances != sorted(distances, reverse=True)


def test_nothing_limits_how_many_candidates_survive() -> None:
    """§67. Seven eligible across the snapshot, and no cap anywhere."""
    result = analysis()

    assert sum(len(entry.eligible) for entry in result.timeframes) == 7
    assert len(result.require(Timeframe.M1).eligible) == 5
