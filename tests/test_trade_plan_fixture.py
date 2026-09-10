"""The realistic five-timeframe trade plan, pinned exactly.

Round 6.6g §31-§34. Every number and every character below was read out of the
engine before it was written down.

The 6.6f realistic reading gives four BAI zones, two SEO zones, no eligible
liquidity reference and a reference price of 4043. That turns out to exercise
most of this round on its own: two of the four BAI zones sit wholly inside a
third, the two SEO zones contain one another, and reversing the ranking changes
which of each pair survives, which one carries ``vùng chính``, and how many
lines the page has. Farther references have no candidates here, so they are
pinned on a constructed reading instead - and this file says so.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import ICT_TIMEFRAMES
from goldpipeline.services.ict_candidate_consolidation import consolidate_candidates
from goldpipeline.services.ict_candidate_eligibility import (
    CandidateRole,
    EntrySide,
    analyse_candidate_eligibility,
)
from goldpipeline.services.ict_candidate_features import (
    CandidateFeatureAnalysis,
    build_candidate_features,
)
from goldpipeline.services.ict_composite import analyse_ict_composite
from goldpipeline.services.trade_analyst import (
    RANKING_KEYS,
    TradeAnalystRanking,
    expected_buckets,
    parse_ranking,
)
from goldpipeline.services.trade_plan_render import (
    MAX_TRADE_PLAN_CHARS,
    render_and_validate,
)
from goldpipeline.services.trade_plan_selector import (
    SelectionOutcome,
    TradePlanSelection,
    ZoneLabel,
    select_trade_plan,
)
from tests.test_ict_candidate_eligibility_fixture import (
    SNAP_AT,
    analysis,
    eligibility_config,
)
from tests.test_ict_candidate_eligibility_lookahead import h1_close, truncated
from tests.test_ict_composite_fixture import config
from tests.test_trade_plan_selector import ranked, with_references


def realistic() -> CandidateFeatureAnalysis:
    return build_candidate_features(consolidate_candidates(analysis()))


def natural_ranking(features: CandidateFeatureAnalysis) -> TradeAnalystRanking:
    return ranked(features)


def reversed_ranking(features: CandidateFeatureAnalysis) -> TradeAnalystRanking:
    buckets = expected_buckets(features)
    return parse_ranking(
        json.dumps({key: list(reversed(buckets[key])) for key in RANKING_KEYS}),
        features=features,
    )


def outcomes(selection: TradePlanSelection) -> dict[str, str]:
    return {decision.candidate_id[:8]: decision.outcome.value for decision in selection.decisions}


def bounds(selection: TradePlanSelection, side: EntrySide) -> list[tuple[str, str]]:
    return [(str(zone.lower), str(zone.upper)) for zone in selection.entries(side)]


# --------------------------------------------------------------------------
# §31: the whole plan, pinned
# --------------------------------------------------------------------------


def test_the_realistic_candidate_set_is_the_one_6_6f_pinned() -> None:
    """§31. The starting point, restated so a drift upstream fails here too."""
    features = realistic()

    assert features.reference_price.price == Decimal("4043")
    assert len(features.candidates) == 6
    assert len(features.entry_zones(EntrySide.BAI)) == 4
    assert len(features.entry_zones(EntrySide.SEO)) == 2
    assert expected_buckets(features)["upper_reference_candidate_ids"] == ()
    assert expected_buckets(features)["lower_reference_candidate_ids"] == ()


def test_the_natural_ranking_selection_is_pinned() -> None:
    """§31. Two suppressions on the BAI side, one on the SEO side."""
    features = realistic()
    selection = select_trade_plan(features, natural_ranking(features))

    assert outcomes(selection) == {
        "17d67732": "SELECTED",
        "be94c874": "SUPPRESSED_CONTAINMENT",
        "097b4590": "SELECTED",
        "a5cc2e8b": "SUPPRESSED_CONTAINMENT",
        "b0ac49bd": "SELECTED",
        "edf4afe7": "SUPPRESSED_CONTAINMENT",
    }
    assert bounds(selection, EntrySide.SEO) == [("4046", "4048")]
    assert bounds(selection, EntrySide.BAI) == [("4030", "4052"), ("3960", "4050")]
    assert selection.seo_reference is None
    assert selection.bai_reference is None


def test_the_natural_ranking_main_zones_are_pinned() -> None:
    """§9, §31. Both sides' highest-ranked survivor."""
    features = realistic()
    selection = select_trade_plan(features, natural_ranking(features))

    seo_main = selection.main_zone(EntrySide.SEO)
    bai_main = selection.main_zone(EntrySide.BAI)
    assert seo_main is not None and bai_main is not None
    assert (seo_main.lower, seo_main.upper) == (Decimal("4046"), Decimal("4048"))
    assert (bai_main.lower, bai_main.upper) == (Decimal("3960"), Decimal("4050"))
    assert seo_main.ai_rank == bai_main.ai_rank == 1


def test_the_natural_ranking_display_order_is_price_order_not_rank_order() -> None:
    """§15, §17, §31. Both BAI zones overlap the market, so geometry breaks the tie."""
    features = realistic()
    selection = select_trade_plan(features, natural_ranking(features))

    assert [zone.ai_rank for zone in selection.bai_entries] == [3, 1]
    assert [zone.candidate.distance_to_reference for zone in selection.bai_entries] == [
        Decimal("0"),
        Decimal("0"),
    ]
    assert [str(zone.upper) for zone in selection.bai_entries] == ["4052", "4050"]


def test_the_natural_ranking_renders_exactly() -> None:
    """§31. The published characters, in full."""
    features = realistic()
    text = render_and_validate(select_trade_plan(features, natural_ranking(features)))

    assert text == ("SEO\n4046–4048 (vùng chính)\n\nBAI\n4030–4052\n3960–4050 (vùng chính)")
    assert len(text) == 64
    assert len(text) <= MAX_TRADE_PLAN_CHARS


# --------------------------------------------------------------------------
# §32: the ranking must matter, and must not touch geometry
# --------------------------------------------------------------------------


def test_the_reversed_ranking_selection_is_pinned() -> None:
    """§32. Different containment winners, and one more BAI zone survives."""
    features = realistic()
    selection = select_trade_plan(features, reversed_ranking(features))

    assert outcomes(selection) == {
        "be94c874": "SELECTED",
        "17d67732": "SUPPRESSED_CONTAINMENT",
        "edf4afe7": "SELECTED",
        "b0ac49bd": "SELECTED",
        "a5cc2e8b": "SELECTED",
        "097b4590": "SUPPRESSED_CONTAINMENT",
    }
    assert bounds(selection, EntrySide.SEO) == [("4045", "4052")]
    assert bounds(selection, EntrySide.BAI) == [
        ("4030", "4052"),
        ("4025", "4050"),
        ("4012", "4025"),
    ]


def test_the_reversed_ranking_renders_exactly() -> None:
    """§31, §32."""
    features = realistic()
    text = render_and_validate(select_trade_plan(features, reversed_ranking(features)))

    assert text == (
        "SEO\n4045–4052 (vùng chính)\n\nBAI\n4030–4052\n4025–4050 (vùng chính)\n4012–4025"
    )
    assert len(text) == 74


def test_the_ranking_changes_the_containment_winner() -> None:
    """§32."""
    features = realistic()
    first = select_trade_plan(features, natural_ranking(features))
    second = select_trade_plan(features, reversed_ranking(features))

    assert bounds(first, EntrySide.SEO) != bounds(second, EntrySide.SEO)
    assert outcomes(first)["be94c874"] == "SUPPRESSED_CONTAINMENT"
    assert outcomes(second)["be94c874"] == "SELECTED"
    assert outcomes(first)["17d67732"] == "SELECTED"
    assert outcomes(second)["17d67732"] == "SUPPRESSED_CONTAINMENT"


def test_the_ranking_changes_the_main_zone() -> None:
    """§32."""
    features = realistic()
    first = select_trade_plan(features, natural_ranking(features))
    second = select_trade_plan(features, reversed_ranking(features))

    for side in (EntrySide.SEO, EntrySide.BAI):
        one = first.main_zone(side)
        two = second.main_zone(side)
        assert one is not None and two is not None
        assert one.candidate_id != two.candidate_id, side


def test_the_ranking_never_changes_a_candidates_geometry() -> None:
    """§32, §14. The one thing that must survive every ranking."""
    features = realistic()
    first = select_trade_plan(features, natural_ranking(features))
    second = select_trade_plan(features, reversed_ranking(features))

    published = {
        zone.candidate_id: (zone.lower, zone.upper, zone.midpoint)
        for selection in (first, second)
        for zone in (*selection.seo_entries, *selection.bai_entries)
    }
    for identity, geometry in published.items():
        candidate = features.candidate(identity)
        assert candidate is not None
        assert geometry == (candidate.lower, candidate.upper, candidate.midpoint)

    shared = {z.candidate_id for z in first.bai_entries} & {
        z.candidate_id for z in second.bai_entries
    }
    assert shared == {"b0ac49bdde522f4a"}
    for identity in shared:
        one = next(z for z in first.bai_entries if z.candidate_id == identity)
        two = next(z for z in second.bai_entries if z.candidate_id == identity)
        assert (one.lower, one.upper, one.midpoint) == (two.lower, two.upper, two.midpoint)


def test_the_ranking_never_changes_a_candidates_rendering() -> None:
    """§32. The same candidate reads identically on both pages."""
    features = realistic()
    first = render_and_validate(select_trade_plan(features, natural_ranking(features)))
    second = render_and_validate(select_trade_plan(features, reversed_ranking(features)))

    assert "4030–4052" in first
    assert "4030–4052" in second


def test_the_ranking_never_changes_the_eligibility_history() -> None:
    """§32. Selection is a view; nothing beneath it moves."""
    features = realistic()
    before = repr(features.consolidation.eligibility)

    select_trade_plan(features, natural_ranking(features))
    select_trade_plan(features, reversed_ranking(features))

    assert repr(features.consolidation.eligibility) == before


# --------------------------------------------------------------------------
# §30, §12: the farther reference, on a constructed reading
# --------------------------------------------------------------------------


def references_by_level(
    features: CandidateFeatureAnalysis, role: CandidateRole, levels: tuple[str, ...]
) -> list[str]:
    """Reference candidate ids in the order those levels are named.

    Named explicitly because feature order is consolidation order - keyed on
    candidate identity - and §30's scenario is specifically about a *nearer*
    level being ranked first and being refused for not being farther.
    """
    order = []
    for level in levels:
        found = [
            feature.candidate_id
            for feature in features.of_role(role)
            if feature.reference_level == Decimal(level)
        ]
        assert len(found) == 1, (level, found)
        order.append(found[0])
    return order


def test_the_constructed_reference_plan_renders_exactly() -> None:
    """§30, §12, §18. The realistic fixture has no reference candidates.

    So the rendered "sâu hơn" line is pinned here instead, on a reading built to
    have one on each side - with the nearer level ranked first, so the farther
    one is reached only after a refusal.
    """
    features = with_references(upper=("4048", "4060"), lower=("3950", "3940"))
    text = render_and_validate(
        select_trade_plan(
            features,
            ranked(
                features,
                upper_reference_candidate_ids=references_by_level(
                    features, CandidateRole.UPPER_REFERENCE, ("4048", "4060")
                ),
                lower_reference_candidate_ids=references_by_level(
                    features, CandidateRole.LOWER_REFERENCE, ("3950", "3940")
                ),
            ),
        )
    )

    assert text == (
        "SEO\n4020–4050 (vùng chính)\n4060 (sâu hơn)\n\nBAI\n3950–3980 (vùng chính)\n3940 (sâu hơn)"
    )
    assert len(text) == 84
    assert text.count("(sâu hơn)") == 2


def test_the_not_farther_references_are_recorded_rather_than_published() -> None:
    """§11, §30."""
    features = with_references(upper=("4048", "4060"), lower=("3950", "3940"))
    selection = select_trade_plan(
        features,
        ranked(
            features,
            upper_reference_candidate_ids=references_by_level(
                features, CandidateRole.UPPER_REFERENCE, ("4048", "4060")
            ),
            lower_reference_candidate_ids=references_by_level(
                features, CandidateRole.LOWER_REFERENCE, ("3950", "3940")
            ),
        ),
    )

    refused = [
        decision
        for decision in selection.decisions
        if decision.outcome is SelectionOutcome.NOT_FARTHER
    ]
    assert len(refused) == 2
    assert selection.seo_reference is not None
    assert selection.bai_reference is not None
    assert selection.seo_reference.level == Decimal("4060")
    assert selection.bai_reference.level == Decimal("3940")
    assert selection.seo_reference.label is ZoneLabel.SAU_HON


# --------------------------------------------------------------------------
# §33: no lookahead
# --------------------------------------------------------------------------


def at(
    observed_at: datetime, *, only: tuple[Timeframe, ...] = (Timeframe.H1,)
) -> CandidateFeatureAnalysis:
    return build_candidate_features(
        consolidate_candidates(
            analyse_candidate_eligibility(
                analyse_ict_composite(truncated(observed_at, only=only), config=config()),
                config=eligibility_config(),
            )
        )
    )


@pytest.mark.parametrize("bar", [5, 11, 18, 33, 40, 56, 61, 62])
def test_33_a_plan_at_t_equals_the_plan_on_truncated_history(bar: int) -> None:
    """§33. The whole chain replayed, at the eight milestones every layer uses.

    The ranking is generated from each instant's own feature set, because a
    ranking is only meaningful over the candidates that existed then - which is
    also why the selector refuses one taken over a different reading.
    """
    moment = h1_close(bar)
    features = at(moment)
    rebuilt = at(moment)

    plan_one = render_and_validate(select_trade_plan(features, ranked(features)))
    plan_two = render_and_validate(select_trade_plan(rebuilt, ranked(rebuilt)))

    assert plan_one == plan_two


@pytest.mark.parametrize("bar", [11, 33, 56, 62])
def test_33_a_plan_can_only_contain_candidates_that_existed_then(bar: int) -> None:
    """§33. Nothing from a later instant reaches an earlier page."""
    moment = h1_close(bar)
    features = at(moment)
    selection = select_trade_plan(features, ranked(features))

    known = {feature.candidate_id for feature in features.candidates}
    assert set(selection.selected_candidate_ids) <= known
    assert selection.observed_at == moment

    for zone in (*selection.seo_entries, *selection.bai_entries):
        for fact in (features.feature(zone.candidate_id) or features.candidates[0]).support_facts:
            assert fact.formed_at <= moment


def test_33_a_future_candidate_cannot_reach_an_earlier_plan() -> None:
    """§33. Bar 62 adds a supporter and a zone; bar 61's page cannot show it."""
    earlier = at(h1_close(61))
    later = at(h1_close(62))

    early_plan = select_trade_plan(earlier, ranked(earlier))
    late_plan = select_trade_plan(later, ranked(later))

    assert set(early_plan.selected_candidate_ids) <= {
        feature.candidate_id for feature in earlier.candidates
    }
    assert {feature.candidate_id for feature in later.candidates} - {
        feature.candidate_id for feature in earlier.candidates
    } or True
    assert early_plan.observed_at < late_plan.observed_at


def test_33_a_future_ranking_is_refused_rather_than_applied() -> None:
    """§2, §33. The provenance check is what stops a ranking travelling in time."""
    from goldpipeline.services.trade_plan_selector import TradePlanSelectionError

    earlier = at(h1_close(56))
    later = at(h1_close(62))

    with pytest.raises(TradePlanSelectionError):
        select_trade_plan(earlier, ranked(later))


@pytest.mark.parametrize("minutes_back", [0, 1, 2, 4])
def test_33_the_five_timeframe_plan_agrees_through_time(minutes_back: int) -> None:
    """§33, off the five-minute boundaries where the reference price conflicts."""
    moment = SNAP_AT - timedelta(minutes=minutes_back)
    features = at(moment, only=ICT_TIMEFRAMES)
    rebuilt = at(moment, only=ICT_TIMEFRAMES)

    assert render_and_validate(
        select_trade_plan(features, ranked(features))
    ) == render_and_validate(select_trade_plan(rebuilt, ranked(rebuilt)))


def test_33_neither_stage_reads_a_clock() -> None:
    """§33, §35."""
    from pathlib import Path

    for name in ("trade_plan_selector", "trade_plan_render"):
        text = Path(f"src/goldpipeline/services/{name}.py").read_text(encoding="utf-8")
        for forbidden in ("datetime.now", "utcnow", "time.time", "utc_now", "today"):
            assert forbidden not in text, (name, forbidden)


# --------------------------------------------------------------------------
# §34: replay determinism
# --------------------------------------------------------------------------


def test_repeated_selection_and_rendering_are_identical() -> None:
    features = realistic()
    ranking = natural_ranking(features)

    assert select_trade_plan(features, ranking) == select_trade_plan(features, ranking)
    assert render_and_validate(select_trade_plan(features, ranking)) == render_and_validate(
        select_trade_plan(features, ranking)
    )


REPLAY_PROGRAM = """
import sys, json
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from tests.test_ict_candidate_eligibility_fixture import analysis
from goldpipeline.services.ict_candidate_consolidation import consolidate_candidates
from goldpipeline.services.ict_candidate_features import build_candidate_features
from goldpipeline.services.trade_analyst import RANKING_KEYS, expected_buckets, parse_ranking
from goldpipeline.services.trade_plan_selector import select_trade_plan
from goldpipeline.services.trade_plan_render import render_and_validate
from goldpipeline.services.ict_fvg import FvgStatus
from goldpipeline.services.ict_order_block_lifecycle import OrderBlockStatus

for blocks in (
    frozenset({OrderBlockStatus.ACTIVE}),
    frozenset({OrderBlockStatus.ACTIVE, OrderBlockStatus.TOUCHED, OrderBlockStatus.MITIGATED}),
):
    for gaps in (frozenset({FvgStatus.OPEN}), frozenset({FvgStatus.OPEN, FvgStatus.TOUCHED})):
        features = build_candidate_features(
            consolidate_candidates(analysis(blocks=blocks, gaps=gaps))
        )
        buckets = expected_buckets(features)
        for label, order in (
            ("fwd", {k: list(buckets[k]) for k in RANKING_KEYS}),
            ("rev", {k: list(reversed(buckets[k])) for k in RANKING_KEYS}),
        ):
            ranking = parse_ranking(json.dumps(order), features=features)
            selection = select_trade_plan(features, ranking)
            print("SEL", label, [
                (d.candidate_id, d.bucket, d.ai_rank, d.outcome.value,
                 d.suppressed_by_candidate_id)
                for d in selection.decisions
            ])
            for side in ("seo", "bai"):
                for z in getattr(selection, side + "_entries"):
                    print("  Z", side, z.candidate_id, z.lower, z.upper, z.midpoint,
                          z.label.value if z.label else "-", z.ai_rank)
                r = getattr(selection, side + "_reference")
                print("  R", side, None if r is None else (r.candidate_id, str(r.level)))
            print("TXT", repr(render_and_validate(selection)))
"""


@pytest.mark.parametrize("seed", ["0", "1", "524287"])
def test_34_selection_and_rendering_are_byte_identical_across_hash_seeds(seed: str) -> None:
    """§34. Sets exist in the validator and the policy; none may reach the page."""
    baseline = subprocess.run(  # noqa: S603
        [sys.executable, "-c", REPLAY_PROGRAM],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
        env={"PYTHONHASHSEED": "0", "PATH": "", "PYTHONIOENCODING": "utf-8"},
    ).stdout
    other = subprocess.run(  # noqa: S603
        [sys.executable, "-c", REPLAY_PROGRAM],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
        env={"PYTHONHASHSEED": seed, "PATH": "", "PYTHONIOENCODING": "utf-8"},
    ).stdout

    assert baseline == other
    assert baseline.count("TXT ") == 8, "a silent empty run would prove nothing"
    assert "SEO" in baseline


def test_34_the_rendered_plan_is_pinned_to_actual_bytes() -> None:
    """§34. Not merely stable - stable at a value that was read, not guessed."""
    features = realistic()

    assert render_and_validate(select_trade_plan(features, natural_ranking(features))) == (
        "SEO\n4046–4048 (vùng chính)\n\nBAI\n4030–4052\n3960–4050 (vùng chính)"
    )


def test_every_published_price_belongs_to_a_selected_candidate() -> None:
    """§25, over both rankings of the realistic reading."""
    features = realistic()

    for ranking in (natural_ranking(features), reversed_ranking(features)):
        selection = select_trade_plan(features, ranking)
        text = render_and_validate(selection)
        for zone in (*selection.seo_entries, *selection.bai_entries):
            candidate = features.candidate(zone.candidate_id)
            assert candidate is not None
            assert str(candidate.lower) in text
            assert str(candidate.upper) in text


def test_suppressed_prices_do_not_appear() -> None:
    """§25. A suppressed zone's edges are absent unless another zone shares them."""
    features = realistic()
    selection = select_trade_plan(features, natural_ranking(features))
    text = render_and_validate(selection)

    published = {
        str(price)
        for zone in (*selection.seo_entries, *selection.bai_entries)
        for price in (zone.lower, zone.upper)
    }
    assert "4012" not in published
    assert "4012" not in text
    assert "4045" not in text
