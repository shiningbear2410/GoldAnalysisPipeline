"""What this layer refuses to do, even though it now knows the price.

Round 6.6e.2b §7, §35-§36, §58, §64-§72. Knowing the current price is what makes
the refusals here worth enforcing: with a reference price in hand, sorting by
distance, capping the list at five, or scoring by proximity are each one line
away, and each would ship a selection policy nobody argued about.

Guards read identifiers out of the parsed module rather than grepping its text,
so prose is free to name what it refuses.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from goldpipeline.services import ict_candidate_eligibility

SOURCE = Path("src/goldpipeline/services/ict_candidate_eligibility.py")

LOWER_ENGINE_ENTRY_POINTS = (
    "confirmed_swings",
    "fair_value_gaps",
    "latest_atr",
    "atr_series",
    "analyse_structure",
    "analyse_liquidity",
    "analyse_fvg_lifecycle",
    "analyse_protected_structure",
    "analyse_dealing_ranges",
    "analyse_order_blocks",
    "analyse_order_block_lifecycle",
    "analyse_ict_composite",
    "analyse_ict_timeframe",
)


def identifiers() -> set[str]:
    """Every name the module uses, string literals excluded."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.arg | ast.keyword) and node.arg is not None:
            names.add(node.arg)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            names.update(alias.name for alias in node.names)
    return names


def imported_modules() -> set[str]:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def called_names() -> set[str]:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def field_names() -> set[str]:
    return (
        set(ict_candidate_eligibility.CandidateEligibilityDecision.__dataclass_fields__)
        | set(ict_candidate_eligibility.CandidateEligibilityTimeframe.__dataclass_fields__)
        | set(ict_candidate_eligibility.CandidateEligibilityAnalysis.__dataclass_fields__)
        | set(ict_candidate_eligibility.CandidateEligibilityConfig.__dataclass_fields__)
        | set(ict_candidate_eligibility.ReferencePrice.__dataclass_fields__)
        | set(ict_candidate_eligibility.ReferencePriceWitness.__dataclass_fields__)
    )


# --------------------------------------------------------------------------
# §58: the projection is the only source seam
# --------------------------------------------------------------------------


@pytest.mark.parametrize("entry_point", LOWER_ENGINE_ENTRY_POINTS)
def test_no_lower_engine_is_ever_called(entry_point: str) -> None:
    """§58. Zero of them, checked by parsing rather than by review."""
    assert entry_point not in called_names(), entry_point
    assert entry_point not in identifiers(), entry_point


def test_the_only_upstream_call_is_the_projection() -> None:
    """§58. It may project sources; it may not analyse anything."""
    called = called_names()

    assert "project_candidate_sources" in called
    assert "project_timeframe_sources" not in called


def test_the_module_calls_nothing_but_its_own_helpers_and_stdlib() -> None:
    """§58, as an exact set - so a new call has to be argued for."""
    assert called_names() == {
        # its own helpers and models
        "_decide",
        "_candidate_id",
        "_entry_reasons",
        "_reference_reasons",
        "_require_same_composite",
        "resolve_reference_price",
        "role_for",
        "reference_level_for",
        "entry_side_for",
        "relation_of_zone",
        "relation_of_level",
        "distance_of_zone",
        "CandidateEligibilityError",
        "CandidateEligibilityAnalysis",
        "CandidateEligibilityTimeframe",
        "CandidateEligibilityDecision",
        "ReferencePrice",
        "ReferencePriceWitness",
        # the one upstream seam
        "project_candidate_sources",
        # a lookup on an object it was handed, not an analysis it ran
        "timeframe",
        # stdlib and the dataclass decorator
        "dataclass",
        "Decimal",
        "frozenset",
        "isinstance",
        "len",
        "max",
        "abs",
        "tuple",
        "next",
        "sorted",
        "append",
        "join",
        "encode",
        "hexdigest",
        "sha256",
        "getLogger",
        "isoformat",
    }


def test_the_entry_point_takes_a_composite_and_never_candles() -> None:
    """§58. Reading the composite's already-computed series is the only raw access."""
    from goldpipeline.services.ict_composite import IctCompositeAnalysis

    signature = inspect.signature(ict_candidate_eligibility.analyse_candidate_eligibility)

    assert signature.parameters["composite"].annotation in (
        IctCompositeAnalysis,
        "IctCompositeAnalysis",
    )
    for forbidden in (
        "IctTimeframeSnapshot",
        "IctMarketSnapshot",
        "OHLCBar",
        "build_timeframe_snapshot",
    ):
        assert forbidden not in identifiers(), forbidden


def test_the_reference_price_reads_only_closed_series() -> None:
    """§3, §58. ``series`` and ``bars[-1]`` are the whole of its raw access."""
    tree = ast.parse(inspect.getsource(ict_candidate_eligibility.resolve_reference_price))
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

    assert {"series", "latest_closed_at", "bars", "close"} <= attributes
    for forbidden in ("high", "low", "open", "volume"):
        assert forbidden not in attributes, forbidden


# --------------------------------------------------------------------------
# §7: lifecycle is not rewritten
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden", ["replace", "evolve", "mutate", "set_status", "with_status", "update"]
)
def test_no_source_status_is_rewritten(forbidden: str) -> None:
    """§7. A newer reference price does not let this layer re-judge an H4 block."""
    assert forbidden not in called_names(), forbidden
    assert forbidden not in identifiers(), forbidden


def test_the_decision_holds_the_source_rather_than_a_copy_of_its_status() -> None:
    """§7, §27. Status stays where the lifecycle engines put it."""
    fields = set(ict_candidate_eligibility.CandidateEligibilityDecision.__dataclass_fields__)

    assert "source" in fields
    for forbidden in ("status", "lifecycle", "mitigation_rule"):
        assert forbidden not in fields, forbidden


# --------------------------------------------------------------------------
# §36, §67: no score, no ranking, no selection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        "score",
        "rank",
        "priority",
        "quality",
        "strength",
        "confidence",
        "confluence",
        "weight",
        "best",
        "top",
    ],
)
def test_no_scoring_of_any_kind(forbidden: str) -> None:
    """§36. Distance is geometry; nothing here turns it into a judgement."""
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden
    assert not any(forbidden in name.lower() for name in field_names()), forbidden
    assert not any(forbidden in name.lower() for name in ict_candidate_eligibility.__all__), (
        forbidden
    )


def test_nothing_is_sorted_by_distance() -> None:
    """§36, §67. The only sort in the module is the fixed reason order.

    Parsed rather than reviewed: a ``sorted`` or ``.sort`` keyed on a price
    would be exactly the selection policy this round refuses.
    """
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    sorts = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "sorted")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "sort")
        )
    ]

    assert len(sorts) == 2, sorts
    for expression in sorts:
        assert "status.value" in expression, "both sorts tidy a config error message"
        assert "distance" not in expression
        assert "price" not in expression


@pytest.mark.parametrize("forbidden", ["limit", "cap", "select", "choose", "pick", "shortlist"])
def test_nothing_selects_a_subset(forbidden: str) -> None:
    """§67. No 3-5 zones, no one main zone, no farther-level cap."""
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden
    assert not any(forbidden in name.lower() for name in field_names()), forbidden


def test_no_narrative_label_appears_anywhere() -> None:
    """§67. No published labels, in any language."""
    text = SOURCE.read_text(encoding="utf-8").lower()

    for forbidden in ("scalp", "vùng", "canh cả", "sâu hơn", "main zone"):  # noqa: RUF001
        assert forbidden not in text, forbidden


# --------------------------------------------------------------------------
# §35, §66: no consolidation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    ["dedupe", "deduplicate", "consolidat", "merge", "collapse", "cluster", "union"],
)
def test_no_consolidation_of_any_kind(forbidden: str) -> None:
    """§35, §66. Two blocks sharing a zone remain two decisions."""
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden
    assert not any(forbidden in name.lower() for name in field_names()), forbidden


def test_overlap_is_a_relation_and_never_a_grouping() -> None:
    """§35, §66, narrowed to its intent.

    ``OVERLAPS`` is where one zone sits relative to one price - a fact about a
    single source. Banning the substring would have forced the relation enum to
    be renamed; what is actually forbidden is an *overlap between candidates*,
    which is the shape consolidation would take.
    """
    assert "OVERLAPS" in {member.name for member in ict_candidate_eligibility.MarketRelation}
    for forbidden in ("overlapping", "overlap_count", "overlaps_with", "overlap_group"):
        assert not any(forbidden in name.lower() for name in identifiers()), forbidden


# --------------------------------------------------------------------------
# §64: no premium/discount filter
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    ["premium", "discount", "in_premium", "in_discount", "range_mismatch", "price_location"],
)
def test_no_range_based_reason_or_filter(forbidden: str) -> None:
    """§64. The active range is carried and never consulted."""
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden
    assert not any(
        forbidden in member.value.lower() for member in ict_candidate_eligibility.EligibilityReason
    ), forbidden


def test_the_active_range_is_carried_and_never_read() -> None:
    """§33, §64. It reaches the result and no branch depends on it.

    ``locate`` is the range engine's own classifier; the module never calls it,
    and ``active_dealing_range`` appears only where the timeframe result is
    assembled.
    """
    assert "locate" not in identifiers()
    assert "active_dealing_range" in field_names()

    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "active_dealing_range"
    ]
    assert len(reads) == 1, "read once, to carry it through"


# --------------------------------------------------------------------------
# §65: no freshness
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    ["fresh", "stale", "expired", "bars_since", "minutes_since", "max_age", "recency"],
)
def test_no_freshness_notion(forbidden: str) -> None:
    """§6, §65. A stale reference price is reported, never rejected."""
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden
    assert not any(forbidden in name.lower() for name in field_names()), forbidden


def test_no_age_is_measured() -> None:
    """§65, token-matched because "age" hides inside ``preimage``."""
    for name in identifiers() | field_names():
        parts = name.lower().split("_")
        assert "age" not in parts, name


# --------------------------------------------------------------------------
# §14: liquidity is never a trade side
# --------------------------------------------------------------------------


def test_no_reference_role_can_receive_an_entry_side() -> None:
    """§14. Enforced in code, not merely by convention."""
    source = inspect.getsource(ict_candidate_eligibility._decide)

    assert "side = None" in source
    assert "entry_side_for" in source


def test_the_side_and_role_enums_do_not_overlap() -> None:
    """§14. BAI/SEO and UPPER/LOWER_REFERENCE are different vocabularies."""
    sides = {member.value for member in ict_candidate_eligibility.EntrySide}
    roles = {member.value for member in ict_candidate_eligibility.CandidateRole}

    assert sides.isdisjoint(roles)
    assert sides == {"BAI", "SEO"}


@pytest.mark.parametrize("forbidden", ["buy_entry", "sell_entry", "trade_direction"])
def test_no_liquidity_side_is_reinterpreted_as_a_trade(forbidden: str) -> None:
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden


# --------------------------------------------------------------------------
# §68: source agnosticism
# --------------------------------------------------------------------------


def test_the_module_reads_only_ict_inputs() -> None:
    """Asserted as an exact set, so a ninth dependency fails rather than sneaks in."""
    inside = {module for module in imported_modules() if module.startswith("goldpipeline")}

    assert inside == {
        "goldpipeline.schemas.common",
        "goldpipeline.services.ict_candidate_source",
        "goldpipeline.services.ict_composite",
        "goldpipeline.services.ict_fvg",
        "goldpipeline.services.ict_liquidity",
        "goldpipeline.services.ict_order_block_lifecycle",
        "goldpipeline.services.ict_primitives",
        "goldpipeline.services.ict_range",
        "goldpipeline.services.ict_structure",
    }


@pytest.mark.parametrize(
    "forbidden",
    [
        "tradingview",
        "metatrader",
        "MetaTrader5",
        "websocket",
        "requests",
        "httpx",
        "anthropic",
        "deepseek",
        "telegram",
        "news",
        "services.levels",
    ],
)
def test_the_module_imports_no_provider(forbidden: str) -> None:
    for module in imported_modules():
        assert forbidden.lower() not in module.lower(), module


def test_the_module_reads_no_clock() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    for forbidden in ("datetime.now", "utcnow", "time.time", "utc_now"):
        assert forbidden not in text


def test_the_module_never_names_float() -> None:
    assert "float" not in identifiers()


def test_no_float_literal_appears_in_the_source() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            raise AssertionError(f"float literal {node.value} at line {node.lineno}")


@pytest.mark.parametrize("forbidden", ["round", "quantize", "scaleb", "tick", "pips", "percent"])
def test_no_rounding_or_normalisation_of_distance(forbidden: str) -> None:
    """§24. No pip conversion, no percentage, no normalisation."""
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden


@pytest.mark.parametrize("forbidden", ["tolerance_of", "epsilon", "approx", "nearly"])
def test_no_tolerance_in_any_comparison(forbidden: str) -> None:
    """§17. Exact boundaries everywhere."""
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden


# --------------------------------------------------------------------------
# §70-§72: nothing shipped changed
# --------------------------------------------------------------------------


def test_the_branch_registry_knows_about_this_module() -> None:
    from tests.test_ict_structure_guards import ICT_BRANCH

    assert "ict_candidate_eligibility.py" in ICT_BRANCH


def test_no_eligibility_code_is_reachable_from_the_shipped_products() -> None:
    """§72."""
    from tests.test_ict_structure_guards import DISPATCH_SEAM, ICT_BRANCH

    # Round 6.6h activated TRADE_PLAN, so the branch is reachable from
    # exactly two files: the orchestrator that dispatches it and the
    # runtime table that tells it to. Both are named, so "only the
    # dispatch we chose" stays a stronger claim than "nobody at all".
    REACHABLE = ICT_BRANCH | DISPATCH_SEAM

    root = Path("src/goldpipeline")
    callers = [
        path
        for path in root.rglob("*.py")
        if "ict_candidate_eligibility" in path.read_text(encoding="utf-8")
        and path.name not in REACHABLE
    ]

    assert callers == []


def test_trade_plan_is_live_and_still_publishes_nothing() -> None:
    """Round 6.6h activated it. What must stay true is everything *else*.

    Ready and dispatchable, and at the same time: no writer prompt, no style
    pass, no repair path, and no automatic publication. Activation was about
    letting a deterministic document reach a human, not about letting anything
    reach a channel.
    """
    from goldpipeline.schemas.article import ArticleType
    from goldpipeline.services.article_routing import SPECS
    from goldpipeline.services.article_runtime import (
        RUNTIMES,
        RevisionRuntime,
        WriteRuntime,
        is_dispatchable,
        runtime_for,
    )
    from goldpipeline.services.review_action import STYLE_ACTIVE_TYPES

    assert SPECS[ArticleType.TRADE_PLAN].ready is True
    assert SPECS[ArticleType.TRADE_PLAN].prompt_id is None
    assert is_dispatchable(ArticleType.TRADE_PLAN) is True
    assert runtime_for(ArticleType.TRADE_PLAN).write is WriteRuntime.TRADE_PLAN
    assert RUNTIMES[ArticleType.TRADE_PLAN].revise is RevisionRuntime.NONE
    assert ArticleType.TRADE_PLAN not in STYLE_ACTIVE_TYPES


def test_the_two_shipped_products_are_untouched() -> None:
    from goldpipeline.schemas.article import ArticleType
    from goldpipeline.services.article_routing import SPECS, writer_prompt_for
    from goldpipeline.services.review_action import STYLE_ACTIVE_TYPES

    assert writer_prompt_for(ArticleType.ANALYSIS) == "gold_writer_v4"
    assert writer_prompt_for(ArticleType.NEWS_DIGEST) == "gold_news_digest_writer_v2"
    assert SPECS[ArticleType.ANALYSIS].ready is True
    assert SPECS[ArticleType.NEWS_DIGEST].ready is True
    assert frozenset({ArticleType.ANALYSIS, ArticleType.NEWS_DIGEST}) == STYLE_ACTIVE_TYPES


def test_no_shipped_prompt_learned_a_word_from_this_round() -> None:
    from goldpipeline.prompts import load_prompt

    for name in (
        "gold_writer_v4",
        "gold_reviewer_v2",
        "gold_finalizer_v2",
        "gold_news_digest_writer_v2",
        "gold_news_digest_finalizer_v1",
        "gold_human_style_v1",
    ):
        text = load_prompt(name).lower()
        # Not "entry zone": the digest writer already forbids entry zones in its
        # own prose, and a prompt refusing something is the opposite of having
        # learned it. These four could only appear if it had.
        for word in ("candidate", "reference price", "eligib", "bai zone"):
            assert word not in text, f"{name} mentions {word!r}"
