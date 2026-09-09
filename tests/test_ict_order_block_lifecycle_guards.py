"""The things the lifecycle engine deliberately stops short of.

Round 6.6d.2 §1, §3, §27-§28, §43, §54-§61. The important absence this round is
usefulness: with a status in hand, "is this one still worth trading?" is one
field away, and writing it would ship a freshness policy nobody has argued
about - the same mistake mitigation would have been if the caller did not have
to name a rule.

Guards read identifiers out of the parsed module rather than grepping its text,
so prose is free to name what it refuses.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from goldpipeline.services import ict_order_block, ict_order_block_lifecycle

SOURCE = Path("src/goldpipeline/services/ict_order_block_lifecycle.py")


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


def field_names() -> set[str]:
    return (
        set(ict_order_block_lifecycle.OrderBlockState.__dataclass_fields__)
        | set(ict_order_block_lifecycle.OrderBlockLifecycleAnalysis.__dataclass_fields__)
        | set(ict_order_block_lifecycle.OrderBlockLifecycleConfig.__dataclass_fields__)
        | set(ict_order_block_lifecycle.BarWitness.__dataclass_fields__)
    )


# --------------------------------------------------------------------------
# §28, §57: no conversion into another object
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    ["breaker", "mitigation_block", "rejection_block", "propulsion", "refined", "inverted"],
)
def test_no_block_conversion_vocabulary(forbidden: str) -> None:
    """§28, §57. An invalidated order block stays an invalidated order block.

    A breaker is a different object with its own formation rules, and minting
    one here would mean this round had quietly defined them.
    """
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in field_names())
    assert not any(forbidden in name.lower() for name in ict_order_block_lifecycle.__all__)
    assert not any(
        forbidden in member.value.lower() for member in ict_order_block_lifecycle.OrderBlockStatus
    )


def test_there_is_no_second_object_type_at_all() -> None:
    """§28. The module defines states and witnesses, and nothing that could be a zone."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}

    assert classes == {
        "OrderBlockStatus",
        "OrderBlockMitigationRule",
        "OrderBlockLifecycleConfig",
        "OrderBlockLifecycleError",
        "BarWitness",
        "OrderBlockState",
        "OrderBlockLifecycleAnalysis",
    }


# --------------------------------------------------------------------------
# §26, §56: no counting, no freshness
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    ["touch_count", "mitigation_count", "reaction", "visits", "revisit", "hits", "tally"],
)
def test_nothing_is_counted(forbidden: str) -> None:
    """§26. How many times price returned is a usefulness question."""
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in field_names())


@pytest.mark.parametrize(
    "forbidden",
    ["fresh", "stale", "expired", "usable", "tradable", "eligible", "preferred", "redundant"],
)
def test_no_freshness_policy(forbidden: str) -> None:
    """§56. Whether a TOUCHED block is still worth using belongs to the candidate engine."""
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in field_names())
    assert not any(forbidden in name.lower() for name in ict_order_block_lifecycle.__all__)


@pytest.mark.parametrize("forbidden", ["dedupe", "deduplicate", "consolidat", "merge", "collapse"])
def test_no_consolidation(forbidden: str) -> None:
    """§29, §30. Two blocks sharing a candle or a zone stay two blocks."""
    assert not any(forbidden in name.lower() for name in identifiers())


# --------------------------------------------------------------------------
# §27, §54, §55: nothing else retires a block
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    ["premium", "discount", "dealing_range", "equilibrium", "price_location", "superseded"],
)
def test_no_dealing_range_participation(forbidden: str) -> None:
    """§27, §54. A range being superseded is a different fact about a different object."""
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize("forbidden", ["fair_value_gap", "fvg", "imbalance", "displacement"])
def test_no_fair_value_gap_participation(forbidden: str) -> None:
    """§55. Confluence is a candidate-engine idea, not a lifecycle input."""
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize("forbidden", ["liquidity", "sweep", "pool", "bsl", "ssl", "close_through"])
def test_no_liquidity_participation(forbidden: str) -> None:
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize(
    "forbidden", ["atr", "wilder", "true_range", "volatility", "volume", "session", "kill_zone"]
)
def test_no_volatility_or_session_context(forbidden: str) -> None:
    """§27. Lifecycle is price interaction, and nothing else."""
    assert not any(forbidden in name.lower() for name in identifiers())


def test_no_structural_event_reaches_the_lifecycle() -> None:
    """§27. Bias flipping, a newer block forming, an anchor moving - none of them.

    The module reads one thing out of the structure package, ``BreakDirection``,
    and uses it to know which edge is far. It never analyses structure.
    """
    names = identifiers()

    assert "BreakDirection" in names
    for forbidden in (
        "analyse_structure",
        "analyse_protected_structure",
        "analyse_dealing_ranges",
        "StructureAnalysis",
        "StructureBias",
        "current_bias",
        "breaks",
    ):
        assert forbidden not in names, forbidden


# --------------------------------------------------------------------------
# §1, §3: one formation authority, and an immutable zone
# --------------------------------------------------------------------------


def test_formation_is_not_reimplemented_here() -> None:
    """§1, §2. The module calls the formation engine and never duplicates it."""
    names = identifiers()

    assert "analyse_order_blocks" in names
    for forbidden in (
        "body_direction",
        "source_body_for",
        "zone_of",
        "_source_candle",
        "_order_block_id",
        "OrderBlockZoneBasis",
    ):
        assert forbidden not in names, f"{forbidden} would be a second formation authority"


def test_the_zone_is_read_and_never_recomputed() -> None:
    """§3. Lifecycle uses ``lower``, ``upper`` and ``midpoint`` exactly as handed over."""
    source = SOURCE.read_text(encoding="utf-8")

    for reading in ("block.lower", "block.upper", "block.midpoint"):
        assert reading in source
    for forbidden in ("source_high", "source_low", "min(", "max(", "full_candle_bounds"):
        assert forbidden not in source, f"{forbidden} would be redrawing the zone"


def test_no_zone_is_widened_narrowed_or_padded() -> None:
    for forbidden in ("widen", "narrow", "pad", "buffer", "expand", "shrink", "adjust"):
        assert not any(forbidden in name.lower() for name in identifiers()), forbidden


def test_the_state_carries_the_block_unchanged() -> None:
    """§3. The formation record travels with the state rather than being copied out."""
    fields = set(ict_order_block_lifecycle.OrderBlockState.__dataclass_fields__)

    assert "order_block" in fields
    for forbidden in ("lower", "upper", "midpoint", "width", "zone_basis"):
        assert forbidden not in fields, f"{forbidden} would be a second copy of the zone"


# --------------------------------------------------------------------------
# §58: no candidate semantics
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        "score",
        "rank",
        "strength",
        "quality",
        "confluence",
        "candidate",
        "seo",
        "bai",
        "entry_price",
        "entry_zone",
        "stop_loss",
        "take_profit",
        "target",
        "support",
        "resistance",
    ],
)
def test_no_trade_vocabulary(forbidden: str) -> None:
    """§58. A mitigated block is a fact about the past, not an instruction."""
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in field_names())
    assert not any(forbidden in name.lower() for name in ict_order_block_lifecycle.__all__)


def test_buy_and_sell_appear_in_no_identifier() -> None:
    for name in identifiers() | field_names():
        parts = name.lower().split("_")
        assert "buy" not in parts, name
        assert "sell" not in parts, name


@pytest.mark.parametrize("forbidden", ["combined", "override", "higher_timeframe", "synthesis"])
def test_no_multi_timeframe_synthesis(forbidden: str) -> None:
    """§53. Whether an H4 state governs an M15 one is a synthesis rule."""
    assert not any(forbidden in name.lower() for name in identifiers())


def test_the_snapshot_entry_point_demands_one_named_timeframe() -> None:
    signature = inspect.signature(ict_order_block_lifecycle.analyse_snapshot_order_block_lifecycle)

    assert "timeframe" in signature.parameters
    assert signature.parameters["timeframe"].default is inspect.Parameter.empty


# --------------------------------------------------------------------------
# §17-§18: one invalidation rule, close-confirmed
# --------------------------------------------------------------------------


def test_there_is_no_second_invalidation_policy() -> None:
    """§18. A wick-based alternative would be a new explicit policy, not a flag."""
    for forbidden in ("wick_invalidation", "invalidation_rule", "invalidation_policy"):
        assert not any(forbidden in name.lower() for name in identifiers()), forbidden


def test_invalidation_reads_the_close_and_nothing_else() -> None:
    """§17, §18. Read as identifiers, so the docstring may say what it refuses."""
    tree = ast.parse(inspect.getsource(ict_order_block_lifecycle.invalidates))
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

    assert {"lower", "upper", "direction"} <= attributes
    for forbidden in ("high", "low", "open", "midpoint"):
        assert forbidden not in attributes, f"invalidation must not consult {forbidden}"


def test_the_touch_predicate_reads_no_close() -> None:
    """§8. Touch is about the traded range, not about where the candle finished."""
    tree = ast.parse(inspect.getsource(ict_order_block_lifecycle.touches))
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

    assert {"lower", "upper"} <= attributes
    assert "close" not in attributes


def test_no_mitigation_rule_reads_a_close_or_a_direction() -> None:
    """§15. All three rules are geometric, over the observed range."""
    for function in (
        ict_order_block_lifecycle.touches,
        ict_order_block_lifecycle.contains_midpoint,
        ict_order_block_lifecycle.covers,
    ):
        tree = ast.parse(inspect.getsource(function))
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert "close" not in attributes, function.__name__
        assert "direction" not in attributes, function.__name__


# --------------------------------------------------------------------------
# §43: source agnosticism
# --------------------------------------------------------------------------


def test_the_module_reads_only_ict_inputs() -> None:
    """Asserted as an exact set, so a sixth dependency fails rather than sneaks in."""
    inside = {module for module in imported_modules() if module.startswith("goldpipeline")}

    assert inside == {
        "goldpipeline.schemas.common",
        "goldpipeline.schemas.ict",
        "goldpipeline.schemas.market",
        "goldpipeline.services.ict_primitives",
        "goldpipeline.services.ict_order_block",
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
    ],
)
def test_the_module_imports_no_provider(forbidden: str) -> None:
    for module in imported_modules():
        assert forbidden.lower() not in module.lower(), module


def test_the_module_reads_no_clock() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    for forbidden in ("datetime.now", "utcnow", "time.time", "utc_now"):
        assert forbidden not in source


def test_the_module_never_names_float() -> None:
    assert "float" not in identifiers()


def test_no_float_literal_appears_in_the_source() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            raise AssertionError(f"float literal {node.value} at line {node.lineno}")


@pytest.mark.parametrize("forbidden", ["round", "quantize", "scaleb", "tick", "pips", "tolerance"])
def test_no_rounding_or_tolerance(forbidden: str) -> None:
    """§8, §17. Exact boundary contact counts, and an exact far-edge close does not."""
    assert not any(forbidden in name.lower() for name in identifiers())


def test_the_branch_registry_knows_about_this_module() -> None:
    from tests.test_ict_structure_guards import ICT_BRANCH

    assert "ict_order_block_lifecycle.py" in ICT_BRANCH


# --------------------------------------------------------------------------
# §1: formation was not disturbed
# --------------------------------------------------------------------------


def test_the_formation_module_gained_no_lifecycle() -> None:
    """§1. Round 6.6d.1's own guards still hold, and its surface is unchanged."""
    surface = set(ict_order_block.OrderBlockAnalysis.__dataclass_fields__) | {
        name for name in vars(ict_order_block.OrderBlockAnalysis) if not name.startswith("_")
    }

    assert surface == {
        "method_version",
        "timeframe",
        "symbol",
        "observed_at",
        "zone_basis",
        "order_blocks",
        "order_block",
        "for_event",
    }
    for forbidden in ("status", "mitigated", "invalidated", "touched"):
        assert not any(
            forbidden in name.lower() for name in ict_order_block.OrderBlock.__dataclass_fields__
        ), forbidden


# --------------------------------------------------------------------------
# §60-§61: nothing shipped changed
# --------------------------------------------------------------------------


def test_trade_plan_is_still_not_a_product() -> None:
    from goldpipeline.domain.errors import ArticleTypeNotReadyError
    from goldpipeline.schemas.article import ArticleType
    from goldpipeline.services.article_routing import SPECS
    from goldpipeline.services.article_runtime import (
        RUNTIMES,
        RevisionRuntime,
        is_dispatchable,
        runtime_for,
    )
    from goldpipeline.services.review_action import STYLE_ACTIVE_TYPES

    assert SPECS[ArticleType.TRADE_PLAN].ready is False
    assert SPECS[ArticleType.TRADE_PLAN].prompt_id is None
    assert is_dispatchable(ArticleType.TRADE_PLAN) is False
    assert RUNTIMES[ArticleType.TRADE_PLAN].revise is RevisionRuntime.NONE
    assert ArticleType.TRADE_PLAN not in STYLE_ACTIVE_TYPES
    with pytest.raises(ArticleTypeNotReadyError):
        runtime_for(ArticleType.TRADE_PLAN)


def test_no_lifecycle_code_is_reachable_from_the_shipped_products() -> None:
    from tests.test_ict_structure_guards import ICT_BRANCH

    root = Path("src/goldpipeline")
    callers = [
        path
        for path in root.rglob("*.py")
        if "ict_order_block_lifecycle" in path.read_text(encoding="utf-8")
        and path.name not in ICT_BRANCH
    ]

    assert callers == []


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
        for word in ("order block", "mitigated", "mitigation", "invalidated"):
            assert word not in text, f"{name} mentions {word!r}"
