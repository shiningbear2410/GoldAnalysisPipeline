"""The things the order-block engine deliberately stops short of.

Round 6.6d.1 §2, §17, §24-§27, §51-§56. The important absence this round is the
lifecycle: with a zone in hand, "has price touched it?" is three lines away, and
writing them would ship a mitigation policy nobody has argued about - including
the part nobody agrees on, which is whether a wick counts.

Guards read identifiers out of the parsed module rather than grepping its text,
so prose is free to name what it refuses.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from goldpipeline.services import ict_order_block

SOURCE = Path("src/goldpipeline/services/ict_order_block.py")


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
        set(ict_order_block.OrderBlock.__dataclass_fields__)
        | set(ict_order_block.OrderBlockAnalysis.__dataclass_fields__)
        | set(ict_order_block.OrderBlockConfig.__dataclass_fields__)
    )


# --------------------------------------------------------------------------
# §52: formation only
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        "mitigated",
        "mitigation",
        "invalidated",
        "invalidation",
        "breaker",
        "rejection_block",
        "propulsion",
        "refined",
        "touch_count",
        "touched",
        "respected",
        "tested",
    ],
)
def test_no_order_block_lifecycle_vocabulary(forbidden: str) -> None:
    """§52. Mitigated by a wick or by a body? Invalidated by which close?

    Round 6.6d.2 owns those questions. A field named ``mitigated`` here would
    answer them by implication, and every later round would inherit the answer
    without anybody having chosen it.
    """
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in field_names())
    assert not any(forbidden in name.lower() for name in ict_order_block.__all__)


def test_an_order_block_carries_no_status_at_all() -> None:
    """§18, §52. There is no lifecycle state, not even a placeholder."""
    for forbidden in ("status", "state", "is_valid", "active", "alive", "fresh", "stale"):
        assert not any(forbidden in name.lower() for name in field_names()), forbidden


def test_the_analysis_reports_no_current_order_block() -> None:
    """§52. "Which one is current?" is a freshness rule, and freshness is later.

    The whole public surface is pinned as an exact set, so a ``current_block`` or
    an ``active_order_block_id`` fails here rather than quietly shipping a
    freshness policy under a lookup's name.
    """
    analysis = ict_order_block.OrderBlockAnalysis
    surface = set(analysis.__dataclass_fields__) | {
        name for name in vars(analysis) if not name.startswith("_")
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


# --------------------------------------------------------------------------
# §17: two bases and no more
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    ["open_to_low", "open_to_high", "fifty", "half_body", "mean_threshold", "wick_only"],
)
def test_no_hybrid_zone_basis(forbidden: str) -> None:
    """§17. Those may be product policies later, if evidence justifies them."""
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(
        forbidden in member.value.lower() for member in ict_order_block.OrderBlockZoneBasis
    )


def test_the_midpoint_is_not_dressed_up_as_a_reading() -> None:
    """§19. It is half way between two edges, and this round defines no use for it."""
    for forbidden in ("consequent", "encroachment", "ce_level", "equilibrium"):
        assert not any(forbidden in name.lower() for name in identifiers()), forbidden
        assert not any(forbidden in name.lower() for name in field_names()), forbidden


# --------------------------------------------------------------------------
# §24-§27: no other engine decides whether a block forms
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        "premium",
        "discount",
        "dealing_range",
        "analyse_dealing_ranges",
        "locate",
        "price_location",
    ],
)
def test_no_dealing_range_filter(forbidden: str) -> None:
    """§24. A bullish block in discount may be more relevant. It is not more real."""
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize("forbidden", ["fair_value_gap", "fvg", "imbalance", "displacement", "gap"])
def test_no_fair_value_gap_requirement(forbidden: str) -> None:
    """§25. An FVG may become a quality signal; it does not decide qualification."""
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize(
    "forbidden", ["liquidity", "sweep", "pool", "bsl", "ssl", "wick_sweep", "close_through"]
)
def test_no_liquidity_sweep_requirement(forbidden: str) -> None:
    """§26. Later ranking may reward that context. Formation is event-scoped."""
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize(
    "forbidden", ["atr", "wilder", "true_range", "volatility", "threshold", "volume"]
)
def test_no_atr_or_displacement_filter(forbidden: str) -> None:
    """§27. The structure engine already decided the event exists."""
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize("forbidden", ["session", "kill_zone", "killzone", "daily", "weekly"])
def test_no_session_or_calendar_context(forbidden: str) -> None:
    assert not any(forbidden in name.lower() for name in identifiers())


def test_no_daylight_saving_reasoning() -> None:
    """Token-matched, because "dst" hides inside ``ProtectedStructureAnalysis``.

    Narrowing the guard rather than the code: the module genuinely does read a
    protected-structure analysis, and a substring rule that forbade that would
    be a rule about spelling.
    """
    for name in identifiers() | field_names():
        parts = name.lower().replace("-", "_").split("_")
        assert "dst" not in parts, name


# --------------------------------------------------------------------------
# §8, §53: no scoring, no candidate semantics
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    ["score", "strength", "rank", "quality", "confluence", "weight", "best", "strongest"],
)
def test_no_scoring_of_any_kind(forbidden: str) -> None:
    """§8, §53. Recency inside the leg is the whole selection rule."""
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in field_names())


@pytest.mark.parametrize(
    "forbidden",
    [
        "entry_price",
        "entry_zone",
        "entry_level",
        "stop_loss",
        "take_profit",
        "seo",
        "bai",
        "candidate",
        "support",
        "resistance",
    ],
)
def test_no_trade_vocabulary(forbidden: str) -> None:
    """§3, §53. A bullish order block is a fact about the past, not an instruction."""
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in field_names())
    assert not any(forbidden in name.lower() for name in ict_order_block.__all__)


def test_buy_and_sell_appear_in_no_identifier() -> None:
    """Token-matched: "buy" hides inside no word here, but "sell" would hide in nothing
    either, so both are checked as whole tokens to keep the guard honest."""
    for name in identifiers() | field_names():
        parts = name.lower().split("_")
        assert "buy" not in parts, name
        assert "sell" not in parts, name


@pytest.mark.parametrize("forbidden", ["combined", "merge", "override", "higher_timeframe"])
def test_no_multi_timeframe_synthesis(forbidden: str) -> None:
    """§50. Which timeframe's zone a plan uses is a synthesis rule with real content."""
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in ict_order_block.__all__)


def test_the_snapshot_entry_point_demands_one_named_timeframe() -> None:
    signature = inspect.signature(ict_order_block.analyse_snapshot_order_blocks)

    assert "timeframe" in signature.parameters
    assert signature.parameters["timeframe"].default is inspect.Parameter.empty


# --------------------------------------------------------------------------
# §2: one structural authority
# --------------------------------------------------------------------------


def test_no_second_structure_authority_is_built_here() -> None:
    """§2. Legs are consumed, never recomputed.

    The module may call the two analysis entry points and read their results. It
    may not reach for the primitives underneath them - choosing a swing, walking
    pivots or detecting a break here would be a second engine that could drift
    from the first.
    """
    names = identifiers()

    assert "analyse_protected_structure" in names
    assert "analyse_structure" in names
    for forbidden in (
        "confirmed_swings",
        "annotate_swings",
        "_latest_of",
        "eligible_before",
        "protected_type_for",
        "current_assignment",
        "bootstrap_bias",
        "classify_break",
    ):
        assert forbidden not in names, f"{forbidden} would be a second authority"


def test_no_swing_identity_is_computed_here() -> None:
    """The narrower half of §2.

    The module compares ``leg.origin_swing_id`` with ``assignment.swing_id`` to
    prove they describe one anchor, which is a consistency check on values it
    was handed. Calling :func:`~goldpipeline.services.ict_structure.swing_id`
    would be different in kind - it would mint an identity here - so the import,
    not the attribute name, is what this forbids.
    """
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)

    assert "swing_id" not in imported
    assert "swing_id" not in {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_the_module_reads_only_ict_inputs() -> None:
    """Asserted as an exact set, so a seventh dependency fails rather than sneaks in."""
    inside = {module for module in imported_modules() if module.startswith("goldpipeline")}

    assert inside == {
        "goldpipeline.schemas.common",
        "goldpipeline.schemas.ict",
        "goldpipeline.schemas.market",
        "goldpipeline.services.ict_primitives",
        "goldpipeline.services.ict_protected",
        "goldpipeline.services.ict_structure",
    }


def test_the_branch_registry_knows_about_this_module() -> None:
    from tests.test_ict_structure_guards import ICT_BRANCH

    assert "ict_order_block.py" in ICT_BRANCH


# --------------------------------------------------------------------------
# §4-§5, §14-§15: exact prices
# --------------------------------------------------------------------------


def test_the_module_never_names_float() -> None:
    """§5. A body direction decided in binary floating point is not reproducible."""
    assert "float" not in identifiers()


def test_no_float_literal_appears_in_the_source() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            raise AssertionError(f"float literal {node.value} at line {node.lineno}")


@pytest.mark.parametrize("forbidden", ["round", "quantize", "scaleb", "tick", "pips", "padding"])
def test_no_rounding_or_padding_of_prices(forbidden: str) -> None:
    """§14. The zone is the candle, exactly."""
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize("forbidden", ["tolerance", "epsilon", "approx", "nearly", "buffer"])
def test_no_tolerance_anywhere(forbidden: str) -> None:
    """§4. An exact doji is neutral, and a one-cent body is a body."""
    assert not any(forbidden in name.lower() for name in identifiers())


def test_body_direction_reads_only_open_and_close() -> None:
    """§5. Not colour, not a provider flag, not the neighbouring candle.

    Read as identifiers, not as text: the docstring is allowed to say what the
    function refuses to look at, which is the point of writing it down.
    """
    tree = ast.parse(inspect.getsource(ict_order_block.body_direction))
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

    assert {"open", "close"} <= attributes
    for forbidden in ("high", "low", "previous", "previous_close", "colour", "color"):
        assert forbidden not in attributes


# --------------------------------------------------------------------------
# §51: source agnosticism
# --------------------------------------------------------------------------


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


def test_provider_metadata_reaches_no_identity() -> None:
    """§40. Provenance is on the snapshot, and never in a preimage."""
    source = inspect.getsource(ict_order_block._order_block_id)

    for forbidden in ("provider", "provider_symbol"):
        assert forbidden not in source


# --------------------------------------------------------------------------
# §55-§56: nothing shipped changed
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


def test_no_order_block_code_is_reachable_from_the_shipped_products() -> None:
    from tests.test_ict_structure_guards import ICT_BRANCH

    root = Path("src/goldpipeline")
    callers = [
        path
        for path in root.rglob("*.py")
        if "ict_order_block" in path.read_text(encoding="utf-8") and path.name not in ICT_BRANCH
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
        for word in ("order block", "order-block", "zone basis", "source candle"):
            assert word not in text, f"{name} mentions {word!r}"
