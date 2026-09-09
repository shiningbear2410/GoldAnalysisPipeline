"""The things the composite deliberately stops short of.

Round 6.6e.1 §13-§14, §21-§22, §39-§40, §45-§47. Two absences matter most here.

The first is **semantics**: this module orchestrates, and the moment it grows a
pivot rule or an ATR formula of its own there are two definitions of the same
thing in the codebase - which is the failure the whole round exists to prevent,
reappearing one layer up.

The second is **synthesis**: with five timeframes in one object, an
``overall_bias`` is three lines away and would ship an unargued trading opinion
wearing the same authority as the geometry.

Guards read identifiers out of the parsed module rather than grepping its text,
so prose is free to name what it refuses.
"""

from __future__ import annotations

import ast
import inspect
from decimal import Decimal
from pathlib import Path

import pytest

from goldpipeline.services import ict_composite
from goldpipeline.services.ict_composite import IctCompositeConfig, IctCompositeError
from goldpipeline.services.ict_order_block import OrderBlockZoneBasis
from goldpipeline.services.ict_order_block_lifecycle import OrderBlockMitigationRule

SOURCE = Path("src/goldpipeline/services/ict_composite.py")


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
        set(ict_composite.IctCompositeAnalysis.__dataclass_fields__)
        | set(ict_composite.IctTimeframeAnalysis.__dataclass_fields__)
        | set(ict_composite.IctCompositeConfig.__dataclass_fields__)
    )


def settings(**kwargs: object) -> IctCompositeConfig:
    base: dict[str, object] = {
        "swing_left_bars": 2,
        "swing_right_bars": 2,
        "atr_period": 14,
        "liquidity_price_tolerance": Decimal("0.5"),
        "order_block_zone_basis": OrderBlockZoneBasis.FULL_CANDLE,
        "order_block_mitigation_rule": OrderBlockMitigationRule.TOUCH,
    }
    base.update(kwargs)
    return IctCompositeConfig(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# §13-§14: the config states everything and defaults nothing
# --------------------------------------------------------------------------


def test_the_config_has_no_defaults_at_all() -> None:
    """§13, §14. Six fields, six required arguments."""
    from dataclasses import MISSING

    fields = IctCompositeConfig.__dataclass_fields__

    assert list(fields) == [
        "swing_left_bars",
        "swing_right_bars",
        "atr_period",
        "liquidity_price_tolerance",
        "order_block_zone_basis",
        "order_block_mitigation_rule",
    ]
    for name, field in fields.items():
        assert field.default is MISSING, name
        assert field.default_factory is MISSING, name

    with pytest.raises(TypeError):
        IctCompositeConfig()  # type: ignore[call-arg]


def test_no_module_level_default_config_exists() -> None:
    """§14. Nothing in ``src`` names a production policy for anyone to inherit."""
    module_level = {
        name for name, value in vars(ict_composite).items() if isinstance(value, IctCompositeConfig)
    }

    assert module_level == set()
    for forbidden in ("DEFAULT_CONFIG", "PRODUCTION_CONFIG", "STANDARD_CONFIG"):
        assert not hasattr(ict_composite, forbidden)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"swing_left_bars": 0}, "pivot windows must be positive"),
        ({"swing_right_bars": 0}, "pivot windows must be positive"),
        ({"swing_left_bars": -1}, "pivot windows must be positive"),
        ({"atr_period": 0}, "ATR period must be positive"),
        ({"atr_period": -3}, "ATR period must be positive"),
        ({"liquidity_price_tolerance": Decimal("-0.01")}, "cannot be negative"),
    ],
)
def test_the_config_validates_its_numbers(kwargs: dict[str, object], match: str) -> None:
    """§13."""
    with pytest.raises(IctCompositeError, match=match):
        settings(**kwargs)


def test_a_float_tolerance_is_refused() -> None:
    """§13. A float tolerance would make pool membership depend on binary rounding."""
    with pytest.raises(IctCompositeError, match="must be a Decimal"):
        settings(liquidity_price_tolerance=0.5)


def test_a_zero_tolerance_is_allowed() -> None:
    """Exact-equality pooling is a real policy, not a mistake."""
    assert settings(liquidity_price_tolerance=Decimal("0")).liquidity_price_tolerance == 0


def test_the_config_reuses_the_existing_enums() -> None:
    """§13. A second ``FULL_CANDLE`` would be a second definition waiting to drift."""
    from goldpipeline.services import ict_order_block, ict_order_block_lifecycle

    # Reached through the module namespace on purpose: these are imports the
    # composite uses, not part of the surface it exports.
    basis = vars(ict_composite)["OrderBlockZoneBasis"]
    rule = vars(ict_composite)["OrderBlockMitigationRule"]

    assert basis is ict_order_block.OrderBlockZoneBasis
    assert rule is ict_order_block_lifecycle.OrderBlockMitigationRule

    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert "OrderBlockZoneBasis" not in defined
    assert "OrderBlockMitigationRule" not in defined
    assert not any(
        isinstance(node, ast.ClassDef)
        and any(isinstance(base, ast.Name) and base.id == "StrEnum" for base in node.bases)
        for node in ast.walk(tree)
    ), "the composite defines no enum of its own"


def test_the_config_hands_each_engine_its_own_config_type() -> None:
    """The nested configs are built from the bundle, not duplicated beside it."""
    bundle = settings()

    assert bundle.liquidity.price_tolerance == bundle.liquidity_price_tolerance
    assert bundle.order_block.zone_basis is bundle.order_block_zone_basis
    assert bundle.order_block_lifecycle.mitigation_rule is bundle.order_block_mitigation_rule


# --------------------------------------------------------------------------
# §22: no semantics are reimplemented here
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        "true_range",
        "atr_series",
        "annotate_swings",
        "eligible_before",
        "swings_known_at",
        "classify_break",
        "bootstrap_bias",
        "body_direction",
        "source_body_for",
        "zone_of",
        "protected_type_for",
        "current_assignment",
        "locate",
        "touches",
        "covers",
        "enters",
        "satisfies",
        "invalidates",
        "contains_midpoint",
        "far_edge",
        "side_of",
    ],
)
def test_no_domain_primitive_is_reached_for(forbidden: str) -> None:
    """§22. It calls authorities; it does not reach past them or copy them."""
    assert forbidden not in identifiers()


@pytest.mark.parametrize(
    "forbidden",
    [
        "pivot",
        "swing_high",
        "swing_low",
        "wilder",
        "smoothing",
        "midpoint",
        "equilibrium",
        "premium",
        "discount",
        "invalidat",
        "tolerance_of",
        "source_candle",
    ],
)
def test_no_semantic_vocabulary_appears_in_an_identifier(forbidden: str) -> None:
    """§22. Orchestration names stages; it does not name their internals.

    ``liquidity_price_tolerance`` is a policy the caller states, not a rule this
    module applies, which is why it survives this list while ``tolerance_of``
    would not.
    """
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden


@pytest.mark.parametrize(
    "forbidden", ["mitigates", "is_mitigated", "mitigation_of", "apply_mitigation"]
)
def test_no_mitigation_rule_is_applied_here(forbidden: str) -> None:
    """§22, narrowed to its intent.

    ``order_block_mitigation_rule`` and the ``order_block_lifecycle`` property
    both name the policy the caller chose and hand it to the engine that owns
    it - which is the opposite of implementing it. Banning the substring would
    have forced the config to describe its own field in code words it does not
    mean, so the guard names the shapes that would actually be a second
    implementation.
    """
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden
    assert "OrderBlockStatus" not in identifiers(), "no status is decided here"


def test_the_module_defines_no_function_that_computes_a_price() -> None:
    """§22. Every function here either builds a result or checks one."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("__")
    }

    assert defined == {
        "liquidity",
        "order_block",
        "order_block_lifecycle",
        "timeframe",
        "require",
        "analyse_ict_timeframe",
        "analyse_ict_composite",
        "_require_consistent",
    }


def test_no_arithmetic_on_prices_happens_here() -> None:
    """§22. No comparison or arithmetic that could constitute a rule."""
    source = SOURCE.read_text(encoding="utf-8")

    for forbidden in ("Decimal(2)", ".high", ".low", ".close", ".open"):
        assert forbidden not in source, forbidden


def test_no_float_literal_appears_in_the_source() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            raise AssertionError(f"float literal {node.value} at line {node.lineno}")


def test_the_module_never_names_float() -> None:
    assert "float" not in identifiers()


# --------------------------------------------------------------------------
# §39-§40: no synthesis, no candidate mapping
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        "overall",
        "higher_timeframe",
        "confluence",
        "agreement",
        "alignment",
        "consensus",
        "dominant",
        "governs",
        "override",
        "combined",
        "merge",
    ],
)
def test_no_multi_timeframe_synthesis(forbidden: str) -> None:
    """§39. Deciding when H4 governs H1 is a real question with real content."""
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in field_names())
    assert not any(forbidden in name.lower() for name in ict_composite.__all__)


def test_the_composite_reports_no_bias_of_its_own() -> None:
    """§39. Five independent readings, and no sixth opinion about them."""
    for forbidden in ("bias", "trend", "direction", "sentiment"):
        assert not any(forbidden in name.lower() for name in field_names()), forbidden


@pytest.mark.parametrize(
    "forbidden",
    [
        "candidate",
        "seo",
        "bai",
        "score",
        "rank",
        "strength",
        "quality",
        "fresh",
        "stale",
        "entry_price",
        "entry_zone",
        "stop_loss",
        "take_profit",
        "target",
        "support",
        "resistance",
    ],
)
def test_no_candidate_or_trade_vocabulary(forbidden: str) -> None:
    """§40. Mapping a bearish block onto a SEO is Round 6.6e.2's decision."""
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in field_names())
    assert not any(forbidden in name.lower() for name in ict_composite.__all__)


def test_buy_and_sell_appear_in_no_identifier() -> None:
    for name in identifiers() | field_names():
        parts = name.lower().split("_")
        assert "buy" not in parts, name
        assert "sell" not in parts, name


def test_the_per_timeframe_result_holds_analyses_not_flattened_fields() -> None:
    """§16. One place for each number, which is the point of the round."""
    fields = set(ict_composite.IctTimeframeAnalysis.__dataclass_fields__)

    assert fields == {
        "method_version",
        "timeframe",
        "symbol",
        "observed_at",
        "series",
        "atr",
        "swings",
        "gaps",
        "fvg_lifecycle",
        "structure",
        "liquidity",
        "protected",
        "dealing_ranges",
        "order_blocks",
        "order_block_lifecycle",
    }


def test_the_composite_result_holds_only_provenance_config_and_timeframes() -> None:
    """§17."""
    assert set(ict_composite.IctCompositeAnalysis.__dataclass_fields__) == {
        "method_version",
        "observed_at",
        "symbol",
        "provider",
        "provider_symbol",
        "config",
        "timeframes",
    }


# --------------------------------------------------------------------------
# §21: source agnosticism
# --------------------------------------------------------------------------


def test_the_module_reads_only_ict_inputs() -> None:
    """Asserted as an exact set, so a tenth dependency fails rather than sneaks in."""
    inside = {module for module in imported_modules() if module.startswith("goldpipeline")}

    assert inside == {
        "goldpipeline.schemas.common",
        "goldpipeline.schemas.ict",
        "goldpipeline.services.ict_fvg",
        "goldpipeline.services.ict_liquidity",
        "goldpipeline.services.ict_order_block",
        "goldpipeline.services.ict_order_block_lifecycle",
        "goldpipeline.services.ict_primitives",
        "goldpipeline.services.ict_protected",
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
def test_the_module_imports_no_provider_or_analysis_path(forbidden: str) -> None:
    """§6, §21. Notably not ``services.levels``, whose ATR serves the ANALYSIS path."""
    for module in imported_modules():
        assert forbidden.lower() not in module.lower(), module


def test_the_module_reads_no_clock() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    for forbidden in ("datetime.now", "utcnow", "time.time", "utc_now"):
        assert forbidden not in source


def test_the_analysis_atr_is_not_the_one_used() -> None:
    """§6. Wilder's ATR from the ICT primitives, not the simple ANALYSIS one."""
    source = inspect.getsource(ict_composite.analyse_ict_timeframe)

    assert "latest_atr" in source
    assert "average_true_range" not in source


# --------------------------------------------------------------------------
# §45-§47: nothing shipped changed
# --------------------------------------------------------------------------


def test_the_branch_registry_knows_about_this_module() -> None:
    from tests.test_ict_structure_guards import ICT_BRANCH

    assert "ict_composite.py" in ICT_BRANCH


def test_no_composite_code_is_reachable_from_the_shipped_products() -> None:
    """§47. The production scheduler must not start doing more work."""
    from tests.test_ict_structure_guards import ICT_BRANCH

    root = Path("src/goldpipeline")
    callers = [
        path
        for path in root.rglob("*.py")
        if "ict_composite" in path.read_text(encoding="utf-8") and path.name not in ICT_BRANCH
    ]

    assert callers == []


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
        for word in ("composite", "order block", "dealing range", "mitigation"):
            assert word not in text, f"{name} mentions {word!r}"


def test_the_analysis_levels_module_still_has_its_own_swing_rule() -> None:
    """§1's locked rule: no domain engine was quietly changed to fit the composite.

    ``services.levels`` resolves pivot ties differently from the ICT primitives
    on purpose - it serves a published claim path. Threading swings through the
    ICT engines must not have touched it.
    """
    from goldpipeline.services import levels

    assert hasattr(levels, "swing_highs")
    source = inspect.getsource(levels)
    assert "confirmed_swings" not in source
    assert "require_swings_for" not in source
