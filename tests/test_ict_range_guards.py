"""The things the dealing-range engine deliberately stops short of.

Round 6.6c.3b §2, §30-§31, §43, §53-§57. The important absence this time is OTE
and the rest of the Fibonacci array: with a lower, an upper and an equilibrium
in hand, 62% and 79% are one line away, and writing them would ship a
retracement policy nobody argued about.

Guards read identifiers out of the parsed module rather than grepping its text.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from goldpipeline.services import ict_range

SOURCE = Path("src/goldpipeline/services/ict_range.py")


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
    return set(ict_range.DealingRange.__dataclass_fields__) | set(
        ict_range.DealingRangeAnalysis.__dataclass_fields__
    )


# --------------------------------------------------------------------------
# §30: only three levels
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    ["optimal_trade", "fibonacci", "sixty_two", "seventy_nine", "golden", "retracement"],
)
def test_no_optimal_trade_entry_or_fibonacci_array(forbidden: str) -> None:
    """§30. Lower, equilibrium and upper. Nothing between them.

    62%, 70.5% and 79% are a retracement policy. Adding them here would settle
    where entries live before any round has argued about it, and they would
    arrive wearing the same authority as the geometry.
    """
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in field_names())
    assert not any(forbidden in name.lower() for name in ict_range.__all__)


def test_ote_is_not_a_name_here() -> None:
    """The same ban, matched as a token rather than a substring.

    ``ote`` lives inside "protected", which this module says constantly. A guard
    that fired on its own vocabulary would teach people to work around guards,
    so this one looks for the abbreviation as an actual name.
    """
    for name in identifiers() | field_names() | set(ict_range.__all__):
        lowered = name.lower()
        assert lowered != "ote"
        assert not lowered.startswith("ote_")
        assert not lowered.endswith("_ote")
        assert "_ote_" not in lowered


def test_a_range_exposes_exactly_three_levels() -> None:
    fields = set(ict_range.DealingRange.__dataclass_fields__)

    assert {"lower", "equilibrium", "upper"} <= fields
    for forbidden in ("level_62", "level_79", "levels", "grid", "zones"):
        assert forbidden not in fields


def test_the_location_enum_has_exactly_five_values() -> None:
    assert {member.value for member in ict_range.PriceLocation} == {
        "BELOW_RANGE",
        "DISCOUNT",
        "EQUILIBRIUM",
        "PREMIUM",
        "ABOVE_RANGE",
    }


def test_the_status_enum_has_exactly_two_values() -> None:
    """§13. A range is replaced, never invalidated, mitigated, broken or filled."""
    assert {member.value for member in ict_range.RangeStatus} == {"ACTIVE", "SUPERSEDED"}


@pytest.mark.parametrize("forbidden", ["invalid", "mitigat", "broken", "filled", "expired"])
def test_no_status_imports_another_concept(forbidden: str) -> None:
    assert not any(forbidden in member.value.lower() for member in ict_range.RangeStatus)
    assert not any(forbidden in name.lower() for name in identifiers())


# --------------------------------------------------------------------------
# §31, §54: no candidate filtering, no order blocks
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden", ["candidate", "confluence", "score", "rank", "quality", "filter_by"]
)
def test_no_candidate_filtering(forbidden: str) -> None:
    """§31. "A bullish gap must be in discount" is a rule for the candidate engine.

    This module exposes where a price sits. Deciding that some zones only count
    in one half of a range is a policy, and it belongs where it can be argued
    with rather than inside the geometry that would make it look inevitable.
    """
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in ict_range.__all__)


@pytest.mark.parametrize(
    "forbidden",
    ["order_block", "orderblock", "breaker", "last_opposite", "fair_value_gap", "fvg"],
)
def test_no_order_block_or_gap_vocabulary(forbidden: str) -> None:
    """§54. Round 6.6d owns order blocks."""
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize(
    "forbidden", ["liquidity", "sweep", "pool", "atr", "volume", "session", "kill_zone", "daily"]
)
def test_no_other_engine_participates(forbidden: str) -> None:
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize(
    "forbidden",
    ["buy", "sell", "seo", "bai", "support", "resistance", "entry_price", "stop_loss"],
)
def test_no_trade_vocabulary(forbidden: str) -> None:
    """§3, §29. Discount is the lower half of a range, not an instruction."""
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in ict_range.__all__)
    assert not any(forbidden in member.value.lower() for member in ict_range.PriceLocation)


@pytest.mark.parametrize("forbidden", ["combined", "merge", "override", "higher_timeframe"])
def test_no_multi_timeframe_synthesis(forbidden: str) -> None:
    """§51. Which timeframe's range premium is measured against is a synthesis rule."""
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in ict_range.__all__)


def test_the_snapshot_entry_point_demands_one_named_timeframe() -> None:
    signature = inspect.signature(ict_range.analyse_snapshot_dealing_ranges)

    assert "timeframe" in signature.parameters
    assert signature.parameters["timeframe"].default is inspect.Parameter.empty


# --------------------------------------------------------------------------
# §2: one protected-anchor authority
# --------------------------------------------------------------------------


def test_no_second_anchor_authority_is_built_here() -> None:
    """§2. Anchors are consumed, never recomputed.

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
        "swing_id",
        "protected_type_for",
        "current_assignment",
    ):
        assert forbidden not in names, f"{forbidden} would be a second authority"


def test_the_module_reads_only_ict_inputs() -> None:
    """Asserted as an exact set, so a sixth dependency fails rather than sneaks in."""
    inside = {module for module in imported_modules() if module.startswith("goldpipeline")}

    assert inside == {
        "goldpipeline.schemas.common",
        "goldpipeline.schemas.ict",
        "goldpipeline.schemas.market",
        "goldpipeline.services.ict_primitives",
        "goldpipeline.services.ict_protected",
        "goldpipeline.services.ict_structure",
    }


# --------------------------------------------------------------------------
# §25, §43: exact prices
# --------------------------------------------------------------------------


def test_the_module_never_names_float() -> None:
    """§25. An equilibrium computed in binary floating point is not reproducible."""
    assert "float" not in identifiers()


def test_no_float_literal_appears_in_the_source() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            raise AssertionError(f"float literal {node.value} at line {node.lineno}")


@pytest.mark.parametrize("forbidden", ["round", "quantize", "scaleb", "tick", "pips"])
def test_no_rounding_of_prices(forbidden: str) -> None:
    """§25. The equilibrium of a two-decimal band may carry a third decimal."""
    assert not any(forbidden in name.lower() for name in identifiers())


def test_no_price_enters_the_range_identity() -> None:
    """§43. So no fourth Decimal canonicalisation helper was introduced.

    The preimage is the symbol, timeframe, method version, assignment id and leg
    id. The extending terminal is deliberately absent too - a range that has run
    further is still the same range.
    """
    source = inspect.getsource(ict_range._range_id)

    assert "normalize" not in source
    joined = source.split('"|".join(')[1]
    for forbidden in ("price", "terminal", "lower", "upper", "equilibrium"):
        assert forbidden not in joined


def test_the_existing_identities_were_not_disturbed() -> None:
    """The other half of §43: nothing was refactored under the older engines."""
    from goldpipeline.services import ict_fvg

    assert "normalize" in inspect.getsource(ict_fvg._canonical)
    liquidity = Path("src/goldpipeline/services/ict_liquidity.py").read_text(encoding="utf-8")
    assert "self.price_tolerance.normalize()" in liquidity


# --------------------------------------------------------------------------
# §53: source agnosticism
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


# --------------------------------------------------------------------------
# §56-§57: nothing shipped changed
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


def test_no_range_code_is_reachable_from_the_shipped_products() -> None:
    from tests.test_ict_structure_guards import ICT_BRANCH

    root = Path("src/goldpipeline")
    callers = [
        path
        for path in root.rglob("*.py")
        if "ict_range" in path.read_text(encoding="utf-8") and path.name not in ICT_BRANCH
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
        for word in ("dealing range", "equilibrium", "premium", "discount"):
            assert word not in text, f"{name} mentions {word!r}"
