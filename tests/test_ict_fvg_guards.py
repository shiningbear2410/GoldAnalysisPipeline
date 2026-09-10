"""The things the FVG lifecycle deliberately does not know.

Round 6.6c.2 §4, §31-§36, §45, §47-§48. Every assertion is about an absence.
Absences rot first: each is a shortcut somebody will take for a good local
reason, and each would turn an observation into a judgement wearing the same
name.

Guards read identifiers out of the parsed module rather than grepping its text
- the Round 6.6a lesson. Prose is allowed to name what it refuses, and this
module's docstring refuses several by name.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from goldpipeline.services import ict_fvg

SOURCE = Path("src/goldpipeline/services/ict_fvg.py")


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


# --------------------------------------------------------------------------
# §4: the status vocabulary
# --------------------------------------------------------------------------


def test_the_status_enum_has_exactly_three_values() -> None:
    assert {member.value for member in ict_fvg.FvgStatus} == {"OPEN", "TOUCHED", "FILLED"}


@pytest.mark.parametrize(
    "forbidden", ["mitigat", "invalid", "inverted", "inversion", "breaker", "rebalanced", "ifvg"]
)
def test_no_status_imports_a_concept_this_round_does_not_implement(forbidden: str) -> None:
    """``MITIGATED`` by which order? ``INVERTED`` for whose entry?

    Each word carries a rule that does not exist yet, and a status named for one
    would be a promise the code cannot keep. ``TOUCHED`` deliberately avoids
    "mitigated" for exactly that reason - the product may define mitigation
    later, and it must be free to define it differently.
    """
    assert not any(forbidden in member.value.lower() for member in ict_fvg.FvgStatus)
    assert not any(forbidden in name.lower() for name in identifiers())


# --------------------------------------------------------------------------
# §18, §33-§36: nothing borrowed from later rounds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("forbidden", ["atr", "true_range", "wilder", "min_size", "minimum_size"])
def test_no_size_or_atr_filter(forbidden: str) -> None:
    """§33. A tiny gap is still a gap.

    Rejecting one below some fraction of an ATR is a quality filter - a real and
    useful idea - wearing the costume of a definition. Detection truth stays the
    strict three-candle rule from Round 6.6a.
    """
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize(
    "forbidden", ["displacement", "quality", "score", "rank", "strength", "grade"]
)
def test_no_quality_scoring(forbidden: str) -> None:
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in ict_fvg.__all__)


@pytest.mark.parametrize("forbidden", ["touch_count", "reaction", "mitigation_count", "visits"])
def test_no_touch_counting(forbidden: str) -> None:
    """§18. First touch and fill only; a reaction count is a quality signal."""
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize(
    "forbidden", ["bos", "mss", "structure", "swing", "market_structure", "choch"]
)
def test_no_structure_requirement(forbidden: str) -> None:
    """§34. A gap is a gap whether or not a break happened.

    Structure may later score relevance. It must never redefine primitive truth,
    and the surest way to keep that promise is for this module not to be able to
    see it.
    """
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize(
    "forbidden", ["liquidity", "sweep", "pool", "bsl", "ssl", "buy_side", "sell_side"]
)
def test_no_liquidity_requirement(forbidden: str) -> None:
    """§35. No sweep is required before a gap, and none is consulted after."""
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize(
    "forbidden",
    ["dealing_range", "equilibrium", "premium", "discount", "protected_high", "protected_low"],
)
def test_no_dealing_range_vocabulary(forbidden: str) -> None:
    """§45. Those need a protected-swing anchor policy, which is the next round.

    Choosing "latest high and latest low" without one would make the arithmetic
    deterministic and the range selection arbitrary, which is the worse of the
    two failures because it looks rigorous.
    """
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize(
    "forbidden", ["order_block", "orderblock", "impulse", "last_opposite", "session", "kill_zone"]
)
def test_no_order_block_or_session_vocabulary(forbidden: str) -> None:
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize(
    "forbidden",
    ["buy_zone", "sell_zone", "support", "resistance", "seo", "bai", "entry_price", "stop_loss"],
)
def test_no_candidate_or_trade_vocabulary(forbidden: str) -> None:
    """§36. An OPEN bullish gap is not a buy zone, and saying so would be a claim."""
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in ict_fvg.__all__)


@pytest.mark.parametrize("forbidden", ["confluence", "combined", "merge", "cross_timeframe"])
def test_no_multi_timeframe_synthesis(forbidden: str) -> None:
    """§43. An H4 gap and an H1 gap at one price is a scoring rule, not geometry."""
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in ict_fvg.__all__)


def test_the_snapshot_entry_point_demands_one_named_timeframe() -> None:
    signature = inspect.signature(ict_fvg.analyse_snapshot_fvg_lifecycle)

    assert "timeframe" in signature.parameters
    assert signature.parameters["timeframe"].default is inspect.Parameter.empty


# --------------------------------------------------------------------------
# §6: exact prices, no float
# --------------------------------------------------------------------------


def test_the_module_never_names_float() -> None:
    assert "float" not in identifiers()


def test_no_float_literal_appears_in_the_source() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            raise AssertionError(f"float literal {node.value} at line {node.lineno}")


@pytest.mark.parametrize("forbidden", ["round", "quantize", "scaleb", "tolerance", "widen"])
def test_the_band_is_never_adjusted(forbidden: str) -> None:
    """§6. The authoritative band is used exactly, with no padding and no tolerance.

    Liquidity pools have an explicit tolerance because two swings at *nearly*
    one price are still one pool. A gap is a single observed band with two exact
    edges, so there is nothing here for a tolerance to mean.
    """
    assert not any(forbidden in name.lower() for name in identifiers())


# --------------------------------------------------------------------------
# §1-§2, §31: source agnosticism and one detection authority
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
def test_the_lifecycle_module_imports_no_provider(forbidden: str) -> None:
    for module in imported_modules():
        assert forbidden.lower() not in module.lower(), module


def test_the_lifecycle_module_reads_no_clock() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    for forbidden in ("datetime.now", "utcnow", "time.time", "utc_now"):
        assert forbidden not in source


def test_the_lifecycle_module_reads_only_ict_inputs() -> None:
    """Asserted as an exact set, so a fifth dependency fails rather than sneaks in."""
    inside = {module for module in imported_modules() if module.startswith("goldpipeline")}

    assert inside == {
        "goldpipeline.schemas.common",
        "goldpipeline.schemas.ict",
        "goldpipeline.schemas.market",
        "goldpipeline.services.ict_primitives",
    }


def test_there_is_one_gap_detection_authority() -> None:
    """§1. The three-candle test is not reimplemented here.

    Checked structurally rather than by name: the module contains no comparison
    of a candle's low against an earlier candle's high, which is what a second
    detector would have to do. What it does have is a call to the one authority.
    """
    names = identifiers()

    assert "fair_value_gaps" in names
    for forbidden in ("detect_fvg", "detect_gaps", "find_gaps", "_gaps_in"):
        assert forbidden not in names


def test_the_lifecycle_module_does_not_reach_structure_or_liquidity() -> None:
    """§34-§35, at the import graph."""
    assert not any("ict_structure" in module for module in imported_modules())
    assert not any("ict_liquidity" in module for module in imported_modules())
    assert not any("levels" in module for module in imported_modules())


# --------------------------------------------------------------------------
# §47-§48: nothing shipped changed
# --------------------------------------------------------------------------


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


def test_no_lifecycle_code_is_reachable_from_the_shipped_products() -> None:
    """An engine no product calls cannot change what anything publishes."""
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
        if "ict_fvg" in path.read_text(encoding="utf-8") and path.name not in REACHABLE
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
        for word in ("fair value gap", "imbalance", "gap fill", "mitigation"):
            assert word not in text, f"{name} mentions {word!r}"


def test_the_6_6a_detector_is_untouched() -> None:
    """§1. Lifecycle was added beside it, not inside it."""
    from goldpipeline.services import ict_primitives

    assert "fair_value_gaps" in ict_primitives.__all__
    assert not hasattr(ict_primitives, "FvgStatus")
    for forbidden in ("FvgStatus", "GapStatus", "lifecycle", "filled", "touched"):
        assert forbidden not in ict_primitives.__all__

    gap_source = inspect.getsource(ict_primitives.fair_value_gaps)
    assert "last.low > first.high" in gap_source, "the strict bullish rule is unchanged"
    assert "last.high < first.low" in gap_source, "and the strict bearish rule"
