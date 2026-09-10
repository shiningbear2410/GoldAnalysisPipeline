"""The things the liquidity engine deliberately does not know.

Round 6.6c.1 §16, §18, §38-§43, §53-§54. Every assertion is about an absence,
and absences rot first: each is a shortcut somebody will take for a good local
reason, and each would turn a geometric observation into a judgement wearing the
same name.

Guards read identifiers out of the parsed module rather than grepping its text,
the lesson from Round 6.6a - prose is allowed to name the thing it refuses, and
this module's docstring refuses several by name.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from goldpipeline.services import ict_liquidity

SOURCE = Path("src/goldpipeline/services/ict_liquidity.py")


def identifiers() -> set[str]:
    """Every name the module uses, docstrings and other string literals excluded."""
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
# §18: geometry, not intent
# --------------------------------------------------------------------------


@pytest.mark.parametrize("forbidden", ["stop_hunt", "stophunt", "manipulation", "smart_money"])
def test_a_sweep_makes_no_claim_about_intent(forbidden: str) -> None:
    """The engine sees a wick beyond a band and a close back inside it.

    Who was on the other side, and whether anyone meant to take those orders, is
    not observable from candles. ``STOP_HUNT`` would be asserting a motive; the
    product may say something friendlier later, from an event that stayed honest.
    """
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in member.value.lower() for member in ict_liquidity.LiquidityEventType)


def test_the_event_vocabulary_is_exactly_two_observations() -> None:
    assert {member.value for member in ict_liquidity.LiquidityEventType} == {
        "WICK_SWEEP",
        "CLOSE_THROUGH",
    }


# --------------------------------------------------------------------------
# §16: pool status vocabulary
# --------------------------------------------------------------------------


def test_the_status_enum_has_exactly_three_values() -> None:
    assert {member.value for member in ict_liquidity.PoolStatus} == {
        "ACTIVE",
        "SWEPT",
        "CLOSED_THROUGH",
    }


@pytest.mark.parametrize("forbidden", ["invalid", "mitigated", "broken_structure", "breaker"])
def test_no_status_imports_a_concept_this_round_does_not_implement(forbidden: str) -> None:
    """``MITIGATED`` by which order? ``BROKEN`` relative to which swing?

    Each of those words carries a rule that does not exist yet, and a status
    named for it would be a promise the code cannot keep.
    """
    assert not any(forbidden in member.value.lower() for member in ict_liquidity.PoolStatus)
    assert not any(forbidden in name.lower() for name in identifiers())


# --------------------------------------------------------------------------
# §39, §41-§43: nothing borrowed from later rounds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("forbidden", ["atr", "true_range", "wilder"])
def test_no_atr_derived_tolerance(forbidden: str) -> None:
    """§39. ATR exists from 6.6a and is deliberately not wired in.

    ``tolerance = ATR * fraction`` may well be the right production policy, but
    the fraction is a product parameter. Deriving it inside the engine would
    make a pool's membership depend on a number nobody chose on purpose.
    """
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize("forbidden", ["dealing_range", "premium", "discount", "equilibrium"])
def test_no_dealing_range_or_premium_discount(forbidden: str) -> None:
    """§41. Pools are not range anchors yet; that is the next round."""
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize("forbidden", ["fair_value_gap", "fvg", "touched", "filled"])
def test_no_fvg_lifecycle(forbidden: str) -> None:
    """§42. Gap detection stays exactly as Round 6.6a left it."""
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize(
    "forbidden", ["order_block", "orderblock", "impulse", "mitigation", "last_opposite"]
)
def test_no_order_block_vocabulary(forbidden: str) -> None:
    """§43. An order block needs structure-break ownership rules that do not exist."""
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize(
    "forbidden", ["session", "kill_zone", "killzone", "daily_high", "weekly_high", "dst"]
)
def test_no_session_or_calendar_liquidity(forbidden: str) -> None:
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize("forbidden", ["confluence", "combined", "merge", "strength", "score"])
def test_no_multi_timeframe_synthesis(forbidden: str) -> None:
    """§40. An H4 pool and an H1 pool at one price is a scoring rule, not geometry."""
    assert not any(forbidden in name.lower() for name in ict_liquidity.__all__)
    assert not any(forbidden in name.lower() for name in identifiers())


def test_the_snapshot_entry_point_demands_one_named_timeframe() -> None:
    signature = inspect.signature(ict_liquidity.analyse_snapshot_liquidity)

    assert "timeframe" in signature.parameters
    assert signature.parameters["timeframe"].default is inspect.Parameter.empty


@pytest.mark.parametrize(
    "forbidden", ["buy_zone", "sell_zone", "entry_price", "stop_loss", "take_profit", "seo", "bai"]
)
def test_no_trade_vocabulary(forbidden: str) -> None:
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in ict_liquidity.__all__)


# --------------------------------------------------------------------------
# §5: no float, anywhere
# --------------------------------------------------------------------------


def test_the_module_never_names_float() -> None:
    """A price band decided by binary floating point is a band nobody can reproduce."""
    assert "float" not in identifiers()


def test_no_float_literal_appears_in_the_source() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            raise AssertionError(f"float literal {node.value} at line {node.lineno}")


@pytest.mark.parametrize("forbidden", ["round", "pips", "quantize", "scaleb"])
def test_no_rounding_of_prices(forbidden: str) -> None:
    """Observed swing prices are reported exactly as the market printed them.

    Bare ``pip`` is not on this list: the package is called ``goldpipeline``,
    which contains it. A guard that fired on its own project name would teach
    people to work around guards.
    """
    assert not any(forbidden in name.lower() for name in identifiers())


# --------------------------------------------------------------------------
# §38: source agnosticism
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
def test_the_liquidity_module_imports_no_provider(forbidden: str) -> None:
    for module in imported_modules():
        assert forbidden.lower() not in module.lower(), module


def test_the_liquidity_module_reads_no_clock() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    for forbidden in ("datetime.now", "utcnow", "time.time", "utc_now"):
        assert forbidden not in source


def test_the_liquidity_module_reads_only_ict_inputs() -> None:
    """Asserted as an exact set, so a fourth dependency fails rather than sneaks in."""
    inside = {module for module in imported_modules() if module.startswith("goldpipeline")}

    assert inside == {
        "goldpipeline.schemas.common",
        "goldpipeline.schemas.ict",
        "goldpipeline.schemas.market",
        "goldpipeline.services.ict_primitives",
        "goldpipeline.services.ict_structure",
    }


def test_there_is_one_swing_authority() -> None:
    """§2. Pivots are never derived here, so structure and liquidity cannot disagree."""
    names = identifiers()

    assert "confirmed_swings" in names, "the 6.6a authority is the one used"
    assert "swing_id" in names, "and the 6.6b identity is the one used"
    for forbidden in ("_pivots", "find_pivots", "detect_swings"):
        assert forbidden not in names


def test_the_liquidity_module_does_not_reach_the_analysis_levels_module() -> None:
    """`levels` serves a published ANALYSIS claim path and uses a different pivot rule."""
    assert not any("levels" in module for module in imported_modules())


def test_liquidity_does_not_consult_structure_state() -> None:
    """§31. Pool formation is geometrically independent of BOS and MSS.

    ``swing_id`` is imported for identity and nothing else - no structure
    analysis, no bias, and no consumed-swing set. A structure break and a
    repeated-swing pool answer different questions, and coupling them would let
    a BOS silently delete liquidity the market can still see.
    """
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    from_structure: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "goldpipeline.services.ict_structure"
        ):
            from_structure.update(alias.name for alias in node.names)

    assert from_structure == {"swing_id"}


# --------------------------------------------------------------------------
# §53-§54: nothing shipped changed
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


def test_no_ict_code_is_reachable_from_the_shipped_products() -> None:
    """An engine no product calls cannot change what anything publishes.

    The branch is allowed to know about itself - this module imports ``swing_id``
    from the structure engine so the project has one swing identity rather than
    two that can drift. Nothing outside the branch may reference any of it, and
    ``test_ict_structure_guards`` states that half.
    """
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
        if "ict_liquidity" in path.read_text(encoding="utf-8") and path.name not in REACHABLE
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
        for word in ("liquidity pool", "buy-side liquidity", "sell-side liquidity", "sweep"):
            assert word not in text, f"{name} mentions {word!r}"


def test_the_6_6a_fair_value_gap_definition_is_untouched() -> None:
    """§42. Detection only, no lifecycle, exactly as that round left it."""
    from goldpipeline.services import ict_primitives

    assert not hasattr(ict_primitives, "GapStatus")
    assert "fair_value_gaps" in ict_primitives.__all__
    for forbidden in ("GapStatus", "gap_lifecycle", "TOUCHED", "FILLED"):
        assert forbidden not in ict_primitives.__all__
