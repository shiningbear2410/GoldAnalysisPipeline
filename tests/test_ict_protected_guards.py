"""The things this round deliberately stops short of.

Round 6.6c.3a §28-§30, §48-§51. The dealing range is the important absence: the
origin and terminal of a structural leg make ``(lower + upper) / 2`` trivial to
write, and writing it would settle the range question by accident - which is
precisely why the brief split the round in two.

Guards read identifiers out of the parsed module rather than grepping its text.
Prose is allowed to name what it refuses, and this module's docstring refuses
several by name.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from goldpipeline.services import ict_protected

SOURCE = Path("src/goldpipeline/services/ict_protected.py")


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
        set(ict_protected.StructuralLeg.__dataclass_fields__)
        | set(ict_protected.ProtectedSwingAssignment.__dataclass_fields__)
        | set(ict_protected.ProtectedStructureAnalysis.__dataclass_fields__)
    )


# --------------------------------------------------------------------------
# §28: the range this round refuses to draw
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    ["dealing_range", "equilibrium", "premium", "discount", "midpoint", "fifty", "retracement"],
)
def test_no_dealing_range_vocabulary(forbidden: str) -> None:
    """§28. The arithmetic is easy; that is exactly the danger.

    A leg has an origin and a terminal, so a midpoint is one line away. Writing
    it here would answer "what range should premium and discount use?" without
    anyone having decided - and the whole reason this round exists separately is
    to settle the anchor semantics first, where they can be argued with.
    """
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in field_names())
    assert not any(forbidden in name.lower() for name in ict_protected.__all__)


def test_a_leg_exposes_a_span_but_no_range() -> None:
    """``span`` is a distance. ``lower``/``upper`` would be a zone."""
    fields = set(ict_protected.StructuralLeg.__dataclass_fields__)

    assert "origin_price" in fields
    assert "terminal_price" in fields
    assert "lower" not in fields
    assert "upper" not in fields


def test_a_leg_does_not_extend_itself() -> None:
    """§27. No name here suggests a terminal that grows."""
    for forbidden in ("extend", "extended", "update_terminal", "grow", "active_range"):
        assert not any(forbidden in name.lower() for name in identifiers())


# --------------------------------------------------------------------------
# §29-§30: no borrowed confirmation, no quality claim
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden", ["fair_value_gap", "fvg", "liquidity", "sweep", "pool", "atr", "volume"]
)
def test_no_confirmation_is_required_from_another_engine(forbidden: str) -> None:
    """§29. A structure event is sufficient to anchor a leg.

    Gaps, pools and displacement may all become context later. None of them may
    decide whether a protected swing exists, or the anchor would silently depend
    on three engines agreeing.
    """
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize(
    "forbidden", ["impulse", "displacement", "strong", "quality", "score", "rank", "strength"]
)
def test_a_leg_makes_no_quality_claim(forbidden: str) -> None:
    """§30. It is a ``StructuralLeg``, not a strong impulse."""
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in ict_protected.__all__)


@pytest.mark.parametrize(
    "forbidden",
    ["support", "resistance", "buy", "sell", "seo", "bai", "entry_price", "stop_loss", "target"],
)
def test_no_trade_vocabulary(forbidden: str) -> None:
    """§3. A protected low is where a bullish leg started, not an instruction."""
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in ict_protected.__all__)
    assert not any(forbidden in member.value.lower() for member in ict_protected.ProtectedSwingType)


@pytest.mark.parametrize(
    "forbidden",
    ["order_block", "orderblock", "breaker", "mitigation", "session", "kill_zone", "candidate"],
)
def test_no_later_round_vocabulary(forbidden: str) -> None:
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize("forbidden", ["confluence", "combined", "merge", "override"])
def test_no_multi_timeframe_synthesis(forbidden: str) -> None:
    """§47. Which timeframe's anchor governs is a synthesis rule with content."""
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not any(forbidden in name.lower() for name in ict_protected.__all__)


def test_the_snapshot_entry_point_demands_one_named_timeframe() -> None:
    signature = inspect.signature(ict_protected.analyse_snapshot_protected_structure)

    assert "timeframe" in signature.parameters
    assert signature.parameters["timeframe"].default is inspect.Parameter.empty


# --------------------------------------------------------------------------
# §13: assignments are facts, not a lifecycle
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden", ["invalidated", "superseded", "broken", "status", "lifecycle", "state_of"]
)
def test_an_assignment_carries_no_lifecycle(forbidden: str) -> None:
    """§13. "Event E established swing S" stays true even after S is taken out.

    A later break may make an assignment stale; it never makes it false. Whether
    a stale one is still usable is a question for the current-anchor rule and,
    later, for whatever builds ranges - not a flag on the historical record.
    """
    assert not any(forbidden in name.lower() for name in field_names())


def test_the_only_status_like_answer_is_derived_not_stored() -> None:
    analysis_fields = set(ict_protected.ProtectedStructureAnalysis.__dataclass_fields__)

    assert "current_protected_assignment_id" in analysis_fields
    assert "current_leg_id" in analysis_fields
    assert callable(ict_protected.current_assignment)


# --------------------------------------------------------------------------
# §49: the Decimal identity question, answered
# --------------------------------------------------------------------------


def test_no_price_enters_any_identity_here() -> None:
    """§49. So no fourth copy of the canonicalisation logic was introduced.

    Assignment identity is built from a swing id and an event id; leg identity
    adds a direction and a timestamp. None of those carries a number, so
    ``Decimal("4000")`` and ``Decimal("4000.00")`` cannot produce two ids for
    one thing - the failure mode Round 6.6c.2 found in the gap identity.

    Extracting a shared helper was therefore not triggered. Doing it anyway
    would have meant touching ``ict_liquidity`` and ``ict_fvg`` for tidiness
    alone, and the brief is explicit that no existing identity may change for
    that reason. It stays in the backlog for the round that needs it.
    """
    digest = inspect.getsource(ict_protected._digest)
    assert "normalize" not in digest

    assignment_source = inspect.getsource(ict_protected._assignment_for)
    leg_source = inspect.getsource(ict_protected._leg_for)
    for source in (assignment_source, leg_source):
        assert "swing_price" not in source.split("_digest(")[-1].split(")")[0]


def test_the_existing_identities_were_not_disturbed() -> None:
    """The other half of §49: nothing was refactored under the older engines."""
    from goldpipeline.services import ict_fvg

    assert "normalize" in inspect.getsource(ict_fvg._canonical)
    liquidity_source = Path("src/goldpipeline/services/ict_liquidity.py").read_text(
        encoding="utf-8"
    )
    assert "self.price_tolerance.normalize()" in liquidity_source


# --------------------------------------------------------------------------
# §48: source agnosticism
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


def test_the_module_reads_only_ict_inputs() -> None:
    """Asserted as an exact set, so a fifth dependency fails rather than sneaks in."""
    inside = {module for module in imported_modules() if module.startswith("goldpipeline")}

    assert inside == {
        "goldpipeline.schemas.common",
        "goldpipeline.schemas.ict",
        "goldpipeline.schemas.market",
        "goldpipeline.services.ict_primitives",
        "goldpipeline.services.ict_structure",
    }


def test_there_is_one_structure_authority() -> None:
    """§1. No parallel approximation of historical active levels was built."""
    names = identifiers()

    assert "analyse_structure" in names
    assert "opposite_active_swing_id" in names
    for forbidden in ("_latest_of", "eligible_before", "confirmed_swings", "annotate_swings"):
        assert forbidden not in names, f"{forbidden} would be a second replay"


def test_no_float_appears(  # noqa: D103
) -> None:
    assert "float" not in identifiers()

    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            raise AssertionError(f"float literal {node.value} at line {node.lineno}")


# --------------------------------------------------------------------------
# §50-§51: nothing shipped changed
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


def test_no_protected_code_is_reachable_from_the_shipped_products() -> None:
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
        if "ict_protected" in path.read_text(encoding="utf-8") and path.name not in REACHABLE
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
        for word in ("protected low", "protected high", "structural leg", "dealing range"):
            assert word not in text, f"{name} mentions {word!r}"
