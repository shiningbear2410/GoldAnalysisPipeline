"""The things this engine deliberately does not know.

Round 6.6b §13, §21, §29-§33, §43, §46-§47. Every assertion here is about an
*absence*, and absences are what rot first: each one is a shortcut somebody
will one day take for a good local reason, and each would quietly turn a
geometric fact into a subjective judgement wearing the same name.

The guards read identifiers out of the parsed module rather than grepping its
text. That distinction earned itself in Round 6.6a, where a text guard failed
on a docstring saying exactly the right thing - that a fair value gap is *not*
a buy zone. Prose is allowed to name the thing it is refusing.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from goldpipeline.services import ict_structure

SOURCE = Path("src/goldpipeline/services/ict_structure.py")


def identifiers() -> set[str]:
    """Every name the structure module actually uses, docstrings excluded."""
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
            # `**kwargs` is an ast.keyword whose arg is None; everything else here
            # names something.
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
# §29-§32: no filters smuggled into the definition of truth
# --------------------------------------------------------------------------


@pytest.mark.parametrize("forbidden", ["atr", "true_range", "wilder"])
def test_no_atr_participates_in_a_structure_break(forbidden: str) -> None:
    """A close one cent beyond a level is beyond it.

    Requiring the break to clear some fraction of an ATR would be a strength
    filter - a real and useful idea - wearing the costume of a definition.
    Later rounds may rank breaks by displacement all they like, with the raw
    event in hand and the filter named as their own.
    """
    assert not any(forbidden in name.lower() for name in identifiers())


@pytest.mark.parametrize("forbidden", ["fair_value_gap", "fvg", "gapdirection", "imbalance"])
def test_a_break_does_not_require_a_gap(forbidden: str) -> None:
    """Displacement may become a quality signal. It is not structural truth."""
    assert not any(forbidden in name.lower() for name in identifiers())


def test_no_volume_confirmation(forbidden: str = "volume") -> None:
    """XAUUSD volume is a broker's tick count, not the market's.

    Different venues report wildly different numbers for the same hour of the
    same instrument, so a rule keyed on it would make structure depend on which
    feed answered - the exact thing the provider-neutrality guard exists to
    prevent.
    """
    assert not any(forbidden in name.lower() for name in identifiers())
    assert not hasattr(ict_structure, "volume")


@pytest.mark.parametrize("forbidden", ["news", "sentiment", "headline", "macro", "calendar"])
def test_no_macro_or_news_reaches_market_geometry(forbidden: str) -> None:
    assert not any(forbidden in name.lower() for name in identifiers())


# --------------------------------------------------------------------------
# §33: one timeframe at a time
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden", ["confluence", "overall_bias", "combined", "align", "higher_timeframe"]
)
def test_no_multi_timeframe_synthesis_exists_yet(forbidden: str) -> None:
    """ "H4 bullish and H1 bullish therefore bullish" is a trading opinion.

    It needs a stated rule for disagreement and for which timeframe gates which,
    and shipping one as a convenience helper would bury that decision where
    nobody argues with it.
    """
    assert not any(forbidden in name.lower() for name in ict_structure.__all__)
    assert not any(forbidden in name.lower() for name in identifiers())


def test_the_snapshot_entry_point_demands_one_named_timeframe() -> None:
    """There is no "analyse everything" call that could imply a combined answer."""
    signature = inspect.signature(ict_structure.analyse_snapshot_structure)

    assert "timeframe" in signature.parameters
    assert signature.parameters["timeframe"].default is inspect.Parameter.empty


# --------------------------------------------------------------------------
# §13: one reversal concept, one name
# --------------------------------------------------------------------------


def test_choch_is_not_a_separate_event_type() -> None:
    """The word means three different things depending on who is teaching.

    Sometimes the first counter-trend break, sometimes any counter-trend break,
    sometimes an exact synonym for MSS. Adding it as an enum member would mean
    picking one of those silently and handing the ambiguity to every stage
    downstream. Product copy can map to it later from a rule written down then.
    """
    assert not any("choch" in name.lower() for name in identifiers())
    assert not any("choch" in member.value.lower() for member in ict_structure.BreakClassification)
    assert {member.value for member in ict_structure.BreakClassification} == {
        "INITIAL_BREAK",
        "BOS",
        "MSS",
    }


def test_the_canonical_long_name_is_recorded_once() -> None:
    assert ict_structure.MARKET_STRUCTURE_SHIFT is ict_structure.BreakClassification.MSS


# --------------------------------------------------------------------------
# §11, §21: honest names
# --------------------------------------------------------------------------


@pytest.mark.parametrize("forbidden", ["protected_high", "protected_low", "protected"])
def test_active_levels_are_not_called_protected(forbidden: str) -> None:
    """No protected-swing rule is implemented, so the word would be a claim.

    A protected high in ICT is a specific structural swing with rules about when
    it stops being protected. Borrowing the label for "the most recent unbroken
    high" would be shipping the connotation without the algorithm.
    """
    assert not any(forbidden in name.lower() for name in identifiers())


def test_the_active_levels_use_plain_names() -> None:
    fields = {field for field in ict_structure.StructureAnalysis.__dataclass_fields__}

    assert "active_high" in fields
    assert "active_low" in fields


@pytest.mark.parametrize("forbidden", ["buy", "sell", "seo", "bai", "long", "short"])
def test_a_direction_is_never_named_as_a_trade(forbidden: str) -> None:
    """§11. Direction describes what price did to a level, not what to do about it."""
    assert not any(forbidden in member.value.lower() for member in ict_structure.BreakDirection)
    assert not any(forbidden in member.value.lower() for member in ict_structure.StructureBias)
    assert not any(forbidden in name.lower() for name in ict_structure.__all__)


@pytest.mark.parametrize(
    "forbidden",
    ["entry_price", "entry_zone", "entry_level", "stop_loss", "take_profit", "target", "risk"],
)
def test_no_trade_parameter_vocabulary_exists_here(forbidden: str) -> None:
    """Bare ``entry`` is not on this list, and deliberately.

    It is the loop variable this codebase has always used for "one item of the
    sequence", including in the ICT snapshot model written a round earlier.
    Banning it would force worse names on ordinary iteration to satisfy a guard,
    which is the guard serving itself. What must never appear is the trade
    vocabulary proper - a price to enter at, a stop, a target.
    """
    assert not any(forbidden in name.lower() for name in identifiers())


def test_the_bias_enum_has_exactly_three_values() -> None:
    """No ``RANGE``. Claiming the market is ranging is more than this engine knows."""
    assert {member.value for member in ict_structure.StructureBias} == {
        "NEUTRAL",
        "BULLISH",
        "BEARISH",
    }


# --------------------------------------------------------------------------
# §43: source agnosticism, and separation from the ANALYSIS helper
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    ["tradingview", "metatrader", "MetaTrader5", "websocket", "requests", "httpx", "anthropic"],
)
def test_the_structure_module_imports_no_provider(forbidden: str) -> None:
    for module in imported_modules():
        assert forbidden.lower() not in module.lower(), module


def test_the_structure_module_reads_no_clock() -> None:
    """Every instant arrives as data, so the same bars always give the same answer."""
    source = SOURCE.read_text(encoding="utf-8")

    for forbidden in ("datetime.now", "utcnow", "time.time", "utc_now"):
        assert forbidden not in source


def test_the_structure_module_does_not_reach_the_analysis_levels_module() -> None:
    """The 6.6b audit's main risk, pinned.

    ``levels.classify_structure`` labels the trend from the last two pivots on
    each side, and its bootstrap arithmetic happens to agree with
    :func:`~goldpipeline.services.ict_structure.bootstrap_bias`. Everything else
    about it differs: a four-value vocabulary including ``RANGE``, a pivot
    definition with a deliberate strict-left/non-strict-right tie-break, no
    confirmation instant, and no state that persists across bars. Importing one
    into the other would make a published ANALYSIS number and an ICT structure
    state move together, which is exactly what neither product wants.
    """
    assert not any("levels" in module for module in imported_modules())


def test_the_structure_module_reads_only_candles_and_swings() -> None:
    inside = {module for module in imported_modules() if module.startswith("goldpipeline")}

    assert inside == {
        "goldpipeline.schemas.common",
        "goldpipeline.schemas.ict",
        "goldpipeline.services.ict_primitives",
    }


# --------------------------------------------------------------------------
# §46-§47: nothing shipped changed
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


ICT_BRANCH = {
    "ict.py",
    "ict_primitives.py",
    "ict_structure.py",
    "ict_liquidity.py",
    "ict_fvg.py",
    "ict_protected.py",
    "ict_range.py",
    "ict_order_block.py",
    "ict_order_block_lifecycle.py",
    "ict_composite.py",
    "ict_candidate_source.py",
    "ict_candidate_eligibility.py",
    "ict_candidate_consolidation.py",
    "ict_candidate_features.py",
    # The Trade Analyst, part of the same TRADE_PLAN branch even though its
    # names do not start with "ict_": the domain service, the provider protocol
    # it is injected through, the offline fakes, and the live Anthropic client.
    "trade_analyst.py",
    "trade_analyst_client.py",
    "fake_trade_analyst.py",
    "anthropic_trade_analyst.py",
    # The deterministic end of the branch: selection and the public renderer.
    "trade_plan_selector.py",
    "trade_plan_render.py",
    # Round 6.6h: the production wiring. Live candles at one instant, the
    # versioned policy, the run stage and its gate.
    "mtf_market.py",
    "trade_plan_policy.py",
    "trade_plan_stage.py",
    "trade_plan_gate.py",
    # Round 6.7: the V2 page and the copywriter that writes words around it.
    "trade_plan_presentation.py",
    "trade_plan_copy.py",
    "plan_copy_client.py",
    "fake_plan_copywriter.py",
}
"""Modules of the TRADE_PLAN branch, which are allowed to know about each other.

The registry every round's reachability guard reads, so adding a branch module
means adding it here once rather than relaxing a guard somewhere.
"""

DISPATCH_SEAM = {"orchestrator.py", "article_runtime.py", "cli.py"}
"""The three files outside the branch that Round 6.6h let learn about it.

Until this round the branch was unreachable from anything a product ran, and
every guard below said so. Activation needed a seam, and it is exactly three
files wide: ``article_runtime`` names the runtime, ``orchestrator`` dispatches
to the stage and its gate, and ``cli`` constructs the market source and the
ranking client the way it constructs every other client. Enumerating them keeps
the guards precise - "nothing reaches this branch except the seam we chose" is a
much stronger statement than relaxing the rule, and a fourth caller still fails.
"""


def test_no_structure_code_is_reachable_from_the_shipped_products() -> None:
    """A structure engine no *product* calls cannot change what anything publishes.

    Round 6.6b wrote this as "no other file mentions it at all", which was an
    accurate way to say it while the structure engine was the branch's only
    consumer. Round 6.6c.1 made that too strong: the liquidity engine imports
    ``swing_id`` from here on purpose, so that there is exactly one swing
    identity in the project rather than two that can drift apart.

    So the guard now says what it always meant. The ICT branch may reference
    itself; nothing outside it may reference the branch at all, which is what
    keeps ANALYSIS and NEWS_DIGEST provably untouched by any of this.
    """
    from tests.test_ict_structure_guards import DISPATCH_SEAM

    reachable = ICT_BRANCH | DISPATCH_SEAM
    root = Path("src/goldpipeline")
    callers = [
        path
        for path in root.rglob("*.py")
        if "ict_structure" in path.read_text(encoding="utf-8") and path.name not in reachable
    ]

    assert callers == []


def test_nothing_outside_the_ict_branch_mentions_any_of_it() -> None:
    """The stronger statement, and the one that actually protects the products.

    Until Round 6.6h this was "nobody outside the branch mentions it at all".
    Activation had to make one seam, so it is now "nobody except the two files
    that dispatch it" - and those two are enumerated in :data:`DISPATCH_SEAM`
    rather than the rule being loosened, so a third caller still fails here.
    """
    root = Path("src/goldpipeline")
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if path.name in ICT_BRANCH | DISPATCH_SEAM:
            continue
        text = path.read_text(encoding="utf-8")
        for module in (
            "ict_primitives",
            "ict_structure",
            "ict_liquidity",
            "ict_fvg",
            "ict_protected",
            "ict_range",
            "ict_order_block",
            "ict_order_block_lifecycle",
            "ict_composite",
            "ict_candidate_source",
            "ict_candidate_eligibility",
            "ict_candidate_consolidation",
            "ict_candidate_features",
            "trade_plan_selector",
            "trade_plan_render",
            "trade_plan_policy",
            "trade_plan_stage",
            "trade_plan_gate",
            "mtf_market",
        ):
            if module in text:
                offenders.append(f"{path.name} -> {module}")

    assert offenders == []


def test_the_dispatch_seam_is_exactly_two_files_and_no_more() -> None:
    """§48. What the seam is allowed to be, checked rather than described.

    The orchestrator may name the stage and the gate; the runtime table may name
    the runtime. Neither may reach into an ICT engine directly - a shortcut past
    the composite would put a market rule back inside the pipeline.
    """
    root = Path("src/goldpipeline")

    orchestrator = (root / "services" / "orchestrator.py").read_text(encoding="utf-8")
    assert "trade_plan_stage" in orchestrator
    assert "trade_plan_gate" in orchestrator
    for forbidden in (
        "ict_composite",
        "ict_candidate_source",
        "ict_candidate_eligibility",
        "ict_candidate_consolidation",
        "ict_candidate_features",
        "trade_plan_selector",
        "trade_plan_render",
        "mtf_market",
    ):
        assert forbidden not in orchestrator, forbidden

    runtime = (root / "services" / "article_runtime.py").read_text(encoding="utf-8")
    assert "trade_plan_stage" in runtime
    for forbidden in ("ict_", "mtf_market", "trade_plan_gate"):
        assert forbidden not in runtime, forbidden

    # The CLI builds clients and nothing else. It may name the market source,
    # the analyst client and the policy those two are configured from; it may
    # not reach an ICT engine, the selector, the renderer or the stage - each of
    # which would be the pipeline being assembled at a command line.
    cli = (root / "cli.py").read_text(encoding="utf-8")
    assert "mtf_market" in cli
    assert "trade_plan_policy" in cli
    assert "anthropic_trade_analyst" in cli
    for forbidden in (
        "ict_composite",
        "ict_candidate_features",
        "trade_plan_selector",
        "trade_plan_render",
        "trade_plan_stage",
        "trade_plan_gate",
    ):
        assert forbidden not in cli, forbidden


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
        for word in ("break of structure", "market structure shift", " bos ", " mss ", "choch"):
            assert word not in text, f"{name} mentions {word!r}"


def test_the_analysis_structure_label_still_says_what_it_said() -> None:
    """Four values, ``RANGE`` among them, and a tie-break the ICT engine refuses."""
    from goldpipeline.schemas.context import MarketStructure
    from goldpipeline.services.levels import classify_structure

    assert {member.value for member in MarketStructure} == {
        "BULLISH",
        "BEARISH",
        "RANGE",
        "INSUFFICIENT_DATA",
    }
    assert classify_structure([], []) is MarketStructure.INSUFFICIENT_DATA
