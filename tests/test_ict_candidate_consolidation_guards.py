"""What this layer refuses to do, now that it can see candidates side by side.

Round 6.6e.2c1 §35-§37, §55-§57, §61, §66-§72. Every earlier layer looked at one
source at a time. This is the first that holds two candidates at once, and that
is precisely what makes the refusals here load-bearing: with a group in hand,
``support_count`` becomes a score, ``len(timeframes)`` becomes confluence,
sorting by ``distance_to_reference`` becomes a shortlist, and merging
"close enough" zones becomes one ``abs(a - b) < tolerance``. None of those has
been argued for, so none of them is one line away by accident.

Guards read identifiers out of the parsed module rather than grepping its text,
so the module's prose stays free to name what it refuses - and it names a great
deal of it.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from goldpipeline.services import ict_candidate_consolidation

SOURCE = Path("src/goldpipeline/services/ict_candidate_consolidation.py")

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
    "project_candidate_sources",
    "project_timeframe_sources",
    "analyse_candidate_eligibility",
    "resolve_reference_price",
)


def tree() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def identifiers() -> set[str]:
    """Every name the module uses, string literals excluded."""
    names: set[str] = set()
    for node in ast.walk(tree()):
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
    modules: set[str] = set()
    for node in ast.walk(tree()):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def called_names() -> set[str]:
    return {
        node.func.id
        for node in ast.walk(tree())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(tree())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def field_names() -> set[str]:
    return set(ict_candidate_consolidation.ConsolidatedCandidate.__dataclass_fields__) | set(
        ict_candidate_consolidation.CandidateConsolidationAnalysis.__dataclass_fields__
    )


def public_names() -> set[str]:
    return (
        field_names()
        | {
            name
            for name in dir(ict_candidate_consolidation.ConsolidatedCandidate)
            if not name.startswith("_")
        }
        | {
            name
            for name in dir(ict_candidate_consolidation.CandidateConsolidationAnalysis)
            if not name.startswith("_")
        }
        | set(ict_candidate_consolidation.__all__)
    )


# --------------------------------------------------------------------------
# §35: the eligibility reading is the only input, and nothing is recomputed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("entry_point", LOWER_ENGINE_ENTRY_POINTS)
def test_no_engine_below_is_ever_called(entry_point: str) -> None:
    """§35. Not the ICT authorities, not the projection, not eligibility itself.

    The reference price in particular: it is *carried* from the reading, never
    resolved again, so a consolidation can never be measured against a price
    its own members were not.
    """
    assert entry_point not in called_names(), entry_point
    assert entry_point not in identifiers(), entry_point


def test_the_module_calls_nothing_but_its_own_helpers_and_stdlib() -> None:
    """§35, as an exact set - so a new call has to be argued for."""
    assert called_names() == {
        # its own helpers and models
        "_group_key",
        "_consolidated_id",
        "_require_agreement",
        "_candidate",
        "_unique",
        "canonical_price",
        "CandidateConsolidationError",
        "CandidateConsolidationAnalysis",
        "ConsolidatedCandidate",
        # enums re-made from a carried value, not analyses re-run
        "CandidateSourceKind",
        "Timeframe",
        # stdlib and the dataclass decorator
        "dataclass",
        "enumerate",
        "format",
        "normalize",
        "len",
        "next",
        "tuple",
        "sort",
        "append",
        "join",
        "encode",
        "hexdigest",
        "sha256",
        "getLogger",
    }


def test_the_entry_point_takes_an_eligibility_reading_and_nothing_else() -> None:
    """§35. One parameter, one type, no config of its own."""
    from goldpipeline.services.ict_candidate_eligibility import CandidateEligibilityAnalysis

    signature = inspect.signature(ict_candidate_consolidation.consolidate_candidates)

    assert list(signature.parameters) == ["eligibility"]
    assert signature.parameters["eligibility"].annotation in (
        CandidateEligibilityAnalysis,
        "CandidateEligibilityAnalysis",
    )


def test_the_module_takes_no_configuration() -> None:
    """§35, §57. Exactness is not a tunable, so there is no dial to turn.

    A config here would be the natural home for a merge tolerance, and the way
    a merge tolerance arrives is by being configurable and defaulting to zero.
    """
    assert not any(
        name.endswith("Config") and name != "IctCompositeConfig"
        for name in ict_candidate_consolidation.__all__
    )
    for forbidden in ("ConsolidationConfig", "CandidateConsolidationConfig"):
        assert not hasattr(ict_candidate_consolidation, forbidden), forbidden


# --------------------------------------------------------------------------
# §36: no raw candles
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        "OHLCBar",
        "IctMarketSnapshot",
        "IctTimeframeSnapshot",
        "build_timeframe_snapshot",
        "bars",
        "series",
        "high",
        "low",
        "close",
        "volume",
    ],
)
def test_no_candle_is_reachable(forbidden: str) -> None:
    """§36. Geometry arrives already computed; there is nothing to re-measure."""
    assert forbidden not in identifiers(), forbidden


def test_no_price_is_computed_from_anything() -> None:
    """§36, §5. Every price on a candidate is copied from a member, not derived.

    The midpoint and the width are the two a merge would recompute, and
    recomputing them is how a merged zone acquires a geometry no source drew.
    """
    source = inspect.getsource(ict_candidate_consolidation._candidate)
    assigned = ast.parse(source.strip())

    for field in ("lower", "upper", "midpoint", "width", "reference_level"):
        assert f"{field}=first.{field}" in ast.unparse(assigned).replace(" ", ""), field

    arithmetic = [
        node
        for node in ast.walk(assigned)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Sub | ast.Div)
    ]
    assert arithmetic == [], "no candidate price is arithmetic"


# --------------------------------------------------------------------------
# §37: no range context
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        "premium",
        "discount",
        "equilibrium",
        "dealing_range",
        "active_dealing_range",
        "locate",
        "price_location",
        "structure_bias",
    ],
)
def test_no_range_context_is_consulted(forbidden: str) -> None:
    """§37. The dealing range stays on the eligibility timeframe entry, unread.

    It is still reachable - ``result.eligibility.require(tf).active_dealing_range``
    - which is the point: carried, not consumed.
    """
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden
    assert not any(forbidden in name.lower() for name in field_names()), forbidden


def test_the_range_remains_reachable_through_the_carried_reading() -> None:
    """§37. Refusing to read it is not the same as throwing it away."""
    from goldpipeline.schemas.common import Timeframe
    from goldpipeline.services.ict_candidate_consolidation import consolidate_candidates
    from goldpipeline.services.ict_candidate_eligibility import analyse_candidate_eligibility
    from goldpipeline.services.ict_composite import analyse_ict_composite
    from tests.test_ict_candidate_eligibility_fixture import (
        eligibility_config,
        staggered_snapshot,
    )
    from tests.test_ict_composite_fixture import config

    result = consolidate_candidates(
        analyse_candidate_eligibility(
            analyse_ict_composite(staggered_snapshot(), config=config()),
            config=eligibility_config(),
        )
    )

    assert result.eligibility.require(Timeframe.H1).active_dealing_range is not None


# --------------------------------------------------------------------------
# §55: exact only - no overlap, no proximity, no tolerance
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        "overlap",
        "intersect",
        "adjacent",
        "nearby",
        "proximity",
        "cluster",
        "bucket",
        "band_merge",
        "fuzzy",
        "approx",
        "epsilon",
        "tolerance",
        "within",
        "atr_multiple",
        "pips",
        "percent",
    ],
)
def test_nothing_soft_reaches_the_grouping(forbidden: str) -> None:
    """§55, §22. Exact equality is the whole rule; everything else is next round."""
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden
    assert not any(forbidden in name.lower() for name in public_names()), forbidden


def test_the_group_key_compares_only_canonical_strings() -> None:
    """§55, §24. No subtraction, no comparison operator, no rounding.

    A tolerance cannot exist in a function whose entire output is a tuple of
    exact strings - which is why the key is strings rather than Decimals.
    """
    parsed = ast.parse(inspect.getsource(ict_candidate_consolidation._group_key).strip())

    for node in ast.walk(parsed):
        assert not isinstance(node, ast.BinOp), ast.unparse(node)
        if isinstance(node, ast.Compare):
            for operator in node.ops:
                assert isinstance(operator, ast.Is | ast.IsNot), ast.unparse(node)


@pytest.mark.parametrize("forbidden", ["round", "quantize", "scaleb", "tick_size", "abs"])
def test_no_price_is_rounded_or_snapped(forbidden: str) -> None:
    """§24. ``normalize`` strips trailing zeros; it does not change a value."""
    assert forbidden not in identifiers(), forbidden


def test_canonicalisation_is_exactly_the_historical_one() -> None:
    """§24. Pinned to the private helper it deliberately did not refactor."""
    from decimal import Decimal

    from goldpipeline.services.ict_fvg import _canonical

    for text in ("4000", "4000.0", "4000.00", "3999.995", "0.5", "1E+3", "4043", "1234.5670"):
        assert ict_candidate_consolidation.canonical_price(Decimal(text)) == _canonical(
            Decimal(text)
        )


# --------------------------------------------------------------------------
# §56: support is a count, never a score
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        "score",
        "rank",
        "rating",
        "priority",
        "quality",
        "strength",
        "confidence",
        "confluence",
        "weight",
        "grade",
        "tier",
        "best",
        "top",
    ],
)
def test_no_scoring_of_any_kind(forbidden: str) -> None:
    """§56. Two supporters is a fact about two supporters and nothing more."""
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden
    assert not any(forbidden in name.lower() for name in public_names()), forbidden


def test_the_two_counts_are_lengths_and_nothing_else() -> None:
    """§56. ``support_count`` and ``timeframe_count`` are ``len`` calls.

    Not weighted, not multiplied by a timeframe rank, not compared to a
    threshold.
    """
    for name in ("support_count", "timeframe_count"):
        body = inspect.getsource(
            getattr(ict_candidate_consolidation.ConsolidatedCandidate, name).fget
        )
        parsed = ast.parse(body.strip().removeprefix("@property").strip())
        returns = [node for node in ast.walk(parsed) if isinstance(node, ast.Return)]

        assert len(returns) == 1
        assert isinstance(returns[0].value, ast.Call)
        assert isinstance(returns[0].value.func, ast.Name)
        assert returns[0].value.func.id == "len", name


def test_no_timeframe_outranks_another() -> None:
    """§56. H4 agreement is not worth more than M1 agreement here."""
    text = SOURCE.read_text(encoding="utf-8")

    for forbidden in ("TIMEFRAME_WEIGHT", "TIMEFRAME_RANK", "higher_timeframe", "htf_bonus"):
        assert forbidden not in text, forbidden
    assert "_ROLE_ORDER" in text and "_SIDE_ORDER" in text, (
        "the only two orders present are the ordering keys"
    )


# --------------------------------------------------------------------------
# §57: no selection, no shortlist, no final answer
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        "select",
        "choose",
        "pick",
        "shortlist",
        "limit",
        "cap",
        "max_candidates",
        "final",
        "primary",
        "main_zone",
        "recommend",
        "publish",
    ],
)
def test_nothing_selects_a_subset(forbidden: str) -> None:
    """§57. Every eligible decision is placed; none is preferred."""
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden
    assert not any(forbidden in name.lower() for name in public_names()), forbidden


def test_the_only_sort_is_the_fixed_display_order() -> None:
    """§33, §57. Parsed, because a sort keyed on distance would be a ranking."""
    sorts = [
        ast.unparse(node)
        for node in ast.walk(tree())
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "sorted")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "sort")
        )
    ]

    assert len(sorts) == 1, sorts
    expression = sorts[0]
    assert "_ROLE_ORDER" in expression
    assert "_SIDE_ORDER" in expression
    assert "consolidated_candidate_id" in expression
    for forbidden in ("distance", "support", "reverse", "price", "width", "timeframe_count"):
        assert forbidden not in expression, forbidden


def test_the_only_slice_in_the_module_truncates_a_hash() -> None:
    """§57. A ``[:5]`` on the candidate list would be a published shortlist.

    Enumerated rather than banned outright, because two slices are honest: the
    identity helper truncates a hash, and the agreement check reads "every
    member after the first". Both are named below; a third slice fails.
    """
    slices = [
        ast.unparse(node)
        for node in ast.walk(tree())
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice)
    ]

    assert slices == [
        # sixteen hex digits off a SHA-256, as every id on this branch is built
        "hashlib.sha256(preimage.encode('utf-8')).hexdigest()[:16]",
        # "every member after the first", inside the agreement check - it drops
        # nothing, it compares the rest against members[0]
        "members[1:]",
    ], slices


def test_no_narrative_label_appears_anywhere() -> None:
    """§57, §67. No published labels, in any language."""
    text = SOURCE.read_text(encoding="utf-8").lower()

    for forbidden in ("scalp", "vùng", "canh cả", "sâu hơn", "main zone", "khuyến nghị"):  # noqa: RUF001
        assert forbidden not in text, forbidden


# --------------------------------------------------------------------------
# §27: no false append-only invariant
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden", ["first_seen", "created_at", "history", "previous", "since", "persisted", "store"]
)
def test_a_group_carries_no_history_of_its_own(forbidden: str) -> None:
    """§27. A group is a view at an instant; it does not remember being older."""
    assert not any(forbidden in name.lower() for name in public_names()), forbidden


@pytest.mark.parametrize("forbidden", ["replace", "evolve", "mutate", "update", "set_status"])
def test_nothing_upstream_is_rewritten(forbidden: str) -> None:
    """§27, §54. Members are held, not edited."""
    assert forbidden not in called_names(), forbidden


# --------------------------------------------------------------------------
# §68: source agnosticism and determinism hygiene
# --------------------------------------------------------------------------


def test_the_module_reads_only_ict_inputs() -> None:
    """Asserted as an exact set, so a fifth dependency fails rather than sneaks in."""
    inside = {module for module in imported_modules() if module.startswith("goldpipeline")}

    assert inside == {
        "goldpipeline.schemas.common",
        "goldpipeline.services.ict_candidate_eligibility",
        "goldpipeline.services.ict_candidate_source",
        "goldpipeline.services.ict_composite",
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


def test_the_module_never_names_float() -> None:
    assert "float" not in identifiers()


def test_no_float_literal_appears_in_the_source() -> None:
    for node in ast.walk(tree()):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            raise AssertionError(f"float literal {node.value} at line {node.lineno}")


def test_no_set_reaches_an_identity_or_an_ordering() -> None:
    """§52. ``_unique`` exists precisely so a set never orders anything."""
    assert "set" not in called_names()
    for node in ast.walk(tree()):
        assert not isinstance(node, ast.SetComp), ast.unparse(node)
        assert not isinstance(node, ast.Set), ast.unparse(node)


# --------------------------------------------------------------------------
# §61, §70-§72: reachability and nothing shipped changed
# --------------------------------------------------------------------------


def test_the_branch_registry_knows_about_this_module() -> None:
    from tests.test_ict_structure_guards import ICT_BRANCH

    assert "ict_candidate_consolidation.py" in ICT_BRANCH


def test_no_consolidation_code_is_reachable_from_the_shipped_products() -> None:
    """§61, §72."""
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
        if "ict_candidate_consolidation" in path.read_text(encoding="utf-8")
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
        for word in ("consolidat", "support_count", "confluence", "candidate"):
            assert word not in text, f"{name} mentions {word!r}"
