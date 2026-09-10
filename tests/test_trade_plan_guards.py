"""What the selector and the renderer refuse to do.

Round 6.6g §14, §23, §35-§37. This is the stage that finally publishes prices,
so the guards are about two things: that no model can reach the text, and that
no price on the page came from anywhere but a deterministic candidate.

The AI-call guard is the sharpest one here. The pipeline already has writers,
reviewers and finalizers, all one import away, and "the renderer must never call
a model" is exactly the kind of rule that stays true only until someone adds a
polish step. So it is parsed rather than trusted.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from goldpipeline.services import trade_plan_render, trade_plan_selector

SELECTOR = Path("src/goldpipeline/services/trade_plan_selector.py")
RENDERER = Path("src/goldpipeline/services/trade_plan_render.py")

MODEL_ENTRY_POINTS = (
    "build_writer_client",
    "build_finalizer_client",
    "WriterClient",
    "FinalizerClient",
    "ReviewerClient",
    "DigestWriterClient",
    "DigestFinalizerClient",
    "TradeAnalystClient",
    "rank_candidates",
    "load_prompt",
    "generate",
    "rank",
    "revise",
    "review",
)

LOWER_ENGINE_ENTRY_POINTS = (
    "confirmed_swings",
    "fair_value_gaps",
    "latest_atr",
    "analyse_structure",
    "analyse_liquidity",
    "analyse_fvg_lifecycle",
    "analyse_protected_structure",
    "analyse_dealing_ranges",
    "analyse_order_blocks",
    "analyse_order_block_lifecycle",
    "analyse_ict_composite",
    "project_candidate_sources",
    "analyse_candidate_eligibility",
    "resolve_reference_price",
    "consolidate_candidates",
    "build_candidate_features",
    "parse_ranking",
    "locate",
)


def tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def identifiers(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree(path)):
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


def imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree(path)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def called_names(path: Path) -> set[str]:
    return {
        node.func.id
        for node in ast.walk(tree(path))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(tree(path))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def public_names(module: object) -> set[str]:
    return {name for name in dir(module) if not name.startswith("_")}


# --------------------------------------------------------------------------
# §23: no model composes the trade plan
# --------------------------------------------------------------------------


@pytest.mark.parametrize("entry_point", MODEL_ENTRY_POINTS)
def test_the_renderer_calls_no_model(entry_point: str) -> None:
    """§23. Parsed, because this is the rule most likely to erode quietly."""
    assert entry_point not in called_names(RENDERER), entry_point
    assert entry_point not in identifiers(RENDERER), entry_point


@pytest.mark.parametrize("entry_point", MODEL_ENTRY_POINTS)
def test_the_selector_calls_no_model(entry_point: str) -> None:
    """§23. The ranking arrives already made; nothing here asks for another."""
    assert entry_point not in called_names(SELECTOR), entry_point
    assert entry_point not in identifiers(SELECTOR), entry_point


@pytest.mark.parametrize(
    "forbidden",
    ["anthropic", "deepseek", "openai", "httpx", "requests", "urllib", "socket", "websocket"],
)
def test_neither_stage_can_reach_a_vendor(forbidden: str) -> None:
    """§23, §35."""
    for path in (SELECTOR, RENDERER):
        for module in imported_modules(path):
            assert forbidden.lower() not in module.lower(), (path.name, module)


def test_the_renderer_imports_only_the_selection_and_the_price_helper() -> None:
    """§23, §35, as an exact set."""
    inside = {module for module in imported_modules(RENDERER) if module.startswith("goldpipeline")}

    assert inside == {
        "goldpipeline.services.ict_candidate_consolidation",
        "goldpipeline.services.ict_candidate_eligibility",
        "goldpipeline.services.trade_plan_selector",
    }


def test_the_selector_imports_only_finished_upstream_models() -> None:
    """§35, as an exact set."""
    inside = {module for module in imported_modules(SELECTOR) if module.startswith("goldpipeline")}

    assert inside == {
        "goldpipeline.services.ict_candidate_consolidation",
        "goldpipeline.services.ict_candidate_eligibility",
        "goldpipeline.services.ict_candidate_features",
        "goldpipeline.services.trade_analyst",
    }


def test_no_prompt_or_llm_text_reaches_the_trade_plan() -> None:
    """§23, §37. There is no TRADE_PLAN writer prompt, and no way to load one."""
    for path in (SELECTOR, RENDERER):
        text = path.read_text(encoding="utf-8")
        for forbidden in ("gold_writer", "gold_finalizer", "gold_reviewer", "prompts"):
            assert forbidden not in text, (path.name, forbidden)


def test_the_renderer_calls_nothing_but_its_own_helpers_and_stdlib() -> None:
    """§23, as an exact set - so a new call has to be argued for."""
    assert called_names(RENDERER) == {
        # its own helpers
        "render_price",
        "render_zone",
        "render_reference",
        "render_trade_plan",
        "validate_trade_plan",
        "_side_lines",
        "_expected_prices",
        "TradePlanRenderError",
        # the shared price representation, reused rather than reimplemented
        "canonical_price",
        # a lookup on the selection it was handed
        "main_zone",
        # stdlib
        "add",
        "any",
        "append",
        "count",
        "frozenset",
        "getLogger",
        "index",
        "info",
        "isdigit",
        "isinstance",
        "join",
        "len",
        "replace",
        "set",
        "sorted",
        "split",
        "strip",
        "sum",
        "type",
    }


# --------------------------------------------------------------------------
# §14, §35: no engine below, no candle, no clock
# --------------------------------------------------------------------------


@pytest.mark.parametrize("entry_point", LOWER_ENGINE_ENTRY_POINTS)
def test_neither_stage_recomputes_anything(entry_point: str) -> None:
    """§35. Both receive finished objects and read them."""
    for path in (SELECTOR, RENDERER):
        assert entry_point not in called_names(path), (path.name, entry_point)


@pytest.mark.parametrize(
    "forbidden",
    ["OHLCBar", "IctMarketSnapshot", "IctTimeframeSnapshot", "bars", "series", "atr", "swings"],
)
def test_no_raw_candle_is_reachable(forbidden: str) -> None:
    """§35."""
    for path in (SELECTOR, RENDERER):
        assert forbidden not in identifiers(path), (path.name, forbidden)


def test_neither_stage_reads_a_clock_or_randomness() -> None:
    """§33, §34."""
    for path in (SELECTOR, RENDERER):
        text = path.read_text(encoding="utf-8")
        for forbidden in (
            "uuid",
            "random",
            "secrets",
            "shuffle",
            "datetime.now",
            "utcnow",
            "time.time",
            "utc_now",
        ):
            assert forbidden not in text, (path.name, forbidden)


def test_neither_stage_names_float() -> None:
    """§19."""
    for path in (SELECTOR, RENDERER):
        assert "float" not in identifiers(path), path.name
        for node in ast.walk(tree(path)):
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                raise AssertionError(f"{path.name}: float literal at line {node.lineno}")


# --------------------------------------------------------------------------
# §14: geometry is copied, never derived
# --------------------------------------------------------------------------


PRICE_TOKENS = ("lower", "upper", "midpoint", "level", "price", "Decimal")


def test_no_arithmetic_touches_a_price() -> None:
    """§14. No round, quantize, offset, padding, ATR or pip adjustment.

    Arithmetic does exist in both files - a tuple concatenation in the selector
    and two set differences in the renderer - and none of it goes near a price,
    which is what the check actually cares about.
    """
    for path in (SELECTOR, RENDERER):
        arithmetic = [
            ast.unparse(node)
            for node in ast.walk(tree(path))
            if isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Add | ast.Sub | ast.Mult | ast.Div)
        ]
        for expression in arithmetic:
            for token in PRICE_TOKENS:
                assert token not in expression, (path.name, expression)


def test_the_only_negation_of_a_price_is_a_sort_key() -> None:
    """§14, §17. BAI descends, so its sort key negates two Decimals.

    Negating a Decimal is exact and the result never leaves the comparison, but
    it is the one place a price is touched by an operator at all - so it is
    enumerated rather than swept in with the rest.
    """
    negations = [
        ast.unparse(node)
        for node in ast.walk(tree(SELECTOR))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
    ]

    assert negations == ["-zone.upper", "-zone.lower"]
    assert not [
        ast.unparse(node)
        for node in ast.walk(tree(RENDERER))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
    ]


@pytest.mark.parametrize(
    "forbidden",
    [
        "round",
        "quantize",
        "scaleb",
        "offset",
        "padding",
        "adjust",
        "atr_multiple",
        "pips",
        "tick_size",
        "percent",
        "tolerance",
        "epsilon",
        "approx",
    ],
)
def test_no_price_is_adjusted(forbidden: str) -> None:
    """§6, §14. Exact everywhere, in both stages."""
    for path in (SELECTOR, RENDERER):
        assert not any(forbidden in name.lower() for name in identifiers(path)), (
            path.name,
            forbidden,
        )


def test_the_selected_zone_copies_the_candidates_prices() -> None:
    """§14. Parsed: the three prices are attribute reads, not expressions."""
    source = inspect.getsource(trade_plan_selector._select_side)
    body = source.split("SelectedEntryZone(")[1]

    for field in ("lower", "upper", "midpoint"):
        assert f"{field}=candidate.{field}," in body.replace(" ", "").replace("\n", "")


def test_the_reference_copies_the_candidates_level() -> None:
    """§14."""
    source = inspect.getsource(trade_plan_selector._select_reference)

    assert "level = candidate.reference_level" in source
    assert "level=level," in source


def test_the_renderer_derives_no_price_of_its_own() -> None:
    """§14. Every published number came in on the selection."""
    source = RENDERER.read_text(encoding="utf-8")

    assert "Decimal(" not in source
    for name in ("render_zone", "render_reference"):
        rendered = inspect.getsource(getattr(trade_plan_render, name))
        assert "render_price(" in rendered


# --------------------------------------------------------------------------
# §6: no softness in the redundancy rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    ["overlap_ratio", "overlap_percent", "nearby", "similar", "proximity", "cluster", "fuzzy"],
)
def test_only_containment_suppresses(forbidden: str) -> None:
    """§6. Partial overlap, touching and near-misses all survive."""
    assert not any(forbidden in name.lower() for name in identifiers(SELECTOR)), forbidden
    assert not any(forbidden in name.lower() for name in public_names(trade_plan_selector)), (
        forbidden
    )


def test_the_containment_test_is_two_comparisons_and_nothing_else() -> None:
    """§5. Parsed, so a tolerance cannot be added without failing here."""
    parsed = ast.parse(inspect.getsource(trade_plan_selector.contains).strip())
    returns = [node for node in ast.walk(parsed) if isinstance(node, ast.Return)]

    assert len(returns) == 1
    rendered = ast.unparse(returns[0])
    assert rendered == "return outer_lower <= inner_lower and outer_upper >= inner_upper"


def test_no_new_geometry_is_constructed_from_a_pair() -> None:
    """§5, §10. Suppression removes a zone; it never makes one.

    Parsed rather than grepped, because the module's prose says plainly that it
    never merges or widens anything - and a guard that punished it for saying so
    would push the explanation out of the file.
    """
    names = identifiers(SELECTOR) | public_names(trade_plan_selector)

    for forbidden in ("intersection", "union", "merge", "widen", "combine", "midpoint_of"):
        assert not any(forbidden in name.lower() for name in names), forbidden


# --------------------------------------------------------------------------
# §4, §9, §10: caps and labels
# --------------------------------------------------------------------------


def test_the_cap_is_five_per_side_and_is_never_a_minimum() -> None:
    """§4. No floor anywhere - nothing counts up to three.

    Identifiers, not prose: the module explicitly explains that it does not
    backfill or pad, and naming a refusal is the opposite of implementing it.
    """
    assert trade_plan_selector.MAX_ENTRY_ZONES_PER_SIDE == 5
    names = identifiers(SELECTOR) | public_names(trade_plan_selector)

    for forbidden in ("min_entry", "minimum", "backfill", "at_least", "quota", "floor"):
        assert not any(forbidden in name.lower() for name in names), forbidden


def test_only_two_labels_can_ever_be_attached() -> None:
    """§10. SCALP, CANH_CA_TUAN and HOLD are not in the vocabulary at all."""
    assert {member.name for member in trade_plan_selector.ZoneLabel} == {
        "VUNG_CHINH",
        "SAU_HON",
    }

    # Identifiers again, because the selector names all three absent labels in
    # order to explain why they are absent.
    for module, path in ((trade_plan_selector, SELECTOR), (trade_plan_render, RENDERER)):
        names = identifiers(path) | public_names(module)
        for forbidden in ("scalp", "canh_ca", "hold", "weekly", "tuan"):
            assert not any(forbidden in name.lower() for name in names), (path.name, forbidden)

    # And nothing published can carry them, whatever the code is called.
    assert "SCALP" not in trade_plan_render.PUBLIC_VOCABULARY


def test_the_main_zone_is_chosen_by_rank_and_not_by_a_second_score() -> None:
    """§9. ``min`` over ``ai_rank``, and nothing else."""
    source = inspect.getsource(trade_plan_selector._label_main)

    assert "min(zones, key=lambda zone: zone.ai_rank)" in source
    for forbidden in ("support_count", "distance", "width", "timeframe", "score"):
        assert forbidden not in source, forbidden


def test_the_public_vocabulary_is_exactly_five_words_and_a_dash() -> None:
    """§22."""
    assert (
        frozenset({"SEO", "BAI", "vùng", "chính", "sâu", "hơn", "—"})
        == trade_plan_render.PUBLIC_VOCABULARY
    )


# --------------------------------------------------------------------------
# §36-§37: nothing is activated, nothing shipped changed
# --------------------------------------------------------------------------


def test_the_branch_registry_knows_about_both_new_modules() -> None:
    """§36."""
    from tests.test_ict_structure_guards import ICT_BRANCH

    assert "trade_plan_selector.py" in ICT_BRANCH
    assert "trade_plan_render.py" in ICT_BRANCH


def test_nothing_outside_the_dormant_branch_calls_either_stage() -> None:
    """§36. Still dormant: no runtime, no scheduler, no CLI reaches these."""
    from tests.test_ict_structure_guards import ICT_BRANCH

    root = Path("src/goldpipeline")
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if path.name not in ICT_BRANCH
        and any(
            module in path.read_text(encoding="utf-8")
            for module in ("trade_plan_selector", "trade_plan_render")
        )
    ]

    assert offenders == []


def test_trade_plan_is_still_not_a_product() -> None:
    """§36. The renderer exists; dispatch does not."""
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
    """§37."""
    from goldpipeline.schemas.article import ArticleType
    from goldpipeline.services.article_routing import SPECS, writer_prompt_for
    from goldpipeline.services.review_action import STYLE_ACTIVE_TYPES

    assert writer_prompt_for(ArticleType.ANALYSIS) == "gold_writer_v4"
    assert writer_prompt_for(ArticleType.NEWS_DIGEST) == "gold_news_digest_writer_v2"
    assert SPECS[ArticleType.ANALYSIS].ready is True
    assert SPECS[ArticleType.NEWS_DIGEST].ready is True
    assert frozenset({ArticleType.ANALYSIS, ArticleType.NEWS_DIGEST}) == STYLE_ACTIVE_TYPES


def test_no_trade_plan_writer_prompt_exists() -> None:
    """§37. There is no LLM writer for this document, and there will not be one."""
    from goldpipeline import prompts

    names = {name for name in dir(prompts) if name.startswith("GOLD_")}
    assert "GOLD_TRADE_PLAN_WRITER_V1" not in names
    assert not any("TRADE_PLAN" in name for name in names)

    files = {path.name for path in Path("src/goldpipeline/prompts").glob("*.md")}
    assert not any("trade_plan" in name for name in files)


def test_no_shipped_prompt_learned_a_word_from_this_round() -> None:
    """§37."""
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
        for word in ("vùng chính", "sâu hơn", "vung_chinh", "sau_hon"):
            assert word not in text, f"{name} mentions {word!r}"


def test_the_market_authority_is_still_tradingview() -> None:
    """§40. Named here so a change to it fails a test rather than a tick."""
    from goldpipeline.cli import PRODUCTION_MARKET_SOURCE

    assert PRODUCTION_MARKET_SOURCE == "tradingview"
