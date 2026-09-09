"""The eight questions this layer deliberately refuses to answer.

Round 6.6e.2a §2-§4, §10, §15-§16, §21-§25, §34, §38, §45-§50. The absences here
are the round's actual content. A projection that quietly decided eligibility, a
side, a score or a merge would look like progress and would have destroyed the
audit boundary the next round depends on.

Guards read identifiers out of the parsed module rather than grepping its text,
so prose is free to name what it refuses.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from goldpipeline.services import ict_candidate_source

SOURCE = Path("src/goldpipeline/services/ict_candidate_source.py")

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
    "analyse_ict_timeframe",
    "analyse_ict_composite",
)


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


def called_names() -> set[str]:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def field_names() -> set[str]:
    return (
        set(ict_candidate_source.CandidateSource.__dataclass_fields__)
        | set(ict_candidate_source.CandidateSourceTimeframe.__dataclass_fields__)
        | set(ict_candidate_source.CandidateSourceProjection.__dataclass_fields__)
        | set(ict_candidate_source.OrderBlockEvidence.__dataclass_fields__)
        | set(ict_candidate_source.FairValueGapEvidence.__dataclass_fields__)
        | set(ict_candidate_source.LiquidityPoolEvidence.__dataclass_fields__)
    )


# --------------------------------------------------------------------------
# §2, §34: the composite is the only input seam
# --------------------------------------------------------------------------


@pytest.mark.parametrize("entry_point", LOWER_ENGINE_ENTRY_POINTS)
def test_no_lower_engine_is_ever_called(entry_point: str) -> None:
    """§34. Zero of them, checked by parsing rather than by review."""
    assert entry_point not in called_names(), entry_point
    assert entry_point not in identifiers(), entry_point


def test_the_module_calls_nothing_but_its_own_helpers_and_stdlib() -> None:
    """§34, as an exact set - so a new call has to be argued for."""
    assert called_names() == {
        # its own helpers and models
        "_source",
        "_candidate_source_id",
        "_check_geometry",
        "project_timeframe_sources",
        "CandidateSource",
        "CandidateSourceTimeframe",
        "CandidateSourceProjection",
        "CandidateSourceError",
        "OrderBlockEvidence",
        "FairValueGapEvidence",
        "LiquidityPoolEvidence",
        # lookups on objects it was handed, not analyses it ran
        "order_block",
        "timeframe",
        # stdlib and the dataclass decorator
        "dataclass",
        "sort",
        "append",
        "tuple",
        "next",
        "join",
        "encode",
        "hexdigest",
        "sha256",
        "getLogger",
    }


def test_the_entry_points_take_composite_objects_and_not_candles() -> None:
    """§2. Raw candles are not the public API of this layer."""
    from goldpipeline.services.ict_composite import (
        IctCompositeAnalysis,
        IctTimeframeAnalysis,
    )

    composite_sig = inspect.signature(ict_candidate_source.project_candidate_sources)
    timeframe_sig = inspect.signature(ict_candidate_source.project_timeframe_sources)

    assert list(composite_sig.parameters) == ["composite"]
    assert composite_sig.parameters["composite"].annotation in (
        IctCompositeAnalysis,
        "IctCompositeAnalysis",
    )
    assert list(timeframe_sig.parameters) == ["entry"]
    assert timeframe_sig.parameters["entry"].annotation in (
        IctTimeframeAnalysis,
        "IctTimeframeAnalysis",
    )


@pytest.mark.parametrize(
    "forbidden",
    ["IctTimeframeSnapshot", "IctMarketSnapshot", "OHLCBar", "build_timeframe_snapshot", "bars"],
)
def test_no_raw_candle_type_reaches_this_layer(forbidden: str) -> None:
    """§2. It reads finished facts, never the series they came from."""
    assert forbidden not in identifiers(), forbidden


# --------------------------------------------------------------------------
# §3-§4: three source kinds, and ranges are not one of them
# --------------------------------------------------------------------------


def test_there_are_exactly_three_source_kinds() -> None:
    """§3."""
    assert [member.value for member in ict_candidate_source.CandidateSourceKind] == [
        "ORDER_BLOCK",
        "FAIR_VALUE_GAP",
        "LIQUIDITY_POOL",
    ]


@pytest.mark.parametrize(
    "forbidden",
    ["DEALING_RANGE", "PROTECTED_SWING", "STRUCTURAL_LEG", "SWING", "ATR", "BOS", "MSS"],
)
def test_no_other_object_became_a_source_kind(forbidden: str) -> None:
    """§3, §4. Those remain context, because their role is undecided."""
    assert forbidden not in {member.value for member in ict_candidate_source.CandidateSourceKind}
    assert forbidden not in {member.name for member in ict_candidate_source.CandidateSourceKind}


def test_no_range_derived_identity_is_minted() -> None:
    """§4. Attaching a range as context is not the same as making it a candidate."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    identity_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_candidate_source_id"
    ]

    assert len(identity_calls) == 1, "one place mints identities"
    for forbidden in ("range_id", "equilibrium", "origin_price", "terminal_price"):
        assert forbidden not in identifiers(), forbidden


def test_there_are_exactly_two_geometry_kinds() -> None:
    """§5."""
    assert [member.value for member in ict_candidate_source.CandidateGeometryKind] == [
        "ZONE",
        "BAND",
    ]


# --------------------------------------------------------------------------
# §10: no fake shared status enum
# --------------------------------------------------------------------------


def test_the_module_defines_no_status_enum_of_its_own() -> None:
    """§10. ``FILLED``, ``INVALIDATED`` and ``SWEPT`` are not synonyms."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    enums = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(isinstance(base, ast.Name) and base.id == "StrEnum" for base in node.bases)
    }

    assert enums == {"CandidateSourceKind", "CandidateGeometryKind"}
    for forbidden in ("LIVE", "USED", "DEAD", "SourceStatus", "CandidateStatus"):
        assert forbidden not in identifiers(), forbidden


def test_each_evidence_type_keeps_its_own_status_type() -> None:
    """§10. Three status systems, three types, no flattening."""
    from goldpipeline.services.ict_fvg import FvgStatus
    from goldpipeline.services.ict_liquidity import PoolStatus
    from goldpipeline.services.ict_order_block_lifecycle import OrderBlockStatus

    hints = {
        name: cls.__dataclass_fields__["status"].type
        for name, cls in (
            ("ob", ict_candidate_source.OrderBlockEvidence),
            ("fvg", ict_candidate_source.FairValueGapEvidence),
            ("pool", ict_candidate_source.LiquidityPoolEvidence),
        )
    }

    assert hints["ob"] in (OrderBlockStatus, "OrderBlockStatus")
    assert hints["fvg"] in (FvgStatus, "FvgStatus")
    assert hints["pool"] in (PoolStatus, "PoolStatus")


def test_the_evidence_union_covers_exactly_the_three_kinds() -> None:
    """§10. A typed union, so a reader cannot silently mishandle a kind."""
    import typing

    members = set(typing.get_args(ict_candidate_source.CandidateEvidence))

    assert members == {
        ict_candidate_source.OrderBlockEvidence,
        ict_candidate_source.FairValueGapEvidence,
        ict_candidate_source.LiquidityPoolEvidence,
    }


# --------------------------------------------------------------------------
# §15-§16: no consolidation, no confluence
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    ["dedupe", "deduplicate", "consolidat", "merge", "collapse", "combine", "union", "intersect"],
)
def test_no_consolidation_of_any_kind(forbidden: str) -> None:
    """§15. One observation, one fact - even with identical bounds."""
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden
    assert not any(forbidden in name.lower() for name in field_names()), forbidden


@pytest.mark.parametrize(
    "forbidden",
    ["confluence", "cluster", "overlap", "support_count", "source_count", "agreement"],
)
def test_no_confluence_of_any_kind(forbidden: str) -> None:
    """§16. Preserve the raw observations first."""
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden
    assert not any(forbidden in name.lower() for name in field_names()), forbidden
    assert not any(forbidden in name.lower() for name in ict_candidate_source.__all__), forbidden


# --------------------------------------------------------------------------
# §21-§24: no price, no side, no score, no freshness
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    ["spot", "current_price", "reference_price", "market_price", "distance", "proximity"],
)
def test_no_current_price_notion(forbidden: str) -> None:
    """§21. The cross-timeframe price authority has not been chosen yet."""
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden
    assert not any(forbidden in name.lower() for name in field_names()), forbidden


@pytest.mark.parametrize(
    "forbidden",
    ["seo", "bai", "entry_side", "trade_direction", "entry_zone", "stop_loss", "take_profit"],
)
def test_no_trade_side_vocabulary(forbidden: str) -> None:
    """§22. Bullish/bearish and buy-side/sell-side stay facts about the past."""
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden
    assert not any(forbidden in name.lower() for name in field_names()), forbidden
    assert not any(forbidden in name.lower() for name in ict_candidate_source.__all__), forbidden


def test_buy_and_sell_appear_in_no_identifier_this_module_defines() -> None:
    """§22, token-matched.

    ``BUY_SIDE`` and ``SELL_SIDE`` live on the liquidity pool, which this module
    carries rather than authors - so the check is on the names defined here.
    """
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    defined = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.ClassDef)
    } | field_names()

    for name in defined:
        parts = name.lower().split("_")
        assert "buy" not in parts, name
        assert "sell" not in parts, name


@pytest.mark.parametrize(
    "forbidden",
    ["score", "rank", "strength", "confidence", "quality", "priority", "weight", "grade"],
)
def test_no_scoring_of_any_kind(forbidden: str) -> None:
    """§23."""
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden
    assert not any(forbidden in name.lower() for name in field_names()), forbidden
    assert not any(forbidden in name.lower() for name in ict_candidate_source.__all__), forbidden


@pytest.mark.parametrize(
    "forbidden", ["fresh", "stale", "expired", "untouched", "tradable", "usable"]
)
def test_no_freshness_policy(forbidden: str) -> None:
    """§24. Lifecycle facts exist; candidate freshness does not."""
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden
    assert not any(forbidden in name.lower() for name in field_names()), forbidden


@pytest.mark.parametrize(
    "forbidden", ["eligible", "ineligible", "eligibility", "reject", "exclude", "accepted"]
)
def test_no_eligibility_notion(forbidden: str) -> None:
    """The round's whole point: this layer decides nothing about usability."""
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden
    assert not any(forbidden in name.lower() for name in field_names()), forbidden
    assert not any(forbidden in name.lower() for name in ict_candidate_source.__all__), forbidden


def test_no_filtering_comparison_exists_in_the_projection_loop() -> None:
    """§9. There is no ``if status is ...`` anywhere that could drop a source.

    Parsed rather than grepped: the module's only comparisons are the geometry
    contract and the one identity join, and this pins that as an exact list.
    """
    tree = ast.parse(inspect.getsource(ict_candidate_source.project_timeframe_sources))
    comparisons = [node for node in ast.walk(tree) if isinstance(node, ast.Compare)]

    assert len(comparisons) == 1, "only the identity join tests anything"
    assert isinstance(comparisons[0].ops[0], ast.Is)


def test_no_vietnamese_narrative_label_appears_anywhere() -> None:
    """§23. The published labels a later round might reach for, in any language.

    Checked against the whole file because a label would be a string literal
    rather than an identifier - but only for words that could not appear
    innocently. "hold" is checked as an identifier token instead, since the
    module legitimately says a projection "holds no M1 sources".
    """
    text = SOURCE.read_text(encoding="utf-8").lower()

    for forbidden in ("scalp", "vùng", "canh cả", "sâu hơn"):  # noqa: RUF001
        assert forbidden not in text, forbidden

    for name in identifiers() | field_names():
        parts = name.lower().split("_")
        for forbidden in ("hold", "scalp", "deep", "main"):
            assert forbidden not in parts, name


# --------------------------------------------------------------------------
# §25: no seventh policy
# --------------------------------------------------------------------------


def test_the_projection_takes_no_config_of_its_own() -> None:
    """§25. It carries the composite's six; it adds none."""
    for function in (
        ict_candidate_source.project_candidate_sources,
        ict_candidate_source.project_timeframe_sources,
    ):
        assert "config" not in inspect.signature(function).parameters

    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert not any("Config" in name for name in classes)


def test_the_carried_config_is_the_composite_s_type() -> None:
    from goldpipeline.services.ict_composite import IctCompositeConfig

    annotation = ict_candidate_source.CandidateSourceProjection.__dataclass_fields__["config"].type

    assert annotation in (IctCompositeConfig, "IctCompositeConfig")


# --------------------------------------------------------------------------
# §38: source agnosticism
# --------------------------------------------------------------------------


def test_the_module_reads_only_ict_inputs() -> None:
    """Asserted as an exact set, so a ninth dependency fails rather than sneaks in."""
    inside = {module for module in imported_modules() if module.startswith("goldpipeline")}

    assert inside == {
        "goldpipeline.schemas.common",
        "goldpipeline.services.ict_composite",
        "goldpipeline.services.ict_fvg",
        "goldpipeline.services.ict_liquidity",
        "goldpipeline.services.ict_order_block",
        "goldpipeline.services.ict_order_block_lifecycle",
        "goldpipeline.services.ict_primitives",
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
def test_the_module_imports_no_provider(forbidden: str) -> None:
    for module in imported_modules():
        assert forbidden.lower() not in module.lower(), module


def test_the_module_reads_no_clock() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    for forbidden in ("datetime.now", "utcnow", "time.time", "utc_now"):
        assert forbidden not in text


def test_the_module_never_names_float() -> None:
    assert "float" not in identifiers()


def test_no_float_literal_appears_in_the_source() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            raise AssertionError(f"float literal {node.value} at line {node.lineno}")


@pytest.mark.parametrize("forbidden", ["round", "quantize", "scaleb", "tick", "pips", "widen"])
def test_no_rounding_or_widening_of_prices(forbidden: str) -> None:
    """§11, §42. A zero-width pool keeps its single price."""
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden


# --------------------------------------------------------------------------
# §45-§50: nothing shipped changed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("forbidden", ["higher_timeframe", "promote", "demote", "synthesis"])
def test_no_multi_timeframe_synthesis(forbidden: str) -> None:
    """§45. Independent facts, each carrying its timeframe."""
    assert not any(forbidden in name.lower() for name in identifiers()), forbidden


def test_the_branch_registry_knows_about_this_module() -> None:
    from tests.test_ict_structure_guards import ICT_BRANCH

    assert "ict_candidate_source.py" in ICT_BRANCH


def test_no_candidate_code_is_reachable_from_the_shipped_products() -> None:
    """§50. The production scheduler must not start doing more work."""
    from tests.test_ict_structure_guards import ICT_BRANCH

    root = Path("src/goldpipeline")
    callers = [
        path
        for path in root.rglob("*.py")
        if "ict_candidate_source" in path.read_text(encoding="utf-8")
        and path.name not in ICT_BRANCH
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
        for word in ("candidate", "order block", "liquidity pool", "fair value gap"):
            assert word not in text, f"{name} mentions {word!r}"
