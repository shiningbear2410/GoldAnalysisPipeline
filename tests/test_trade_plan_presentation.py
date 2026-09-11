"""Round 6.7 §P-§AM: the V2 page, the Plan Copywriter, and the V2 gate.

The page is a pure function of three documents, so most of this file needs no
Run at all: a :class:`PlanDocument` in, one exact string out. The copywriter
tests use the real staggered reading and the real selector, because what a copy
may say depends on which zones were actually selected.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from goldpipeline.adapters.fake_plan_copywriter import EchoPlanCopywriter, ScriptedPlanCopywriter
from goldpipeline.adapters.fake_trade_analyst import EchoTradeAnalyst
from goldpipeline.adapters.trade_analyst_client import TradeAnalystRequest
from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.manifest import RunStatus
from goldpipeline.schemas.news import CuratedItem, CuratedNews
from goldpipeline.schemas.orchestration import PipelineMode
from goldpipeline.schemas.publish import Decision, PublishDecision
from goldpipeline.services.ict_candidate_consolidation import consolidate_candidates
from goldpipeline.services.ict_candidate_eligibility import analyse_candidate_eligibility
from goldpipeline.services.ict_candidate_features import build_candidate_features
from goldpipeline.services.ict_composite import analyse_ict_composite
from goldpipeline.services.orchestrator import PipelineClients, resume_pipeline
from goldpipeline.services.publish_gate import DECISION_FILENAME
from goldpipeline.services.trade_analyst import (
    build_trade_analyst_input,
    build_trade_analyst_prompt,
    parse_ranking,
)
from goldpipeline.services.trade_plan_copy import (
    COPY_KEYS,
    PLAN_COPY_MODEL,
    PlanCopyError,
    build_plan_copy_request,
    parse_plan_copy,
)
from goldpipeline.services.trade_plan_gate import TRADE_PLAN_GATE_VERSION, gate_trade_plan
from goldpipeline.services.trade_plan_policy import PRODUCTION_POLICY_V1
from goldpipeline.services.trade_plan_presentation import (
    CLOSING,
    DISCLAIMER,
    MAX_PLAN_CHARS,
    NEWS_HEADING,
    NO_NEWS_LINE,
    TITLE_PREFIX,
    PlanDocument,
    PlanNewsItem,
    PlanPresentationError,
    PlanSide,
    ZoneLine,
    news_display,
    plan_date,
    plan_news_items,
    prose_problem,
    render_plan,
    validate_plan,
)
from goldpipeline.services.trade_plan_selector import TradePlanSelection, ZoneLabel
from goldpipeline.services.trade_plan_stage import (
    ANALYST_REQUEST_FILENAME,
    COPY_FILENAME,
    COPY_REQUEST_FILENAME,
    FINAL_ARTICLE_FILENAME,
    TRADE_PLAN_ARTIFACTS,
    write_trade_plan,
)
from goldpipeline.storage.run_store import RunStore
from tests.conftest import make_normalized_run
from tests.test_trade_plan_live import observation, observe

NOW = datetime(2026, 9, 10, 1, 0, tzinfo=UTC)

DOCUMENT = PlanDocument(
    date="11.09.2026",
    market_view="Khung H4 vẫn giữ cấu trúc tăng, còn M15 đang điều chỉnh về vùng cân bằng.",
    seo=PlanSide(
        heading_note="chờ hồi lên mới tính",
        zones=(
            ZoneLine("4325", "4328", True, "hợp lưu H1 và H4"),
            ZoneLine("4340", "4345.5", False, None),
        ),
        deep_level="4360",
        deep_note="chỉ để tham chiếu",
    ),
    bai=PlanSide(
        heading_note=None,
        zones=(ZoneLine("4290", "4295", True, None),),
        deep_level=None,
        deep_note=None,
    ),
    news=("Fed giữ nguyên lãi suất",),
)

EXPECTED = "\n".join(
    [
        "🎯 KẾ HOẠCH VÀNG — 11.09.2026",
        "Khung H4 vẫn giữ cấu trúc tăng, còn M15 đang điều chỉnh về vùng cân bằng.",
        "",
        "🔴 SEO · chờ hồi lên mới tính",
        "▸ Vùng 1 · 4325 – 4328  (vùng chủ đạo, hợp lưu H1 và H4)",
        "▸ Vùng 2 · 4340 – 4345.5",
        "▸ Vùng sâu chờ sẵn: 4360  (chỉ để tham chiếu)",
        "",
        "🟢 BAI",
        "▸ Vùng 1 · 4290 – 4295  (vùng chủ đạo)",
        "",
        "📰 Tin cần chú ý",
        "- Fed giữ nguyên lãi suất",
        "",
        "🔴 Trên đây là nhận định cá nhân của mình, thông tin mang tính tham khảo. "
        "Không phải lời khuyên đầu tư, tư vấn tài chính",
        "👉 Chúc mọi người một phiên giao dịch kỷ luật.",
    ]
)

PRICES = frozenset({"4325", "4328", "4340", "4345.5", "4360", "4290", "4295"})
NEWS_DISPLAYS = frozenset({"Fed giữ nguyên lãi suất"})
IDS = frozenset({"abcdefabcdefabcd"})


def validate(text: str, document: PlanDocument = DOCUMENT, **overrides: Any) -> None:
    validate_plan(
        text,
        document,
        allowed_prices=overrides.get("prices", PRICES),
        candidate_ids=overrides.get("ids", IDS),
        news_displays=overrides.get("news", NEWS_DISPLAYS),
    )


def with_view(view: str) -> PlanDocument:
    return PlanDocument(
        date=DOCUMENT.date, market_view=view, seo=DOCUMENT.seo, bai=DOCUMENT.bai, news=DOCUMENT.news
    )


# --------------------------------------------------------------------------
# AP 10-15: the page
# --------------------------------------------------------------------------


def test_10_the_page_is_exactly_this() -> None:
    """§P. Pinned in full: every emoji, separator and blank line."""
    text = render_plan(DOCUMENT)

    assert text == EXPECTED
    validate(text)


def test_10_the_date_is_the_market_instant_in_vietnam() -> None:
    """§Q. Seventeen hundred UTC is already tomorrow in Hà Nội."""
    assert plan_date(datetime(2026, 9, 10, 18, 30, tzinfo=UTC)) == "11.09.2026"
    assert plan_date(datetime(2026, 9, 10, 16, 59, tzinfo=UTC)) == "10.09.2026"
    with pytest.raises(PlanPresentationError):
        plan_date(datetime(2026, 9, 10, 16, 59))  # noqa: DTZ001 - naive on purpose


def test_11_the_market_view_is_one_paragraph_on_its_own_line() -> None:
    lines = render_plan(DOCUMENT).split("\n")

    assert lines[1] == DOCUMENT.market_view
    assert lines[2] == ""


def test_12_prices_are_printed_exactly_as_selected() -> None:
    """§AI. No rounding, no padding, one separator."""
    text = render_plan(DOCUMENT)

    assert "4340 – 4345.5" in text
    assert "4345.50" not in text
    assert "4325-4328" not in text


def test_12_a_price_changed_on_the_page_is_refused() -> None:
    with pytest.raises(PlanPresentationError):
        validate(EXPECTED.replace("4328", "4329"))


def test_12_a_price_the_selection_did_not_choose_is_refused() -> None:
    with pytest.raises(PlanPresentationError, match="not a selected price"):
        validate(EXPECTED, prices=PRICES - {"4345.5"})
    with pytest.raises(PlanPresentationError, match="missing"):
        validate(EXPECTED, prices=PRICES | {"4400"})


def test_13_the_main_label_and_the_note_share_one_bracket() -> None:
    """§AB. Merged, and never written twice."""
    said_it = PlanSide(
        heading_note=None,
        zones=(ZoneLine("4290", "4295", True, "vùng chủ đạo khung H4"),),
        deep_level=None,
        deep_note=None,
    )
    document = PlanDocument(
        date=DOCUMENT.date, market_view="Chờ phản ứng.", seo=DOCUMENT.seo, bai=said_it, news=()
    )

    text = render_plan(document)

    assert "▸ Vùng 1 · 4290 – 4295  (vùng chủ đạo khung H4)" in text
    assert text.count("vùng chủ đạo") == 2  # one per side


def test_14_the_deep_level_is_one_price_after_the_zones() -> None:
    lines = render_plan(DOCUMENT).split("\n")

    assert lines[6] == "▸ Vùng sâu chờ sẵn: 4360  (chỉ để tham chiếu)"
    assert lines[7] == ""


def test_an_empty_side_is_a_dash_with_no_note_and_no_deep_level() -> None:
    """§AL."""
    empty = PlanSide(heading_note=None, zones=(), deep_level=None, deep_note=None)
    document = PlanDocument(
        date=DOCUMENT.date, market_view="Chờ.", seo=empty, bai=DOCUMENT.bai, news=()
    )

    text = render_plan(document)

    assert "🔴 SEO\n—\n\n🟢 BAI" in text
    validate(text, document, prices=frozenset({"4290", "4295"}), news=frozenset())


def test_15_no_news_says_so_honestly() -> None:
    """§AF."""
    document = PlanDocument(
        date=DOCUMENT.date, market_view="Chờ.", seo=DOCUMENT.seo, bai=DOCUMENT.bai, news=()
    )

    text = render_plan(document)

    assert f"{NEWS_HEADING}\n{NO_NEWS_LINE}\n" in text
    validate(text, document, news=frozenset())


def test_a_news_line_that_was_not_offered_is_refused() -> None:
    with pytest.raises(PlanPresentationError, match="not an offered item"):
        validate(EXPECTED, news=frozenset({"Một tin khác"}))


def test_the_disclaimer_and_the_closing_appear_once_and_last() -> None:
    """§AG, §AH."""
    lines = EXPECTED.split("\n")

    assert lines[-2:] == [DISCLAIMER, CLOSING]
    assert EXPECTED.count(DISCLAIMER) == 1
    assert EXPECTED.count(CLOSING) == 1


def test_the_page_is_capped_and_never_cut() -> None:
    """§AJ."""
    with pytest.raises(PlanPresentationError, match="cap"):
        render_plan(with_view("Chờ phản ứng rõ ràng. " * 200))
    assert len(EXPECTED) < MAX_PLAN_CHARS


def test_a_candidate_id_in_the_page_is_refused() -> None:
    document = with_view("Theo dõi abcdefabcdefabcd trước khi vào lệnh.")

    with pytest.raises(PlanPresentationError, match="leaked"):
        validate(render_plan(document), document)


def test_the_page_is_plain_text() -> None:
    """§AM. No code fence, no JSON, no table."""
    assert "`" not in EXPECTED
    assert "|" not in EXPECTED
    assert not EXPECTED.startswith("{")


# --------------------------------------------------------------------------
# prose rules
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("Giá có thể về 4300", "a number"),
        ("chờ 3 vùng", "a number"),
        ("khung M30", "a number"),
        ("giá ４３００", "a number"),
        ("đặt SL dưới đáy", "stop loss"),
        ("chốt lời một phần", "take profit"),
        ("tỷ lệ R:R đẹp", "risk-reward"),
        ("RSI đang quá mua", "indicator"),
        ("đây là vùng chính", "the retired label"),
        ("xem https://example.com", "a link"),
        ("{json}", "markup"),
    ],
)
def test_prose_rules_name_what_they_refuse(text: str, problem: str) -> None:
    assert prose_problem(text) == problem


@pytest.mark.parametrize(
    "text", ["Khung H4 giữ xu hướng tăng", "M1 M5 M15 H1 H4 đồng thuận", "vùng chủ đạo phía mua"]
)
def test_timeframe_names_are_the_only_digits_allowed(text: str) -> None:
    assert prose_problem(text) is None


def test_a_news_headline_is_its_first_line_without_links() -> None:
    assert news_display("Fed giữ lãi suất https://t.me/x\nchi tiết") == "Fed giữ lãi suất"
    assert news_display("\n\n  Vàng tăng  \n") == "Vàng tăng"
    assert news_display("https://only.link") == ""

    long = news_display("vàng " * 100)
    assert len(long) <= 160
    assert long.endswith("…")
    assert "  " not in long


def test_news_items_take_their_identity_from_the_source() -> None:
    items = plan_news_items(news_fixture())

    assert [item.news_item_id for item in items] == ["kitco_vn/11", "goldnews/12"]
    assert items[0].display == "Fed giữ nguyên lãi suất"
    assert plan_news_items(None) == ()


# --------------------------------------------------------------------------
# AP 16-20: the copywriter
# --------------------------------------------------------------------------


def news_fixture() -> CuratedNews:
    return CuratedNews(
        items=[
            CuratedItem(
                channel="kitco_vn",
                message_id=11,
                published_at=NOW,
                text="Fed giữ nguyên lãi suất\nChi tiết cuộc họp và phản ứng của thị trường.",
                relevance_score=1.0,
                source_count=2,
            ),
            CuratedItem(
                channel="goldnews",
                message_id=12,
                published_at=NOW,
                text="Vàng giữ vững trên mốc 4300 USD sau dữ liệu việc làm",
                relevance_score=0.8,
                source_count=1,
            ),
            CuratedItem(
                channel="goldnews",
                message_id=13,
                published_at=NOW,
                text="https://t.me/only-a-link",
                relevance_score=0.5,
                source_count=1,
            ),
        ],
        item_limit=5,
        chars_per_item=500,
    )


def real_selection() -> TradePlanSelection:
    """The staggered reading, ranked by the echo analyst, selected for real."""
    from goldpipeline.services.trade_plan_selector import select_trade_plan

    shot = observation()
    policy = PRODUCTION_POLICY_V1
    composite = analyse_ict_composite(shot.snapshot, config=policy.composite_config())
    features = build_candidate_features(
        consolidate_candidates(
            analyse_candidate_eligibility(composite, config=policy.eligibility_config())
        )
    )
    prompt = build_trade_analyst_prompt(build_trade_analyst_input(features))
    response = EchoTradeAnalyst().rank(
        TradeAnalystRequest(system=prompt.system, user=prompt.user, max_tokens=4000)
    )
    return select_trade_plan(features, parse_ranking(response.text, features=features))


@pytest.fixture(scope="module")
def selection() -> TradePlanSelection:
    chosen = real_selection()
    assert chosen.seo_entries or chosen.bai_entries, "the fixture must publish a zone"
    return chosen


@pytest.fixture
def news_items() -> tuple[PlanNewsItem, ...]:
    return plan_news_items(news_fixture())


def echo_answer(selection: TradePlanSelection, items: tuple[PlanNewsItem, ...]) -> dict[str, Any]:
    request = build_plan_copy_request(selection, items)
    parsed: dict[str, Any] = json.loads(EchoPlanCopywriter().write(request).text)
    return parsed


def parse(answer: dict[str, Any] | str, selection: TradePlanSelection, items: Any) -> Any:
    text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
    return parse_plan_copy(text, selection=selection, news_items=items)


def entry_ids(selection: TradePlanSelection) -> list[str]:
    return [zone.candidate_id for zone in (*selection.seo_entries, *selection.bai_entries)]


def test_the_echo_copy_is_accepted(
    selection: TradePlanSelection, news_items: tuple[PlanNewsItem, ...]
) -> None:
    copy = parse(echo_answer(selection, news_items), selection, news_items)

    assert set(copy.zone_notes) <= set(entry_ids(selection))
    assert copy.news_item_ids == ("kitco_vn/11", "goldnews/12")


def test_16_a_news_id_never_offered_is_refused(
    selection: TradePlanSelection, news_items: tuple[PlanNewsItem, ...]
) -> None:
    answer = echo_answer(selection, news_items)
    answer["news_item_ids"] = ["reuters/999"]

    with pytest.raises(PlanCopyError, match="never offered"):
        parse(answer, selection, news_items)


def test_16_more_than_three_news_items_are_refused(
    selection: TradePlanSelection, news_items: tuple[PlanNewsItem, ...]
) -> None:
    answer = echo_answer(selection, news_items)
    answer["news_item_ids"] = ["kitco_vn/11"] * 4

    with pytest.raises(PlanCopyError):
        parse(answer, selection, news_items)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("market_view", "Giá có thể quay về 4300 trước khi tăng."),
        ("market_view", "Ba kịch bản: 1 là tăng, 2 là giảm."),
        ("seo_heading_note", "chờ 2 phiên"),
        ("bai_heading_note", "canh mua ở 4290"),
    ],
)
def test_17_a_digit_in_the_prose_is_refused(
    selection: TradePlanSelection,
    news_items: tuple[PlanNewsItem, ...],
    field: str,
    value: str,
) -> None:
    answer = echo_answer(selection, news_items)
    answer[field] = value

    with pytest.raises(PlanCopyError, match="a number"):
        parse(answer, selection, news_items)


def test_17_a_digit_in_a_zone_note_is_refused(
    selection: TradePlanSelection, news_items: tuple[PlanNewsItem, ...]
) -> None:
    answer = echo_answer(selection, news_items)
    answer["zone_notes"] = {entry_ids(selection)[0]: "quanh 4300"}

    with pytest.raises(PlanCopyError, match="a number"):
        parse(answer, selection, news_items)


def test_18_a_note_for_an_unselected_id_is_refused(
    selection: TradePlanSelection, news_items: tuple[PlanNewsItem, ...]
) -> None:
    unselected = [
        candidate.candidate_id
        for candidate in selection.features.candidates
        if candidate.candidate_id not in selection.selected_candidate_ids
    ]
    answer = echo_answer(selection, news_items)
    for identity in [*unselected[:1], "deadbeefdeadbeef"]:
        answer["zone_notes"] = {identity: "vùng phản ứng"}
        with pytest.raises(PlanCopyError, match="not a selected id"):
            parse(answer, selection, news_items)


def test_18_only_the_main_zone_may_be_called_main(
    selection: TradePlanSelection, news_items: tuple[PlanNewsItem, ...]
) -> None:
    others = [
        zone.candidate_id
        for zone in (*selection.seo_entries, *selection.bai_entries)
        if zone.label is not ZoneLabel.VUNG_CHINH
    ]
    if not others:
        pytest.skip("the fixture has no secondary zone")
    answer = echo_answer(selection, news_items)
    answer["zone_notes"] = {others[0]: "vùng chủ đạo mới"}

    with pytest.raises(PlanCopyError, match="vùng chủ đạo"):
        parse(answer, selection, news_items)


@pytest.mark.parametrize(
    "text",
    [
        "Here is the commentary",
        "[]",
        '```json\n{"market_view": "x"}\n```',
        "{}",
    ],
)
def test_19_malformed_answers_are_refused(
    selection: TradePlanSelection, news_items: tuple[PlanNewsItem, ...], text: str
) -> None:
    with pytest.raises(PlanCopyError):
        parse(text, selection, news_items)


@pytest.mark.parametrize("key", ["price", "seo_entries", "stop_loss", "zones"])
def test_20_the_copy_has_no_field_that_can_hold_a_price(
    selection: TradePlanSelection, news_items: tuple[PlanNewsItem, ...], key: str
) -> None:
    answer = echo_answer(selection, news_items)
    answer[key] = "4300"

    with pytest.raises(PlanCopyError, match="unexpected"):
        parse(answer, selection, news_items)
    assert key not in COPY_KEYS


def test_20_a_missing_key_is_refused(
    selection: TradePlanSelection, news_items: tuple[PlanNewsItem, ...]
) -> None:
    answer = echo_answer(selection, news_items)
    del answer["reference_notes"]

    with pytest.raises(PlanCopyError, match="missing"):
        parse(answer, selection, news_items)


@pytest.mark.parametrize(
    ("field", "limit"),
    [("market_view", 1200), ("seo_heading_note", 70), ("bai_heading_note", 70)],
)
def test_lengths_are_hard_limits(
    selection: TradePlanSelection,
    news_items: tuple[PlanNewsItem, ...],
    field: str,
    limit: int,
) -> None:
    answer = echo_answer(selection, news_items)
    answer[field] = "a" * (limit + 1)

    with pytest.raises(PlanCopyError, match="over"):
        parse(answer, selection, news_items)


def test_a_multi_line_market_view_is_refused(
    selection: TradePlanSelection, news_items: tuple[PlanNewsItem, ...]
) -> None:
    answer = echo_answer(selection, news_items)
    answer["market_view"] = "Đoạn một.\nĐoạn hai."

    with pytest.raises(PlanCopyError, match="single line"):
        parse(answer, selection, news_items)


def test_the_copy_request_is_compact_and_carries_only_the_selection(
    selection: TradePlanSelection, news_items: tuple[PlanNewsItem, ...]
) -> None:
    """§W. The selected zones and references, not the candidate set or its graph."""
    request = build_plan_copy_request(selection, news_items)
    unselected = [
        candidate.candidate_id
        for candidate in selection.features.candidates
        if candidate.candidate_id not in selection.selected_candidate_ids
    ]

    assert "<PLAN_DATA>" in request.user
    assert "pair_relations" not in request.user
    assert all(identity not in request.user for identity in unselected)
    assert all(identity in request.user for identity in selection.selected_candidate_ids)
    assert len(request.user) < 20_000
    assert "# OUTPUT CONTRACT" in request.system
    assert PLAN_COPY_MODEL == "claude-sonnet-5"


# --------------------------------------------------------------------------
# AP 21: the stage and the V2 gate
# --------------------------------------------------------------------------


@pytest.fixture
def plan_run(runs_dir: Path, tmp_path: Path) -> str:
    created = make_normalized_run(runs_dir, tmp_path)
    run = RunStore(runs_dir).open(created.run_id)
    manifest = run.load_manifest()
    assert manifest.provenance is not None
    manifest.provenance.article_type = ArticleType.TRADE_PLAN
    run.save_manifest(manifest)
    return str(created.run_id)


def clients(*, news: CuratedNews | None = None, copywriter: Any = None) -> PipelineClients:
    return PipelineClients(
        trade_plan_market=observe(),
        trade_plan_analyst=lambda selection: EchoTradeAnalyst(),
        trade_plan_copywriter=lambda selection: copywriter or EchoPlanCopywriter(),
        trade_plan_news=(lambda: news) if news is not None else None,
    )


def article_of(runs_dir: Path, run_id: str) -> str:
    raw = RunStore(runs_dir).open(run_id).read_artifact_bytes(FINAL_ARTICLE_FILENAME)
    return raw.decode("utf-8").rstrip("\n")


def test_21_a_plan_run_reaches_ready_through_the_v2_gate(runs_dir: Path, plan_run: str) -> None:
    result = resume_pipeline(
        run_id=plan_run,
        store=RunStore(runs_dir),
        clients=clients(news=news_fixture()),
        mode=PipelineMode.READY_FOR_PUBLISH,
    )
    run = RunStore(runs_dir).open(plan_run)
    decision = PublishDecision.model_validate_json(
        run.read_artifact_bytes(DECISION_FILENAME).decode("utf-8")
    )
    article = article_of(runs_dir, plan_run)

    assert result.result.run_status is RunStatus.READY_TO_PUBLISH
    assert decision.decision is Decision.APPROVED
    assert decision.gate_version == TRADE_PLAN_GATE_VERSION == "gold_trade_plan_gate_v2"
    assert article.startswith(TITLE_PREFIX + plan_date(observation().market_observed_at))
    assert "- Fed giữ nguyên lãi suất" in article
    assert "- Vàng giữ vững trên mốc 4300 USD sau dữ liệu việc làm" in article
    assert article.split("\n")[-2:] == [DISCLAIMER, CLOSING]
    for name in TRADE_PLAN_ARTIFACTS:
        assert run.has_artifact(name), name


def test_21_without_news_the_page_says_so(runs_dir: Path, plan_run: str) -> None:
    resume_pipeline(
        run_id=plan_run,
        store=RunStore(runs_dir),
        clients=clients(),
        mode=PipelineMode.READY_FOR_PUBLISH,
    )

    assert NO_NEWS_LINE in article_of(runs_dir, plan_run)


def test_21_an_edited_copy_blocks_the_gate(runs_dir: Path, plan_run: str) -> None:
    store = RunStore(runs_dir)
    write_trade_plan(
        run_id=plan_run,
        store=store,
        observe=observe(),
        analyst=EchoTradeAnalyst(),
        copywriter=EchoPlanCopywriter(),
    )
    path = store.open(plan_run).artifact_path(COPY_FILENAME)
    edited = json.loads(path.read_text(encoding="utf-8"))
    edited["market_view"] = "Mua ngay, chắc chắn thắng."
    path.write_text(json.dumps(edited, ensure_ascii=False), encoding="utf-8")

    decision = gate_trade_plan(run_id=plan_run, store=store)

    assert decision.decision is Decision.BLOCKED
    assert store.open(plan_run).load_manifest().status is RunStatus.PUBLISH_BLOCKED


def test_19_a_bad_copy_fails_the_stage_and_leaves_nothing(runs_dir: Path, plan_run: str) -> None:
    store = RunStore(runs_dir)

    result = write_trade_plan(
        run_id=plan_run,
        store=store,
        observe=observe(),
        analyst=EchoTradeAnalyst(),
        copywriter=ScriptedPlanCopywriter(text="Đây là bình luận của tôi."),
    )

    assert not result.succeeded
    assert "did not return JSON" in str(result.error)
    run = store.open(plan_run)
    assert run.load_manifest().status is RunStatus.NORMALIZED
    for name in TRADE_PLAN_ARTIFACTS:
        assert not run.has_artifact(name), name


def test_the_copy_request_is_a_small_fraction_of_the_ranking_request(
    runs_dir: Path, plan_run: str
) -> None:
    store = RunStore(runs_dir)
    write_trade_plan(
        run_id=plan_run,
        store=store,
        observe=observe(),
        analyst=EchoTradeAnalyst(),
        copywriter=EchoPlanCopywriter(),
    )
    run = store.open(plan_run)

    copy_request = run.read_artifact_bytes(COPY_REQUEST_FILENAME)
    analyst_request = run.read_artifact_bytes(ANALYST_REQUEST_FILENAME)

    assert len(copy_request) < len(analyst_request)
