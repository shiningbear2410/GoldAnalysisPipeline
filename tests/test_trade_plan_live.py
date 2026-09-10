"""The whole TRADE_PLAN product, offline, from a normalized Run to a review copy.

Round 6.6h §21-§25. Everything the live smoke will do, with a fixture market, a
fake analyst and a fake Telegram - so that a defect is found here rather than
against three external services.

Two properties matter more than the happy path. The first is that a trade plan
reaching a human is *exactly once*: one reservation, one send, one result, and
silence on every tick afterwards. The second is that every way this can fail
does so without leaving a half-finished document behind - a Run holding a page
with no evidence, or evidence with no page, would be worse than a Run that
failed outright.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from goldpipeline.adapters.fake_publisher import FakePublisherClient
from goldpipeline.adapters.fake_trade_analyst import (
    EchoTradeAnalyst,
    HallucinatingTradeAnalyst,
    MalformedTradeAnalyst,
    ScriptedTradeAnalyst,
)
from goldpipeline.adapters.mtf_market import (
    MultiTimeframeError,
    MultiTimeframeObservation,
    TimeframeFetch,
)
from goldpipeline.adapters.trade_analyst_client import (
    TradeAnalystRequest,
    TradeAnalystResponse,
)
from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.manifest import RunStatus
from goldpipeline.schemas.orchestration import PipelineMode, PipelineStatus
from goldpipeline.schemas.publish import Decision
from goldpipeline.schemas.review_delivery import (
    INTENT_FILENAME,
    RESULT_FILENAME,
    ReviewDeliveryStatus,
)
from goldpipeline.services.orchestrator import PipelineClients, resume_pipeline
from goldpipeline.services.publish_gate import DECISION_FILENAME
from goldpipeline.services.review_delivery import deliver_review, is_eligible
from goldpipeline.services.trade_plan_policy import PRODUCTION_POLICY_V1
from goldpipeline.services.trade_plan_stage import (
    ANALYST_REQUEST_FILENAME,
    ANALYST_RESPONSE_FILENAME,
    CANDIDATES_FILENAME,
    FINAL_ARTICLE_FILENAME,
    MARKET_FILENAME,
    POLICY_FILENAME,
    RANKING_FILENAME,
    SELECTION_FILENAME,
    TRADE_PLAN_ARTIFACTS,
    write_trade_plan,
)
from goldpipeline.storage.run_store import RunStore
from tests.conftest import make_normalized_run
from tests.test_ict_candidate_eligibility_fixture import staggered_snapshot

REVIEW_CHAT = "-1001234567890"


# --------------------------------------------------------------------------
# a fixture observation, built the way the live adapter builds one
# --------------------------------------------------------------------------


def observation(snapshot: Any | None = None) -> MultiTimeframeObservation:
    """The staggered five-timeframe snapshot, wrapped as a live observation.

    Reuses the reading six rounds of tests were written against rather than
    inventing a new one, so the offline product is the product those rounds
    proved - reference price 4043, six consolidated candidates.
    """
    shot = snapshot if snapshot is not None else staggered_snapshot()
    return MultiTimeframeObservation(
        market_observed_at=shot.observed_at,
        symbol=shot.symbol,
        provider=shot.provider,
        provider_symbol=PRODUCTION_POLICY_V1.provider_symbol,
        snapshot=shot,
        fetches=tuple(
            TimeframeFetch(
                timeframe=entry.timeframe,
                provider=shot.provider,
                provider_symbol=PRODUCTION_POLICY_V1.provider_symbol,
                requested_bars=PRODUCTION_POLICY_V1.bars_per_timeframe,
                received_bars=entry.bar_count,
                closed_bars=entry.bar_count,
                first_bar_open_time=entry.bars[0].timestamp,
                latest_bar_open_time=entry.bars[-1].timestamp,
                latest_closed_at=entry.latest_closed_at,
            )
            for entry in shot.timeframes
        ),
    )


def observe(snapshot: Any | None = None) -> Any:
    return lambda: observation(snapshot)


def failing_observe(message: str = "H1 could not be fetched") -> Any:
    def raise_it() -> MultiTimeframeObservation:
        raise MultiTimeframeError(message, timeframe=Timeframe.H1)

    return raise_it


# --------------------------------------------------------------------------
# a normalized TRADE_PLAN Run
# --------------------------------------------------------------------------


@pytest.fixture
def trade_plan_run(runs_dir: Path, tmp_path: Path) -> str:
    """A real NORMALIZED Run whose recorded product mode is TRADE_PLAN."""
    created = make_normalized_run(runs_dir, tmp_path)
    store = RunStore(runs_dir)
    run = store.open(created.run_id)
    manifest = run.load_manifest()
    assert manifest.provenance is not None
    manifest.provenance.article_type = ArticleType.TRADE_PLAN
    run.save_manifest(manifest)
    return str(created.run_id)


def clients(
    *,
    analyst: Any | None = None,
    market: Any | None = None,
    news: Any | None = None,
) -> PipelineClients:
    """Pipeline clients with every prose stage deliberately absent.

    A TRADE_PLAN Run must never reach a writer, a reviewer or a finalizer, and
    leaving those factories ``None`` turns "did not call one" into "could not
    have": the orchestrator raises rather than quietly building one.
    """
    return PipelineClients(
        trade_plan_market=market if market is not None else observe(),
        trade_plan_analyst=lambda selection: analyst or EchoTradeAnalyst(),
        trade_plan_news=news,
    )


def drive(runs_dir: Path, run_id: str, **kwargs: Any) -> Any:
    return resume_pipeline(
        run_id=run_id,
        store=RunStore(runs_dir),
        clients=clients(**kwargs),
        mode=PipelineMode.READY_FOR_PUBLISH,
    )


# --------------------------------------------------------------------------
# §24: the whole flow
# --------------------------------------------------------------------------


def test_a_trade_plan_run_reaches_ready_to_publish(runs_dir: Path, trade_plan_run: str) -> None:
    """§24. Event → Run → market → candidates → ranking → plan → gate → ready."""
    result = drive(runs_dir, trade_plan_run)
    manifest = RunStore(runs_dir).open(trade_plan_run).load_manifest()

    assert result.status is not PipelineStatus.FAILED
    assert manifest.status is RunStatus.READY_TO_PUBLISH


def test_every_audit_artifact_is_written(runs_dir: Path, trade_plan_run: str) -> None:
    """§15. The chain a reader walks back from a published price."""
    drive(runs_dir, trade_plan_run)
    run = RunStore(runs_dir).open(trade_plan_run)

    for name in TRADE_PLAN_ARTIFACTS:
        assert run.has_artifact(name), name
    assert run.has_artifact(DECISION_FILENAME)

    recorded = {ref.name for ref in run.load_manifest().artifact_files}
    assert set(TRADE_PLAN_ARTIFACTS) <= recorded


def test_no_prose_stage_runs_or_is_even_constructed(runs_dir: Path, trade_plan_run: str) -> None:
    """§18. No fake writer, reviewer or finalizer artifact is invented."""
    drive(runs_dir, trade_plan_run)
    run = RunStore(runs_dir).open(trade_plan_run)

    for name in (
        "claude_writer.json",
        "claude_draft.md",
        "gpt_review.json",
        "claude_finalizer.json",
    ):
        assert not run.has_artifact(name), name


def test_the_run_skips_drafted_and_reviewed(runs_dir: Path, trade_plan_run: str) -> None:
    """§18. The manifest records what happened, not a borrowed history."""
    drive(runs_dir, trade_plan_run)
    stages = [
        event.stage for event in RunStore(runs_dir).open(trade_plan_run).load_manifest().events
    ]

    assert "trade_plan" in stages
    assert "trade_plan_gate" in stages
    assert not any(stage.startswith("writer") for stage in stages)
    assert not any(stage.startswith("reviewer") for stage in stages)


def test_the_policy_snapshot_is_the_production_policy(runs_dir: Path, trade_plan_run: str) -> None:
    """§14, and the locked V1 values."""
    drive(runs_dir, trade_plan_run)
    body = json.loads(RunStore(runs_dir).open(trade_plan_run).read_artifact_bytes(POLICY_FILENAME))

    assert body == PRODUCTION_POLICY_V1.snapshot()
    assert body["swing_left_bars"] == body["swing_right_bars"] == 2
    assert body["atr_period"] == 14
    assert body["liquidity_price_tolerance"] == "0.5"
    assert body["order_block_zone_basis"] == "FULL_CANDLE"
    assert body["order_block_mitigation_rule"] == "MIDPOINT"
    assert body["allowed_order_block_statuses"] == ["ACTIVE", "MITIGATED", "TOUCHED"]
    assert body["allowed_fvg_statuses"] == ["OPEN", "TOUCHED"]
    assert body["bars_per_timeframe"] == 500
    assert body["timeframes"] == ["H4", "H1", "M15", "M5", "M1"]
    assert body["analyst_provider"] == "anthropic"
    assert body["analyst_model"] == "claude-sonnet-5"


def test_the_five_timeframes_share_one_observation_instant(
    runs_dir: Path, trade_plan_run: str
) -> None:
    """§2, §5."""
    drive(runs_dir, trade_plan_run)
    body = json.loads(RunStore(runs_dir).open(trade_plan_run).read_artifact_bytes(MARKET_FILENAME))
    observed = body["provenance"]["market_observed_at"]

    assert len(body["provenance"]["timeframes"]) == 5
    assert len(body["composite"]) == 5
    for entry in body["provenance"]["timeframes"]:
        assert entry["latest_closed_at"] <= observed
    assert {entry["timeframe"] for entry in body["composite"]} == {
        "H4",
        "H1",
        "M15",
        "M5",
        "M1",
    }


def test_a_published_price_walks_back_to_a_candle(runs_dir: Path, trade_plan_run: str) -> None:
    """§15. The one question the audit trail exists to answer."""
    drive(runs_dir, trade_plan_run)
    run = RunStore(runs_dir).open(trade_plan_run)
    selection = json.loads(run.read_artifact_bytes(SELECTION_FILENAME))
    candidates = json.loads(run.read_artifact_bytes(CANDIDATES_FILENAME))

    zone = (selection["seo_entries"] + selection["bai_entries"])[0]
    consolidated = next(
        entry
        for entry in candidates["consolidated"]
        if entry["consolidated_candidate_id"] == zone["candidate_id"]
    )
    assert (consolidated["lower"], consolidated["upper"]) == (zone["lower"], zone["upper"])

    supporter = consolidated["supporting_candidate_ids"][0]
    decision = next(
        entry for entry in candidates["decisions"] if entry["candidate_id"] == supporter
    )
    assert decision["eligible"] is True
    assert decision["source_id"] in consolidated["supporting_source_ids"]
    assert decision["source_kind"] in {"ORDER_BLOCK", "FAIR_VALUE_GAP", "LIQUIDITY_POOL"}
    assert decision["source_formed_at"] <= selection["observed_at"]


def test_the_final_article_is_the_rendered_plan(runs_dir: Path, trade_plan_run: str) -> None:
    """§16. The canonical filename holds the deterministic page and nothing else."""
    drive(runs_dir, trade_plan_run)
    run = RunStore(runs_dir).open(trade_plan_run)
    article = run.read_artifact_bytes(FINAL_ARTICLE_FILENAME).decode("utf-8").rstrip("\n")
    selection = json.loads(run.read_artifact_bytes(SELECTION_FILENAME))

    assert article == selection["final_text"]
    assert article.startswith("SEO\n")
    assert "\nBAI\n" in article
    assert len(article) <= 650


def test_the_gate_approves_and_names_itself(runs_dir: Path, trade_plan_run: str) -> None:
    """§17."""
    drive(runs_dir, trade_plan_run)
    decision = json.loads(
        RunStore(runs_dir).open(trade_plan_run).read_artifact_bytes(DECISION_FILENAME)
    )

    assert decision["decision"] == Decision.APPROVED.value
    assert decision["gate_version"] == "gold_trade_plan_gate_v1"
    assert decision["stage"] == "trade_plan_gate"
    assert decision["blockers"] == []
    # No review and no finalization happened, and the decision says so rather
    # than filling in a value that would imply one did.
    assert decision["review_status"] is None
    assert decision["finalization_mode"] is None


def test_the_analyst_request_carries_every_candidate_id(
    runs_dir: Path, trade_plan_run: str
) -> None:
    """§16."""
    drive(runs_dir, trade_plan_run)
    run = RunStore(runs_dir).open(trade_plan_run)
    payload = run.read_artifact_bytes(ANALYST_REQUEST_FILENAME).decode("utf-8")
    ranking = json.loads(run.read_artifact_bytes(RANKING_FILENAME))

    for values in ranking["offered"].values():
        for identity in values:
            assert identity in payload


def test_the_raw_analyst_response_is_kept(runs_dir: Path, trade_plan_run: str) -> None:
    """§15. What the model actually said, before anything validated it."""
    drive(runs_dir, trade_plan_run)
    body = json.loads(
        RunStore(runs_dir).open(trade_plan_run).read_artifact_bytes(ANALYST_RESPONSE_FILENAME)
    )

    assert body["provider"] == "fake"
    assert json.loads(body["text"]).keys() >= {"bai_entry_candidate_ids"}


# --------------------------------------------------------------------------
# §10: news context
# --------------------------------------------------------------------------


def curated(text: str = "Fed holds rates steady.") -> Any:
    from datetime import UTC

    from goldpipeline.schemas.news import CuratedItem, CuratedNews

    return CuratedNews(
        items=[
            CuratedItem(
                channel="example",
                message_id=11,
                published_at=datetime(2026, 9, 7, 11, 0, tzinfo=UTC),
                text=text,
                relevance_score=1.0,
                source_count=1,
            )
        ],
        item_limit=5,
        chars_per_item=400,
    )


def test_news_context_is_threaded_and_its_provenance_recorded(
    runs_dir: Path, trade_plan_run: str
) -> None:
    """§10."""
    news = curated()
    drive(runs_dir, trade_plan_run, news=lambda: news)
    run = RunStore(runs_dir).open(trade_plan_run)

    payload = json.loads(
        run.read_artifact_bytes(ANALYST_REQUEST_FILENAME)
        .decode("utf-8")
        .split("<CANDIDATE_DATA>")[1]
        .split("</CANDIDATE_DATA>")[0]
    )
    assert payload["news_context"]["trust_level"] == "UNTRUSTED"
    assert payload["news_context"]["items"][0]["text"] == "Fed holds rates steady."

    selection = json.loads(run.read_artifact_bytes(SELECTION_FILENAME))
    assert selection["news_context"] == {
        "item_count": 1,
        "omitted_count": 0,
        "truncated_count": 0,
        "channels": ["example"],
        "trust_level": "UNTRUSTED",
    }


def test_absent_news_is_recorded_as_absent_and_the_plan_still_builds(
    runs_dir: Path, trade_plan_run: str
) -> None:
    """§10, §25. ``None`` is a legitimate production state, not a degraded one."""
    drive(runs_dir, trade_plan_run)
    run = RunStore(runs_dir).open(trade_plan_run)

    assert json.loads(run.read_artifact_bytes(SELECTION_FILENAME))["news_context"] is None
    assert run.load_manifest().status is RunStatus.READY_TO_PUBLISH


# --------------------------------------------------------------------------
# §20-§21: review delivery, exactly once
# --------------------------------------------------------------------------


def deliver(runs_dir: Path, run_id: str, client: FakePublisherClient, **kwargs: Any) -> Any:
    return deliver_review(
        run_id=run_id,
        store=RunStore(runs_dir),
        client=client,
        target_chat=REVIEW_CHAT,
        sleep=lambda _: None,
        **kwargs,
    )


def test_a_finished_trade_plan_is_delivered_exactly_once(
    runs_dir: Path, trade_plan_run: str
) -> None:
    """§20, §21. One reservation, one send, one result - then silence."""
    drive(runs_dir, trade_plan_run)
    client = FakePublisherClient()

    first = deliver(runs_dir, trade_plan_run, client)
    second = deliver(runs_dir, trade_plan_run, client)
    third = deliver(runs_dir, trade_plan_run, client)

    assert first.status is ReviewDeliveryStatus.DELIVERED
    assert second.status is ReviewDeliveryStatus.SKIPPED
    assert third.status is ReviewDeliveryStatus.SKIPPED
    assert len(client.sent) == 1

    run = RunStore(runs_dir).open(trade_plan_run)
    assert run.has_artifact(INTENT_FILENAME)
    assert run.has_artifact(RESULT_FILENAME)


def test_the_delivered_message_carries_the_plan_verbatim(
    runs_dir: Path, trade_plan_run: str
) -> None:
    """§16, §20. The reviewer sees exactly what the renderer produced."""
    drive(runs_dir, trade_plan_run)
    client = FakePublisherClient()
    deliver(runs_dir, trade_plan_run, client)

    run = RunStore(runs_dir).open(trade_plan_run)
    article = run.read_artifact_bytes(FINAL_ARTICLE_FILENAME).decode("utf-8").strip()
    body = "".join(str(entry) for entry in client.sent)

    assert article in body
    assert "READY_TO_PUBLISH" in body


def test_delivery_does_not_publish_and_does_not_move_the_run(
    runs_dir: Path, trade_plan_run: str
) -> None:
    """§20, §35. Review-only, and the Run stays where the gate left it."""
    drive(runs_dir, trade_plan_run)
    deliver(runs_dir, trade_plan_run, FakePublisherClient())

    run = RunStore(runs_dir).open(trade_plan_run)
    assert run.load_manifest().status is RunStatus.READY_TO_PUBLISH
    assert not run.has_artifact("publish_intent.json")
    assert not run.has_artifact("publish_result.json")


def test_a_delivered_plan_is_no_longer_eligible(runs_dir: Path, trade_plan_run: str) -> None:
    """§21, §22. The cheap check the worker runs on every Run every tick."""
    from goldpipeline.schemas.common import utc_now

    drive(runs_dir, trade_plan_run)
    run = RunStore(runs_dir).open(trade_plan_run)
    manifest = run.load_manifest()

    assert is_eligible(run, manifest, now=manifest.created_at, max_run_age_minutes=60) is None
    deliver(runs_dir, trade_plan_run, FakePublisherClient())

    run = RunStore(runs_dir).open(trade_plan_run)
    assert (
        is_eligible(run, run.load_manifest(), now=utc_now(), max_run_age_minutes=60)
        == "already delivered"
    )


# --------------------------------------------------------------------------
# §22: an idle tick costs a trade plan nothing
# --------------------------------------------------------------------------


def test_a_run_that_is_not_due_never_reaches_the_market_or_the_analyst(
    runs_dir: Path, trade_plan_run: str
) -> None:
    """§22. Building the seam opens nothing; only being due calls it."""
    calls: list[str] = []

    def explode() -> Any:
        calls.append("market")
        raise AssertionError("an idle Run must not fetch five timeframes")

    def no_analyst(selection: Any) -> Any:
        calls.append("analyst")
        raise AssertionError("an idle Run must not call the analyst")

    drive(runs_dir, trade_plan_run)  # finishes the Run
    assert RunStore(runs_dir).open(trade_plan_run).load_manifest().status is (
        RunStatus.READY_TO_PUBLISH
    )

    resume_pipeline(
        run_id=trade_plan_run,
        store=RunStore(runs_dir),
        clients=PipelineClients(trade_plan_market=explode, trade_plan_analyst=no_analyst),
        mode=PipelineMode.READY_FOR_PUBLISH,
    )

    assert calls == []


def test_the_market_seam_is_a_callable_not_a_source(runs_dir: Path) -> None:
    """§22. A source constructed per tick would connect per tick."""
    import inspect

    field = PipelineClients.__dataclass_fields__["trade_plan_market"]

    assert "Callable" in str(field.type)
    assert inspect.signature(write_trade_plan).parameters["observe"] is not None


# --------------------------------------------------------------------------
# §25: the failure matrix
# --------------------------------------------------------------------------


def stage(runs_dir: Path, run_id: str, **kwargs: Any) -> Any:
    return write_trade_plan(
        run_id=run_id,
        store=RunStore(runs_dir),
        observe=kwargs.pop("observe", observe()),
        analyst=kwargs.pop("analyst", EchoTradeAnalyst()),
        **kwargs,
    )


def assert_failed_cleanly(runs_dir: Path, run_id: str) -> None:
    """No artifact, no status change, and nothing half-written."""
    run = RunStore(runs_dir).open(run_id)

    assert run.load_manifest().status is RunStatus.NORMALIZED
    for name in TRADE_PLAN_ARTIFACTS:
        assert not run.has_artifact(name), name
    assert not run.has_artifact(DECISION_FILENAME)


def test_1_a_timeframe_fetch_failure_stops_the_run(runs_dir: Path, trade_plan_run: str) -> None:
    """§25. Four timeframes is a different analysis, not a smaller one."""
    result = stage(runs_dir, trade_plan_run, observe=failing_observe())

    assert not result.succeeded
    assert result.error is not None
    assert_failed_cleanly(runs_dir, trade_plan_run)


def test_2_a_conflicting_reference_price_stops_the_run(runs_dir: Path, trade_plan_run: str) -> None:
    """§6, §25. Two feeds disagreeing at one instant is a data-integrity refusal.

    The unstaggered five-path snapshot genuinely has three different closes at
    one instant, which Round 6.6e.2b discovered and kept as a free fixture.
    """
    from tests.test_ict_composite_fixture import divergent_snapshot

    result = stage(runs_dir, trade_plan_run, observe=observe(divergent_snapshot()))

    assert not result.succeeded
    assert result.error is not None
    assert "closing price" in str(result.error)
    assert_failed_cleanly(runs_dir, trade_plan_run)


def test_3_invalid_ranking_json_stops_the_run(runs_dir: Path, trade_plan_run: str) -> None:
    result = stage(runs_dir, trade_plan_run, analyst=MalformedTradeAnalyst())

    assert not result.succeeded
    assert "did not return JSON" in str(result.error)
    assert_failed_cleanly(runs_dir, trade_plan_run)


def test_4_an_unknown_candidate_id_stops_the_run(runs_dir: Path, trade_plan_run: str) -> None:
    result = stage(runs_dir, trade_plan_run, analyst=HallucinatingTradeAnalyst())

    assert not result.succeeded
    assert "not a deterministic candidate" in str(result.error)
    assert_failed_cleanly(runs_dir, trade_plan_run)


class DroppingAnalyst:
    """Returns a ranking with one candidate quietly missing."""

    provider = "fake"
    model = "dropping"

    def rank(self, request: TradeAnalystRequest) -> TradeAnalystResponse:
        from goldpipeline.adapters.fake_trade_analyst import BUCKET_KEYS, _payload_of

        buckets = _payload_of(request)["buckets"]
        answer = {key: list(buckets[key]) for key in BUCKET_KEYS}
        answer["bai_entry_candidate_ids"] = answer["bai_entry_candidate_ids"][1:]
        return TradeAnalystResponse(text=json.dumps(answer), model=self.model, provider="fake")


class SwappingAnalyst:
    """Puts a BAI candidate in the SEO bucket."""

    provider = "fake"
    model = "swapping"

    def rank(self, request: TradeAnalystRequest) -> TradeAnalystResponse:
        from goldpipeline.adapters.fake_trade_analyst import BUCKET_KEYS, _payload_of

        buckets = _payload_of(request)["buckets"]
        answer = {key: list(buckets[key]) for key in BUCKET_KEYS}
        moved = answer["bai_entry_candidate_ids"].pop()
        answer["seo_entry_candidate_ids"].append(moved)
        return TradeAnalystResponse(text=json.dumps(answer), model=self.model, provider="fake")


def test_5_a_missing_candidate_id_stops_the_run(runs_dir: Path, trade_plan_run: str) -> None:
    result = stage(runs_dir, trade_plan_run, analyst=DroppingAnalyst())

    assert not result.succeeded
    assert "missing candidates" in str(result.error)
    assert_failed_cleanly(runs_dir, trade_plan_run)


def test_6_a_bucket_mismatch_stops_the_run(runs_dir: Path, trade_plan_run: str) -> None:
    """§12, §25. Moving one id between buckets is refused in full.

    The message names the bucket that is now short rather than the one that
    gained an id, because the validator walks the buckets in order and BAI is
    first. Either way the answer is a refusal and nothing is repaired - which is
    the property that matters.
    """
    result = stage(runs_dir, trade_plan_run, analyst=SwappingAnalyst())

    assert not result.succeeded
    assert "missing candidates" in str(result.error)
    assert_failed_cleanly(runs_dir, trade_plan_run)


def test_7_a_price_in_the_ranking_stops_the_run(runs_dir: Path, trade_plan_run: str) -> None:
    """§13. The model may return ids. Anything else is refused in full."""
    result = stage(
        runs_dir,
        trade_plan_run,
        analyst=ScriptedTradeAnalyst(text=json.dumps({"entry": "4010", "stop_loss": "3990"})),
    )

    assert not result.succeeded
    assert_failed_cleanly(runs_dir, trade_plan_run)


def test_8_a_gate_failure_leaves_the_run_blocked(runs_dir: Path, trade_plan_run: str) -> None:
    """§17. A tampered page never reaches READY_TO_PUBLISH."""
    from goldpipeline.services.trade_plan_gate import gate_trade_plan

    stage(runs_dir, trade_plan_run)
    run = RunStore(runs_dir).open(trade_plan_run)
    # Rewrite the published page behind the manifest's back.
    run.artifact_path(FINAL_ARTICLE_FILENAME).write_text(
        "SEO\n9999–9999\n\nBAI\n—\n", encoding="utf-8"
    )

    decision = gate_trade_plan(run_id=trade_plan_run, store=RunStore(runs_dir))

    assert decision.decision is Decision.BLOCKED
    assert decision.blockers
    assert RunStore(runs_dir).open(trade_plan_run).load_manifest().status is (
        RunStatus.PUBLISH_BLOCKED
    )


def test_9_a_blocked_plan_is_never_delivered(runs_dir: Path, trade_plan_run: str) -> None:
    """§17, §21."""
    from goldpipeline.services.trade_plan_gate import gate_trade_plan

    stage(runs_dir, trade_plan_run)
    run = RunStore(runs_dir).open(trade_plan_run)
    run.artifact_path(FINAL_ARTICLE_FILENAME).write_text("SEO\n—\n\nBAI\n—\n", encoding="utf-8")
    gate_trade_plan(run_id=trade_plan_run, store=RunStore(runs_dir))

    client = FakePublisherClient()
    outcome = deliver(runs_dir, trade_plan_run, client)

    assert outcome.status is ReviewDeliveryStatus.SKIPPED
    assert client.sent == []


def test_10_a_run_with_no_candidates_still_produces_a_valid_page(
    runs_dir: Path, trade_plan_run: str
) -> None:
    """§25, §21. Nothing to publish is a state, not a failure."""
    from goldpipeline.schemas.ict import IctMarketSnapshot

    shot = staggered_snapshot()
    quiet = IctMarketSnapshot(
        observed_at=shot.observed_at,
        symbol=shot.symbol,
        provider=shot.provider,
        provider_symbol=shot.provider_symbol,
        timeframes=shot.timeframes[:1],
    )
    result = stage(runs_dir, trade_plan_run, observe=observe(quiet))

    assert result.succeeded
    article = (
        RunStore(runs_dir)
        .open(trade_plan_run)
        .read_artifact_bytes(FINAL_ARTICLE_FILENAME)
        .decode("utf-8")
        .rstrip("\n")
    )
    assert "SEO" in article and "BAI" in article
    assert len(article) <= 650


def test_11_one_empty_side_renders_the_em_dash(runs_dir: Path, trade_plan_run: str) -> None:
    """§25. The realistic reading has no eligible reference and one thin side."""
    drive(runs_dir, trade_plan_run)
    article = (
        RunStore(runs_dir)
        .open(trade_plan_run)
        .read_artifact_bytes(FINAL_ARTICLE_FILENAME)
        .decode("utf-8")
    )

    assert "SEO" in article
    assert "BAI" in article


def test_12_a_review_delivery_failure_leaves_the_run_recoverable(
    runs_dir: Path, trade_plan_run: str
) -> None:
    """§21, §25. Reuses the existing reservation semantics unchanged."""
    from goldpipeline.adapters.fake_publisher import ambiguous_client

    drive(runs_dir, trade_plan_run)
    outcome = deliver(runs_dir, trade_plan_run, ambiguous_client())

    assert outcome.status in {
        ReviewDeliveryStatus.UNCERTAIN,
        ReviewDeliveryStatus.FAILED,
    }
    run = RunStore(runs_dir).open(trade_plan_run)
    assert run.load_manifest().status is RunStatus.READY_TO_PUBLISH
    assert not run.has_artifact("publish_intent.json")


def test_13_a_second_stage_attempt_is_refused(runs_dir: Path, trade_plan_run: str) -> None:
    """Runs are immutable. A second plan would overwrite evidence."""
    assert stage(runs_dir, trade_plan_run).succeeded
    RunStore(runs_dir).open(trade_plan_run)

    second = write_trade_plan(
        run_id=trade_plan_run,
        store=RunStore(runs_dir),
        observe=observe(),
        analyst=EchoTradeAnalyst(),
    )

    assert not second.succeeded
    assert "already has trade plan artifacts" in str(second.error) or "needs" in str(second.error)


# --------------------------------------------------------------------------
# §23: the other two products are untouched
# --------------------------------------------------------------------------


def _recording_market(calls: list[str]) -> Any:
    def observe_it() -> MultiTimeframeObservation:
        calls.append("market")
        return observation()

    return observe_it


def _recording_analyst(calls: list[str]) -> Any:
    def build(selection: Any) -> Any:
        calls.append("analyst")
        return EchoTradeAnalyst()

    return build


def test_an_analysis_run_still_goes_through_its_writer(runs_dir: Path, tmp_path: Path) -> None:
    """§23. A trade plan's branch is reached only by a trade plan."""
    routing = importlib.import_module("tests.test_article_routing_pipeline")

    created = make_normalized_run(runs_dir, tmp_path)
    tracked = routing.make_tracked_clients()
    run_it = routing.run_it
    result = run_it(runs_dir, created.run_id, tracked)

    assert result.status is not PipelineStatus.FAILED
    assert "writer" in tracked.built
    assert RunStore(runs_dir).open(created.run_id).has_artifact("claude_draft.md")


def test_only_a_trade_plan_run_reaches_the_market_seam(runs_dir: Path, tmp_path: Path) -> None:
    """§23. An ANALYSIS Run never fetches five timeframes."""
    routing = importlib.import_module("tests.test_article_routing_pipeline")

    created = make_normalized_run(runs_dir, tmp_path)
    calls: list[str] = []

    base = routing.make_tracked_clients().as_pipeline_clients()
    result = resume_pipeline(
        run_id=created.run_id,
        store=RunStore(runs_dir),
        clients=replace(
            base,
            trade_plan_market=_recording_market(calls),
            trade_plan_analyst=_recording_analyst(calls),
        ),
        mode=PipelineMode.READY_FOR_PUBLISH,
    )

    assert result.status is not PipelineStatus.FAILED
    assert calls == []
