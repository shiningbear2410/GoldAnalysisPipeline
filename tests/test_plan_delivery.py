"""Round 6.7 §I, §M-§O, AP 22-26: answering a ``/plan``, once, without touching the Run.

Two halves. The first drives :func:`deliver_plan_replies` directly against real
Runs in each state a request can end in. The second drives a whole scheduler
tick with the command bot on, which is the only place the brief's promises about
idle ticks, review delivery and publishing can actually be observed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from goldpipeline.adapters.fake_command_bot import (
    FAKE_BOT_ID,
    FAKE_BOT_USERNAME,
    FAKE_CHAT_ID,
    FakeCommandBot,
    plan_update,
)
from goldpipeline.adapters.fake_plan_copywriter import EchoPlanCopywriter
from goldpipeline.adapters.fake_trade_analyst import EchoTradeAnalyst
from goldpipeline.domain.errors import PublisherTransportAmbiguousError
from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.automation import RetryClass
from goldpipeline.schemas.ingestion import LedgerEntry
from goldpipeline.schemas.manifest import RunStatus
from goldpipeline.schemas.orchestration import PipelineMode
from goldpipeline.schemas.plan_command import (
    CommandDeliveryIntent,
    DeliveryKind,
    DeliveryStatus,
    PlanCommandSettings,
    PlanReceipt,
)
from goldpipeline.schemas.review_delivery import INTENT_FILENAME, RESULT_FILENAME
from goldpipeline.services.automation import run_tick
from goldpipeline.services.automation_state import AutomationStore
from goldpipeline.services.inbox import INDEX, Inbox, Ledger
from goldpipeline.services.orchestrator import PipelineClients, resume_pipeline
from goldpipeline.services.plan_command import (
    ACK_TEXT,
    RESPONSE_KEY,
    PlanCommandStore,
    plan_event_id,
)
from goldpipeline.services.plan_delivery import (
    FAILURE_TEXT,
    GIVE_UP_AFTER,
    deliver_plan_replies,
)
from goldpipeline.services.review_delivery import is_eligible
from goldpipeline.services.trade_plan_gate import gate_trade_plan
from goldpipeline.services.trade_plan_stage import FINAL_ARTICLE_FILENAME, write_trade_plan
from goldpipeline.storage.run_store import RunStore
from tests.conftest import INGEST_NOW, make_mt5_source, make_normalized_run, make_worker_context
from tests.test_trade_plan_live import observe

NOW = datetime(2026, 9, 10, 1, 0, tzinfo=UTC)
EVENT = plan_event_id(bot_id=FAKE_BOT_ID, chat_id=FAKE_CHAT_ID, message_id=100)
SETTINGS = PlanCommandSettings(
    enabled=True,
    bot_id=FAKE_BOT_ID,
    bot_username=FAKE_BOT_USERNAME,
    authorised_chat_ids=[FAKE_CHAT_ID],
)


def explode(name: str) -> Any:
    def fail(*_: Any) -> Any:
        raise AssertionError(f"an idle tick must not reach the {name}")

    return fail


def plan_clients() -> PipelineClients:
    return PipelineClients(
        trade_plan_market=observe(),
        trade_plan_analyst=lambda selection: EchoTradeAnalyst(),
        trade_plan_copywriter=lambda selection: EchoPlanCopywriter(),
        trade_plan_news=None,
    )


def normalized_plan_run(runs_dir: Path, tmp_path: Path) -> str:
    created = make_normalized_run(runs_dir, tmp_path)
    run = RunStore(runs_dir).open(created.run_id)
    manifest = run.load_manifest()
    assert manifest.provenance is not None
    manifest.provenance.article_type = ArticleType.TRADE_PLAN
    run.save_manifest(manifest)
    return str(created.run_id)


def ready_plan_run(runs_dir: Path, tmp_path: Path) -> str:
    run_id = normalized_plan_run(runs_dir, tmp_path)
    resume_pipeline(
        run_id=run_id,
        store=RunStore(runs_dir),
        clients=plan_clients(),
        mode=PipelineMode.READY_FOR_PUBLISH,
    )
    assert RunStore(runs_dir).open(run_id).load_manifest().status is RunStatus.READY_TO_PUBLISH
    return run_id


@dataclass
class Desk:
    """One accepted request, its ledger mapping, and the means to answer it."""

    runs_dir: Path
    inbox: Inbox
    ledger: Ledger
    store: PlanCommandStore
    automation: AutomationStore
    bot: FakeCommandBot
    built: list[str]

    def deliver(self, *, now: datetime = NOW, settings: PlanCommandSettings = SETTINGS) -> Any:
        def factory() -> FakeCommandBot:
            self.built.append("bot")
            return self.bot

        return deliver_plan_replies(
            settings=settings,
            store=self.store,
            runs=RunStore(self.runs_dir),
            inbox=self.inbox,
            ledger=self.ledger,
            automation=self.automation,
            bot_factory=factory,
            now=now,
        )

    def map_to(self, run_id: str) -> None:
        self.ledger.reserve(
            LedgerEntry(
                event_id=EVENT,
                source="telegram_plan_command",
                payload_sha256="0" * 64,
                run_id=run_id,
            )
        )


@pytest.fixture
def desk(runs_dir: Path, tmp_path: Path) -> Desk:
    inbox = Inbox(tmp_path / "inbox")
    inbox.ensure_layout()
    store = PlanCommandStore(tmp_path / "automation")
    store.write_receipt(
        PlanReceipt(
            event_id=EVENT,
            update_id=1,
            bot_id=FAKE_BOT_ID,
            chat_id=FAKE_CHAT_ID,
            message_id=100,
            message_date=NOW,
            accepted_at=NOW,
        )
    )
    return Desk(
        runs_dir=runs_dir,
        inbox=inbox,
        ledger=Ledger(inbox.directory(INDEX)),
        store=store,
        automation=AutomationStore(tmp_path / "automation"),
        bot=FakeCommandBot(),
        built=[],
    )


def article(runs_dir: Path, run_id: str) -> str:
    raw = RunStore(runs_dir).open(run_id).read_artifact_bytes(FINAL_ARTICLE_FILENAME)
    return raw.decode("utf-8").rstrip("\n")


def run_fingerprint(runs_dir: Path, run_id: str) -> dict[str, str]:
    root = RunStore(runs_dir).open(run_id).path
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# --------------------------------------------------------------------------
# AP 22-26: the plan, once, and nothing else moves
# --------------------------------------------------------------------------


def test_22_the_plan_is_sent_to_the_requesting_chat_exactly_once(
    desk: Desk, runs_dir: Path, tmp_path: Path
) -> None:
    run_id = ready_plan_run(runs_dir, tmp_path)
    desk.map_to(run_id)

    first = desk.deliver()
    second = desk.deliver()

    assert [(result.kind, result.status) for result in first] == [
        (DeliveryKind.PLAN, DeliveryStatus.DELIVERED)
    ]
    assert second == []
    assert desk.bot.sent == [(FAKE_CHAT_ID, article(runs_dir, run_id))]
    assert desk.bot.calls == ["sendMessage"]


def test_22_the_plan_is_sent_exactly_as_rendered(
    desk: Desk, runs_dir: Path, tmp_path: Path
) -> None:
    """§AM. No header, no code fence, no wrapper: the approved page, verbatim."""
    run_id = ready_plan_run(runs_dir, tmp_path)
    desk.map_to(run_id)

    desk.deliver()
    [(_, text)] = desk.bot.sent

    assert text == article(runs_dir, run_id)
    assert text.startswith("🎯 KẾ HOẠCH VÀNG — ")
    assert "`" not in text


def test_23_an_uncertain_reply_is_never_resent(desk: Desk, runs_dir: Path, tmp_path: Path) -> None:
    desk.map_to(ready_plan_run(runs_dir, tmp_path))
    desk.bot.send_failure = PublisherTransportAmbiguousError("timed out", reason="timeout")

    [result] = desk.deliver()
    desk.bot.send_failure = None
    again = desk.deliver()

    assert result.status is DeliveryStatus.UNCERTAIN
    assert again == []
    assert desk.bot.calls == ["sendMessage"]


def test_23_an_orphaned_reply_intent_is_closed_without_sending(
    desk: Desk, runs_dir: Path, tmp_path: Path
) -> None:
    desk.map_to(ready_plan_run(runs_dir, tmp_path))
    desk.store.reserve(
        f"{EVENT}.{RESPONSE_KEY}",
        CommandDeliveryIntent(
            event_id=EVENT,
            kind=DeliveryKind.PLAN,
            attempt_id="crashed",
            created_at=NOW,
            chat_id=FAKE_CHAT_ID,
            text_sha256="0" * 64,
            char_count=10,
        ),
    )

    [result] = desk.deliver()

    assert result.status is DeliveryStatus.UNCERTAIN
    assert desk.bot.sent == []


def test_24_the_run_stays_ready_to_publish_and_is_not_touched(
    desk: Desk, runs_dir: Path, tmp_path: Path
) -> None:
    run_id = ready_plan_run(runs_dir, tmp_path)
    desk.map_to(run_id)
    before = run_fingerprint(runs_dir, run_id)

    desk.deliver()

    assert run_fingerprint(runs_dir, run_id) == before
    assert RunStore(runs_dir).open(run_id).load_manifest().status is RunStatus.READY_TO_PUBLISH


def test_25_review_delivery_is_untouched(desk: Desk, runs_dir: Path, tmp_path: Path) -> None:
    """The command reply consumes nothing of the review copy's one-attempt budget."""
    run_id = ready_plan_run(runs_dir, tmp_path)
    desk.map_to(run_id)

    desk.deliver()
    run = RunStore(runs_dir).open(run_id)

    assert not run.has_artifact(INTENT_FILENAME)
    assert not run.has_artifact(RESULT_FILENAME)
    reason = is_eligible(run, run.load_manifest(), now=datetime.now(UTC), max_run_age_minutes=10**7)
    assert reason is None


def test_26_nothing_is_published(desk: Desk, runs_dir: Path, tmp_path: Path) -> None:
    run_id = ready_plan_run(runs_dir, tmp_path)
    desk.map_to(run_id)

    desk.deliver()
    run = RunStore(runs_dir).open(run_id)

    assert not run.has_artifact("publish_intent.json")
    assert not run.has_artifact("publish_result.json")
    assert run.load_manifest().status is not RunStatus.PUBLISHED


# --------------------------------------------------------------------------
# §O: one failure notice, and only for a real ending
# --------------------------------------------------------------------------


def test_a_blocked_plan_gets_one_failure_notice(desk: Desk, runs_dir: Path, tmp_path: Path) -> None:
    run_id = normalized_plan_run(runs_dir, tmp_path)
    store = RunStore(runs_dir)
    write_trade_plan(
        run_id=run_id,
        store=store,
        observe=observe(),
        analyst=EchoTradeAnalyst(),
        copywriter=EchoPlanCopywriter(),
    )
    store.open(run_id).artifact_path(FINAL_ARTICLE_FILENAME).write_text(
        "tampered", encoding="utf-8"
    )
    gate_trade_plan(run_id=run_id, store=store)
    desk.map_to(run_id)

    first = desk.deliver()
    second = desk.deliver()

    assert [result.kind for result in first] == [DeliveryKind.FAILURE_NOTICE]
    assert second == []
    assert desk.bot.sent == [(FAKE_CHAT_ID, FAILURE_TEXT)]
    assert FAILURE_TEXT == "Không tạo được kế hoạch lúc này. Hãy thử lại sau."


def test_no_notice_while_a_bounded_retry_is_still_running(
    desk: Desk, runs_dir: Path, tmp_path: Path
) -> None:
    run_id = normalized_plan_run(runs_dir, tmp_path)
    desk.map_to(run_id)
    desk.automation.record_failure(
        run_id, failure_code="WRITER_TIMEOUT", retry_class=RetryClass.TRANSIENT, now=NOW
    )

    assert desk.deliver(now=NOW + GIVE_UP_AFTER * 2) == []
    assert desk.built == []


@pytest.mark.parametrize("retry_class", [RetryClass.PERMANENT, RetryClass.TERMINAL])
def test_a_retry_that_will_never_happen_gets_the_notice(
    desk: Desk, runs_dir: Path, tmp_path: Path, retry_class: RetryClass
) -> None:
    run_id = normalized_plan_run(runs_dir, tmp_path)
    desk.map_to(run_id)
    desk.automation.record_failure(run_id, failure_code="X", retry_class=retry_class, now=NOW)

    [result] = desk.deliver()

    assert result.kind is DeliveryKind.FAILURE_NOTICE
    assert desk.bot.sent == [(FAKE_CHAT_ID, FAILURE_TEXT)]


def test_an_expired_request_gets_the_notice(desk: Desk) -> None:
    expired = desk.inbox.root / "expired"
    expired.mkdir()
    (expired / f"{EVENT}.json").write_text("{}", encoding="utf-8")

    [result] = desk.deliver()

    assert result.kind is DeliveryKind.FAILURE_NOTICE
    assert result.run_id is None


def test_a_waiting_request_is_left_alone_until_the_give_up_window(desk: Desk) -> None:
    assert desk.deliver() == []
    assert desk.deliver(now=NOW + GIVE_UP_AFTER) == []
    assert desk.built == []

    [result] = desk.deliver(now=NOW + GIVE_UP_AFTER + timedelta(seconds=1))
    assert result.kind is DeliveryKind.FAILURE_NOTICE


def test_a_chat_no_longer_authorised_is_recorded_and_not_sent(
    desk: Desk, runs_dir: Path, tmp_path: Path
) -> None:
    desk.map_to(ready_plan_run(runs_dir, tmp_path))
    other = SETTINGS.model_copy(update={"authorised_chat_ids": ["42"]})

    [result] = desk.deliver(settings=other)

    assert result.status is DeliveryStatus.SKIPPED
    assert desk.bot.sent == []


# --------------------------------------------------------------------------
# §I: whole ticks
# --------------------------------------------------------------------------


def tick_context(
    tmp_path: Path,
    runs_dir: Path,
    *,
    bot: FakeCommandBot,
    clients: PipelineClients,
    enabled: bool = True,
) -> Any:
    inbox = Inbox(tmp_path / "tick-inbox")
    inbox.ensure_layout()
    context = make_worker_context(
        inbox,
        runs_dir,
        tmp_path / "tick-automation",
        clients=clients,
        market_source=make_mt5_source(),
    )
    return replace(
        context,
        plan_command=SETTINGS if enabled else PlanCommandSettings(),
        plan_bot=(lambda: bot) if enabled else None,
    )


def idle_clients() -> PipelineClients:
    return PipelineClients(
        trade_plan_market=explode("market"),
        trade_plan_analyst=explode("analyst"),
        trade_plan_copywriter=explode("copywriter"),
        trade_plan_news=explode("news"),
    )


def test_an_idle_tick_makes_exactly_one_poll_and_nothing_else(
    tmp_path: Path, runs_dir: Path
) -> None:
    """§I. One getUpdates; no market, no news, no model, no Run, no send."""
    bot = FakeCommandBot()
    context = tick_context(tmp_path, runs_dir, bot=bot, clients=idle_clients())

    result = run_tick(context, now=INGEST_NOW)

    assert bot.calls == ["getUpdates"]
    assert result.plan_command_polled is True
    assert result.plan_commands == []
    assert result.plan_replies == []
    assert RunStore(runs_dir).list_run_ids() == []


def test_with_the_command_off_a_tick_makes_no_telegram_request(
    tmp_path: Path, runs_dir: Path
) -> None:
    """The tick this worker did before Round 6.7, byte for byte."""
    bot = FakeCommandBot(updates=[plan_update(1)])
    context = tick_context(tmp_path, runs_dir, bot=bot, clients=idle_clients(), enabled=False)

    result = run_tick(context, now=INGEST_NOW)

    assert bot.calls == []
    assert result.plan_command_polled is False
    assert context.inbox.pending() == []


def test_an_unauthorised_plan_reaches_nothing(tmp_path: Path, runs_dir: Path) -> None:
    """§D. No Run, no TradingView, no Anthropic, no reply."""
    bot = FakeCommandBot(updates=[plan_update(1, chat_id="5550001234")])
    context = tick_context(tmp_path, runs_dir, bot=bot, clients=idle_clients())

    run_tick(context, now=INGEST_NOW)

    assert bot.calls == ["getUpdates"]
    assert RunStore(runs_dir).list_run_ids() == []


def test_a_plan_request_becomes_a_ready_plan_in_the_requesting_chat(
    tmp_path: Path, runs_dir: Path
) -> None:
    """The whole flow in one tick: poll, submit, ingest, stage, gate, reply."""
    date = int(INGEST_NOW.timestamp())
    bot = FakeCommandBot(updates=[plan_update(1, date=date)])
    context = tick_context(tmp_path, runs_dir, bot=bot, clients=plan_clients())

    first = run_tick(context, now=INGEST_NOW)
    [run_id] = RunStore(runs_dir).list_run_ids()
    run = RunStore(runs_dir).open(run_id)

    assert run.load_manifest().status is RunStatus.READY_TO_PUBLISH
    assert bot.sent == [(FAKE_CHAT_ID, ACK_TEXT), (FAKE_CHAT_ID, article(runs_dir, run_id))]
    assert [item.code for item in first.plan_replies] == ["PLAN_DELIVERED"]
    assert not run.has_artifact(INTENT_FILENAME)
    assert not run.has_artifact("publish_intent.json")

    second = run_tick(context, now=INGEST_NOW + timedelta(minutes=1))

    assert bot.calls == ["getUpdates", "sendMessage", "sendMessage", "getUpdates"]
    assert second.plan_replies == []
    assert RunStore(runs_dir).list_run_ids() == [run_id]


def test_a_poll_failure_costs_one_line_and_the_tick_still_runs(
    tmp_path: Path, runs_dir: Path
) -> None:
    from goldpipeline.adapters.telegram_command_bot import CommandBotTransportError

    bot = FakeCommandBot(poll_failure=CommandBotTransportError("down", method="getUpdates"))
    context = tick_context(tmp_path, runs_dir, bot=bot, clients=idle_clients())

    result = run_tick(context, now=INGEST_NOW)

    assert [item.code for item in result.plan_commands] == ["COMMAND_BOT_TRANSPORT"]
    assert "COMMAND_BOT_TRANSPORT" in result.errors
