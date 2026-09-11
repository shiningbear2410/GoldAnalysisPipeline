"""Round 6.7 §F-§L: a ``/plan`` tapped in Telegram becomes exactly one inbox event.

Everything here runs against :class:`FakeCommandBot`: scripted updates in,
recorded sends out, no network. The properties that matter are the ones a real
scheduler would break first - a replayed update, a crash between two writes, an
acknowledgement whose outcome is unknown - so each has its own test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from goldpipeline.adapters.fake_command_bot import (
    FAKE_BOT_ID,
    FAKE_BOT_USERNAME,
    FAKE_CHAT_ID,
    MESSAGE_DATE,
    FakeCommandBot,
    plan_update,
)
from goldpipeline.adapters.telegram_command_bot import CommandBotTransportError
from goldpipeline.domain.errors import (
    PipelineError,
    PublisherRejectedError,
    PublisherTransportAmbiguousError,
)
from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.inbox import EVENT_ID_PATTERN, AnalysisEvent
from goldpipeline.schemas.ingestion import LedgerEntry
from goldpipeline.schemas.plan_command import (
    CommandDeliveryIntent,
    DeliveryKind,
    DeliveryStatus,
    IgnoreReason,
    PlanCommandSettings,
)
from goldpipeline.services.inbox import INDEX, Inbox, Ledger
from goldpipeline.services.plan_command import (
    ACK_KEY,
    ACK_TEXT,
    PlanCommandStore,
    PollReport,
    parse_plan_command,
    plan_event_id,
    poll_plan_commands,
)

NOW = datetime(2026, 9, 10, 0, 27, tzinfo=UTC)
STRANGER = "5550001234"
SETTINGS = PlanCommandSettings(
    enabled=True,
    bot_id=FAKE_BOT_ID,
    bot_username=FAKE_BOT_USERNAME,
    authorised_chat_ids=[FAKE_CHAT_ID],
)


@dataclass
class World:
    inbox: Inbox
    store: PlanCommandStore
    ledger: Ledger
    bot: FakeCommandBot

    def poll(self) -> PollReport:
        return poll_plan_commands(
            settings=SETTINGS,
            store=self.store,
            inbox=self.inbox,
            ledger=self.ledger,
            bot=self.bot,
            now=NOW,
        )

    def events(self) -> list[dict[str, Any]]:
        return [json.loads(path.read_text(encoding="utf-8")) for path in self.inbox.pending()]

    def forget_offset(self) -> None:
        (self.store.state / "offset.json").unlink()


@pytest.fixture
def world(tmp_path: Path) -> World:
    inbox = Inbox(tmp_path / "inbox")
    inbox.ensure_layout()
    return World(
        inbox=inbox,
        store=PlanCommandStore(tmp_path / "automation"),
        ledger=Ledger(inbox.directory(INDEX)),
        bot=FakeCommandBot(),
    )


# --------------------------------------------------------------------------
# AP 1-2: the command
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text", ["/plan", "  /plan  ", "\n/plan\t", "/plan@pcplanbot", "/plan@PCPlanBot"]
)
def test_1_the_parser_accepts_exactly_plan(text: str) -> None:
    assert parse_plan_command(text, bot_username="pcplanbot")


@pytest.mark.parametrize(
    "text",
    [
        "/plan now",
        "/plans",
        "/Plan",
        "plan",
        "/plan@otherbot",
        "/plan@",
        "/start",
        "/help",
        "",
        "   ",
        "/plan /plan",
        "/plan@pcplanbot extra",
    ],
)
def test_1_the_parser_refuses_everything_else(text: str) -> None:
    assert not parse_plan_command(text, bot_username="pcplanbot")


@pytest.mark.parametrize("value", [None, 5, {"text": "/plan"}, ["/plan"]])
def test_1_the_parser_refuses_what_is_not_text(value: object) -> None:
    assert not parse_plan_command(value, bot_username="pcplanbot")


def test_2_the_addressed_form_is_accepted(world: World) -> None:
    world.bot.updates = [plan_update(1, text="/plan@pcplanbot")]

    report = world.poll()

    assert len(report.accepted) == 1
    assert len(world.events()) == 1


# --------------------------------------------------------------------------
# AP 3: authorisation
# --------------------------------------------------------------------------


def test_3_an_unauthorised_chat_is_ignored_silently(world: World) -> None:
    """§D. No reply, no event, no receipt - one audit line without the chat id."""
    world.bot.updates = [plan_update(1, chat_id=STRANGER)]

    report = world.poll()

    assert report.accepted == []
    assert report.ignored == [IgnoreReason.UNAUTHORISED_CHAT]
    assert world.events() == []
    assert world.bot.sent == []
    assert world.bot.calls == ["getUpdates"]
    assert world.store.receipts() == []
    assert world.store.read_offset() == 2

    audit = (world.store.state / "ignored" / "1.json").read_text(encoding="utf-8")
    assert "UNAUTHORISED_CHAT" in audit
    assert f'"{STRANGER}"' not in audit
    assert STRANGER not in json.loads(audit).values()


@pytest.mark.parametrize(
    ("update", "reason"),
    [
        (plan_update(1, text="/start"), IgnoreReason.NOT_A_PLAN_COMMAND),
        (plan_update(1, text="làm kế hoạch đi"), IgnoreReason.NOT_A_PLAN_COMMAND),
        (plan_update(1, text=None), IgnoreReason.NOT_A_PLAN_COMMAND),
        (plan_update(1, from_bot=True), IgnoreReason.FROM_A_BOT),
        ({"update_id": 1, "edited_message": {}}, IgnoreReason.NOT_A_MESSAGE),
        ({"update_id": 1, "message": {"chat": {}}}, IgnoreReason.MALFORMED),
    ],
)
def test_3_everything_else_is_ignored_too(
    world: World, update: dict[str, Any], reason: IgnoreReason
) -> None:
    world.bot.updates = [update]

    report = world.poll()

    assert report.ignored == [reason]
    assert world.events() == []
    assert world.bot.sent == []


# --------------------------------------------------------------------------
# AP 4-5: idempotency and crash replay
# --------------------------------------------------------------------------


def test_4_the_same_message_twice_is_one_event(world: World) -> None:
    world.bot.updates = [plan_update(1, message_id=77), plan_update(2, message_id=77)]

    report = world.poll()

    assert len(report.accepted) == 1
    assert len(report.duplicates) == 1
    assert len(world.events()) == 1
    assert world.bot.sent == [(FAKE_CHAT_ID, ACK_TEXT)]


def test_4_two_taps_are_two_requests(world: World) -> None:
    world.bot.updates = [plan_update(1, message_id=77), plan_update(2, message_id=78)]

    report = world.poll()

    assert len(report.accepted) == 2
    assert len(world.events()) == 2
    assert world.bot.sent == [(FAKE_CHAT_ID, ACK_TEXT), (FAKE_CHAT_ID, ACK_TEXT)]


def test_4_a_replayed_update_is_not_a_second_request(world: World) -> None:
    world.bot.updates = [plan_update(1)]
    world.poll()
    world.forget_offset()

    report = world.poll()

    assert report.accepted == []
    assert len(report.duplicates) == 1
    assert len(world.events()) == 1


def test_5_a_crash_before_the_receipt_replays_without_a_second_event(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§J. Submission is durable before anything else; the offset moves last."""
    world.bot.updates = [plan_update(1)]
    original = PlanCommandStore.write_receipt

    def crash(self: PlanCommandStore, receipt: Any) -> None:
        raise OSError("the process died here")

    monkeypatch.setattr(PlanCommandStore, "write_receipt", crash)
    with pytest.raises(OSError, match="died"):
        world.poll()

    assert len(world.events()) == 1
    assert world.store.read_offset() is None
    assert world.bot.sent == []

    monkeypatch.setattr(PlanCommandStore, "write_receipt", original)
    report = world.poll()

    assert report.accepted == []
    assert len(report.duplicates) == 1
    assert len(world.events()) == 1
    assert world.store.read_offset() == 2
    assert world.bot.sent == [(FAKE_CHAT_ID, ACK_TEXT)]


def test_5_an_event_already_processed_is_never_resubmitted(world: World) -> None:
    """The worker moved the event on; the command bot's own state was then lost."""
    world.bot.updates = [plan_update(1)]
    world.poll()
    claimed = world.inbox.claim(world.inbox.pending()[0])
    assert claimed is not None
    world.inbox.complete(claimed)
    world.forget_offset()
    for receipt in (world.store.state / "receipts").iterdir():
        receipt.unlink()

    report = world.poll()

    assert report.accepted == []
    assert len(report.duplicates) == 1
    assert world.inbox.pending() == []
    assert world.bot.calls.count("sendMessage") == 1


def test_5_a_ledger_entry_alone_prevents_resubmission(world: World) -> None:
    """An event the ingestion ledger has seen never becomes a second Run."""
    event_id = plan_event_id(bot_id=FAKE_BOT_ID, chat_id=FAKE_CHAT_ID, message_id=100)
    world.ledger.reserve(
        LedgerEntry(event_id=event_id, source="x", payload_sha256="0" * 64, run_id="r1")
    )
    world.bot.updates = [plan_update(1, message_id=100)]

    report = world.poll()

    assert report.accepted == []
    assert report.duplicates == [event_id]
    assert world.events() == []


def test_a_poll_failure_submits_nothing_and_keeps_the_offset(world: World) -> None:
    world.store.write_offset(5)
    world.bot.poll_failure = CommandBotTransportError("gone", method="getUpdates")

    with pytest.raises(PipelineError):
        world.poll()

    assert world.store.read_offset() == 5
    assert world.events() == []


def test_the_offset_moves_after_the_last_update_in_order(world: World) -> None:
    world.bot.updates = [plan_update(7, message_id=2), plan_update(5, message_id=1)]

    world.poll()

    assert world.store.read_offset() == 8
    assert [receipt.update_id for receipt in world.store.receipts()] == [5, 7]


# --------------------------------------------------------------------------
# AP 6: the acknowledgement
# --------------------------------------------------------------------------


def test_6_the_acknowledgement_is_sent_once(world: World) -> None:
    world.bot.updates = [plan_update(1)]

    world.poll()
    world.poll()
    world.forget_offset()
    world.poll()

    assert world.bot.sent == [(FAKE_CHAT_ID, ACK_TEXT)]
    assert ACK_TEXT == "🎯 Đã nhận /plan. Mình đang tạo kế hoạch vàng…"


def test_6_an_uncertain_acknowledgement_is_never_resent(world: World) -> None:
    world.bot.updates = [plan_update(1)]
    world.bot.send_failure = PublisherTransportAmbiguousError("timed out", reason="timeout")

    report = world.poll()

    assert [ack.status for ack in report.acknowledgements] == [DeliveryStatus.UNCERTAIN]
    world.bot.send_failure = None
    world.forget_offset()
    world.poll()

    assert world.bot.calls.count("sendMessage") == 1
    assert len(world.events()) == 1


def test_6_a_refused_acknowledgement_does_not_lose_the_request(world: World) -> None:
    world.bot.updates = [plan_update(1)]
    world.bot.send_failure = PublisherRejectedError("no", status_code=400)

    report = world.poll()

    assert [ack.status for ack in report.acknowledgements] == [DeliveryStatus.FAILED]
    assert len(world.events()) == 1


def test_6_an_orphaned_intent_is_closed_as_uncertain_without_sending(world: World) -> None:
    event_id = plan_event_id(bot_id=FAKE_BOT_ID, chat_id=FAKE_CHAT_ID, message_id=100)
    world.store.reserve(
        f"{event_id}.{ACK_KEY}",
        CommandDeliveryIntent(
            event_id=event_id,
            kind=DeliveryKind.ACKNOWLEDGEMENT,
            attempt_id="a1",
            created_at=NOW,
            chat_id=FAKE_CHAT_ID,
            text_sha256="0" * 64,
            char_count=5,
        ),
    )
    world.bot.updates = [plan_update(1, message_id=100)]

    world.poll()

    assert world.bot.sent == []
    result = world.store.read_result(f"{event_id}.{ACK_KEY}")
    assert result is not None
    assert result.status is DeliveryStatus.UNCERTAIN
    assert result.failure_code == "ORPHAN_REPLY_INTENT"


# --------------------------------------------------------------------------
# AP 7: the event
# --------------------------------------------------------------------------


def test_7_the_event_is_a_normal_trade_plan_event(world: World) -> None:
    world.bot.updates = [plan_update(41, message_id=4242)]

    world.poll()
    [payload] = world.events()
    event = AnalysisEvent.model_validate(payload)

    assert event.article_type is ArticleType.TRADE_PLAN
    assert event.raw_text == "/plan"
    assert event.source == "telegram_plan_command"
    assert event.event_id == plan_event_id(
        bot_id=FAKE_BOT_ID, chat_id=FAKE_CHAT_ID, message_id=4242
    )
    assert event.created_at == datetime.fromtimestamp(MESSAGE_DATE, UTC)
    assert event.metadata == {
        "request_source": "telegram_plan",
        "request_chat_id": FAKE_CHAT_ID,
        "request_message_id": 4242,
        "request_update_id": 41,
    }
    assert "token" not in json.dumps(payload).lower()


def test_7_the_idempotency_key_is_bot_chat_and_message() -> None:
    base = plan_event_id(bot_id=1, chat_id="2", message_id=3)

    assert base == plan_event_id(bot_id=1, chat_id="2", message_id=3)
    assert base != plan_event_id(bot_id=9, chat_id="2", message_id=3)
    assert base != plan_event_id(bot_id=1, chat_id="9", message_id=3)
    assert base != plan_event_id(bot_id=1, chat_id="2", message_id=9)
    assert EVENT_ID_PATTERN.fullmatch(base)


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------


def test_an_enabled_command_needs_a_bot_and_a_chat() -> None:
    with pytest.raises(ValueError, match="needs"):
        PlanCommandSettings(enabled=True)
    with pytest.raises(ValueError, match="bot username"):
        PlanCommandSettings(bot_username="notabotname")
    with pytest.raises(ValueError, match="numeric"):
        PlanCommandSettings(authorised_chat_ids=["@somechannel"])

    assert PlanCommandSettings(bot_username="@pcplanbot").bot_username == "pcplanbot"


def test_absent_settings_mean_off(tmp_path: Path) -> None:
    read = PlanCommandStore(tmp_path).read_settings()

    assert read.settings.enabled is False
    assert read.problem is None


def test_damaged_settings_mean_off_and_say_why(tmp_path: Path) -> None:
    store = PlanCommandStore(tmp_path)
    store.settings_path.write_text('{"enabled": true', encoding="utf-8")

    read = store.read_settings()

    assert read.settings.enabled is False
    assert read.problem is not None


def test_the_settings_have_nowhere_to_hold_a_token() -> None:
    fields = set(PlanCommandSettings.model_fields)

    assert not any("token" in name or "secret" in name for name in fields)
    with pytest.raises(ValueError):
        PlanCommandSettings.model_validate({"bot_token": "123:abc"})


def test_settings_round_trip_through_the_store(tmp_path: Path) -> None:
    store = PlanCommandStore(tmp_path)

    store.write_settings(SETTINGS)

    assert store.read_settings().settings == SETTINGS
