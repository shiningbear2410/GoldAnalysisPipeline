"""The ``/plan`` command: poll, authorise, submit durably, acknowledge once.

Round 6.7. A person taps ``/plan`` in the command bot, and a normal
``TRADE_PLAN`` event lands in the inbox the scheduler already drains. Nothing
here runs a pipeline, creates a Run or calls a model: the event is submitted and
the existing worker does the rest, exactly as it would for any other event.

The order is the safety argument:

1.  read one short batch of updates, oldest first;
2.  decide: a ``/plan`` from an authorised chat, or something to ignore;
3.  **submit the event**, atomically, under an id derived from the bot, the chat
    and the message - the same message can never become two events;
4.  write the receipt, then advance the offset;
5.  acknowledge, once, behind an intent.

**The offset moves last.** A crash anywhere before it means Telegram hands back
the same update next tick, and step 3 recognises the event it already submitted
- by its receipt, its inbox file or its ledger entry - so the replay creates
nothing. Moving the offset first would make a crash lose the request instead.

**Unauthorised chats get silence.** No reply, no Run, no market fetch, no model
call; only an audit line holding the update id, the reason and a digest of the
chat id. A stranger learns nothing, not even that the bot is listening.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from goldpipeline.adapters.telegram_command_bot import (
    CommandBotClient,
    CommandBotError,
    CommandBotResponseError,
    CommandBotTransportError,
)
from goldpipeline.domain.errors import (
    InboxPayloadError,
    LedgerError,
    PipelineError,
    PublisherError,
    PublisherResponseError,
    PublisherTransportAmbiguousError,
)
from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.inbox import AnalysisEvent
from goldpipeline.schemas.plan_command import (
    EVENT_SOURCE,
    REQUEST_SOURCE,
    CommandDeliveryIntent,
    CommandDeliveryResult,
    DeliveryKind,
    DeliveryStatus,
    IgnoredUpdate,
    IgnoreReason,
    PlanCommandSettings,
    PlanReceipt,
)
from goldpipeline.services.inbox import FAILED, INCOMING, PROCESSED, PROCESSING, Inbox, Ledger
from goldpipeline.storage.atomic import atomic_write_bytes, encode_json, sha256_bytes

logger = logging.getLogger(__name__)

ACK_TEXT = "🎯 Đã nhận /plan. Mình đang tạo kế hoạch vàng…"
PLAN_TEXT = "/plan"

PLAN_COMMAND_FILENAME = "plan_command.json"
"""The optional settings file, beside the automation state. Absent means off."""

STATE_DIRNAME = "plan_command"

EVENT_DIRECTORIES = (INCOMING, PROCESSING, PROCESSED, FAILED, "deferred", "expired")
"""Every inbox directory an event can be in. ``deferred`` and ``expired`` are the
automation worker's own two states, named identically there."""

ACK_KEY = "ack"
RESPONSE_KEY = "response"


class PlanCommandStateError(PipelineError):
    """The command bot's own state could not be read. Nothing is advanced."""

    code = "PLAN_COMMAND_STATE_UNREADABLE"


# --------------------------------------------------------------------------
# the command
# --------------------------------------------------------------------------


def parse_plan_command(text: object, *, bot_username: str) -> bool:
    """Exactly ``/plan``, or ``/plan@<this bot>``, with surrounding whitespace allowed.

    Nothing else: no arguments, no other command, no other bot's address. The
    command word is matched exactly; the bot's username, as Telegram does,
    without regard to case.
    """
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped or any(character.isspace() for character in stripped):
        return False
    command, at, target = stripped.partition("@")
    if command != PLAN_TEXT:
        return False
    if not at:
        return True
    return bool(target) and target.casefold() == bot_username.casefold()


def plan_event_id(*, bot_id: int, chat_id: str, message_id: int) -> str:
    """The idempotency key: one message in one chat to one bot is one event."""
    digest = hashlib.sha256(f"{bot_id}:{chat_id}:{message_id}".encode()).hexdigest()
    return f"tgplan-{digest[:32]}"


def chat_ref(chat_id: str) -> str:
    """A stable digest of a chat id, for audit lines that must not hold the id."""
    return hashlib.sha256(f"chat:{chat_id}".encode()).hexdigest()[:16]


@dataclass(frozen=True)
class PlanRequest:
    update_id: int
    chat_id: str
    message_id: int
    message_date: datetime


def classify_update(
    update: dict[str, Any], *, settings: PlanCommandSettings
) -> PlanRequest | tuple[IgnoreReason, str | None]:
    """A request to act on, or the reason not to. Authorisation comes first."""
    message = update.get("message")
    if not isinstance(message, dict):
        return IgnoreReason.NOT_A_MESSAGE, None

    chat = message.get("chat")
    raw_chat = chat.get("id") if isinstance(chat, dict) else None
    message_id = message.get("message_id")
    date = message.get("date")
    if (
        not isinstance(raw_chat, int)
        or not isinstance(message_id, int)
        or not isinstance(date, int)
    ):
        return IgnoreReason.MALFORMED, None

    chat_id = str(raw_chat)
    if chat_id not in settings.authorised_chat_ids:
        return IgnoreReason.UNAUTHORISED_CHAT, chat_id
    sender = message.get("from")
    if isinstance(sender, dict) and sender.get("is_bot") is True:
        return IgnoreReason.FROM_A_BOT, chat_id
    if not parse_plan_command(message.get("text"), bot_username=settings.bot_username or ""):
        return IgnoreReason.NOT_A_PLAN_COMMAND, chat_id

    return PlanRequest(
        update_id=int(update["update_id"]),
        chat_id=chat_id,
        message_id=message_id,
        message_date=datetime.fromtimestamp(date, UTC),
    )


def plan_event(request: PlanRequest, *, event_id: str) -> AnalysisEvent:
    """A normal TRADE_PLAN event, carrying where the request came from as data."""
    return AnalysisEvent(
        source=EVENT_SOURCE,
        event_id=event_id,
        created_at=request.message_date,
        raw_text=PLAN_TEXT,
        article_type=ArticleType.TRADE_PLAN,
        chat_id=int(request.chat_id),
        message_id=request.message_id,
        metadata={
            "request_source": REQUEST_SOURCE,
            "request_chat_id": request.chat_id,
            "request_message_id": request.message_id,
            "request_update_id": request.update_id,
        },
    )


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanCommandRead:
    settings: PlanCommandSettings
    problem: str | None = None


class PlanCommandStore:
    """Settings, offset, receipts, audit lines and reply records. All local files."""

    def __init__(self, automation_dir: Path | str) -> None:
        self.root = Path(automation_dir)
        self.state = self.root / STATE_DIRNAME

    @property
    def settings_path(self) -> Path:
        return self.root / PLAN_COMMAND_FILENAME

    # -- settings ----------------------------------------------------------

    def read_settings(self) -> PlanCommandRead:
        """The settings, or "off" - with the reason when the file is unusable.

        Never raises. A damaged file turns the command off rather than stopping
        the worker; the problem is reported on the tick so a person sees it.
        """
        if not self.settings_path.is_file():
            return PlanCommandRead(PlanCommandSettings())
        try:
            settings = PlanCommandSettings.model_validate_json(
                self.settings_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            logger.warning("plan_command.settings unusable: %s", type(exc).__name__)
            return PlanCommandRead(PlanCommandSettings(), problem=type(exc).__name__)
        return PlanCommandRead(settings)

    def write_settings(self, settings: PlanCommandSettings) -> Path:
        checked = PlanCommandSettings.model_validate(settings.model_dump(mode="json"))
        atomic_write_bytes(self.settings_path, encode_json(checked))
        return self.settings_path

    # -- offset ------------------------------------------------------------

    def read_offset(self) -> int | None:
        path = self.state / "offset.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))["next_offset"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise PlanCommandStateError("the command bot's offset is unreadable") from exc
        if not isinstance(value, int):
            raise PlanCommandStateError("the command bot's offset is not an integer")
        return value

    def write_offset(self, value: int) -> None:
        atomic_write_bytes(self.state / "offset.json", encode_json({"next_offset": value}))

    # -- receipts ----------------------------------------------------------

    def _receipt_path(self, event_id: str) -> Path:
        return self.state / "receipts" / f"{event_id}.json"

    def read_receipt(self, event_id: str) -> PlanReceipt | None:
        path = self._receipt_path(event_id)
        if not path.is_file():
            return None
        try:
            return PlanReceipt.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PlanCommandStateError(f"receipt {event_id} is unreadable") from exc

    def write_receipt(self, receipt: PlanReceipt) -> None:
        if not self._receipt_path(receipt.event_id).is_file():
            atomic_write_bytes(self._receipt_path(receipt.event_id), encode_json(receipt))

    def receipts(self) -> list[PlanReceipt]:
        directory = self.state / "receipts"
        if not directory.is_dir():
            return []
        found = [
            self.read_receipt(path.stem)
            for path in sorted(directory.iterdir())
            if path.is_file() and path.suffix == ".json"
        ]
        return sorted(
            (receipt for receipt in found if receipt is not None),
            key=lambda receipt: (receipt.accepted_at, receipt.event_id),
        )

    def record_ignored(self, entry: IgnoredUpdate) -> None:
        atomic_write_bytes(self.state / "ignored" / f"{entry.update_id}.json", encode_json(entry))

    # -- replies -----------------------------------------------------------

    def _reply_path(self, key: str, part: str) -> Path:
        return self.state / "replies" / f"{key}.{part}.json"

    def reserve(self, key: str, intent: CommandDeliveryIntent) -> bool:
        """Create the intent exclusively. ``False`` when one already exists."""
        path = self._reply_path(key, "intent")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(handle, "wb") as stream:
            stream.write(encode_json(intent))
            stream.flush()
            os.fsync(stream.fileno())
        return True

    def read_intent(self, key: str) -> CommandDeliveryIntent | None:
        path = self._reply_path(key, "intent")
        if not path.is_file():
            return None
        try:
            return CommandDeliveryIntent.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def has_intent(self, key: str) -> bool:
        return self._reply_path(key, "intent").is_file()

    def read_result(self, key: str) -> CommandDeliveryResult | None:
        path = self._reply_path(key, "result")
        if not path.is_file():
            return None
        try:
            return CommandDeliveryResult.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PlanCommandStateError(f"reply record {key} is unreadable") from exc

    def has_result(self, key: str) -> bool:
        return self._reply_path(key, "result").is_file()

    def settle(self, key: str, result: CommandDeliveryResult) -> None:
        atomic_write_bytes(self._reply_path(key, "result"), encode_json(result))

    def orphan_keys(self) -> list[str]:
        """Intents with no result: attempts that never recorded how they ended."""
        directory = self.state / "replies"
        if not directory.is_dir():
            return []
        keys = [
            path.name.removesuffix(".intent.json")
            for path in sorted(directory.iterdir())
            if path.name.endswith(".intent.json")
        ]
        return [key for key in keys if not self.has_result(key)]


# --------------------------------------------------------------------------
# replying, at most once
# --------------------------------------------------------------------------


def close_orphans(store: PlanCommandStore, *, now: datetime) -> list[CommandDeliveryResult]:
    """Record every unfinished reply as ``UNCERTAIN``. Sends nothing.

    The crash window the intent exists for: a previous tick reserved a reply and
    stopped, so Telegram may already hold the message. Resending it every minute
    would be the defect; recording that nobody knows is the fix.
    """
    closed: list[CommandDeliveryResult] = []
    for key in store.orphan_keys():
        intent = store.read_intent(key)
        event_id, _, suffix = key.rpartition(".")
        result = CommandDeliveryResult(
            event_id=intent.event_id if intent else event_id,
            kind=intent.kind
            if intent
            else (DeliveryKind.ACKNOWLEDGEMENT if suffix == ACK_KEY else DeliveryKind.PLAN),
            attempt_id=intent.attempt_id if intent else "unknown",
            status=DeliveryStatus.UNCERTAIN,
            chat_id=intent.chat_id if intent else "unknown",
            run_id=intent.run_id if intent else None,
            started_at=now,
            completed_at=now,
            failure_code="ORPHAN_REPLY_INTENT",
            detail="a previous attempt recorded an intent and no result; it is not resent",
        )
        store.settle(key, result)
        closed.append(result)
        logger.warning("plan_command.reply key=%s status=UNCERTAIN orphan", key)
    return closed


def send_once(
    *,
    store: PlanCommandStore,
    bot: CommandBotClient,
    event_id: str,
    kind: DeliveryKind,
    key_suffix: str,
    chat_id: str,
    text: str,
    run_id: str | None,
    now: datetime,
) -> CommandDeliveryResult | None:
    """Reserve, send, record. ``None`` when an earlier attempt owns this reply."""
    key = f"{event_id}.{key_suffix}"
    if store.has_result(key) or store.has_intent(key):
        return None

    attempt = secrets.token_hex(8)
    intent = CommandDeliveryIntent(
        event_id=event_id,
        kind=kind,
        attempt_id=attempt,
        created_at=now,
        chat_id=chat_id,
        run_id=run_id,
        text_sha256=sha256_bytes(text.encode("utf-8")),
        char_count=len(text),
    )
    if not store.reserve(key, intent):
        return None

    status = DeliveryStatus.DELIVERED
    message_id: int | None = None
    failure_code: str | None = None
    detail: str | None = None
    try:
        message_id = bot.send_message(chat_id=chat_id, text=text).message_id
    except (
        PublisherTransportAmbiguousError,
        PublisherResponseError,
        CommandBotTransportError,
        CommandBotResponseError,
    ) as exc:
        status, failure_code = DeliveryStatus.UNCERTAIN, exc.code
        detail = "the reply may or may not have arrived; it is not resent"
    except (PublisherError, CommandBotError) as exc:
        status, failure_code = DeliveryStatus.FAILED, exc.code
        detail = "Telegram refused the reply; nothing was delivered"

    result = CommandDeliveryResult(
        event_id=event_id,
        kind=kind,
        attempt_id=attempt,
        status=status,
        chat_id=chat_id,
        run_id=run_id,
        started_at=now,
        completed_at=now,
        message_id=message_id,
        failure_code=failure_code,
        detail=detail,
    )
    store.settle(key, result)
    logger.info("plan_command.reply key=%s kind=%s status=%s", key, kind, status)
    return result


# --------------------------------------------------------------------------
# polling
# --------------------------------------------------------------------------


@dataclass
class PollReport:
    received: int = 0
    accepted: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    ignored: list[IgnoreReason] = field(default_factory=list)
    acknowledgements: list[CommandDeliveryResult] = field(default_factory=list)


def _already_submitted(
    store: PlanCommandStore, inbox: Inbox, ledger: Ledger, event_id: str
) -> bool:
    if store.read_receipt(event_id) is not None:
        return True
    if any((inbox.directory(name) / f"{event_id}.json").is_file() for name in EVENT_DIRECTORIES):
        return True
    try:
        return ledger.read(event_id) is not None
    except LedgerError:
        # An unreadable history is not an absent one. Refusing to resubmit is
        # the direction that cannot create a second Run.
        return True


def poll_plan_commands(
    *,
    settings: PlanCommandSettings,
    store: PlanCommandStore,
    inbox: Inbox,
    ledger: Ledger,
    bot: CommandBotClient,
    now: datetime,
) -> PollReport:
    """One lightweight ``getUpdates``, then act on what it returned.

    Raises:
        PipelineError: The poll itself failed, or the bot's state is unreadable.
            Nothing was submitted and the offset did not move.
    """
    if not settings.enabled or settings.bot_id is None:
        raise ValueError("the /plan command is not enabled")

    close_orphans(store, now=now)
    offset = store.read_offset()
    updates = bot.get_updates(offset=offset)
    if not all(isinstance(update.get("update_id"), int) for update in updates):
        raise CommandBotResponseError("an update carries no id", method="getUpdates")

    report = PollReport(received=len(updates))
    for update in sorted(updates, key=lambda entry: int(entry["update_id"])):
        update_id = int(update["update_id"])
        if offset is not None and update_id < offset:
            continue

        decision = classify_update(update, settings=settings)
        if not isinstance(decision, PlanRequest):
            reason, chat_id = decision
            store.record_ignored(
                IgnoredUpdate(
                    update_id=update_id,
                    reason=reason,
                    chat_ref=None if chat_id is None else chat_ref(chat_id),
                    recorded_at=now,
                )
            )
            store.write_offset(update_id + 1)
            report.ignored.append(reason)
            logger.info("plan_command.ignored update=%d reason=%s", update_id, reason)
            continue

        event_id = plan_event_id(
            bot_id=settings.bot_id, chat_id=decision.chat_id, message_id=decision.message_id
        )
        if _already_submitted(store, inbox, ledger, event_id):
            report.duplicates.append(event_id)
        else:
            try:
                inbox.submit(
                    plan_event(decision, event_id=event_id).model_dump(mode="json"),
                    event_id=event_id,
                )
            except InboxPayloadError:
                report.duplicates.append(event_id)
            else:
                report.accepted.append(event_id)
                logger.info("plan_command.accepted update=%d event=%s", update_id, event_id)

        # Durable submission first, then the receipt, then the offset.
        store.write_receipt(
            PlanReceipt(
                event_id=event_id,
                update_id=update_id,
                bot_id=settings.bot_id,
                chat_id=decision.chat_id,
                message_id=decision.message_id,
                message_date=decision.message_date,
                accepted_at=now,
            )
        )
        store.write_offset(update_id + 1)

        acknowledgement = send_once(
            store=store,
            bot=bot,
            event_id=event_id,
            kind=DeliveryKind.ACKNOWLEDGEMENT,
            key_suffix=ACK_KEY,
            chat_id=decision.chat_id,
            text=ACK_TEXT,
            run_id=None,
            now=now,
        )
        if acknowledgement is not None:
            report.acknowledgements.append(acknowledgement)

    return report


__all__ = [
    "ACK_KEY",
    "ACK_TEXT",
    "EVENT_DIRECTORIES",
    "PLAN_COMMAND_FILENAME",
    "RESPONSE_KEY",
    "STATE_DIRNAME",
    "PlanCommandRead",
    "PlanCommandStateError",
    "PlanCommandStore",
    "PlanRequest",
    "PollReport",
    "chat_ref",
    "classify_update",
    "close_orphans",
    "parse_plan_command",
    "plan_event",
    "plan_event_id",
    "poll_plan_commands",
    "send_once",
]
