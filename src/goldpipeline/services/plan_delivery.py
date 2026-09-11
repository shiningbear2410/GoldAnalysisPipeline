"""Answering a ``/plan``: the finished plan, or one short apology, exactly once.

Round 6.7. **This is command response delivery, not review delivery and not
publishing.** It sends the approved trade plan to the chat that asked for it,
through the command bot, and records that it did so beside the request's
receipt. It never touches the Run: no artifact is written into it, no manifest
event is added, and its status stays ``READY_TO_PUBLISH``. Review delivery keeps
its own artifacts, its own bot and its own switch, and none of them is read here.

**One reply per request, whatever it says.** The reply is either the plan or a
failure notice, and both share one reservation - so a request can never receive
both, nor either twice. A reply whose outcome is unknown is recorded as
``UNCERTAIN`` and never resent.

**A failure notice waits for a real ending.** A Run on a bounded transient retry
is not a failure yet, and saying so would be premature. The notice goes out when
the event was refused or expired, when the Run was blocked or failed, when its
retries are exhausted or were never allowed, or - for a request stuck on
something no retry will fix, such as a missing credential - after a fixed wait.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from goldpipeline.adapters.telegram_command_bot import CommandBotClient
from goldpipeline.domain.errors import LedgerError, PipelineError
from goldpipeline.schemas.automation import RetryClass
from goldpipeline.schemas.ingestion import LedgerState
from goldpipeline.schemas.manifest import RunStatus
from goldpipeline.schemas.plan_command import (
    CommandDeliveryResult,
    DeliveryKind,
    DeliveryStatus,
    PlanCommandSettings,
    PlanReceipt,
)
from goldpipeline.schemas.publish import Decision, PublishDecision
from goldpipeline.services.automation_state import AutomationStore
from goldpipeline.services.inbox import FAILED, Inbox, Ledger
from goldpipeline.services.plan_command import (
    RESPONSE_KEY,
    PlanCommandStore,
    close_orphans,
    send_once,
)
from goldpipeline.storage.atomic import sha256_bytes
from goldpipeline.storage.run_store import RunDirectory, RunStore

logger = logging.getLogger(__name__)

FAILURE_TEXT = "Không tạo được kế hoạch lúc này. Hãy thử lại sau."

GIVE_UP_AFTER = timedelta(minutes=90)
"""How long a request may wait on something no retry is going to fix.

Longer than the transient retry schedule, which is exhausted within an hour, so
this never pre-empts a bounded retry that is still running.
"""

FINAL_ARTICLE_FILENAME = "claude_final.md"
DECISION_FILENAME = "publish_decision.json"
EXPIRED_DIRECTORY = "expired"

PLAN_READY_STATUSES = (RunStatus.READY_TO_PUBLISH, RunStatus.PUBLISHED)
TERMINAL_STATUSES = (RunStatus.PUBLISH_BLOCKED, RunStatus.FAILED)


@dataclass(frozen=True)
class DueReply:
    kind: DeliveryKind
    run_id: str | None
    text: str


def _failure(run_id: str | None) -> DueReply:
    return DueReply(kind=DeliveryKind.FAILURE_NOTICE, run_id=run_id, text=FAILURE_TEXT)


def _give_up(receipt: PlanReceipt, run_id: str | None, now: datetime) -> DueReply | None:
    return _failure(run_id) if now - receipt.accepted_at > GIVE_UP_AFTER else None


def approved_plan(run: RunDirectory) -> str | None:
    """The gated page, exactly as approved - or ``None`` if it cannot be proven so."""
    if not run.has_artifact(DECISION_FILENAME) or not run.has_artifact(FINAL_ARTICLE_FILENAME):
        return None
    decision = PublishDecision.model_validate_json(
        run.read_artifact_bytes(DECISION_FILENAME).decode("utf-8")
    )
    article = run.read_artifact_bytes(FINAL_ARTICLE_FILENAME)
    if decision.decision is not Decision.APPROVED:
        return None
    if decision.final_article_sha256 != sha256_bytes(article):
        return None
    return article.decode("utf-8").rstrip("\n")


def reply_due(
    receipt: PlanReceipt,
    *,
    runs: RunStore,
    inbox: Inbox,
    ledger: Ledger,
    automation: AutomationStore,
    now: datetime,
) -> DueReply | None:
    """What this request should be answered with now, or ``None`` to keep waiting."""
    try:
        entry = ledger.read(receipt.event_id)
    except LedgerError:
        return None

    if entry is None:
        refused = any(
            (inbox.directory(name) / f"{receipt.event_id}.json").is_file()
            for name in (FAILED, EXPIRED_DIRECTORY)
        )
        return _failure(None) if refused else _give_up(receipt, None, now)

    run_id = entry.run_id
    try:
        run = runs.open(run_id)
        manifest = run.load_manifest()
    except (FileNotFoundError, ValueError, PipelineError):
        if entry.state is LedgerState.ABANDONED:
            return _failure(run_id)
        return _give_up(receipt, run_id, now)

    if manifest.status in PLAN_READY_STATUSES:
        text = approved_plan(run)
        return _failure(run_id) if text is None else DueReply(DeliveryKind.PLAN, run_id, text)
    if manifest.status in TERMINAL_STATUSES:
        return _failure(run_id)

    retry = automation.read_retry(run_id)
    if retry is not None:
        if retry.exhausted or retry.retry_class in (RetryClass.PERMANENT, RetryClass.TERMINAL):
            return _failure(run_id)
        if retry.retry_class is RetryClass.TRANSIENT:
            return None
    return _give_up(receipt, run_id, now)


def deliver_plan_replies(
    *,
    settings: PlanCommandSettings,
    store: PlanCommandStore,
    runs: RunStore,
    inbox: Inbox,
    ledger: Ledger,
    automation: AutomationStore,
    bot_factory: Callable[[], CommandBotClient],
    now: datetime,
) -> list[CommandDeliveryResult]:
    """Answer every request whose answer is due. Local reads unless one is.

    The bot is built only when there is something to send, so a tick with no
    finished request makes no Telegram call from here at all.
    """
    results = close_orphans(store, now=now)
    bot: CommandBotClient | None = None

    for receipt in store.receipts():
        key = f"{receipt.event_id}.{RESPONSE_KEY}"
        if store.has_result(key) or store.has_intent(key):
            continue

        due = reply_due(
            receipt, runs=runs, inbox=inbox, ledger=ledger, automation=automation, now=now
        )
        if due is None:
            continue

        if receipt.chat_id not in settings.authorised_chat_ids:
            # Authorisation was withdrawn after the request. Recorded, not sent.
            skipped = CommandDeliveryResult(
                event_id=receipt.event_id,
                kind=due.kind,
                attempt_id="none",
                status=DeliveryStatus.SKIPPED,
                chat_id=receipt.chat_id,
                run_id=due.run_id,
                started_at=now,
                completed_at=now,
                detail="the chat is no longer authorised",
            )
            store.settle(key, skipped)
            results.append(skipped)
            continue

        if bot is None:
            bot = bot_factory()
        result = send_once(
            store=store,
            bot=bot,
            event_id=receipt.event_id,
            kind=due.kind,
            key_suffix=RESPONSE_KEY,
            chat_id=receipt.chat_id,
            text=due.text,
            run_id=due.run_id,
            now=now,
        )
        if result is not None:
            results.append(result)
    return results


__all__ = [
    "FAILURE_TEXT",
    "GIVE_UP_AFTER",
    "DueReply",
    "approved_plan",
    "deliver_plan_replies",
    "reply_due",
]
