"""Telegram review delivery: showing an approved article to a human.

The gap this closes is small and real. A Run reaches ``READY_TO_PUBLISH`` and
then waits, and the only way to read it was to open a file on the machine that
produced it. So the approved article is delivered to the operator's own chat.

The thing these tests exist to defend is that **review delivery is not
publishing**. They share a transport and nothing else: different destination,
different artifacts, different trigger, and only one of them changes a Run's
status. If that distinction ever blurs, a Run a human merely *read* would look
like one that had been posted to an audience - and the publisher's one-attempt
budget would already be spent.

The second hazard is the scheduler. It fires every minute, so a delivery that is
not durably recorded becomes sixty duplicate messages an hour. The idempotency
model is copied from the publisher deliberately: an intent committed before the
first request, and an outcome that cannot be determined is never retried.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from conftest import CLEAN_ARTICLE, make_published_ready_run

from goldpipeline.adapters.publisher_client import SendOutcome, SendRequest
from goldpipeline.config import ReviewDeliverySettings
from goldpipeline.domain.errors import (
    PublisherAuthenticationError,
    PublisherConfigurationError,
    PublisherPermissionError,
    PublisherRateLimitError,
    PublisherRejectedError,
    PublisherTransportAmbiguousError,
)
from goldpipeline.schemas.common import utc_now
from goldpipeline.schemas.manifest import RunStatus
from goldpipeline.schemas.review_delivery import (
    INTENT_FILENAME,
    RESULT_FILENAME,
    ReviewDeliveryStatus,
)
from goldpipeline.services.review_delivery import HEADER, _compose, deliver_review
from goldpipeline.storage.run_store import RunStore

REVIEW_CHAT = "7387726751"
PUBLISH_CHAT = "@pcfxsn"


class RecordingClient:
    """A transport that records what it was asked to send, and never sends."""

    provider = "telegram"

    def __init__(self, *, raises: Exception | None = None, raise_times: int = 0) -> None:
        self.sent: list[SendRequest] = []
        self.raises = raises
        self.remaining = raise_times if raises is not None else 0
        self._next_id = 5000

    def send(self, request: SendRequest) -> SendOutcome:
        if self.raises is not None and (self.remaining > 0 or self.remaining == -1):
            if self.remaining > 0:
                self.remaining -= 1
            raise self.raises
        self.sent.append(request)
        self._next_id += 1
        return SendOutcome(message_id=self._next_id, chat_id=request.target_chat)

    @property
    def calls(self) -> int:
        return len(self.sent)


def always_raises(error: Exception) -> RecordingClient:
    client = RecordingClient(raises=error)
    client.remaining = -1
    return client


def ready_run(runs_dir: Any, tmp_path: Path, **kwargs: Any) -> Any:
    """A Run sitting at READY_TO_PUBLISH with an APPROVED gate decision."""
    return make_published_ready_run(runs_dir, tmp_path, **kwargs)


def run_dir_of(runs_dir: Any, run_id: str) -> Path:
    return Path(runs_dir) / run_id


def deliver(runs_dir: Any, run_id: str, client: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("target_chat", REVIEW_CHAT)
    kwargs.setdefault("sleep", lambda _seconds: None)
    return deliver_review(run_id=run_id, store=RunStore(runs_dir), client=client, **kwargs)


# --- eligibility ----------------------------------------------------------


def test_an_approved_ready_run_is_delivered(runs_dir: Any, tmp_path: Path) -> None:
    """Requirement 12.1."""
    run = ready_run(runs_dir, tmp_path)
    client = RecordingClient()

    outcome = deliver(runs_dir, run.run_id, client)

    assert outcome.status is ReviewDeliveryStatus.DELIVERED
    assert outcome.delivered
    assert client.calls == 1
    assert client.sent[0].target_chat == REVIEW_CHAT


@pytest.mark.parametrize(
    "status",
    [RunStatus.DRAFTED, RunStatus.REVIEWED, RunStatus.FINALIZED, RunStatus.PUBLISH_BLOCKED],
    ids=lambda s: str(s),
)
def test_a_run_that_is_not_ready_is_never_delivered(
    status: RunStatus, runs_dir: Any, tmp_path: Path
) -> None:
    """Requirements 12.2 and 12.3.

    Only the final approved article is shown. An article the gate declined, or
    one still mid-pipeline, is not something to put in front of a human as
    though it were finished.
    """
    run = ready_run(runs_dir, tmp_path)
    store = RunStore(runs_dir)
    directory = store.open(run.run_id)
    manifest = directory.load_manifest()
    manifest.status = status
    directory.save_manifest(manifest)

    client = RecordingClient()
    outcome = deliver(runs_dir, run.run_id, client)

    assert outcome.status is ReviewDeliveryStatus.SKIPPED
    assert client.calls == 0


def test_a_run_older_than_the_window_is_not_delivered(runs_dir: Any, tmp_path: Path) -> None:
    """Requirement 12.11 - the backlog guard.

    Switching review delivery on must not post every finished Run the machine
    happens to be holding. Expressed as an age limit rather than an activation
    marker: it needs no extra state, and it is the right rule anyway - an
    article about this morning's candles is not worth reading tonight.
    """
    run = ready_run(runs_dir, tmp_path)
    client = RecordingClient()

    outcome = deliver(
        runs_dir,
        run.run_id,
        client,
        now=utc_now() + timedelta(hours=3),
        max_run_age_minutes=60,
    )

    assert outcome.status is ReviewDeliveryStatus.SKIPPED
    assert "older than" in (outcome.reason or "")
    assert client.calls == 0


def test_a_fresh_run_inside_the_window_is_delivered(runs_dir: Any, tmp_path: Path) -> None:
    """Requirement 12.12 - the other half of the same rule."""
    run = ready_run(runs_dir, tmp_path)
    client = RecordingClient()

    outcome = deliver(
        runs_dir,
        run.run_id,
        client,
        now=utc_now() + timedelta(minutes=5),
        max_run_age_minutes=60,
    )

    assert outcome.status is ReviewDeliveryStatus.DELIVERED


# --- the Run is not published ---------------------------------------------


def test_delivery_leaves_the_run_ready_to_publish(runs_dir: Any, tmp_path: Path) -> None:
    """Requirements 12.4, 12.13 and 12.14 - the whole point of the feature.

    A human has read the article. A human has not yet decided anything.
    """
    run = ready_run(runs_dir, tmp_path)
    deliver(runs_dir, run.run_id, RecordingClient())

    directory = RunStore(runs_dir).open(run.run_id)
    assert directory.load_manifest().status is RunStatus.READY_TO_PUBLISH

    run_dir = (
        Path(directory.path) if hasattr(directory, "path") else run_dir_of(runs_dir, run.run_id)
    )
    assert not (run_dir / "publish_intent.json").exists()
    assert not (run_dir / "publish_result.json").exists()
    assert (run_dir / RESULT_FILENAME).exists()


def test_the_manifest_records_the_delivery_without_publish_semantics(
    runs_dir: Any, tmp_path: Path
) -> None:
    """Requirement 7: safe events, untouched publish meaning."""
    run = ready_run(runs_dir, tmp_path)
    deliver(runs_dir, run.run_id, RecordingClient())

    manifest = json.loads(
        (run_dir_of(runs_dir, run.run_id) / "manifest.json").read_text(encoding="utf-8")
    )
    stages = [event["stage"] for event in manifest["events"]]

    assert "review_delivery.intent" in stages
    assert "review_delivery.sent" in stages
    assert not any(stage.startswith("publish.") for stage in stages)
    assert manifest["status"] == RunStatus.READY_TO_PUBLISH.value


# --- idempotency ----------------------------------------------------------


def test_the_same_run_is_never_delivered_twice(runs_dir: Any, tmp_path: Path) -> None:
    """Requirement 12.6."""
    run = ready_run(runs_dir, tmp_path)
    client = RecordingClient()

    first = deliver(runs_dir, run.run_id, client)
    second = deliver(runs_dir, run.run_id, client)

    assert first.status is ReviewDeliveryStatus.DELIVERED
    assert second.status is ReviewDeliveryStatus.SKIPPED
    assert second.reason == "already delivered"
    assert client.calls == 1


def test_repeated_scheduler_ticks_do_not_resend(runs_dir: Any, tmp_path: Path) -> None:
    """Requirement 12.7, and the reason idempotency is durable rather than in memory.

    Sixty ticks an hour against one approved Run is the realistic load, so the
    guard is exercised at that shape rather than once.
    """
    run = ready_run(runs_dir, tmp_path)
    client = RecordingClient()

    for _ in range(30):
        deliver(runs_dir, run.run_id, client)

    assert client.calls == 1


def test_idempotency_survives_a_restart(runs_dir: Any, tmp_path: Path) -> None:
    """Requirement 12.25.

    A fresh ``RunStore`` holds no memory of the first delivery; the artifact on
    disk is what stops the second.
    """
    run = ready_run(runs_dir, tmp_path)
    deliver(runs_dir, run.run_id, RecordingClient())

    fresh_client = RecordingClient()
    outcome = deliver_review(
        run_id=run.run_id,
        store=RunStore(runs_dir),
        client=fresh_client,
        target_chat=REVIEW_CHAT,
        sleep=lambda _s: None,
    )

    assert outcome.status is ReviewDeliveryStatus.SKIPPED
    assert fresh_client.calls == 0


def test_a_successful_delivery_records_a_durable_result(runs_dir: Any, tmp_path: Path) -> None:
    """Requirement 12.5."""
    run = ready_run(runs_dir, tmp_path)
    deliver(runs_dir, run.run_id, RecordingClient())

    stored = json.loads(
        (run_dir_of(runs_dir, run.run_id) / RESULT_FILENAME).read_text(encoding="utf-8")
    )

    assert stored["status"] == ReviewDeliveryStatus.DELIVERED.value
    assert stored["target_chat"] == REVIEW_CHAT
    assert stored["confirmed_count"] == stored["chunk_count"] == 1
    assert stored["messages"][0]["message_id"] > 0


# --- ambiguous failures ---------------------------------------------------


def test_an_ambiguous_transport_failure_becomes_uncertain(runs_dir: Any, tmp_path: Path) -> None:
    """Requirement 12.8 - the dangerous case.

    The request may have arrived. Resending would duplicate it, and a reader
    cannot un-see a duplicate, so the attempt is closed and a human decides.
    """
    run = ready_run(runs_dir, tmp_path)
    client = always_raises(PublisherTransportAmbiguousError("timeout"))

    outcome = deliver(runs_dir, run.run_id, client)

    assert outcome.status is ReviewDeliveryStatus.UNCERTAIN
    assert outcome.result is not None
    assert outcome.result.failure is not None
    assert outcome.result.failure.safe_code == "PUBLISHER_TRANSPORT_AMBIGUOUS"
    assert outcome.result.failure.category.value == "TRANSPORT_AMBIGUOUS"


def test_an_uncertain_delivery_is_never_retried(runs_dir: Any, tmp_path: Path) -> None:
    """The consequence that matters: the next tick sends nothing."""
    run = ready_run(runs_dir, tmp_path)
    first = always_raises(PublisherTransportAmbiguousError("timeout"))
    deliver(runs_dir, run.run_id, first)

    second = RecordingClient()
    for _ in range(5):
        outcome = deliver(runs_dir, run.run_id, second)

    assert outcome.status is ReviewDeliveryStatus.SKIPPED
    assert second.calls == 0


def test_an_orphan_intent_is_closed_as_uncertain(runs_dir: Any, tmp_path: Path) -> None:
    """The crash window: an intent written, the process gone before the result."""
    run = ready_run(runs_dir, tmp_path)
    client = always_raises(PublisherTransportAmbiguousError("boom"))
    deliver(runs_dir, run.run_id, client)

    run_dir = run_dir_of(runs_dir, run.run_id)
    assert (run_dir / INTENT_FILENAME).exists()
    assert (run_dir / RESULT_FILENAME).exists()

    stored = json.loads((run_dir / RESULT_FILENAME).read_text(encoding="utf-8"))
    assert stored["status"] == ReviewDeliveryStatus.UNCERTAIN.value


def test_flood_control_is_retried_then_bounded(runs_dir: Any, tmp_path: Path) -> None:
    """Requirement 12.9.

    A 429 is Telegram stating it did *not* accept the request - the one
    condition where retrying cannot duplicate anything. Bounded anyway.
    """
    run = ready_run(runs_dir, tmp_path)
    client = RecordingClient(raises=PublisherRateLimitError("flood", retry_after=1), raise_times=1)
    waits: list[float] = []

    outcome = deliver(runs_dir, run.run_id, client, sleep=waits.append)

    assert outcome.status is ReviewDeliveryStatus.DELIVERED
    assert waits == [1.0], "waited exactly the delay Telegram asked for"
    assert client.calls == 1


def test_persistent_flood_control_gives_up_without_sending(runs_dir: Any, tmp_path: Path) -> None:
    run = ready_run(runs_dir, tmp_path)
    client = always_raises(PublisherRateLimitError("flood", retry_after=1))

    outcome = deliver(runs_dir, run.run_id, client, sleep=lambda _s: None)

    assert outcome.status is ReviewDeliveryStatus.FAILED
    assert outcome.result is not None
    assert outcome.result.failure is not None
    assert outcome.result.failure.safe_code == "RATE_LIMIT_RETRIES_EXHAUSTED"


def test_an_explicit_rejection_is_a_clean_failure(runs_dir: Any, tmp_path: Path) -> None:
    """Telegram said no. Nothing arrived, and that is certain."""
    run = ready_run(runs_dir, tmp_path)
    client = always_raises(PublisherRejectedError("chat not found"))

    outcome = deliver(runs_dir, run.run_id, client)

    assert outcome.status is ReviewDeliveryStatus.FAILED
    assert outcome.result is not None
    assert outcome.result.confirmed_count == 0


# --- destination ----------------------------------------------------------


def test_review_delivery_never_falls_back_to_the_publish_target() -> None:
    """Requirements 12.10 and 11 - fail closed on a missing destination.

    The two destinations are different audiences and one of them is public, so
    "no review chat configured" must never silently become "use the channel".
    """
    with pytest.raises(PublisherConfigurationError) as caught:
        ReviewDeliverySettings.from_env({"GOLDPIPELINE_TELEGRAM_REVIEW_ENABLED": "true"})

    assert caught.value.details["setting"] == "GOLDPIPELINE_TELEGRAM_REVIEW_CHAT_ID"


def test_the_configured_review_chat_is_the_one_used(runs_dir: Any, tmp_path: Path) -> None:
    """The destination comes from configuration and nowhere else."""
    run = ready_run(runs_dir, tmp_path)
    client = RecordingClient()

    deliver(runs_dir, run.run_id, client, target_chat=REVIEW_CHAT)

    assert every_target(client) == {REVIEW_CHAT}
    assert PUBLISH_CHAT not in every_target(client)


def every_target(client: RecordingClient) -> set[str]:
    return {request.target_chat for request in client.sent}


def test_review_settings_are_independent_of_auto_publish() -> None:
    """Requirement 12.16 and section 2.

    Enabling one must never imply the other. Asserted at the settings boundary,
    because that is where a future refactor would be tempted to merge them.
    """
    from goldpipeline.config import AutomationSettings

    env = {
        "GOLDPIPELINE_TELEGRAM_REVIEW_ENABLED": "true",
        "GOLDPIPELINE_TELEGRAM_REVIEW_CHAT_ID": REVIEW_CHAT,
        "GOLDPIPELINE_AUTOPUBLISH_ENABLED": "false",
    }
    review = ReviewDeliverySettings.from_env(env)
    automation = AutomationSettings.from_env(env)

    assert review.enabled is True
    assert automation.auto_publish_enabled is False


# --- message content ------------------------------------------------------


def test_the_message_carries_what_a_reviewer_needs(runs_dir: Any, tmp_path: Path) -> None:
    """Requirements 12.17, 12.18 and 12.19."""
    run = ready_run(runs_dir, tmp_path)
    client = RecordingClient()

    deliver(runs_dir, run.run_id, client)
    text = client.sent[0].text
    article = (run_dir_of(runs_dir, run.run_id) / "claude_final.md").read_text(encoding="utf-8")

    assert text.startswith(HEADER)
    assert run.run_id in text
    assert "Review:" in text and "Score:" in text
    assert "Gate: APPROVED" in text
    assert article.strip() in text, "the full final article, verbatim"
    assert "READY_TO_PUBLISH" in text
    assert "Chưa được publish tự động." in text


def test_a_long_article_is_chunked_and_reassembles(runs_dir: Any, tmp_path: Path) -> None:
    """Requirement 12.20 - the project's existing exact chunking, reused."""
    long_article = "\n\n".join(f"Đoạn {index}. {CLEAN_ARTICLE}" for index in range(12))
    run = ready_run(runs_dir, tmp_path, article=long_article)
    client = RecordingClient()

    outcome = deliver(runs_dir, run.run_id, client, chunk_limit=600)

    assert outcome.status is ReviewDeliveryStatus.DELIVERED
    assert client.calls > 1, "the message needed splitting"

    # Reassembly is checked against the message the service actually composed,
    # not against the chunks themselves - joining a list to itself would pass
    # even if chunking dropped half the article.
    expected = _compose(
        run_id=run.run_id,
        run=RunStore(runs_dir).open(run.run_id),
        article=long_article,
    )
    assert "".join(request.text for request in client.sent) == expected
    assert long_article.strip() in expected


def test_no_parse_mode_is_requested(runs_dir: Any, tmp_path: Path) -> None:
    """Requirement 12.21.

    ``SendRequest`` carries no parse mode at all, so article content can never
    be interpreted as markup - which is what would let a Vietnamese article
    containing ``*`` or ``_`` fail to send, or render wrongly.
    """
    run = ready_run(runs_dir, tmp_path)
    client = RecordingClient()

    deliver(runs_dir, run.run_id, client)

    assert not hasattr(client.sent[0], "parse_mode")
    assert set(vars(client.sent[0])) == {"target_chat", "text", "chunk_index"}


# --- costs nothing it should not ------------------------------------------


def test_delivery_makes_no_ai_or_market_call(
    runs_dir: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirements 12.22 and 12.23.

    Delivery reads finished artifacts. There is nothing left to generate, and a
    Telegram problem must never cost an Anthropic call.
    """
    import socket

    run = ready_run(runs_dir, tmp_path)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("review delivery must open no socket of its own")

    monkeypatch.setattr(socket.socket, "connect", refuse)

    outcome = deliver(runs_dir, run.run_id, RecordingClient())

    assert outcome.status is ReviewDeliveryStatus.DELIVERED


# --- the scheduled worker -------------------------------------------------


def worker_with_review(
    inbox: Any,
    runs_dir: Path,
    automation_dir: Path,
    client: RecordingClient,
    *,
    enabled: bool = True,
    max_run_age_minutes: int = 60,
) -> Any:
    """A worker context whose review transport is the recording stand-in."""
    from conftest import make_worker_context

    context = make_worker_context(inbox, runs_dir, automation_dir, enabled=True)
    return replace_review(
        context,
        ReviewDeliverySettings(
            enabled=enabled,
            chat_id=REVIEW_CHAT,
            max_run_age_minutes=max_run_age_minutes,
        ),
        lambda: (client, REVIEW_CHAT),
    )


def replace_review(context: Any, settings: Any, factory: Any) -> Any:
    from dataclasses import replace as dc_replace

    return dc_replace(context, review_delivery=settings, review_client=factory)


def test_a_scheduled_tick_delivers_an_approved_run(
    inbox: Any, runs_dir: Path, automation_dir: Path, tmp_path: Path
) -> None:
    """The wiring: the worker finds the approved Run and shows it, once."""
    from goldpipeline.services.automation import run_tick

    run = ready_run(runs_dir, tmp_path)
    client = RecordingClient()

    result = run_tick(worker_with_review(inbox, runs_dir, automation_dir, client))

    assert client.calls == 1
    assert len(result.review_deliveries) == 1
    assert result.review_deliveries[0].identifier == run.run_id
    assert result.review_deliveries[0].code == ReviewDeliveryStatus.DELIVERED.value
    assert RunStore(runs_dir).open(run.run_id).load_manifest().status is (
        RunStatus.READY_TO_PUBLISH
    )


def test_repeated_ticks_deliver_once(
    inbox: Any, runs_dir: Path, automation_dir: Path, tmp_path: Path
) -> None:
    """Requirement 12.7 at the level that actually runs every minute."""
    from goldpipeline.services.automation import run_tick

    ready_run(runs_dir, tmp_path)
    client = RecordingClient()
    context = worker_with_review(inbox, runs_dir, automation_dir, client)

    for _ in range(10):
        run_tick(context)

    assert client.calls == 1


def test_a_tick_with_review_disabled_sends_nothing(
    inbox: Any, runs_dir: Path, automation_dir: Path, tmp_path: Path
) -> None:
    """Off by default, and off means off."""
    from goldpipeline.services.automation import run_tick

    ready_run(runs_dir, tmp_path)
    client = RecordingClient()

    result = run_tick(worker_with_review(inbox, runs_dir, automation_dir, client, enabled=False))

    assert client.calls == 0
    assert result.review_deliveries == []


def test_an_idle_tick_makes_no_telegram_call(
    inbox: Any, runs_dir: Path, automation_dir: Path
) -> None:
    """Requirement 12.24 - no work, no cost.

    The scheduler fires 1,440 times a day; only an approved Run may cost
    anything.
    """
    from goldpipeline.services.automation import run_tick

    client = RecordingClient()
    result = run_tick(worker_with_review(inbox, runs_dir, automation_dir, client))

    assert client.calls == 0
    assert result.review_deliveries == []
    assert not result.did_work


def test_the_tick_never_publishes(
    inbox: Any, runs_dir: Path, automation_dir: Path, tmp_path: Path
) -> None:
    """Requirements 12.15 and 12.26.

    Review delivery runs with auto-publish off, which is the only mode this
    machine uses. The publisher must remain untouched.
    """
    from goldpipeline.services.automation import run_tick

    run = ready_run(runs_dir, tmp_path)
    client = RecordingClient()
    context = worker_with_review(inbox, runs_dir, automation_dir, client)

    assert context.settings.auto_publish_enabled is False
    result = run_tick(context)

    assert result.auto_publish_enabled is False
    assert result.mode == "READY_FOR_PUBLISH"
    run_dir = run_dir_of(runs_dir, run.run_id)
    assert not (run_dir / "publish_intent.json").exists()
    assert not (run_dir / "publish_result.json").exists()


# --- regressions found by the first live send -----------------------------


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (PublisherPermissionError("forbidden"), "PUBLISHER_PERMISSION_ERROR"),
        (PublisherAuthenticationError("bad token"), "PUBLISHER_AUTHENTICATION_ERROR"),
        (PublisherRejectedError("chat not found"), "PUBLISHER_REJECTED"),
    ],
    ids=["permission", "authentication", "rejected"],
)
def test_every_refusal_still_records_a_result(
    error: Exception, expected_code: str, runs_dir: Any, tmp_path: Path
) -> None:
    """The bug the first real send found.

    ``_send`` originally caught three exception types and Telegram raised a
    fourth: a 403, because a bot may not open a conversation with a user who has
    never messaged it. The exception escaped, no result was written, and the
    orphaned intent then made the Run permanently ineligible. Every refusal must
    end in a durable result.
    """
    run = ready_run(runs_dir, tmp_path)

    outcome = deliver(runs_dir, run.run_id, always_raises(error))

    assert outcome.status is ReviewDeliveryStatus.FAILED
    assert (run_dir_of(runs_dir, run.run_id) / RESULT_FILENAME).exists()
    assert outcome.result is not None
    assert outcome.result.failure is not None
    assert outcome.result.failure.safe_code == expected_code
    assert outcome.result.confirmed_count == 0


def test_a_refused_delivery_is_not_retried_next_tick(runs_dir: Any, tmp_path: Path) -> None:
    """A recorded refusal closes the Run for delivery, like any other outcome."""
    run = ready_run(runs_dir, tmp_path)
    deliver(runs_dir, run.run_id, always_raises(PublisherPermissionError("forbidden")))

    second = RecordingClient()
    outcome = deliver(runs_dir, run.run_id, second)

    assert outcome.status is ReviewDeliveryStatus.SKIPPED
    assert second.calls == 0


def test_the_worker_closes_an_orphaned_intent(
    inbox: Any, runs_dir: Path, automation_dir: Path, tmp_path: Path
) -> None:
    """The second half of the same production failure.

    With an intent and no result the Run is ineligible, so a worker that skipped
    every ineligible Run could never close it - leaving it stuck forever after
    one bad attempt. The orphan is the one ineligibility the worker must act on.
    """
    from goldpipeline.services.automation import run_tick

    run = ready_run(runs_dir, tmp_path)
    run_dir = run_dir_of(runs_dir, run.run_id)

    # Simulate the crash window: an intent committed, no result.
    deliver(runs_dir, run.run_id, always_raises(PublisherTransportAmbiguousError("boom")))
    (run_dir / RESULT_FILENAME).unlink()
    assert (run_dir / INTENT_FILENAME).exists()

    client = RecordingClient()
    result = run_tick(worker_with_review(inbox, runs_dir, automation_dir, client))

    assert client.calls == 0, "an orphan is closed, never resent"
    assert (run_dir / RESULT_FILENAME).exists()
    stored = json.loads((run_dir / RESULT_FILENAME).read_text(encoding="utf-8"))
    assert stored["status"] == ReviewDeliveryStatus.UNCERTAIN.value
    assert len(result.review_deliveries) == 1
