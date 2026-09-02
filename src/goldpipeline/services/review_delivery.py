"""Sending an approved article to a human, without publishing it.

The gap this closes: a Run reaches ``READY_TO_PUBLISH`` and then waits, and the
only way to read it is to open a file on the machine that produced it. So the
approved article is delivered to the operator's own Telegram chat, and the Run
stays exactly where it was.

**Nothing here changes a Run's status.** That is the whole point. The manifest
gains ``review_delivery.*`` events so an auditor can see what was shown to whom,
and the status remains ``READY_TO_PUBLISH`` because a human has still not
decided anything.

The order below is the safety argument, and it is the publisher's argument with
different filenames:

1.  refuse unless the Run is ``READY_TO_PUBLISH`` and the gate said ``APPROVED``;
2.  refuse if a result already exists - one delivery per Run;
3.  refuse if an intent exists without a result, and record the uncertainty;
4.  refuse if the Run is older than the configured window;
5.  read ``claude_final.md`` once and plan the chunks;
6.  **commit the intent**;
7.  only now, send.

Steps 1-5 touch no network and write nothing. Step 6 is the hinge.

**Why an unknown outcome is never retried.** A minute scheduler turns one
ambiguous send into sixty duplicates an hour. Telegram offers no idempotency key
this pipeline can use, so the intent is what distinguishes "never sent" from
"sent, reply lost" - and when it cannot be distinguished, the answer is to stop
and let a person look.
"""

from __future__ import annotations

import logging
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from goldpipeline.adapters.publisher_client import PublisherClient, SendRequest
from goldpipeline.domain.errors import (
    PublisherRateLimitError,
    PublisherRejectedError,
    PublisherTransportAmbiguousError,
)
from goldpipeline.schemas.common import utc_now
from goldpipeline.schemas.manifest import RunManifest, RunStatus
from goldpipeline.schemas.publish import Decision, PublishDecision
from goldpipeline.schemas.publisher import (
    ChunkPlan,
    DeliveredMessage,
    FailureCategory,
    PublishFailure,
)
from goldpipeline.schemas.review import ReviewResult
from goldpipeline.schemas.review_delivery import (
    INTENT_FILENAME,
    RESULT_FILENAME,
    ReviewDeliveryIntent,
    ReviewDeliveryResult,
    ReviewDeliveryStatus,
)
from goldpipeline.services.chunking import SAFE_CHUNK_LIMIT, plan_chunks, utf16_length, verify_plan
from goldpipeline.storage.atomic import sha256_bytes
from goldpipeline.storage.run_store import PreparedArtifact, RunDirectory, RunStore

logger = logging.getLogger(__name__)

FINAL_ARTICLE_FILENAME = "claude_final.md"
DECISION_FILENAME = "publish_decision.json"
REVIEW_FILENAME = "gpt_review.json"

CHUNK_PACING_SECONDS = 1.0
MAX_RATE_LIMIT_RETRIES = 2
"""Flood control is the one condition safe to retry: Telegram is stating that it
did *not* accept the request. Bounded anyway - a review copy is not worth
waiting on forever."""

HEADER = "🟡 BÀI CHỜ DUYỆT"


@dataclass(frozen=True)
class ReviewDeliveryOutcome:
    """What one call did, including deciding not to act."""

    run_id: str
    status: ReviewDeliveryStatus
    result: ReviewDeliveryResult | None = None
    reason: str | None = None

    @property
    def delivered(self) -> bool:
        return self.status is ReviewDeliveryStatus.DELIVERED


def is_eligible(
    run: RunDirectory,
    manifest: RunManifest,
    *,
    now: datetime,
    max_run_age_minutes: int,
) -> str | None:
    """Why this Run should not be delivered, or ``None`` if it should.

    Read-only and cheap: the worker calls it for every Run on every tick, so it
    opens no artifact it does not need and never touches the network.
    """
    if manifest.status is not RunStatus.READY_TO_PUBLISH:
        return f"status is {manifest.status}, not READY_TO_PUBLISH"
    if run.has_artifact(RESULT_FILENAME):
        return "already delivered"
    if run.has_artifact(INTENT_FILENAME):
        return "a previous delivery outcome is unresolved"
    if not run.has_artifact(FINAL_ARTICLE_FILENAME):
        return "no final article"
    if not run.has_artifact(DECISION_FILENAME):
        return "no publish decision"

    age = now - manifest.created_at
    if age > timedelta(minutes=max_run_age_minutes):
        # The backlog guard. Switching review delivery on must not post every
        # finished Run the machine happens to be holding.
        return f"older than {max_run_age_minutes} minutes"
    return None


def deliver_review(
    *,
    run_id: str,
    store: RunStore,
    client: PublisherClient,
    target_chat: str,
    now: datetime | None = None,
    max_run_age_minutes: int = 60,
    sleep: Callable[[float], None] = time.sleep,
    chunk_limit: int = SAFE_CHUNK_LIMIT,
    attempt_id: str | None = None,
) -> ReviewDeliveryOutcome:
    """Show one approved Run to a human, at most once.

    Args:
        run_id: The Run to deliver. Must be ``READY_TO_PUBLISH`` and approved.
        store: Where Runs live.
        client: Any :class:`PublisherClient` - the transport is shared with the
            publisher; nothing else about publishing is.
        target_chat: The review destination, from configuration. Never the
            publish target, and never anything derived from article content.
        now: Injection point for tests.
        max_run_age_minutes: Older Runs are skipped.
        sleep: Injection point for pacing, so tests spend no real seconds.
        chunk_limit: Maximum UTF-16 units per message.
        attempt_id: Force an id; generated when omitted.

    Returns:
        A :class:`ReviewDeliveryOutcome`. Ineligibility is a normal ``SKIPPED``
        return, not an error - the worker asks about every Run every minute.
    """
    run = store.open(run_id)
    manifest = run.load_manifest()
    started = now or utc_now()

    reason = is_eligible(run, manifest, now=started, max_run_age_minutes=max_run_age_minutes)
    if reason is not None:
        if run.has_artifact(INTENT_FILENAME) and not run.has_artifact(RESULT_FILENAME):
            return _record_uncertain(run, manifest, started=started, attempt=attempt_id)
        return ReviewDeliveryOutcome(
            run_id=run_id, status=ReviewDeliveryStatus.SKIPPED, reason=reason
        )

    decision = PublishDecision.model_validate_json(
        run.read_artifact_bytes(DECISION_FILENAME).decode("utf-8")
    )
    if decision.decision is not Decision.APPROVED:
        return ReviewDeliveryOutcome(
            run_id=run_id,
            status=ReviewDeliveryStatus.SKIPPED,
            reason=f"gate decision is {decision.decision}, not APPROVED",
        )

    article_bytes = run.read_artifact_bytes(FINAL_ARTICLE_FILENAME)
    article = article_bytes.decode("utf-8")
    message = _compose(run_id=run_id, run=run, article=article)

    chunks = plan_chunks(message, chunk_limit)
    verify_plan(message, chunks, chunk_limit)

    attempt = attempt_id or secrets.token_hex(8)
    intent = ReviewDeliveryIntent(
        run_id=run_id,
        attempt_id=attempt,
        created_at=started,
        provider=client.provider,
        target_chat=target_chat,
        decision_sha256=sha256_bytes(run.read_artifact_bytes(DECISION_FILENAME)),
        final_article_sha256=sha256_bytes(article_bytes),
        chunk_count=len(chunks),
        chunks=[
            ChunkPlan(
                index=index,
                text_sha256=sha256_bytes(chunk.encode("utf-8")),
                char_count=len(chunk),
                utf16_units=utf16_length(chunk),
            )
            for index, chunk in enumerate(chunks)
        ],
    )

    # Nothing above has touched the network. Nothing below may run until the
    # intent is durable.
    intent_artifact = PreparedArtifact.from_json(INTENT_FILENAME, intent)
    run.commit_artifacts([intent_artifact], manifest)
    manifest.record_event("review_delivery.intent", "OK", f"attempt={attempt} chunks={len(chunks)}")
    run.save_manifest(manifest)
    logger.info(
        "run=%s attempt=%s stage=review_delivery.intent chunks=%d", run_id, attempt, len(chunks)
    )

    delivery = _send(
        run_id=run_id,
        attempt=attempt,
        client=client,
        target_chat=target_chat,
        chunks=chunks,
        sleep=sleep,
    )

    result = ReviewDeliveryResult(
        run_id=run_id,
        attempt_id=attempt,
        status=delivery.status,
        provider=client.provider,
        target_chat=target_chat,
        started_at=started,
        completed_at=now or utc_now(),
        final_article_sha256=intent.final_article_sha256,
        review_intent_sha256=intent_artifact.sha256,
        chunk_count=len(chunks),
        confirmed_count=len(delivery.messages),
        messages=delivery.messages,
        failure=delivery.failure,
        warnings=delivery.warnings,
    )
    return _commit(run, manifest, result)


# --------------------------------------------------------------------------
# message
# --------------------------------------------------------------------------


def _compose(*, run_id: str, run: RunDirectory, article: str) -> str:
    """Build the operator's message: what this is, then the article verbatim.

    Plain text. No parse mode is requested anywhere in this path, so a Vietnamese
    article containing ``*`` or ``_`` cannot be mangled into a formatting error -
    and article content can never become markup.
    """
    verdict = "unknown"
    score: object = "-"
    try:
        review = ReviewResult.model_validate_json(
            run.read_artifact_bytes(REVIEW_FILENAME).decode("utf-8")
        )
        verdict = str(review.status)
        score = review.score
    except Exception:  # noqa: BLE001 - a missing review must not block the copy
        logger.warning("run=%s review metadata unreadable for the review copy", run_id)

    return "\n".join(
        [
            HEADER,
            "",
            f"Run: {run_id}",
            f"Review: {verdict} | Score: {score}",
            "Gate: APPROVED",
            "",
            article.strip(),
            "",
            "---",
            "Trạng thái: READY_TO_PUBLISH",
            "Chưa được publish tự động.",
        ]
    )


# --------------------------------------------------------------------------
# sending
# --------------------------------------------------------------------------


@dataclass
class _Delivery:
    status: ReviewDeliveryStatus
    messages: list[DeliveredMessage]
    failure: PublishFailure | None = None
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []


def _send(
    *,
    run_id: str,
    attempt: str,
    client: PublisherClient,
    target_chat: str,
    chunks: list[str],
    sleep: Callable[[float], None],
) -> _Delivery:
    """Send the chunks in order, stopping at the first non-success."""
    messages: list[DeliveredMessage] = []
    warnings: list[str] = []

    for index, text in enumerate(chunks):
        if index > 0:
            sleep(CHUNK_PACING_SECONDS)

        retries = 0
        while True:
            try:
                outcome = client.send(
                    SendRequest(target_chat=target_chat, text=text, chunk_index=index)
                )
            except PublisherRateLimitError as exc:
                if retries >= MAX_RATE_LIMIT_RETRIES:
                    return _Delivery(
                        status=(
                            ReviewDeliveryStatus.PARTIAL
                            if messages
                            else ReviewDeliveryStatus.FAILED
                        ),
                        messages=messages,
                        failure=PublishFailure(
                            category=FailureCategory.RATE_LIMITED,
                            safe_code="RATE_LIMIT_RETRIES_EXHAUSTED",
                            safe_message=(
                                f"Flood control persisted after {retries} retries. "
                                "Nothing further was sent."
                            ),
                            failed_chunk_index=index,
                        ),
                        warnings=warnings,
                    )
                delay = float(exc.details.get("retry_after") or 1)
                retries += 1
                warnings.append(f"Chunk {index} was rate limited; waited {delay:g}s.")
                sleep(delay)
                continue
            except PublisherTransportAmbiguousError:
                # The dangerous one. It may have arrived; never send it again.
                logger.error(
                    "run=%s attempt=%s chunk=%d stage=review_delivery status=UNCERTAIN",
                    run_id,
                    attempt,
                    index,
                )
                return _Delivery(
                    status=ReviewDeliveryStatus.UNCERTAIN,
                    messages=messages,
                    failure=PublishFailure(
                        category=FailureCategory.TRANSPORT_AMBIGUOUS,
                        safe_code="TRANSPORT_AMBIGUOUS",
                        safe_message=(
                            "The transport did not confirm this chunk. It is not resent, "
                            "because a duplicate review copy cannot be withdrawn."
                        ),
                        failed_chunk_index=index,
                    ),
                    warnings=warnings,
                )
            except PublisherRejectedError as exc:
                return _Delivery(
                    status=(
                        ReviewDeliveryStatus.PARTIAL if messages else ReviewDeliveryStatus.FAILED
                    ),
                    messages=messages,
                    failure=PublishFailure(
                        category=FailureCategory.REJECTED,
                        safe_code=exc.code,
                        safe_message="The provider refused the request. Nothing was delivered.",
                        failed_chunk_index=index,
                    ),
                    warnings=warnings,
                )

            messages.append(
                DeliveredMessage(
                    chunk_index=index,
                    chunk_sha256=sha256_bytes(text.encode("utf-8")),
                    message_id=outcome.message_id,
                    telegram_date=outcome.telegram_date,
                    retry_count=retries,
                )
            )
            break

    return _Delivery(status=ReviewDeliveryStatus.DELIVERED, messages=messages, warnings=warnings)


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def _commit(
    run: RunDirectory, manifest: RunManifest, result: ReviewDeliveryResult
) -> ReviewDeliveryOutcome:
    """Persist the outcome. The Run's status is deliberately left alone."""
    artifact = PreparedArtifact.from_json(RESULT_FILENAME, result)
    run.commit_artifacts([artifact], manifest)
    manifest.record_event(
        "review_delivery.sent",
        str(result.status),
        f"attempt={result.attempt_id} confirmed={result.confirmed_count}/{result.chunk_count}",
    )
    run.save_manifest(manifest)
    logger.info(
        "run=%s attempt=%s stage=review_delivery status=%s confirmed=%d/%d",
        result.run_id,
        result.attempt_id,
        result.status,
        result.confirmed_count,
        result.chunk_count,
    )
    return ReviewDeliveryOutcome(run_id=result.run_id, status=result.status, result=result)


def _record_uncertain(
    run: RunDirectory, manifest: RunManifest, *, started: datetime, attempt: str | None
) -> ReviewDeliveryOutcome:
    """Close an intent that never got a result. Sends nothing.

    The crash window the intent exists for: a previous process wrote one and
    stopped, so Telegram may already hold the message.
    """
    previous: ReviewDeliveryIntent | None = None
    try:
        previous = ReviewDeliveryIntent.model_validate_json(
            run.read_artifact_bytes(INTENT_FILENAME).decode("utf-8")
        )
    except Exception:  # noqa: BLE001 - an unreadable intent is still an intent
        logger.warning("run=%s review delivery intent unreadable", run.run_id)

    result = ReviewDeliveryResult(
        run_id=run.run_id,
        attempt_id=previous.attempt_id if previous else (attempt or "unknown"),
        status=ReviewDeliveryStatus.UNCERTAIN,
        provider=previous.provider if previous else "unknown",
        target_chat=previous.target_chat if previous else "unknown",
        started_at=started,
        completed_at=started,
        final_article_sha256=previous.final_article_sha256 if previous else None,
        review_intent_sha256=sha256_bytes(run.read_artifact_bytes(INTENT_FILENAME)),
        chunk_count=previous.chunk_count if previous else 0,
        confirmed_count=0,
        failure=PublishFailure(
            category=FailureCategory.TRANSPORT_AMBIGUOUS,
            safe_code="ORPHAN_REVIEW_INTENT",
            safe_message=(
                "A previous attempt recorded an intent and no result, so it is unknown "
                "whether the review copy arrived. It is not resent."
            ),
        ),
    )
    return _commit(run, manifest, result)


__all__ = [
    "HEADER",
    "ReviewDeliveryOutcome",
    "deliver_review",
    "is_eligible",
]
