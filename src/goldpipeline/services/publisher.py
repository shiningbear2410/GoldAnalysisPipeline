"""Publisher orchestration.

The order below is the whole safety argument:

1.  refuse unless the Run is ``READY_TO_PUBLISH``;
2.  refuse if a result already exists - one attempt per Run;
3.  refuse if an *intent* exists without a result, and record why;
4.  verify the decision is ``APPROVED`` from a gate version we support;
5.  verify the artifact digests still match what the gate approved;
6.  read ``claude_final.md`` **once**, and hold those bytes;
7.  plan the chunks and prove they reassemble into exactly that text;
8.  build the transport config - a missing token fails here, before any intent;
9.  **commit ``publish_intent.json``**;
10. only now, send.

Steps 1-8 touch no network and write nothing. Step 9 is the hinge: after it,
the Run carries durable evidence that a request may have gone out.

**Why the intent must exist first.** Telegram's ``sendMessage`` has no
idempotency key this pipeline can use. If the process dies between sending and
recording, nothing else distinguishes "never sent" from "sent, acknowledgement
lost". The intent makes that distinction: a later run sees it, knows the outcome
is unknown, and refuses to send again. A duplicate post is not recoverable -
readers have already seen it - so the pipeline prefers a Run that needs a human
over one that might post twice.

**Ambiguity is never retried.** A timeout, a reset, a 5xx or an unparseable
reply all end the attempt as ``UNCERTAIN``. Only an explicit 429 is retried,
because that is Telegram stating it did *not* accept the request.
"""

from __future__ import annotations

import logging
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from goldpipeline.adapters.publisher_client import PublisherClient, SendRequest
from goldpipeline.domain.errors import (
    ArtifactIntegrityError,
    PublisherArtifactExistsError,
    PublisherAuthenticationError,
    PublisherError,
    PublisherIntegrityError,
    PublisherNotApprovedError,
    PublisherPermissionError,
    PublisherRateLimitError,
    PublisherRejectedError,
    PublisherResponseError,
    PublisherTransportAmbiguousError,
)
from goldpipeline.schemas.common import utc_now
from goldpipeline.schemas.manifest import RunError, RunManifest, RunStatus
from goldpipeline.schemas.publish import Decision, PublishDecision
from goldpipeline.schemas.publisher import (
    SUPPORTED_GATE_VERSIONS,
    ChunkPlan,
    DeliveredMessage,
    FailureCategory,
    PublishFailure,
    PublishIntent,
    PublishResult,
    PublishStatus,
)
from goldpipeline.services.chunking import SAFE_CHUNK_LIMIT, plan_chunks, utf16_length, verify_plan
from goldpipeline.services.finalizer import FINAL_FILENAME, FINALIZER_FILENAME
from goldpipeline.services.integrity import verify_artifact
from goldpipeline.services.pipeline import CONTEXT_FILENAME
from goldpipeline.services.publish_gate import DECISION_FILENAME
from goldpipeline.storage.atomic import sha256_bytes
from goldpipeline.storage.run_store import PreparedArtifact, RunDirectory, RunStore

logger = logging.getLogger(__name__)

INTENT_FILENAME = "publish_intent.json"
RESULT_FILENAME = "publish_result.json"

PUBLISHABLE_STATUSES = (RunStatus.READY_TO_PUBLISH,)
"""Only a gate-approved Run may be published."""

MAX_RATE_LIMIT_RETRIES = 2
"""How many times one chunk may be re-sent after an explicit 429.

Bounded so a channel in a long flood-control window ends the attempt rather than
holding an open intent indefinitely.
"""

CHUNK_PACING_SECONDS = 1.1
"""Pause between confirmed chunks in the same chat.

Telegram's guidance is roughly one message per second per chat; a little over
that keeps a multi-part article from tripping flood control mid-publication.
"""

_STATUS_BY_OUTCOME = {
    PublishStatus.PUBLISHED: RunStatus.PUBLISHED,
    PublishStatus.FAILED: RunStatus.PUBLISH_FAILED,
    PublishStatus.PARTIAL: RunStatus.PARTIALLY_PUBLISHED,
    PublishStatus.UNCERTAIN: RunStatus.PUBLISH_UNCERTAIN,
}


@dataclass(frozen=True)
class PublishRunResult:
    """Outcome of a publish attempt."""

    run_id: str
    run_dir: Path
    status: RunStatus
    result: PublishResult
    result_path: Path

    @property
    def published(self) -> bool:
        """Whether every chunk was confirmed delivered."""
        return self.result.status is PublishStatus.PUBLISHED


def publish_run(
    *,
    run_id: str,
    store: RunStore,
    client: PublisherClient,
    target_chat: str,
    now: datetime | None = None,
    sleep: Callable[[float], None] = time.sleep,
    chunk_limit: int = SAFE_CHUNK_LIMIT,
    attempt_id: str | None = None,
) -> PublishRunResult:
    """Publish an approved Run, once.

    Args:
        run_id: The Run to publish. Must be ``READY_TO_PUBLISH``.
        store: Where Runs live.
        client: Any :class:`PublisherClient` - Telegram or the offline fake.
        target_chat: Destination, from configuration. Never from article content.
        now: Injection point for tests.
        sleep: Injection point for pacing and flood-control waits, so tests do
            not spend real seconds waiting.
        chunk_limit: Maximum UTF-16 units per message.
        attempt_id: Force an id; generated when omitted.

    Returns:
        A :class:`PublishRunResult`. Every delivery outcome - including
        ``FAILED``, ``PARTIAL`` and ``UNCERTAIN`` - is a normal return with a
        persisted result.

    Raises:
        PublisherNotApprovedError: The Run is not cleared to publish.
        PublisherArtifactExistsError: It was already attempted.
        PublisherIntegrityError: An artifact changed after the gate approved it.
    """
    run = store.open(run_id)
    manifest = run.load_manifest()
    started = now or utc_now()
    attempt = attempt_id or _new_attempt_id()

    _require_no_result(run)
    orphan = _check_orphan_intent(run, manifest, attempt=attempt, started=started)
    if orphan is not None:
        return orphan

    _require_publishable(run, manifest)
    approved = _load_approved(run, manifest)

    article = approved.article
    chunks = plan_chunks(article, chunk_limit)
    verify_plan(article, chunks, chunk_limit)

    intent = PublishIntent(
        run_id=run.run_id,
        attempt_id=attempt,
        created_at=started,
        provider=client.provider,
        target_chat=target_chat,
        gate_version=approved.decision.gate_version,
        decision_sha256=approved.decision_sha256,
        final_article_sha256=approved.article_sha256,
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

    # Nothing above this line has touched the network. Nothing below may run
    # until the intent is durable.
    intent_artifact = PreparedArtifact.from_json(INTENT_FILENAME, intent)
    run.commit_artifacts([intent_artifact], manifest)
    manifest.status = RunStatus.PUBLISHING
    manifest.record_event(
        "publish.intent", "OK", f"attempt={attempt} chunks={len(chunks)} target={target_chat}"
    )
    run.save_manifest(manifest)
    logger.info(
        "run=%s attempt=%s stage=publish.intent chunks=%d target=%s",
        run.run_id,
        attempt,
        len(chunks),
        target_chat,
    )

    delivery = _deliver(
        run_id=run.run_id,
        attempt=attempt,
        client=client,
        target_chat=target_chat,
        chunks=chunks,
        intent=intent,
        sleep=sleep,
    )

    result = PublishResult(
        run_id=run.run_id,
        attempt_id=attempt,
        status=delivery.status,
        provider=client.provider,
        target_chat=target_chat,
        started_at=started,
        completed_at=now or utc_now(),
        decision_sha256=approved.decision_sha256,
        final_article_sha256=approved.article_sha256,
        publish_intent_sha256=intent_artifact.sha256,
        chunk_count=len(chunks),
        confirmed_count=len(delivery.messages),
        messages=delivery.messages,
        failure=delivery.failure,
        warnings=delivery.warnings,
    )
    return _commit_result(run=run, manifest=manifest, result=result)


# --------------------------------------------------------------------------
# preconditions
# --------------------------------------------------------------------------


def _new_attempt_id() -> str:
    return secrets.token_hex(8)


def _require_no_result(run: RunDirectory) -> None:
    """One attempt per Run.

    A second attempt cannot know what the first delivered, so allowing one is
    how an article gets posted twice.
    """
    if run.has_artifact(RESULT_FILENAME):
        raise PublisherArtifactExistsError(
            f"run {run.run_id} has already been attempted; publishing is one attempt "
            "per Run, and a retry cannot know what the first attempt delivered",
            run_id=run.run_id,
            artifact=RESULT_FILENAME,
        )


def _check_orphan_intent(
    run: RunDirectory, manifest: RunManifest, *, attempt: str, started: datetime
) -> PublishRunResult | None:
    """Handle an intent with no result: record uncertainty, send nothing.

    This is the crash window the intent exists for. The previous process wrote
    an intent and never wrote a result, so Telegram may already hold the
    message. Sending again could duplicate it, and readers cannot un-see a
    duplicate - so the Run is closed as ``UNCERTAIN`` for a human to reconcile.
    """
    if not run.has_artifact(INTENT_FILENAME):
        return None

    logger.error(
        "run=%s stage=publish.orphan_intent status=UNCERTAIN - refusing to resend",
        run.run_id,
    )

    previous = _read_intent(run)
    result = PublishResult(
        run_id=run.run_id,
        attempt_id=previous.attempt_id if previous else attempt,
        status=PublishStatus.UNCERTAIN,
        provider=previous.provider if previous else "unknown",
        target_chat=previous.target_chat if previous else "unknown",
        started_at=started,
        completed_at=started,
        decision_sha256=previous.decision_sha256 if previous else None,
        final_article_sha256=previous.final_article_sha256 if previous else None,
        publish_intent_sha256=sha256_bytes(run.read_artifact_bytes(INTENT_FILENAME)),
        chunk_count=previous.chunk_count if previous else 0,
        confirmed_count=0,
        messages=[],
        failure=PublishFailure(
            category=FailureCategory.ORPHAN_PUBLISH_INTENT,
            safe_code="ORPHAN_PUBLISH_INTENT",
            safe_message=(
                "A previous attempt wrote a publish intent but never recorded a result. "
                "The provider may already hold the message, so it will not be sent "
                "again. Check the channel and reconcile by hand."
            ),
        ),
        warnings=["No request was made during this invocation."],
    )
    return _commit_result(run=run, manifest=manifest, result=result)


def _read_intent(run: RunDirectory) -> PublishIntent | None:
    """Read a previous intent, tolerating one that no longer parses."""
    try:
        return PublishIntent.model_validate_json(
            run.read_artifact_bytes(INTENT_FILENAME).decode("utf-8")
        )
    except (PydanticValidationError, UnicodeDecodeError, OSError):
        return None


def _require_publishable(run: RunDirectory, manifest: RunManifest) -> None:
    """Refuse anything the gate has not cleared."""
    if manifest.status not in PUBLISHABLE_STATUSES:
        raise PublisherNotApprovedError(
            f"run {run.run_id} is {manifest.status}; publishing needs {RunStatus.READY_TO_PUBLISH}",
            run_id=run.run_id,
            status=str(manifest.status),
        )


@dataclass(frozen=True)
class _Approved:
    """The verified inputs a publish attempt is built from."""

    decision: PublishDecision
    decision_sha256: str
    article: str
    article_sha256: str


def _load_approved(run: RunDirectory, manifest: RunManifest) -> _Approved:
    """Re-verify the approval and read the article exactly once.

    The gate's approval names specific bytes. This checks those bytes are still
    on disk, then takes a snapshot: everything afterwards - chunking, hashing,
    sending - works from the snapshot, so a file edited mid-publish cannot
    change what goes out.
    """
    try:
        decision_artifact = verify_artifact(run, manifest, DECISION_FILENAME)
        article_artifact = verify_artifact(run, manifest, FINAL_FILENAME)
        finalizer_artifact = verify_artifact(run, manifest, FINALIZER_FILENAME)
        context_artifact = verify_artifact(run, manifest, CONTEXT_FILENAME)
    except ArtifactIntegrityError as exc:
        # `exc.details` already carries `run_id`, so it is not passed again -
        # duplicating it turns a clean refusal into a TypeError.
        raise PublisherIntegrityError(
            f"an artifact changed after the gate approved it: {exc.message}",
            **{"run_id": run.run_id, **exc.details},
        ) from exc

    try:
        decision = PublishDecision.model_validate_json(decision_artifact.text)
    except (PydanticValidationError, UnicodeDecodeError) as exc:
        raise PublisherIntegrityError(
            f"{DECISION_FILENAME} does not satisfy the publish decision schema",
            run_id=run.run_id,
        ) from exc

    if decision.run_id != run.run_id:
        raise PublisherIntegrityError(
            f"{DECISION_FILENAME} belongs to a different run", run_id=run.run_id
        )

    if decision.decision is not Decision.APPROVED:
        raise PublisherNotApprovedError(
            f"the publish decision for run {run.run_id} is {decision.decision}",
            run_id=run.run_id,
            decision=str(decision.decision),
        )

    if decision.gate_version not in SUPPORTED_GATE_VERSIONS:
        raise PublisherNotApprovedError(
            f"the decision came from gate {decision.gate_version!r}, which this "
            "publisher does not support; its checks may not mean what this "
            "publisher assumes",
            run_id=run.run_id,
            gate_version=decision.gate_version,
        )

    for label, recorded, actual in (
        ("final_article_sha256", decision.final_article_sha256, article_artifact.sha256),
        (
            "finalizer_metadata_sha256",
            decision.finalizer_metadata_sha256,
            finalizer_artifact.sha256,
        ),
        ("context_sha256", decision.context_sha256, context_artifact.sha256),
    ):
        if recorded != actual:
            raise PublisherIntegrityError(
                f"the decision's {label} no longer matches the artifact on disk; "
                "the approval does not cover these bytes",
                run_id=run.run_id,
                field=label,
            )

    article = article_artifact.text
    if not article.strip():
        raise PublisherIntegrityError(f"{FINAL_FILENAME} is empty", run_id=run.run_id)

    return _Approved(
        decision=decision,
        decision_sha256=decision_artifact.sha256,
        article=article,
        article_sha256=article_artifact.sha256,
    )


# --------------------------------------------------------------------------
# delivery
# --------------------------------------------------------------------------


@dataclass
class _Delivery:
    """What the send loop established."""

    status: PublishStatus
    messages: list[DeliveredMessage]
    failure: PublishFailure | None = None
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []


def _deliver(
    *,
    run_id: str,
    attempt: str,
    client: PublisherClient,
    target_chat: str,
    chunks: list[str],
    intent: PublishIntent,
    sleep: Callable[[float], None],
) -> _Delivery:
    """Send the chunks in order, stopping at the first thing that is not success.

    Stopping matters. Once one chunk's fate is unknown or refused, continuing
    would post the rest of an article whose earlier part may be missing - a
    worse outcome than an incomplete post someone can finish by hand.
    """
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
                        status=PublishStatus.PARTIAL if messages else PublishStatus.FAILED,
                        messages=messages,
                        failure=PublishFailure(
                            category=FailureCategory.RATE_LIMITED,
                            safe_code="RATE_LIMIT_RETRIES_EXHAUSTED",
                            safe_message=(
                                f"The provider kept applying flood control after "
                                f"{retries} retries. Nothing further was sent."
                            ),
                            failed_chunk_index=index,
                        ),
                        warnings=warnings,
                    )
                delay = float(exc.details.get("retry_after") or 1)
                retries += 1
                warnings.append(
                    f"Chunk {index} was rate limited; waited {delay:g}s before retry {retries}."
                )
                logger.warning(
                    "run=%s attempt=%s chunk=%d rate limited, retry %d after %.0fs",
                    run_id,
                    attempt,
                    index,
                    retries,
                    delay,
                )
                sleep(delay)
                continue

            except (
                PublisherTransportAmbiguousError,
                PublisherResponseError,
            ) as exc:
                # The message may already be posted. Never retried.
                return _Delivery(
                    status=PublishStatus.UNCERTAIN,
                    messages=messages,
                    failure=PublishFailure(
                        category=(
                            FailureCategory.RESPONSE_INVALID
                            if isinstance(exc, PublisherResponseError)
                            else FailureCategory.TRANSPORT_AMBIGUOUS
                        ),
                        safe_code=exc.code,
                        safe_message=exc.message,
                        failed_chunk_index=index,
                    ),
                    warnings=warnings,
                )

            except (
                PublisherAuthenticationError,
                PublisherPermissionError,
                PublisherRejectedError,
            ) as exc:
                # An explicit refusal: nothing was delivered for this chunk.
                return _Delivery(
                    status=PublishStatus.PARTIAL if messages else PublishStatus.FAILED,
                    messages=messages,
                    failure=PublishFailure(
                        category=_category_for(exc),
                        safe_code=exc.code,
                        safe_message=exc.message,
                        failed_chunk_index=index,
                    ),
                    warnings=warnings,
                )

            messages.append(
                DeliveredMessage(
                    chunk_index=index,
                    chunk_sha256=intent.chunks[index].text_sha256,
                    message_id=outcome.message_id,
                    telegram_date=outcome.telegram_date,
                    retry_count=retries,
                )
            )
            logger.info(
                "run=%s attempt=%s stage=publish.sent chunk=%d/%d message_id=%s",
                run_id,
                attempt,
                index + 1,
                len(chunks),
                outcome.message_id,
            )
            break

    return _Delivery(status=PublishStatus.PUBLISHED, messages=messages, warnings=warnings)


def _category_for(exc: PublisherError) -> FailureCategory:
    if isinstance(exc, PublisherAuthenticationError):
        return FailureCategory.AUTHENTICATION
    if isinstance(exc, PublisherPermissionError):
        return FailureCategory.PERMISSION
    return FailureCategory.REJECTED


# --------------------------------------------------------------------------
# commit
# --------------------------------------------------------------------------


def _commit_result(
    *, run: RunDirectory, manifest: RunManifest, result: PublishResult
) -> PublishRunResult:
    """Write the result, then move the Run to match it.

    The artifact lands first, and the manifest only afterwards. A manifest
    saying ``PUBLISHED`` with no result beside it would be a claim nobody can
    check - and, because the next run refuses an already-attempted Run, it would
    also be unrecoverable.
    """
    if result.status is PublishStatus.PUBLISHED and not result.fully_delivered:
        raise PublisherIntegrityError(
            "refusing to record PUBLISHED without every chunk confirmed",
            run_id=run.run_id,
            confirmed=result.confirmed_count,
            expected=result.chunk_count,
        )

    artifact = PreparedArtifact.from_json(RESULT_FILENAME, result)
    run.commit_artifacts([artifact], manifest)

    status = _STATUS_BY_OUTCOME[result.status]
    manifest.status = status
    manifest.record_event(
        "publish.complete",
        str(result.status),
        f"attempt={result.attempt_id} confirmed={result.confirmed_count}/{result.chunk_count}",
    )
    if result.failure is not None:
        manifest.error = RunError(
            code=result.failure.safe_code,
            message=result.failure.safe_message,
            details={"category": str(result.failure.category)},
        )
    run.save_manifest(manifest)

    logger.info(
        "run=%s attempt=%s stage=publish.complete status=%s confirmed=%d/%d",
        run.run_id,
        result.attempt_id,
        result.status,
        result.confirmed_count,
        result.chunk_count,
    )

    return PublishRunResult(
        run_id=run.run_id,
        run_dir=run.path,
        status=status,
        result=result,
        result_path=run.artifact_path(RESULT_FILENAME),
    )


__all__ = [
    "CHUNK_PACING_SECONDS",
    "INTENT_FILENAME",
    "MAX_RATE_LIMIT_RETRIES",
    "RESULT_FILENAME",
    "PublishRunResult",
    "publish_run",
]
