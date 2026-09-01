"""Reviewer stage orchestration.

The order, and why it is this order:

1. open the Run and refuse if it is not drafted, or already reviewed;
2. verify all three input artifacts against the manifest, and cross-check the
   digests the writer recorded about them;
3. run the deterministic prechecks;
4. render the versioned prompt, carrying those findings;
5. call the provider;
6. validate the answer as a self-consistent review of *this* Run;
7. apply policy - merge deterministic findings in, escalate the verdict if they
   demand it;
8. serialize the artifact and commit it;
9. update the manifest.

Steps 1-3 cost nothing and happen before any provider is contacted, so a
tampered or unreviewable Run never spends money. Nothing is written before step
8: a failure anywhere earlier leaves the Run exactly as Round 2 produced it, plus
one failure event on the ledger, and the stage can simply be run again.

A provider failure is never a verdict. It says nothing about the article, so it
raises rather than being recorded as a REJECT.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from goldpipeline.adapters.reviewer_client import ReviewerClient, ReviewRequest
from goldpipeline.domain.errors import (
    ArtifactIntegrityError,
    ReviewArtifactExistsError,
    ReviewError,
    ReviewSchemaError,
    RunNotReviewableError,
)
from goldpipeline.prompts import DEFAULT_REVIEWER_PROMPT
from goldpipeline.schemas.common import utc_now
from goldpipeline.schemas.context import AnalysisContext
from goldpipeline.schemas.manifest import RunError, RunManifest, RunStatus
from goldpipeline.schemas.review import ReviewResult
from goldpipeline.schemas.writer import WriterResult
from goldpipeline.services.integrity import (
    VerifiedArtifact,
    require_digest_match,
    verify_artifact,
)
from goldpipeline.services.pipeline import CONTEXT_FILENAME
from goldpipeline.services.precheck import run_prechecks
from goldpipeline.services.review_policy import apply_policy, validate_response
from goldpipeline.services.reviewer_prompt import build_reviewer_prompt
from goldpipeline.services.writer import DRAFT_FILENAME, WRITER_FILENAME
from goldpipeline.storage.run_store import PreparedArtifact, RunDirectory, RunStore

logger = logging.getLogger(__name__)

REVIEW_FILENAME = "gpt_review.json"

REVIEWABLE_STATUSES = (RunStatus.DRAFTED,)
"""Only a drafted Run has something to review."""

ReviewFailure = ReviewError | ArtifactIntegrityError
"""The two families of expected failure.

A reviewer failure is about the provider or its answer; an integrity failure
is about the Run's own artifacts. Both stop the stage without writing anything,
and both are reported rather than raised, so a caller can point a human at the
Run either way.
"""


@dataclass(frozen=True)
class ReviewRunResult:
    """Outcome of a reviewer stage attempt."""

    run_id: str
    run_dir: Path
    status: RunStatus
    result: ReviewResult | None = None
    review_path: Path | None = None
    error: ReviewFailure | None = None

    @property
    def succeeded(self) -> bool:
        """Whether a review artifact was committed."""
        return self.status is RunStatus.REVIEWED and self.result is not None


@dataclass(frozen=True)
class ReviewInputs:
    """The three verified artifacts a review is built from."""

    context: AnalysisContext
    context_sha256: str
    article: str
    draft_sha256: str
    writer_result: WriterResult
    writer_metadata_sha256: str


def review_draft(
    *,
    run_id: str,
    store: RunStore,
    client: ReviewerClient,
    prompt_version: str = DEFAULT_REVIEWER_PROMPT,
    max_output_tokens: int = 8000,
    now: datetime | None = None,
) -> ReviewRunResult:
    """Audit an existing drafted Run and persist the verdict.

    Args:
        run_id: The Run to review. Must already be ``DRAFTED``.
        store: Where Runs live.
        client: Any :class:`ReviewerClient` - real or fake.
        prompt_version: Versioned prompt template id, recorded on the artifact.
        max_output_tokens: Output ceiling for the provider call.
        now: Injection point for tests.

    Returns:
        A :class:`ReviewRunResult`. Expected failures are reported through
        ``result.error`` rather than raised, so a caller can still point a human
        at the Run. Programming errors and disk failures propagate.
    """
    run = store.open(run_id)
    manifest = run.load_manifest()

    try:
        return _execute(
            run=run,
            manifest=manifest,
            client=client,
            prompt_version=prompt_version,
            max_output_tokens=max_output_tokens,
            now=now,
        )
    except ReviewError as exc:
        _record_failure(run, manifest, exc)
        return ReviewRunResult(
            run_id=run.run_id, run_dir=run.path, status=manifest.status, error=exc
        )
    except ArtifactIntegrityError as exc:
        # Raised by the shared integrity helpers, which predate this stage.
        _record_failure(run, manifest, exc)
        return ReviewRunResult(
            run_id=run.run_id, run_dir=run.path, status=manifest.status, error=exc
        )


def _execute(
    *,
    run: RunDirectory,
    manifest: RunManifest,
    client: ReviewerClient,
    prompt_version: str,
    max_output_tokens: int,
    now: datetime | None,
) -> ReviewRunResult:
    """Do the work. Raises on any expected failure."""
    _require_reviewable(run, manifest)

    inputs = load_verified_inputs(run, manifest)
    logger.info("run=%s stage=review.start status=OK model=%s", run.run_id, client.model)
    manifest.record_event("review.start", "OK", f"provider={client.provider} model={client.model}")

    try:
        report = run_prechecks(
            context=inputs.context, writer_result=inputs.writer_result, article=inputs.article
        )
    except PydanticValidationError as exc:
        # A deterministic finding built from the Run's own artifacts - a
        # claim's resolved value, most likely - did not fit the schema
        # PrecheckFinding declares. This is local computation, not the
        # network: no request has been sent, and the same bytes will fail the
        # same way on the next attempt. Never persist the rejected value
        # itself, only where in the schema it was rejected.
        raise ReviewSchemaError(
            f"run {run.run_id}: a deterministic precheck finding failed its own schema",
            run_id=run.run_id,
            phase="PREPARE",
            errors=[
                {"loc": ".".join(str(part) for part in err["loc"]), "type": err["type"]}
                for err in exc.errors()
            ],
        ) from exc
    logger.info(
        "run=%s stage=review.precheck status=OK findings=%d blocking=%d",
        run.run_id,
        len(report.findings),
        len(report.blocking),
    )

    prompt = build_reviewer_prompt(
        context=inputs.context,
        writer_result=inputs.writer_result,
        article=inputs.article,
        report=report,
        prompt_version=prompt_version,
    )

    response = client.review(
        ReviewRequest(prompt=prompt, run_id=run.run_id, max_output_tokens=max_output_tokens)
    )

    validate_response(response.output, run_id=run.run_id)
    outcome = apply_policy(response.output, report)

    result = ReviewResult(
        run_id=run.run_id,
        status=outcome.status,
        score=response.output.score,
        summary=response.output.summary,
        issues=outcome.issues,
        revision_instructions=list(response.output.revision_instructions),
        model_status=outcome.model_status,
        verdict_source=outcome.verdict_source,
        policy_notes=outcome.notes,
        deterministic_findings=list(report.findings),
        model=response.model,
        provider=response.provider,
        prompt_version=prompt.prompt_version,
        reviewed_at=now or utc_now(),
        context_sha256=inputs.context_sha256,
        draft_sha256=inputs.draft_sha256,
        writer_metadata_sha256=inputs.writer_metadata_sha256,
        usage=response.usage,
    )

    run.commit_artifacts([PreparedArtifact.from_json(REVIEW_FILENAME, result)], manifest)

    manifest.status = RunStatus.REVIEWED
    manifest.record_event(
        "review.complete",
        str(result.status),
        f"score={result.score} issues={len(result.issues)} verdict_source={result.verdict_source}",
    )
    run.save_manifest(manifest)
    logger.info(
        "run=%s stage=review.complete status=%s score=%d issues=%d source=%s",
        run.run_id,
        result.status,
        result.score,
        len(result.issues),
        result.verdict_source,
    )

    return ReviewRunResult(
        run_id=run.run_id,
        run_dir=run.path,
        status=RunStatus.REVIEWED,
        result=result,
        review_path=run.artifact_path(REVIEW_FILENAME),
    )


# --------------------------------------------------------------------------
# preconditions
# --------------------------------------------------------------------------


def _require_reviewable(run: RunDirectory, manifest: RunManifest) -> None:
    """Refuse unless the Run is drafted and not already reviewed."""
    if run.has_artifact(REVIEW_FILENAME):
        raise ReviewArtifactExistsError(
            f"run {run.run_id} already has {REVIEW_FILENAME}. Runs are immutable; "
            "a verdict someone may have acted on is never replaced.",
            run_id=run.run_id,
            artifact=REVIEW_FILENAME,
        )

    if manifest.status not in REVIEWABLE_STATUSES:
        raise RunNotReviewableError(
            f"run {run.run_id} is {manifest.status}, the reviewer needs {RunStatus.DRAFTED}",
            run_id=run.run_id,
            status=str(manifest.status),
        )

    missing = [
        name
        for name in (CONTEXT_FILENAME, DRAFT_FILENAME, WRITER_FILENAME)
        if not run.has_artifact(name)
    ]
    if missing:
        raise RunNotReviewableError(
            f"run {run.run_id} is missing artifacts the reviewer needs: {missing}",
            run_id=run.run_id,
            missing=missing,
        )


def load_verified_inputs(run: RunDirectory, manifest: RunManifest) -> ReviewInputs:
    """Load and cross-check every artifact the review is built from.

    Three separate proofs, because each catches a different kind of tampering:

    * each file matches the digest the manifest recorded for it;
    * the writer result names the same context and draft digests, so the
      metadata and the article still describe each other;
    * every document agrees on the run id.

    Raises:
        ArtifactIntegrityError: If any of those fail.
    """
    context_artifact = verify_artifact(run, manifest, CONTEXT_FILENAME)
    draft_artifact = verify_artifact(run, manifest, DRAFT_FILENAME)
    writer_artifact = verify_artifact(run, manifest, WRITER_FILENAME)

    context = _parse_context(run, context_artifact)
    writer_result = _parse_writer_result(run, writer_artifact)

    require_digest_match(
        label=f"{WRITER_FILENAME}.article_sha256 vs {DRAFT_FILENAME}",
        expected=writer_result.article_sha256,
        actual=draft_artifact.sha256,
        run_id=run.run_id,
        artifact=DRAFT_FILENAME,
    )
    require_digest_match(
        label=f"{WRITER_FILENAME}.context_sha256 vs {CONTEXT_FILENAME}",
        expected=writer_result.context_sha256,
        actual=context_artifact.sha256,
        run_id=run.run_id,
        artifact=CONTEXT_FILENAME,
    )

    for label, value in (
        (CONTEXT_FILENAME, context.run_id),
        (WRITER_FILENAME, writer_result.run_id),
    ):
        if value != run.run_id:
            raise ArtifactIntegrityError(
                f"{label} belongs to run {value}, not {run.run_id}",
                run_id=run.run_id,
                artifact=label,
                found_run_id=value,
            )

    article = draft_artifact.text.strip()
    if not article:
        raise ArtifactIntegrityError(
            f"{DRAFT_FILENAME} is empty", run_id=run.run_id, artifact=DRAFT_FILENAME
        )

    return ReviewInputs(
        context=context,
        context_sha256=context_artifact.sha256,
        article=article,
        draft_sha256=draft_artifact.sha256,
        writer_result=writer_result,
        writer_metadata_sha256=writer_artifact.sha256,
    )


def _parse_context(run: RunDirectory, artifact: VerifiedArtifact) -> AnalysisContext:
    try:
        return AnalysisContext.model_validate_json(artifact.text)
    except (PydanticValidationError, UnicodeDecodeError) as exc:
        raise ArtifactIntegrityError(
            f"{artifact.name} does not satisfy the context schema",
            run_id=run.run_id,
            artifact=artifact.name,
        ) from exc


def _parse_writer_result(run: RunDirectory, artifact: VerifiedArtifact) -> WriterResult:
    try:
        return WriterResult.model_validate_json(artifact.text)
    except (PydanticValidationError, UnicodeDecodeError) as exc:
        raise ArtifactIntegrityError(
            f"{artifact.name} does not satisfy the writer result schema",
            run_id=run.run_id,
            artifact=artifact.name,
        ) from exc


def _record_failure(run: RunDirectory, manifest: RunManifest, exc: ReviewFailure) -> None:
    """Append a failure event to the ledger.

    The Run's status is left alone. A reviewer failure does not invalidate the
    draft, and marking the Run ``FAILED`` would wrongly imply it cannot be
    retried - which is exactly what this stage supports.
    """
    manifest.error = RunError(code=exc.code, message=exc.message, details=exc.details)
    manifest.record_event("review.failed", exc.code, exc.message)
    run.save_manifest(manifest)
    logger.error(
        "run=%s stage=review.failed status=%s message=%s", run.run_id, exc.code, exc.message
    )


__all__ = [
    "REVIEW_FILENAME",
    "ReviewFailure",
    "ReviewInputs",
    "ReviewRunResult",
    "load_verified_inputs",
    "review_draft",
]
