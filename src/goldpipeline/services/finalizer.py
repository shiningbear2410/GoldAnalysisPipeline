"""Finalizer stage orchestration.

The **action** decides the path, and two of the three never reach a provider.
Since Round 6.4g the action is not the same thing as the review verdict:
:func:`~goldpipeline.services.review_action.effective_action` reads the content
verdict *and* the human-style verdict beside it and answers one question - does
a model run? The verdict on the artifact is never rewritten to make that
answer come out a particular way.

* ``PASS_THROUGH`` - nothing needs repair. The draft is copied byte for byte and
  the stage records ``PASSTHROUGH``. Calling a model here would spend money to
  introduce drift into something that passed, which is the opposite of the job.
* ``FINALIZE`` - one call. Content corrections, style repairs, or both together.
* ``REJECT`` - blocked. A reviewer judged the piece unsalvageable, and
  "ask a model to rescue it" is not a recovery strategy. The Run waits for a
  human.

For the revision path the order is:

1. refuse unless the Run is reviewed and not already finalized;
2. verify all four artifacts against the manifest, and the review's own
   cross-references against them;
3. re-run the deterministic checks on the draft, for a baseline;
4. render the versioned prompt, carrying the style findings to repair;
5. call the provider - **once**;
6. validate the resolutions - every content issue answered, severe ones
   actually applied, every style finding answered and none left unresolved;
7. re-run the checks on the revision and compare against the baseline;
8. run the final deterministic checks on the finished article: output contract,
   authoritative date, numeric claims, and the sections the revision had no
   licence to touch;
9. commit both artifacts atomically;
10. update the manifest.

Nothing is written before step 9. A failure anywhere earlier leaves the Run
exactly as Round 3 produced it, plus one failure event on the ledger - and it is
the end of the automatic path. There is no step that calls the model again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from goldpipeline.adapters.finalizer_client import FinalizerClient, FinalizeRequest
from goldpipeline.domain.errors import (
    ArtifactIntegrityError,
    FinalizationBlockedError,
    FinalizeArtifactExistsError,
    FinalizeError,
    FinalizePostcheckError,
    RunNotFinalizableError,
)
from goldpipeline.prompts import DEFAULT_FINALIZER_PROMPT
from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.common import utc_now
from goldpipeline.schemas.context import AnalysisContext
from goldpipeline.schemas.finalizer import (
    FinalizationMode,
    FinalizerResult,
    FinalizerUsage,
    FinalizerWarning,
    IssueResolution,
    StyleResolution,
)
from goldpipeline.schemas.manifest import RunError, RunManifest, RunStatus
from goldpipeline.schemas.review import HumanStyleFinding, PrecheckFinding, ReviewResult
from goldpipeline.schemas.writer import WriterResult
from goldpipeline.services.final_postcheck import (
    FinalPostcheckReport,
    check_final_article,
    describe,
)
from goldpipeline.services.finalizer_policy import (
    compare_findings,
    require_clean_postcheck,
    validate_resolutions,
    validate_style_resolutions,
)
from goldpipeline.services.finalizer_prompt import build_finalizer_prompt
from goldpipeline.services.integrity import (
    VerifiedArtifact,
    require_digest_match,
    verify_artifact,
)
from goldpipeline.services.market_facts import article_date
from goldpipeline.services.pipeline import CONTEXT_FILENAME
from goldpipeline.services.precheck import run_prechecks
from goldpipeline.services.review_action import ActionDecision, ReviewAction, effective_action
from goldpipeline.services.reviewer import REVIEW_FILENAME
from goldpipeline.services.writer import DRAFT_FILENAME, WRITER_FILENAME
from goldpipeline.storage.run_store import PreparedArtifact, RunDirectory, RunStore

logger = logging.getLogger(__name__)

FINAL_FILENAME = "claude_final.md"
FINALIZER_FILENAME = "claude_finalizer.json"

FINALIZER_ARTIFACTS = (FINAL_FILENAME, FINALIZER_FILENAME)

FinalizeFailure = FinalizeError | ArtifactIntegrityError
"""The two families of expected failure: the stage's own, and artifact tampering."""


@dataclass(frozen=True)
class FinalizeRunResult:
    """Outcome of a finalizer stage attempt."""

    run_id: str
    run_dir: Path
    status: RunStatus
    result: FinalizerResult | None = None
    final_path: Path | None = None
    metadata_path: Path | None = None
    error: FinalizeFailure | None = None

    @property
    def succeeded(self) -> bool:
        """Whether both artifacts were committed."""
        return self.status is RunStatus.FINALIZED and self.result is not None

    @property
    def blocked(self) -> bool:
        """Whether the review verdict forbade finalization.

        Distinct from a failure: nothing went wrong, the pipeline declined.
        """
        return isinstance(self.error, FinalizationBlockedError)


@dataclass(frozen=True)
class FinalizeInputs:
    """The four verified artifacts a finalization is built from."""

    context: AnalysisContext
    context_sha256: str
    draft_bytes: bytes
    draft_sha256: str
    writer_result: WriterResult
    writer_metadata_sha256: str
    review: ReviewResult
    review_sha256: str

    @property
    def article(self) -> str:
        """The draft as text."""
        return self.draft_bytes.decode("utf-8").strip()


def finalize_run(
    *,
    run_id: str,
    store: RunStore,
    client: FinalizerClient | None = None,
    prompt_version: str = DEFAULT_FINALIZER_PROMPT,
    max_tokens: int = 8000,
    now: datetime | None = None,
) -> FinalizeRunResult:
    """Produce the final article for a reviewed Run.

    Args:
        run_id: The Run to finalize. Must already be ``REVIEWED``.
        store: Where Runs live.
        client: Any :class:`FinalizerClient`. Only needed when the verdict is
            ``NEEDS_REVISION``; a passthrough or a block never touches it, so
            ``None`` is legitimate for those.
        prompt_version: Versioned prompt template id.
        max_tokens: Output ceiling for the provider call.
        now: Injection point for tests.

    Returns:
        A :class:`FinalizeRunResult`. Expected failures - including a blocking
        ``REJECT`` - are reported through ``result.error`` rather than raised.
    """
    run = store.open(run_id)
    manifest = run.load_manifest()

    try:
        return _execute(
            run=run,
            manifest=manifest,
            client=client,
            prompt_version=prompt_version,
            max_tokens=max_tokens,
            now=now,
        )
    except (FinalizeError, ArtifactIntegrityError) as exc:
        _record_failure(run, manifest, exc)
        return FinalizeRunResult(
            run_id=run.run_id, run_dir=run.path, status=manifest.status, error=exc
        )


def _execute(
    *,
    run: RunDirectory,
    manifest: RunManifest,
    client: FinalizerClient | None,
    prompt_version: str,
    max_tokens: int,
    now: datetime | None,
) -> FinalizeRunResult:
    """Do the work. Raises on any expected failure."""
    _require_finalizable(run, manifest)
    inputs = load_verified_inputs(run, manifest)

    # Provenance is absent on Runs created before it existed, and those Runs
    # were all ANALYSIS - which is what the manifest field itself defaults to.
    article_type = manifest.provenance.article_type if manifest.provenance else ArticleType.ANALYSIS

    # The one place that decides. `review.status` is what was *judged* and is
    # never rewritten; this is what the pipeline *does* about it. Round 6.4g
    # separated the two so that a content PASS with a style problem can reach
    # the finalizer without the artifact having to claim the content failed.
    decision = effective_action(inputs.review, article_type=article_type)
    verdict = decision.content_status

    logger.info(
        "run=%s stage=finalize.start status=OK action=%s content=%s style=%s findings=%d",
        run.run_id,
        decision.action,
        verdict,
        decision.style_verdict,
        len(decision.style_findings),
    )
    manifest.record_event(
        "finalize.start", str(decision.action), "; ".join(decision.reasons) or str(verdict)
    )

    if decision.action is ReviewAction.REJECT:
        raise FinalizationBlockedError(
            f"review verdict is {verdict}; finalization is blocked. "
            "A rejected article needs a human, not another model.",
            run_id=run.run_id,
            review_status=str(verdict),
            issue_count=len(inputs.review.issues),
        )

    if decision.action is ReviewAction.PASS_THROUGH:
        return _finalize_passthrough(run=run, manifest=manifest, inputs=inputs, now=now)

    return _finalize_revision(
        run=run,
        manifest=manifest,
        inputs=inputs,
        decision=decision,
        article_type=article_type,
        client=client,
        prompt_version=prompt_version,
        max_tokens=max_tokens,
        now=now,
    )


# --------------------------------------------------------------------------
# the two accepting paths
# --------------------------------------------------------------------------


def _finalize_passthrough(
    *,
    run: RunDirectory,
    manifest: RunManifest,
    inputs: FinalizeInputs,
    now: datetime | None,
) -> FinalizeRunResult:
    """Copy the draft to the final article, byte for byte.

    Not "re-serialize the text" - the exact bytes. Normalizing whitespace or a
    trailing newline here would mean the published article differs from the one
    that was reviewed, which is the whole thing a passthrough exists to prevent.
    """
    final = PreparedArtifact.from_bytes(FINAL_FILENAME, inputs.draft_bytes)

    result = _build_result(
        inputs=inputs,
        run_id=run.run_id,
        mode=FinalizationMode.PASSTHROUGH,
        provider_called=False,
        final=final,
        article=inputs.article,
        resolutions=[],
        warnings=[],
        postcheck_findings=[],
        model=None,
        provider=None,
        prompt_version=None,
        usage=FinalizerUsage(),
        now=now,
    )
    return _commit(run=run, manifest=manifest, final=final, result=result)


def _finalize_revision(
    *,
    run: RunDirectory,
    manifest: RunManifest,
    inputs: FinalizeInputs,
    decision: ActionDecision,
    article_type: ArticleType,
    client: FinalizerClient | None,
    prompt_version: str,
    max_tokens: int,
    now: datetime | None,
) -> FinalizeRunResult:
    """One model call, then every deterministic check that can be made.

    **Exactly one call.** There is a single ``client.finalize`` below, no loop,
    no retry, and no other path that reaches it. Everything after it either
    accepts the article or stops the Run: a revision that fails a check is never
    sent back for another attempt, because a model told "your last answer was
    rejected" changes things nobody asked it to, and because there is no way for
    deterministic code to adjudicate a second opinion about prose.

    Content and style are repaired together, in this one call. Splitting them
    into two would double the cost, double the drift, and leave the second pass
    editing an article the first pass had already changed underneath it.
    """
    if client is None:
        raise RunNotFinalizableError(
            f"run {run.run_id} needs revision, which requires a finalizer client",
            run_id=run.run_id,
        )

    baseline = run_prechecks(
        context=inputs.context,
        writer_result=inputs.writer_result,
        article=inputs.article,
        check_claims=False,
    )

    prompt = build_finalizer_prompt(
        context=inputs.context,
        article=inputs.article,
        review=inputs.review,
        report=baseline,
        style_findings=decision.style_findings,
        prompt_version=prompt_version,
    )

    response = client.finalize(
        FinalizeRequest(prompt=prompt, run_id=run.run_id, max_tokens=max_tokens)
    )
    validate_resolutions(response.output, inputs.review, run_id=run.run_id)
    validate_style_resolutions(response.output, decision.style_findings, run_id=run.run_id)

    article = response.output.article.strip()
    revised = run_prechecks(
        context=inputs.context,
        writer_result=inputs.writer_result,
        article=article,
        check_claims=False,
    )
    outcome = compare_findings(
        original=baseline, revised=revised, review=inputs.review, final_article=article
    )
    require_clean_postcheck(outcome)

    final_report = _require_clean_final_article(
        run_id=run.run_id,
        article=article,
        inputs=inputs,
        article_type=article_type,
        style_findings=decision.style_findings,
    )

    logger.info(
        "run=%s stage=finalize.postcheck status=OK findings=%d chars=%d->%d "
        "changed=%s symptoms=%d->%d",
        run.run_id,
        len(outcome.findings),
        len(inputs.article),
        len(article),
        [str(key) for key in final_report.changed_sections],
        final_report.symptoms_before,
        final_report.symptoms_after,
    )
    if final_report.symptoms_worse_by:
        # Recorded, never fatal. A symptom is a countable pattern, not a
        # judgement, and refusing a revision over one more of them would put
        # deterministic code in charge of prose - which this round refuses.
        logger.info(
            "run=%s stage=finalize.symptoms note=worse_by=%d",
            run.run_id,
            final_report.symptoms_worse_by,
        )

    final = PreparedArtifact.from_text(FINAL_FILENAME, article)
    result = _build_result(
        inputs=inputs,
        run_id=run.run_id,
        mode=FinalizationMode.REVISED,
        provider_called=True,
        final=final,
        article=article,
        resolutions=list(response.output.issue_resolutions),
        style_resolutions=list(response.output.style_resolutions),
        warnings=list(response.output.warnings),
        postcheck_findings=outcome.findings,
        model=response.model,
        provider=response.provider,
        selection_id=response.selection_id,
        prompt_version=prompt.prompt_version,
        usage=response.usage,
        chars_before=len(inputs.article),
        chars_after=len(article),
        changed_sections=[str(key) for key in final_report.changed_sections],
        symptoms_before=final_report.symptoms_before,
        symptoms_after=final_report.symptoms_after,
        now=now,
    )
    return _commit(run=run, manifest=manifest, final=final, result=result)


def _require_clean_final_article(
    *,
    run_id: str,
    article: str,
    inputs: FinalizeInputs,
    article_type: ArticleType,
    style_findings: tuple[HumanStyleFinding, ...],
) -> FinalPostcheckReport:
    """The last deterministic word on the finished article.

    Terminal by design. Every failure here stops the Run for a person; none of
    them is a reason to call the model again.
    """
    report = check_final_article(
        article=article,
        draft=inputs.article,
        context=inputs.context,
        writer_result=inputs.writer_result,
        article_type=article_type,
        expected_date=article_date(inputs.context.timing.latest_candle_at),
        style_findings=style_findings,
    )
    if report.ok:
        return report

    raise FinalizePostcheckError(
        "the finished article failed the final deterministic checks; the Run "
        "stops here rather than spending a second model call",
        run_id=run_id,
        **describe(report),
    )


def _build_result(
    *,
    inputs: FinalizeInputs,
    run_id: str,
    mode: FinalizationMode,
    provider_called: bool,
    final: PreparedArtifact,
    article: str,
    resolutions: list[IssueResolution],
    warnings: list[FinalizerWarning],
    postcheck_findings: list[PrecheckFinding],
    model: str | None,
    provider: str | None,
    prompt_version: str | None,
    selection_id: str | None = None,
    usage: FinalizerUsage,
    style_resolutions: list[StyleResolution] | None = None,
    chars_before: int | None = None,
    chars_after: int | None = None,
    changed_sections: list[str] | None = None,
    symptoms_before: int | None = None,
    symptoms_after: int | None = None,
    now: datetime | None,
) -> FinalizerResult:
    """Stamp the metadata artifact. Every provenance field is set here.

    ``review_status`` still records the *content* verdict, unchanged. Round 6.4g
    added a second reason a revision can happen; it did not change what that
    field has always meant, and a reader of an old artifact and a new one is
    looking at the same thing.
    """
    return FinalizerResult(
        run_id=run_id,
        finalization_mode=mode,
        review_status=inputs.review.status,
        provider_called=provider_called,
        final_file=FINAL_FILENAME,
        final_article_sha256=final.sha256,
        article_chars=len(article),
        issue_resolutions=resolutions,
        style_resolutions=list(style_resolutions or []),
        warnings=warnings,
        postcheck_findings=postcheck_findings,
        chars_before=chars_before,
        chars_after=chars_after,
        changed_sections=list(changed_sections or []),
        style_symptoms_before=symptoms_before,
        style_symptoms_after=symptoms_after,
        model=model,
        provider=provider,
        selection_id=selection_id,
        prompt_version=prompt_version,
        created_at=now or utc_now(),
        context_sha256=inputs.context_sha256,
        original_draft_sha256=inputs.draft_sha256,
        writer_metadata_sha256=inputs.writer_metadata_sha256,
        review_sha256=inputs.review_sha256,
        usage=usage,
    )


def _commit(
    *,
    run: RunDirectory,
    manifest: RunManifest,
    final: PreparedArtifact,
    result: FinalizerResult,
) -> FinalizeRunResult:
    """Write both artifacts as one unit, then move the Run to FINALIZED."""
    metadata = PreparedArtifact.from_json(FINALIZER_FILENAME, result)
    run.commit_artifacts([final, metadata], manifest)

    manifest.status = RunStatus.FINALIZED
    manifest.record_event(
        "finalize.complete",
        str(result.finalization_mode),
        f"{result.article_chars} chars, {result.applied_count}/"
        f"{len(result.issue_resolutions)} issues applied, "
        f"provider_called={result.provider_called}",
    )
    run.save_manifest(manifest)
    logger.info(
        "run=%s stage=finalize.complete status=%s chars=%d provider_called=%s",
        run.run_id,
        result.finalization_mode,
        result.article_chars,
        result.provider_called,
    )

    return FinalizeRunResult(
        run_id=run.run_id,
        run_dir=run.path,
        status=RunStatus.FINALIZED,
        result=result,
        final_path=run.artifact_path(FINAL_FILENAME),
        metadata_path=run.artifact_path(FINALIZER_FILENAME),
    )


# --------------------------------------------------------------------------
# preconditions
# --------------------------------------------------------------------------


def _require_finalizable(run: RunDirectory, manifest: RunManifest) -> None:
    """Refuse unless the Run is reviewed and not already finalized."""
    existing = [name for name in FINALIZER_ARTIFACTS if run.has_artifact(name)]
    if existing:
        raise FinalizeArtifactExistsError(
            f"run {run.run_id} already has finalizer artifacts: {existing}. "
            "Runs are immutable; a published article is never silently replaced.",
            run_id=run.run_id,
            artifacts=existing,
        )

    if manifest.status is not RunStatus.REVIEWED:
        raise RunNotFinalizableError(
            f"run {run.run_id} is {manifest.status}, the finalizer needs {RunStatus.REVIEWED}",
            run_id=run.run_id,
            status=str(manifest.status),
        )

    missing = [
        name
        for name in (CONTEXT_FILENAME, DRAFT_FILENAME, WRITER_FILENAME, REVIEW_FILENAME)
        if not run.has_artifact(name)
    ]
    if missing:
        raise RunNotFinalizableError(
            f"run {run.run_id} is missing artifacts the finalizer needs: {missing}",
            run_id=run.run_id,
            missing=missing,
        )


def load_verified_inputs(run: RunDirectory, manifest: RunManifest) -> FinalizeInputs:
    """Load and cross-check every artifact the finalization is built from.

    Four proofs, each catching a different kind of tampering:

    * each file matches the digest the manifest recorded for it;
    * the writer result still describes the draft and the context it named;
    * the review still describes the same three inputs it judged;
    * every document agrees on the run id.

    Raises:
        ArtifactIntegrityError: If any of those fail.
    """
    context_artifact = verify_artifact(run, manifest, CONTEXT_FILENAME)
    draft_artifact = verify_artifact(run, manifest, DRAFT_FILENAME)
    writer_artifact = verify_artifact(run, manifest, WRITER_FILENAME)
    review_artifact = verify_artifact(run, manifest, REVIEW_FILENAME)

    context = _parse(run, context_artifact, AnalysisContext, "context")
    writer_result = _parse(run, writer_artifact, WriterResult, "writer result")
    review = _parse(run, review_artifact, ReviewResult, "review")

    require_digest_match(
        label=f"{WRITER_FILENAME}.article_sha256 vs {DRAFT_FILENAME}",
        expected=writer_result.article_sha256,
        actual=draft_artifact.sha256,
        run_id=run.run_id,
        artifact=DRAFT_FILENAME,
    )
    require_digest_match(
        label=f"{REVIEW_FILENAME}.draft_sha256 vs {DRAFT_FILENAME}",
        expected=review.draft_sha256,
        actual=draft_artifact.sha256,
        run_id=run.run_id,
        artifact=DRAFT_FILENAME,
    )
    require_digest_match(
        label=f"{REVIEW_FILENAME}.context_sha256 vs {CONTEXT_FILENAME}",
        expected=review.context_sha256,
        actual=context_artifact.sha256,
        run_id=run.run_id,
        artifact=CONTEXT_FILENAME,
    )
    require_digest_match(
        label=f"{REVIEW_FILENAME}.writer_metadata_sha256 vs {WRITER_FILENAME}",
        expected=review.writer_metadata_sha256,
        actual=writer_artifact.sha256,
        run_id=run.run_id,
        artifact=WRITER_FILENAME,
    )

    for label, value in (
        (CONTEXT_FILENAME, context.run_id),
        (WRITER_FILENAME, writer_result.run_id),
        (REVIEW_FILENAME, review.run_id),
    ):
        if value != run.run_id:
            raise ArtifactIntegrityError(
                f"{label} belongs to run {value}, not {run.run_id}",
                run_id=run.run_id,
                artifact=label,
                found_run_id=value,
            )

    if not draft_artifact.text.strip():
        raise ArtifactIntegrityError(
            f"{DRAFT_FILENAME} is empty", run_id=run.run_id, artifact=DRAFT_FILENAME
        )

    return FinalizeInputs(
        context=context,
        context_sha256=context_artifact.sha256,
        draft_bytes=draft_artifact.payload,
        draft_sha256=draft_artifact.sha256,
        writer_result=writer_result,
        writer_metadata_sha256=writer_artifact.sha256,
        review=review,
        review_sha256=review_artifact.sha256,
    )


def _parse[ModelT: BaseModel](
    run: RunDirectory, artifact: VerifiedArtifact, model: type[ModelT], label: str
) -> ModelT:
    """Parse an artifact, turning a schema failure into an integrity failure."""
    try:
        return model.model_validate_json(artifact.text)
    except (PydanticValidationError, UnicodeDecodeError) as exc:
        raise ArtifactIntegrityError(
            f"{artifact.name} does not satisfy the {label} schema",
            run_id=run.run_id,
            artifact=artifact.name,
        ) from exc


def _record_failure(run: RunDirectory, manifest: RunManifest, exc: FinalizeFailure) -> None:
    """Append a failure event to the ledger.

    The Run's status is left at ``REVIEWED``. A blocked or failed finalization
    does not invalidate the review, and a ``REJECT`` in particular is a correct
    outcome that a human still needs to see.
    """
    manifest.error = RunError(code=exc.code, message=exc.message, details=exc.details)
    manifest.record_event("finalize.failed", exc.code, exc.message)
    run.save_manifest(manifest)

    log = logger.warning if isinstance(exc, FinalizationBlockedError) else logger.error
    log("run=%s stage=finalize.failed status=%s message=%s", run.run_id, exc.code, exc.message)


__all__ = [
    "FINALIZER_ARTIFACTS",
    "FINALIZER_FILENAME",
    "FINAL_FILENAME",
    "FinalizeFailure",
    "FinalizeInputs",
    "FinalizeRunResult",
    "finalize_run",
    "load_verified_inputs",
]
