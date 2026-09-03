"""Writer stage orchestration.

The order below is what makes the stage safe to retry:

1. open the Run and check it is ready (``NORMALIZED``, no writer artifacts yet);
2. load ``context.json`` and verify it against the digest in the manifest;
3. screen the source text for prices the market data contradicts;
4. render the versioned prompt;
5. call the provider;
6. validate the answer against this Run;
7. serialize both artifacts **in memory**;
8. commit them as one atomic unit;
9. update the manifest.

Nothing is written before step 8. A failure anywhere in 1-7 leaves the Run
byte-for-byte as Round 1 produced it, apart from a failure event appended to the
manifest ledger - so the stage can simply be run again.

The Run's own status only moves to ``DRAFTED`` after the artifacts are on disk.
A manifest that claims the writer completed therefore always describes files
that exist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from goldpipeline.adapters.writer_client import WriterClient, WriterRequest
from goldpipeline.domain.errors import (
    ContextIntegrityError,
    RunNotReadyError,
    WriterArtifactExistsError,
    WriterError,
    WriterResponseError,
)
from goldpipeline.prompts import DEFAULT_WRITER_PROMPT
from goldpipeline.schemas.common import utc_now
from goldpipeline.schemas.context import AnalysisContext
from goldpipeline.schemas.manifest import RunError, RunManifest, RunStatus
from goldpipeline.schemas.quality import QualityStatus
from goldpipeline.schemas.writer import (
    WarningCode,
    WriterModelOutput,
    WriterResult,
    WriterStatus,
    WriterWarning,
)
from goldpipeline.services.claim_resolver import ClaimPathError, resolve_path
from goldpipeline.services.pipeline import CONTEXT_FILENAME
from goldpipeline.services.source_guard import (
    SourceGuardReport,
    build_guard_warnings,
    screen_source_prices,
)
from goldpipeline.services.writer_prompt import build_writer_prompt
from goldpipeline.storage.atomic import sha256_bytes
from goldpipeline.storage.run_store import PreparedArtifact, RunDirectory, RunStore

logger = logging.getLogger(__name__)

DRAFT_FILENAME = "claude_draft.md"
WRITER_FILENAME = "claude_writer.json"

WRITER_ARTIFACTS = (DRAFT_FILENAME, WRITER_FILENAME)

MIN_ARTICLE_CHARS = 40
"""Below this an "article" is a stub, not a publishable piece."""


@dataclass(frozen=True)
class WriterRunResult:
    """Outcome of a writer stage attempt."""

    run_id: str
    run_dir: Path
    status: RunStatus
    result: WriterResult | None = None
    draft_path: Path | None = None
    metadata_path: Path | None = None
    error: WriterError | None = None

    @property
    def succeeded(self) -> bool:
        """Whether both artifacts were committed."""
        return self.status is RunStatus.DRAFTED and self.result is not None


def write_draft(
    *,
    run_id: str,
    store: RunStore,
    client: WriterClient,
    prompt_version: str = DEFAULT_WRITER_PROMPT,
    max_tokens: int = 8000,
    now: datetime | None = None,
) -> WriterRunResult:
    """Generate and persist a draft for an existing normalized Run.

    Args:
        run_id: The Run to write for. Must already be ``NORMALIZED``.
        store: Where Runs live.
        client: Any :class:`WriterClient` - real or fake.
        prompt_version: Versioned prompt template id, recorded on the artifact.
        max_tokens: Output ceiling for the provider call.
        now: Injection point for tests.

    Returns:
        A :class:`WriterRunResult`. Expected failures are reported through
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
            max_tokens=max_tokens,
            now=now,
        )
    except WriterError as exc:
        _record_failure(run, manifest, exc)
        return WriterRunResult(
            run_id=run.run_id,
            run_dir=run.path,
            status=manifest.status,
            error=exc,
        )


def _execute(
    *,
    run: RunDirectory,
    manifest: RunManifest,
    client: WriterClient,
    prompt_version: str,
    max_tokens: int,
    now: datetime | None,
) -> WriterRunResult:
    """Do the work. Raises :class:`WriterError` on any expected failure."""
    _require_ready(run, manifest)

    context, context_digest = load_verified_context(run, manifest)
    logger.info("run=%s stage=writer.start status=OK model=%s", run.run_id, client.model)
    manifest.record_event("writer.start", "OK", f"provider={client.provider} model={client.model}")

    guard = screen_source_prices(context)
    prompt = build_writer_prompt(context, guard_report=guard, prompt_version=prompt_version)

    response = client.generate(
        WriterRequest(prompt=prompt, run_id=run.run_id, max_tokens=max_tokens)
    )
    output = _validate_output(response.output, context)

    warnings = _merge_warnings(output, context, guard)
    article = output.article.strip()

    draft = PreparedArtifact.from_text(DRAFT_FILENAME, article)
    result = WriterResult(
        run_id=run.run_id,
        status=output.status,
        title=output.title,
        model=response.model,
        provider=response.provider,
        selection_id=response.selection_id,
        prompt_version=prompt.prompt_version,
        context_sha256=context_digest,
        draft_file=DRAFT_FILENAME,
        article_sha256=draft.sha256,
        article_chars=len(article),
        created_at=now or utc_now(),
        source_claims=list(output.source_claims),
        news_claims=list(output.news_claims),
        warnings=warnings,
        usage=response.usage,
    )
    metadata = PreparedArtifact.from_json(WRITER_FILENAME, result)

    run.commit_artifacts([draft, metadata], manifest)

    manifest.status = RunStatus.DRAFTED
    manifest.record_event(
        "writer.complete",
        "OK",
        f"{result.article_chars} chars, {len(result.source_claims)} claims, "
        f"{len(result.warnings)} warnings",
    )
    run.save_manifest(manifest)
    logger.info(
        "run=%s stage=writer.complete status=OK chars=%d warnings=%d",
        run.run_id,
        result.article_chars,
        len(result.warnings),
    )

    return WriterRunResult(
        run_id=run.run_id,
        run_dir=run.path,
        status=RunStatus.DRAFTED,
        result=result,
        draft_path=run.artifact_path(DRAFT_FILENAME),
        metadata_path=run.artifact_path(WRITER_FILENAME),
    )


# --------------------------------------------------------------------------
# preconditions
# --------------------------------------------------------------------------


def _require_ready(run: RunDirectory, manifest: RunManifest) -> None:
    """Refuse to run unless the Run is normalized and not already drafted."""
    existing = [name for name in WRITER_ARTIFACTS if run.has_artifact(name)]
    if existing:
        raise WriterArtifactExistsError(
            f"run {run.run_id} already has writer artifacts: {existing}. "
            "Runs are immutable; create a new Run instead.",
            run_id=run.run_id,
            artifacts=existing,
        )

    if manifest.status is not RunStatus.NORMALIZED:
        raise RunNotReadyError(
            f"run {run.run_id} is {manifest.status}, the writer needs {RunStatus.NORMALIZED}",
            run_id=run.run_id,
            status=str(manifest.status),
        )

    if not run.has_artifact(CONTEXT_FILENAME):
        raise RunNotReadyError(f"run {run.run_id} has no {CONTEXT_FILENAME}", run_id=run.run_id)


def load_verified_context(run: RunDirectory, manifest: RunManifest) -> tuple[AnalysisContext, str]:
    """Load ``context.json``, proving it is the file Round 1 wrote.

    The manifest records a digest for every artifact. Checking it here is what
    makes "the writer worked from the Run's real inputs" a fact rather than an
    assumption - a context edited by hand after normalization is rejected
    instead of silently becoming the basis of a published article.

    Returns:
        The parsed context and its SHA-256.

    Raises:
        ContextIntegrityError: If the bytes do not match the manifest, or the
            document no longer satisfies its schema.
    """
    raw = run.read_artifact_bytes(CONTEXT_FILENAME)
    digest = sha256_bytes(raw)

    recorded = next((ref for ref in manifest.artifact_files if ref.name == CONTEXT_FILENAME), None)
    if recorded is None:
        raise ContextIntegrityError(
            f"manifest for run {run.run_id} does not record {CONTEXT_FILENAME}",
            run_id=run.run_id,
        )
    if recorded.sha256 != digest:
        raise ContextIntegrityError(
            f"{CONTEXT_FILENAME} has changed since it was written; refusing to use it",
            run_id=run.run_id,
            expected_sha256=recorded.sha256,
            actual_sha256=digest,
        )

    try:
        context = AnalysisContext.model_validate_json(raw.decode("utf-8"))
    except (PydanticValidationError, UnicodeDecodeError) as exc:
        raise ContextIntegrityError(
            f"{CONTEXT_FILENAME} does not satisfy the context schema", run_id=run.run_id
        ) from exc

    if context.run_id != run.run_id:
        raise ContextIntegrityError(
            f"{CONTEXT_FILENAME} belongs to run {context.run_id}, not {run.run_id}",
            run_id=run.run_id,
            context_run_id=context.run_id,
        )
    return context, digest


# --------------------------------------------------------------------------
# response validation
# --------------------------------------------------------------------------

_MAX_REPORTED_PATHS = 10
"""How many bad paths the error names. Enough to diagnose, bounded for the manifest."""


def _validate_output(output: WriterModelOutput, context: AnalysisContext) -> WriterModelOutput:
    """Check a provider answer against *this* Run.

    Schema validity is already guaranteed by the client. What is checked here is
    whether the answer is about the right Run and substantial enough to publish -
    conditions no schema can express.
    """
    if output.run_id != context.run_id:
        raise WriterResponseError(
            "response run_id does not match this run",
            expected=context.run_id,
            actual=output.run_id,
        )

    article = output.article.strip()
    if not article:
        raise WriterResponseError("response contains an empty article")
    if len(article) < MIN_ARTICLE_CHARS:
        raise WriterResponseError(
            f"article is {len(article)} characters, minimum is {MIN_ARTICLE_CHARS}",
            article_chars=len(article),
        )
    if not output.title.strip():
        raise WriterResponseError("response contains an empty title")

    _require_resolvable_claims(output, context)

    return output


def _require_resolvable_claims(output: WriterModelOutput, context: AnalysisContext) -> None:
    """Refuse a draft whose claims cite paths that do not exist.

    The invariant this establishes: **a Run never reaches DRAFTED carrying an
    unresolvable source path.** Before this check a production Run committed
    seventeen claims of which sixteen pointed at invented fields; the failure
    surfaced two stages later as fourteen HIGH reviewer findings against an
    article that was factually fine, and a finalizer repaired something that was
    never broken. Catching it here costs one loop and localises the fault where
    it happened.

    Raised as a :class:`WriterResponseError` rather than a new class because
    that is exactly what it is - the provider returned a structurally invalid
    answer. That also gives it the retry policy the existing design already
    chose for malformed responses: bounded, five attempts, because a model is
    not deterministic and a second attempt genuinely differs. It is not a
    configuration fault; nothing about the machine needs fixing.

    Deliberately no repair. Guessing which real path a hallucinated one meant
    would substitute our arithmetic for the writer's citation and produce a
    claim nobody made.
    """
    invalid: list[str] = []
    for claim in output.source_claims:
        try:
            resolve_path(context, claim.source)
        except ClaimPathError:
            invalid.append(claim.source)

    if invalid:
        shown = ", ".join(repr(path) for path in invalid[:_MAX_REPORTED_PATHS])
        if len(invalid) > _MAX_REPORTED_PATHS:
            shown += f", and {len(invalid) - _MAX_REPORTED_PATHS} more"
        raise WriterResponseError(
            f"{len(invalid)} of {len(output.source_claims)} source_claims cite paths that "
            f"do not resolve against this context: {shown}. Source paths must be copied "
            "from the VALID SOURCE PATHS catalog.",
            invalid_paths=invalid[:_MAX_REPORTED_PATHS],
            invalid_count=len(invalid),
            claim_count=len(output.source_claims),
        )


def _merge_warnings(
    output: WriterModelOutput, context: AnalysisContext, guard: SourceGuardReport
) -> list[WriterWarning]:
    """Combine deterministic warnings with the ones the model reported.

    The deterministic ones come first and are never dropped: they were computed
    in Python and do not depend on the model having noticed anything.
    """
    warnings: list[WriterWarning] = []
    warnings.extend(build_guard_warnings(guard))

    if context.data_quality.status is not QualityStatus.OK:
        codes = ", ".join(str(w.code) for w in context.data_quality.warnings) or "none"
        warnings.append(
            WriterWarning(
                code=WarningCode.DEGRADED_INPUT_QUALITY,
                message=f"Context quality is {context.data_quality.status} ({codes}).",
            )
        )

    if output.status is WriterStatus.INSUFFICIENT_DATA:
        warnings.append(
            WriterWarning(
                code=WarningCode.SOURCE_TOO_THIN,
                message="Writer reported the inputs were too thin for a full article.",
            )
        )

    seen = {(w.code, w.message) for w in warnings}
    warnings.extend(
        warning for warning in output.warnings if (warning.code, warning.message) not in seen
    )
    return warnings


def _record_failure(run: RunDirectory, manifest: RunManifest, exc: WriterError) -> None:
    """Append a failure event to the ledger.

    The Run's *status* is left alone. A writer failure does not invalidate the
    Run's inputs, and marking it ``FAILED`` would wrongly imply it cannot be
    retried - which is exactly what this stage supports.
    """
    manifest.error = RunError(code=exc.code, message=exc.message, details=exc.details)
    manifest.record_event("writer.failed", exc.code, exc.message)
    run.save_manifest(manifest)
    logger.error(
        "run=%s stage=writer.failed status=%s message=%s", run.run_id, exc.code, exc.message
    )


__all__ = [
    "DRAFT_FILENAME",
    "MIN_ARTICLE_CHARS",
    "WRITER_ARTIFACTS",
    "WRITER_FILENAME",
    "WriterRunResult",
    "load_verified_context",
    "write_draft",
]
