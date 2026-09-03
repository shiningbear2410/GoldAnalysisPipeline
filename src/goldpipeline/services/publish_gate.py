"""The deterministic final publish gate.

The last automated boundary before anything leaves the machine. It calls no
model, opens no socket, and reads no credential - every decision here is made by
code that a reviewer can read and a test can pin down.

It answers one question: *may Round 6 publish ``claude_final.md`` automatically?*
The answer is ``APPROVED`` or ``BLOCKED``, and the gate never edits, sanitises or
retries. An article it will not approve needs a human or a new Run. Cleaning one
up and approving it would make the boundary decorative.

**Fail closed.** Anything uncertain about credentials, model-control prose, a
foreign instrument, an unsupported indicator or a suspicious price blocks. A
false block costs someone a look; a false approval publishes it.

Two failure modes, deliberately distinguished:

* A tampered-but-readable Run yields a ``BLOCKED`` **decision**. The Run is still
  identifiable, the block is meaningful, and an auditor gets a record. The
  tampered content is never used as evidence.
* A Run whose manifest cannot be parsed raises :class:`UntrustworthyRunError`.
  There is no trustworthy identity to attach a decision to, so writing one would
  be inventing provenance.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from goldpipeline.domain.errors import (
    ArtifactIntegrityError,
    PublishDecisionExistsError,
    RunNotGateableError,
    UntrustworthyRunError,
)
from goldpipeline.schemas.common import utc_now
from goldpipeline.schemas.context import AnalysisContext
from goldpipeline.schemas.finalizer import FinalizerResult, ResolutionStatus
from goldpipeline.schemas.manifest import RunManifest, RunStatus
from goldpipeline.schemas.publish import (
    BlockerCode,
    CheckId,
    CheckStatus,
    Decision,
    GateCheck,
    GateFinding,
    PublishDecision,
)
from goldpipeline.schemas.review import (
    BLOCKING_SEVERITIES,
    FindingCode,
    PrecheckFinding,
    ReviewResult,
    ReviewStatus,
    Severity,
)
from goldpipeline.schemas.writer import WriterResult
from goldpipeline.services import content_safety as safety
from goldpipeline.services.finalizer import FINAL_FILENAME, FINALIZER_FILENAME
from goldpipeline.services.finalizer_policy import MAX_EVIDENCE_PROBE_CHARS, FindingKey
from goldpipeline.services.integrity import verify_artifact
from goldpipeline.services.news_provenance import NewsProvenance
from goldpipeline.services.news_provenance import verify as verify_news_claims
from goldpipeline.services.pipeline import (
    ANALYSIS_SOURCE_FILENAME,
    CONTEXT_FILENAME,
    MARKET_SOURCE_FILENAME,
)
from goldpipeline.services.precheck import run_prechecks
from goldpipeline.services.reviewer import REVIEW_FILENAME
from goldpipeline.services.writer import DRAFT_FILENAME, WRITER_FILENAME
from goldpipeline.storage.run_store import PreparedArtifact, RunDirectory, RunStore

logger = logging.getLogger(__name__)

DECISION_FILENAME = "publish_decision.json"

GATEABLE_STATUSES = (RunStatus.FINALIZED,)
"""Only a finalized Run has an article to gate."""

REQUIRED_ARTIFACTS = (
    ANALYSIS_SOURCE_FILENAME,
    MARKET_SOURCE_FILENAME,
    CONTEXT_FILENAME,
    DRAFT_FILENAME,
    WRITER_FILENAME,
    REVIEW_FILENAME,
    FINAL_FILENAME,
    FINALIZER_FILENAME,
)
"""The whole chain. Every one is verified before a single word is read."""

_PRECHECK_TO_CHECK = {
    FindingCode.FOREIGN_SYMBOL_MENTIONED: CheckId.FOREIGN_SYMBOL,
    FindingCode.UNSUPPORTED_INDICATOR_MENTIONED: CheckId.UNSUPPORTED_INDICATOR,
    FindingCode.UNKNOWN_PRICE_LIKE_NUMBER: CheckId.SUSPICIOUS_PRICE,
    FindingCode.NUMBER_OUTSIDE_MARKET_RANGE: CheckId.SUSPICIOUS_PRICE,
    FindingCode.ABSOLUTE_RISK_LANGUAGE: CheckId.RISK_LANGUAGE,
    FindingCode.SYMBOL_NOT_MENTIONED: CheckId.CONTEXT_CONSISTENCY,
}

_GATE_ESCALATIONS = {
    FindingCode.UNKNOWN_PRICE_LIKE_NUMBER: Severity.HIGH,
}
"""Findings this gate rates more severely than the review did.

Round 3 calls a price-like number it cannot account for MEDIUM, and that is the
right call mid-pipeline: a near-miss is worth a reviewer's attention but not a
halt, and the finalizer may yet explain or remove it.

At the boundary the calculus inverts. A number in an article about to be
published that appears in neither the market data, nor the recorded claims, nor
the analyst's note is unexplained *for good* - there is no later stage to catch
it. Publishing an invented price is precisely the failure this pipeline exists
to prevent, so here it blocks.
"""

_PRECHECK_TO_BLOCKER = {
    FindingCode.FOREIGN_SYMBOL_MENTIONED: BlockerCode.FOREIGN_SYMBOL_MENTIONED,
    FindingCode.UNSUPPORTED_INDICATOR_MENTIONED: BlockerCode.UNSUPPORTED_INDICATOR_MENTIONED,
    FindingCode.UNKNOWN_PRICE_LIKE_NUMBER: BlockerCode.SUSPICIOUS_PRICE,
    FindingCode.NUMBER_OUTSIDE_MARKET_RANGE: BlockerCode.SUSPICIOUS_PRICE,
    FindingCode.ABSOLUTE_RISK_LANGUAGE: BlockerCode.ABSOLUTE_RISK_LANGUAGE,
    FindingCode.SYMBOL_NOT_MENTIONED: BlockerCode.SYMBOL_CONTRADICTS_CONTEXT,
}


@dataclass(frozen=True)
class GateResult:
    """Outcome of a gate run."""

    run_id: str
    run_dir: Path
    status: RunStatus
    decision: PublishDecision
    decision_path: Path

    @property
    def approved(self) -> bool:
        """Whether Round 6 may publish."""
        return self.decision.approved


@dataclass
class _Inputs:
    """Artifacts the gate managed to read and trust."""

    context: AnalysisContext | None = None
    writer_result: WriterResult | None = None
    review: ReviewResult | None = None
    finalization: FinalizerResult | None = None
    draft: str | None = None
    article: str | None = None
    digests: dict[str, str] = field(default_factory=dict)
    provenance: NewsProvenance | None = None
    """What was established about the article's news statements, if anything."""


def gate_publish(
    *,
    run_id: str,
    store: RunStore,
    now: datetime | None = None,
) -> GateResult:
    """Decide whether a finalized Run may be published automatically.

    Args:
        run_id: The Run to gate. Must be ``FINALIZED``.
        store: Where Runs live.
        now: Injection point for tests.

    Returns:
        A :class:`GateResult`. Both outcomes are successes of this function -
        ``BLOCKED`` is a decision, not an error.

    Raises:
        PublishDecisionExistsError: The Run already has a decision.
        RunNotGateableError: The Run is not finalized, or is missing artifacts.
        UntrustworthyRunError: The manifest cannot be parsed.
    """
    run = store.open(run_id)
    manifest = _load_manifest(run)

    _require_gateable(run, manifest)

    checks, inputs = _run_checks(run, manifest)
    decision = _build_decision(run.run_id, checks, inputs, now=now)

    return _commit(run=run, manifest=manifest, decision=decision)


# --------------------------------------------------------------------------
# preconditions
# --------------------------------------------------------------------------


def _load_manifest(run: RunDirectory) -> RunManifest:
    """Read the manifest, or refuse to reason about the Run at all."""
    try:
        return run.load_manifest()
    except (PydanticValidationError, ValueError, OSError, UnicodeDecodeError) as exc:
        raise UntrustworthyRunError(
            f"manifest for run {run.run_id} cannot be read; no trustworthy decision "
            "can be made about this Run",
            run_id=run.run_id,
        ) from exc


def _require_gateable(run: RunDirectory, manifest: RunManifest) -> None:
    """Refuse unless the Run is finalized and has no decision yet."""
    if run.has_artifact(DECISION_FILENAME):
        raise PublishDecisionExistsError(
            f"run {run.run_id} already has {DECISION_FILENAME}; publish decisions are "
            "immutable, so re-evaluating would silently replace one verdict with another",
            run_id=run.run_id,
            artifact=DECISION_FILENAME,
        )

    if manifest.status not in GATEABLE_STATUSES:
        raise RunNotGateableError(
            f"run {run.run_id} is {manifest.status}, the gate needs {RunStatus.FINALIZED}",
            run_id=run.run_id,
            status=str(manifest.status),
        )


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------


def _run_checks(run: RunDirectory, manifest: RunManifest) -> tuple[list[GateCheck], _Inputs]:
    """Run every gate check, in order.

    The integrity check comes first and, when it fails, is the only one that
    runs: every later check reads article content, and reading content the
    manifest disowns would be judging a document nobody wrote.
    """
    inputs = _Inputs()
    integrity = _check_integrity(run, manifest, inputs)
    checks = [integrity]

    if integrity.status is CheckStatus.FAIL:
        return checks, inputs

    assert inputs.context is not None
    assert inputs.review is not None
    assert inputs.finalization is not None
    assert inputs.writer_result is not None
    assert inputs.article is not None
    assert inputs.draft is not None

    checks.append(_check_run_state(manifest))
    checks.append(_check_review_verdict_state(inputs.review, inputs.finalization))
    checks.append(_check_issue_closure(inputs.review, inputs.finalization))
    checks.append(_check_correction_closure(inputs.review, inputs.article))
    checks.append(_check_structure(inputs.article))
    checks.append(_check_telegram_compatibility(inputs.article))
    checks.append(_check_instruction_text(inputs.article))
    checks.append(_check_credentials(inputs.article))

    # Judged against the *final* article, not the draft: provenance describes
    # the sentences about to be published, and the finalizer may have changed
    # them since the writer vouched for them.
    inputs.provenance = verify_news_claims(
        inputs.context, inputs.writer_result.news_claims, inputs.article
    )
    checks.append(_check_external_claims(inputs.article, inputs.provenance))

    article_findings = run_prechecks(
        context=inputs.context,
        writer_result=inputs.writer_result,
        article=inputs.article,
        check_claims=False,
    ).findings
    checks.extend(_checks_from_precheck(article_findings))

    checks.append(_check_context_consistency(inputs.context, inputs.article))
    checks.append(
        _check_no_new_regression(
            context=inputs.context,
            writer_result=inputs.writer_result,
            draft=inputs.draft,
            article_findings=article_findings,
        )
    )
    return checks, inputs


def _check_integrity(run: RunDirectory, manifest: RunManifest, inputs: _Inputs) -> GateCheck:
    """Verify the whole artifact chain before reading a word of it."""
    findings: list[GateFinding] = []

    def fail(message: str, source: str | None = None) -> GateCheck:
        findings.append(
            GateFinding(
                code=BlockerCode.ARTIFACT_INTEGRITY_FAILURE,
                severity=Severity.CRITICAL,
                message=message,
                source=source,
            )
        )
        return GateCheck(
            check_id=CheckId.ARTIFACT_CHAIN_INTEGRITY,
            status=CheckStatus.FAIL,
            description="Every artifact matches the digest recorded for it.",
            findings=findings,
        )

    verified: dict[str, str] = {}
    for name in REQUIRED_ARTIFACTS:
        try:
            artifact = verify_artifact(run, manifest, name)
        except ArtifactIntegrityError as exc:
            # Deliberately generic: the message is written here, and the
            # tampered bytes are never quoted back into the decision.
            return fail(exc.message, source=name)
        verified[name] = artifact.sha256

    inputs.digests = verified

    try:
        inputs.context = AnalysisContext.model_validate_json(
            run.read_artifact_bytes(CONTEXT_FILENAME).decode("utf-8")
        )
        inputs.writer_result = WriterResult.model_validate_json(
            run.read_artifact_bytes(WRITER_FILENAME).decode("utf-8")
        )
        inputs.review = ReviewResult.model_validate_json(
            run.read_artifact_bytes(REVIEW_FILENAME).decode("utf-8")
        )
        inputs.finalization = FinalizerResult.model_validate_json(
            run.read_artifact_bytes(FINALIZER_FILENAME).decode("utf-8")
        )
        inputs.draft = run.read_artifact_bytes(DRAFT_FILENAME).decode("utf-8")
        inputs.article = run.read_artifact_bytes(FINAL_FILENAME).decode("utf-8")
    except (PydanticValidationError, UnicodeDecodeError) as exc:
        return fail(f"an artifact does not satisfy its schema: {type(exc).__name__}")

    for label, value in (
        (CONTEXT_FILENAME, inputs.context.run_id),
        (WRITER_FILENAME, inputs.writer_result.run_id),
        (REVIEW_FILENAME, inputs.review.run_id),
        (FINALIZER_FILENAME, inputs.finalization.run_id),
    ):
        if value != run.run_id:
            return fail(f"{label} belongs to a different run", source=label)

    cross_checks = (
        ("writer_metadata.article_sha256", inputs.writer_result.article_sha256, DRAFT_FILENAME),
        ("writer_metadata.context_sha256", inputs.writer_result.context_sha256, CONTEXT_FILENAME),
        ("review.draft_sha256", inputs.review.draft_sha256, DRAFT_FILENAME),
        ("review.context_sha256", inputs.review.context_sha256, CONTEXT_FILENAME),
        ("review.writer_metadata_sha256", inputs.review.writer_metadata_sha256, WRITER_FILENAME),
        ("finalizer.context_sha256", inputs.finalization.context_sha256, CONTEXT_FILENAME),
        (
            "finalizer.original_draft_sha256",
            inputs.finalization.original_draft_sha256,
            DRAFT_FILENAME,
        ),
        ("finalizer.review_sha256", inputs.finalization.review_sha256, REVIEW_FILENAME),
        (
            "finalizer.writer_metadata_sha256",
            inputs.finalization.writer_metadata_sha256,
            WRITER_FILENAME,
        ),
        (
            "finalizer.final_article_sha256",
            inputs.finalization.final_article_sha256,
            FINAL_FILENAME,
        ),
    )
    for label, recorded, target in cross_checks:
        if recorded != verified[target]:
            return fail(f"{label} does not match {target}", source=target)

    return GateCheck(
        check_id=CheckId.ARTIFACT_CHAIN_INTEGRITY,
        status=CheckStatus.PASS,
        description=f"All {len(REQUIRED_ARTIFACTS)} artifacts verified against the manifest.",
    )


def _check_run_state(manifest: RunManifest) -> GateCheck:
    """The manifest must say the Run is finalized."""
    ok = manifest.status is RunStatus.FINALIZED
    return GateCheck(
        check_id=CheckId.RUN_STATE,
        status=CheckStatus.PASS if ok else CheckStatus.FAIL,
        description="The Run reached FINALIZED.",
        findings=[]
        if ok
        else [
            GateFinding(
                code=BlockerCode.RUN_NOT_FINALIZED,
                severity=Severity.CRITICAL,
                message=f"Run status is {manifest.status}, not FINALIZED.",
            )
        ],
    )


def _check_review_verdict_state(review: ReviewResult, finalization: FinalizerResult) -> GateCheck:
    """A rejected review can never have produced a finalized article.

    Reaching this state means an earlier guard was bypassed or an artifact was
    assembled by hand. Either way the Run is not what it claims to be.
    """
    findings: list[GateFinding] = []

    if review.status is ReviewStatus.REJECT:
        findings.append(
            GateFinding(
                code=BlockerCode.IMPOSSIBLE_REVIEW_STATE,
                severity=Severity.CRITICAL,
                message=(
                    "The review verdict is REJECT, yet the Run is finalized. The finalizer "
                    "refuses rejected articles, so this state should be unreachable."
                ),
                source=REVIEW_FILENAME,
            )
        )

    if finalization.review_status is not review.status:
        findings.append(
            GateFinding(
                code=BlockerCode.IMPOSSIBLE_REVIEW_STATE,
                severity=Severity.CRITICAL,
                message=(
                    f"The finalization records review status {finalization.review_status}, "
                    f"but the review says {review.status}."
                ),
                source=FINALIZER_FILENAME,
            )
        )

    return GateCheck(
        check_id=CheckId.REVIEW_VERDICT_STATE,
        status=CheckStatus.FAIL if findings else CheckStatus.PASS,
        description="The review verdict and the finalization agree, and it is not REJECT.",
        findings=findings,
    )


def _check_issue_closure(review: ReviewResult, finalization: FinalizerResult) -> GateCheck:
    """Every review issue was answered, and the severe ones were fixed.

    Round 4 enforces this on its own response; checking it again here is
    defence in depth. If that check ever regresses, or an artifact was written
    by an older version, this still holds the line.
    """
    findings: list[GateFinding] = []
    resolved = {item.issue_id: item for item in finalization.issue_resolutions}

    for issue in review.issues:
        resolution = resolved.get(issue.issue_id)
        if resolution is None:
            findings.append(
                GateFinding(
                    code=BlockerCode.UNRESOLVED_REVIEW_ISSUE,
                    severity=Severity.CRITICAL,
                    message=f"Review issue {issue.issue_id} has no resolution.",
                    source=FINALIZER_FILENAME,
                )
            )
            continue

        if (
            issue.severity in BLOCKING_SEVERITIES
            and resolution.resolution is not ResolutionStatus.APPLIED
        ):
            findings.append(
                GateFinding(
                    code=BlockerCode.UNRESOLVED_REVIEW_ISSUE,
                    severity=Severity.CRITICAL,
                    message=(
                        f"Review issue {issue.issue_id} is {issue.severity} but was "
                        f"resolved as {resolution.resolution}."
                    ),
                    source=FINALIZER_FILENAME,
                )
            )

    return GateCheck(
        check_id=CheckId.REVIEW_ISSUE_CLOSURE,
        status=CheckStatus.FAIL if findings else CheckStatus.PASS,
        description="Every review issue is accounted for; severe ones were applied.",
        findings=findings,
    )


def _check_correction_closure(review: ReviewResult, article: str) -> GateCheck:
    """A value the review called wrong must be gone from the published text.

    Metadata claiming ``APPLIED`` is a statement about the world; this is the
    check on it. If the wrong number is still there, the fix did not happen
    however confidently it was reported.
    """
    findings: list[GateFinding] = []

    for issue in review.issues:
        evidence = issue.evidence
        if evidence is None or not evidence.actual:
            continue
        probe = evidence.actual.strip()
        if not probe or len(probe) > MAX_EVIDENCE_PROBE_CHARS:
            continue
        if probe in article:
            findings.append(
                GateFinding(
                    code=BlockerCode.CORRECTION_NOT_APPLIED,
                    severity=Severity.HIGH,
                    message=(
                        f"Issue {issue.issue_id} flagged {probe!r} as wrong, and it is still "
                        "in the final article."
                    ),
                    evidence=probe,
                    source=evidence.source_path,
                    position=article.find(probe),
                )
            )

    return GateCheck(
        check_id=CheckId.CORRECTION_CLOSURE,
        status=CheckStatus.FAIL if findings else CheckStatus.PASS,
        description="No value the review called wrong survives in the final article.",
        findings=findings,
    )


def _check_structure(article: str) -> GateCheck:
    """Basic sanity: is this prose at all, and of a publishable size?"""
    findings: list[GateFinding] = []
    stripped = article.strip()
    length = len(stripped)

    if not stripped:
        findings.append(
            GateFinding(
                code=BlockerCode.ARTICLE_EMPTY,
                severity=Severity.CRITICAL,
                message="The final article is empty or whitespace only.",
                source=FINAL_FILENAME,
            )
        )
    elif length < safety.MIN_ARTICLE_CHARS:
        findings.append(
            GateFinding(
                code=BlockerCode.ARTICLE_TOO_SHORT,
                severity=Severity.HIGH,
                message=(
                    f"The final article is {length} characters; the minimum is "
                    f"{safety.MIN_ARTICLE_CHARS}."
                ),
                source=FINAL_FILENAME,
            )
        )
    elif length > safety.MAX_ARTICLE_CHARS:
        findings.append(
            GateFinding(
                code=BlockerCode.ARTICLE_TOO_LONG,
                severity=Severity.HIGH,
                message=(
                    f"The final article is {length} characters; the maximum is "
                    f"{safety.MAX_ARTICLE_CHARS}."
                ),
                source=FINAL_FILENAME,
            )
        )

    if safety.looks_like_json(stripped):
        findings.append(
            GateFinding(
                code=BlockerCode.ARTICLE_LOOKS_LIKE_JSON,
                severity=Severity.CRITICAL,
                message="The final article is a serialized object, not prose.",
                source=FINAL_FILENAME,
            )
        )

    findings.extend(
        GateFinding(
            code=BlockerCode.ARTICLE_CONTAINS_TRACEBACK,
            severity=Severity.CRITICAL,
            message=f"The final article contains error output ({match.label!r}).",
            evidence=match.matched,
            source=FINAL_FILENAME,
            position=match.position,
        )
        for match in safety.find_tracebacks(article)
    )

    findings.extend(
        GateFinding(
            code=BlockerCode.ARTICLE_CONTAINS_CODE_BLOCK,
            severity=Severity.HIGH,
            message="The final article contains a fenced code block.",
            source=FINAL_FILENAME,
            position=match.position,
        )
        for match in safety.find_code_blocks(article)[:3]
    )

    return GateCheck(
        check_id=CheckId.ARTICLE_STRUCTURE,
        status=CheckStatus.FAIL if findings else CheckStatus.PASS,
        description="The article is prose of a publishable length, with no dumps.",
        findings=findings,
    )


def _check_telegram_compatibility(article: str) -> GateCheck:
    """Can Round 6 send this as plain text without surprises?

    Deliberately shallow. The publisher will send plain text, so there is no
    markup to parse here - what matters is that the bytes are clean UTF-8 and
    that someone downstream knows whether the piece needs splitting.
    """
    findings: list[GateFinding] = []

    try:
        article.encode("utf-8")
    except UnicodeEncodeError:
        findings.append(
            GateFinding(
                code=BlockerCode.ARTICLE_NOT_UTF8,
                severity=Severity.CRITICAL,
                message="The final article is not encodable as UTF-8.",
                source=FINAL_FILENAME,
            )
        )

    control = safety.find_control_characters(article)
    findings.extend(
        GateFinding(
            code=BlockerCode.ARTICLE_CONTROL_CHARACTERS,
            severity=Severity.CRITICAL,
            message=f"The final article contains a control character ({match.label}).",
            source=FINAL_FILENAME,
            position=match.position,
        )
        for match in control[:5]
    )

    if len(article) > safety.TELEGRAM_MESSAGE_LIMIT and not findings:
        findings.append(
            GateFinding(
                code=BlockerCode.UNEXPECTED_ARTIFACT,
                severity=Severity.LOW,
                message=(
                    f"The article is {len(article)} characters, above Telegram's "
                    f"{safety.TELEGRAM_MESSAGE_LIMIT}-character message limit; the "
                    "publisher will need to split it."
                ),
                source=FINAL_FILENAME,
            )
        )

    blocking = [finding for finding in findings if finding.is_blocking]
    return GateCheck(
        check_id=CheckId.TELEGRAM_COMPATIBILITY,
        status=CheckStatus.FAIL
        if blocking
        else (CheckStatus.WARN if findings else CheckStatus.PASS),
        description="Clean UTF-8, no control characters, length noted for the publisher.",
        findings=findings,
    )


def _check_instruction_text(article: str) -> GateCheck:
    """No published sentence may read as an attempt to steer a model.

    This is the gap Round 4 left open by design: a finalizer making the minimum
    necessary edit leaves such a sentence alone unless a review issue named it.
    """
    findings = [
        GateFinding(
            code=BlockerCode.INSTRUCTION_SHAPED_TEXT,
            severity=Severity.CRITICAL,
            message=f"The final article contains model-control text ({match.label!r}).",
            evidence=match.matched,
            source=FINAL_FILENAME,
            position=match.position,
        )
        for match in safety.find_instruction_text(article)
    ]
    return GateCheck(
        check_id=CheckId.INSTRUCTION_SHAPED_TEXT,
        status=CheckStatus.FAIL if findings else CheckStatus.PASS,
        description="The article contains no instruction-shaped or model-control prose.",
        findings=findings,
    )


def _check_credentials(article: str) -> GateCheck:
    """No credential-shaped value may be published.

    The finding carries a redaction, never the token: a decision artifact is
    read, logged and shared, and it must not become a second copy of a secret.
    """
    findings = [
        GateFinding(
            code=BlockerCode.POSSIBLE_CREDENTIAL_EXPOSURE,
            severity=Severity.CRITICAL,
            message=f"The final article contains a value shaped like a {match.label}.",
            evidence=match.matched,
            source=FINAL_FILENAME,
            position=match.position,
        )
        for match in safety.find_credentials(article)
    ]
    return GateCheck(
        check_id=CheckId.CREDENTIAL_EXPOSURE,
        status=CheckStatus.FAIL if findings else CheckStatus.PASS,
        description="The article contains no credential-shaped values.",
        findings=findings,
    )


def _check_external_claims(article: str, provenance: NewsProvenance) -> GateCheck:
    """Every asserted economic event must be sourced, or the article does not go.

    The rule has not been relaxed - it has been given a way to be satisfied. An
    article whose Run carries no producer brief has no verified statements, every
    occurrence is uncovered, and this behaves exactly as it did before news
    provenance existed. An article that faithfully quotes a collected item, cites
    it by id, and still says so in the final text is covered *at that sentence*.

    Coverage is per occurrence, and the finding is reported at the first one that
    is not covered. Sourcing the first mention of the Fed does not license the
    second: an article that relays one real item and invents a second event is
    still an article that invents an event.
    """
    findings: list[GateFinding] = []
    seen: set[str] = set()

    for occurrence in safety.external_claim_occurrences(article):
        if occurrence.entity in seen or provenance.covers(occurrence.start, occurrence.end):
            continue
        seen.add(occurrence.entity)
        findings.append(
            GateFinding(
                code=BlockerCode.EXTERNAL_FACT_WITHOUT_SOURCE,
                severity=Severity.HIGH,
                message=(
                    f"The final article asserts an economic event ({occurrence.label}) with "
                    "no verified news source."
                ),
                evidence=safety.excerpt(article, occurrence.start, occurrence.end),
                source=FINAL_FILENAME,
                position=occurrence.start,
            )
        )

    return GateCheck(
        check_id=CheckId.EXTERNAL_FACT_WITHOUT_SOURCE,
        status=CheckStatus.FAIL if findings else CheckStatus.PASS,
        description="Every asserted news event is supported by a cited producer-brief item.",
        findings=findings,
    )


def _checks_from_precheck(findings: list[PrecheckFinding]) -> list[GateCheck]:
    """Fold the shared deterministic scanners into gate checks.

    Reuses Round 3's scanners rather than restating them: one definition of
    "foreign symbol" or "suspicious price" for the whole pipeline, so the gate
    cannot come to disagree with the review about what those words mean.
    """
    grouped: dict[CheckId, list[GateFinding]] = {
        CheckId.FOREIGN_SYMBOL: [],
        CheckId.UNSUPPORTED_INDICATOR: [],
        CheckId.SUSPICIOUS_PRICE: [],
        CheckId.RISK_LANGUAGE: [],
    }

    for finding in findings:
        check_id = _PRECHECK_TO_CHECK.get(finding.code)
        if check_id is None or check_id not in grouped:
            continue
        grouped[check_id].append(
            GateFinding(
                code=_PRECHECK_TO_BLOCKER[finding.code],
                severity=_GATE_ESCALATIONS.get(finding.code, finding.severity),
                message=finding.message,
                evidence=finding.excerpt or finding.actual,
                source=finding.source_path or FINAL_FILENAME,
            )
        )

    descriptions = {
        CheckId.FOREIGN_SYMBOL: "The article names no instrument other than the Run's.",
        CheckId.UNSUPPORTED_INDICATOR: "The article states no indicator the context lacks.",
        CheckId.SUSPICIOUS_PRICE: "Every price-like number is accounted for by the data.",
        CheckId.RISK_LANGUAGE: "The article makes no absolute or guaranteed claim.",
    }

    checks: list[GateCheck] = []
    for check_id, found in grouped.items():
        blocking = [finding for finding in found if finding.is_blocking]
        checks.append(
            GateCheck(
                check_id=check_id,
                status=CheckStatus.FAIL
                if blocking
                else (CheckStatus.WARN if found else CheckStatus.PASS),
                description=descriptions[check_id],
                findings=found,
            )
        )
    return checks


def _check_context_consistency(context: AnalysisContext, article: str) -> GateCheck:
    """The article must not name a timeframe other than the Run's.

    The other two halves of context consistency are covered elsewhere and not
    duplicated here: a foreign *instrument* is caught by ``FOREIGN_SYMBOL``, and
    a contradicted *price* by ``SUSPICIOUS_PRICE``, both through the shared
    scanners. What is left is the timeframe, which nothing else looks at.

    Omission is fine - an article need not name any of them. What is refused is
    an explicit statement that disagrees with the source of truth.
    """
    findings: list[GateFinding] = []
    folded = article.casefold()

    for timeframe in ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"):
        if timeframe == str(context.market.timeframe):
            continue
        if re.search(rf"\b{timeframe.casefold()}\b", folded):
            findings.append(
                GateFinding(
                    code=BlockerCode.TIMEFRAME_CONTRADICTS_CONTEXT,
                    severity=Severity.HIGH,
                    message=(
                        f"The article names timeframe {timeframe}, but this Run is "
                        f"{context.market.timeframe}."
                    ),
                    evidence=timeframe,
                    source="context.market.timeframe",
                )
            )

    return GateCheck(
        check_id=CheckId.CONTEXT_CONSISTENCY,
        status=CheckStatus.FAIL if findings else CheckStatus.PASS,
        description=f"The article names no timeframe other than {context.market.timeframe}.",
        findings=findings,
    )


def _check_no_new_regression(
    *,
    context: AnalysisContext,
    writer_result: WriterResult,
    draft: str,
    article_findings: list[PrecheckFinding],
) -> GateCheck:
    """The final article must be no worse than the draft it came from.

    Round 4 checks this too. Doing it again independently is the point: if that
    check regresses, or an artifact was produced by an older build, a new
    HIGH-severity problem still cannot reach publication.
    """
    baseline = run_prechecks(
        context=context, writer_result=writer_result, article=draft, check_claims=False
    )
    before = {FindingKey.of(finding) for finding in baseline.findings}

    findings = [
        GateFinding(
            code=BlockerCode.NEW_REGRESSION_SINCE_DRAFT,
            severity=finding.severity,
            message=(
                f"The final article has a problem the draft did not: {finding.code} "
                f"({finding.actual!r})."
            ),
            evidence=finding.excerpt or finding.actual,
            source=FINAL_FILENAME,
        )
        for finding in article_findings
        if finding.severity in BLOCKING_SEVERITIES and FindingKey.of(finding) not in before
    ]

    return GateCheck(
        check_id=CheckId.NO_NEW_REGRESSION,
        status=CheckStatus.FAIL if findings else CheckStatus.PASS,
        description="The final article introduces no severe problem the draft lacked.",
        findings=findings,
    )


# --------------------------------------------------------------------------
# decision and commit
# --------------------------------------------------------------------------


def _build_decision(
    run_id: str, checks: list[GateCheck], inputs: _Inputs, *, now: datetime | None
) -> PublishDecision:
    """Fold the checks into one verdict.

    The policy is a single sentence: any blocking finding blocks. Warnings are
    recorded and do not.
    """
    blockers = [finding for check in checks for finding in check.blocking_findings]
    warnings = [
        finding for check in checks for finding in check.findings if not finding.is_blocking
    ]

    digests = inputs.digests
    return PublishDecision(
        run_id=run_id,
        decision=Decision.BLOCKED if blockers else Decision.APPROVED,
        created_at=now or utc_now(),
        checks=checks,
        blockers=blockers,
        warnings=warnings,
        review_status=inputs.review.status if inputs.review else None,
        finalization_mode=inputs.finalization.finalization_mode if inputs.finalization else None,
        article_chars=len(inputs.article.strip()) if inputs.article else None,
        context_sha256=digests.get(CONTEXT_FILENAME),
        draft_sha256=digests.get(DRAFT_FILENAME),
        writer_metadata_sha256=digests.get(WRITER_FILENAME),
        review_sha256=digests.get(REVIEW_FILENAME),
        final_article_sha256=digests.get(FINAL_FILENAME),
        finalizer_metadata_sha256=digests.get(FINALIZER_FILENAME),
        news_provenance=inputs.provenance.report() if inputs.provenance else None,
    )


def _commit(*, run: RunDirectory, manifest: RunManifest, decision: PublishDecision) -> GateResult:
    """Write the decision, then move the Run to match it.

    The artifact lands first. A manifest saying ``READY_TO_PUBLISH`` with no
    decision beside it would be an approval nobody can audit.
    """
    artifact = PreparedArtifact.from_json(DECISION_FILENAME, decision)
    run.commit_artifacts([artifact], manifest)

    status = RunStatus.READY_TO_PUBLISH if decision.approved else RunStatus.PUBLISH_BLOCKED
    passed, warned, failed = decision.counts

    manifest.status = status
    manifest.record_event(
        "publish_gate",
        str(decision.decision),
        f"{passed} passed, {warned} warnings, {failed} failed; {len(decision.blockers)} blocker(s)",
    )
    run.save_manifest(manifest)

    logger.info(
        "run=%s stage=publish_gate status=%s checks=%d/%d/%d blockers=%d",
        run.run_id,
        decision.decision,
        passed,
        warned,
        failed,
        len(decision.blockers),
    )

    return GateResult(
        run_id=run.run_id,
        run_dir=run.path,
        status=status,
        decision=decision,
        decision_path=run.artifact_path(DECISION_FILENAME),
    )


__all__ = [
    "DECISION_FILENAME",
    "GATEABLE_STATUSES",
    "REQUIRED_ARTIFACTS",
    "GateResult",
    "gate_publish",
]
