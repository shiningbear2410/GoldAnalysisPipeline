"""Deciding how far to trust a reviewer's verdict.

Two different kinds of disagreement, handled differently on purpose.

**The response contradicts itself.** A PASS that lists a HIGH issue, a
NEEDS_REVISION with nothing to revise, a REJECT with no reason. These are not
lenient verdicts - they are broken ones, and a broken answer is not evidence
about the article. The response is rejected and no artifact is written.

**The response is more generous than the deterministic evidence allows.** The
prechecks found a wrong claim or a foreign instrument, and the reviewer still
said PASS. Here the pipeline knows something the verdict does not, so the
verdict is escalated and the escalation is recorded on the artifact
(``verdict_source: POLICY_ESCALATED``, plus a note saying why).

Escalating rather than rejecting is deliberate: the findings are real and Round
4 needs them. Throwing away a complete review because the reviewer was too kind
would discard the evidence along with the verdict. Nothing is silent - a reader
of ``gpt_review.json`` can always see that the reviewer and the pipeline
disagreed, and by how much.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from goldpipeline.domain.errors import ReviewResponseError
from goldpipeline.schemas.review import (
    FACTUAL_CATEGORIES,
    PASS_MIN_SCORE,
    IssueCategory,
    PrecheckFinding,
    ReviewIssue,
    ReviewModelOutput,
    ReviewStatus,
    Severity,
    VerdictSource,
)
from goldpipeline.services.precheck import PrecheckReport

_STATUS_RANK = {
    ReviewStatus.PASS: 0,
    ReviewStatus.NEEDS_REVISION: 1,
    ReviewStatus.REJECT: 2,
}

PRECHECK_ISSUE_PREFIX = "precheck"
"""Issue ids minted from deterministic findings carry this prefix."""


@dataclass
class PolicyOutcome:
    """The verdict after policy has been applied."""

    status: ReviewStatus
    model_status: ReviewStatus
    verdict_source: VerdictSource
    issues: list[ReviewIssue]
    notes: list[str] = field(default_factory=list)


def validate_response(output: ReviewModelOutput, *, run_id: str) -> None:
    """Reject a response that contradicts itself or belongs to another Run.

    Raises:
        ReviewResponseError: If the answer cannot be trusted as a review.
    """
    if output.run_id != run_id:
        raise ReviewResponseError(
            "response run_id does not match the run under review",
            expected=run_id,
            actual=output.run_id,
        )

    blocking = output.blocking_issues
    if output.status is ReviewStatus.PASS and blocking:
        raise ReviewResponseError(
            "response is PASS but lists issues that rule out a pass",
            severities=sorted({str(issue.severity) for issue in blocking}),
            issue_ids=[issue.issue_id for issue in blocking],
        )

    if output.status is ReviewStatus.PASS and output.revision_instructions:
        raise ReviewResponseError(
            "response is PASS but asks for revisions; a passing article needs no edits",
            instruction_count=len(output.revision_instructions),
        )

    if output.status is ReviewStatus.PASS and output.score < PASS_MIN_SCORE:
        raise ReviewResponseError(
            f"response is PASS but scores {output.score}, below the {PASS_MIN_SCORE} threshold",
            score=output.score,
            minimum=PASS_MIN_SCORE,
        )

    if output.status is not ReviewStatus.PASS and not output.issues:
        raise ReviewResponseError(
            f"response is {output.status} but lists no issues to justify it",
            status=str(output.status),
        )

    missing_evidence = [
        issue.issue_id
        for issue in output.issues
        if issue.category in FACTUAL_CATEGORIES
        and issue.severity in {Severity.HIGH, Severity.CRITICAL}
        and issue.evidence is None
    ]
    if missing_evidence:
        raise ReviewResponseError(
            "factual issues must cite evidence from the context, not just assert a problem",
            issue_ids=missing_evidence,
        )


def apply_policy(output: ReviewModelOutput, report: PrecheckReport) -> PolicyOutcome:
    """Merge deterministic findings in and escalate the verdict if they demand it.

    Args:
        output: The model's response, already validated as self-consistent.
        report: What the deterministic pass established.
    """
    issues = list(output.issues)
    notes: list[str] = []

    merged = _findings_as_issues(report, existing=issues)
    if merged:
        issues.extend(merged)
        notes.append(
            f"{len(merged)} deterministic finding(s) were added as issues; the reviewer "
            "did not report them."
        )

    required = _status_required_by(report)
    status = output.status
    source = VerdictSource.MODEL

    if _STATUS_RANK[required] > _STATUS_RANK[status]:
        notes.append(
            f"Verdict escalated from {status} to {required}: deterministic checks found "
            f"{report.worst_severity} evidence the review did not account for."
        )
        status = required
        source = VerdictSource.POLICY_ESCALATED

    return PolicyOutcome(
        status=status,
        model_status=output.status,
        verdict_source=source,
        issues=issues,
        notes=notes,
    )


def _status_required_by(report: PrecheckReport) -> ReviewStatus:
    """The mildest verdict the deterministic findings permit.

    A CRITICAL finding - a foreign instrument, say - means the article is about
    the wrong thing and editing cannot rescue it. A HIGH finding is usually a
    number that the finalizer can correct.
    """
    severities = {finding.severity for finding in report.findings}
    if Severity.CRITICAL in severities:
        return ReviewStatus.REJECT
    if Severity.HIGH in severities:
        return ReviewStatus.NEEDS_REVISION
    return ReviewStatus.PASS


def _findings_as_issues(
    report: PrecheckReport, *, existing: list[ReviewIssue]
) -> list[ReviewIssue]:
    """Turn blocking findings the reviewer missed into issues.

    Only blocking ones: a LOW or MEDIUM finding is already visible in
    ``deterministic_findings`` on the artifact, and promoting every heuristic
    warning into a formal issue would bury the real ones.
    """
    already_cited = {
        (issue.evidence.source_path, issue.evidence.actual)
        for issue in existing
        if issue.evidence is not None
    }
    already_mentioned = " ".join(issue.message for issue in existing).casefold()

    issues: list[ReviewIssue] = []
    for index, finding in enumerate(report.blocking, start=1):
        if (finding.source_path, finding.actual) in already_cited:
            continue
        if finding.actual and finding.actual.casefold() in already_mentioned:
            continue
        issues.append(
            ReviewIssue(
                issue_id=f"{PRECHECK_ISSUE_PREFIX}-{index}-{finding.code.lower()}",
                category=_category_for(finding),
                severity=finding.severity,
                message=finding.message,
                article_excerpt=finding.excerpt,
                evidence=None,
                suggested_fix=None,
            )
        )
    return issues


def _category_for(finding: PrecheckFinding) -> IssueCategory:
    """Map a deterministic finding onto the shared issue taxonomy."""
    from goldpipeline.schemas.review import FindingCode

    mapping = {
        FindingCode.CLAIM_VALUE_MISMATCH: IssueCategory.DATA_MISMATCH,
        FindingCode.CLAIM_SOURCE_NOT_FOUND: IssueCategory.UNSUPPORTED_CLAIM,
        FindingCode.UNKNOWN_PRICE_LIKE_NUMBER: IssueCategory.UNSUPPORTED_CLAIM,
        FindingCode.NUMBER_OUTSIDE_MARKET_RANGE: IssueCategory.DATA_MISMATCH,
        FindingCode.FOREIGN_SYMBOL_MENTIONED: IssueCategory.DATA_MISMATCH,
        FindingCode.SYMBOL_NOT_MENTIONED: IssueCategory.STYLE,
        FindingCode.UNSUPPORTED_INDICATOR_MENTIONED: IssueCategory.UNSUPPORTED_CLAIM,
        FindingCode.ABSOLUTE_RISK_LANGUAGE: IssueCategory.RISK_LANGUAGE,
        FindingCode.NO_SOURCE_CLAIMS: IssueCategory.OTHER,
    }
    return mapping.get(finding.code, IssueCategory.OTHER)


__all__ = [
    "PRECHECK_ISSUE_PREFIX",
    "PolicyOutcome",
    "apply_policy",
    "validate_response",
]
