"""What the pipeline demands of a revision before it will keep it.

Three gates, applied in order, and each rejects rather than repairs.

**1. Every issue is answered.** No missing resolution, no duplicate, no id the
review never raised. A revision that quietly skips an issue is indistinguishable
from one that fixed it, and this is what makes the difference visible.

**2. Severe issues are actually fixed.** A HIGH or CRITICAL issue must be
``APPLIED``. Letting a model mark a wrong price "not applicable" would hand it
the one escape hatch that makes the whole review chain decorative.

**3. The article got better, not different.** The deterministic checks are re-run
on the revised text and compared against the original. A finding the original
did not have is a regression; a severe finding that survived is a fix that did
not happen. Either way the revision is refused and nothing is written.

Gate 3 is the one that earns its keep. A model asked to remove an invented RSI
reading will comply and, in the same breath, add an EMA200 - fluently, and
undetectably to anything that only reads the resolutions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from goldpipeline.domain.errors import FinalizePostcheckError, FinalizeResponseError
from goldpipeline.schemas.finalizer import (
    MANDATORY_SEVERITIES,
    MIN_ARTICLE_CHARS,
    FinalizerModelOutput,
    IssueResolution,
    ResolutionStatus,
)
from goldpipeline.schemas.review import (
    BLOCKING_SEVERITIES,
    PrecheckFinding,
    ReviewIssue,
    ReviewResult,
)
from goldpipeline.services.precheck import PrecheckReport

MAX_EVIDENCE_PROBE_CHARS = 64
"""Longest ``evidence.actual`` worth searching for literally in the revision.

A short one - ``3325.20``, ``BTCUSD``, ``RSI`` - is a token whose continued
presence proves the correction was not made. A long one is a paraphrase, and
looking for it verbatim would produce noise rather than evidence.
"""


@dataclass(frozen=True)
class FindingKey:
    """Identity of a deterministic finding, for comparing two articles.

    Keyed on the code and the offending value rather than the message, so
    rewording a finding does not make it look like a different one.
    """

    code: str
    actual: str | None

    @classmethod
    def of(cls, finding: PrecheckFinding) -> FindingKey:
        return cls(code=str(finding.code), actual=finding.actual)


@dataclass
class PostcheckOutcome:
    """The result of comparing the revision against the original."""

    findings: list[PrecheckFinding] = field(default_factory=list)
    introduced: list[PrecheckFinding] = field(default_factory=list)
    persisted: list[PrecheckFinding] = field(default_factory=list)
    unapplied_corrections: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the revision is safe to keep."""
        return not (self.introduced or self.persisted or self.unapplied_corrections)


def validate_resolutions(
    output: FinalizerModelOutput, review: ReviewResult, *, run_id: str
) -> None:
    """Check the response is a complete, honest account of this Run's review.

    Raises:
        FinalizeResponseError: If it is not.
    """
    if output.run_id != run_id:
        raise FinalizeResponseError(
            "response run_id does not match the run being finalized",
            expected=run_id,
            actual=output.run_id,
        )

    article = output.article.strip()
    if not article:
        raise FinalizeResponseError("response contains an empty article")
    if len(article) < MIN_ARTICLE_CHARS:
        raise FinalizeResponseError(
            f"final article is {len(article)} characters, minimum is {MIN_ARTICLE_CHARS}",
            article_chars=len(article),
        )

    expected = {issue.issue_id: issue for issue in review.issues}
    resolved = {item.issue_id: item for item in output.issue_resolutions}

    missing = sorted(expected.keys() - resolved.keys())
    if missing:
        raise FinalizeResponseError(
            "every review issue must be accounted for; these were not",
            missing_issue_ids=missing,
        )

    unknown = sorted(resolved.keys() - expected.keys())
    if unknown:
        raise FinalizeResponseError(
            "response resolves issues the review never raised",
            unknown_issue_ids=unknown,
        )

    _require_mandatory_fixes(expected, output.issue_resolutions)


def _require_mandatory_fixes(
    issues: dict[str, ReviewIssue], resolutions: list[IssueResolution]
) -> None:
    """Refuse a response that declines a HIGH or CRITICAL issue."""
    declined = [
        {
            "issue_id": item.issue_id,
            "severity": str(issues[item.issue_id].severity),
            "category": str(issues[item.issue_id].category),
            "resolution": str(item.resolution),
        }
        for item in resolutions
        if issues[item.issue_id].severity in MANDATORY_SEVERITIES
        and item.resolution is not ResolutionStatus.APPLIED
    ]
    if declined:
        raise FinalizeResponseError(
            "HIGH and CRITICAL issues must be fixed, not declined",
            declined=declined,
        )


def compare_findings(
    *,
    original: PrecheckReport,
    revised: PrecheckReport,
    review: ReviewResult,
    final_article: str,
) -> PostcheckOutcome:
    """Compare the revision against the original it was meant to improve.

    Args:
        original: Deterministic findings on the draft.
        revised: Deterministic findings on the final article.
        review: The verdict being acted on, for its evidence.
        final_article: The revised text, searched for corrections that were
            claimed but not made.
    """
    before = {FindingKey.of(finding) for finding in original.findings}

    introduced = [
        finding
        for finding in revised.findings
        if finding.severity in BLOCKING_SEVERITIES and FindingKey.of(finding) not in before
    ]
    persisted = [
        finding
        for finding in revised.findings
        if finding.severity in BLOCKING_SEVERITIES and FindingKey.of(finding) in before
    ]

    return PostcheckOutcome(
        findings=list(revised.findings),
        introduced=introduced,
        persisted=persisted,
        unapplied_corrections=_unapplied_corrections(review, final_article),
    )


def _unapplied_corrections(review: ReviewResult, final_article: str) -> list[str]:
    """Find factual corrections the response claimed but the text does not show.

    When an issue cites ``evidence.actual`` - the wrong value as it appeared in
    the draft - and the finalizer says it applied the fix, that value should be
    gone. If it is still there verbatim, the claim is false and the article
    still carries the error.

    Only short, token-like values are checked; see
    :data:`MAX_EVIDENCE_PROBE_CHARS`.
    """
    unapplied: list[str] = []
    for issue in review.issues:
        evidence = issue.evidence
        if evidence is None or not evidence.actual:
            continue
        probe = evidence.actual.strip()
        if not probe or len(probe) > MAX_EVIDENCE_PROBE_CHARS:
            continue
        if probe in final_article:
            unapplied.append(issue.issue_id)
    return unapplied


def require_clean_postcheck(outcome: PostcheckOutcome) -> None:
    """Raise unless the revision improved the article without breaking it.

    Raises:
        FinalizePostcheckError: On a regression, a surviving flaw, or a
            correction claimed but not visible in the text.
    """
    if outcome.ok:
        return

    details: dict[str, object] = {}
    problems: list[str] = []

    if outcome.introduced:
        details["introduced"] = [
            {"code": str(f.code), "severity": str(f.severity), "actual": f.actual}
            for f in outcome.introduced
        ]
        problems.append(f"{len(outcome.introduced)} new problem(s) the original draft did not have")

    if outcome.persisted:
        details["persisted"] = [
            {"code": str(f.code), "severity": str(f.severity), "actual": f.actual}
            for f in outcome.persisted
        ]
        problems.append(f"{len(outcome.persisted)} flagged problem(s) still present")

    if outcome.unapplied_corrections:
        details["unapplied_corrections"] = outcome.unapplied_corrections
        problems.append(
            f"{len(outcome.unapplied_corrections)} correction(s) reported as applied "
            "but still visible in the article"
        )

    raise FinalizePostcheckError(
        "the revised article did not pass the deterministic checks: " + "; ".join(problems),
        **details,
    )


__all__ = [
    "MAX_EVIDENCE_PROBE_CHARS",
    "FindingKey",
    "PostcheckOutcome",
    "compare_findings",
    "require_clean_postcheck",
    "validate_resolutions",
]
