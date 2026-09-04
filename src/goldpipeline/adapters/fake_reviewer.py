"""Offline reviewer client.

Every test runs against this. Beyond keeping the suite free of networks, keys
and cost, it makes verdicts that are awkward to provoke from a real reviewer -
a PASS that contradicts its own issue list, a response about the wrong Run -
reproducible on demand.

The default behaviour is not a canned PASS: it reads the deterministic findings
the prompt carries and answers consistently with them. A fake that always passed
would make the whole pipeline look healthy in exactly the cases that matter.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from goldpipeline.adapters.reviewer_client import ReviewRequest, ReviewResponse
from goldpipeline.domain.errors import (
    ReviewError,
    ReviewProviderError,
    ReviewResponseError,
    ReviewTimeoutError,
)
from goldpipeline.schemas.review import (
    Evidence,
    HumanStyleAssessment,
    IssueCategory,
    ReviewIssue,
    ReviewModelOutput,
    ReviewStatus,
    ReviewUsage,
    Severity,
)

FAKE_PROVIDER = "fake"
FAKE_MODEL = "fake-reviewer-v1"

_FINDING_RE = re.compile(r"^\s*\d+\.\s*\[(?P<severity>[A-Z]+)\]\s*(?P<code>[A-Z_]+):", re.MULTILINE)
_CLEAN_MARKER = "No deterministic problems were found"

_STYLE_IN_SCOPE_MARKER = "Human style **is** in scope"
"""How the fake tells whether it was asked for a style judgement.

Read back from the rendered prompt, exactly like the precheck findings above and
for the same reason: a fake that decided this from its own configuration would
keep answering correctly after the plumbing that carries the question broke.
"""


def style_in_scope(request: ReviewRequest) -> bool:
    """Whether the prompt asked this review for a human-style judgement."""
    return _STYLE_IN_SCOPE_MARKER in request.prompt.user


def clean_style_assessment(score: int = 92) -> HumanStyleAssessment:
    """A style judgement with nothing to report.

    The default on purpose. The fake could mirror the deterministic symptoms the
    prompt carries into style findings, and it deliberately does not: a symptom
    is a hint and a finding is a judgement, and a fake that collapsed the two
    would quietly teach every test the exact confusion the round is guarding
    against. Tests that want findings supply them.
    """
    return HumanStyleAssessment(
        style_score=score,
        summary="Bài viết đọc tự nhiên, có quan điểm rõ ràng, không thừa chữ.",
        findings=[],
    )


@dataclass(frozen=True)
class _Finding:
    severity: Severity
    code: str


def _read_findings(request: ReviewRequest) -> list[_Finding]:
    """Recover the deterministic findings from the rendered prompt.

    Reading back the prompt keeps the fake honest: it answers about the Run it
    was actually given rather than a fixed script, so a plumbing mistake that
    sends the wrong findings shows up as a wrong verdict.
    """
    if _CLEAN_MARKER in request.prompt.user:
        return []
    return [
        _Finding(severity=Severity(match["severity"]), code=match["code"])
        for match in _FINDING_RE.finditer(request.prompt.user)
        if match["severity"] in Severity.__members__
    ]


def _category_for(code: str) -> IssueCategory:
    if "SYMBOL" in code or "NUMBER" in code or "MISMATCH" in code:
        return IssueCategory.DATA_MISMATCH
    if "INDICATOR" in code or "CLAIM" in code:
        return IssueCategory.UNSUPPORTED_CLAIM
    if "RISK" in code:
        return IssueCategory.RISK_LANGUAGE
    return IssueCategory.OTHER


@dataclass
class FakeReviewerClient:
    """Deterministic, offline implementation of :class:`ReviewerClient`.

    Configure at most one behaviour:

    * default - answer consistently with the prompt's precheck findings;
    * ``raises`` - raise that error instead of answering;
    * ``output`` - return a specific :class:`ReviewModelOutput`, for contract
      violations such as a wrong run id or a self-contradictory verdict;
    * ``output_factory`` - compute the output from the request.
    """

    output: ReviewModelOutput | None = None
    output_factory: Callable[[ReviewRequest], ReviewModelOutput] | None = None
    raises: ReviewError | None = None
    usage: ReviewUsage = field(
        default_factory=lambda: ReviewUsage(input_tokens=2400, output_tokens=380, total_tokens=2780)
    )
    model_name: str = FAKE_MODEL
    calls: list[ReviewRequest] = field(default_factory=list)
    """Every request seen, so tests can assert on what was actually sent."""

    @property
    def provider(self) -> str:
        return FAKE_PROVIDER

    @property
    def model(self) -> str:
        return self.model_name

    def review(self, request: ReviewRequest) -> ReviewResponse:
        """Return the configured response, or one derived from the prompt."""
        self.calls.append(request)

        if self.raises is not None:
            raise self.raises

        if self.output is not None:
            output = self.output
        elif self.output_factory is not None:
            output = self.output_factory(request)
        else:
            output = self._build_default(request)

        return ReviewResponse(
            output=output, model=self.model, provider=self.provider, usage=self.usage
        )

    def _build_default(self, request: ReviewRequest) -> ReviewModelOutput:
        findings = _read_findings(request)
        style = clean_style_assessment() if style_in_scope(request) else None

        if not findings:
            return ReviewModelOutput(
                run_id=request.run_id,
                status=ReviewStatus.PASS,
                score=95,
                summary=(
                    "Bài viết bám sát dữ liệu trong context, không phát hiện sai lệch "
                    "số liệu hay claim không có căn cứ."
                ),
                issues=[],
                revision_instructions=[],
                style_review=style,
            )

        severities = {finding.severity for finding in findings}
        if Severity.CRITICAL in severities:
            status, score = ReviewStatus.REJECT, 30
        elif Severity.HIGH in severities:
            status, score = ReviewStatus.NEEDS_REVISION, 62
        else:
            status, score = ReviewStatus.NEEDS_REVISION, 82

        issues = [
            ReviewIssue(
                issue_id=f"fake-{index}-{finding.code.lower()}",
                category=_category_for(finding.code),
                severity=finding.severity,
                message=f"Precheck reported {finding.code}; confirmed against the context.",
                evidence=(
                    Evidence(
                        source_path="context.price.latest_close",
                        expected="see context",
                        actual="see article",
                    )
                    if finding.severity in {Severity.HIGH, Severity.CRITICAL}
                    else None
                ),
            )
            for index, finding in enumerate(findings, start=1)
        ]

        return ReviewModelOutput(
            run_id=request.run_id,
            status=status,
            score=score,
            summary=(
                f"Phát hiện {len(issues)} vấn đề cần xử lý trước khi xuất bản, "
                "trong đó có sai lệch so với dữ liệu nguồn."
            ),
            issues=issues,
            revision_instructions=[
                f"Xử lý vấn đề {issue.issue_id} theo đúng dữ liệu trong context."
                for issue in issues
            ],
            style_review=style,
        )


def passing_client(score: int = 95) -> FakeReviewerClient:
    """A reviewer that always passes, whatever the prompt says.

    Used to prove that policy escalation works: a generous verdict must not be
    able to overrule a deterministic finding.
    """
    return FakeReviewerClient(
        output_factory=lambda request: ReviewModelOutput(
            run_id=request.run_id,
            status=ReviewStatus.PASS,
            score=score,
            summary="Không phát hiện vấn đề nào.",
            issues=[],
            revision_instructions=[],
            style_review=clean_style_assessment() if style_in_scope(request) else None,
        )
    )


def failing_client(error: ReviewError) -> FakeReviewerClient:
    """A client that always raises *error*."""
    return FakeReviewerClient(raises=error)


def timing_out_client(seconds: float = 120.0) -> FakeReviewerClient:
    """A client that always times out."""
    return failing_client(
        ReviewTimeoutError(f"provider did not respond within {seconds:g}s", timeout_seconds=seconds)
    )


def erroring_client(message: str = "provider returned HTTP 500") -> FakeReviewerClient:
    """A client that always reports a provider failure."""
    return failing_client(ReviewProviderError(message, status_code=500))


def malformed_client(message: str = "response was not valid JSON") -> FakeReviewerClient:
    """A client that always reports an unparseable answer."""
    return failing_client(ReviewResponseError(message))


__all__ = [
    "FAKE_MODEL",
    "FAKE_PROVIDER",
    "FakeReviewerClient",
    "clean_style_assessment",
    "erroring_client",
    "failing_client",
    "malformed_client",
    "passing_client",
    "style_in_scope",
    "timing_out_client",
]
