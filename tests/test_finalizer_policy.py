"""Resolution validation and postcheck regression policy."""

from __future__ import annotations

from typing import Any

import pytest

from goldpipeline.domain.errors import FinalizePostcheckError, FinalizeResponseError
from goldpipeline.schemas.finalizer import (
    FinalizerModelOutput,
    IssueResolution,
    ResolutionStatus,
)
from goldpipeline.schemas.review import (
    Evidence,
    FindingCode,
    IssueCategory,
    PrecheckFinding,
    ReviewIssue,
    ReviewResult,
    ReviewStatus,
    Severity,
)
from goldpipeline.services.finalizer_policy import (
    FindingKey,
    compare_findings,
    require_clean_postcheck,
    validate_resolutions,
)
from goldpipeline.services.precheck import PrecheckReport

RUN_ID = "20260828_022701_a83f2c"
DIGEST = "a" * 64

ARTICLE = "🕯 NHẬN ĐỊNH VÀNG\n\nGiá gần nhất quanh 3305.90, thị trường đang tích luỹ trong biên hẹp."


def issue(
    *,
    issue_id: str = "i1",
    severity: Severity = Severity.HIGH,
    category: IssueCategory = IssueCategory.DATA_MISMATCH,
    evidence: Evidence | None = None,
) -> ReviewIssue:
    return ReviewIssue(
        issue_id=issue_id,
        category=category,
        severity=severity,
        message="Something to fix.",
        evidence=evidence,
    )


def review(
    *issues: ReviewIssue, status: ReviewStatus = ReviewStatus.NEEDS_REVISION
) -> ReviewResult:
    return ReviewResult(
        run_id=RUN_ID,
        status=status,
        score=60,
        summary="Cần sửa vài chỗ.",
        issues=list(issues),
        model_status=status,
        model="fake",
        provider="fake",
        prompt_version="gold_reviewer_v1",
        context_sha256=DIGEST,
        draft_sha256=DIGEST,
        writer_metadata_sha256=DIGEST,
    )


def output(*resolutions: IssueResolution, article: str = ARTICLE, run_id: str = RUN_ID) -> Any:
    return FinalizerModelOutput(run_id=run_id, article=article, issue_resolutions=list(resolutions))


def resolution(
    issue_id: str = "i1", status: ResolutionStatus = ResolutionStatus.APPLIED
) -> IssueResolution:
    return IssueResolution(
        issue_id=issue_id, resolution=status, description="Đã xử lý theo review."
    )


def finding(
    code: FindingCode = FindingCode.UNSUPPORTED_INDICATOR_MENTIONED,
    severity: Severity = Severity.HIGH,
    actual: str | None = "RSI",
) -> PrecheckFinding:
    return PrecheckFinding(
        code=code, severity=severity, message=f"{code} on {actual}.", actual=actual
    )


def report(*findings: PrecheckFinding) -> PrecheckReport:
    return PrecheckReport(findings=list(findings))


# --- resolution completeness ---------------------------------------------


def test_a_complete_account_is_accepted() -> None:
    validate_resolutions(output(resolution()), review(issue()), run_id=RUN_ID)


def test_wrong_run_id_is_rejected() -> None:
    """Requirement 27.11."""
    with pytest.raises(FinalizeResponseError, match="run_id"):
        validate_resolutions(
            output(resolution(), run_id="20200101_000000_aaaaaa"),
            review(issue()),
            run_id=RUN_ID,
        )


def test_a_missing_resolution_is_rejected() -> None:
    """Requirement 27.12: silence about an issue is not an answer."""
    with pytest.raises(FinalizeResponseError, match="must be accounted for") as exc:
        validate_resolutions(
            output(resolution("i1")),
            review(issue(issue_id="i1"), issue(issue_id="i2")),
            run_id=RUN_ID,
        )
    assert exc.value.details["missing_issue_ids"] == ["i2"]


def test_a_duplicate_resolution_is_rejected() -> None:
    """Requirement 27.13. Caught by the schema, before policy is reached."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="resolved once"):
        output(resolution("i1"), resolution("i1"))


def test_an_unknown_issue_id_is_rejected() -> None:
    """Requirement 27.14: the finalizer cannot invent issues to look diligent."""
    with pytest.raises(FinalizeResponseError, match="never raised") as exc:
        validate_resolutions(
            output(resolution("i1"), resolution("ghost")),
            review(issue(issue_id="i1")),
            run_id=RUN_ID,
        )
    assert exc.value.details["unknown_issue_ids"] == ["ghost"]


def test_an_empty_article_is_rejected() -> None:
    """Requirement 27.18. Whitespace never reaches the schema, so build it raw."""
    raw = FinalizerModelOutput.model_construct(
        run_id=RUN_ID, article="   \n\t ", issue_resolutions=[resolution()], warnings=[]
    )
    with pytest.raises(FinalizeResponseError, match="empty article"):
        validate_resolutions(raw, review(issue()), run_id=RUN_ID)


def test_a_stub_article_is_rejected() -> None:
    with pytest.raises(FinalizeResponseError, match="minimum"):
        validate_resolutions(
            output(resolution(), article="Ngắn quá."), review(issue()), run_id=RUN_ID
        )


# --- mandatory fixes ------------------------------------------------------


@pytest.mark.parametrize("severity", [Severity.HIGH, Severity.CRITICAL])
@pytest.mark.parametrize("declined", [ResolutionStatus.NOT_APPLICABLE, ResolutionStatus.BLOCKED])
def test_a_severe_issue_cannot_be_declined(severity: Severity, declined: ResolutionStatus) -> None:
    """Requirements 27.15 and 27.16: the one escape hatch that must not exist."""
    with pytest.raises(FinalizeResponseError, match="must be fixed, not declined") as exc:
        validate_resolutions(
            output(resolution(status=declined)),
            review(issue(severity=severity)),
            run_id=RUN_ID,
        )
    assert exc.value.details["declined"][0]["severity"] == str(severity)


@pytest.mark.parametrize("severity", [Severity.LOW, Severity.MEDIUM])
@pytest.mark.parametrize("declined", [ResolutionStatus.NOT_APPLICABLE, ResolutionStatus.BLOCKED])
def test_a_minor_issue_may_be_declined_with_a_reason(
    severity: Severity, declined: ResolutionStatus
) -> None:
    """Requirement 27.17: judgement is allowed where the stakes are low."""
    validate_resolutions(
        output(resolution(status=declined)),
        review(issue(severity=severity, category=IssueCategory.STYLE)),
        run_id=RUN_ID,
    )


def test_a_declined_resolution_still_needs_a_description() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        IssueResolution(issue_id="i1", resolution=ResolutionStatus.NOT_APPLICABLE, description="  ")


# --- the schema will not carry a second analysis -------------------------


def test_the_schema_has_no_field_for_a_verdict_or_score() -> None:
    """The finalizer edits; it does not re-judge."""
    forbidden = {"score", "status", "verdict", "issues", "summary"}
    assert forbidden.isdisjoint(FinalizerModelOutput.model_fields)

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        FinalizerModelOutput.model_validate({"run_id": RUN_ID, "article": ARTICLE, "score": 100})


# --- regression comparison ------------------------------------------------


def test_a_clean_revision_passes() -> None:
    outcome = compare_findings(
        original=report(finding()),
        revised=report(),
        review=review(issue()),
        final_article=ARTICLE,
    )
    assert outcome.ok
    require_clean_postcheck(outcome)


def test_a_surviving_flaw_fails() -> None:
    """Requirement 27.36: an issue that is still there was not fixed."""
    outcome = compare_findings(
        original=report(finding()),
        revised=report(finding()),
        review=review(issue()),
        final_article=ARTICLE,
    )
    assert not outcome.ok
    assert outcome.persisted
    with pytest.raises(FinalizePostcheckError, match="still present"):
        require_clean_postcheck(outcome)


def test_a_new_problem_fails() -> None:
    """Requirement 27.37: a revision that breaks something new is worse."""
    outcome = compare_findings(
        original=report(),
        revised=report(finding(actual="EMA")),
        review=review(issue()),
        final_article=ARTICLE,
    )
    assert outcome.introduced
    with pytest.raises(FinalizePostcheckError, match="the original draft did not have"):
        require_clean_postcheck(outcome)


def test_swapping_one_flaw_for_another_fails() -> None:
    """Removing RSI and adding EMA is not a fix."""
    outcome = compare_findings(
        original=report(finding(actual="RSI")),
        revised=report(finding(actual="EMA")),
        review=review(issue()),
        final_article=ARTICLE,
    )
    assert [f.actual for f in outcome.introduced] == ["EMA"]
    assert outcome.persisted == []


def test_low_and_medium_findings_do_not_fail_the_revision() -> None:
    """Only blocking severities gate a revision; the rest are recorded."""
    outcome = compare_findings(
        original=report(),
        revised=report(finding(severity=Severity.MEDIUM, actual="3325.20")),
        review=review(issue()),
        final_article=ARTICLE,
    )
    assert outcome.ok
    assert len(outcome.findings) == 1


def test_findings_are_matched_on_code_and_value_not_wording() -> None:
    first = PrecheckFinding(
        code=FindingCode.UNSUPPORTED_INDICATOR_MENTIONED,
        severity=Severity.HIGH,
        message="one wording",
        actual="RSI",
    )
    second = PrecheckFinding(
        code=FindingCode.UNSUPPORTED_INDICATOR_MENTIONED,
        severity=Severity.HIGH,
        message="a completely different wording",
        actual="RSI",
    )
    assert FindingKey.of(first) == FindingKey.of(second)


# --- corrections claimed but not made ------------------------------------


def test_a_correction_claimed_but_not_made_is_caught() -> None:
    """Requirement 18: the wrong value should be gone from the text."""
    bad = ReviewIssue(
        issue_id="i1",
        category=IssueCategory.DATA_MISMATCH,
        severity=Severity.HIGH,
        message="Sai giá.",
        evidence=Evidence(
            source_path="context.price.latest_close", expected="3305.90", actual="3325.20"
        ),
    )
    outcome = compare_findings(
        original=report(),
        revised=report(),
        review=review(bad),
        final_article="Giá gần nhất vẫn là 3325.20, không đổi gì cả.",
    )
    assert outcome.unapplied_corrections == ["i1"]
    with pytest.raises(FinalizePostcheckError, match="still visible"):
        require_clean_postcheck(outcome)


def test_a_correction_actually_made_passes() -> None:
    bad = ReviewIssue(
        issue_id="i1",
        category=IssueCategory.DATA_MISMATCH,
        severity=Severity.HIGH,
        message="Sai giá.",
        evidence=Evidence(
            source_path="context.price.latest_close", expected="3305.90", actual="3325.20"
        ),
    )
    outcome = compare_findings(
        original=report(), revised=report(), review=review(bad), final_article=ARTICLE
    )
    assert outcome.unapplied_corrections == []
    assert outcome.ok


def test_a_long_evidence_value_is_not_probed_literally() -> None:
    """A paraphrase searched verbatim would be noise, not evidence."""
    paraphrase = ReviewIssue(
        issue_id="i1",
        category=IssueCategory.SOURCE_CONTRADICTION,
        severity=Severity.HIGH,
        message="Phóng đại quan điểm nguồn.",
        evidence=Evidence(
            source_path="context.raw_analysis.text",
            expected="ưu tiên bán",
            actual="x" * 200,
        ),
    )
    outcome = compare_findings(
        original=report(),
        revised=report(),
        review=review(paraphrase),
        final_article=ARTICLE + "x" * 200,
    )
    assert outcome.unapplied_corrections == []


def test_an_issue_without_evidence_is_not_probed() -> None:
    outcome = compare_findings(
        original=report(),
        revised=report(),
        review=review(issue(category=IssueCategory.STYLE, severity=Severity.LOW)),
        final_article=ARTICLE,
    )
    assert outcome.unapplied_corrections == []
