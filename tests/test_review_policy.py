"""Response validation and verdict policy.

Two separate mechanisms, and the tests keep them separate:

* ``validate_response`` rejects an answer that contradicts itself;
* ``apply_policy`` escalates a verdict the deterministic evidence outranks.
"""

from __future__ import annotations

from typing import Any

import pytest

from goldpipeline.domain.errors import ReviewResponseError
from goldpipeline.schemas.review import (
    Evidence,
    FindingCode,
    IssueCategory,
    PrecheckFinding,
    ReviewIssue,
    ReviewModelOutput,
    ReviewStatus,
    Severity,
    VerdictSource,
)
from goldpipeline.services.precheck import PrecheckReport
from goldpipeline.services.review_policy import apply_policy, validate_response

RUN_ID = "20260828_022701_a83f2c"


def issue(
    *,
    severity: Severity = Severity.MEDIUM,
    category: IssueCategory = IssueCategory.STYLE,
    issue_id: str = "i1",
    evidence: Evidence | None = None,
) -> ReviewIssue:
    return ReviewIssue(
        issue_id=issue_id,
        category=category,
        severity=severity,
        message="Something to fix.",
        evidence=evidence,
    )


def output(**overrides: Any) -> ReviewModelOutput:
    fields: dict[str, Any] = {
        "run_id": RUN_ID,
        "status": ReviewStatus.PASS,
        "score": 95,
        "summary": "Không phát hiện vấn đề đáng kể.",
        "issues": [],
        "revision_instructions": [],
    }
    fields.update(overrides)
    return ReviewModelOutput(**fields)


def report(*findings: PrecheckFinding) -> PrecheckReport:
    return PrecheckReport(findings=list(findings))


def finding(
    severity: Severity, code: FindingCode = FindingCode.CLAIM_VALUE_MISMATCH
) -> PrecheckFinding:
    return PrecheckFinding(
        code=code, severity=severity, message=f"{code} at {severity}.", actual="3325.20"
    )


# --- response validity ----------------------------------------------------


def test_a_clean_pass_is_accepted() -> None:
    validate_response(output(), run_id=RUN_ID)


def test_wrong_run_id_is_rejected() -> None:
    """Requirement 27.16: an answer about another Run is not this Run's review."""
    with pytest.raises(ReviewResponseError, match="run_id"):
        validate_response(output(run_id="20200101_000000_aaaaaa"), run_id=RUN_ID)


@pytest.mark.parametrize("severity", [Severity.HIGH, Severity.CRITICAL])
def test_pass_with_a_blocking_issue_is_rejected(severity: Severity) -> None:
    """Requirement 27.18: a PASS cannot list a problem that rules out passing."""
    with pytest.raises(ReviewResponseError, match="rule out a pass"):
        validate_response(
            output(
                status=ReviewStatus.PASS,
                issues=[
                    issue(
                        severity=severity,
                        category=IssueCategory.DATA_MISMATCH,
                        evidence=Evidence(
                            source_path="context.price.latest_close",
                            expected="3305.90",
                            actual="3325.20",
                        ),
                    )
                ],
            ),
            run_id=RUN_ID,
        )


def test_pass_with_low_severity_issues_is_allowed() -> None:
    """Style nits do not stop an article passing."""
    validate_response(
        output(
            issues=[
                issue(severity=Severity.LOW),
                issue(severity=Severity.MEDIUM, issue_id="i2"),
            ]
        ),
        run_id=RUN_ID,
    )


def test_pass_with_revision_instructions_is_rejected() -> None:
    """Requirement 27.19: a passing article needs no edits."""
    with pytest.raises(ReviewResponseError, match="asks for revisions"):
        validate_response(output(revision_instructions=["Sửa lại câu mở đầu."]), run_id=RUN_ID)


def test_pass_below_the_score_threshold_is_rejected() -> None:
    with pytest.raises(ReviewResponseError, match="below the 90"):
        validate_response(output(score=72), run_id=RUN_ID)


@pytest.mark.parametrize("status", [ReviewStatus.NEEDS_REVISION, ReviewStatus.REJECT])
def test_a_non_pass_verdict_without_issues_is_rejected(status: ReviewStatus) -> None:
    """Requirements 27.20 and 19: a verdict must justify itself."""
    with pytest.raises(ReviewResponseError, match="no issues to justify"):
        validate_response(output(status=status, score=50, issues=[]), run_id=RUN_ID)


def test_a_factual_issue_without_evidence_is_rejected() -> None:
    """Requirement 9: 'the number looks wrong' is not a finding."""
    with pytest.raises(ReviewResponseError, match="cite evidence"):
        validate_response(
            output(
                status=ReviewStatus.NEEDS_REVISION,
                score=60,
                issues=[
                    issue(
                        severity=Severity.HIGH,
                        category=IssueCategory.DATA_MISMATCH,
                        evidence=None,
                    )
                ],
                revision_instructions=["Sửa số liệu."],
            ),
            run_id=RUN_ID,
        )


def test_a_factual_issue_with_evidence_is_accepted() -> None:
    validate_response(
        output(
            status=ReviewStatus.NEEDS_REVISION,
            score=60,
            issues=[
                issue(
                    severity=Severity.HIGH,
                    category=IssueCategory.DATA_MISMATCH,
                    evidence=Evidence(
                        source_path="context.price.latest_close",
                        expected="3305.90",
                        actual="3325.20",
                    ),
                )
            ],
            revision_instructions=["Sửa giá gần nhất thành 3305.90."],
        ),
        run_id=RUN_ID,
    )


def test_a_style_issue_needs_no_evidence() -> None:
    """Only factual categories carry the evidence obligation."""
    validate_response(
        output(
            status=ReviewStatus.NEEDS_REVISION,
            score=80,
            issues=[issue(severity=Severity.HIGH, category=IssueCategory.STYLE)],
            revision_instructions=["Rút ngắn phần mở đầu."],
        ),
        run_id=RUN_ID,
    )


# --- schema-level protections --------------------------------------------


def test_the_schema_has_no_field_for_a_rewritten_article() -> None:
    """Requirement 10: the reviewer must not be able to return a new draft."""
    forbidden = {"revised_article", "final_article", "better_version", "rewritten_text"}
    assert forbidden.isdisjoint(ReviewModelOutput.model_fields)

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ReviewModelOutput.model_validate(
            {
                "run_id": RUN_ID,
                "status": "PASS",
                "score": 95,
                "summary": "ok",
                "revised_article": "🕯 PHÂN TÍCH VÀNG ...",
            }
        )


def test_an_instruction_long_enough_to_be_a_draft_is_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="must not rewrite"):
        output(status=ReviewStatus.NEEDS_REVISION, score=60, revision_instructions=["x" * 900])


def test_duplicate_issue_ids_are_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="unique"):
        output(
            status=ReviewStatus.NEEDS_REVISION,
            score=60,
            issues=[issue(issue_id="dup"), issue(issue_id="dup")],
        )


def test_unknown_severity_and_category_are_rejected() -> None:
    from pydantic import ValidationError

    for field, value in (("severity", "CATASTROPHIC"), ("category", "VIBES")):
        with pytest.raises(ValidationError):
            ReviewIssue.model_validate(
                {
                    "issue_id": "i1",
                    "category": "STYLE",
                    "severity": "LOW",
                    "message": "m",
                    field: value,
                }
            )


def test_unknown_status_is_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        output(status="LOOKS_FINE")


# --- verdict escalation ---------------------------------------------------


def test_a_clean_precheck_leaves_the_verdict_alone() -> None:
    outcome = apply_policy(output(), report())
    assert outcome.status is ReviewStatus.PASS
    assert outcome.verdict_source is VerdictSource.MODEL
    assert outcome.notes == []


def test_a_high_finding_escalates_a_pass() -> None:
    """Requirement 27.34: deterministic evidence outranks a generous verdict."""
    outcome = apply_policy(output(), report(finding(Severity.HIGH)))

    assert outcome.status is ReviewStatus.NEEDS_REVISION
    assert outcome.model_status is ReviewStatus.PASS
    assert outcome.verdict_source is VerdictSource.POLICY_ESCALATED
    assert any("escalated" in note.lower() for note in outcome.notes)


def test_a_critical_finding_escalates_to_reject() -> None:
    outcome = apply_policy(
        output(), report(finding(Severity.CRITICAL, FindingCode.FOREIGN_SYMBOL_MENTIONED))
    )
    assert outcome.status is ReviewStatus.REJECT
    assert outcome.verdict_source is VerdictSource.POLICY_ESCALATED


def test_low_and_medium_findings_do_not_escalate() -> None:
    outcome = apply_policy(output(), report(finding(Severity.LOW), finding(Severity.MEDIUM)))
    assert outcome.status is ReviewStatus.PASS
    assert outcome.verdict_source is VerdictSource.MODEL


def test_a_stricter_model_verdict_is_never_softened() -> None:
    """Policy only ever escalates. It must not talk a REJECT down."""
    strict = output(
        status=ReviewStatus.REJECT,
        score=20,
        issues=[issue(severity=Severity.CRITICAL, issue_id="x1")],
        revision_instructions=["Viết lại toàn bộ phần số liệu."],
    )
    outcome = apply_policy(strict, report(finding(Severity.HIGH)))

    assert outcome.status is ReviewStatus.REJECT
    assert outcome.verdict_source is VerdictSource.MODEL


def test_missed_blocking_findings_become_issues() -> None:
    outcome = apply_policy(output(), report(finding(Severity.HIGH)))

    assert len(outcome.issues) == 1
    added = outcome.issues[0]
    assert added.issue_id.startswith("precheck-")
    assert added.severity is Severity.HIGH
    assert added.category is IssueCategory.DATA_MISMATCH


def test_findings_the_reviewer_already_reported_are_not_duplicated() -> None:
    reported = ReviewIssue(
        issue_id="model-1",
        category=IssueCategory.DATA_MISMATCH,
        severity=Severity.HIGH,
        message="The article says 3325.20 but the context says 3305.90.",
        evidence=Evidence(
            source_path="context.price.latest_close", expected="3305.90", actual="3325.20"
        ),
    )
    outcome = apply_policy(
        output(
            status=ReviewStatus.NEEDS_REVISION,
            score=60,
            issues=[reported],
            revision_instructions=["Sửa giá."],
        ),
        report(finding(Severity.HIGH)),
    )

    assert len(outcome.issues) == 1
    assert outcome.issues[0].issue_id == "model-1"


def test_non_blocking_findings_are_not_promoted_to_issues() -> None:
    """They stay visible in deterministic_findings without burying real issues."""
    outcome = apply_policy(output(), report(finding(Severity.MEDIUM)))
    assert outcome.issues == []


# --- deterministic severity authority (Round 9.3.4A) -----------------------
#
# A model may report its own, milder issue about a finding the deterministic
# pass already classified as blocking. Effective severity must never fall
# below the deterministic one - the model may only ever raise it.


def cited_issue(*, severity: Severity, actual: str = "3325.20") -> ReviewIssue:
    """A model issue that explicitly cites the same evidence as a finding."""
    return issue(
        severity=severity,
        category=IssueCategory.DATA_MISMATCH,
        evidence=Evidence(
            source_path="context.price.latest_close", expected="3305.90", actual=actual
        ),
    )


def mentioning_issue(*, severity: Severity, actual: str = "3325.20") -> ReviewIssue:
    """A model issue that only mentions the finding's value in its own words."""
    return ReviewIssue(
        issue_id="i1",
        category=IssueCategory.DATA_MISMATCH,
        severity=severity,
        message=f"The article states {actual!r}; the paraphrase, not the raw claim, is at fault.",
    )


def cited_finding(severity: Severity, actual: str = "3325.20") -> PrecheckFinding:
    """A finding whose (source_path, actual) matches ``cited_issue``'s evidence exactly."""
    return PrecheckFinding(
        code=FindingCode.CLAIM_VALUE_MISMATCH,
        severity=severity,
        message=f"CLAIM_VALUE_MISMATCH at {severity}.",
        source_path="context.price.latest_close",
        actual=actual,
    )


@pytest.mark.parametrize(
    ("deterministic", "model", "effective"),
    [
        (Severity.HIGH, Severity.MEDIUM, Severity.HIGH),
        (Severity.HIGH, Severity.LOW, Severity.HIGH),
        (Severity.HIGH, Severity.HIGH, Severity.HIGH),
        (Severity.HIGH, Severity.CRITICAL, Severity.CRITICAL),
        (Severity.CRITICAL, Severity.LOW, Severity.CRITICAL),
        (Severity.CRITICAL, Severity.HIGH, Severity.CRITICAL),
    ],
)
def test_effective_severity_is_never_below_the_deterministic_one(
    deterministic: Severity, model: Severity, effective: Severity
) -> None:
    """The production gap: a model must not launder a blocking finding's
    severity down by citing it in a milder issue of its own."""
    outcome = apply_policy(
        output(
            status=ReviewStatus.NEEDS_REVISION,
            score=60,
            issues=[cited_issue(severity=model)],
            revision_instructions=["Sửa giá."],
        ),
        report(cited_finding(deterministic)),
    )

    assert len(outcome.issues) == 1, "citing the finding must not duplicate it"
    assert outcome.issues[0].severity is effective


def test_a_model_issue_can_still_escalate_a_medium_finding() -> None:
    """Escalation is always allowed - only downgrading is forbidden."""
    outcome = apply_policy(
        output(
            status=ReviewStatus.NEEDS_REVISION,
            score=60,
            issues=[issue(severity=Severity.HIGH, category=IssueCategory.STYLE)],
        ),
        report(finding(Severity.MEDIUM)),
    )

    assert outcome.issues[0].severity is Severity.HIGH, (
        "an unrelated MEDIUM finding must not touch it"
    )


def test_low_deterministic_and_low_model_stay_low() -> None:
    outcome = apply_policy(output(), report(finding(Severity.LOW)))
    assert outcome.issues == []


def test_severity_escalation_via_mention_not_exact_citation() -> None:
    """The looser of the two existing association signals also participates."""
    outcome = apply_policy(
        output(
            status=ReviewStatus.NEEDS_REVISION,
            score=60,
            issues=[mentioning_issue(severity=Severity.MEDIUM)],
            revision_instructions=["Sửa giá."],
        ),
        report(finding(Severity.HIGH)),
    )

    assert len(outcome.issues) == 1
    assert outcome.issues[0].severity is Severity.HIGH
    assert any("severity raised" in note for note in outcome.notes)


def test_an_unrelated_model_issue_does_not_suppress_the_finding() -> None:
    """No reliable association exists, so the finding is promoted as before -
    it must never simply disappear because some unrelated issue exists."""
    outcome = apply_policy(
        output(
            status=ReviewStatus.NEEDS_REVISION,
            score=60,
            issues=[issue(severity=Severity.LOW, issue_id="unrelated")],
            revision_instructions=["Sửa văn phong."],
        ),
        report(finding(Severity.HIGH)),
    )

    assert len(outcome.issues) == 2
    ids = {i.issue_id for i in outcome.issues}
    assert "unrelated" in ids
    promoted = next(i for i in outcome.issues if i.issue_id != "unrelated")
    assert promoted.severity is Severity.HIGH


def test_verdict_escalation_is_unaffected_by_severity_reconciliation() -> None:
    """This round changes issue closure, not verdict policy."""
    outcome = apply_policy(
        output(
            status=ReviewStatus.PASS,
            issues=[cited_issue(severity=Severity.MEDIUM)],
        ),
        report(cited_finding(Severity.HIGH)),
    )

    assert outcome.status is ReviewStatus.NEEDS_REVISION
    assert outcome.verdict_source is VerdictSource.POLICY_ESCALATED
    assert outcome.issues[0].severity is Severity.HIGH


def test_production_shaped_note_paraphrase_regression() -> None:
    """Shaped after the real Round 9.3.3 finding: a HIGH CLAIM_VALUE_MISMATCH
    on ``context.raw_analysis.text``, cited by a model issue the model itself
    filed as MEDIUM. Must come out HIGH, not silently stay MEDIUM."""
    note_finding = PrecheckFinding(
        code=FindingCode.CLAIM_VALUE_MISMATCH,
        severity=Severity.HIGH,
        message="Precheck 17 (HIGH): claim paraphrases raw_analysis.text instead of quoting it.",
        source_path="context.raw_analysis.text",
        actual="ghi chú là dữ liệu kiểm thử hệ thống, không phải tín hiệu giao dịch",
    )
    model_issue = ReviewIssue(
        issue_id="PRECHECK-NOTE-PARAPHRASE",
        category=IssueCategory.SOURCE_CONTRADICTION,
        severity=Severity.MEDIUM,
        message="Precheck 17 (HIGH): the note is paraphrased, not quoted verbatim.",
        evidence=Evidence(
            source_path="context.raw_analysis.text",
            expected="verbatim raw_analysis.text",
            actual="ghi chú là dữ liệu kiểm thử hệ thống, không phải tín hiệu giao dịch",
        ),
    )

    outcome = apply_policy(
        output(
            status=ReviewStatus.NEEDS_REVISION,
            score=78,
            issues=[model_issue],
            revision_instructions=["Trích nguyên văn ghi chú nguồn."],
        ),
        report(note_finding),
    )

    assert len(outcome.issues) == 1
    assert outcome.issues[0].severity is Severity.HIGH
    assert outcome.issues[0].issue_id == "PRECHECK-NOTE-PARAPHRASE"
