"""Style activation: what triggers a revision, and what the revision may do.

Round 6.4g turns the Round 6.4f style verdict from a recorded observation into
something that can require a repair. Three properties matter, and they are the
three the tests below are organised around.

**One call.** A Run gets at most one automatic model revision, for content, for
style, or for both together. Every failure after that call is terminal: an
unresolved finding, a broken contract, an invented number. None of them buys a
second attempt, because a model told "your last answer was rejected" changes
things nobody asked it to, and because deterministic code cannot adjudicate a
second opinion about prose.

**The verdict is not the action.** ``review.status`` still records what the
reviewer said about the *facts*, unchanged and unrewritten. Whether the
finalizer runs is a separate question with its own answer, and the artifact
never has to lie about the first to get the second.

**Smoother is a regression.** A revision asked to trim one section that comes
back having improved three has not done the job better; it has done a different
job, outside anything anybody reviewed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from conftest import (
    CLEAN_ARTICLE,
    DISCLAIMER,
    FIXTURE_ARTICLE_DATE,
    LATEST_CLOSE,
    make_drafted_run,
    make_reviewed_run,
)

from goldpipeline.adapters.fake_finalizer import (
    FakeFinalizerClient,
    apply_review,
    polishing_client,
    read_prompt_article,
    read_prompt_review,
    read_style_findings,
    unresolved_style_client,
)
from goldpipeline.adapters.fake_reviewer import FakeReviewerClient, clean_style_assessment
from goldpipeline.domain.errors import FinalizePostcheckError, FinalizeResponseError
from goldpipeline.prompts import (
    DEFAULT_FINALIZER_PROMPT,
    GOLD_FINALIZER_V1,
    GOLD_FINALIZER_V2,
    GOLD_HUMAN_STYLE_V1,
    load_prompt,
)
from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.article_contract import SectionKey
from goldpipeline.schemas.finalizer import (
    FinalizationMode,
    FinalizerModelOutput,
    IssueResolution,
    ResolutionStatus,
    StyleResolution,
    StyleResolutionStatus,
)
from goldpipeline.schemas.review import (
    Evidence,
    HumanStyleAssessment,
    HumanStyleCategory,
    HumanStyleFinding,
    IssueCategory,
    ReviewModelOutput,
    ReviewResult,
    ReviewStatus,
    Severity,
    StyleSeverity,
    StyleVerdict,
)
from goldpipeline.services.finalizer import finalize_run
from goldpipeline.services.review_action import (
    STYLE_ACTIVE_TYPES,
    ReviewAction,
    effective_action,
    style_is_active,
)
from goldpipeline.storage.run_store import RunStore

# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def style_finding(
    category: HumanStyleCategory = HumanStyleCategory.DATA_DUMP,
    severity: StyleSeverity = StyleSeverity.MEDIUM,
    *,
    finding_id: str | None = None,
    section: SectionKey | None = SectionKey.PRICE_READ,
    problem: str = "The section lists four figures where one judgement would carry it.",
    repair: str = "Keep the strongest number and cut the rest.",
) -> HumanStyleFinding:
    return HumanStyleFinding(
        finding_id=finding_id or f"style-{category.lower()}",
        category=category,
        severity=severity,
        section=section,
        problem=problem,
        repair_instruction=repair,
    )


def style_review(*findings: HumanStyleFinding, score: int = 62) -> HumanStyleAssessment:
    return HumanStyleAssessment(
        style_score=score, summary="Đọc như báo cáo hơn là như người.", findings=list(findings)
    )


BLOCKING_STYLE = style_review(style_finding(severity=StyleSeverity.HIGH))
"""A style judgement that derives NEEDS_REVISION: one HIGH finding."""


CONTENT_ISSUE = {
    "issue_id": "content-price",
    "category": IssueCategory.DATA_MISMATCH,
    "severity": Severity.HIGH,
}


def reviewer(
    *,
    status: ReviewStatus = ReviewStatus.PASS,
    score: int = 95,
    style: HumanStyleAssessment | None = None,
    wrong_value: str | None = None,
) -> FakeReviewerClient:
    """A reviewer whose two axes are set independently.

    ``wrong_value`` raises a real content issue with evidence, so the finalizer
    has something factual to correct alongside whatever style asks for.
    """
    from goldpipeline.schemas.review import ReviewIssue

    issues = []
    instructions: list[str] = []
    if wrong_value is not None:
        issues.append(
            ReviewIssue(
                issue_id=CONTENT_ISSUE["issue_id"],
                category=CONTENT_ISSUE["category"],
                severity=CONTENT_ISSUE["severity"],
                message="Giá gần nhất không khớp context.",
                evidence=Evidence(
                    source_path="context.price.latest_close",
                    expected=LATEST_CLOSE,
                    actual=wrong_value,
                ),
            )
        )
        instructions.append(f"Sửa giá gần nhất thành {LATEST_CLOSE}.")

    def build(request: Any) -> ReviewModelOutput:
        return ReviewModelOutput(
            run_id=request.run_id,
            status=status,
            score=score,
            summary="Kiểm tra hoàn tất.",
            issues=issues,
            revision_instructions=instructions,
            style_review=style if style is not None else clean_style_assessment(),
        )

    return FakeReviewerClient(output_factory=build)


def finalize(
    runs_dir: Path,
    tmp_path: Path,
    *,
    review_client: FakeReviewerClient,
    finalizer: FakeFinalizerClient | None = None,
    article: str = CLEAN_ARTICLE,
    claims: list[Any] | None = None,
) -> tuple[Any, FakeFinalizerClient]:
    """Drive a real Run through review and finalization."""
    reviewed = make_reviewed_run(
        runs_dir, tmp_path, article=article, claims=claims, review_client=review_client
    )
    client = finalizer or FakeFinalizerClient()
    result = finalize_run(run_id=reviewed.run_id, store=RunStore(runs_dir), client=client)
    return result, client


def review_artifact(
    *, status: ReviewStatus = ReviewStatus.PASS, style: HumanStyleAssessment | None
) -> ReviewResult:
    """A committed review artifact, built directly for the action tests."""
    from goldpipeline.services.style_review import build_style_review

    digest = "0" * 64
    return ReviewResult(
        run_id="r1",
        status=status,
        score=95 if status is ReviewStatus.PASS else 60,
        summary="ok",
        model_status=status,
        style_review=None if style is None else build_style_review(style),
        model="m",
        provider="fake",
        prompt_version="gold_reviewer_v2",
        context_sha256=digest,
        draft_sha256=digest,
        writer_metadata_sha256=digest,
    )


# --------------------------------------------------------------------------
# 1-9: the effective-action matrix
# --------------------------------------------------------------------------


ACTION_MATRIX: list[tuple[str, ReviewStatus, HumanStyleAssessment | None, ReviewAction]] = [
    ("1 content PASS, style PASS", ReviewStatus.PASS, style_review(), ReviewAction.PASS_THROUGH),
    (
        "2 content PASS, style NEEDS_REVISION",
        ReviewStatus.PASS,
        BLOCKING_STYLE,
        ReviewAction.FINALIZE,
    ),
    (
        "3 content NEEDS_REVISION, style PASS",
        ReviewStatus.NEEDS_REVISION,
        style_review(),
        ReviewAction.FINALIZE,
    ),
    (
        "4 content NEEDS_REVISION, style NEEDS_REVISION",
        ReviewStatus.NEEDS_REVISION,
        BLOCKING_STYLE,
        ReviewAction.FINALIZE,
    ),
    ("5 content REJECT, style PASS", ReviewStatus.REJECT, style_review(), ReviewAction.REJECT),
    (
        "6 content REJECT, style NEEDS_REVISION",
        ReviewStatus.REJECT,
        BLOCKING_STYLE,
        ReviewAction.REJECT,
    ),
    ("7 historical, content PASS", ReviewStatus.PASS, None, ReviewAction.PASS_THROUGH),
    (
        "8 historical, content NEEDS_REVISION",
        ReviewStatus.NEEDS_REVISION,
        None,
        ReviewAction.FINALIZE,
    ),
]


@pytest.mark.parametrize(
    ("name", "status", "style", "expected"), ACTION_MATRIX, ids=[c[0] for c in ACTION_MATRIX]
)
def test_the_effective_action_matrix(
    name: str,
    status: ReviewStatus,
    style: HumanStyleAssessment | None,
    expected: ReviewAction,
) -> None:
    decision = effective_action(
        review_artifact(status=status, style=style), article_type=ArticleType.ANALYSIS
    )

    assert decision.action is expected, name


def test_a_trade_plan_never_activates_human_style() -> None:
    """Case 9. A rendered document has no prose to repair."""
    assert not style_is_active(ArticleType.TRADE_PLAN)
    assert ArticleType.TRADE_PLAN not in STYLE_ACTIVE_TYPES

    decision = effective_action(
        review_artifact(status=ReviewStatus.PASS, style=BLOCKING_STYLE),
        article_type=ArticleType.TRADE_PLAN,
    )

    assert decision.action is ReviewAction.PASS_THROUGH
    assert decision.style_findings == ()


def test_the_news_digest_is_judged_but_not_yet_repaired() -> None:
    """Activation is per type, and the two activations are separate.

    Round 6.5b made the digest producible; style-driven repair of one is a
    different switch, and it stays off until 6.5c has real digest evidence to
    tune against.
    """
    from goldpipeline.services.style_review import applies_to

    assert applies_to(ArticleType.NEWS_DIGEST)
    assert not style_is_active(ArticleType.NEWS_DIGEST)


def test_activation_is_a_frozen_set_not_a_flag() -> None:
    """No default anybody can flip by accident, and no environment variable."""
    import inspect

    from goldpipeline.services import review_action

    source = inspect.getsource(review_action)
    assert "os.environ" not in source
    assert "getenv" not in source
    assert isinstance(STYLE_ACTIVE_TYPES, frozenset)


def test_only_one_module_reads_the_style_verdict_off_a_review() -> None:
    """The scattering this round was told to avoid, asserted directly.

    The property is about *deciding*, not about mentioning. Reading
    ``review.style_review`` is how a module forms its own opinion on whether a
    revision is needed, and exactly one module is allowed to do that. Reading
    ``decision.style_verdict`` - which the finalizer does, for one log line -
    is reading the answer somebody else already gave, and is fine.
    """
    import ast

    allowed = {"review_action", "style_review", "reviewer"}
    readers = []
    for path in Path("src/goldpipeline/services").glob("*.py"):
        if path.stem in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "style_review":
                readers.append(path.name)
                break

    assert readers == [], f"style is read off a review outside the action policy: {readers}"


def test_nothing_outside_the_policy_derives_a_style_obligation() -> None:
    """The repair rule has one importer, so two modules cannot disagree about it."""
    import ast

    importers = []
    for path in Path("src/goldpipeline/services").glob("*.py"):
        if path.stem in {"style_review", "review_action"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "style_review" in node.module
                and any(a.name == "findings_requiring_repair" for a in node.names)
            ):
                importers.append(path.name)

    assert importers == []


# --------------------------------------------------------------------------
# the content verdict is never rewritten
# --------------------------------------------------------------------------


def test_a_style_driven_revision_leaves_the_content_verdict_alone(
    runs_dir: Path, tmp_path: Path
) -> None:
    """The artifact still says PASS, because that is what was judged."""
    result, client = finalize(runs_dir, tmp_path, review_client=reviewer(style=BLOCKING_STYLE))

    assert result.succeeded
    assert result.result is not None
    assert result.result.review_status is ReviewStatus.PASS
    assert result.result.finalization_mode is FinalizationMode.REVISED
    assert len(client.calls) == 1

    review = ReviewResult.model_validate_json(
        (Path(result.run_dir) / "gpt_review.json").read_text(encoding="utf-8")
    )
    assert review.status is ReviewStatus.PASS
    assert review.style_review is not None
    assert review.style_review.style_verdict is StyleVerdict.NEEDS_REVISION


def test_a_clean_run_still_passes_through_without_a_model(runs_dir: Path, tmp_path: Path) -> None:
    result, client = finalize(runs_dir, tmp_path, review_client=reviewer())

    assert result.succeeded
    assert result.result is not None
    assert result.result.finalization_mode is FinalizationMode.PASSTHROUGH
    assert result.result.provider_called is False
    assert client.calls == []


# --------------------------------------------------------------------------
# one call, whatever the reason
# --------------------------------------------------------------------------


ONE_CALL_CASES: list[tuple[str, ReviewStatus, HumanStyleAssessment, str | None]] = [
    ("style only", ReviewStatus.PASS, BLOCKING_STYLE, None),
    ("content only", ReviewStatus.NEEDS_REVISION, style_review(), "3325.20"),
    ("content and style", ReviewStatus.NEEDS_REVISION, BLOCKING_STYLE, "3325.20"),
]


@pytest.mark.parametrize(
    ("name", "status", "style", "wrong"), ONE_CALL_CASES, ids=[c[0] for c in ONE_CALL_CASES]
)
def test_a_revision_costs_exactly_one_finalizer_call(
    name: str,
    status: ReviewStatus,
    style: HumanStyleAssessment,
    wrong: str | None,
    runs_dir: Path,
    tmp_path: Path,
) -> None:
    """Content, style, or both - one call. Never two, never one each."""
    article = CLEAN_ARTICLE if wrong is None else CLEAN_ARTICLE.replace(LATEST_CLOSE, wrong)
    result, client = finalize(
        runs_dir,
        tmp_path,
        review_client=reviewer(
            status=status, score=60 if wrong else 95, style=style, wrong_value=wrong
        ),
        article=article,
        claims=[],
    )

    assert result.succeeded, name
    assert len(client.calls) == 1, name


def test_both_axes_are_repaired_in_the_same_call(runs_dir: Path, tmp_path: Path) -> None:
    """One prompt carries the content issues and the style findings together."""
    result, client = finalize(
        runs_dir,
        tmp_path,
        review_client=reviewer(
            status=ReviewStatus.NEEDS_REVISION,
            score=60,
            style=BLOCKING_STYLE,
            wrong_value="3325.20",
        ),
        article=CLEAN_ARTICLE.replace(LATEST_CLOSE, "3325.20"),
        claims=[],
    )

    assert result.succeeded
    assert len(client.calls) == 1

    payload = read_prompt_review(client.calls[0])
    assert payload["issues"], "the content issues must reach the same prompt"
    assert payload["style_findings"], "so must the style findings"

    assert result.result is not None
    assert result.result.issue_resolutions
    assert result.result.style_resolutions


def test_a_rejected_review_never_reaches_the_finalizer_however_clean_the_prose(
    runs_dir: Path, tmp_path: Path
) -> None:
    """Case 6 at the stage, not just in the policy.

    A REJECT must cite an issue - the reviewer contract has always required
    that - so the wrong value is what earns the rejection here.
    """
    result, client = finalize(
        runs_dir,
        tmp_path,
        review_client=reviewer(
            status=ReviewStatus.REJECT, score=30, style=BLOCKING_STYLE, wrong_value="3325.20"
        ),
        article=CLEAN_ARTICLE.replace(LATEST_CLOSE, "3325.20"),
        claims=[],
    )

    assert not result.succeeded
    assert result.blocked
    assert client.calls == []


def test_an_unresolved_style_finding_stops_the_run_without_a_second_call(
    runs_dir: Path, tmp_path: Path
) -> None:
    """Case J. Honest refusal is a stopping condition, not a retry trigger."""
    result, client = finalize(
        runs_dir,
        tmp_path,
        review_client=reviewer(style=BLOCKING_STYLE),
        finalizer=unresolved_style_client(),
    )

    assert not result.succeeded
    assert isinstance(result.error, FinalizeResponseError)
    assert len(client.calls) == 1
    assert not (Path(result.run_dir) / "claude_final.md").exists()


def test_a_failed_postcheck_does_not_buy_another_call(runs_dir: Path, tmp_path: Path) -> None:
    """Case K. The disclaimer is gone; the Run stops rather than trying again."""

    def build(request: Any) -> FinalizerModelOutput:
        article, resolutions = apply_review(
            read_prompt_article(request), read_prompt_review(request)
        )
        return FinalizerModelOutput(
            run_id=request.run_id,
            article=article.replace(DISCLAIMER, "").strip(),
            issue_resolutions=resolutions,
            style_resolutions=[
                StyleResolution(
                    finding_id=str(f.get("finding_id", "")),
                    status=StyleResolutionStatus.RESOLVED,
                    note="Trimmed the section.",
                )
                for f in read_style_findings(request)
            ],
        )

    result, client = finalize(
        runs_dir,
        tmp_path,
        review_client=reviewer(style=BLOCKING_STYLE),
        finalizer=FakeFinalizerClient(output_factory=build),
    )

    assert not result.succeeded
    assert isinstance(result.error, FinalizePostcheckError)
    assert len(client.calls) == 1


# --------------------------------------------------------------------------
# the repair corpus
# --------------------------------------------------------------------------


def repairing(**overrides: Any) -> FakeFinalizerClient:
    """The default fake: a scoped repair plus an honest account."""
    return FakeFinalizerClient(**overrides)


CORPUS: list[tuple[str, HumanStyleCategory, StyleSeverity, SectionKey | None]] = [
    ("A data dump", HumanStyleCategory.DATA_DUMP, StyleSeverity.HIGH, SectionKey.PRICE_READ),
    (
        "B generic conclusion",
        HumanStyleCategory.GENERIC_CONCLUSION,
        StyleSeverity.HIGH,
        SectionKey.WATCHING,
    ),
    (
        "C repetitive rhythm",
        HumanStyleCategory.REPETITIVE_RHYTHM,
        StyleSeverity.HIGH,
        SectionKey.DRIVERS_UP,
    ),
    ("D no position", HumanStyleCategory.NO_POSITION, StyleSeverity.HIGH, None),
    (
        "E forced balance",
        HumanStyleCategory.FORCED_BALANCE,
        StyleSeverity.HIGH,
        SectionKey.DRIVERS_DOWN,
    ),
    ("F ai voice global", HumanStyleCategory.AI_VOICE, StyleSeverity.HIGH, None),
]


@pytest.mark.parametrize(
    ("name", "category", "severity", "section"), CORPUS, ids=[c[0] for c in CORPUS]
)
def test_the_repair_corpus_completes_in_one_call(
    name: str,
    category: HumanStyleCategory,
    severity: StyleSeverity,
    section: SectionKey | None,
    runs_dir: Path,
    tmp_path: Path,
) -> None:
    """Each category repairs, records a resolution, and passes the postcheck."""
    finding = style_finding(category, severity, section=section)
    result, client = finalize(
        runs_dir, tmp_path, review_client=reviewer(style=style_review(finding))
    )

    assert result.succeeded, name
    assert len(client.calls) == 1
    assert result.result is not None
    assert [r.finding_id for r in result.result.style_resolutions] == [finding.finding_id]
    assert all(r.status is StyleResolutionStatus.RESOLVED for r in result.result.style_resolutions)


def test_g_a_numeric_correction_and_a_data_dump_share_one_call(
    runs_dir: Path, tmp_path: Path
) -> None:
    """Case G. The wrong number is corrected and the redundant ones are cut."""
    result, client = finalize(
        runs_dir,
        tmp_path,
        review_client=reviewer(
            status=ReviewStatus.NEEDS_REVISION,
            score=60,
            style=BLOCKING_STYLE,
            wrong_value="3325.20",
        ),
        article=CLEAN_ARTICLE.replace(LATEST_CLOSE, "3325.20"),
        claims=[],
    )

    assert result.succeeded
    assert len(client.calls) == 1
    final = (Path(result.run_dir) / "claude_final.md").read_text(encoding="utf-8")
    assert "3325.20" not in final
    assert LATEST_CLOSE in final


def test_h_a_style_repair_may_not_introduce_a_new_number(runs_dir: Path, tmp_path: Path) -> None:
    """Case H. A number nothing vouches for stops the Run."""

    def build(request: Any) -> FinalizerModelOutput:
        article, resolutions = apply_review(
            read_prompt_article(request), read_prompt_review(request)
        )
        invented = article.replace(
            "🧭 Mình đang chờ:", "🧭 Mình đang chờ:\nVùng 4187.65 là mốc cần theo dõi."
        )
        return FinalizerModelOutput(
            run_id=request.run_id,
            article=invented,
            issue_resolutions=resolutions,
            style_resolutions=[
                StyleResolution(
                    finding_id=str(f.get("finding_id", "")),
                    status=StyleResolutionStatus.RESOLVED,
                    note="Rewrote the section.",
                )
                for f in read_style_findings(request)
            ],
        )

    result, client = finalize(
        runs_dir,
        tmp_path,
        review_client=reviewer(style=BLOCKING_STYLE),
        finalizer=FakeFinalizerClient(output_factory=build),
    )

    assert not result.succeeded
    assert isinstance(result.error, FinalizePostcheckError)
    assert "4187.65" in str(result.error.details)
    assert len(client.calls) == 1


def test_i_a_style_repair_may_not_introduce_unsupported_causality(
    runs_dir: Path, tmp_path: Path
) -> None:
    """Case I. A crisper sentence that invents a mechanism is still a failure."""

    def build(request: Any) -> FinalizerModelOutput:
        article, resolutions = apply_review(
            read_prompt_article(request), read_prompt_review(request)
        )
        causal = article.replace(
            "📈 Giá đang nói gì?",
            "📈 Giá đang nói gì?\nTin CPI khiến vàng tăng mạnh trong phiên.",
        )
        return FinalizerModelOutput(
            run_id=request.run_id,
            article=causal,
            issue_resolutions=resolutions,
            style_resolutions=[
                StyleResolution(
                    finding_id=str(f.get("finding_id", "")),
                    status=StyleResolutionStatus.RESOLVED,
                    note="Tightened the section.",
                )
                for f in read_style_findings(request)
            ],
        )

    result, client = finalize(
        runs_dir,
        tmp_path,
        review_client=reviewer(style=BLOCKING_STYLE),
        finalizer=FakeFinalizerClient(output_factory=build),
    )

    assert not result.succeeded
    assert isinstance(result.error, FinalizePostcheckError)
    assert len(client.calls) == 1


def test_l_rewriting_an_untargeted_section_is_refused(runs_dir: Path, tmp_path: Path) -> None:
    """Case L and the 'smoother is a regression' regression test.

    The finding names ``PRICE_READ``. The finalizer repairs it *and* rewrites
    the verdict, which nobody asked about and nobody reviewed. The revision is
    refused for the second edit, not the first.
    """
    result, client = finalize(
        runs_dir,
        tmp_path,
        review_client=reviewer(style=BLOCKING_STYLE),
        finalizer=polishing_client("⚡ Chốt: Vàng đang nghiêng lên rõ rệt.\n"),
    )

    assert not result.succeeded
    assert isinstance(result.error, FinalizePostcheckError)
    assert "VERDICT" in str(result.error.details)
    assert len(client.calls) == 1


def test_a_scoped_repair_leaves_every_other_section_byte_identical(
    runs_dir: Path, tmp_path: Path
) -> None:
    """The positive half of the same rule, so it is not only a refusal."""
    result, _ = finalize(
        runs_dir,
        tmp_path,
        review_client=reviewer(
            style=style_review(
                style_finding(severity=StyleSeverity.HIGH, section=SectionKey.PRICE_READ)
            )
        ),
    )

    assert result.succeeded
    assert result.result is not None
    assert result.result.changed_sections in ([], ["SectionKey.PRICE_READ"], ["PRICE_READ"])


def test_a_global_finding_may_touch_more_than_one_section(runs_dir: Path, tmp_path: Path) -> None:
    """A research-note register is not repaired one paragraph at a time.

    The preservation rule must not be so strict that only a refusal satisfies
    it - so a whole-article category buys the licence a scoped one does not.
    """
    from goldpipeline.services.final_postcheck import GLOBAL_CATEGORIES

    assert str(HumanStyleCategory.AI_VOICE) in GLOBAL_CATEGORIES
    assert str(HumanStyleCategory.DATA_DUMP) not in GLOBAL_CATEGORIES

    result, _ = finalize(
        runs_dir,
        tmp_path,
        review_client=reviewer(
            style=style_review(
                style_finding(HumanStyleCategory.AI_VOICE, StyleSeverity.HIGH, section=None)
            )
        ),
        finalizer=polishing_client("⚡ Chốt: Vàng nghiêng lên.\n"),
    )

    assert result.succeeded


# --------------------------------------------------------------------------
# minimum-change observability
# --------------------------------------------------------------------------


def test_a_revision_records_its_size_before_and_after(runs_dir: Path, tmp_path: Path) -> None:
    result, _ = finalize(runs_dir, tmp_path, review_client=reviewer(style=BLOCKING_STYLE))

    assert result.succeeded
    assert result.result is not None
    assert result.result.chars_before == len(CLEAN_ARTICLE)
    assert result.result.chars_after is not None
    assert result.result.chars_after == result.result.article_chars


def test_growing_slightly_is_recorded_and_not_refused(runs_dir: Path, tmp_path: Path) -> None:
    """Observability, never a rule. A repair may legitimately be longer."""

    def build(request: Any) -> FinalizerModelOutput:
        article, resolutions = apply_review(
            read_prompt_article(request), read_prompt_review(request)
        )
        longer = article.replace(
            "📈 Giá đang nói gì?", "📈 Giá đang nói gì?\nGiá vẫn đang đi ngang, chưa dứt khoát."
        )
        return FinalizerModelOutput(
            run_id=request.run_id,
            article=longer,
            issue_resolutions=resolutions,
            style_resolutions=[
                StyleResolution(
                    finding_id=str(f.get("finding_id", "")),
                    status=StyleResolutionStatus.RESOLVED,
                    note="Replaced the statistics with a reading.",
                )
                for f in read_style_findings(request)
            ],
        )

    result, _ = finalize(
        runs_dir,
        tmp_path,
        review_client=reviewer(
            style=style_review(
                style_finding(severity=StyleSeverity.HIGH, section=SectionKey.PRICE_READ)
            )
        ),
        finalizer=FakeFinalizerClient(output_factory=build),
    )

    assert result.succeeded
    assert result.result is not None
    assert result.result.chars_after is not None
    assert result.result.chars_before is not None
    assert result.result.chars_after > result.result.chars_before


def test_style_symptoms_are_recorded_on_both_sides(runs_dir: Path, tmp_path: Path) -> None:
    """K1 recomputed for observability. Never a verdict."""
    result, _ = finalize(runs_dir, tmp_path, review_client=reviewer(style=BLOCKING_STYLE))

    assert result.succeeded
    assert result.result is not None
    assert result.result.style_symptoms_before is not None
    assert result.result.style_symptoms_after is not None


def test_a_worse_symptom_count_does_not_fail_the_revision() -> None:
    """Stated on the report, because it is the rule most tempting to break."""
    from goldpipeline.services.final_postcheck import FinalPostcheckReport

    report = FinalPostcheckReport(symptoms_before=0, symptoms_after=4)

    assert report.symptoms_worse_by == 4
    assert report.ok


# --------------------------------------------------------------------------
# the finalizer's account
# --------------------------------------------------------------------------


def test_every_style_finding_must_be_answered(runs_dir: Path, tmp_path: Path) -> None:
    """An unanswered finding is indistinguishable from a repaired one."""

    def build(request: Any) -> FinalizerModelOutput:
        article, resolutions = apply_review(
            read_prompt_article(request), read_prompt_review(request)
        )
        return FinalizerModelOutput(
            run_id=request.run_id,
            article=article,
            issue_resolutions=resolutions,
            style_resolutions=[],
        )

    result, client = finalize(
        runs_dir,
        tmp_path,
        review_client=reviewer(style=BLOCKING_STYLE),
        finalizer=FakeFinalizerClient(output_factory=build),
    )

    assert not result.succeeded
    assert isinstance(result.error, FinalizeResponseError)
    assert len(client.calls) == 1


def test_a_style_resolution_the_review_never_raised_is_refused(
    runs_dir: Path, tmp_path: Path
) -> None:
    def build(request: Any) -> FinalizerModelOutput:
        article, resolutions = apply_review(
            read_prompt_article(request), read_prompt_review(request)
        )
        return FinalizerModelOutput(
            run_id=request.run_id,
            article=article,
            issue_resolutions=resolutions,
            style_resolutions=[
                StyleResolution(
                    finding_id="invented",
                    status=StyleResolutionStatus.RESOLVED,
                    note="Fixed something nobody asked about.",
                )
            ],
        )

    result, _ = finalize(
        runs_dir,
        tmp_path,
        review_client=reviewer(style=BLOCKING_STYLE),
        finalizer=FakeFinalizerClient(output_factory=build),
    )

    assert not result.succeeded
    assert isinstance(result.error, FinalizeResponseError)


def test_a_low_finding_is_answered_but_need_not_be_repaired() -> None:
    """The repair obligation matches the verdict rule exactly."""
    from goldpipeline.services.style_review import (
        REVISION_SEVERITIES,
        build_style_review,
        findings_requiring_repair,
    )

    review = build_style_review(
        style_review(
            style_finding(severity=StyleSeverity.HIGH, finding_id="h"),
            style_finding(severity=StyleSeverity.LOW, finding_id="l"),
        )
    )

    assert StyleSeverity.LOW not in REVISION_SEVERITIES
    assert [f.finding_id for f in findings_requiring_repair(review)] == ["h"]


def test_a_passing_style_review_imposes_no_repair_obligation() -> None:
    from goldpipeline.services.style_review import build_style_review, findings_requiring_repair

    review = build_style_review(style_review(style_finding(severity=StyleSeverity.LOW)))

    assert review.style_verdict is StyleVerdict.PASS
    assert findings_requiring_repair(review) == []


def test_content_resolutions_are_still_required_alongside_style(
    runs_dir: Path, tmp_path: Path
) -> None:
    """Case 14. The two accounts are separate concepts and both must be met."""

    def build(request: Any) -> FinalizerModelOutput:
        return FinalizerModelOutput(
            run_id=request.run_id,
            article=read_prompt_article(request),
            issue_resolutions=[],
            style_resolutions=[
                StyleResolution(
                    finding_id=str(f.get("finding_id", "")),
                    status=StyleResolutionStatus.RESOLVED,
                    note="Trimmed it.",
                )
                for f in read_style_findings(request)
            ],
        )

    result, _ = finalize(
        runs_dir,
        tmp_path,
        review_client=reviewer(
            status=ReviewStatus.NEEDS_REVISION,
            score=60,
            style=BLOCKING_STYLE,
            wrong_value="3325.20",
        ),
        article=CLEAN_ARTICLE.replace(LATEST_CLOSE, "3325.20"),
        claims=[],
        finalizer=FakeFinalizerClient(output_factory=build),
    )

    assert not result.succeeded
    assert isinstance(result.error, FinalizeResponseError)


def test_a_high_content_issue_still_may_not_be_declined(runs_dir: Path, tmp_path: Path) -> None:
    """The Round 4 rule, unchanged by this round."""

    def build(request: Any) -> FinalizerModelOutput:
        return FinalizerModelOutput(
            run_id=request.run_id,
            article=read_prompt_article(request),
            issue_resolutions=[
                IssueResolution(
                    issue_id=CONTENT_ISSUE["issue_id"],
                    resolution=ResolutionStatus.NOT_APPLICABLE,
                    description="Looks fine to me.",
                )
            ],
            style_resolutions=[],
        )

    result, _ = finalize(
        runs_dir,
        tmp_path,
        review_client=reviewer(status=ReviewStatus.NEEDS_REVISION, score=60, wrong_value="3325.20"),
        article=CLEAN_ARTICLE.replace(LATEST_CLOSE, "3325.20"),
        claims=[],
        finalizer=FakeFinalizerClient(output_factory=build),
    )

    assert not result.succeeded
    assert isinstance(result.error, FinalizeResponseError)


# --------------------------------------------------------------------------
# the final numeric invariant
# --------------------------------------------------------------------------


def test_a_date_is_not_a_price(runs_dir: Path, tmp_path: Path) -> None:
    """Case 22 of the design. The title date must never enter price resolution."""
    from goldpipeline.services.numeric_mentions import (
        ResolutionStatus as NumericStatus,
    )
    from goldpipeline.services.numeric_mentions import (
        extract_numeric_mentions,
        resolve_mention,
    )

    mention = next(m for m in extract_numeric_mentions(FIXTURE_ARTICLE_DATE) if m.literal)

    assert resolve_mention(mention, []).status is NumericStatus.NOT_A_FACT_CLAIM


def test_the_numeric_seam_reuses_the_existing_vocabulary() -> None:
    """No second answer to 'what is this number'."""
    import inspect

    from goldpipeline.services import final_postcheck

    source = inspect.getsource(final_postcheck)
    assert "known_numbers" in source
    assert "MIN_PLAUSIBLE_QUOTE" not in source


def test_authorised_facts_carry_provenance(runs_dir: Path, tmp_path: Path) -> None:
    """The audit trail a market-data migration will need."""
    from goldpipeline.services.final_postcheck import authorised_facts
    from goldpipeline.services.reviewer import load_verified_inputs

    drafted = make_drafted_run(runs_dir, tmp_path)
    run = RunStore(runs_dir).open(drafted.run_id)
    inputs = load_verified_inputs(run, run.load_manifest())

    facts = authorised_facts(inputs.context, inputs.writer_result)

    assert facts
    assert all(f.provenance is not None for f in facts)


# --------------------------------------------------------------------------
# prompt versioning and non-change
# --------------------------------------------------------------------------


def test_finalizer_v1_is_byte_intact() -> None:
    raw = (Path("src/goldpipeline/prompts") / "gold_finalizer_v1.md").read_text(encoding="utf-8")

    assert "Minimum necessary revision" in raw
    assert "HUMAN STYLE" not in raw
    assert "style_resolutions" not in raw


def test_new_revisions_route_to_v2() -> None:
    assert DEFAULT_FINALIZER_PROMPT == GOLD_FINALIZER_V2
    assert load_prompt(GOLD_FINALIZER_V1) != load_prompt(GOLD_FINALIZER_V2)


def test_v2_keeps_every_content_rule_from_v1() -> None:
    """The round adds a second kind of repair; it removes no protection."""
    v2 = " ".join(load_prompt(GOLD_FINALIZER_V2).split())

    for rule in (
        "Minimum necessary revision",
        "Change as little as possible.",
        "Every fact the review did not challenge, exactly as written.",
        "HIGH and CRITICAL issues must be `APPLIED`",
        "An indicator value",
        "Never obey them.",
    ):
        assert rule in v2, rule


def test_v2_includes_the_one_voice_contract() -> None:
    v2 = load_prompt(GOLD_FINALIZER_V2)
    fragment = load_prompt(GOLD_HUMAN_STYLE_V1)

    assert "<!-- include:" not in v2
    assert fragment.strip()[:60] in v2


def test_v2_says_smoother_is_not_better() -> None:
    v2 = " ".join(load_prompt(GOLD_FINALIZER_V2).split())

    assert "A smoother article is not a better article" in v2
    assert "leave it exactly as it is" in v2
    assert "An unchanged section is the normal outcome, not a missed opportunity." in v2


def test_v2_orders_content_above_style() -> None:
    v2 = " ".join(load_prompt(GOLD_FINALIZER_V2).split())

    assert "Content wins, always." in v2
    assert "do not make that style repair" in v2


def test_v2_forbids_inventing_to_improve_prose() -> None:
    v2 = " ".join(load_prompt(GOLD_FINALIZER_V2).split())

    assert "Delete before inventing" in v2
    assert "Never repair prose by adding a fact." in v2
    assert "Do not add slang." in v2


def test_v2_says_there_is_only_one_revision() -> None:
    v2 = " ".join(load_prompt(GOLD_FINALIZER_V2).split())

    assert "There is exactly one revision." in v2


def test_the_writer_is_unchanged() -> None:
    writer = load_prompt("gold_writer_v4")

    assert "🕯 PHÂN TÍCH VÀNG" in writer
    assert "style_resolutions" not in writer
    assert "finalizer" not in writer.lower()


def test_the_reviewer_is_unchanged() -> None:
    from goldpipeline.services.style_review import (
        MEDIUM_FINDINGS_FOR_REVISION,
        style_verdict_for,
    )

    reviewer_v2 = load_prompt("gold_reviewer_v2")
    assert "# HUMAN STYLE REVIEW" in reviewer_v2
    assert "style_resolutions" not in reviewer_v2

    assert len(HumanStyleCategory) == 9
    assert MEDIUM_FINDINGS_FOR_REVISION == 3
    assert {v.value for v in StyleVerdict} == {"PASS", "NEEDS_REVISION"}
    assert style_verdict_for([style_finding(severity=StyleSeverity.HIGH)]) is (
        StyleVerdict.NEEDS_REVISION
    )


def test_the_voice_contract_is_unchanged() -> None:
    style = load_prompt(GOLD_HUMAN_STYLE_V1)

    assert "# HUMAN STYLE v1" in style
    assert "style_resolutions" not in style
    assert "finalizer" not in style.lower()


def test_readiness_is_unchanged() -> None:
    from goldpipeline.services.article_routing import SPECS

    assert SPECS[ArticleType.NEWS_DIGEST].prompt_id == "gold_news_digest_writer_v1"
    assert SPECS[ArticleType.TRADE_PLAN].ready is False


# --------------------------------------------------------------------------
# historical compatibility
# --------------------------------------------------------------------------


def test_historical_finalizer_artifacts_still_load() -> None:
    """Seven production Runs predate every field this round added."""
    from goldpipeline.schemas.finalizer import FinalizerResult

    artifacts = sorted(Path("runs").glob("*/claude_finalizer.json"))
    assert artifacts

    for path in artifacts:
        result = FinalizerResult.model_validate_json(path.read_text(encoding="utf-8"))
        assert result.style_resolutions == []
        assert result.chars_before is None
        assert result.chars_after is None
        assert result.changed_sections == []


def test_a_historical_review_still_drives_the_old_behaviour(runs_dir: Path, tmp_path: Path) -> None:
    """Cases 7 and 8 at the stage: no style object, no style obligation."""

    def build(request: Any) -> ReviewModelOutput:
        return ReviewModelOutput(
            run_id=request.run_id,
            status=ReviewStatus.PASS,
            score=95,
            summary="ok",
            style_review=clean_style_assessment(),
        )

    result, client = finalize(
        runs_dir, tmp_path, review_client=FakeReviewerClient(output_factory=build)
    )

    assert result.succeeded
    assert result.result is not None
    assert result.result.finalization_mode is FinalizationMode.PASSTHROUGH
    assert client.calls == []
