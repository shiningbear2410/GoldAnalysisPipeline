"""The human-style axis, and the proof that it decides nothing yet.

Round 6.4f. The reviewer now judges writing as well as truth, and the two
judgements are kept apart on purpose: the finalizer has not been taught to
repair prose, so a style verdict that could reach the pipeline would hand a
factually sound article to a stage whose instructions are about wrong numbers.

Two kinds of test here, and the second kind matters more.

* The **corpus** covers the writing failures the reviewer is meant to name -
  one case per category, plus the cases where the right answer is to say
  nothing. Every one is offline, built from constructed reviewer output rather
  than from a model, because what can be pinned deterministically is the
  vocabulary, the verdict rule and the instructions in the prompt - not whether
  a particular model picks a particular label on a particular day.

* The **shadow-mode invariants** prove the connection is absent. They vary the
  style judgement across its whole range and assert that the content verdict,
  the pipeline transition, the finalizer's input and the gate cannot tell the
  difference. Those are the tests that would fail the day somebody wires this
  up early.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import CLEAN_ARTICLE, make_drafted_run, make_reviewed_run

from goldpipeline.adapters.fake_reviewer import (
    FakeReviewerClient,
    clean_style_assessment,
    style_in_scope,
)
from goldpipeline.domain.errors import ReviewResponseError
from goldpipeline.prompts import (
    DEFAULT_REVIEWER_PROMPT,
    GOLD_HUMAN_STYLE_V1,
    GOLD_REVIEWER_V1,
    GOLD_REVIEWER_V2,
    load_prompt,
)
from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.article_contract import HUMAN_STYLE_TYPES, SectionKey
from goldpipeline.schemas.review import (
    HumanStyleAssessment,
    HumanStyleCategory,
    HumanStyleFinding,
    HumanStyleReview,
    IssueCategory,
    ReviewModelOutput,
    ReviewResult,
    ReviewStatus,
    StyleSeverity,
    StyleVerdict,
)
from goldpipeline.services.reviewer import REVIEW_FILENAME, review_draft
from goldpipeline.services.style_review import (
    MEDIUM_FINDINGS_FOR_REVISION,
    STYLE_AWARE_PROMPTS,
    applies_to,
    build_style_review,
    requires_style_review,
    resolve_style_review,
    style_verdict_for,
)
from goldpipeline.storage.run_store import RunStore

# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def finding(
    category: HumanStyleCategory,
    severity: StyleSeverity = StyleSeverity.MEDIUM,
    *,
    finding_id: str | None = None,
    section: SectionKey | None = None,
    problem: str = "The section restates the verdict instead of adding to it.",
    repair: str = "Delete the repeated sentence.",
) -> HumanStyleFinding:
    """One style finding, with everything a Round 6.4g repair would need."""
    return HumanStyleFinding(
        finding_id=finding_id or f"style-{category.lower()}",
        category=category,
        severity=severity,
        section=section,
        problem=problem,
        repair_instruction=repair,
    )


def assessment(
    *findings: HumanStyleFinding, score: int = 78, summary: str = "Đọc hơi cứng."
) -> HumanStyleAssessment:
    return HumanStyleAssessment(style_score=score, summary=summary, findings=list(findings))


def reviewer_returning(
    *,
    status: ReviewStatus = ReviewStatus.PASS,
    score: int = 95,
    style: HumanStyleAssessment | None = None,
    omit_style: bool = False,
    issues: list[Any] | None = None,
    instructions: list[str] | None = None,
) -> FakeReviewerClient:
    """A reviewer whose two axes can be set independently.

    The point of the signature: content and style are separate arguments, so a
    test can hold one still and move the other. That is the whole experiment.
    """

    def build(request: Any) -> ReviewModelOutput:
        resolved: HumanStyleAssessment | None
        if omit_style:
            resolved = None
        elif style is not None:
            resolved = style
        else:
            resolved = clean_style_assessment() if style_in_scope(request) else None
        return ReviewModelOutput(
            run_id=request.run_id,
            status=status,
            score=score,
            summary="Kiểm tra nội dung hoàn tất.",
            issues=list(issues or []),
            revision_instructions=list(instructions or []),
            style_review=resolved,
        )

    return FakeReviewerClient(output_factory=build)


def review_of(runs_dir: Path, tmp_path: Path, client: FakeReviewerClient) -> ReviewResult:
    """Drive the real reviewer stage and return the committed artifact."""
    reviewed = make_reviewed_run(runs_dir, tmp_path, review_client=client)
    result = reviewed.result
    assert isinstance(result, ReviewResult)
    return result


# --------------------------------------------------------------------------
# 1-11: the corpus, one case per writing failure
# --------------------------------------------------------------------------


CORPUS: list[tuple[str, list[HumanStyleFinding], StyleVerdict]] = [
    # 1. a natural, good article: the correct answer is nothing at all.
    ("natural good analysis", [], StyleVerdict.PASS),
    # 2. reads like a wire report.
    (
        "news-desk article",
        [finding(HumanStyleCategory.NEWS_DESK_VOICE, StyleSeverity.HIGH)],
        StyleVerdict.NEEDS_REVISION,
    ),
    # 3. bank-research register: the same category, milder.
    (
        "bank-research register",
        [finding(HumanStyleCategory.NEWS_DESK_VOICE, StyleSeverity.MEDIUM)],
        StyleVerdict.PASS,
    ),
    # 4. every sentence the same shape.
    (
        "repetitive rhythm",
        [finding(HumanStyleCategory.REPETITIVE_RHYTHM, StyleSeverity.MEDIUM)],
        StyleVerdict.PASS,
    ),
    # 5. connective-heavy prose: material only when it is material.
    (
        "connective heavy",
        [finding(HumanStyleCategory.AI_VOICE, StyleSeverity.MEDIUM)],
        StyleVerdict.PASS,
    ),
    # 6. correct but padded.
    (
        "verbose but factual",
        [finding(HumanStyleCategory.VERBOSITY, StyleSeverity.MEDIUM)],
        StyleVerdict.PASS,
    ),
    # 7. the price section as a statistics table.
    (
        "price data dump",
        [
            finding(
                HumanStyleCategory.DATA_DUMP, StyleSeverity.MEDIUM, section=SectionKey.PRICE_READ
            )
        ],
        StyleVerdict.PASS,
    ),
    # 8. an ending that would fit any asset.
    (
        "generic conclusion",
        [finding(HumanStyleCategory.GENERIC_CONCLUSION, StyleSeverity.MEDIUM)],
        StyleVerdict.PASS,
    ),
    # 9. never commits to anything.
    (
        "no position",
        [finding(HumanStyleCategory.NO_POSITION, StyleSeverity.HIGH)],
        StyleVerdict.NEEDS_REVISION,
    ),
    # 10. a side invented to fill a heading.
    (
        "forced balance",
        [finding(HumanStyleCategory.FORCED_BALANCE, StyleSeverity.HIGH)],
        StyleVerdict.NEEDS_REVISION,
    ),
    # 11. an honest one-sided day: nothing to report.
    ("one-sided honest article", [], StyleVerdict.PASS),
]


@pytest.mark.parametrize(("name", "findings", "expected"), CORPUS, ids=[c[0] for c in CORPUS])
def test_the_corpus_derives_the_expected_style_verdict(
    name: str, findings: list[HumanStyleFinding], expected: StyleVerdict
) -> None:
    """Every corpus case, through the real derivation."""
    assert style_verdict_for(findings) is expected, name


@pytest.mark.parametrize(("name", "findings", "expected"), CORPUS, ids=[c[0] for c in CORPUS])
def test_every_corpus_case_round_trips_through_the_artifact(
    name: str, findings: list[HumanStyleFinding], expected: StyleVerdict
) -> None:
    """The vocabulary can express each case and survive serialization."""
    review = build_style_review(assessment(*findings))
    restored = HumanStyleReview.model_validate_json(review.model_dump_json())

    assert restored.style_verdict is expected
    assert [f.category for f in restored.findings] == [f.category for f in findings]


def test_a_good_article_may_return_no_findings_at_all() -> None:
    """Case 1 and 11, stated as the property they are really about.

    There is no quota. An empty finding list is a complete answer, and a
    reviewer that could not produce one would make the axis worthless.
    """
    review = build_style_review(
        HumanStyleAssessment(style_score=94, summary="Đọc tự nhiên.", findings=[])
    )

    assert review.findings == []
    assert review.style_verdict is StyleVerdict.PASS
    assert review.style_score == 94


def test_an_honest_one_sided_article_is_not_forced_balance() -> None:
    """Case 11 explicitly: asymmetry is correct reporting, never a style fault.

    The prompt carries the rule; this pins that it is there, because the day it
    is dropped the reviewer starts punishing the writer for obeying Round 6.4e.
    """
    system = load_prompt(GOLD_REVIEWER_V2)

    assert "one-sided day honestly reported is" in system
    assert "must never be flagged as unbalanced" in system


# --------------------------------------------------------------------------
# 12-14: length is not a style rule
# --------------------------------------------------------------------------


def test_a_short_quiet_day_article_is_never_penalised() -> None:
    """Case 12. Below the target is correct on a quiet day, not a finding."""
    system = load_prompt(GOLD_REVIEWER_V2)

    assert "Never raise a finding for brevity." in system
    assert "short piece beats padding" in system


def test_crossing_the_target_is_not_automatically_a_finding() -> None:
    """Case 13. 1050-1100 chars sits over target_max and under the hard cap.

    The deterministic contract does not block it, and the reviewer must not
    turn a guidance range into a hidden ceiling.
    """
    system = load_prompt(GOLD_REVIEWER_V2)

    assert "not thereby a problem" in system
    assert "Judge the prose, never the character count." in system


def test_bloat_is_judged_as_prose_not_as_length() -> None:
    """Case 14. Under the cap and still padded is a real VERBOSITY finding.

    The category exists and carries a repair; what must not exist is a rule
    that derives it from a character count.
    """
    verbose = finding(
        HumanStyleCategory.VERBOSITY,
        StyleSeverity.MEDIUM,
        problem="The drivers section explains what a weaker dollar means for gold twice.",
        repair="Delete the second explanation.",
    )

    assert verbose.category is HumanStyleCategory.VERBOSITY
    assert "Delete" in verbose.repair_instruction


# --------------------------------------------------------------------------
# 15-18: which axis owns which defect
# --------------------------------------------------------------------------


def test_supported_but_excessive_numbers_are_style_not_content(
    runs_dir: Path, tmp_path: Path
) -> None:
    """Case 15. Content passes; style may still call it a data dump."""
    style = assessment(
        finding(HumanStyleCategory.DATA_DUMP, StyleSeverity.MEDIUM, section=SectionKey.PRICE_READ)
    )
    result = review_of(runs_dir, tmp_path, reviewer_returning(style=style))

    assert result.status is ReviewStatus.PASS
    assert result.issues == []
    assert result.style_review is not None
    assert result.style_review.findings[0].category is HumanStyleCategory.DATA_DUMP


def test_the_prompt_gives_unsupported_numbers_to_content_only() -> None:
    """Case 16. A numeric defect is not also a style finding for being numeric."""
    system = load_prompt(GOLD_REVIEWER_V2)

    assert "If a number is unsupported, that is a **content** issue." in system
    assert "Do not also file `DATA_DUMP` because it was numeric." in unwrapped(GOLD_REVIEWER_V2)


def test_the_prompt_gives_unsupported_causality_to_content_only() -> None:
    """Case 17. A confident wrong sentence is a content issue, not AI_VOICE."""
    system = load_prompt(GOLD_REVIEWER_V2)

    assert "unsupported causal claim, that is a **content** issue" in system
    assert "One defect, one axis, whichever owns it." in system


def test_allowed_temporal_language_produces_no_finding_on_either_axis(
    runs_dir: Path, tmp_path: Path
) -> None:
    """Case 18. The Round 6.4e wording the writer is told to use stays clean."""
    result = review_of(runs_dir, tmp_path, reviewer_returning())

    assert result.status is ReviewStatus.PASS
    assert result.issues == []
    assert result.style_review is not None
    assert result.style_review.findings == []


# --------------------------------------------------------------------------
# 19-20: deterministic symptoms are hints, not judgements
# --------------------------------------------------------------------------


def test_style_findings_are_possible_with_no_deterministic_symptom(
    runs_dir: Path, tmp_path: Path
) -> None:
    """Case 19. K1 silent, K2 speaks.

    The fixture article produces no symptoms; the reviewer still reports
    `AI_VOICE`. If the pipeline ever derived style findings from symptoms, this
    would be impossible - which is the point of asserting it.
    """
    style = assessment(finding(HumanStyleCategory.AI_VOICE, StyleSeverity.MEDIUM))
    result = review_of(runs_dir, tmp_path, reviewer_returning(style=style))

    assert result.style_review is not None
    assert result.style_review.findings[0].category is HumanStyleCategory.AI_VOICE


def test_a_deterministic_symptom_need_not_become_a_style_finding(
    runs_dir: Path, tmp_path: Path
) -> None:
    """Case 20. K1 speaks, K2 declines - and that is a valid review.

    An article carrying a countable symptom can read perfectly naturally. The
    reviewer is allowed to look and disagree, and nothing in the pipeline
    promotes a symptom into a finding on its own.
    """
    repetitive = "\n\n".join([f"Câu số {n} dài đúng bằng câu trước." for n in range(1, 7)])
    article = CLEAN_ARTICLE + "\n\n" + repetitive

    drafted = make_drafted_run(runs_dir, tmp_path, article=article, enforce_contract=False)
    reviewed = review_draft(
        run_id=drafted.run_id,
        store=RunStore(runs_dir),
        client=reviewer_returning(),
    )

    assert reviewed.succeeded
    assert reviewed.result is not None
    assert reviewed.result.style_review is not None
    assert reviewed.result.style_review.findings == []
    assert reviewed.result.style_review.style_verdict is StyleVerdict.PASS


def test_the_prompt_states_the_hint_rule_in_both_directions() -> None:
    """Neither direction may be collapsed: hints do not create or suppress."""
    system = load_prompt(GOLD_REVIEWER_V2)

    assert "hints, not verdicts" in system
    assert "raise a finding just because a symptom was listed" in system
    assert "zero symptoms and still sound like a research desk" in system


def test_symptoms_reach_the_prompt_labelled_as_hints(runs_dir: Path, tmp_path: Path) -> None:
    """The user turn must carry them, and must carry the caveat with them."""
    client = reviewer_returning()
    make_reviewed_run(runs_dir, tmp_path, review_client=client)
    user = client.calls[0].prompt.user

    assert "# DETERMINISTIC STYLE SYMPTOMS" in user
    assert "hints" in user
    assert "Do not raise a style finding merely because a" in user


# --------------------------------------------------------------------------
# 21: article-type scope
# --------------------------------------------------------------------------


def test_a_deterministic_trade_plan_gets_no_style_review() -> None:
    """Case 21. A rendered document has no voice, so there is nothing to judge."""
    assert not applies_to(ArticleType.TRADE_PLAN)
    assert ArticleType.TRADE_PLAN not in HUMAN_STYLE_TYPES
    assert not requires_style_review(
        prompt_version=GOLD_REVIEWER_V2, article_type=ArticleType.TRADE_PLAN
    )


def test_a_volunteered_style_review_for_a_trade_plan_is_dropped() -> None:
    """Recording a verdict nobody was asked for would be a fabricated judgement."""
    output = ReviewModelOutput(
        run_id="r1",
        status=ReviewStatus.PASS,
        score=95,
        summary="ok",
        style_review=clean_style_assessment(),
    )

    resolved = resolve_style_review(
        output, prompt_version=GOLD_REVIEWER_V2, article_type=ArticleType.TRADE_PLAN
    )

    assert resolved is None


def test_analysis_is_in_scope_and_the_digest_will_be() -> None:
    """The digest is not producible yet, and is already reviewable when it is."""
    assert applies_to(ArticleType.ANALYSIS)
    assert applies_to(ArticleType.NEWS_DIGEST)


def test_the_prompt_says_out_of_scope_out_loud(runs_dir: Path, tmp_path: Path) -> None:
    """Silence about scope would leave the model guessing what to omit.

    Both turns of the same builder, one Run, two article types - so the only
    difference between the two prompts is the thing under test.
    """
    from goldpipeline.services.reviewer import load_verified_inputs
    from goldpipeline.services.reviewer_prompt import build_reviewer_prompt

    drafted = make_drafted_run(runs_dir, tmp_path)
    run = RunStore(runs_dir).open(drafted.run_id)
    inputs = load_verified_inputs(run, run.load_manifest())

    def rendered(article_type: ArticleType) -> str:
        return build_reviewer_prompt(
            context=inputs.context,
            writer_result=inputs.writer_result,
            article=inputs.article,
            report=_empty_report(),
            prompt_version=GOLD_REVIEWER_V2,
            article_type=article_type,
        ).user

    assert "Human style **is** in scope" in rendered(ArticleType.ANALYSIS)

    out = rendered(ArticleType.TRADE_PLAN)
    assert "Human style is **not** in scope" in out
    assert "Omit `style_review` entirely." in out
    assert "# DETERMINISTIC STYLE SYMPTOMS" not in out


# --------------------------------------------------------------------------
# 22-23: schema compatibility, both directions
# --------------------------------------------------------------------------


def test_a_historical_v1_review_loads_with_no_style_review() -> None:
    """Case 22. Seven production reviews predate this field.

    Read from the repository rather than from a hand-written fixture: a
    compatibility claim about real artifacts should be tested against the real
    artifacts.
    """
    runs = Path("runs")
    reviews = sorted(runs.glob("*/gpt_review.json"))
    assert reviews, "no historical reviews found to check compatibility against"

    for path in reviews:
        result = ReviewResult.model_validate_json(path.read_text(encoding="utf-8"))
        assert result.style_review is None, path
        assert result.prompt_version == GOLD_REVIEWER_V1, path
        assert result.schema_version == "1.0.0", path


def test_a_v2_analysis_review_without_a_style_review_is_refused() -> None:
    """Case 23. The one place the axis touches production, and it is deliberate.

    A missing object is a plumbing failure - the prompt asked and the model did
    not answer - not a judgement about the article. Accepting it silently would
    mean the axis quietly stops being computed and nobody learns until a later
    round depends on it.
    """
    output = ReviewModelOutput(
        run_id="r1", status=ReviewStatus.PASS, score=95, summary="ok", style_review=None
    )

    with pytest.raises(ReviewResponseError) as caught:
        resolve_style_review(
            output, prompt_version=GOLD_REVIEWER_V2, article_type=ArticleType.ANALYSIS
        )

    assert "style_review" in caught.value.message


def test_a_v1_review_of_an_analysis_is_not_required_to_carry_style() -> None:
    """An older prompt was never asked, so its silence is not a failure."""
    output = ReviewModelOutput(
        run_id="r1", status=ReviewStatus.PASS, score=95, summary="ok", style_review=None
    )

    assert (
        resolve_style_review(
            output, prompt_version=GOLD_REVIEWER_V1, article_type=ArticleType.ANALYSIS
        )
        is None
    )


def test_the_new_field_is_additive_on_the_artifact(runs_dir: Path, tmp_path: Path) -> None:
    """`gpt_review.json` keeps its filename, its fields and its readers."""
    result = review_of(runs_dir, tmp_path, reviewer_returning())
    payload = json.loads(result.model_dump_json())

    for field in ("run_id", "status", "score", "summary", "issues", "revision_instructions"):
        assert field in payload
    assert payload["style_review"]["style_verdict"] == "PASS"


# --------------------------------------------------------------------------
# 24-25: THE SHADOW-MODE INVARIANT
# --------------------------------------------------------------------------


SHADOW_CASES: list[tuple[str, HumanStyleAssessment]] = [
    ("clean style", clean_style_assessment()),
    ("one LOW", assessment(finding(HumanStyleCategory.VERBOSITY, StyleSeverity.LOW))),
    (
        "three MEDIUM",
        assessment(
            finding(HumanStyleCategory.AI_VOICE, StyleSeverity.MEDIUM, finding_id="a"),
            finding(HumanStyleCategory.VERBOSITY, StyleSeverity.MEDIUM, finding_id="b"),
            finding(HumanStyleCategory.DATA_DUMP, StyleSeverity.MEDIUM, finding_id="c"),
        ),
    ),
    (
        "one HIGH",
        assessment(finding(HumanStyleCategory.NEWS_DESK_VOICE, StyleSeverity.HIGH), score=41),
    ),
]


@pytest.mark.parametrize(("name", "style"), SHADOW_CASES, ids=[c[0] for c in SHADOW_CASES])
def test_style_never_moves_a_content_pass(
    name: str, style: HumanStyleAssessment, runs_dir: Path, tmp_path: Path
) -> None:
    """Case 24, across the whole style range.

    Content PASS stays PASS whatever style says - including a HIGH finding that
    derives a style NEEDS_REVISION. The verdict, the score and the instruction
    list are identical in every case.
    """
    result = review_of(runs_dir, tmp_path, reviewer_returning(style=style))

    assert result.status is ReviewStatus.PASS
    assert result.model_status is ReviewStatus.PASS
    assert result.score == 95
    assert result.issues == []
    assert result.revision_instructions == []
    assert result.style_review is not None


def test_a_style_needs_revision_still_leaves_a_content_pass(runs_dir: Path, tmp_path: Path) -> None:
    """The headline case, stated on its own so a failure names itself."""
    style = assessment(finding(HumanStyleCategory.NO_POSITION, StyleSeverity.HIGH), score=38)
    result = review_of(runs_dir, tmp_path, reviewer_returning(style=style))

    assert result.style_review is not None
    assert result.style_review.style_verdict is StyleVerdict.NEEDS_REVISION
    assert result.status is ReviewStatus.PASS


def test_the_content_verdict_is_byte_identical_across_style_judgements(
    runs_dir: Path, tmp_path: Path
) -> None:
    """The strongest form: serialize both reviews and diff everything but style.

    Not "the status matched" but "no content-bearing field moved at all". A
    future change that let style nudge the score would fail here even if the
    verdict happened to survive.
    """
    clean = review_of(runs_dir, tmp_path, reviewer_returning(style=clean_style_assessment()))
    harsh = review_of(
        runs_dir,
        tmp_path,
        reviewer_returning(
            style=assessment(finding(HumanStyleCategory.AI_VOICE, StyleSeverity.HIGH), score=30)
        ),
    )

    ignored = {"run_id", "reviewed_at", "context_sha256", "draft_sha256", "writer_metadata_sha256"}
    a = {k: v for k, v in json.loads(clean.model_dump_json()).items() if k not in ignored}
    b = {k: v for k, v in json.loads(harsh.model_dump_json()).items() if k not in ignored}

    assert a.pop("style_review") != b.pop("style_review"), "the experiment must actually differ"
    assert a == b


def test_a_content_needs_revision_is_unchanged_by_a_clean_style(
    runs_dir: Path, tmp_path: Path
) -> None:
    """Case 25. The existing revision path continues exactly as before."""
    from goldpipeline.schemas.review import Evidence, ReviewIssue, Severity

    issue = ReviewIssue(
        issue_id="content-1",
        category=IssueCategory.DATA_MISMATCH,
        severity=Severity.HIGH,
        message="Giá gần nhất không khớp context.",
        evidence=Evidence(
            source_path="context.price.latest_close", expected="3314.20", actual="3325.20"
        ),
    )
    client = reviewer_returning(
        status=ReviewStatus.NEEDS_REVISION,
        score=60,
        issues=[issue],
        instructions=["Sửa giá gần nhất."],
        style=clean_style_assessment(),
    )
    result = review_of(runs_dir, tmp_path, client)

    assert result.status is ReviewStatus.NEEDS_REVISION
    assert [i.issue_id for i in result.issues] == ["content-1"]
    assert result.revision_instructions == ["Sửa giá gần nhất."]
    assert result.style_review is not None
    assert result.style_review.style_verdict is StyleVerdict.PASS


def test_the_orchestrator_never_reads_the_style_verdict() -> None:
    """Structural, not behavioural: there is no code path to break.

    A behavioural test proves the wire is not carrying current today. This
    proves the wire is not there.
    """
    import ast

    src = Path("src/goldpipeline/services")
    for module in ("orchestrator.py", "review_policy.py", "publish_gate.py", "finalizer_prompt.py"):
        text = (src / module).read_text(encoding="utf-8")
        tree = ast.parse(text)
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        assert "style_review" not in names, f"{module} reads style_review"
        assert "style_verdict" not in names, f"{module} reads style_verdict"


def test_the_finalizer_prompt_never_sees_a_style_finding(runs_dir: Path, tmp_path: Path) -> None:
    """Behavioural counterpart: a HIGH style finding reaches no finalizer turn."""
    from goldpipeline.services.finalizer_prompt import build_finalizer_prompt

    marker = "PRICE_READ restates four market numbers"
    style = assessment(
        finding(
            HumanStyleCategory.DATA_DUMP,
            StyleSeverity.HIGH,
            section=SectionKey.PRICE_READ,
            problem=marker,
        ),
        score=35,
    )
    reviewed = make_reviewed_run(runs_dir, tmp_path, review_client=reviewer_returning(style=style))
    assert reviewed.result is not None

    from goldpipeline.services.finalizer import load_verified_inputs

    run = RunStore(runs_dir).open(reviewed.run_id)
    inputs = load_verified_inputs(run, run.load_manifest())
    prompt = build_finalizer_prompt(
        context=inputs.context,
        review=inputs.review,
        article=inputs.article,
        report=_empty_report(),
    )

    assert marker not in prompt.user
    assert "DATA_DUMP" not in prompt.user
    assert "style_review" not in prompt.user


# --------------------------------------------------------------------------
# the derivation rule itself
# --------------------------------------------------------------------------


def test_one_high_finding_is_enough() -> None:
    assert (
        style_verdict_for([finding(HumanStyleCategory.AI_VOICE, StyleSeverity.HIGH)])
        is StyleVerdict.NEEDS_REVISION
    )


def test_two_mediums_are_not_enough_and_three_are() -> None:
    """The threshold, from both sides, so an off-by-one cannot hide."""
    mediums = [
        finding(HumanStyleCategory.VERBOSITY, StyleSeverity.MEDIUM, finding_id=f"m{n}")
        for n in range(MEDIUM_FINDINGS_FOR_REVISION)
    ]

    assert style_verdict_for(mediums[:-1]) is StyleVerdict.PASS
    assert style_verdict_for(mediums) is StyleVerdict.NEEDS_REVISION


def test_low_findings_never_accumulate_into_a_revision() -> None:
    """Attention is not a defect. Six small notes must not become a rewrite."""
    lows = [
        finding(HumanStyleCategory.VERBOSITY, StyleSeverity.LOW, finding_id=f"l{n}")
        for n in range(6)
    ]

    assert style_verdict_for(lows) is StyleVerdict.PASS


def test_the_verdict_is_derived_and_never_taken_from_the_model() -> None:
    """The model has nowhere to put a verdict, which is why it cannot disagree."""
    assert "style_verdict" not in HumanStyleAssessment.model_fields
    assert "style_verdict" in HumanStyleReview.model_fields


def test_there_is_no_style_reject() -> None:
    """A writing problem is editable; an unsalvageable one is a content problem."""
    assert {v.value for v in StyleVerdict} == {"PASS", "NEEDS_REVISION"}
    assert "REJECT" not in {v.value for v in StyleVerdict}


def test_style_severity_has_no_critical() -> None:
    assert {s.value for s in StyleSeverity} == {"LOW", "MEDIUM", "HIGH"}


# --------------------------------------------------------------------------
# vocabulary and finding shape
# --------------------------------------------------------------------------


def test_the_style_vocabulary_is_separate_from_the_content_one() -> None:
    """Overloading `IssueCategory` is what would have put style on the verdict path."""
    content = {c.value for c in IssueCategory}
    style = {c.value for c in HumanStyleCategory}

    assert not (content & style)


def test_the_vocabulary_is_compact_and_every_member_is_used_by_the_corpus() -> None:
    """Nine categories, and none of them decorative.

    `FORMAT_DRIFT` is the exception the assertion allows: it exists for drift
    the deterministic contract cannot already see, which by definition the
    contract-valid corpus articles do not exhibit.
    """
    used = {f.category for _, findings, _ in CORPUS for f in findings}
    unused = set(HumanStyleCategory) - used

    assert len(HumanStyleCategory) == 9
    assert unused == {HumanStyleCategory.FORMAT_DRIFT}


def test_over_explanation_was_folded_into_verbosity() -> None:
    """The audit Round 6.4f was asked to do, pinned so it is not silently undone."""
    assert not hasattr(HumanStyleCategory, "OVER_EXPLANATION")


def test_a_finding_must_say_what_is_wrong_and_what_to_do() -> None:
    """Both fields required, both non-blank: 'improve the style' is not a finding."""
    with pytest.raises(ValueError):
        HumanStyleFinding(
            finding_id="x",
            category=HumanStyleCategory.AI_VOICE,
            severity=StyleSeverity.LOW,
            problem="   ",
            repair_instruction="Delete it.",
        )
    with pytest.raises(ValueError):
        HumanStyleFinding(
            finding_id="x",
            category=HumanStyleCategory.AI_VOICE,
            severity=StyleSeverity.LOW,
            problem="Too formal.",
            repair_instruction="  ",
        )


def test_a_repair_instruction_cannot_be_a_rewritten_paragraph() -> None:
    """The cap is what stops the reviewer becoming a shadow finalizer."""
    from goldpipeline.schemas.review import MAX_STYLE_REPAIR_CHARS

    assert MAX_STYLE_REPAIR_CHARS < 500
    with pytest.raises(ValueError):
        HumanStyleFinding(
            finding_id="x",
            category=HumanStyleCategory.AI_VOICE,
            severity=StyleSeverity.LOW,
            problem="Too formal.",
            repair_instruction="x" * (MAX_STYLE_REPAIR_CHARS + 1),
        )


def test_finding_ids_are_unique_within_one_review() -> None:
    with pytest.raises(ValueError):
        HumanStyleAssessment(
            style_score=70,
            summary="ok",
            findings=[
                finding(HumanStyleCategory.AI_VOICE, finding_id="dup"),
                finding(HumanStyleCategory.VERBOSITY, finding_id="dup"),
            ],
        )


def test_a_finding_can_name_the_section_it_is_in() -> None:
    """Reusing the contract's own vocabulary, not a second list of headings."""
    located = finding(HumanStyleCategory.DATA_DUMP, section=SectionKey.PRICE_READ)

    assert located.section is SectionKey.PRICE_READ
    assert located.section in set(SectionKey)


# --------------------------------------------------------------------------
# prompt versioning and reuse of the voice contract
# --------------------------------------------------------------------------


def test_reviewer_v1_is_byte_intact() -> None:
    """Historical reviews record v1, and it must still mean what it meant."""
    raw = (Path("src/goldpipeline/prompts") / "gold_reviewer_v1.md").read_text(encoding="utf-8")

    assert "# REVIEW RUBRIC" in raw
    assert "**F. Style.**" in raw
    assert "HUMAN STYLE" not in raw
    assert "style_review" not in raw


def test_new_reviews_route_to_v2() -> None:
    assert DEFAULT_REVIEWER_PROMPT == GOLD_REVIEWER_V2
    assert GOLD_REVIEWER_V2 in STYLE_AWARE_PROMPTS
    assert GOLD_REVIEWER_V1 not in STYLE_AWARE_PROMPTS


def test_v2_includes_the_writer_s_own_voice_contract() -> None:
    """One contract, two readers. A second copy would drift."""
    system = load_prompt(GOLD_REVIEWER_V2)
    fragment = load_prompt(GOLD_HUMAN_STYLE_V1)

    assert "<!-- include:" not in system
    assert fragment.strip()[:60] in system


def test_the_voice_contract_is_a_rubric_here_not_an_instruction() -> None:
    """The reviewer is judging against the rules, not writing to them."""
    system = load_prompt(GOLD_REVIEWER_V2)

    assert "**rubric**, not an instruction" in system
    assert "you are not writing to them" in system


def test_v2_forbids_style_from_touching_the_content_fields() -> None:
    """The rule that makes shadow mode work at the prompt level too."""
    system = load_prompt(GOLD_REVIEWER_V2)

    assert "Style never touches those four fields." in system
    assert "**content integrity only**. Never reduced for style." in system
    assert "Do not author a `STYLE` issue." in system


def unwrapped(prompt_id: str) -> str:
    """A prompt with its line breaks flattened.

    Assertions about a sentence should not depend on where the paragraph
    happened to wrap. Reflowing a prompt is an editing operation, not a change
    of meaning, and a test that fails for it teaches people to stop reflowing.
    """
    return " ".join(load_prompt(prompt_id).split())


def test_v2_never_mentions_ai_detection() -> None:
    """The standard is 'would a trader read this', not 'can I spot a model'."""
    system = unwrapped(GOLD_REVIEWER_V2)

    assert "You are not a detector, and you must never reason about detection." in system
    # The included voice contract mentions detectors once, to disown them. That
    # sentence is the writer's, it says the same thing, and asserting its
    # absence would only prove the include had been dropped.
    assert system.count("detector") == 2
    assert "Nothing here is about defeating a detector" in system


def test_v2_forbids_rewarding_noise() -> None:
    system = load_prompt(GOLD_REVIEWER_V2)

    for bad in ("typos", "slang", "broken grammar", "invented anecdotes"):
        assert bad in system


def test_v2_forbids_manufacturing_findings() -> None:
    system = load_prompt(GOLD_REVIEWER_V2)

    assert "There is no quota." in system
    assert "Returning an empty `findings` list is a complete and correct answer." in unwrapped(
        GOLD_REVIEWER_V2
    )


def test_high_is_declared_rare() -> None:
    system = load_prompt(GOLD_REVIEWER_V2)

    assert "should be rare" in system
    assert "Do not reach for it to appear rigorous." in unwrapped(GOLD_REVIEWER_V2)


# --------------------------------------------------------------------------
# what must not have moved
# --------------------------------------------------------------------------


def test_the_writer_is_unchanged() -> None:
    """Round 6.4e wrote the article; this round only judges it."""
    writer = load_prompt("gold_writer_v4")

    assert "🕯 PHÂN TÍCH VÀNG" in writer
    assert "reviewer" not in writer.lower()
    assert "style_review" not in writer


def test_the_voice_contract_is_unchanged() -> None:
    style = load_prompt(GOLD_HUMAN_STYLE_V1)

    assert "# HUMAN STYLE v1" in style
    assert "style_review" not in style
    assert "reviewer" not in style.lower()


def test_the_finalizer_prompt_is_unchanged() -> None:
    finalizer = load_prompt("gold_finalizer_v1")

    assert "Minimum necessary revision" in finalizer
    assert "HUMAN STYLE" not in finalizer
    assert "style_review" not in finalizer


def test_the_finalizer_schema_has_no_style_field() -> None:
    from goldpipeline.schemas.finalizer import FinalizerModelOutput

    assert "style_review" not in FinalizerModelOutput.model_fields
    assert "style_verdict" not in FinalizerModelOutput.model_fields


def test_the_content_verdict_thresholds_are_unchanged() -> None:
    from goldpipeline.schemas.review import BLOCKING_SEVERITIES, PASS_MIN_SCORE, Severity

    assert PASS_MIN_SCORE == 90
    assert set(BLOCKING_SEVERITIES) == {Severity.HIGH, Severity.CRITICAL}
    assert {s.value for s in ReviewStatus} == {"PASS", "NEEDS_REVISION", "REJECT"}


def test_the_reviewer_still_takes_no_generation_selection() -> None:
    """Reviewer independence, unchanged by this round."""
    import inspect

    from goldpipeline import cli
    from goldpipeline.services import generation

    assert "selection" not in inspect.signature(cli._reviewer_client).parameters
    assert not hasattr(generation, "build_reviewer_client")


def test_the_reviewer_prompt_is_not_chosen_by_a_preference() -> None:
    """No `/model`, no `UserPreferences`, no `GenerationSelection` over review."""
    text = Path("src/goldpipeline/services/reviewer.py").read_text(encoding="utf-8")

    assert "GenerationSelection" not in text
    assert "UserPreferences" not in text


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _empty_report() -> Any:
    from goldpipeline.services.precheck import PrecheckReport

    return PrecheckReport(findings=[])


def test_the_review_artifact_filename_is_unchanged() -> None:
    """Readers of `gpt_review.json` predate this round and outlive it."""
    assert REVIEW_FILENAME == "gpt_review.json"
