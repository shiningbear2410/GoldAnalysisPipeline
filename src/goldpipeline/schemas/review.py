"""Reviewer stage contracts.

The same split as Round 2, for the same reason:

* :class:`ReviewModelOutput` is what the reviewer model may author - a verdict,
  a score, issues, and instructions for whoever fixes them.
* :class:`ReviewResult` is the artifact this pipeline stamps - identity, model,
  timestamps, digests, deterministic findings, usage.

**The reviewer never rewrites the article.** There is no field here for a
revised, corrected or improved version, and ``extra="forbid"`` means a model
that invents one has its answer rejected. Rewriting belongs to Round 4; a
reviewer that also rewrites is no longer an independent auditor of the text it
produced. The instruction fields are length-capped for the same reason: an
instruction is a sentence, not a smuggled draft.

**Two axes since Round 6.4f, and only one of them decides anything.** Content
integrity - ``status``, ``score``, ``issues``, ``revision_instructions`` - is
what it always was and still governs the pipeline. Human style is a second,
independent judgement carried in ``style_review``: its own closed vocabulary,
its own severities, its own verdict, and no path to the transition. The
separation is structural rather than advisory, because the alternative is a
reviewer that blocks on voice while the finalizer has not yet been taught to
repair voice - which would rewrite production articles unpredictably.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import Field, field_validator

from goldpipeline.schemas.article_contract import SectionKey
from goldpipeline.schemas.common import StrictModel, UtcDatetime, utc_now

REVIEW_SCHEMA_VERSION = "1.1.0"
"""Version of the review artifact contract.

1.1.0 adds the optional ``style_review`` object. Additive: every 1.0.0 artifact
still validates, and ``schema_version`` is a plain string rather than a literal
so a historical file keeps recording the version it was actually written under.
"""

MAX_ISSUES = 40
MAX_INSTRUCTIONS = 30
MAX_INSTRUCTION_CHARS = 500
"""An instruction longer than this is prose, not a fix - and possibly a rewrite."""

MAX_SUMMARY_CHARS = 2000
MAX_EXCERPT_CHARS = 500

PASS_MIN_SCORE = 90
"""Below this a review cannot be a PASS, however clean the issue list looks."""

MAX_STYLE_FINDINGS = 20
"""More than this is not a review, it is a list of everything the model noticed."""

MAX_STYLE_PROBLEM_CHARS = 500
MAX_STYLE_REPAIR_CHARS = 300
"""Deliberately tighter than an issue's ``suggested_fix``.

A style repair says *what to cut or state plainly*. Three hundred characters is
enough for that and not enough for a rewritten section, which is the failure
mode a style-aware reviewer is most likely to drift into.
"""

MAX_STYLE_SUMMARY_CHARS = 1000


@dataclass(frozen=True)
class ReviewerPrompt:
    """A rendered reviewer prompt.

    Lives here rather than beside its builder for the same reason as
    :class:`~goldpipeline.schemas.writer.WriterPrompt`: the adapter layer sends
    it and the service layer builds it, and schemas is the one layer both may
    depend on.

    ``system`` is the versioned template, byte for byte. ``user`` is the only
    place Run content ever appears - the context, the writer metadata, the
    analyst's note and the article, the last two fenced with ``nonce``.
    """

    system: str
    user: str
    prompt_version: str
    nonce: str

    @property
    def sections(self) -> tuple[str, ...]:
        """Upper-case headings across both turns, for assertions and debugging."""
        combined = "\n".join((self.system, self.user))
        return tuple(
            line for line in combined.splitlines() if line.startswith("# ") and line.isupper()
        )


class ReviewStatus(StrEnum):
    """The verdict.

    * ``PASS`` - nothing worth fixing was found.
    * ``NEEDS_REVISION`` - usable, but a later stage must correct specific
      things. This is the normal outcome for a wrong number.
    * ``REJECT`` - not safe to continue automatically: a critical factual error,
      a wrong instrument, or an article that small edits cannot rescue.

    A provider or network failure is **not** a verdict. It says nothing about
    the article, so it raises rather than producing a ``REJECT``.
    """

    PASS = "PASS"
    NEEDS_REVISION = "NEEDS_REVISION"
    REJECT = "REJECT"


class Severity(StrEnum):
    """How much an issue matters."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


BLOCKING_SEVERITIES = frozenset({Severity.HIGH, Severity.CRITICAL})
"""Severities that make a PASS impossible."""


class IssueCategory(StrEnum):
    """What kind of problem an issue describes.

    Small and closed: a downstream stage branches on these, and a free-text
    category from a language model is not something you can branch on.
    """

    DATA_MISMATCH = "DATA_MISMATCH"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    SOURCE_CONTRADICTION = "SOURCE_CONTRADICTION"
    LOGIC = "LOGIC"
    STYLE = "STYLE"
    FORMAT = "FORMAT"
    RISK_LANGUAGE = "RISK_LANGUAGE"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    OTHER = "OTHER"


FACTUAL_CATEGORIES = frozenset(
    {
        IssueCategory.DATA_MISMATCH,
        IssueCategory.UNSUPPORTED_CLAIM,
        IssueCategory.SOURCE_CONTRADICTION,
    }
)
"""Categories that assert something about the facts, and so must carry evidence."""


class FindingCode(StrEnum):
    """Deterministic pre-review findings, produced by Python rather than a model."""

    CLAIM_SOURCE_NOT_FOUND = "CLAIM_SOURCE_NOT_FOUND"
    CLAIM_VALUE_MISMATCH = "CLAIM_VALUE_MISMATCH"
    UNKNOWN_PRICE_LIKE_NUMBER = "UNKNOWN_PRICE_LIKE_NUMBER"
    NUMBER_OUTSIDE_MARKET_RANGE = "NUMBER_OUTSIDE_MARKET_RANGE"
    SYMBOL_NOT_MENTIONED = "SYMBOL_NOT_MENTIONED"
    FOREIGN_SYMBOL_MENTIONED = "FOREIGN_SYMBOL_MENTIONED"
    UNSUPPORTED_INDICATOR_MENTIONED = "UNSUPPORTED_INDICATOR_MENTIONED"
    ABSOLUTE_RISK_LANGUAGE = "ABSOLUTE_RISK_LANGUAGE"
    NO_SOURCE_CLAIMS = "NO_SOURCE_CLAIMS"


class Evidence(StrictModel):
    """Why a factual issue is an issue.

    "The number looks wrong" is not a review finding. This is what turns an
    assertion into something a later stage can act on without re-deriving it.
    """

    source_path: str = Field(
        min_length=1,
        max_length=200,
        description="Dotted path into the context, e.g. 'context.price.latest_close'.",
    )
    expected: str = Field(max_length=400, description="What the source of truth says.")
    actual: str = Field(max_length=400, description="What the article says.")


class ReviewIssue(StrictModel):
    """One problem found in the article."""

    issue_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
        description="Stable identifier, unique within one review.",
    )
    category: IssueCategory
    severity: Severity
    message: str = Field(min_length=1, max_length=1000, description="What is wrong.")
    claim: str | None = Field(
        default=None, max_length=500, description="The statement being challenged."
    )
    article_excerpt: str | None = Field(
        default=None, max_length=MAX_EXCERPT_CHARS, description="Where in the article it appears."
    )
    evidence: Evidence | None = Field(default=None, description="Required for factual categories.")
    suggested_fix: str | None = Field(
        default=None,
        max_length=MAX_INSTRUCTION_CHARS,
        description="A specific correction. Never a rewritten article.",
    )


# --------------------------------------------------------------------------
# human style: the second axis
# --------------------------------------------------------------------------


class HumanStyleCategory(StrEnum):
    """What kind of writing problem a style finding describes.

    A separate enum from :class:`IssueCategory` on purpose. Overloading the
    content vocabulary would put style findings on the path that decides
    verdicts - a HIGH content issue makes a PASS impossible - and the whole
    point of this round is that style decides nothing yet.

    Nine, not the ten the design sketched. ``OVER_EXPLANATION`` was folded into
    :attr:`VERBOSITY`: both said "these words could go", the boundary between
    them was a judgement call every time, and two categories that a reviewer
    cannot reliably tell apart produce inconsistent findings rather than finer
    ones. The pairs that remain are kept because each has a distinct repair.
    """

    AI_VOICE = "AI_VOICE"
    """The formulaic register: stacked connectives, throat-clearing openers,
    every paragraph built the same way. Repair: cut the scaffolding."""

    NEWS_DESK_VOICE = "NEWS_DESK_VOICE"
    """Institutional register - a wire report or a bank research note. Formal,
    impersonal, hedged. Distinct from `AI_VOICE`: this one reads as written by
    a person, just the wrong person. Repair: say it the way a trader would."""

    REPETITIVE_RHYTHM = "REPETITIVE_RHYTHM"
    """Sentences or paragraphs of one length and one shape. Repair: vary."""

    VERBOSITY = "VERBOSITY"
    """Text that could be deleted without losing anything a trader would use -
    including explaining what the reader already knows. Repair: delete."""

    DATA_DUMP = "DATA_DUMP"
    """Statistics standing in for interpretation. The numbers may all be true
    and supported; the section still never says what they mean. Repair: keep
    the strongest figure, state the judgement."""

    NO_POSITION = "NO_POSITION"
    """Never commits. Every observation is immediately qualified away. Repair:
    state the leaning the evidence already supports."""

    FORCED_BALANCE = "FORCED_BALANCE"
    """A side invented to fill a heading. Distinct from `NO_POSITION`: the
    article may take a view and still manufacture a counterweight it does not
    believe. Repair: say there is nothing material on that side."""

    GENERIC_CONCLUSION = "GENERIC_CONCLUSION"
    """An ending that restates the opening, or that would fit any asset on any
    day. Repair: name the condition that would change the view."""

    FORMAT_DRIFT = "FORMAT_DRIFT"
    """Editorial drift toward another product - digest chronology, plan-like
    levels - that the deterministic contract does not already make impossible.
    Never used to restate a blocking structural finding."""


class StyleSeverity(StrEnum):
    """How much a style finding matters.

    No ``CRITICAL``, deliberately. A writing problem is editable; a piece that
    cannot be rescued by editing is a content problem and belongs to
    :class:`Severity`. ``HIGH`` is for a whole-article failure of voice, and is
    meant to be rare - a reviewer that reaches for it to look rigorous produces
    a revision queue nobody trusts.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class StyleVerdict(StrEnum):
    """The style-only verdict.

    Two values, not three. There is no style ``REJECT``: fabricated or unsafe
    content is what stops an article, and that judgement lives on the content
    axis.
    """

    PASS = "PASS"
    NEEDS_REVISION = "NEEDS_REVISION"


class HumanStyleFinding(StrictModel):
    """One writing problem, shaped so a later round can repair it minimally.

    Every field exists because the Round 6.4g finalizer will need it: *what*
    kind of problem, *how much* it matters, *where* it is, *why* it is a
    problem, and *what to do* - without being handed a rewritten paragraph.
    """

    finding_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
        description="Stable identifier, unique within one style review.",
    )
    category: HumanStyleCategory
    severity: StyleSeverity
    section: SectionKey | None = Field(
        default=None,
        description="Which part of the article. None means the piece as a whole.",
    )
    problem: str = Field(
        min_length=1,
        max_length=MAX_STYLE_PROBLEM_CHARS,
        description="What is wrong, concretely. Never 'the style could be improved'.",
    )
    repair_instruction: str = Field(
        min_length=1,
        max_length=MAX_STYLE_REPAIR_CHARS,
        description="What to change. An instruction, never replacement prose.",
    )
    article_excerpt: str | None = Field(
        default=None,
        max_length=MAX_EXCERPT_CHARS,
        description="The text the finding is about, quoted from the article.",
    )

    @field_validator("problem", "repair_instruction")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a style finding must say something")
        return value.strip()


class HumanStyleAssessment(StrictModel):
    """What the reviewer model may author about style.

    Note the absence of a verdict. The model supplies observations - a score,
    findings, a sentence of summary - and the verdict is derived from them in
    Python by :func:`~goldpipeline.services.style_review.style_verdict_for`.

    That is not distrust for its own sake. A verdict the model states could
    disagree with the findings it listed, and the only ways to handle the
    disagreement are to reject the whole review or to silently overrule it.
    Rejecting would let the shadow axis fail a review whose content verdict was
    sound, which is exactly the production coupling this round exists to avoid;
    overruling silently would leave an artifact whose verdict nobody can derive.
    Deriving it removes the disagreement instead of resolving it.
    """

    style_score: int = Field(
        ge=0,
        le=100,
        description=(
            "Editorial judgement of fit with the product voice, 0-100. "
            "Not a count of banned phrases."
        ),
    )
    summary: str = Field(
        min_length=1,
        max_length=MAX_STYLE_SUMMARY_CHARS,
        description="A sentence or two on how the piece reads.",
    )
    findings: list[HumanStyleFinding] = Field(default_factory=list, max_length=MAX_STYLE_FINDINGS)

    @field_validator("summary")
    @classmethod
    def _reject_blank_summary(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("style summary must not be empty")
        return value.strip()

    @field_validator("findings")
    @classmethod
    def _finding_ids_are_unique(cls, value: list[HumanStyleFinding]) -> list[HumanStyleFinding]:
        seen = [finding.finding_id for finding in value]
        duplicates = sorted({item for item in seen if seen.count(item) > 1})
        if duplicates:
            raise ValueError(f"finding_id values must be unique; repeated: {duplicates}")
        return value


class HumanStyleReview(StrictModel):
    """The style judgement as it is stamped on the artifact.

    The model's assessment plus the derived verdict. Absent entirely on a review
    of an article type that has no voice to judge - a rendered ``TRADE_PLAN``
    has no style to review, and recording ``PASS`` for it would be a claim about
    a judgement nobody made.
    """

    style_score: int = Field(ge=0, le=100)
    style_verdict: StyleVerdict = Field(
        description="Derived from the findings, never taken from the model."
    )
    summary: str = Field(min_length=1, max_length=MAX_STYLE_SUMMARY_CHARS)
    findings: list[HumanStyleFinding] = Field(default_factory=list)

    @property
    def blocking_findings(self) -> list[HumanStyleFinding]:
        """Findings that would drive a revision once Round 6.4g activates this.

        Named for what it will mean, and deliberately unused by anything that
        decides something today.
        """
        return [f for f in self.findings if f.severity is StyleSeverity.HIGH]


class ReviewModelOutput(StrictModel):
    """The structured response a reviewer model must return.

    Note what is absent: any field carrying a corrected article. That is the
    contract, not an oversight.
    """

    run_id: str = Field(
        min_length=1,
        max_length=64,
        description="Echo of the run id under review, checked against the real one.",
    )
    status: ReviewStatus
    score: int = Field(ge=0, le=100, description="Overall quality, 0-100.")
    summary: str = Field(
        min_length=1, max_length=MAX_SUMMARY_CHARS, description="A few sentences of verdict."
    )
    issues: list[ReviewIssue] = Field(default_factory=list, max_length=MAX_ISSUES)
    revision_instructions: list[str] = Field(
        default_factory=list,
        max_length=MAX_INSTRUCTIONS,
        description="Specific edits for the finalizer. Instructions only, never prose to publish.",
    )
    style_review: HumanStyleAssessment | None = Field(
        default=None,
        description=(
            "Human-style judgement. Required for an ANALYSIS review under a "
            "style-aware prompt, absent for article types that have no voice. "
            "Nothing here affects status, score, issues or instructions."
        ),
    )

    @field_validator("summary")
    @classmethod
    def _reject_blank_summary(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("summary must not be empty")
        return value.strip()

    @field_validator("revision_instructions")
    @classmethod
    def _instructions_are_instructions(cls, value: list[str]) -> list[str]:
        """Reject blanks, and anything long enough to be a draft in disguise."""
        cleaned: list[str] = []
        for instruction in value:
            text = instruction.strip()
            if not text:
                raise ValueError("revision instructions must not be blank")
            if len(text) > MAX_INSTRUCTION_CHARS:
                raise ValueError(
                    f"a revision instruction may not exceed {MAX_INSTRUCTION_CHARS} characters; "
                    "the reviewer must not rewrite the article"
                )
            cleaned.append(text)
        return cleaned

    @field_validator("issues")
    @classmethod
    def _issue_ids_are_unique(cls, value: list[ReviewIssue]) -> list[ReviewIssue]:
        seen = [issue.issue_id for issue in value]
        duplicates = sorted({item for item in seen if seen.count(item) > 1})
        if duplicates:
            raise ValueError(f"issue_id values must be unique; repeated: {duplicates}")
        return value

    @property
    def blocking_issues(self) -> list[ReviewIssue]:
        """Issues severe enough to rule out a PASS."""
        return [issue for issue in self.issues if issue.severity in BLOCKING_SEVERITIES]


class PrecheckFinding(StrictModel):
    """One deterministic observation made before the model was consulted.

    These are facts about the artifacts, computed in Python. They do not depend
    on a model having noticed anything, and they are never discarded because a
    reviewer failed to mention them.
    """

    code: FindingCode
    severity: Severity
    message: str = Field(min_length=1, max_length=1000)
    source_path: str | None = Field(default=None, max_length=200)
    expected: str | None = Field(default=None, max_length=400)
    actual: str | None = Field(default=None, max_length=400)
    excerpt: str | None = Field(default=None, max_length=MAX_EXCERPT_CHARS)

    @property
    def is_blocking(self) -> bool:
        """Whether this finding alone rules out a PASS."""
        return self.severity in BLOCKING_SEVERITIES


class VerdictSource(StrEnum):
    """Where the final verdict came from.

    ``POLICY_ESCALATED`` records that Python overrode a more generous model
    verdict because deterministic evidence contradicted it. Recorded rather than
    applied silently: a reader of the artifact should be able to see that the
    reviewer and the pipeline disagreed.
    """

    MODEL = "MODEL"
    POLICY_ESCALATED = "POLICY_ESCALATED"


class ReviewUsage(StrictModel):
    """Provider usage metadata. Counts and opaque ids only."""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    request_id: str | None = Field(default=None, max_length=200)
    response_status: str | None = Field(default=None, max_length=64)


class ReviewResult(StrictModel):
    """The ``gpt_review.json`` artifact.

    Carries the digests of all three inputs, so a later stage can prove the
    review describes the artifacts it is looking at rather than earlier versions
    of them.
    """

    schema_version: str = Field(default=REVIEW_SCHEMA_VERSION)
    run_id: str
    stage: str = Field(default="gpt_reviewer")

    status: ReviewStatus
    score: int = Field(ge=0, le=100)
    summary: str
    issues: list[ReviewIssue] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)

    model_status: ReviewStatus = Field(
        description="The verdict the model itself returned, before policy was applied."
    )
    verdict_source: VerdictSource = Field(default=VerdictSource.MODEL)
    policy_notes: list[str] = Field(
        default_factory=list, description="Why the final verdict differs from the model's."
    )
    deterministic_findings: list[PrecheckFinding] = Field(default_factory=list)

    style_review: HumanStyleReview | None = Field(
        default=None,
        description=(
            "The style judgement, when the article type has a voice to judge. "
            "None on every review written before Round 6.4f, and on any article "
            "type outside HUMAN_STYLE_TYPES. Shadow-mode: recorded, never acted on."
        ),
    )

    model: str = Field(description="Model id that produced the review.")
    provider: str = Field(description="Which client produced it, e.g. 'openai', 'fake'.")
    prompt_version: str = Field(description="Version of the reviewer prompt used.")
    reviewed_at: UtcDatetime = Field(default_factory=utc_now)

    context_sha256: str = Field(min_length=64, max_length=64)
    draft_sha256: str = Field(min_length=64, max_length=64)
    writer_metadata_sha256: str = Field(min_length=64, max_length=64)

    usage: ReviewUsage = Field(default_factory=ReviewUsage)

    @property
    def blocking_issues(self) -> list[ReviewIssue]:
        """Issues severe enough to rule out a PASS."""
        return [issue for issue in self.issues if issue.severity in BLOCKING_SEVERITIES]


__all__ = [
    "BLOCKING_SEVERITIES",
    "FACTUAL_CATEGORIES",
    "MAX_INSTRUCTION_CHARS",
    "MAX_STYLE_FINDINGS",
    "MAX_STYLE_PROBLEM_CHARS",
    "MAX_STYLE_REPAIR_CHARS",
    "MAX_STYLE_SUMMARY_CHARS",
    "PASS_MIN_SCORE",
    "REVIEW_SCHEMA_VERSION",
    "Evidence",
    "FindingCode",
    "HumanStyleAssessment",
    "HumanStyleCategory",
    "HumanStyleFinding",
    "HumanStyleReview",
    "IssueCategory",
    "PrecheckFinding",
    "ReviewIssue",
    "ReviewModelOutput",
    "ReviewResult",
    "ReviewStatus",
    "ReviewUsage",
    "ReviewerPrompt",
    "SectionKey",
    "Severity",
    "StyleSeverity",
    "StyleVerdict",
    "VerdictSource",
]
