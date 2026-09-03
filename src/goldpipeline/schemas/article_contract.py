"""What each article type is, in terms a deterministic check can hold it to.

Three products share one pipeline, and until now they shared one prose
contract too - the writer prompt - which is a document a model reads, not a
document code can enforce. This module is the enforceable half: per type, how
long the piece may be, which sections it must and must not have, whether it
carries a disclaimer, and whether a language model writes it at all.

**One registry.** :data:`CONTRACTS` is the only authority. The routing table in
:mod:`goldpipeline.services.article_routing` says whether a type *runs*; this
says what it must *look like* when it does. A test holds the two consistent, so
a routing change that hands a writer prompt to a type contracted as
deterministic fails before it ships.

**Targets are guidance; hard caps are limits.** A naturally short article that
says everything it has to say is not a defect, and no check here treats it as
one. Only the hard cap is a finding. Padding to a minimum is precisely the
machine-written register the style work exists to remove.

**Not wired.** Nothing in production reads this module yet. It exists so the
rounds that activate the new shapes have a contract to activate against, and
so the contract can be reviewed on its own before any prose changes.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.common import StrictModel

ARTICLE_CONTRACT_VERSION = "1"

DISCLAIMER_TEXT = "🔴 Nhận định cá nhân, không phải lời khuyên đầu tư."
"""The one disclaimer string. Exactly this, once, where the contract asks for it."""

TRADE_PLAN_SIDE_LABELS = ("SEO", "BAI")
"""The operator's literal product terminology for the two sides of a plan.

Not abbreviations of SELL and BUY, and never normalised to them. A check that
finds ``SELL`` or ``BUY`` standing where these belong reports a violation.
"""


class GenerationMode(StrEnum):
    """Who produces the final public document."""

    LLM = "LLM"
    """A writer model drafts it from a prompt; review and finalization follow."""

    DETERMINISTIC = "DETERMINISTIC"
    """Code renders it from validated data. There is no writer prompt.

    This is a statement about the *final document*, not about the process
    behind it. An upstream analyst - human or model - may still weigh candidate
    zones, judge session relevance or read structure; what it may not do is put
    a number into the public article. Every level in a deterministic document
    was computed and validated by code, and the renderer prints what it is
    given.
    """


class SectionKey(StrEnum):
    """Contract vocabulary for the parts of an article.

    Names, not prose. The heading a reader sees belongs to the prompt or the
    renderer; this enum is how a contract says "a verdict must exist" without
    fixing the words that introduce it. A later round maps each key to the cues
    a check looks for.
    """

    # ANALYSIS - what matters, why to gold, does price agree, what changes it
    ANALYSIS_TITLE = "ANALYSIS_TITLE"
    VERDICT = "VERDICT"
    DRIVERS_UP = "DRIVERS_UP"
    DRIVERS_DOWN = "DRIVERS_DOWN"
    PRICE_READ = "PRICE_READ"
    """Does price agree with the thesis. Never a restatement of the digest's range."""
    WATCHING = "WATCHING"

    # NEWS_DIGEST - what happened
    DIGEST_TITLE = "DIGEST_TITLE"
    DIGEST_WINDOW = "DIGEST_WINDOW"
    DIGEST_ITEMS = "DIGEST_ITEMS"
    PRICE_REACTION = "PRICE_REACTION"
    BALANCE = "BALANCE"

    # TRADE_PLAN - where to watch
    PLAN_TITLE = "PLAN_TITLE"
    SEO = "SEO"
    BAI = "BAI"

    DISCLAIMER = "DISCLAIMER"


class StructureKind(StrEnum):
    """Shapes of text a contract can forbid, independent of any heading.

    A section is forbidden by name; a structure is forbidden by form. A digest
    that grows an ``if/then`` scenario block has not added a heading, it has
    changed product, and the contract needs a word for that.
    """

    NEWS_ITEM_LIST = "NEWS_ITEM_LIST"
    """Time-stamped, per-item lines. The digest's body; foreign to an analysis."""

    TRADE_ZONE_LIST = "TRADE_ZONE_LIST"
    """Numeric ``low – high`` zone lines. The plan's body; foreign to prose pieces."""

    SCENARIO_ESSAY = "SCENARIO_ESSAY"
    """Conditional outlook prose. Belongs in an analysis, not in a digest."""

    EXPLANATORY_PROSE = "EXPLANATORY_PROSE"
    """Sentences that explain. A plan names levels and stops."""

    DATE_LINE = "DATE_LINE"
    """A dated title or window line. A plan has none."""

    RISK_PARAMETERS = "RISK_PARAMETERS"
    """Stop loss, take profit, risk-reward. A plan lists zones only."""


class DisclaimerPolicy(StrictModel):
    """How many disclaimers an article carries. Per type, never global.

    ``expected_count`` is a count, not a boolean, because "at most one" and
    "exactly one" are different rules and the difference is exactly what a
    duplicated disclaimer violates.
    """

    expected_count: int = Field(ge=0, le=1)
    text: str = Field(default=DISCLAIMER_TEXT, min_length=1)


class ArticleContract(StrictModel):
    """The enforceable shape of one article type."""

    schema_version: Literal["1"] = "1"

    article_type: ArticleType
    question: str = Field(
        min_length=1,
        max_length=160,
        description="The one question this product answers. If two types share it, merge them.",
    )
    generation_mode: GenerationMode
    human_style: bool = Field(
        description=(
            "Whether the human-style contract applies. True for prose a model writes; "
            "False for a rendered document, which has no voice to judge."
        )
    )

    target_min_chars: int = Field(ge=0, description="Guidance. Falling short is not a finding.")
    target_max_chars: int = Field(ge=0, description="Guidance. Exceeding it is not a finding.")
    hard_max_chars: int = Field(gt=0, description="The limit. Exceeding it is a finding.")

    required_sections: tuple[SectionKey, ...] = Field(
        description="In reading order. Every one must be present."
    )
    forbidden_sections: frozenset[SectionKey] = Field(default_factory=frozenset)
    forbidden_structures: frozenset[StructureKind] = Field(default_factory=frozenset)

    disclaimer: DisclaimerPolicy

    @model_validator(mode="after")
    def _coherent(self) -> ArticleContract:
        if not self.target_min_chars <= self.target_max_chars <= self.hard_max_chars:
            raise ValueError("expected target_min <= target_max <= hard_max")
        if len(set(self.required_sections)) != len(self.required_sections):
            raise ValueError("required_sections repeats a key")
        if set(self.required_sections) & self.forbidden_sections:
            raise ValueError("a section is both required and forbidden")
        if self.generation_mode is GenerationMode.DETERMINISTIC and self.human_style:
            # A rendered document has no author whose voice could be judged.
            raise ValueError("a deterministic document cannot carry the human-style contract")
        expects_disclaimer = self.disclaimer.expected_count > 0
        if expects_disclaimer != (SectionKey.DISCLAIMER in self.required_sections):
            raise ValueError("disclaimer policy and required_sections disagree")
        return self


CONTRACTS: dict[ArticleType, ArticleContract] = {
    ArticleType.NEWS_DIGEST: ArticleContract(
        article_type=ArticleType.NEWS_DIGEST,
        question="What happened?",
        generation_mode=GenerationMode.LLM,
        human_style=True,
        target_min_chars=900,
        target_max_chars=1500,
        hard_max_chars=1900,
        required_sections=(
            SectionKey.DIGEST_TITLE,
            SectionKey.DIGEST_WINDOW,
            SectionKey.DIGEST_ITEMS,
            SectionKey.PRICE_REACTION,
            SectionKey.BALANCE,
            SectionKey.DISCLAIMER,
        ),
        forbidden_sections=frozenset(
            {
                SectionKey.SEO,
                SectionKey.BAI,
                SectionKey.PLAN_TITLE,
                SectionKey.VERDICT,
                SectionKey.WATCHING,
            }
        ),
        forbidden_structures=frozenset(
            {StructureKind.TRADE_ZONE_LIST, StructureKind.SCENARIO_ESSAY}
        ),
        disclaimer=DisclaimerPolicy(expected_count=1),
    ),
    ArticleType.ANALYSIS: ArticleContract(
        article_type=ArticleType.ANALYSIS,
        question="What matters, why to gold, does price agree, and what could change it?",
        generation_mode=GenerationMode.LLM,
        human_style=True,
        target_min_chars=600,
        target_max_chars=1000,
        hard_max_chars=1300,
        required_sections=(
            SectionKey.ANALYSIS_TITLE,
            SectionKey.VERDICT,
            SectionKey.DRIVERS_UP,
            SectionKey.DRIVERS_DOWN,
            SectionKey.PRICE_READ,
            SectionKey.WATCHING,
            SectionKey.DISCLAIMER,
        ),
        forbidden_sections=frozenset(
            {
                SectionKey.SEO,
                SectionKey.BAI,
                SectionKey.PLAN_TITLE,
                SectionKey.DIGEST_ITEMS,
                SectionKey.DIGEST_WINDOW,
            }
        ),
        forbidden_structures=frozenset(
            {StructureKind.TRADE_ZONE_LIST, StructureKind.NEWS_ITEM_LIST}
        ),
        disclaimer=DisclaimerPolicy(expected_count=1),
    ),
    ArticleType.TRADE_PLAN: ArticleContract(
        article_type=ArticleType.TRADE_PLAN,
        question="Where do I watch for SEO / BAI?",
        generation_mode=GenerationMode.DETERMINISTIC,
        human_style=False,
        target_min_chars=200,
        target_max_chars=450,
        hard_max_chars=650,
        required_sections=(SectionKey.PLAN_TITLE, SectionKey.SEO, SectionKey.BAI),
        forbidden_sections=frozenset(
            {
                SectionKey.DISCLAIMER,
                SectionKey.VERDICT,
                SectionKey.DRIVERS_UP,
                SectionKey.DRIVERS_DOWN,
                SectionKey.PRICE_READ,
                SectionKey.WATCHING,
                SectionKey.DIGEST_TITLE,
                SectionKey.DIGEST_WINDOW,
                SectionKey.DIGEST_ITEMS,
                SectionKey.PRICE_REACTION,
                SectionKey.BALANCE,
            }
        ),
        forbidden_structures=frozenset(
            {
                StructureKind.EXPLANATORY_PROSE,
                StructureKind.NEWS_ITEM_LIST,
                StructureKind.SCENARIO_ESSAY,
                StructureKind.DATE_LINE,
                StructureKind.RISK_PARAMETERS,
            }
        ),
        disclaimer=DisclaimerPolicy(expected_count=0),
    ),
}
"""Every article type, once. A test asserts the keys equal the enum."""


def contract_for(article_type: ArticleType) -> ArticleContract:
    """The contract for *article_type*."""
    return CONTRACTS[article_type]


HUMAN_STYLE_TYPES = frozenset(t for t, c in CONTRACTS.items() if c.human_style)
"""Types the human-style contract will apply to. ``TRADE_PLAN`` is not one."""


__all__ = [
    "ARTICLE_CONTRACT_VERSION",
    "CONTRACTS",
    "DISCLAIMER_TEXT",
    "HUMAN_STYLE_TYPES",
    "TRADE_PLAN_SIDE_LABELS",
    "ArticleContract",
    "DisclaimerPolicy",
    "GenerationMode",
    "SectionKey",
    "StructureKind",
    "contract_for",
]
