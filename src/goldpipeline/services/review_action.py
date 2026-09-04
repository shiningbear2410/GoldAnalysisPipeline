"""What the pipeline does about a review. One authority, one decision.

Round 6.4g, and this module exists to keep two things apart that Round 6.4f
deliberately separated and that it would be very easy to merge back together.

``ReviewResult.status`` is the **content integrity verdict**. It is what the
reviewer said about whether the article is true, and it is stamped on an
immutable artifact. Nothing here rewrites it. A Run whose facts are clean and
whose prose is not still has ``status == PASS`` on disk forever, because that is
what was judged.

:class:`ReviewAction` is the **orchestration decision**: given that verdict and
the style judgement beside it, does the finalizer run? Those are different
questions, and conflating them would mean either lying in the artifact ("we said
NEEDS_REVISION so the finalizer would fire") or scattering ``if
review.style_review`` through the orchestrator until nobody can say what
actually triggers a revision.

**Style activation lives here and nowhere else.** :data:`STYLE_ACTIVE_TYPES` is
the switch, and it is a frozen set of article types rather than an environment
flag or a boolean default: a set cannot be flipped by a stale variable on one
machine, and adding a type to it is a visible line in a diff. Round 6.4f built
the style verdict and left it with no caller; this module is that caller, and
the only one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.review import (
    HumanStyleFinding,
    ReviewResult,
    ReviewStatus,
    StyleVerdict,
)
from goldpipeline.services.style_review import applies_to, findings_requiring_repair

STYLE_ACTIVE_TYPES = frozenset({ArticleType.ANALYSIS})
"""Article types whose style verdict may require a revision.

Narrower than :data:`~goldpipeline.schemas.article_contract.HUMAN_STYLE_TYPES`,
and the gap is intentional. ``NEWS_DIGEST`` has a voice worth *judging* the day
it becomes producible, but activating a repair path for an article type that
cannot yet be written would be a rule nobody could test against a real Run.
``TRADE_PLAN`` is in neither: a rendered document has no prose to repair.

Adding a type here is the whole of activating style revision for it.
"""


class ReviewAction(StrEnum):
    """What the pipeline should do next, having read the review.

    Deliberately not named after the verdict values it partly mirrors. A reader
    who sees ``PASS_THROUGH`` should not have to wonder whether it means the
    content passed - it means no model will be called, whatever the reason.
    """

    PASS_THROUGH = "PASS_THROUGH"
    """Copy the draft byte for byte. Nothing needs repair."""

    FINALIZE = "FINALIZE"
    """One revision pass, repairing everything asked for in a single call."""

    REJECT = "REJECT"
    """Stop. A human decides. Never a model."""


@dataclass(frozen=True)
class ActionDecision:
    """The action, and enough of the reasoning to put on the ledger."""

    action: ReviewAction
    content_status: ReviewStatus
    style_verdict: StyleVerdict | None
    """``None`` when the review carries no style judgement at all - every review
    written before Round 6.4f, and every article type outside the style scope."""

    style_findings: tuple[HumanStyleFinding, ...] = ()
    """The findings the finalizer must repair. Empty unless style drove this."""

    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def calls_provider(self) -> bool:
        """Whether this decision spends a model call."""
        return self.action is ReviewAction.FINALIZE

    @property
    def style_driven(self) -> bool:
        """Whether style is part of why the finalizer is running.

        True for a style-only revision *and* for a combined one. The finalizer
        needs the findings either way, and asking "was it only style?" is a
        question nothing downstream should be branching on.
        """
        return bool(self.style_findings)


def style_is_active(article_type: ArticleType) -> bool:
    """Whether a style verdict may require a revision for this article type.

    Both conditions: the type must have a voice at all (the Round 6.4f rule),
    and its repair path must be switched on here. A type that could be judged
    but not repaired would produce a verdict nothing acts on, which is the
    shadow mode this round is ending - deliberately, for one type at a time.
    """
    return applies_to(article_type) and article_type in STYLE_ACTIVE_TYPES


def effective_action(review: ReviewResult, *, article_type: ArticleType) -> ActionDecision:
    """Decide what happens to this Run, without touching what was judged.

    Args:
        review: The committed review artifact. Read, never modified.
        article_type: What the Run is writing.

    Returns:
        An :class:`ActionDecision`. Content integrity has precedence at both
        ends: a rejection is never softened by clean prose, and a content
        revision happens whether or not style also asked for one - in the same
        single call.
    """
    content = review.status

    if content is ReviewStatus.REJECT:
        # No style judgement can rescue this, and none can make it worse. A
        # rejected article needs a person, and calling a model to tidy its
        # sentences would be spending money on something nobody will publish.
        return ActionDecision(
            action=ReviewAction.REJECT,
            content_status=content,
            style_verdict=_verdict_of(review),
            reasons=("content integrity verdict is REJECT",),
        )

    style = review.style_review
    active = style_is_active(article_type)
    repairs: tuple[HumanStyleFinding, ...] = ()
    reasons: list[str] = []

    if content is not ReviewStatus.PASS:
        reasons.append(f"content integrity verdict is {content}")

    if active and style is not None and style.style_verdict is StyleVerdict.NEEDS_REVISION:
        repairs = tuple(findings_requiring_repair(style))
        reasons.append(
            f"human style verdict is {style.style_verdict} ({len(repairs)} finding(s) to repair)"
        )

    if content is not ReviewStatus.PASS or repairs:
        return ActionDecision(
            action=ReviewAction.FINALIZE,
            content_status=content,
            style_verdict=_verdict_of(review),
            style_findings=repairs,
            reasons=tuple(reasons),
        )

    return ActionDecision(
        action=ReviewAction.PASS_THROUGH,
        content_status=content,
        style_verdict=_verdict_of(review),
        reasons=("content integrity PASS with nothing to repair",),
    )


def _verdict_of(review: ReviewResult) -> StyleVerdict | None:
    """The style verdict, or ``None`` when the review carries none.

    A historical review has no style object, and reporting ``PASS`` for it would
    claim a judgement nobody made.
    """
    return None if review.style_review is None else review.style_review.style_verdict


__all__ = [
    "STYLE_ACTIVE_TYPES",
    "ActionDecision",
    "ReviewAction",
    "effective_action",
    "style_is_active",
]
