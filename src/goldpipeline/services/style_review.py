"""The human-style axis: when it applies, what it decides, and what it does not.

Round 6.4f, and the shape of this module is the round's whole safety argument.

The reviewer can now judge writing. The finalizer has not yet been taught to
repair writing. If style could reach the verdict today, a NEEDS_REVISION raised
for voice would be handed to a finalizer whose instructions are about wrong
numbers, and it would rewrite a factually sound article by guesswork. So style
is computed for real, recorded in full, and connected to nothing.

**Shadow mode is structural, not a flag.** There is no boolean here whose
default could drift, and no branch that a future edit might flip by accident.
The style verdict is simply never read by
:mod:`goldpipeline.services.review_policy`, by the orchestrator, or by the
finalizer prompt - and tests assert each of those absences directly. Round 6.4g
activates this by *adding* a caller, which is a visible change in a diff, rather
than by changing a default, which is not.

The one place the style axis can affect production is deliberate and narrow:
under a style-aware prompt, an ANALYSIS review that comes back with no style
object at all is refused. That is a plumbing failure - the prompt asked for
something and the model did not return it - not a judgement about the article,
and refusing is how every other missing required field in this pipeline behaves.
Silently accepting it would mean the axis quietly stops being computed and
nobody finds out until a later round depends on it.
"""

from __future__ import annotations

from goldpipeline.domain.errors import ReviewResponseError
from goldpipeline.prompts import GOLD_REVIEWER_V2
from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.article_contract import HUMAN_STYLE_TYPES
from goldpipeline.schemas.review import (
    HumanStyleAssessment,
    HumanStyleFinding,
    HumanStyleReview,
    ReviewModelOutput,
    StyleSeverity,
    StyleVerdict,
)

STYLE_AWARE_PROMPTS = frozenset({GOLD_REVIEWER_V2})
"""Reviewer prompts that ask for a style judgement.

A set rather than a version comparison: prompt ids are names, not an ordering,
and "v2 or later" is not something a string can be asked. A future v3 that keeps
the style axis joins this set explicitly.
"""

MEDIUM_FINDINGS_FOR_REVISION = 3
"""Three MEDIUM findings mean the piece reads wrong, not that it slipped once.

Below this the findings are worth recording and not worth a rewrite: an article
with two ordinary imperfections is an ordinary article. The threshold is stated
here rather than inside the verdict function so that Round 6.4g can argue with
the number without having to re-derive the rule.
"""


def applies_to(article_type: ArticleType) -> bool:
    """Whether this article type has a voice worth judging.

    Reuses :data:`HUMAN_STYLE_TYPES` from the article contract rather than
    declaring a second list. A rendered ``TRADE_PLAN`` has no prose voice, and
    a round that added one here while the contract said otherwise would have
    two answers to the same question.

    ``NEWS_DIGEST`` is in that set and is not yet producible at all, so in
    practice this is ``ANALYSIS`` today. That is deliberate: the day the digest
    becomes producible it should already be reviewable for voice, rather than
    needing this function edited as part of an unrelated round.
    """
    return article_type in HUMAN_STYLE_TYPES


def requires_style_review(*, prompt_version: str, article_type: ArticleType) -> bool:
    """Whether this review must come back carrying a style judgement.

    Both conditions, not either: an older prompt was never asked for one, and an
    article type with no voice must not be given one.
    """
    return prompt_version in STYLE_AWARE_PROMPTS and applies_to(article_type)


def style_verdict_for(findings: list[HumanStyleFinding]) -> StyleVerdict:
    """Derive the style verdict from the findings.

    One HIGH finding means the piece reads as the wrong product. Three MEDIUM
    findings mean no single sentence is the problem and the article still does
    not sound like a person. Anything less is recorded and left alone.

    LOW findings never accumulate into a revision however many there are. A
    reviewer noticing six small things is a reviewer paying attention, and
    turning attention into a rewrite is how a style gate becomes a nuisance
    nobody leaves switched on.
    """
    if any(finding.severity is StyleSeverity.HIGH for finding in findings):
        return StyleVerdict.NEEDS_REVISION

    mediums = sum(1 for finding in findings if finding.severity is StyleSeverity.MEDIUM)
    if mediums >= MEDIUM_FINDINGS_FOR_REVISION:
        return StyleVerdict.NEEDS_REVISION

    return StyleVerdict.PASS


REVISION_SEVERITIES = frozenset({StyleSeverity.HIGH, StyleSeverity.MEDIUM})
"""Severities a revision must actually repair.

Exactly the severities :func:`style_verdict_for` counts, and that is the point
of stating it here rather than in the finalizer: the set of findings that *can*
cause a revision and the set the revision *must* fix are the same set, and two
modules deciding that separately would drift apart the first time either
threshold moved.

``LOW`` is excluded for the reason Round 6.4f gave for not accumulating it: a
reviewer noticing six small things is a reviewer paying attention, and turning
attention into mandatory edits is how a style gate becomes one nobody leaves on.
A LOW finding is still sent to the finalizer and still answered - it may simply
be answered with "left alone".
"""


def findings_requiring_repair(review: HumanStyleReview) -> list[HumanStyleFinding]:
    """The findings a style-driven revision must resolve.

    Empty when the verdict is PASS: a review nobody is acting on imposes no
    obligations, even if it recorded observations.
    """
    if review.style_verdict is not StyleVerdict.NEEDS_REVISION:
        return []
    return [f for f in review.findings if f.severity in REVISION_SEVERITIES]


def build_style_review(assessment: HumanStyleAssessment) -> HumanStyleReview:
    """Stamp the model's assessment with the verdict its findings imply."""
    return HumanStyleReview(
        style_score=assessment.style_score,
        style_verdict=style_verdict_for(assessment.findings),
        summary=assessment.summary,
        findings=list(assessment.findings),
    )


def resolve_style_review(
    output: ReviewModelOutput,
    *,
    prompt_version: str,
    article_type: ArticleType,
) -> HumanStyleReview | None:
    """The style judgement to record, or ``None`` when there is none to record.

    Args:
        output: The model's validated response.
        prompt_version: Which reviewer prompt produced it.
        article_type: What the Run is writing.

    Returns:
        A :class:`HumanStyleReview` when one was asked for and returned, or
        ``None`` when this article type has no voice, or when an older prompt
        never requested one.

    Raises:
        ReviewResponseError: A style-aware prompt reviewed an article type with
            a voice and the response carried no style object. See the module
            docstring: this is a plumbing failure, not a verdict.
    """
    if not requires_style_review(prompt_version=prompt_version, article_type=article_type):
        # An assessment volunteered for a type with no voice is dropped rather
        # than stored. Recording a style verdict for a rendered document would
        # be a claim about a judgement nobody was asked to make.
        return None

    if output.style_review is None:
        raise ReviewResponseError(
            f"prompt {prompt_version} reviews {article_type} for human style, "
            "but the response carried no style_review",
            prompt_version=prompt_version,
            article_type=str(article_type),
        )

    return build_style_review(output.style_review)


__all__ = [
    "MEDIUM_FINDINGS_FOR_REVISION",
    "REVISION_SEVERITIES",
    "STYLE_AWARE_PROMPTS",
    "applies_to",
    "build_style_review",
    "findings_requiring_repair",
    "requires_style_review",
    "resolve_style_review",
    "style_verdict_for",
]
