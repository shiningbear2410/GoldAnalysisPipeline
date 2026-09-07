"""What a digest revision may contain, and therefore what it cannot.

Round 6.5c.3. The analysis finalizer returns an article, and everything that
protects the published text is a *check* applied afterwards: did the date
survive, is the disclaimer still there, did a price change. Each of those is a
rule that could be forgotten, and each is a rule the model could break.

A digest revision returns none of that. It returns items, a balance, and
claims - the same three things the writer was asked for - and code renders the
article around them from the immutable snapshot. So the questions the analysis
finalizer has to *ask* are questions a digest revision cannot *raise*:

* the title cannot drift, because there is no field to put a title in;
* the window cannot move, because the model is never given one to return;
* an item's timestamp cannot change, because the renderer reads it from the
  source item and the revision has nowhere to write one;
* the price block cannot be edited, because it is not part of this schema;
* the impact labels cannot be reworded, because the model returns an enum and
  Python owns the phrase;
* the disclaimer cannot be dropped, because the renderer places it.

``extra="forbid"`` on :class:`~goldpipeline.schemas.common.StrictModel` is what
makes that a guarantee rather than a habit: a model that decides to return an
``article`` field has its response refused, not partially honoured.

What *is* here is the editorial judgement a repair legitimately changes, plus
the account of what was repaired - the same two resolution vocabularies the
analysis finalizer uses, reused rather than re-invented so that "answered" means
one thing across both products.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from goldpipeline.schemas.common import StrictModel
from goldpipeline.schemas.finalizer import (
    MAX_RESOLUTIONS,
    FinalizerWarning,
    IssueResolution,
    StyleResolution,
)
from goldpipeline.schemas.news_digest import (
    MAX_BALANCE_CHARS,
    MAX_DIGEST_ITEMS,
    MIN_DIGEST_ITEMS,
    DigestItem,
)
from goldpipeline.schemas.writer import NewsClaim, WriterStatus

DIGEST_FINALIZER_SCHEMA_VERSION = "1.0.0"


class DigestEditorialRevision(StrictModel):
    """The editorial content of a repaired digest. Nothing else.

    Deliberately a separate type from
    :class:`~goldpipeline.schemas.news_digest.DigestEditorial` rather than a
    reuse of it. That one carries ``run_id`` and ``status`` because it is a
    complete answer to "write me a digest"; this one is a *fragment* handed back
    inside a larger response, and giving it a second run id would create two
    places for them to disagree.

    The three fields are exactly the three the writer owned. A repair that
    wanted to change anything else is asking for something the pipeline does not
    permit, and there is no field for it to ask in.
    """

    items: tuple[DigestItem, ...] = Field(
        min_length=MIN_DIGEST_ITEMS,
        max_length=MAX_DIGEST_ITEMS,
        description=(
            "The selected items after repair, in reading order. Every "
            "`news_item_id` must still name one of the originally collected "
            "items - a repair may drop or reorder, never introduce."
        ),
    )
    balance: str = Field(
        min_length=1,
        max_length=MAX_BALANCE_CHARS,
        description="The 🧭 Cán cân paragraph after repair.",
    )
    news_claims: tuple[NewsClaim, ...] = Field(
        default=(),
        max_length=MAX_RESOLUTIONS,
        description=(
            "Provenance for the repaired text. Re-declared rather than carried "
            "over: a claim quotes a statement, and a statement that was edited "
            "is no longer the one the original claim named."
        ),
    )

    @field_validator("items")
    @classmethod
    def _ids_are_unique(cls, value: tuple[DigestItem, ...]) -> tuple[DigestItem, ...]:
        """One item, one bullet.

        A duplicate is not a formatting slip - it is the same story reported
        twice, which the writer prompt forbids and which a reader would read as
        two separate events.
        """
        seen = [item.news_item_id for item in value]
        duplicates = sorted({name for name in seen if seen.count(name) > 1})
        if duplicates:
            raise ValueError(f"an item may appear once: {', '.join(duplicates)}")
        return value


class DigestFinalizerModelOutput(StrictModel):
    """The structured response a digest finalizer model must return.

    Note what is absent, and that its absence is the design: no article, no
    title, no window, no timestamps, no price figures, no disclaimer. The model
    revises editorial content and reports what it did; the pipeline renders
    everything else from facts it captured before any model was consulted.
    """

    run_id: str = Field(
        min_length=1,
        max_length=64,
        description="Echo of the run id, checked against the real one.",
    )
    status: WriterStatus = Field(
        description="COMPLETED when the repair was made, or the honest alternative."
    )
    editorial: DigestEditorialRevision
    issue_resolutions: list[IssueResolution] = Field(
        default_factory=list,
        max_length=MAX_RESOLUTIONS,
        description="One entry per content issue the review raised.",
    )
    style_resolutions: list[StyleResolution] = Field(
        default_factory=list,
        max_length=MAX_RESOLUTIONS,
        description="One entry per human-style finding requiring repair.",
    )
    warnings: list[FinalizerWarning] = Field(default_factory=list, max_length=MAX_RESOLUTIONS)

    @field_validator("issue_resolutions")
    @classmethod
    def _issue_ids_are_unique(cls, value: list[IssueResolution]) -> list[IssueResolution]:
        seen = [item.issue_id for item in value]
        duplicates = sorted({name for name in seen if seen.count(name) > 1})
        if duplicates:
            raise ValueError(f"an issue may be resolved once: {', '.join(duplicates)}")
        return value

    @field_validator("style_resolutions")
    @classmethod
    def _style_ids_are_unique(cls, value: list[StyleResolution]) -> list[StyleResolution]:
        seen = [item.finding_id for item in value]
        duplicates = sorted({name for name in seen if seen.count(name) > 1})
        if duplicates:
            raise ValueError(f"a finding may be resolved once: {', '.join(duplicates)}")
        return value


__all__ = [
    "DIGEST_FINALIZER_SCHEMA_VERSION",
    "DigestEditorialRevision",
    "DigestFinalizerModelOutput",
]
