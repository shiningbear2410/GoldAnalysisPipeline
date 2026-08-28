"""Finalizer stage contracts.

The finalizer is an **editor**, not a second analyst. Everything in this module
is shaped by that: the model returns a revised article and an account of what it
did to each issue, and nothing else. It does not get to restate the facts, add
claims, or re-score the piece.

The account is the important half. A revision that silently ignores an issue
looks identical to one that fixed it, so every issue the reviewer raised must be
answered by name - and for the severe ones, answered with ``APPLIED``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import Field, field_validator

from goldpipeline.schemas.common import StrictModel, UtcDatetime, utc_now
from goldpipeline.schemas.review import (
    PrecheckFinding,
    ReviewStatus,
    Severity,
)

FINALIZER_SCHEMA_VERSION = "1.0.0"
"""Version of the finalizer artifact contract."""

MAX_ARTICLE_CHARS = 20_000
MAX_RESOLUTIONS = 40
MAX_DESCRIPTION_CHARS = 500
MIN_ARTICLE_CHARS = 40


@dataclass(frozen=True)
class FinalizerPrompt:
    """A rendered finalizer prompt.

    Lives here for the same reason as the writer and reviewer prompts: the
    adapter layer sends it, the service layer builds it, and schemas is the one
    layer both may depend on.
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


class FinalizationMode(StrEnum):
    """How the final article was produced.

    * ``PASSTHROUGH`` - the review passed, so the draft is the final article,
      byte for byte. No model was called.
    * ``REVISED`` - the review asked for changes and a model made them.
    """

    PASSTHROUGH = "PASSTHROUGH"
    REVISED = "REVISED"


class ResolutionStatus(StrEnum):
    """What the finalizer did about one issue.

    * ``APPLIED`` - the article was changed to address it.
    * ``NOT_APPLICABLE`` - on inspection the issue does not hold. Permitted only
      for LOW and MEDIUM issues, and only with a reason.
    * ``BLOCKED`` - the issue is real but cannot be fixed by editing alone.
      Also LOW/MEDIUM only; a severe issue that cannot be edited away means the
      article should not have been finalized.
    """

    APPLIED = "APPLIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    BLOCKED = "BLOCKED"


MANDATORY_SEVERITIES = frozenset({Severity.HIGH, Severity.CRITICAL})
"""Severities that must be resolved with ``APPLIED``.

An editor is not entitled to decide that a wrong price, an invented indicator or
a foreign instrument is "not applicable". Those are exactly the failures the
review exists to catch, and declining them is the one escape hatch that would
make the whole chain decorative.
"""


class IssueResolution(StrictModel):
    """The finalizer's account of one review issue."""

    issue_id: str = Field(
        min_length=1,
        max_length=64,
        description="Must match an issue_id from gpt_review.json exactly.",
    )
    resolution: ResolutionStatus
    description: str = Field(
        min_length=1,
        max_length=MAX_DESCRIPTION_CHARS,
        description="What was changed, or why nothing was. One or two sentences.",
    )

    @field_validator("description")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a resolution must say what was done")
        return value.strip()


class FinalizerWarning(StrictModel):
    """Something the finalizer noticed while editing."""

    message: str = Field(min_length=1, max_length=600)


class FinalizerModelOutput(StrictModel):
    """The structured response a finalizer model must return.

    Note what is absent: no score, no verdict, no new claims, no metadata. The
    model revises text and reports what it did; everything else is stamped by
    the pipeline.
    """

    run_id: str = Field(
        min_length=1,
        max_length=64,
        description="Echo of the run id, checked against the real one.",
    )
    article: str = Field(max_length=MAX_ARTICLE_CHARS, description="The revised article.")
    issue_resolutions: list[IssueResolution] = Field(
        default_factory=list, max_length=MAX_RESOLUTIONS
    )
    warnings: list[FinalizerWarning] = Field(default_factory=list, max_length=MAX_RESOLUTIONS)

    @field_validator("article")
    @classmethod
    def _reject_blank_article(cls, value: str) -> str:
        """An article of whitespace is an empty article, not a short one."""
        if not value.strip():
            raise ValueError("the final article must not be empty")
        return value.strip()

    @field_validator("issue_resolutions")
    @classmethod
    def _resolution_ids_are_unique(cls, value: list[IssueResolution]) -> list[IssueResolution]:
        seen = [item.issue_id for item in value]
        duplicates = sorted({item for item in seen if seen.count(item) > 1})
        if duplicates:
            raise ValueError(f"each issue may be resolved once; repeated: {duplicates}")
        return value


class FinalizerUsage(StrictModel):
    """Provider usage metadata. Counts and opaque ids only."""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    request_id: str | None = Field(default=None, max_length=200)
    stop_reason: str | None = Field(default=None, max_length=64)


class FinalizerResult(StrictModel):
    """The ``claude_finalizer.json`` artifact.

    Carries the digests of every input and of the article it produced, so a
    later stage can prove the finalization describes the artifacts in front of
    it rather than earlier versions of them.
    """

    schema_version: str = Field(default=FINALIZER_SCHEMA_VERSION)
    run_id: str
    stage: str = Field(default="claude_finalizer")

    finalization_mode: FinalizationMode
    review_status: ReviewStatus = Field(description="The verdict that led here.")
    provider_called: bool = Field(
        description="False for a passthrough. The audit trail for cost and for drift."
    )

    final_file: str = Field(description="Name of the markdown artifact holding the article.")
    final_article_sha256: str = Field(min_length=64, max_length=64)
    article_chars: int = Field(ge=1)

    issue_resolutions: list[IssueResolution] = Field(default_factory=list)
    warnings: list[FinalizerWarning] = Field(default_factory=list)
    postcheck_findings: list[PrecheckFinding] = Field(
        default_factory=list,
        description="Deterministic findings on the final article, after revision.",
    )

    model: str | None = Field(
        default=None, description="Model that revised it, or null for a passthrough."
    )
    provider: str | None = Field(default=None)
    prompt_version: str | None = Field(
        default=None, description="Null for a passthrough - no prompt was rendered."
    )
    created_at: UtcDatetime = Field(default_factory=utc_now)

    context_sha256: str = Field(min_length=64, max_length=64)
    original_draft_sha256: str = Field(min_length=64, max_length=64)
    writer_metadata_sha256: str = Field(min_length=64, max_length=64)
    review_sha256: str = Field(min_length=64, max_length=64)

    usage: FinalizerUsage = Field(default_factory=FinalizerUsage)

    @property
    def applied_count(self) -> int:
        """How many issues were actually fixed."""
        return sum(
            1 for item in self.issue_resolutions if item.resolution is ResolutionStatus.APPLIED
        )


__all__ = [
    "FINALIZER_SCHEMA_VERSION",
    "MANDATORY_SEVERITIES",
    "MAX_ARTICLE_CHARS",
    "MIN_ARTICLE_CHARS",
    "FinalizationMode",
    "FinalizerModelOutput",
    "FinalizerPrompt",
    "FinalizerResult",
    "FinalizerUsage",
    "FinalizerWarning",
    "IssueResolution",
    "ResolutionStatus",
]
