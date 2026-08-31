"""Orchestration contracts.

The orchestrator decides *nothing* about an article. Every verdict in here was
reached by a stage that already existed - the reviewer's, the finalizer's, the
gate's, the publisher's - and this module only gives those verdicts a shape a
caller can read in one place.

That is why there is no ``PipelineVerdict`` and no severity scale here. A
pipeline execution has three interesting properties and no more: how far it got,
why it stopped, and what each stage said on the way.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from goldpipeline.schemas.common import StrictModel, UtcDatetime, utc_now
from goldpipeline.schemas.manifest import RunStatus
from goldpipeline.schemas.publish import Decision
from goldpipeline.schemas.publisher import PublishStatus

ORCHESTRATION_SCHEMA_VERSION = "1.0.0"
"""Version of the pipeline execution contract."""


class PipelineMode(StrEnum):
    """How far one invocation is allowed to go.

    The mode is a *ceiling*, not a target: a Run that stops early because a gate
    declined has still honoured its mode. Publishing is its own value rather
    than a flag on the others, so that nothing can reach Telegram without the
    word appearing in the request.
    """

    GENERATE_ONLY = "GENERATE_ONLY"
    """Stop at ``FINALIZED``. No publish decision is made."""

    READY_FOR_PUBLISH = "READY_FOR_PUBLISH"
    """Stop at ``READY_TO_PUBLISH`` or ``PUBLISH_BLOCKED``. The safe default."""

    PUBLISH = "PUBLISH"
    """Continue into the publisher, but only from an approved decision."""


class PipelineStage(StrEnum):
    """The six stages, in the only order they may run."""

    NORMALIZE = "NORMALIZE"
    WRITE = "WRITE"
    REVIEW = "REVIEW"
    FINALIZE = "FINALIZE"
    GATE = "GATE"
    PUBLISH = "PUBLISH"


class StageOutcome(StrEnum):
    """What happened to one stage in one invocation.

    ``SKIPPED`` and ``BLOCKED`` are both non-events, and conflating them would
    hide the thing an operator most needs to see. ``SKIPPED`` means the stage
    was already done, or the mode did not reach it; ``BLOCKED`` means it ran, or
    would have run, and the pipeline deliberately stopped there.
    """

    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class PipelineStatus(StrEnum):
    """How the invocation as a whole ended.

    The split that matters is between *a gate spoke* and *something broke*:

    * ``COMPLETED`` - the pipeline reached the ceiling its mode allowed.
    * ``ALREADY_COMPLETED`` - it was already there. Nothing ran, nothing was
      called, no provider or network was touched.
    * ``BLOCKED`` - a stage declined: a ``REJECT``, a blocked gate, or a
      delivery that was not fully confirmed. Retrying will not help.
    * ``NOT_RESUMABLE`` - the Run is in a publish-side state a human must
      resolve. Distinct from ``BLOCKED`` because nothing declined anything;
      the orchestrator simply refuses to touch it.
    * ``FAILED`` - an execution failure. See ``error``.
    """

    COMPLETED = "COMPLETED"
    ALREADY_COMPLETED = "ALREADY_COMPLETED"
    BLOCKED = "BLOCKED"
    NOT_RESUMABLE = "NOT_RESUMABLE"
    FAILED = "FAILED"


class PipelineEvent(StrEnum):
    """Audit vocabulary written into the Run manifest.

    Small on purpose. These are recorded alongside the events the stages write
    for themselves, so a manifest reads as one continuous story rather than two
    parallel ones.
    """

    RUN_CREATED = "RUN_CREATED"
    WRITER_COMPLETED = "WRITER_COMPLETED"
    REVIEW_COMPLETED = "REVIEW_COMPLETED"
    FINALIZER_COMPLETED = "FINALIZER_COMPLETED"
    GATE_APPROVED = "GATE_APPROVED"
    GATE_BLOCKED = "GATE_BLOCKED"
    PUBLISH_COMPLETED = "PUBLISH_COMPLETED"
    PIPELINE_STOPPED = "PIPELINE_STOPPED"
    PIPELINE_FAILED = "PIPELINE_FAILED"


class StageExecution(StrictModel):
    """One stage's slot in one invocation."""

    stage: PipelineStage
    started_at: UtcDatetime
    completed_at: UtcDatetime
    outcome: StageOutcome
    detail: str | None = Field(
        default=None,
        description=(
            "The stage's own word for what it decided - PASS, PASSTHROUGH, "
            "APPROVED, PUBLISHED - so a summary can quote it rather than "
            "re-derive it."
        ),
    )


class PipelineExecutionResult(StrictModel):
    """Everything one orchestrator invocation did.

    Deliberately not a copy of the Run. The artifacts are the record of *what
    the pipeline produced*; this is the record of *how one invocation ran*, and
    duplicating article text or digests here would create a second source of
    truth that could disagree with the first.
    """

    schema_version: str = Field(default=ORCHESTRATION_SCHEMA_VERSION)
    run_id: str
    started_at: UtcDatetime = Field(default_factory=utc_now)
    completed_at: UtcDatetime = Field(default_factory=utc_now)
    mode: PipelineMode
    final_stage: PipelineStage | None = Field(
        default=None,
        description="The last stage that ran, or None when nothing needed to run.",
    )
    status: PipelineStatus
    run_status: RunStatus = Field(
        description=(
            "Where the Run itself ended up. Distinct from `status`, which "
            "describes the invocation: a Run can be READY_TO_PUBLISH while the "
            "invocation that got it there is COMPLETED, and both matter."
        )
    )
    stages: list[StageExecution] = Field(default_factory=list)
    publish_decision: Decision | None = None
    publish_status: PublishStatus | None = None
    error: dict[str, object] | None = Field(
        default=None,
        description="Serialized PipelineError, present only when status is FAILED.",
    )

    @property
    def succeeded(self) -> bool:
        """Whether the invocation ended without a stop or a failure."""
        return self.status in (PipelineStatus.COMPLETED, PipelineStatus.ALREADY_COMPLETED)


__all__ = [
    "ORCHESTRATION_SCHEMA_VERSION",
    "PipelineEvent",
    "PipelineExecutionResult",
    "PipelineMode",
    "PipelineStage",
    "PipelineStatus",
    "StageExecution",
    "StageOutcome",
]
