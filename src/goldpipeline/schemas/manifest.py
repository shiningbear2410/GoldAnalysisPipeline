"""Run manifest - the ledger of what a Run contains and how it got there.

Immutability model:

* **source files and artifacts are write-once.** Once ``ohlc.json`` exists in a
  Run it is never rewritten; the storage layer refuses.
* **the manifest is append-oriented.** It records status transitions and the
  files produced, so it is rewritten as the Run progresses. It is the only
  mutable file in a Run directory, and it only ever grows.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from goldpipeline import CONTEXT_SCHEMA_VERSION, PIPELINE_VERSION
from goldpipeline.schemas.common import UtcDatetime, utc_now


class RunStatus(StrEnum):
    """Where a Run has got to.

    Kept small on purpose - this is a progress marker, not a workflow engine.
    Each stage adds at most one value:

    * ``CREATED``    - directory exists, inputs not yet validated
    * ``NORMALIZED`` - Round 1 done, ``context.json`` written
    * ``DRAFTED``    - Round 2 done, writer artifacts written
    * ``REVIEWED``   - Round 3 done, ``gpt_review.json`` written
    * ``FINALIZED``  - Round 4 done, finalizer artifacts written
    * ``READY_TO_PUBLISH`` - Round 5 approved it for publication
    * ``PUBLISH_BLOCKED``  - Round 5 refused it
    * ``FAILED``     - a stage failed; see ``error`` for which and why

    A failed writer or reviewer stage leaves the Run at its previous status, not
    ``FAILED``: the earlier artifacts are still valid and the stage can be
    retried. ``FAILED`` is reserved for a Run whose own inputs are unusable.

    ``REVIEWED`` says a review happened, **not** that it passed. The verdict
    lives in ``gpt_review.json``; a Run may legitimately be ``REVIEWED`` with a
    status of ``REJECT`` - and such a Run never reaches ``FINALIZED``, because
    the finalizer refuses to auto-correct it.

    ``PUBLISH_BLOCKED`` is likewise a completed outcome, not a failure: the gate
    ran and said no. Neither it nor ``READY_TO_PUBLISH`` means anything was sent
    anywhere - publication is Round 6, and nothing here is ever called
    ``PUBLISHED``.
    """

    CREATED = "CREATED"
    NORMALIZED = "NORMALIZED"
    DRAFTED = "DRAFTED"
    REVIEWED = "REVIEWED"
    FINALIZED = "FINALIZED"
    READY_TO_PUBLISH = "READY_TO_PUBLISH"
    PUBLISH_BLOCKED = "PUBLISH_BLOCKED"
    FAILED = "FAILED"


class MutableModel(BaseModel):
    """Manifest models are mutable: the manifest is a ledger updated per stage."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ArtifactRef(MutableModel):
    """A file stored inside a Run, with the digest needed to prove it unchanged."""

    name: str = Field(description="File name relative to the Run directory.")
    sha256: str = Field(min_length=64, max_length=64, description="Digest of the bytes on disk.")
    size_bytes: int = Field(ge=0)
    written_at: UtcDatetime = Field(default_factory=utc_now)


class RunEvent(MutableModel):
    """One stage transition, for tracing a Run after the fact."""

    at: UtcDatetime = Field(default_factory=utc_now)
    stage: str
    status: str
    message: str | None = None


class RunError(MutableModel):
    """The failure that stopped a Run, in serializable form."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class RunManifest(MutableModel):
    """Top-level description of a Run directory."""

    run_id: str
    created_at: UtcDatetime = Field(default_factory=utc_now)
    updated_at: UtcDatetime = Field(default_factory=utc_now)
    pipeline_version: str = Field(default=PIPELINE_VERSION)
    schema_version: str = Field(default=CONTEXT_SCHEMA_VERSION)
    status: RunStatus = Field(default=RunStatus.CREATED)
    source_files: list[ArtifactRef] = Field(
        default_factory=list, description="Immutable inputs captured for audit."
    )
    artifact_files: list[ArtifactRef] = Field(
        default_factory=list, description="Files this pipeline derived."
    )
    events: list[RunEvent] = Field(default_factory=list)
    error: RunError | None = None

    def record_event(self, stage: str, status: str, message: str | None = None) -> None:
        """Append a stage transition and bump ``updated_at``."""
        self.events.append(RunEvent(stage=stage, status=status, message=message))
        self.updated_at = utc_now()


__all__ = ["ArtifactRef", "MutableModel", "RunError", "RunEvent", "RunManifest", "RunStatus"]
