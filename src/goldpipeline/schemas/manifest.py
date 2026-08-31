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
    * ``PUBLISHING``       - Round 6 has written its intent and may be sending
    * ``PUBLISHED``        - every chunk confirmed by Telegram
    * ``PARTIALLY_PUBLISHED`` - some chunks confirmed, then an explicit refusal
    * ``PUBLISH_FAILED``   - explicitly refused before anything was delivered
    * ``PUBLISH_UNCERTAIN``- delivery could not be determined; needs a human
    * ``FAILED``     - a stage failed; see ``error`` for which and why

    A failed writer or reviewer stage leaves the Run at its previous status, not
    ``FAILED``: the earlier artifacts are still valid and the stage can be
    retried. ``FAILED`` is reserved for a Run whose own inputs are unusable.

    ``REVIEWED`` says a review happened, **not** that it passed. The verdict
    lives in ``gpt_review.json``; a Run may legitimately be ``REVIEWED`` with a
    status of ``REJECT`` - and such a Run never reaches ``FINALIZED``, because
    the finalizer refuses to auto-correct it.

    ``PUBLISH_BLOCKED`` is likewise a completed outcome, not a failure: the gate
    ran and said no. ``READY_TO_PUBLISH`` means Round 5 approved it, not that
    anything was sent.

    ``PUBLISHING`` is the one status that is not terminal. It exists so that a
    process killed mid-send leaves a trace: the next run sees an intent with no
    result and refuses to send again, because Telegram may already have the
    message.
    """

    CREATED = "CREATED"
    NORMALIZED = "NORMALIZED"
    DRAFTED = "DRAFTED"
    REVIEWED = "REVIEWED"
    FINALIZED = "FINALIZED"
    READY_TO_PUBLISH = "READY_TO_PUBLISH"
    PUBLISH_BLOCKED = "PUBLISH_BLOCKED"
    PUBLISHING = "PUBLISHING"
    PUBLISHED = "PUBLISHED"
    PARTIALLY_PUBLISHED = "PARTIALLY_PUBLISHED"
    PUBLISH_FAILED = "PUBLISH_FAILED"
    PUBLISH_UNCERTAIN = "PUBLISH_UNCERTAIN"
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


class RunProvenance(MutableModel):
    """Where a Run's inputs came from, in the words of the adapters that fetched them.

    Separate from the source *files*, which record what a provider said. This
    records the act of asking: which inbox event, which broker symbol, when the
    request went out and when the answer came back. Together they answer the
    question an audit actually has - *which analysis, and which candles, is this
    article built on?*

    The per-adapter halves are open dictionaries on purpose. A live Telegram
    reader and a MetaTrader terminal have almost nothing in common to describe,
    and a union type covering both would grow a branch for every provider ever
    added. Adapters are expected to keep credentials out of them; the ones here
    record a setting's name when they need to refer to it, never its value.
    """

    analysis_origin: str = Field(description="Human-readable source of the analysis.")
    market_origin: str = Field(description="Human-readable source of the market data.")
    analysis: dict[str, Any] = Field(default_factory=dict)
    market: dict[str, Any] = Field(default_factory=dict)


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
    provenance: RunProvenance | None = Field(
        default=None,
        description="Where the inputs came from. Absent on Runs created before Round 8.",
    )
    error: RunError | None = None

    def record_event(self, stage: str, status: str, message: str | None = None) -> None:
        """Append a stage transition and bump ``updated_at``."""
        self.events.append(RunEvent(stage=stage, status=status, message=message))
        self.updated_at = utc_now()


__all__ = [
    "ArtifactRef",
    "MutableModel",
    "RunError",
    "RunEvent",
    "RunManifest",
    "RunProvenance",
    "RunStatus",
]
