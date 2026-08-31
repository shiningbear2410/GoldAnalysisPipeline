"""Ingestion contracts: the ledger, and what one ingestion attempt concluded.

The ledger answers exactly one question - *which Run was created for this
event?* - and it answers it durably, before the Run exists. That ordering is the
whole design, and it is the same argument Round 6 makes about the publish
intent: a process can die between "decided to act" and "recorded that it acted",
and the only way to tell those apart afterwards is to have written something
down first.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from goldpipeline.schemas.common import StrictModel, UtcDatetime, utc_now

INGESTION_SCHEMA_VERSION = "1.0.0"


class LedgerState(StrEnum):
    """How far one event got.

    ``RESERVED`` is the state that matters. It says a run id was allocated and a
    Run was about to be created; it does *not* say whether that succeeded. An
    event stuck in ``RESERVED`` is never re-ingested automatically, because
    doing so is how one analysis becomes two articles.
    """

    RESERVED = "RESERVED"
    INGESTED = "INGESTED"
    ABANDONED = "ABANDONED"


class IngestOutcome(StrEnum):
    """What one ingestion attempt concluded.

    ``ALREADY_INGESTED`` is a success, not an error. A producer that retries
    after a lost acknowledgement must get a calm answer with the original run
    id, or it will retry forever.
    """

    INGESTED = "INGESTED"
    ALREADY_INGESTED = "ALREADY_INGESTED"
    CONFLICT = "CONFLICT"
    UNRESOLVED = "UNRESOLVED"
    MARKET_UNAVAILABLE = "MARKET_UNAVAILABLE"
    INVALID_PAYLOAD = "INVALID_PAYLOAD"
    RUN_FAILED = "RUN_FAILED"
    NOTHING_TO_DO = "NOTHING_TO_DO"


class LedgerEntry(StrictModel):
    """The durable ``event_id -> run_id`` mapping.

    Written once when the run id is allocated, then completed in place. The
    identity fields - ``event_id``, ``payload_sha256``, ``run_id`` - are never
    rewritten; only ``state`` and the closing timestamp move.
    """

    schema_version: str = Field(default=INGESTION_SCHEMA_VERSION)
    event_id: str
    source: str
    payload_sha256: str = Field(min_length=64, max_length=64)
    run_id: str = Field(description="Allocated before the Run is created, never reassigned.")
    state: LedgerState = LedgerState.RESERVED
    reserved_at: UtcDatetime = Field(default_factory=utc_now)
    settled_at: UtcDatetime | None = Field(
        default=None, description="When the entry left RESERVED, either way."
    )
    note: str | None = Field(
        default=None, description="Why an entry was abandoned. Safe text only."
    )


class IngestResult(StrictModel):
    """The outcome of one ingestion attempt, for a CLI or a caller to act on."""

    schema_version: str = Field(default=INGESTION_SCHEMA_VERSION)
    outcome: IngestOutcome
    event_id: str | None = None
    run_id: str | None = Field(
        default=None,
        description="Present whenever a Run exists for this event, including on a replay.",
    )
    payload_sha256: str | None = None
    source_path: str | None = Field(default=None, description="Where the event file ended up.")
    failure_code: str | None = Field(
        default=None,
        description=(
            "The originating error's own code, structured. A scheduler has to "
            "tell a closed market from a misconfigured symbol, and parsing that "
            "out of `detail` would make the difference a formatting accident."
        ),
    )
    detail: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether a Run exists for this event as a result of - or before - this attempt."""
        return self.outcome in (IngestOutcome.INGESTED, IngestOutcome.ALREADY_INGESTED)


__all__ = [
    "INGESTION_SCHEMA_VERSION",
    "IngestOutcome",
    "IngestResult",
    "LedgerEntry",
    "LedgerState",
]
