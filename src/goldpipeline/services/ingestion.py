"""Turning an inbox event into a Run, exactly once.

The order below is the whole safety argument:

1.  claim the event with an atomic rename - one consumer owns it;
2.  parse it and hash the producer's exact bytes;
3.  consult the ledger, which can say *already done*, *conflict*, or *unknown*;
4.  fetch the market data **before** anything durable is written;
5.  allocate a run id and **reserve it in the ledger**;
6.  create the Run under that id;
7.  record provenance, settle the ledger, move the event to ``processed/``.

Steps 1-4 write nothing that survives a crash. Step 5 is the hinge: after it,
there is a durable record saying "a Run with this id was about to exist".

**Why the reservation names the run id.** ``create_run`` accepts an explicit id,
so the id can be chosen *before* the Run is created rather than learned after.
That single detail is what makes a crash recoverable without guessing: an event
found stranded in ``processing/`` names exactly one Run, and looking at that
Run's manifest answers whether the work happened. Without it, a stranded event
and a duplicate article are the same observation.

**Nothing is retried automatically.** A market provider that was briefly down
leaves no trace and the event goes back to ``incoming/`` - that is provably safe,
because nothing was reserved. Everything else waits for ``inbox-reconcile`` and a
human. There is no scheduler here and no loop; deciding *when* to run this is
Round 9's job.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from goldpipeline.adapters.base import MarketDataSource
from goldpipeline.adapters.inbox_source import InboxAnalysisSource, parse_event
from goldpipeline.domain.errors import (
    InboxPayloadError,
    MarketDataError,
    PipelineError,
)
from goldpipeline.domain.run_id import generate_run_id
from goldpipeline.schemas.common import utc_now
from goldpipeline.schemas.inbox import AnalysisEvent
from goldpipeline.schemas.ingestion import (
    IngestOutcome,
    IngestResult,
    LedgerEntry,
    LedgerState,
)
from goldpipeline.schemas.manifest import RunStatus
from goldpipeline.services.inbox import ClaimedEvent, Inbox, Ledger
from goldpipeline.services.pipeline import create_run
from goldpipeline.storage.run_store import RunStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionContext:
    """Everything one ingestion attempt needs.

    The market source is passed in rather than built here so the caller decides
    between a live terminal and a file - and so tests can hand over a fake
    without this module knowing what MetaTrader is.
    """

    inbox: Inbox
    store: RunStore
    market_source: MarketDataSource
    expected_symbol: str | None = None

    @property
    def ledger(self) -> Ledger:
        return Ledger(self.inbox.directory("index"))


def ingest_next(context: IngestionContext, *, now: datetime | None = None) -> IngestResult:
    """Claim and ingest the oldest waiting event, if there is one."""
    for candidate in context.inbox.pending():
        claimed = context.inbox.claim(candidate)
        if claimed is not None:
            return ingest_claimed(context, claimed, now=now)
        # Another consumer took it between listing and claiming. Perfectly
        # ordinary; try the next one.
    return IngestResult(outcome=IngestOutcome.NOTHING_TO_DO, detail="no events waiting")


def ingest_file(
    context: IngestionContext, path: Path, *, now: datetime | None = None
) -> IngestResult:
    """Submit a payload file into the inbox and ingest it immediately.

    The convenience path behind ``pipeline-ingest``: it still goes through the
    inbox, so a manually triggered run leaves the same audit trail as one the
    producer submitted.
    """
    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise InboxPayloadError(f"analysis payload could not be read: {path}") from exc
    if not isinstance(payload, dict):
        raise InboxPayloadError(f"analysis payload must be a JSON object: {path}")

    event = parse_event(payload)
    submitted = context.inbox.submit(payload, event_id=event.event_id)
    claimed = context.inbox.claim(submitted)
    if claimed is None:  # pragma: no cover - we created it a line ago
        raise InboxPayloadError(
            f"event {event.event_id!r} was taken by another consumer", event_id=event.event_id
        )
    return ingest_claimed(context, claimed, now=now)


def ingest_claimed(
    context: IngestionContext, path: Path, *, now: datetime | None = None
) -> IngestResult:
    """Ingest an event this process already holds in ``processing/``."""
    moment = now or utc_now()

    try:
        claimed = context.inbox.read(path)
        event = parse_event(claimed.payload)
    except InboxPayloadError as exc:
        landed = context.inbox.reject(path, code=exc.code, reason=exc.message, **exc.details)
        logger.warning("ingest.invalid path=%s code=%s", path.name, exc.code)
        return IngestResult(
            outcome=IngestOutcome.INVALID_PAYLOAD,
            source_path=str(landed),
            failure_code=exc.code,
            detail=exc.message,
        )

    digest = claimed.sha256
    settled = _check_ledger(context, path, event.event_id, digest)
    if settled is not None:
        return settled

    # Fetch before reserving. A terminal that is briefly unreachable must leave
    # no ledger entry and no Run - only then is putting the event back safe.
    try:
        context.market_source.load()
    except MarketDataError as exc:
        returned = context.inbox.release(path)
        logger.warning("ingest.market_unavailable event=%s code=%s", event.event_id, exc.code)
        return IngestResult(
            outcome=IngestOutcome.MARKET_UNAVAILABLE,
            event_id=event.event_id,
            payload_sha256=digest,
            source_path=str(returned),
            failure_code=exc.code,
            detail=f"[{exc.code}] {exc.message}",
        )

    run_id = generate_run_id()
    context.ledger.reserve(
        LedgerEntry(
            event_id=event.event_id,
            source=event.source,
            payload_sha256=digest,
            run_id=run_id,
            state=LedgerState.RESERVED,
            reserved_at=moment,
        )
    )

    return _create(context, claimed, event, run_id=run_id, now=moment)


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------


def _check_ledger(
    context: IngestionContext, path: Path, event_id: str, digest: str
) -> IngestResult | None:
    """Decide what the ledger already knows about this event.

    Three answers matter, and they are not the same:

    * *seen, same bytes, finished* - a producer retry. Answer calmly with the
      original run id, or it will retry forever.
    * *seen, different bytes* - an id was reused for different content. Refuse
      and change nothing; the original mapping is the audit trail for an article
      that may already be published.
    * *seen, unfinished* - a previous attempt died mid-flight. Whether a Run
      exists is knowable, but not from here.
    """
    entry = context.ledger.read(event_id)
    if entry is None:
        return None

    if entry.payload_sha256 != digest:
        landed = context.inbox.reject(
            path,
            code="EVENT_ID_CONFLICT",
            reason=(
                f"event_id {event_id!r} was already used for different content. "
                "The original mapping is left untouched; resubmit under a new event_id."
            ),
            event_id=event_id,
            recorded_sha256=entry.payload_sha256,
            submitted_sha256=digest,
            recorded_run_id=entry.run_id,
        )
        logger.warning("ingest.conflict event=%s run=%s", event_id, entry.run_id)
        return IngestResult(
            outcome=IngestOutcome.CONFLICT,
            event_id=event_id,
            run_id=entry.run_id,
            payload_sha256=digest,
            source_path=str(landed),
            detail="event_id reused with different content",
        )

    if entry.state is LedgerState.RESERVED:
        landed = context.inbox.reject(
            path,
            code="EVENT_UNRESOLVED",
            reason=(
                f"a previous attempt at event {event_id!r} never finished. Run "
                "`inbox-reconcile` to determine whether run "
                f"{entry.run_id} was created before submitting this again."
            ),
            event_id=event_id,
            reserved_run_id=entry.run_id,
        )
        return IngestResult(
            outcome=IngestOutcome.UNRESOLVED,
            event_id=event_id,
            run_id=entry.run_id,
            payload_sha256=digest,
            source_path=str(landed),
            detail="a previous attempt is unresolved",
        )

    landed = context.inbox.complete(path)
    logger.info("ingest.replay event=%s run=%s state=%s", event_id, entry.run_id, entry.state)
    return IngestResult(
        outcome=(
            IngestOutcome.ALREADY_INGESTED
            if entry.state is LedgerState.INGESTED
            else IngestOutcome.RUN_FAILED
        ),
        event_id=event_id,
        run_id=entry.run_id,
        payload_sha256=digest,
        source_path=str(landed),
        detail=f"already seen; ledger state is {entry.state}",
    )


def _create(
    context: IngestionContext,
    claimed: ClaimedEvent,
    event: AnalysisEvent,
    *,
    run_id: str,
    now: datetime,
) -> IngestResult:
    """Create the Run under the reserved id and settle everything behind it."""
    event_id = event.event_id
    analysis_source = InboxAnalysisSource(event, raw=claimed.raw, origin=f"inbox:{event_id}")

    result = create_run(
        analysis_source=analysis_source,
        market_source=context.market_source,
        store=context.store,
        expected_symbol=context.expected_symbol,
        run_id=run_id,
        now=now,
    )

    if not result.succeeded:
        detail = f"[{result.error.code}] {result.error.message}" if result.error else "unknown"
        context.ledger.settle(event_id, state=LedgerState.ABANDONED, note=detail[:500], now=now)
        landed = context.inbox.reject(
            path=claimed.path,
            code=result.error.code if result.error else "RUN_FAILED",
            reason=result.error.message if result.error else "the Run did not normalize",
            event_id=event_id,
            run_id=run_id,
        )
        logger.warning("ingest.run_failed event=%s run=%s", event_id, run_id)
        return IngestResult(
            outcome=IngestOutcome.RUN_FAILED,
            event_id=event_id,
            run_id=run_id,
            payload_sha256=claimed.sha256,
            source_path=str(landed),
            failure_code=result.error.code if result.error else "RUN_FAILED",
            detail=detail,
        )

    # The Run exists and normalized. Only now is it true that this event has
    # been ingested, so only now does anything say so.
    context.ledger.settle(event_id, state=LedgerState.INGESTED, now=now)
    landed = context.inbox.complete(claimed.path)
    logger.info("ingest.ok event=%s run=%s", event_id, run_id)

    return IngestResult(
        outcome=IngestOutcome.INGESTED,
        event_id=event_id,
        run_id=run_id,
        payload_sha256=claimed.sha256,
        source_path=str(landed),
        detail=f"run {run_id} normalized",
    )


# --------------------------------------------------------------------------
# reconciliation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OrphanReport:
    """What became - or should become - of one interrupted event."""

    event_id: str
    run_id: str | None
    run_status: RunStatus | None
    resolution: str
    recovered_to: str | None = None


def reconcile(context: IngestionContext, *, recover: bool = False) -> list[OrphanReport]:
    """Work out what happened to events stranded in ``processing/``.

    Deterministic, because the ledger named the run id before the Run was
    created. For each stranded event there are exactly two cases:

    * the reserved Run exists and normalized - the work was done, so the ledger
      is completed and the event moves to ``processed/``;
    * the Run is absent or failed - the work did not happen, and the event moves
      to ``failed/`` rather than back into circulation. Retrying means
      resubmitting under a new ``event_id``, which keeps the original mapping
      intact and makes the retry visible.

    Nothing moves unless *recover* is set; the default is a report.
    """
    reports: list[OrphanReport] = []

    for path in context.inbox.orphans():
        event_id = path.stem
        entry = context.ledger.read(event_id)

        if entry is None:
            # Claimed, then interrupted before anything was reserved. No Run can
            # exist, so returning it to the queue is provably safe.
            reports.append(
                OrphanReport(
                    event_id=event_id,
                    run_id=None,
                    run_status=None,
                    resolution="no reservation was made; safe to re-queue",
                    recovered_to=str(context.inbox.release(path)) if recover else None,
                )
            )
            continue

        status = _run_status(context.store, entry.run_id)
        if status is not None and status is not RunStatus.FAILED:
            if recover:
                if entry.state is LedgerState.RESERVED:
                    context.ledger.settle(event_id, state=LedgerState.INGESTED)
                landed: str | None = str(context.inbox.complete(path))
            else:
                landed = None
            reports.append(
                OrphanReport(
                    event_id=event_id,
                    run_id=entry.run_id,
                    run_status=status,
                    resolution=f"run {entry.run_id} exists and reached {status}",
                    recovered_to=landed,
                )
            )
            continue

        if recover:
            if entry.state is LedgerState.RESERVED:
                context.ledger.settle(
                    event_id,
                    state=LedgerState.ABANDONED,
                    note=f"reserved run {entry.run_id} is {status or 'missing'}",
                )
            landed = str(
                context.inbox.reject(
                    path,
                    code="EVENT_ABANDONED",
                    reason=(
                        f"reserved run {entry.run_id} is {status or 'missing'}. Resubmit "
                        "under a new event_id to retry; the original mapping is kept."
                    ),
                    event_id=event_id,
                    run_id=entry.run_id,
                )
            )
        else:
            landed = None
        reports.append(
            OrphanReport(
                event_id=event_id,
                run_id=entry.run_id,
                run_status=status,
                resolution=f"run {entry.run_id} is {status or 'missing'}; needs a new event_id",
                recovered_to=landed,
            )
        )

    return reports


def _run_status(store: RunStore, run_id: str) -> RunStatus | None:
    """Read a Run's status, or ``None`` when the Run is not there at all."""
    try:
        return store.open(run_id).load_manifest().status
    except (FileNotFoundError, ValueError, PipelineError):
        return None


__all__ = [
    "IngestionContext",
    "OrphanReport",
    "ingest_claimed",
    "ingest_file",
    "ingest_next",
    "reconcile",
]
