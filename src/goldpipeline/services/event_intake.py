"""Admitting remote events into the durable inbox, exactly once.

This service does one thing: it turns bytes from a remote producer into
:meth:`Inbox.submit` calls. It creates no Run, fetches no market data, calls no
model and sends no message - the inbox stays the only handoff into the pipeline,
so everything downstream is the code that was already proven.

**Why local admission can be exactly-once over an at-least-once transport.**
The ledger already answers "have I seen this analysis?" using ``event_id`` plus
the SHA-256 of the exact bytes that were written, and
:mod:`goldpipeline.services.admission` asks it. A producer may hand back the
same event on every poll for a week and it will be admitted once. Nothing here
invents a second notion of payload identity; there is exactly one, it lives in
the ledger, and the internal producer consults the same one.

**Remote producers may only ask for ANALYSIS.** The restriction lives here
rather than in the schema because only this layer knows the event arrived
over a network; a local producer added later may legitimately request all
three modes through the same inbox.

**Per-event admission, not per-batch.** One malformed event does not discard the
valid ones beside it. That mirrors the inbox: each file is accepted or refused on
its own, and a refusal is written down rather than allowed to spoil its
neighbours. The one exception is a malformed *envelope*, which the transport
refuses whole - that is evidence about the producer, not about any event.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from goldpipeline.adapters.event_transport import EventTransport
from goldpipeline.adapters.inbox_source import parse_event
from goldpipeline.domain.errors import InboxPayloadError
from goldpipeline.services.admission import AdmissionState
from goldpipeline.services.admission import resolve as resolve_admission
from goldpipeline.services.article_routing import REMOTE_ALLOWED_TYPES
from goldpipeline.services.inbox import Inbox, Ledger

logger = logging.getLogger(__name__)


@dataclass
class IntakeConflict:
    """One ``event_id`` offered with content that does not match what we hold."""

    event_id: str
    known_sha256: str
    offered_sha256: str
    where: str


@dataclass(frozen=True)
class RejectedEvent:
    """One event refused by policy rather than by schema."""

    event_id: str
    reason: str


@dataclass
class IntakeReport:
    """What one intake pass did. Counts and ids only - never analysis text."""

    received: int = 0
    submitted: list[str] = field(default_factory=list)
    duplicate: list[str] = field(default_factory=list)
    invalid: int = 0
    conflicts: list[IntakeConflict] = field(default_factory=list)
    rejected: list[RejectedEvent] = field(default_factory=list)

    @property
    def admitted(self) -> int:
        return len(self.submitted)


def intake(
    *,
    transport: EventTransport,
    inbox: Inbox,
    ledger: Ledger,
    limit: int,
) -> IntakeReport:
    """Fetch pending events and admit the ones not already known.

    Transport failures propagate: deciding whether an unreachable producer is
    worth reporting belongs to the caller, which knows what else the tick did.
    Per-event problems do not propagate - they are counted here.
    """
    payloads = transport.fetch_pending(limit=limit)
    report = IntakeReport(received=len(payloads))

    for payload in payloads:
        _admit_one(payload, inbox=inbox, ledger=ledger, report=report)

    logger.info(
        "intake.pass received=%d submitted=%d duplicate=%d invalid=%d conflict=%d rejected=%d",
        report.received,
        len(report.submitted),
        len(report.duplicate),
        report.invalid,
        len(report.conflicts),
        len(report.rejected),
    )
    return report


def _admit_one(
    payload: dict[str, object],
    *,
    inbox: Inbox,
    ledger: Ledger,
    report: IntakeReport,
) -> None:
    """Decide the fate of one offered event."""
    try:
        event = parse_event(payload)
    except InboxPayloadError:
        # Refused at the door, exactly as a bad local file would be, and never
        # written to incoming/. The payload is not logged: an invalid event
        # still contains the analyst's text.
        report.invalid += 1
        logger.warning("intake.invalid_event refused by schema validation")
        return

    if event.article_type not in REMOTE_ALLOWED_TYPES:
        # Enforced here because this is the only layer that knows the event came
        # over a network. The payload cannot be asked - a producer describing its
        # own origin is describing what it would like to be believed. Refused
        # before submit(), so no inbox file and no Run ever exist for it.
        report.rejected.append(
            RejectedEvent(
                event_id=event.event_id,
                reason=f"remote producers may not request {event.article_type}",
            )
        )
        logger.warning(
            "intake.article_type_refused event=%s type=%s",
            event.event_id,
            event.article_type,
        )
        return

    event_id = event.event_id
    admission = resolve_admission(payload, event_id=event_id, inbox=inbox, ledger=ledger)

    if admission.state is AdmissionState.UNREADABLE_HISTORY:
        # An unreadable entry is not "never seen". Treating it as new would
        # re-ingest an event whose history is merely damaged.
        report.invalid += 1
        logger.warning("intake.ledger_unreadable event=%s", event_id)
        return

    if admission.state is AdmissionState.DUPLICATE:
        report.duplicate.append(event_id)
        if admission.known_sha256 is None:
            # Present but unreadable on disk. Not a conflict - we cannot claim
            # the content differs - and certainly not grounds to submit a second
            # copy, which is the one outcome that cannot be undone.
            logger.warning(
                "intake.existing_unreadable event=%s source=%s", event_id, admission.where
            )
        return

    if admission.state is AdmissionState.CONFLICT:
        report.conflicts.append(
            IntakeConflict(
                event_id=event_id,
                known_sha256=admission.known_sha256 or "",
                offered_sha256=admission.offered_sha256,
                where=admission.where or "",
            )
        )
        logger.warning("intake.conflict event=%s source=%s", event_id, admission.where)
        return

    try:
        inbox.submit(payload, event_id=event_id)
    except InboxPayloadError:
        # Lost a race with another writer between the check and the submit.
        # The other copy is already waiting, so this is a duplicate, not a fault.
        report.duplicate.append(event_id)
        return

    report.submitted.append(event_id)


__all__ = ["IntakeConflict", "IntakeReport", "RejectedEvent", "intake"]
