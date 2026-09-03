"""Deciding whether an event is already known, before submitting it.

One question - *have we seen this exact analysis under this id?* - and one place
that answers it. Both producers ask it: the remote intake service, which may be
offered the same event on every poll for a week, and the internal producer,
whose caller may retry after a lost acknowledgement. Two implementations of this
would be two definitions of "the same event", and the first day they disagreed
would be the day one analysis became two articles.

**Identity is the ledger's, not a new one.** The ledger records ``event_id``
plus the SHA-256 of the exact bytes the inbox wrote. This module hashes with the
same ``encode_json`` and ``sha256_bytes``, so "same payload" means the same
thing here, in the inbox and in the ledger. Nothing invents a second notion of
payload identity.

**Absence of a ledger entry is not absence of the event.** An event submitted a
minute ago is waiting in ``incoming/`` with no ledger entry at all - the entry
appears when a consumer reserves a run id. So all four inbox directories are
searched too, and for the same reason ``processed/`` is searched as well as
``incoming/``: the question is "is a copy already here", not "is one still
pending".

Read-only. Nothing here moves, writes or rewrites a file.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from goldpipeline.domain.errors import LedgerError
from goldpipeline.services.inbox import FAILED, INCOMING, PROCESSED, PROCESSING, Inbox, Ledger
from goldpipeline.storage.atomic import encode_json, sha256_bytes

LEDGER = "ledger"
"""Where a decision came from, when it came from the ledger rather than a directory."""

SEARCHED = (INCOMING, PROCESSING, PROCESSED, FAILED)
"""Inbox directories a copy of an event may sit in.

All four, because "already here" is the question, not "already finished". An
event waiting in ``incoming/`` and one already moved to ``processed/`` are both
reasons not to submit a second copy.
"""


class AdmissionState(StrEnum):
    """What is already known about an offered event."""

    NEW = "NEW"
    """No copy and no ledger entry. Safe to submit."""

    DUPLICATE = "DUPLICATE"
    """A copy with the same bytes is already here, or was already ingested."""

    CONFLICT = "CONFLICT"
    """This id already holds *different* content.

    Never resolved by writing a second copy or by minting a new id. The recorded
    mapping is the audit trail for an article that may already be published, and
    the resubmission is the thing in question, not the record.
    """

    UNREADABLE_HISTORY = "UNREADABLE_HISTORY"
    """A ledger entry exists but cannot be parsed.

    Deliberately its own answer rather than folded into ``NEW`` or ``CONFLICT``.
    Treating damaged history as "never seen" is how an event gets ingested
    twice; calling it a conflict would assert the content differs, which is
    exactly what could not be determined.
    """


@dataclass(frozen=True)
class Admission:
    """What one lookup concluded about one ``event_id``."""

    state: AdmissionState
    event_id: str
    offered_sha256: str
    known_sha256: str | None = None
    where: str | None = None
    run_id: str | None = None

    @property
    def may_submit(self) -> bool:
        return self.state is AdmissionState.NEW


def payload_bytes(payload: dict[str, Any]) -> bytes:
    """The exact bytes the inbox would write for *payload*."""
    return encode_json(payload)


def resolve(
    payload: dict[str, Any],
    *,
    event_id: str,
    inbox: Inbox,
    ledger: Ledger,
) -> Admission:
    """Decide whether *payload* may be submitted under *event_id*.

    The ledger is consulted first because it is the durable record: it survives
    an event moving between directories, and it is what the ingestion service
    itself checks. The directories are the fallback for the window before any
    consumer has reserved a run id.

    Raises nothing. An unreadable ledger entry is reported as
    :attr:`AdmissionState.UNREADABLE_HISTORY` rather than raised, so a caller
    that has to answer a person can say what it found.
    """
    offered = sha256_bytes(payload_bytes(payload))

    try:
        entry = ledger.read(event_id)
    except LedgerError:
        return Admission(
            state=AdmissionState.UNREADABLE_HISTORY,
            event_id=event_id,
            offered_sha256=offered,
            where=LEDGER,
        )

    if entry is not None:
        state = (
            AdmissionState.DUPLICATE if entry.payload_sha256 == offered else AdmissionState.CONFLICT
        )
        return Admission(
            state=state,
            event_id=event_id,
            offered_sha256=offered,
            known_sha256=entry.payload_sha256,
            where=LEDGER,
            run_id=entry.run_id,
        )

    existing = existing_copy(inbox, event_id)
    if existing is None:
        return Admission(state=AdmissionState.NEW, event_id=event_id, offered_sha256=offered)

    where, raw = existing
    if raw is None:
        # Present but unreadable. Not a conflict - we cannot claim the content
        # differs - and certainly not grounds to submit a second copy, which is
        # the one outcome that cannot be undone.
        return Admission(
            state=AdmissionState.DUPLICATE,
            event_id=event_id,
            offered_sha256=offered,
            where=where,
        )

    known = sha256_bytes(raw)
    return Admission(
        state=AdmissionState.DUPLICATE if known == offered else AdmissionState.CONFLICT,
        event_id=event_id,
        offered_sha256=offered,
        known_sha256=known,
        where=where,
    )


def existing_copy(inbox: Inbox, event_id: str) -> tuple[str, bytes | None] | None:
    """Find an already-present copy of this event, and return its bytes.

    ``None`` for the bytes means "present, but could not be read" - a different
    fact from "absent", and the caller must not mistake it for a conflict.

    Read-only: this looks, and never moves or rewrites anything.
    """
    for name in SEARCHED:
        candidate = inbox.directory(name) / f"{event_id}.json"
        if candidate.is_file():
            try:
                return name, candidate.read_bytes()
            except OSError:
                return name, None
    return None


__all__ = [
    "LEDGER",
    "SEARCHED",
    "Admission",
    "AdmissionState",
    "existing_copy",
    "payload_bytes",
    "resolve",
]
