"""The durable inbox and its ledger.

Two properties carry the design, and most of this file is about them:

* **a consumer never sees a partial file**, because submission renames into
  place rather than writing in place; and
* **two consumers never hold the same event**, because claiming is a rename and
  the kernel only lets one of them win.

The rest is about not losing production input: nothing is deleted, a refusal
keeps the payload and writes the reason beside it, and a reused id is refused
rather than resolved.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from conftest import SAMPLE_EVENT_ID, make_event_payload, submit_event

from goldpipeline.domain.errors import InboxPayloadError, LedgerError
from goldpipeline.schemas.ingestion import LedgerEntry, LedgerState
from goldpipeline.services.inbox import (
    DIRECTORIES,
    FAILED,
    INCOMING,
    PROCESSED,
    PROCESSING,
    REASON_SUFFIX,
    Inbox,
    Ledger,
)

# --- layout and submission ------------------------------------------------


def test_the_layout_is_created_on_demand(tmp_path: Path) -> None:
    box = Inbox(tmp_path / "inbox")
    box.ensure_layout()

    assert sorted(p.name for p in (tmp_path / "inbox").iterdir()) == sorted(DIRECTORIES)


def test_a_submitted_event_lands_in_incoming(inbox: Inbox) -> None:
    """Requirement 1."""
    path = submit_event(inbox, make_event_payload())

    assert path.parent.name == INCOMING
    assert path.name == f"{SAMPLE_EVENT_ID}.json"
    assert json.loads(path.read_text(encoding="utf-8"))["event_id"] == SAMPLE_EVENT_ID


def test_submission_leaves_no_temporary_behind(inbox: Inbox) -> None:
    """Requirement 4 of the durable-inbox section.

    A consumer listing ``incoming/`` must see finished documents and nothing
    else - no ``.tmp``, no zero-byte placeholder.
    """
    submit_event(inbox, make_event_payload())

    names = [p.name for p in inbox.directory(INCOMING).iterdir()]
    assert names == [f"{SAMPLE_EVENT_ID}.json"]


def test_a_partial_write_never_becomes_visible(inbox: Inbox, monkeypatch: Any) -> None:
    """The reason submission renames instead of writing in place.

    The write is made to fail after the temporary file exists. What must not
    happen is a half-written event appearing in ``incoming/``.
    """

    def explode(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", explode)
    with pytest.raises(OSError, match="disk full"):
        submit_event(inbox, make_event_payload())

    assert list(inbox.directory(INCOMING).iterdir()) == []


def test_resubmitting_a_waiting_event_is_refused(inbox: Inbox) -> None:
    submit_event(inbox, make_event_payload())

    with pytest.raises(InboxPayloadError):
        submit_event(inbox, make_event_payload())


def test_vietnamese_survives_the_round_trip(inbox: Inbox) -> None:
    """Requirement 14."""
    text = "Vàng đang giằng co quanh vùng hỗ trợ — chưa có tín hiệu dứt khoát."
    submitted = submit_event(inbox, make_event_payload(raw_text=text))
    assert "\\u" not in submitted.read_text(encoding="utf-8"), "stored as escapes, not characters"

    claimed = inbox.read(inbox.claim(submitted))  # type: ignore[arg-type]
    assert claimed.payload["raw_text"] == text


# --- claiming -------------------------------------------------------------


def test_claiming_moves_the_event_to_processing(inbox: Inbox) -> None:
    """Requirement 6."""
    path = submit_event(inbox, make_event_payload())
    claimed = inbox.claim(path)

    assert claimed is not None
    assert claimed.parent.name == PROCESSING
    assert not path.exists()


def test_a_second_consumer_cannot_claim_the_same_event(inbox: Inbox) -> None:
    """Requirement 7.

    The rename is the exclusion. Whoever loses finds the source gone, which is
    the same answer they would get if the event had never existed - and either
    way they must not touch it.
    """
    path = submit_event(inbox, make_event_payload())

    first = inbox.claim(path)
    second = inbox.claim(path)

    assert first is not None
    assert second is None
    assert len(list(inbox.directory(PROCESSING).iterdir())) == 1


def test_pending_lists_only_finished_events(inbox: Inbox) -> None:
    submit_event(inbox, make_event_payload(event_id="event-aaaaaaaa"))
    submit_event(inbox, make_event_payload(event_id="event-bbbbbbbb"))
    (inbox.directory(INCOMING) / ".half-written.tmp").write_text("{", encoding="utf-8")

    assert [p.stem for p in inbox.pending()] == ["event-aaaaaaaa", "event-bbbbbbbb"]


# --- reading --------------------------------------------------------------


def test_malformed_json_is_refused(inbox: Inbox) -> None:
    """Requirement 2."""
    path = inbox.directory(PROCESSING) / "broken.json"
    path.write_text('{"event_id": "x"', encoding="utf-8")

    with pytest.raises(InboxPayloadError, match="not valid JSON"):
        inbox.read(path)


def test_a_json_array_is_refused(inbox: Inbox) -> None:
    path = inbox.directory(PROCESSING) / "array.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(InboxPayloadError, match="must be a JSON object"):
        inbox.read(path)


def test_non_utf8_is_refused(inbox: Inbox) -> None:
    path = inbox.directory(PROCESSING) / "latin.json"
    path.write_bytes(b'{"raw_text": "\xff\xfe"}')

    with pytest.raises(InboxPayloadError, match="not valid UTF-8"):
        inbox.read(path)


def test_the_digest_is_of_the_producers_own_bytes(inbox: Inbox) -> None:
    """Requirement 15.

    Hashed as written, not as re-serialized. A digest of our own rendering
    would change the day the JSON indentation changed, and every stored mapping
    would quietly stop matching.
    """
    import hashlib

    path = submit_event(inbox, make_event_payload())
    raw = path.read_bytes()
    claimed = inbox.read(inbox.claim(path))  # type: ignore[arg-type]

    assert claimed.sha256 == hashlib.sha256(raw).hexdigest()


# --- terminal moves -------------------------------------------------------


def test_completing_moves_to_processed(inbox: Inbox) -> None:
    path = inbox.claim(submit_event(inbox, make_event_payload()))
    assert path is not None

    landed = inbox.complete(path)

    assert landed.parent.name == PROCESSED
    assert list(inbox.directory(PROCESSING).iterdir()) == []


def test_releasing_returns_the_event_to_the_queue(inbox: Inbox) -> None:
    path = inbox.claim(submit_event(inbox, make_event_payload()))
    assert path is not None

    landed = inbox.release(path)

    assert landed.parent.name == INCOMING
    assert [p.stem for p in inbox.pending()] == [SAMPLE_EVENT_ID]


def test_a_rejected_event_keeps_its_payload_and_gains_a_reason(inbox: Inbox) -> None:
    """Requirement 9.

    Production input a machine could not understand is exactly the input a human
    most needs to read, so the payload survives untouched and the explanation
    goes beside it.
    """
    original = submit_event(inbox, make_event_payload())
    payload_before = original.read_bytes()
    path = inbox.claim(original)
    assert path is not None

    landed = inbox.reject(path, code="TEST_REFUSAL", reason="because", detail=42)

    assert landed.parent.name == FAILED
    assert landed.read_bytes() == payload_before
    note = json.loads((landed.parent / f"{landed.stem}{REASON_SUFFIX}").read_text(encoding="utf-8"))
    assert note["code"] == "TEST_REFUSAL"
    assert note["reason"] == "because"
    assert note["details"] == {"detail": 42}


def test_nothing_is_ever_deleted(inbox: Inbox) -> None:
    """Every terminal move is a move. There is no delete on this class."""
    assert not hasattr(inbox, "delete")
    assert not hasattr(inbox, "purge")


def test_orphans_are_what_is_left_in_processing(inbox: Inbox) -> None:
    """Requirement 10."""
    inbox.claim(submit_event(inbox, make_event_payload(event_id="event-stranded")))
    inbox.complete(inbox.claim(submit_event(inbox, make_event_payload(event_id="event-finished"))))  # type: ignore[arg-type]

    assert [p.stem for p in inbox.orphans()] == ["event-stranded"]


# --- the ledger -----------------------------------------------------------


def entry(**overrides: Any) -> LedgerEntry:
    payload: dict[str, Any] = {
        "event_id": SAMPLE_EVENT_ID,
        "source": "gold_analysis_bot",
        "payload_sha256": "a" * 64,
        "run_id": "20260828_030000_abc123",
    }
    payload.update(overrides)
    return LedgerEntry(**payload)


def test_an_unseen_event_has_no_entry(tmp_path: Path) -> None:
    assert Ledger(tmp_path / "index").read("never-seen-before") is None


def test_reserving_records_the_run_id_before_the_run_exists(tmp_path: Path) -> None:
    """Requirement 16, and the reason recovery is deterministic."""
    ledger = Ledger(tmp_path / "index")
    ledger.reserve(entry())

    stored = ledger.read(SAMPLE_EVENT_ID)
    assert stored is not None
    assert stored.run_id == "20260828_030000_abc123"
    assert stored.state is LedgerState.RESERVED
    assert stored.settled_at is None


def test_a_second_reservation_is_refused(tmp_path: Path) -> None:
    """Requirement 7, at the ledger rather than the directory.

    Exclusive by ``O_CREAT | O_EXCL``, so two consumers that somehow both got
    past the claim still cannot both allocate a Run.
    """
    ledger = Ledger(tmp_path / "index")
    ledger.reserve(entry())

    with pytest.raises(LedgerError, match="already has a ledger entry"):
        ledger.reserve(entry(run_id="20260828_030001_def456"))

    stored = ledger.read(SAMPLE_EVENT_ID)
    assert stored is not None
    assert stored.run_id == "20260828_030000_abc123"


def test_settling_keeps_every_identity_field(tmp_path: Path) -> None:
    """Requirement 41: the mapping is never overwritten, only closed out."""
    ledger = Ledger(tmp_path / "index")
    ledger.reserve(entry())

    settled = ledger.settle(SAMPLE_EVENT_ID, state=LedgerState.INGESTED)

    assert settled.event_id == SAMPLE_EVENT_ID
    assert settled.payload_sha256 == "a" * 64
    assert settled.run_id == "20260828_030000_abc123"
    assert settled.state is LedgerState.INGESTED
    assert settled.settled_at is not None


def test_abandoning_records_why(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "index")
    ledger.reserve(entry())

    settled = ledger.settle(SAMPLE_EVENT_ID, state=LedgerState.ABANDONED, note="run never appeared")

    assert settled.state is LedgerState.ABANDONED
    assert settled.note == "run never appeared"
    assert settled.run_id == "20260828_030000_abc123"


def test_settling_an_unknown_event_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(LedgerError, match="no ledger entry to settle"):
        Ledger(tmp_path / "index").settle("never-seen-before", state=LedgerState.INGESTED)


def test_an_unreadable_entry_is_not_treated_as_a_new_event(tmp_path: Path) -> None:
    """The most dangerous way to get this wrong.

    A corrupt entry means the history is unknown, not absent. Reporting "never
    seen" would re-ingest an event that may already have produced an article.
    """
    index = tmp_path / "index"
    index.mkdir()
    (index / f"{SAMPLE_EVENT_ID}.json").write_text("{ not json", encoding="utf-8")

    with pytest.raises(LedgerError, match="unreadable"):
        Ledger(index).read(SAMPLE_EVENT_ID)


def test_entries_are_listed_oldest_first(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    ledger = Ledger(tmp_path / "index")
    ledger.reserve(entry(event_id="event-second", reserved_at=datetime(2026, 8, 28, 4, tzinfo=UTC)))
    ledger.reserve(entry(event_id="event-first", reserved_at=datetime(2026, 8, 28, 3, tzinfo=UTC)))

    assert [e.event_id for e in ledger.entries()] == ["event-first", "event-second"]
