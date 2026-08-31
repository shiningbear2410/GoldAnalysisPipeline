"""The durable analysis inbox.

Four directories and an index::

    inbox/
        incoming/    producer writes here
        processing/  exactly one consumer holds each file here
        processed/   a Run exists for this event
        failed/      refused, with a reason beside it
        index/       event_id -> run_id, write-once

**Producers never write a partial file into ``incoming/``.** :meth:`Inbox.submit`
writes a temporary file in the same directory, fsyncs it, and renames it into
place. A consumer therefore only ever sees whole documents, and does not need to
guess whether a short file is truncated or merely short.

**Claiming is a rename, and that is what makes it exclusive.** Two consumers
racing for the same event both call ``rename(incoming/x, processing/x)``; the
kernel lets one win and the other gets "no such file". There is no lock to leak
and no lease to expire. This is the maildir trick, and it is enough here because
the only thing being protected is which process owns one small file.

**Nothing is ever deleted.** A refused event moves to ``failed/`` with a reason
file beside it. Production input that a machine could not understand is exactly
the input a human most needs to look at.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from goldpipeline.domain.errors import InboxPayloadError, LedgerError
from goldpipeline.schemas.common import utc_now
from goldpipeline.schemas.ingestion import LedgerEntry, LedgerState
from goldpipeline.storage.atomic import encode_json, sha256_bytes

logger = logging.getLogger(__name__)

INCOMING = "incoming"
PROCESSING = "processing"
PROCESSED = "processed"
FAILED = "failed"
INDEX = "index"

DIRECTORIES = (INCOMING, PROCESSING, PROCESSED, FAILED, INDEX)

REASON_SUFFIX = ".reason.json"
"""Written beside a failed event, never inside it - the payload stays verbatim."""


@dataclass(frozen=True)
class ClaimedEvent:
    """One event file, held by this process."""

    path: Path
    payload: dict[str, Any]
    raw: bytes

    @property
    def sha256(self) -> str:
        """Digest of the exact bytes the producer wrote."""
        return sha256_bytes(self.raw)


class Inbox:
    """A directory tree the producer writes to and one consumer drains."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    # -- layout ------------------------------------------------------------

    def ensure_layout(self) -> None:
        """Create the directories if they are not there yet."""
        for name in DIRECTORIES:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def directory(self, name: str) -> Path:
        return self.root / name

    # -- producer side -----------------------------------------------------

    def submit(self, payload: dict[str, Any], *, event_id: str) -> Path:
        """Place *payload* in ``incoming/`` atomically.

        This is the API the producing bot calls - directly, or through
        ``inbox-submit``. It writes a temporary file beside the target, flushes
        and fsyncs it, and only then renames, so a consumer polling the
        directory can never read a half-written analysis.

        Raises:
            InboxPayloadError: An event with this id is already waiting.
        """
        self.ensure_layout()
        target = self.directory(INCOMING) / f"{event_id}.json"
        if target.exists():
            raise InboxPayloadError(
                f"an event with id {event_id!r} is already waiting in the inbox",
                event_id=event_id,
            )

        payload_bytes = encode_json(payload)
        handle, name = tempfile.mkstemp(
            prefix=f".{event_id}.", suffix=".tmp", dir=str(target.parent)
        )
        temporary = Path(name)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(payload_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

        logger.info("inbox.submit event=%s bytes=%d", event_id, len(payload_bytes))
        return target

    # -- consumer side -----------------------------------------------------

    def pending(self) -> list[Path]:
        """Events waiting in ``incoming/``, oldest name first."""
        directory = self.directory(INCOMING)
        if not directory.is_dir():
            return []
        return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix == ".json")

    def claim(self, path: Path) -> Path | None:
        """Move one event into ``processing/``, or report that someone else did.

        Returns:
            The new path, or ``None`` when another consumer got there first.
        """
        self.ensure_layout()
        target = self.directory(PROCESSING) / path.name
        try:
            os.rename(path, target)
        except (FileNotFoundError, FileExistsError):
            # Lost the race, or the event is already being processed. Either
            # way this consumer does not own it and must not touch it.
            return None
        logger.info("inbox.claim event=%s", path.stem)
        return target

    def read(self, path: Path) -> ClaimedEvent:
        """Read and JSON-parse a claimed event.

        Raises:
            InboxPayloadError: The file is unreadable, not UTF-8, not JSON, or
                not a JSON object.
        """
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise InboxPayloadError(f"inbox event could not be read: {path.name}") from exc

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InboxPayloadError(f"inbox event is not valid UTF-8: {path.name}") from exc

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InboxPayloadError(
                f"inbox event is not valid JSON: {path.name} ({exc.msg} at line {exc.lineno})"
            ) from exc

        if not isinstance(payload, dict):
            raise InboxPayloadError(
                f"inbox event must be a JSON object, got {type(payload).__name__}: {path.name}"
            )
        return ClaimedEvent(path=path, payload=payload, raw=raw)

    # -- terminal moves ----------------------------------------------------

    def complete(self, path: Path) -> Path:
        """Move a processed event to ``processed/``."""
        return self._move(path, PROCESSED)

    def release(self, path: Path) -> Path:
        """Return an event to ``incoming/``.

        Only for the case where nothing was written and nothing was reserved -
        a market provider that was briefly unreachable. Anything that touched
        the ledger goes to ``failed/`` instead and waits for a human.
        """
        return self._move(path, INCOMING)

    def reject(self, path: Path, *, code: str, reason: str, **details: Any) -> Path:
        """Move an event to ``failed/`` and write a reason beside it."""
        moved = self._move(path, FAILED)
        note = {
            "code": code,
            "reason": reason,
            "failed_at": utc_now().isoformat().replace("+00:00", "Z"),
            "details": details,
        }
        (moved.parent / f"{moved.stem}{REASON_SUFFIX}").write_bytes(encode_json(note))
        logger.warning("inbox.reject event=%s code=%s", moved.stem, code)
        return moved

    def orphans(self) -> list[Path]:
        """Events left in ``processing/`` by an interrupted consumer."""
        directory = self.directory(PROCESSING)
        if not directory.is_dir():
            return []
        return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix == ".json")

    def _move(self, path: Path, destination: str) -> Path:
        self.ensure_layout()
        target = self.directory(destination) / path.name
        os.replace(path, target)
        return target


class Ledger:
    """The write-once ``event_id -> run_id`` index.

    One small JSON file per event. A database would be a reasonable choice at a
    thousand events an hour; at a handful a day it would be infrastructure to
    operate for no benefit, and a directory of files is greppable at 3am.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path_for(self, event_id: str) -> Path:
        return self.root / f"{event_id}.json"

    def read(self, event_id: str) -> LedgerEntry | None:
        """Return the entry for *event_id*, or ``None`` if it has never been seen.

        Raises:
            LedgerError: The entry exists but cannot be parsed. Deliberately not
                treated as "never seen": that would re-ingest an event whose
                history is merely unreadable.
        """
        path = self.path_for(event_id)
        if not path.is_file():
            return None
        try:
            return LedgerEntry.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, PydanticValidationError) as exc:
            raise LedgerError(
                f"ingestion ledger entry for {event_id!r} is unreadable; "
                "refusing to treat it as a new event",
                event_id=event_id,
            ) from exc

    def reserve(self, entry: LedgerEntry) -> None:
        """Record a run id for an event, before the Run exists.

        Exclusive by construction: the file is created with ``O_CREAT | O_EXCL``,
        so two consumers cannot both reserve the same event.

        Raises:
            LedgerError: The event already has an entry.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        payload = encode_json(entry)
        try:
            handle = os.open(self.path_for(entry.event_id), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise LedgerError(
                f"event {entry.event_id!r} already has a ledger entry",
                event_id=entry.event_id,
            ) from exc

        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        logger.info("ledger.reserve event=%s run=%s", entry.event_id, entry.run_id)

    def settle(
        self,
        event_id: str,
        *,
        state: LedgerState,
        note: str | None = None,
        now: datetime | None = None,
    ) -> LedgerEntry:
        """Close out a reservation, keeping every identity field as written.

        Only ``state``, ``settled_at`` and ``note`` change. The event id, the
        payload digest and the run id are the mapping itself, and rewriting any
        of them would detach a published article from the analysis it came from.

        Raises:
            LedgerError: There is no entry to settle.
        """
        entry = self.read(event_id)
        if entry is None:
            raise LedgerError(
                f"no ledger entry to settle for event {event_id!r}", event_id=event_id
            )

        settled = entry.model_copy(
            update={"state": state, "settled_at": now or utc_now(), "note": note}
        )
        _atomic_write(self.path_for(event_id), encode_json(settled))
        logger.info("ledger.settle event=%s state=%s run=%s", event_id, state, entry.run_id)
        return settled

    def entries(self) -> list[LedgerEntry]:
        """Every entry, oldest reservation first."""
        if not self.root.is_dir():
            return []
        found = [
            self.read(path.stem)
            for path in sorted(self.root.iterdir())
            if path.is_file() and path.suffix == ".json"
        ]
        return sorted((e for e in found if e is not None), key=lambda e: e.reserved_at)


def _atomic_write(path: Path, payload: bytes) -> None:
    """Replace *path* with *payload*, atomically."""
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "DIRECTORIES",
    "FAILED",
    "INCOMING",
    "INDEX",
    "PROCESSED",
    "PROCESSING",
    "REASON_SUFFIX",
    "ClaimedEvent",
    "Inbox",
    "Ledger",
]
