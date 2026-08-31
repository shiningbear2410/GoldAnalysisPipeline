"""Operational state for the automation worker.

Everything here is *runtime* state, and that is the distinction worth holding
onto. A Run's artifacts are immutable evidence of what the pipeline produced;
these files describe what a scheduler is currently doing about them. They are
overwritten, they can be deleted without losing an article, and none of them is
ever consulted to decide whether something is safe to publish.

::

    automation/
        state.json      the dashboard: counts and the last tick
        history/        one immutable record per tick
        retries/        backoff state, keyed by run id
        .worker.lock    held for the duration of one tick

**History is per-tick files, not an append log.** A process killed mid-append
leaves a truncated final line that every later reader has to cope with. A
process killed mid-write of ``history/<tick>.json`` leaves a temporary file that
nothing looks at.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from goldpipeline.schemas.automation import (
    BACKOFF_MINUTES,
    CONFIGURATION_BACKOFF_MINUTES,
    AutomationState,
    AutomationTickResult,
    DeferRecord,
    RetryClass,
    RetryRecord,
)
from goldpipeline.storage.atomic import encode_json

logger = logging.getLogger(__name__)

STATE_FILENAME = "state.json"
HISTORY_DIR = "history"
RETRIES_DIR = "retries"

DEFER_SUFFIX = ".defer.json"
"""Sidecar naming for a deferred event.

Beside the payload, never inside it. The bytes a producer wrote are evidence,
and scheduling metadata is not part of the evidence.
"""

HISTORY_KEEP = 500
"""How many tick records to keep.

A minute-by-minute scheduler writes 1,440 files a day. Keeping roughly the last
eight hours is enough to reconstruct an incident and few enough that the
directory stays listable.
"""


class AutomationStore:
    """Reads and writes the worker's operational state."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    # -- layout ------------------------------------------------------------

    def ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / HISTORY_DIR).mkdir(exist_ok=True)
        (self.root / RETRIES_DIR).mkdir(exist_ok=True)

    # -- the dashboard -----------------------------------------------------

    def read_state(self) -> AutomationState:
        """The last known snapshot, or a blank one.

        A corrupt state file is *not* a reason to fail a tick: it holds counts,
        not decisions. It is replaced and the tick proceeds.
        """
        path = self.root / STATE_FILENAME
        if not path.is_file():
            return AutomationState()
        try:
            return AutomationState.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, PydanticValidationError):
            logger.warning("automation state was unreadable; starting a fresh snapshot")
            return AutomationState()

    def write_state(self, state: AutomationState) -> Path:
        """Overwrite the snapshot atomically."""
        self.ensure_layout()
        path = self.root / STATE_FILENAME
        _atomic_write(path, encode_json(state))
        return path

    # -- history -----------------------------------------------------------

    def record_tick(self, result: AutomationTickResult) -> Path:
        """Write one immutable tick record and prune the oldest."""
        self.ensure_layout()
        stamp = result.started_at.strftime("%Y%m%dT%H%M%SZ")
        path = self.root / HISTORY_DIR / f"{stamp}_{result.tick_id}.json"
        _atomic_write(path, encode_json(result))
        self._prune_history()
        return path

    def history(self, limit: int = 20) -> list[Path]:
        """The most recent tick records, newest first."""
        directory = self.root / HISTORY_DIR
        if not directory.is_dir():
            return []
        return sorted(directory.glob("*.json"), reverse=True)[:limit]

    def _prune_history(self) -> None:
        records = sorted((self.root / HISTORY_DIR).glob("*.json"))
        for stale in records[:-HISTORY_KEEP]:
            stale.unlink(missing_ok=True)

    # -- retry backoff -----------------------------------------------------

    def read_retry(self, run_id: str) -> RetryRecord | None:
        """Backoff state for one Run, if it has any."""
        path = self.root / RETRIES_DIR / f"{run_id}.json"
        if not path.is_file():
            return None
        try:
            return RetryRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, PydanticValidationError):
            logger.warning("retry record for run=%s was unreadable; discarding it", run_id)
            return None

    def record_failure(
        self,
        run_id: str,
        *,
        failure_code: str,
        retry_class: RetryClass,
        now: datetime,
    ) -> RetryRecord:
        """Note a failure and work out when - or whether - to try again.

        The schedule lengthens with each attempt and then stops. An automation
        layer that retried forever would spend real money discovering the same
        thing every minute, and would bury the one failure a human needs to see
        under a thousand identical ones.
        """
        self.ensure_layout()
        previous = self.read_retry(run_id)
        attempts = (previous.attempts + 1) if previous else 1

        if retry_class is RetryClass.CONFIGURATION:
            # Never exhausted. A human fixing the environment should not also
            # have to clear a retry record before the work resumes.
            delay, exhausted = CONFIGURATION_BACKOFF_MINUTES, False
        elif retry_class is RetryClass.TRANSIENT:
            index = min(attempts - 1, len(BACKOFF_MINUTES) - 1)
            delay = BACKOFF_MINUTES[index]
            exhausted = attempts > len(BACKOFF_MINUTES)
        else:
            # PERMANENT and TERMINAL never come back on their own.
            delay, exhausted = 0, True

        record = RetryRecord(
            run_id=run_id,
            attempts=attempts,
            retry_class=retry_class,
            failure_code=failure_code,
            first_failed_at=previous.first_failed_at if previous else now,
            last_failed_at=now,
            next_attempt_at=now + timedelta(minutes=delay),
            exhausted=exhausted,
        )
        _atomic_write(self.root / RETRIES_DIR / f"{run_id}.json", encode_json(record))
        logger.info(
            "automation.retry run=%s class=%s attempts=%d next=%s exhausted=%s",
            run_id,
            retry_class,
            attempts,
            record.next_attempt_at.isoformat(),
            exhausted,
        )
        return record

    def clear_retry(self, run_id: str) -> None:
        """Forget a Run's backoff state, because it moved forward.

        Called on any successful advance. Without it a Run that failed twice and
        then succeeded would carry a five-minute delay into its next stage for
        no reason.
        """
        (self.root / RETRIES_DIR / f"{run_id}.json").unlink(missing_ok=True)

    def retry_allows(self, run_id: str, moment: datetime) -> bool:
        """Whether this Run may be attempted now."""
        record = self.read_retry(run_id)
        return record is None or record.due(moment)


# --------------------------------------------------------------------------
# deferred events
# --------------------------------------------------------------------------


def defer_path(event_path: Path) -> Path:
    """Where the sidecar for *event_path* lives."""
    return event_path.parent / f"{event_path.stem}{DEFER_SUFFIX}"


def read_defer(event_path: Path) -> DeferRecord | None:
    """Read a deferred event's sidecar, if it has a readable one."""
    path = defer_path(event_path)
    if not path.is_file():
        return None
    try:
        return DeferRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, PydanticValidationError):
        logger.warning("defer record for %s was unreadable", event_path.name)
        return None


def write_defer(event_path: Path, record: DeferRecord) -> Path:
    """Write a deferred event's sidecar beside it."""
    path = defer_path(event_path)
    _atomic_write(path, encode_json(record))
    return path


def next_defer(
    event_path: Path, *, reason_code: str, reason: str, now: datetime, retry_minutes: int
) -> DeferRecord:
    """Build the sidecar for one more deferral, counting the attempts so far."""
    previous = read_defer(event_path)
    return DeferRecord(
        event_id=event_path.stem,
        reason_code=reason_code,
        reason=reason,
        deferred_at=now,
        next_attempt_at=now + timedelta(minutes=retry_minutes),
        attempt_count=(previous.attempt_count + 1) if previous else 1,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    """Replace *path* with *payload*, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
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


def read_json(path: Path) -> dict[str, object] | None:
    """Read a small JSON object, tolerating anything unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


__all__ = [
    "DEFER_SUFFIX",
    "HISTORY_DIR",
    "HISTORY_KEEP",
    "RETRIES_DIR",
    "STATE_FILENAME",
    "AutomationStore",
    "defer_path",
    "next_defer",
    "read_defer",
    "read_json",
    "write_defer",
]
