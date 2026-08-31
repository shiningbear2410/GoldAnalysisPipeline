"""Per-Run execution lock.

One lock file per Run directory, and no global lock: two invocations working on
two different Runs have nothing to contend over, and a global lock would make
them queue for no reason.

Acquisition is ``O_CREAT | O_EXCL``, which is atomic on NTFS and POSIX alike -
the file either did not exist and is now ours, or it existed and is not. There
is no window between checking and creating.

**A stale lock is never removed automatically.** A lock left behind by a killed
process looks exactly like a lock held by a process that is still mid-publish,
and this pipeline's most expensive mistake is posting the same article twice. So
a held lock is reported with everything known about its holder, and a human
decides. Deleting the file is a deliberate act, not a fallback.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import socket
from datetime import datetime
from pathlib import Path
from types import TracebackType

from goldpipeline.domain.errors import RunLockedError
from goldpipeline.schemas.common import utc_now

logger = logging.getLogger(__name__)

LOCK_FILENAME = ".pipeline.lock"
"""Name of the lock file inside a Run directory.

Dotted so it sorts away from the artifacts, and deliberately *not* registered in
the manifest: it is not part of the Run's content, it is a runtime detail with a
shorter lifetime than the directory around it.
"""

WORKER_LOCK_FILENAME = ".worker.lock"
"""Name of the automation worker's intake lock.

A different lock guarding a different thing. The per-Run lock stops two
processes driving one article; this one stops two scheduled ticks both deciding
what to work on next. A Run being driven by hand is unaffected by it.
"""


class RunLock:
    """Exclusive claim on one Run, for the duration of one invocation.

    Use it as a context manager::

        with RunLock(run_dir):
            ...

    The lock is released on the way out whether the body succeeded or raised.
    It is not released on a hard kill - that is the case the stale-lock policy
    above exists for.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        filename: str = LOCK_FILENAME,
        now: datetime | None = None,
        pid: int | None = None,
        hostname: str | None = None,
    ) -> None:
        """Prepare a lock inside *run_dir*.

        Args:
            run_dir: The directory to place the lock in.
            filename: Which lock. Defaults to the per-Run lock; the automation
                worker passes :data:`WORKER_LOCK_FILENAME` so its intake lock and
                a Run lock can be held at the same time without one standing in
                for the other.
            now: Injection point for tests.
            pid: Injection point for tests; defaults to this process.
            hostname: Injection point for tests; defaults to this host.
        """
        self.path = run_dir / filename
        self._now = now
        self._pid = pid if pid is not None else os.getpid()
        self._hostname = hostname if hostname is not None else socket.gethostname()
        self._token = secrets.token_hex(8)
        self._held = False

    @property
    def held(self) -> bool:
        """Whether this instance currently holds the lock."""
        return self._held

    def acquire(self) -> None:
        """Take the lock, or explain who has it.

        Raises:
            RunLockedError: Another invocation holds this Run.
        """
        payload = json.dumps(
            {
                "pid": self._pid,
                "hostname": self._hostname,
                "created_at": (self._now or utc_now()).isoformat().replace("+00:00", "Z"),
                "token": self._token,
            },
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

        try:
            handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise self._held_by_someone_else() from None

        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        self._held = True
        logger.debug("lock acquired path=%s pid=%s", self.path, self._pid)

    def release(self) -> None:
        """Give the lock up, if this instance is the one holding it.

        Ownership is re-checked against the token in the file. Without that
        check, a lock this process never took - one a human recreated, say -
        could be deleted on the way out of an unrelated failure.
        """
        if not self._held:
            return
        self._held = False

        if self._read_holder().get("token") != self._token:
            logger.warning(
                "lock at %s is no longer ours; leaving it in place for a human", self.path
            )
            return

        self.path.unlink(missing_ok=True)
        logger.debug("lock released path=%s", self.path)

    def __enter__(self) -> RunLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    # -- internals ---------------------------------------------------------

    def _held_by_someone_else(self) -> RunLockedError:
        """Build the refusal, carrying whatever the holder recorded."""
        holder = self._read_holder()
        return RunLockedError(
            f"run is already being executed by another process (lock: {self.path}). "
            "If that process is gone, inspect the lock file and remove it by hand - "
            "it is never cleared automatically, because a crashed publisher and a "
            "running one look identical from here.",
            lock_path=str(self.path),
            holder_pid=holder.get("pid"),
            holder_hostname=holder.get("hostname"),
            holder_created_at=holder.get("created_at"),
        )

    def _read_holder(self) -> dict[str, object]:
        """Read the lock file, tolerating a partial or corrupt one.

        A lock written by a process that died mid-write is still a lock. Failing
        to parse it must not turn into "therefore it is free".
        """
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}


__all__ = ["LOCK_FILENAME", "WORKER_LOCK_FILENAME", "RunLock"]
