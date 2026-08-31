"""Per-Run execution lock.

The lock exists for one scenario: two orchestrator invocations reaching the
publisher for the same Run at the same time. Everything here is about making
that impossible without inventing a second failure mode - a lock that clears
itself would turn a crashed publisher into a duplicated article.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from goldpipeline.domain.errors import RunLockedError
from goldpipeline.services.run_lock import LOCK_FILENAME, RunLock


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    target = tmp_path / "20260828_120000_abcdef"
    target.mkdir()
    return target


def test_acquiring_creates_the_lock_file(run_dir: Path) -> None:
    """Requirement 35."""
    lock = RunLock(run_dir)
    lock.acquire()

    assert (run_dir / LOCK_FILENAME).is_file()
    assert lock.held
    lock.release()


def test_the_lock_records_who_holds_it(run_dir: Path) -> None:
    """Whoever finds a stale lock has to decide about a process, so name it."""
    with RunLock(run_dir, pid=4242, hostname="build-agent-7"):
        holder = json.loads((run_dir / LOCK_FILENAME).read_text(encoding="utf-8"))

    assert holder["pid"] == 4242
    assert holder["hostname"] == "build-agent-7"
    assert holder["created_at"].endswith("Z")


def test_a_second_holder_is_refused(run_dir: Path) -> None:
    """Requirement 36: the whole point."""
    first = RunLock(run_dir)
    first.acquire()

    with pytest.raises(RunLockedError):
        RunLock(run_dir).acquire()

    first.release()


def test_the_refusal_carries_the_holder_details(run_dir: Path) -> None:
    with RunLock(run_dir, pid=99, hostname="other-host"), pytest.raises(RunLockedError) as exc:
        RunLock(run_dir).acquire()

    assert exc.value.details["holder_pid"] == 99
    assert exc.value.details["holder_hostname"] == "other-host"
    assert LOCK_FILENAME in exc.value.details["lock_path"]


def test_two_different_runs_lock_independently(tmp_path: Path) -> None:
    """Requirements 37 and 41: per-Run, and no global lock anywhere.

    Two Runs have nothing to contend over. If this ever fails, someone has
    introduced a shared lock and serialized the whole pipeline.
    """
    first_dir = tmp_path / "20260828_120000_aaaaaa"
    second_dir = tmp_path / "20260828_120001_bbbbbb"
    first_dir.mkdir()
    second_dir.mkdir()

    with RunLock(first_dir), RunLock(second_dir):
        assert (first_dir / LOCK_FILENAME).is_file()
        assert (second_dir / LOCK_FILENAME).is_file()


def test_the_lock_lives_inside_the_run_it_guards(tmp_path: Path) -> None:
    """No lock directory, no registry, nothing outside the Run."""
    run_dir = tmp_path / "20260828_120000_abcdef"
    run_dir.mkdir()

    with RunLock(run_dir):
        siblings = [p.name for p in tmp_path.iterdir()]

    assert siblings == [run_dir.name]


def test_a_successful_body_releases_the_lock(run_dir: Path) -> None:
    """Requirement 38."""
    with RunLock(run_dir):
        pass

    assert not (run_dir / LOCK_FILENAME).exists()


def test_a_failing_body_releases_the_lock(run_dir: Path) -> None:
    """Requirement 39: a crashed stage must not wedge the Run permanently."""
    with pytest.raises(RuntimeError), RunLock(run_dir):
        raise RuntimeError("stage blew up")

    assert not (run_dir / LOCK_FILENAME).exists()
    RunLock(run_dir).acquire()  # and the Run is usable again


def test_a_stale_lock_is_never_removed_automatically(run_dir: Path) -> None:
    """Requirement 40.

    The lock names a process that no longer exists, which is indistinguishable
    from one that is mid-``sendMessage``. The pipeline refuses either way.
    """
    (run_dir / LOCK_FILENAME).write_text(
        json.dumps({"pid": 999999, "hostname": "gone", "created_at": "2020-01-01T00:00:00Z"}),
        encoding="utf-8",
    )

    with pytest.raises(RunLockedError):
        RunLock(run_dir).acquire()

    assert (run_dir / LOCK_FILENAME).is_file()


def test_an_unparseable_lock_is_still_a_lock(run_dir: Path) -> None:
    """A process killed mid-write leaves a truncated file. That is not 'free'."""
    (run_dir / LOCK_FILENAME).write_text('{"pid": 12', encoding="utf-8")

    with pytest.raises(RunLockedError) as exc:
        RunLock(run_dir).acquire()

    assert exc.value.details["holder_pid"] is None
    assert (run_dir / LOCK_FILENAME).is_file()


def test_releasing_a_lock_we_no_longer_own_leaves_it_alone(run_dir: Path) -> None:
    """Someone else's lock is not ours to delete on the way out of our failure."""
    lock = RunLock(run_dir)
    lock.acquire()

    # A human cleared ours and a different invocation took the Run.
    (run_dir / LOCK_FILENAME).write_text(
        json.dumps({"pid": 7, "hostname": "elsewhere", "token": "not-ours"}), encoding="utf-8"
    )
    lock.release()

    assert (run_dir / LOCK_FILENAME).is_file()


def test_releasing_without_acquiring_is_a_no_op(run_dir: Path) -> None:
    RunLock(run_dir).release()

    assert not (run_dir / LOCK_FILENAME).exists()


def test_release_is_idempotent(run_dir: Path) -> None:
    lock = RunLock(run_dir)
    lock.acquire()
    lock.release()
    lock.release()

    assert not lock.held
