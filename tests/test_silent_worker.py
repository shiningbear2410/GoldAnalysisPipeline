"""Round 9.2.2: the scheduled worker runs without a console window.

``python.exe`` is a console-subsystem binary. Started by Task Scheduler under an
interactive token it has no console to inherit, so Windows allocates one, and
that console is a real visible window - once a minute, all day. Measured, not
assumed: a probe registered as a scheduled task reported
``GetConsoleWindow() != 0`` with ``IsWindowVisible() == 1`` under ``python.exe``,
and ``GetConsoleWindow() == 0`` under ``pythonw.exe``.

The fix is the same interpreter built for the GUI subsystem. Nothing about the
pipeline changes - same module, same arguments, same working directory, same
strict configuration contract, same exit codes.

What *does* change is dangerous enough to be the larger half of this file:
under ``pythonw.exe`` there are no standard streams at all. ``sys.stdout`` and
``sys.stderr`` are ``None``, so everything the worker printed now goes nowhere.
A failing tick that only wrote to ``stderr`` would become a silent non-zero exit
code - which is a smaller version of exactly the defect Round 9.2.1 existed to
remove. So the tests here assert twice over: that correctness never depends on a
stream, and that a refusal leaves durable evidence on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import make_event_payload, write_json

from goldpipeline.cli import EXIT_ERROR, EXIT_OK, main
from goldpipeline.schemas.runtime_config import ConfigMode
from goldpipeline.services.task_plan import (
    VENV_PYTHON,
    VENV_PYTHONW,
    WORKER_COMMAND,
    build_plan,
)


def invoke(args: list[str]) -> int:
    return main(args)


@pytest.fixture
def dirs(tmp_path: Path) -> dict[str, Path]:
    return {
        "inbox": tmp_path / "inbox",
        "runs": tmp_path / "runs",
        "automation": tmp_path / "automation",
    }


def worker_args(dirs: dict[str, Path], *extra: str) -> list[str]:
    return [
        "automation-worker-tick",
        "--inbox-dir",
        str(dirs["inbox"]),
        "--runs-dir",
        str(dirs["runs"]),
        "--automation-dir",
        str(dirs["automation"]),
        "--fake-mt5",
        "--fake-ai",
        *extra,
    ]


def submit(dirs: dict[str, Path], tmp_path: Path) -> None:
    from datetime import UTC, datetime

    payload = make_event_payload(created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"))
    path = write_json(tmp_path / "event.json", payload)
    invoke(["inbox-submit", "--file", str(path), "--inbox-dir", str(dirs["inbox"])])


@pytest.fixture
def venv(tmp_path: Path) -> Path:
    """A checkout whose virtualenv has both interpreters."""
    root = tmp_path / "project"
    (root / VENV_PYTHONW.parent).mkdir(parents=True)
    (root / VENV_PYTHONW).write_text("", encoding="utf-8")
    (root / VENV_PYTHON).write_text("", encoding="utf-8")
    return root


def history(dirs: dict[str, Path]) -> list[dict[str, Any]]:
    records = sorted((dirs["automation"] / "history").glob("*.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in records]


# --- the task definition --------------------------------------------------


def test_the_plan_uses_the_silent_interpreter(venv: Path) -> None:
    """Requirements 8.1 and 8.2."""
    plan = build_plan(project_root=venv)

    assert plan.executable == venv / VENV_PYTHONW
    assert plan.executable.name == "pythonw.exe"
    assert plan.executable_exists


def test_the_console_interpreter_is_only_a_fallback(tmp_path: Path) -> None:
    """A checkout without ``pythonw.exe`` still gets a working task.

    A scheduler that flickers beats no scheduler at all, so the console build is
    a fallback rather than a hard requirement - but it is second, never first.
    """
    root = tmp_path / "console-only"
    (root / VENV_PYTHON.parent).mkdir(parents=True)
    (root / VENV_PYTHON).write_text("", encoding="utf-8")

    plan = build_plan(project_root=root)

    assert plan.executable == root / VENV_PYTHON


def test_everything_else_about_the_definition_is_unchanged(venv: Path) -> None:
    """Requirements 8.3 to 8.6.

    The point of this round is that *only* the interpreter moved. Each of these
    was a deliberate decision in an earlier round and none of them is affected
    by which subsystem the binary was built for.
    """
    plan = build_plan(project_root=venv)
    xml = plan.to_xml()

    assert plan.arguments == " ".join(WORKER_COMMAND) == "-m goldpipeline automation-worker-tick"
    assert plan.working_directory == venv.resolve()
    assert f"<WorkingDirectory>{venv.resolve()}</WorkingDirectory>" in xml
    assert "<MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>" in xml
    assert "<LogonType>InteractiveToken</LogonType>" in xml
    assert "<Interval>PT1M</Interval>" in xml


def test_the_hidden_flag_was_not_used_as_the_fix(venv: Path) -> None:
    """``Hidden`` hides a task from a listing; it does not remove a console.

    Recorded as a test because it is the plausible-looking wrong answer, and a
    later reader deserves to know it was considered rather than missed.
    """
    assert "<Hidden>false</Hidden>" in build_plan(project_root=venv).to_xml()


def test_the_definition_carries_no_secret(venv: Path) -> None:
    """Requirement 8.7."""
    from conftest import FAKE_API_KEY, FAKE_OPENAI_KEY, TELEGRAM_TOKEN_SENTINEL

    xml = build_plan(project_root=venv).to_xml()

    for sentinel in (FAKE_API_KEY, FAKE_OPENAI_KEY, TELEGRAM_TOKEN_SENTINEL):
        assert sentinel not in xml
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TELEGRAM_BOT_TOKEN", "password"):
        assert name not in xml


# --- correctness does not depend on a console -----------------------------


class _NoStreams:
    """Replaces ``sys.stdout``/``sys.stderr`` with ``None``, as pythonw does.

    Not a mock of a stream - the *absence* of one. A stand-in that swallowed
    writes would still let a test pass while the real thing raised
    ``AttributeError`` on ``None.write``.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setattr(sys, "stdout", None)
        monkeypatch.setattr(sys, "stderr", None)


def test_a_healthy_tick_works_with_no_streams(
    dirs: dict[str, Path],
    tmp_path: Path,
    production_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirements 8.8, 8.9, 8.13 and 8.14."""
    submit(dirs, tmp_path)
    production_config(GOLDPIPELINE_AUTOMATION_ENABLED="true")
    _NoStreams(monkeypatch)

    code = invoke(worker_args(dirs))

    assert code == EXIT_OK
    records = history(dirs)
    assert len(records) == 1
    assert records[0]["status"] == "OK"
    assert records[0]["config_mode"] == ConfigMode.STRICT_PERSISTENT.value
    assert records[0]["config_sha256"], "the tick still names the configuration it read"
    assert len(records[0]["processed_events"]) == 1


def test_a_disabled_tick_exits_zero_with_no_streams(
    dirs: dict[str, Path],
    tmp_path: Path,
    production_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirement 8.12."""
    submit(dirs, tmp_path)
    production_config(GOLDPIPELINE_AUTOMATION_ENABLED="false")
    _NoStreams(monkeypatch)

    assert invoke(worker_args(dirs)) == EXIT_OK


def test_a_strict_config_failure_exits_non_zero_with_no_streams(
    dirs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 8.11.

    The exit code is the only thing Task Scheduler records, so it has to carry
    the whole signal when nothing can be printed.
    """
    submit(dirs, tmp_path)
    _NoStreams(monkeypatch)

    assert invoke(worker_args(dirs)) == EXIT_ERROR


# --- durable failure evidence ---------------------------------------------


def test_a_strict_config_failure_is_recorded_on_disk(
    dirs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 8.10, and the reason this round is not a one-line change.

    Before the silent worker, a refusal explained itself on stderr. With no
    console that explanation reaches nobody, leaving "non-zero, every minute"
    and no way to tell which failure it was. So the refusal is written where
    every other tick is written.
    """
    submit(dirs, tmp_path)
    _NoStreams(monkeypatch)

    assert invoke(worker_args(dirs)) == EXIT_ERROR

    records = history(dirs)
    assert len(records) == 1, "the refusal left exactly one record"
    assert records[0]["status"] == "FAILED"
    assert records[0]["errors"] == ["PERSISTENT_CONFIG_NOT_FOUND"]
    assert records[0]["automation_enabled"] is False
    assert records[0]["auto_publish_enabled"] is False

    state = json.loads((dirs["automation"] / "state.json").read_text(encoding="utf-8"))
    assert state["last_error_safe"] == "PERSISTENT_CONFIG_NOT_FOUND"
    assert state["last_tick_status"] == "FAILED"


def test_the_failure_record_names_no_values(
    dirs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A code, never a message. The record is written unattended."""
    from conftest import FAKE_API_KEY, FAKE_OPENAI_KEY, TELEGRAM_TOKEN_SENTINEL

    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI_KEY)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TELEGRAM_TOKEN_SENTINEL)
    submit(dirs, tmp_path)

    assert invoke(worker_args(dirs)) == EXIT_ERROR

    raw = json.dumps(history(dirs))
    for sentinel in (FAKE_API_KEY, FAKE_OPENAI_KEY, TELEGRAM_TOKEN_SENTINEL):
        assert sentinel not in raw


def test_an_incomplete_configuration_is_recorded_by_code(
    dirs: dict[str, Path], tmp_path: Path, production_config: Any
) -> None:
    """Each refusal is distinguishable in the history, not merely 'failed'."""
    production_config(GOLDPIPELINE_AUTOPUBLISH_ENABLED=None)

    assert invoke(worker_args(dirs)) == EXIT_ERROR
    assert history(dirs)[0]["errors"] == ["PERSISTENT_CONFIG_INCOMPLETE"]


def test_recording_a_failure_never_masks_the_failure(
    dirs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the evidence cannot be written, the exit code still stands.

    The recorder runs on a path where something is already wrong. A second
    failure inside it must not turn a clean non-zero exit into a traceback that,
    on the silent worker, nobody would ever see.
    """
    from goldpipeline import cli

    def explode(*args: Any, **kwargs: Any) -> None:
        raise OSError(28, "no space left on device")

    monkeypatch.setattr(cli.AutomationStore, "record_tick", explode)
    submit(dirs, tmp_path)

    assert invoke(worker_args(dirs)) == EXIT_ERROR


# --- nothing else moved ---------------------------------------------------


def test_the_silent_worker_still_publishes_nothing(
    dirs: dict[str, Path],
    tmp_path: Path,
    production_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirements 8.15 and 8.16."""
    submit(dirs, tmp_path)
    production_config(GOLDPIPELINE_AUTOMATION_ENABLED="true")
    _NoStreams(monkeypatch)

    assert invoke(worker_args(dirs)) == EXIT_OK

    record = history(dirs)[0]
    assert record["auto_publish_enabled"] is False
    assert record["mode"] == "READY_FOR_PUBLISH"

    runs = sorted(p.name for p in dirs["runs"].iterdir()) if dirs["runs"].is_dir() else []
    for run_id in runs:
        manifest = json.loads((dirs["runs"] / run_id / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] != "PUBLISHED", "nothing is published with auto-publish off"


def test_the_silent_worker_opens_no_socket(
    dirs: dict[str, Path],
    tmp_path: Path,
    production_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirement 8.17."""
    import socket

    submit(dirs, tmp_path)
    production_config(GOLDPIPELINE_AUTOMATION_ENABLED="true")

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("an offline tick must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", explode)
    _NoStreams(monkeypatch)

    assert invoke(worker_args(dirs)) == EXIT_OK
