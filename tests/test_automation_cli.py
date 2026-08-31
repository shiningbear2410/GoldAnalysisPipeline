"""CLI integration for the automation worker.

Every invocation here is offline: fake terminal, fake AI, fake transport, and a
socket guard to prove it. Nothing in this file registers a Windows task, calls a
provider, or reaches Telegram.

The distinction the tests keep returning to is between the two entry points.
``automation-worker-tick`` is what Task Scheduler runs and it obeys the kill
switch; ``automation-run-once`` is a person at a keyboard and it does not,
because the person is the authorisation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import SAMPLE_EVENT_ID, make_event_payload, write_json

from goldpipeline.cli import EXIT_BLOCKED, EXIT_ERROR, EXIT_OK, main


def invoke(args: list[str]) -> int:
    return main(args)


@pytest.fixture
def dirs(tmp_path: Path) -> dict[str, Path]:
    return {
        "inbox": tmp_path / "inbox",
        "runs": tmp_path / "runs",
        "automation": tmp_path / "automation",
    }


def automation_args(dirs: dict[str, Path], command: str, *extra: str) -> list[str]:
    return [
        command,
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


def run_ids(dirs: dict[str, Path]) -> list[str]:
    """Runs created so far.

    Tolerates a missing directory, because "nothing ran" and "the directory was
    never created" are the same observation - and the dry run in particular must
    leave no trace at all, not even an empty folder.
    """
    root = dirs["runs"]
    return sorted(p.name for p in root.iterdir()) if root.is_dir() else []


def submit(dirs: dict[str, Path], tmp_path: Path) -> None:
    """Put one fresh event in the inbox through the ordinary command."""
    from datetime import UTC, datetime

    payload = make_event_payload(created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"))
    path = write_json(tmp_path / "event.json", payload)
    invoke(["inbox-submit", "--file", str(path), "--inbox-dir", str(dirs["inbox"])])


# --- the smoke path --------------------------------------------------------


def test_an_idle_tick_exits_zero(dirs: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    """Golden case A, through the command."""
    code = invoke(automation_args(dirs, "automation-run-once"))
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "nothing to do" in out
    assert "Auto publish: OFF" in out


def test_a_fresh_event_runs_the_whole_pipeline(
    dirs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Golden case B, through the command."""
    submit(dirs, tmp_path)
    capsys.readouterr()

    code = invoke(automation_args(dirs, "automation-run-once", "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["status"] == "OK"
    assert payload["mode"] == "READY_FOR_PUBLISH"
    assert payload["auto_publish_enabled"] is False
    assert [item["outcome"] for item in payload["processed_events"]] == ["INGESTED"]
    assert [item["outcome"] for item in payload["resumed_runs"]] == ["COMPLETED"]


def test_the_tick_publishes_nothing_by_default(
    dirs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 4 of the spec, at the command boundary."""
    submit(dirs, tmp_path)
    capsys.readouterr()
    invoke(automation_args(dirs, "automation-run-once", "--json"))
    payload = json.loads(capsys.readouterr().out)

    run_dir = dirs["runs"] / payload["resumed_runs"][0]["identifier"]
    assert not (run_dir / "publish_intent.json").exists()
    assert not (run_dir / "publish_result.json").exists()


def test_a_whole_tick_opens_no_socket(
    dirs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 52 of Round 8, still true with a scheduler on top."""
    import socket

    submit(dirs, tmp_path)

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("an offline tick must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", explode)
    assert invoke(automation_args(dirs, "automation-run-once")) == EXIT_OK


# --- the kill switch -------------------------------------------------------


def test_the_scheduled_worker_does_nothing_when_automation_is_off(
    dirs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 50 of the spec.

    Off is the default, so registering the task is not the same act as
    switching the system on.
    """
    submit(dirs, tmp_path)
    capsys.readouterr()

    code = invoke(automation_args(dirs, "automation-worker-tick"))
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "disabled" in out
    assert "GOLDPIPELINE_AUTOMATION_ENABLED" in out
    assert run_ids(dirs) == [], "no Run was created"


def test_the_scheduled_worker_runs_when_automation_is_on(
    dirs: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submit(dirs, tmp_path)
    capsys.readouterr()
    monkeypatch.setenv("GOLDPIPELINE_AUTOMATION_ENABLED", "true")

    code = invoke(automation_args(dirs, "automation-worker-tick", "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["status"] == "OK"
    assert len(payload["processed_events"]) == 1


def test_run_once_ignores_the_kill_switch(
    dirs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 51.

    The switch exists to stop the *scheduler*, not to stop an operator
    investigating. A person typing this command is the authorisation.
    """
    submit(dirs, tmp_path)
    capsys.readouterr()

    code = invoke(automation_args(dirs, "automation-run-once", "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert len(payload["processed_events"]) == 1


# --- the dry run -----------------------------------------------------------


def test_a_dry_run_reports_work_without_doing_any(
    dirs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 33 of the spec.

    It reads directories and manifests. It claims nothing, creates nothing,
    calls nothing, and - importantly - writes no automation state either.
    """
    submit(dirs, tmp_path)
    capsys.readouterr()

    code = invoke(automation_args(dirs, "automation-run-once", "--dry-run", "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["dry_run"] is True
    assert payload["pending_events"] == [SAMPLE_EVENT_ID]
    assert payload["auto_publish_enabled"] is False
    assert run_ids(dirs) == []
    assert not (dirs["automation"] / "state.json").exists()
    assert (dirs["inbox"] / "incoming" / f"{SAMPLE_EVENT_ID}.json").is_file()


# --- status ----------------------------------------------------------------


def test_status_counts_the_queue_and_the_runs(
    dirs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirements 73, 74 and 75."""
    submit(dirs, tmp_path)
    invoke(automation_args(dirs, "automation-run-once"))
    capsys.readouterr()

    code = invoke(automation_args(dirs, "automation-status", "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["pending_events"] == 0
    assert payload["run_status_counts"]["READY_TO_PUBLISH"] == 1
    assert payload["auto_publish_enabled"] is False
    assert payload["last_tick_status"] == "OK"
    assert payload["last_error_safe"] is None


def test_status_reads_only(
    dirs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 32 of the spec: read-only, and no network."""
    import socket

    submit(dirs, tmp_path)
    invoke(automation_args(dirs, "automation-run-once"))
    before = run_ids(dirs)

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("status must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", explode)
    assert invoke(automation_args(dirs, "automation-status")) == EXIT_OK
    assert run_ids(dirs) == before


def test_status_prints_the_headline_counts(
    dirs: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    invoke(automation_args(dirs, "automation-status"))
    out = capsys.readouterr().out

    for line in ("Automation:", "Auto publish:", "Pending inbox:", "PUBLISH_UNCERTAIN:"):
        assert line in out


# --- preflight -------------------------------------------------------------


def test_preflight_never_prints_a_credential(
    dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 53.

    An operator will paste this into a chat window to ask what is wrong, so it
    has to be safe to paste.
    """
    from conftest import FAKE_API_KEY, FAKE_OPENAI_KEY, TELEGRAM_TOKEN_SENTINEL

    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI_KEY)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TELEGRAM_TOKEN_SENTINEL)
    monkeypatch.setenv("TELEGRAM_TARGET_CHAT_ID", "@gold_signals_test")

    invoke(automation_args(dirs, "automation-preflight"))
    captured = capsys.readouterr()

    for secret in (FAKE_API_KEY, FAKE_OPENAI_KEY, TELEGRAM_TOKEN_SENTINEL):
        assert secret not in captured.out
        assert secret not in captured.err
    assert "configured" in captured.out


def test_preflight_reports_what_is_missing(
    dirs: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 45 of the spec."""
    code = invoke(automation_args(dirs, "automation-preflight", "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_BLOCKED
    assert payload["mt5"] == "available", "the offline stand-in answers"
    assert payload["anthropic"] == "missing"
    assert payload["task_readiness"] == "NOT_READY"
    assert any("credentials are missing" in blocker for blocker in payload["blockers"])


def test_publishing_credentials_are_not_required_when_publishing_is_off(
    dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirements 46 and 50.

    An operator generating articles but not publishing them needs no Telegram
    credentials, and must not be told otherwise.
    """
    from conftest import FAKE_API_KEY, FAKE_OPENAI_KEY

    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI_KEY)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    code = invoke(automation_args(dirs, "automation-preflight", "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["telegram"] == "missing"
    assert payload["task_readiness"] == "READY"
    assert payload["blockers"] == []


def test_a_target_mismatch_shows_up_in_preflight(
    dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Golden case G, caught before anything runs."""
    from conftest import FAKE_API_KEY, FAKE_OPENAI_KEY, TELEGRAM_TOKEN_SENTINEL

    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI_KEY)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TELEGRAM_TOKEN_SENTINEL)
    monkeypatch.setenv("TELEGRAM_TARGET_CHAT_ID", "@configured_channel")
    monkeypatch.setenv("GOLDPIPELINE_AUTOPUBLISH_ENABLED", "true")
    monkeypatch.setenv("GOLDPIPELINE_AUTOPUBLISH_ALLOWED_TARGET", "@different_channel")

    code = invoke(automation_args(dirs, "automation-preflight", "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_BLOCKED
    assert payload["allowed_target"] == "@different_channel"
    assert payload["configured_target"] == "@configured_channel"
    assert any("differ" in blocker for blocker in payload["blockers"])


# --- the task plan ---------------------------------------------------------


def test_the_task_plan_prints_and_registers_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirements 34 and 64."""
    code = invoke(["automation-task-plan"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "NOT registered" in out
    assert ".venv" in out
    assert "IgnoreNew" in out
    assert "never SYSTEM" in out


def test_the_task_plan_json_names_the_policy(capsys: pytest.CaptureFixture[str]) -> None:
    invoke(["automation-task-plan", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["multiple_instances_policy"] == "IgnoreNew"
    assert payload["principal"] == "InteractiveToken"
    assert payload["interval_minutes"] == 1
    assert payload["arguments"] == "-m goldpipeline automation-worker-tick"
    assert payload["registered"] is False


def test_there_is_no_install_command() -> None:
    """Requirement 42 of the spec, read strictly.

    An install command is only permitted behind an explicit ``--apply``. Rather
    than build one and guard it, this round ships none: registering a
    minute-by-minute task is a decision to make while reading the XML.
    """
    from goldpipeline.cli import build_parser

    parser = build_parser()
    for forbidden in ("automation-task-install", "automation-task-remove"):
        with pytest.raises(SystemExit):
            parser.parse_args([forbidden])


def test_the_plan_command_makes_no_windows_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 67: no scheduled task is created or deleted, ever."""
    import subprocess

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the plan command must not shell out")

    monkeypatch.setattr(subprocess, "run", explode)
    monkeypatch.setattr(subprocess, "Popen", explode)
    assert invoke(["automation-task-plan", "--xml"]) == EXIT_OK


# --- state stays safe ------------------------------------------------------


def test_the_runtime_state_holds_no_secret(
    dirs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 56."""
    from conftest import FAKE_API_KEY, TELEGRAM_TOKEN_SENTINEL

    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TELEGRAM_TOKEN_SENTINEL)
    submit(dirs, tmp_path)
    invoke(automation_args(dirs, "automation-run-once"))

    written = "".join(
        path.read_text(encoding="utf-8") for path in dirs["automation"].rglob("*.json")
    )
    assert FAKE_API_KEY not in written
    assert TELEGRAM_TOKEN_SENTINEL not in written


def test_utf8_paths_and_content_survive(
    dirs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 72."""
    from datetime import UTC, datetime

    payload = make_event_payload(
        created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        raw_text="Vàng giằng co quanh hỗ trợ — chưa dứt khoát.",
    )
    path = write_json(tmp_path / "sự-kiện.json", payload)
    invoke(["inbox-submit", "--file", str(path), "--inbox-dir", str(dirs["inbox"])])
    capsys.readouterr()

    code = invoke(automation_args(dirs, "automation-run-once", "--json"))
    result = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    run_dir = dirs["runs"] / result["resumed_runs"][0]["identifier"]
    stored = json.loads((run_dir / "telegram_input.json").read_text(encoding="utf-8"))
    assert stored["raw_text"] == "Vàng giằng co quanh hỗ trợ — chưa dứt khoát."


# --- exit codes ------------------------------------------------------------


def test_a_deferred_event_still_exits_zero(
    dirs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 55 of the spec.

    A closed market is normal. Exiting non-zero for it would paint the Task
    Scheduler history red every weekend and teach an operator to ignore it - so
    an expired event, which is also normal, exits zero too.
    """
    from datetime import UTC, datetime, timedelta

    old = datetime.now(UTC) - timedelta(hours=8)
    payload = make_event_payload(created_at=old.isoformat().replace("+00:00", "Z"))
    path = write_json(tmp_path / "old.json", payload)
    invoke(["inbox-submit", "--file", str(path), "--inbox-dir", str(dirs["inbox"])])
    capsys.readouterr()

    code = invoke(automation_args(dirs, "automation-run-once", "--json"))
    result = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert [item["outcome"] for item in result["expired_events"]] == ["EXPIRED"]


def test_the_earlier_commands_still_work(
    dirs: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Round 1-8 regression, at the command surface."""
    fixtures = Path(__file__).resolve().parent.parent / "fixtures"
    code = invoke(
        [
            "pipeline-run",
            "--telegram",
            str(fixtures / "telegram_sample.json"),
            "--ohlc",
            str(fixtures / "ohlc_sample.json"),
            "--runs-dir",
            str(dirs["runs"]),
            "--now",
            "2026-08-28T03:00:00Z",
            "--fake-ai",
        ]
    )
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "Final status: READY_TO_PUBLISH" in out


def test_a_bad_flag_combination_is_refused(
    dirs: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """There is no `--publish` on an automation command at all."""
    from goldpipeline.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(automation_args(dirs, "automation-run-once", "--publish"))
    capsys.readouterr()

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            automation_args(dirs, "automation-run-once", "--confirm-real-publish")
        )
    capsys.readouterr()


def test_a_missing_ohlc_path_is_refused(
    dirs: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    code = invoke(
        [
            "automation-run-once",
            "--inbox-dir",
            str(dirs["inbox"]),
            "--runs-dir",
            str(dirs["runs"]),
            "--automation-dir",
            str(dirs["automation"]),
            "--market-source",
            "file",
        ]
    )

    assert code == EXIT_ERROR
    assert "--ohlc" in capsys.readouterr().err
