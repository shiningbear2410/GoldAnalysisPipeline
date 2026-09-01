"""Registering, inspecting and removing the scheduled task.

Every test drives :class:`FakeTaskScheduler`. Nothing here creates, changes or
deletes a real scheduled task, and nothing shells out.

Two properties carry the design:

* **nothing changes without ``--apply``** - registering something that runs
  every minute deserves to be read before it happens; and
* **an existing task is never silently replaced** - it might be an older
  definition or something a person made by hand, and overwriting either on the
  way past is the kind of helpfulness that loses work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import FAKE_API_KEY, FAKE_OPENAI_KEY, TELEGRAM_TOKEN_SENTINEL

from goldpipeline import cli
from goldpipeline.adapters.task_scheduler import (
    FakeTaskScheduler,
    PowerShellTaskScheduler,
    TaskInfo,
    compare,
)
from goldpipeline.cli import EXIT_BLOCKED, EXIT_OK, main
from goldpipeline.domain.errors import (
    TaskDefinitionMismatchError,
    TaskSchedulerError,
    TaskSchedulerUnavailableError,
)
from goldpipeline.services.task_plan import DEFAULT_TASK_NAME, build_plan

SENTINELS = (FAKE_API_KEY, FAKE_OPENAI_KEY, TELEGRAM_TOKEN_SENTINEL)


def invoke(args: list[str]) -> int:
    return main(args)


# --- nothing happens without --apply ---------------------------------------


def test_install_without_apply_changes_nothing(
    task_scheduler: FakeTaskScheduler, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirements 12 and 20.

    The default is a plan. A minute-by-minute task is a decision to make while
    looking at the definition, not a side effect of typing the command.
    """
    code = invoke(["automation-task-install"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "Would register" in out
    assert "Nothing was changed" in out
    assert task_scheduler.installed == []
    assert task_scheduler.tasks == {}


def test_remove_without_apply_changes_nothing(
    task_scheduler: FakeTaskScheduler, capsys: pytest.CaptureFixture[str]
) -> None:
    task_scheduler.tasks[DEFAULT_TASK_NAME] = TaskInfo(installed=True, enabled=True)

    code = invoke(["automation-task-remove"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "Would remove" in out
    assert task_scheduler.removed == []
    assert DEFAULT_TASK_NAME in task_scheduler.tasks


def test_apply_defaults_to_false() -> None:
    from goldpipeline.cli import build_parser

    for command in ("automation-task-install", "automation-task-remove"):
        assert build_parser().parse_args([command]).apply is False


# --- registering -----------------------------------------------------------


def test_apply_registers_the_reviewed_definition(
    task_scheduler: FakeTaskScheduler, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 21."""
    code = invoke(["automation-task-install", "--apply", "--json"])
    payload = json.loads(capsys.readouterr().out)
    plan = build_plan()

    assert code == EXIT_OK
    assert len(task_scheduler.installed) == 1
    name, xml = task_scheduler.installed[0]
    assert name == DEFAULT_TASK_NAME
    assert payload["installed"] is True
    assert payload["executable"] == str(plan.executable)
    assert payload["working_directory"] == str(plan.working_directory)


def test_the_registered_definition_names_the_venv_interpreter(
    task_scheduler: FakeTaskScheduler, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 15.

    Task Scheduler's PATH has little to do with an operator's shell, so "python"
    on it could be an interpreter with none of this project's dependencies.
    """
    invoke(["automation-task-install", "--apply"])
    capsys.readouterr()
    _, xml = task_scheduler.installed[0]

    assert ".venv" in xml
    # Round 9.2.2: the silent build. `python.exe` under an interactive-token
    # task has no console to inherit, so Windows makes one - and it is visible.
    assert "pythonw.exe" in xml
    assert "<Command>python<" not in xml
    assert "<Command>pythonw<" not in xml


def test_the_registered_definition_sets_the_working_directory(
    task_scheduler: FakeTaskScheduler, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 16.

    Runs, the inbox and the automation state all resolve relative to the current
    directory. Without this the worker would build a second, empty set of them.
    """
    invoke(["automation-task-install", "--apply"])
    capsys.readouterr()
    _, xml = task_scheduler.installed[0]

    assert f"<WorkingDirectory>{build_plan().working_directory}" in xml


def test_the_registered_definition_uses_the_interactive_user(
    task_scheduler: FakeTaskScheduler, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 17.

    MetaTrader 5 is a desktop application bound to a logged-in session, and the
    credentials are encrypted against that user's login. SYSTEM has neither.
    """
    invoke(["automation-task-install", "--apply"])
    capsys.readouterr()
    _, xml = task_scheduler.installed[0]

    assert "<LogonType>InteractiveToken</LogonType>" in xml
    assert "SYSTEM" not in xml


def test_the_registered_definition_skips_an_overlapping_run(
    task_scheduler: FakeTaskScheduler, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 18.

    ``StopExisting`` would kill a stage mid-request, possibly mid-``sendMessage``
    - precisely the ambiguity Round 6 exists to avoid.
    """
    invoke(["automation-task-install", "--apply"])
    capsys.readouterr()
    _, xml = task_scheduler.installed[0]

    assert "<MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>" in xml
    assert "StopExisting" not in xml


def test_the_registered_definition_carries_no_credential(
    task_scheduler: FakeTaskScheduler,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirement 19, re-checked with credentials present in the environment."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI_KEY)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TELEGRAM_TOKEN_SENTINEL)

    invoke(["automation-task-install", "--apply"])
    captured = capsys.readouterr()
    _, xml = task_scheduler.installed[0]

    for sentinel in SENTINELS:
        assert sentinel not in xml
        assert sentinel not in captured.out
        assert sentinel not in captured.err
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TELEGRAM_BOT_TOKEN", "password"):
        assert name not in xml


def test_a_task_registered_as_system_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 23 of the spec.

    A SYSTEM task cannot see the MetaTrader window or read this user's
    credentials. It would look healthy and do nothing useful - the worst shape a
    failure can take.
    """

    class SystemScheduler(FakeTaskScheduler):
        def install(self, task_name: str, xml: str) -> None:
            super().install(task_name, xml)
            self.tasks[task_name] = TaskInfo(
                installed=True, enabled=True, task_user="NT AUTHORITY\\SYSTEM"
            )

    scheduler = SystemScheduler()
    monkeypatch.setattr(cli, "_task_scheduler", lambda: scheduler)

    code = invoke(["automation-task-install", "--apply"])
    err = capsys.readouterr().err

    assert code == EXIT_BLOCKED
    assert "SYSTEM" in err
    assert "automation-task-remove" in err


# --- an existing task ------------------------------------------------------


def test_a_matching_existing_task_is_reported_not_replaced(
    task_scheduler: FakeTaskScheduler, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirements 11 of the spec and 22."""
    invoke(["automation-task-install", "--apply"])
    capsys.readouterr()

    code = invoke(["automation-task-install", "--apply"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "Already installed" in out
    assert len(task_scheduler.installed) == 1, "not registered a second time"


def test_a_conflicting_existing_task_fails_closed(
    task_scheduler: FakeTaskScheduler, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 23.

    It might be an older definition from a previous round, or something a person
    created by hand. Either way it is left exactly where it is.
    """
    task_scheduler.tasks[DEFAULT_TASK_NAME] = TaskInfo(
        installed=True,
        enabled=True,
        executable=r"C:\Python\python.exe",
        arguments="-m something_else",
        working_directory=r"C:\elsewhere",
    )

    code = invoke(["automation-task-install", "--apply"])
    err = capsys.readouterr().err

    assert code != EXIT_OK
    assert task_scheduler.installed == []
    assert "already registered" in err


def test_the_mismatch_names_what_differs() -> None:
    existing = TaskInfo(
        installed=True,
        executable=r"C:\Python\python.exe",
        arguments="-m goldpipeline automation-worker-tick",
        working_directory=r"C:\elsewhere",
    )

    with pytest.raises(TaskDefinitionMismatchError) as exc:
        compare(
            existing,
            executable=r"D:\project\.venv\Scripts\python.exe",
            arguments="-m goldpipeline automation-worker-tick",
            working_directory=r"D:\project",
        )

    differences = exc.value.details["differences"]
    assert any("executable" in item for item in differences)
    assert any("working directory" in item for item in differences)
    assert not any("arguments" in item for item in differences)


def test_a_trailing_separator_is_not_a_mismatch() -> None:
    """Windows reports a working directory with or without a trailing slash."""
    compare(
        TaskInfo(
            installed=True,
            executable=r"D:\p\.venv\Scripts\python.exe",
            arguments="-m goldpipeline automation-worker-tick",
            working_directory="D:\\p\\",
        ),
        executable=r"D:\p\.venv\Scripts\python.exe",
        arguments="-m goldpipeline automation-worker-tick",
        working_directory=r"D:\p",
    )


# --- status ----------------------------------------------------------------


def test_status_reports_a_missing_task(
    task_scheduler: FakeTaskScheduler, capsys: pytest.CaptureFixture[str]
) -> None:
    code = invoke(["automation-task-status"])
    out = capsys.readouterr().out

    assert code == EXIT_BLOCKED
    assert "Installed: NO" in out


def test_status_reports_the_registered_definition(
    task_scheduler: FakeTaskScheduler, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 24."""
    invoke(["automation-task-install", "--apply"])
    capsys.readouterr()

    code = invoke(["automation-task-status", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["installed"] is True
    assert payload["logon_type"] == "InteractiveToken"
    assert payload["multiple_instances_policy"] == "IgnoreNew"
    assert payload["arguments"] == "-m goldpipeline automation-worker-tick"


def test_status_shows_the_last_run_and_result(
    task_scheduler: FakeTaskScheduler, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 21 of the spec: what proves the scheduler actually fired."""
    task_scheduler.tasks[DEFAULT_TASK_NAME] = TaskInfo(
        installed=True,
        enabled=True,
        state="Ready",
        task_user="PC\\operator",
        logon_type="InteractiveToken",
        multiple_instances_policy="IgnoreNew",
        last_run_time="2026-08-31T14:05:00",
        last_result=0,
        next_run_time="2026-08-31T14:06:00",
    )

    invoke(["automation-task-status"])
    out = capsys.readouterr().out

    assert "Last run:           2026-08-31T14:05:00" in out
    assert "Last result:        0" in out
    assert "Next run:           2026-08-31T14:06:00" in out


def test_status_never_prints_a_credential(
    task_scheduler: FakeTaskScheduler,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TELEGRAM_TOKEN_SENTINEL)
    invoke(["automation-task-install", "--apply"])
    capsys.readouterr()

    invoke(["automation-task-status"])
    captured = capsys.readouterr()

    for sentinel in SENTINELS:
        assert sentinel not in captured.out
        assert sentinel not in captured.err


# --- removal ---------------------------------------------------------------


def test_apply_removes_the_task(
    task_scheduler: FakeTaskScheduler, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 25."""
    invoke(["automation-task-install", "--apply"])
    capsys.readouterr()

    code = invoke(["automation-task-remove", "--apply"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "Removed" in out
    assert task_scheduler.removed == [DEFAULT_TASK_NAME]
    assert task_scheduler.tasks == {}


def test_removing_an_absent_task_is_calm(
    task_scheduler: FakeTaskScheduler, capsys: pytest.CaptureFixture[str]
) -> None:
    code = invoke(["automation-task-remove", "--apply"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "Nothing to remove" in out
    assert task_scheduler.removed == []


# --- the real adapter, without touching Windows ----------------------------


def test_the_real_adapter_parses_a_scheduler_answer() -> None:
    """Requirement 24, against the shape PowerShell actually returns."""
    answer = json.dumps(
        {
            "installed": True,
            "enabled": True,
            "state": "Ready",
            "task_user": "PC\\operator",
            "logon_type": "InteractiveToken",
            "multiple_instances_policy": "IgnoreNew",
            "executable": r"D:\p\.venv\Scripts\python.exe",
            "arguments": "-m goldpipeline automation-worker-tick",
            "working_directory": r"D:\p",
            "last_run_time": "2026-08-31T14:05:00.0000000",
            "last_result": 0,
            "next_run_time": "2026-08-31T14:06:00.0000000",
        }
    )
    info = PowerShellTaskScheduler(runner=lambda script: answer).query("whatever")

    assert info.installed
    assert info.last_result == 0
    assert info.logon_type == "InteractiveToken"
    assert not info.runs_as_system


def test_the_real_adapter_reports_an_absent_task() -> None:
    info = PowerShellTaskScheduler(runner=lambda s: '{"installed": false}').query("nope")

    assert not info.installed


@pytest.mark.parametrize("user", ["NT AUTHORITY\\SYSTEM", "S-1-5-18", "system"])
def test_a_system_principal_is_recognised(user: str) -> None:
    assert TaskInfo(installed=True, task_user=user).runs_as_system


def test_an_unreachable_scheduler_is_reported_clearly() -> None:
    def missing(script: str) -> str:
        raise FileNotFoundError("powershell.exe")

    with pytest.raises(TaskSchedulerUnavailableError):
        PowerShellTaskScheduler(runner=missing).query("whatever")


def test_a_scheduler_failure_is_summarised_not_dumped() -> None:
    def failing(script: str) -> str:
        raise RuntimeError("Access is denied.\n" + "stack " * 200)

    with pytest.raises(TaskSchedulerError) as exc:
        PowerShellTaskScheduler(runner=failing).remove("whatever")

    assert len(str(exc.value)) < 400


def test_an_unreadable_answer_is_refused() -> None:
    with pytest.raises(TaskSchedulerError):
        PowerShellTaskScheduler(runner=lambda s: "not json at all").query("whatever")


def test_nothing_in_the_tests_shells_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 30: no network, and no Windows mutation either."""
    import subprocess

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the task tests must not shell out")

    monkeypatch.setattr(subprocess, "run", explode)
    assert PowerShellTaskScheduler(runner=lambda s: '{"installed": false}').query("x") is not None


# --- the worker still respects the flags -----------------------------------


def test_a_registered_task_does_nothing_while_automation_is_off(
    tmp_path: Path, production_config: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 26, and the whole of Phase B.

    Registering the task and switching the system on are two separate acts. The
    scheduler may fire immediately; the worker must decline - and since Round
    9.2.1 it declines because a complete configuration told it to, not because
    it failed to find one.
    """
    production_config(GOLDPIPELINE_AUTOMATION_ENABLED="false")
    code = invoke(
        [
            "automation-worker-tick",
            "--inbox-dir",
            str(tmp_path / "inbox"),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--automation-dir",
            str(tmp_path / "automation"),
            "--fake-mt5",
            "--fake-ai",
        ]
    )
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "disabled" in out
    assert not (tmp_path / "runs").exists()


def test_auto_publish_stays_off_through_activation(config_store: Any) -> None:
    """Requirements 27 and 28.

    Registering a task, persisting configuration and enabling the worker are
    three separate decisions, and none of them is permission to publish.
    """
    from goldpipeline.adapters.config_store import LayeredConfig
    from goldpipeline.config import AutomationSettings
    from goldpipeline.schemas.runtime_config import ConfigKey

    config_store.set(ConfigKey.AUTOMATION_ENABLED, "true")
    config_store.set(ConfigKey.AUTOPUBLISH_ALLOWED_TARGET, "@pcfxsn")
    settings = AutomationSettings.from_env(LayeredConfig({}, config_store.load()))

    assert settings.enabled is True
    assert settings.auto_publish_enabled is False, "enabling the worker is not enabling publishing"
