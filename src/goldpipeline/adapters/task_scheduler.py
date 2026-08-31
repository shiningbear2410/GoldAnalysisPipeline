"""Registering, inspecting and removing the Windows scheduled task.

Round 9 generated the definition; this registers it. The abstraction exists so
every test stays offline - :class:`FakeTaskScheduler` is what the suite drives,
and no test in this repository creates, changes or deletes a real scheduled task.

**Queries go through PowerShell, not ``schtasks /Query``.** The latter prints a
table whose field labels are localised: on a Vietnamese Windows, parsing "Last
Run Time" finds nothing, and a status command that silently reports "unknown" is
worse than one that fails. ``Get-ScheduledTask`` returns objects with stable
property names in any locale.

**An existing task is never silently replaced.** If one is already registered
under this name, its definition is compared against the intended one. A match is
reported as already installed; a mismatch fails closed, because a task someone
else created - or an older definition from a previous round - is not something to
overwrite on the way past.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from goldpipeline.domain.errors import (
    TaskDefinitionMismatchError,
    TaskSchedulerError,
    TaskSchedulerUnavailableError,
)

logger = logging.getLogger(__name__)

POWERSHELL_TIMEOUT_SECONDS = 60.0
"""Bounded, so a wedged scheduler service cannot hang a status command."""


@dataclass(frozen=True)
class TaskInfo:
    """What the operating system says about a registered task.

    Every field is safe to print. The task definition carries a command line and
    a schedule and no credential, which is the whole reason credentials live in
    the credential store.
    """

    installed: bool
    enabled: bool = False
    state: str = "Unknown"
    task_user: str = ""
    logon_type: str = ""
    multiple_instances_policy: str = ""
    executable: str = ""
    arguments: str = ""
    working_directory: str = ""
    last_run_time: str | None = None
    last_result: int | None = None
    next_run_time: str | None = None

    @property
    def runs_as_system(self) -> bool:
        """Whether this task runs as SYSTEM rather than a logged-in user.

        A SYSTEM task cannot see the interactive desktop's MetaTrader window,
        and cannot read credentials stored against the user's login. It would
        look healthy and do nothing useful, which is the worst failure shape.
        """
        upper = self.task_user.upper()
        return "SYSTEM" in upper or upper.endswith("S-1-5-18")


class TaskSchedulerAdapter(Protocol):
    """Anything that can register, inspect and remove a scheduled task."""

    def query(self, task_name: str) -> TaskInfo: ...
    def install(self, task_name: str, xml: str) -> None: ...
    def remove(self, task_name: str) -> None: ...


class PowerShellTaskScheduler:
    """The real Windows Task Scheduler, driven through PowerShell cmdlets."""

    def __init__(self, runner: Any = None) -> None:
        """Build an adapter.

        Args:
            runner: Callable taking a PowerShell script and returning its
                stdout. Injected by tests; the default shells out.
        """
        self._runner = runner if runner is not None else _run_powershell

    def query(self, task_name: str) -> TaskInfo:
        """Read a task's definition and last-run information."""
        script = _QUERY_SCRIPT.replace("{{TASK_NAME}}", _ps_literal(task_name))
        payload = self._json(script, "query the scheduled task")
        if not payload.get("installed"):
            return TaskInfo(installed=False)
        return TaskInfo(
            installed=True,
            enabled=bool(payload.get("enabled")),
            state=str(payload.get("state") or "Unknown"),
            task_user=str(payload.get("task_user") or ""),
            logon_type=str(payload.get("logon_type") or ""),
            multiple_instances_policy=str(payload.get("multiple_instances_policy") or ""),
            executable=str(payload.get("executable") or ""),
            arguments=str(payload.get("arguments") or ""),
            working_directory=str(payload.get("working_directory") or ""),
            last_run_time=_optional(payload.get("last_run_time")),
            last_result=_optional_int(payload.get("last_result")),
            next_run_time=_optional(payload.get("next_run_time")),
        )

    def install(self, task_name: str, xml: str) -> None:
        """Register the task from its XML definition.

        The XML goes through a temporary file rather than the command line: it
        is multi-line, and a definition inlined into a shell command is one
        quoting bug away from registering something different from what was
        reviewed.
        """
        directory = Path(tempfile.mkdtemp(prefix="goldpipeline-task-"))
        definition = directory / "task.xml"
        try:
            definition.write_text(xml, encoding="utf-16")
            script = _INSTALL_SCRIPT.replace("{{TASK_NAME}}", _ps_literal(task_name)).replace(
                "{{XML_PATH}}", _ps_literal(str(definition))
            )
            self._run(script, "register the scheduled task")
        finally:
            definition.unlink(missing_ok=True)
            directory.rmdir()
        logger.info("task.install name=%s", task_name)

    def remove(self, task_name: str) -> None:
        """Unregister the task."""
        script = _REMOVE_SCRIPT.replace("{{TASK_NAME}}", _ps_literal(task_name))
        self._run(script, "remove the scheduled task")
        logger.info("task.remove name=%s", task_name)

    # -- internals ---------------------------------------------------------

    def _run(self, script: str, what: str) -> str:
        try:
            return str(self._runner(script))
        except FileNotFoundError as exc:
            raise TaskSchedulerUnavailableError(
                "PowerShell is not available, so the Windows Task Scheduler "
                "cannot be reached from here.",
            ) from exc
        except Exception as exc:  # noqa: BLE001 - shell errors are undocumented
            raise TaskSchedulerError(f"could not {what}", detail=_safe(exc)) from None

    def _json(self, script: str, what: str) -> dict[str, Any]:
        raw = self._run(script, what).strip()
        if not raw:
            return {"installed": False}
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            raise TaskSchedulerError(
                f"could not {what}: the scheduler returned something unreadable"
            ) from exc
        return parsed if isinstance(parsed, dict) else {"installed": False}


class FakeTaskScheduler:
    """Offline stand-in. Every test in this repository drives this one."""

    def __init__(self, tasks: dict[str, TaskInfo] | None = None) -> None:
        self.tasks = dict(tasks or {})
        self.installed: list[tuple[str, str]] = []
        """Every registration, so a test can prove one did or did not happen."""
        self.removed: list[str] = []

    def query(self, task_name: str) -> TaskInfo:
        return self.tasks.get(task_name, TaskInfo(installed=False))

    def install(self, task_name: str, xml: str) -> None:
        self.installed.append((task_name, xml))
        self.tasks[task_name] = _info_from_xml(task_name, xml)

    def remove(self, task_name: str) -> None:
        self.removed.append(task_name)
        self.tasks.pop(task_name, None)


def compare(existing: TaskInfo, *, executable: str, arguments: str, working_directory: str) -> None:
    """Refuse to touch a task whose definition is not the one we intend.

    A task already registered under this name might be an older definition from
    a previous round, or something a person created by hand. Overwriting either
    on the way past would be the kind of "helpful" that loses work.

    Raises:
        TaskDefinitionMismatchError: The registered task differs.
    """
    differences = [
        f"{field}: registered {found!r}, expected {wanted!r}"
        for field, found, wanted in (
            ("executable", existing.executable, executable),
            ("arguments", existing.arguments, arguments),
            ("working directory", existing.working_directory, working_directory),
        )
        if found.strip().rstrip("\\") != wanted.strip().rstrip("\\")
    ]
    if differences:
        raise TaskDefinitionMismatchError(
            "a task with this name is already registered but does not match the "
            "intended definition. It was left untouched; inspect it and remove "
            "it deliberately if it is stale.",
            differences=differences,
        )


def _info_from_xml(task_name: str, xml: str) -> TaskInfo:
    """Derive a plausible registered task from the XML the fake was handed."""
    return TaskInfo(
        installed=True,
        enabled=True,
        state="Ready",
        task_user="TEST\\operator",
        logon_type="InteractiveToken",
        multiple_instances_policy="IgnoreNew",
        executable=_between(xml, "<Command>", "</Command>"),
        arguments=_between(xml, "<Arguments>", "</Arguments>"),
        working_directory=_between(xml, "<WorkingDirectory>", "</WorkingDirectory>"),
        last_run_time=None,
        last_result=None,
        next_run_time=None,
    )


def _between(text: str, start: str, end: str) -> str:
    try:
        return text.split(start, 1)[1].split(end, 1)[0]
    except IndexError:
        return ""


def _run_powershell(script: str) -> str:
    """Execute a PowerShell script and return its stdout."""
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=POWERSHELL_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"exit {completed.returncode}")
    return completed.stdout


def _ps_literal(value: str) -> str:
    """Quote a value as a PowerShell single-quoted string."""
    return "'" + value.replace("'", "''") + "'"


def _safe(exc: BaseException) -> str:
    """A short, safe description of a shell failure."""
    text = str(exc).strip().splitlines()
    return text[0][:200] if text else type(exc).__name__


def _optional(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_QUERY_SCRIPT = """
$ErrorActionPreference = 'Stop'
$name = {{TASK_NAME}}
$t = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
if (-not $t) { '{"installed": false}'; exit 0 }
$i = Get-ScheduledTaskInfo -TaskName $name -ErrorAction SilentlyContinue
$a = $t.Actions | Select-Object -First 1
[ordered]@{
  installed = $true
  enabled = ($t.State -ne 'Disabled')
  state = [string]$t.State
  task_user = [string]$t.Principal.UserId
  logon_type = [string]$t.Principal.LogonType
  multiple_instances_policy = [string]$t.Settings.MultipleInstances
  executable = [string]$a.Execute
  arguments = [string]$a.Arguments
  working_directory = [string]$a.WorkingDirectory
  last_run_time = if ($i -and $i.LastRunTime) { $i.LastRunTime.ToString('o') } else { '' }
  last_result = if ($i) { [int]$i.LastTaskResult } else { $null }
  next_run_time = if ($i -and $i.NextRunTime) { $i.NextRunTime.ToString('o') } else { '' }
} | ConvertTo-Json -Compress
"""

_INSTALL_SCRIPT = """
$ErrorActionPreference = 'Stop'
$xml = Get-Content -Path {{XML_PATH}} -Raw -Encoding Unicode
Register-ScheduledTask -TaskName {{TASK_NAME}} -Xml $xml | Out-Null
'registered'
"""

_REMOVE_SCRIPT = """
$ErrorActionPreference = 'Stop'
Unregister-ScheduledTask -TaskName {{TASK_NAME}} -Confirm:$false
'removed'
"""


__all__ = [
    "FakeTaskScheduler",
    "PowerShellTaskScheduler",
    "TaskInfo",
    "TaskSchedulerAdapter",
    "compare",
]
