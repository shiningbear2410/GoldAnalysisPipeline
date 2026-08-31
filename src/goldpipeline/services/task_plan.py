"""Windows Task Scheduler definition for the automation worker.

Generates the XML; registering it is a separate, deliberate act. The four
choices in here are the ones that matter, and each is a decision rather than a
default:

**The exact interpreter, never ``python``.** Task Scheduler runs with a PATH
that has little to do with the one in an operator's shell, and "python" on that
PATH may be a different interpreter with none of this project's dependencies.
The plan names ``.venv\\Scripts\\python.exe`` by absolute path.

**An explicit working directory.** Runs, the inbox and the automation state are
all resolved relative to the current directory. Without this the worker would
create a second, empty set of them wherever Task Scheduler happened to start.

**IgnoreNew, never StopExisting.** A tick can outlast a minute: a writer call
takes as long as it takes. ``StopExisting`` would kill a stage mid-request -
possibly mid-``sendMessage``, which is exactly the ambiguity Round 6 exists to
avoid. A skipped tick costs one minute; a killed publish costs certainty.

**The interactive user, never SYSTEM.** MetaTrader 5 is a desktop application
bound to a logged-in session. A SYSTEM task cannot see it, so it would defer
every event forever while looking healthy.

**No secret ever appears here.** The definition carries a command line and a
schedule. Credentials come from the environment the task inherits, which is why
activating this needs a deliberate credential strategy rather than a field in
this file.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

DEFAULT_TASK_NAME = "GoldAnalysisPipeline Automation"
DEFAULT_INTERVAL_MINUTES = 1
WORKER_COMMAND = ("-m", "goldpipeline", "automation-worker-tick")
"""What the scheduler runs.

The *worker* entry point, not ``automation-run-once``: this one honours the kill
switch, so a registered task can be switched off without unregistering it.
"""

VENV_PYTHON = Path(".venv") / "Scripts" / "python.exe"


@dataclass(frozen=True)
class TaskPlan:
    """A deterministic Task Scheduler definition, and what it would do."""

    task_name: str
    executable: Path
    arguments: str
    working_directory: Path
    interval_minutes: int
    executable_exists: bool

    @property
    def command_line(self) -> str:
        """How the task would be invoked, for a human to read and check."""
        return f'"{self.executable}" {self.arguments}'

    def to_xml(self) -> str:
        """Render the Task Scheduler XML.

        ``RepetitionPattern`` with an unbounded duration is the shape Windows
        uses for "every N minutes, forever". The task is registered against the
        interactive user; no account name or password appears, and none is asked
        for - a task that needed a stored password would put one in a file.
        """
        return _TEMPLATE.format(
            interval=f"PT{self.interval_minutes}M",
            command=escape(str(self.executable)),
            arguments=escape(self.arguments),
            working_directory=escape(str(self.working_directory)),
            description=escape(
                "Runs one finite GoldAnalysisPipeline automation tick and exits. "
                "Honours GOLDPIPELINE_AUTOMATION_ENABLED; publishes nothing unless "
                "unattended publishing is separately enabled and allowlisted."
            ),
        )


def build_plan(
    *,
    project_root: Path | None = None,
    task_name: str = DEFAULT_TASK_NAME,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    executable: Path | None = None,
) -> TaskPlan:
    """Work out the exact task definition for this checkout.

    Args:
        project_root: Where the project lives. Defaults to the current
            directory, resolved absolutely - a relative path in a scheduled task
            means whatever Task Scheduler's own directory happens to be.
        task_name: Name to register under.
        interval_minutes: How often to wake the worker.
        executable: Override the interpreter. Defaults to this checkout's
            virtualenv, falling back to the running interpreter when there is no
            ``.venv`` - reported either way so an operator can see which.
    """
    root = (project_root or Path.cwd()).resolve()
    chosen = executable if executable is not None else _resolve_interpreter(root)
    return TaskPlan(
        task_name=task_name,
        executable=chosen,
        arguments=" ".join(WORKER_COMMAND),
        working_directory=root,
        interval_minutes=interval_minutes,
        executable_exists=chosen.is_file(),
    )


def _resolve_interpreter(root: Path) -> Path:
    """Prefer this checkout's virtualenv over whatever is on PATH."""
    candidate = root / VENV_PYTHON
    if candidate.is_file():
        return candidate
    # No virtualenv here. Naming the running interpreter is still far better
    # than "python", which under Task Scheduler could be anything at all.
    return Path(sys.executable)


_TEMPLATE = """<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>{description}</Description>
  </RegistrationInfo>
  <Triggers>
    <TimeTrigger>
      <Repetition>
        <Interval>{interval}</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
      <StartBoundary>2026-01-01T00:00:00</StartBoundary>
      <Enabled>true</Enabled>
    </TimeTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>false</AllowHardTerminate>
    <StartWhenAvailable>false</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{command}</Command>
      <Arguments>{arguments}</Arguments>
      <WorkingDirectory>{working_directory}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


__all__ = [
    "DEFAULT_INTERVAL_MINUTES",
    "DEFAULT_TASK_NAME",
    "VENV_PYTHON",
    "WORKER_COMMAND",
    "TaskPlan",
    "build_plan",
]
