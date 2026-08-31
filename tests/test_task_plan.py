"""The Windows Task Scheduler definition.

Generated, never registered. Every assertion here is about one of the four
choices that make an unattended worker safe on Windows, and each is the kind of
thing that fails silently at 3am if it drifts: the wrong interpreter, the wrong
directory, an overlap policy that kills a publish mid-request, or a principal
that cannot see the MetaTrader window.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import pytest

from goldpipeline.services.task_plan import (
    DEFAULT_INTERVAL_MINUTES,
    DEFAULT_TASK_NAME,
    VENV_PYTHON,
    WORKER_COMMAND,
    build_plan,
)

NAMESPACE = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A checkout complete with a virtualenv interpreter, at a path with a space."""
    root = tmp_path / "07. PC Fx" / "01. GoldAnalysisPipeline"
    (root / VENV_PYTHON.parent).mkdir(parents=True)
    (root / VENV_PYTHON).write_text("", encoding="utf-8")
    return root


def parse(xml: str) -> ElementTree.Element:
    return ElementTree.fromstring(xml)


def text(xml: str, path: str) -> str:
    node = parse(xml).find(path, NAMESPACE)
    assert node is not None and node.text is not None, f"missing {path}"
    return node.text


# --- the interpreter -------------------------------------------------------


def test_the_plan_names_this_checkouts_own_interpreter(project: Path) -> None:
    """Requirement 57.

    Task Scheduler runs with a PATH that has little to do with an operator's
    shell, so "python" on it may be a different interpreter with none of this
    project's dependencies. The absolute path is the whole point.
    """
    plan = build_plan(project_root=project)

    assert plan.executable == project / VENV_PYTHON
    assert plan.executable.is_absolute()
    assert plan.executable_exists
    assert plan.executable.name == "python.exe"


def test_a_missing_interpreter_is_reported_rather_than_guessed(tmp_path: Path) -> None:
    """No ``.venv`` here, so the plan names the running interpreter and says so."""
    plan = build_plan(project_root=tmp_path)

    assert plan.executable.is_absolute()
    assert plan.executable.name != "python"


def test_the_command_line_quotes_a_path_containing_spaces(project: Path) -> None:
    """Requirement 68: this project lives under `07. PC Fx`."""
    plan = build_plan(project_root=project)

    assert " " in str(plan.executable)
    assert plan.command_line.startswith('"')
    assert plan.command_line.endswith(" ".join(WORKER_COMMAND))


def test_the_scheduler_runs_the_worker_entry_point(project: Path) -> None:
    """Not ``automation-run-once``.

    The worker entry point honours the kill switch, so a registered task can be
    switched off without unregistering it - which matters, because unregistering
    is the step people forget to undo.
    """
    plan = build_plan(project_root=project)

    assert plan.arguments == "-m goldpipeline automation-worker-tick"
    assert "run-once" not in plan.arguments


# --- the working directory -------------------------------------------------


def test_the_working_directory_is_explicit_and_absolute(project: Path) -> None:
    """Requirements 58 and 36 of the spec.

    Runs, the inbox and the automation state all resolve relative to the current
    directory. Without this the worker would quietly build a second, empty set
    of them wherever Task Scheduler happened to start.
    """
    plan = build_plan(project_root=project)
    xml = plan.to_xml()

    assert plan.working_directory == project.resolve()
    assert text(xml, ".//t:WorkingDirectory") == str(project.resolve())


# --- overlap and principal -------------------------------------------------


def test_the_overlap_policy_skips_rather_than_kills(project: Path) -> None:
    """Requirements 60 and 37 of the spec.

    A tick can outlast a minute. ``StopExisting`` would kill a stage
    mid-request - possibly mid-``sendMessage``, which is precisely the ambiguity
    Round 6 exists to avoid. A skipped tick costs a minute; a killed publish
    costs certainty.
    """
    xml = build_plan(project_root=project).to_xml()

    assert text(xml, ".//t:MultipleInstancesPolicy") == "IgnoreNew"
    assert "StopExisting" not in xml
    assert "Parallel" not in xml
    assert text(xml, ".//t:AllowHardTerminate") == "false"


def test_the_task_runs_as_the_interactive_user(project: Path) -> None:
    """Requirement 61.

    MetaTrader 5 is a desktop application bound to a logged-in session. A SYSTEM
    task cannot see it, so it would defer every event forever while looking
    perfectly healthy.
    """
    xml = build_plan(project_root=project).to_xml()

    assert text(xml, ".//t:LogonType") == "InteractiveToken"
    assert "SYSTEM" not in xml
    assert "S-1-5-18" not in xml, "the SYSTEM account SID"


def test_the_definition_carries_no_credential(project: Path) -> None:
    """Requirements 62, 63, 54 and 55.

    The definition carries a command line and a schedule, and nothing else. A
    task needing a stored password would put one in a file, which is why this
    one is registered against the interactive session instead.

    The two path elements are excluded before scanning: they are wherever the
    checkout happens to live, which is environment rather than content, and a
    temp directory named after this very test would otherwise fail it.
    """
    plan = build_plan(project_root=project)
    lowered = (
        plan.to_xml()
        .replace(str(plan.executable), "")
        .replace(str(plan.working_directory), "")
        .lower()
    )

    # Named precisely rather than by keyword: "InteractiveToken" is the
    # principal, and a blanket ban on the word "token" would fail on the very
    # element that keeps a password out of this file.
    for forbidden in (
        "password",
        "anthropic_api_key",
        "openai_api_key",
        "telegram_bot_token",
        "api_key",
        "apikey",
        "<userid>",
        "sk-",
    ):
        assert forbidden not in lowered, forbidden

    assert "interactivetoken" in lowered, "the principal that makes a password unnecessary"


# --- the schedule ----------------------------------------------------------


def test_the_schedule_is_every_minute(project: Path) -> None:
    """Requirement 59."""
    plan = build_plan(project_root=project)

    assert plan.interval_minutes == DEFAULT_INTERVAL_MINUTES == 1
    assert text(plan.to_xml(), ".//t:Interval") == "PT1M"


def test_the_interval_is_configurable(project: Path) -> None:
    plan = build_plan(project_root=project, interval_minutes=5)

    assert text(plan.to_xml(), ".//t:Interval") == "PT5M"


def test_the_repetition_does_not_stop_itself(project: Path) -> None:
    xml = build_plan(project_root=project).to_xml()

    assert text(xml, ".//t:StopAtDurationEnd") == "false"
    assert text(xml, ".//t:Enabled") == "true"


def test_there_is_no_execution_time_limit(project: Path) -> None:
    """``PT0S`` means unlimited.

    A limit would let Windows terminate a tick mid-provider-call, which is the
    same hazard as ``StopExisting``. The worker bounds itself with its own soft
    deadline instead.
    """
    assert text(build_plan(project_root=project).to_xml(), ".//t:ExecutionTimeLimit") == "PT0S"


# --- generation is not registration ----------------------------------------


def test_generating_a_plan_touches_nothing(project: Path, tmp_path: Path) -> None:
    """Requirements 64, 65 and 40 of the spec.

    Building a plan reads the filesystem and returns a string. There is no
    install command and no ``--apply``: registering a task that runs every
    minute is a decision to make while looking at the definition, not a side
    effect of printing it.
    """
    import goldpipeline.services.task_plan as module

    before = sorted(p.name for p in project.iterdir())
    plan = build_plan(project_root=project)
    plan.to_xml()

    assert sorted(p.name for p in project.iterdir()) == before
    assert not hasattr(module, "install")
    assert not hasattr(module, "register")
    assert not hasattr(module, "remove")


def test_nothing_here_imports_a_windows_scheduler_api() -> None:
    """Requirement 66: the module is importable and testable on any platform."""
    from pathlib import Path as _Path

    source = (
        _Path(__file__).resolve().parent.parent
        / "src"
        / "goldpipeline"
        / "services"
        / "task_plan.py"
    ).read_text(encoding="utf-8")

    for forbidden in ("win32com", "subprocess", "schtasks", "pywin32", "os.system"):
        assert forbidden not in source, forbidden


# --- the document itself ---------------------------------------------------


def test_the_xml_parses_and_is_a_task(project: Path) -> None:
    root = parse(build_plan(project_root=project).to_xml())

    assert root.tag.endswith("Task")
    assert root.attrib["version"] == "1.4"


def test_the_plan_is_deterministic(project: Path) -> None:
    """Two builds of the same checkout produce identical bytes.

    A definition that drifted between runs would make "did anything change?"
    unanswerable during an incident.
    """
    assert build_plan(project_root=project).to_xml() == build_plan(project_root=project).to_xml()


def test_the_default_name_is_recognisable() -> None:
    assert DEFAULT_TASK_NAME == "GoldAnalysisPipeline Automation"
