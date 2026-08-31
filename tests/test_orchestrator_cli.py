"""CLI integration for the orchestrator.

Nothing here reaches Telegram, Anthropic or OpenAI. Every invocation is offline,
and a socket guard proves the claim rather than asserting it.

The command this file is really testing is the *default* one: the flags a tired
operator types at the end of the day. It must run every check and send nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from goldpipeline.cli import EXIT_BLOCKED, EXIT_ERROR, EXIT_INVALID_DATA, EXIT_OK, main

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

FROZEN_NOW = "2026-08-28T03:00:00Z"
"""Shortly after the shipped fixtures end, so recency checks stay quiet."""


def invoke(args: list[str]) -> int:
    return main(args)


def run_args(runs_dir: Path, *extra: str) -> list[str]:
    return [
        "pipeline-run",
        "--telegram",
        str(FIXTURES / "telegram_sample.json"),
        "--ohlc",
        str(FIXTURES / "ohlc_sample.json"),
        "--symbol",
        "XAUUSD",
        "--runs-dir",
        str(runs_dir),
        "--now",
        FROZEN_NOW,
        "--fake-ai",
        *extra,
    ]


def resume_args(runs_dir: Path, run_id: str, *extra: str) -> list[str]:
    return [
        "pipeline-resume",
        "--run-id",
        run_id,
        "--runs-dir",
        str(runs_dir),
        "--fake-ai",
        *extra,
    ]


def json_run(runs_dir: Path, capsys: pytest.CaptureFixture[str], *extra: str) -> dict[str, Any]:
    """Run the pipeline and return the parsed JSON summary."""
    invoke(run_args(runs_dir, "--json", *extra))
    payload: dict[str, Any] = json.loads(capsys.readouterr().out)
    return payload


# --- the offline smoke path -----------------------------------------------


def test_offline_pipeline_smoke(runs_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Requirement 56: one command, JSON inputs in, READY_TO_PUBLISH out."""
    code = invoke(run_args(runs_dir))
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "Final status: READY_TO_PUBLISH" in out
    assert "Mode: READY_FOR_PUBLISH" in out


def test_the_execution_summary_names_every_stage(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 58."""
    invoke(run_args(runs_dir))
    out = capsys.readouterr().out

    assert "NORMALIZE  COMPLETED" in out
    assert "WRITER     COMPLETED" in out
    assert "REVIEW     PASS" in out
    assert "FINALIZE   PASSTHROUGH" in out
    assert "GATE       APPROVED" in out


def test_the_summary_does_not_dump_the_article(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 28 of the spec: concise by default."""
    invoke(run_args(runs_dir))
    out = capsys.readouterr().out

    assert "NHẬN ĐỊNH VÀNG" not in out
    assert len(out.splitlines()) < 20


def test_the_json_summary_is_machine_readable(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 29 of the spec."""
    payload = json_run(runs_dir, capsys)

    assert payload["status"] == "COMPLETED"
    assert payload["run_status"] == "READY_TO_PUBLISH"
    assert payload["mode"] == "READY_FOR_PUBLISH"
    assert payload["publish_decision"] == "APPROVED"
    assert payload["publish_status"] is None
    assert [stage["stage"] for stage in payload["stages"]] == [
        "NORMALIZE",
        "WRITE",
        "REVIEW",
        "FINALIZE",
        "GATE",
    ]


def test_the_full_fake_pipeline_opens_no_socket(
    runs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 52.

    ``--fake-ai --fake-publisher --publish`` is the most network-shaped command
    this CLI offers, and it must still be entirely offline.
    """
    import socket

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the offline pipeline must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", explode)
    assert invoke(run_args(runs_dir, "--fake-publisher", "--publish")) == EXIT_OK


# --- publishing is never the default --------------------------------------


def test_the_default_command_never_publishes(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 49.

    No ``--publish``, no ``--mode``: the Run reaches the gate's approval and
    stops there, with no intent and no result on disk.
    """
    payload = json_run(runs_dir, capsys)
    run_dir = runs_dir / payload["run_id"]

    assert payload["publish_status"] is None
    assert not (run_dir / "publish_intent.json").exists()
    assert not (run_dir / "publish_result.json").exists()


def test_real_publishing_requires_the_confirmation_flag(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirements 20 (spec) and 51.

    ``--publish`` on its own is refused before anything runs, so the guard costs
    nothing and cannot be reached by accident. No network is involved in
    proving it.
    """
    code = invoke(run_args(runs_dir, "--publish"))
    err = capsys.readouterr().err

    assert code == EXIT_ERROR
    assert "--confirm-real-publish" in err
    assert list(runs_dir.iterdir()) == []


@pytest.mark.parametrize("rejected", ["--all", "--chat-id=@somewhere", "--yes"])
def test_no_catch_all_flag_exists(
    runs_dir: Path, capsys: pytest.CaptureFixture[str], rejected: str
) -> None:
    """Requirement 20 of the spec.

    There is no ``--all``, no synonym for one, and no way to redirect the
    destination from a command line. Publishing for real takes two flags that
    each say what they do.
    """
    from goldpipeline.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(run_args(runs_dir, rejected))
    capsys.readouterr()

    parsed = build_parser().parse_args(run_args(runs_dir, "--publish", "--confirm-real-publish"))
    assert parsed.publish is True
    assert parsed.confirm_real_publish is True


def test_publish_mode_with_the_offline_transport_needs_no_confirmation(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 50, and the reason the guard is scoped to the real transport.

    ``--fake-publisher`` cannot reach anyone, so demanding a confirmation for it
    would train operators to type the confirmation reflexively.
    """
    payload = json_run(runs_dir, capsys, "--publish", "--fake-publisher")

    assert payload["mode"] == "PUBLISH"
    assert payload["publish_status"] == "PUBLISHED"
    assert payload["run_status"] == "PUBLISHED"


def test_the_offline_transport_wins_over_the_confirmation_flag(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both flags together is allowed, and nothing is sent.

    Rejecting the pair would push an operator towards dropping
    ``--fake-publisher``, which is the wrong flag to drop.
    """
    payload = json_run(runs_dir, capsys, "--publish", "--fake-publisher", "--confirm-real-publish")

    assert payload["publish_status"] == "PUBLISHED"
    assert payload["run_status"] == "PUBLISHED"


def test_fake_full_publish_smoke(runs_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Requirement 57."""
    code = invoke(run_args(runs_dir, "--publish", "--fake-publisher"))
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "PUBLISH    PUBLISHED" in out
    assert "Final status: PUBLISHED" in out


def test_generate_only_stops_before_the_gate(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = json_run(runs_dir, capsys, "--mode", "generate-only")

    assert payload["run_status"] == "FINALIZED"
    assert payload["publish_decision"] is None


# --- exit codes -----------------------------------------------------------


def test_a_clean_run_exits_zero(runs_dir: Path) -> None:
    """Requirement 59, first case."""
    assert invoke(run_args(runs_dir)) == EXIT_OK


def test_a_blocked_gate_exits_blocked(
    runs_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 59: a gate declining is not a failure.

    Driven through a real Run that the gate refuses, then resumed through the
    CLI so the exit code comes from the command rather than the service.
    """
    from conftest import make_tracked_clients, run_orchestrated

    clients = make_tracked_clients()
    blocked = run_orchestrated(
        runs_dir,
        tmp_path,
        clients,
        article="Vàng đang giằng co trong biên hẹp, chưa có tín hiệu rõ ràng.",
    )
    capsys.readouterr()

    code = invoke(resume_args(runs_dir, blocked.run_id, "--publish", "--fake-publisher"))

    assert code == EXIT_BLOCKED
    assert "Final status: PUBLISH_BLOCKED" in capsys.readouterr().err


def test_bad_input_data_exits_invalid_data(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 59: a broken input is a different problem from a gate."""
    code = invoke(
        [
            "pipeline-run",
            "--telegram",
            str(FIXTURES / "telegram_sample.json"),
            "--ohlc",
            str(FIXTURES / "ohlc_invalid_duplicate.json"),
            "--runs-dir",
            str(runs_dir),
            "--fake-ai",
        ]
    )
    err = capsys.readouterr().err

    assert code == EXIT_INVALID_DATA
    assert "Final status: FAILED" in err


def test_a_missing_run_exits_error(runs_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = invoke(resume_args(runs_dir, "20260828_010101_abcdef"))

    assert code == EXIT_ERROR
    assert "no such run" in capsys.readouterr().err


def test_a_locked_run_exits_error_and_names_the_holder(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A concurrency conflict is neither bad data nor a gate: its own code."""
    from goldpipeline.services.run_lock import RunLock

    invoke(run_args(runs_dir, "--mode", "generate-only"))
    run_id = next(p.name for p in runs_dir.iterdir())
    capsys.readouterr()

    with RunLock(runs_dir / run_id, pid=31337, hostname="other-box"):
        code = invoke(resume_args(runs_dir, run_id))

    err = capsys.readouterr().err
    assert code == EXIT_ERROR
    assert "already being executed" in err
    assert "31337" in err


# --- resuming through the CLI ---------------------------------------------


def test_resume_continues_where_the_run_stopped(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invoke(run_args(runs_dir, "--mode", "generate-only"))
    run_id = next(p.name for p in runs_dir.iterdir())
    capsys.readouterr()

    code = invoke(resume_args(runs_dir, run_id, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert [stage["stage"] for stage in payload["stages"]] == ["GATE"]
    assert payload["run_status"] == "READY_TO_PUBLISH"


def test_resuming_a_published_run_is_a_quiet_success(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invoke(run_args(runs_dir, "--publish", "--fake-publisher"))
    run_id = next(p.name for p in runs_dir.iterdir())
    capsys.readouterr()

    code = invoke(resume_args(runs_dir, run_id, "--publish", "--fake-publisher", "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["status"] == "ALREADY_COMPLETED"
    assert payload["stages"] == []


# --- the existing commands still work -------------------------------------


def test_the_single_stage_commands_are_unchanged(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 5 of the spec: adding orchestration breaks no existing path."""
    codes = [
        invoke(
            [
                "create-run",
                "--telegram",
                str(FIXTURES / "telegram_sample.json"),
                "--ohlc",
                str(FIXTURES / "ohlc_sample.json"),
                "--runs-dir",
                str(runs_dir),
                "--now",
                FROZEN_NOW,
            ]
        )
    ]
    run_id = next(p.name for p in runs_dir.iterdir())
    for command, flag in (
        ("write-draft", "--fake-writer"),
        ("review-draft", "--fake-reviewer"),
        ("finalize", "--fake-finalizer"),
    ):
        codes.append(invoke([command, "--run-id", run_id, "--runs-dir", str(runs_dir), flag]))
    codes.append(invoke(["gate-publish", "--run-id", run_id, "--runs-dir", str(runs_dir)]))
    codes.append(
        invoke(
            [
                "publish",
                "--run-id",
                run_id,
                "--runs-dir",
                str(runs_dir),
                "--fake-publisher",
            ]
        )
    )
    capsys.readouterr()

    assert codes == [EXIT_OK] * 6
