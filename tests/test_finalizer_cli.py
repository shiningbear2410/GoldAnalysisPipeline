"""CLI integration for the finalizer stage, offline throughout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import FAKE_API_KEY, make_drafted_run, make_reviewed_run

from goldpipeline.cli import EXIT_BLOCKED, EXIT_ERROR, EXIT_INVALID_DATA, EXIT_OK, main

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def invoke(args: list[str]) -> int:
    return main(args)


def finalize_args(run_id: str, runs_dir: Path, *extra: str) -> list[str]:
    return [
        "finalize",
        "--run-id",
        run_id,
        "--runs-dir",
        str(runs_dir),
        "--fake-finalizer",
        *extra,
    ]


# --- PASS -----------------------------------------------------------------


def test_passthrough_smoke(
    reviewed_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 27.41: the documented PASS path, with no API call."""
    code = invoke(finalize_args(reviewed_run.run_id, runs_dir))
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert f"Run: {reviewed_run.run_id}" in out
    assert "Review: PASS" in out
    assert "Finalization: PASSTHROUGH" in out
    assert "Provider called: No" in out
    assert "claude_final.md" in out

    run_dir = Path(reviewed_run.run_dir)
    assert (run_dir / "claude_final.md").read_bytes() == (run_dir / "claude_draft.md").read_bytes()


def test_passthrough_needs_no_key_and_no_fake_flag(
    reviewed_run: Any, runs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PASS is a byte copy; demanding credentials for it would be wrong."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code = invoke(["finalize", "--run-id", reviewed_run.run_id, "--runs-dir", str(runs_dir)])
    assert code == EXIT_OK


def test_passthrough_json_output(
    reviewed_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = invoke(finalize_args(reviewed_run.run_id, runs_dir, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["status"] == "FINALIZED"
    assert payload["review"] == "PASS"
    assert payload["finalization"] == "PASSTHROUGH"
    assert payload["provider_called"] is False
    assert payload["model"] is None
    assert payload["prompt_version"] is None


# --- NEEDS_REVISION -------------------------------------------------------


def test_revision_smoke(
    revisable_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 27.40."""
    code = invoke(finalize_args(revisable_run.run_id, runs_dir))
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "Review: NEEDS_REVISION" in out
    assert "Finalization: REVISED" in out
    assert "Provider called: Yes" in out
    assert "Issues resolved:" in out
    assert "[APPLIED]" in out

    final = (Path(revisable_run.run_dir) / "claude_final.md").read_text(encoding="utf-8")
    assert "RSI" not in final


def test_revision_json_output(
    revisable_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = invoke(finalize_args(revisable_run.run_id, runs_dir, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["finalization"] == "REVISED"
    assert payload["provider_called"] is True
    assert payload["issues_applied"] == payload["issues_total"]
    assert payload["model"] == "fake-finalizer-v1"
    assert payload["prompt_version"] == "gold_finalizer_v1"


def test_a_revision_without_a_key_is_a_configuration_error(
    revisable_run: Any,
    runs_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirement 25: only the revision path demands credentials."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code = invoke(["finalize", "--run-id", revisable_run.run_id, "--runs-dir", str(runs_dir)])
    err = capsys.readouterr().err

    assert code == EXIT_ERROR
    assert "Configuration error" in err
    assert "ANTHROPIC_API_KEY" in err
    assert not (Path(revisable_run.run_dir) / "claude_final.md").exists()


# --- REJECT ---------------------------------------------------------------


def test_a_rejected_review_is_blocked(
    rejected_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 27.7: a distinct exit code, because retrying will not help."""
    code = invoke(finalize_args(rejected_run.run_id, runs_dir))
    err = capsys.readouterr().err

    assert code == EXIT_BLOCKED
    assert "Review: REJECT" in err
    assert "Finalization blocked." in err
    assert not (Path(rejected_run.run_dir) / "claude_final.md").exists()


def test_a_block_is_reported_in_json_too(
    rejected_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = invoke(finalize_args(rejected_run.run_id, runs_dir, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_BLOCKED
    assert payload["finalization"] == "BLOCKED"
    assert payload["error"]["code"] == "FINALIZATION_BLOCKED"
    assert payload["status"] == "REVIEWED"


def test_a_block_needs_no_key(
    rejected_run: Any, runs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 27.10, through the command an operator runs."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code = invoke(["finalize", "--run-id", rejected_run.run_id, "--runs-dir", str(runs_dir)])
    assert code == EXIT_BLOCKED


# --- guards ---------------------------------------------------------------


def test_rerun_fails_without_overwriting(
    reviewed_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invoke(finalize_args(reviewed_run.run_id, runs_dir))
    capsys.readouterr()

    code = invoke(finalize_args(reviewed_run.run_id, runs_dir, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_INVALID_DATA
    assert payload["error"]["code"] == "FINALIZE_ARTIFACT_EXISTS"


def test_an_unreviewed_run_is_reported(
    runs_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    drafted = make_drafted_run(runs_dir, tmp_path)
    code = invoke(finalize_args(drafted.run_id, runs_dir, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_INVALID_DATA
    assert payload["error"]["code"] == "RUN_NOT_FINALIZABLE"


def test_show_run_lists_the_final_artifacts(
    reviewed_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invoke(finalize_args(reviewed_run.run_id, runs_dir))
    capsys.readouterr()

    assert invoke(["show-run", reviewed_run.run_id, "--runs-dir", str(runs_dir)]) == EXIT_OK
    out = capsys.readouterr().out

    assert "Status: FINALIZED" in out
    assert "claude_final.md" in out
    assert "claude_finalizer.json" in out


def test_the_cli_never_prints_the_key(
    revisable_run: Any,
    runs_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)
    invoke(finalize_args(revisable_run.run_id, runs_dir, "--json"))

    captured = capsys.readouterr()
    assert FAKE_API_KEY not in captured.out
    assert FAKE_API_KEY not in captured.err


# --- the whole pipeline ---------------------------------------------------


def test_all_four_stages_run_end_to_end(runs_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Round 1 through 4, over the shipped fixtures, in one go."""
    assert (
        invoke(
            [
                "create-run",
                "--telegram",
                str(FIXTURES / "telegram_sample.json"),
                "--ohlc",
                str(FIXTURES / "ohlc_sample.json"),
                "--symbol",
                "XAUUSD",
                "--runs-dir",
                str(runs_dir),
                "--now",
                "2026-08-28T02:20:12Z",
            ]
        )
        == EXIT_OK
    )
    run_id = capsys.readouterr().out.splitlines()[0].removeprefix("Run created: ")

    for stage in (
        ["write-draft", "--run-id", run_id, "--runs-dir", str(runs_dir), "--fake-writer"],
        ["review-draft", "--run-id", run_id, "--runs-dir", str(runs_dir), "--fake-reviewer"],
    ):
        assert invoke(stage) == EXIT_OK
        capsys.readouterr()

    assert invoke(finalize_args(run_id, runs_dir, "--json")) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "FINALIZED"
    assert payload["finalization"] in {"PASSTHROUGH", "REVISED"}

    run_dir = runs_dir / run_id
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "claude_draft.md",
        "claude_final.md",
        "claude_finalizer.json",
        "claude_writer.json",
        "context.json",
        "gpt_review.json",
        "manifest.json",
        "ohlc.json",
        "telegram_input.json",
    ]


def test_the_injection_fixture_never_reaches_a_final_article(
    runs_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 28, through the command an operator runs."""
    article = (FIXTURES / "article_injection.md").read_text(encoding="utf-8")
    reviewed = make_reviewed_run(
        runs_dir, tmp_path, article=article, claims=[], enforce_contract=False
    )

    code = invoke(finalize_args(reviewed.run_id, runs_dir, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_BLOCKED
    assert payload["finalization"] == "BLOCKED"
    assert not (Path(reviewed.run_dir) / "claude_final.md").exists()
