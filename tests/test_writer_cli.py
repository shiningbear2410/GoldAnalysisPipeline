"""CLI integration for the writer stage, offline throughout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import FAKE_API_KEY, make_analysis_payload, make_normalized_run

from goldpipeline.cli import EXIT_INVALID_DATA, EXIT_OK, main
from goldpipeline.prompts import DEFAULT_WRITER_PROMPT

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def invoke(args: list[str]) -> int:
    return main(args)


def draft_args(run_id: str, runs_dir: Path, *extra: str) -> list[str]:
    return [
        "write-draft",
        "--run-id",
        run_id,
        "--runs-dir",
        str(runs_dir),
        "--fake-writer",
        *extra,
    ]


def test_fake_smoke_produces_a_draft(
    normalized_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 22.20: the documented smoke path works with no API call."""
    code = invoke(draft_args(normalized_run.run_id, runs_dir))
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert f"Run: {normalized_run.run_id}" in out
    assert "Writer: fake" in out
    assert "Status: COMPLETED" in out
    assert "claude_draft.md" in out
    assert "claude_writer.json" in out

    run_dir = Path(normalized_run.run_dir)
    assert (run_dir / "claude_draft.md").is_file()
    assert (run_dir / "claude_writer.json").is_file()


def test_fake_mode_needs_no_credentials(
    normalized_run: Any, runs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--fake-writer short-circuits before any credential is read."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert invoke(draft_args(normalized_run.run_id, runs_dir)) == EXIT_OK


def test_json_output_reports_the_essentials(
    normalized_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = invoke(draft_args(normalized_run.run_id, runs_dir, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["run_id"] == normalized_run.run_id
    assert payload["status"] == "DRAFTED"
    assert payload["writer_status"] == "COMPLETED"
    assert payload["provider"] == "fake"
    assert payload["prompt_version"] == DEFAULT_WRITER_PROMPT
    assert payload["article_chars"] > 0


def test_rerun_fails_with_exit_code_2(
    normalized_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invoke(draft_args(normalized_run.run_id, runs_dir))
    capsys.readouterr()

    code = invoke(draft_args(normalized_run.run_id, runs_dir, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_INVALID_DATA
    assert payload["error"]["code"] == "WRITER_ARTIFACT_EXISTS"


def test_unknown_run_is_reported(runs_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(FileNotFoundError):
        invoke(draft_args("20260828_022701_a83f2c", runs_dir))


def test_real_writer_without_a_key_fails_before_any_call(
    normalized_run: Any,
    runs_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without --fake-writer and without a key, the CLI must not reach out."""
    from goldpipeline.cli import EXIT_ERROR

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code = invoke(
        [
            "write-draft",
            "--run-id",
            normalized_run.run_id,
            "--runs-dir",
            str(runs_dir),
        ]
    )
    err = capsys.readouterr().err

    # Exit 1, not 2: this is a configuration problem, not bad data.
    assert code == EXIT_ERROR
    assert "Configuration error" in err
    assert "ANTHROPIC_API_KEY" in err
    assert not (Path(normalized_run.run_dir) / "claude_draft.md").exists()


def test_show_run_lists_the_writer_artifacts(
    normalized_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invoke(draft_args(normalized_run.run_id, runs_dir))
    capsys.readouterr()

    assert invoke(["show-run", normalized_run.run_id, "--runs-dir", str(runs_dir)]) == EXIT_OK
    out = capsys.readouterr().out

    assert "Status: DRAFTED" in out
    assert "claude_draft.md" in out
    assert "claude_writer.json" in out


def test_cli_never_prints_the_key(
    normalized_run: Any,
    runs_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)
    invoke(draft_args(normalized_run.run_id, runs_dir, "--json"))

    captured = capsys.readouterr()
    assert FAKE_API_KEY not in captured.out
    assert FAKE_API_KEY not in captured.err


def test_full_pipeline_from_fixtures(runs_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Round 1 then Round 2, over the shipped fixtures, in one go."""
    code = invoke(
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
    assert code == EXIT_OK
    run_id = capsys.readouterr().out.splitlines()[0].removeprefix("Run created: ")

    assert invoke(draft_args(run_id, runs_dir, "--json")) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "DRAFTED"
    draft = Path(payload["draft"]).read_text(encoding="utf-8")
    assert "NHẬN ĐỊNH VÀNG" in draft
    assert "3314.20" in draft


def test_injection_fixture_runs_end_to_end(
    runs_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The adversarial message produces a normal Run and a normal draft."""
    payload = json.loads((FIXTURES / "telegram_injection.json").read_text(encoding="utf-8"))
    run = make_normalized_run(runs_dir, tmp_path, analysis=payload)

    assert invoke(draft_args(run.run_id, runs_dir, "--json")) == EXIT_OK
    result = json.loads(capsys.readouterr().out)

    assert result["status"] == "DRAFTED"
    assert "SOURCE_PRICE_OUT_OF_RANGE" in result["warnings"]

    draft = Path(result["draft"]).read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY" not in draft
    assert "BTCUSD" not in draft
    assert "9999" not in draft


def test_warnings_are_shown_to_the_operator(
    runs_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = make_normalized_run(
        runs_dir,
        tmp_path,
        analysis=make_analysis_payload(raw_text="Giá vàng hiện tại là 9999, mua ngay."),
    )
    assert invoke(draft_args(run.run_id, runs_dir)) == EXIT_OK
    out = capsys.readouterr().out
    assert "SOURCE_PRICE_OUT_OF_RANGE" in out
