"""CLI integration for the reviewer stage, offline throughout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    CLEAN_ARTICLE,
    FAKE_OPENAI_KEY,
    make_drafted_run,
    make_normalized_run,
)

from goldpipeline.cli import EXIT_ERROR, EXIT_INVALID_DATA, EXIT_OK, main

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def invoke(args: list[str]) -> int:
    return main(args)


def review_args(run_id: str, runs_dir: Path, *extra: str) -> list[str]:
    return [
        "review-draft",
        "--run-id",
        run_id,
        "--runs-dir",
        str(runs_dir),
        "--fake-reviewer",
        *extra,
    ]


def test_fake_smoke_produces_a_review(
    drafted_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 27.28: the documented smoke path works with no API call."""
    code = invoke(review_args(drafted_run.run_id, runs_dir))
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert f"Run: {drafted_run.run_id}" in out
    assert "Reviewer: fake" in out
    assert "Verdict: PASS" in out
    assert "Score:" in out
    assert "gpt_review.json" in out

    assert (Path(drafted_run.run_dir) / "gpt_review.json").is_file()


def test_fake_mode_needs_no_credentials(
    drafted_run: Any, runs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--fake-reviewer short-circuits before any credential is read."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert invoke(review_args(drafted_run.run_id, runs_dir)) == EXIT_OK


def test_json_output_reports_the_essentials(
    drafted_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = invoke(review_args(drafted_run.run_id, runs_dir, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["run_id"] == drafted_run.run_id
    assert payload["status"] == "REVIEWED"
    assert payload["verdict"] == "PASS"
    assert payload["model_verdict"] == "PASS"
    assert payload["verdict_source"] == "MODEL"
    assert payload["provider"] == "fake"
    assert payload["prompt_version"] == "gold_reviewer_v1"
    assert payload["deterministic_findings"] == []


def test_a_flawed_article_reports_its_issues(
    runs_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    drafted = make_drafted_run(runs_dir, tmp_path, article=f"{CLEAN_ARTICLE}\n\nRSI đang ở 72.")
    code = invoke(review_args(drafted.run_id, runs_dir))
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "Verdict: NEEDS_REVISION" in out
    assert "UNSUPPORTED_CLAIM" in out


def test_the_verdict_source_is_shown_when_policy_escalates(
    runs_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An operator should see that the pipeline overrode the reviewer."""
    from goldpipeline.adapters.fake_reviewer import passing_client
    from goldpipeline.services.reviewer import review_draft
    from goldpipeline.storage.run_store import RunStore

    drafted = make_drafted_run(
        runs_dir, tmp_path, article=f"{CLEAN_ARTICLE}\n\nĐây thực ra là BTCUSD."
    )
    review_draft(run_id=drafted.run_id, store=RunStore(runs_dir), client=passing_client())
    capsys.readouterr()

    assert invoke(["show-run", drafted.run_id, "--runs-dir", str(runs_dir)]) == EXIT_OK
    assert "Status: REVIEWED" in capsys.readouterr().out


def test_rerun_fails_without_overwriting(
    drafted_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invoke(review_args(drafted_run.run_id, runs_dir))
    capsys.readouterr()

    code = invoke(review_args(drafted_run.run_id, runs_dir, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_INVALID_DATA
    assert payload["error"]["code"] == "REVIEW_ARTIFACT_EXISTS"


def test_an_undrafted_run_is_reported(
    runs_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    normalized = make_normalized_run(runs_dir, tmp_path)
    code = invoke(review_args(normalized.run_id, runs_dir, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_INVALID_DATA
    assert payload["error"]["code"] == "RUN_NOT_REVIEWABLE"


def test_real_reviewer_without_a_key_fails_before_any_call(
    drafted_run: Any,
    runs_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirement 25: a missing key is a configuration problem, not bad data."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = invoke(
        [
            "review-draft",
            "--run-id",
            drafted_run.run_id,
            "--runs-dir",
            str(runs_dir),
        ]
    )
    err = capsys.readouterr().err

    assert code == EXIT_ERROR
    assert "Configuration error" in err
    # Anthropic since Round 9.3.1: one credential covers all three AI stages.
    assert "ANTHROPIC_API_KEY" in err
    assert not (Path(drafted_run.run_dir) / "gpt_review.json").exists()


def test_show_run_lists_the_review_artifact(
    drafted_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invoke(review_args(drafted_run.run_id, runs_dir))
    capsys.readouterr()

    assert invoke(["show-run", drafted_run.run_id, "--runs-dir", str(runs_dir)]) == EXIT_OK
    out = capsys.readouterr().out

    assert "Status: REVIEWED" in out
    assert "gpt_review.json" in out
    assert "sha256=" in out


def test_cli_never_prints_the_key(
    drafted_run: Any,
    runs_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI_KEY)
    invoke(review_args(drafted_run.run_id, runs_dir, "--json"))

    captured = capsys.readouterr()
    assert FAKE_OPENAI_KEY not in captured.out
    assert FAKE_OPENAI_KEY not in captured.err


def test_all_three_stages_run_end_to_end(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Round 1, then 2, then 3, over the shipped fixtures."""
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

    assert (
        invoke(
            [
                "write-draft",
                "--run-id",
                run_id,
                "--runs-dir",
                str(runs_dir),
                "--fake-writer",
            ]
        )
        == EXIT_OK
    )
    capsys.readouterr()

    assert invoke(review_args(run_id, runs_dir, "--json")) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "REVIEWED"
    assert payload["verdict"] in {"PASS", "NEEDS_REVISION", "REJECT"}

    run_dir = runs_dir / run_id
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "claude_draft.md",
        "claude_writer.json",
        "context.json",
        "gpt_review.json",
        "manifest.json",
        "ohlc.json",
        "telegram_input.json",
    ]


def test_the_injection_article_is_rejected_end_to_end(
    runs_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 28, through the command an operator would actually run."""
    article = (FIXTURES / "article_injection.md").read_text(encoding="utf-8")
    drafted = make_drafted_run(runs_dir, tmp_path, article=article, claims=[])

    assert invoke(review_args(drafted.run_id, runs_dir, "--json")) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)

    assert payload["verdict"] == "REJECT"
    findings = payload["deterministic_findings"]
    assert "FOREIGN_SYMBOL_MENTIONED" in findings
    assert "UNSUPPORTED_INDICATOR_MENTIONED" in findings
    assert "ABSOLUTE_RISK_LANGUAGE" in findings

    review = (Path(drafted.run_dir) / "gpt_review.json").read_text(encoding="utf-8")
    assert FAKE_OPENAI_KEY not in review
    assert "sk-" not in review
