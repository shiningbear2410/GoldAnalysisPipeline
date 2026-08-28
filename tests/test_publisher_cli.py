"""CLI integration for the publisher.

Nothing here reaches Telegram. ``--fake-publisher`` is the only path these tests
take, and a socket guard proves it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    CLEAN_ARTICLE,
    TELEGRAM_TOKEN_SENTINEL,
    make_finalized_run,
    republish_article,
)

from goldpipeline.cli import EXIT_BLOCKED, EXIT_ERROR, EXIT_INVALID_DATA, EXIT_OK, main

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def invoke(args: list[str]) -> int:
    return main(args)


def publish_args(run_id: str, runs_dir: Path, *extra: str) -> list[str]:
    return [
        "publish",
        "--run-id",
        run_id,
        "--runs-dir",
        str(runs_dir),
        "--fake-publisher",
        *extra,
    ]


# --- the offline smoke path -----------------------------------------------


def test_fake_publish_smoke(
    ready_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 39.70."""
    code = invoke(publish_args(ready_run.run_id, runs_dir))
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert f"Run: {ready_run.run_id}" in out
    assert "Publisher: fake" in out
    assert "Status: PUBLISHED" in out
    assert "Delivered: 1/1" in out
    assert "publish_result.json" in out


def test_fake_publish_json_output(
    ready_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = invoke(publish_args(ready_run.run_id, runs_dir, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["publish_status"] == "PUBLISHED"
    assert payload["status"] == "PUBLISHED"
    assert payload["provider"] == "fake"
    assert payload["confirmed_count"] == payload["chunk_count"] == 1
    assert payload["message_ids"] == [1000]
    assert payload["failure"] is None


def test_the_offline_path_needs_no_credentials(
    ready_run: Any, runs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 34: a smoke run must not need a bot token."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_TARGET_CHAT_ID", raising=False)
    assert invoke(publish_args(ready_run.run_id, runs_dir)) == EXIT_OK


def test_the_fake_target_is_obviously_not_real(
    ready_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An offline attempt must never be mistakable for one that reached Telegram."""
    invoke(publish_args(ready_run.run_id, runs_dir, "--json"))
    payload = json.loads(capsys.readouterr().out)
    assert payload["target_chat"] == "@fake_offline_channel"


def test_the_cli_opens_no_socket(
    ready_run: Any, runs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 39.68."""
    import socket

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the fake publisher must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", explode)
    assert invoke(publish_args(ready_run.run_id, runs_dir)) == EXIT_OK


# --- configuration --------------------------------------------------------


@pytest.mark.parametrize("missing", ["TELEGRAM_BOT_TOKEN", "TELEGRAM_TARGET_CHAT_ID"])
def test_real_mode_without_config_fails_before_any_intent(
    ready_run: Any,
    runs_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    missing: str,
) -> None:
    """Requirements 39.27-39.28 and 35."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TELEGRAM_TOKEN_SENTINEL)
    monkeypatch.setenv("TELEGRAM_TARGET_CHAT_ID", "@gold_signals_test")
    monkeypatch.delenv(missing, raising=False)

    code = invoke(["publish", "--run-id", ready_run.run_id, "--runs-dir", str(runs_dir)])
    err = capsys.readouterr().err

    assert code == EXIT_ERROR
    assert "Configuration error" in err
    assert missing in err
    assert not (Path(ready_run.run_dir) / "publish_intent.json").exists()
    assert not (Path(ready_run.run_dir) / "publish_result.json").exists()


def test_an_invalid_target_is_refused_before_any_intent(
    ready_run: Any,
    runs_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirement 39.67."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TELEGRAM_TOKEN_SENTINEL)
    monkeypatch.setenv("TELEGRAM_TARGET_CHAT_ID", "https://evil.example/hook")

    code = invoke(["publish", "--run-id", ready_run.run_id, "--runs-dir", str(runs_dir)])

    assert code == EXIT_ERROR
    assert "TELEGRAM_TARGET_CHAT_ID" in capsys.readouterr().err
    assert not (Path(ready_run.run_dir) / "publish_intent.json").exists()


@pytest.mark.parametrize("flag", ["--chat-id", "--target", "--target-chat"])
def test_there_is_no_flag_to_redirect_the_destination(
    ready_run: Any, runs_dir: Path, flag: str
) -> None:
    """Requirement 7: a destination flag is one typo from the wrong channel.

    Asserted as behaviour rather than by inspecting the parser: what matters is
    that the command refuses the argument, however the parser is built.
    """
    with pytest.raises(SystemExit) as exit_info:
        invoke(publish_args(ready_run.run_id, runs_dir, flag, "@somewhere_else"))

    assert exit_info.value.code != 0
    assert not (Path(ready_run.run_dir) / "publish_intent.json").exists()


def test_the_cli_never_prints_the_token(
    ready_run: Any,
    runs_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirement 39.30."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TELEGRAM_TOKEN_SENTINEL)
    invoke(publish_args(ready_run.run_id, runs_dir, "--json"))

    captured = capsys.readouterr()
    assert TELEGRAM_TOKEN_SENTINEL not in captured.out
    assert TELEGRAM_TOKEN_SENTINEL not in captured.err


# --- refusals -------------------------------------------------------------


def test_an_ungated_run_is_refused(
    runs_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 39.71."""
    finalized = make_finalized_run(runs_dir, tmp_path)
    code = invoke(publish_args(finalized.run_id, runs_dir))

    assert code == EXIT_INVALID_DATA
    assert "PUBLISHER_NOT_APPROVED" in capsys.readouterr().err
    assert not (Path(finalized.run_dir) / "publish_intent.json").exists()


def test_a_blocked_run_is_refused(
    runs_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from goldpipeline.services.publish_gate import gate_publish
    from goldpipeline.storage.run_store import RunStore

    finalized = make_finalized_run(runs_dir, tmp_path)
    republish_article(runs_dir, finalized.run_id, f"{CLEAN_ARTICLE}\n\nThực ra đây là BTCUSD.")
    gate_publish(run_id=finalized.run_id, store=RunStore(runs_dir))

    code = invoke(publish_args(finalized.run_id, runs_dir))
    assert code == EXIT_INVALID_DATA
    assert "PUBLISHER_NOT_APPROVED" in capsys.readouterr().err


def test_a_second_attempt_is_refused(
    ready_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invoke(publish_args(ready_run.run_id, runs_dir))
    capsys.readouterr()

    code = invoke(publish_args(ready_run.run_id, runs_dir))
    assert code == EXIT_INVALID_DATA
    assert "PUBLISHER_ARTIFACT_EXISTS" in capsys.readouterr().err


def test_an_orphan_intent_reports_uncertainty(
    ready_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 39.72 / golden case F, through the command."""
    from goldpipeline.schemas.manifest import RunStatus
    from goldpipeline.storage.run_store import RunStore

    invoke(publish_args(ready_run.run_id, runs_dir))
    capsys.readouterr()

    run_dir = Path(ready_run.run_dir)
    (run_dir / "publish_result.json").unlink()
    store = RunStore(runs_dir)
    run = store.open(ready_run.run_id)
    manifest = run.load_manifest()
    manifest.artifact_files = [
        ref for ref in manifest.artifact_files if ref.name != "publish_result.json"
    ]
    manifest.status = RunStatus.PUBLISHING
    run.save_manifest(manifest)

    code = invoke(publish_args(ready_run.run_id, runs_dir, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_BLOCKED
    assert payload["publish_status"] == "UNCERTAIN"
    assert payload["status"] == "PUBLISH_UNCERTAIN"
    assert payload["failure"]["code"] == "ORPHAN_PUBLISH_INTENT"
    assert payload["message_ids"] == []


def test_show_run_lists_the_publisher_artifacts(
    ready_run: Any, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invoke(publish_args(ready_run.run_id, runs_dir))
    capsys.readouterr()

    assert invoke(["show-run", ready_run.run_id, "--runs-dir", str(runs_dir)]) == EXIT_OK
    out = capsys.readouterr().out

    assert "Status: PUBLISHED" in out
    assert "publish_intent.json" in out
    assert "publish_result.json" in out


# --- the whole pipeline ---------------------------------------------------


def test_all_six_stages_run_end_to_end(
    runs_dir: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every stage over the shipped fixtures, with no credentials anywhere."""
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TELEGRAM_BOT_TOKEN"):
        monkeypatch.delenv(name, raising=False)

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
        ["finalize", "--run-id", run_id, "--runs-dir", str(runs_dir), "--fake-finalizer"],
        ["gate-publish", "--run-id", run_id, "--runs-dir", str(runs_dir)],
    ):
        assert invoke(stage) == EXIT_OK
        capsys.readouterr()

    assert invoke(publish_args(run_id, runs_dir, "--json")) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["publish_status"] == "PUBLISHED"

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
        "publish_decision.json",
        "publish_intent.json",
        "publish_result.json",
        "telegram_input.json",
    ]


def test_no_earlier_stage_publishes_anything(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 33: each stage is independent; only `publish` sends."""
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
    run_dir = runs_dir / run_id

    for stage in (
        ["write-draft", "--run-id", run_id, "--runs-dir", str(runs_dir), "--fake-writer"],
        ["review-draft", "--run-id", run_id, "--runs-dir", str(runs_dir), "--fake-reviewer"],
        ["finalize", "--run-id", run_id, "--runs-dir", str(runs_dir), "--fake-finalizer"],
        ["gate-publish", "--run-id", run_id, "--runs-dir", str(runs_dir)],
    ):
        invoke(stage)
        capsys.readouterr()
        assert not (run_dir / "publish_intent.json").exists()
        assert not (run_dir / "publish_result.json").exists()
