"""CLI integration for live ingestion and the MetaTrader diagnostic.

Nothing here reaches a terminal, a broker, Telegram, Anthropic or OpenAI. Every
invocation runs against the offline stand-ins, and a socket guard proves the
claim rather than asserting it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import SAMPLE_EVENT_ID, make_event_payload, write_json

from goldpipeline.cli import EXIT_BLOCKED, EXIT_ERROR, EXIT_INVALID_DATA, EXIT_OK, main


def invoke(args: list[str]) -> int:
    return main(args)


@pytest.fixture
def event_file(tmp_path: Path) -> Path:
    return write_json(tmp_path / "event.json", make_event_payload())


@pytest.fixture
def inbox_dir(tmp_path: Path) -> Path:
    return tmp_path / "inbox"


def ingest_args(event: Path, inbox_dir: Path, runs_dir: Path, *extra: str) -> list[str]:
    return [
        "pipeline-ingest",
        "--analysis",
        str(event),
        "--inbox-dir",
        str(inbox_dir),
        "--runs-dir",
        str(runs_dir),
        "--fake-mt5",
        "--fake-ai",
        *extra,
    ]


# --- the ingestion smoke path ---------------------------------------------


def test_ingesting_an_event_reaches_ready_to_publish(
    event_file: Path, inbox_dir: Path, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 31 of the spec: the whole flow behind one command."""
    code = invoke(ingest_args(event_file, inbox_dir, runs_dir))
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert f"Event: {SAMPLE_EVENT_ID} (INGESTED)" in out
    assert "Final status: READY_TO_PUBLISH" in out


def test_the_default_ingestion_never_publishes(
    event_file: Path, inbox_dir: Path, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirements 32 and 39."""
    invoke(ingest_args(event_file, inbox_dir, runs_dir, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "READY_FOR_PUBLISH"
    assert payload["publish_status"] is None
    assert payload["ingest"]["outcome"] == "INGESTED"
    run_dir = runs_dir / payload["run_id"]
    assert not (run_dir / "publish_intent.json").exists()
    assert not (run_dir / "publish_result.json").exists()


def test_normalize_only_stops_at_the_run(
    event_file: Path, inbox_dir: Path, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = invoke(ingest_args(event_file, inbox_dir, runs_dir, "--normalize-only", "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["outcome"] == "INGESTED"
    manifest = json.loads(
        (runs_dir / payload["run_id"] / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "NORMALIZED"


def test_the_whole_ingestion_opens_no_socket(
    event_file: Path, inbox_dir: Path, runs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 44."""
    import socket

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("offline ingestion must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", explode)
    assert invoke(ingest_args(event_file, inbox_dir, runs_dir)) == EXIT_OK


def test_publishing_still_takes_two_deliberate_flags(
    event_file: Path, inbox_dir: Path, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 32 of the spec: Round 7's friction is unchanged here."""
    code = invoke(ingest_args(event_file, inbox_dir, runs_dir, "--publish"))

    assert code == EXIT_ERROR
    assert "--confirm-real-publish" in capsys.readouterr().err
    assert list(runs_dir.iterdir()) == []


def test_a_file_market_source_still_works(
    event_file: Path, inbox_dir: Path, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The JSON path remains available, for replaying a captured snapshot."""
    fixtures = Path(__file__).resolve().parent.parent / "fixtures"
    code = invoke(
        [
            "pipeline-ingest",
            "--analysis",
            str(event_file),
            "--inbox-dir",
            str(inbox_dir),
            "--runs-dir",
            str(runs_dir),
            "--market-source",
            "file",
            "--ohlc",
            str(fixtures / "ohlc_sample.json"),
            "--normalize-only",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["outcome"] == "INGESTED"


def test_a_file_market_source_without_a_path_is_refused(
    event_file: Path, inbox_dir: Path, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = invoke(
        [
            "pipeline-ingest",
            "--analysis",
            str(event_file),
            "--inbox-dir",
            str(inbox_dir),
            "--runs-dir",
            str(runs_dir),
            "--market-source",
            "file",
        ]
    )

    assert code == EXIT_ERROR
    assert "--ohlc" in capsys.readouterr().err


# --- duplicates and conflicts through the CLI -----------------------------


def test_a_duplicate_ingestion_exits_zero_and_names_the_original_run(
    event_file: Path, inbox_dir: Path, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Golden case B, and requirement 37 of the spec.

    Exit zero matters: a producer wrapper that retries on a non-zero code would
    otherwise loop forever over an event that is already done.
    """
    invoke(ingest_args(event_file, inbox_dir, runs_dir, "--normalize-only", "--json"))
    first = json.loads(capsys.readouterr().out)

    code = invoke(ingest_args(event_file, inbox_dir, runs_dir, "--normalize-only", "--json"))
    second = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert second["outcome"] == "ALREADY_INGESTED"
    assert second["run_id"] == first["run_id"]
    assert len(list(runs_dir.iterdir())) == 1


def test_a_conflicting_event_exits_blocked(
    event_file: Path,
    inbox_dir: Path,
    runs_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Golden case C."""
    invoke(ingest_args(event_file, inbox_dir, runs_dir, "--normalize-only"))
    capsys.readouterr()

    other = write_json(
        tmp_path / "conflict.json", make_event_payload(raw_text="Nội dung hoàn toàn khác.")
    )
    code = invoke(ingest_args(other, inbox_dir, runs_dir, "--normalize-only"))
    err = capsys.readouterr().err

    assert code == EXIT_BLOCKED
    assert "CONFLICT" in err
    assert len(list(runs_dir.iterdir())) == 1


def test_an_invalid_payload_exits_invalid_data(
    inbox_dir: Path, runs_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A payload that fails its schema never reaches the inbox at all."""
    payload = make_event_payload()
    del payload["raw_text"]
    bad = write_json(tmp_path / "bad.json", payload)

    code = invoke(ingest_args(bad, inbox_dir, runs_dir))
    err = capsys.readouterr().err

    assert code == EXIT_INVALID_DATA
    assert "schema validation" in err
    assert list(runs_dir.iterdir()) == []


# --- the inbox commands ---------------------------------------------------


def test_submit_then_process_one(
    event_file: Path, inbox_dir: Path, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The production shape: the producer submits, an operator drains.

    Round 9 automates the second half; nothing here loops or waits.
    """
    submitted = invoke(["inbox-submit", "--file", str(event_file), "--inbox-dir", str(inbox_dir)])
    out = capsys.readouterr().out
    assert submitted == EXIT_OK
    assert SAMPLE_EVENT_ID in out
    assert (inbox_dir / "incoming" / f"{SAMPLE_EVENT_ID}.json").is_file()

    code = invoke(
        [
            "inbox-process-one",
            "--inbox-dir",
            str(inbox_dir),
            "--runs-dir",
            str(runs_dir),
            "--fake-mt5",
            "--fake-ai",
        ]
    )

    assert code == EXIT_OK
    assert "Final status: READY_TO_PUBLISH" in capsys.readouterr().out
    assert (inbox_dir / "processed" / f"{SAMPLE_EVENT_ID}.json").is_file()


def test_processing_an_empty_inbox_is_quiet(
    inbox_dir: Path, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = invoke(
        [
            "inbox-process-one",
            "--inbox-dir",
            str(inbox_dir),
            "--runs-dir",
            str(runs_dir),
            "--fake-mt5",
        ]
    )

    assert code == EXIT_OK
    assert "NOTHING_TO_DO" in capsys.readouterr().out


def test_reconcile_reports_and_then_recovers(
    event_file: Path, inbox_dir: Path, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invoke(["inbox-submit", "--file", str(event_file), "--inbox-dir", str(inbox_dir)])
    capsys.readouterr()
    # Strand it exactly as an interrupted consumer would.
    (inbox_dir / "processing").mkdir(parents=True, exist_ok=True)
    (inbox_dir / "incoming" / f"{SAMPLE_EVENT_ID}.json").rename(
        inbox_dir / "processing" / f"{SAMPLE_EVENT_ID}.json"
    )

    code = invoke(["inbox-reconcile", "--inbox-dir", str(inbox_dir), "--runs-dir", str(runs_dir)])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "Interrupted events: 1" in out
    assert "--recover" in out
    assert (inbox_dir / "processing" / f"{SAMPLE_EVENT_ID}.json").is_file()

    invoke(
        [
            "inbox-reconcile",
            "--inbox-dir",
            str(inbox_dir),
            "--runs-dir",
            str(runs_dir),
            "--recover",
        ]
    )
    capsys.readouterr()

    assert (inbox_dir / "incoming" / f"{SAMPLE_EVENT_ID}.json").is_file()


def test_reconcile_says_so_when_there_is_nothing_to_do(
    inbox_dir: Path, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = invoke(["inbox-reconcile", "--inbox-dir", str(inbox_dir), "--runs-dir", str(runs_dir)])

    assert code == EXIT_OK
    assert "Nothing to reconcile" in capsys.readouterr().out


# --- the diagnostic -------------------------------------------------------


def test_mt5_check_reports_the_configuration_and_the_latest_closed_candle(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirement 30 of the spec."""
    code = invoke(["mt5-check", "--fake-mt5"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "MT5 terminal:      connected" in out
    assert "Configured symbol: XAUUSD" in out
    assert "Timeframe:         M15" in out
    assert "Requested bars:    20" in out
    assert "Latest closed:" in out
    assert "Data quality:      OK" in out


def test_mt5_check_prints_no_account_details(capsys: pytest.CaptureFixture[str]) -> None:
    """A diagnostic an operator will paste into a chat must be safe to paste."""
    from goldpipeline.adapters.fake_mt5 import TERMINAL_SECRET_SENTINEL

    invoke(["mt5-check", "--fake-mt5", "--json"])
    captured = capsys.readouterr()

    assert TERMINAL_SECRET_SENTINEL not in captured.out
    assert TERMINAL_SECRET_SENTINEL not in captured.err
    for word in ("password", "login", "token", "account"):
        assert word not in captured.out.lower()


def test_mt5_check_writes_no_run(runs_dir: Path) -> None:
    """Read-only means read-only: no Run, no artifact, no side effect."""
    invoke(["mt5-check", "--fake-mt5"])

    assert list(runs_dir.iterdir()) == []


def test_mt5_check_reports_a_missing_symbol_without_substituting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 13 of the spec: candidates are listed, never chosen."""
    monkeypatch.setenv("GOLDPIPELINE_MT5_SYMBOL", "XAUUSD_NOPE")

    code = invoke(["mt5-check", "--fake-mt5"])
    err = capsys.readouterr().err

    assert code == EXIT_INVALID_DATA
    assert "SYMBOL_NOT_FOUND" in err
    assert "never substituted" in err


def test_mt5_check_json_is_machine_readable(capsys: pytest.CaptureFixture[str]) -> None:
    invoke(["mt5-check", "--fake-mt5", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["connected"] is True
    assert payload["provider_symbol"] == "XAUUSD"
    assert payload["canonical_symbol"] == "XAUUSD"
    assert payload["returned_bars"] == 20
    assert payload["latest_closed_candle"].endswith("Z")
    assert payload["provenance"]["start_pos"] == 1


# --- nothing older broke --------------------------------------------------


def test_the_earlier_commands_are_unchanged(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 45 and 46."""
    fixtures = Path(__file__).resolve().parent.parent / "fixtures"
    code = invoke(
        [
            "pipeline-run",
            "--telegram",
            str(fixtures / "telegram_sample.json"),
            "--ohlc",
            str(fixtures / "ohlc_sample.json"),
            "--runs-dir",
            str(runs_dir),
            "--now",
            "2026-08-28T03:00:00Z",
            "--fake-ai",
        ]
    )
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "Final status: READY_TO_PUBLISH" in out
