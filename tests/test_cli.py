"""CLI integration: the smoke path a human actually runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import make_analysis_payload, make_market_payload, make_series, write_json

from goldpipeline.cli import EXIT_INVALID_DATA, EXIT_OK, main

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def invoke(args: list[str]) -> int:
    return main(args)


def test_create_run_smoke(
    tmp_path: Path, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The documented command produces the documented output."""
    analysis = write_json(tmp_path / "telegram.json", make_analysis_payload())
    market = write_json(tmp_path / "ohlc.json", make_market_payload())

    code = invoke(
        [
            "create-run",
            "--telegram",
            str(analysis),
            "--ohlc",
            str(market),
            "--symbol",
            "XAUUSD",
            "--runs-dir",
            str(runs_dir),
        ]
    )
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "Run created: " in out
    assert "Status: NORMALIZED" in out
    assert "context.json" in out

    run_dirs = list(runs_dir.iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "context.json").exists()


def test_create_run_over_the_shipped_fixtures(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fixtures in the repository must actually work."""
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
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["status"] == "NORMALIZED"
    assert payload["bar_count"] == 20


def test_minimal_fixture_is_accepted(runs_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A message with no optional metadata still produces a usable Run."""
    code = invoke(
        [
            "create-run",
            "--telegram",
            str(FIXTURES / "telegram_minimal.json"),
            "--ohlc",
            str(FIXTURES / "ohlc_sample.json"),
            "--runs-dir",
            str(runs_dir),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert "MISSING_TELEGRAM_METADATA" in payload["warnings"]
    assert payload["quality_status"] == "WARNING"


def test_invalid_fixture_fails_with_exit_code_2(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = invoke(
        [
            "create-run",
            "--telegram",
            str(FIXTURES / "telegram_sample.json"),
            "--ohlc",
            str(FIXTURES / "ohlc_invalid_duplicate.json"),
            "--runs-dir",
            str(runs_dir),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_INVALID_DATA
    assert payload["status"] == "FAILED"
    assert payload["error"]["code"] == "DUPLICATE_TIMESTAMP"
    assert not (Path(payload["run_dir"]) / "context.json").exists()


def test_dry_run_leaves_no_run_behind(
    tmp_path: Path, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    analysis = write_json(tmp_path / "telegram.json", make_analysis_payload())
    market = write_json(tmp_path / "ohlc.json", make_market_payload())

    code = invoke(
        [
            "create-run",
            "--telegram",
            str(analysis),
            "--ohlc",
            str(market),
            "--runs-dir",
            str(runs_dir),
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "no Run was created" in out
    assert list(runs_dir.iterdir()) == []


def test_dry_run_reports_invalid_data(
    tmp_path: Path, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bars = make_series(4)
    bars.append(dict(bars[1]))
    analysis = write_json(tmp_path / "telegram.json", make_analysis_payload())
    market = write_json(tmp_path / "ohlc.json", make_market_payload(bars=bars))

    code = invoke(
        [
            "create-run",
            "--telegram",
            str(analysis),
            "--ohlc",
            str(market),
            "--runs-dir",
            str(runs_dir),
            "--dry-run",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_INVALID_DATA
    assert payload["valid"] is False
    assert payload["error"]["code"] == "DUPLICATE_TIMESTAMP"
    assert list(runs_dir.iterdir()) == []


def test_show_and_list_runs(
    tmp_path: Path, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    analysis = write_json(tmp_path / "telegram.json", make_analysis_payload())
    market = write_json(tmp_path / "ohlc.json", make_market_payload())
    invoke(
        [
            "create-run",
            "--telegram",
            str(analysis),
            "--ohlc",
            str(market),
            "--runs-dir",
            str(runs_dir),
        ]
    )
    run_id = capsys.readouterr().out.splitlines()[0].removeprefix("Run created: ")

    assert invoke(["list-runs", "--runs-dir", str(runs_dir)]) == EXIT_OK
    assert run_id in capsys.readouterr().out

    assert invoke(["show-run", run_id, "--runs-dir", str(runs_dir)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "Status: NORMALIZED" in out
    assert "context.json" in out
    assert "sha256=" in out


def test_missing_source_file_is_reported(
    tmp_path: Path, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    market = write_json(tmp_path / "ohlc.json", make_market_payload())
    code = invoke(
        [
            "create-run",
            "--telegram",
            str(tmp_path / "nope.json"),
            "--ohlc",
            str(market),
            "--runs-dir",
            str(runs_dir),
            "--json",
        ]
    )
    assert code == EXIT_INVALID_DATA
    assert "INPUT_VALIDATION_ERROR" in capsys.readouterr().out


def test_now_flag_makes_runs_reproducible(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--now pins the recency checks so a fixture-driven Run stays clean."""
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
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["quality_status"] == "OK"
    assert payload["warnings"] == []


def test_now_flag_rejects_naive_datetimes(runs_dir: Path) -> None:
    with pytest.raises(SystemExit):
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
                "2026-08-28T02:20:12",
            ]
        )


def test_fixture_recency_is_actually_checked(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Running the same fixture much later flags it as stale rather than fresh."""
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
            "2026-09-05T00:00:00Z",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert "STALE_DATA" in payload["warnings"]


def test_vietnamese_paths_do_not_crash_the_cli(
    tmp_path: Path, runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Error messages embed file paths, and this user's paths contain Vietnamese."""
    vietnamese_dir = tmp_path / "phân tích vàng"
    vietnamese_dir.mkdir()
    market = write_json(vietnamese_dir / "ohlc.json", make_market_payload())

    code = invoke(
        [
            "create-run",
            "--telegram",
            str(vietnamese_dir / "không tồn tại.json"),
            "--ohlc",
            str(market),
            "--runs-dir",
            str(runs_dir),
        ]
    )
    assert code == EXIT_INVALID_DATA
    assert "không tồn tại.json" in capsys.readouterr().err
