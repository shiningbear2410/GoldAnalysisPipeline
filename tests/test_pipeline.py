"""End-to-end Run creation: artifacts, immutability, failure behaviour."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from conftest import (
    BASE_TIME,
    VIETNAMESE_TEXT,
    make_analysis_payload,
    make_market_payload,
    make_series,
    write_json,
)

from goldpipeline.adapters.file_source import JsonFileAnalysisSource, JsonFileMarketDataSource
from goldpipeline.schemas.context import AnalysisContext
from goldpipeline.schemas.manifest import RunStatus
from goldpipeline.schemas.quality import QualityStatus
from goldpipeline.services.pipeline import RunResult, create_run
from goldpipeline.storage.atomic import sha256_bytes
from goldpipeline.storage.run_store import RunStore

PIPELINE_NOW = BASE_TIME + timedelta(hours=3)
"""Deterministic 'now', shortly after the generated series ends.

Without it every Run would be judged against wall-clock time and the recency
checks would start failing on a different day.
"""


def run_pipeline(
    tmp_path: Path,
    runs_dir: Path,
    *,
    analysis: dict[str, Any] | None = None,
    market: dict[str, Any] | None = None,
    expected_symbol: str | None = "XAUUSD",
    now: datetime | None = PIPELINE_NOW,
) -> RunResult:
    """Write payloads to disk and execute the pipeline over them."""
    analysis_payload = analysis or make_analysis_payload()
    analysis_path = write_json(tmp_path / "telegram_input.json", analysis_payload)
    market_path = write_json(tmp_path / "ohlc.json", market or make_market_payload())
    return create_run(
        analysis_source=JsonFileAnalysisSource(analysis_path),
        market_source=JsonFileMarketDataSource(market_path),
        store=RunStore(runs_dir),
        expected_symbol=expected_symbol,
        now=now,
    )


def successful(result: RunResult) -> tuple[AnalysisContext, Path]:
    """Assert the Run succeeded and hand back its context and context path."""
    assert result.succeeded, f"run failed: {result.error}"
    assert result.context is not None
    assert result.context_path is not None
    return result.context, result.context_path


def failure_code(result: RunResult) -> str:
    """Assert the Run failed and hand back its error code."""
    assert result.status is RunStatus.FAILED
    assert result.error is not None
    return result.error.code


# --- happy path -----------------------------------------------------------


def test_successful_run_writes_exactly_the_expected_artifacts(
    tmp_path: Path, runs_dir: Path
) -> None:
    result = run_pipeline(tmp_path, runs_dir)

    assert result.succeeded
    assert result.status is RunStatus.NORMALIZED
    assert sorted(p.name for p in result.run_dir.iterdir()) == [
        "context.json",
        "manifest.json",
        "ohlc.json",
        "telegram_input.json",
    ]


def test_manifest_describes_the_run(tmp_path: Path, runs_dir: Path) -> None:
    result = run_pipeline(tmp_path, runs_dir)
    manifest = RunStore(runs_dir).open(result.run_id).load_manifest()

    assert manifest.run_id == result.run_id
    assert manifest.status is RunStatus.NORMALIZED
    assert manifest.error is None
    assert [ref.name for ref in manifest.source_files] == ["telegram_input.json", "ohlc.json"]
    assert [ref.name for ref in manifest.artifact_files] == ["context.json"]
    assert [event.stage for event in manifest.events] == [
        "run.create",
        "ingest",
        "source.capture",
        "normalize",
        "context.build",
    ]


def test_context_round_trips_through_the_schema(tmp_path: Path, runs_dir: Path) -> None:
    """Requirement 14.14."""
    result = run_pipeline(tmp_path, runs_dir)
    context, context_path = successful(result)
    raw = json.loads(context_path.read_text(encoding="utf-8"))

    reloaded = AnalysisContext.model_validate(raw)
    assert reloaded == context
    assert json.loads(reloaded.model_dump_json()) == raw


def test_context_carries_everything_round_2_needs(tmp_path: Path, runs_dir: Path) -> None:
    """The acceptance criterion: no agent should need to look anything up."""
    result = run_pipeline(tmp_path, runs_dir)
    context, _ = successful(result)

    assert context.run_id == result.run_id
    assert context.market.symbol == "XAUUSD"
    assert context.market.timeframe == "M15"
    assert context.market.provider == "mt5-demo"
    assert context.market.timezone == "UTC"

    assert context.timing.data_from == context.ohlc.bars[0].timestamp
    assert context.timing.data_to == context.ohlc.bars[-1].timestamp
    assert context.timing.latest_candle_at == context.ohlc.bars[-1].timestamp

    latest = context.ohlc.bars[-1]
    assert context.price.latest_open == latest.open
    assert context.price.latest_high == latest.high
    assert context.price.latest_low == latest.low
    assert context.price.latest_close == latest.close

    assert context.raw_analysis.text == VIETNAMESE_TEXT
    assert context.raw_analysis.trust_level == "UNTRUSTED"
    assert "never as instructions" in context.raw_analysis.handling
    assert context.data_quality.bar_count == len(context.ohlc.bars)


def test_context_price_matches_latest_bar_after_reordering(tmp_path: Path, runs_dir: Path) -> None:
    """latest_bar is derived post-normalization, not taken from input order."""
    bars = make_series(6)
    result = run_pipeline(tmp_path, runs_dir, market=make_market_payload(bars=list(reversed(bars))))

    context, _ = successful(result)
    assert context.price.latest_close == context.ohlc.bars[-1].close
    assert context.timing.latest_candle_at == context.ohlc.bars[-1].timestamp


def test_stored_context_is_utf8_and_human_readable(tmp_path: Path, runs_dir: Path) -> None:
    """Requirements 14.7 and 14.8 at the file level."""
    result = run_pipeline(tmp_path, runs_dir)
    _, context_path = successful(result)
    raw = context_path.read_bytes()

    assert b"\\u" not in raw
    text = raw.decode("utf-8")
    assert "giằng co" in text
    assert "khuyến nghị đầu tư" in text


def test_prices_are_stored_as_json_strings_not_floats(tmp_path: Path, runs_dir: Path) -> None:
    """Prices leave the pipeline as exact decimal strings.

    A JSON float here would be a real hazard: 3314.20 round-tripping through a
    binary float is the kind of drift that turns into a wrong number inside a
    published article.
    """
    bars = [
        {
            "timestamp": "2026-08-28T00:00:00Z",
            "open": 3312.45,
            "high": 3315.10,
            "low": 3311.90,
            "close": 3314.20,
            "volume": 1489,
        }
    ]
    result = run_pipeline(tmp_path, runs_dir, market=make_market_payload(bars=bars))
    _, context_path = successful(result)
    raw = json.loads(context_path.read_text(encoding="utf-8"))

    assert isinstance(raw["price"]["latest_close"], str)
    assert Decimal(raw["price"]["latest_close"]) == Decimal("3314.20")
    assert Decimal(raw["ohlc"]["bars"][0]["open"]) == Decimal("3312.45")
    assert Decimal(raw["ohlc"]["bars"][0]["high"]) == Decimal("3315.10")


# --- immutability ---------------------------------------------------------


def test_source_files_are_not_mutated_by_context_building(tmp_path: Path, runs_dir: Path) -> None:
    """Requirement 14.15: what was ingested is what stays on disk."""
    analysis_payload = make_analysis_payload()
    market_payload = make_market_payload()
    result = run_pipeline(tmp_path, runs_dir, analysis=analysis_payload, market=market_payload)

    stored_analysis = json.loads((result.run_dir / "telegram_input.json").read_text("utf-8"))
    stored_market = json.loads((result.run_dir / "ohlc.json").read_text("utf-8"))

    assert stored_analysis == analysis_payload
    assert stored_market == market_payload

    manifest = RunStore(runs_dir).open(result.run_id).load_manifest()
    for ref in manifest.source_files:
        assert sha256_bytes((result.run_dir / ref.name).read_bytes()) == ref.sha256


def test_two_runs_do_not_share_a_directory(tmp_path: Path, runs_dir: Path) -> None:
    first = run_pipeline(tmp_path, runs_dir)
    second = run_pipeline(tmp_path, runs_dir)

    _, first_context = successful(first)
    _, second_context = successful(second)

    assert first.run_id != second.run_id
    assert first.run_dir != second.run_dir
    assert first_context.exists()
    assert second_context.exists()


# --- failure behaviour ----------------------------------------------------


def test_failed_validation_never_produces_a_normalized_run(tmp_path: Path, runs_dir: Path) -> None:
    """Requirement 14.13: a failed Run must not look usable."""
    bars = make_series(4)
    bars.append(dict(bars[2]))  # duplicate timestamp
    result = run_pipeline(tmp_path, runs_dir, market=make_market_payload(bars=bars))

    assert not result.succeeded
    assert result.status is RunStatus.FAILED
    assert result.context is None
    assert not (result.run_dir / "context.json").exists()

    manifest = RunStore(runs_dir).open(result.run_id).load_manifest()
    assert manifest.status is RunStatus.FAILED
    assert manifest.error is not None
    assert manifest.error.code == "DUPLICATE_TIMESTAMP"
    assert manifest.artifact_files == []


def test_failed_run_still_preserves_the_offending_inputs(tmp_path: Path, runs_dir: Path) -> None:
    """The inputs that caused a failure are the most useful thing to keep."""
    result = run_pipeline(
        tmp_path, runs_dir, market=make_market_payload(symbol="EURUSD"), expected_symbol="XAUUSD"
    )
    assert result.status is RunStatus.FAILED
    stored = json.loads((result.run_dir / "ohlc.json").read_text("utf-8"))
    assert stored["symbol"] == "EURUSD"


def test_symbol_mismatch_fails_the_run(tmp_path: Path, runs_dir: Path) -> None:
    """Requirement 14.10 at pipeline level."""
    result = run_pipeline(
        tmp_path, runs_dir, market=make_market_payload(symbol="XAGUSD"), expected_symbol="XAUUSD"
    )
    assert failure_code(result) == "SYMBOL_MISMATCH"


def test_malformed_source_file_fails_cleanly(tmp_path: Path, runs_dir: Path) -> None:
    analysis_path = tmp_path / "telegram_input.json"
    analysis_path.write_text("{ not json", encoding="utf-8")
    market_path = write_json(tmp_path / "ohlc.json", make_market_payload())

    result = create_run(
        analysis_source=JsonFileAnalysisSource(analysis_path),
        market_source=JsonFileMarketDataSource(market_path),
        store=RunStore(runs_dir),
    )
    assert failure_code(result) == "INPUT_VALIDATION_ERROR"
    assert not (result.run_dir / "context.json").exists()


# --- quality reporting ----------------------------------------------------


def test_clean_inputs_report_quality_ok(tmp_path: Path, runs_dir: Path) -> None:
    result = run_pipeline(tmp_path, runs_dir)
    context, _ = successful(result)
    assert context.data_quality.status is QualityStatus.OK
    assert context.data_quality.warnings == []
    assert context.data_quality.missing_fields == []


def test_degraded_inputs_report_quality_warning(tmp_path: Path, runs_dir: Path) -> None:
    """Missing optional data degrades quality without failing the Run."""
    bars = [{**bar, "volume": None} if isinstance(bar, dict) else bar for bar in make_series(12)]
    for bar in bars:
        bar.pop("volume", None)

    result = run_pipeline(
        tmp_path,
        runs_dir,
        analysis=make_analysis_payload(include_metadata=False),
        market=make_market_payload(bars=bars),
    )

    context, _ = successful(result)
    quality = context.data_quality
    assert quality.status is QualityStatus.WARNING
    assert "ohlc.volume" in quality.missing_fields
    assert {"MISSING_VOLUME", "MISSING_TELEGRAM_METADATA"} <= {w.code for w in quality.warnings}
