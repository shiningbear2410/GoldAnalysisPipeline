"""Shared test fixtures and payload builders.

Tests build payloads through these helpers rather than hand-writing dicts, so a
schema change surfaces in one place instead of thirty.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

BASE_TIME = datetime(2026, 8, 28, 0, 0, tzinfo=UTC)
"""Anchor for generated series. Chosen to match the shipped fixtures."""

VIETNAMESE_TEXT = (
    "Vàng đang giằng co quanh vùng 3.314 sau khi bật lên từ hỗ trợ 3.305.\n"
    "Kịch bản ưu tiên: chờ điều chỉnh về 3.309 - 3.311 để tìm cơ hội mua lên.\n"
    "Lưu ý: đây là quan điểm cá nhân, không phải khuyến nghị đầu tư."
)
"""Text with a full range of Vietnamese diacritics, for encoding assertions."""


def make_bar(
    *,
    minute_offset: int = 0,
    open_: str = "3312.45",
    high: str = "3315.10",
    low: str = "3311.90",
    close: str = "3314.20",
    volume: str | None = "1489",
    timestamp: str | None = None,
    symbol: str | None = None,
) -> dict[str, Any]:
    """Build one valid OHLC bar payload."""
    when = timestamp or (BASE_TIME + timedelta(minutes=minute_offset)).isoformat().replace(
        "+00:00", "Z"
    )
    bar: dict[str, Any] = {
        "timestamp": when,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    }
    if volume is not None:
        bar["volume"] = volume
    if symbol is not None:
        bar["symbol"] = symbol
    return bar


def make_series(count: int = 12, *, step_minutes: int = 15) -> list[dict[str, Any]]:
    """Build a contiguous, ascending, internally valid series of *count* bars.

    Prices drift upward by 0.50 per bar so that no two bars are identical and
    every bar satisfies the OHLC ordering invariants.
    """
    bars: list[dict[str, Any]] = []
    for index in range(count):
        base = 3300 + index * 0.5
        bars.append(
            make_bar(
                minute_offset=index * step_minutes,
                open_=f"{base:.2f}",
                high=f"{base + 1.20:.2f}",
                low=f"{base - 0.80:.2f}",
                close=f"{base + 0.40:.2f}",
                volume=str(1500 + index * 10),
            )
        )
    return bars


def make_market_payload(
    *,
    bars: list[dict[str, Any]] | None = None,
    symbol: str = "XAUUSD",
    provider: str = "mt5-demo",
    timeframe: str = "M15",
    timezone: str | None = None,
    requested_at: str | None = "2026-08-28T03:00:00Z",
    **extra: Any,
) -> dict[str, Any]:
    """Build a market data payload, with any field overridable.

    ``timezone`` is omitted by default: the generated bars carry explicit ``Z``
    offsets, so no source timezone is needed to interpret them.
    """
    payload: dict[str, Any] = {
        "symbol": symbol,
        "provider": provider,
        "timeframe": timeframe,
        "bars": bars if bars is not None else make_series(),
    }
    if timezone is not None:
        payload["timezone"] = timezone
    if requested_at is not None:
        payload["requested_at"] = requested_at
    payload.update(extra)
    return payload


def make_analysis_payload(
    *,
    raw_text: str = VIETNAMESE_TEXT,
    include_metadata: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    """Build a raw analysis payload."""
    payload: dict[str, Any] = {"source": "telegram", "raw_text": raw_text}
    if include_metadata:
        payload.update(
            {
                "chat_id": -1002145890733,
                "message_id": 48217,
                "message_date": "2026-08-28T02:18:44Z",
                "received_at": "2026-08-28T02:20:12Z",
                "author": {"id": 5512340098, "username": "gold_desk_vn"},
                "metadata": {"chat_title": "Gold Signals VN"},
            }
        )
    payload.update(extra)
    return payload


def write_json(path: Path, payload: Any) -> Path:
    """Write *payload* as UTF-8 JSON and return the path."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    """An isolated runs root, so tests never touch the repository's runs/."""
    target = tmp_path / "runs"
    target.mkdir()
    return target


@pytest.fixture
def source_files(tmp_path: Path) -> tuple[Path, Path]:
    """A valid (analysis, market) pair of JSON files on disk."""
    analysis = write_json(tmp_path / "telegram_input.json", make_analysis_payload())
    market = write_json(tmp_path / "ohlc.json", make_market_payload())
    return analysis, market


@pytest.fixture
def frozen_now() -> datetime:
    """A fixed 'now' shortly after the generated series ends."""
    return BASE_TIME + timedelta(hours=3)


# --------------------------------------------------------------------------
# Round 2 helpers
# --------------------------------------------------------------------------

WRITER_NOW = datetime(2026, 8, 28, 3, 30, tzinfo=UTC)
"""Deterministic 'now' for writer artifacts."""

FAKE_API_KEY = "anthropic-test-credential-DO-NOT-LEAK-0123456789abcdef"
"""A recognisable stand-in credential for the Anthropic stages.

Tests grep artifacts and logs for this exact string, so a regression that starts
writing the key somewhere fails loudly instead of quietly.

Deliberately *not* shaped like a real key. Nothing here depends on the ``sk-ant-``
prefix, and a fake value carrying it would trip secret scanners and push
protection forever after - a false alarm that teaches people to ignore the real
ones.
"""


def make_normalized_run(
    runs_dir: Path,
    tmp_path: Path,
    *,
    analysis: dict[str, Any] | None = None,
    market: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> Any:
    """Create a real NORMALIZED Run on disk, the way Round 1 would.

    The writer stage is tested against genuine Run directories rather than
    hand-built ones, so the two rounds stay honestly coupled.
    """
    from goldpipeline.adapters.file_source import (
        JsonFileAnalysisSource,
        JsonFileMarketDataSource,
    )
    from goldpipeline.services.pipeline import create_run
    from goldpipeline.storage.run_store import RunStore

    sources = tmp_path / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    analysis_path = write_json(sources / "telegram_input.json", analysis or make_analysis_payload())
    market_path = write_json(sources / "ohlc.json", market or make_market_payload())

    result = create_run(
        analysis_source=JsonFileAnalysisSource(analysis_path),
        market_source=JsonFileMarketDataSource(market_path),
        store=RunStore(runs_dir),
        expected_symbol="XAUUSD",
        now=now or (BASE_TIME + timedelta(hours=3)),
    )
    assert result.succeeded, f"fixture run failed to normalize: {result.error}"
    return result


@pytest.fixture
def normalized_run(runs_dir: Path, tmp_path: Path) -> Any:
    """A freshly normalized Run, ready for the writer stage."""
    return make_normalized_run(runs_dir, tmp_path)


@pytest.fixture
def sample_context(normalized_run: Any) -> Any:
    """The AnalysisContext of a freshly normalized Run."""
    return normalized_run.context


# --------------------------------------------------------------------------
# Round 3 helpers
# --------------------------------------------------------------------------

REVIEW_NOW = datetime(2026, 8, 28, 4, 15, tzinfo=UTC)
"""Deterministic 'now' for review artifacts."""

FAKE_OPENAI_KEY = "openai-test-credential-DO-NOT-LEAK-fedcba9876543210"
"""A recognisable stand-in credential for the reviewer stage.

Same reasoning as :data:`FAKE_API_KEY`: unique and greppable, but not shaped
like a vendor key.
"""

_LATEST_BAR = make_series()[-1]

LATEST_CLOSE = _LATEST_BAR["close"]
LATEST_HIGH = _LATEST_BAR["high"]
LATEST_LOW = _LATEST_BAR["low"]
"""The last candle of the default generated series.

Derived rather than written down: an article quoting a number the series does
not contain is precisely what the reviewer stage flags, so the fixtures must not
be able to drift into that state by accident.
"""

CLEAN_ARTICLE = (
    "\U0001f56f NHẬN ĐỊNH VÀNG\n"
    "\n"
    "⚡ Chốt nhanh\n"
    f"Giá gần nhất trong dữ liệu quanh {LATEST_CLOSE}, thị trường đang tích luỹ.\n"
    "\n"
    "📍 Giá đang ở đâu\n"
    f"Nến M15 gần nhất của XAUUSD đóng cửa tại {LATEST_CLOSE}, "
    f"đỉnh {LATEST_HIGH} và đáy {LATEST_LOW}.\n"
    "\n"
    "🎯 Kịch bản\n"
    "Ưu tiên quan sát thêm. Nếu giá giữ được vùng hỗ trợ, kịch bản tăng vẫn còn hiệu lực.\n"
    "\n"
    "⚠️ Lưu ý\n"
    "Đây là quan điểm cá nhân, không phải khuyến nghị đầu tư."
)
"""An article whose every number comes from the generated fixture series."""


def make_drafted_run(
    runs_dir: Path,
    tmp_path: Path,
    *,
    article: str = CLEAN_ARTICLE,
    claims: list[Any] | None = None,
    warnings: list[Any] | None = None,
    analysis: dict[str, Any] | None = None,
    market: dict[str, Any] | None = None,
    writer_status: Any = None,
) -> Any:
    """Create a real DRAFTED Run whose article is *article*.

    Built by driving the actual writer stage with a fake client rather than by
    writing files and patching digests: the reviewer stage should be tested
    against Runs the pipeline itself produced, hashes and all.
    """
    from goldpipeline.adapters.fake_writer import FakeWriterClient
    from goldpipeline.schemas.writer import (
        ClaimType,
        SourceClaim,
        WriterModelOutput,
        WriterStatus,
    )
    from goldpipeline.services.writer import write_draft
    from goldpipeline.storage.run_store import RunStore

    normalized = make_normalized_run(runs_dir, tmp_path, analysis=analysis, market=market)

    default_claims = [
        SourceClaim(type=ClaimType.PRICE, value=LATEST_CLOSE, source="context.price.latest_close"),
        SourceClaim(type=ClaimType.MARKET_META, value="XAUUSD", source="context.market.symbol"),
    ]

    def build(request: Any) -> WriterModelOutput:
        return WriterModelOutput(
            run_id=request.run_id,
            status=writer_status or WriterStatus.COMPLETED,
            title="Nhận định vàng",
            article=article,
            source_claims=default_claims if claims is None else claims,
            warnings=warnings or [],
        )

    drafted = write_draft(
        run_id=normalized.run_id,
        store=RunStore(runs_dir),
        client=FakeWriterClient(output_factory=build),
        now=WRITER_NOW,
    )
    assert drafted.succeeded, f"fixture run failed to draft: {drafted.error}"
    return drafted


@pytest.fixture
def drafted_run(runs_dir: Path, tmp_path: Path) -> Any:
    """A freshly drafted Run with a clean article, ready for review."""
    return make_drafted_run(runs_dir, tmp_path)


def tamper(run_dir: Path, filename: str, content: str) -> None:
    """Overwrite an artifact *without* updating the manifest.

    Simulates an edit made outside the pipeline - which is exactly what the
    integrity checks exist to catch, so the manifest is deliberately left stale.
    """
    (Path(run_dir) / filename).write_text(content, encoding="utf-8")


def load_review(run_dir: Path) -> Any:
    """Read a Run's review artifact back through its schema."""
    from goldpipeline.schemas.review import ReviewResult

    return ReviewResult.model_validate_json(
        (Path(run_dir) / "gpt_review.json").read_text(encoding="utf-8")
    )


# --------------------------------------------------------------------------
# Round 4 helpers
# --------------------------------------------------------------------------

FINALIZE_NOW = datetime(2026, 8, 28, 5, 0, tzinfo=UTC)
"""Deterministic 'now' for finalizer artifacts."""

RSI_ARTICLE = f"{CLEAN_ARTICLE}\n\nRSI đang ở 72, tín hiệu tăng rõ ràng."
"""A draft with one invented indicator: HIGH, so the review asks for revision."""

BTCUSD_ARTICLE = f"{CLEAN_ARTICLE}\n\nThực ra đây là phân tích BTCUSD."
"""A draft naming a foreign instrument: CRITICAL, so the review rejects it."""


def make_reviewed_run(
    runs_dir: Path,
    tmp_path: Path,
    *,
    article: str = CLEAN_ARTICLE,
    claims: list[Any] | None = None,
    analysis: dict[str, Any] | None = None,
    market: dict[str, Any] | None = None,
    review_client: Any = None,
) -> Any:
    """Create a real REVIEWED Run, ready for the finalizer.

    Built by driving the actual writer and reviewer stages rather than by
    writing files and patching digests: the finalizer verifies four artifacts
    against each other, so it must be tested against Runs the pipeline itself
    produced.
    """
    from goldpipeline.adapters.fake_reviewer import FakeReviewerClient
    from goldpipeline.services.reviewer import review_draft
    from goldpipeline.storage.run_store import RunStore

    drafted = make_drafted_run(
        runs_dir, tmp_path, article=article, claims=claims, analysis=analysis, market=market
    )
    reviewed = review_draft(
        run_id=drafted.run_id,
        store=RunStore(runs_dir),
        client=review_client or FakeReviewerClient(),
        now=REVIEW_NOW,
    )
    assert reviewed.succeeded, f"fixture run failed to review: {reviewed.error}"
    return reviewed


@pytest.fixture
def reviewed_run(runs_dir: Path, tmp_path: Path) -> Any:
    """A reviewed Run whose verdict is PASS."""
    return make_reviewed_run(runs_dir, tmp_path)


@pytest.fixture
def revisable_run(runs_dir: Path, tmp_path: Path) -> Any:
    """A reviewed Run whose verdict is NEEDS_REVISION."""
    from goldpipeline.schemas.review import ReviewStatus

    reviewed = make_reviewed_run(runs_dir, tmp_path, article=RSI_ARTICLE, claims=[])
    assert reviewed.result is not None
    assert reviewed.result.status is ReviewStatus.NEEDS_REVISION
    return reviewed


@pytest.fixture
def rejected_run(runs_dir: Path, tmp_path: Path) -> Any:
    """A reviewed Run whose verdict is REJECT."""
    from goldpipeline.schemas.review import ReviewStatus

    reviewed = make_reviewed_run(runs_dir, tmp_path, article=BTCUSD_ARTICLE, claims=[])
    assert reviewed.result is not None
    assert reviewed.result.status is ReviewStatus.REJECT
    return reviewed


def load_finalization(run_dir: Path) -> Any:
    """Read a Run's finalizer artifact back through its schema."""
    from goldpipeline.schemas.finalizer import FinalizerResult

    return FinalizerResult.model_validate_json(
        (Path(run_dir) / "claude_finalizer.json").read_text(encoding="utf-8")
    )
