"""Shared test fixtures and payload builders.

Tests build payloads through these helpers rather than hand-writing dicts, so a
schema change surfaces in one place instead of thirty.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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


# --------------------------------------------------------------------------
# Round 5 helpers
# --------------------------------------------------------------------------

GATE_NOW = datetime(2026, 8, 28, 6, 0, tzinfo=UTC)
"""Deterministic 'now' for publish decisions."""


def make_finalized_run(
    runs_dir: Path,
    tmp_path: Path,
    *,
    article: str = CLEAN_ARTICLE,
    claims: list[Any] | None = None,
    analysis: dict[str, Any] | None = None,
    market: dict[str, Any] | None = None,
    review_client: Any = None,
    finalizer_client: Any = None,
) -> Any:
    """Create a real FINALIZED Run, ready for the publish gate.

    Driven through all four real stages rather than assembled by hand: the gate
    verifies eight artifacts against each other and against the manifest, so
    only a Run the pipeline itself produced is a fair test of it.
    """
    from goldpipeline.adapters.fake_finalizer import FakeFinalizerClient
    from goldpipeline.services.finalizer import finalize_run
    from goldpipeline.storage.run_store import RunStore

    reviewed = make_reviewed_run(
        runs_dir,
        tmp_path,
        article=article,
        claims=claims,
        analysis=analysis,
        market=market,
        review_client=review_client,
    )
    finalized = finalize_run(
        run_id=reviewed.run_id,
        store=RunStore(runs_dir),
        client=finalizer_client or FakeFinalizerClient(),
        now=FINALIZE_NOW,
    )
    assert finalized.succeeded, f"fixture run failed to finalize: {finalized.error}"
    return finalized


@pytest.fixture
def finalized_run(runs_dir: Path, tmp_path: Path) -> Any:
    """A finalized Run whose article is clean and whose review passed."""
    return make_finalized_run(runs_dir, tmp_path)


def republish_article(runs_dir: Path, run_id: str, article: str) -> None:
    """Replace a Run's final article and re-stamp every digest that names it.

    The gate's first check is the artifact chain, so a test about *content*
    must leave that chain consistent - otherwise every such test would trip the
    integrity check instead of the scanner it is aiming at. This rewrites the
    final article and updates the manifest and the finalizer metadata to match,
    which is exactly what an honest pipeline producing that article would have
    written.
    """
    import json

    from goldpipeline.storage.atomic import encode_text, sha256_bytes
    from goldpipeline.storage.run_store import RunStore

    store = RunStore(runs_dir)
    run = store.open(run_id)
    run_dir = Path(run.path)

    payload = encode_text(article)
    (run_dir / "claude_final.md").write_bytes(payload)
    digest = sha256_bytes(payload)

    metadata = json.loads((run_dir / "claude_finalizer.json").read_text(encoding="utf-8"))
    metadata["final_article_sha256"] = digest
    # `article_chars` is constrained to >= 1, so an empty-article test would
    # otherwise fail schema validation and trip the integrity check instead of
    # reaching the structure check it is aiming at.
    metadata["article_chars"] = max(1, len(article.strip()))
    encoded = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    (run_dir / "claude_finalizer.json").write_bytes(encoded)

    manifest = run.load_manifest()
    for ref in manifest.artifact_files:
        if ref.name == "claude_final.md":
            ref.sha256, ref.size_bytes = digest, len(payload)
        elif ref.name == "claude_finalizer.json":
            ref.sha256, ref.size_bytes = sha256_bytes(encoded), len(encoded)
    run.save_manifest(manifest)


def load_decision(run_dir: Path) -> Any:
    """Read a Run's publish decision back through its schema."""
    from goldpipeline.schemas.publish import PublishDecision

    return PublishDecision.model_validate_json(
        (Path(run_dir) / "publish_decision.json").read_text(encoding="utf-8")
    )


# --------------------------------------------------------------------------
# Round 6 helpers
# --------------------------------------------------------------------------

PUBLISH_NOW = datetime(2026, 8, 28, 7, 30, tzinfo=UTC)
"""Deterministic 'now' for publish artifacts."""

TELEGRAM_TOKEN_SENTINEL = "telegram-test-credential-DO-NOT-LEAK"
"""A stand-in bot token, greppable across artifacts, logs and CLI output.

Deliberately not shaped like a real Telegram token (`<digits>:AA<35 chars>`):
a fake carrying that shape would trip secret scanners and push protection
forever after, and nothing here depends on the format.
"""

TEST_TARGET_CHAT = "@gold_signals_test"


def make_published_ready_run(
    runs_dir: Path,
    tmp_path: Path,
    *,
    article: str = CLEAN_ARTICLE,
    claims: list[Any] | None = None,
    analysis: dict[str, Any] | None = None,
) -> Any:
    """Create a real READY_TO_PUBLISH Run, gate approval and all.

    Driven through all five stages rather than assembled by hand: the publisher
    re-verifies four artifacts against the gate's own digests, so only a Run the
    pipeline produced is a fair test of it.
    """
    from goldpipeline.services.publish_gate import gate_publish
    from goldpipeline.storage.run_store import RunStore

    finalized = make_finalized_run(
        runs_dir, tmp_path, article=article, claims=claims, analysis=analysis
    )
    gated = gate_publish(run_id=finalized.run_id, store=RunStore(runs_dir), now=GATE_NOW)
    assert gated.approved, f"fixture run was blocked: {gated.decision.blockers}"
    return gated


@pytest.fixture
def ready_run(runs_dir: Path, tmp_path: Path) -> Any:
    """A Run the gate approved, ready for the publisher."""
    return make_published_ready_run(runs_dir, tmp_path)


class RecordingSleep:
    """Captures pacing and flood-control waits instead of spending them."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)

    @property
    def total(self) -> float:
        return sum(self.waits)


def load_publish_result(run_dir: Path) -> Any:
    """Read a Run's publish result back through its schema."""
    from goldpipeline.schemas.publisher import PublishResult

    return PublishResult.model_validate_json(
        (Path(run_dir) / "publish_result.json").read_text(encoding="utf-8")
    )


def load_publish_intent(run_dir: Path) -> Any:
    """Read a Run's publish intent back through its schema."""
    from goldpipeline.schemas.publisher import PublishIntent

    return PublishIntent.model_validate_json(
        (Path(run_dir) / "publish_intent.json").read_text(encoding="utf-8")
    )


# --------------------------------------------------------------------------
# Round 7 helpers
# --------------------------------------------------------------------------

PIPELINE_NOW = BASE_TIME + timedelta(hours=3)
"""Deterministic 'now' for a whole orchestrated run.

One instant for every stage, shortly after the generated series ends, so an
orchestrated Run produces the same artifacts on every machine and never trips
the recency checks.
"""


def exploding_factory(name: str) -> Any:
    """A client factory that fails if anything ever calls it.

    The sharpest way to state the lazy-configuration requirement: a test asserts
    that a stage was not reached by proving its client was never *built*, which
    is also the moment a real client would read a credential.
    """

    def build() -> Any:
        raise AssertionError(
            f"the {name} client was constructed, but this invocation should never need one"
        )

    return build


@dataclass
class TrackedClients:
    """The four offline fakes, plus a record of which were handed out.

    ``calls`` on each fake says whether a stage *ran*; ``built`` says whether its
    client was even constructed. The two are different questions - the second is
    the one that decides whether an API key had to be present.
    """

    writer: Any
    reviewer: Any
    finalizer: Any
    publisher: Any
    target_chat: str = "@fake_offline_channel"
    built: list[str] = field(default_factory=list)

    def _hand_out(self, name: str, client: Any) -> Any:
        self.built.append(name)
        return client

    def as_pipeline_clients(self) -> Any:
        """Wrap the fakes as the orchestrator's lazy factories."""
        from goldpipeline.services.orchestrator import PipelineClients

        return PipelineClients(
            writer=lambda: self._hand_out("writer", self.writer),
            reviewer=lambda: self._hand_out("reviewer", self.reviewer),
            finalizer=lambda: self._hand_out("finalizer", self.finalizer),
            publisher=lambda: (
                self._hand_out("publisher", self.publisher),
                self.target_chat,
            ),
        )


def make_tracked_clients(
    *,
    writer: Any = None,
    reviewer: Any = None,
    finalizer: Any = None,
    publisher: Any = None,
    target_chat: str = "@fake_offline_channel",
) -> TrackedClients:
    """Build the four fakes, overriding any of them."""
    from goldpipeline.adapters.fake_finalizer import FakeFinalizerClient
    from goldpipeline.adapters.fake_publisher import FakePublisherClient
    from goldpipeline.adapters.fake_reviewer import FakeReviewerClient
    from goldpipeline.adapters.fake_writer import FakeWriterClient

    return TrackedClients(
        writer=writer if writer is not None else FakeWriterClient(),
        reviewer=reviewer if reviewer is not None else FakeReviewerClient(),
        finalizer=finalizer if finalizer is not None else FakeFinalizerClient(),
        publisher=publisher if publisher is not None else FakePublisherClient(),
        target_chat=target_chat,
    )


@pytest.fixture
def tracked_clients() -> TrackedClients:
    """The default set of offline clients for an orchestrated run."""
    return make_tracked_clients()


def run_orchestrated(
    runs_dir: Path,
    tmp_path: Path,
    clients: TrackedClients,
    *,
    mode: Any = None,
    article: str | None = None,
    analysis: dict[str, Any] | None = None,
    market: dict[str, Any] | None = None,
) -> Any:
    """Drive a fresh Run end to end through the orchestrator.

    Sources are written to disk and read back through the real file adapters, so
    the test exercises the same path the CLI does rather than a shortcut.
    """
    from goldpipeline.adapters.file_source import (
        JsonFileAnalysisSource,
        JsonFileMarketDataSource,
    )
    from goldpipeline.services.orchestrator import DEFAULT_MODE, run_pipeline
    from goldpipeline.storage.run_store import RunStore

    if article is not None:
        clients.writer = _writer_returning(article)

    sources = tmp_path / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    analysis_path = write_json(sources / "telegram_input.json", analysis or make_analysis_payload())
    market_path = write_json(sources / "ohlc.json", market or make_market_payload())

    return run_pipeline(
        analysis_source=JsonFileAnalysisSource(analysis_path),
        market_source=JsonFileMarketDataSource(market_path),
        store=RunStore(runs_dir),
        clients=clients.as_pipeline_clients(),
        mode=mode or DEFAULT_MODE,
        expected_symbol="XAUUSD",
        now=PIPELINE_NOW,
    )


def _writer_returning(article: str) -> Any:
    """A fake writer that always drafts *article*, with no claims to check."""
    from goldpipeline.adapters.fake_writer import FakeWriterClient
    from goldpipeline.schemas.writer import WriterModelOutput, WriterStatus

    def build(request: Any) -> Any:
        return WriterModelOutput(
            run_id=request.run_id,
            status=WriterStatus.COMPLETED,
            title="Nhận định vàng",
            article=article,
            source_claims=[],
            warnings=[],
        )

    return FakeWriterClient(output_factory=build)


# --------------------------------------------------------------------------
# Round 8 helpers
# --------------------------------------------------------------------------

INGEST_NOW = datetime(2026, 8, 28, 3, 0, tzinfo=UTC)
"""Deterministic 'now' for ingestion.

The same instant the generated candle series is aligned to, so an ingested Run
is byte-identical on every machine and never trips the recency checks.
"""

SAMPLE_EVENT_ID = "gold-20260828-0300-a1b2c3"
"""A realistically shaped producer id: date, time, and enough randomness."""


def make_event_payload(
    *,
    event_id: str = SAMPLE_EVENT_ID,
    raw_text: str = VIETNAMESE_TEXT,
    source: str = "gold_analysis_bot",
    created_at: str = "2026-08-28T03:00:00Z",
    include_optional: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    """Build one inbox payload, the way the producing bot would write it."""
    payload: dict[str, Any] = {
        "schema_version": "1",
        "source": source,
        "event_id": event_id,
        "created_at": created_at,
        "raw_text": raw_text,
    }
    if include_optional:
        payload.update(
            {
                "message_date": "2026-08-28T02:58:00Z",
                "chat_id": -1002145890733,
                "message_id": 48217,
                "author": "gold_desk_vn",
                "metadata": {"strategy": "intraday"},
            }
        )
    payload.update(extra)
    return payload


@pytest.fixture
def inbox(tmp_path: Path) -> Any:
    """An isolated inbox, laid out and empty."""
    from goldpipeline.services.inbox import Inbox

    box = Inbox(tmp_path / "inbox")
    box.ensure_layout()
    return box


def make_mt5_source(*, module: Any = None, now: datetime | None = None, **overrides: Any) -> Any:
    """A MetaTrader source wired to the offline module.

    Never touches a terminal, and never imports the vendor package: the fake is
    injected, which is the whole reason the adapter takes one.
    """
    from goldpipeline.adapters.fake_mt5 import FakeMt5Module, make_rates
    from goldpipeline.adapters.mt5_market import MetaTrader5MarketDataSource
    from goldpipeline.config import MarketDataSettings

    moment = now or INGEST_NOW
    settings = MarketDataSettings(**overrides)
    return MetaTrader5MarketDataSource(
        settings,
        module=module
        if module is not None
        else FakeMt5Module(
            known_symbols=(settings.provider_symbol,),
            rates=make_rates(now=moment, timeframe=settings.timeframe),
        ),
        now=moment,
    )


def make_ingestion_context(inbox: Any, runs_dir: Path, *, market_source: Any = None) -> Any:
    """An ingestion context whose market data comes from the offline module."""
    from goldpipeline.services.ingestion import IngestionContext
    from goldpipeline.storage.run_store import RunStore

    return IngestionContext(
        inbox=inbox,
        store=RunStore(runs_dir),
        market_source=market_source if market_source is not None else make_mt5_source(),
        expected_symbol="XAUUSD",
    )


def submit_event(inbox: Any, payload: dict[str, Any]) -> Path:
    """Put *payload* in the inbox the way a producer would."""
    submitted: Path = inbox.submit(payload, event_id=payload["event_id"])
    return submitted


def read_ledger(inbox: Any, event_id: str) -> Any:
    """Read one ledger entry back through its schema."""
    from goldpipeline.services.inbox import Ledger

    return Ledger(inbox.directory("index")).read(event_id)


# --------------------------------------------------------------------------
# Round 9 helpers
# --------------------------------------------------------------------------

AUTOMATION_NOW = INGEST_NOW
"""Logical 'now' for a tick.

The same instant the candle series and the sample event are aligned to, so a
tick is deterministic and neither the market nor the analysis is stale.
"""


class FrozenElapsed:
    """A stand-in for the tick's elapsed-time clock.

    Separate from logical time on purpose: the deadline is about how long the
    worker has actually been running, which a pinned wall clock cannot answer.
    Advance it to make a tick believe it is out of time.
    """

    def __init__(self, seconds: float = 0.0, *, step: float = 0.0) -> None:
        self.seconds = seconds
        self.step = step

    def __call__(self) -> float:
        """Return the current reading, then advance by *step*.

        The step is what lets a test put a tick past its deadline *during* the
        tick rather than before it - the worker reads this clock once at the
        start and again before each new item.
        """
        current = self.seconds
        self.seconds += self.step
        return current

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


def make_worker_context(
    inbox: Any,
    runs_dir: Path,
    automation_dir: Path,
    *,
    clients: Any = None,
    market_source: Any = None,
    settings: Any = None,
    publisher_target: str | None = None,
    elapsed: Any = None,
    **overrides: Any,
) -> Any:
    """Assemble a worker context whose every dependency is offline."""
    from goldpipeline.config import AutomationSettings
    from goldpipeline.services.automation import WorkerContext
    from goldpipeline.services.automation_state import AutomationStore
    from goldpipeline.storage.run_store import RunStore

    tracked = clients if clients is not None else make_tracked_clients()
    resolved = settings if settings is not None else AutomationSettings(**overrides)
    return WorkerContext(
        inbox=inbox,
        store=RunStore(runs_dir),
        automation=AutomationStore(automation_dir),
        settings=resolved,
        market_source=market_source if market_source is not None else make_mt5_source(),
        clients=tracked.as_pipeline_clients() if isinstance(tracked, TrackedClients) else tracked,
        expected_symbol="XAUUSD",
        publisher_target=publisher_target,
        elapsed=elapsed if elapsed is not None else FrozenElapsed(),
    )


@pytest.fixture
def automation_dir(tmp_path: Path) -> Path:
    """An isolated automation state root."""
    target = tmp_path / "automation"
    target.mkdir()
    return target


def event_aged(minutes: int, *, event_id: str | None = None, **extra: Any) -> dict[str, Any]:
    """An event payload created *minutes* before :data:`AUTOMATION_NOW`."""
    created = AUTOMATION_NOW - timedelta(minutes=minutes)
    return make_event_payload(
        event_id=event_id or SAMPLE_EVENT_ID,
        created_at=created.isoformat().replace("+00:00", "Z"),
        **extra,
    )


def age_run(runs_dir: Path, run_id: str, created_at: datetime) -> None:
    """Rewrite a Run's creation timestamp.

    ``create_run`` stamps the manifest from the wall clock, not from its ``now``
    argument - that one is the data-recency clock. Tests about how *old* a Run
    is therefore have to say so explicitly rather than rely on a pinned logical
    clock the manifest never saw.
    """
    from goldpipeline.storage.run_store import RunStore

    run = RunStore(runs_dir).open(run_id)
    manifest = run.load_manifest()
    manifest.created_at = created_at
    run.save_manifest(manifest)


# --------------------------------------------------------------------------
# Round 9.1 helpers
# --------------------------------------------------------------------------


class _FakeBackend:
    """Stands in for a keyring backend, named to match a real one."""

    def __init__(self, name: str) -> None:
        module, _, cls = name.rpartition(".")
        self.__class__ = type(cls, (_FakeBackend,), {"__module__": module})
        self._name = name


class FakeKeyringModule:
    """Offline stand-in for the keyring module.

    Every credential test in this repository runs against this, so no test reads,
    writes or lists an entry in a real credential manager. It models the two
    behaviours that matter: a backend with a *name* the adapter must judge, and
    failures raised rather than returned.
    """

    def __init__(
        self,
        stored: dict[tuple[str, str], str] | None = None,
        *,
        backend_name: str = "keyring.backends.Windows.WinVaultKeyring",
        backend_error: Exception | None = None,
        read_error: Exception | None = None,
        write_error: Exception | None = None,
        delete_error: Exception | None = None,
    ) -> None:
        self.stored = dict(stored or {})
        self.backend_name = backend_name
        self.backend_error = backend_error
        self.read_error = read_error
        self.write_error = write_error
        self.delete_error = delete_error
        self.reads: list[tuple[str, str]] = []
        """Every lookup, so a test can assert on entry naming."""

    def get_keyring(self) -> Any:
        if self.backend_error is not None:
            raise self.backend_error
        return _FakeBackend(self.backend_name)

    def get_password(self, service: str, entry: str) -> str | None:
        self.reads.append((service, entry))
        if self.read_error is not None:
            raise self.read_error
        return self.stored.get((service, entry))

    def set_password(self, service: str, entry: str, value: str) -> None:
        if self.write_error is not None:
            raise self.write_error
        self.stored[(service, entry)] = value

    def delete_password(self, service: str, entry: str) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        if (service, entry) not in self.stored:
            raise _PasswordDeleteError("not found")
        del self.stored[(service, entry)]


class _PasswordDeleteError(Exception):
    """Shaped like keyring's own missing-entry error, which the adapter reads."""


_PasswordDeleteError.__name__ = "PasswordDeleteError"


def fail_backend_module(**kwargs: Any) -> FakeKeyringModule:
    """keyring's own no-op backend: reports success, stores nothing."""
    return FakeKeyringModule(backend_name="keyring.backends.fail.Keyring", **kwargs)


def plaintext_backend_module(**kwargs: Any) -> FakeKeyringModule:
    """A backend that would keep credentials in a readable file."""
    return FakeKeyringModule(backend_name="keyrings.alt.file.PlaintextKeyring", **kwargs)


@pytest.fixture(autouse=True)
def no_real_credential_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test away from the machine's real credential manager.

    ``keyring`` is installed in this environment, so without this the CLI's
    composite provider would consult the operating system's actual vault on any
    miss. That read is harmless, but a suite that touches real credential
    storage is no longer a suite anyone can run anywhere - so the backend is
    reported unavailable by default and tests that want a store substitute their
    own offline one.
    """
    from goldpipeline import cli
    from goldpipeline.adapters.windows_credentials import BackendReport

    monkeypatch.setattr(
        cli,
        "inspect_backend",
        lambda *args, **kwargs: BackendReport(
            available=False,
            secure=False,
            backend="none",
            detail="tests never reach the machine's credential store",
        ),
    )


# --------------------------------------------------------------------------
# Round 9.2 helpers
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_real_machine_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep every test away from the machine's real config file and task.

    The CLI resolves both by default, so without this a test run would read the
    operator's actual persisted configuration and query the real Task Scheduler.
    Neither is destructive, but a suite whose results depend on one machine's
    state is no longer a suite anyone can trust.

    Both the interactive store and the strict production loader are redirected
    to the *same* temporary file, because they describe one machine. Redirecting
    only one would let a test write settings through ``config-set`` and then
    watch the worker read the operator's real ``%LOCALAPPDATA%`` instead - which
    is precisely the split-brain this round exists to make impossible.
    """
    from goldpipeline import cli
    from goldpipeline.adapters import production_config
    from goldpipeline.adapters.config_store import RuntimeConfigStore
    from goldpipeline.adapters.task_scheduler import FakeTaskScheduler

    path = tmp_path / "appdata" / "config.json"
    store = RuntimeConfigStore(path)
    scheduler = FakeTaskScheduler()
    monkeypatch.setattr(cli, "_config_store", lambda: store)
    monkeypatch.setattr(cli, "_task_scheduler", lambda: scheduler)
    monkeypatch.setattr(
        production_config,
        "production_config_path",
        lambda env=None, *, windows=None: path,
    )


COMPLETE_PRODUCTION_CONFIG: dict[str, str] = {
    "TELEGRAM_TARGET_CHAT_ID": "@testchannel",
    "GOLDPIPELINE_MT5_SYMBOL": "XAUUSD",
    "GOLDPIPELINE_CANONICAL_SYMBOL": "XAUUSD",
    "GOLDPIPELINE_OHLC_TIMEFRAME": "M15",
    "GOLDPIPELINE_OHLC_BARS": "20",
    "GOLDPIPELINE_MAX_DATA_AGE_MINUTES": "90",
    "GOLDPIPELINE_MAX_ANALYSIS_EVENT_AGE_MINUTES": "60",
    "GOLDPIPELINE_DEFER_RETRY_MINUTES": "5",
    "GOLDPIPELINE_AUTOMATION_MAX_EVENTS_PER_TICK": "3",
    "GOLDPIPELINE_AUTOMATION_MAX_TICK_MINUTES": "10",
    "GOLDPIPELINE_AUTOMATION_ENABLED": "false",
    "GOLDPIPELINE_AUTOPUBLISH_ENABLED": "false",
    "GOLDPIPELINE_AUTOPUBLISH_ALLOWED_TARGET": "@testchannel",
    "GOLDPIPELINE_AUTOPUBLISH_MAX_RUN_AGE_MINUTES": "30",
}
"""Every approved key, explicitly. Off by default, like the real machine.

Spelled out rather than generated from :class:`ConfigKey` so that a key added
without a deliberate decision about its production value breaks these tests
instead of silently inheriting one.
"""


def write_production_config(path: Path, **overrides: str) -> Path:
    """Write a complete production configuration, with optional changes.

    Passing ``None`` for a key removes it, which is how the incompleteness tests
    express "this one setting disappeared".
    """
    values = dict(COMPLETE_PRODUCTION_CONFIG)
    for key, value in overrides.items():
        if value is None:
            values.pop(key, None)
        else:
            values[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": "1.0.0", "values": values}, indent=2),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def production_config(tmp_path: Path) -> Any:
    """A complete production configuration at the guarded path.

    Returns the writer so a test can re-write it with different settings; the
    path is the one :func:`no_real_machine_state` already redirected both
    readers to.
    """
    path = tmp_path / "appdata" / "config.json"

    def write(**overrides: str) -> Path:
        return write_production_config(path, **overrides)

    write()
    return write


@pytest.fixture
def config_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """An isolated persisted-configuration file, wired into the CLI."""
    from goldpipeline import cli
    from goldpipeline.adapters.config_store import RuntimeConfigStore

    store = RuntimeConfigStore(tmp_path / "appdata" / "config.json")
    monkeypatch.setattr(cli, "_config_store", lambda: store)
    return store


@pytest.fixture
def task_scheduler(monkeypatch: pytest.MonkeyPatch) -> Any:
    """An offline Task Scheduler, wired into the CLI."""
    from goldpipeline import cli
    from goldpipeline.adapters.task_scheduler import FakeTaskScheduler

    scheduler = FakeTaskScheduler()
    monkeypatch.setattr(cli, "_task_scheduler", lambda: scheduler)
    return scheduler
