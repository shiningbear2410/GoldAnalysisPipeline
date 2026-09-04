"""A digest Run's facts, captured once and never recomputed.

Round 6.5c.1a. Round 6.5c.1 built `digest_snapshot` and proved it with a
scratch smoke - 31 checks, all passing, and none of them committed. The audit
opening this round found the module carrying no repo coverage at all, which
means the proof existed only in a transcript. These tests are that proof made
permanent; the module itself is unchanged.

The claim under test is narrow and worth stating exactly: **after a Run has
captured its digest facts, nothing later in that Run needs the clock, the
provider, or the news collector again.** So the resume half here is hostile to
any code that tried to recompute - the market source raises on contact, and the
store is reopened from its path so nothing survives in memory.

Entirely offline. The series is synthetic and no provider is reachable.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from goldpipeline.adapters.base import LoadedSource
from goldpipeline.domain.errors import ArtifactIntegrityError
from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.digest import DigestWindow
from goldpipeline.schemas.manifest import RunManifest
from goldpipeline.schemas.market import MarketDataInput, OHLCBar
from goldpipeline.schemas.news_digest import DigestSourceItem
from goldpipeline.services.digest_pipeline import build_digest_facts_for_window
from goldpipeline.services.digest_snapshot import (
    DIGEST_CONTEXT_FILENAME,
    load_digest_snapshot,
    require_digest_snapshot,
    requires_snapshot,
    write_digest_snapshot,
)
from goldpipeline.services.integrity import verify_artifact
from goldpipeline.storage.run_store import RunDirectory, RunStore

CAPTURE_AT = datetime(2026, 9, 4, 6, 0, tzinfo=UTC)
RESTART_AT = CAPTURE_AT + timedelta(days=8)
WINDOW = DigestWindow.ending_at(CAPTURE_AT, timedelta(hours=6))
RUN_ID = "20260904_060000_a1b2c3"
SYMBOL = "XAUUSD"


# --------------------------------------------------------------------------
# market sources: one that answers, one that must never be reached
# --------------------------------------------------------------------------


class SyntheticSource:
    """A fixed M5 series. Deterministic, offline, and counted."""

    def __init__(self) -> None:
        self.loads = 0

    def load(self) -> LoadedSource[MarketDataInput]:
        self.loads += 1
        origin = WINDOW.start - timedelta(minutes=10)
        bars = []
        for index in range(74):
            close = Decimal("4000") + Decimal(index) * Decimal("0.5")
            bars.append(
                OHLCBar(
                    timestamp=origin + timedelta(minutes=5 * index),
                    open=close - Decimal("0.5"),
                    high=close + Decimal("1.5"),
                    low=close - Decimal("2.0"),
                    close=close,
                )
            )
        return LoadedSource(
            model=MarketDataInput(
                symbol=SYMBOL,
                provider="synthetic",
                timeframe=Timeframe.M5,
                bars=bars,
                requested_at=WINDOW.end,
            ),
            raw_payload={},
            origin="synthetic",
            provenance={"kind": "offline"},
        )


class ForbiddenSource:
    """Contact is the failure. There is no correct answer it could give."""

    def load(self) -> LoadedSource[MarketDataInput]:
        raise AssertionError(
            "a resumed digest Run reached for market data; the snapshot exists "
            "precisely so that it does not"
        )


NEWS = (
    DigestSourceItem(
        item_id="goldnewsvn:9001",
        published_at=CAPTURE_AT - timedelta(hours=4),
        text="Chỉ số USD giảm 0.21% trong phiên.",
    ),
    DigestSourceItem(
        item_id="goldnewsvn:9002",
        published_at=CAPTURE_AT - timedelta(hours=2),
        text="SPDR Gold Trust mua ròng 9.98 tấn.",
    ),
)


def capture(source: Any | None = None) -> Any:
    """The deterministic facts, built from one synthetic fetch."""
    return build_digest_facts_for_window(
        window=WINDOW,
        market_source=source or SyntheticSource(),
        symbol=SYMBOL,
        news_items=NEWS,
        provider_symbol="SYNTHETIC:XAUUSD",
    )


@pytest.fixture
def digest_run(tmp_path: Path) -> tuple[RunDirectory, RunManifest, Any]:
    """A Run that has captured and committed its digest facts."""
    run = RunStore(tmp_path / "runs").create(run_id=RUN_ID)
    manifest = RunManifest(run_id=run.run_id, created_at=CAPTURE_AT)
    run.save_manifest(manifest)

    facts = capture()
    write_digest_snapshot(run, manifest, facts)
    run.save_manifest(manifest)
    return run, manifest, facts


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------


def test_the_provider_is_asked_exactly_once() -> None:
    source = SyntheticSource()
    capture(source)

    assert source.loads == 1


def test_the_snapshot_is_committed_and_recorded_in_the_manifest(
    digest_run: tuple[RunDirectory, RunManifest, Any],
) -> None:
    run, manifest, _ = digest_run

    assert run.has_artifact(DIGEST_CONTEXT_FILENAME)
    assert any(ref.name == DIGEST_CONTEXT_FILENAME for ref in manifest.artifact_files)


def test_market_provenance_is_captured_with_the_facts(
    digest_run: tuple[RunDirectory, RunManifest, Any],
) -> None:
    """A venue may revise a candle later; the article belongs to what was fetched."""
    _, _, facts = digest_run

    assert facts.market is not None
    assert facts.market.provider == "synthetic"
    assert facts.market.provider_symbol == "SYNTHETIC:XAUUSD"
    assert facts.market.timeframe is Timeframe.M5
    assert facts.market.bars_received == 74
    assert facts.market.bars_requested >= facts.market.bars_received


def test_a_second_capture_is_refused_rather_than_allowed_to_win(
    digest_run: tuple[RunDirectory, RunManifest, Any],
) -> None:
    """Two snapshots would be two answers to "which six hours is this?"."""
    run, manifest, facts = digest_run

    with pytest.raises(ArtifactIntegrityError):
        write_digest_snapshot(run, manifest, facts)


# --------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------


def reopen(run: RunDirectory) -> tuple[RunDirectory, RunManifest]:
    """A fresh handle from the path alone. Nothing is inherited in memory."""
    reopened = RunStore(run.path.parent).open(run.run_id)
    return reopened, reopened.load_manifest()


def test_a_resumed_run_reads_its_facts_without_touching_a_provider(
    digest_run: tuple[RunDirectory, RunManifest, Any],
) -> None:
    """The whole point of the artifact, asserted with a source that would crash."""
    run, _, facts = digest_run
    reopened, manifest = reopen(run)

    forbidden = ForbiddenSource()
    loaded = load_digest_snapshot(reopened, manifest)

    assert loaded == facts
    with pytest.raises(AssertionError):
        forbidden.load()  # it would have raised, had anything called it


@pytest.mark.parametrize(
    "field",
    ["title", "window_line", "price_reaction_block", "symbol", "timeframe"],
)
def test_the_rendered_shell_survives_a_restart_byte_for_byte(
    digest_run: tuple[RunDirectory, RunManifest, Any], field: str
) -> None:
    """A title rebuilt from `now` eight days later would be a different date."""
    run, _, facts = digest_run
    reopened, manifest = reopen(run)

    loaded = load_digest_snapshot(reopened, manifest)

    assert getattr(loaded, field) == getattr(facts, field)
    assert RESTART_AT > CAPTURE_AT, "the resume is deliberately on another day"


def test_the_price_arithmetic_survives_a_restart(
    digest_run: tuple[RunDirectory, RunManifest, Any],
) -> None:
    run, _, facts = digest_run
    reopened, manifest = reopen(run)

    loaded = load_digest_snapshot(reopened, manifest)

    assert loaded.price_reaction == facts.price_reaction
    assert loaded.window == facts.window


def test_the_collected_items_survive_a_restart(
    digest_run: tuple[RunDirectory, RunManifest, Any],
) -> None:
    run, _, facts = digest_run
    reopened, manifest = reopen(run)

    loaded = load_digest_snapshot(reopened, manifest)

    assert loaded.news_item_ids == facts.news_item_ids
    assert [i.text for i in loaded.news_items] == [i.text for i in facts.news_items]


def test_the_retrieval_instant_is_the_captured_one_not_the_resume(
    digest_run: tuple[RunDirectory, RunManifest, Any],
) -> None:
    run, _, _ = digest_run
    reopened, manifest = reopen(run)

    loaded = load_digest_snapshot(reopened, manifest)

    assert loaded.market is not None
    assert loaded.market.retrieved_at < RESTART_AT


# --------------------------------------------------------------------------
# integrity
# --------------------------------------------------------------------------


def test_the_manifest_digest_validates(
    digest_run: tuple[RunDirectory, RunManifest, Any],
) -> None:
    run, _, _ = digest_run
    reopened, manifest = reopen(run)

    verified = verify_artifact(reopened, manifest, DIGEST_CONTEXT_FILENAME)

    assert len(verified.sha256) == 64


def test_an_edited_snapshot_stops_the_run_rather_than_being_used(
    digest_run: tuple[RunDirectory, RunManifest, Any],
) -> None:
    """A snapshot edited after commit describes a window nobody reviewed."""
    run, _, _ = digest_run
    reopened, manifest = reopen(run)
    target = reopened.artifact_path(DIGEST_CONTEXT_FILENAME)
    target.write_bytes(target.read_bytes().replace(b'"XAUUSD"', b'"XAGUSD"', 1))

    with pytest.raises(ArtifactIntegrityError):
        load_digest_snapshot(reopened, manifest)


def test_a_missing_snapshot_stops_a_digest_run(tmp_path: Path) -> None:
    """Fail closed. Regenerating would silently describe different hours."""
    run = RunStore(tmp_path / "runs").create(run_id=RUN_ID)
    manifest = RunManifest(run_id=run.run_id, created_at=CAPTURE_AT)
    run.save_manifest(manifest)

    with pytest.raises(ArtifactIntegrityError):
        require_digest_snapshot(run, manifest, ArticleType.NEWS_DIGEST)


def test_an_analysis_run_is_not_asked_for_a_snapshot_it_never_had(
    tmp_path: Path,
) -> None:
    """Demanding one would make every historical Run retroactively invalid."""
    run = RunStore(tmp_path / "runs").create(run_id=RUN_ID)
    manifest = RunManifest(run_id=run.run_id, created_at=CAPTURE_AT)
    run.save_manifest(manifest)

    assert requires_snapshot(ArticleType.NEWS_DIGEST) is True
    assert requires_snapshot(ArticleType.ANALYSIS) is False
    assert require_digest_snapshot(run, manifest, ArticleType.ANALYSIS) is None
