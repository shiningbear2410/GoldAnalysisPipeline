"""A news digest, driven through the real production dispatch.

Round 6.5c.2. Everything before this round could build a digest out of its
parts; nothing ran one. These tests enter through ``run_pipeline`` and
``resume_pipeline`` - the same functions the scheduled worker calls - with a
scratch Run store and offline clients, and they assert on the Run that comes
out rather than on the services that made it.

That distinction is the whole point of the file. The digest services were
already covered by their own suites and every one of those passed while the
orchestrator was still refusing to dispatch a digest at all. A library that
works and a pipeline that runs it are different claims, and only the second one
publishes anything.

**Two invariants get most of the attention here**, because they are the two a
resumed Run can silently break:

* an artifact that exists is loaded, never rebuilt - proven by handing the
  resumed Run a market source, a writer and a reviewer that raise on contact;
* a content ``NEEDS_REVISION`` digest stops, rather than being handed to a
  finalizer built to repair a different product.

Offline throughout. No provider, no Telegram, no clock-dependent assertion.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    PIPELINE_NOW,
    make_analysis_payload,
    make_market_payload,
    make_tracked_clients,
    write_json,
)
from test_human_style_review import reviewer_returning

from goldpipeline.adapters.base import LoadedSource
from goldpipeline.adapters.fake_digest_writer import FakeDigestWriterClient
from goldpipeline.adapters.fake_reviewer import FakeReviewerClient
from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.digest import DigestWindow
from goldpipeline.schemas.manifest import RunStatus
from goldpipeline.schemas.market import MarketDataInput, OHLCBar
from goldpipeline.schemas.producer import PRODUCER_SOURCE
from goldpipeline.schemas.review import ReviewStatus
from goldpipeline.services.digest_context import NEWS_WINDOW_METADATA_KEY
from goldpipeline.services.digest_snapshot import DIGEST_CONTEXT_FILENAME
from goldpipeline.services.digest_stage import DIGEST_EDITORIAL_FILENAME
from goldpipeline.services.finalizer import DIGEST_FINAL_EDITORIAL_FILENAME
from goldpipeline.services.news_collector import curate
from goldpipeline.services.producer_brief import news_item_id, render_brief
from goldpipeline.services.writer import DRAFT_FILENAME, WRITER_FILENAME
from goldpipeline.storage.run_store import RunStore
from tests.test_producer import make_collection, make_item, request_for

LOOKBACK = timedelta(hours=6)
FINAL_FILENAME = "claude_final.md"

USD_TEXT = "Chi so USD giam 0.21 phan tram trong phien."
ETF_TEXT = "SPDR Gold Trust mua rong 9.98 tan trong phien gan nhat."
FED_TEXT = "Fed Williams noi loi suat tang gan day khong phan anh ky vong lam phat cao hon."

ITEMS = (
    make_item(channel="tintucvnws", message_id=41, text=USD_TEXT, minutes_ago=200),
    make_item(channel="pcnewsfx", message_id=42, text=ETF_TEXT, minutes_ago=150),
    make_item(channel="tintucvnws", message_id=43, text=FED_TEXT, minutes_ago=90),
)
ITEM_IDS = (
    news_item_id("tintucvnws", 41),
    news_item_id("pcnewsfx", 42),
    news_item_id("tintucvnws", 43),
)


# --------------------------------------------------------------------------
# a digest event, as the inbox would present it
# --------------------------------------------------------------------------


def digest_brief() -> str:
    collection = make_collection(items=list(ITEMS))
    return render_brief(request_for(), collection, curate(collection))


class DigestAnalysisSource:
    """The real file adapter, declaring what a digest event declares.

    Wrapping rather than faking: the payload still goes through
    ``JsonFileAnalysisSource``, so the source file the Run stores is the one
    production would store. What is added is the article type and the
    provenance the inbox adapter records - the two things a digest Run needs
    and an analysis Run does not.
    """

    def __init__(self, path: Path, *, created_at: datetime) -> None:
        from goldpipeline.adapters.file_source import JsonFileAnalysisSource

        self._inner = JsonFileAnalysisSource(path)
        self._created_at = created_at

    def load(self) -> Any:
        loaded = self._inner.load()
        return replace(
            loaded,
            article_type=ArticleType.NEWS_DIGEST,
            provenance={
                **loaded.provenance,
                "kind": "inbox",
                "event_created_at": self._created_at.isoformat().replace("+00:00", "Z"),
                "article_type": str(ArticleType.NEWS_DIGEST),
            },
        )


class OfflineM5Source:
    """A deterministic M5 series covering the digest window. Counted."""

    def __init__(self, window: DigestWindow) -> None:
        self.window = window
        self.loads = 0

    def load(self) -> LoadedSource[MarketDataInput]:
        self.loads += 1
        origin = self.window.start - timedelta(minutes=10)
        bars = []
        for index in range(74):
            close = Decimal("4000") - Decimal(index) * Decimal("0.4")
            bars.append(
                OHLCBar(
                    timestamp=origin + timedelta(minutes=5 * index),
                    open=close + Decimal("0.4"),
                    high=close + Decimal("1.5"),
                    low=close - Decimal("1.5"),
                    close=close,
                )
            )
        return LoadedSource(
            model=MarketDataInput(
                symbol="XAUUSD",
                provider="tradingview",
                timeframe=Timeframe.M5,
                bars=bars,
                requested_at=self.window.end,
            ),
            raw_payload={},
            origin="offline-m5",
            provenance={"kind": "offline"},
        )


class ForbiddenMarket:
    """Contact is the failure. A resumed digest must not fetch."""

    def load(self) -> Any:
        raise AssertionError("a resumed digest Run fetched market data")


class ForbiddenWriter:
    """A digest writer that must never be called again."""

    provider = "forbidden"
    model = "forbidden"

    def generate(self, request: Any) -> Any:
        raise AssertionError("a resumed digest Run called the writer again")


class ForbiddenReviewer:
    """A reviewer that must never be called again."""

    provider = "forbidden"
    model = "forbidden"

    def review(self, request: Any) -> Any:
        raise AssertionError("a resumed digest Run called the reviewer again")


@pytest.fixture
def digest_sources(tmp_path: Path) -> tuple[Path, Path, datetime]:
    """The two source files a digest Run is created from."""
    created_at = PIPELINE_NOW
    payload = make_analysis_payload()
    payload["source"] = PRODUCER_SOURCE
    payload["raw_text"] = digest_brief()
    payload["metadata"] = {NEWS_WINDOW_METADATA_KEY: int(LOOKBACK.total_seconds())}

    sources = tmp_path / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    return (
        write_json(sources / "telegram_input.json", payload),
        write_json(sources / "ohlc.json", make_market_payload()),
        created_at,
    )


def _severity(name: str) -> Any:
    from goldpipeline.schemas.review import Severity

    return Severity(name)


def style_needing_revision() -> Any:
    """A style assessment whose derived verdict is NEEDS_REVISION."""
    from goldpipeline.schemas.review import (
        HumanStyleAssessment,
        HumanStyleCategory,
        HumanStyleFinding,
        StyleSeverity,
    )

    return HumanStyleAssessment(
        style_score=48,
        summary="Doc nhu ban tin may.",
        findings=[
            HumanStyleFinding(
                finding_id="style-high",
                category=HumanStyleCategory.DATA_DUMP,
                severity=StyleSeverity.HIGH,
                problem="Phan can can lap lai moi so lieu da co o tren.",
                repair_instruction="Rut gon can can thanh mot nhan dinh, bo cac so lieu.",
            )
        ],
    )


def content_issue(severity: Any) -> Any:
    """One issue, so a non-PASS verdict is one the reviewer justified.

    `validate_response` refuses a NEEDS_REVISION or REJECT carrying no issues,
    which is correct and means a test cannot ask for one without saying why.
    """
    from goldpipeline.schemas.review import Evidence, IssueCategory, ReviewIssue

    return ReviewIssue(
        issue_id="digest-1",
        category=IssueCategory.UNSUPPORTED_CLAIM,
        severity=severity,
        message="Mot cau trong ban tin khong truy duoc ve item nao.",
        claim="ETF mua vi lo lam phat tang.",
        evidence=Evidence(
            source_path="collected_news",
            expected="mot item noi ve dong co nay",
            actual="khong item nao noi ve dong co",
        ),
    )


def market_factory(seen: list[OfflineM5Source]) -> Any:
    def build(window: DigestWindow) -> OfflineM5Source:
        source = OfflineM5Source(window)
        seen.append(source)
        return source

    return build


def run_digest(
    tmp_path: Path,
    digest_sources: tuple[Path, Path, datetime],
    *,
    clients: Any = None,
    mode: Any = None,
) -> tuple[Any, Any, list[OfflineM5Source]]:
    """Drive a NEWS_DIGEST Run through the production dispatch seam."""
    from goldpipeline.adapters.file_source import JsonFileMarketDataSource
    from goldpipeline.services.orchestrator import DEFAULT_MODE, run_pipeline

    analysis_path, market_path, created_at = digest_sources
    seen: list[OfflineM5Source] = []
    tracked = clients or make_tracked_clients(digest_market_factory=market_factory(seen))
    if tracked.digest_market_factory is None:
        tracked.digest_market_factory = market_factory(seen)

    outcome = run_pipeline(
        analysis_source=DigestAnalysisSource(analysis_path, created_at=created_at),
        market_source=JsonFileMarketDataSource(market_path),
        store=RunStore(tmp_path / "runs"),
        clients=tracked.as_pipeline_clients(),
        mode=mode or DEFAULT_MODE,
        expected_symbol="XAUUSD",
        now=PIPELINE_NOW,
    )
    return outcome, tracked, seen


def manifest_of(tmp_path: Path, run_id: str) -> Any:
    return RunStore(tmp_path / "runs").open(run_id).load_manifest()


# --------------------------------------------------------------------------
# §27: production dispatch, end to end
# --------------------------------------------------------------------------


def test_a_clean_digest_reaches_ready_to_publish(
    tmp_path: Path, digest_sources: tuple[Path, Path, datetime]
) -> None:
    """The claim this round exists to make, through the seam the worker uses."""
    outcome, tracked, seen = run_digest(tmp_path, digest_sources)

    assert outcome.error is None, outcome.error
    manifest = manifest_of(tmp_path, outcome.run_id)
    assert manifest.status is RunStatus.READY_TO_PUBLISH
    assert manifest.provenance is not None
    assert manifest.provenance.article_type is ArticleType.NEWS_DIGEST
    assert len(seen) == 1 and seen[0].loads == 1, "exactly one market fetch"


def test_the_digest_writer_is_dispatched_and_the_analysis_writer_is_not(
    tmp_path: Path, digest_sources: tuple[Path, Path, datetime]
) -> None:
    """§3: the fail-closed guard was replaced, not deleted."""
    _, tracked, _ = run_digest(tmp_path, digest_sources)

    assert tracked.digest_writer.calls, "the digest writer produced the draft"
    assert not tracked.writer.calls, "the analysis writer was never called"
    assert "writer" not in tracked.built, "and its client was never even built"


def test_every_digest_artifact_is_written_once(
    tmp_path: Path, digest_sources: tuple[Path, Path, datetime]
) -> None:
    outcome, _, _ = run_digest(tmp_path, digest_sources)
    run = RunStore(tmp_path / "runs").open(outcome.run_id)

    for name in (
        DIGEST_CONTEXT_FILENAME,
        DIGEST_EDITORIAL_FILENAME,
        DRAFT_FILENAME,
        WRITER_FILENAME,
        FINAL_FILENAME,
    ):
        assert run.has_artifact(name), name

    manifest = run.load_manifest()
    names = [ref.name for ref in manifest.artifact_files]
    assert len(names) == len(set(names)), f"an artifact was recorded twice: {names}"


def test_the_snapshot_holds_the_window_the_event_was_accepted_under(
    tmp_path: Path, digest_sources: tuple[Path, Path, datetime]
) -> None:
    """§18: Run facts are snapshot facts. Nothing recomputes them from a clock."""
    from goldpipeline.services.digest_snapshot import load_digest_snapshot

    outcome, _, _ = run_digest(tmp_path, digest_sources)
    run = RunStore(tmp_path / "runs").open(outcome.run_id)
    facts = load_digest_snapshot(run, run.load_manifest())

    _, _, created_at = digest_sources
    assert facts.window.end == created_at
    assert facts.window.lookback_seconds == int(LOOKBACK.total_seconds())
    assert facts.timeframe is Timeframe.M5
    assert facts.market is not None
    assert facts.market.provider == "tradingview"
    assert set(facts.news_item_ids) == set(ITEM_IDS)


def test_the_published_shell_is_deterministic_not_authored(
    tmp_path: Path, digest_sources: tuple[Path, Path, datetime]
) -> None:
    """§10: the title, window, price block and disclaimer come from code."""
    from goldpipeline.schemas.article_contract import contract_for
    from goldpipeline.services.digest_snapshot import load_digest_snapshot

    outcome, _, _ = run_digest(tmp_path, digest_sources)
    run = RunStore(tmp_path / "runs").open(outcome.run_id)
    facts = load_digest_snapshot(run, run.load_manifest())
    article = run.read_artifact_bytes(FINAL_FILENAME).decode("utf-8")

    for line in facts.deterministic_lines:
        assert line in article, line
    disclaimer = contract_for(ArticleType.NEWS_DIGEST).disclaimer.text
    assert article.count(disclaimer) == 1


def test_the_final_article_is_the_reviewed_draft_byte_for_byte(
    tmp_path: Path, digest_sources: tuple[Path, Path, datetime]
) -> None:
    """§16: no model edits a digest in this round, so nothing may differ."""
    outcome, tracked, _ = run_digest(tmp_path, digest_sources)
    run = RunStore(tmp_path / "runs").open(outcome.run_id)

    assert run.read_artifact_bytes(FINAL_FILENAME) == run.read_artifact_bytes(DRAFT_FILENAME)
    assert not tracked.finalizer.calls, "no finalizer model call"


def test_the_reviewer_was_given_the_digest_user_turn(
    tmp_path: Path, digest_sources: tuple[Path, Path, datetime]
) -> None:
    """§12: never the analysis builder, which says the pipeline collected none."""
    _, tracked, _ = run_digest(tmp_path, digest_sources)

    assert tracked.reviewer.calls, "the reviewer ran"
    user = tracked.reviewer.calls[-1].prompt.user
    assert "**This Run collected news.**" in user
    assert '"available_news": []' not in user
    assert str(ArticleType.NEWS_DIGEST) in user


# --------------------------------------------------------------------------
# §20-24: resume, and what must not be touched again
# --------------------------------------------------------------------------


def resume(tmp_path: Path, run_id: str, clients: Any, *, mode: Any = None) -> Any:
    from goldpipeline.services.orchestrator import DEFAULT_MODE, resume_pipeline

    return resume_pipeline(
        run_id=run_id,
        store=RunStore(tmp_path / "runs"),
        clients=clients.as_pipeline_clients(),
        mode=mode or DEFAULT_MODE,
        now=PIPELINE_NOW,
    )


def test_b_a_snapshot_survives_a_failed_writer_and_is_not_rebuilt(
    tmp_path: Path, digest_sources: tuple[Path, Path, datetime]
) -> None:
    """§21. Stage B: the facts are captured, the writer then fails.

    The snapshot is committed before the provider is called, so this is a real
    state a Run can be left in - and the state where a refetch would be most
    tempting and most wrong. The resumed Run is given a market source that
    raises, and must not care.
    """
    from goldpipeline.domain.errors import WriterProviderError

    broken = make_tracked_clients(
        digest_writer=FakeDigestWriterClient(raises=WriterProviderError("socket closed")),
        digest_market_factory=None,
    )
    outcome, _, seen = run_digest(tmp_path, digest_sources, clients=broken)

    run = RunStore(tmp_path / "runs").open(outcome.run_id)
    assert run.load_manifest().status is RunStatus.NORMALIZED
    assert run.has_artifact(DIGEST_CONTEXT_FILENAME), "the facts were captured first"
    assert not run.has_artifact(DRAFT_FILENAME)
    assert seen[0].loads == 1
    captured = run.read_artifact_bytes(DIGEST_CONTEXT_FILENAME)

    hostile = make_tracked_clients(digest_market_factory=lambda window: ForbiddenMarket())
    resumed = resume(tmp_path, outcome.run_id, hostile)

    assert resumed.error is None, resumed.error
    assert run.load_manifest().status is RunStatus.READY_TO_PUBLISH
    assert run.read_artifact_bytes(DIGEST_CONTEXT_FILENAME) == captured, "not rebuilt"


def test_cd_a_resumed_run_with_a_draft_never_calls_the_writer_again(
    tmp_path: Path, digest_sources: tuple[Path, Path, datetime]
) -> None:
    """§22. Stages C and D: the editorial artifact and the rendered draft exist."""
    from goldpipeline.domain.errors import ReviewProviderError

    broken = make_tracked_clients(
        reviewer=FakeReviewerClient(raises=ReviewProviderError("socket closed")),
        digest_market_factory=None,
    )
    outcome, _, _ = run_digest(tmp_path, digest_sources, clients=broken)

    run = RunStore(tmp_path / "runs").open(outcome.run_id)
    assert run.load_manifest().status is RunStatus.DRAFTED
    assert run.has_artifact(DIGEST_EDITORIAL_FILENAME)
    before = run.read_artifact_bytes(DRAFT_FILENAME)

    hostile = make_tracked_clients(
        digest_writer=ForbiddenWriter(),
        digest_market_factory=lambda window: ForbiddenMarket(),
    )
    resumed = resume(tmp_path, outcome.run_id, hostile)

    assert resumed.error is None, resumed.error
    assert run.load_manifest().status is RunStatus.READY_TO_PUBLISH
    assert run.read_artifact_bytes(DRAFT_FILENAME) == before


def test_ef_a_resumed_run_past_review_calls_neither_writer_nor_reviewer(
    tmp_path: Path, digest_sources: tuple[Path, Path, datetime]
) -> None:
    """§23. Stages E and F: the review and the final article are on disk."""
    from goldpipeline.schemas.orchestration import PipelineMode

    outcome, _, _ = run_digest(tmp_path, digest_sources, mode=PipelineMode.GENERATE_ONLY)
    run = RunStore(tmp_path / "runs").open(outcome.run_id)
    assert run.load_manifest().status is RunStatus.FINALIZED
    assert run.has_artifact("gpt_review.json")
    before = run.read_artifact_bytes(FINAL_FILENAME)

    hostile = make_tracked_clients(
        digest_writer=ForbiddenWriter(),
        reviewer=ForbiddenReviewer(),
        digest_market_factory=lambda window: ForbiddenMarket(),
    )
    resumed = resume(tmp_path, outcome.run_id, hostile)

    assert resumed.error is None, resumed.error
    assert run.load_manifest().status is RunStatus.READY_TO_PUBLISH
    assert run.read_artifact_bytes(FINAL_FILENAME) == before


def test_g_a_run_already_ready_to_publish_repeats_nothing(
    tmp_path: Path, digest_sources: tuple[Path, Path, datetime]
) -> None:
    outcome, _, _ = run_digest(tmp_path, digest_sources)
    run = RunStore(tmp_path / "runs").open(outcome.run_id)
    before = run.read_artifact_bytes(FINAL_FILENAME)

    hostile = make_tracked_clients(
        digest_writer=ForbiddenWriter(),
        reviewer=ForbiddenReviewer(),
        digest_market_factory=lambda window: ForbiddenMarket(),
    )
    resumed = resume(tmp_path, outcome.run_id, hostile)

    assert resumed.error is None, resumed.error
    assert run.read_artifact_bytes(FINAL_FILENAME) == before


def test_style_needs_revision_now_buys_exactly_one_digest_repair(
    tmp_path: Path, digest_sources: tuple[Path, Path, datetime]
) -> None:
    """Round 6.5c.3 inverted this. The same review, a different outcome.

    Until this round a HIGH style finding on a digest bought nothing, because
    the only rewriter available was built for a different product. Now it buys
    one repair, through a finalizer that returns editorial content and cannot
    reach the deterministic shell.
    """
    clients = make_tracked_clients(
        reviewer=reviewer_returning(style=style_needing_revision()),
        digest_market_factory=None,
    )
    outcome, tracked, _ = run_digest(tmp_path, digest_sources, clients=clients)

    assert outcome.error is None, outcome.error
    run = RunStore(tmp_path / "runs").open(outcome.run_id)
    assert run.load_manifest().status is RunStatus.READY_TO_PUBLISH

    assert len(tracked.digest_finalizer.calls) == 1, "exactly one repair"
    assert not tracked.finalizer.calls, "and never the analysis finalizer"
    assert run.has_artifact(DIGEST_FINAL_EDITORIAL_FILENAME)


def test_a_repaired_digest_keeps_its_deterministic_shell(
    tmp_path: Path, digest_sources: tuple[Path, Path, datetime]
) -> None:
    """Not preserved by checking - preserved because the repair cannot reach it."""
    from goldpipeline.schemas.article_contract import contract_for
    from goldpipeline.services.digest_snapshot import load_digest_snapshot

    clients = make_tracked_clients(
        reviewer=reviewer_returning(style=style_needing_revision()),
        digest_market_factory=None,
    )
    outcome, _, _ = run_digest(tmp_path, digest_sources, clients=clients)

    run = RunStore(tmp_path / "runs").open(outcome.run_id)
    facts = load_digest_snapshot(run, run.load_manifest())
    draft = run.read_artifact_bytes(DRAFT_FILENAME).decode("utf-8")
    final = run.read_artifact_bytes(FINAL_FILENAME).decode("utf-8")

    assert final != draft, "the repair did change something"
    for line in facts.deterministic_lines:
        assert line in final, line
    disclaimer = contract_for(ArticleType.NEWS_DIGEST).disclaimer.text
    assert final.count(disclaimer) == 1


def test_item_timestamps_survive_a_repair_because_they_are_source_owned(
    tmp_path: Path, digest_sources: tuple[Path, Path, datetime]
) -> None:
    """The repair has no timestamp field, and the renderer reads the item."""
    import re

    clients = make_tracked_clients(
        reviewer=reviewer_returning(style=style_needing_revision()),
        digest_market_factory=None,
    )
    outcome, _, _ = run_digest(tmp_path, digest_sources, clients=clients)

    run = RunStore(tmp_path / "runs").open(outcome.run_id)
    draft = run.read_artifact_bytes(DRAFT_FILENAME).decode("utf-8")
    final = run.read_artifact_bytes(FINAL_FILENAME).decode("utf-8")

    stamps = re.compile(r"\d{2}:\d{2} \u2014")
    assert stamps.findall(final) == stamps.findall(draft)


def test_the_writers_own_editorial_survives_the_repair_byte_for_byte(
    tmp_path: Path, digest_sources: tuple[Path, Path, datetime]
) -> None:
    """What the writer wrote and what the repair changed stay two artifacts."""
    from goldpipeline.services.integrity import verify_artifact

    clients = make_tracked_clients(
        reviewer=reviewer_returning(style=style_needing_revision()),
        digest_market_factory=None,
    )
    outcome, _, _ = run_digest(tmp_path, digest_sources, clients=clients)

    run = RunStore(tmp_path / "runs").open(outcome.run_id)
    manifest = run.load_manifest()
    recorded = {ref.name: ref.sha256 for ref in manifest.artifact_files}

    verify_artifact(run, manifest, DIGEST_EDITORIAL_FILENAME)
    assert DIGEST_EDITORIAL_FILENAME in recorded
    assert DIGEST_FINAL_EDITORIAL_FILENAME in recorded
    assert recorded[DIGEST_EDITORIAL_FILENAME] != recorded[DIGEST_FINAL_EDITORIAL_FILENAME]


def test_content_needs_revision_now_repairs_in_one_call(
    tmp_path: Path, digest_sources: tuple[Path, Path, datetime]
) -> None:
    """Round 6.5c.2 stopped here with REVISION_UNAVAILABLE. It no longer does."""
    clients = make_tracked_clients(
        reviewer=reviewer_returning(
            status=ReviewStatus.NEEDS_REVISION,
            score=61,
            issues=[content_issue(_severity("HIGH"))],
            instructions=["Xoa cau khong co nguon."],
        ),
        digest_market_factory=None,
    )
    outcome, tracked, _ = run_digest(tmp_path, digest_sources, clients=clients)

    assert outcome.error is None, outcome.error
    run = RunStore(tmp_path / "runs").open(outcome.run_id)
    assert run.load_manifest().status is RunStatus.READY_TO_PUBLISH
    assert len(tracked.digest_finalizer.calls) == 1
    assert not tracked.finalizer.calls


def test_content_and_style_are_repaired_in_the_same_single_call(
    tmp_path: Path, digest_sources: tuple[Path, Path, datetime]
) -> None:
    """One prompt, one output, one render - never two sequential repairs."""
    from goldpipeline.schemas.finalizer import FinalizerResult

    clients = make_tracked_clients(
        reviewer=reviewer_returning(
            status=ReviewStatus.NEEDS_REVISION,
            score=58,
            issues=[content_issue(_severity("HIGH"))],
            style=style_needing_revision(),
            instructions=["Xoa cau khong co nguon."],
        ),
        digest_market_factory=None,
    )
    outcome, tracked, _ = run_digest(tmp_path, digest_sources, clients=clients)

    assert outcome.error is None, outcome.error
    assert len(tracked.digest_finalizer.calls) == 1, "both axes, one call"

    run = RunStore(tmp_path / "runs").open(outcome.run_id)
    result = FinalizerResult.model_validate_json(
        run.read_artifact_bytes("claude_finalizer.json").decode("utf-8")
    )
    assert result.issue_resolutions, "the content issue was answered"
    assert result.style_resolutions, "and so was the style finding"
    assert result.review_status is ReviewStatus.NEEDS_REVISION, "the judgement is untouched"


def test_a_review_status_is_never_rewritten_by_a_repair(
    tmp_path: Path, digest_sources: tuple[Path, Path, datetime]
) -> None:
    """A style repair on a content PASS leaves the recorded verdict a PASS."""
    from goldpipeline.schemas.review import ReviewResult

    clients = make_tracked_clients(
        reviewer=reviewer_returning(style=style_needing_revision()),
        digest_market_factory=None,
    )
    outcome, _, _ = run_digest(tmp_path, digest_sources, clients=clients)

    run = RunStore(tmp_path / "runs").open(outcome.run_id)
    review = ReviewResult.model_validate_json(
        run.read_artifact_bytes("gpt_review.json").decode("utf-8")
    )
    assert review.status is ReviewStatus.PASS


# --------------------------------------------------------------------------
# §28-29: what must not have changed
# --------------------------------------------------------------------------


def test_28_an_analysis_run_is_untouched_by_any_of_this(tmp_path: Path, runs_dir: Path) -> None:
    """The same orchestrator, the same result it always produced."""
    from conftest import run_orchestrated

    seen: list[OfflineM5Source] = []
    clients = make_tracked_clients(digest_market_factory=market_factory(seen))
    outcome = run_orchestrated(runs_dir, tmp_path, clients)

    assert outcome.error is None, outcome.error
    run = RunStore(runs_dir).open(outcome.run_id)
    manifest = run.load_manifest()

    assert manifest.status is RunStatus.READY_TO_PUBLISH
    assert manifest.provenance is not None
    assert manifest.provenance.article_type is ArticleType.ANALYSIS
    assert clients.writer.calls, "the analysis writer wrote it"
    assert not clients.digest_writer.calls, "the digest writer did not"
    assert not seen, "no digest market source was ever built"
    assert not run.has_artifact(DIGEST_CONTEXT_FILENAME)
    assert not run.has_artifact(DIGEST_EDITORIAL_FILENAME)


def test_29_trade_plan_is_still_refused_by_production_dispatch() -> None:
    from goldpipeline.domain.errors import ArticleTypeNotReadyError
    from goldpipeline.services.article_runtime import is_dispatchable, runtime_for

    assert is_dispatchable(ArticleType.ANALYSIS) is True
    assert is_dispatchable(ArticleType.NEWS_DIGEST) is True
    assert is_dispatchable(ArticleType.TRADE_PLAN) is False

    with pytest.raises(ArticleTypeNotReadyError):
        runtime_for(ArticleType.TRADE_PLAN)


def test_news_digest_style_activation_is_on_and_trade_plan_is_not() -> None:
    from goldpipeline.services.review_action import STYLE_ACTIVE_TYPES

    assert ArticleType.ANALYSIS in STYLE_ACTIVE_TYPES
    assert ArticleType.NEWS_DIGEST in STYLE_ACTIVE_TYPES
    assert ArticleType.TRADE_PLAN not in STYLE_ACTIVE_TYPES


def test_the_venue_symbol_is_recorded_from_the_adapter_not_invented(
    tmp_path: Path, digest_sources: tuple[Path, Path, datetime]
) -> None:
    """Provenance answers "which instrument, at which venue?" - both parts.

    The first live dispatch recorded `XAUUSD` here, which is the canonical
    symbol rather than the venue's own. Not wrong, but not what the field
    documents either, and provenance that is nearly right is the kind that
    stops being checked. The adapter already puts its venue symbol on the
    loaded source; this reads it from there rather than from an argument, so
    any provider gets the same treatment.
    """
    from goldpipeline.services.digest_snapshot import load_digest_snapshot

    class VenueAwareM5(OfflineM5Source):
        def load(self) -> Any:
            loaded = super().load()
            return replace(
                loaded, provenance={**loaded.provenance, "provider_symbol": "OANDA:XAUUSD"}
            )

    clients = make_tracked_clients(digest_market_factory=lambda window: VenueAwareM5(window))
    outcome, _, _ = run_digest(tmp_path, digest_sources, clients=clients)

    run = RunStore(tmp_path / "runs").open(outcome.run_id)
    facts = load_digest_snapshot(run, run.load_manifest())

    assert facts.market is not None
    assert facts.market.provider_symbol == "OANDA:XAUUSD"
    assert facts.market.provider == "tradingview"


def test_a_source_without_a_venue_symbol_falls_back_to_the_canonical_one(
    tmp_path: Path, digest_sources: tuple[Path, Path, datetime]
) -> None:
    """No invented string, and no empty field either."""
    from goldpipeline.services.digest_snapshot import load_digest_snapshot

    outcome, _, _ = run_digest(tmp_path, digest_sources)
    run = RunStore(tmp_path / "runs").open(outcome.run_id)
    facts = load_digest_snapshot(run, run.load_manifest())

    assert facts.market is not None
    assert facts.market.provider_symbol == "XAUUSD"
