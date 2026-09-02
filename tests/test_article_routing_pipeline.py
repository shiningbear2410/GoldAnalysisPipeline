"""Routing through a real Run: ANALYSIS proceeds, the other two stop early.

``TrackedClients`` distinguishes two questions the fakes can answer: whether a
stage *ran*, and whether its client was even *constructed*. The second is the
stronger claim, and the one asserted here - an unimplemented article type must
not reach the point where a provider client is built, because that is the point
where a credential would be needed and a bill would start.

Offline throughout: no MT5, no model, no network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from conftest import make_normalized_run, make_tracked_clients

from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.orchestration import PipelineMode, PipelineStatus
from goldpipeline.storage.run_store import RunStore


def set_article_type(runs_dir: Path, run_id: str, kind: ArticleType) -> None:
    """Stamp a Run's recorded product mode, as ingestion would have."""
    store = RunStore(runs_dir)
    run = store.open(run_id)
    manifest = run.load_manifest()
    assert manifest.provenance is not None
    manifest.provenance.article_type = kind
    run.save_manifest(manifest)


def run_it(runs_dir: Path, run_id: str, clients: Any) -> Any:
    from goldpipeline.services.orchestrator import resume_pipeline

    return resume_pipeline(
        run_id=run_id,
        store=RunStore(runs_dir),
        clients=clients.as_pipeline_clients(),
        mode=PipelineMode.READY_FOR_PUBLISH,
    )


class TestAnalysisIsUnchanged:
    def test_analysis_runs_and_uses_the_production_prompt(
        self, runs_dir: Path, tmp_path: Path
    ) -> None:
        from goldpipeline.prompts import DEFAULT_WRITER_PROMPT

        normalized = make_normalized_run(runs_dir, tmp_path)
        clients = make_tracked_clients()

        result = run_it(runs_dir, normalized.run_id, clients)

        assert result.status is not PipelineStatus.FAILED
        assert "writer" in clients.built
        written = (
            RunStore(runs_dir).open(normalized.run_id).read_artifact_bytes("claude_writer.json")
        )
        assert DEFAULT_WRITER_PROMPT.encode() in written

    def test_a_run_with_no_recorded_type_still_runs(self, runs_dir: Path, tmp_path: Path) -> None:
        """Historical Runs carry no article type; they are ANALYSIS."""
        normalized = make_normalized_run(runs_dir, tmp_path)
        store = RunStore(runs_dir)
        run = store.open(normalized.run_id)
        manifest = run.load_manifest()
        manifest.provenance = None
        run.save_manifest(manifest)

        clients = make_tracked_clients()
        result = run_it(runs_dir, normalized.run_id, clients)

        assert result.status is not PipelineStatus.FAILED
        assert "writer" in clients.built


@pytest.mark.parametrize("kind", [ArticleType.TRADE_PLAN, ArticleType.NEWS_DIGEST])
class TestUnimplementedTypesFailClosed:
    def test_the_run_fails_with_an_explicit_reason(
        self, runs_dir: Path, tmp_path: Path, kind: ArticleType
    ) -> None:
        normalized = make_normalized_run(runs_dir, tmp_path)
        set_article_type(runs_dir, normalized.run_id, kind)

        result = run_it(runs_dir, normalized.run_id, make_tracked_clients())

        assert result.status is PipelineStatus.FAILED
        assert result.error is not None
        assert result.error.code == "ARTICLE_TYPE_NOT_READY"

    def test_no_provider_client_is_ever_built(
        self, runs_dir: Path, tmp_path: Path, kind: ArticleType
    ) -> None:
        """The strong claim: not merely 'did not run' but 'was never constructed'."""
        normalized = make_normalized_run(runs_dir, tmp_path)
        set_article_type(runs_dir, normalized.run_id, kind)

        clients = make_tracked_clients()
        run_it(runs_dir, normalized.run_id, clients)

        assert clients.built == []

    def test_no_stage_runs(self, runs_dir: Path, tmp_path: Path, kind: ArticleType) -> None:
        normalized = make_normalized_run(runs_dir, tmp_path)
        set_article_type(runs_dir, normalized.run_id, kind)

        clients = make_tracked_clients()
        run_it(runs_dir, normalized.run_id, clients)

        for fake in (clients.writer, clients.reviewer, clients.finalizer, clients.publisher):
            assert not getattr(fake, "calls", [])

    def test_no_article_artifacts_are_written(
        self, runs_dir: Path, tmp_path: Path, kind: ArticleType
    ) -> None:
        """Nothing that looks like a produced article may exist."""
        normalized = make_normalized_run(runs_dir, tmp_path)
        set_article_type(runs_dir, normalized.run_id, kind)
        run_it(runs_dir, normalized.run_id, make_tracked_clients())

        directory = Path(runs_dir) / normalized.run_id
        for name in (
            "claude_writer.json",
            "claude_draft.md",
            "gpt_review.json",
            "claude_final.md",
            "publish_decision.json",
            "publish_intent.json",
            "review_delivery_intent.json",
        ):
            assert not (directory / name).exists(), f"{name} should not exist"

    def test_it_does_not_become_ready_to_publish(
        self, runs_dir: Path, tmp_path: Path, kind: ArticleType
    ) -> None:
        from goldpipeline.schemas.manifest import RunStatus

        normalized = make_normalized_run(runs_dir, tmp_path)
        set_article_type(runs_dir, normalized.run_id, kind)
        run_it(runs_dir, normalized.run_id, make_tracked_clients())

        manifest = RunStore(runs_dir).open(normalized.run_id).load_manifest()
        assert manifest.status is not RunStatus.READY_TO_PUBLISH
        assert manifest.status is not RunStatus.PUBLISHED

    def test_retrying_does_not_eventually_let_it_through(
        self, runs_dir: Path, tmp_path: Path, kind: ArticleType
    ) -> None:
        """Deterministic refusal: the tenth attempt fails like the first."""
        normalized = make_normalized_run(runs_dir, tmp_path)
        set_article_type(runs_dir, normalized.run_id, kind)

        for _ in range(3):
            clients = make_tracked_clients()
            result = run_it(runs_dir, normalized.run_id, clients)
            assert result.status is PipelineStatus.FAILED
            assert clients.built == []


class TestRetryClassification:
    @pytest.mark.parametrize("kind", [ArticleType.TRADE_PLAN, ArticleType.NEWS_DIGEST])
    def test_not_ready_is_permanent_not_transient(self, kind: ArticleType) -> None:
        """A deploy fixes this, not a backoff. Retrying it every minute is waste."""
        from goldpipeline.domain.errors import ArticleTypeNotReadyError
        from goldpipeline.schemas.automation import RetryClass
        from goldpipeline.services.automation import classify

        assert classify(ArticleTypeNotReadyError("x")) is RetryClass.PERMANENT
