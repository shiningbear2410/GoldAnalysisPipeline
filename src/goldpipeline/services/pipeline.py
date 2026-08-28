"""Run orchestration - the one place that decides what a Run is.

Ordering, and why:

1. generate ``run_id`` and create the directory (fails fast if it exists);
2. write the manifest with status ``CREATED``;
3. **persist the raw source payloads verbatim**;
4. normalize and validate;
5. build the context;
6. write ``context.json``;
7. flip the manifest to ``NORMALIZED``.

Step 3 deliberately happens *before* validation. When a Run fails, the inputs
that caused the failure are the most valuable thing to keep - a Run directory
holding the offending ``ohlc.json`` and a manifest explaining exactly which
invariant it broke is far more useful for debugging than no directory at all.

The safety property that matters is preserved either way: a failed Run ends at
status ``FAILED`` and has **no** ``context.json``. Nothing downstream can
mistake it for usable input.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from goldpipeline.adapters.base import AnalysisSource, MarketDataSource
from goldpipeline.domain.errors import PipelineError
from goldpipeline.schemas.context import AnalysisContext
from goldpipeline.schemas.manifest import RunError, RunManifest, RunStatus
from goldpipeline.services.context_builder import build_context
from goldpipeline.services.normalizer import normalize_analysis, normalize_market_data
from goldpipeline.storage.run_store import RunDirectory, RunStore

logger = logging.getLogger(__name__)

ANALYSIS_SOURCE_FILENAME = "telegram_input.json"
MARKET_SOURCE_FILENAME = "ohlc.json"
CONTEXT_FILENAME = "context.json"


@dataclass(frozen=True)
class RunResult:
    """Outcome of a Run creation attempt."""

    run_id: str
    run_dir: Path
    status: RunStatus
    manifest: RunManifest
    context: AnalysisContext | None = None
    context_path: Path | None = None
    error: PipelineError | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the Run reached ``NORMALIZED`` with a context on disk."""
        return self.status is RunStatus.NORMALIZED and self.context is not None


def create_run(
    *,
    analysis_source: AnalysisSource,
    market_source: MarketDataSource,
    store: RunStore,
    expected_symbol: str | None = None,
    run_id: str | None = None,
    now: datetime | None = None,
) -> RunResult:
    """Execute the Round 1 pipeline and persist an immutable Run.

    Args:
        analysis_source: Where the raw human analysis comes from.
        market_source: Where the OHLC data comes from.
        store: Run storage root.
        expected_symbol: Instrument the caller expects; mismatches are fatal.
        run_id: Force a specific id. Mainly for tests; collides loudly.
        now: Injection point for tests; defaults to the current UTC time.

    Returns:
        A :class:`RunResult`. Validation failures are reported through
        ``result.error`` and ``status=FAILED`` rather than raised, so that the
        caller can still point a human at the Run directory. Unexpected
        exceptions (bugs, disk errors) propagate.
    """
    run = store.create(run_id=run_id)
    manifest = RunManifest(run_id=run.run_id)
    manifest.record_event("run.create", "OK", f"run directory created at {run.path}")
    run.save_manifest(manifest)
    logger.info("run=%s stage=run.create status=OK path=%s", run.run_id, run.path)

    try:
        context = _execute(
            run=run,
            manifest=manifest,
            analysis_source=analysis_source,
            market_source=market_source,
            expected_symbol=expected_symbol,
            now=now,
        )
    except PipelineError as exc:
        _mark_failed(run, manifest, exc)
        return RunResult(
            run_id=run.run_id,
            run_dir=run.path,
            status=RunStatus.FAILED,
            manifest=manifest,
            error=exc,
        )

    return RunResult(
        run_id=run.run_id,
        run_dir=run.path,
        status=RunStatus.NORMALIZED,
        manifest=manifest,
        context=context,
        context_path=run.artifact_path(CONTEXT_FILENAME),
    )


def _execute(
    *,
    run: RunDirectory,
    manifest: RunManifest,
    analysis_source: AnalysisSource,
    market_source: MarketDataSource,
    expected_symbol: str | None,
    now: datetime | None,
) -> AnalysisContext:
    """Ingest, normalize and persist. Raises :class:`PipelineError` on bad data."""
    # --- ingest -----------------------------------------------------------
    loaded_analysis = analysis_source.load()
    loaded_market = market_source.load()
    manifest.record_event(
        "ingest",
        "OK",
        f"analysis={loaded_analysis.origin} market={loaded_market.origin}",
    )
    logger.info("run=%s stage=ingest status=OK", run.run_id)

    # --- capture sources verbatim, before any judgement about them --------
    run.write_source(ANALYSIS_SOURCE_FILENAME, loaded_analysis.raw_payload, manifest)
    run.write_source(MARKET_SOURCE_FILENAME, loaded_market.raw_payload, manifest)
    manifest.record_event("source.capture", "OK", "raw payloads stored")
    run.save_manifest(manifest)

    # --- normalize --------------------------------------------------------
    market = normalize_market_data(loaded_market.model, expected_symbol=expected_symbol, now=now)
    analysis = normalize_analysis(loaded_analysis.model)
    warning_count = len(market.warnings) + len(analysis.warnings)
    manifest.record_event(
        "normalize", "OK", f"{market.snapshot.bar_count} bars, {warning_count} warnings"
    )
    logger.info(
        "run=%s stage=normalize status=OK bars=%d warnings=%d",
        run.run_id,
        market.snapshot.bar_count,
        warning_count,
    )

    # --- build + persist context -----------------------------------------
    context = build_context(run_id=run.run_id, market=market, analysis=analysis, generated_at=now)
    run.write_artifact(CONTEXT_FILENAME, context, manifest)

    manifest.status = RunStatus.NORMALIZED
    manifest.record_event("context.build", "OK", f"quality={context.data_quality.status}")
    run.save_manifest(manifest)
    logger.info(
        "run=%s stage=context.build status=OK quality=%s",
        run.run_id,
        context.data_quality.status,
    )
    return context


def _mark_failed(run: RunDirectory, manifest: RunManifest, exc: PipelineError) -> None:
    """Record a fatal error on the manifest and persist it."""
    manifest.status = RunStatus.FAILED
    manifest.error = RunError(code=exc.code, message=exc.message, details=exc.details)
    manifest.record_event("run.fail", exc.code, exc.message)
    run.save_manifest(manifest)
    logger.error("run=%s stage=run.fail status=%s message=%s", run.run_id, exc.code, exc.message)


def validate_sources(
    *,
    analysis_source: AnalysisSource,
    market_source: MarketDataSource,
    expected_symbol: str | None = None,
    now: datetime | None = None,
) -> AnalysisContext:
    """Run ingestion and normalization without touching the filesystem.

    Backs the CLI's ``--dry-run``: it answers "would this data produce a valid
    Run?" without leaving a Run behind.

    Raises:
        PipelineError: On any fatal inconsistency.
    """
    loaded_analysis = analysis_source.load()
    loaded_market = market_source.load()
    market = normalize_market_data(loaded_market.model, expected_symbol=expected_symbol, now=now)
    analysis = normalize_analysis(loaded_analysis.model)
    return build_context(run_id="dry-run", market=market, analysis=analysis, generated_at=now)


__all__ = [
    "ANALYSIS_SOURCE_FILENAME",
    "CONTEXT_FILENAME",
    "MARKET_SOURCE_FILENAME",
    "RunResult",
    "create_run",
    "validate_sources",
]
