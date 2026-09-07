"""The write and review stages, as a news digest performs them.

Round 6.5c.2. Everything before this round could build a digest; nothing
dispatched one. The orchestrator stopped a ``NEWS_DIGEST`` Run at the write
stage on purpose, because the alternative was handing a digest prompt to a
writer that parses articles. This module is what that guard was waiting for.

**It is a stage implementation, not a second pipeline.** The Run statuses, the
artifact names the rest of the pipeline reads, the write-once semantics, the
manifest events and the digests are all the existing ones. What differs is only
what happens between "the Run is NORMALIZED" and "``claude_draft.md`` exists":

* an analysis asks a model for an article and checks it afterwards;
* a digest computes its facts first, asks a model for *editorial content only*,
  and assembles the article around the answer in code.

**Three artifacts, and each earns its place.** ``claude_draft.md`` is the
rendered digest, because review, finalize and the gate all read that name.
``claude_writer.json`` is a real :class:`WriterResult` - it describes the *write*
(which model, which prompt, which draft digest, which news claims), which is
true of a digest write as much as an analysis one, and every downstream stage
depends on it existing. ``digest_editorial.json`` is the model's structured
answer, kept because it is the only thing the rendered article cannot be
reduced back to: a resumed Run rebuilding the reviewer prompt needs the impact
markers and the claims, not prose.

**The snapshot is built here, once.** Immediately before the writer request and
never after: the window, the M5 candles and the collected items are captured
into ``digest_context.json`` and become the Run's authority. A resumed Run loads
it and touches no provider - which is why the market source is optional on this
call, and why passing one that raises is a legitimate test.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from goldpipeline.adapters.base import MarketDataSource
from goldpipeline.adapters.digest_writer_client import DigestRequest, DigestWriterClient
from goldpipeline.domain.errors import RunNotReadyError, WriterResponseError
from goldpipeline.prompts import DEFAULT_DIGEST_WRITER_PROMPT, DEFAULT_REVIEWER_PROMPT
from goldpipeline.schemas.context import AnalysisContext
from goldpipeline.schemas.digest import DigestWindow
from goldpipeline.schemas.manifest import RunManifest
from goldpipeline.schemas.news_digest import DigestEditorial, DigestSourceItem
from goldpipeline.schemas.review import PrecheckFinding, ReviewerPrompt
from goldpipeline.schemas.writer import WriterUsage
from goldpipeline.services.digest_context import (
    NEWS_WINDOW_METADATA_KEY,
    DigestFacts,
)
from goldpipeline.services.digest_pipeline import (
    build_digest_facts_for_window,
)
from goldpipeline.services.digest_review import build_digest_reviewer_prompt
from goldpipeline.services.digest_snapshot import (
    DIGEST_CONTEXT_FILENAME,
    load_digest_snapshot,
    write_digest_snapshot,
)
from goldpipeline.services.digest_writer import (
    assemble_digest,
    build_digest_prompt,
    digest_precheck,
    validate_editorial,
)
from goldpipeline.services.news_provenance import authenticate
from goldpipeline.services.precheck import PrecheckReport
from goldpipeline.storage.run_store import RunDirectory

logger = logging.getLogger(__name__)

DigestMarketSource = Callable[[DigestWindow], MarketDataSource]
"""Builds the M5 source for one window.

Takes the window because the bar count is derived from it, and is a factory
rather than an instance so that a resumed Run - which must never fetch -
does not construct one at all. Nothing is opened, no credential is read, and
there is no socket to leave unused.
"""

DIGEST_EDITORIAL_FILENAME = "digest_editorial.json"
"""The model's structured answer, kept because prose cannot be reduced back to it."""

EVENT_CREATED_AT_KEY = "event_created_at"
"""Where the manifest records the instant the producer accepted the window."""

ANALYSIS_SOURCE_FILENAME = "telegram_input.json"


# --------------------------------------------------------------------------
# recovering the Run's own digest inputs
# --------------------------------------------------------------------------


def digest_window_of(run: RunDirectory, manifest: RunManifest) -> DigestWindow:
    """The window this Run was accepted under, from its immutable record.

    Two immutable places, and both are needed. The *end* is the producer's own
    observation instant, recorded in manifest provenance when the inbox adapter
    handed the event over; the *length* is the lookback the collector used,
    carried on the stored source file. Neither is recomputed from the clock -
    that is the whole reason a digest has a snapshot at all.

    Raises:
        RunNotReadyError: Either is missing. A digest Run without a window is
            not a digest Run, and choosing a default would describe a span
            nobody collected news for.
    """
    provenance = manifest.provenance.analysis if manifest.provenance else {}
    raw_end = provenance.get(EVENT_CREATED_AT_KEY)
    if not isinstance(raw_end, str):
        raise RunNotReadyError(
            f"run {run.run_id} records no {EVENT_CREATED_AT_KEY}; its digest window "
            "cannot be recovered, and the clock is not an acceptable substitute",
            run_id=run.run_id,
        )
    try:
        end = datetime.fromisoformat(raw_end.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError as exc:
        raise RunNotReadyError(
            f"run {run.run_id}: {EVENT_CREATED_AT_KEY} is not a timestamp",
            run_id=run.run_id,
        ) from exc

    source = _source_payload(run)
    raw_lookback = (source.get("metadata") or {}).get(NEWS_WINDOW_METADATA_KEY)
    if raw_lookback is None:
        raise RunNotReadyError(
            f"run {run.run_id} carries no {NEWS_WINDOW_METADATA_KEY}; a digest needs "
            "the lookback its news was collected over",
            run_id=run.run_id,
        )
    try:
        seconds = int(raw_lookback)
    except (TypeError, ValueError) as exc:
        raise RunNotReadyError(
            f"run {run.run_id}: {NEWS_WINDOW_METADATA_KEY} is not a whole number of seconds",
            run_id=run.run_id,
        ) from exc

    try:
        return DigestWindow.ending_at(end, timedelta(seconds=seconds))
    except ValueError as exc:
        # The bounds are the news collector's own, and a window outside them
        # describes a span the collector would never have gathered for.
        raise RunNotReadyError(
            f"run {run.run_id}: {NEWS_WINDOW_METADATA_KEY}={seconds} is not a window "
            "this pipeline collects news over",
            run_id=run.run_id,
        ) from exc


def _source_payload(run: RunDirectory) -> dict[str, Any]:
    """The stored analysis source, as the producer wrote it."""
    if not run.has_artifact(ANALYSIS_SOURCE_FILENAME):
        raise RunNotReadyError(
            f"run {run.run_id} has no {ANALYSIS_SOURCE_FILENAME}", run_id=run.run_id
        )
    loaded: dict[str, Any] = json.loads(
        run.read_artifact_bytes(ANALYSIS_SOURCE_FILENAME).decode("utf-8")
    )
    return loaded


def digest_sources_of(run: RunDirectory, context: AnalysisContext) -> tuple[DigestSourceItem, ...]:
    """The closed list of items this Run may cite.

    Recovered from the producer brief the Run was created with, through the
    same authentication an analysis uses. A Run whose input is not an authentic
    brief has no items, and a digest with nothing to report is refused rather
    than written from whatever text happened to arrive.

    Raises:
        RunNotReadyError: No authentic brief, or a brief with no items.
    """
    state, parsed = authenticate(context)
    if parsed is None:
        raise RunNotReadyError(
            f"run {run.run_id} has no authentic producer brief ({state}); a digest "
            "has nothing to select from without one",
            run_id=run.run_id,
        )
    if not parsed.items:
        raise RunNotReadyError(
            f"run {run.run_id}: the producer brief declares no items", run_id=run.run_id
        )

    return tuple(
        DigestSourceItem(
            item_id=item.item_id,
            published_at=_published(item.published, run_id=run.run_id, item_id=item.item_id),
            text=item.text,
        )
        for item in parsed.items
    )


def _published(stamp: str, *, run_id: str, item_id: str) -> datetime:
    """One item's publication instant, as the brief recorded it."""
    if stamp in {"", "-"}:
        raise RunNotReadyError(
            f"run {run_id}: item {item_id} carries no publication time",
            run_id=run_id,
        )
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError as exc:
        raise RunNotReadyError(
            f"run {run_id}: item {item_id} has an unreadable publication time",
            run_id=run_id,
        ) from exc


# --------------------------------------------------------------------------
# the snapshot, built once
# --------------------------------------------------------------------------


def ensure_digest_snapshot(
    run: RunDirectory,
    manifest: RunManifest,
    context: AnalysisContext,
    *,
    market_source: DigestMarketSource | None,
) -> DigestFacts:
    """The Run's digest facts: loaded if captured, captured if not.

    The order matters and is the invariant this round exists to establish. An
    existing snapshot is *loaded and verified* - never rebuilt, never
    supplemented, never refreshed. Only a Run that has none reaches a provider,
    and it does so exactly once.

    Raises:
        RunNotReadyError: No snapshot and no market source to build one with.
            That combination is a resumed Run whose capture never completed, and
            fetching now would describe a different six hours than the one the
            event was accepted for.
        ArtifactIntegrityError: A snapshot that does not match its digest.
    """
    if run.has_artifact(DIGEST_CONTEXT_FILENAME):
        facts = load_digest_snapshot(run, manifest)
        logger.info("run=%s stage=digest.snapshot status=REUSED", run.run_id)
        manifest.record_event("digest.snapshot", "REUSED", "loaded from the Run")
        return facts

    if market_source is None:
        raise RunNotReadyError(
            f"run {run.run_id} has no {DIGEST_CONTEXT_FILENAME} and no market source to "
            "capture one with; a digest's facts are captured when the Run is written, "
            "not recovered later from a moved market",
            run_id=run.run_id,
        )

    window = digest_window_of(run, manifest)
    sources = digest_sources_of(run, context)
    # The window first, the source second. A provider-neutral source for a
    # digest is *derived* from the span being described - how many M5 bars a
    # six-hour window needs is not the same as a seven-day one - so there is no
    # source to build until the window is known, and none is built at all on a
    # Run that already has its snapshot.
    facts = build_digest_facts_for_window(
        window=window,
        market_source=market_source(window),
        symbol=context.market.symbol,
        news_items=sources,
    )
    write_digest_snapshot(run, manifest, facts)
    manifest.record_event(
        "digest.snapshot",
        "OK",
        f"{len(sources)} items, {facts.price_reaction.market_activity}",
    )
    logger.info(
        "run=%s stage=digest.snapshot status=OK items=%d activity=%s",
        run.run_id,
        len(sources),
        facts.price_reaction.market_activity,
    )
    return facts


# --------------------------------------------------------------------------
# write
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DigestDraft:
    """A digest, written and rendered, before anything is persisted.

    Deliberately not artifacts. The filenames a Run uses belong to
    :mod:`~goldpipeline.services.writer`, which is the module every other stage
    already reads them from; this one produces the *content* and lets that one
    write it under the pipeline's own conventions. The alternative had the two
    importing each other.
    """

    article: str
    editorial: DigestEditorial
    facts: DigestFacts
    title: str
    model: str
    provider: str
    selection_id: str | None
    prompt_version: str
    usage: WriterUsage


def build_digest_draft(
    *,
    run: RunDirectory,
    manifest: RunManifest,
    context: AnalysisContext,
    client: DigestWriterClient,
    market_source: DigestMarketSource | None,
    prompt_version: str = DEFAULT_DIGEST_WRITER_PROMPT,
    max_tokens: int = 8000,
) -> DigestDraft:
    """Capture the facts, ask for editorial content, render the digest.

    Raises:
        WriterResponseError: The editorial answer failed a deterministic check.
            No draft is produced and the writer is **not** called again: a
            response that cites an uncollected item, or quantifies something
            nothing supports, is wrong in a way a retry cannot fix.
        DigestMarketDataError: The snapshot could not be captured.
    """
    facts = ensure_digest_snapshot(run, manifest, context, market_source=market_source)

    prompt = build_digest_prompt(facts, run_id=run.run_id, prompt_version=prompt_version)
    response = client.generate(
        DigestRequest(prompt=prompt, run_id=run.run_id, max_tokens=max_tokens)
    )

    editorial = response.output
    validate_editorial(editorial, facts, run_id=run.run_id)
    article = assemble_digest(editorial, facts)

    logger.info(
        "run=%s stage=digest.write status=OK items=%d chars=%d claims=%d",
        run.run_id,
        len(editorial.items),
        len(article),
        len(editorial.news_claims),
    )
    return DigestDraft(
        article=article,
        editorial=editorial,
        facts=facts,
        title=facts.title,
        model=response.model,
        provider=response.provider,
        selection_id=response.selection_id,
        prompt_version=prompt.prompt_version,
        usage=response.usage,
    )


# --------------------------------------------------------------------------
# review preparation
# --------------------------------------------------------------------------


def load_digest_editorial(run: RunDirectory) -> DigestEditorial:
    """The structured answer this Run's draft was rendered from."""
    if not run.has_artifact(DIGEST_EDITORIAL_FILENAME):
        raise RunNotReadyError(
            f"run {run.run_id} has no {DIGEST_EDITORIAL_FILENAME}; its digest cannot be "
            "reviewed without the editorial content the draft was built from",
            run_id=run.run_id,
        )
    return DigestEditorial.model_validate_json(
        run.read_artifact_bytes(DIGEST_EDITORIAL_FILENAME).decode("utf-8")
    )


def prepare_digest_review(
    *,
    run: RunDirectory,
    manifest: RunManifest,
    article: str,
    prompt_version: str = DEFAULT_REVIEWER_PROMPT,
) -> tuple[ReviewerPrompt, PrecheckReport]:
    """The reviewer prompt and deterministic report for a digest.

    The analysis prechecks are not run, and that is a correctness decision
    rather than a shortcut. They resolve claims against ``context.json`` and
    scan for price-like numbers the M15 context does not vouch for - and a
    digest's prices come from its *own* M5 snapshot, so every figure in its
    price block would be reported as unsupported. The digest's equivalent is
    :func:`~goldpipeline.services.digest_writer.digest_precheck`, re-run here
    against the persisted artifacts so a draft edited after it was written is
    caught rather than reviewed.
    """
    facts = load_digest_snapshot(run, manifest)
    editorial = load_digest_editorial(run)
    report = digest_precheck(editorial, facts, article=article)

    findings: list[PrecheckFinding] = []
    if not report.ok:
        # Reaching here means an artifact changed after it was committed: these
        # checks are a precondition of the draft existing at all.
        raise WriterResponseError(
            f"run {run.run_id}: the persisted digest no longer satisfies its own "
            "deterministic checks; it was not produced by this pipeline as it stands",
            unknown_item_ids=list(report.unknown_item_ids),
            unsupported_numbers=list(report.unsupported_numbers),
            altered_lines=list(report.altered_lines),
            unsupported_claims=[str(c.verdict) for c in report.unsupported_claims],
        )

    prompt = build_digest_reviewer_prompt(
        facts=facts,
        editorial=editorial,
        article=article,
        run_id=run.run_id,
        precheck=report,
        prompt_version=prompt_version,
    )
    return prompt, PrecheckReport(findings=findings)


__all__ = [
    "DIGEST_EDITORIAL_FILENAME",
    "DigestDraft",
    "build_digest_draft",
    "digest_sources_of",
    "digest_window_of",
    "ensure_digest_snapshot",
    "load_digest_editorial",
    "prepare_digest_review",
]
