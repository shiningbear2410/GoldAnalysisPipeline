"""The automation worker: one finite tick, then exit.

**There is no loop here, and that is the design.** Windows Task Scheduler owns
the clock; this owns one unit of work. A daemon would need its own supervision,
its own restart policy, and its own answer for what happens when it wedges -
three problems the operating system already solved. So the worker starts, does a
bounded amount of work, writes down what it did, and exits.

The order within a tick::

    1. take the worker lock          one intake at a time
    2. reconcile stranded events     finish what a killed tick started
    3. promote due deferrals         events whose market may now be open
    4. resume existing Runs          oldest first
    5. process new events            oldest first, capped
    6. write state and history
    7. release the lock and exit

**Existing Runs come before new events.** A Run that already reached the
reviewer has spent real money on providers; abandoning it half-finished while
starting a fresh one turns a backlog into a bill.

**Two clocks, not one.** ``MAX_DATA_AGE`` asks whether the candles are current;
``MAX_ANALYSIS_EVENT_AGE`` asks whether the analyst's note still describes a
market anyone is looking at. Weekends fall out of those two facts without a
calendar: on Saturday the market data is stale, so events defer; by the time it
reopens the note has expired, so it is never paired with Monday's bars. No day of
the week appears anywhere in this file.

**Publish outcomes never reach the retry classifier.** Round 6 decided that an
ambiguous delivery is terminal, and a scheduler that could second-guess that
would be the single most expensive bug available here. ``UNCERTAIN`` arrives as
``BLOCKED`` and is written down, not retried.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from goldpipeline.adapters.base import MarketDataSource
from goldpipeline.adapters.inbox_source import parse_event
from goldpipeline.config import AutomationSettings
from goldpipeline.domain.errors import (
    ArtifactIntegrityError,
    AutoPublishNotAllowedError,
    AutoPublishTargetMismatchError,
    ContextIntegrityError,
    FinalizeConfigurationError,
    InboxPayloadError,
    MarketDataConfigurationError,
    MarketDataError,
    PipelineError,
    PublisherConfigurationError,
    ReviewConfigurationError,
    RunLockedError,
    WriterConfigurationError,
)
from goldpipeline.schemas.automation import (
    AutomationState,
    AutomationTickResult,
    RetryClass,
    TickStatus,
    WorkItem,
    WorkOutcome,
)
from goldpipeline.schemas.common import utc_now
from goldpipeline.schemas.ingestion import IngestOutcome, IngestResult
from goldpipeline.schemas.manifest import RunStatus
from goldpipeline.schemas.orchestration import PipelineMode, PipelineStatus
from goldpipeline.services.automation_state import (
    AutomationStore,
    next_defer,
    read_defer,
    write_defer,
)
from goldpipeline.services.inbox import INCOMING, Inbox
from goldpipeline.services.ingestion import IngestionContext, ingest_claimed, reconcile
from goldpipeline.services.orchestrator import (
    PipelineClients,
    PipelineRunResult,
    resume_pipeline,
)
from goldpipeline.services.run_lock import WORKER_LOCK_FILENAME, RunLock
from goldpipeline.storage.atomic import encode_json
from goldpipeline.storage.run_store import RunStore

logger = logging.getLogger(__name__)

DEFERRED = "deferred"
EXPIRED = "expired"
"""Two more inbox states, both auditable and neither destructive.

``deferred/`` is "not yet"; ``expired/`` is "no longer". Keeping them apart
matters because one is waiting for the world and the other is waiting for a
human.
"""

RESUMABLE_STATUSES = (
    RunStatus.NORMALIZED,
    RunStatus.DRAFTED,
    RunStatus.REVIEWED,
    RunStatus.FINALIZED,
)
"""Statuses the worker will pick up unprompted.

Everything else is either finished or needs a person. ``READY_TO_PUBLISH`` is
handled separately, because whether it is resumable depends on whether
unattended publishing has been authorised.
"""

DEFERRABLE_MARKET_CODES = frozenset(
    {"STALE_MARKET_DATA", "MT5_INITIALIZE_FAILED", "MT5_NOT_INSTALLED", "MT5_PROVIDER_ERROR"}
)
"""Market failures that mean *not now* rather than *not ever*.

A closed market, a terminal that is not running, a transient IPC error. None of
these say anything is wrong with the event, so the event waits. A missing symbol
or an unusable configuration is not here: those need a person.
"""


@dataclass(frozen=True)
class WorkerContext:
    """Everything one tick needs, all injected.

    The market source and the client factories are passed in rather than built
    here so that a test drives a whole tick with no terminal, no keys and no
    network - and so that this module never learns what MetaTrader is.
    """

    inbox: Inbox
    store: RunStore
    automation: AutomationStore
    settings: AutomationSettings
    market_source: MarketDataSource
    clients: PipelineClients
    expected_symbol: str | None = None
    publisher_target: str | None = None
    """The configured Telegram destination, read once by the caller.

    Compared against the allowlist. Never taken from a Run, an event, or an
    argument that a payload could reach.
    """

    elapsed: Callable[[], float] = time.monotonic
    """Source for the tick deadline, in seconds.

    Deliberately *not* the same clock as ``now``. ``now`` is logical time and
    is pinned in tests so artifacts come out identical; this measures how long
    the tick has actually been running, which a pinned clock cannot answer.
    """


@dataclass
class _Tick:
    """Accumulator for one invocation."""

    tick_id: str
    started: datetime
    started_elapsed: float
    mode: PipelineMode
    auto_publish: bool
    reconciled: list[WorkItem] = field(default_factory=list)
    resumed: list[WorkItem] = field(default_factory=list)
    processed: list[WorkItem] = field(default_factory=list)
    deferred: list[WorkItem] = field(default_factory=list)
    expired: list[WorkItem] = field(default_factory=list)
    blocked: list[WorkItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def finish(self, status: TickStatus) -> AutomationTickResult:
        return AutomationTickResult(
            tick_id=self.tick_id,
            started_at=self.started,
            completed_at=utc_now(),
            status=status,
            mode=str(self.mode),
            auto_publish_enabled=self.auto_publish,
            reconciled=self.reconciled,
            resumed_runs=self.resumed,
            processed_events=self.processed,
            deferred_events=self.deferred,
            expired_events=self.expired,
            blocked_runs=self.blocked,
            errors=self.errors,
        )


def run_tick(context: WorkerContext, *, now: datetime | None = None) -> AutomationTickResult:
    """Do one finite unit of automation work, then return.

    Args:
        context: Injected inbox, storage, settings, market source and clients.
        now: Injection point for tests; the whole tick uses one instant.

    Returns:
        An :class:`AutomationTickResult`. Business outcomes - a blocked gate, a
        deferred event, an expired one - are ordinary results, not errors.

    Raises:
        AutoPublishNotAllowedError: Unattended publishing is switched on but not
            authorised for the configured destination. Refused before any work,
            because a misconfigured allowlist must not quietly publish anything.
    """
    moment = now or utc_now()
    tick = _Tick(
        tick_id=secrets.token_hex(4),
        started=moment,
        started_elapsed=context.elapsed(),
        mode=PipelineMode.READY_FOR_PUBLISH,
        auto_publish=False,
    )

    # Resolved before the lock: an unauthorised publish configuration is a
    # refusal to start, not something to discover halfway through a tick.
    if context.settings.auto_publish_enabled:
        _require_auto_publish_allowed(context.settings, context.publisher_target)
        tick.mode = PipelineMode.PUBLISH
        tick.auto_publish = True

    context.automation.ensure_layout()
    lock = RunLock(context.automation.root, filename=WORKER_LOCK_FILENAME, now=moment)
    try:
        lock.acquire()
    except RunLockedError:
        # Another tick is already deciding what to work on. Overlap is expected
        # with a minute schedule and a stage that can take longer than a minute;
        # it is not an error and must not be reported as one.
        logger.info("automation.tick=%s status=SKIPPED another worker holds the lock", tick.tick_id)
        return tick.finish(TickStatus.SKIPPED)

    try:
        _reconcile(context, tick)
        _promote_deferred(context, tick, moment)
        _resume_runs(context, tick, moment)
        _process_events(context, tick, moment)
        status = TickStatus.BLOCKED if tick.blocked else TickStatus.OK
    except PipelineError as exc:
        # A worker-level failure. The stage-level ones are handled where they
        # happen and become retry state rather than ending the tick.
        logger.error("automation.tick=%s status=FAILED code=%s", tick.tick_id, exc.code)
        tick.errors.append(exc.code)
        status = TickStatus.FAILED
    finally:
        lock.release()

    result = tick.finish(status)
    _persist(context, result)
    logger.info(
        "automation.tick=%s status=%s resumed=%d processed=%d deferred=%d expired=%d blocked=%d",
        result.tick_id,
        result.status,
        len(result.resumed_runs),
        len(result.processed_events),
        len(result.deferred_events),
        len(result.expired_events),
        len(result.blocked_runs),
    )
    return result


# --------------------------------------------------------------------------
# the auto-publish guard
# --------------------------------------------------------------------------


def _require_auto_publish_allowed(settings: AutomationSettings, target: str | None) -> None:
    """Refuse unattended publishing unless it is authorised for *this* channel.

    Two settings have to agree. Enabling publishing is one decision; naming the
    channel it may publish to is a second, and requiring both means a copied
    ``.env`` or an inherited environment cannot silently redirect the pipeline at
    someone else's channel. They must match exactly - no prefix, no
    normalisation, no "close enough".
    """
    allowed = settings.auto_publish_allowed_target
    if not allowed:
        raise AutoPublishNotAllowedError(
            "unattended publishing is enabled but no destination is allowlisted. "
            "Set GOLDPIPELINE_AUTOPUBLISH_ALLOWED_TARGET to the channel it may "
            "post to; enabling publishing without naming a target is not enough.",
            setting="GOLDPIPELINE_AUTOPUBLISH_ALLOWED_TARGET",
        )
    if not target:
        raise AutoPublishNotAllowedError(
            "unattended publishing is enabled but TELEGRAM_TARGET_CHAT_ID is not configured",
            setting="TELEGRAM_TARGET_CHAT_ID",
        )
    if allowed != target:
        raise AutoPublishTargetMismatchError(
            "the allowlisted automation target does not match the configured "
            "destination, so nothing will be published. Make "
            "GOLDPIPELINE_AUTOPUBLISH_ALLOWED_TARGET and TELEGRAM_TARGET_CHAT_ID "
            "identical, or turn unattended publishing off.",
            allowlisted=allowed,
            configured=target,
        )


# --------------------------------------------------------------------------
# 2. reconcile
# --------------------------------------------------------------------------


def _reconcile(context: WorkerContext, tick: _Tick) -> None:
    """Finish what an interrupted tick started, before starting anything new."""
    ingestion = _ingestion_context(context)
    for report in reconcile(ingestion, recover=True):
        tick.reconciled.append(
            WorkItem(
                kind="event",
                identifier=report.event_id,
                outcome=WorkOutcome.COMPLETED if report.run_status else WorkOutcome.SKIPPED,
                detail=report.resolution,
            )
        )


# --------------------------------------------------------------------------
# 3. deferrals
# --------------------------------------------------------------------------


def _promote_deferred(context: WorkerContext, tick: _Tick, moment: datetime) -> None:
    """Return deferred events to the queue once their wait is over.

    Promotion rather than in-place retry keeps the rest of the tick uniform:
    after this step there is exactly one place to look for work.
    """
    directory = context.inbox.directory(DEFERRED)
    if not directory.is_dir():
        return

    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(".defer.json"):
            continue
        record = read_defer(path)
        if record is not None and not record.due(moment):
            continue
        # Only the event moves. The sidecar stays behind in `deferred/`, which
        # keeps `incoming/` holding nothing but events - and means a second
        # deferral finds the first one and counts up from it.
        promoted = _move(path, context.inbox.directory(INCOMING))
        logger.info("automation.promote event=%s", promoted.stem)


def _defer(
    context: WorkerContext,
    tick: _Tick,
    event_path: Path,
    *,
    code: str,
    reason: str,
    moment: datetime,
) -> None:
    """Set an event aside until the market may be ready for it.

    The payload is moved, never rewritten. A sidecar records why and when to
    look again, so an operator can see a queue waiting on a closed market rather
    than a queue of failures.
    """
    landed = _move(event_path, context.inbox.directory(DEFERRED))
    record = next_defer(
        landed,
        reason_code=code,
        reason=reason,
        now=moment,
        retry_minutes=context.settings.defer_retry_minutes,
    )
    write_defer(landed, record)
    tick.deferred.append(
        WorkItem(
            kind="event",
            identifier=landed.stem,
            outcome=WorkOutcome.DEFERRED,
            code=code,
            detail=reason,
            next_attempt_at=record.next_attempt_at,
        )
    )
    logger.info(
        "automation.defer event=%s code=%s next=%s",
        landed.stem,
        code,
        record.next_attempt_at.isoformat(),
    )


def _expire(
    context: WorkerContext, tick: _Tick, event_path: Path, *, age_minutes: int, moment: datetime
) -> None:
    """Retire an analysis that waited too long to be worth writing about.

    Never requeued and never paired with fresher candles. That pairing - a
    Saturday note against Monday's opening bars - is the specific accident this
    whole age policy exists to prevent.
    """
    landed = _move(event_path, context.inbox.directory(EXPIRED))
    reason = (
        f"the analysis was {age_minutes} minutes old, past the "
        f"{context.settings.max_event_age_minutes} minute limit. It is not paired with "
        "newer market data; submit a fresh analysis instead."
    )
    _write_reason(landed, code="EXPIRED_ANALYSIS_EVENT", reason=reason, moment=moment)
    tick.expired.append(
        WorkItem(
            kind="event",
            identifier=landed.stem,
            outcome=WorkOutcome.EXPIRED,
            code="EXPIRED_ANALYSIS_EVENT",
            detail=reason,
        )
    )
    logger.info("automation.expire event=%s age=%d", landed.stem, age_minutes)


# --------------------------------------------------------------------------
# 4. resume existing Runs
# --------------------------------------------------------------------------


def _resume_runs(context: WorkerContext, tick: _Tick, moment: datetime) -> None:
    """Advance Runs that already exist, oldest first."""
    for run_id in _resumable_runs(context, moment):
        if _out_of_time(context, tick):
            return
        _advance(context, tick, run_id, moment)


def _resumable_runs(context: WorkerContext, moment: datetime) -> list[str]:
    """Which Runs this tick may pick up, oldest created first.

    Oldest first because a backlog worked newest-first never clears its tail.
    """
    candidates: list[tuple[datetime, str]] = []
    for run_id in context.store.list_run_ids():
        try:
            manifest = context.store.open(run_id).load_manifest()
        except (FileNotFoundError, ValueError, PipelineError):
            continue

        # Anything else is terminal or waiting for a human. Silently skipped
        # rather than logged as an error every minute for the rest of its life.
        if not may_resume(manifest.status, context.settings.auto_publish_enabled):
            continue

        if not context.automation.retry_allows(run_id, moment):
            continue
        candidates.append((manifest.created_at, run_id))

    return [run_id for _, run_id in sorted(candidates)]


def may_resume(status: RunStatus, auto_publish: bool) -> bool:
    """Whether the worker may pick up a Run in this state unprompted.

    ``READY_TO_PUBLISH`` is the only status whose answer depends on
    configuration: with unattended publishing off it is finished work, and with
    it on there is one more stage to run.
    """
    if status in RESUMABLE_STATUSES:
        return True
    return status is RunStatus.READY_TO_PUBLISH and auto_publish


def _advance(context: WorkerContext, tick: _Tick, run_id: str, moment: datetime) -> None:
    """Drive one Run as far as the mode allows, and record what happened."""
    mode = tick.mode
    if mode is PipelineMode.PUBLISH:
        refusal = _publish_refusal(context, run_id, moment)
        if refusal is not None:
            tick.blocked.append(refusal)
            return

    try:
        outcome = resume_pipeline(
            run_id=run_id, store=context.store, clients=context.clients, mode=mode
        )
    except RunLockedError:
        # Someone is driving this Run by hand. Their lock, their Run.
        tick.resumed.append(
            WorkItem(kind="run", identifier=run_id, outcome=WorkOutcome.SKIPPED, code="RUN_LOCKED")
        )
        return
    except PipelineError as exc:
        _record_stage_failure(context, tick, run_id, exc, moment)
        return

    _record_outcome(context, tick, run_id, outcome, moment)


def _record_outcome(
    context: WorkerContext,
    tick: _Tick,
    run_id: str,
    outcome: PipelineRunResult,
    moment: datetime,
) -> None:
    """Translate a pipeline result into a work item and any retry state."""
    result = outcome.result
    status: PipelineStatus = result.status

    if status in (PipelineStatus.COMPLETED, PipelineStatus.ALREADY_COMPLETED):
        context.automation.clear_retry(run_id)
        published = result.run_status is RunStatus.PUBLISHED
        tick.resumed.append(
            WorkItem(
                kind="run",
                identifier=run_id,
                outcome=WorkOutcome.PUBLISHED if published else WorkOutcome.COMPLETED,
                detail=str(result.run_status),
            )
        )
        return

    if status in (PipelineStatus.BLOCKED, PipelineStatus.NOT_RESUMABLE):
        # A gate spoke, or a publish attempt reached a state Round 6 calls
        # terminal. Recorded once and never retried - most importantly
        # PUBLISH_UNCERTAIN, where Telegram may already hold the article.
        context.automation.record_failure(
            run_id,
            failure_code=str(result.run_status),
            retry_class=RetryClass.TERMINAL,
            now=moment,
        )
        tick.blocked.append(
            WorkItem(
                kind="run",
                identifier=run_id,
                outcome=WorkOutcome.BLOCKED,
                code=str(result.run_status),
                detail=str(status),
            )
        )
        return

    error = outcome.error
    if error is not None:
        _record_stage_failure(context, tick, run_id, error, moment)
        return

    tick.resumed.append(
        WorkItem(kind="run", identifier=run_id, outcome=WorkOutcome.FAILED, code=str(status))
    )


def _record_stage_failure(
    context: WorkerContext, tick: _Tick, run_id: str, error: PipelineError, moment: datetime
) -> None:
    """Classify a stage failure and schedule - or refuse - another attempt."""
    retry_class = classify(error)
    record = context.automation.record_failure(
        run_id, failure_code=error.code, retry_class=retry_class, now=moment
    )
    tick.resumed.append(
        WorkItem(
            kind="run",
            identifier=run_id,
            outcome=(WorkOutcome.RETRY_SCHEDULED if not record.exhausted else WorkOutcome.FAILED),
            code=error.code,
            detail=str(retry_class),
            next_attempt_at=None if record.exhausted else record.next_attempt_at,
        )
    )


def classify(error: PipelineError) -> RetryClass:
    """Decide whether a failure is worth another attempt.

    The three interesting boundaries:

    * **integrity** is permanent. A Run whose artifacts no longer match their
      digests will not repair itself, and retrying builds on bytes nobody wrote.
    * **configuration** waits rather than exhausts. A missing key is a human's
      job, and the work should resume when they finish it - without anyone
      having to clear a retry file too.
    * **everything else from a provider** is transient with a bounded backoff.
      That includes a malformed structured response and a failed finalizer
      postcheck: models are not deterministic, so a second attempt genuinely
      differs - but only five of them, spread over three quarters of an hour,
      after which a human looks.

    Publish outcomes never arrive here. They come back as ``BLOCKED`` and are
    classified ``TERMINAL`` at the call site, which is what keeps Round 6's
    one-attempt rule above any policy written in this module.
    """
    if isinstance(error, ArtifactIntegrityError | ContextIntegrityError):
        return RetryClass.PERMANENT
    if isinstance(
        error,
        WriterConfigurationError
        | ReviewConfigurationError
        | FinalizeConfigurationError
        | PublisherConfigurationError
        | MarketDataConfigurationError,
    ):
        return RetryClass.CONFIGURATION
    if isinstance(error, MarketDataError):
        return RetryClass.TRANSIENT
    if isinstance(error, InboxPayloadError):
        return RetryClass.PERMANENT
    return RetryClass.TRANSIENT


def _publish_refusal(context: WorkerContext, run_id: str, moment: datetime) -> WorkItem | None:
    """Refuse to publish a Run that is too old, without touching it.

    The guard against the worst automation accident on offer: switching
    unattended publishing on and having last week's approved backlog go out at
    once. Age is measured from the Run's creation - the oldest, most
    conservative timestamp available - and an article past the cutoff is left
    exactly where it is for a human to publish deliberately.
    """
    try:
        manifest = context.store.open(run_id).load_manifest()
    except (FileNotFoundError, ValueError, PipelineError):
        return None
    if manifest.status is not RunStatus.READY_TO_PUBLISH:
        return None

    age = int((moment - manifest.created_at).total_seconds() // 60)
    if age <= context.settings.auto_publish_max_run_age_minutes:
        return None

    return WorkItem(
        kind="run",
        identifier=run_id,
        outcome=WorkOutcome.BLOCKED,
        code="AUTO_PUBLISH_TOO_OLD",
        detail=(
            f"approved {age} minutes ago, past the "
            f"{context.settings.auto_publish_max_run_age_minutes} minute cutoff for "
            "unattended publishing"
        ),
    )


# --------------------------------------------------------------------------
# 5. new events
# --------------------------------------------------------------------------


def _process_events(context: WorkerContext, tick: _Tick, moment: datetime) -> None:
    """Ingest waiting events, oldest first and no more than the cap."""
    ingestion = _ingestion_context(context)
    processed = 0

    for candidate in context.inbox.pending():
        if processed >= context.settings.max_events_per_tick:
            return
        if _out_of_time(context, tick):
            return

        claimed = context.inbox.claim(candidate)
        if claimed is None:
            continue

        processed += 1
        if _expire_if_too_old(context, tick, claimed, moment):
            continue
        _ingest_one(context, tick, ingestion, claimed, moment)


def _expire_if_too_old(
    context: WorkerContext, tick: _Tick, claimed: Path, moment: datetime
) -> bool:
    """Retire the event if the analysis has aged out. Returns whether it did.

    Checked *before* the market is consulted, so an expired note never causes a
    terminal round-trip and can never be rescued by fresh candles.
    """
    try:
        event = parse_event(context.inbox.read(claimed).payload)
    except InboxPayloadError:
        # Unparseable. The ingestion service refuses it properly, with a reason.
        return False

    age = int((moment - event.created_at).total_seconds() // 60)
    if age <= context.settings.max_event_age_minutes:
        return False

    _expire(context, tick, claimed, age_minutes=age, moment=moment)
    return True


def _ingest_one(
    context: WorkerContext,
    tick: _Tick,
    ingestion: IngestionContext,
    claimed: Path,
    moment: datetime,
) -> None:
    """Ingest one claimed event, then advance the Run it produced."""
    result = ingest_claimed(ingestion, claimed, now=moment)

    if result.outcome is IngestOutcome.MARKET_UNAVAILABLE:
        # The ingestion service put the event back in the queue. Whether it
        # waits or needs a person depends on *which* market failure it was.
        _handle_market_unavailable(context, tick, result, moment)
        return

    tick.processed.append(
        WorkItem(
            kind="event",
            identifier=result.event_id or claimed.stem,
            outcome=(
                WorkOutcome.INGESTED
                if result.succeeded
                else WorkOutcome.BLOCKED
                if result.outcome is not IngestOutcome.INVALID_PAYLOAD
                else WorkOutcome.FAILED
            ),
            code=result.failure_code or str(result.outcome),
            detail=result.detail,
        )
    )

    if result.succeeded and result.run_id:
        _advance(context, tick, result.run_id, moment)


def _handle_market_unavailable(
    context: WorkerContext, tick: _Tick, result: IngestResult, moment: datetime
) -> None:
    """Defer, or refuse, depending on the market failure.

    A closed market is *not now*; a symbol that does not exist on the broker is
    *not ever* without a person. Both arrive as the same ingestion outcome, and
    telling them apart is the whole reason the outcome carries a structured code.
    """
    code = result.failure_code or "MARKET_DATA_ERROR"
    queued = context.inbox.directory(INCOMING) / f"{result.event_id}.json"
    if not queued.is_file():  # pragma: no cover - the service just put it there
        return

    if code not in DEFERRABLE_MARKET_CODES:
        tick.blocked.append(
            WorkItem(
                kind="event",
                identifier=result.event_id or queued.stem,
                outcome=WorkOutcome.BLOCKED,
                code=code,
                detail="market data is misconfigured; deferring would not help",
            )
        )
        tick.errors.append(code)
        return

    claimed = context.inbox.claim(queued)
    if claimed is None:  # pragma: no cover - only under a concurrent consumer
        return
    _defer(context, tick, claimed, code=code, reason=result.detail or code, moment=moment)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _ingestion_context(context: WorkerContext) -> IngestionContext:
    return IngestionContext(
        inbox=context.inbox,
        store=context.store,
        market_source=context.market_source,
        expected_symbol=context.expected_symbol,
    )


def _out_of_time(context: WorkerContext, tick: _Tick) -> bool:
    """Whether to stop starting new work.

    A soft deadline: nothing in flight is interrupted and no network call is
    cut short. It only stops the worker *beginning* something it probably
    cannot finish, and the next tick picks up where this one left off. Task
    Scheduler's overlap policy covers a tick that runs past the next one.
    """
    spent = context.elapsed() - tick.started_elapsed
    return spent >= context.settings.max_tick_minutes * 60


def _move(path: Path, destination: Path) -> Path:
    """Move one file into *destination*, creating it if needed."""
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / path.name
    os.replace(path, target)
    return target


def _write_reason(path: Path, *, code: str, reason: str, moment: datetime) -> None:
    """Write a reason sidecar beside a retired event."""
    note = {
        "code": code,
        "reason": reason,
        "recorded_at": moment.isoformat().replace("+00:00", "Z"),
    }
    (path.parent / f"{path.stem}.reason.json").write_bytes(encode_json(note))


def _persist(context: WorkerContext, result: AutomationTickResult) -> None:
    """Update the dashboard and add one immutable tick record."""
    previous = context.automation.read_state()
    state = AutomationState(
        last_tick_id=result.tick_id,
        last_tick_started_at=result.started_at,
        last_tick_completed_at=result.completed_at,
        last_tick_status=result.status,
        events_seen=previous.events_seen
        + len(result.processed_events)
        + len(result.deferred_events)
        + len(result.expired_events),
        events_processed=previous.events_processed + len(result.processed_events),
        events_deferred=previous.events_deferred + len(result.deferred_events),
        events_expired=previous.events_expired + len(result.expired_events),
        runs_resumed=previous.runs_resumed + len(result.resumed_runs),
        runs_completed=previous.runs_completed
        + sum(
            1
            for item in result.resumed_runs
            if item.outcome in (WorkOutcome.COMPLETED, WorkOutcome.PUBLISHED)
        ),
        runs_blocked=previous.runs_blocked + len(result.blocked_runs),
        last_error_safe=result.errors[-1] if result.errors else previous.last_error_safe,
    )
    context.automation.write_state(state)
    context.automation.record_tick(result)


__all__ = [
    "DEFERRABLE_MARKET_CODES",
    "DEFERRED",
    "EXPIRED",
    "RESUMABLE_STATUSES",
    "WorkerContext",
    "classify",
    "may_resume",
    "run_tick",
]
