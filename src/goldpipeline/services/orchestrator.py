"""End-to-end pipeline orchestration.

This module contains no business logic and is meant to stay that way. It knows
the order of the six stages, which one a Run is due for, and when to stop. Every
judgement about an article - is it accurate, does it need revision, is it safe to
publish, did it reach Telegram - is made by the stage that already owned that
question, and read back here as a verdict.

**The state machine is the Run's own status.** There is no second progress model
to drift out of sync with the manifest::

    NORMALIZED       -> WRITE
    DRAFTED          -> REVIEW
    REVIEWED         -> FINALIZE
    FINALIZED        -> GATE
    READY_TO_PUBLISH -> PUBLISH   (only in PUBLISH mode)

Anything else is terminal for this orchestrator, and the publish-side states are
terminal *in different ways* - see :func:`_terminal_result`.

**Stages are never re-run.** The status decides what is due, and the stage
services themselves refuse to overwrite artifacts and re-verify digests before
trusting anything. Status alone is not treated as proof: it selects the next
step, and the step then checks the bytes.

**Nothing publishes by default.** :class:`PipelineMode` is a ceiling, and the
default ceiling is the gate. Reaching Telegram takes an explicit mode.

**No execution artifact is written.** A ``pipeline_execution.json`` was
considered and rejected: a Run may legitimately be orchestrated more than once
(create today, resume and publish tomorrow), and artifacts in this pipeline are
write-once, so the second invocation would have to either fail or overwrite -
and both break the model the rest of the rounds depend on. The audit trail lives
where it already did, in the manifest's event log, which this module appends to;
the machine-readable view is the returned
:class:`~goldpipeline.schemas.orchestration.PipelineExecutionResult`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from goldpipeline.adapters.base import AnalysisSource, MarketDataSource
from goldpipeline.adapters.finalizer_client import FinalizerClient, LazyFinalizerClient
from goldpipeline.adapters.publisher_client import PublisherClient
from goldpipeline.adapters.reviewer_client import ReviewerClient
from goldpipeline.adapters.writer_client import WriterClient
from goldpipeline.domain.errors import (
    ArticleTypeNotReadyError,
    PipelineError,
    RunNotResumableError,
)
from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.common import utc_now
from goldpipeline.schemas.manifest import RunStatus
from goldpipeline.schemas.orchestration import (
    PipelineEvent,
    PipelineExecutionResult,
    PipelineMode,
    PipelineStage,
    PipelineStatus,
    StageExecution,
    StageOutcome,
)
from goldpipeline.schemas.publish import Decision
from goldpipeline.schemas.publisher import PublishStatus
from goldpipeline.schemas.review import ReviewStatus
from goldpipeline.services.article_routing import writer_prompt_for
from goldpipeline.services.finalizer import finalize_run, load_verified_inputs
from goldpipeline.services.pipeline import create_run
from goldpipeline.services.publish_gate import gate_publish
from goldpipeline.services.publisher import publish_run
from goldpipeline.services.reviewer import review_draft
from goldpipeline.services.run_lock import RunLock
from goldpipeline.services.writer import write_draft
from goldpipeline.storage.run_store import RunStore

logger = logging.getLogger(__name__)

DEFAULT_MODE = PipelineMode.READY_FOR_PUBLISH
"""The safe default: run every check, publish nothing.

Chosen over ``GENERATE_ONLY`` because the publish decision is itself a check
worth having, and it costs nothing - the gate makes no provider call and opens
no socket.
"""

STAGE_ORDER = (
    PipelineStage.NORMALIZE,
    PipelineStage.WRITE,
    PipelineStage.REVIEW,
    PipelineStage.FINALIZE,
    PipelineStage.GATE,
    PipelineStage.PUBLISH,
)

_CEILING: dict[PipelineMode, PipelineStage] = {
    PipelineMode.GENERATE_ONLY: PipelineStage.FINALIZE,
    PipelineMode.READY_FOR_PUBLISH: PipelineStage.GATE,
    PipelineMode.PUBLISH: PipelineStage.PUBLISH,
}

_DUE_STAGE: dict[RunStatus, PipelineStage] = {
    RunStatus.NORMALIZED: PipelineStage.WRITE,
    RunStatus.DRAFTED: PipelineStage.REVIEW,
    RunStatus.REVIEWED: PipelineStage.FINALIZE,
    RunStatus.FINALIZED: PipelineStage.GATE,
    RunStatus.READY_TO_PUBLISH: PipelineStage.PUBLISH,
}

_NOT_RESUMABLE: dict[RunStatus, str] = {
    RunStatus.PUBLISHING: (
        "a publish attempt is in flight or was interrupted. Telegram may already "
        "hold the article, so nothing here will send it again."
    ),
    RunStatus.PUBLISH_UNCERTAIN: (
        "a publish attempt ended without a confirmed outcome. This is never "
        "retried automatically - a human must check the channel and reconcile."
    ),
    RunStatus.PARTIALLY_PUBLISHED: (
        "some chunks were delivered and then the provider refused. Resending "
        "would duplicate what already reached readers."
    ),
    RunStatus.PUBLISH_FAILED: (
        "the publish attempt was explicitly refused. One attempt per Run: fix "
        "the cause and create a new Run."
    ),
    RunStatus.CREATED: "the Run never finished normalizing; it has no usable context.",
    RunStatus.FAILED: "the Run failed during normalization and its inputs are unusable.",
}
"""Statuses the orchestrator refuses to drive, and why.

Every entry is a place where continuing could either duplicate a published
article or build on data the pipeline already rejected.
"""


@dataclass(frozen=True)
class PipelineClients:
    """Client factories, called at most once and only when a stage needs one.

    Factories rather than instances so that a Run which stops before the writer
    - or resumes at the gate - never has to have an API key present. This is the
    whole of the lazy-configuration requirement: nothing is read from the
    environment until the stage that depends on it is about to run.

    The publisher factory returns its destination alongside its client, because
    the destination is part of the transport's configuration. There is
    deliberately no way to pass a target chat into this module: an orchestrator
    that accepted one would put the destination within reach of a command line,
    a config file, or in the worst case a Run's own content.
    """

    writer: Callable[[], WriterClient] | None = None
    reviewer: Callable[[], ReviewerClient] | None = None
    finalizer: Callable[[], FinalizerClient] | None = None
    publisher: Callable[[], tuple[PublisherClient, str]] | None = None


@dataclass
class _Execution:
    """Mutable accumulator for one invocation.

    Holds only what the returned result needs. Nothing here is authoritative -
    the Run's manifest and artifacts remain the record of what happened; this is
    a transcript of one pass over them.
    """

    store: RunStore
    run_id: str
    run_dir: Path
    mode: PipelineMode
    started: datetime
    stages: list[StageExecution] = field(default_factory=list)
    publish_decision: Decision | None = None
    publish_status: PublishStatus | None = None

    def ran(self, stage: PipelineStage) -> bool:
        """Whether *stage* already has a slot in this invocation."""
        return any(record.stage is stage for record in self.stages)

    def record(
        self,
        stage: PipelineStage,
        started: datetime,
        outcome: StageOutcome,
        detail: str | None = None,
    ) -> None:
        """Append one stage's slot."""
        self.stages.append(
            StageExecution(
                stage=stage,
                started_at=started,
                completed_at=utc_now(),
                outcome=outcome,
                detail=detail,
            )
        )

    def finish(
        self, status: PipelineStatus, *, error: PipelineError | None = None
    ) -> PipelineRunResult:
        """Freeze the transcript into a result.

        The Run's status is re-read here rather than tracked alongside, so the
        reported status is whatever the manifest actually says at the end - not
        what this module believed on the way through.
        """
        result = PipelineExecutionResult(
            run_id=self.run_id,
            started_at=self.started,
            completed_at=utc_now(),
            mode=self.mode,
            final_stage=self.stages[-1].stage if self.stages else None,
            status=status,
            run_status=self.store.open(self.run_id).load_manifest().status,
            stages=list(self.stages),
            publish_decision=self.publish_decision,
            publish_status=self.publish_status,
            error=error.to_dict() if error is not None else None,
        )
        logger.info(
            "run=%s stage=pipeline status=%s mode=%s stages=%d",
            self.run_id,
            status,
            self.mode,
            len(self.stages),
        )
        return PipelineRunResult(result=result, run_dir=self.run_dir, error=error)


@dataclass(frozen=True)
class PipelineRunResult:
    """Outcome of one orchestrator invocation.

    Mirrors the shape every stage returns: a serializable result, plus the live
    exception when there was one, so a caller can classify a missing API key
    differently from a malformed candle.
    """

    result: PipelineExecutionResult
    run_dir: Path
    error: PipelineError | None = None

    @property
    def run_id(self) -> str:
        return self.result.run_id

    @property
    def status(self) -> PipelineStatus:
        return self.result.status

    @property
    def succeeded(self) -> bool:
        """Whether the invocation ended without a stop or a failure."""
        return self.result.succeeded


def run_pipeline(
    *,
    analysis_source: AnalysisSource,
    market_source: MarketDataSource,
    store: RunStore,
    clients: PipelineClients,
    mode: PipelineMode = DEFAULT_MODE,
    expected_symbol: str | None = None,
    run_id: str | None = None,
    now: datetime | None = None,
) -> PipelineRunResult:
    """Create a Run from fresh inputs and drive it as far as *mode* allows.

    Args:
        analysis_source: Where the raw human analysis comes from.
        market_source: Where the OHLC data comes from.
        store: Run storage root.
        clients: Factories for whichever stages this mode will reach.
        mode: How far the invocation may go. Defaults to the safe ceiling.
        expected_symbol: Instrument the caller expects; a mismatch fails the Run.
        run_id: Force a specific id. Mainly for tests.
        now: Injection point for tests; threaded into every stage.

    Returns:
        A :class:`PipelineRunResult`. Business stops - a rejected review, a
        blocked gate, an unconfirmed delivery - are ordinary returns, not
        exceptions.

    Raises:
        RunLockedError: Another process is already driving this Run.
    """
    started = now or utc_now()
    created = create_run(
        analysis_source=analysis_source,
        market_source=market_source,
        store=store,
        expected_symbol=expected_symbol,
        run_id=run_id,
        now=now,
    )

    execution = _Execution(
        store=store, run_id=created.run_id, run_dir=created.run_dir, mode=mode, started=started
    )

    if not created.succeeded:
        assert created.error is not None
        execution.record(PipelineStage.NORMALIZE, started, StageOutcome.FAILED, str(created.status))
        logger.info("run=%s stage=pipeline.normalize status=FAILED", created.run_id)
        return execution.finish(PipelineStatus.FAILED, error=created.error)

    execution.record(PipelineStage.NORMALIZE, started, StageOutcome.COMPLETED, "COMPLETED")
    _record_event(store, created.run_id, PipelineEvent.RUN_CREATED, f"mode={mode}")

    # The Run directory did not exist a moment ago and its id is unique, so
    # there was nothing to contend over until now.
    with RunLock(created.run_dir):
        return _drive(store=store, clients=clients, execution=execution, now=now)


def resume_pipeline(
    *,
    run_id: str,
    store: RunStore,
    clients: PipelineClients,
    mode: PipelineMode = DEFAULT_MODE,
    now: datetime | None = None,
) -> PipelineRunResult:
    """Continue an existing Run from wherever it stopped.

    Only the stage the Run is *due for* runs; completed stages are neither
    repeated nor re-verified here, because the stage that owns each artifact
    re-verifies it on the way past.

    Args:
        run_id: The Run to continue.
        store: Where Runs live.
        clients: Factories for whichever stages this mode will reach.
        mode: How far the invocation may go. Defaults to the safe ceiling.
        now: Injection point for tests.

    Returns:
        A :class:`PipelineRunResult`.

    Raises:
        FileNotFoundError: No such Run.
        RunLockedError: Another process is already driving this Run.
    """
    run = store.open(run_id)
    execution = _Execution(
        store=store, run_id=run.run_id, run_dir=run.path, mode=mode, started=now or utc_now()
    )
    with RunLock(run.path):
        return _drive(store=store, clients=clients, execution=execution, now=now)


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


def _drive(
    *,
    store: RunStore,
    clients: PipelineClients,
    execution: _Execution,
    now: datetime | None,
) -> PipelineRunResult:
    """Advance the Run one stage at a time until something says stop."""
    while True:
        status = store.open(execution.run_id).load_manifest().status

        terminal = _terminal_result(status, execution)
        if terminal is not None:
            return terminal

        stage = _DUE_STAGE[status]
        if STAGE_ORDER.index(stage) > STAGE_ORDER.index(_CEILING[execution.mode]):
            # The Run is further along than this mode asked for. Not a stop -
            # the mode's ceiling was reached, which is success.
            return execution.finish(
                PipelineStatus.COMPLETED if execution.stages else PipelineStatus.ALREADY_COMPLETED
            )

        stop = _RUNNERS[stage](store, clients, execution, now)
        if stop is not None:
            return stop


def _terminal_result(status: RunStatus, execution: _Execution) -> PipelineRunResult | None:
    """Decide whether *status* ends the invocation before any stage runs."""
    if status is RunStatus.PUBLISHED:
        # Terminal and successful. Deliberately not an error: re-running a
        # finished pipeline should be a no-op, not a failure a script has to
        # special-case.
        execution.publish_status = PublishStatus.PUBLISHED
        return execution.finish(
            PipelineStatus.COMPLETED if execution.stages else PipelineStatus.ALREADY_COMPLETED
        )

    if status is RunStatus.PUBLISH_BLOCKED:
        # The decision is immutable. Re-running the gate over the same artifacts
        # would produce the same answer, and re-running it over different ones
        # would mean the Run was tampered with. Either way: a new Run.
        execution.publish_decision = Decision.BLOCKED
        return execution.finish(PipelineStatus.BLOCKED)

    reason = _NOT_RESUMABLE.get(status)
    if reason is not None:
        return execution.finish(
            PipelineStatus.NOT_RESUMABLE,
            error=RunNotResumableError(
                f"run {execution.run_id} is {status} and will not be resumed: {reason}",
                run_id=execution.run_id,
                status=str(status),
            ),
        )

    return None


# --------------------------------------------------------------------------
# stage runners
#
# Each returns None to continue, or a finished result to stop. None of them
# decides anything: they call one service and translate what it answered.
# --------------------------------------------------------------------------


def _run_write(
    store: RunStore,
    clients: PipelineClients,
    execution: _Execution,
    now: datetime | None,
) -> PipelineRunResult | None:
    started = utc_now()

    # Routing decided here, before a client is built or a provider is called.
    # An article type with no implementation stops the Run at this line: the
    # writer, reviewer, finalizer, gate and publisher all sit behind it, so a
    # mode that is not ready costs nothing and produces nothing. It is never
    # substituted with ANALYSIS - writing a different article than the one asked
    # for is a silent wrong answer, and refusing is a visible one.
    try:
        prompt_id = writer_prompt_for(_article_type_of(store, execution.run_id))
    except ArticleTypeNotReadyError as exc:
        execution.record(PipelineStage.WRITE, started, StageOutcome.FAILED)
        return _fail(store, execution, PipelineStage.WRITE, exc)

    result = write_draft(
        run_id=execution.run_id,
        store=store,
        client=_require(clients.writer, "writer")(),
        prompt_version=prompt_id,
        now=now,
    )

    if not result.succeeded:
        assert result.error is not None
        execution.record(PipelineStage.WRITE, started, StageOutcome.FAILED)
        return _fail(store, execution, PipelineStage.WRITE, result.error)

    assert result.result is not None
    execution.record(
        PipelineStage.WRITE, started, StageOutcome.COMPLETED, str(result.result.status)
    )
    _record_event(
        store,
        execution.run_id,
        PipelineEvent.WRITER_COMPLETED,
        f"{result.result.article_chars} chars",
    )
    return None


def _run_review(
    store: RunStore,
    clients: PipelineClients,
    execution: _Execution,
    now: datetime | None,
) -> PipelineRunResult | None:
    started = utc_now()
    result = review_draft(
        run_id=execution.run_id,
        store=store,
        client=_require(clients.reviewer, "reviewer")(),
        now=now,
    )

    if not result.succeeded:
        assert result.error is not None
        execution.record(PipelineStage.REVIEW, started, StageOutcome.FAILED)
        return _fail(store, execution, PipelineStage.REVIEW, result.error)

    assert result.result is not None
    verdict = result.result.status
    _record_event(store, execution.run_id, PipelineEvent.REVIEW_COMPLETED, str(verdict))

    if verdict is ReviewStatus.REJECT:
        # Stop here rather than at the finalizer. Round 4 would refuse anyway,
        # but calling it to be refused would report the wrong stage as the one
        # that ended the pipeline.
        execution.record(PipelineStage.REVIEW, started, StageOutcome.BLOCKED, str(verdict))
        return _stopped(store, execution, f"review verdict {verdict}")

    execution.record(PipelineStage.REVIEW, started, StageOutcome.COMPLETED, str(verdict))
    return None


def _run_finalize(
    store: RunStore,
    clients: PipelineClients,
    execution: _Execution,
    now: datetime | None,
) -> PipelineRunResult | None:
    started = utc_now()

    if not execution.ran(PipelineStage.REVIEW):
        # Resumed at REVIEWED, so this invocation has not seen the verdict.
        # Read it through the finalizer's own loader, which verifies all four
        # artifacts - a tampered Run must fail here, not quietly proceed.
        stop = _check_resumed_verdict(store, execution, started)
        if stop is not None:
            return stop

    client = clients.finalizer
    result = finalize_run(
        run_id=execution.run_id,
        store=store,
        # Wrapped, not built: a PASS is a byte copy and must not require a key.
        client=LazyFinalizerClient(client) if client is not None else None,
        now=now,
    )

    if not result.succeeded:
        assert result.error is not None
        execution.record(PipelineStage.FINALIZE, started, StageOutcome.FAILED)
        return _fail(store, execution, PipelineStage.FINALIZE, result.error)

    assert result.result is not None
    mode = result.result.finalization_mode
    execution.record(PipelineStage.FINALIZE, started, StageOutcome.COMPLETED, str(mode))
    _record_event(
        store,
        execution.run_id,
        PipelineEvent.FINALIZER_COMPLETED,
        f"{mode} provider_called={result.result.provider_called}",
    )
    return None


def _run_gate(
    store: RunStore,
    clients: PipelineClients,
    execution: _Execution,
    now: datetime | None,
) -> PipelineRunResult | None:
    started = utc_now()
    result = gate_publish(run_id=execution.run_id, store=store, now=now)
    decision = result.decision
    execution.publish_decision = decision.decision

    if not result.approved:
        blockers = len(decision.blockers)
        execution.record(
            PipelineStage.GATE,
            started,
            StageOutcome.BLOCKED,
            f"{decision.decision} ({blockers} blocker{'' if blockers == 1 else 's'})",
        )
        _record_event(
            store,
            execution.run_id,
            PipelineEvent.GATE_BLOCKED,
            f"{len(decision.blockers)} blocker(s)",
        )
        return _stopped(store, execution, f"gate blocked with {len(decision.blockers)} blocker(s)")

    execution.record(PipelineStage.GATE, started, StageOutcome.COMPLETED, str(decision.decision))
    _record_event(store, execution.run_id, PipelineEvent.GATE_APPROVED, decision.gate_version)
    return None


def _run_publish(
    store: RunStore,
    clients: PipelineClients,
    execution: _Execution,
    now: datetime | None,
) -> PipelineRunResult | None:
    started = utc_now()
    client, target_chat = _require(clients.publisher, "publisher")()

    outcome = publish_run(
        run_id=execution.run_id,
        store=store,
        client=client,
        target_chat=target_chat,
        now=now,
    )
    execution.publish_status = outcome.result.status

    if not outcome.published:
        # FAILED, PARTIAL and UNCERTAIN all end here, unretried. UNCERTAIN
        # especially: the publisher could not tell whether Telegram has the
        # article, and guessing is how one article becomes two.
        execution.record(
            PipelineStage.PUBLISH, started, StageOutcome.BLOCKED, str(outcome.result.status)
        )
        return _stopped(store, execution, f"publish ended {outcome.result.status}")

    execution.record(
        PipelineStage.PUBLISH, started, StageOutcome.COMPLETED, str(outcome.result.status)
    )
    _record_event(
        store,
        execution.run_id,
        PipelineEvent.PUBLISH_COMPLETED,
        f"{outcome.result.confirmed_count}/{outcome.result.chunk_count} chunk(s)",
    )
    return None


_Runner = Callable[
    [RunStore, PipelineClients, _Execution, "datetime | None"], "PipelineRunResult | None"
]

_RUNNERS: dict[PipelineStage, _Runner] = {
    PipelineStage.WRITE: _run_write,
    PipelineStage.REVIEW: _run_review,
    PipelineStage.FINALIZE: _run_finalize,
    PipelineStage.GATE: _run_gate,
    PipelineStage.PUBLISH: _run_publish,
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _article_type_of(store: RunStore, run_id: str) -> ArticleType:
    """The product mode this Run was created for.

    Read from the manifest rather than passed along, so a resumed Run routes the
    same way the original did. A Run written before article types existed has no
    provenance or no field, and both mean ANALYSIS - which is what it was.
    """
    manifest = store.open(run_id).load_manifest()
    if manifest.provenance is None:
        return ArticleType.ANALYSIS
    return manifest.provenance.article_type


def _check_resumed_verdict(
    store: RunStore, execution: _Execution, started: datetime
) -> PipelineRunResult | None:
    """Read a resumed Run's review verdict and stop if it was a rejection."""
    run = store.open(execution.run_id)
    manifest = run.load_manifest()
    try:
        inputs = load_verified_inputs(run, manifest)
    except PipelineError as exc:
        execution.record(PipelineStage.FINALIZE, started, StageOutcome.FAILED)
        return _fail(store, execution, PipelineStage.FINALIZE, exc)

    if inputs.review.status is ReviewStatus.REJECT:
        execution.record(
            PipelineStage.REVIEW, started, StageOutcome.BLOCKED, str(inputs.review.status)
        )
        return _stopped(store, execution, f"review verdict {inputs.review.status}")
    return None


def _require[ClientT](factory: Callable[[], ClientT] | None, name: str) -> Callable[[], ClientT]:
    """Fail loudly when a mode was asked for without the client it needs.

    A wiring mistake, not a pipeline outcome, so it raises rather than becoming a
    ``FAILED`` execution result.
    """
    if factory is None:
        raise ValueError(f"this pipeline mode reaches the {name} stage, but no {name} was provided")
    return factory


def _record_event(
    store: RunStore, run_id: str, event: PipelineEvent, message: str | None = None
) -> None:
    """Append an orchestration event to the Run's manifest.

    Re-read before writing: the stage that just finished saved its own manifest,
    and holding a stale copy here would erase the events it recorded.
    """
    run = store.open(run_id)
    manifest = run.load_manifest()
    manifest.record_event("pipeline", str(event), message)
    run.save_manifest(manifest)
    logger.info("run=%s stage=pipeline event=%s %s", run_id, event, message or "")


def _fail(
    store: RunStore, execution: _Execution, stage: PipelineStage, error: PipelineError
) -> PipelineRunResult:
    """End the invocation on an execution failure."""
    _record_event(store, execution.run_id, PipelineEvent.PIPELINE_FAILED, f"{stage}: {error.code}")
    return execution.finish(PipelineStatus.FAILED, error=error)


def _stopped(store: RunStore, execution: _Execution, reason: str) -> PipelineRunResult:
    """End the invocation on a business stop - a gate spoke, nothing broke."""
    _record_event(store, execution.run_id, PipelineEvent.PIPELINE_STOPPED, reason)
    return execution.finish(PipelineStatus.BLOCKED)


__all__ = [
    "DEFAULT_MODE",
    "STAGE_ORDER",
    "PipelineClients",
    "PipelineRunResult",
    "resume_pipeline",
    "run_pipeline",
]
