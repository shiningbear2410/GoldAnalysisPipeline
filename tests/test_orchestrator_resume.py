"""Resuming a Run: what may continue, what may not, and what it costs.

Two themes. First, **state awareness**: the Run's own status decides which
stage is due, so a resumed Run repeats nothing and needs no credentials for
stages it has already passed. Second, **fail closed on the publish side**: every
state except ``READY_TO_PUBLISH`` is refused, because on that side of the
pipeline a wrong guess is visible to readers and cannot be taken back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    CLEAN_ARTICLE,
    PIPELINE_NOW,
    TEST_TARGET_CHAT,
    RecordingSleep,
    TrackedClients,
    exploding_factory,
    make_drafted_run,
    make_finalized_run,
    make_normalized_run,
    make_published_ready_run,
    make_reviewed_run,
    make_tracked_clients,
    run_orchestrated,
    tamper,
)

from goldpipeline.adapters.fake_publisher import FakePublisherClient, ambiguous_client
from goldpipeline.domain.errors import (
    PublisherPermissionError,
    RunLockedError,
    RunNotResumableError,
)
from goldpipeline.schemas.manifest import RunStatus
from goldpipeline.schemas.orchestration import PipelineMode, PipelineStage, PipelineStatus
from goldpipeline.schemas.publisher import PublishStatus
from goldpipeline.services.orchestrator import (
    DEFAULT_MODE,
    PipelineClients,
    resume_pipeline,
)
from goldpipeline.services.publisher import publish_run
from goldpipeline.services.run_lock import LOCK_FILENAME, RunLock
from goldpipeline.storage.run_store import RunStore


def resume(
    runs_dir: Path,
    run_id: str,
    clients: TrackedClients | PipelineClients,
    *,
    mode: PipelineMode = DEFAULT_MODE,
) -> Any:
    """Continue *run_id*, with either tracked fakes or raw factories."""
    wired = clients.as_pipeline_clients() if isinstance(clients, TrackedClients) else clients
    return resume_pipeline(
        run_id=run_id,
        store=RunStore(runs_dir),
        clients=wired,
        mode=mode,
        now=PIPELINE_NOW,
    )


# --- resuming from each ordinary status -----------------------------------


def test_resume_from_normalized_runs_every_remaining_stage(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 15."""
    normalized = make_normalized_run(runs_dir, tmp_path)
    clients = make_tracked_clients()

    outcome = resume(runs_dir, normalized.run_id, clients)

    assert outcome.status is PipelineStatus.COMPLETED
    assert outcome.result.run_status is RunStatus.READY_TO_PUBLISH
    assert [record.stage for record in outcome.result.stages] == [
        PipelineStage.WRITE,
        PipelineStage.REVIEW,
        PipelineStage.FINALIZE,
        PipelineStage.GATE,
    ]


def test_resume_from_drafted_does_not_call_the_writer(runs_dir: Path, tmp_path: Path) -> None:
    """Requirements 16 and 26, and golden case E: writer calls must be zero."""
    drafted = make_drafted_run(runs_dir, tmp_path)
    before = (Path(drafted.run_dir) / "claude_draft.md").read_bytes()
    clients = make_tracked_clients()

    outcome = resume(runs_dir, drafted.run_id, clients)

    assert clients.writer.calls == []
    assert "writer" not in clients.built
    assert (Path(drafted.run_dir) / "claude_draft.md").read_bytes() == before
    assert outcome.result.run_status is RunStatus.READY_TO_PUBLISH
    assert [record.stage for record in outcome.result.stages][0] is PipelineStage.REVIEW


def test_resume_from_reviewed_starts_at_the_finalizer(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 17."""
    reviewed = make_reviewed_run(runs_dir, tmp_path)
    clients = make_tracked_clients()

    outcome = resume(runs_dir, reviewed.run_id, clients)

    assert clients.writer.calls == []
    assert clients.reviewer.calls == []
    assert [record.stage for record in outcome.result.stages] == [
        PipelineStage.FINALIZE,
        PipelineStage.GATE,
    ]


def test_resume_from_finalized_only_runs_the_gate(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 18."""
    finalized = make_finalized_run(runs_dir, tmp_path)
    clients = make_tracked_clients()

    outcome = resume(runs_dir, finalized.run_id, clients)

    assert [record.stage for record in outcome.result.stages] == [PipelineStage.GATE]
    assert outcome.result.run_status is RunStatus.READY_TO_PUBLISH


def test_resume_from_ready_to_publish_without_publish_is_a_no_op(
    runs_dir: Path, tmp_path: Path
) -> None:
    """Requirement 19.

    Success, not an error: a scheduler re-running the safe mode over a Run that
    already passed the gate should be quiet, not noisy.
    """
    ready = make_published_ready_run(runs_dir, tmp_path)
    clients = make_tracked_clients()

    outcome = resume(runs_dir, ready.run_id, clients)

    assert outcome.status is PipelineStatus.ALREADY_COMPLETED
    assert outcome.succeeded
    assert outcome.result.stages == []
    assert outcome.result.final_stage is None
    assert clients.built == []
    assert not (Path(ready.run_dir) / "publish_intent.json").exists()


def test_resume_from_ready_to_publish_with_publish_delivers(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 20."""
    ready = make_published_ready_run(runs_dir, tmp_path)
    clients = make_tracked_clients()

    outcome = resume(runs_dir, ready.run_id, clients, mode=PipelineMode.PUBLISH)

    assert outcome.status is PipelineStatus.COMPLETED
    assert outcome.result.publish_status is PublishStatus.PUBLISHED
    assert outcome.result.run_status is RunStatus.PUBLISHED
    assert len(clients.publisher.calls) == 1


# --- terminal and refused states ------------------------------------------


def test_a_published_run_is_terminal(runs_dir: Path, tmp_path: Path) -> None:
    """Requirements 14 (spec) and 21: zero provider calls, and not an error."""
    clients = make_tracked_clients()
    first = run_orchestrated(runs_dir, tmp_path, clients, mode=PipelineMode.PUBLISH)
    assert first.result.run_status is RunStatus.PUBLISHED

    second_clients = make_tracked_clients()
    outcome = resume(runs_dir, first.run_id, second_clients, mode=PipelineMode.PUBLISH)

    assert outcome.status is PipelineStatus.ALREADY_COMPLETED
    assert outcome.succeeded
    assert second_clients.built == []
    assert second_clients.publisher.calls == []


def test_a_blocked_run_does_not_rerun_the_gate(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 22.

    The decision is immutable. Round 5 would raise rather than write a second
    one, so re-running the gate could only ever produce noise; the orchestrator
    reports the standing decision instead.
    """
    clients = make_tracked_clients()
    blocked = run_orchestrated(
        runs_dir,
        tmp_path,
        clients,
        article="Vàng đang giằng co trong biên hẹp, chưa có tín hiệu rõ ràng.",
        enforce_contract=False,
    )
    assert blocked.result.run_status is RunStatus.PUBLISH_BLOCKED
    decision_before = (Path(blocked.run_dir) / "publish_decision.json").read_bytes()

    outcome = resume(runs_dir, blocked.run_id, make_tracked_clients(), mode=PipelineMode.PUBLISH)

    assert outcome.status is PipelineStatus.BLOCKED
    assert outcome.result.stages == []
    assert (Path(blocked.run_dir) / "publish_decision.json").read_bytes() == decision_before


def test_an_uncertain_run_is_never_resumed(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 23, and the second half of golden case F.

    The one that matters most. Telegram may already hold the article; a second
    attempt cannot find out, so there is not one.
    """
    clients = make_tracked_clients(publisher=ambiguous_client())
    uncertain = run_orchestrated(runs_dir, tmp_path, clients, mode=PipelineMode.PUBLISH)
    assert uncertain.result.run_status is RunStatus.PUBLISH_UNCERTAIN

    second = make_tracked_clients()
    outcome = resume(runs_dir, uncertain.run_id, second, mode=PipelineMode.PUBLISH)

    assert outcome.status is PipelineStatus.NOT_RESUMABLE
    assert isinstance(outcome.error, RunNotResumableError)
    assert second.publisher.calls == []
    assert second.built == []


def test_a_partially_published_run_is_never_resumed(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 24."""
    paragraph = (
        "Vàng tiếp tục tích luỹ trong biên hẹp, thanh khoản mỏng dần về cuối phiên. "
        "Phe mua vẫn giữ được vùng hỗ trợ nhưng chưa tạo ra động lực rõ ràng nào."
    )
    clients = make_tracked_clients(
        publisher=FakePublisherClient(
            failures={1: PublisherPermissionError("forbidden", status_code=403)}
        )
    )
    partial = run_orchestrated(
        runs_dir,
        tmp_path,
        clients,
        article=CLEAN_ARTICLE + "\n\n" + "\n\n".join([paragraph] * 30),
        mode=PipelineMode.PUBLISH,
        enforce_contract=False,
    )
    assert partial.result.run_status is RunStatus.PARTIALLY_PUBLISHED

    second = make_tracked_clients()
    outcome = resume(runs_dir, partial.run_id, second, mode=PipelineMode.PUBLISH)

    assert outcome.status is PipelineStatus.NOT_RESUMABLE
    assert second.publisher.calls == []


def test_a_run_stuck_publishing_is_never_resumed(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 25.

    ``PUBLISHING`` is what a process killed mid-send leaves behind. It looks
    exactly like a send in flight, so it is treated as one.
    """
    ready = make_published_ready_run(runs_dir, tmp_path)
    run = RunStore(runs_dir).open(ready.run_id)
    manifest = run.load_manifest()
    manifest.status = RunStatus.PUBLISHING
    run.save_manifest(manifest)

    clients = make_tracked_clients()
    outcome = resume(runs_dir, ready.run_id, clients, mode=PipelineMode.PUBLISH)

    assert outcome.status is PipelineStatus.NOT_RESUMABLE
    assert clients.publisher.calls == []


def test_a_refused_publish_is_not_reattempted(runs_dir: Path, tmp_path: Path) -> None:
    """One attempt per Run holds through the orchestrator too."""
    from goldpipeline.adapters.fake_publisher import rejecting_client

    clients = make_tracked_clients(publisher=rejecting_client())
    failed = run_orchestrated(runs_dir, tmp_path, clients, mode=PipelineMode.PUBLISH)
    assert failed.result.run_status is RunStatus.PUBLISH_FAILED

    second = make_tracked_clients()
    outcome = resume(runs_dir, failed.run_id, second, mode=PipelineMode.PUBLISH)

    assert outcome.status is PipelineStatus.NOT_RESUMABLE
    assert second.publisher.calls == []


def test_a_failed_run_is_not_resumed(runs_dir: Path, tmp_path: Path) -> None:
    """A Run whose own inputs were rejected has nothing to continue from."""
    from conftest import make_analysis_payload

    clients = make_tracked_clients()
    failed = run_orchestrated(
        runs_dir, tmp_path, clients, analysis=make_analysis_payload(symbol="BTCUSD")
    )
    assert failed.result.run_status is RunStatus.FAILED

    outcome = resume(runs_dir, failed.run_id, make_tracked_clients())

    assert outcome.status is PipelineStatus.NOT_RESUMABLE


# --- integrity ------------------------------------------------------------


def test_a_tampered_artifact_fails_before_the_next_stage(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 27.

    The review no longer matches its recorded digest, so the finalizer must not
    be handed a verdict nobody can vouch for - and its client is never even
    built.
    """
    reviewed = make_reviewed_run(runs_dir, tmp_path)
    tamper(Path(reviewed.run_dir), "gpt_review.json", '{"status": "PASS"}')
    clients = make_tracked_clients()

    outcome = resume(runs_dir, reviewed.run_id, clients)

    assert outcome.status is PipelineStatus.FAILED
    assert outcome.result.final_stage is PipelineStage.FINALIZE
    assert clients.finalizer.calls == []
    assert clients.built == []
    assert not (Path(reviewed.run_dir) / "claude_final.md").exists()


def test_a_resumed_run_leaves_its_sources_untouched(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 53."""
    normalized = make_normalized_run(runs_dir, tmp_path)
    run_dir = Path(normalized.run_dir)
    before = {name: (run_dir / name).read_bytes() for name in ("telegram_input.json", "ohlc.json")}

    resume(runs_dir, normalized.run_id, make_tracked_clients(), mode=PipelineMode.PUBLISH)

    for name, payload in before.items():
        assert (run_dir / name).read_bytes() == payload


# --- lazy provider configuration ------------------------------------------


def test_resuming_at_the_gate_needs_no_ai_clients(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 42.

    Stated the strongest way available: the factories *explode* if called, so a
    passing test proves no credential was ever read.
    """
    finalized = make_finalized_run(runs_dir, tmp_path)
    clients = PipelineClients(
        writer=lambda _selection: exploding_factory("writer")(),
        reviewer=exploding_factory("reviewer"),
        finalizer=lambda _selection: exploding_factory("finalizer")(),
        publisher=exploding_factory("publisher"),
    )

    outcome = resume(runs_dir, finalized.run_id, clients)

    assert outcome.status is PipelineStatus.COMPLETED
    assert outcome.result.run_status is RunStatus.READY_TO_PUBLISH


def test_publishing_an_approved_run_needs_no_ai_clients(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 43."""
    ready = make_published_ready_run(runs_dir, tmp_path)
    publisher = FakePublisherClient()
    clients = PipelineClients(
        writer=lambda _selection: exploding_factory("writer")(),
        reviewer=exploding_factory("reviewer"),
        finalizer=lambda _selection: exploding_factory("finalizer")(),
        publisher=lambda: (publisher, TEST_TARGET_CHAT),
    )

    outcome = resume(runs_dir, ready.run_id, clients, mode=PipelineMode.PUBLISH)

    assert outcome.result.publish_status is PublishStatus.PUBLISHED
    assert publisher.targets == [TEST_TARGET_CHAT]


def test_the_default_mode_needs_no_telegram_configuration(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 44.

    A Telegram token is read when the publisher client is built. In the default
    mode it never is, so an operator generating articles needs no publishing
    credentials at all.
    """
    normalized = make_normalized_run(runs_dir, tmp_path)
    tracked = make_tracked_clients()
    wired = tracked.as_pipeline_clients()
    clients = PipelineClients(
        writer=wired.writer,
        reviewer=wired.reviewer,
        finalizer=wired.finalizer,
        publisher=exploding_factory("publisher"),
    )

    outcome = resume(runs_dir, normalized.run_id, clients)

    assert outcome.result.run_status is RunStatus.READY_TO_PUBLISH


# --- concurrency ----------------------------------------------------------


def test_the_lock_is_held_while_a_stage_runs(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 35.

    Observed from inside the invocation rather than after it: a factory checks
    the lock file at the moment its stage is about to run.
    """
    normalized = make_normalized_run(runs_dir, tmp_path)
    run_dir = Path(normalized.run_dir)
    seen: list[bool] = []
    tracked = make_tracked_clients()
    wired = tracked.as_pipeline_clients()

    def watching_writer(selection: Any = None) -> Any:
        seen.append((run_dir / LOCK_FILENAME).is_file())
        assert wired.writer is not None
        return wired.writer(selection)

    clients = PipelineClients(
        writer=watching_writer, reviewer=wired.reviewer, finalizer=wired.finalizer
    )
    resume(runs_dir, normalized.run_id, clients, mode=PipelineMode.GENERATE_ONLY)

    assert seen == [True]
    assert not (run_dir / LOCK_FILENAME).exists()


def test_a_second_invocation_on_the_same_run_is_refused(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 36."""
    normalized = make_normalized_run(runs_dir, tmp_path)
    clients = make_tracked_clients()

    with RunLock(Path(normalized.run_dir)), pytest.raises(RunLockedError):
        resume(runs_dir, normalized.run_id, clients)

    assert clients.built == []


def test_two_different_runs_do_not_block_each_other(runs_dir: Path, tmp_path: Path) -> None:
    """Requirements 37 and 41: the lock is per-Run, and there is no global one."""
    held = make_normalized_run(runs_dir, tmp_path)
    other = make_normalized_run(runs_dir, tmp_path)

    with RunLock(Path(held.run_dir)):
        outcome = resume(runs_dir, other.run_id, make_tracked_clients())

    assert outcome.status is PipelineStatus.COMPLETED


def test_the_lock_is_released_after_a_failed_invocation(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 39, through the orchestrator rather than the lock alone."""
    from goldpipeline.adapters.fake_writer import FakeWriterClient
    from goldpipeline.domain.errors import WriterTimeoutError

    normalized = make_normalized_run(runs_dir, tmp_path)
    clients = make_tracked_clients(writer=FakeWriterClient(raises=WriterTimeoutError("timeout")))

    outcome = resume(runs_dir, normalized.run_id, clients)

    assert outcome.status is PipelineStatus.FAILED
    assert not (Path(normalized.run_dir) / LOCK_FILENAME).exists()
    # ... and the Run can be picked up again once the cause is fixed.
    assert resume(runs_dir, normalized.run_id, make_tracked_clients()).succeeded


def test_the_lock_is_not_recorded_as_an_artifact(runs_dir: Path, tmp_path: Path) -> None:
    """It is a runtime detail, not part of what the Run produced."""
    normalized = make_normalized_run(runs_dir, tmp_path)
    outcome = resume(runs_dir, normalized.run_id, make_tracked_clients())

    manifest = json.loads((outcome.run_dir / "manifest.json").read_text(encoding="utf-8"))
    names = [ref["name"] for ref in [*manifest["source_files"], *manifest["artifact_files"]]]
    assert LOCK_FILENAME not in names


# --- the publisher's own guard still applies ------------------------------


def test_an_orphan_intent_still_ends_uncertain(runs_dir: Path, tmp_path: Path) -> None:
    """Round 6's crash guard is not bypassed by orchestrating the stage.

    The Run is put back into exactly the state a process killed mid-send leaves:
    an intent on disk, no result. Resuming must reach the publisher's own
    refusal, not the transport.
    """
    ready = make_published_ready_run(runs_dir, tmp_path)
    publish_run(
        run_id=ready.run_id,
        store=RunStore(runs_dir),
        client=FakePublisherClient(),
        target_chat=TEST_TARGET_CHAT,
        sleep=RecordingSleep(),
    )

    run = RunStore(runs_dir).open(ready.run_id)
    (Path(ready.run_dir) / "publish_result.json").unlink()
    manifest = run.load_manifest()
    manifest.artifact_files = [
        ref for ref in manifest.artifact_files if ref.name != "publish_result.json"
    ]
    manifest.status = RunStatus.PUBLISHING
    run.save_manifest(manifest)

    clients = make_tracked_clients()
    outcome = resume(runs_dir, ready.run_id, clients, mode=PipelineMode.PUBLISH)

    assert outcome.status is PipelineStatus.NOT_RESUMABLE
    assert clients.publisher.calls == []
