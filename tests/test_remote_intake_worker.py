"""Remote intake inside a real tick: isolation, and doing nothing when off.

The property under test is not "intake works" - that is
``test_remote_intake.py``. It is that **an optional upstream source cannot harm
the pipeline that does not need it.** A producer which is unreachable,
unauthorised or lying must leave local inbox processing exactly as it was.

Offline throughout: no HTTP, no MT5, no model, no Telegram.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from goldpipeline.config import IngestSettings
from goldpipeline.domain.errors import (
    RemoteIntakeConfigurationError,
    RemoteIntakeResponseError,
    RemoteIntakeTransportError,
)
from goldpipeline.schemas.automation import TickStatus, WorkOutcome
from goldpipeline.services.automation import run_tick
from goldpipeline.services.inbox import INCOMING, Inbox
from tests.conftest import AUTOMATION_NOW, event_aged, make_worker_context

ENABLED = IngestSettings(enabled=True, url="https://producer.example", max_events=10)


def remote_event(event_id: str) -> dict[str, Any]:
    return event_aged(2, event_id=event_id)


class RecordingTransport:
    def __init__(self, batch: list[dict] | None = None, error: Exception | None = None) -> None:
        self.batch = batch or []
        self.error = error
        self.calls = 0

    def fetch_pending(self, *, limit: int) -> list[dict[str, Any]]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return list(self.batch)


class ExplodingFactory:
    """Fails loudly if a disabled feature tries to build anything."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> Any:
        self.calls += 1
        raise AssertionError("transport built while remote intake is disabled")


@pytest.fixture
def inbox(tmp_path: Path) -> Inbox:
    box = Inbox(tmp_path / "inbox")
    box.ensure_layout()
    return box


def tick_with(inbox: Inbox, tmp_path: Path, automation_dir: Path, **over: Any):
    context = make_worker_context(inbox, tmp_path / "runs", automation_dir, max_events_per_tick=5)
    return replace(context, **over)


# ------------------------------------------------------------- disabled
class TestDisabledChangesNothing:
    def test_no_transport_is_built(self, inbox: Inbox, tmp_path: Path, automation_dir: Path):
        factory = ExplodingFactory()
        context = tick_with(
            inbox, tmp_path, automation_dir, ingest=IngestSettings(), event_transport=factory
        )
        result = run_tick(context, now=AUTOMATION_NOW)
        assert factory.calls == 0
        assert result.status is TickStatus.OK

    def test_tick_reports_no_remote_activity(
        self, inbox: Inbox, tmp_path: Path, automation_dir: Path
    ):
        context = tick_with(inbox, tmp_path, automation_dir)
        result = run_tick(context, now=AUTOMATION_NOW)
        assert result.remote_fetch_attempted is False
        assert result.remote_fetch_status is None
        assert result.remote_events_received == 0
        assert result.remote_events_submitted == 0
        assert result.remote_intake == []
        assert result.errors == []

    def test_enabled_without_a_factory_still_does_nothing(
        self, inbox: Inbox, tmp_path: Path, automation_dir: Path
    ):
        """Belt and braces: settings on, wiring absent, no attempt."""
        context = tick_with(inbox, tmp_path, automation_dir, ingest=ENABLED, event_transport=None)
        result = run_tick(context, now=AUTOMATION_NOW)
        assert result.remote_fetch_attempted is False

    def test_local_events_still_process_normally(
        self, inbox: Inbox, tmp_path: Path, automation_dir: Path
    ):
        inbox.submit(remote_event("local_event_00000001"), event_id="local_event_00000001")
        context = tick_with(inbox, tmp_path, automation_dir)
        result = run_tick(context, now=AUTOMATION_NOW)
        assert len(result.processed_events) == 1
        assert result.remote_fetch_attempted is False


# --------------------------------------------------------- failure isolation
class TestFailureIsolation:
    """A broken producer must never stop local work."""

    @pytest.mark.parametrize(
        "error",
        [
            RemoteIntakeTransportError("unreachable", reason="connection"),
            RemoteIntakeTransportError("timed out", reason="timeout"),
            RemoteIntakeConfigurationError("refused", setting="INGEST_TOKEN"),
            RemoteIntakeResponseError("garbage"),
            RuntimeError("something nobody predicted"),
        ],
    )
    def test_local_inbox_still_processes(
        self, inbox: Inbox, tmp_path: Path, automation_dir: Path, error: Exception
    ):
        inbox.submit(remote_event("local_event_00000001"), event_id="local_event_00000001")
        transport = RecordingTransport(error=error)
        context = tick_with(
            inbox,
            tmp_path,
            automation_dir,
            ingest=ENABLED,
            event_transport=lambda: transport,
        )

        result = run_tick(context, now=AUTOMATION_NOW)

        # The local event was ingested despite the remote source failing.
        assert len(result.processed_events) == 1
        assert result.processed_events[0].outcome is WorkOutcome.INGESTED
        # And the failure was recorded rather than swallowed.
        assert result.remote_fetch_attempted is True
        assert result.remote_fetch_status is not None
        assert result.remote_fetch_status != "OK"
        assert result.errors, "a remote failure must leave a safe code behind"

    def test_tick_does_not_fail_because_the_producer_did(
        self, inbox: Inbox, tmp_path: Path, automation_dir: Path
    ):
        transport = RecordingTransport(error=RemoteIntakeTransportError("down", reason="timeout"))
        context = tick_with(
            inbox, tmp_path, automation_dir, ingest=ENABLED, event_transport=lambda: transport
        )
        result = run_tick(context, now=AUTOMATION_NOW)
        assert result.status is TickStatus.OK

    def test_unexpected_exception_is_contained(
        self, inbox: Inbox, tmp_path: Path, automation_dir: Path
    ):
        def factory() -> Any:
            raise ValueError("a third-party stack misbehaved")

        context = tick_with(
            inbox, tmp_path, automation_dir, ingest=ENABLED, event_transport=factory
        )
        result = run_tick(context, now=AUTOMATION_NOW)
        assert result.status is TickStatus.OK
        assert result.remote_fetch_status == "REMOTE_INTAKE_UNEXPECTED"

    def test_no_secret_or_analysis_text_in_the_record(
        self, inbox: Inbox, tmp_path: Path, automation_dir: Path
    ):
        secret = "bearer-token-value-should-never-appear"
        transport = RecordingTransport(
            error=RemoteIntakeTransportError(f"failed talking to {secret}", reason="connection")
        )
        context = tick_with(
            inbox, tmp_path, automation_dir, ingest=ENABLED, event_transport=lambda: transport
        )
        result = run_tick(context, now=AUTOMATION_NOW)
        # Only safe codes are recorded, never provider messages.
        assert secret not in result.model_dump_json()


# ------------------------------------------------------------- happy path
class TestRemoteEventsReachThePipeline:
    def test_fetched_event_is_ingested_in_the_same_tick(
        self, inbox: Inbox, tmp_path: Path, automation_dir: Path
    ):
        transport = RecordingTransport([remote_event("remote_event_00000001")])
        context = tick_with(
            inbox, tmp_path, automation_dir, ingest=ENABLED, event_transport=lambda: transport
        )

        result = run_tick(context, now=AUTOMATION_NOW)

        assert result.remote_fetch_status == "OK"
        assert result.remote_events_received == 1
        assert result.remote_events_submitted == 1
        # Step 0 runs before local processing, so the same tick picks it up.
        assert len(result.processed_events) == 1
        assert result.processed_events[0].identifier == "remote_event_00000001"

    def test_repeated_ticks_admit_one_event_once(
        self, inbox: Inbox, tmp_path: Path, automation_dir: Path
    ):
        """The producer offers the same event on every tick, as it is allowed to."""
        transport = RecordingTransport([remote_event("remote_event_00000001")])
        context = tick_with(
            inbox, tmp_path, automation_dir, ingest=ENABLED, event_transport=lambda: transport
        )

        first = run_tick(context, now=AUTOMATION_NOW)
        assert first.remote_events_submitted == 1

        for _ in range(3):
            later = run_tick(context, now=AUTOMATION_NOW)
            assert later.remote_events_submitted == 0
            assert later.remote_events_duplicate == 1

        assert transport.calls == 4
        assert list(inbox.directory(INCOMING).glob("*.json")) == []

    def test_conflict_is_named_in_the_tick_record(
        self, inbox: Inbox, tmp_path: Path, automation_dir: Path
    ):
        original = remote_event("remote_event_00000001")
        inbox.submit(original, event_id="remote_event_00000001")
        changed = dict(original)
        changed["raw_text"] = "hoan toan khac"

        transport = RecordingTransport([changed])
        context = tick_with(
            inbox, tmp_path, automation_dir, ingest=ENABLED, event_transport=lambda: transport
        )
        result = run_tick(context, now=AUTOMATION_NOW)

        assert result.remote_events_conflict == 1
        conflicts = [w for w in result.remote_intake if w.code == "EVENT_CONFLICT"]
        assert len(conflicts) == 1
        assert conflicts[0].identifier == "remote_event_00000001"
        assert "hoan toan khac" not in (conflicts[0].detail or "")

    def test_invalid_remote_event_is_counted_and_local_work_continues(
        self, inbox: Inbox, tmp_path: Path, automation_dir: Path
    ):
        inbox.submit(remote_event("local_event_00000001"), event_id="local_event_00000001")
        transport = RecordingTransport([{"schema_version": "1", "nonsense": True}])
        context = tick_with(
            inbox, tmp_path, automation_dir, ingest=ENABLED, event_transport=lambda: transport
        )

        result = run_tick(context, now=AUTOMATION_NOW)

        assert result.remote_events_invalid == 1
        assert result.remote_events_submitted == 0
        assert len(result.processed_events) == 1
