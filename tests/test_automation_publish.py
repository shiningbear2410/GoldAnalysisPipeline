"""Unattended publishing: three guards, all of which must agree.

This is the part of the system that can do something irreversible without a
person present, so the guards are deliberately redundant:

1. **off by default** - the absence of configuration means no;
2. **an allowlisted channel that must equal the configured one** - enabling
   publishing and naming where it may publish are two separate decisions, so a
   copied environment cannot silently redirect the pipeline;
3. **an age cutoff** - switching automation on must not empty last week's
   approved backlog into the channel at once.

Every test here uses the offline publisher. Nothing in this file can reach
Telegram, and the round it belongs to does not enable publishing for real.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    AUTOMATION_NOW,
    GATE_BLOCKED_ARTICLE,
    age_run,
    event_aged,
    make_published_ready_run,
    make_tracked_clients,
    make_worker_context,
    submit_event,
)

from goldpipeline.config import AutomationSettings
from goldpipeline.domain.errors import (
    AutoPublishNotAllowedError,
    AutoPublishTargetMismatchError,
)
from goldpipeline.schemas.automation import TickStatus, WorkOutcome
from goldpipeline.schemas.manifest import RunStatus
from goldpipeline.services.automation import run_tick
from goldpipeline.services.inbox import Inbox
from goldpipeline.storage.run_store import RunStore

ALLOWED = "@gold_signals_test"


def publishing_context(
    inbox: Inbox,
    runs_dir: Path,
    automation_dir: Path,
    *,
    allowed: str | None = ALLOWED,
    configured: str | None = ALLOWED,
    clients: Any = None,
    **overrides: Any,
) -> Any:
    """A worker with unattended publishing switched on, and an offline transport."""
    return make_worker_context(
        inbox,
        runs_dir,
        automation_dir,
        clients=clients if clients is not None else make_tracked_clients(),
        auto_publish_enabled=True,
        auto_publish_allowed_target=allowed,
        publisher_target=configured,
        **overrides,
    )


# --- off by default --------------------------------------------------------


def test_publishing_is_off_unless_it_is_configured_on() -> None:
    """Requirement 37, at the configuration boundary.

    The absence of a setting means no. Only an explicit affirmative turns it on,
    so a typo or an empty string reads as off.
    """
    assert AutomationSettings().auto_publish_enabled is False
    assert AutomationSettings.from_env({}).auto_publish_enabled is False
    assert (
        AutomationSettings.from_env(
            {"GOLDPIPELINE_AUTOPUBLISH_ENABLED": "maybe"}
        ).auto_publish_enabled
        is False
    )
    assert (
        AutomationSettings.from_env(
            {"GOLDPIPELINE_AUTOPUBLISH_ENABLED": "true"}
        ).auto_publish_enabled
        is True
    )


def test_an_approved_run_is_left_alone_when_publishing_is_off(
    inbox: Inbox, runs_dir: Path, tmp_path: Path, automation_dir: Path
) -> None:
    """Requirements 38 and 47.

    It is finished work, not a problem. The worker leaves it, quietly, however
    many times it looks at it.
    """
    ready = make_published_ready_run(runs_dir, tmp_path)
    clients = make_tracked_clients()
    context = make_worker_context(inbox, runs_dir, automation_dir, clients=clients)

    for minute in range(3):
        result = run_tick(context, now=AUTOMATION_NOW + timedelta(minutes=minute))
        assert not result.did_work
        assert result.errors == []

    assert clients.publisher.calls == []
    assert RunStore(runs_dir).open(ready.run_id).load_manifest().status is (
        RunStatus.READY_TO_PUBLISH
    )


# --- the allowlist ---------------------------------------------------------


def test_enabling_without_an_allowlisted_target_refuses_to_start(
    inbox: Inbox, runs_dir: Path, tmp_path: Path, automation_dir: Path
) -> None:
    """Requirement 39.

    Refused before any work at all: a misconfigured allowlist must not be
    something the worker discovers halfway through a tick.
    """
    make_published_ready_run(runs_dir, tmp_path)
    clients = make_tracked_clients()
    context = publishing_context(inbox, runs_dir, automation_dir, allowed=None, clients=clients)

    with pytest.raises(AutoPublishNotAllowedError) as exc:
        run_tick(context, now=AUTOMATION_NOW)

    assert "GOLDPIPELINE_AUTOPUBLISH_ALLOWED_TARGET" in str(exc.value)
    assert clients.publisher.calls == []


def test_a_target_mismatch_refuses_to_start(
    inbox: Inbox, runs_dir: Path, tmp_path: Path, automation_dir: Path
) -> None:
    """Requirement 40 and golden case G.

    The guard against the accident that matters: an inherited or copied
    environment pointing unattended publishing at a channel nobody authorised.
    """
    make_published_ready_run(runs_dir, tmp_path)
    clients = make_tracked_clients()
    context = publishing_context(
        inbox,
        runs_dir,
        automation_dir,
        allowed="@allowlisted_channel",
        configured="@somewhere_else",
        clients=clients,
    )

    with pytest.raises(AutoPublishTargetMismatchError) as exc:
        run_tick(context, now=AUTOMATION_NOW)

    assert exc.value.code == "AUTO_PUBLISH_TARGET_MISMATCH"
    assert exc.value.details["allowlisted"] == "@allowlisted_channel"
    assert exc.value.details["configured"] == "@somewhere_else"
    assert clients.publisher.calls == []


def test_a_missing_telegram_destination_refuses_to_start(
    inbox: Inbox, runs_dir: Path, automation_dir: Path
) -> None:
    context = publishing_context(inbox, runs_dir, automation_dir, configured=None)

    with pytest.raises(AutoPublishNotAllowedError) as exc:
        run_tick(context, now=AUTOMATION_NOW)

    assert "TELEGRAM_TARGET_CHAT_ID" in str(exc.value)


def test_the_match_must_be_exact(
    inbox: Inbox, runs_dir: Path, tmp_path: Path, automation_dir: Path
) -> None:
    """No normalisation, no prefix matching, no close enough."""
    make_published_ready_run(runs_dir, tmp_path)
    context = publishing_context(
        inbox, runs_dir, automation_dir, allowed="@pcfxsn", configured="@pcfxsn_test"
    )

    with pytest.raises(AutoPublishTargetMismatchError):
        run_tick(context, now=AUTOMATION_NOW)


# --- golden case F: publishing, offline ------------------------------------


def test_a_fresh_approved_run_is_published(
    inbox: Inbox, runs_dir: Path, tmp_path: Path, automation_dir: Path
) -> None:
    """Requirements 41 and 43, and golden case F. Offline throughout."""
    ready = make_published_ready_run(runs_dir, tmp_path)
    age_run(runs_dir, ready.run_id, AUTOMATION_NOW - timedelta(minutes=2))
    clients = make_tracked_clients()
    context = publishing_context(inbox, runs_dir, automation_dir, clients=clients)

    result = run_tick(context, now=AUTOMATION_NOW)

    assert result.auto_publish_enabled is True
    assert result.mode == "PUBLISH"
    assert [item.outcome for item in result.resumed_runs] == [WorkOutcome.PUBLISHED]
    assert len(clients.publisher.calls) == 1
    assert RunStore(runs_dir).open(ready.run_id).load_manifest().status is RunStatus.PUBLISHED


def test_what_is_published_is_what_the_gate_approved(
    inbox: Inbox, runs_dir: Path, tmp_path: Path, automation_dir: Path
) -> None:
    """Requirement 48.

    Automation adds a layer between the gate and the transport, so the
    exact-content invariant is re-proven here rather than assumed.
    """
    ready = make_published_ready_run(runs_dir, tmp_path)
    clients = make_tracked_clients()
    context = publishing_context(inbox, runs_dir, automation_dir, clients=clients)

    run_tick(context, now=AUTOMATION_NOW)
    approved = (Path(ready.run_dir) / "claude_final.md").read_text(encoding="utf-8")

    assert "".join(clients.publisher.sent) == approved


def test_the_destination_comes_only_from_configuration(
    inbox: Inbox, runs_dir: Path, tmp_path: Path, automation_dir: Path
) -> None:
    """Requirement 45 of Round 7, still true with a scheduler on top."""
    make_published_ready_run(runs_dir, tmp_path)
    clients = make_tracked_clients(target_chat=ALLOWED)
    context = publishing_context(inbox, runs_dir, automation_dir, clients=clients)

    run_tick(context, now=AUTOMATION_NOW)

    assert clients.publisher.targets == [ALLOWED]


# --- the age cutoff --------------------------------------------------------


def test_an_old_approved_run_is_not_published(
    inbox: Inbox, runs_dir: Path, tmp_path: Path, automation_dir: Path
) -> None:
    """Requirement 42, and the worst accident this round could enable.

    Someone switches automation on and a backlog of last week's approved
    articles goes out at once. Age is measured from the Run's creation - the
    oldest and most conservative timestamp available.
    """
    ready = make_published_ready_run(runs_dir, tmp_path)
    age_run(runs_dir, ready.run_id, AUTOMATION_NOW - timedelta(days=3))
    clients = make_tracked_clients()
    context = publishing_context(inbox, runs_dir, automation_dir, clients=clients)

    result = run_tick(context, now=AUTOMATION_NOW)

    assert result.status is TickStatus.BLOCKED
    assert result.blocked_runs[0].code == "AUTO_PUBLISH_TOO_OLD"
    assert clients.publisher.calls == []
    assert RunStore(runs_dir).open(ready.run_id).load_manifest().status is (
        RunStatus.READY_TO_PUBLISH
    ), "left exactly where it was, for a human to publish deliberately"


def test_the_cutoff_is_configurable(
    inbox: Inbox, runs_dir: Path, tmp_path: Path, automation_dir: Path
) -> None:
    ready = make_published_ready_run(runs_dir, tmp_path)
    age_run(runs_dir, ready.run_id, AUTOMATION_NOW - timedelta(days=3))
    clients = make_tracked_clients()
    context = publishing_context(
        inbox,
        runs_dir,
        automation_dir,
        clients=clients,
        auto_publish_max_run_age_minutes=10_000,
    )

    result = run_tick(context, now=AUTOMATION_NOW)

    assert [item.outcome for item in result.resumed_runs] == [WorkOutcome.PUBLISHED]


def test_the_default_cutoff_is_conservative() -> None:
    assert AutomationSettings().auto_publish_max_run_age_minutes == 30


# --- blocked and terminal Runs are still refused ---------------------------


def test_a_blocked_decision_is_never_published(
    inbox: Inbox, runs_dir: Path, tmp_path: Path, automation_dir: Path
) -> None:
    """Requirement 44: enabling automation does not soften the gate."""
    from conftest import run_orchestrated

    blocked = run_orchestrated(
        runs_dir,
        tmp_path,
        make_tracked_clients(),
        article=GATE_BLOCKED_ARTICLE,
        enforce_contract=False,
    )
    assert blocked.result.run_status is RunStatus.PUBLISH_BLOCKED

    clients = make_tracked_clients()
    context = publishing_context(inbox, runs_dir, automation_dir, clients=clients)
    result = run_tick(context, now=AUTOMATION_NOW)

    assert not result.did_work
    assert clients.publisher.calls == []


def test_an_uncertain_run_stays_terminal_with_publishing_on(
    inbox: Inbox, runs_dir: Path, tmp_path: Path, automation_dir: Path
) -> None:
    """Requirement 49, and the line no scheduling policy may cross."""
    from conftest import run_orchestrated

    from goldpipeline.adapters.fake_publisher import ambiguous_client
    from goldpipeline.schemas.orchestration import PipelineMode

    uncertain = run_orchestrated(
        runs_dir,
        tmp_path,
        make_tracked_clients(publisher=ambiguous_client()),
        mode=PipelineMode.PUBLISH,
    )
    assert uncertain.result.run_status is RunStatus.PUBLISH_UNCERTAIN
    intent = (Path(uncertain.run_dir) / "publish_intent.json").read_bytes()

    clients = make_tracked_clients()
    context = publishing_context(inbox, runs_dir, automation_dir, clients=clients)
    for minute in range(4):
        run_tick(context, now=AUTOMATION_NOW + timedelta(minutes=minute))

    assert clients.publisher.calls == []
    assert (Path(uncertain.run_dir) / "publish_intent.json").read_bytes() == intent


# --- a full event, published, offline --------------------------------------


def test_a_fresh_event_can_go_all_the_way(
    inbox: Inbox, runs_dir: Path, automation_dir: Path
) -> None:
    """The production shape, rehearsed entirely offline.

    Round 9 implements this path and does not switch it on: no test here, and
    no smoke in this round, publishes for real.
    """
    submit_event(inbox, event_aged(2))
    clients = make_tracked_clients()
    context = publishing_context(inbox, runs_dir, automation_dir, clients=clients)

    result = run_tick(context, now=AUTOMATION_NOW)

    assert [item.outcome for item in result.processed_events] == [WorkOutcome.INGESTED]
    assert [item.outcome for item in result.resumed_runs] == [WorkOutcome.PUBLISHED]
    assert len(clients.publisher.calls) == 1
