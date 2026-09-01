"""Automation contracts: what one scheduled tick did, and what it may do next.

The worker decides *scheduling* and nothing else - when to look, what to look at
first, and when to stop. Every verdict it reports was reached by a stage that
already owned that question.

The type doing the most work here is :class:`RetryClass`. A scheduler firing
every minute is a machine for turning one bad decision into sixty an hour, so
each failure has to be sorted into "try again", "a human must fix the
environment", or "never". Getting a publish outcome into the first bucket would
be the most expensive mistake this package can make, which is why publish
outcomes never reach the classifier at all.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field

from goldpipeline.schemas.common import StrictModel, UtcDatetime, utc_now
from goldpipeline.schemas.runtime_config import ConfigMode

AUTOMATION_SCHEMA_VERSION = "1.1.0"
"""Bumped when a tick record gained configuration provenance.

Additive: every new field has a default, so records written by 1.0.0 still
read. The version moved anyway, because "which fields should I expect in an
incident?" is a question worth being able to answer from the file itself.
"""

BACKOFF_MINUTES = (1, 2, 5, 10, 30)
"""Delay after each successive transient failure, then the item is exhausted.

Bounded rather than unbounded: five attempts spread over three quarters of an
hour is enough to ride out a provider hiccup, and anything still failing after
that is not a hiccup. There is no jitter because there is one worker, so there
is no thundering herd to spread out.
"""

CONFIGURATION_BACKOFF_MINUTES = 30
"""How long to wait before re-checking a missing credential.

Never exhausted: a human fixing the environment should not also have to clear a
retry record. But not one minute either - an absent API key will still be absent
sixty seconds from now, and logging that fact 1,440 times a day is how an
operator learns to ignore the log.
"""


class TickStatus(StrEnum):
    """How one invocation of the worker ended.

    ``BLOCKED`` is deliberately not a failure. A gate declining an article is
    the system working; a scheduler that treated it as an error would light up
    an alert every time the reviewer did its job.
    """

    OK = "OK"
    BLOCKED = "BLOCKED"
    DISABLED = "DISABLED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


class WorkOutcome(StrEnum):
    """What became of one Run or one event during a tick."""

    COMPLETED = "COMPLETED"
    """Reached the ceiling the mode allowed - usually ``READY_TO_PUBLISH``."""

    PUBLISHED = "PUBLISHED"
    INGESTED = "INGESTED"
    BLOCKED = "BLOCKED"
    """A stage declined. Not retried, and not an error."""

    DEFERRED = "DEFERRED"
    """Nothing was wrong with the event; the market was not ready for it."""

    EXPIRED = "EXPIRED"
    """The analysis grew too old to write about while it waited."""

    RETRY_SCHEDULED = "RETRY_SCHEDULED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


class RetryClass(StrEnum):
    """Whether, and how soon, a failure is worth trying again.

    * ``TERMINAL`` - a stage reached a conclusion. Nothing failed, so there is
      nothing to retry. Every publish outcome lands here.
    * ``PERMANENT`` - something is wrong with the Run itself: tampered
      artifacts, unusable inputs. Retrying reproduces it.
    * ``CONFIGURATION`` - the environment is incomplete. A long backoff, and no
      exhaustion, because a human will fix it.
    * ``TRANSIENT`` - a provider timed out or answered badly. Bounded backoff.
    """

    TERMINAL = "TERMINAL"
    PERMANENT = "PERMANENT"
    CONFIGURATION = "CONFIGURATION"
    TRANSIENT = "TRANSIENT"


class WorkItem(StrictModel):
    """One Run or event the worker touched, and what came of it."""

    kind: Literal["run", "event"]
    identifier: str = Field(description="Run id or event id.")
    outcome: WorkOutcome
    code: str | None = Field(
        default=None, description="Safe error or reason code. Never a provider message."
    )
    detail: str | None = Field(default=None, description="Short, safe explanation.")
    next_attempt_at: UtcDatetime | None = None


class AutomationTickResult(StrictModel):
    """Everything one scheduled invocation did.

    Counts and identifiers only. No article text, no provider payloads, no
    prompts - a tick record is written every minute and read during an incident,
    and both of those argue for it staying small.
    """

    schema_version: str = Field(default=AUTOMATION_SCHEMA_VERSION)
    tick_id: str
    started_at: UtcDatetime = Field(default_factory=utc_now)
    completed_at: UtcDatetime = Field(default_factory=utc_now)
    status: TickStatus
    mode: str = Field(description="Pipeline ceiling this tick ran at.")
    auto_publish_enabled: bool = False
    automation_enabled: bool = False

    config_mode: ConfigMode | None = Field(
        default=None, description="How this process resolved its non-secret configuration."
    )
    config_path: str | None = Field(
        default=None, description="The configuration file this tick actually read."
    )
    config_sha256: str | None = Field(
        default=None, description="SHA-256 of that file's bytes. Contains no secret material."
    )
    config_schema_version: str | None = None
    code_version: str | None = Field(
        default=None, description="Build constant, so an audit names configuration *and* code."
    )
    """Provenance, recorded because ``exit 0`` turned out not to mean anything.

    A worker reading no configuration and a worker reading the right one
    produced identical evidence: green history, nothing done. The fingerprint
    settles it - if these fields disagree with the operator's file, the
    scheduler was never reading it.

    Deliberately a fingerprint and not the settings themselves. A tick record is
    written every minute and read during an incident; both argue for keeping it
    small, and the file it names is readable by anyone who needs the values.
    """

    reconciled: list[WorkItem] = Field(default_factory=list)
    resumed_runs: list[WorkItem] = Field(default_factory=list)
    processed_events: list[WorkItem] = Field(default_factory=list)
    deferred_events: list[WorkItem] = Field(default_factory=list)
    expired_events: list[WorkItem] = Field(default_factory=list)
    blocked_runs: list[WorkItem] = Field(default_factory=list)
    errors: list[str] = Field(
        default_factory=list, description="Safe error codes only, never values or messages."
    )

    @property
    def did_work(self) -> bool:
        """Whether anything at all happened this tick."""
        return bool(
            self.reconciled
            or self.resumed_runs
            or self.processed_events
            or self.deferred_events
            or self.expired_events
            or self.blocked_runs
        )


class AutomationState(StrictModel):
    """The operational snapshot in ``automation/state.json``.

    Mutable by design, and the only file in this project that is: it describes
    *now*, not what happened. Run artifacts remain immutable; this is the
    dashboard, and a dashboard that could not be overwritten would be useless.
    """

    schema_version: str = Field(default=AUTOMATION_SCHEMA_VERSION)
    last_tick_id: str | None = None
    last_tick_started_at: UtcDatetime | None = None
    last_tick_completed_at: UtcDatetime | None = None
    last_tick_status: TickStatus | None = None

    events_seen: int = 0
    events_processed: int = 0
    events_deferred: int = 0
    events_expired: int = 0

    runs_resumed: int = 0
    runs_completed: int = 0
    runs_blocked: int = 0

    last_error_safe: str | None = Field(
        default=None, description="Safe code of the last failure. Never a provider message."
    )


class DeferRecord(StrictModel):
    """Why an event is waiting, and when to look at it again.

    Written beside the event rather than into it. The payload the producer wrote
    is evidence, and scheduling metadata is not part of it.
    """

    schema_version: str = Field(default=AUTOMATION_SCHEMA_VERSION)
    event_id: str
    reason_code: str
    reason: str
    deferred_at: UtcDatetime
    next_attempt_at: UtcDatetime
    attempt_count: int = Field(ge=1)

    def due(self, moment: datetime) -> bool:
        """Whether this event may be looked at again yet."""
        return moment >= self.next_attempt_at


class RetryRecord(StrictModel):
    """Backoff state for one Run the worker failed to advance.

    Kept in ``automation/retries/``, outside the Run. A Run's artifacts describe
    what the pipeline produced; how many times a scheduler has tried to nudge it
    is an operational detail with a shorter life than the Run.
    """

    schema_version: str = Field(default=AUTOMATION_SCHEMA_VERSION)
    run_id: str
    attempts: int = Field(default=1, ge=1)
    retry_class: RetryClass
    failure_code: str
    first_failed_at: UtcDatetime
    last_failed_at: UtcDatetime
    next_attempt_at: UtcDatetime
    exhausted: bool = False

    def due(self, moment: datetime) -> bool:
        """Whether this Run may be attempted again yet."""
        return not self.exhausted and moment >= self.next_attempt_at


__all__ = [
    "AUTOMATION_SCHEMA_VERSION",
    "BACKOFF_MINUTES",
    "CONFIGURATION_BACKOFF_MINUTES",
    "AutomationState",
    "AutomationTickResult",
    "DeferRecord",
    "RetryClass",
    "RetryRecord",
    "TickStatus",
    "WorkItem",
    "WorkOutcome",
]
