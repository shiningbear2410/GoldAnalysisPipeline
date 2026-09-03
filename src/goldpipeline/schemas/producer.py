"""What the internal producer is asked for, and what it answers.

The producer turns a request into at most one :class:`AnalysisEvent` in the
durable inbox. Nothing here runs anything, and nothing here is configuration - a
request selects a product mode and a news window, and that is the whole of its
authority. It cannot name a model, a provider, a prompt, a reviewer, a Telegram
destination, a filesystem path or a publish behaviour, because no field for any
of those exists.

**The request is the idempotency key.** ``request_id`` is supplied by the caller
and becomes the event id through one pure function, :func:`event_id_for`. A
caller that retries after a lost acknowledgement sends the same ``request_id``
and gets the same event id, which the ledger already knows how to recognise. No
second identity, no second dedupe system.

**The requested time is the caller's, not the clock's.** ``requested_at`` is the
end of the news window *and* the event's ``created_at``. Collection may take two
seconds or twenty; that is runtime, not product semantics, and letting it into
the payload would mean two retries of the same request produced different bytes
and therefore a conflict where there is no disagreement.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator

from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.common import StrictModel, UtcDatetime
from goldpipeline.schemas.news import (
    DEFAULT_LOOKBACK,
    MAX_LOOKBACK,
    MIN_LOOKBACK,
    CollectionOutcome,
)

PRODUCER_SCHEMA_VERSION = "1"
"""Version of the producer request and result contracts."""

PRODUCER_VERSION = "internal_producer_v1"
"""Recorded on every event this producer writes.

A version, not a feature flag: it says which code shaped the brief, so a Run
written last month can be read against the renderer that wrote it.
"""

PRODUCER_SOURCE = "internal_producer"
"""``AnalysisEvent.source`` for everything this producer submits.

One stable, code-defined name. Deliberately *not* a scraped channel: ``source``
answers "which producer wrote this event", and a channel that happened to supply
one of forty news items did not write anything. Attributing an event to a
channel would also let the sources decide how their own events are labelled.
"""

EVENT_ID_PREFIX = "internal_"
"""Namespace for producer event ids.

Keeps this producer's ids from ever colliding with a remote producer's, and
makes the origin of an id readable in a directory listing at 3am.
"""

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{5,47}$")
"""What a ``request_id`` may look like: 6-48 characters, no separators.

Narrower than the event id pattern it feeds, because a prefix is added and the
result still has to satisfy that pattern. Deliberately excludes whitespace,
slashes, dots-at-the-front and everything else that could turn an id into a path
- and, just as deliberately, excludes free text. A request id is a handle a
caller generates, not a sentence a user typed.
"""

DEFAULT_NEWS_LOOKBACK = DEFAULT_LOOKBACK
MIN_NEWS_LOOKBACK = MIN_LOOKBACK
MAX_NEWS_LOOKBACK = MAX_LOOKBACK
"""The collector's bounds, under this module's names.

Aliases rather than copies. A request that could never be honoured is refused
while it is still only a request, and the numbers it is refused against are the
ones the collector will actually enforce - not a second set that happens to
agree today.
"""

MAX_CLOCK_SKEW = timedelta(minutes=5)
"""How far into the future ``requested_at`` may sit.

A window ending in the future asks for news that has not been published. A few
minutes of tolerance covers an ordinary clock difference between a caller and
this machine; anything beyond it is a wrong clock or a wrong request, and both
are better refused than silently collected against.
"""


class ProducerOutcome(StrEnum):
    """What one production attempt concluded.

    ``ALREADY_SUBMITTED`` is a success, not an error, for the same reason
    ``ALREADY_INGESTED`` is: a caller that retries after a lost acknowledgement
    must get a calm answer naming the original event, or it will retry forever
    and each retry is a candidate second article.
    """

    SUBMITTED = "SUBMITTED"
    ALREADY_SUBMITTED = "ALREADY_SUBMITTED"

    CONFLICT = "CONFLICT"
    """This ``request_id`` already names an event holding different content.

    Never resolved by minting a fresh event id: that would turn "you asked the
    same question twice and got two different answers" into two Runs.
    """

    NEWS_COLLECTION_FAILED = "NEWS_COLLECTION_FAILED"
    """No source produced anything. Explicitly not an empty brief."""

    ARTICLE_TYPE_NOT_READY = "ARTICLE_TYPE_NOT_READY"
    BRIEF_TOO_LARGE = "BRIEF_TOO_LARGE"
    INVALID_REQUEST = "INVALID_REQUEST"

    @property
    def submitted_or_known(self) -> bool:
        """Whether an event for this request exists as a result of - or before - the attempt."""
        return self in (ProducerOutcome.SUBMITTED, ProducerOutcome.ALREADY_SUBMITTED)


class ProducerRequest(StrictModel):
    """One request to produce an analysis event.

    Four fields, and the two that matter are the id and the instant. Everything
    a later bot might want to vary - which model writes it, who reviews it,
    where it is published - is not here and will not be: those are pipeline
    configuration, and an event-generation request has no business carrying
    them.
    """

    schema_version: Literal["1"] = "1"
    request_id: str = Field(description="Caller-supplied idempotency handle. Becomes the event id.")
    requested_at: UtcDatetime = Field(
        description="When the caller asked. The END of the news window, and the event's created_at."
    )
    article_type: ArticleType = ArticleType.ANALYSIS
    news_lookback_seconds: int = Field(
        default=int(DEFAULT_NEWS_LOOKBACK.total_seconds()),
        ge=int(MIN_NEWS_LOOKBACK.total_seconds()),
        le=int(MAX_NEWS_LOOKBACK.total_seconds()),
        description="How far back the news window reaches from requested_at.",
    )

    @field_validator("request_id")
    @classmethod
    def _usable_as_an_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not REQUEST_ID_PATTERN.fullmatch(cleaned):
            raise ValueError(
                "request_id must be 6-48 characters of letters, digits, dot, dash or "
                "underscore, starting with a letter or digit"
            )
        return cleaned

    @field_validator("requested_at")
    @classmethod
    def _aware_utc(cls, value: datetime) -> datetime:
        """Require an explicit offset, then normalize to UTC.

        A naive datetime here would be read as this machine's local time, which
        on a Windows box in Asia/Ho_Chi_Minh puts the news window seven hours
        from where the caller meant it. There is no context at this layer to
        interpret a naive value with, so it is refused rather than guessed.
        """
        if value.tzinfo is None:
            raise ValueError("requested_at must carry an explicit UTC offset")
        return value.astimezone(UTC)

    @property
    def news_lookback(self) -> timedelta:
        return timedelta(seconds=self.news_lookback_seconds)

    @property
    def window_end(self) -> datetime:
        return self.requested_at

    @property
    def window_start(self) -> datetime:
        return self.requested_at - self.news_lookback

    @property
    def event_id(self) -> str:
        return event_id_for(self.request_id)


def event_id_for(request_id: str) -> str:
    """The event id a request maps to. Pure, total and stable.

    One function, called everywhere, so "which event does this request mean?"
    has exactly one answer - including on the retry path, where a second answer
    would mean a second Run.
    """
    return f"{EVENT_ID_PREFIX}{request_id}"


class ProducerResult(StrictModel):
    """The outcome of one production attempt.

    Counts, ids and codes. Never news text: this travels into logs, a CLI, and
    eventually a Telegram reply, and untrusted third-party prose has no business
    in any of them.
    """

    schema_version: Literal["1"] = "1"
    outcome: ProducerOutcome
    request_id: str
    event_id: str | None = None
    run_id: str | None = Field(
        default=None,
        description="Present when the ledger already names a Run for this event.",
    )

    news_outcome: CollectionOutcome | None = None
    coverage_complete: bool | None = Field(
        default=None,
        description="Whether every source reached back past the requested window start.",
    )
    item_count: int | None = Field(
        default=None, ge=0, description="Curated items placed in the brief."
    )
    news_window_seconds: int | None = Field(default=None, ge=1)
    brief_chars: int | None = Field(default=None, ge=0)

    conflict_source: str | None = Field(
        default=None,
        description="Where the differing copy was found: the ledger, or an inbox directory.",
    )
    detail: str | None = Field(default=None, description="Safe operator text. Never news content.")

    @property
    def succeeded(self) -> bool:
        return self.outcome.submitted_or_known


__all__ = [
    "DEFAULT_NEWS_LOOKBACK",
    "EVENT_ID_PREFIX",
    "MAX_CLOCK_SKEW",
    "MAX_NEWS_LOOKBACK",
    "MIN_NEWS_LOOKBACK",
    "PRODUCER_SCHEMA_VERSION",
    "PRODUCER_SOURCE",
    "PRODUCER_VERSION",
    "REQUEST_ID_PATTERN",
    "ProducerOutcome",
    "ProducerRequest",
    "ProducerResult",
    "event_id_for",
]
