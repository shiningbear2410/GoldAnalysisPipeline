"""What the news collector produces.

Data only. Nothing in this module runs anything, and nothing in it may be read
as configuration - a :class:`NewsItem` carries what a channel published and
where it came from, never a setting, a model, a destination or an instruction.

**All of it is untrusted.** The text arrives from public channels this pipeline
does not control, and it will contain, sooner or later, a sentence engineered to
read as an order. That is fine as long as it stays *content*: it is marked
untrusted here for the same reason ``raw_analysis.text`` is, so that the fencing
and content-safety layers a later round hands it to cannot forget what it is.

**Absence is recorded, never smoothed over.** A source that failed, a window that
could not be covered, an item whose timestamp would not parse - each has a field
saying so. A collection that quietly returns fewer items than the caller asked
about is indistinguishable from a quiet day, and those are very different facts.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal

from pydantic import Field

from goldpipeline.schemas.common import StrictModel, UtcDatetime

NEWS_SCHEMA_VERSION = "1.0.0"

DEFAULT_LOOKBACK = timedelta(hours=24)
MIN_LOOKBACK = timedelta(hours=1)
MAX_LOOKBACK = timedelta(days=7)
"""The one authority on how far back a news window may reach.

Three modules need these bounds - the collector that walks the pages, the
producer request that names a window, and the preference that remembers one -
and until they lived here there were two copies with a comment on each saying it
matched the other. That is the arrangement where the third copy disagrees.

Seven days is the ceiling for a concrete reason: a preview page holds twenty
messages, so a month of a busy channel is dozens of pages of news nobody will
read against this morning's candles. One hour is the floor because a shorter
window is a question about the last few posts, not about the market.
"""

MAX_ITEM_CHARS = 4000
"""Hard ceiling on one stored item's normalized text.

Well above a Telegram post that says anything, and far below the size at which a
single hostile message could dominate a collection.
"""


class NewsCategory(StrEnum):
    """Why an item might matter to gold.

    A taxonomy rather than a keyword list: each member owns its own terms and
    weight, so a term can be added or re-weighted in one place and tested on its
    own. A flat list of two hundred words is untestable and, in practice,
    unmaintained.
    """

    GOLD = "GOLD"
    MONETARY_POLICY = "MONETARY_POLICY"
    INFLATION = "INFLATION"
    USD_DXY = "USD_DXY"
    TREASURY_YIELDS = "TREASURY_YIELDS"
    ETF_FLOWS = "ETF_FLOWS"
    GEOPOLITICS_RISK = "GEOPOLITICS_RISK"
    US_MACRO = "US_MACRO"
    CENTRAL_BANKS = "CENTRAL_BANKS"


class SourceOutcome(StrEnum):
    """How one channel's collection ended."""

    OK = "OK"
    """Every requested page was fetched and parsed."""

    INCOMPLETE = "INCOMPLETE"
    """Usable items, but the window was not fully covered - page cap or a gap."""

    FAILED = "FAILED"
    """Nothing usable came back from this channel."""


class CollectionOutcome(StrEnum):
    """How the collection as a whole ended."""

    OK = "OK"
    PARTIAL = "PARTIAL"
    """At least one source worked and at least one did not."""

    FAILED = "FAILED"
    """No source produced anything. Explicitly not an empty success."""


class StopReason(StrEnum):
    """Why pagination stopped walking one channel backwards.

    Closed, because "did we cover the window?" is answered from this and not
    from a page count. Fetching forty pages proves effort, not coverage; only
    two of these reasons are evidence that the requested history was reached.
    """

    CUTOFF_REACHED = "CUTOFF_REACHED"
    """Walked past the requested start. The window is genuinely covered."""

    SOURCE_EXHAUSTED = "SOURCE_EXHAUSTED"
    """The channel has no older messages. Covered by definition - there is no
    more history to miss."""

    PAGE_CAP_REACHED = "PAGE_CAP_REACHED"
    """This source's circuit breaker tripped before the cutoff."""

    GLOBAL_BUDGET_REACHED = "GLOBAL_BUDGET_REACHED"
    """The run's total request budget was spent, possibly by other sources."""

    REPEATED_CURSOR = "REPEATED_CURSOR"
    """The next page did not move backwards. Telegram answers a ``before`` past
    the beginning with the same page, so this is how a walk would loop."""

    EMPTY_PAGE = "EMPTY_PAGE"
    """A later page carried no messages. Probably the end of the channel, but
    not provably so, which is why it does not count as coverage."""

    FETCH_FAILED = "FETCH_FAILED"
    RATE_LIMITED = "RATE_LIMITED"
    """The server asked us to stop. Honoured immediately, without retrying."""

    PARSE_FAILED = "PARSE_FAILED"
    """The first page did not parse - the markup is no longer what we read."""


COVERING_STOPS = frozenset({StopReason.CUTOFF_REACHED, StopReason.SOURCE_EXHAUSTED})
"""The only two reasons that count as having covered the window.

Kept as a set beside the enum so the rule lives in one place: coverage is a
property of *why* the walk ended, never of how much it fetched.
"""


class NewsItem(StrictModel):
    """One published message, normalized. UNTRUSTED content throughout."""

    channel: str = Field(description="Configured source channel. Never read from page text.")
    message_id: int = Field(ge=1)
    url: str = Field(description="Canonical permalink, derived from channel and message id.")
    published_at: UtcDatetime = Field(description="Publication time, in UTC.")
    text: str = Field(max_length=MAX_ITEM_CHARS, description="Normalized text. UNTRUSTED.")
    trust_level: Literal["UNTRUSTED"] = "UNTRUSTED"

    relevance_score: float = Field(default=0.0, ge=0.0)
    matched_categories: list[NewsCategory] = Field(default_factory=list)

    corroborating_channels: list[str] = Field(
        default_factory=list,
        description="Other channels that carried the same story, after deduplication.",
    )
    duplicate_count: int = Field(
        default=0, ge=0, description="How many further copies were merged into this item."
    )

    @property
    def source_count(self) -> int:
        """Distinct channels carrying this story, this one included.

        Counted in channels, not messages, so a chatty channel repeating itself
        does not look like independent confirmation.
        """
        return 1 + len(self.corroborating_channels)


class SourceReport(StrictModel):
    """What happened when one channel was collected."""

    channel: str
    outcome: SourceOutcome
    pages_fetched: int = Field(default=0, ge=0)
    items_parsed: int = Field(default=0, ge=0)
    items_in_window: int = Field(default=0, ge=0)
    items_skipped: int = Field(
        default=0, ge=0, description="Parsed but unusable - an unreadable timestamp, say."
    )
    covered_window: bool = Field(
        default=False,
        description="Whether pagination reached back past the requested start.",
    )
    stop_reason: StopReason | None = Field(
        default=None, description="Why the walk ended. The basis for covered_window."
    )
    requested_start: UtcDatetime | None = Field(
        default=None, description="The cutoff this source was asked to reach."
    )
    newest_seen: UtcDatetime | None = None
    oldest_seen: UtcDatetime | None = Field(
        default=None,
        description="Oldest message observed, in or out of the window. How far back we got.",
    )
    error_code: str | None = Field(
        default=None, description="Safe code. Never a provider message or a URL."
    )


class NewsCollection(StrictModel):
    """The result of one collection pass."""

    schema_version: str = Field(default=NEWS_SCHEMA_VERSION)
    collected_at: UtcDatetime
    window_start: UtcDatetime
    window_end: UtcDatetime
    lookback_seconds: int = Field(ge=1)

    outcome: CollectionOutcome
    items: list[NewsItem] = Field(default_factory=list)
    sources: list[SourceReport] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def window(self) -> timedelta:
        return timedelta(seconds=self.lookback_seconds)

    @property
    def complete(self) -> bool:
        """Whether every source covered the whole requested window.

        The honest question a caller has to ask before saying "no news": an
        incomplete collection and a quiet market look identical in the item
        count alone.
        """
        return bool(self.sources) and all(s.covered_window for s in self.sources)

    def covered_from(self) -> datetime | None:
        """The earliest point the collection can actually speak about."""
        if not self.sources:
            return None
        return self.window_start if self.complete else None


class CuratedItem(StrictModel):
    """One item as a later prompt would see it: clipped, and honest about it."""

    channel: str
    message_id: int
    published_at: UtcDatetime
    text: str
    text_truncated: bool = False
    relevance_score: float = Field(ge=0.0)
    matched_categories: list[NewsCategory] = Field(default_factory=list)
    source_count: int = Field(ge=1)
    trust_level: Literal["UNTRUSTED"] = "UNTRUSTED"


class CuratedNews(StrictModel):
    """A bounded subset of a collection, sized for a prompt.

    Separate from :class:`NewsCollection` on purpose. The collection is the
    record of what was published; this is a selection made for one purpose, and
    conflating them would mean the evidence changed shape whenever the prompt
    budget did.
    """

    schema_version: str = Field(default=NEWS_SCHEMA_VERSION)
    items: list[CuratedItem] = Field(default_factory=list)
    item_limit: int = Field(ge=1)
    chars_per_item: int = Field(ge=1)
    omitted_count: int = Field(
        default=0, ge=0, description="Items ranked below the cut. Recorded, never silent."
    )
    truncated_count: int = Field(default=0, ge=0)

    @property
    def truncated(self) -> bool:
        return bool(self.omitted_count or self.truncated_count)


__all__ = [
    "DEFAULT_LOOKBACK",
    "MAX_ITEM_CHARS",
    "MAX_LOOKBACK",
    "MIN_LOOKBACK",
    "NEWS_SCHEMA_VERSION",
    "CollectionOutcome",
    "CuratedItem",
    "CuratedNews",
    "NewsCategory",
    "NewsCollection",
    "NewsItem",
    "COVERING_STOPS",
    "SourceOutcome",
    "SourceReport",
    "StopReason",
]
