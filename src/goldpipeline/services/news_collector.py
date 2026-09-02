"""Collecting gold-relevant news from public Telegram preview pages.

fetch → parse → time-filter → score → deduplicate → rank → bound.

**Data only.** Nothing here calls a model, writes a Run, submits an event or
sends a message. The output is a :class:`NewsCollection`, and a later round
decides what to do with it. Keeping that boundary means this can be developed
and tested against saved HTML with no part of the pipeline running.

**Per-source isolation is the design, not a nicety.** Four channels, and any of
them may be down, renamed, rate-limited or newly unparseable on any given
morning. One failing must cost its own items and nothing else, so every source
is fetched inside its own guard and reports its own outcome. Only if *every*
source fails does the collection fail - and it says ``FAILED`` rather than
returning an empty success, because "no news today" and "we could not look" are
different answers to the same question.

**Coverage is reported, never assumed.** Pagination walks backwards until the
window is covered, the page cap is hit, or the channel runs out. Hitting the cap
leaves a usable collection marked ``INCOMPLETE`` - a caller that wants to say
"nothing happened overnight" needs to know whether anyone actually looked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from goldpipeline.adapters.telegram_preview import (
    NewsPageFetcher,
    RawMessage,
    parse_preview_page,
    validate_channel,
)
from goldpipeline.domain.errors import NewsConfigurationError, NewsError
from goldpipeline.schemas.common import utc_now
from goldpipeline.schemas.news import (
    MAX_ITEM_CHARS,
    CollectionOutcome,
    CuratedItem,
    CuratedNews,
    NewsCollection,
    NewsItem,
    SourceOutcome,
    SourceReport,
)
from goldpipeline.services.news_dedup import DEFAULT_SIMILARITY, deduplicate
from goldpipeline.services.news_taxonomy import (
    DEFAULT_RELEVANCE_THRESHOLD,
    is_relevant,
    score_text,
)

logger = logging.getLogger(__name__)

DEFAULT_CHANNELS: tuple[str, ...] = ("tintucvnws", "pcnewsfx", "ktnews24", "UGLibrary")
"""The configured sources.

Server-side configuration, never read from a page. A channel that appears in
scraped text is content, not a source.
"""

DEFAULT_LOOKBACK = timedelta(hours=24)
MIN_LOOKBACK = timedelta(hours=1)
MAX_LOOKBACK = timedelta(days=7)
"""Bounds on the requested window.

Seven days is the ceiling because a preview page holds twenty messages: asking
for a month means paginating a busy channel dozens of times for news nobody will
read against this morning's candles.
"""

DEFAULT_MAX_PAGES_PER_SOURCE = 6
"""Roughly 120 messages per channel. Enough for a day on a busy feed, and a hard
stop on a channel that would otherwise be walked back for ever."""

DEFAULT_MAX_ITEMS = 400
"""Ceiling on stored items across all sources, before curation."""

DEFAULT_CURATED_ITEMS = 45
DEFAULT_CHARS_PER_ITEM = 650
"""What a later prompt receives. Both recorded on the curated result, so a
reader can tell how much was left out."""


@dataclass(frozen=True)
class NewsSettings:
    """Collector configuration.

    Code-level for now, deliberately. ``REQUIRED_PRODUCTION_KEYS`` is derived
    from ``ConfigKey``, so a new member becomes mandatory the moment it ships and
    would fail the running worker closed before the persisted file could be
    updated. These become persisted settings in the round that activates
    NEWS_DIGEST, as one migration.
    """

    channels: tuple[str, ...] = DEFAULT_CHANNELS
    lookback: timedelta = DEFAULT_LOOKBACK
    max_pages_per_source: int = DEFAULT_MAX_PAGES_PER_SOURCE
    max_items: int = DEFAULT_MAX_ITEMS
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    similarity_threshold: float = DEFAULT_SIMILARITY
    curated_items: int = DEFAULT_CURATED_ITEMS
    chars_per_item: int = DEFAULT_CHARS_PER_ITEM

    def validated(self) -> NewsSettings:
        """Clamp the window and refuse settings that cannot work.

        Raises:
            NewsConfigurationError: No sources, or a nonsensical bound.
        """
        if not self.channels:
            raise NewsConfigurationError("no news sources are configured")
        for channel in self.channels:
            validate_channel(channel)
        if self.max_pages_per_source < 1:
            raise NewsConfigurationError("max_pages_per_source must be at least 1")
        if self.curated_items < 1 or self.chars_per_item < 1:
            raise NewsConfigurationError("curated bounds must be positive")

        window = max(MIN_LOOKBACK, min(MAX_LOOKBACK, self.lookback))
        return replace(self, lookback=window)


# --------------------------------------------------------------------------
# one source
# --------------------------------------------------------------------------


@dataclass
class _SourceResult:
    report: SourceReport
    items: list[NewsItem]


def collect_source(
    channel: str,
    *,
    fetcher: NewsPageFetcher,
    since: datetime,
    settings: NewsSettings,
) -> _SourceResult:
    """Walk one channel backwards until *since* is covered or a bound stops it.

    Never raises for an ordinary failure: the outcome is on the report, because
    a collection with three good sources and one bad one is a useful collection.
    """
    items: list[NewsItem] = []
    pages = 0
    parsed = 0
    skipped = 0
    covered = False
    before: int | None = None
    seen_cursors: set[int] = set()
    error_code: str | None = None

    while pages < settings.max_pages_per_source:
        try:
            html = fetcher.fetch(channel, before=before)
            page = parse_preview_page(html)
        except NewsError as exc:
            error_code = exc.code
            logger.warning("news.source_failed channel=%s code=%s", channel, exc.code)
            break

        pages += 1
        parsed += len(page.messages)

        for raw in page.messages:
            item = _to_item(channel, raw)
            if item is None:
                skipped += 1
                continue
            if item.published_at >= since:
                items.append(item)

        oldest = page.oldest_id
        oldest_time = min(
            (m.published_at for m in page.messages if m.published_at is not None),
            default=None,
        )
        if oldest_time is not None and oldest_time < since:
            covered = True
            break
        if oldest is None or oldest in seen_cursors or oldest <= 1:
            # No cursor, or a page that does not move backwards. Telegram
            # answers a `before` past the beginning with the same page, and a
            # loop that trusted it would run until the cap every time.
            break
        seen_cursors.add(oldest)
        before = oldest

    if error_code is not None and not items:
        outcome = SourceOutcome.FAILED
    elif covered:
        outcome = SourceOutcome.OK
    else:
        outcome = SourceOutcome.INCOMPLETE

    return _SourceResult(
        report=SourceReport(
            channel=channel,
            outcome=outcome,
            pages_fetched=pages,
            items_parsed=parsed,
            items_in_window=len(items),
            items_skipped=skipped,
            covered_window=covered,
            error_code=error_code,
        ),
        items=items,
    )


def _to_item(channel: str, raw: RawMessage) -> NewsItem | None:
    """Turn a parsed message into an item, or drop it for a stated reason.

    Dropped when the timestamp would not parse or the text is empty. Neither is
    recoverable by guessing: an invented time puts a story in the wrong day, and
    an empty item is a photo post with nothing to read.
    """
    if raw.published_at is None or not raw.text.strip():
        return None

    text = raw.text.strip()[:MAX_ITEM_CHARS]
    scored = score_text(text)
    return NewsItem(
        channel=channel,
        message_id=raw.message_id,
        url=f"https://t.me/{channel}/{raw.message_id}",
        published_at=raw.published_at,
        text=text,
        relevance_score=scored.score,
        matched_categories=list(scored.categories),
    )


# --------------------------------------------------------------------------
# the collection
# --------------------------------------------------------------------------


def collect_news(
    *,
    fetcher: NewsPageFetcher,
    settings: NewsSettings | None = None,
    now: datetime | None = None,
) -> NewsCollection:
    """Collect, score, deduplicate and rank news across every configured source."""
    resolved = (settings or NewsSettings()).validated()
    end = now or utc_now()
    since = end - resolved.lookback

    reports: list[SourceReport] = []
    gathered: list[NewsItem] = []
    warnings: list[str] = []

    for channel in resolved.channels:
        result = collect_source(channel, fetcher=fetcher, since=since, settings=resolved)
        reports.append(result.report)
        gathered.extend(result.items)
        if result.report.outcome is SourceOutcome.FAILED:
            warnings.append(f"source {channel} failed: {result.report.error_code}")
        elif not result.report.covered_window:
            warnings.append(f"source {channel} did not cover the requested window")

    relevant = [item for item in gathered if is_relevant_item(item, resolved.relevance_threshold)]
    merged = deduplicate(relevant, threshold=resolved.similarity_threshold)
    ranked = rank(merged)[: resolved.max_items]

    if len(merged) > resolved.max_items:
        warnings.append(f"kept {resolved.max_items} of {len(merged)} items after ranking")

    succeeded = [r for r in reports if r.outcome is not SourceOutcome.FAILED]
    if not succeeded:
        outcome = CollectionOutcome.FAILED
    elif len(succeeded) < len(reports) or any(not r.covered_window for r in reports):
        outcome = CollectionOutcome.PARTIAL
    else:
        outcome = CollectionOutcome.OK

    logger.info(
        "news.collect sources=%d ok=%d items=%d outcome=%s",
        len(reports),
        len(succeeded),
        len(ranked),
        outcome,
    )
    return NewsCollection(
        collected_at=end,
        window_start=since,
        window_end=end,
        lookback_seconds=int(resolved.lookback.total_seconds()),
        outcome=outcome,
        items=ranked,
        sources=reports,
        warnings=warnings,
    )


def is_relevant_item(item: NewsItem, threshold: float) -> bool:
    """Whether an item clears the relevance bar."""
    from goldpipeline.services.news_taxonomy import RelevanceResult

    return is_relevant(
        RelevanceResult(score=item.relevance_score, categories=tuple(item.matched_categories)),
        threshold,
    )


def rank(items: list[NewsItem]) -> list[NewsItem]:
    """Order by importance, deterministically.

    Relevance first, because that is what the reader is here for. Then recency,
    because a fresher telling of an equally relevant story is the more useful
    one. Corroboration breaks ties *after* those two - it is evidence that a
    story is real, not evidence that it matters, and letting it lead would rank
    whatever the most channels happened to repost.

    The final key is ``(channel, message_id)``, which cannot tie, so the order is
    total and two runs over the same input agree exactly.
    """
    return sorted(
        items,
        key=lambda item: (
            -item.relevance_score,
            -item.published_at.timestamp(),
            -item.source_count,
            item.channel,
            item.message_id,
        ),
    )


def curate(
    collection: NewsCollection,
    *,
    item_limit: int = DEFAULT_CURATED_ITEMS,
    chars_per_item: int = DEFAULT_CHARS_PER_ITEM,
) -> CuratedNews:
    """The bounded subset a prompt would receive.

    The collection keeps everything it found; this is a selection for one
    purpose. Both what was dropped and what was clipped are counted, so nothing
    shrinks silently - a prompt built from forty of two hundred items is a
    different piece of evidence from one built from all of them.
    """
    kept = collection.items[:item_limit]
    truncated = 0
    items: list[CuratedItem] = []

    for item in kept:
        text = item.text
        clipped = len(text) > chars_per_item
        if clipped:
            text = text[:chars_per_item]
            truncated += 1
        items.append(
            CuratedItem(
                channel=item.channel,
                message_id=item.message_id,
                published_at=item.published_at,
                text=text,
                text_truncated=clipped,
                relevance_score=item.relevance_score,
                matched_categories=list(item.matched_categories),
                source_count=item.source_count,
            )
        )

    return CuratedNews(
        items=items,
        item_limit=item_limit,
        chars_per_item=chars_per_item,
        omitted_count=max(len(collection.items) - len(kept), 0),
        truncated_count=truncated,
    )


__all__ = [
    "DEFAULT_CHANNELS",
    "DEFAULT_CHARS_PER_ITEM",
    "DEFAULT_CURATED_ITEMS",
    "DEFAULT_LOOKBACK",
    "DEFAULT_MAX_PAGES_PER_SOURCE",
    "MAX_LOOKBACK",
    "MIN_LOOKBACK",
    "NewsSettings",
    "collect_news",
    "collect_source",
    "curate",
    "rank",
]
