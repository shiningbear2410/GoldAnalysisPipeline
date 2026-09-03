"""The internal producer: a request in, at most one inbox event out.

    ProducerRequest -> news collection -> producer brief -> AnalysisEvent
                    -> Inbox.submit()

That arrow stops at the inbox, and the stop is the design. This module calls no
writer, no reviewer, no finalizer, no gate, no publisher and no Telegram API. It
does not create a Run and it does not touch MetaTrader. Everything after the
event lands is the automation worker's job - the same path a remote producer's
event takes, and the same path that has been running in production.

**Why no MT5 here.** The pipeline already fetches authoritative closed candles
when it ingests the event. A producer that fetched its own would give the Run
two technical snapshots taken at different instants, and the first time they
disagreed someone would have to work out which one the article was written
from. There is one authoritative technical read, and it is not this one.

**Why the news window is the caller's instant, not the collector's.** The
window ends at ``requested_at`` and the event's ``created_at`` is that same
instant. Collection runtime is not product semantics: a retry two seconds later
must render the same bytes, or the ledger would see a conflict where the caller
only ever asked one question. Nothing ephemeral reaches the payload - not the
collection's own ``collected_at``, not a page count, not a duration.

**Order matters more than any single check.** A request is validated, then the
article type is refused, and only then is anything fetched. An unfinished mode
costs no request to a public server and leaves no file anywhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any, Protocol

from goldpipeline.adapters.telegram_preview import HttpPreviewFetcher, NewsPageFetcher
from goldpipeline.domain.errors import (
    ArticleTypeNotReadyError,
    InboxPayloadError,
    ProducerRequestError,
)
from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.common import utc_now
from goldpipeline.schemas.inbox import AnalysisEvent
from goldpipeline.schemas.news import CollectionOutcome, CuratedNews, NewsCollection
from goldpipeline.schemas.producer import (
    MAX_CLOCK_SKEW,
    PRODUCER_SOURCE,
    PRODUCER_VERSION,
    ProducerOutcome,
    ProducerRequest,
    ProducerResult,
)
from goldpipeline.schemas.telegram import MAX_RAW_TEXT_CHARS
from goldpipeline.services.admission import AdmissionState
from goldpipeline.services.admission import resolve as resolve_admission
from goldpipeline.services.article_routing import require_ready
from goldpipeline.services.inbox import Inbox, Ledger
from goldpipeline.services.news_collector import NewsSettings, collect_news, curate
from goldpipeline.services.producer_brief import PRODUCER_BRIEF_VERSION, render_brief

logger = logging.getLogger(__name__)

PRODUCER_ARTICLE_TYPES = frozenset({ArticleType.ANALYSIS})
"""Modes this producer knows how to build a brief for.

Not a second readiness table - readiness is
:func:`~goldpipeline.services.article_routing.require_ready`'s to decide, and it
is consulted first. This is a narrower statement: *what this producer can
construct*. A news brief plus market facts is an ANALYSIS; the day NEWS_DIGEST
becomes ready it will want a different document, and the producer should say so
rather than quietly shipping an analysis brief under a digest's name.
"""


class NewsCollector(Protocol):
    """Something that can collect a news window.

    A protocol rather than the collector function itself, so the producer can be
    tested end to end - request through inbox file - with no HTTP, no fixtures
    on disk and no waiting.
    """

    def collect(self, *, window_end: datetime, lookback: timedelta) -> NewsCollection: ...


@dataclass(frozen=True)
class LiveNewsCollector:
    """The real collector, bound to the configured channels.

    Holds settings rather than reading them, so the caller decides. The window
    is the request's, and it overrides whatever lookback the settings carry -
    the request is the thing being answered.
    """

    settings: NewsSettings = field(default_factory=NewsSettings)
    fetcher: NewsPageFetcher | None = None

    def collect(self, *, window_end: datetime, lookback: timedelta) -> NewsCollection:
        fetcher = self.fetcher if self.fetcher is not None else HttpPreviewFetcher()
        return collect_news(
            fetcher=fetcher,
            settings=replace(self.settings, lookback=lookback),
            now=window_end,
        )


def build_request(
    *,
    request_id: str,
    requested_at: datetime,
    article_type: ArticleType = ArticleType.ANALYSIS,
    lookback: timedelta,
) -> ProducerRequest:
    """Construct a request, turning schema complaints into one pipeline error.

    Raises:
        ProducerRequestError: The inputs do not form a request. Nothing has been
            collected and nothing written at this point, which is the reason
            this is a distinct step rather than a branch inside
            :func:`produce`.
    """
    try:
        return ProducerRequest(
            request_id=request_id,
            requested_at=requested_at,
            article_type=article_type,
            news_lookback_seconds=int(lookback.total_seconds()),
        )
    except ValueError as exc:
        raise ProducerRequestError(f"producer request is not usable: {exc}") from exc


def produce(
    request: ProducerRequest,
    *,
    collector: NewsCollector,
    inbox: Inbox,
    ledger: Ledger,
    now: datetime | None = None,
) -> ProducerResult:
    """Turn one request into at most one inbox event.

    Never raises for a routine refusal. An unfinished article type, a failed
    collection, a repeated request and a conflicting one are all outcomes on the
    returned :class:`ProducerResult`, because a bot on the other end has to say
    something useful about each of them. Genuinely exceptional conditions - a
    damaged ledger entry, a disk that will not write - still raise.
    """
    invalid = _reject_unusable(request, now=now or utc_now())
    if invalid is not None:
        return invalid

    not_ready = _reject_unready_type(request)
    if not_ready is not None:
        return not_ready

    collection = collector.collect(window_end=request.window_end, lookback=request.news_lookback)
    if collection.outcome is CollectionOutcome.FAILED:
        # No source produced anything. An article written from an empty brief
        # would be indistinguishable from one written about a quiet market, and
        # only one of those is true.
        logger.warning("producer.collection_failed request=%s", request.request_id)
        return ProducerResult(
            outcome=ProducerOutcome.NEWS_COLLECTION_FAILED,
            request_id=request.request_id,
            news_outcome=collection.outcome,
            coverage_complete=collection.complete,
            item_count=0,
            news_window_seconds=request.news_lookback_seconds,
            detail="no news source produced anything for the requested window",
        )

    curated = curate(collection)
    brief = render_brief(request, collection, curated)

    too_large = _reject_oversized(request, collection, curated, brief)
    if too_large is not None:
        return too_large

    event = _build_event(request, collection, curated, brief)
    payload = event.model_dump(mode="json")

    return _submit(request, event, payload, collection, curated, brief, inbox=inbox, ledger=ledger)


# --------------------------------------------------------------------------
# refusals, in the order they are cheapest to make
# --------------------------------------------------------------------------


def _reject_unusable(request: ProducerRequest, *, now: datetime) -> ProducerResult | None:
    """Refuse a request whose window could not have happened yet.

    The schema has already checked the shape; what it cannot check is the
    request against this machine's clock. A window ending in the future asks for
    news nobody has published, and collecting against it would return a window's
    worth of nothing that looks exactly like a quiet market.

    A window ending in the *past* is not refused here. It produces an event the
    worker may then expire as stale, which is that layer's decision to make with
    its own configured limit - and duplicating the limit here would give the
    system two answers to "how old is too old".
    """
    ahead = request.requested_at - now
    if ahead > MAX_CLOCK_SKEW:
        return ProducerResult(
            outcome=ProducerOutcome.INVALID_REQUEST,
            request_id=request.request_id,
            news_window_seconds=request.news_lookback_seconds,
            detail=(
                f"requested_at is {int(ahead.total_seconds())}s in the future, past the "
                f"{int(MAX_CLOCK_SKEW.total_seconds())}s tolerance"
            ),
        )
    return None


def _reject_unready_type(request: ProducerRequest) -> ProducerResult | None:
    """Refuse an article type that cannot be produced, before anything is fetched.

    Two gates, and they answer different questions. ``require_ready`` is the
    pipeline's authority on whether a mode has an implementation at all;
    :data:`PRODUCER_ARTICLE_TYPES` says whether *this* producer knows what
    document that mode would need. Neither substitutes ANALYSIS for the mode
    that was asked for - writing a different kind of article than the one
    requested is precisely the silent failure the routing table exists to
    prevent.
    """
    try:
        require_ready(request.article_type)
    except ArticleTypeNotReadyError as exc:
        logger.info("producer.article_type_not_ready type=%s", request.article_type)
        return ProducerResult(
            outcome=ProducerOutcome.ARTICLE_TYPE_NOT_READY,
            request_id=request.request_id,
            news_window_seconds=request.news_lookback_seconds,
            detail=exc.message,
        )

    if request.article_type not in PRODUCER_ARTICLE_TYPES:
        return ProducerResult(
            outcome=ProducerOutcome.ARTICLE_TYPE_NOT_READY,
            request_id=request.request_id,
            news_window_seconds=request.news_lookback_seconds,
            detail=(
                f"the internal producer builds no brief for {request.article_type}; "
                "it produces " + ", ".join(sorted(str(t) for t in PRODUCER_ARTICLE_TYPES))
            ),
        )
    return None


def _reject_oversized(
    request: ProducerRequest,
    collection: NewsCollection,
    curated: CuratedNews,
    brief: str,
) -> ProducerResult | None:
    """Refuse a brief the event schema would not accept.

    Belt and braces: the curation bounds make this unreachable today - a test
    pins the worst case at well under the ceiling - and it is here anyway
    because the alternative to failing now is failing inside
    :meth:`Inbox.submit`, after the caller has been told a collection happened.
    Nothing is trimmed to fit: dropping bytes outside the deterministic curation
    policy would mean the brief no longer says what was collected.
    """
    if len(brief) <= MAX_RAW_TEXT_CHARS:
        return None

    logger.error("producer.brief_too_large request=%s chars=%d", request.request_id, len(brief))
    return ProducerResult(
        outcome=ProducerOutcome.BRIEF_TOO_LARGE,
        request_id=request.request_id,
        news_outcome=collection.outcome,
        coverage_complete=collection.complete,
        item_count=len(curated.items),
        news_window_seconds=request.news_lookback_seconds,
        brief_chars=len(brief),
        detail=f"brief is {len(brief)} characters, limit is {MAX_RAW_TEXT_CHARS}",
    )


# --------------------------------------------------------------------------
# the event
# --------------------------------------------------------------------------


def _build_event(
    request: ProducerRequest,
    collection: NewsCollection,
    curated: CuratedNews,
    brief: str,
) -> AnalysisEvent:
    """Assemble the event. Deterministic in the request and the collection.

    ``created_at`` is ``requested_at`` deliberately. It is when the caller
    asked, which is what "this analysis is ninety minutes old" should mean to
    the worker's staleness check, and it is stable across retries - a wall-clock
    ``now()`` here would give the same request different bytes every time and
    turn every retry into a conflict.
    """
    return AnalysisEvent(
        source=PRODUCER_SOURCE,
        event_id=request.event_id,
        created_at=request.requested_at,
        raw_text=brief,
        article_type=request.article_type,
        metadata=_metadata(request, collection, curated),
    )


def _metadata(
    request: ProducerRequest, collection: NewsCollection, curated: CuratedNews
) -> dict[str, Any]:
    """Stable provenance, and nothing else.

    Every value is derived from the request or the collection, so it is the same
    on a retry. Nothing here is read as configuration anywhere in the pipeline -
    the inbox schema carries metadata through to the Run as data, and the
    routing table a producer would have to reach lives in application code - so
    this cannot select a model, a prompt, a destination or a path. It exists so
    that a Run can be traced back to the request that caused it.
    """
    return {
        "producer_version": PRODUCER_VERSION,
        "producer_brief_version": PRODUCER_BRIEF_VERSION,
        "request_id": request.request_id,
        "news_window_seconds": request.news_lookback_seconds,
        "news_collection_outcome": str(collection.outcome),
        "news_coverage_complete": collection.complete,
        "news_item_count": len(curated.items),
    }


def _submit(
    request: ProducerRequest,
    event: AnalysisEvent,
    payload: dict[str, Any],
    collection: NewsCollection,
    curated: CuratedNews,
    brief: str,
    *,
    inbox: Inbox,
    ledger: Ledger,
) -> ProducerResult:
    """Place the event in the inbox, unless it is already known.

    The identity check is the shared one in
    :mod:`goldpipeline.services.admission` - the same ledger entry and the same
    payload digest the ingestion service uses. There is no producer-side
    journal, because there is nothing a journal would know that the ledger does
    not.
    """

    def answer(outcome: ProducerOutcome, **extra: Any) -> ProducerResult:
        return ProducerResult(
            outcome=outcome,
            request_id=request.request_id,
            event_id=event.event_id,
            news_outcome=collection.outcome,
            coverage_complete=collection.complete,
            item_count=len(curated.items),
            news_window_seconds=request.news_lookback_seconds,
            brief_chars=len(brief),
            **extra,
        )

    admission = resolve_admission(payload, event_id=event.event_id, inbox=inbox, ledger=ledger)

    if admission.state is AdmissionState.DUPLICATE:
        logger.info("producer.duplicate request=%s event=%s", request.request_id, event.event_id)
        return answer(
            ProducerOutcome.ALREADY_SUBMITTED,
            run_id=admission.run_id,
            detail=f"an identical event is already recorded ({admission.where})",
        )

    if admission.state is AdmissionState.CONFLICT:
        # The same request id, different bytes. Almost always a source message
        # edited or deleted between the first attempt and the retry. The event
        # already submitted is left exactly as it is: overwriting it would
        # rewrite the input of a Run that may already be published, and minting
        # a fresh event id would produce a second article for one request.
        logger.warning(
            "producer.conflict request=%s event=%s where=%s",
            request.request_id,
            event.event_id,
            admission.where,
        )
        return answer(
            ProducerOutcome.CONFLICT,
            run_id=admission.run_id,
            conflict_source=admission.where,
            detail=(
                "this request_id already names an event with different content; "
                "the existing event is unchanged. Retry under a new request_id."
            ),
        )

    if admission.state is AdmissionState.UNREADABLE_HISTORY:
        return answer(
            ProducerOutcome.CONFLICT,
            conflict_source=admission.where,
            detail=(
                "the ledger entry for this event cannot be read, so it is not known "
                "whether a Run exists. Nothing was submitted; this needs a person."
            ),
        )

    try:
        inbox.submit(payload, event_id=event.event_id)
    except InboxPayloadError:
        # Lost a race with another writer between the check and the submit. The
        # other copy is already waiting, so this is a duplicate, not a fault.
        logger.info("producer.race request=%s event=%s", request.request_id, event.event_id)
        return answer(
            ProducerOutcome.ALREADY_SUBMITTED,
            detail="another writer submitted this event first",
        )

    logger.info(
        "producer.submitted request=%s event=%s items=%d outcome=%s",
        request.request_id,
        event.event_id,
        len(curated.items),
        collection.outcome,
    )
    return answer(ProducerOutcome.SUBMITTED)


__all__ = [
    "PRODUCER_ARTICLE_TYPES",
    "LiveNewsCollector",
    "NewsCollector",
    "build_request",
    "produce",
]
