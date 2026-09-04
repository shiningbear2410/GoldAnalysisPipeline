"""Pagination capacity: reaching the cutoff, and stopping honestly when it cannot.

The round this file belongs to exists because a live sample disproved an
assumption. ``max_pages_per_source = 6`` was chosen as "roughly 120 messages -
enough for a day on a busy feed"; ``pcnewsfx`` turned out to publish about 1,200
messages a day, so six pages covered roughly two hours of a twenty-four hour
request. The collector said INCOMPLETE, which was honest and useless.

So the property under test throughout is the separation of two ideas:

* the **target** is the requested cutoff, and the walk stops the moment it is
  reached - a high cap must never make a quiet channel fetch more;
* the **caps** are circuit breakers, and hitting one is never evidence of
  coverage however many pages it took.

Offline throughout: pages are generated in-process, no network of any kind.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from goldpipeline.domain.errors import NewsFetchError
from goldpipeline.schemas.news import (
    COVERING_STOPS,
    CollectionOutcome,
    SourceOutcome,
    StopReason,
)
from goldpipeline.services.news_collector import (
    DEFAULT_MAX_PAGES_PER_SOURCE,
    NewsSettings,
    NewsSourceSpec,
    _Budget,
    collect_news,
    collect_source,
)

NOW = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)


def message(mid: int, text: str, when: datetime, channel: str) -> str:
    return (
        f'<div class="tgme_widget_message" data-post="{channel}/{mid}">'
        f'<div class="tgme_widget_message_text">{text}</div>'
        f'<time datetime="{when.isoformat()}"></time></div>'
    )


def page(*blocks: str) -> str:
    return "<html><body>" + "".join(blocks) + "</body></html>"


class VolumeFetcher:
    """Generates pages on demand at a configurable posting rate.

    Models the channel that motivated this round: 20 messages per page, and a
    rate expressed in seconds per message, so a page covers minutes rather than
    hours and a day needs dozens of pages.
    """

    def __init__(
        self,
        *,
        newest_id: int = 100_000,
        per_page: int = 20,
        seconds_per_message: float = 72.0,
        oldest_id: int | None = None,
        fail_after: int | None = None,
        error: Exception | None = None,
    ) -> None:
        self.newest_id = newest_id
        self.per_page = per_page
        self.seconds_per_message = seconds_per_message
        self.oldest_id = oldest_id
        self.fail_after = fail_after
        self.error = error
        self.requests: list[int | None] = []

    def fetch(self, channel: str, *, before: int | None = None) -> str:
        self.requests.append(before)
        if self.fail_after is not None and len(self.requests) > self.fail_after:
            raise self.error or NewsFetchError("boom", channel=channel)

        top = self.newest_id if before is None else before - 1
        blocks = []
        for offset in range(self.per_page):
            mid = top - offset
            if self.oldest_id is not None and mid < self.oldest_id:
                break
            age = (self.newest_id - mid) * self.seconds_per_message
            blocks.append(
                message(
                    mid,
                    f"Gold and the Fed react to development {mid}",
                    NOW - timedelta(seconds=age),
                    channel,
                )
            )
        return page(*blocks) if blocks else "<html><body></body></html>"


def walk(fetcher: object, *, hours: int, **settings_kwargs: object):
    """Run one source walk and return its report."""
    settings = NewsSettings(channels=("pcnewsfx",), **settings_kwargs).validated()  # type: ignore[arg-type]
    return collect_source(
        "pcnewsfx",
        fetcher=fetcher,  # type: ignore[arg-type]
        since=NOW - timedelta(hours=hours),
        settings=settings,
    ).report


# ------------------------------------------------------------ high volume
class TestHighVolumeCoverage:
    def test_six_pages_cannot_cover_a_day(self) -> None:
        """The old fixed cap, shown failing on the channel that exposed it."""
        report = walk(VolumeFetcher(), hours=24, max_pages_per_source=6)
        assert report.pages_fetched == 6
        assert report.covered_window is False
        assert report.stop_reason is StopReason.PAGE_CAP_REACHED
        assert report.outcome is SourceOutcome.INCOMPLETE

    def test_the_walk_continues_past_six_when_allowed(self) -> None:
        assert walk(VolumeFetcher(), hours=24).pages_fetched > 6

    def test_a_day_is_covered_with_the_new_default(self) -> None:
        report = walk(VolumeFetcher(), hours=24)
        assert report.covered_window is True
        assert report.stop_reason is StopReason.CUTOFF_REACHED
        assert report.outcome is SourceOutcome.OK
        assert report.oldest_seen is not None
        assert report.oldest_seen < NOW - timedelta(hours=24)

    def test_it_stops_as_soon_as_the_cutoff_is_passed(self) -> None:
        """Not one page more: the target is coverage, not the cap."""
        report = walk(VolumeFetcher(), hours=24)
        # 1,200 messages a day at 20 a page is 60 pages, plus the boundary
        # page that crosses the cutoff. Well inside the 80-page ceiling,
        # which is the headroom the cap exists to provide.
        assert 59 <= report.pages_fetched <= 62
        assert report.pages_fetched < DEFAULT_MAX_PAGES_PER_SOURCE

    def test_the_cursor_strictly_decreases(self) -> None:
        fetcher = VolumeFetcher()
        walk(fetcher, hours=6)
        cursors = [c for c in fetcher.requests if c is not None]
        assert cursors == sorted(cursors, reverse=True)
        assert len(cursors) == len(set(cursors)), "a cursor was requested twice"

    def test_no_message_is_collected_twice(self) -> None:
        result = collect_source(
            "pcnewsfx",
            fetcher=VolumeFetcher(),
            since=NOW - timedelta(hours=6),
            settings=NewsSettings(channels=("pcnewsfx",)).validated(),
        )
        ids = [item.message_id for item in result.items]
        assert len(ids) == len(set(ids))

    def test_a_cap_below_the_need_still_returns_usable_data(self) -> None:
        report = walk(VolumeFetcher(), hours=24, max_pages_per_source=10)
        assert report.stop_reason is StopReason.PAGE_CAP_REACHED
        assert report.covered_window is False
        assert report.items_in_window > 0

    @pytest.mark.parametrize("hours", [6, 12, 24, 48, 72, 168])
    def test_every_supported_window_terminates(self, hours: int) -> None:
        """A 7-day request on a busy channel may be incomplete, never unbounded."""
        report = walk(VolumeFetcher(), hours=hours)
        assert report.pages_fetched <= DEFAULT_MAX_PAGES_PER_SOURCE
        if not report.covered_window:
            assert report.stop_reason is StopReason.PAGE_CAP_REACHED


# ------------------------------------------------------------- low volume
class TestLowVolumeCoverage:
    """A quiet channel must not be walked to the cap just because it may be."""

    def test_a_quiet_channel_stops_after_the_pages_it_needs(self) -> None:
        report = walk(VolumeFetcher(seconds_per_message=3600.0), hours=6)
        assert report.pages_fetched == 1, "six hours fits on one page at one an hour"
        assert report.covered_window is True
        assert report.stop_reason is StopReason.CUTOFF_REACHED

    def test_a_high_cap_does_not_force_extra_fetches(self) -> None:
        report = walk(VolumeFetcher(seconds_per_message=3600.0), hours=6, max_pages_per_source=500)
        assert report.pages_fetched == 1

    def test_a_short_channel_is_exhausted_not_capped(self) -> None:
        """Nothing older exists, so the window is covered by definition."""
        short = VolumeFetcher(newest_id=30, oldest_id=1, seconds_per_message=60.0)
        report = walk(short, hours=168)
        assert report.stop_reason is StopReason.SOURCE_EXHAUSTED
        assert report.covered_window is True
        assert report.outcome is SourceOutcome.OK


# --------------------------------------------------------- global budget
class TestGlobalBudget:
    def test_the_run_budget_stops_a_source_mid_walk(self) -> None:
        result = collect_source(
            "pcnewsfx",
            fetcher=VolumeFetcher(),
            since=NOW - timedelta(hours=24),
            settings=NewsSettings(channels=("pcnewsfx",)).validated(),
            budget=_Budget(3),
        )
        assert result.report.pages_fetched == 3
        assert result.report.stop_reason is StopReason.GLOBAL_BUDGET_REACHED
        assert result.report.covered_window is False
        assert result.items, "what was already collected stays usable"

    def test_the_budget_is_shared_across_sources(self) -> None:
        """A later source can be starved by an earlier one - visibly, not silently."""
        collection = collect_news(
            fetcher=VolumeFetcher(),
            settings=NewsSettings(
                channels=("pcnewsfx", "ktnews24"), global_page_budget=4, max_items=50
            ),
            now=NOW,
        )
        assert sum(r.pages_fetched for r in collection.sources) == 4
        assert StopReason.GLOBAL_BUDGET_REACHED in {r.stop_reason for r in collection.sources}
        assert collection.outcome is CollectionOutcome.PARTIAL
        assert collection.complete is False

    def test_an_exhausted_budget_never_reports_coverage(self) -> None:
        result = collect_source(
            "pcnewsfx",
            fetcher=VolumeFetcher(),
            since=NOW - timedelta(hours=24),
            settings=NewsSettings(channels=("pcnewsfx",)).validated(),
            budget=_Budget(0),
        )
        assert result.report.pages_fetched == 0
        assert result.report.covered_window is False


# -------------------------------------------------------- per-source caps
class TestPerSourceCaps:
    def test_a_source_specific_cap_overrides_the_default(self) -> None:
        report = walk(
            VolumeFetcher(),
            hours=24,
            max_pages_per_source=60,
            source_specs=(NewsSourceSpec(channel="pcnewsfx", hard_page_cap=4),),
        )
        assert report.pages_fetched == 4
        assert report.stop_reason is StopReason.PAGE_CAP_REACHED

    def test_a_spec_for_another_channel_does_not_apply(self) -> None:
        report = walk(
            VolumeFetcher(seconds_per_message=3600.0),
            hours=6,
            source_specs=(NewsSourceSpec(channel="ktnews24", hard_page_cap=1),),
        )
        assert report.covered_window is True

    def test_shipped_sources_carry_no_guessed_rate(self) -> None:
        """One morning's sample is not a constant worth freezing into code."""
        assert NewsSettings().source_specs == ()


# ------------------------------------------------------- stops & failures
class TestStopReasonsAndFailures:
    def test_a_rate_limit_stops_that_source_without_retrying(self) -> None:
        fetcher = VolumeFetcher(
            fail_after=2,
            error=NewsFetchError("rate limited", channel="pcnewsfx", status_code=429),
        )
        report = walk(fetcher, hours=24)
        assert report.stop_reason is StopReason.RATE_LIMITED
        assert len(fetcher.requests) == 3, "stopped on the first refusal, no retries"
        assert report.items_in_window > 0

    def test_a_timeout_midway_keeps_what_was_collected(self) -> None:
        fetcher = VolumeFetcher(fail_after=2, error=NewsFetchError("timeout", channel="x"))
        report = walk(fetcher, hours=24)
        assert report.stop_reason is StopReason.FETCH_FAILED
        assert report.outcome is SourceOutcome.INCOMPLETE
        assert report.items_in_window > 0

    def test_a_failure_on_the_first_page_is_a_failed_source(self) -> None:
        report = walk(
            VolumeFetcher(fail_after=0, error=NewsFetchError("down", channel="x")), hours=24
        )
        assert report.outcome is SourceOutcome.FAILED
        assert report.pages_fetched == 0

    def test_broken_markup_on_the_first_page_is_a_parse_failure(self) -> None:
        class Broken:
            def fetch(self, channel: str, *, before: int | None = None) -> str:
                return "<html><body>not telegram any more</body></html>"

        report = walk(Broken(), hours=24)
        assert report.stop_reason is StopReason.PARSE_FAILED
        assert report.outcome is SourceOutcome.FAILED

    def test_an_empty_later_page_ends_the_walk_without_claiming_coverage(self) -> None:
        """Probably the end of the channel - but not provably, so not coverage."""

        class RunsOut:
            def __init__(self) -> None:
                self.calls = 0

            def fetch(self, channel: str, *, before: int | None = None) -> str:
                self.calls += 1
                if self.calls == 1:
                    return page(
                        message(500, "Gold and the Fed", NOW - timedelta(minutes=1), "pcnewsfx"),
                        message(499, "Gold and Fed again", NOW - timedelta(minutes=2), "pcnewsfx"),
                    )
                return "<html><body></body></html>"

        report = walk(RunsOut(), hours=24)
        assert report.stop_reason is StopReason.EMPTY_PAGE
        assert report.covered_window is False

    def test_a_repeated_cursor_is_named_as_such(self) -> None:
        same = page(message(700, "Gold and the Fed", NOW - timedelta(minutes=1), "pcnewsfx"))

        class Stuck:
            def fetch(self, channel: str, *, before: int | None = None) -> str:
                return same

        report = walk(Stuck(), hours=24)
        assert report.stop_reason is StopReason.REPEATED_CURSOR
        assert report.covered_window is False
        assert report.pages_fetched <= 2

    @pytest.mark.parametrize(
        "reason",
        [
            StopReason.PAGE_CAP_REACHED,
            StopReason.GLOBAL_BUDGET_REACHED,
            StopReason.REPEATED_CURSOR,
            StopReason.EMPTY_PAGE,
            StopReason.FETCH_FAILED,
            StopReason.RATE_LIMITED,
            StopReason.PARSE_FAILED,
        ],
    )
    def test_only_two_reasons_ever_mean_covered(self, reason: StopReason) -> None:
        assert reason not in COVERING_STOPS


# -------------------------------------------------------- coverage fields
class TestCoverageFields:
    def test_the_report_carries_the_evidence(self) -> None:
        report = walk(VolumeFetcher(seconds_per_message=3600.0), hours=6)
        assert report.requested_start == NOW - timedelta(hours=6)
        assert report.newest_seen is not None
        assert report.oldest_seen is not None
        assert report.oldest_seen <= report.newest_seen
        assert report.items_parsed >= report.items_in_window

    def test_the_same_fixture_produces_the_same_report(self) -> None:
        assert (
            walk(VolumeFetcher(), hours=12).model_dump_json()
            == walk(VolumeFetcher(), hours=12).model_dump_json()
        )

    def test_collection_completeness_follows_the_sources(self) -> None:
        collection = collect_news(
            fetcher=VolumeFetcher(seconds_per_message=3600.0),
            settings=NewsSettings(channels=("ktnews24",), max_items=50),
            now=NOW,
        )
        assert collection.complete is True
        assert collection.covered_from() == collection.window_start


class TestNewsDigestStillNotReady:
    def test_coverage_work_did_not_activate_the_article_type(self) -> None:
        from goldpipeline.schemas.article import ArticleType
        from goldpipeline.services.article_routing import READY_TYPES, spec_for

        # Coverage accounting is not the same as being able to publish a
        # digest. The type was activated later, by the round that wrote the
        # digest writer - never as a side effect of coverage work.
        assert spec_for(ArticleType.TRADE_PLAN).ready is False
        assert ArticleType.TRADE_PLAN not in READY_TYPES
