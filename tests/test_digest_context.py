"""The deterministic foundation a news digest will be built from.

Round 6.5a. No model, no network, no production wiring - so what these tests
pin is arithmetic and rendering, which is exactly the half of a digest that
should never have depended on a model in the first place.

Three properties carry most of the file.

**Net change is not the range.** They are both distances in price units, and a
piece that reports one under the other's name is wrong in a way no reader can
catch. The corpus keeps a fixture where the two differ by a lot precisely so a
regression that swapped them could not hide behind a coincidence.

**A window that cannot be measured is not a zero.** A weekend, a six-minute
window inside one candle, a series that starts after the boundary - each is a
distinct state with its own sentence, and none of them is "price did not move".

**The window is immutable and never recomputed.** A Run resumed an hour later
must describe the same hours, so nothing here reads the clock.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from goldpipeline.prompts import DEFAULT_DIGEST_WRITER_PROMPT
from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.digest import (
    DIGEST_HARD_CAP_CHARS,
    DIGEST_TARGET_MAX_CHARS,
    DIGEST_TARGET_MIN_CHARS,
    DigestWindow,
    MarketActivity,
    PriceReaction,
    PriceReference,
)
from goldpipeline.schemas.inbox import AnalysisEvent
from goldpipeline.schemas.market import OHLCBar
from goldpipeline.schemas.news import DEFAULT_LOOKBACK, MAX_LOOKBACK, MIN_LOOKBACK
from goldpipeline.schemas.news_digest import DigestSourceItem
from goldpipeline.services.digest_context import (
    NEWS_WINDOW_METADATA_KEY,
    DigestFacts,
    build_digest_facts,
    digest_window_from_event,
)
from goldpipeline.services.digest_render import (
    PRICE_REACTION_HEADING,
    RANGE_LABEL,
    digest_title,
    digest_window_line,
    format_move,
    format_percent,
    render_price_reaction,
)
from goldpipeline.services.price_reaction import (
    PREFERRED_DIGEST_TIMEFRAME,
    calculate_price_reaction,
    close_time,
)

SYMBOL = "XAUUSD"
TF = Timeframe.M5
STEP = timedelta(minutes=5)

BASE = datetime(2026, 9, 4, 6, 0, tzinfo=UTC)
"""13:00 in Vietnam. Far from a date boundary, so a test that is not about
calendar edges cannot accidentally become one."""


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def bar(
    minute: int,
    close: str,
    *,
    high: str | None = None,
    low: str | None = None,
    open_: str | None = None,
    origin: datetime = BASE,
) -> OHLCBar:
    """One M5 candle opening *minute* minutes after *origin*.

    Highs and lows default to a band around the close so that every bar
    satisfies the OHLC invariants without each test having to restate them.
    """
    body = Decimal(close)
    return OHLCBar(
        timestamp=origin + timedelta(minutes=minute),
        open=Decimal(open_) if open_ else body,
        high=Decimal(high) if high else max(body, Decimal(open_) if open_ else body) + 1,
        low=Decimal(low) if low else min(body, Decimal(open_) if open_ else body) - 1,
        close=body,
    )


def series(closes: list[str], *, origin: datetime = BASE) -> list[OHLCBar]:
    """A contiguous M5 series, one bar per close."""
    return [bar(index * 5, value, origin=origin) for index, value in enumerate(closes)]


def ramp(count: int, *, first: int = 100) -> list[OHLCBar]:
    """*count* contiguous M5 bars whose closes rise by one each time."""
    return series([str(first + index) for index in range(count)])


def window(*, start_minute: int, end_minute: int, origin: datetime = BASE) -> DigestWindow:
    """A window expressed in minutes from *origin*, for readability.

    Spans must clear the one-hour floor the news collector sets, so the boundary
    cases below are written as a wide window whose *start* lands where the test
    needs it - which is also what a real digest looks like. A minutes-long
    digest window is not a thing this product has.
    """
    start = origin + timedelta(minutes=start_minute)
    end = origin + timedelta(minutes=end_minute)
    return DigestWindow(start=start, end=end, lookback_seconds=int((end - start).total_seconds()))


def react(bars: list[OHLCBar], win: DigestWindow, *, provider: str | None = None) -> PriceReaction:
    return calculate_price_reaction(
        bars, timeframe=TF, window=win, symbol=SYMBOL, provider=provider
    )


# --------------------------------------------------------------------------
# the window
# --------------------------------------------------------------------------


LOOKBACKS = [
    ("6h", timedelta(hours=6)),
    ("12h", timedelta(hours=12)),
    ("24h default", DEFAULT_LOOKBACK),
    ("48h", timedelta(hours=48)),
    ("72h", timedelta(hours=72)),
    ("7d", timedelta(days=7)),
    ("custom 90m", timedelta(minutes=90)),
    ("lower bound", MIN_LOOKBACK),
    ("upper bound", MAX_LOOKBACK),
]


@pytest.mark.parametrize(("name", "lookback"), LOOKBACKS, ids=[c[0] for c in LOOKBACKS])
def test_every_supported_lookback_builds_a_window(name: str, lookback: timedelta) -> None:
    win = DigestWindow.ending_at(BASE, lookback)

    assert win.end == BASE
    assert win.start == BASE - lookback
    assert win.lookback == lookback


def test_a_lookback_below_the_floor_is_refused() -> None:
    with pytest.raises(ValueError):
        DigestWindow.ending_at(BASE, MIN_LOOKBACK - timedelta(seconds=1))


def test_a_lookback_above_the_ceiling_is_refused() -> None:
    with pytest.raises(ValueError):
        DigestWindow.ending_at(BASE, MAX_LOOKBACK + timedelta(seconds=1))


def test_the_bounds_are_the_news_collector_s_own() -> None:
    """One authority for the range, not a second copy that will drift."""
    import inspect

    from goldpipeline.schemas import digest

    source = inspect.getsource(digest)
    assert "MIN_LOOKBACK" in source
    assert "MAX_LOOKBACK" in source
    assert timedelta(hours=1) == MIN_LOOKBACK
    assert timedelta(days=7) == MAX_LOOKBACK


def test_a_window_is_immutable() -> None:
    win = DigestWindow.ending_at(BASE, DEFAULT_LOOKBACK)

    with pytest.raises(ValueError):
        win.end = BASE + timedelta(hours=1)  # type: ignore[misc]


def test_a_window_whose_fields_disagree_is_refused() -> None:
    """The stored span and the declared lookback can never diverge."""
    with pytest.raises(ValueError):
        DigestWindow(start=BASE - timedelta(hours=24), end=BASE, lookback_seconds=3600)


def test_a_window_must_end_after_it_starts() -> None:
    with pytest.raises(ValueError):
        DigestWindow(start=BASE, end=BASE, lookback_seconds=0)


def test_the_boundaries_are_utc_whatever_the_caller_passed() -> None:
    """Authority is UTC; Vietnam is a rendering concern.

    Built from a genuine +07:00 instant, so the assertion is that the offset was
    normalized away rather than that it was never there.
    """
    saigon = datetime(2026, 9, 4, 13, 0, tzinfo=timezone(timedelta(hours=7)))
    win = DigestWindow.ending_at(saigon, DEFAULT_LOOKBACK)

    assert win.end.utcoffset() == timedelta(0)
    assert win.start.utcoffset() == timedelta(0)
    assert win.end == datetime(2026, 9, 4, 6, 0, tzinfo=UTC)


def test_the_window_is_half_open_at_the_start() -> None:
    """An instant on the boundary belongs to the previous digest, not both."""
    win = window(start_minute=0, end_minute=60)

    assert not win.covers(win.start)
    assert win.covers(win.start + timedelta(seconds=1))
    assert win.covers(win.end)


def test_nothing_in_the_digest_layer_reads_the_clock() -> None:
    """A Run resumed an hour later must describe the same hours.

    Asserted structurally rather than behaviourally: a test that called a
    function twice and got the same answer would pass even if the clock were
    read, as long as it were read quickly.
    """
    import inspect

    from goldpipeline.services import digest_context, digest_render, price_reaction

    for module in (digest_context, digest_render, price_reaction):
        source = inspect.getsource(module)
        assert "utc_now" not in source, module.__name__
        assert "datetime.now" not in source, module.__name__
        assert "date.today" not in source, module.__name__


# --------------------------------------------------------------------------
# the window is read from what the producer already recorded
# --------------------------------------------------------------------------


def event(**overrides: object) -> AnalysisEvent:
    payload: dict[str, object] = {
        "source": "gold_analysis_bot",
        "event_id": "gold-20260904-0600-abc123",
        "created_at": BASE,
        "raw_text": "tin vàng",
    }
    payload.update(overrides)
    return AnalysisEvent.model_validate(payload)


def test_the_window_comes_from_the_producer_s_own_record() -> None:
    """`created_at` is the end; the recorded lookback is the span."""
    win = digest_window_from_event(
        event(metadata={NEWS_WINDOW_METADATA_KEY: int(DEFAULT_LOOKBACK.total_seconds())})
    )

    assert win is not None
    assert win.end == BASE
    assert win.lookback == DEFAULT_LOOKBACK


def test_an_event_recording_no_lookback_yields_no_window() -> None:
    """Every ANALYSIS event, and every hand-submitted one. Absence, not failure."""
    assert digest_window_from_event(event()) is None
    assert digest_window_from_event(event(metadata={"purpose": "smoke"})) is None


def test_a_malformed_lookback_is_refused_rather_than_guessed() -> None:
    """Missing and unusable are different things and get different answers."""
    with pytest.raises(ValueError):
        digest_window_from_event(event(metadata={NEWS_WINDOW_METADATA_KEY: "soon"}))
    with pytest.raises(ValueError):
        digest_window_from_event(event(metadata={NEWS_WINDOW_METADATA_KEY: True}))


def test_an_out_of_range_lookback_is_refused() -> None:
    with pytest.raises(ValueError):
        digest_window_from_event(event(metadata={NEWS_WINDOW_METADATA_KEY: 30}))
    with pytest.raises(ValueError):
        digest_window_from_event(event(metadata={NEWS_WINDOW_METADATA_KEY: 999_999}))


def test_the_producer_and_the_reader_agree_on_the_metadata_key() -> None:
    """An untyped dict key written in one module and read in another."""
    import inspect

    from goldpipeline.services import producer

    assert NEWS_WINDOW_METADATA_KEY in inspect.getsource(producer)


def test_no_second_window_authority_was_created() -> None:
    """The producer request already defines the window; this round reads it."""
    from goldpipeline.schemas.producer import ProducerRequest

    request = ProducerRequest(
        request_id="gold-20260904-0600",
        requested_at=BASE,
        news_lookback_seconds=int(DEFAULT_LOOKBACK.total_seconds()),
    )
    derived = DigestWindow.ending_at(request.window_end, request.news_lookback)

    assert derived.start == request.window_start
    assert derived.end == request.window_end


# --------------------------------------------------------------------------
# the title
# --------------------------------------------------------------------------


def test_a_same_day_window_carries_one_date() -> None:
    win = window(start_minute=0, end_minute=360)

    assert digest_title(win) == "📰 TIN VÀNG 04.09.2026"


def test_a_cross_day_window_shows_both_ends() -> None:
    win = DigestWindow.ending_at(BASE, DEFAULT_LOOKBACK)

    assert digest_title(win) == "📰 TIN VÀNG 03.09 → 04.09.2026"


def test_a_cross_month_window_shows_both_months() -> None:
    end = datetime(2026, 10, 1, 6, 0, tzinfo=UTC)
    win = DigestWindow.ending_at(end, DEFAULT_LOOKBACK)

    assert digest_title(win) == "📰 TIN VÀNG 30.09 → 01.10.2026"


def test_a_cross_year_window_writes_the_year_twice() -> None:
    """The one case where a bare day.month is genuinely ambiguous."""
    end = datetime(2026, 1, 1, 6, 0, tzinfo=UTC)
    win = DigestWindow.ending_at(end, DEFAULT_LOOKBACK)

    assert digest_title(win) == "📰 TIN VÀNG 31.12.2025 → 01.01.2026"


def test_the_title_is_dated_in_vietnam_not_utc() -> None:
    """23:30 UTC on the 3rd is 06:30 on the 4th for the reader.

    The whole point of the rule: the two calendars disagree for the seven hours
    of the morning that matter most, and the reader's own is the one that counts.
    """
    end = datetime(2026, 9, 3, 23, 30, tzinfo=UTC)
    win = DigestWindow.ending_at(end, timedelta(hours=6))

    assert end.strftime("%d") == "03"
    assert digest_title(win).endswith("04.09.2026")


# --------------------------------------------------------------------------
# the window line
# --------------------------------------------------------------------------


def test_the_window_line_is_exact_to_the_minute() -> None:
    end = datetime(2026, 9, 4, 7, 7, tzinfo=UTC)
    win = DigestWindow.ending_at(end, DEFAULT_LOOKBACK)

    assert digest_window_line(win) == "🕐 03/09 14:07 → 04/09 14:07 (giờ VN)"


def test_the_window_line_never_rounds() -> None:
    """A window that ran to 14:07 did not run to 14:00."""
    end = datetime(2026, 9, 4, 7, 7, 43, tzinfo=UTC)
    win = DigestWindow.ending_at(end, timedelta(hours=6))

    assert "14:07" in digest_window_line(win)
    assert "14:00" not in digest_window_line(win)


def test_the_window_line_names_the_timezone() -> None:
    assert digest_window_line(window(start_minute=0, end_minute=60)).endswith("(giờ VN)")


# --------------------------------------------------------------------------
# price reaction: the semantic that matters most
# --------------------------------------------------------------------------


def test_net_change_and_range_are_different_numbers() -> None:
    """The Round 6.4 design's own example, pinned.

    start 4323, end 4428, high 4460, low 4321 → net +105, range 139. The two
    differ by enough that a regression swapping them cannot pass by coincidence.
    """
    bars = [
        bar(0, "4323", high="4325", low="4322"),
        bar(5, "4400", high="4460", low="4321"),
        bar(10, "4428", high="4430", low="4395"),
    ]
    win = window(start_minute=5, end_minute=65)

    result = react(bars, win)

    assert result.market_activity is MarketActivity.NORMAL
    assert result.start_reference is not None
    assert result.start_reference.close == Decimal("4323")
    assert result.end_reference is not None
    assert result.end_reference.close == Decimal("4428")
    assert result.net_change == Decimal("105")
    assert result.price_range == Decimal("139")
    assert result.net_change != result.price_range


def test_a_rising_window_reports_a_positive_change() -> None:
    result = react(series(["100", "101", "102"]), window(start_minute=5, end_minute=65))

    assert result.net_change == Decimal("2")
    assert result.direction == 1


def test_a_falling_window_reports_a_negative_change() -> None:
    result = react(series(["102", "101", "100"]), window(start_minute=5, end_minute=65))

    assert result.net_change == Decimal("-2")
    assert result.direction == -1


def test_a_flat_window_reports_zero_and_says_so() -> None:
    """Zero here is observed, not assumed: both boundaries were measured."""
    result = react(series(["100", "105", "100"]), window(start_minute=5, end_minute=65))

    assert result.net_change == Decimal("0")
    assert result.direction == 0
    assert result.market_activity is MarketActivity.NORMAL


def test_a_start_boundary_exactly_on_a_close_uses_that_candle() -> None:
    """ "At or before", so a candle closing on the boundary is the boundary price."""
    bars = series(["100", "101", "102"])
    win = window(start_minute=5, end_minute=65)

    result = react(bars, win)

    assert result.start_reference is not None
    assert result.start_reference.candle_close_at == win.start
    assert result.start_reference.close == Decimal("100")


def test_a_start_boundary_between_closes_uses_the_earlier_candle() -> None:
    """The last price actually known at the boundary, never the next one."""
    bars = series(["100", "101", "102", "103"])
    win = window(start_minute=7, end_minute=67)

    result = react(bars, win)

    assert result.start_reference is not None
    assert result.start_reference.close == Decimal("100")
    assert result.start_reference.candle_close_at == BASE + timedelta(minutes=5)


def test_an_end_boundary_exactly_on_a_close_uses_that_candle() -> None:
    bars = ramp(14)
    win = window(start_minute=5, end_minute=65)

    result = react(bars, win)

    assert result.end_reference is not None
    assert result.end_reference.candle_close_at == win.end
    assert result.end_reference.close == Decimal("112")


def test_a_forming_candle_is_never_read() -> None:
    """Its close is not yet a fact, and reading it would move after publication."""
    # The last bar opens at 65 and would close at 70 - after the window ends,
    # so it is still forming. Its close and its spike must both be invisible.
    bars = [*ramp(13), bar(65, "999", high="1000", low="998")]
    win = window(start_minute=5, end_minute=67)

    result = react(bars, win)

    assert result.end_reference is not None
    assert result.end_reference.close == Decimal("112")
    assert Decimal("999") not in {result.net_change, result.window_high}
    assert result.window_high != Decimal("1000")


def test_an_end_boundary_inside_a_forming_candle_falls_back_to_the_last_close() -> None:
    bars = ramp(13)
    win = window(start_minute=5, end_minute=67)

    result = react(bars, win)

    assert result.end_reference is not None
    assert result.end_reference.candle_close_at == BASE + timedelta(minutes=65)


# --------------------------------------------------------------------------
# the range inclusion rule
# --------------------------------------------------------------------------


def test_a_candle_entirely_before_the_window_is_excluded_from_the_range() -> None:
    """The start-reference candle must not also donate its high and low.

    It closed at or before the boundary, so it happened before the window. A
    spike inside it belongs to the previous digest.
    """
    bars = [
        bar(0, "100", high="9999", low="1"),
        bar(5, "101", high="102", low="100"),
        bar(10, "102", high="103", low="101"),
    ]
    win = window(start_minute=5, end_minute=65)

    result = react(bars, win)

    assert result.window_high == Decimal("103")
    assert result.window_low == Decimal("100")
    assert result.price_range == Decimal("3")


def test_a_candle_overlapping_the_window_is_included_in_the_range() -> None:
    """A bar still running at the end boundary has observed highs and lows."""
    bars = [
        bar(0, "100", high="101", low="99"),
        bar(5, "101", high="120", low="100"),
        bar(10, "102", high="103", low="101"),
    ]
    win = window(start_minute=7, end_minute=67)

    result = react(bars, win)

    # The bar opening at 5 straddles the start boundary: it opened before the
    # window and closed inside it, so its swing belongs to this window.
    assert result.window_high == Decimal("120")


def test_the_range_never_borrows_the_net_change_s_meaning() -> None:
    """A window where price ended where it began but swung hard in between."""
    bars = [
        bar(0, "100", high="100", low="100", open_="100"),
        bar(5, "150", high="150", low="100"),
        bar(10, "100", high="150", low="100"),
    ]
    win = window(start_minute=5, end_minute=65)

    result = react(bars, win)

    assert result.net_change == Decimal("0")
    assert result.price_range == Decimal("50")


# --------------------------------------------------------------------------
# market activity states
# --------------------------------------------------------------------------


def test_insufficient_history_when_nothing_closed_before_the_start() -> None:
    """The first available candle is not a stand-in for the boundary price."""
    bars = series(["100", "101"])
    win = window(start_minute=-60, end_minute=15)

    result = react(bars, win)

    assert result.market_activity is MarketActivity.INSUFFICIENT_HISTORY
    assert result.start_reference is None
    assert result.net_change is None
    assert result.percent_change is None


def test_no_new_closed_bar_when_the_window_sits_inside_one_candle() -> None:
    """A one-hour window inside a four-hour candle.

    Written on H4 rather than M5 because the digest floor is an hour: a window
    shorter than a single bar is only reachable on a coarse series, which is
    exactly where a real digest would meet this state.
    """
    bars = [
        OHLCBar(
            timestamp=BASE + timedelta(hours=4 * index),
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("99"),
            close=Decimal(100 + index),
        )
        for index in range(2)
    ]
    win = DigestWindow(
        start=BASE + timedelta(hours=5),
        end=BASE + timedelta(hours=6),
        lookback_seconds=3600,
    )

    result = calculate_price_reaction(bars, timeframe=Timeframe.H4, window=win, symbol=SYMBOL)

    assert result.market_activity is MarketActivity.NO_NEW_CLOSED_BAR
    assert result.start_reference is not None
    assert result.net_change is None


def test_a_weekend_window_reports_no_market_activity_not_a_zero_move() -> None:
    """The state the round exists to prevent being reported as ``0``."""
    friday = series(["100", "101", "102"])
    win = window(start_minute=600, end_minute=960)

    result = react(friday, win)

    assert result.market_activity is MarketActivity.NO_MARKET_ACTIVITY
    assert result.net_change is None
    assert result.price_range is None
    assert result.start_reference is not None


def test_a_daily_rollover_gap_is_not_filled() -> None:
    """Observed bars only. No synthetic candle bridges the hole."""
    before = series(["100", "101"])
    after = series(["110", "111"], origin=BASE + timedelta(hours=8))
    win = DigestWindow(
        start=BASE + timedelta(minutes=10),
        end=BASE + timedelta(hours=8, minutes=10),
        lookback_seconds=int(timedelta(hours=8).total_seconds()),
    )

    result = react(before + after, win)

    assert result.market_activity is MarketActivity.NORMAL
    assert result.closed_bars_in_window == 2
    assert result.net_change == Decimal("10")


def test_only_normal_carries_a_measured_change() -> None:
    """The schema refuses a half-filled object, so a state cannot lie."""
    win = window(start_minute=0, end_minute=60)
    ref = PriceReference(candle_open_at=BASE, candle_close_at=BASE + STEP, close=Decimal("100"))

    with pytest.raises(ValueError):
        PriceReaction(
            window=win,
            symbol=SYMBOL,
            timeframe=TF,
            market_activity=MarketActivity.NO_MARKET_ACTIVITY,
            start_reference=ref,
            end_reference=ref,
            net_change=Decimal("0"),
        )


def test_a_net_change_requires_both_boundaries() -> None:
    win = window(start_minute=0, end_minute=60)

    with pytest.raises(ValueError):
        PriceReaction(
            window=win,
            symbol=SYMBOL,
            timeframe=TF,
            market_activity=MarketActivity.NORMAL,
            net_change=Decimal("5"),
        )


# --------------------------------------------------------------------------
# arithmetic quality
# --------------------------------------------------------------------------


def test_percentage_is_decimal_and_unrounded_in_the_domain() -> None:
    bars = [bar(0, "4000"), bar(5, "4020"), bar(10, "4040")]
    win = window(start_minute=5, end_minute=65)

    result = react(bars, win)

    assert isinstance(result.percent_change, Decimal)
    assert result.net_change == Decimal("40")
    assert result.percent_change == (Decimal("40") / Decimal("4000")) * Decimal(100)


def test_no_binary_float_reaches_the_calculation() -> None:
    import inspect

    from goldpipeline.services import price_reaction

    source = inspect.getsource(price_reaction)
    assert "float(" not in source


def test_boundary_metadata_records_which_candles_were_used() -> None:
    bars = series(["100", "101", "102"])
    win = window(start_minute=5, end_minute=65)

    result = react(bars, win)

    assert result.start_reference is not None
    assert result.end_reference is not None
    assert result.start_reference.candle_open_at == BASE
    assert result.start_reference.candle_close_at == BASE + STEP
    assert result.end_reference.candle_open_at == BASE + timedelta(minutes=10)
    assert result.closed_bars_in_window == 2


def test_unsorted_input_is_normalized_not_misread() -> None:
    """The upstream model guarantees order; this must not depend on it."""
    ordered = series(["100", "101", "102"])
    win = window(start_minute=5, end_minute=65)

    assert react(list(reversed(ordered)), win).net_change == react(ordered, win).net_change


def test_a_calendar_timeframe_is_refused_rather_than_guessed() -> None:
    """A month is not a fixed number of seconds, so no close can be derived."""
    with pytest.raises(ValueError):
        close_time(bar(0, "100"), Timeframe.MN1)


# --------------------------------------------------------------------------
# source agnosticism
# --------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["tradingview", "mt5", "file", None])
def test_the_same_bars_give_the_same_answer_whatever_supplied_them(
    provider: str | None,
) -> None:
    bars = series(["100", "101", "102"])
    win = window(start_minute=5, end_minute=65)

    result = react(bars, win, provider=provider)

    assert result.net_change == Decimal("2")
    assert result.provider == provider


def test_the_calculator_imports_no_provider() -> None:
    """Provenance is metadata; it may never become numeric meaning."""
    import ast
    from pathlib import Path

    tree = ast.parse(
        Path("src/goldpipeline/services/price_reaction.py").read_text(encoding="utf-8")
    )
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }

    assert not any("tradingview" in name or "mt5" in name for name in imported)
    assert "MetaTrader5" not in {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def normal_reaction(net: str, span: str = "10") -> PriceReaction:
    """A NORMAL reaction with a chosen net change and range."""
    start = Decimal("4000")
    end = start + Decimal(net)
    low = min(start, end) - Decimal("1")
    return PriceReaction(
        window=window(start_minute=0, end_minute=60),
        symbol=SYMBOL,
        timeframe=TF,
        market_activity=MarketActivity.NORMAL,
        start_reference=PriceReference(
            candle_open_at=BASE - STEP, candle_close_at=BASE, close=start
        ),
        end_reference=PriceReference(
            candle_open_at=BASE + timedelta(minutes=55),
            candle_close_at=BASE + timedelta(minutes=60),
            close=end,
        ),
        window_high=low + Decimal(span),
        window_low=low,
        net_change=Decimal(net),
        price_range=Decimal(span),
        percent_change=(Decimal(net) / start) * Decimal(100),
        closed_bars_in_window=12,
        overlapping_bars=12,
    )


def test_a_rise_is_rendered_as_tang() -> None:
    text = render_price_reaction(normal_reaction("105"))

    assert PRICE_REACTION_HEADING in text
    assert "tăng" in text
    assert "giảm" not in text


def test_a_fall_is_rendered_as_giam() -> None:
    text = render_price_reaction(normal_reaction("-105"))

    assert "giảm" in text
    assert "tăng" not in text


def test_a_flat_window_is_not_called_a_rise() -> None:
    """Magnitude without a sign check is how ``0`` becomes "tăng"."""
    text = render_price_reaction(normal_reaction("0"))

    assert "không đổi" in text
    assert "tăng" not in text
    assert "giảm" not in text


def test_the_range_is_labelled_bien_do_and_the_net_change_is_not() -> None:
    text = render_price_reaction(normal_reaction("105", span="139"))

    assert RANGE_LABEL.capitalize() in text
    move_sentence = text.split(RANGE_LABEL.capitalize())[0]
    assert "105" in move_sentence
    assert RANGE_LABEL not in move_sentence.lower()
    assert "139" in text.split(RANGE_LABEL.capitalize())[1]


def test_the_renderer_never_explains_why() -> None:
    """Observation, not explanation. The two belong to different stages."""
    for net in ("105", "-105", "0"):
        text = render_price_reaction(normal_reaction(net))
        for causal in ("do ", "khiến", "bởi", "nhờ", "vì "):
            assert causal not in text.lower()


def test_the_renderer_never_names_the_provider() -> None:
    reaction = normal_reaction("105").model_copy(update={"provider": "tradingview"})
    text = render_price_reaction(reaction)

    for name in ("TradingView", "OANDA", "MT5", "MetaTrader"):
        assert name.lower() not in text.lower()


def test_every_unmeasurable_state_has_its_own_sentence() -> None:
    """None of them may read as "price did not move"."""
    win = window(start_minute=0, end_minute=60)
    ref = PriceReference(candle_open_at=BASE - STEP, candle_close_at=BASE, close=Decimal("4000"))
    seen = set()

    for state in (
        MarketActivity.NO_NEW_CLOSED_BAR,
        MarketActivity.NO_MARKET_ACTIVITY,
        MarketActivity.INSUFFICIENT_HISTORY,
    ):
        reaction = PriceReaction(
            window=win,
            symbol=SYMBOL,
            timeframe=TF,
            market_activity=state,
            start_reference=None if state is MarketActivity.INSUFFICIENT_HISTORY else ref,
        )
        text = render_price_reaction(reaction)
        assert "không đổi" not in text, state
        assert "tăng" not in text and "giảm" not in text, state
        seen.add(text)

    assert len(seen) == 3, "each state needs its own wording"


# --------------------------------------------------------------------------
# display policy
# --------------------------------------------------------------------------


MOVES = [
    ("105.00", "105"),
    ("105.30", "105.3"),
    ("105.25", "105.25"),
    ("0.125", "0.13"),
    ("-105.00", "105"),
    ("139", "139"),
    ("4451.824", "4451.82"),
]


@pytest.mark.parametrize(("raw", "expected"), MOVES, ids=[m[0] for m in MOVES])
def test_a_move_is_rendered_without_visual_noise(raw: str, expected: str) -> None:
    assert format_move(Decimal(raw)) == expected


def test_a_move_never_carries_its_own_sign() -> None:
    """Direction is the verb's job; "tăng -105" is not a sentence."""
    assert "-" not in format_move(Decimal("-105"))


def test_a_small_move_is_not_rounded_away() -> None:
    assert format_move(Decimal("0.13")) == "0.13"
    assert format_move(Decimal("0.004")) == "0"


def test_a_percentage_carries_an_explicit_sign() -> None:
    assert format_percent(Decimal("0.9832")) == "+0.98%"
    assert format_percent(Decimal("-1.2")) == "-1.20%"


def test_display_never_changes_the_domain_value() -> None:
    reaction = normal_reaction("105.004")

    render_price_reaction(reaction)

    assert reaction.net_change == Decimal("105.004")


# --------------------------------------------------------------------------
# the context seam
# --------------------------------------------------------------------------


def facts(**overrides: object) -> DigestFacts:
    win = window(start_minute=0, end_minute=60)
    payload: dict[str, object] = {
        "window": win,
        "price_reaction": normal_reaction("105"),
        "symbol": SYMBOL,
        "timeframe": TF,
        # Round 6.5b: the seam carries whole items, not just ids. The
        # timestamp travels with the item so the renderer never asks a model
        # what time something happened.
        "news_items": (
            DigestSourceItem(
                item_id="goldnewsvn:901", published_at=BASE, text="Fed giữ nguyên lãi suất."
            ),
            DigestSourceItem(item_id="goldnewsvn:902", published_at=BASE, text="USD giảm 0.21%."),
        ),
    }
    payload.update(overrides)
    return build_digest_facts(**payload)  # type: ignore[arg-type]


def test_the_seam_carries_both_the_facts_and_their_rendering() -> None:
    built = facts()

    assert built.title.startswith("📰 TIN VÀNG")
    assert built.window_line.endswith("(giờ VN)")
    assert PRICE_REACTION_HEADING in built.price_reaction_block
    assert built.price_reaction.net_change == Decimal("105")


def test_the_deterministic_lines_are_named_as_a_group() -> None:
    """What Round 6.5b will require a writer to reproduce unchanged."""
    built = facts()

    assert built.deterministic_lines == (
        built.title,
        built.window_line,
        built.price_reaction_block,
    )


def test_the_seam_refuses_a_reaction_about_a_different_window() -> None:
    """Two windows in one digest is how a title stops matching its numbers."""
    other = window(start_minute=0, end_minute=120)

    with pytest.raises(ValueError):
        facts(window=other)


def test_news_items_are_offered_as_a_closed_list_not_a_ranking() -> None:
    """Selection is editorial and belongs to the writer, not to this round."""
    built = facts()

    assert built.news_item_ids == ("goldnewsvn:901", "goldnewsvn:902")
    assert not hasattr(built, "selected_items")
    assert not hasattr(built, "impact")


def test_the_seam_is_immutable() -> None:
    built = facts()

    with pytest.raises(ValueError):
        built.title = "📰 TIN VÀNG hôm nay"  # type: ignore[misc]


def test_the_length_contract_is_recorded_but_not_enforced() -> None:
    """Agreed now so the activation round adopts it rather than re-deciding."""
    assert (DIGEST_TARGET_MIN_CHARS, DIGEST_TARGET_MAX_CHARS) == (900, 1500)
    assert DIGEST_HARD_CAP_CHARS == 1900

    # Recorded in the digest schema and nowhere else: the article contract is
    # untouched, so nothing enforces these on a stage that cannot run yet.
    import inspect

    from goldpipeline.schemas import article_contract

    assert "DIGEST_HARD_CAP_CHARS" not in inspect.getsource(article_contract)


# --------------------------------------------------------------------------
# nothing in production moved
# --------------------------------------------------------------------------


def test_news_digest_became_ready_with_its_own_prompt() -> None:
    """Round 6.5b turned this on. TRADE_PLAN stayed off, which is the point.

    Activating one article type is exactly the moment another can be switched
    on by accident, so the assertion covers both.
    """
    from goldpipeline.schemas.article import ArticleType
    from goldpipeline.services.article_routing import SPECS

    assert SPECS[ArticleType.NEWS_DIGEST].ready is True
    assert SPECS[ArticleType.NEWS_DIGEST].prompt_id == DEFAULT_DIGEST_WRITER_PROMPT
    assert DEFAULT_DIGEST_WRITER_PROMPT.startswith("gold_news_digest_writer_")
    assert SPECS[ArticleType.TRADE_PLAN].ready is False
    assert SPECS[ArticleType.TRADE_PLAN].prompt_id is None


def test_style_activation_did_not_reach_the_digest() -> None:
    from goldpipeline.schemas.article import ArticleType
    from goldpipeline.services.review_action import style_is_active

    assert not style_is_active(ArticleType.NEWS_DIGEST)


def test_no_prompt_changed() -> None:
    from goldpipeline.prompts import load_prompt

    assert "🕯 PHÂN TÍCH VÀNG" in load_prompt("gold_writer_v4")
    assert "# HUMAN STYLE v1" in load_prompt("gold_human_style_v1")
    assert "# HUMAN STYLE REVIEW" in load_prompt("gold_reviewer_v2")
    assert "# HUMAN STYLE REPAIR" in load_prompt("gold_finalizer_v2")

    for prompt in ("gold_writer_v4", "gold_reviewer_v2", "gold_finalizer_v2"):
        text = load_prompt(prompt)
        assert "TIN VÀNG" not in text, prompt
        assert "Giá phản ứng" not in text, prompt


def test_only_digest_modules_reach_the_digest_layer() -> None:
    """The digest is wired now, and only to itself.

    Round 6.5a asserted nothing reached this code, because nothing could. Now
    something does, and the property worth keeping is narrower and more useful:
    the ANALYSIS stages must not import any of it. A writer that could reach a
    digest renderer, or a finalizer that knew about price reactions, is how one
    article type starts quietly shaping another.
    """
    import ast
    from pathlib import Path

    digest_modules = {
        "digest",
        "digest_context",
        "digest_pipeline",
        "digest_render",
        "digest_writer",
        "news_digest",
        "price_reaction",
    }
    analysis_stages = (
        "services/writer.py",
        "services/writer_prompt.py",
        "services/reviewer.py",
        "services/reviewer_prompt.py",
        "services/finalizer.py",
        "services/finalizer_prompt.py",
        "services/final_postcheck.py",
        "services/orchestrator.py",
        "services/publish_gate.py",
        "services/analysis_contract.py",
    )

    root = Path("src/goldpipeline")
    for relative in analysis_stages:
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        reached = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.rsplit(".", 1)[-1] in digest_modules
        }
        assert not reached, f"{relative} reaches the digest layer: {reached}"


def test_the_preferred_digest_timeframe_is_recorded_not_imposed() -> None:
    """M5 is the product choice; the calculator still takes what it is given."""
    assert PREFERRED_DIGEST_TIMEFRAME is Timeframe.M5

    bars = [
        OHLCBar(
            timestamp=BASE + timedelta(minutes=15 * i),
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("99"),
            close=Decimal(100 + i),
        )
        for i in range(4)
    ]
    win = DigestWindow(
        start=BASE + timedelta(minutes=15),
        end=BASE + timedelta(minutes=75),
        lookback_seconds=3600,
    )

    result = calculate_price_reaction(bars, timeframe=Timeframe.M15, window=win, symbol=SYMBOL)

    assert result.timeframe is Timeframe.M15
    assert result.market_activity is MarketActivity.NORMAL
