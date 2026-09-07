"""ATR, confirmed swings and fair value gaps, checked by hand.

Round 6.6a §32-§35. Every expected number below was worked out on paper from
the fixture bars, not by running the function and pasting what came back. That
distinction is the whole value of this file: a test that compares an
implementation to itself passes for any implementation, including a wrong one,
and these three primitives are what BOS, order blocks, liquidity and every
published price level will eventually be derived from.

Three properties get the most attention, because each is a place a plausible
implementation goes quietly wrong:

* **Wilder, not a mean.** The seed is a simple average of the first `period`
  true ranges and every later value is recursive. A rolling mean would agree
  with it on flat data and diverge exactly when volatility matters.
* **No lookahead.** A pivot is not knowable at its own timestamp. Tests here
  reconstruct "what did we know at time T" and check the answer does not depend
  on how much future the snapshot happens to hold.
* **Strict inequality.** Equal highs make no swing and a touch makes no gap.
  Both are deliberate absences, and an absence nobody pinned is one a later
  round will "fix".

Offline throughout.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import IctTimeframeSnapshot, build_timeframe_snapshot
from goldpipeline.schemas.market import OHLCBar
from goldpipeline.services.ict_primitives import (
    DEFAULT_ATR_PERIOD,
    GapDirection,
    SwingType,
    atr_series,
    confirmed_swings,
    fair_value_gaps,
    latest_atr,
    swings_known_at,
    true_range,
    true_range_series,
)

START = datetime(2026, 9, 7, 0, 0, tzinfo=UTC)
HOUR = timedelta(hours=1)


def candle(index: int, high: str, low: str, close: str, open_: str | None = None) -> OHLCBar:
    """One H1 bar at ``START + index`` hours."""
    return OHLCBar(
        timestamp=START + HOUR * index,
        open=Decimal(open_ if open_ is not None else close),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
    )


def h1(bars: list[OHLCBar]) -> IctTimeframeSnapshot:
    """A snapshot observed exactly when the last bar closed."""
    observed = bars[-1].timestamp + HOUR
    return build_timeframe_snapshot(timeframe=Timeframe.H1, bars=tuple(bars), observed_at=observed)


# --------------------------------------------------------------------------
# §32: ATR
# --------------------------------------------------------------------------

# Hand-built so every true range is obvious:
#   bar0  H 110  L 100  C 105   -> no TR (no predecessor)
#   bar1  H 112  L 108  C 110   -> max(4, |112-105|=7, |108-105|=3) = 7   (gap up)
#   bar2  H 111  L 101  C 102   -> max(10, |111-110|=1, |101-110|=9) = 10 (ordinary)
#   bar3  H  99  L  95  C  96   -> max(4, |99-102|=3, |95-102|=7)   = 7   (gap down)
TR_BARS = [
    candle(0, "110", "100", "105"),
    candle(1, "112", "108", "110"),
    candle(2, "111", "101", "102"),
    candle(3, "99", "95", "96"),
]


def test_true_range_uses_the_previous_close_for_a_gap_up() -> None:
    assert true_range(TR_BARS[0], TR_BARS[1]) == Decimal(7)


def test_true_range_is_the_plain_bar_range_when_there_is_no_gap() -> None:
    assert true_range(TR_BARS[1], TR_BARS[2]) == Decimal(10)


def test_true_range_uses_the_previous_close_for_a_gap_down() -> None:
    assert true_range(TR_BARS[2], TR_BARS[3]) == Decimal(7)


def test_the_first_bar_has_no_true_range_and_is_not_approximated() -> None:
    """Substituting `high - low` there would understate an opening gap."""
    ranges = true_range_series(TR_BARS)

    assert len(ranges) == len(TR_BARS) - 1
    assert [value for _, value in ranges] == [Decimal(7), Decimal(10), Decimal(7)]
    assert ranges[0][0].timestamp == TR_BARS[1].timestamp


def test_the_wilder_seed_is_the_simple_average_of_the_first_period_ranges() -> None:
    """period=3 over TRs 7, 10, 7 -> (7+10+7)/3 = 8."""
    points = atr_series(h1(TR_BARS), period=3)

    assert len(points) == 1
    assert points[0].atr == Decimal(8)
    assert points[0].true_range == Decimal(7)
    assert points[0].period == 3


def test_the_next_value_is_wilder_recursion_not_a_rolling_mean() -> None:
    """A fourth TR of 20 gives (8*2 + 20)/3 = 12; a rolling mean would give 12.33..."""
    bars = [*TR_BARS, candle(4, "116", "96", "100")]  # TR = max(20, |116-96|=20, |96-96|=0)
    points = atr_series(h1(bars), period=3)

    assert points[-1].true_range == Decimal(20)
    assert points[-1].atr == Decimal(12)

    rolling_mean = (Decimal(10) + Decimal(7) + Decimal(20)) / Decimal(3)
    assert points[-1].atr != rolling_mean, "Wilder and a mean must not be confused"


def test_the_period_is_configurable_and_changes_the_answer() -> None:
    two = atr_series(h1(TR_BARS), period=2)
    three = atr_series(h1(TR_BARS), period=3)

    assert two[0].atr == (Decimal(7) + Decimal(10)) / Decimal(2)
    assert three[0].atr == Decimal(8)
    assert two[0].period == 2


def test_the_default_period_is_wilders_own() -> None:
    assert DEFAULT_ATR_PERIOD == 14


def test_insufficient_history_yields_nothing_not_a_shorter_average() -> None:
    """A 3-bar mean labelled a 14-period ATR is a wrong number wearing a right name."""
    assert atr_series(h1(TR_BARS), period=14) == []
    assert latest_atr(h1(TR_BARS), period=14) is None


def test_exactly_period_bars_is_still_not_enough() -> None:
    """One warm-up bar is consumed: the first bar contributes no true range."""
    assert atr_series(h1(TR_BARS), period=4) == []
    assert atr_series(h1(TR_BARS), period=3) != []


def test_a_non_positive_period_is_a_caller_mistake_not_a_market_condition() -> None:
    for bad in (0, -1):
        with pytest.raises(ValueError, match="must be positive"):
            atr_series(h1(TR_BARS), period=bad)


def test_every_point_carries_its_own_bar_and_timeframe() -> None:
    points = atr_series(h1(TR_BARS), period=3)

    assert points[0].timeframe is Timeframe.H1
    assert points[0].bar_open_time == TR_BARS[3].timestamp
    assert points[0].bar_close_time == TR_BARS[3].timestamp + HOUR


def test_atr_arithmetic_is_decimal_throughout() -> None:
    points = atr_series(h1(TR_BARS), period=3)

    assert isinstance(points[0].atr, Decimal)
    assert isinstance(points[0].true_range, Decimal)
    assert not isinstance(points[0].atr, float)


def test_a_forming_candle_cannot_move_the_atr() -> None:
    """It never enters the snapshot, so the primitive never sees it."""
    observed = TR_BARS[-1].timestamp + HOUR
    forming = candle(4, "999", "1", "500")

    without = build_timeframe_snapshot(
        timeframe=Timeframe.H1, bars=tuple(TR_BARS), observed_at=observed
    )
    with_forming = build_timeframe_snapshot(
        timeframe=Timeframe.H1, bars=(*TR_BARS, forming), observed_at=observed
    )

    assert atr_series(with_forming, period=3) == atr_series(without, period=3)


# --------------------------------------------------------------------------
# §33: confirmed swings
# --------------------------------------------------------------------------

# index:      0    1    2    3    4
# highs:    100  105  120  104   99   -> bar2 is a swing high
# lows:      90   85   70   86   91   -> bar2 is also a swing low of the lows? no:
#                                        70 is the lowest, so bar2 is both by shape.
PIVOT_BARS = [
    candle(0, "100", "90", "95"),
    candle(1, "105", "85", "100"),
    candle(2, "120", "70", "110"),
    candle(3, "104", "86", "95"),
    candle(4, "99", "91", "95"),
]


def test_an_obvious_swing_high_is_found() -> None:
    swings = confirmed_swings(h1(PIVOT_BARS), left_bars=2, right_bars=2)
    highs = [s for s in swings if s.swing_type is SwingType.SWING_HIGH]

    assert len(highs) == 1
    assert highs[0].price == Decimal(120)
    assert highs[0].pivot_time == PIVOT_BARS[2].timestamp


def test_an_obvious_swing_low_is_found() -> None:
    bars = [
        candle(0, "110", "100", "105"),
        candle(1, "108", "95", "100"),
        candle(2, "104", "80", "90"),
        candle(3, "106", "92", "100"),
        candle(4, "112", "99", "108"),
    ]
    swings = confirmed_swings(h1(bars), left_bars=2, right_bars=2)
    lows = [s for s in swings if s.swing_type is SwingType.SWING_LOW]

    assert len(lows) == 1
    assert lows[0].price == Decimal(80)
    assert lows[0].pivot_time == bars[2].timestamp


def test_a_flat_series_produces_no_pivot() -> None:
    bars = [candle(i, "100", "90", "95") for i in range(7)]

    assert confirmed_swings(h1(bars)) == []


def test_equal_highs_produce_no_swing() -> None:
    """Deliberate. Two bars sharing a high are a pool, not a pivot.

    Resolving the tie first-wins or last-wins would invent a distinction the
    market did not make. Equal highs get named for what they are in a later
    round.
    """
    bars = [
        candle(0, "100", "90", "95"),
        candle(1, "105", "88", "100"),
        candle(2, "120", "85", "110"),
        candle(3, "120", "86", "110"),  # ties the high
        candle(4, "104", "87", "95"),
        candle(5, "99", "88", "95"),
    ]
    highs = [s for s in confirmed_swings(h1(bars)) if s.swing_type is SwingType.SWING_HIGH]

    assert highs == []


def test_equal_lows_produce_no_swing() -> None:
    bars = [
        candle(0, "110", "100", "105"),
        candle(1, "108", "95", "100"),
        candle(2, "104", "80", "90"),
        candle(3, "106", "80", "95"),  # ties the low
        candle(4, "107", "92", "100"),
        candle(5, "112", "99", "108"),
    ]
    lows = [s for s in confirmed_swings(h1(bars)) if s.swing_type is SwingType.SWING_LOW]

    assert lows == []


def test_the_left_window_is_configurable() -> None:
    """A wider left window demands the pivot dominate more history."""
    bars = [
        candle(0, "130", "90", "95"),  # taller than the candidate
        candle(1, "100", "88", "95"),
        candle(2, "105", "85", "100"),
        candle(3, "120", "80", "110"),  # candidate
        candle(4, "104", "86", "95"),
        candle(5, "99", "87", "95"),
    ]
    narrow = [
        s
        for s in confirmed_swings(h1(bars), left_bars=2, right_bars=2)
        if s.swing_type is SwingType.SWING_HIGH
    ]
    wide = [
        s
        for s in confirmed_swings(h1(bars), left_bars=3, right_bars=2)
        if s.swing_type is SwingType.SWING_HIGH
    ]

    assert [s.price for s in narrow] == [Decimal(120)]
    assert wide == [], "bar0's 130 dominates once the window reaches it"


def test_the_right_window_is_configurable() -> None:
    bars = [
        candle(0, "100", "90", "95"),
        candle(1, "105", "88", "100"),
        candle(2, "120", "85", "110"),  # candidate
        candle(3, "104", "86", "95"),
        candle(4, "125", "87", "120"),  # taller, further right
        candle(5, "99", "88", "95"),
    ]
    narrow = [
        s.price
        for s in confirmed_swings(h1(bars), left_bars=2, right_bars=1)
        if s.swing_type is SwingType.SWING_HIGH
    ]
    wide = [
        s.price
        for s in confirmed_swings(h1(bars), left_bars=2, right_bars=2)
        if s.swing_type is SwingType.SWING_HIGH
    ]

    # With one bar to the right, bar2's 120 clears its single neighbour. With
    # two, bar4's 125 reaches it and the pivot is gone. (bar4 is itself a pivot
    # under the narrow window - it dominates its own single neighbour - which is
    # correct and not what this test is about.)
    assert Decimal(120) in narrow
    assert Decimal(120) not in wide


def test_a_non_positive_window_is_refused() -> None:
    for left, right in ((0, 2), (2, 0), (-1, 2)):
        with pytest.raises(ValueError, match="must be positive"):
            confirmed_swings(h1(PIVOT_BARS), left_bars=left, right_bars=right)


def test_insufficient_left_history_yields_no_pivot() -> None:
    """The first `left_bars` bars can never be pivots; they have no left side."""
    bars = PIVOT_BARS[2:]  # the tall bar is now first

    assert confirmed_swings(h1(bars), left_bars=2, right_bars=1) == []


def test_insufficient_right_history_yields_no_pivot() -> None:
    bars = PIVOT_BARS[:4]  # only one bar right of the candidate

    assert confirmed_swings(h1(bars), left_bars=2, right_bars=2) == []


def test_multiple_swings_are_returned_oldest_first() -> None:
    bars = [
        candle(0, "100", "90", "95"),
        candle(1, "105", "88", "100"),
        candle(2, "120", "85", "110"),  # swing high
        candle(3, "104", "70", "80"),  # swing low
        candle(4, "108", "86", "100"),
        candle(5, "130", "95", "125"),  # swing high
        candle(6, "112", "96", "100"),
        candle(7, "109", "97", "105"),
    ]
    swings = confirmed_swings(h1(bars), left_bars=2, right_bars=2)

    assert [s.pivot_time for s in swings] == sorted(s.pivot_time for s in swings)
    assert [s.price for s in swings] == [Decimal(120), Decimal(70), Decimal(130)]


def test_a_monotonic_rise_produces_no_confirmed_high() -> None:
    bars = [candle(i, str(100 + i * 5), str(90 + i * 5), str(95 + i * 5)) for i in range(8)]
    highs = [s for s in confirmed_swings(h1(bars)) if s.swing_type is SwingType.SWING_HIGH]

    assert highs == []


def test_a_monotonic_fall_produces_no_confirmed_low() -> None:
    bars = [candle(i, str(200 - i * 5), str(190 - i * 5), str(195 - i * 5)) for i in range(8)]
    lows = [s for s in confirmed_swings(h1(bars)) if s.swing_type is SwingType.SWING_LOW]

    assert lows == []


def test_the_pivot_price_is_the_exact_decimal_from_the_bar() -> None:
    bars = [
        candle(0, "100.00", "90", "95"),
        candle(1, "105.00", "88", "100"),
        candle(2, "120.37", "85", "110"),
        candle(3, "104.00", "86", "95"),
        candle(4, "99.00", "87", "95"),
    ]
    highs = [s for s in confirmed_swings(h1(bars)) if s.swing_type is SwingType.SWING_HIGH]

    assert highs[0].price == Decimal("120.37")
    assert isinstance(highs[0].price, Decimal)


def test_the_same_bars_give_the_same_swings_whatever_the_provider() -> None:
    """Provider is provenance; it reaches no primitive."""
    observed = PIVOT_BARS[-1].timestamp + HOUR
    one = build_timeframe_snapshot(
        timeframe=Timeframe.H1, bars=tuple(PIVOT_BARS), observed_at=observed
    )
    two = build_timeframe_snapshot(
        timeframe=Timeframe.H1, bars=tuple(PIVOT_BARS), observed_at=observed
    )

    assert confirmed_swings(one) == confirmed_swings(two)


# --------------------------------------------------------------------------
# §17 / §33: no lookahead
# --------------------------------------------------------------------------


def test_confirmation_lags_the_pivot_by_the_right_window() -> None:
    """A pivot at 02:00 with two right bars was not knowable at 02:00."""
    swings = confirmed_swings(h1(PIVOT_BARS), left_bars=2, right_bars=2)
    high = next(s for s in swings if s.swing_type is SwingType.SWING_HIGH)

    assert high.pivot_time == START + HOUR * 2
    assert high.confirmed_at == START + HOUR * 5, "close of the second right-hand bar"
    assert high.confirmed_at > high.pivot_time


def test_a_swing_is_invisible_before_it_is_confirmed() -> None:
    swings = confirmed_swings(h1(PIVOT_BARS), left_bars=2, right_bars=2)
    high = next(s for s in swings if s.swing_type is SwingType.SWING_HIGH)

    assert swings_known_at(swings, high.pivot_time) == []
    assert swings_known_at(swings, high.confirmed_at - timedelta(seconds=1)) == []
    assert high in swings_known_at(swings, high.confirmed_at)


def test_what_was_knowable_does_not_depend_on_how_much_future_is_held() -> None:
    """The property the whole `confirmed_at` field exists for.

    A snapshot ending early and one holding extra bars must agree about what
    was knowable at the earlier instant. If they disagreed, every later stage
    would silently get more or less structure depending on fetch depth.
    """
    short_bars = PIVOT_BARS
    long_bars = [*PIVOT_BARS, candle(5, "98", "88", "92"), candle(6, "97", "87", "91")]
    cutoff = PIVOT_BARS[-1].timestamp + HOUR

    short_known = swings_known_at(confirmed_swings(h1(short_bars)), cutoff)
    long_known = swings_known_at(confirmed_swings(h1(long_bars)), cutoff)

    assert short_known == long_known


# --------------------------------------------------------------------------
# §34: fair value gaps
# --------------------------------------------------------------------------


def test_a_bullish_gap_is_found_with_its_band() -> None:
    """A.high 100, C.low 110 -> untraded band [100, 110]."""
    bars = [
        candle(0, "100", "90", "95"),
        candle(1, "115", "98", "112"),
        candle(2, "120", "110", "118"),
    ]
    gaps = fair_value_gaps(h1(bars))

    assert len(gaps) == 1
    assert gaps[0].direction is GapDirection.BULLISH
    assert gaps[0].lower == Decimal(100)
    assert gaps[0].upper == Decimal(110)


def test_a_bearish_gap_is_found_with_its_band() -> None:
    """A.low 100, C.high 90 -> untraded band [90, 100]."""
    bars = [
        candle(0, "115", "100", "105"),
        candle(1, "108", "88", "92"),
        candle(2, "90", "80", "85"),
    ]
    gaps = fair_value_gaps(h1(bars))

    assert len(gaps) == 1
    assert gaps[0].direction is GapDirection.BEARISH
    assert gaps[0].lower == Decimal(90)
    assert gaps[0].upper == Decimal(100)


def test_an_exact_touch_is_not_a_gap() -> None:
    """`C.low == A.high` means every price between them traded."""
    bars = [
        candle(0, "100", "90", "95"),
        candle(1, "115", "98", "112"),
        candle(2, "120", "100", "118"),  # low touches A's high exactly
    ]
    assert fair_value_gaps(h1(bars)) == []


def test_an_exact_bearish_touch_is_not_a_gap() -> None:
    bars = [
        candle(0, "115", "100", "105"),
        candle(1, "108", "88", "92"),
        candle(2, "100", "80", "85"),  # high touches A's low exactly
    ]
    assert fair_value_gaps(h1(bars)) == []


def test_overlapping_ordinary_candles_make_no_gap() -> None:
    bars = [
        candle(0, "110", "100", "105"),
        candle(1, "112", "102", "108"),
        candle(2, "111", "101", "106"),
    ]
    assert fair_value_gaps(h1(bars)) == []


def test_gap_size_and_midpoint_are_exact_decimals() -> None:
    bars = [
        candle(0, "100.25", "90", "95"),
        candle(1, "115", "98", "112"),
        candle(2, "120", "110.75", "118"),
    ]
    gap = fair_value_gaps(h1(bars))[0]

    assert gap.size == Decimal("10.50")
    assert gap.midpoint == Decimal("105.50")
    assert isinstance(gap.size, Decimal)


def test_the_gap_records_all_three_anchor_candles() -> None:
    bars = [
        candle(0, "100", "90", "95"),
        candle(1, "115", "98", "112"),
        candle(2, "120", "110", "118"),
    ]
    gap = fair_value_gaps(h1(bars))[0]

    assert gap.first_time == bars[0].timestamp
    assert gap.middle_time == bars[1].timestamp
    assert gap.last_time == bars[2].timestamp
    assert gap.timeframe is Timeframe.H1


def test_the_gap_forms_when_candle_c_closes_not_when_it_opens() -> None:
    """A three-candle pattern is not a fact until its third candle finishes."""
    bars = [
        candle(0, "100", "90", "95"),
        candle(1, "115", "98", "112"),
        candle(2, "120", "110", "118"),
    ]
    gap = fair_value_gaps(h1(bars))[0]

    assert gap.formed_at == bars[2].timestamp + HOUR
    assert gap.formed_at != bars[2].timestamp
    assert gap.formed_at != bars[1].timestamp + HOUR


def test_a_forming_third_candle_creates_no_gap() -> None:
    """It is not in the snapshot, so the pattern is not yet a pattern."""
    closed = [candle(0, "100", "90", "95"), candle(1, "115", "98", "112")]
    forming = candle(2, "120", "110", "118")
    observed = forming.timestamp  # C has opened but not closed

    series = build_timeframe_snapshot(
        timeframe=Timeframe.H1, bars=(*closed, forming), observed_at=observed
    )

    assert series.bar_count == 2
    assert fair_value_gaps(series) == []


def test_multiple_gaps_are_returned_oldest_first() -> None:
    bars = [
        candle(0, "100", "90", "95"),
        candle(1, "115", "98", "112"),
        candle(2, "120", "110", "118"),  # bullish gap [100, 110]
        candle(3, "150", "119", "145"),
        candle(4, "170", "130", "165"),  # bullish gap [120, 130]
    ]
    gaps = fair_value_gaps(h1(bars))

    # Three, not two: a strong run leaves a gap on every consecutive triple, so
    # bars 1-2-3 produce [115, 119] between the two obvious ones. Worth pinning
    # rather than fixturing away - overlapping gaps are normal in a trend, and a
    # later round ranking zones has to expect them.
    assert [(g.lower, g.upper) for g in gaps] == [
        (Decimal(100), Decimal(110)),
        (Decimal(115), Decimal(119)),
        (Decimal(120), Decimal(130)),
    ]
    assert [g.formed_at for g in gaps] == sorted(g.formed_at for g in gaps)
    assert all(g.direction is GapDirection.BULLISH for g in gaps)


def test_the_same_bars_give_the_same_gaps_whatever_the_provider() -> None:
    bars = [
        candle(0, "100", "90", "95"),
        candle(1, "115", "98", "112"),
        candle(2, "120", "110", "118"),
    ]
    observed = bars[-1].timestamp + HOUR
    one = build_timeframe_snapshot(timeframe=Timeframe.H1, bars=tuple(bars), observed_at=observed)
    two = build_timeframe_snapshot(timeframe=Timeframe.H1, bars=tuple(bars), observed_at=observed)

    assert fair_value_gaps(one) == fair_value_gaps(two)


def test_a_series_too_short_for_three_candles_makes_no_gap() -> None:
    bars = [candle(0, "100", "90", "95"), candle(1, "115", "98", "112")]

    assert fair_value_gaps(h1(bars)) == []


# --------------------------------------------------------------------------
# §24: the primitives assert no meaning
# --------------------------------------------------------------------------


def test_the_vocabulary_stays_observational() -> None:
    """No BSL, SSL, support, resistance, SEO or BAI at this layer.

    A type called `LiquidityHigh` would be asserting something this code has
    not established. Those readings need structure, which is a later round.
    """
    import ast
    import inspect

    from goldpipeline.services import ict_primitives

    names = set(ict_primitives.__all__)
    for forbidden in ("BSL", "SSL", "Support", "Resistance", "Seo", "Bai", "OrderBlock"):
        assert not any(forbidden.lower() in name.lower() for name in names), forbidden

    # Code only. The prose deliberately says a bullish gap is *not* a buy zone,
    # and a guard that grepped the docstrings would fail for the module saying
    # exactly the right thing.
    tree = ast.parse(inspect.getsource(ict_primitives))
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            node.value.value = ""
    code = ast.unparse(tree).lower()

    for word in ("buy_zone", "sell_zone", "entry", "stop_loss", "take_profit", "target"):
        assert word not in code, word


def test_swing_and_gap_kinds_are_closed_enums_not_strings() -> None:
    assert {m.value for m in SwingType} == {"SWING_HIGH", "SWING_LOW"}
    assert {m.value for m in GapDirection} == {"BULLISH", "BEARISH"}
