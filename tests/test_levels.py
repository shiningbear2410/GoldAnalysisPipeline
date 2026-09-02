"""Deterministic technical measurements.

Offline throughout: no MT5, no model, no network. Every expected value here is
either hand-calculated in the test or derivable by reading the bars, because a
test that asserts whatever the code happened to return proves only that the code
is consistent with itself.

The invariant these tests exist to defend: **the model may never originate a
level.** That only holds if the numbers in the context are reproducible, so the
tests check reproducibility rather than plausibility.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.context import MarketStructure, ZoneKind
from goldpipeline.schemas.market import OHLCBar
from goldpipeline.services.levels import (
    DEFAULT_ATR_PERIOD,
    MAX_PIVOT_WINDOW,
    LevelSettings,
    average_true_range,
    build_levels,
    classify_structure,
    cluster_zones,
    swing_highs,
    swing_lows,
    true_ranges,
)

START = datetime(2026, 9, 2, 0, 0, tzinfo=UTC)


def bar(index: int, high: str, low: str, close: str | None = None, open_: str | None = None):
    """One candle. Defaults keep OHLC invariants satisfied."""
    high_d, low_d = Decimal(high), Decimal(low)
    close_d = Decimal(close) if close is not None else (high_d + low_d) / 2
    open_d = Decimal(open_) if open_ is not None else close_d
    return OHLCBar(
        timestamp=START + timedelta(minutes=15 * index),
        open=open_d,
        high=high_d,
        low=low_d,
        close=close_d,
    )


def series(spec: list[tuple[str, str]]) -> list[OHLCBar]:
    return [bar(i, high, low) for i, (high, low) in enumerate(spec)]


# ------------------------------------------------------------------- ATR
class TestTrueRange:
    def test_first_bar_has_no_true_range(self) -> None:
        """No previous close means no TR - omitted, not approximated."""
        bars = series([("10", "8"), ("11", "9")])
        assert len(true_ranges(bars)) == 1

    def test_plain_range_when_no_gap(self) -> None:
        bars = [bar(0, "10", "8", close="9"), bar(1, "11", "9", close="10")]
        # high-low = 2; |11-9| = 2; |9-9| = 0  -> 2
        assert true_ranges(bars) == [Decimal("2")]

    def test_gap_up_uses_previous_close(self) -> None:
        """The case a naive high-low misses entirely."""
        bars = [bar(0, "10", "8", close="9"), bar(1, "20", "19", close="19.5")]
        # high-low = 1; |20-9| = 11; |19-9| = 10  -> 11
        assert true_ranges(bars) == [Decimal("11")]

    def test_gap_down_uses_previous_close(self) -> None:
        bars = [bar(0, "20", "18", close="19"), bar(1, "10", "9", close="9.5")]
        # high-low = 1; |10-19| = 9; |9-19| = 10  -> 10
        assert true_ranges(bars) == [Decimal("10")]


class TestAverageTrueRange:
    def test_hand_calculated_mean(self) -> None:
        """Five bars, four true ranges, averaged over period 4.

        closes 9, 10, 11, 12, 13
        TRs:  bar1 |11-9|=2 -> 2 ; bar2 2 ; bar3 2 ; bar4 2   -> mean 2
        """
        bars = [
            bar(0, "10", "8", close="9"),
            bar(1, "11", "9", close="10"),
            bar(2, "12", "10", close="11"),
            bar(3, "13", "11", close="12"),
            bar(4, "14", "12", close="13"),
        ]
        assert average_true_range(bars, period=4) == Decimal("2")

    def test_mixed_hand_calculated(self) -> None:
        bars = [
            bar(0, "100", "90", close="95"),
            bar(1, "110", "100", close="105"),  # TR max(10, 15, 5)  = 15
            bar(2, "108", "104", close="106"),  # TR max(4, 3, 1)    = 4
            bar(3, "120", "118", close="119"),  # TR max(2, 14, 12)  = 14
        ]
        assert true_ranges(bars) == [Decimal("15"), Decimal("4"), Decimal("14")]
        assert average_true_range(bars, period=3) == Decimal("11")  # 33/3

    def test_uses_only_the_last_period_ranges(self) -> None:
        bars = [
            bar(0, "100", "90", close="95"),
            bar(1, "200", "190", close="195"),  # huge TR, outside the window
            bar(2, "196", "194", close="195"),
            bar(3, "196", "194", close="195"),
        ]
        # last two TRs are 2 and 2
        assert average_true_range(bars, period=2) == Decimal("2")

    def test_insufficient_bars_returns_none(self) -> None:
        """Absent beats approximate."""
        bars = series([("10", "8"), ("11", "9"), ("12", "10")])
        assert average_true_range(bars, period=DEFAULT_ATR_PERIOD) is None

    def test_exactly_enough_bars(self) -> None:
        bars = [bar(i, str(10 + i), str(8 + i), close=str(9 + i)) for i in range(5)]
        assert average_true_range(bars, period=4) is not None
        assert average_true_range(bars, period=5) is None

    def test_precision_is_preserved(self) -> None:
        """Decimal in, Decimal out - no float rounding on the way through."""
        bars = [
            bar(0, "2000.123", "2000.003", close="2000.063"),
            bar(1, "2000.223", "2000.103", close="2000.163"),
            bar(2, "2000.323", "2000.203", close="2000.263"),
        ]
        atr = average_true_range(bars, period=2)
        assert isinstance(atr, Decimal)
        assert atr == Decimal("0.16")


# ----------------------------------------------------------------- pivots
class TestSwingPoints:
    def test_confirmed_swing_high(self) -> None:
        bars = series([("10", "5"), ("11", "6"), ("15", "7"), ("11", "6"), ("10", "5")])
        highs = swing_highs(bars, window=2)
        assert [p.bar_index for p in highs] == [2]
        assert highs[0].price == Decimal("15")
        assert highs[0].timestamp == bars[2].timestamp

    def test_confirmed_swing_low(self) -> None:
        bars = series([("10", "5"), ("9", "4"), ("8", "1"), ("9", "4"), ("10", "5")])
        lows = swing_lows(bars, window=2)
        assert [p.bar_index for p in lows] == [2]
        assert lows[0].price == Decimal("1")

    def test_edge_bars_are_never_pivots(self) -> None:
        """The highest bar sits at the right edge: unconfirmed, so not a pivot."""
        bars = series([("10", "5"), ("11", "6"), ("12", "7"), ("13", "8"), ("99", "9")])
        assert swing_highs(bars, window=2) == []

    def test_left_edge_is_never_a_pivot(self) -> None:
        bars = series([("99", "5"), ("11", "6"), ("12", "7"), ("13", "8"), ("10", "9")])
        assert [p.bar_index for p in swing_highs(bars, window=2)] == []

    def test_equal_highs_resolve_to_the_earlier_bar(self) -> None:
        """Strict-left / non-strict-right: exactly one pivot, deterministically."""
        bars = series(
            [("10", "5"), ("11", "6"), ("15", "7"), ("15", "7"), ("11", "6"), ("10", "5")]
        )
        highs = swing_highs(bars, window=2)
        assert [p.bar_index for p in highs] == [2]

    def test_equal_lows_resolve_to_the_earlier_bar(self) -> None:
        bars = series([("10", "9"), ("9", "8"), ("8", "1"), ("8", "1"), ("9", "8"), ("10", "9")])
        lows = swing_lows(bars, window=2)
        assert [p.bar_index for p in lows] == [2]

    def test_flat_series_has_no_pivots(self) -> None:
        bars = series([("10", "5")] * 9)
        assert swing_highs(bars, window=2) == []
        assert swing_lows(bars, window=2) == []

    def test_insufficient_bars(self) -> None:
        bars = series([("10", "5"), ("11", "6"), ("12", "7")])
        assert swing_highs(bars, window=2) == []

    def test_result_is_deterministic_across_repeats(self) -> None:
        bars = series([("10", "5"), ("14", "6"), ("15", "7"), ("14", "6"), ("10", "5")])
        first = swing_highs(bars, window=2)
        for _ in range(5):
            assert swing_highs(bars, window=2) == first

    @pytest.mark.parametrize("window", [1, 2, 3, MAX_PIVOT_WINDOW])
    def test_window_sizes_never_report_edge_bars(self, window: int) -> None:
        bars = [bar(i, str(10 + i), str(5 + i)) for i in range(20)]
        for pivot in swing_highs(bars, window=window):
            assert window <= pivot.bar_index <= len(bars) - 1 - window


# -------------------------------------------------------------- structure
class TestStructure:
    def build(self, highs: list[str], lows: list[str]):
        h = [p for p in swing_highs(_hl_series(highs, high=True), window=1)]
        return h

    def test_higher_highs_and_higher_lows_is_bullish(self) -> None:
        highs = _points([("10", 1), ("12", 5)])
        lows = _points([("5", 2), ("7", 6)])
        assert classify_structure(highs, lows) is MarketStructure.BULLISH

    def test_lower_highs_and_lower_lows_is_bearish(self) -> None:
        highs = _points([("12", 1), ("10", 5)])
        lows = _points([("7", 2), ("5", 6)])
        assert classify_structure(highs, lows) is MarketStructure.BEARISH

    def test_broadening_is_range_not_a_trend(self) -> None:
        """Higher high with a lower low is not bullish, and must not be called so."""
        highs = _points([("10", 1), ("12", 5)])
        lows = _points([("7", 2), ("5", 6)])
        assert classify_structure(highs, lows) is MarketStructure.RANGE

    def test_equal_high_is_not_a_higher_high(self) -> None:
        highs = _points([("10", 1), ("10", 5)])
        lows = _points([("5", 2), ("7", 6)])
        assert classify_structure(highs, lows) is MarketStructure.RANGE

    @pytest.mark.parametrize(
        ("n_highs", "n_lows"),
        [(0, 0), (1, 1), (2, 1), (1, 2), (2, 0)],
    )
    def test_insufficient_pivots(self, n_highs: int, n_lows: int) -> None:
        highs = _points([("10", i) for i in range(n_highs)])
        lows = _points([("5", i) for i in range(n_lows)])
        assert classify_structure(highs, lows) is MarketStructure.INSUFFICIENT_DATA


def _points(spec: list[tuple[str, int]]):
    from goldpipeline.schemas.context import SwingPoint

    return [
        SwingPoint(
            bar_index=index,
            timestamp=START + timedelta(minutes=15 * index),
            price=Decimal(price),
        )
        for price, index in spec
    ]


def _hl_series(values: list[str], *, high: bool) -> list[OHLCBar]:
    return [bar(i, v, str(Decimal(v) - 5)) for i, v in enumerate(values)]


# ------------------------------------------------------------------ zones
class TestZones:
    def test_nearby_pivots_form_one_zone(self) -> None:
        pivots = _points([("100", 1), ("100.4", 3), ("100.2", 5)])
        zones = cluster_zones(pivots, kind=ZoneKind.RESISTANCE, threshold=Decimal("0.5"))
        assert len(zones) == 1
        assert zones[0].lower == Decimal("100")
        assert zones[0].upper == Decimal("100.4")
        assert zones[0].width == Decimal("0.4")
        assert zones[0].pivot_count == 3

    def test_isolated_pivots_form_separate_zones(self) -> None:
        pivots = _points([("100", 1), ("140", 3)])
        zones = cluster_zones(pivots, kind=ZoneKind.SUPPORT, threshold=Decimal("0.5"))
        assert len(zones) == 2
        assert [z.pivot_count for z in zones] == [1, 1]

    def test_single_pivot_zone_is_degenerate_not_widened(self) -> None:
        """A lone pivot is a level. Padding it to ATR would invent territory."""
        zones = cluster_zones(_points([("100", 1)]), kind=ZoneKind.SUPPORT, threshold=Decimal("5"))
        assert zones[0].lower == zones[0].upper == Decimal("100")
        assert zones[0].width == Decimal("0")

    def test_zones_never_overlap(self) -> None:
        pivots = _points([(str(100 + i), i) for i in range(10)])
        zones = cluster_zones(pivots, kind=ZoneKind.SUPPORT, threshold=Decimal("1"))
        for earlier, later in zip(zones, zones[1:], strict=False):
            assert earlier.upper < later.lower

    def test_bounds_are_observed_prices_only(self) -> None:
        prices = {Decimal("100"), Decimal("100.4")}
        zones = cluster_zones(
            _points([("100", 1), ("100.4", 2)]), kind=ZoneKind.SUPPORT, threshold=Decimal("1")
        )
        assert zones[0].lower in prices and zones[0].upper in prices

    def test_ordering_is_stable(self) -> None:
        pivots = _points([("140", 1), ("100", 2), ("120", 3)])
        zones = cluster_zones(pivots, kind=ZoneKind.SUPPORT, threshold=Decimal("1"))
        assert [z.lower for z in zones] == sorted(z.lower for z in zones)

    def test_no_pivots_no_zones(self) -> None:
        assert cluster_zones([], kind=ZoneKind.SUPPORT, threshold=Decimal("1")) == []

    def test_timestamps_span_the_cluster(self) -> None:
        pivots = _points([("100", 5), ("100.2", 1)])
        zone = cluster_zones(pivots, kind=ZoneKind.SUPPORT, threshold=Decimal("1"))[0]
        assert zone.first_timestamp < zone.last_timestamp


# ------------------------------------------------------------- assembly
def trending_bars(count: int = 24) -> list[OHLCBar]:
    """A zig-zag with a rising bias, so pivots exist on both sides."""
    bars = []
    for i in range(count):
        base = Decimal(100) + Decimal(i)
        swing = Decimal(4) if i % 4 == 2 else Decimal(0)
        bars.append(bar(i, str(base + swing + 2), str(base - swing - 2), close=str(base)))
    return bars


class TestBuildLevels:
    def test_window_extremes_are_exact(self) -> None:
        bars = series([("10", "5"), ("30", "6"), ("12", "1"), ("11", "7")])
        levels = build_levels(bars)
        assert levels.window_high == Decimal("30")
        assert levels.window_high_at == bars[1].timestamp
        assert levels.window_low == Decimal("1")
        assert levels.window_low_at == bars[2].timestamp

    def test_equal_extremes_pick_the_earlier_bar(self) -> None:
        bars = series([("30", "5"), ("30", "5")])
        levels = build_levels(bars)
        assert levels.window_high_at == bars[0].timestamp
        assert levels.window_low_at == bars[0].timestamp

    def test_empty_bars_yield_an_empty_block(self) -> None:
        levels = build_levels([])
        assert levels.bars_considered == 0
        assert levels.atr is None
        assert levels.swing_highs == []
        assert levels.structure is MarketStructure.INSUFFICIENT_DATA
        assert levels.support_zones == []

    def test_short_window_still_produces_a_valid_block(self) -> None:
        """One unavailable measurement must not fail the whole Run."""
        levels = build_levels(series([("10", "8"), ("11", "9"), ("12", "10")]))
        assert levels.atr is None
        assert levels.structure is MarketStructure.INSUFFICIENT_DATA
        assert levels.support_zones == []
        assert levels.window_high == Decimal("12")

    def test_no_zones_without_an_atr(self) -> None:
        """Without ATR there is no defensible clustering threshold."""
        levels = build_levels(series([("10", "8")] * 6))
        assert levels.atr is None
        assert levels.support_zones == []
        assert levels.resistance_zones == []

    def test_full_block_on_a_realistic_window(self) -> None:
        levels = build_levels(trending_bars())
        assert levels.bars_considered == 24
        assert levels.atr is not None and levels.atr > 0
        assert levels.swing_highs and levels.swing_lows
        assert levels.structure in set(MarketStructure)
        assert levels.method_version == "1"

    def test_zone_count_is_bounded(self) -> None:
        settings = LevelSettings(max_zones_per_side=2)
        levels = build_levels(trending_bars(60), settings)
        assert len(levels.support_zones) <= 2
        assert len(levels.resistance_zones) <= 2

    def test_pivot_window_is_clamped(self) -> None:
        levels = build_levels(trending_bars(), LevelSettings(pivot_window=99))
        assert levels.pivot_window == MAX_PIVOT_WINDOW

    def test_recomputation_is_identical(self) -> None:
        bars = trending_bars()
        assert build_levels(bars).model_dump_json() == build_levels(bars).model_dump_json()


# ------------------------------------------------------- closed-bar safety
class TestFormingCandleCannotChangeAnything:
    """The core safety property: only bars handed in are ever considered.

    The adapter already excludes the forming candle. These tests prove the
    measurements have no other route to it - appending or mutating a candle
    beyond the window changes nothing, because nothing here looks there.
    """

    def test_appending_a_candle_changes_nothing_for_the_original_window(self) -> None:
        closed = trending_bars()
        before = build_levels(closed).model_dump_json()

        forming = bar(999, "9999", "1", close="5000")
        _ = build_levels([*closed, forming])  # a different window, deliberately

        assert build_levels(closed).model_dump_json() == before

    @pytest.mark.parametrize(
        "measurement",
        ["atr", "structure", "window_high", "window_low", "swing_highs", "support_zones"],
    )
    def test_each_measurement_is_a_function_of_the_bars_given(self, measurement: str) -> None:
        closed = trending_bars()
        first = getattr(build_levels(closed), measurement)
        second = getattr(build_levels(list(closed)), measurement)
        assert first == second

    def test_an_extreme_forming_candle_does_not_leak_backwards(self) -> None:
        """A wild unclosed candle must not move yesterday's numbers."""
        closed = trending_bars()
        baseline = build_levels(closed)
        wild = bar(1000, "50000", "0.01", close="25000")
        extended = build_levels([*closed, wild])

        # The extended window legitimately differs; the closed one must not.
        assert build_levels(closed) == baseline
        assert extended.window_high != baseline.window_high

    def test_pivots_never_include_the_final_bar(self) -> None:
        """Whatever the last bar is, it cannot be confirmed."""
        bars = trending_bars()
        last = len(bars) - 1
        assert all(p.bar_index != last for p in build_levels(bars).swing_highs)
        assert all(p.bar_index != last for p in build_levels(bars).swing_lows)
