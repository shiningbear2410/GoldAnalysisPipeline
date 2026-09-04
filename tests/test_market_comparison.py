"""Comparing two market sources, on synthetic data with known answers.

Every series here is built by hand, so each metric has an expected value that
does not depend on a network, a broker, or the time of day.

Two themes carry most of the tests:

* **a difference must not be invented.** When two sources anchor their bars at
  different offsets - a real and expected thing on higher timeframes - pairing
  candles by position would report a four-hour time shift as a price
  disagreement. The comparison must refuse to pair them and say why.
* **a difference must not be hidden.** No metric rounds, averages away, or
  thresholds a disagreement. A single bad bar has to survive into the maximum.

Offline throughout: no MT5, no websocket, no model, no clock dependency.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from goldpipeline.domain.errors import IncomparableSourcesError
from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.market import MarketDataInput
from goldpipeline.services.market_comparison import (
    SESSION_BREAK_MINIMUM,
    AlignmentKind,
    GapKind,
    VolumeShape,
    compare_market_sources,
    describe_series,
)

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
MINUTES = {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240}


def series(
    *,
    count: int = 20,
    timeframe: str = "M15",
    provider: str = "alpha",
    provider_symbol: str | None = "XAUUSD",
    symbol: str = "XAUUSD",
    close: float = 4300.00,
    step: float = 0.50,
    offset: float = 0.0,
    newest_open: datetime | None = None,
    anchor_shift: timedelta = timedelta(0),
    skip: set[int] | None = None,
    duplicate: int | None = None,
    reverse: bool = False,
    with_volume: bool = True,
    overrides: dict[int, dict[str, float]] | None = None,
) -> MarketDataInput:
    """A clean ascending series, plus whichever defect a test asks for.

    ``newest_open`` defaults to the last bar that has closed by :data:`NOW`, so
    a series is fresh unless a test deliberately makes it stale.
    """
    period = timedelta(minutes=MINUTES[timeframe])
    if newest_open is None:
        # The newest bar that has *closed* on this series' own grid. Adding the
        # shift to a fixed instant instead would produce a forming bar, and
        # then every freshness assertion would be measuring the fixture.
        span = int(period.total_seconds())
        grid = int(anchor_shift.total_seconds()) % span
        moment = int(NOW.timestamp())
        current = moment - ((moment - grid) % span)
        newest_open = datetime.fromtimestamp(current - span, tz=UTC)
    else:
        newest_open = newest_open + anchor_shift

    bars: list[dict[str, Any]] = []
    for index in range(count):
        if skip and index in skip:
            continue
        opened = newest_open - period * (count - 1 - index)
        base = Decimal(str(close)) - Decimal(str(step)) * (count - 1 - index) + Decimal(str(offset))
        bar: dict[str, Any] = {
            "timestamp": opened.isoformat().replace("+00:00", "Z"),
            "open": str(base - Decimal("0.20")),
            "high": str(base + Decimal("0.40")),
            "low": str(base - Decimal("0.40")),
            "close": str(base),
        }
        if with_volume:
            bar["volume"] = str(100 + index)
        if overrides and index in overrides:
            for field, value in overrides[index].items():
                bar[field] = str(Decimal(str(value)))
            # Keep the bar a legal candle. A test that moves a close is saying
            # "the sources disagree here", not "this provider sent a bar whose
            # high is below its close" - that case has its own tests elsewhere.
            prices = [Decimal(bar[name]) for name in ("open", "high", "low", "close")]
            bar["high"] = str(max(prices))
            bar["low"] = str(min(prices))
        bars.append(bar)

    if duplicate is not None:
        bars.append(dict(bars[duplicate]))
    if reverse:
        bars.reverse()

    return MarketDataInput.model_validate(
        {
            "symbol": symbol,
            "provider": provider,
            "provider_symbol": provider_symbol,
            "timeframe": timeframe,
            "timezone": None,
            "bars": bars,
        }
    )


def compare(a: MarketDataInput, b: MarketDataInput, **kwargs: Any) -> Any:
    return compare_market_sources(a, b, observed_at=NOW, **kwargs)


# --- refusals -------------------------------------------------------------


class TestRefusals:
    def test_different_instruments_are_refused_not_compared(self) -> None:
        with pytest.raises(IncomparableSourcesError, match="different instruments"):
            compare(series(), series(symbol="XAGUSD"))

    def test_different_timeframes_are_refused(self) -> None:
        with pytest.raises(IncomparableSourcesError, match="different timeframes"):
            compare(series(timeframe="M5"), series(timeframe="M15"))

    def test_a_requested_timeframe_must_match_the_payloads(self) -> None:
        with pytest.raises(IncomparableSourcesError, match="not the one these payloads"):
            compare(series(timeframe="M15"), series(timeframe="M15"), timeframe=Timeframe.H1)

    def test_a_matching_requested_timeframe_is_accepted(self) -> None:
        result = compare(series(), series(), timeframe=Timeframe.M15)
        assert result.timeframe is Timeframe.M15

    def test_a_naive_observation_time_is_refused(self) -> None:
        with pytest.raises(IncomparableSourcesError, match="naive"):
            compare_market_sources(series(), series(), observed_at=datetime(2026, 9, 3, 12, 0))

    def test_provider_symbols_may_differ_when_the_canonical_symbol_matches(self) -> None:
        """The expected case: one broker calls gold XAUUSD, one venue OANDA:XAUUSD."""
        result = compare(
            series(provider="metatrader5", provider_symbol="XAUUSD"),
            series(provider="tradingview", provider_symbol="OANDA:XAUUSD"),
        )
        assert result.canonical_symbol == "XAUUSD"
        assert result.a.provider_symbol == "XAUUSD"
        assert result.b.provider_symbol == "OANDA:XAUUSD"
        assert result.alignment is AlignmentKind.ALIGNED


# --- the easy cases -------------------------------------------------------


class TestPerfectMatch:
    def test_identical_series_agree_on_everything(self) -> None:
        result = compare(series(), series(provider="beta"))
        assert result.alignment is AlignmentKind.ALIGNED
        assert result.intersection_count == 20
        assert result.intersection_ratio == 1
        assert result.only_in_a == result.only_in_b == 0
        for metric in (result.close, result.open, result.high, result.low, result.bar_range):
            assert metric is not None
            assert metric.median_abs == metric.p95_abs == metric.max_abs == 0
        assert result.directional is not None
        assert result.directional.ratio == 1
        assert result.directional.flat_ties == 0

    def test_the_comparison_is_symmetric(self) -> None:
        a, b = series(), series(provider="beta", offset=0.7)
        forward, backward = compare(a, b), compare(b, a)
        assert forward.close is not None and backward.close is not None
        assert forward.close.max_abs == backward.close.max_abs
        assert forward.intersection_count == backward.intersection_count
        assert forward.anchor_offset_seconds == -(backward.anchor_offset_seconds or 0)


class TestPriceDifferences:
    def test_a_constant_offset_shows_up_everywhere_undiminished(self) -> None:
        """A venue trading 0.75 higher is not a defect, but it must be visible."""
        result = compare(series(), series(provider="beta", offset=0.75))
        assert result.close is not None
        assert result.close.median_abs == Decimal("0.75")
        assert result.close.p95_abs == Decimal("0.75")
        assert result.close.max_abs == Decimal("0.75")
        assert result.close.sample_count == 20
        # A constant offset cancels in the range and in the direction.
        assert result.bar_range is not None and result.bar_range.max_abs == 0
        assert result.directional is not None and result.directional.ratio == 1

    def test_one_bad_bar_survives_into_the_maximum(self) -> None:
        result = compare(series(), series(provider="beta", overrides={7: {"close": 9999.0}}))
        assert result.close is not None
        assert result.close.median_abs == 0
        assert result.close.max_abs > Decimal("5000")
        assert result.close.max_at is not None

    def test_small_varying_differences_are_not_rounded_away(self) -> None:
        tweaks = {
            index: {"close": 4300.00 - 0.50 * (19 - index) + 0.01 * index} for index in range(20)
        }
        result = compare(series(), series(provider="beta", overrides=tweaks))
        assert result.close is not None
        assert result.close.max_abs == Decimal("0.19")
        assert result.close.median_abs > 0

    def test_range_coherence_is_reported_separately_from_level(self) -> None:
        wider = {index: {"high": 4300.00 - 0.50 * (19 - index) + 2.00} for index in range(20)}
        result = compare(series(), series(provider="beta", overrides=wider))
        assert result.bar_range is not None
        assert result.bar_range.median_abs == Decimal("1.60")
        assert result.close is not None and result.close.max_abs == 0


class TestDirectionalAgreement:
    def test_opposite_direction_is_caught(self) -> None:
        falling = {index: {"close": 4300.00 + 0.50 * (19 - index)} for index in range(20)}
        result = compare(series(), series(provider="beta", overrides=falling))
        assert result.directional is not None
        assert result.directional.comparable_intervals == 19
        assert result.directional.agreed == 0
        assert result.directional.ratio == 0

    def test_flat_intervals_are_excluded_not_counted_as_agreement(self) -> None:
        flat = {index: {"close": 4300.00} for index in range(20)}
        result = compare(series(), series(provider="beta", overrides=flat))
        assert result.directional is not None
        assert result.directional.flat_ties == 19
        assert result.directional.comparable_intervals == 0
        assert result.directional.ratio is None

    def test_moves_across_a_gap_are_not_counted(self) -> None:
        """A weekend move compared to a one-bar move would be a false disagreement."""
        gapped = series(count=20, skip={10, 11, 12})
        result = compare(gapped, series(count=20, skip={10, 11, 12}, provider="beta"))
        assert result.directional is not None
        assert result.directional.comparable_intervals == 15


# --- completeness ---------------------------------------------------------


class TestCompleteness:
    def test_a_missing_active_session_candle_is_visible_on_one_side_only(self) -> None:
        result = compare(series(), series(provider="beta", skip={9}))
        assert result.only_in_a == 1
        assert result.only_in_b == 0
        assert result.intersection_count == 19
        assert len(result.only_in_a_samples) == 1
        assert result.b.quiet_intervals == 1
        assert result.b.session_breaks == 0

    def test_duplicates_are_counted_and_noted(self) -> None:
        result = compare(series(), series(provider="beta", duplicate=4))
        assert result.b.bar_count == 21
        assert result.b.unique_timestamps == 20
        assert result.b.duplicate_timestamps == 1
        assert any("repeated a timestamp" in note for note in result.notes)
        # A duplicate must not become a second candle in the comparison.
        assert result.intersection_count == 20

    def test_unsorted_input_is_reported_but_still_comparable(self) -> None:
        result = compare(series(), series(provider="beta", reverse=True))
        assert result.a.ascending is True
        assert result.b.ascending is False
        assert result.intersection_count == 20
        assert result.close is not None and result.close.max_abs == 0

    def test_different_bar_counts_do_not_look_like_missing_data(self) -> None:
        """A longer series reaches further back; it is not missing anything."""
        result = compare(series(count=40), series(count=20, provider="beta"))
        assert result.a.bar_count == 40
        assert result.b.bar_count == 20
        assert result.intersection_count == 20
        assert result.comparable_window_bars == 20
        assert result.intersection_ratio == 1
        assert result.only_in_a == 0

    def test_partial_overlap(self) -> None:
        older = series(count=20, newest_open=NOW - timedelta(minutes=15 * 10))
        result = compare(older, series(count=20, provider="beta"))
        assert 0 < result.intersection_count < 20
        assert result.alignment is AlignmentKind.ALIGNED

    def test_disjoint_windows_have_no_overlap(self) -> None:
        ancient = series(count=10, newest_open=NOW - timedelta(days=5))
        result = compare(ancient, series(count=10, provider="beta"))
        assert result.alignment is AlignmentKind.NO_OVERLAP
        assert result.intersection_count == 0
        assert result.close is None
        assert any("disjoint" in note for note in result.notes)

    def test_an_empty_series_is_reported_not_crashed(self) -> None:
        empty = MarketDataInput.model_validate(
            {
                "symbol": "XAUUSD",
                "provider": "beta",
                "timeframe": "M15",
                "timezone": None,
                "bars": [],
            }
        )
        result = compare(series(), empty)
        assert result.alignment is AlignmentKind.NO_OVERLAP
        assert result.b.bar_count == 0
        assert result.b.newest is None
        assert result.b.newest_is_closed is False
        assert any("no bars" in note for note in result.notes)


class TestSessionGaps:
    def test_a_weekend_shaped_gap_is_a_session_break_not_corruption(self) -> None:
        weekend = series(count=30, timeframe="H1", skip=set(range(5, 20)))
        quality = describe_series(weekend, timeframe=Timeframe.H1, observed_at=NOW)
        assert quality.session_breaks == 1
        assert quality.quiet_intervals == 0
        assert quality.gaps[0].kind is GapKind.SESSION_BREAK
        assert quality.gaps[0].missing_periods == 15

    def test_a_single_missing_minute_is_a_quiet_interval(self) -> None:
        thin = series(count=30, timeframe="M1", skip={12})
        quality = describe_series(thin, timeframe=Timeframe.M1, observed_at=NOW)
        assert quality.quiet_intervals == 1
        assert quality.session_breaks == 0
        assert quality.gaps[0].kind is GapKind.QUIET_INTERVAL

    def test_the_session_break_boundary_is_where_it_is_documented(self) -> None:
        # Skipping N consecutive bars leaves a span of (N + 1) periods, so the
        # documented boundary falls between N = periods - 2 and N = periods - 1.
        periods = int(SESSION_BREAK_MINIMUM / timedelta(minutes=15))
        just_under = series(count=60, skip=set(range(10, 10 + periods - 2)))
        just_over = series(count=60, skip=set(range(10, 10 + periods - 1)))
        under = describe_series(just_under, timeframe=Timeframe.M15, observed_at=NOW)
        over = describe_series(just_over, timeframe=Timeframe.M15, observed_at=NOW)
        assert under.gaps[0].kind is GapKind.QUIET_INTERVAL
        assert over.gaps[0].kind is GapKind.SESSION_BREAK

    def test_a_contiguous_series_has_no_gaps(self) -> None:
        quality = describe_series(series(count=50), timeframe=Timeframe.M15, observed_at=NOW)
        assert quality.gaps == ()


# --- freshness ------------------------------------------------------------


class TestFreshness:
    def test_a_fresh_series_is_the_expected_latest_closed_interval(self) -> None:
        quality = describe_series(series(), timeframe=Timeframe.M15, observed_at=NOW)
        assert quality.newest_is_closed is True
        assert quality.newest_age_seconds == 0
        assert quality.newest_is_expected_latest is True

    def test_a_stale_newest_candle_is_measured_not_excused(self) -> None:
        stale = series(newest_open=NOW - timedelta(hours=4))
        quality = describe_series(stale, timeframe=Timeframe.M15, observed_at=NOW)
        assert quality.newest_is_closed is True
        assert quality.newest_age_seconds == int(timedelta(hours=3, minutes=45).total_seconds())
        assert quality.newest_is_expected_latest is False

    def test_a_forming_newest_candle_is_detected_by_arithmetic(self) -> None:
        """A provider label is never trusted; the bar's own open time decides."""
        forming = series(newest_open=NOW - timedelta(minutes=5))
        quality = describe_series(forming, timeframe=Timeframe.M15, observed_at=NOW)
        assert quality.newest_is_closed is False
        assert quality.newest_age_seconds is None

    def test_a_forming_bar_on_either_side_is_noted(self) -> None:
        result = compare(series(), series(provider="beta", newest_open=NOW - timedelta(minutes=5)))
        assert any("had not closed" in note for note in result.notes)

    @pytest.mark.parametrize("timeframe", ["M1", "M5", "M15", "H1", "H4"])
    def test_freshness_uses_each_sources_own_grid(self, timeframe: str) -> None:
        """A feed anchored at 01:00 is not judged against one anchored at 00:00."""
        period = timedelta(minutes=MINUTES[timeframe])
        shifted = series(timeframe=timeframe, anchor_shift=period / 4)
        quality = describe_series(shifted, timeframe=Timeframe(timeframe), observed_at=NOW)
        assert quality.newest_is_expected_latest is True
        assert quality.off_grid_bars == 0


# --- anchors --------------------------------------------------------------


class TestSessionAnchors:
    def test_the_same_anchor_aligns(self) -> None:
        result = compare(
            series(timeframe="H4", count=20), series(timeframe="H4", count=20, provider="beta")
        )
        assert result.alignment is AlignmentKind.ALIGNED
        assert result.anchor_offset_seconds == 0
        assert result.close is not None

    def test_a_different_h4_anchor_is_named_not_treated_as_a_fault(self) -> None:
        mt5_like = series(timeframe="H4", count=20, provider="metatrader5")
        venue_like = series(
            timeframe="H4",
            count=20,
            provider="tradingview",
            anchor_shift=timedelta(hours=1),
        )
        result = compare(mt5_like, venue_like)
        assert result.alignment is AlignmentKind.DIFFERENT_SESSION_ANCHOR
        assert result.anchor_offset_seconds == 3600

    def test_a_different_anchor_withholds_per_bar_prices_rather_than_faking_them(self) -> None:
        result = compare(
            series(timeframe="H4", count=20),
            series(timeframe="H4", count=20, provider="beta", anchor_shift=timedelta(hours=2)),
        )
        assert result.close is None
        assert result.open is None
        assert result.high is None
        assert result.low is None
        assert result.bar_range is None
        assert result.directional is None
        assert any("withheld" in note for note in result.notes)

    def test_window_coherence_still_works_when_anchors_differ(self) -> None:
        """The only honest price comparison left: same clock time, no pairing."""
        result = compare(
            series(timeframe="H4", count=20),
            series(timeframe="H4", count=20, provider="beta", anchor_shift=timedelta(hours=1)),
        )
        assert result.window is not None
        assert result.window.bars_a > 0 and result.window.bars_b > 0
        assert result.window.high_difference >= 0
        # The two 'last closes' in the shared window cannot coincide when the
        # grids are offset, and they cannot be a whole period apart either.
        period = int(timedelta(hours=4).total_seconds())
        assert 0 < result.window.last_close_gap_seconds < period
        assert any("apart" in note for note in result.notes)

    def test_the_modal_offset_defines_the_grid_not_one_stray_bar(self) -> None:
        clean = series(count=20, timeframe="H1")
        bars: list[dict[str, Any]] = []
        for index, bar in enumerate(clean.bars):
            moment = bar.timestamp + (timedelta(minutes=7) if index == 3 else timedelta(0))
            bars.append(
                {
                    "timestamp": moment.isoformat().replace("+00:00", "Z"),
                    "open": str(bar.open),
                    "high": str(bar.high),
                    "low": str(bar.low),
                    "close": str(bar.close),
                }
            )
        mutated = MarketDataInput.model_validate(
            {
                "symbol": "XAUUSD",
                "provider": "alpha",
                "timeframe": "H1",
                "timezone": None,
                "bars": bars,
            }
        )
        quality = describe_series(mutated, timeframe=Timeframe.H1, observed_at=NOW)
        assert quality.off_grid_bars == 1
        assert quality.grid_offset_seconds == 0


# --- volume ---------------------------------------------------------------


class TestVolume:
    def test_volume_presence_is_recorded_per_source(self) -> None:
        result = compare(series(), series(provider="beta", with_volume=False))
        assert result.a.volume_shape is VolumeShape.PRESENT
        assert result.b.volume_shape is VolumeShape.ABSENT

    def test_no_volume_metric_is_produced_at_all(self) -> None:
        """Tick counts and venue activity are different quantities."""
        result = compare(series(), series(provider="beta", with_volume=False))
        fields = set(type(result).model_fields)
        assert not any("volume" in name for name in fields)
        dumped = result.model_dump()
        assert "volume" not in str(dumped.get("close"))

    def test_a_volume_disagreement_changes_no_price_metric(self) -> None:
        loud = series(provider="beta")
        quiet = series(provider="beta", with_volume=False)
        assert compare(series(), loud).close == compare(series(), quiet).close


# --- precision ------------------------------------------------------------


class TestPrecision:
    def test_price_precision_is_recorded_so_deltas_can_be_read_in_context(self) -> None:
        two = series()
        three = series(provider="beta", overrides={i: {"close": 4300.125} for i in range(20)})
        result = compare(two, three)
        assert result.a.price_decimals_max == 2
        assert result.b.price_decimals_max == 3

    def test_a_sub_cent_difference_is_not_rounded_to_zero(self) -> None:
        nudged = {index: {"close": 4300.00 - 0.50 * (19 - index) + 0.001} for index in range(20)}
        result = compare(series(), series(provider="beta", overrides=nudged))
        assert result.close is not None
        assert result.close.max_abs == Decimal("0.001")


# --- boundary -------------------------------------------------------------


class TestBoundary:
    MODULE = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "goldpipeline"
        / "services"
        / "market_comparison.py"
    )

    def test_it_imports_no_provider_no_ai_and_no_transport(self) -> None:
        tree = ast.parse(self.MODULE.read_text(encoding="utf-8"))
        forbidden = (
            "goldpipeline.adapters",
            "websocket",
            "MetaTrader5",
            "goldpipeline.prompts",
            "goldpipeline.services.telegram",
            "goldpipeline.services.news_collector",
            "goldpipeline.schemas.telegram",
        )
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert not any(name.startswith(bad) for bad in forbidden), name

    def test_it_names_no_wire_or_terminal_detail(self) -> None:
        text = self.MODULE.read_text(encoding="utf-8")
        for word in (
            "~m~",
            "resolve_symbol",
            "create_series",
            "copy_rates_from_pos",
            "symbol_select",
        ):
            assert word not in text, word

    def test_it_defines_no_acceptance_threshold(self) -> None:
        """Observation only: nothing here decides what difference is tolerable."""
        text = self.MODULE.read_text(encoding="utf-8")
        for word in ("TOLERANCE", "MAX_DIFFERENCE", "ACCEPTABLE", "def is_valid", "PASS", "FAIL"):
            assert word not in text, word

    def test_neither_side_is_assumed_to_be_a_particular_provider(self) -> None:
        text = self.MODULE.read_text(encoding="utf-8")
        assert "metatrader5" not in text.lower()
        assert '"tradingview"' not in text


class TestProductionUnchanged:
    def test_the_cli_selects_tradingview_as_the_production_authority(self) -> None:
        cli = (Path(__file__).resolve().parents[1] / "src" / "goldpipeline" / "cli.py").read_text(
            encoding="utf-8"
        )
        assert 'PRODUCTION_MARKET_SOURCE = "tradingview"' in cli
        assert 'default="mt5"' not in cli
        # MT5 stays selectable, and is never a fallback.
        assert '"tradingview", "mt5", "file"' in cli

    def test_nothing_wires_the_comparison_into_the_pipeline(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "goldpipeline"
        for path in root.rglob("*.py"):
            if path.name == "market_comparison.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert "market_comparison" not in node.module, path.name

    def test_article_readiness_is_unchanged(self) -> None:
        from goldpipeline.schemas.article import ArticleType
        from goldpipeline.services.article_routing import READY_TYPES, SPECS

        assert {ArticleType.ANALYSIS, ArticleType.NEWS_DIGEST} == READY_TYPES
        # NEWS_DIGEST was activated by Round 6.5b, with its own writer.
        assert SPECS[ArticleType.NEWS_DIGEST].prompt_id == "gold_news_digest_writer_v1"
        assert SPECS[ArticleType.TRADE_PLAN].ready is False
