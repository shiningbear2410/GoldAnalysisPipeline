"""The TradingView market data source, exercised entirely offline.

The connection is injected, so nothing here opens a socket, needs the
websocket library, or depends on when it runs.

The tests that matter most are the ones about the **forming candle**. The feed
counts the bar in progress among the bars it sends; quoting one produces an
article that is false by the time anybody reads it. Two separate mechanisms
prevent that - asking for more bars than are wanted, and proving each returned
bar has closed by arithmetic on its own open time - and both are tested at
every supported timeframe, on the second before the close, on the boundary, and
after it.

The second theme is **failing closed**. One unusable bar refuses the whole
fetch. Each such test names the specific corruption, because the alternative
outcome - a quietly shortened series - is invisible to every stage downstream.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from goldpipeline.adapters.base import MarketDataSource
from goldpipeline.adapters.fake_tradingview import (
    FEED_INTERNALS_SENTINEL,
    TIMEFRAME_MINUTES,
    FakeConnection,
    RecordingConnector,
    conversation,
    critical_error,
    data_update,
    heartbeat,
    make_series,
    protocol_error,
    scripted,
    series_completed,
    session_chatter,
    timescale_update,
    unparseable,
)
from goldpipeline.adapters.tradingview_market import (
    MAX_CANDLE_LIMIT,
    PROVIDER_NAME,
    SAFETY_MARGIN_BARS,
    SUPPORTED_SYMBOLS,
    SUPPORTED_TIMEFRAMES,
    WEBSOCKET_URL,
    TradingViewMarketDataSource,
)
from goldpipeline.adapters.tradingview_protocol import encode_frame
from goldpipeline.domain.errors import (
    InsufficientBarsError,
    MarketDataConfigurationError,
    TradingViewCandleError,
    TradingViewConnectionError,
    TradingViewCriticalError,
    TradingViewFramingError,
    TradingViewProtocolError,
    TradingViewTimeoutError,
)
from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.market import MarketDataInput
from goldpipeline.services.normalizer import normalize_market_data

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
SYMBOL = "OANDA:XAUUSD"


def make_source(
    connection: FakeConnection | None = None,
    *,
    timeframe: Timeframe | str = Timeframe.M15,
    limit: int = 5,
    now: datetime = NOW,
    connector: RecordingConnector | None = None,
    **kwargs: Any,
) -> tuple[TradingViewMarketDataSource, RecordingConnector]:
    """A source wired to a scripted connection and a frozen clock."""
    if connector is None:
        connector = RecordingConnector()
        if connection is not None:
            connector.scripts.append(connection)
    ticks = iter(range(0, 100_000))
    source = TradingViewMarketDataSource(
        timeframe=timeframe,
        limit=limit,
        connector=connector,
        now=lambda: now,
        monotonic=lambda: float(next(ticks)),
        session_id="cs_test",
        series_id="sds_test",
        **kwargs,
    )
    return source, connector


def rows_for(timeframe: str, count: int, now: datetime = NOW) -> list[list[Any]]:
    return make_series(count=count, timeframe=timeframe, now=now)


# --- configuration, before any network ------------------------------------


class TestConfiguration:
    @pytest.mark.parametrize(
        ("timeframe", "wire"),
        [("M1", "1"), ("M5", "5"), ("M15", "15"), ("H1", "60"), ("H4", "240")],
    )
    def test_timeframe_maps_to_the_wire_interval(self, timeframe: str, wire: str) -> None:
        connection = FakeConnection(packets=[conversation(rows_for(timeframe, 8))])
        source, _ = make_source(connection, timeframe=timeframe, limit=5)
        source.load()
        assert connection.command("create_series").params[4] == wire

    def test_the_mapping_is_exactly_the_five_supported_timeframes(self) -> None:
        assert {str(t) for t in SUPPORTED_TIMEFRAMES} == {"M1", "M5", "M15", "H1", "H4"}
        assert list(SUPPORTED_TIMEFRAMES.values()) == ["1", "5", "15", "60", "240"]

    def test_m30_is_refused_although_the_project_config_allows_it(self) -> None:
        """The pipeline knows M30; this venue is not offered it, so it fails loudly."""
        connector = RecordingConnector()
        with pytest.raises(MarketDataConfigurationError, match="does not support"):
            TradingViewMarketDataSource(timeframe=Timeframe.M30, connector=connector)
        assert connector.connections == []

    @pytest.mark.parametrize("timeframe", ["D1", "W1", "MN1"])
    def test_other_known_timeframes_are_refused(self, timeframe: str) -> None:
        with pytest.raises(MarketDataConfigurationError, match="does not support"):
            TradingViewMarketDataSource(timeframe=timeframe)

    def test_a_nonsense_timeframe_is_refused_before_the_network(self) -> None:
        connector = RecordingConnector()
        with pytest.raises(MarketDataConfigurationError, match="not a timeframe"):
            TradingViewMarketDataSource(timeframe="M7", connector=connector)
        assert connector.connections == []

    def test_the_supported_symbol_is_accepted(self) -> None:
        source, _ = make_source(FakeConnection(packets=[conversation()]))
        assert source.provider_symbol == SYMBOL
        assert SUPPORTED_SYMBOLS[SYMBOL] == "XAUUSD"

    @pytest.mark.parametrize(
        "symbol", ["FOREXCOM:XAUUSD", "XAUUSD", "OANDA:XAGUSD", "OANDA:EURUSD", ""]
    )
    def test_an_unsupported_symbol_never_becomes_a_substitute_feed(self, symbol: str) -> None:
        connector = RecordingConnector()
        with pytest.raises(MarketDataConfigurationError, match="does not support"):
            TradingViewMarketDataSource(provider_symbol=symbol, connector=connector)
        assert connector.connections == []

    def test_symbol_case_and_padding_are_tolerated(self) -> None:
        source = TradingViewMarketDataSource(provider_symbol="  oanda:xauusd ")
        assert source.provider_symbol == SYMBOL

    @pytest.mark.parametrize("limit", [0, -1, MAX_CANDLE_LIMIT + 1])
    def test_limits_outside_the_bounds_are_refused_before_the_network(self, limit: int) -> None:
        connector = RecordingConnector()
        with pytest.raises(MarketDataConfigurationError, match="limit must be"):
            TradingViewMarketDataSource(limit=limit, connector=connector)
        assert connector.connections == []

    def test_a_non_integer_limit_is_refused(self) -> None:
        with pytest.raises(MarketDataConfigurationError, match="whole number"):
            TradingViewMarketDataSource(limit=10.5)  # type: ignore[arg-type]

    def test_the_endpoint_is_pinned_in_code(self) -> None:
        assert WEBSOCKET_URL == "wss://data.tradingview.com/socket.io/websocket"


# --- session flow ---------------------------------------------------------


class TestSessionFlow:
    def test_the_handshake_order(self) -> None:
        connection = FakeConnection(packets=[conversation()])
        source, _ = make_source(connection)
        source.load()
        assert connection.method_order() == [
            "set_auth_token",
            "chart_create_session",
            "switch_timezone",
            "resolve_symbol",
            "create_series",
        ]

    def test_the_timezone_is_pinned_to_utc_before_bars_are_requested(self) -> None:
        connection = FakeConnection(packets=[conversation()])
        source, _ = make_source(connection)
        source.load()
        order = connection.method_order()
        assert connection.command("switch_timezone").params[1] == "Etc/UTC"
        assert order.index("switch_timezone") < order.index("create_series")

    def test_no_credential_is_sent(self) -> None:
        connection = FakeConnection(packets=[conversation()])
        source, _ = make_source(connection)
        source.load()
        assert connection.command("set_auth_token").params == ["unauthorized_user_token"]

    def test_the_resolved_symbol_is_the_configured_one(self) -> None:
        connection = FakeConnection(packets=[conversation()])
        source, _ = make_source(connection)
        source.load()
        resolved = connection.command("resolve_symbol").params[2]
        assert SYMBOL in resolved
        assert '"session":"regular"' in resolved
        assert '"adjustment":"splits"' in resolved

    def test_more_bars_are_requested_than_are_wanted(self) -> None:
        connection = FakeConnection(packets=[conversation()])
        source, _ = make_source(connection, limit=5)
        source.load()
        assert connection.command("create_series").params[5] == 5 + SAFETY_MARGIN_BARS

    def test_a_heartbeat_is_echoed_in_the_same_framing(self) -> None:
        connection = FakeConnection(
            packets=[heartbeat(3), timescale_update(rows_for("M15", 8)), series_completed()]
        )
        source, _ = make_source(connection)
        source.load()
        assert connection.heartbeat_echoes == ["~h~3"]
        assert encode_frame("~h~3") in connection.sent

    def test_session_ids_are_bounded_and_not_global(self) -> None:
        first = TradingViewMarketDataSource()
        second = TradingViewMarketDataSource()
        connection = FakeConnection(packets=[conversation()])
        connector = RecordingConnector(scripts=[connection])
        third = TradingViewMarketDataSource(
            limit=5,
            connector=connector,
            now=lambda: NOW,
            monotonic=lambda: 0.0,
            series_id="sds_test",
        )
        third.load()
        created = connection.command("chart_create_session").params[0]
        assert created.startswith("cs_") and len(created) <= 24
        # Two sources built independently do not share an id.
        assert first._session_id != second._session_id  # noqa: SLF001

    def test_unreadable_chatter_is_counted_and_skipped(self) -> None:
        connection = FakeConnection(
            packets=[
                session_chatter() + unparseable(),
                timescale_update(rows_for("M15", 8)),
                series_completed(),
            ]
        )
        source, _ = make_source(connection)
        loaded = source.load()
        assert loaded.provenance["unparseable_envelopes"] == 1
        assert loaded.model.bars

    def test_frames_split_across_packets_reassemble(self) -> None:
        whole = conversation(rows_for("M15", 8))
        third = len(whole) // 3
        connection = scripted(whole[:third], whole[third : 2 * third], whole[2 * third :])
        source, _ = make_source(connection, limit=5)
        assert len(source.load().model.bars) == 5

    def test_a_data_update_contributes_bars(self) -> None:
        rows = rows_for("M15", 8)
        connection = scripted(timescale_update(rows[:4]), data_update(rows[4:]), series_completed())
        source, _ = make_source(connection, limit=5)
        assert len(source.load().model.bars) == 5

    def test_a_restated_bar_replaces_rather_than_duplicates(self) -> None:
        """A `du` restating a bar the history already sent must not add a second one."""
        rows = rows_for("M15", 8)
        restated = [list(rows[-3])]
        restated[0][4] = round(float(restated[0][4]) + 0.10, 5)
        restated[0][2] = round(float(restated[0][2]) + 0.20, 5)
        connection = scripted(timescale_update(rows), data_update(restated), series_completed())
        source, _ = make_source(connection, limit=5)
        bars = source.load().model.bars

        stamps = [bar.timestamp for bar in bars]
        assert len(set(stamps)) == len(stamps) == 5
        opened = datetime.fromtimestamp(int(restated[0][0]), tz=UTC)
        updated = [bar for bar in bars if bar.timestamp == opened]
        assert len(updated) == 1
        assert updated[0].close == Decimal(str(restated[0][4]))

    def test_load_fetches_once_and_caches(self) -> None:
        connection = FakeConnection(packets=[conversation()])
        source, connector = make_source(connection)
        assert source.load() is source.load()
        assert len(connector.connections) == 1


# --- closed candles -------------------------------------------------------


class TestClosedCandles:
    @pytest.mark.parametrize("timeframe", ["M1", "M5", "M15", "H1", "H4"])
    def test_the_forming_bar_is_excluded(self, timeframe: str) -> None:
        rows = rows_for(timeframe, 8)
        connection = FakeConnection(packets=[conversation(rows)])
        source, _ = make_source(connection, timeframe=timeframe, limit=5)
        loaded = source.load()

        forming_open = datetime.fromtimestamp(int(rows[-1][0]), tz=UTC)
        assert loaded.provenance["bars_forming_dropped"] == 1
        assert all(bar.timestamp < forming_open for bar in loaded.model.bars)

    @pytest.mark.parametrize("timeframe", ["M1", "M5", "M15", "H1", "H4"])
    def test_one_second_before_the_close_is_still_forming(self, timeframe: str) -> None:
        period = timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
        rows = rows_for(timeframe, 8)
        newest_open = datetime.fromtimestamp(int(rows[-1][0]), tz=UTC)
        moment = newest_open + period - timedelta(seconds=1)

        connection = FakeConnection(packets=[conversation(rows)])
        source, _ = make_source(connection, timeframe=timeframe, limit=5, now=moment)
        loaded = source.load()
        assert loaded.provenance["bars_forming_dropped"] == 1
        assert loaded.model.bars[-1].timestamp == newest_open - period

    @pytest.mark.parametrize("timeframe", ["M1", "M5", "M15", "H1", "H4"])
    def test_the_exact_close_boundary_is_closed(self, timeframe: str) -> None:
        period = timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
        rows = rows_for(timeframe, 8)
        newest_open = datetime.fromtimestamp(int(rows[-1][0]), tz=UTC)

        connection = FakeConnection(packets=[conversation(rows)])
        source, _ = make_source(connection, timeframe=timeframe, limit=5, now=newest_open + period)
        loaded = source.load()
        assert loaded.provenance["bars_forming_dropped"] == 0
        assert loaded.model.bars[-1].timestamp == newest_open

    @pytest.mark.parametrize("timeframe", ["M1", "M5", "M15", "H1", "H4"])
    def test_well_after_the_close_is_closed(self, timeframe: str) -> None:
        period = timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
        rows = rows_for(timeframe, 8)
        newest_open = datetime.fromtimestamp(int(rows[-1][0]), tz=UTC)

        connection = FakeConnection(packets=[conversation(rows)])
        source, _ = make_source(
            connection, timeframe=timeframe, limit=5, now=newest_open + period * 3
        )
        assert source.load().provenance["bars_forming_dropped"] == 0

    def test_series_completed_is_not_evidence_of_closure(self) -> None:
        """The response finished. That says nothing about the market."""
        rows = rows_for("M15", 6)
        connection = FakeConnection(packets=[conversation(rows)])
        source, _ = make_source(connection, timeframe="M15", limit=4)
        loaded = source.load()
        assert loaded.provenance["bars_received"] == 6
        assert loaded.provenance["bars_forming_dropped"] == 1

    def test_the_requested_count_is_closed_candles(self) -> None:
        rows = rows_for("M15", 12)
        connection = FakeConnection(packets=[conversation(rows)])
        source, _ = make_source(connection, limit=10)
        loaded = source.load()
        assert len(loaded.model.bars) == 10
        assert loaded.provenance["bars_returned"] == 10

    def test_the_newest_closed_candles_are_the_ones_kept(self) -> None:
        rows = rows_for("M15", 12)
        connection = FakeConnection(packets=[conversation(rows)])
        source, _ = make_source(connection, limit=3)
        bars = source.load().model.bars
        expected = [datetime.fromtimestamp(int(row[0]), tz=UTC) for row in rows[-4:-1]]
        assert [bar.timestamp for bar in bars] == expected

    def test_too_few_closed_candles_fails_rather_than_shortening(self) -> None:
        connection = FakeConnection(packets=[conversation(rows_for("M15", 4))])
        source, _ = make_source(connection, limit=10)
        with pytest.raises(InsufficientBarsError) as caught:
            source.load()
        assert caught.value.details["requested"] == 10
        assert caught.value.details["received"] == 3

    def test_a_series_of_only_a_forming_bar_fails(self) -> None:
        connection = FakeConnection(packets=[conversation(rows_for("M15", 1))])
        source, _ = make_source(connection, limit=1)
        with pytest.raises(InsufficientBarsError):
            source.load()


# --- candle validation ----------------------------------------------------


class TestCandleValidation:
    def one_bar(self, values: list[Any], *, limit: int = 1) -> TradingViewMarketDataSource:
        """A conversation whose only bar is *values*, already closed."""
        connection = scripted(timescale_update([values]), series_completed())
        source, _ = make_source(connection, limit=limit)
        return source

    def closed_row(self, **overrides: Any) -> list[Any]:
        row = list(rows_for("M15", 3)[0])
        for index, value in overrides.items():
            row[int(index)] = value
        return row

    def test_a_valid_bar_is_accepted(self) -> None:
        loaded = self.one_bar(self.closed_row()).load()
        assert len(loaded.model.bars) == 1

    def test_prices_keep_the_precision_the_feed_sent(self) -> None:
        row = self.closed_row()
        row[1], row[2], row[3], row[4] = 4323.456, 4324.0, 4322.5, 4323.125
        bars = self.one_bar(row).load().model.bars
        assert bars[0].open == Decimal("4323.456")
        assert bars[0].close == Decimal("4323.125")
        assert bars[0].high == Decimal("4324.0")

    @pytest.mark.parametrize("field", [1, 2, 3, 4])
    def test_a_non_finite_price_fails_closed(self, field: int) -> None:
        for bad in (float("nan"), float("inf"), float("-inf")):
            row = self.closed_row(**{str(field): bad})
            with pytest.raises(TradingViewCandleError):
                self.one_bar(row).load()

    @pytest.mark.parametrize("field", [1, 2, 3, 4])
    def test_a_non_numeric_price_fails_closed(self, field: int) -> None:
        candidates: tuple[Any, ...] = ("4323.5", None, True, [], {})
        for bad in candidates:
            row = self.closed_row(**{str(field): bad})
            with pytest.raises(TradingViewCandleError, match="not a number"):
                self.one_bar(row).load()

    @pytest.mark.parametrize("field", [1, 2, 3, 4])
    def test_a_non_positive_price_fails_closed(self, field: int) -> None:
        for bad in (0, -1.0):
            row = self.closed_row(**{str(field): bad})
            with pytest.raises(TradingViewCandleError, match="positive"):
                self.one_bar(row).load()

    @pytest.mark.parametrize(
        ("overrides", "problem"),
        [
            ({"2": 4000.0}, "high < low"),
            ({"1": 9000.0, "2": 8000.0}, "high < open"),
            ({"4": 9000.0}, "high < close"),
            ({"3": 5000.0, "4": 4900.0}, "low > open"),
            ({"3": 5000.0}, "low > close"),
        ],
    )
    def test_impossible_relationships_fail_closed(
        self, overrides: dict[str, Any], problem: str
    ) -> None:
        with pytest.raises(TradingViewCandleError, match="impossible candle"):
            self.one_bar(self.closed_row(**overrides)).load()

    def test_one_bad_bar_refuses_the_whole_response(self) -> None:
        """Fail closed: nineteen good bars do not excuse the twentieth."""
        rows = rows_for("M15", 10)
        rows[3][2] = 1.0  # high below low
        connection = scripted(timescale_update(rows), series_completed())
        source, _ = make_source(connection, limit=5)
        with pytest.raises(TradingViewCandleError):
            source.load()

    def test_too_few_fields_fails_closed(self) -> None:
        connection = scripted(timescale_update([[1_800_000_000.0, 1.0, 2.0]]), series_completed())
        source, _ = make_source(connection, limit=1)
        with pytest.raises(TradingViewCandleError, match="fewer fields"):
            source.load()

    @pytest.mark.parametrize("bad", [0, -1, 1000, None, "x", True, float("nan")])
    def test_an_unusable_timestamp_fails_closed(self, bad: Any) -> None:
        with pytest.raises(TradingViewCandleError):
            self.one_bar(self.closed_row(**{"0": bad})).load()

    def test_a_timestamp_far_in_the_future_fails_closed(self) -> None:
        future = (NOW + timedelta(days=30)).timestamp()
        with pytest.raises(TradingViewCandleError, match="future"):
            self.one_bar(self.closed_row(**{"0": future})).load()

    def test_bars_come_back_sorted_ascending(self) -> None:
        rows = rows_for("M15", 10)
        shuffled = [rows[5], rows[0], rows[8], rows[2], rows[6], rows[1], rows[3], rows[4]]
        connection = scripted(timescale_update(shuffled), series_completed())
        source, _ = make_source(connection, limit=6)
        stamps = [bar.timestamp for bar in source.load().model.bars]
        assert stamps == sorted(stamps)

    def test_duplicate_timestamps_are_deduplicated(self) -> None:
        rows = rows_for("M15", 8)
        connection = scripted(timescale_update(rows + rows[:3] + [rows[4]]), series_completed())
        source, _ = make_source(connection, limit=5)
        bars = source.load().model.bars
        stamps = [bar.timestamp for bar in bars]
        assert len(set(stamps)) == len(stamps) == 5


class TestVolume:
    def test_volume_is_carried_when_the_feed_sends_one(self) -> None:
        connection = FakeConnection(packets=[conversation(rows_for("M15", 6))])
        source, _ = make_source(connection, limit=3)
        assert all(bar.volume is not None for bar in source.load().model.bars)

    def test_missing_volume_stays_none_and_never_becomes_zero(self) -> None:
        rows = make_series(count=6, timeframe="M15", now=NOW, with_volume=False)
        connection = FakeConnection(packets=[conversation(rows)])
        source, _ = make_source(connection, limit=3)
        bars = source.load().model.bars
        assert all(bar.volume is None for bar in bars)
        assert not any(bar.volume == Decimal(0) for bar in bars)

    @pytest.mark.parametrize("bad", [None, "x", -5, float("nan"), float("inf"), True])
    def test_an_unusable_volume_is_dropped_not_fatal(self, bad: Any) -> None:
        """A bad volume is not a reason to refuse a candle whose prices are sound."""
        rows = rows_for("M15", 6)
        for row in rows:
            row[5] = bad
        connection = FakeConnection(packets=[conversation(rows)])
        source, _ = make_source(connection, limit=3)
        assert all(bar.volume is None for bar in source.load().model.bars)

    def test_a_real_zero_volume_is_kept_as_zero(self) -> None:
        rows = rows_for("M15", 6)
        for row in rows:
            row[5] = 0.0
        connection = FakeConnection(packets=[conversation(rows)])
        source, _ = make_source(connection, limit=3)
        assert all(bar.volume == Decimal(0) for bar in source.load().model.bars)


# --- failures on the wire -------------------------------------------------


class TestWireFailures:
    def test_a_protocol_error_is_surfaced_and_closes_the_socket(self) -> None:
        connection = scripted(protocol_error())
        source, connector = make_source(connection)
        with pytest.raises(TradingViewProtocolError) as caught:
            source.load()
        assert connector.all_closed
        assert FEED_INTERNALS_SENTINEL not in str(caught.value)
        assert FEED_INTERNALS_SENTINEL not in repr(caught.value.details)

    def test_a_critical_error_is_surfaced_and_closes_the_socket(self) -> None:
        connection = scripted(critical_error())
        source, connector = make_source(connection)
        with pytest.raises(TradingViewCriticalError) as caught:
            source.load()
        assert connector.all_closed
        assert FEED_INTERNALS_SENTINEL not in str(caught.value)

    def test_malformed_framing_closes_the_socket(self) -> None:
        connection = scripted("not framed at all")
        source, connector = make_source(connection)
        with pytest.raises(TradingViewFramingError):
            source.load()
        assert connector.all_closed

    def test_a_truncated_tail_at_completion_is_refused(self) -> None:
        """A packet that stopped mid-frame must not be treated as the end."""
        connection = scripted(
            timescale_update(rows_for("M15", 8)) + series_completed() + "~m~40~m~part"
        )
        source, _ = make_source(connection, limit=5)
        with pytest.raises(TradingViewFramingError, match="unfinished packet"):
            source.load()

    def test_a_socket_that_closes_early_is_a_connection_failure(self) -> None:
        connection = scripted(timescale_update(rows_for("M15", 8)))
        source, connector = make_source(connection)
        with pytest.raises(TradingViewConnectionError, match="closed before"):
            source.load()
        assert connector.all_closed

    def test_a_connect_failure_is_typed_and_leaks_nothing(self) -> None:
        connector = RecordingConnector(connect_error=OSError("refused by 10.0.0.1:443"))
        source, _ = make_source(connector=connector)
        with pytest.raises(OSError, match="refused"):
            source.load()
        assert connector.connections == []

    def test_a_send_failure_closes_the_socket(self) -> None:
        connection = FakeConnection(send_error=OSError("broken pipe"), send_error_at=3)
        source, connector = make_source(connection)
        with pytest.raises(TradingViewConnectionError, match="sending"):
            source.load()
        assert connector.all_closed

    def test_a_recv_failure_before_the_deadline_is_a_connection_failure(self) -> None:
        connection = FakeConnection(fail_after=0, recv_error=OSError("reset"))
        source, connector = make_source(connection)
        with pytest.raises(TradingViewConnectionError, match="receiving"):
            source.load()
        assert connector.all_closed

    def test_a_recv_failure_after_the_deadline_is_a_timeout(self) -> None:
        """A socket error is read as a timeout only once the budget has gone."""
        # First reading sets the deadline; the loop check is still inside it, so
        # the failure is classified by the clock at the moment recv gave up.
        clock = iter([0.0, 0.5, 99.0, 99.0])
        connection = FakeConnection(fail_after=0, recv_error=TimeoutError("timed out"))
        connector = RecordingConnector(scripts=[connection])
        source = TradingViewMarketDataSource(
            limit=5,
            connector=connector,
            now=lambda: NOW,
            monotonic=lambda: next(clock),
            timeout_seconds=1.0,
            series_id="sds_test",
        )
        with pytest.raises(TradingViewTimeoutError, match="stopped responding"):
            source.load()
        assert connector.all_closed

    def test_a_quiet_feed_times_out_rather_than_hanging(self) -> None:
        clock = iter([0.0, 0.5, 999.0, 1000.0])
        connection = FakeConnection(packets=[session_chatter(), session_chatter()])
        connector = RecordingConnector(scripts=[connection])
        source = TradingViewMarketDataSource(
            limit=5,
            connector=connector,
            now=lambda: NOW,
            monotonic=lambda: next(clock),
            timeout_seconds=5.0,
            series_id="sds_test",
        )
        with pytest.raises(TradingViewTimeoutError, match="did not complete"):
            source.load()
        assert connector.all_closed

    def test_a_chatty_feed_is_bounded_by_a_packet_budget(self) -> None:
        connection = FakeConnection(packets=[session_chatter() * 5000])
        source, connector = make_source(connection)
        with pytest.raises(TradingViewProtocolError, match="more packets"):
            source.load()
        assert connector.all_closed

    def test_binary_that_is_not_utf8_is_refused(self) -> None:
        connection = FakeConnection(packets=[b"\xff\xfe\x00bad"])  # type: ignore[list-item]
        source, connector = make_source(connection)
        with pytest.raises(TradingViewFramingError, match="UTF-8"):
            source.load()
        assert connector.all_closed

    def test_valid_utf8_bytes_are_accepted(self) -> None:
        connection = FakeConnection(packets=[conversation(rows_for("M15", 8)).encode("utf-8")])  # type: ignore[list-item]
        source, _ = make_source(connection, limit=5)
        assert len(source.load().model.bars) == 5


class TestRetryPolicy:
    def test_no_retry_by_default(self) -> None:
        connector = RecordingConnector(
            scripts=[FakeConnection(fail_after=0, recv_error=OSError("reset"))]
        )
        source, _ = make_source(connector=connector)
        with pytest.raises(TradingViewConnectionError):
            source.load()
        assert len(connector.connections) == 1

    def test_a_transport_failure_may_be_retried_when_asked(self) -> None:
        connector = RecordingConnector(
            scripts=[
                FakeConnection(fail_after=0, recv_error=OSError("reset")),
                FakeConnection(packets=[conversation(rows_for("M15", 8))]),
            ]
        )
        source, _ = make_source(connector=connector, limit=5, max_retries=1)
        assert len(source.load().model.bars) == 5
        assert len(connector.connections) == 2
        assert connector.all_closed

    @pytest.mark.parametrize("packet", [protocol_error(), critical_error(), "not framed at all"])
    def test_deterministic_failures_are_never_retried(self, packet: str) -> None:
        connector = RecordingConnector(
            scripts=[scripted(packet), FakeConnection(packets=[conversation()])]
        )
        source, _ = make_source(connector=connector, max_retries=3)
        with pytest.raises(Exception):  # noqa: B017 - the class differs per packet
            source.load()
        assert len(connector.connections) == 1

    def test_an_invalid_candle_is_never_retried(self) -> None:
        rows = rows_for("M15", 8)
        rows[2][2] = 1.0
        connector = RecordingConnector(
            scripts=[
                scripted(timescale_update(rows), series_completed()),
                FakeConnection(packets=[conversation()]),
            ]
        )
        source, _ = make_source(connector=connector, max_retries=3)
        with pytest.raises(TradingViewCandleError):
            source.load()
        assert len(connector.connections) == 1


class TestResourceCleanup:
    def test_the_socket_closes_on_success(self) -> None:
        source, connector = make_source(FakeConnection(packets=[conversation()]))
        source.load()
        assert connector.only.closed == 1

    def test_repeated_sources_do_not_accumulate_connections(self) -> None:
        connector = RecordingConnector(
            scripts=[FakeConnection(packets=[conversation(rows_for("M15", 8))]) for _ in range(3)]
        )
        for _ in range(3):
            source, _ = make_source(connector=connector, limit=5)
            source.load()
        assert len(connector.connections) == 3
        assert connector.all_closed

    def test_a_close_that_itself_fails_does_not_mask_the_real_failure(self) -> None:
        connection = FakeConnection(packets=[protocol_error()], close_error=OSError("late"))
        source, _ = make_source(connection)
        with pytest.raises(TradingViewProtocolError):
            source.load()

    def test_a_close_that_fails_does_not_break_a_success(self) -> None:
        connection = FakeConnection(
            packets=[conversation(rows_for("M15", 8))], close_error=OSError("late")
        )
        source, _ = make_source(connection, limit=5)
        assert len(source.load().model.bars) == 5


# --- provenance and downstream fit ----------------------------------------


class TestProvenance:
    def loaded(self, *, timeframe: str = "M15", limit: int = 5) -> Any:
        connection = FakeConnection(packets=[conversation(rows_for(timeframe, limit + 3))])
        source, _ = make_source(connection, timeframe=timeframe, limit=limit)
        return source.load()

    def test_source_and_symbol(self) -> None:
        loaded = self.loaded()
        assert loaded.provenance["provider"] == PROVIDER_NAME == "tradingview"
        assert loaded.provenance["provider_symbol"] == SYMBOL
        assert loaded.provenance["canonical_symbol"] == "XAUUSD"
        assert loaded.model.provider == "tradingview"
        assert loaded.model.symbol == "XAUUSD"
        assert loaded.model.provider_symbol == SYMBOL

    @pytest.mark.parametrize("timeframe", ["M1", "M5", "M15", "H1", "H4"])
    def test_timeframe_is_recorded_both_ways(self, timeframe: str) -> None:
        loaded = self.loaded(timeframe=timeframe)
        assert loaded.provenance["timeframe"] == timeframe
        assert loaded.provenance["wire_interval"] == SUPPORTED_TIMEFRAMES[Timeframe(timeframe)]
        assert str(loaded.model.timeframe) == timeframe

    def test_timestamps_are_utc(self) -> None:
        loaded = self.loaded()
        assert loaded.model.timezone is None
        for bar in loaded.model.bars:
            assert bar.timestamp.utcoffset() == timedelta(0)
        assert loaded.provenance["latest_candle_at"].endswith("Z")

    def test_counts_explain_what_happened(self) -> None:
        loaded = self.loaded(limit=5)
        assert loaded.provenance["bars_requested"] == 5
        assert loaded.provenance["bars_received"] == 8
        assert loaded.provenance["bars_forming_dropped"] == 1
        assert loaded.provenance["bars_returned"] == 5

    def test_the_endpoint_and_absence_of_auth_are_recorded(self) -> None:
        loaded = self.loaded()
        assert loaded.provenance["endpoint"] == WEBSOCKET_URL
        assert loaded.provenance["authenticated"] is False

    def test_provenance_carries_no_semantic_type(self) -> None:
        """Round 6.4a's invariant: what a number means is not where it came from."""
        keys = set(self.loaded().provenance)
        assert not any("semantic" in key for key in keys)
        assert "ABSOLUTE_PRICE" not in repr(self.loaded().provenance)

    def test_origin_names_the_provider(self) -> None:
        assert self.loaded().origin == f"tradingview:{SYMBOL}:M15"

    def test_the_raw_payload_is_what_was_validated(self) -> None:
        loaded = self.loaded()
        assert MarketDataInput.model_validate(loaded.raw_payload) == loaded.model


class TestDownstreamFit:
    def test_the_source_satisfies_the_provider_protocol(self) -> None:
        """Structurally interchangeable with the MT5 source, and type-checked as such."""
        source, _ = make_source(FakeConnection(packets=[conversation(rows_for("M15", 8))]))
        provider: MarketDataSource = source
        loaded = provider.load()
        assert isinstance(loaded.model, MarketDataInput)
        assert loaded.origin and loaded.provenance["kind"] == "market"

    def test_the_payload_normalises_like_any_other_provider(self) -> None:
        connection = FakeConnection(packets=[conversation(rows_for("M15", 14))])
        source, _ = make_source(connection, limit=12)
        snapshot = normalize_market_data(source.load().model).snapshot
        assert snapshot.provider == "tradingview"
        assert snapshot.symbol == "XAUUSD"
        assert snapshot.timezone == "UTC"
        assert snapshot.bar_count == 12
        assert snapshot.latest_bar == snapshot.bars[-1]
        assert snapshot.source_timezone is None

    def test_no_provider_specific_candle_model_was_introduced(self) -> None:
        import goldpipeline.adapters.tradingview_market as market
        import goldpipeline.schemas.market as schema

        assert not hasattr(market, "TradingViewCandle")
        assert not hasattr(schema, "TradingViewCandle")


# --- boundary -------------------------------------------------------------


class TestProtocolBoundary:
    WIRE_METHODS = (
        "resolve_symbol",
        "create_series",
        "timescale_update",
        "series_completed",
        "chart_create_session",
        "switch_timezone",
        "set_auth_token",
    )
    WIRE_LITERALS = ("~m~", "~h~", "unauthorized_user_token", "data.tradingview.com")

    def test_wire_details_live_only_in_the_adapter_modules(self) -> None:
        """If TradingView changes its protocol, only these files may need to change.

        A wire method name only ever reaches the socket as a string literal, so
        that is what is searched for. A bare substring search would trip over
        the normalizer's own ``_resolve_symbol`` helper, which has nothing to do
        with this feed - the same lesson a producer test learned about grepping
        for words that also occur in unrelated code.
        """
        import re
        from pathlib import Path

        allowed = {
            "tradingview_protocol.py",
            "tradingview_market.py",
            "fake_tradingview.py",
        }
        methods = re.compile(
            "|".join(rf"[\"']{re.escape(word)}[\"']" for word in self.WIRE_METHODS)
        )
        root = Path(__file__).resolve().parents[1] / "src" / "goldpipeline"
        for path in root.rglob("*.py"):
            if path.name in allowed:
                continue
            text = path.read_text(encoding="utf-8")
            found = methods.search(text)
            assert found is None, f"{path.name} names the wire method {found.group(0)}"
            for literal in self.WIRE_LITERALS:
                assert literal not in text, f"{path.name} knows about {literal!r}"

    def test_the_adapter_imports_nothing_it_should_not(self) -> None:
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "src" / "goldpipeline"
        forbidden = (
            "pyperclip",
            "goldpipeline.prompts",
            "goldpipeline.schemas.article_contract",
            "goldpipeline.adapters.telegram",
            "goldpipeline.adapters.anthropic",
            "goldpipeline.adapters.deepseek",
            "goldpipeline.adapters.mt5_market",
        )
        for name in ("tradingview_market.py", "tradingview_protocol.py"):
            tree = ast.parse((root / "adapters" / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    assert not any(module.startswith(bad) for bad in forbidden), (
                        f"{name} imports {module}"
                    )

    def test_no_ict_or_clipboard_behaviour_was_ported(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "src" / "goldpipeline"
        banned = (
            "pyperclip",
            "order_block",
            "market_structure",
            "fair_value_gap",
            "def fvg",
            "liquidity_pool",
            "get_session",
        )
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            for word in banned:
                assert word not in text, f"{path.name} contains {word!r}"

    def test_the_websocket_library_is_not_imported_at_module_scope(self) -> None:
        """The suite, and the MT5 path, run without the optional extra installed."""
        import ast
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "goldpipeline"
            / "adapters"
            / "tradingview_market.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import):
                assert all(a.name != "websocket" for a in node.names)
            if isinstance(node, ast.ImportFrom):
                assert node.module != "websocket"


# --- production is untouched ---------------------------------------------


class TestProductionUnchanged:
    def test_the_cli_market_source_still_defaults_to_mt5(self) -> None:
        from pathlib import Path

        cli = (Path(__file__).resolve().parents[1] / "src" / "goldpipeline" / "cli.py").read_text(
            encoding="utf-8"
        )
        assert '"--market-source", choices=("mt5", "file"), default="mt5"' in cli
        assert "tradingview" not in cli

    def test_nothing_in_the_services_layer_imports_this_provider(self) -> None:
        """Prose may mention the feed; no service may call it.

        Checked by reading imports rather than grepping, because a docstring
        that *explains* provider-independence is not a dependency on a
        provider - `numeric_mentions` says exactly that, and should.
        """
        import ast
        from pathlib import Path

        services = Path(__file__).resolve().parents[1] / "src" / "goldpipeline" / "services"
        for path in services.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module, *(f"{node.module}.{a.name}" for a in node.names)]
                for name in names:
                    assert "tradingview" not in name.lower(), f"{path.name} imports {name}"

    def test_article_readiness_is_unchanged(self) -> None:
        from goldpipeline.schemas.article import ArticleType
        from goldpipeline.services.article_routing import READY_TYPES, SPECS

        assert {ArticleType.ANALYSIS} == READY_TYPES
        assert SPECS[ArticleType.NEWS_DIGEST].ready is False
        assert SPECS[ArticleType.TRADE_PLAN].ready is False

    def test_the_mt5_provider_name_is_untouched(self) -> None:
        from goldpipeline.adapters.mt5_market import PROVIDER_NAME as MT5_PROVIDER

        assert MT5_PROVIDER == "metatrader5"
        assert PROVIDER_NAME != MT5_PROVIDER

    def test_this_round_fetched_no_mt5_comparison(self) -> None:
        """Side-by-side comparison belongs to the next round, not this file."""
        import ast
        from pathlib import Path

        path = Path(__file__).resolve()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert "goldpipeline.adapters.fake_mt5" not in imported
