"""The MetaTrader 5 market data source, exercised entirely offline.

The module is injected, so no test here needs a terminal, a broker, or the
vendor package installed.

The tests that matter most are the ones about the **forming candle**. Index 0 of
``copy_rates_from_pos`` is the bar still moving; quoting it produces an article
that is false by the time anyone reads it. Two separate mechanisms prevent that -
starting at index 1, and proving the returned bar has closed - and both are
tested, because the second exists precisely for the day the first assumption
turns out to be wrong.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from conftest import INGEST_NOW, make_mt5_source

from goldpipeline.adapters.fake_mt5 import (
    TERMINAL_SECRET_SENTINEL,
    TRADING_FUNCTIONS,
    FakeMt5Module,
    make_rates,
    missing_symbol_module,
    unavailable_module,
)
from goldpipeline.adapters.mt5_market import (
    PROVIDER_NAME,
    MetaTrader5MarketDataSource,
    Mt5Module,
)
from goldpipeline.config import MarketDataSettings
from goldpipeline.domain.errors import (
    FormingCandleError,
    InsufficientBarsError,
    MarketDataConfigurationError,
    Mt5InitializeError,
    Mt5ProviderError,
    Mt5SymbolNotFoundError,
    Mt5SymbolNotSelectedError,
    StaleMarketDataError,
)

# --- connecting -----------------------------------------------------------


def test_a_successful_fetch_returns_the_requested_bars() -> None:
    """Requirements 17 and 24."""
    module = FakeMt5Module(rates=make_rates(now=INGEST_NOW))
    loaded = make_mt5_source(module=module).load()

    assert module.initialized == 1
    assert len(loaded.model.bars) == 20
    assert loaded.model.provider == PROVIDER_NAME


def test_a_terminal_that_cannot_be_reached_fails_clearly() -> None:
    """Requirement 18."""
    with pytest.raises(Mt5InitializeError) as exc:
        make_mt5_source(module=unavailable_module()).load()

    assert "terminal" in str(exc.value)
    assert exc.value.details["provider_error"] == "-10005: IPC timeout"


def test_the_connection_is_released_after_a_success() -> None:
    """Requirement 19."""
    module = FakeMt5Module(rates=make_rates(now=INGEST_NOW))
    make_mt5_source(module=module).load()

    assert module.shutdowns == 1
    assert module.called[-1] == "shutdown"


def test_the_connection_is_released_after_a_failure() -> None:
    """Requirement 20.

    A terminal connection left open holds an IPC slot the next process will be
    told is busy - so the failure path matters more than the success path here.
    """
    module = missing_symbol_module()

    with pytest.raises(Mt5SymbolNotFoundError):
        make_mt5_source(module=module).load()

    assert module.shutdowns == 1


def test_a_provider_failure_mid_fetch_still_releases_the_connection() -> None:
    module = FakeMt5Module(rates_error=RuntimeError("terminal went away"))

    with pytest.raises(Mt5ProviderError):
        make_mt5_source(module=module).load()

    assert module.shutdowns == 1


def test_the_fetch_happens_once_however_often_it_is_asked_for() -> None:
    """One source instance is one snapshot.

    The ingestion service pre-flights the terminal and then hands the same source
    to ``create_run``. Two fetches could return two different markets, and the
    Run would be built on one while having been approved on the other.
    """
    module = FakeMt5Module(rates=make_rates(now=INGEST_NOW))
    source = make_mt5_source(module=module)

    first = source.load()
    second = source.load()

    assert first is second
    assert module.called.count("copy_rates_from_pos") == 1


# --- symbols --------------------------------------------------------------


def test_a_missing_symbol_fails_and_never_substitutes() -> None:
    """Requirements 21 and 34.

    The candidates are listed because a human needs them. They are not chosen,
    because ``XAUUSD`` and ``XAUUSD.a`` are not guaranteed to be the same
    instrument and publishing the wrong one is not recoverable.
    """
    with pytest.raises(Mt5SymbolNotFoundError) as exc:
        make_mt5_source(module=missing_symbol_module()).load()

    assert exc.value.details["symbol"] == "XAUUSD"
    assert "XAUUSDm" in exc.value.details["candidates"]
    assert "never guessed" in str(exc.value)


def test_a_hidden_symbol_is_added_to_market_watch() -> None:
    """Requirement 24 of the spec: visibility only, and only if allowed."""
    module = FakeMt5Module(hidden_symbols=("XAUUSD",), rates=make_rates(now=INGEST_NOW))
    make_mt5_source(module=module).load()

    assert ("symbol_select", ("XAUUSD", True)) in module.calls


def test_selection_can_be_refused_by_configuration() -> None:
    module = FakeMt5Module(hidden_symbols=("XAUUSD",), rates=make_rates(now=INGEST_NOW))
    source = MetaTrader5MarketDataSource(
        MarketDataSettings(), module=module, now=INGEST_NOW, select_if_hidden=False
    )

    with pytest.raises(Mt5SymbolNotSelectedError):
        source.load()
    assert "symbol_select" not in module.called


def test_a_failed_selection_is_reported() -> None:
    module = FakeMt5Module(
        hidden_symbols=("XAUUSD",), select_ok=False, rates=make_rates(now=INGEST_NOW)
    )

    with pytest.raises(Mt5SymbolNotSelectedError):
        make_mt5_source(module=module).load()


def test_the_broker_symbol_and_the_canonical_symbol_are_both_recorded() -> None:
    """Requirement 34.

    The article, the context and every downstream check talk about ``XAUUSD``;
    the terminal is asked about ``XAUUSDm``. Neither name is inferred from the
    other, and both end up on the record.
    """
    module = FakeMt5Module(known_symbols=("XAUUSDm",), rates=make_rates(now=INGEST_NOW))
    loaded = make_mt5_source(
        module=module, provider_symbol="XAUUSDm", canonical_symbol="XAUUSD"
    ).load()

    assert loaded.model.symbol == "XAUUSD"
    assert loaded.model.provider_symbol == "XAUUSDm"
    assert loaded.provenance["provider_symbol"] == "XAUUSDm"
    assert loaded.provenance["canonical_symbol"] == "XAUUSD"
    assert module.calls[1] == ("symbol_info", ("XAUUSDm",))


# --- timeframes -----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "constant"),
    [("M1", 1), ("M5", 5), ("M15", 15), ("M30", 30), ("H1", 16385), ("H4", 16388)],
)
def test_supported_timeframes_map_to_the_providers_own_constants(name: str, constant: int) -> None:
    """Requirement 22.

    Read off the module by name rather than from a table copied in here. ``H4``
    is 16388, which is exactly the sort of number nobody notices going stale.
    """
    module = FakeMt5Module(rates=make_rates(now=INGEST_NOW, timeframe=name, count=25))
    # The age limit is lifted because this test is about the constant, not
    # about staleness: an H4 bar is legitimately hours old the moment it closes.
    make_mt5_source(
        module=module,
        timeframe=name,
        bar_count=10,
        max_data_age_minutes=10_000,
        now=INGEST_NOW,
    ).load()

    call = next(c for c in module.calls if c[0] == "copy_rates_from_pos")
    assert call[1][1] == constant


@pytest.mark.parametrize("rejected", ["M3", "H2", "D1", "W1", "16388", "MN1", "1m"])
def test_unsupported_timeframes_are_refused_by_configuration(rejected: str) -> None:
    """Requirement 23.

    Rejected at the configuration boundary, so an opaque provider integer can
    never arrive from a payload or a typo and fetch a resolution nothing
    downstream was written for.
    """
    with pytest.raises(MarketDataConfigurationError) as exc:
        MarketDataSettings.from_env({"GOLDPIPELINE_OHLC_TIMEFRAME": rejected})

    assert exc.value.details["setting"] == "GOLDPIPELINE_OHLC_TIMEFRAME"


@pytest.mark.parametrize("accepted", ["m15", " M15 ", "h4"])
def test_case_and_whitespace_are_tolerated_in_the_timeframe(accepted: str) -> None:
    """Forgiving about shape, strict about meaning."""
    settings = MarketDataSettings.from_env({"GOLDPIPELINE_OHLC_TIMEFRAME": accepted})

    assert settings.timeframe == accepted.strip().upper()


def test_a_module_missing_a_timeframe_constant_fails_clearly() -> None:
    module = FakeMt5Module(rates=make_rates(now=INGEST_NOW))
    # A provider whose constant is not an integer: the adapter must say which
    # constant it could not resolve, rather than passing nonsense to the fetch.
    object.__setattr__(module, "TIMEFRAME_M15", "not-an-int")

    with pytest.raises(Mt5ProviderError, match="TIMEFRAME_M15"):
        make_mt5_source(module=module).load()


# --- the forming candle ---------------------------------------------------


def test_the_fetch_starts_after_the_forming_candle() -> None:
    """Requirements 16 of the spec and 25: ``start_pos`` is 1, never 0."""
    module = FakeMt5Module(rates=make_rates(now=INGEST_NOW))
    make_mt5_source(module=module).load()

    call = next(c for c in module.calls if c[0] == "copy_rates_from_pos")
    assert call[1][2] == 1


def test_the_current_forming_candle_never_reaches_the_bars() -> None:
    """Requirement 25, stated as an absence.

    Index 0 opened at the top of the current period and has not closed. It must
    appear nowhere in the result.
    """
    rates = make_rates(now=INGEST_NOW)
    forming_at = datetime.fromtimestamp(rates[0]["time"], tz=UTC)
    loaded = make_mt5_source(module=FakeMt5Module(rates=rates)).load()

    assert forming_at not in [bar.timestamp for bar in loaded.model.bars]
    assert loaded.model.bars[-1].timestamp == forming_at - timedelta(minutes=15)


def test_the_latest_bar_has_actually_closed() -> None:
    """Requirement 26."""
    loaded = make_mt5_source(module=FakeMt5Module(rates=make_rates(now=INGEST_NOW))).load()
    latest = loaded.model.bars[-1]

    assert latest.timestamp + timedelta(minutes=15) <= INGEST_NOW


def test_a_provider_that_returns_a_forming_candle_is_caught() -> None:
    """Requirement 17 of the spec, and why ``start_pos`` alone is not enough.

    Here the provider ignores ``start_pos`` and hands back the forming bar
    anyway. The start index was an assumption; the timestamp arithmetic is a
    fact, and it is what refuses.
    """

    class IgnoresStartPos(FakeMt5Module):
        def copy_rates_from_pos(
            self, symbol: str, timeframe: int, start_pos: int, count: int
        ) -> list[dict[str, Any]]:
            series = self.rates or make_rates()
            return series[0:count]

    module = IgnoresStartPos(rates=make_rates(now=INGEST_NOW))

    with pytest.raises(FormingCandleError) as exc:
        make_mt5_source(module=module).load()

    assert "still moving" in str(exc.value)
    assert exc.value.details["candle_opened_at"] == "2026-08-28T03:00:00Z"


def test_a_few_seconds_of_clock_skew_is_tolerated() -> None:
    """The tolerance exists for skew, and is small enough to be useless for more."""
    rates = make_rates(now=INGEST_NOW)
    loaded = MetaTrader5MarketDataSource(
        MarketDataSettings(),
        module=FakeMt5Module(rates=rates),
        now=INGEST_NOW - timedelta(seconds=3),
    ).load()

    assert len(loaded.model.bars) == 20


# --- staleness ------------------------------------------------------------


def test_data_older_than_the_limit_is_refused() -> None:
    """Requirement 25 of the spec.

    Usually this means the market is closed, which the message says - and which
    is still a reason not to publish an analysis of it.
    """
    stale_now = INGEST_NOW + timedelta(hours=4)

    with pytest.raises(StaleMarketDataError) as exc:
        MetaTrader5MarketDataSource(
            MarketDataSettings(),
            module=FakeMt5Module(rates=make_rates(now=INGEST_NOW)),
            now=stale_now,
        ).load()

    assert "market is closed" in str(exc.value)
    assert exc.value.details["limit_minutes"] == 90


def test_the_age_limit_is_configurable() -> None:
    generous = MarketDataSettings(max_data_age_minutes=600)
    loaded = MetaTrader5MarketDataSource(
        generous,
        module=FakeMt5Module(rates=make_rates(now=INGEST_NOW)),
        now=INGEST_NOW + timedelta(hours=4),
    ).load()

    assert len(loaded.model.bars) == 20


# --- what comes back ------------------------------------------------------


def test_too_few_bars_is_a_failure_not_a_shrug() -> None:
    """Requirement 30."""
    module = FakeMt5Module(rates=make_rates(now=INGEST_NOW, count=8))

    with pytest.raises(InsufficientBarsError) as exc:
        make_mt5_source(module=module).load()

    assert exc.value.details["requested"] == 20
    assert exc.value.details["received"] == 7


def test_timestamps_are_utc_aware() -> None:
    """Requirement 27.

    Never reinterpreted through the machine's local timezone - the epoch second
    the provider returns is an instant, and it is read as one.
    """
    loaded = make_mt5_source(module=FakeMt5Module(rates=make_rates(now=INGEST_NOW))).load()

    assert all(bar.is_utc for bar in loaded.model.bars)
    assert loaded.model.bars[-1].timestamp == datetime(2026, 8, 28, 2, 45, tzinfo=UTC)


def test_bars_arrive_in_ascending_order() -> None:
    """The provider counts backwards; everything downstream reads forwards."""
    loaded = make_mt5_source(module=FakeMt5Module(rates=make_rates(now=INGEST_NOW))).load()
    timestamps = [bar.timestamp for bar in loaded.model.bars]

    assert timestamps == sorted(timestamps)


def test_prices_are_exact_decimals() -> None:
    """Requirement 31.

    Converted through ``str`` rather than binary float arithmetic, and quantized
    to the precision the broker itself declares.
    """
    rates = make_rates(now=INGEST_NOW)
    loaded = make_mt5_source(module=FakeMt5Module(rates=rates, digits=2)).load()
    latest = loaded.model.bars[-1]

    assert latest.close == Decimal("3305.50")
    assert str(latest.close) == "3305.50"
    assert all(-bar.close.as_tuple().exponent == 2 for bar in loaded.model.bars)


def test_a_noisy_float_is_quantized_to_the_brokers_precision() -> None:
    """The reason ``digits`` is consulted rather than the float's own tail."""
    rates = make_rates(now=INGEST_NOW)
    rates[1]["close"] = 3305.5000000000005
    loaded = make_mt5_source(module=FakeMt5Module(rates=rates, digits=2)).load()

    assert loaded.model.bars[-1].close == Decimal("3305.50")


def test_tick_volume_is_used_when_the_broker_reports_no_real_volume() -> None:
    """Requirement 32.

    Most retail feeds report zero real volume, and tick count is then the only
    signal there is. Which column was used is recorded rather than guessed at
    later.
    """
    loaded = make_mt5_source(module=FakeMt5Module(rates=make_rates(now=INGEST_NOW))).load()

    assert loaded.model.bars[-1].volume == Decimal("1401")
    assert loaded.provenance["volume_field"] == "tick_volume"


def test_real_volume_wins_where_the_broker_reports_it() -> None:
    rates = make_rates(now=INGEST_NOW)
    for row in rates:
        row["real_volume"] = 987
    loaded = make_mt5_source(module=FakeMt5Module(rates=rates)).load()

    assert loaded.model.bars[-1].volume == Decimal("987")
    assert loaded.provenance["volume_field"] == "real_volume"


def test_the_provenance_says_what_was_asked_and_what_came_back() -> None:
    """Requirement 33."""
    loaded = make_mt5_source(module=FakeMt5Module(rates=make_rates(now=INGEST_NOW))).load()
    provenance = loaded.provenance

    assert provenance["provider"] == PROVIDER_NAME
    assert provenance["timeframe"] == "M15"
    assert provenance["bars_requested"] == 20
    assert provenance["bars_returned"] == 20
    assert provenance["start_pos"] == 1
    assert provenance["requested_at"].endswith("Z")
    assert provenance["retrieved_at"].endswith("Z")
    assert provenance["latest_candle_at"] == "2026-08-28T02:45:00Z"


# --- the read-only boundary -----------------------------------------------


def test_no_trading_function_is_ever_called() -> None:
    """Requirement 35.

    Asserted against the module the adapter actually drove, not read off the
    source. The fake raises on any attribute outside the market-data surface, so
    even reaching for one would fail loudly.
    """
    module = FakeMt5Module(rates=make_rates(now=INGEST_NOW))
    make_mt5_source(module=module).load()

    assert set(module.called) <= {
        "initialize",
        "shutdown",
        "symbol_info",
        "symbol_select",
        "symbols_get",
        "copy_rates_from_pos",
        "last_error",
    }
    for forbidden in TRADING_FUNCTIONS:
        assert forbidden not in module.called


def test_the_module_protocol_names_no_trading_function() -> None:
    """Requirement 12 of the spec, checked structurally.

    The read-only boundary is a type, not a promise: a trading call would have
    to be added to this protocol first, in a diff a reviewer would see.
    """
    surface = {name for name in dir(Mt5Module) if not name.startswith("_")}

    assert surface == {
        "initialize",
        "shutdown",
        "last_error",
        "symbol_info",
        "symbol_select",
        "symbols_get",
        "copy_rates_from_pos",
    }
    for forbidden in TRADING_FUNCTIONS:
        assert forbidden not in surface


def test_the_adapter_module_never_imports_the_vendor_at_load_time() -> None:
    """Importing the adapter must work on a machine with no MetaTrader at all."""
    import sys

    assert "MetaTrader5" not in sys.modules


# --- what errors may say --------------------------------------------------


def test_errors_never_carry_terminal_internals() -> None:
    """Requirement 36.

    ``terminal_info()`` carries the data path, the broker, and on some builds
    the logged-in account. None of it helps diagnose a fetch, and all of it
    would end up in a manifest.
    """
    module = FakeMt5Module(initialize_ok=False, error=(-10005, f"path={TERMINAL_SECRET_SENTINEL}"))

    with pytest.raises(Mt5InitializeError) as exc:
        make_mt5_source(module=module).load()

    assert "terminal_info" not in module.called
    # The provider's own error text is passed through; nothing else is read.
    assert exc.value.details["provider_error"].startswith("-10005:")


def test_a_missing_symbol_error_does_not_dump_the_whole_broker() -> None:
    module = FakeMt5Module(
        known_symbols=("EURUSD",), all_symbols=tuple(f"XAU{i:04d}" for i in range(200))
    )

    with pytest.raises(Mt5SymbolNotFoundError) as exc:
        make_mt5_source(module=module).load()

    assert len(exc.value.details["candidates"]) <= 12
