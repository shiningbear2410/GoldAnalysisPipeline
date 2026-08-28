"""Market data validation and normalization policy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from conftest import BASE_TIME, make_bar, make_market_payload, make_series
from pydantic import ValidationError

from goldpipeline.domain.errors import (
    DuplicateTimestampError,
    EmptyBarsError,
    LatestBarMismatchError,
    NaiveTimestampError,
    SymbolMismatchError,
    UnknownTimezoneError,
)
from goldpipeline.schemas.market import MarketDataInput, OHLCBar
from goldpipeline.schemas.quality import WarningCode
from goldpipeline.services.normalizer import NormalizedMarketData, normalize_market_data


def normalize(
    payload: dict[str, Any],
    *,
    expected_symbol: str | None = None,
    now: datetime | None = None,
) -> NormalizedMarketData:
    """Parse and normalize in one step."""
    return normalize_market_data(
        MarketDataInput.model_validate(payload), expected_symbol=expected_symbol, now=now
    )


def warning_codes(result: NormalizedMarketData) -> set[WarningCode]:
    return {warning.code for warning in result.warnings}


# --- valid data -----------------------------------------------------------


def test_valid_ohlc_is_accepted(frozen_now: datetime) -> None:
    """Requirement 14.2."""
    result = normalize(make_market_payload(), now=frozen_now)
    assert result.snapshot.bar_count == 12
    assert result.snapshot.symbol == "XAUUSD"
    assert result.warnings == []


def test_prices_survive_as_exact_decimals(frozen_now: datetime) -> None:
    """No float noise: 3312.45 must stay 3312.45, not 3312.4500000000003."""
    bars = [make_bar(open_="3312.45", high="3315.10", low="3311.90", close="3314.20")]
    result = normalize(make_market_payload(bars=bars), now=frozen_now)
    latest = result.snapshot.latest_bar
    assert latest.close == Decimal("3314.20")
    assert str(latest.open) == "3312.45"


def test_numeric_json_prices_do_not_gain_float_noise(frozen_now: datetime) -> None:
    """Real providers send JSON numbers, not strings. Those must stay exact.

    Trailing zeros are not preserved (3315.10 becomes 3315.1) because the
    pipeline does not invent or pad precision. The *value* is exact, which is
    what every downstream price claim depends on.
    """
    bars = [
        {
            "timestamp": "2026-08-28T00:00:00Z",
            "open": 3312.45,
            "high": 3315.10,
            "low": 3311.90,
            "close": 3314.20,
            "volume": 1489,
        }
    ]
    result = normalize(make_market_payload(bars=bars), now=frozen_now)
    latest = result.snapshot.latest_bar
    assert latest.open == Decimal("3312.45")
    assert latest.high == Decimal("3315.10")
    assert latest.close == Decimal("3314.20")
    assert "0000" not in str(latest.open)


# --- invalid OHLC ---------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("high", "3310.00", "high below open and close"),
        ("low", "3313.00", "low above open"),
        ("high", "3311.00", "high below low"),
    ],
)
def test_invalid_ohlc_is_rejected(field: str, value: str, reason: str) -> None:
    """Requirement 14.3: broken candles never enter the pipeline."""
    bar = make_bar(open_="3312.45", high="3315.10", low="3311.90", close="3314.20")
    bar[field] = value
    with pytest.raises(ValidationError):
        MarketDataInput.model_validate(make_market_payload(bars=[bar]))


def test_non_positive_price_is_rejected() -> None:
    bar = make_bar(low="0")
    with pytest.raises(ValidationError):
        MarketDataInput.model_validate(make_market_payload(bars=[bar]))


def test_empty_series_is_rejected(frozen_now: datetime) -> None:
    with pytest.raises(EmptyBarsError):
        normalize(make_market_payload(bars=[]), now=frozen_now)


# --- duplicates and ordering ---------------------------------------------


def test_duplicate_candles_are_rejected(frozen_now: datetime) -> None:
    """Requirement 14.4: duplicates fail loudly, never silently de-duplicated."""
    bars = make_series(4)
    bars.append(dict(bars[2]))
    with pytest.raises(DuplicateTimestampError) as exc:
        normalize(make_market_payload(bars=bars), now=frozen_now)
    assert exc.value.details["duplicates"] == ["2026-08-28T00:30:00+00:00"]


def test_duplicate_detection_survives_mixed_offsets(frozen_now: datetime) -> None:
    """Same instant written two ways is still a duplicate."""
    bars = [
        make_bar(timestamp="2026-08-28T00:00:00Z"),
        make_bar(timestamp="2026-08-28T07:00:00+07:00"),
    ]
    with pytest.raises(DuplicateTimestampError):
        normalize(make_market_payload(bars=bars), now=frozen_now)


def test_unsorted_bars_are_sorted_and_flagged(frozen_now: datetime) -> None:
    """Requirement 14.5: the declared policy is sort + warn."""
    bars = make_series(6)
    shuffled = [bars[3], bars[0], bars[5], bars[1], bars[4], bars[2]]
    result = normalize(make_market_payload(bars=shuffled), now=frozen_now)

    timestamps = [bar.timestamp for bar in result.snapshot.bars]
    assert timestamps == sorted(timestamps)
    assert result.snapshot.bar_count == 6
    assert WarningCode.BARS_REORDERED in warning_codes(result)


# --- latest_bar -----------------------------------------------------------


def test_latest_bar_matches_last_bar(frozen_now: datetime) -> None:
    """Requirement 14.6."""
    result = normalize(make_market_payload(bars=make_series(8)), now=frozen_now)
    snapshot = result.snapshot
    assert snapshot.latest_bar == snapshot.bars[-1]
    assert snapshot.latest_bar.close == snapshot.bars[-1].close
    assert snapshot.data_to == snapshot.latest_bar.timestamp


def test_latest_bar_reflects_sorting_not_input_order(frozen_now: datetime) -> None:
    bars = make_series(5)
    reversed_bars = list(reversed(bars))
    result = normalize(make_market_payload(bars=reversed_bars), now=frozen_now)
    assert result.snapshot.latest_bar.timestamp == BASE_TIME + timedelta(minutes=60)


def test_provider_latest_bar_disagreement_is_fatal(frozen_now: datetime) -> None:
    """A provider asserting a different last candle is a contradiction, not a hint."""
    bars = make_series(5)
    lying = make_bar(minute_offset=60, close="9999.00", high="9999.50")
    with pytest.raises(LatestBarMismatchError):
        normalize(make_market_payload(bars=bars, latest_bar=lying), now=frozen_now)


def test_provider_latest_bar_agreement_is_accepted(frozen_now: datetime) -> None:
    bars = make_series(5)
    result = normalize(make_market_payload(bars=bars, latest_bar=dict(bars[-1])), now=frozen_now)
    assert result.snapshot.latest_bar.timestamp == BASE_TIME + timedelta(minutes=60)


def test_snapshot_schema_itself_rejects_inconsistent_latest_bar(frozen_now: datetime) -> None:
    """Defence in depth: even hand-constructing a bad snapshot fails."""
    from goldpipeline.schemas.market import MarketDataSnapshot

    bars = [OHLCBar.model_validate(bar) for bar in make_series(3)]
    with pytest.raises(ValidationError, match="latest_bar"):
        MarketDataSnapshot(
            symbol="XAUUSD",
            provider="test",
            timeframe="M15",
            source_timezone="UTC",
            requested_at=frozen_now,
            data_from=bars[0].timestamp,
            data_to=bars[-1].timestamp,
            bars=bars,
            latest_bar=bars[0],
        )


# --- symbols --------------------------------------------------------------


def test_expected_symbol_mismatch_is_rejected(frozen_now: datetime) -> None:
    """Requirement 14.10."""
    with pytest.raises(SymbolMismatchError) as exc:
        normalize(make_market_payload(symbol="EURUSD"), expected_symbol="XAUUSD", now=frozen_now)
    assert exc.value.details == {"expected": "XAUUSD", "actual": "EURUSD"}


def test_per_bar_symbol_mismatch_is_rejected(frozen_now: datetime) -> None:
    bars = make_series(3)
    bars[1]["symbol"] = "XAGUSD"
    with pytest.raises(SymbolMismatchError):
        normalize(make_market_payload(bars=bars), expected_symbol="XAUUSD", now=frozen_now)


def test_symbol_separators_are_normalized(frozen_now: datetime) -> None:
    """XAU/USD and xauusd are the same instrument, not a mismatch."""
    result = normalize(
        make_market_payload(symbol="XAU/USD"), expected_symbol=" xauusd ", now=frozen_now
    )
    assert result.snapshot.symbol == "XAUUSD"


# --- timezones ------------------------------------------------------------


def test_offset_timestamps_are_converted_to_utc(frozen_now: datetime) -> None:
    bars = [make_bar(timestamp="2026-08-28T07:00:00+07:00")]
    result = normalize(make_market_payload(bars=bars), now=frozen_now)
    assert result.snapshot.latest_bar.timestamp == datetime(2026, 8, 28, 0, 0, tzinfo=UTC)


def test_naive_timestamps_use_the_declared_source_timezone(frozen_now: datetime) -> None:
    bars = [make_bar(timestamp="2026-08-28T07:00:00")]
    result = normalize(make_market_payload(bars=bars, timezone="+07:00"), now=frozen_now)
    snapshot = result.snapshot
    assert snapshot.latest_bar.timestamp == datetime(2026, 8, 28, 0, 0, tzinfo=UTC)
    assert snapshot.timezone == "UTC"
    assert snapshot.source_timezone == "+07:00"


def test_naive_timestamp_without_timezone_is_rejected(frozen_now: datetime) -> None:
    """Never assumed to be UTC - guessing would shift every candle silently."""
    bars = [make_bar(timestamp="2026-08-28T07:00:00")]
    with pytest.raises(NaiveTimestampError):
        normalize(make_market_payload(bars=bars), now=frozen_now)


def test_aware_timestamps_need_no_declared_timezone(frozen_now: datetime) -> None:
    result = normalize(make_market_payload(), now=frozen_now)
    assert result.snapshot.source_timezone is None
    assert result.snapshot.timezone == "UTC"


def test_unresolvable_timezone_is_rejected(frozen_now: datetime) -> None:
    with pytest.raises(UnknownTimezoneError):
        normalize(make_market_payload(timezone="Mars/Olympus_Mons"), now=frozen_now)


# --- quality warnings -----------------------------------------------------


def test_missing_volume_is_a_warning_not_an_error(frozen_now: datetime) -> None:
    bars = [make_bar(minute_offset=i * 15, volume=None) for i in range(12)]
    result = normalize(make_market_payload(bars=bars), now=frozen_now)
    assert WarningCode.MISSING_VOLUME in warning_codes(result)
    assert "ohlc.volume" in result.missing_fields
    assert result.snapshot.bar_count == 12


def test_gaps_are_flagged(frozen_now: datetime) -> None:
    bars = make_series(12)
    del bars[5:7]
    result = normalize(make_market_payload(bars=bars), now=frozen_now)
    assert WarningCode.BAR_GAPS in warning_codes(result)


def test_low_bar_count_is_flagged(frozen_now: datetime) -> None:
    result = normalize(make_market_payload(bars=make_series(3)), now=frozen_now)
    assert WarningCode.LOW_BAR_COUNT in warning_codes(result)


def test_future_dated_candles_are_flagged() -> None:
    """A candle that has not closed yet must never be quoted silently."""
    result = normalize(make_market_payload(), now=BASE_TIME - timedelta(hours=6))
    assert WarningCode.FUTURE_DATA in warning_codes(result)


def test_stale_candles_are_flagged() -> None:
    result = normalize(make_market_payload(), now=BASE_TIME + timedelta(days=3))
    assert WarningCode.STALE_DATA in warning_codes(result)


def test_declared_range_is_corrected_and_flagged(frozen_now: datetime) -> None:
    payload = make_market_payload(
        bars=make_series(6),
        data_from="2026-08-01T00:00:00Z",
        data_to="2026-09-01T00:00:00Z",
    )
    result = normalize(payload, now=frozen_now)
    assert WarningCode.DECLARED_RANGE_ADJUSTED in warning_codes(result)
    assert result.snapshot.data_from == BASE_TIME
    assert result.snapshot.data_to == BASE_TIME + timedelta(minutes=75)


def test_missing_requested_at_defaults_and_is_flagged(frozen_now: datetime) -> None:
    result = normalize(make_market_payload(requested_at=None), now=frozen_now)
    assert result.snapshot.requested_at == frozen_now
    assert WarningCode.REQUESTED_AT_DEFAULTED in warning_codes(result)
    assert "market.requested_at" in result.missing_fields


def test_unknown_payload_field_is_rejected() -> None:
    """A typo in a provider payload must not be silently dropped."""
    with pytest.raises(ValidationError):
        MarketDataInput.model_validate(make_market_payload(timefrmae="M15"))
