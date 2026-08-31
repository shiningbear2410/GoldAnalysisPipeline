"""Market data schemas.

Two models, on purpose:

* :class:`MarketDataInput` - what a provider or fixture hands us. Lenient about
  derived fields (``data_from``, ``data_to``, ``latest_bar`` may be absent) and
  tolerant of naive timestamps, which the normalizer resolves.
* :class:`MarketDataSnapshot` - the normalized, self-consistent result. Every
  derived field is present and cross-checked. If a snapshot exists, its
  invariants hold; there is no such thing as a snapshot whose ``latest_bar``
  disagrees with ``bars[-1]``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Self

from pydantic import Field, field_validator, model_validator

from goldpipeline.schemas.common import (
    LenientModel,
    Price,
    StrictModel,
    Timeframe,
    UtcDatetime,
    Volume,
    normalize_symbol,
    validate_price_exponent,
)


class OHLCBar(StrictModel):
    """A single candle.

    Invariants enforced at construction::

        high >= open,  high >= close,  high >= low
        low  <= open,  low  <= close

    ``timestamp`` denotes the bar's **open** time. It may be naive at input
    time; :class:`MarketDataSnapshot` requires it to be aware UTC.
    """

    timestamp: UtcDatetime = Field(description="Bar open time.")
    open: Price
    high: Price
    low: Price
    close: Price
    volume: Volume | None = Field(
        default=None, description="Traded volume; null when the provider omits it."
    )
    symbol: str | None = Field(
        default=None, description="Optional per-bar symbol echo, cross-checked on normalization."
    )

    @field_validator("open", "high", "low", "close")
    @classmethod
    def _sane_price(cls, value: Decimal) -> Decimal:
        return validate_price_exponent(value, "price")

    @field_validator("symbol")
    @classmethod
    def _canonical_symbol(cls, value: str | None) -> str | None:
        return None if value is None else normalize_symbol(value)

    @model_validator(mode="after")
    def _check_ohlc_ordering(self) -> Self:
        problems: list[str] = []
        if self.high < self.low:
            problems.append(f"high ({self.high}) < low ({self.low})")
        if self.high < self.open:
            problems.append(f"high ({self.high}) < open ({self.open})")
        if self.high < self.close:
            problems.append(f"high ({self.high}) < close ({self.close})")
        if self.low > self.open:
            problems.append(f"low ({self.low}) > open ({self.open})")
        if self.low > self.close:
            problems.append(f"low ({self.low}) > close ({self.close})")
        if problems:
            raise ValueError(f"invalid OHLC bar at {self.timestamp}: " + "; ".join(problems))
        return self

    @property
    def is_utc(self) -> bool:
        """Whether the timestamp is timezone-aware and expressed in UTC."""
        return self.timestamp.tzinfo is not None and self.timestamp.utcoffset() == UTC.utcoffset(
            None
        )


class MarketDataInput(LenientModel):
    """Raw market data payload as delivered by a provider or fixture."""

    symbol: str = Field(description="Instrument symbol, e.g. 'XAUUSD'.")
    provider: str = Field(description="Where the data came from, e.g. 'mt5', 'twelvedata'.")
    timeframe: Timeframe
    timezone: str | None = Field(
        default=None,
        description=(
            "Timezone the provider's naive timestamps are expressed in. Leave null when "
            "every timestamp carries an explicit offset. It is never defaulted to UTC: "
            "silently assuming UTC for a provider that reports local time would shift "
            "every candle by hours without anyone noticing."
        ),
    )
    provider_symbol: str | None = Field(
        default=None,
        description=(
            "The provider's own name for this instrument, when it differs from "
            "`symbol`. Brokers rename gold - XAUUSDm, GOLD, XAUUSD.a - and the "
            "two names are recorded separately rather than one being inferred "
            "from the other. Null when the provider uses the canonical name."
        ),
    )
    requested_at: datetime | None = Field(
        default=None, description="When the data was fetched; defaults to ingestion time."
    )
    retrieved_at: datetime | None = Field(
        default=None,
        description=(
            "When the provider's answer came back. Recorded alongside "
            "`requested_at` so an audit can tell how long a fetch took, and how "
            "old the data already was when it arrived."
        ),
    )
    data_from: datetime | None = Field(
        default=None, description="Declared start of coverage; derived from bars when absent."
    )
    data_to: datetime | None = Field(
        default=None, description="Declared end of coverage; derived from bars when absent."
    )
    bars: list[OHLCBar] = Field(default_factory=list, description="Candles, any order.")
    latest_bar: OHLCBar | None = Field(
        default=None,
        description="Optional provider assertion about the last bar; cross-checked, never trusted.",
    )

    @field_validator("symbol")
    @classmethod
    def _canonical_symbol(cls, value: str) -> str:
        return normalize_symbol(value)

    @field_validator("provider")
    @classmethod
    def _non_empty_provider(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("provider must not be empty")
        return cleaned


class MarketDataSnapshot(StrictModel):
    """Normalized market data. Every invariant below is guaranteed to hold."""

    symbol: str
    provider: str
    timeframe: Timeframe
    timezone: str = Field(
        default="UTC", description="Timezone of the stored timestamps. Always 'UTC'."
    )
    source_timezone: str | None = Field(
        description=(
            "Timezone the provider declared for naive timestamps, kept for audit. "
            "Null means every incoming timestamp already carried an explicit offset."
        )
    )
    requested_at: UtcDatetime
    data_from: UtcDatetime = Field(description="Timestamp of the first bar.")
    data_to: UtcDatetime = Field(description="Timestamp of the last bar.")
    bars: list[OHLCBar] = Field(min_length=1, description="Candles, ascending, unique, UTC.")
    latest_bar: OHLCBar = Field(description="Always identical to bars[-1].")

    @field_validator("timezone")
    @classmethod
    def _must_be_utc(cls, value: str) -> str:
        if value != "UTC":
            raise ValueError("normalized snapshots always store timestamps in UTC")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        timestamps = [bar.timestamp for bar in self.bars]

        for bar in self.bars:
            if not bar.is_utc:
                raise ValueError(f"bar {bar.timestamp} is not timezone-aware UTC")
            if bar.symbol is not None and bar.symbol != self.symbol:
                raise ValueError(
                    f"bar at {bar.timestamp} declares symbol {bar.symbol!r}, "
                    f"snapshot declares {self.symbol!r}"
                )

        if timestamps != sorted(timestamps):
            raise ValueError("bars must be sorted ascending by timestamp")
        if len(set(timestamps)) != len(timestamps):
            raise ValueError("bars must not contain duplicate timestamps")

        if self.latest_bar != self.bars[-1]:
            raise ValueError("latest_bar must be identical to bars[-1]")
        if self.data_from != timestamps[0]:
            raise ValueError("data_from must equal the first bar timestamp")
        if self.data_to != timestamps[-1]:
            raise ValueError("data_to must equal the last bar timestamp")
        return self

    @property
    def bar_count(self) -> int:
        """Number of candles in the snapshot."""
        return len(self.bars)


__all__ = ["MarketDataInput", "MarketDataSnapshot", "OHLCBar"]
