"""Five timeframes of one instrument, observed at one instant.

Round 6.6h. The ICT branch has always taken an :class:`IctMarketSnapshot` whose
timeframes share a single ``observed_at``; every no-lookahead proof in six
rounds rests on that. This is where a live feed is made to satisfy it.

**One clock reading, five fetches.** The instant is taken once, before the first
socket opens, and every series is then filtered against it. Letting each
timeframe call the clock would mean M1 was observed a second later than H4, and
the reference-price resolver - which compares closes at the *same* instant -
would be comparing two different market moments and calling the difference a
data conflict.

**Nothing is resampled.** H4 comes from the provider's own H4, on the provider's
own grid, and so does every other timeframe. Building H4 out of H1 bars would
silently impose a 00/04/08 grid on a feed that publishes 01/05/09, and the
resulting order blocks would be at prices no candle ever drew.

**Forming candles are dropped, never truncated.** That rule lives in
``build_timeframe_snapshot`` and is reused rather than restated.

**One provider.** TradingView, and no MT5 fallback: a plan built from two feeds
would have no single answer to "what was the price".
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from goldpipeline.adapters.tradingview_market import TradingViewMarketDataSource
from goldpipeline.schemas.common import Timeframe, utc_now
from goldpipeline.schemas.ict import IctMarketSnapshot, build_timeframe_snapshot
from goldpipeline.schemas.market import OHLCBar
from goldpipeline.services.normalizer import normalize_market_data

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 1


class MultiTimeframeError(RuntimeError):
    """One timeframe could not be acquired, so the snapshot was not built.

    Deliberately fatal rather than partial: an ICT composite over four
    timeframes is a different analysis from one over five, and quietly
    producing it would publish a plan whose missing evidence nobody can see.
    """

    def __init__(self, message: str, *, timeframe: Timeframe | None = None) -> None:
        super().__init__(message)
        self.timeframe = timeframe


@dataclass(frozen=True)
class TimeframeFetch:
    """What one timeframe's fetch produced. Provenance, not analysis."""

    timeframe: Timeframe
    provider: str
    provider_symbol: str
    requested_bars: int
    received_bars: int
    closed_bars: int
    first_bar_open_time: datetime
    latest_bar_open_time: datetime
    latest_closed_at: datetime


@dataclass(frozen=True)
class MultiTimeframeObservation:
    """Five closed series and the single instant they were observed at."""

    market_observed_at: datetime
    symbol: str
    provider: str
    provider_symbol: str
    snapshot: IctMarketSnapshot
    fetches: tuple[TimeframeFetch, ...]

    def fetch(self, timeframe: Timeframe) -> TimeframeFetch | None:
        return next((entry for entry in self.fetches if entry.timeframe is timeframe), None)

    def provenance(self) -> dict[str, object]:
        """The JSON-safe record of how these candles were obtained."""
        return {
            "market_observed_at": self.market_observed_at.isoformat(),
            "symbol": self.symbol,
            "provider": self.provider,
            "provider_symbol": self.provider_symbol,
            "timeframes": [
                {
                    "timeframe": entry.timeframe.value,
                    "requested_bars": entry.requested_bars,
                    "received_bars": entry.received_bars,
                    "closed_bars": entry.closed_bars,
                    "first_bar_open_time": entry.first_bar_open_time.isoformat(),
                    "latest_bar_open_time": entry.latest_bar_open_time.isoformat(),
                    "latest_closed_at": entry.latest_closed_at.isoformat(),
                }
                for entry in self.fetches
            ],
        }


class MultiTimeframeMarketSource:
    """Fetches every configured timeframe and freezes them at one instant.

    Composes the existing single-timeframe TradingView source rather than
    reimplementing the wire protocol: closed-candle proof, retry classification,
    symbol validation and staleness all stay where they were tested.
    """

    def __init__(
        self,
        *,
        provider_symbol: str,
        timeframes: tuple[Timeframe, ...],
        bars_per_timeframe: int,
        build_source: Callable[[Timeframe, int], object] | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_data_age_minutes: int | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        """Build a source. Opens nothing.

        Args:
            provider_symbol: TradingView's name for the instrument.
            timeframes: Which series to fetch, slowest to fastest.
            bars_per_timeframe: Closed bars wanted from each.
            build_source: Makes one single-timeframe source. Injected by tests
                so the whole five-timeframe flow runs with no network; the
                default builds the real TradingView adapter.
            timeout_seconds: Per-receive timeout for each fetch.
            max_retries: Transport retries per timeframe, inside the adapter.
            max_data_age_minutes: Staleness guard, passed through unchanged.
            now: Clock. Called exactly once, and a test asserts that.
        """
        if not timeframes:
            raise MultiTimeframeError("a snapshot needs at least one timeframe")

        self._provider_symbol = provider_symbol
        self._timeframes = timeframes
        self._bars = bars_per_timeframe
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._max_data_age = max_data_age_minutes
        self._now = now if now is not None else utc_now
        self._build_source = build_source if build_source is not None else self._tradingview

    def staleness_allowance(self, timeframe: Timeframe) -> int | None:
        """The freshness limit for one timeframe, in minutes.

        The configured limit is a *lateness* budget: how far behind the market
        the newest closed candle may be beyond the point at which one could
        possibly exist. On M15 - the timeframe the setting was chosen for - the
        two are nearly the same thing, and the distinction never mattered.

        On H4 it is the whole story. An H4 bar is 240 minutes long, so on a
        perfectly live market its newest *closed* bar is routinely two or three
        hours old; measuring that against a 90-minute limit would call a healthy
        feed stale every time. So each timeframe gets the configured budget plus
        its own bar duration, which asks the question the setting was always
        meant to ask: is this series later than it should be?
        """
        if self._max_data_age is None:
            return None
        duration = timeframe.duration
        if duration is None:  # pragma: no cover - every ICT timeframe has one
            raise MultiTimeframeError(
                f"{timeframe.value} has no fixed duration", timeframe=timeframe
            )
        return self._max_data_age + int(duration.total_seconds() // 60)

    def set_build_source(self, build: Callable[[Timeframe, int], object]) -> None:
        """Replace the per-timeframe builder after construction.

        The production builder needs :meth:`staleness_allowance`, which is a
        property of a configured source, so it cannot be written before one
        exists. Kept explicit rather than making the constructor accept a
        callback that takes the half-built object.
        """
        self._build_source = build

    def _tradingview(self, timeframe: Timeframe, bars: int) -> object:
        return TradingViewMarketDataSource(
            provider_symbol=self._provider_symbol,
            timeframe=timeframe,
            limit=bars,
            timeout_seconds=self._timeout,
            max_retries=self._max_retries,
            max_data_age_minutes=self.staleness_allowance(timeframe),
        )

    def observe(self) -> MultiTimeframeObservation:
        """Fetch every timeframe and freeze them at one shared instant.

        Raises:
            MultiTimeframeError: A timeframe failed, returned a different
                instrument, or has no bar closed by the observation instant.
        """
        # Once. Every series below is filtered against this exact value, so the
        # five of them describe one market moment rather than five adjacent ones.
        market_observed_at = self._now()

        fetches: list[TimeframeFetch] = []
        series: list[tuple[Timeframe, tuple[OHLCBar, ...]]] = []
        symbol: str | None = None
        provider: str | None = None

        for timeframe in self._timeframes:
            source = self._build_source(timeframe, self._bars)
            try:
                loaded = source.load()  # type: ignore[attr-defined]
            except Exception as exc:
                raise MultiTimeframeError(
                    f"{timeframe.value} could not be fetched: {exc}", timeframe=timeframe
                ) from exc

            normalized = normalize_market_data(loaded.model)
            bars = tuple(normalized.snapshot.bars)
            if not bars:
                raise MultiTimeframeError(
                    f"{timeframe.value} returned no candles", timeframe=timeframe
                )

            if symbol is None:
                symbol = normalized.snapshot.symbol
                provider = normalized.snapshot.provider
            elif normalized.snapshot.symbol != symbol:
                raise MultiTimeframeError(
                    f"{timeframe.value} returned {normalized.snapshot.symbol}, "
                    f"but earlier timeframes returned {symbol}",
                    timeframe=timeframe,
                )

            try:
                closed = build_timeframe_snapshot(
                    timeframe=timeframe, bars=bars, observed_at=market_observed_at
                )
            except ValueError as exc:
                raise MultiTimeframeError(
                    f"{timeframe.value} has no bar closed by "
                    f"{market_observed_at.isoformat()}: {exc}",
                    timeframe=timeframe,
                ) from exc

            series.append((timeframe, closed.bars))
            fetches.append(
                TimeframeFetch(
                    timeframe=timeframe,
                    provider=normalized.snapshot.provider,
                    provider_symbol=self._provider_symbol,
                    requested_bars=self._bars,
                    received_bars=len(bars),
                    closed_bars=closed.bar_count,
                    first_bar_open_time=closed.bars[0].timestamp,
                    latest_bar_open_time=closed.bars[-1].timestamp,
                    latest_closed_at=closed.latest_closed_at,
                )
            )
            logger.info(
                "mtf.fetch timeframe=%s received=%d closed=%d latest_closed_at=%s",
                timeframe.value,
                len(bars),
                closed.bar_count,
                closed.latest_closed_at.isoformat(),
            )

        assert symbol is not None and provider is not None

        snapshot = IctMarketSnapshot(
            observed_at=market_observed_at,
            symbol=symbol,
            provider=provider,
            provider_symbol=self._provider_symbol,
            timeframes=tuple(
                build_timeframe_snapshot(
                    timeframe=timeframe, bars=bars, observed_at=market_observed_at
                )
                for timeframe, bars in series
            ),
        )

        return MultiTimeframeObservation(
            market_observed_at=market_observed_at,
            symbol=symbol,
            provider=provider,
            provider_symbol=self._provider_symbol,
            snapshot=snapshot,
            fetches=tuple(fetches),
        )


__all__ = [
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TIMEOUT_SECONDS",
    "MultiTimeframeError",
    "MultiTimeframeMarketSource",
    "MultiTimeframeObservation",
    "TimeframeFetch",
]
