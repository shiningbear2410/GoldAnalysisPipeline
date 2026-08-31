"""Offline stand-in for the MetaTrader5 module.

Every test in this repository runs against this, and so does ``--fake-mt5``.
Nothing here opens a socket or looks for a terminal, which is what lets the
suite run on a build machine that has never had MetaTrader installed.

It models the two details that make the real module awkward, because those are
exactly the ones worth testing:

* **index 0 is the candle still forming.** :func:`make_rates` builds a series
  whose newest entry has not closed, so a test can prove the adapter skipped it
  rather than trusting that it did.
* **failure is a return value, not an exception.** ``initialize`` answers
  ``False`` and ``symbol_info`` answers ``None``; the reason lives behind a
  separate ``last_error()`` call. Code written against a fake that raised would
  be wrong against the real thing.

The fake also records every attribute anyone touched, so a test can assert that
no trading function was so much as looked up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

FAKE_DIGITS = 2
"""Gold is quoted to two decimals on most brokers."""

TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}
"""Minutes per timeframe, used to lay out a plausible series."""

TERMINAL_SECRET_SENTINEL = "account-1234567-DO-NOT-LEAK"
"""Planted where a careless adapter might pick it up.

The real ``terminal_info()`` carries the data path, the broker and on some
builds the logged-in account. Tests grep error text for this string, so an
adapter that starts reporting terminal internals fails loudly.
"""


@dataclass(frozen=True)
class FakeSymbolInfo:
    """What ``symbol_info`` returns for a symbol that exists."""

    name: str
    digits: int = FAKE_DIGITS
    visible: bool = True
    path: str = f"Forex\\{TERMINAL_SECRET_SENTINEL}"


def make_rates(
    *,
    count: int = 21,
    timeframe: str = "M15",
    now: datetime | None = None,
    close: float = 3305.90,
    step: float = 0.40,
) -> list[dict[str, Any]]:
    """Build a descending-index candle series, newest first.

    Laid out the way the provider does it: index 0 is the bar that opened most
    recently and has *not* closed, index 1 is the newest closed bar, and so on
    backwards. A caller asking for ``start_pos=1`` therefore gets closed bars
    only - and a caller who forgets gets a forming one, which is the mistake
    these tests exist to catch.
    """
    minutes = TIMEFRAME_MINUTES[timeframe]
    moment = now or datetime.now(UTC)
    # Align to the current period so index 0 is genuinely mid-flight.
    epoch = int(moment.timestamp())
    period = minutes * 60
    current_open = epoch - (epoch % period)

    rates: list[dict[str, Any]] = []
    for index in range(count):
        opened = current_open - index * period
        base = close - index * step
        rates.append(
            {
                "time": opened,
                "open": round(base - step / 2, FAKE_DIGITS),
                "high": round(base + step, FAKE_DIGITS),
                "low": round(base - step, FAKE_DIGITS),
                "close": round(base, FAKE_DIGITS),
                "tick_volume": 1400 + index,
                "spread": 12,
                "real_volume": 0,
            }
        )
    return rates


@dataclass
class FakeMt5Module:
    """Deterministic, offline implementation of the market-data surface.

    Configure a failure with ``initialize_ok``, ``known_symbols``,
    ``select_ok`` or ``rates``; anything left alone behaves.
    """

    initialize_ok: bool = True
    select_ok: bool = True
    known_symbols: tuple[str, ...] = ("XAUUSD",)
    hidden_symbols: tuple[str, ...] = ()
    all_symbols: tuple[str, ...] = ("XAUUSD", "XAUUSDm", "XAUEUR", "EURUSD")
    rates: list[dict[str, Any]] | None = None
    rates_error: Exception | None = None
    error: tuple[int, str] = (0, "Success")
    digits: int = FAKE_DIGITS

    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)
    """Every call made, so tests can assert on order and on absence."""

    initialized: int = 0
    shutdowns: int = 0

    # MetaTrader5 exposes these as module constants. The values are the real
    # ones, so a mistake in the adapter's lookup shows up here rather than only
    # against a live terminal.
    TIMEFRAME_M1: int = 1
    TIMEFRAME_M5: int = 5
    TIMEFRAME_M15: int = 15
    TIMEFRAME_M30: int = 30
    TIMEFRAME_H1: int = 16385
    TIMEFRAME_H4: int = 16388

    def initialize(self, *args: Any, **kwargs: Any) -> bool:
        self.calls.append(("initialize", args))
        if self.initialize_ok:
            self.initialized += 1
        return self.initialize_ok

    def shutdown(self) -> None:
        self.calls.append(("shutdown", ()))
        self.shutdowns += 1

    def last_error(self) -> tuple[int, str]:
        self.calls.append(("last_error", ()))
        return self.error

    def symbol_info(self, symbol: str) -> FakeSymbolInfo | None:
        self.calls.append(("symbol_info", (symbol,)))
        if symbol not in self.known_symbols:
            return None
        return FakeSymbolInfo(
            name=symbol, digits=self.digits, visible=symbol not in self.hidden_symbols
        )

    def symbol_select(self, symbol: str, enable: bool) -> bool:
        self.calls.append(("symbol_select", (symbol, enable)))
        return self.select_ok

    def symbols_get(self, *args: Any, **kwargs: Any) -> list[FakeSymbolInfo]:
        self.calls.append(("symbols_get", args))
        return [FakeSymbolInfo(name=name) for name in self.all_symbols]

    def copy_rates_from_pos(
        self, symbol: str, timeframe: int, start_pos: int, count: int
    ) -> list[dict[str, Any]] | None:
        self.calls.append(("copy_rates_from_pos", (symbol, timeframe, start_pos, count)))
        if self.rates_error is not None:
            raise self.rates_error
        series = self.rates if self.rates is not None else make_rates()
        return series[start_pos : start_pos + count]

    # -- assertions tests lean on ------------------------------------------

    @property
    def called(self) -> list[str]:
        """Names of every function called, in order."""
        return [name for name, _ in self.calls]

    def __getattr__(self, name: str) -> Any:
        """Refuse anything outside the market-data surface.

        The real module would happily hand back ``order_send``. This one raises,
        so a test that asserts the adapter never reaches for a trading call
        cannot pass by accident on a typo.

        Dunders are let through to the normal lookup failure: copy, pickle and
        pytest all probe for optional protocol methods, and answering those
        probes with a lecture breaks tooling for no safety gain.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(f"{name!r} is not part of the market-data surface this pipeline uses")


def unavailable_module(code: int = -10005, message: str = "IPC timeout") -> FakeMt5Module:
    """A terminal that cannot be reached."""
    return FakeMt5Module(initialize_ok=False, error=(code, message))


def missing_symbol_module(known: str = "EURUSD") -> FakeMt5Module:
    """A broker that does not offer the configured symbol."""
    return FakeMt5Module(known_symbols=(known,))


TRADING_FUNCTIONS = (
    "order_send",
    "order_check",
    "order_calc_margin",
    "order_calc_profit",
    "positions_get",
    "position_close",
    "history_orders_get",
)
"""Names a read-only adapter must never call. Asserted against, not documented at."""


__all__ = [
    "FAKE_DIGITS",
    "TERMINAL_SECRET_SENTINEL",
    "TIMEFRAME_MINUTES",
    "TRADING_FUNCTIONS",
    "FakeMt5Module",
    "FakeSymbolInfo",
    "make_rates",
    "missing_symbol_module",
    "unavailable_module",
]
