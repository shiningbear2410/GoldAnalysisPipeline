"""MetaTrader 5 as a live OHLC provider.

**Read-only, and structurally so.** :class:`Mt5Module` names the seven functions
this adapter may call, and every one of them reads. ``order_send``,
``order_check`` and the position APIs are not in the protocol, are not imported,
and are not reachable from here. That is not a promise in a docstring - it is the
type the adapter is written against, and a test asserts the module surface.

**The forming candle is the whole problem.** ``copy_rates_from_pos`` counts from
zero, and index 0 is the bar that is still moving: its high, low and close all
change until the period ends. An article quoting it is false before anyone reads
it. So this adapter starts at index 1 *and* then proves the bar it got has
actually closed, because a start index is an assumption and arithmetic on the
timestamp is a fact.

**Symbols are not guessed.** Brokers call gold ``XAUUSD``, ``XAUUSDm``, ``GOLD``
and ``XAUUSD.a``, and those are not always the same instrument. A missing symbol
fails and lists what the terminal does offer; it never substitutes something
that looks close.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Protocol, cast

from goldpipeline.adapters.base import LoadedSource
from goldpipeline.config import MarketDataSettings
from goldpipeline.domain.errors import (
    FormingCandleError,
    InsufficientBarsError,
    Mt5InitializeError,
    Mt5NotInstalledError,
    Mt5ProviderError,
    Mt5SymbolNotFoundError,
    Mt5SymbolNotSelectedError,
    StaleMarketDataError,
)
from goldpipeline.schemas.common import Timeframe, utc_now
from goldpipeline.schemas.market import MarketDataInput

logger = logging.getLogger(__name__)

PROVIDER_NAME = "metatrader5"

CLOSED_CANDLE_TOLERANCE = timedelta(seconds=5)
"""How far the terminal's clock may run ahead of ours before we care.

Small on purpose. This exists for clock skew of a few seconds between this
process and the broker's server, not to wave through a bar that is genuinely
still open.
"""

MAX_PRICE_DECIMALS = 8
"""Ceiling the price schema enforces; broker digits are clamped to it."""

_CANDIDATE_LIMIT = 12
"""How many symbol names a "not found" error may list, so a diagnostic on a
broker with 3000 instruments stays readable."""


class Mt5Module(Protocol):
    """The market-data subset of the MetaTrader5 module.

    Written out in full rather than typed as ``Any`` so that the read-only
    boundary is checkable. If someone later needs a trading call, they have to
    add it here first, in a diff a reviewer will see.
    """

    def initialize(self, *args: Any, **kwargs: Any) -> bool: ...
    def shutdown(self) -> None: ...
    def last_error(self) -> tuple[int, str]: ...
    def symbol_info(self, symbol: str) -> Any: ...
    def symbol_select(self, symbol: str, enable: bool) -> bool: ...
    def symbols_get(self, *args: Any, **kwargs: Any) -> Any: ...
    def copy_rates_from_pos(
        self, symbol: str, timeframe: int, start_pos: int, count: int
    ) -> Any: ...


def load_mt5_module() -> Mt5Module:
    """Import the real MetaTrader5 package.

    Imported here rather than at module scope so that the schemas, the services
    and the whole test suite work on a machine that has never seen a terminal.

    Raises:
        Mt5NotInstalledError: The package is not importable.
    """
    try:
        import MetaTrader5
    except ImportError as exc:  # pragma: no cover - exercised by injecting a fake
        raise Mt5NotInstalledError(
            "the MetaTrader5 package is not installed in this interpreter. "
            'Install it with `pip install -e ".[mt5]"` on Windows; it is an '
            "optional extra because it ships Windows-only wheels and needs a "
            "terminal to talk to.",
            setting="MetaTrader5",
        ) from exc
    return cast(Mt5Module, MetaTrader5)


class MetaTrader5MarketDataSource:
    """Fetches the latest *closed* candles for one instrument.

    One instance fetches once. :meth:`load` caches its result, so a caller can
    pre-flight the terminal and then hand the same source to ``create_run``
    without a second round trip - and without the two disagreeing about what the
    market was doing.
    """

    def __init__(
        self,
        settings: MarketDataSettings,
        *,
        module: Mt5Module | None = None,
        now: datetime | None = None,
        select_if_hidden: bool = True,
    ) -> None:
        """Build a source.

        Args:
            settings: Symbol, timeframe and bar count. Holds no credential.
            module: The MetaTrader5 module, or a stand-in. Injected by tests so
                the whole adapter runs with no terminal anywhere.
            now: Injection point for tests; used for the closed-candle proof.
            select_if_hidden: Whether a symbol missing from Market Watch may be
                added to it. This changes terminal *visibility* only - it places
                no order and moves no money - but it is a write to the user's
                terminal, so it can be turned off.
        """
        self._settings = settings
        self._module = module
        self._now = now
        self._select_if_hidden = select_if_hidden
        self._cached: LoadedSource[MarketDataInput] | None = None

    @property
    def settings(self) -> MarketDataSettings:
        return self._settings

    def load(self) -> LoadedSource[MarketDataInput]:
        """Fetch the latest closed candles, once.

        Raises:
            Mt5NotInstalledError: The package is missing.
            Mt5InitializeError: The terminal could not be reached.
            Mt5SymbolNotFoundError: The configured symbol is not on this broker.
            Mt5SymbolNotSelectedError: It exists but is not visible.
            InsufficientBarsError: Fewer candles came back than were asked for.
            FormingCandleError: The newest bar has not closed.
            StaleMarketDataError: The newest closed bar is too old to write about.
        """
        if self._cached is None:
            self._cached = self._fetch()
        return self._cached

    # -- the fetch ---------------------------------------------------------

    def _fetch(self) -> LoadedSource[MarketDataInput]:
        module = self._module if self._module is not None else load_mt5_module()
        settings = self._settings
        requested_at = self._now or utc_now()

        if not module.initialize():
            raise Mt5InitializeError(
                "could not connect to the MetaTrader 5 terminal. Start the terminal, "
                "log in to an account, and make sure it is not blocked by another client.",
                provider_error=_last_error(module),
            )

        try:
            digits = self._require_symbol(module, settings.provider_symbol)
            rates = self._copy_rates(module, settings)
            retrieved_at = self._now or utc_now()
        finally:
            # Always, on every path. A terminal connection left open holds an
            # IPC slot the next process will be told is busy.
            _shutdown_quietly(module)

        converted = [self._to_bar(row, digits) for row in rates]
        converted.sort(key=lambda pair: pair[0]["timestamp"])
        bars = [bar for bar, _ in converted]
        volume_source = converted[-1][1]
        self._require_closed(bars[-1], settings)

        payload: dict[str, Any] = {
            "symbol": settings.canonical_symbol,
            "provider": PROVIDER_NAME,
            "provider_symbol": settings.provider_symbol,
            "timeframe": settings.timeframe,
            # Every timestamp below carries an explicit offset, so there is no
            # provider timezone for the normalizer to resolve.
            "timezone": None,
            "requested_at": _iso(requested_at),
            "retrieved_at": _iso(retrieved_at),
            "bars": bars,
        }
        model = MarketDataInput.model_validate(payload)

        logger.info(
            "mt5.fetch symbol=%s timeframe=%s bars=%d latest=%s",
            settings.provider_symbol,
            settings.timeframe,
            len(bars),
            bars[-1]["timestamp"],
        )

        return LoadedSource(
            model=model,
            raw_payload=payload,
            origin=f"{PROVIDER_NAME}:{settings.provider_symbol}:{settings.timeframe}",
            provenance={
                "kind": "market",
                "provider": PROVIDER_NAME,
                "provider_symbol": settings.provider_symbol,
                "canonical_symbol": settings.canonical_symbol,
                "timeframe": settings.timeframe,
                "bars_requested": settings.bar_count,
                "bars_returned": len(bars),
                "start_pos": 1,
                "requested_at": _iso(requested_at),
                "retrieved_at": _iso(retrieved_at),
                "latest_candle_at": bars[-1]["timestamp"],
                "volume_field": volume_source,
            },
        )

    # -- steps -------------------------------------------------------------

    def _require_symbol(self, module: Mt5Module, symbol: str) -> int | None:
        """Check the symbol exists and is visible. Returns its price precision."""
        info = module.symbol_info(symbol)
        if info is None:
            raise Mt5SymbolNotFoundError(
                f"the configured symbol {symbol!r} does not exist on this broker. "
                "Set GOLDPIPELINE_MT5_SYMBOL to the broker's own name for gold - "
                "it is never guessed, because similarly named symbols are not "
                "always the same instrument.",
                setting="GOLDPIPELINE_MT5_SYMBOL",
                symbol=symbol,
                candidates=_candidates(module, symbol),
            )

        if not getattr(info, "visible", True):
            if not self._select_if_hidden:
                raise Mt5SymbolNotSelectedError(
                    f"symbol {symbol!r} is not in Market Watch and this source was "
                    "asked not to add it. Add it in the terminal, or allow selection.",
                    symbol=symbol,
                )
            # Market Watch visibility only. This places no order and changes no
            # account state; the terminal simply will not serve history for a
            # symbol it is not tracking.
            if not module.symbol_select(symbol, True):
                raise Mt5SymbolNotSelectedError(
                    f"symbol {symbol!r} exists but could not be added to Market Watch",
                    symbol=symbol,
                    provider_error=_last_error(module),
                )

        digits = getattr(info, "digits", None)
        return int(digits) if isinstance(digits, int) else None

    def _copy_rates(self, module: Mt5Module, settings: MarketDataSettings) -> list[Any]:
        """Ask for *bar_count* candles starting at the first closed one."""
        timeframe = _timeframe_constant(module, settings.timeframe)
        try:
            # start_pos=1 skips index 0, the candle still forming.
            rates = module.copy_rates_from_pos(
                settings.provider_symbol, timeframe, 1, settings.bar_count
            )
        except Exception:  # noqa: BLE001 - vendor errors are undocumented
            # `from None`: the vendor exception's text is not reproduced, only
            # the provider's own numeric error is passed along.
            raise Mt5ProviderError(
                "the terminal failed while returning candles",
                provider_error=_last_error(module),
            ) from None

        rows = [] if rates is None else list(rates)
        if len(rows) < settings.bar_count:
            raise InsufficientBarsError(
                f"asked for {settings.bar_count} closed candles and received {len(rows)}. "
                "The terminal may not have downloaded enough history for this "
                "symbol and timeframe yet - open its chart once and try again.",
                symbol=settings.provider_symbol,
                timeframe=settings.timeframe,
                requested=settings.bar_count,
                received=len(rows),
                provider_error=_last_error(module) if not rows else None,
            )
        return rows

    def _to_bar(self, row: Any, digits: int | None) -> tuple[dict[str, Any], str | None]:
        """Map one provider row onto Round 1's bar payload.

        Returns the bar and which volume column it came from - the latter is
        provenance, not part of the bar, so it is returned rather than smuggled
        into a payload the schema would reject.
        """
        opened = datetime.fromtimestamp(int(_field(row, "time")), tz=UTC)

        real_volume = _optional_int(_field(row, "real_volume", required=False))
        tick_volume = _optional_int(_field(row, "tick_volume", required=False))
        # Real volume is the honest number where a broker reports it; most
        # retail feeds report zero and tick count is the only signal there is.
        volume, volume_source = (
            (real_volume, "real_volume")
            if real_volume
            else ((tick_volume, "tick_volume") if tick_volume is not None else (None, None))
        )

        bar: dict[str, Any] = {
            "timestamp": _iso(opened),
            "open": _price(_field(row, "open"), digits),
            "high": _price(_field(row, "high"), digits),
            "low": _price(_field(row, "low"), digits),
            "close": _price(_field(row, "close"), digits),
        }
        if volume is not None:
            bar["volume"] = str(volume)
        return bar, volume_source

    def _require_closed(self, latest: dict[str, Any], settings: MarketDataSettings) -> None:
        """Prove the newest bar has closed, and is recent enough to write about.

        ``start_pos=1`` is an assumption about the provider's indexing. This is
        arithmetic on the timestamp the provider actually returned, which is why
        both exist.
        """
        moment = self._now or utc_now()
        opened = datetime.fromisoformat(latest["timestamp"])
        duration = Timeframe(settings.timeframe).duration
        assert duration is not None, "supported timeframes all have a fixed duration"

        closes_at = opened + duration
        if closes_at > moment + CLOSED_CANDLE_TOLERANCE:
            raise FormingCandleError(
                "the newest candle has not closed yet, so its high, low and close "
                "are still moving. Nothing is written from a bar that can change.",
                symbol=settings.provider_symbol,
                timeframe=settings.timeframe,
                candle_opened_at=latest["timestamp"],
                candle_closes_at=_iso(closes_at),
                now=_iso(moment),
            )

        age_minutes = (moment - closes_at).total_seconds() / 60
        if age_minutes > settings.max_data_age_minutes:
            raise StaleMarketDataError(
                f"the newest closed candle is {int(age_minutes)} minutes old, past the "
                f"{settings.max_data_age_minutes} minute limit. Often this just means the "
                "market is closed - which is still a reason not to publish an analysis "
                "of it. Raise GOLDPIPELINE_MAX_DATA_AGE_MINUTES deliberately if that is "
                "what you intend.",
                setting="GOLDPIPELINE_MAX_DATA_AGE_MINUTES",
                symbol=settings.provider_symbol,
                candle_closed_at=_iso(closes_at),
                age_minutes=int(age_minutes),
                limit_minutes=settings.max_data_age_minutes,
            )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _timeframe_constant(module: Mt5Module, name: str) -> int:
    """Resolve ``M15`` to the provider's own constant.

    Read off the module by name rather than hardcoded. The provider's timeframe
    values are opaque integers (``H4`` is 16388), and a table of them copied into
    this repository would be a silent liability the day one changed.
    """
    constant = getattr(module, f"TIMEFRAME_{name}", None)
    if not isinstance(constant, int):
        raise Mt5ProviderError(
            f"the MetaTrader5 module does not define a TIMEFRAME_{name} constant",
            timeframe=name,
        )
    return constant


def _candidates(module: Mt5Module, symbol: str) -> list[str]:
    """Names on this broker that look related, purely as a hint for a human.

    Listed, never chosen. The whole point of failing here is that the pipeline
    must not decide that ``XAUUSD.a`` is what you meant.
    """
    stem = symbol.strip().upper()[:3] or "XAU"
    try:
        found = module.symbols_get()
    except Exception:  # noqa: BLE001 - a diagnostic must not raise over a diagnostic
        return []
    if not found:
        return []
    names = [str(getattr(item, "name", "")) for item in found]
    return sorted(name for name in names if name and stem in name.upper())[:_CANDIDATE_LIMIT]


def _field(row: Any, name: str, *, required: bool = True) -> Any:
    """Read one column from a provider row.

    Rows are numpy records from the real package and plain mappings from the
    offline fake; both index by name, and neither is worth a dependency to
    abstract over.
    """
    try:
        return row[name]
    except (KeyError, IndexError, TypeError, ValueError):
        value = getattr(row, name, None)
        if value is None and required:
            raise Mt5ProviderError(f"provider row is missing the {name!r} field") from None
        return value


def _price(value: Any, digits: int | None) -> str:
    """Convert a provider price to an exact decimal string.

    Via ``str`` rather than binary float arithmetic, so nothing picks up
    representation noise on the way in. Where the broker states the instrument's
    precision, the value is quantized to it - that is the broker's own answer to
    how many digits are real, and it beats inventing one here.
    """
    number = Decimal(str(float(value)))
    exponent = number.as_tuple().exponent
    natural = -exponent if isinstance(exponent, int) else MAX_PRICE_DECIMALS
    places = max(0, min(digits if digits is not None else natural, MAX_PRICE_DECIMALS))
    return str(number.quantize(Decimal(1).scaleb(-places)))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _last_error(module: Mt5Module) -> str:
    """The provider's own error, and nothing else.

    Deliberately not ``terminal_info()``: that carries the data path, the
    company, and in some builds the logged-in account. None of it helps diagnose
    a fetch, and all of it would end up in a manifest.
    """
    try:
        code, description = module.last_error()
    except Exception:  # noqa: BLE001 - never fail while describing a failure
        return "unavailable"
    return f"{int(code)}: {description}"


def _shutdown_quietly(module: Mt5Module) -> None:
    """Release the terminal connection, and never mask the real failure."""
    try:
        module.shutdown()
    except Exception:  # noqa: BLE001
        logger.warning("mt5.shutdown failed; the terminal connection may be left open")


__all__ = [
    "CLOSED_CANDLE_TOLERANCE",
    "PROVIDER_NAME",
    "MetaTrader5MarketDataSource",
    "Mt5Module",
    "load_mt5_module",
]
