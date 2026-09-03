"""TradingView as an additional live OHLC provider.

Implements the same :class:`~goldpipeline.adapters.base.MarketDataSource`
protocol the MetaTrader 5 source implements, and returns the same
:class:`~goldpipeline.schemas.market.MarketDataInput`. Nothing downstream can
tell which provider filled it in beyond reading ``provider``, which is the
point: this round adds a feed, it does not change what the pipeline believes.

**Production authority is unchanged.** No caller is migrated here. The CLI's
``--market-source`` still defaults to ``mt5``, the scheduled worker still uses
the MT5 path, and whether TradingView should ever become authoritative is a
question for the next round, answered by comparing the two - not by this file
existing.

**Read-only and unauthenticated.** One host, ``wss://data.tradingview.com``,
opened with the public no-account token. There is no login, no cookie, no
stored session, and nothing here attempts to reach data a signed-out browser
could not. The socket is opened, a series is requested, and it is closed.

**Two independent guards against the forming candle.** The pipeline analyses
closed bars; a bar still forming has a high, low and close that keep moving, so
an article quoting one is false before anybody reads it. TradingView will
happily send the current bar, so:

1. more bars are requested than are wanted, and
2. every returned bar is proved closed by arithmetic on its own open time -
   ``open + duration <= now`` - and the still-forming ones are dropped.

``series_completed`` is *not* used as evidence of closure. It means the
response finished, which is a statement about the socket, not the market.

**Fail closed on bad data.** One unusable bar fails the whole fetch rather than
being dropped with a warning. That is the stricter of the two options the round
allowed, and it matches what the normalizer already does with a bad bar. The
reasoning: a feed that sends an impossible candle is not behaving as
documented, and the alternative - quietly returning nineteen bars where twenty
were required - hides that from every stage downstream. A refused fetch is
visible; a short series is not.

**No threads.** The reference prototype runs ``WebSocketApp`` on a daemon
thread and collects bars in a callback. This adapter blocks on a synchronous
connection instead, because the rest of this pipeline is synchronous and
because a fetch that owns no thread cannot leak one. The socket is closed in a
``finally`` on every path - success, timeout, protocol failure - exactly as the
MT5 source shuts down its terminal connection.
"""

from __future__ import annotations

import json
import logging
import math
import secrets
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol, cast

from goldpipeline.adapters.base import LoadedSource
from goldpipeline.adapters.tradingview_protocol import (
    AUTH_TOKEN,
    Message,
    MessageKind,
    classify,
    decode_frames,
    encode_frame,
    encode_message,
    error_summary,
    extract_raw_bars,
)
from goldpipeline.domain.errors import (
    InsufficientBarsError,
    MarketDataConfigurationError,
    TradingViewCandleError,
    TradingViewConnectionError,
    TradingViewCriticalError,
    TradingViewFramingError,
    TradingViewNotInstalledError,
    TradingViewProtocolError,
    TradingViewTimeoutError,
)
from goldpipeline.schemas.common import Timeframe, utc_now
from goldpipeline.schemas.market import MarketDataInput

logger = logging.getLogger(__name__)

PROVIDER_NAME = "tradingview"

WEBSOCKET_URL = "wss://data.tradingview.com/socket.io/websocket"
"""The only host this provider may reach.

A constant, not configuration. A redirectable or overridable endpoint on a
component that feeds trading analysis is an attack surface with no upside, so
the address is pinned in code where a reviewer sees any change to it.
"""

ORIGIN = "https://data.tradingview.com"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
"""A browser-shaped agent string, as the reference flow sends.

The endpoint refuses a bare client. This is not an attempt to appear to be
something it is not beyond what the public data socket requires to answer.
"""

SUPPORTED_TIMEFRAMES: dict[Timeframe, str] = {
    Timeframe.M1: "1",
    Timeframe.M5: "5",
    Timeframe.M15: "15",
    Timeframe.H1: "60",
    Timeframe.H4: "240",
}
"""Timeframe to wire interval. Explicit, exhaustive, and the only mapping.

No prefix arithmetic: ``H1`` is ``"60"`` and ``M15`` is ``"15"``, and deriving
one from the other by stripping a letter and multiplying is the kind of clever
that turns into an hour of wrong candles. A timeframe absent from this table is
a configuration failure raised before the socket opens - notably ``M30``, which
the project's own config accepts and this provider does not.
"""

SUPPORTED_SYMBOLS: dict[str, str] = {"OANDA:XAUUSD": "XAUUSD"}
"""Provider symbol to the canonical symbol the pipeline reasons about.

One entry, deliberately. Gold on another venue is a different instrument with a
different spread, and this provider will refuse an unknown name rather than
serve the nearest thing that answers - the same rule the MT5 source applies to
broker symbol variants.
"""

MIN_CANDLE_LIMIT = 1
MAX_CANDLE_LIMIT = 5000
"""Bounds on how many closed candles may be requested.

The ceiling matches the reference implementation's own limit and exists so a
mistyped request is refused locally instead of asking a public endpoint for
something unreasonable. The pipeline's production floor is stricter still:
``MarketDataSettings`` will not accept fewer than ten.
"""

SAFETY_MARGIN_BARS = 2
"""Extra bars requested so filtering the forming candle cannot shorten the answer.

``limit`` means *closed* candles. TradingView counts the bar in progress among
the bars it sends, so asking for exactly ``limit`` and then dropping the
forming one returns one too few. Two rather than one because a request that
lands exactly on a period boundary can see the newest bar roll over between
the request and the reply.
"""

DEFAULT_TIMEOUT_SECONDS = 20.0
MAX_MESSAGES = 4000
"""How many packets one fetch will read before giving up.

A second bound beside the clock: a socket that stays busy sending things that
are not the requested series still ends the fetch, rather than reading forever
under a deadline that keeps being reset.
"""

MAX_PRICE_DECIMALS = 8
"""What the price schema accepts. Feed precision is clamped to it."""

MIN_TIMESTAMP = 946_684_800
"""2000-01-01T00:00:00Z. Below this a bar timestamp is not a date, it is a bug."""

MAX_TIMESTAMP_SKEW = timedelta(days=2)
"""How far ahead of our clock a bar may be stamped before it is rejected."""

_MIN_BAR_FIELDS = 5
"""``[timestamp, open, high, low, close]``. Volume, at index 5, is optional."""


# --------------------------------------------------------------------------
# the socket seam
# --------------------------------------------------------------------------


class WebSocketConnection(Protocol):
    """The three operations this adapter needs from a websocket.

    Narrow on purpose, and the reason the whole session flow is testable
    offline: a fake implementing these three methods replaces the network
    entirely, with no monkeypatching and no library installed.
    """

    def send(self, payload: str) -> None: ...
    def recv(self) -> str | bytes: ...
    def close(self) -> None: ...


Connector = Callable[[float], WebSocketConnection]
"""Opens a connection, given a per-operation timeout in seconds."""


def connect_tradingview(timeout: float) -> WebSocketConnection:
    """Open the real websocket.

    The library is imported here rather than at module scope so that the
    schemas, the services and the entire test suite run without it - the same
    arrangement :func:`goldpipeline.adapters.mt5_market.load_mt5_module` uses
    for the vendor terminal package.

    Raises:
        TradingViewNotInstalledError: The websocket client is not installed.
        TradingViewConnectionError: The socket could not be opened.
    """
    try:
        from websocket import create_connection
    except ImportError as exc:  # pragma: no cover - exercised by injecting a fake
        raise TradingViewNotInstalledError(
            "the websocket-client package is not installed in this interpreter. "
            'Install it with `pip install -e ".[tradingview]"`; it is an optional '
            "extra because every test and the MT5 market path run without it.",
            setting="websocket-client",
        ) from exc

    try:
        return cast(
            WebSocketConnection,
            create_connection(
                WEBSOCKET_URL,
                timeout=timeout,
                origin=ORIGIN,
                header=[f"User-Agent: {USER_AGENT}"],
                suppress_origin=False,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - library exceptions are undocumented
        # `from None`: the library's message can carry the full handshake, and
        # this text reaches an operator. Only the failure type is reported.
        raise TradingViewConnectionError(
            "could not open the TradingView data socket",
            host="data.tradingview.com",
            failure=type(exc).__name__,
        ) from None


# --------------------------------------------------------------------------
# the source
# --------------------------------------------------------------------------


class TradingViewMarketDataSource:
    """Fetches the latest *closed* candles for one instrument from TradingView.

    One instance fetches once; :meth:`load` caches, so a caller may pre-flight
    the feed and then reuse the same source without a second round trip and
    without the two answers disagreeing. Same contract as the MT5 source.
    """

    def __init__(
        self,
        *,
        provider_symbol: str = "OANDA:XAUUSD",
        timeframe: Timeframe | str = Timeframe.M15,
        limit: int = 20,
        connector: Connector | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = 0,
        now: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
        session_id: str | None = None,
        series_id: str | None = None,
    ) -> None:
        """Build a source. Validates everything locally; opens nothing.

        Args:
            provider_symbol: TradingView's own name for the instrument. Must be
                a key of :data:`SUPPORTED_SYMBOLS`.
            timeframe: One of :data:`SUPPORTED_TIMEFRAMES`.
            limit: How many *closed* candles are wanted.
            connector: Opens the socket. Injected by tests so the whole session
                flow runs with no network anywhere.
            timeout_seconds: Per-receive timeout handed to the connection.
            max_retries: Extra attempts after a *transport* failure only. Zero
                by default - see the class of failure each error names.
            now: Clock for the closed-candle proof. Injected so tests do not
                depend on when they run.
            monotonic: Clock for the overall deadline. Separate from ``now`` on
                purpose: a test may freeze market time without the fetch loop
                becoming unbounded.
            session_id: Fixed chart session id, for deterministic tests.
            series_id: Fixed series handle, for deterministic tests. Both ids
                are correlation handles the feed echoes back, not secrets, and
                both are generated per instance so nothing global is mutated.

        Raises:
            MarketDataConfigurationError: Unsupported symbol or timeframe, or a
                limit outside the accepted bounds. Raised here, before any
                network call.
        """
        self._interval = _require_timeframe(timeframe)
        self._timeframe = Timeframe(timeframe)
        self._provider_symbol = _require_symbol(provider_symbol)
        self._canonical_symbol = SUPPORTED_SYMBOLS[self._provider_symbol]
        self._limit = _require_limit(limit)
        self._connector = connector if connector is not None else connect_tradingview
        self._timeout = float(timeout_seconds)
        self._max_retries = max(0, int(max_retries))
        self._now = now if now is not None else utc_now
        self._monotonic = monotonic if monotonic is not None else time.monotonic
        self._session_id = session_id or f"cs_{secrets.token_hex(6)}"
        self._series_id = series_id or f"sds_{secrets.token_hex(6)}"
        self._cached: LoadedSource[MarketDataInput] | None = None

    @property
    def timeframe(self) -> Timeframe:
        return self._timeframe

    @property
    def provider_symbol(self) -> str:
        return self._provider_symbol

    def load(self) -> LoadedSource[MarketDataInput]:
        """Fetch the latest closed candles, once.

        Only transport failures are retried, and only up to ``max_retries``. A
        protocol fault, a critical error from the feed, a malformed frame or an
        unusable candle are all deterministic: the same request would produce
        the same failure, so retrying it only delays the report.

        Raises:
            TradingViewNotInstalledError: The websocket client is missing.
            TradingViewConnectionError: The socket failed or closed early.
            TradingViewTimeoutError: The feed went quiet before completing.
            TradingViewFramingError: A malformed packet arrived.
            TradingViewProtocolError: The feed reported a protocol fault, or
                the series never completed within the fetch budget.
            TradingViewCriticalError: The feed reported a critical error.
            TradingViewCandleError: A returned bar was not usable.
            InsufficientBarsError: Fewer closed candles than were requested.
        """
        if self._cached is not None:
            return self._cached

        attempts = self._max_retries + 1
        last: TradingViewConnectionError | TradingViewTimeoutError | None = None
        for attempt in range(1, attempts + 1):
            try:
                self._cached = self._fetch()
                return self._cached
            except (TradingViewConnectionError, TradingViewTimeoutError) as exc:
                last = exc
                if attempt == attempts:
                    raise
                logger.warning(
                    "tradingview.retry attempt=%d/%d code=%s", attempt, attempts, exc.code
                )
        raise last if last is not None else TradingViewConnectionError("no attempt was made")

    # -- one attempt -------------------------------------------------------

    def _fetch(self) -> LoadedSource[MarketDataInput]:
        requested_at = self._now()
        deadline = self._monotonic() + self._timeout
        series_id = self._series_id
        symbol_id = f"sym_{secrets.token_hex(6)}"

        connection = self._connector(self._timeout)
        try:
            self._open_series(connection, series_id=series_id, symbol_id=symbol_id)
            raw, stats = self._read_series(connection, series_id=series_id, deadline=deadline)
        finally:
            # Every path. A socket left open holds a file descriptor and a
            # server-side session; the MT5 source shuts its terminal link down
            # the same way and for the same reason.
            _close_quietly(connection)

        retrieved_at = self._now()
        bars = self._normalise(raw)
        closed = self._closed_only(bars)

        if len(closed) < self._limit:
            raise InsufficientBarsError(
                f"asked for {self._limit} closed candles and could use {len(closed)} of the "
                f"{len(bars)} the feed returned. The market may be quiet, or this "
                f"timeframe may not have enough history on this venue yet.",
                symbol=self._provider_symbol,
                timeframe=str(self._timeframe),
                requested=self._limit,
                received=len(closed),
            )

        selected = closed[-self._limit :]
        payload = self._payload(selected, requested_at, retrieved_at)
        model = MarketDataInput.model_validate(payload)

        logger.info(
            "tradingview.fetch symbol=%s timeframe=%s closed=%d latest=%s",
            self._provider_symbol,
            self._timeframe,
            len(selected),
            selected[-1]["timestamp"],
        )

        return LoadedSource(
            model=model,
            raw_payload=payload,
            origin=f"{PROVIDER_NAME}:{self._provider_symbol}:{self._timeframe}",
            provenance={
                "kind": "market",
                "provider": PROVIDER_NAME,
                "provider_symbol": self._provider_symbol,
                "canonical_symbol": self._canonical_symbol,
                "timeframe": str(self._timeframe),
                "wire_interval": self._interval,
                "bars_requested": self._limit,
                "bars_received": len(bars),
                "bars_forming_dropped": len(bars) - len(closed),
                "bars_returned": len(selected),
                "requested_at": _iso(requested_at),
                "retrieved_at": _iso(retrieved_at),
                "latest_candle_at": selected[-1]["timestamp"],
                "endpoint": WEBSOCKET_URL,
                "authenticated": False,
                "messages_read": stats.messages,
                "heartbeats_echoed": stats.heartbeats,
                "unparseable_envelopes": stats.unparseable,
            },
        )

    # -- session -----------------------------------------------------------

    def _open_series(
        self, connection: WebSocketConnection, *, series_id: str, symbol_id: str
    ) -> None:
        """Send the handshake and ask for the series.

        The order matters and is the reference flow's: authorise, create the
        chart session, pin the timezone to UTC *before* asking for bars, resolve
        the symbol, then request the series.
        """
        resolve = json.dumps(
            {"symbol": self._provider_symbol, "adjustment": "splits", "session": "regular"},
            separators=(",", ":"),
        )
        commands = (
            ("set_auth_token", [AUTH_TOKEN]),
            ("chart_create_session", [self._session_id, ""]),
            # UTC at the source, so no timestamp in this pipeline is ever a
            # local time that merely looks like a UTC one.
            ("switch_timezone", [self._session_id, "Etc/UTC"]),
            ("resolve_symbol", [self._session_id, symbol_id, f"={resolve}"]),
            (
                "create_series",
                [
                    self._session_id,
                    series_id,
                    "s1",
                    symbol_id,
                    self._interval,
                    self._limit + SAFETY_MARGIN_BARS,
                    "",
                ],
            ),
        )
        for method, params in commands:
            self._send(connection, encode_message(method, params))

    def _send(self, connection: WebSocketConnection, frame: str) -> None:
        try:
            connection.send(frame)
        except Exception as exc:  # noqa: BLE001 - library exceptions are undocumented
            raise TradingViewConnectionError(
                "the TradingView socket failed while sending a request",
                failure=type(exc).__name__,
            ) from None

    def _read_series(
        self, connection: WebSocketConnection, *, series_id: str, deadline: float
    ) -> tuple[list[tuple[Any, ...]], _ReadStats]:
        """Read until the series completes, collecting bar rows.

        Bars are keyed by timestamp as they arrive, so a bar restated by a
        later ``du`` replaces the earlier copy rather than appearing twice.
        Deduplication therefore happens before validation, which is what stops
        one repeated row from being mistaken for two candles.
        """
        collected: dict[Any, tuple[Any, ...]] = {}
        stats = _ReadStats()
        buffer = ""

        while True:
            if stats.messages >= MAX_MESSAGES:
                raise TradingViewProtocolError(
                    "the feed sent more packets than one fetch will read without "
                    "completing the requested series",
                    packets=stats.messages,
                    ceiling=MAX_MESSAGES,
                )
            if self._monotonic() > deadline:
                raise TradingViewTimeoutError(
                    "the requested series did not complete within the timeout",
                    timeout_seconds=self._timeout,
                    symbol=self._provider_symbol,
                    timeframe=str(self._timeframe),
                )

            buffer += self._recv(connection, deadline)
            payloads, buffer = decode_frames(buffer)

            for payload in payloads:
                stats.messages += 1
                message = classify(payload)

                if message.kind is MessageKind.HEARTBEAT:
                    self._send(connection, encode_frame(message.raw))
                    stats.heartbeats += 1
                    continue
                if message.kind is MessageKind.UNPARSEABLE:
                    stats.unparseable += 1
                    continue
                if message.kind is MessageKind.CRITICAL_ERROR:
                    raise TradingViewCriticalError(
                        "the feed reported a critical error",
                        reported=error_summary(message),
                    )
                if message.kind is MessageKind.PROTOCOL_ERROR:
                    raise TradingViewProtocolError(
                        "the feed reported a protocol error",
                        reported=error_summary(message),
                    )
                if message.kind in (MessageKind.TIMESCALE_UPDATE, MessageKind.DATA_UPDATE):
                    self._collect(message, series_id, collected)
                    continue
                if message.kind is MessageKind.SERIES_COMPLETED:
                    if buffer:
                        raise TradingViewFramingError(
                            "the series completed with an unfinished packet still buffered",
                            buffered=len(buffer),
                        )
                    return list(collected.values()), stats

    def _collect(self, message: Message, series_id: str, into: dict[Any, tuple[Any, ...]]) -> None:
        for bar in extract_raw_bars(message, series_id):
            if len(bar.values) < _MIN_BAR_FIELDS:
                raise TradingViewCandleError(
                    "a candle row carried fewer fields than a candle needs",
                    fields=len(bar.values),
                    required=_MIN_BAR_FIELDS,
                    symbol=self._provider_symbol,
                )
            into[bar.values[0]] = bar.values

    def _recv(self, connection: WebSocketConnection, deadline: float) -> str:
        """One packet, as text.

        A receive failure is read as a timeout when the deadline has passed and
        as a transport failure otherwise. That distinction is worth making
        because only one of the two is ever retried.
        """
        try:
            data = connection.recv()
        except Exception as exc:  # noqa: BLE001 - library exceptions are undocumented
            if self._monotonic() > deadline:
                raise TradingViewTimeoutError(
                    "the feed stopped responding before the series completed",
                    timeout_seconds=self._timeout,
                    failure=type(exc).__name__,
                ) from None
            raise TradingViewConnectionError(
                "the TradingView socket failed while receiving",
                failure=type(exc).__name__,
            ) from None

        if isinstance(data, bytes):
            try:
                data = data.decode("utf-8")
            except UnicodeDecodeError:
                raise TradingViewFramingError(
                    "the feed sent a packet that is not valid UTF-8 text"
                ) from None
        if not data:
            raise TradingViewConnectionError(
                "the TradingView socket closed before the series completed"
            )
        return data

    # -- normalisation -----------------------------------------------------

    def _normalise(self, rows: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
        """Validate every row and turn it into a bar payload.

        Sorted ascending at the end, and the timestamps are already unique
        because collection was keyed by them. Both properties are re-checked by
        :class:`~goldpipeline.schemas.market.MarketDataSnapshot` downstream;
        doing it here as well means a fault is attributed to this adapter
        rather than surfacing later as a normalization error.
        """
        bars = [self._to_bar(row) for row in rows]
        bars.sort(key=lambda bar: bar["timestamp"])
        stamps = [bar["timestamp"] for bar in bars]
        if len(set(stamps)) != len(stamps):  # pragma: no cover - keyed collection prevents it
            raise TradingViewCandleError(
                "the feed returned two candles with the same timestamp",
                symbol=self._provider_symbol,
            )
        return bars

    def _to_bar(self, row: tuple[Any, ...]) -> dict[str, Any]:
        opened = self._timestamp(row[0])
        bar: dict[str, Any] = {
            "timestamp": _iso(opened),
            "open": self._price(row[1], "open"),
            "high": self._price(row[2], "high"),
            "low": self._price(row[3], "low"),
            "close": self._price(row[4], "close"),
        }
        volume = self._volume(row[5]) if len(row) > _MIN_BAR_FIELDS else None
        if volume is not None:
            # Absent volume stays absent. `OHLCBar.volume` is optional, so
            # there is no need to invent a zero - and a zero would be a lie
            # with the same shape as a real reading of no activity.
            bar["volume"] = volume
        self._check_ohlc(bar)
        return bar

    def _timestamp(self, value: Any) -> datetime:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TradingViewCandleError(
                "a candle timestamp was not a number", symbol=self._provider_symbol
            )
        if not math.isfinite(value) or value < MIN_TIMESTAMP:
            raise TradingViewCandleError(
                "a candle timestamp is not a usable date",
                symbol=self._provider_symbol,
            )
        moment = datetime.fromtimestamp(int(value), tz=UTC)
        if moment > self._now() + MAX_TIMESTAMP_SKEW:
            raise TradingViewCandleError(
                "a candle is stamped further into the future than clock skew explains",
                symbol=self._provider_symbol,
                candle_opened_at=_iso(moment),
            )
        return moment

    def _price(self, value: Any, field: str) -> str:
        """One price as an exact decimal string.

        Via ``str`` rather than binary float arithmetic, so nothing picks up
        representation noise on the way in - the same conversion the MT5 source
        uses. Precision is whatever the feed sent, clamped to what the schema
        accepts; no house rounding is applied, because a rounded price is a
        number that appears nowhere in the data and the gate compares article
        prices against the data exactly.
        """
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TradingViewCandleError(
                f"a candle {field} was not a number",
                symbol=self._provider_symbol,
                field=field,
            )
        if not math.isfinite(value):
            raise TradingViewCandleError(
                f"a candle {field} was not finite",
                symbol=self._provider_symbol,
                field=field,
            )
        if value <= 0:
            raise TradingViewCandleError(
                f"a candle {field} was not a positive price",
                symbol=self._provider_symbol,
                field=field,
            )
        try:
            number = Decimal(str(float(value)))
        except (InvalidOperation, ValueError, OverflowError):
            raise TradingViewCandleError(
                f"a candle {field} could not be read as a decimal",
                symbol=self._provider_symbol,
                field=field,
            ) from None
        exponent = number.as_tuple().exponent
        natural = -exponent if isinstance(exponent, int) else MAX_PRICE_DECIMALS
        places = max(0, min(natural, MAX_PRICE_DECIMALS))
        return str(number.quantize(Decimal(1).scaleb(-places)))

    def _volume(self, value: Any) -> str | None:
        """Volume, when the feed sent a usable one.

        Whatever this number is, it is not the physical volume of gold traded:
        OANDA reports its own activity, not the market's. It is carried because
        the schema has a place for it and dropped rather than guessed when it
        is unusable - a bad volume is not a reason to refuse a candle whose
        prices are sound.
        """
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        if not math.isfinite(value) or value < 0:
            return None
        return str(Decimal(str(float(value))))

    def _check_ohlc(self, bar: dict[str, Any]) -> None:
        """The candle relationships, checked here so the error names this feed.

        ``OHLCBar`` enforces the same invariants and would raise a validation
        error a few lines later. Doing it here yields a typed market-data
        failure that says which provider sent the impossible bar.
        """
        high = Decimal(bar["high"])
        low = Decimal(bar["low"])
        open_ = Decimal(bar["open"])
        close = Decimal(bar["close"])
        problems: list[str] = []
        if high < low:
            problems.append("high < low")
        if high < open_:
            problems.append("high < open")
        if high < close:
            problems.append("high < close")
        if low > open_:
            problems.append("low > open")
        if low > close:
            problems.append("low > close")
        if problems:
            raise TradingViewCandleError(
                "the feed returned an impossible candle: " + ", ".join(problems),
                symbol=self._provider_symbol,
                candle_opened_at=bar["timestamp"],
            )

    def _closed_only(self, bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop every bar whose period has not finished.

        ``open + duration <= now``, in UTC, with no tolerance. The MT5 source
        allows a few seconds of slack because it is second-guessing a bar the
        terminal has already labelled closed; here the only evidence is the
        feed's own timestamp against our clock, so slack would only ever admit
        a bar that is still moving. Being strict can cost one candle. Being
        lenient costs an article that was wrong when it was published.
        """
        moment = self._now()
        duration = self._timeframe.duration
        assert duration is not None, "every supported timeframe has a fixed duration"  # noqa: S101
        return [
            bar for bar in bars if datetime.fromisoformat(bar["timestamp"]) + duration <= moment
        ]

    def _payload(
        self, bars: list[dict[str, Any]], requested_at: datetime, retrieved_at: datetime
    ) -> dict[str, Any]:
        return {
            "symbol": self._canonical_symbol,
            "provider": PROVIDER_NAME,
            "provider_symbol": self._provider_symbol,
            "timeframe": str(self._timeframe),
            # Every timestamp above carries an explicit offset, so the
            # normalizer has no provider timezone to resolve.
            "timezone": None,
            "requested_at": _iso(requested_at),
            "retrieved_at": _iso(retrieved_at),
            "bars": bars,
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


class _ReadStats:
    """Counters recorded as provenance, so a quiet oddity is still visible."""

    __slots__ = ("heartbeats", "messages", "unparseable")

    def __init__(self) -> None:
        self.messages = 0
        self.heartbeats = 0
        self.unparseable = 0


def _require_timeframe(timeframe: Timeframe | str) -> str:
    try:
        resolved = Timeframe(timeframe)
    except ValueError:
        raise MarketDataConfigurationError(
            f"{timeframe!r} is not a timeframe this pipeline knows",
            setting="timeframe",
            supported=sorted(str(t) for t in SUPPORTED_TIMEFRAMES),
        ) from None
    interval = SUPPORTED_TIMEFRAMES.get(resolved)
    if interval is None:
        raise MarketDataConfigurationError(
            f"the TradingView provider does not support {resolved}",
            setting="timeframe",
            timeframe=str(resolved),
            supported=sorted(str(t) for t in SUPPORTED_TIMEFRAMES),
        )
    return interval


def _require_symbol(provider_symbol: str) -> str:
    cleaned = provider_symbol.strip().upper()
    if cleaned not in SUPPORTED_SYMBOLS:
        raise MarketDataConfigurationError(
            f"the TradingView provider does not support {provider_symbol!r}. It is never "
            "substituted for a similar feed: gold on another venue is a different "
            "instrument with a different spread.",
            setting="provider_symbol",
            supported=sorted(SUPPORTED_SYMBOLS),
        )
    return cleaned


def _require_limit(limit: int) -> int:
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise MarketDataConfigurationError("limit must be a whole number", setting="limit")
    if not MIN_CANDLE_LIMIT <= limit <= MAX_CANDLE_LIMIT:
        raise MarketDataConfigurationError(
            f"limit must be between {MIN_CANDLE_LIMIT} and {MAX_CANDLE_LIMIT}",
            setting="limit",
            minimum=MIN_CANDLE_LIMIT,
            maximum=MAX_CANDLE_LIMIT,
            requested=limit,
        )
    return limit


def _close_quietly(connection: WebSocketConnection) -> None:
    """Release the socket, and never mask the failure that led here."""
    try:
        connection.close()
    except Exception:  # noqa: BLE001
        logger.warning("tradingview.close failed; the socket may be left open")


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "MAX_CANDLE_LIMIT",
    "MIN_CANDLE_LIMIT",
    "ORIGIN",
    "PROVIDER_NAME",
    "SAFETY_MARGIN_BARS",
    "SUPPORTED_SYMBOLS",
    "SUPPORTED_TIMEFRAMES",
    "WEBSOCKET_URL",
    "Connector",
    "TradingViewMarketDataSource",
    "WebSocketConnection",
    "connect_tradingview",
]
