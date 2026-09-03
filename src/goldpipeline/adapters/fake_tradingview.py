"""Offline stand-in for the TradingView data socket.

Every test of this provider runs against this, so the suite needs no network
and not even the websocket library installed. It models the details that make
the real socket awkward, because those are the ones worth testing:

* **the newest bar is still forming.** :func:`make_series` lays out a series
  whose last entry has not closed, so a test can prove the adapter dropped it
  rather than trusting that it did;
* **several frames arrive in one packet, and packets split mid-frame.** The
  fake can deliver a whole conversation as one string, one frame per packet, or
  chopped at arbitrary byte offsets, which is how the buffering path is
  exercised;
* **heartbeats interleave with data** and must be echoed or the server would
  hang up.

It also records everything it was sent, so a test can assert on the session
flow - that the timezone was pinned before the series was requested, that the
symbol asked for was the one configured, that a heartbeat came back.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from goldpipeline.adapters.tradingview_protocol import encode_frame, encode_message

TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240}

DEFAULT_SERIES_ID = "sds_test"

FEED_INTERNALS_SENTINEL = "tv-session-cookie-DO-NOT-LEAK"
"""Planted in error payloads where a careless adapter might pick it up.

Real ``protocol_error`` frames echo request content and server-side detail.
Tests grep operator-facing error text for this string, so an adapter that
starts reporting raw feed payloads fails loudly.
"""


def make_series(
    *,
    count: int = 22,
    timeframe: str = "M15",
    now: datetime | None = None,
    close: float = 4323.50,
    step: float = 0.40,
    with_volume: bool = True,
) -> list[list[Any]]:
    """Bar rows in wire order, oldest first, newest still forming.

    The last row is the bar ``now`` falls inside: it opens at the most recent
    period boundary at or before ``now``, so ``open + period`` is still in the
    future and the bar has *not* closed. That mirrors the real feed, which
    counts the bar in progress among the bars it sends, and it means a caller
    asking for ``count - 1`` closed candles gets them while a caller asking for
    ``count`` does not.
    """
    moment = (now or datetime(2026, 9, 3, 12, 0, tzinfo=UTC)).astimezone(UTC)
    minutes = TIMEFRAME_MINUTES[timeframe]
    period = timedelta(minutes=minutes)
    # Floor `now` onto the period grid: that bar is the one still forming.
    seconds = int(moment.timestamp())
    newest_open = datetime.fromtimestamp(seconds - seconds % int(period.total_seconds()), tz=UTC)

    rows: list[list[Any]] = []
    for index in range(count):
        opened = newest_open - period * (count - 1 - index)
        base = close - step * (count - 1 - index)
        row: list[Any] = [
            float(int(opened.timestamp())),
            round(base - step / 2, 5),
            round(base + step, 5),
            round(base - step, 5),
            round(base, 5),
        ]
        if with_volume:
            row.append(float(100 + index))
        rows.append(row)
    return rows


def timescale_update(rows: Iterable[list[Any]], *, series_id: str = DEFAULT_SERIES_ID) -> str:
    """A framed ``timescale_update`` carrying *rows*."""
    return encode_message(
        "timescale_update",
        [
            "cs_test",
            {series_id: {"s": [{"i": i, "v": row} for i, row in enumerate(rows)]}},
        ],
    )


def data_update(rows: Iterable[list[Any]], *, series_id: str = DEFAULT_SERIES_ID) -> str:
    """A framed ``du`` carrying *rows*, in the same nesting as history."""
    return encode_message(
        "du",
        ["cs_test", {series_id: {"s": [{"i": i, "v": row} for i, row in enumerate(rows)]}}],
    )


def series_completed(*, series_id: str = DEFAULT_SERIES_ID) -> str:
    return encode_message("series_completed", ["cs_test", series_id, "streaming", "s1"])


def protocol_error() -> str:
    return encode_message("protocol_error", [FEED_INTERNALS_SENTINEL])


def critical_error() -> str:
    return encode_message("critical_error", [FEED_INTERNALS_SENTINEL])


def heartbeat(number: int = 1) -> str:
    return encode_frame(f"~h~{number}")


def session_chatter() -> str:
    """A recognised envelope this fetch has no use for."""
    return encode_message("quote_completed", ["qs_test", "OANDA:XAUUSD"])


def unparseable() -> str:
    """A well-framed payload that is not readable JSON."""
    return encode_frame("{not json at all")


def conversation(
    rows: list[list[Any]] | None = None,
    *,
    series_id: str = DEFAULT_SERIES_ID,
    include_heartbeat: bool = True,
    include_chatter: bool = True,
) -> str:
    """A complete successful conversation as one string."""
    parts: list[str] = []
    if include_chatter:
        parts.append(session_chatter())
    if include_heartbeat:
        parts.append(heartbeat())
    parts.append(timescale_update(rows if rows is not None else make_series(), series_id=series_id))
    parts.append(series_completed(series_id=series_id))
    return "".join(parts)


@dataclass
class SentCommand:
    """One command the adapter sent, already decoded."""

    method: str
    params: list[Any]


@dataclass
class FakeConnection:
    """A scripted websocket. Hands back ``packets`` one ``recv`` at a time.

    Args:
        packets: What each ``recv`` returns, in order. A packet may hold any
            number of whole frames, or part of one.
        recv_error: Raised instead of returning, once ``packets`` is exhausted
            or immediately when ``fail_after`` is zero.
        fail_after: How many successful receives happen before ``recv_error``.
        send_error: Raised by ``send`` on the ``send_error_at``-th call.
    """

    packets: list[str] = field(default_factory=list)
    recv_error: BaseException | None = None
    fail_after: int | None = None
    send_error: BaseException | None = None
    send_error_at: int = 1
    close_error: BaseException | None = None

    sent: list[str] = field(default_factory=list)
    commands: list[SentCommand] = field(default_factory=list)
    heartbeat_echoes: list[str] = field(default_factory=list)
    closed: int = 0
    received: int = 0

    def send(self, payload: str) -> None:
        self.sent.append(payload)
        if self.send_error is not None and len(self.sent) == self.send_error_at:
            raise self.send_error
        body = _frame_body(payload)
        if body.startswith("~h~"):
            self.heartbeat_echoes.append(body)
            return
        try:
            decoded = json.loads(body)
        except ValueError:
            return
        if isinstance(decoded, dict):
            self.commands.append(
                SentCommand(method=str(decoded.get("m", "")), params=list(decoded.get("p") or []))
            )

    def recv(self) -> str | bytes:
        if self.fail_after is not None and self.received >= self.fail_after:
            raise self.recv_error or ConnectionError("fake socket failed")
        if self.received < len(self.packets):
            packet = self.packets[self.received]
            self.received += 1
            return packet
        if self.recv_error is not None:
            raise self.recv_error
        return ""  # socket closed

    def close(self) -> None:
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error

    # -- assertions a test finds useful -----------------------------------

    def method_order(self) -> list[str]:
        return [command.method for command in self.commands]

    def command(self, method: str) -> SentCommand:
        for entry in self.commands:
            if entry.method == method:
                return entry
        raise AssertionError(f"the adapter never sent {method!r}; sent {self.method_order()}")


@dataclass
class RecordingConnector:
    """Hands out :class:`FakeConnection` instances and remembers them.

    So a test can prove that a retry opened a second socket, and that every
    socket it handed out was closed.
    """

    connections: list[FakeConnection] = field(default_factory=list)
    scripts: list[FakeConnection] = field(default_factory=list)
    timeouts: list[float] = field(default_factory=list)
    connect_error: BaseException | None = None

    def __call__(self, timeout: float) -> FakeConnection:
        self.timeouts.append(timeout)
        if self.connect_error is not None:
            raise self.connect_error
        # A test always scripts what it needs; an unscripted call gets a socket
        # that closes immediately, which fails loudly rather than hanging.
        connection = self.scripts.pop(0) if self.scripts else FakeConnection()
        self.connections.append(connection)
        return connection

    @property
    def only(self) -> FakeConnection:
        if len(self.connections) != 1:
            raise AssertionError(f"expected exactly one connection, got {len(self.connections)}")
        return self.connections[0]

    @property
    def all_closed(self) -> bool:
        return all(connection.closed >= 1 for connection in self.connections)


def scripted(*packets: str) -> FakeConnection:
    """A connection that returns each *packet* from one ``recv``."""
    return FakeConnection(packets=list(packets))


def single_packet(rows: list[list[Any]] | None = None, **kwargs: Any) -> FakeConnection:
    """A connection that delivers a whole successful conversation at once."""
    return scripted(conversation(rows, **kwargs))


def _frame_body(frame: str) -> str:
    parts = frame.split("~m~")
    return parts[2] if len(parts) >= 3 else frame


__all__ = [
    "DEFAULT_SERIES_ID",
    "FEED_INTERNALS_SENTINEL",
    "TIMEFRAME_MINUTES",
    "FakeConnection",
    "RecordingConnector",
    "SentCommand",
    "conversation",
    "critical_error",
    "data_update",
    "heartbeat",
    "make_series",
    "protocol_error",
    "scripted",
    "series_completed",
    "session_chatter",
    "single_packet",
    "timescale_update",
    "unparseable",
]
