"""TradingView's websocket wire format, and nothing else.

**This module is a boundary, not a feature.** TradingView's data socket speaks
a private, undocumented protocol: length-prefixed ``~m~`` frames, heartbeats,
session handles, ``resolve_symbol``, ``create_series``. None of that is a
project API and none of it is stable. It can change without notice, because
nobody promised it would not.

So every detail of it lives here and in
:mod:`goldpipeline.adapters.tradingview_market`. Outside those two modules,
nothing knows that a frame has a length header, that a session id looks like
``cs_ab12cd34ef56``, or that ``unauthorized_user_token`` is a string anyone
sends. The rest of the pipeline sees :class:`~goldpipeline.schemas.market.OHLCBar`
and a provider name.

The consequence worth stating plainly: if TradingView changes this protocol,
the fix is confined to these two files. Article contracts, numeric semantics,
the analysis stages, a future ICT engine and the trade plan cannot be affected,
because none of them can see any of this.

**No I/O here.** Pure functions over strings, so the awkward parts - a
truncated tail, a length header that lies, several messages in one packet - are
tested directly rather than through a socket.

The wire format, for the record::

    ~m~<character length of payload>~m~<payload>

A payload is either JSON (``{"m": method, "p": params}``) or a heartbeat
(``~h~<n>``), which must be echoed back inside the same framing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from goldpipeline.domain.errors import TradingViewFramingError

FRAME_DELIMITER = "~m~"
HEARTBEAT_PREFIX = "~h~"

MAX_FRAME_LENGTH = 4 * 1024 * 1024
"""Largest payload a single frame may declare.

A length header is the one number this module has to trust before it can read
anything, so it is bounded. A frame claiming to be gigabytes is a malformed
frame, not a large one, and finding that out from a header is much cheaper
than finding it out from memory.
"""

_HEADER_RE = re.compile(r"^~m~(\d{1,9})~m~")

AUTH_TOKEN = "unauthorized_user_token"
"""The public, unauthenticated token the reference flow uses.

Not a credential. It is the literal string that means "no account", it grants
nothing, and this provider performs no login of any kind.
"""


class MessageKind(StrEnum):
    """What an incoming payload is, as far as a candle fetch cares."""

    HEARTBEAT = "HEARTBEAT"
    """Must be echoed, or the server closes the socket."""

    TIMESCALE_UPDATE = "TIMESCALE_UPDATE"
    """The bulk history answer to ``create_series``. Carries bars."""

    DATA_UPDATE = "DATA_UPDATE"
    """An incremental update (``du``). Carries bars in the same shape."""

    SERIES_COMPLETED = "SERIES_COMPLETED"
    """The requested series finished sending.

    Says the *response* is complete. It says nothing about whether the newest
    bar in it has closed, which is a separate question answered by arithmetic
    in the market source.
    """

    PROTOCOL_ERROR = "PROTOCOL_ERROR"
    CRITICAL_ERROR = "CRITICAL_ERROR"

    OTHER = "OTHER"
    """A recognised envelope this fetch has no use for: quotes, session status."""

    UNPARSEABLE = "UNPARSEABLE"
    """Well-framed but not JSON we can read.

    Counted and skipped rather than fatal. The socket carries chatter unrelated
    to the series, and one unreadable envelope cannot contribute a bar - so it
    cannot corrupt the result either. A malformed *bar*, inside a message we did
    recognise, is a different matter and fails the fetch.
    """


_KINDS: dict[str, MessageKind] = {
    "timescale_update": MessageKind.TIMESCALE_UPDATE,
    "du": MessageKind.DATA_UPDATE,
    "series_completed": MessageKind.SERIES_COMPLETED,
    "protocol_error": MessageKind.PROTOCOL_ERROR,
    "critical_error": MessageKind.CRITICAL_ERROR,
}


@dataclass(frozen=True)
class Message:
    """One decoded payload."""

    kind: MessageKind
    method: str = ""
    params: tuple[Any, ...] = ()
    raw: str = ""
    """The framed payload, kept only for heartbeat echo. Never logged."""


# --------------------------------------------------------------------------
# framing
# --------------------------------------------------------------------------


def encode_frame(payload: str) -> str:
    """Wrap *payload* in one ``~m~`` frame.

    The length is in characters, matching what the server sends and what
    :func:`decode_frames` reads back.
    """
    return f"{FRAME_DELIMITER}{len(payload)}{FRAME_DELIMITER}{payload}"


def encode_message(method: str, params: list[Any]) -> str:
    """One framed JSON command.

    ``separators`` is pinned so the encoding is byte-stable: a test that asserts
    on a sent frame should not start failing because a default changed.
    """
    return encode_frame(json.dumps({"m": method, "p": params}, separators=(",", ":")))


def decode_frames(data: str) -> tuple[list[str], str]:
    """Split *data* into complete payloads plus any incomplete tail.

    Returns ``(payloads, remainder)``. The remainder is a frame that has begun
    but not arrived in full; a caller holding a stream keeps it and prepends it
    to the next read. A caller that has reached the end of the conversation
    treats a non-empty remainder as a truncated packet - which the market
    source does.

    Buffering the tail rather than rejecting it is the one place this module is
    lenient, and it is lenient about *arrival*, never about *content*: a
    complete frame whose header is unusable raises, because a length that
    cannot be trusted means the position of every following frame is unknown.

    Raises:
        TradingViewFramingError: A frame is present but malformed - no header
            where one must be, a length of zero, or a length past the ceiling.
    """
    payloads: list[str] = []
    cursor = 0
    length = len(data)

    while cursor < length:
        rest = data[cursor:]
        header = _HEADER_RE.match(rest)
        if header is None:
            if _partial_header(rest):
                # The delimiter or its length digits are still arriving.
                return payloads, rest
            raise TradingViewFramingError(
                "packet does not start with a frame header",
                offset=cursor,
            )

        declared = int(header.group(1))
        if declared == 0:
            raise TradingViewFramingError("frame declares a length of zero", offset=cursor)
        if declared > MAX_FRAME_LENGTH:
            raise TradingViewFramingError(
                "frame declares a length past the accepted ceiling",
                declared=declared,
                ceiling=MAX_FRAME_LENGTH,
            )

        start = cursor + header.end()
        end = start + declared
        if end > length:
            return payloads, rest  # incomplete body; wait for more
        payloads.append(data[start:end])
        cursor = end

    return payloads, ""


def _partial_header(rest: str) -> bool:
    """Whether *rest* could still become a valid header once more data arrives.

    Two ways it can: the opening delimiter itself is half-arrived (``"~m"``),
    or the delimiter is complete and its digits are (``"~m~12"``). Anything
    else is not an unfinished header, it is a bad one - and the caller raises,
    because guessing where the next frame starts is how misaligned bytes become
    candle history.
    """
    if len(rest) < len(FRAME_DELIMITER):
        return FRAME_DELIMITER.startswith(rest)
    if not rest.startswith(FRAME_DELIMITER):
        return False
    tail = rest[len(FRAME_DELIMITER) :]
    digits = tail[:9]
    if not digits:
        return True
    consumed = 0
    for char in digits:
        if not char.isdigit():
            break
        consumed += 1
    if consumed == 0:
        return False
    # Digits, then either nothing yet or the start of the closing delimiter.
    return FRAME_DELIMITER.startswith(tail[consumed : consumed + len(FRAME_DELIMITER)])


# --------------------------------------------------------------------------
# messages
# --------------------------------------------------------------------------


def is_heartbeat(payload: str) -> bool:
    """Whether this payload is a heartbeat rather than a command."""
    return payload.startswith(HEARTBEAT_PREFIX)


def classify(payload: str) -> Message:
    """Read one framed payload into a :class:`Message`.

    Never raises. An envelope this function cannot read becomes
    :attr:`MessageKind.UNPARSEABLE`, and the caller decides what that is worth;
    raising here would let unrelated socket chatter fail a valid fetch.
    """
    if is_heartbeat(payload):
        return Message(kind=MessageKind.HEARTBEAT, raw=payload)

    try:
        decoded = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return Message(kind=MessageKind.UNPARSEABLE)

    if not isinstance(decoded, dict):
        return Message(kind=MessageKind.UNPARSEABLE)

    method = decoded.get("m")
    if not isinstance(method, str):
        return Message(kind=MessageKind.UNPARSEABLE)

    params = decoded.get("p")
    return Message(
        kind=_KINDS.get(method, MessageKind.OTHER),
        method=method,
        params=tuple(params) if isinstance(params, list) else (),
    )


def error_summary(message: Message) -> str:
    """A short, safe description of a server-reported error.

    Deliberately not the payload. An error frame can echo back what was sent
    and carry server-side detail of no diagnostic value, and this string goes
    into an operator-facing exception. Only the method name and the shape of
    what came with it are reported.
    """
    return f"{message.method} with {len(message.params)} parameter(s)"


# --------------------------------------------------------------------------
# series extraction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RawBar:
    """One bar exactly as the wire gave it, before any validation.

    Values stay ``Any`` on purpose. Deciding whether ``"4323"`` or ``None`` or
    ``float("nan")`` is an acceptable price is the market source's job, and
    doing it here would put a trading-data policy inside a codec.
    """

    values: tuple[Any, ...]


def extract_raw_bars(message: Message, series_id: str) -> list[RawBar]:
    """Pull the bar rows for *series_id* out of a series message.

    Both ``timescale_update`` and ``du`` nest their bars the same way::

        p[1][<series id>]["s"] -> [{"i": index, "v": [ts, o, h, l, c, vol]}, ...]

    Anything not shaped like that yields nothing. A missing series is normal -
    the socket carries messages about other series - and an empty list lets the
    caller keep reading rather than treating chatter as failure.
    """
    if len(message.params) < 2:
        return []
    body = message.params[1]
    if not isinstance(body, dict):
        return []
    series = body.get(series_id)
    if not isinstance(series, dict):
        return []
    rows = series.get("s")
    if not isinstance(rows, list):
        return []

    bars: list[RawBar] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        values = row.get("v")
        if isinstance(values, list):
            bars.append(RawBar(values=tuple(values)))
    return bars


__all__ = [
    "AUTH_TOKEN",
    "FRAME_DELIMITER",
    "HEARTBEAT_PREFIX",
    "MAX_FRAME_LENGTH",
    "Message",
    "MessageKind",
    "RawBar",
    "classify",
    "decode_frames",
    "encode_frame",
    "encode_message",
    "error_summary",
    "extract_raw_bars",
    "is_heartbeat",
]
