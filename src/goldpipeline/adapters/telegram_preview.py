"""Reading a Telegram channel's public preview page.

``https://t.me/s/<channel>`` is the page a browser shows anyone, with no account
and no token. That is the entire reason this transport exists: the Bot API does
not deliver messages authored by other bots, so a bot cannot read the channels
this pipeline needs, however it is configured. The public page can be read by
anybody, including us.

**A client, never a server**, and an unauthenticated one. No credential is sent,
so none can leak; a page that comes back is public information either way.

**Parsed with the standard library's HTML parser, not a regex.** Telegram's
markup nests and its text contains angle brackets, emoji and Vietnamese - the
cases where a regex silently returns half a message. When the markup changes,
this fails to find messages and says so, rather than inventing fields from a
partial match.

**The channel name comes from configuration, never from the page.** A page can
claim to be anything; which channels are trusted is a decision made on this
machine.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from typing import Any, Protocol

from goldpipeline.domain.errors import NewsFetchError, NewsParseError

logger = logging.getLogger(__name__)

PREVIEW_BASE = "https://t.me/s"
"""The public preview endpoint. HTTPS only, and the only host this reads."""

DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_MAX_BYTES = 2_097_152
"""2 MiB. A preview page of twenty messages is far smaller; this is the ceiling
past which something is wrong rather than merely long."""

USER_AGENT = "GoldAnalysisPipeline/1.0 (+news-collector; read-only)"
"""Deterministic and honest. Not a browser impersonation."""

_CHANNEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{3,31}$")
"""What a Telegram channel name may look like. Applied to configuration."""


@dataclass(frozen=True)
class RawMessage:
    """One message as the page presented it, before any interpretation."""

    message_id: int
    published_at: datetime | None
    """``None`` when the page carried no parseable time. Never invented."""

    text: str


@dataclass
class ParsedPage:
    """One preview page."""

    messages: list[RawMessage] = field(default_factory=list)

    @property
    def oldest_id(self) -> int | None:
        """The lowest message id on the page - where the next page starts."""
        return min((m.message_id for m in self.messages), default=None)


class NewsPageFetcher(Protocol):
    """Fetches one preview page. A Protocol so tests hand over saved HTML."""

    def fetch(self, channel: str, *, before: int | None = None) -> str:
        """Return the page's HTML.

        Raises:
            NewsFetchError: The page could not be retrieved.
        """
        ...


def validate_channel(name: str) -> str:
    """Check a configured channel name before it reaches a URL.

    Configuration is trusted more than page content, but not blindly: a name
    with a slash or a dot would change which URL is requested.
    """
    cleaned = name.strip().lstrip("@")
    if not _CHANNEL_RE.fullmatch(cleaned):
        raise NewsFetchError(f"invalid channel name: {name!r}", channel=name)
    return cleaned


class HttpPreviewFetcher:
    """Fetches preview pages over HTTPS."""

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_bytes: int = DEFAULT_MAX_BYTES,
        client: Any | None = None,
    ) -> None:
        self._timeout = timeout_seconds
        self._max_bytes = max_bytes
        self._client = client

    def url_for(self, channel: str, before: int | None = None) -> str:
        base = f"{PREVIEW_BASE}/{validate_channel(channel)}"
        return f"{base}?before={before}" if before is not None else base

    def fetch(self, channel: str, *, before: int | None = None) -> str:
        """Retrieve one page. See :class:`NewsPageFetcher`."""
        import httpx2

        url = self.url_for(channel, before)
        headers = {"User-Agent": USER_AGENT, "Accept": "text/html"}

        try:
            if self._client is not None:
                response = self._client.get(url, headers=headers)
            else:
                # Redirects are followed only because t.me answers some channel
                # forms with one; the request carries no credential, so a
                # redirect can leak nothing. Nothing is sent, only read.
                with httpx2.Client(
                    timeout=self._timeout, follow_redirects=True, verify=True
                ) as client:
                    response = client.get(url, headers=headers)
        except Exception as exc:  # noqa: BLE001 - the message may carry the URL
            raise NewsFetchError(
                "the preview page could not be retrieved", channel=channel
            ) from _Scrubbed(exc)

        status = int(getattr(response, "status_code", 0) or 0)
        if status != 200:
            raise NewsFetchError(
                f"the preview page answered HTTP {status}", channel=channel, status_code=status
            )

        content = getattr(response, "content", b"") or b""
        if len(content) > self._max_bytes:
            raise NewsFetchError(
                "the preview page exceeds the size cap",
                channel=channel,
                limit_bytes=self._max_bytes,
            )
        return content.decode("utf-8", errors="replace")


class _PreviewParser(HTMLParser):
    """Collects messages from Telegram preview markup.

    Structure relied upon, and nothing else:

    * ``data-post="channel/12345"`` - the message id;
    * ``<time datetime="...">`` inside the message - the publication time;
    * ``class="tgme_widget_message_text"`` - the body.

    Three attributes rather than a shape, so a layout change that keeps them
    keeps working. If they disappear the parser finds nothing, which the caller
    reports as a failed source - the honest outcome when a page stops being the
    page we were written against.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.messages: list[RawMessage] = []
        self._current_id: int | None = None
        self._current_time: datetime | None = None
        self._text_depth = 0
        self._chunks: list[str] = []
        self._seen_ids: set[int] = set()

    # -- tags ---------------------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: (value or "") for key, value in attrs}
        classes = values.get("class", "").split()

        post = values.get("data-post")
        if post and "/" in post:
            self._flush()
            self._current_id = _int_or_none(post.rsplit("/", 1)[1])
            self._current_time = None

        if tag == "time" and "datetime" in values and self._current_time is None:
            self._current_time = parse_timestamp(values["datetime"])

        if any(cls.startswith("tgme_widget_message_text") for cls in classes):
            self._text_depth = 1
            return

        if self._text_depth:
            self._text_depth += 1
            if tag == "br":
                self._chunks.append("\n")
                self._text_depth -= 1

    def handle_endtag(self, tag: str) -> None:
        if self._text_depth:
            self._text_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._text_depth:
            self._chunks.append(data)

    # -- assembly -----------------------------------------------------------

    def _flush(self) -> None:
        if self._current_id is None:
            self._chunks.clear()
            return
        if self._current_id in self._seen_ids:
            self._chunks.clear()
            self._current_id = None
            return

        text = re.sub(r"[ \t]+", " ", "".join(self._chunks)).strip()
        self._seen_ids.add(self._current_id)
        self.messages.append(
            RawMessage(
                message_id=self._current_id,
                published_at=self._current_time,
                text=text,
            )
        )
        self._chunks.clear()
        self._current_id = None

    def close(self) -> None:
        super().close()
        self._flush()


def parse_preview_page(html: str) -> ParsedPage:
    """Extract messages from one preview page.

    Raises:
        NewsParseError: The page contains no recognisable message at all.
            Deliberately an error rather than an empty page: "no messages here"
            and "this is no longer the markup we parse" need different answers,
            and only one of them is a normal Saturday.
    """
    parser = _PreviewParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # noqa: BLE001 - malformed markup, not our bug
        raise NewsParseError("the preview page could not be parsed") from _Scrubbed(exc)

    if not parser.messages:
        raise NewsParseError("the preview page contained no recognisable messages")
    return ParsedPage(messages=parser.messages)


def parse_timestamp(value: str) -> datetime | None:
    """Parse Telegram's ``datetime`` attribute into aware UTC.

    Returns ``None`` rather than guessing. An item whose time cannot be read is
    an item that cannot be placed in a window, and inventing one would put a
    story in a day it did not happen.
    """
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # A naive stamp has no meaning here: the machine's local zone is not the
        # publisher's, and assuming otherwise silently shifts every item.
        return None
    from datetime import UTC

    return parsed.astimezone(UTC)


def _int_or_none(value: str) -> int | None:
    try:
        number = int(value)
    except ValueError:
        return None
    return number if number > 0 else None


class _Scrubbed(Exception):
    """Carries an exception's type but not its text.

    The same guard the publisher and the event transport use: a cause is worth
    keeping, but the message attached to it may hold a URL, and these travel
    into logs.
    """

    def __init__(self, original: BaseException) -> None:
        super().__init__(type(original).__name__)


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "PREVIEW_BASE",
    "USER_AGENT",
    "HttpPreviewFetcher",
    "NewsPageFetcher",
    "ParsedPage",
    "RawMessage",
    "parse_preview_page",
    "parse_timestamp",
    "validate_channel",
]
