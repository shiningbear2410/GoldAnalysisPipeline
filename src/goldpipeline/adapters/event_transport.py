"""Fetching events from an optional remote producer.

**This is a client, and never a server.** The pipeline reaches out; nothing
reaches in. That single decision removes the whole class of problems a receiver
would bring - an inbound port on a machine behind NAT, a listener to supervise,
a daemon whose restart policy is its own invention - and it costs nothing,
because the machine that should be reachable is the producer's, not this one.

**The transport is allowed to be dumb.** It may hand back the same events on
every call, forever. Admission is decided locally by
:mod:`goldpipeline.services.event_intake` against the ingestion ledger, which
already keys on ``event_id`` and the payload digest. So the hard guarantee -
exactly one Run per analysis - is not something this module has to provide, and
therefore not something a network can take away. At-least-once is enough.

**Nothing here trusts the answer.** A response is bytes from another machine: it
is size-capped before parsing, shape-checked before use, and every event in it
is validated by the same ``parse_event`` a local file goes through. The producer
cannot name a path, a model, a chat or a Run - the schema has no field for any of
those.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol
from urllib.parse import urlencode, urlsplit, urlunsplit

from goldpipeline.domain.errors import (
    RemoteIntakeConfigurationError,
    RemoteIntakeResponseError,
    RemoteIntakeTransportError,
)

logger = logging.getLogger(__name__)

PENDING_PATH = "/outbox/pending"
"""The one endpoint this pipeline knows how to call."""

_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
"""Answers worth trying again on the next tick.

A fetch is free to retry - it delivered nothing and changed nothing - so this
list is generous compared with the publisher's. The distinction that matters is
against 401/403, where retrying every minute would achieve nothing except noise
in someone's auth log.
"""


class EventTransport(Protocol):
    """Where pending events come from.

    A Protocol, so tests supply a list and production supplies HTTPS, with no
    conditional in the service between them.
    """

    def fetch_pending(self, *, limit: int) -> list[dict[str, Any]]:
        """Return up to *limit* events, oldest first.

        Returning the same events again on a later call is expected and safe.

        Raises:
            RemoteIntakeTransportError: The producer could not be reached.
            RemoteIntakeResponseError: The producer answered unusably.
            RemoteIntakeConfigurationError: The credential was refused.
        """
        ...


class HttpOutboxTransport:
    """Reads ``GET /outbox/pending`` from the producer over HTTPS.

    Args:
        base_url: The producer's HTTPS origin, validated by
            :class:`~goldpipeline.config.IngestSettings`.
        token: Bearer credential, from the credential store. Never logged.
        timeout_seconds: Bounded well below the worker's tick budget.
        max_bytes: Hard cap on the response body.
        client: Injected for tests; production builds its own.
    """

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        timeout_seconds: float,
        max_bytes: int,
        client: Any | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout_seconds
        self._max_bytes = max_bytes
        self._client = client

    # -- request -----------------------------------------------------------

    def _url(self, limit: int) -> str:
        parts = urlsplit(self._base_url + PENDING_PATH)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode({"limit": limit}), ""))

    def _get(self, url: str) -> Any:
        """Issue the request.

        Every exception is replaced rather than wrapped. The HTTP layer puts
        request detail - including headers - into the text it produces, and this
        message travels into a tick record.
        """
        import httpx2

        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }
        try:
            if self._client is not None:
                return self._client.get(url, headers=headers)
            # follow_redirects is off deliberately. A redirect to another origin
            # would re-send the Authorization header there, handing the token to
            # whoever controls the redirect target.
            with httpx2.Client(
                timeout=self._timeout, follow_redirects=False, verify=True
            ) as client:
                return client.get(url, headers=headers)
        except httpx2.TimeoutException as exc:
            raise RemoteIntakeTransportError(
                "the producer did not answer within the timeout", reason="timeout"
            ) from _Scrubbed(exc)
        except httpx2.HTTPError as exc:
            raise RemoteIntakeTransportError(
                "the connection to the producer failed", reason="connection"
            ) from _Scrubbed(exc)
        except Exception as exc:  # noqa: BLE001 - anything here may carry the header
            raise RemoteIntakeTransportError(
                "the request to the producer failed in an unexpected way",
                reason="unexpected",
            ) from _Scrubbed(exc)

    # -- response ----------------------------------------------------------

    def fetch_pending(self, *, limit: int) -> list[dict[str, Any]]:
        """Fetch and shape-check one batch. See :class:`EventTransport`."""
        response = self._get(self._url(limit))
        status = int(getattr(response, "status_code", 0) or 0)

        if status in (401, 403):
            # A human must fix this; retrying every minute only fills a log.
            raise RemoteIntakeConfigurationError(
                f"the producer refused the ingest credential (HTTP {status})",
                setting="INGEST_TOKEN",
            )
        if 300 <= status < 400:
            raise RemoteIntakeResponseError(
                f"the producer answered with a redirect (HTTP {status}); redirects are "
                "not followed, because the Authorization header must not travel to "
                "another origin",
                status_code=status,
            )
        if status in _RETRYABLE_STATUS:
            raise RemoteIntakeTransportError(
                f"the producer answered HTTP {status}", reason="status", status_code=status
            )
        if status != 200:
            raise RemoteIntakeResponseError(
                f"the producer answered HTTP {status}", status_code=status
            )

        body = self._body(response)
        return _envelope_events(body, limit=limit)

    def _body(self, response: Any) -> Any:
        """Read the body, refusing anything past the cap before parsing it."""
        declared = getattr(response, "headers", {}) or {}
        length = declared.get("content-length") or declared.get("Content-Length")
        if length is not None:
            try:
                if int(length) > self._max_bytes:
                    raise RemoteIntakeResponseError(
                        "the producer's response exceeds the size cap",
                        limit_bytes=self._max_bytes,
                    )
            except (TypeError, ValueError):
                # An unparseable Content-Length is not itself fatal; the real
                # check below works on the bytes that actually arrived.
                pass

        content = getattr(response, "content", b"") or b""
        if len(content) > self._max_bytes:
            raise RemoteIntakeResponseError(
                "the producer's response exceeds the size cap", limit_bytes=self._max_bytes
            )

        import json

        try:
            return json.loads(content.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise RemoteIntakeResponseError("the producer's response is not UTF-8") from _Scrubbed(
                exc
            )
        except json.JSONDecodeError as exc:
            raise RemoteIntakeResponseError(
                "the producer's response is not valid JSON"
            ) from _Scrubbed(exc)


def _envelope_events(body: Any, *, limit: int) -> list[dict[str, Any]]:
    """Check the envelope only. Event contents are validated by ``parse_event``.

    Refused whole rather than partially: a response whose *shape* is wrong tells
    us the producer is not speaking this contract, and picking usable-looking
    fragments out of it would be guessing.
    """
    if not isinstance(body, dict):
        raise RemoteIntakeResponseError(
            f"the producer's response must be a JSON object, got {type(body).__name__}"
        )
    if "events" not in body:
        raise RemoteIntakeResponseError("the producer's response has no 'events' field")

    events = body["events"]
    if not isinstance(events, list):
        raise RemoteIntakeResponseError(f"'events' must be a list, got {type(events).__name__}")
    if len(events) > limit:
        raise RemoteIntakeResponseError(
            f"the producer returned {len(events)} events, more than the requested {limit}",
            limit=limit,
        )
    for index, item in enumerate(events):
        if not isinstance(item, dict):
            raise RemoteIntakeResponseError(
                f"event at index {index} must be a JSON object, got {type(item).__name__}"
            )
    return list(events)


class _Scrubbed(Exception):
    """Carries an exception's type but not its text.

    The same guard the publisher uses. A cause is worth keeping for a traceback;
    the message attached to it may contain the request URL or the Authorization
    header, and this one ends up in a tick record.
    """

    def __init__(self, original: BaseException) -> None:
        super().__init__(type(original).__name__)


__all__ = ["PENDING_PATH", "EventTransport", "HttpOutboxTransport"]
