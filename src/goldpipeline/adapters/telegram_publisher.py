"""Telegram transport for the publisher.

The only module that holds a bot token or opens a socket to Telegram.

**The token is in the URL.** Telegram authenticates by path -
``https://api.telegram.org/bot<TOKEN>/sendMessage`` - which makes every HTTP
exception a potential credential leak, because client libraries put the request
URL in their messages. So no exception from the HTTP layer is ever allowed to
propagate: each is caught and replaced with a message written here. Nothing that
leaves this module has seen the token.

**Plain text only.** No ``parse_mode``. The article was approved as exact
characters, and a markup parser is a second thing that can change what readers
see - an unescaped ``*`` or ``_`` would silently restyle or swallow text that
nobody reviewed in that form.
"""

from __future__ import annotations

import logging
from typing import Any

from goldpipeline.adapters.publisher_client import SendOutcome, SendRequest
from goldpipeline.config import TelegramSettings
from goldpipeline.domain.errors import (
    PublisherAuthenticationError,
    PublisherConfigurationError,
    PublisherPermissionError,
    PublisherRateLimitError,
    PublisherRejectedError,
    PublisherTransportAmbiguousError,
)

logger = logging.getLogger(__name__)

TELEGRAM_PROVIDER = "telegram"
API_BASE = "https://api.telegram.org"

MAX_RETRY_AFTER_SECONDS = 300
"""Longest ``retry_after`` this publisher will honour.

Telegram occasionally returns very long flood-control windows. Waiting them out
inside a publish attempt means holding an uncommitted intent for minutes; beyond
this the attempt ends and a human decides when to try again.
"""


class TelegramPublisherClient:
    """Posts one message at a time via the Bot API."""

    def __init__(self, settings: TelegramSettings, *, http: Any | None = None) -> None:
        """Build a client.

        Args:
            settings: Token, destination and timeout. Never logged.
            http: A pre-built HTTP client. Injected by tests so every branch of
                the response and error handling below runs without a network.
        """
        self._settings = settings
        self._http = http if http is not None else _build_http_client(settings)

    @property
    def provider(self) -> str:
        return TELEGRAM_PROVIDER

    def send(self, request: SendRequest) -> SendOutcome:
        """Deliver one message and confirm it was posted."""
        logger.info(
            "publisher.send chunk=%d chars=%d target=%s",
            request.chunk_index,
            len(request.text),
            request.target_chat,
        )
        payload = {
            "chat_id": request.target_chat,
            "text": request.text,
            # Plain text: no parse_mode, and no preview card generated from a
            # URL the article happens to contain.
            "link_preview_options": {"is_disabled": True},
        }
        response = self._post(payload)
        return self._confirm(response)

    # -- transport ---------------------------------------------------------

    def _post(self, payload: dict[str, Any]) -> Any:
        """Issue the request.

        Every exception is replaced, never wrapped with its original message:
        the HTTP layer puts the token-bearing URL in the text it produces.
        """
        import httpx2

        url = f"{API_BASE}/bot{self._settings.bot_token}/sendMessage"
        try:
            return self._http.post(url, json=payload)
        except httpx2.TimeoutException as exc:
            raise PublisherTransportAmbiguousError(
                "the request to the provider timed out; the message may or may not "
                "have been delivered",
                reason="timeout",
            ) from _Scrubbed(exc)
        except httpx2.HTTPError as exc:
            raise PublisherTransportAmbiguousError(
                "the connection to the provider failed; the message may or may not "
                "have been delivered",
                reason="connection",
            ) from _Scrubbed(exc)
        except Exception as exc:  # noqa: BLE001 - anything here may carry the URL
            raise PublisherTransportAmbiguousError(
                "the request to the provider failed in an unexpected way; the message "
                "may or may not have been delivered",
                reason="unexpected",
            ) from _Scrubbed(exc)

    # -- response handling -------------------------------------------------

    def _confirm(self, response: Any) -> SendOutcome:
        """Turn a reply into a confirmed delivery, or the right kind of failure."""
        status = int(getattr(response, "status_code", 0) or 0)

        try:
            body = response.json()
        except Exception:  # noqa: BLE001 - a body that will not parse tells us nothing
            if status >= 400:
                raise self._explicit_failure(status, {}) from None
            raise PublisherTransportAmbiguousError(
                f"the provider replied with HTTP {status} but the body could not be "
                "parsed; delivery cannot be confirmed",
                reason="unparseable_body",
                status_code=status,
            ) from None

        if not isinstance(body, dict):
            raise PublisherTransportAmbiguousError(
                "the provider reply was not an object; delivery cannot be confirmed",
                reason="unexpected_body",
                status_code=status,
            )

        if status == 429 or body.get("error_code") == 429:
            raise PublisherRateLimitError(
                "the provider applied flood control",
                retry_after=_retry_after(body),
                status_code=429,
            )

        if status >= 400 or body.get("ok") is not True:
            raise self._explicit_failure(status, body)

        if status >= 500:  # pragma: no cover - covered by the branch above
            raise PublisherTransportAmbiguousError(
                f"the provider replied with HTTP {status}", reason="server_error"
            )

        result = body.get("result")
        if not isinstance(result, dict):
            raise PublisherTransportAmbiguousError(
                "the provider reported success without describing the message; "
                "delivery cannot be confirmed",
                reason="missing_result",
            )

        message_id = result.get("message_id")
        if not isinstance(message_id, int):
            raise PublisherTransportAmbiguousError(
                "the provider reported success without a message id; delivery cannot be confirmed",
                reason="missing_message_id",
            )

        chat = result.get("chat")
        chat_id = str(chat["id"]) if isinstance(chat, dict) and "id" in chat else None
        date = result.get("date")

        return SendOutcome(
            message_id=message_id,
            chat_id=chat_id,
            telegram_date=date if isinstance(date, int) else None,
        )

    @staticmethod
    def _explicit_failure(status: int, body: dict[str, Any]) -> Exception:
        """Map a refusal onto the taxonomy.

        A refusal is knowable: nothing was delivered. The description Telegram
        returns is not echoed - it can quote the request, and this text is
        stored and printed.
        """
        code = int(body.get("error_code") or status or 0)

        if code == 401:
            return PublisherAuthenticationError(
                "the provider rejected the bot token; check TELEGRAM_BOT_TOKEN",
                setting="TELEGRAM_BOT_TOKEN",
                status_code=code,
            )
        if code == 403:
            return PublisherPermissionError(
                "the bot may not post to the configured target. Add the bot to the "
                "channel and grant it permission to post messages.",
                status_code=code,
            )
        if code == 400:
            return PublisherRejectedError(
                "the provider rejected the request as invalid; check "
                "TELEGRAM_TARGET_CHAT_ID and the message content",
                status_code=code,
            )
        if code >= 500:
            return PublisherTransportAmbiguousError(
                f"the provider replied with HTTP {code}; delivery cannot be confirmed",
                reason="server_error",
                status_code=code,
            )
        return PublisherRejectedError(
            f"the provider refused the request (HTTP {code})", status_code=code
        )


class _Scrubbed(Exception):
    """A stand-in cause that carries no provider text.

    ``raise ... from exc`` would attach the original exception, and a traceback
    printed anywhere would then show the token-bearing URL. This preserves the
    chain's shape while dropping its contents.
    """

    def __init__(self, original: BaseException) -> None:
        super().__init__(f"{type(original).__name__} (details withheld)")


def _retry_after(body: dict[str, Any]) -> int:
    """Read the flood-control delay, clamped to something a human would wait.

    A missing or nonsensical value becomes a small default rather than an error:
    the response already told us the request was refused, which is the part that
    matters.
    """
    parameters = body.get("parameters")
    raw = parameters.get("retry_after") if isinstance(parameters, dict) else None
    if not isinstance(raw, int) or raw < 0:
        return 1
    return min(raw, MAX_RETRY_AFTER_SECONDS)


def _build_http_client(settings: TelegramSettings) -> Any:
    """Construct the HTTP client.

    Retries are disabled. A retried POST is how one approved article becomes two
    published messages - the transport must never make that decision, because
    only the publisher knows whether a resend is safe.
    """
    try:
        import httpx2
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise PublisherConfigurationError(
            "the httpx2 package is not installed; install it to publish",
            setting="httpx2",
        ) from exc

    return httpx2.Client(
        timeout=httpx2.Timeout(settings.timeout_seconds, connect=10.0),
        transport=httpx2.HTTPTransport(retries=0),
        follow_redirects=False,
    )


__all__ = ["API_BASE", "MAX_RETRY_AFTER_SECONDS", "TELEGRAM_PROVIDER", "TelegramPublisherClient"]
