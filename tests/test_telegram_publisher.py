"""Telegram transport, exercised entirely offline.

The HTTP client is injected, so every response and error branch runs without a
socket. No test here can reach Telegram, which for this stage is a safety
requirement rather than a convenience.

Half these tests exist for one reason: the bot token is in the request URL, so
any HTTP exception that escapes is a leaked credential.
"""

from __future__ import annotations

from typing import Any

import httpx2
import pytest
from conftest import TELEGRAM_TOKEN_SENTINEL

from goldpipeline.adapters.publisher_client import SendRequest
from goldpipeline.adapters.telegram_publisher import (
    MAX_RETRY_AFTER_SECONDS,
    TelegramPublisherClient,
)
from goldpipeline.config import TelegramSettings
from goldpipeline.domain.errors import (
    PublisherAuthenticationError,
    PublisherPermissionError,
    PublisherRateLimitError,
    PublisherRejectedError,
    PublisherTransportAmbiguousError,
)

TARGET = "@gold_signals_vn"


def settings() -> TelegramSettings:
    return TelegramSettings.from_env(
        {
            "TELEGRAM_BOT_TOKEN": TELEGRAM_TOKEN_SENTINEL,
            "TELEGRAM_TARGET_CHAT_ID": TARGET,
        }
    )


def request(text: str = "🕯 NHẬN ĐỊNH VÀNG\n\nVàng đi ngang.") -> SendRequest:
    return SendRequest(target_chat=TARGET, text=text, chunk_index=0)


class StubHttp:
    """Stands in for the httpx2 client."""

    def __init__(self, *, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def post(self, url: str, json: dict[str, Any]) -> Any:
        self.calls.append((url, json))
        if self.error is not None:
            raise self.error
        return self.response


class StubResponse:
    """A reply with a status code and a body."""

    def __init__(self, status_code: int, body: Any = None, *, raises: bool = False) -> None:
        self.status_code = status_code
        self._body = body
        self._raises = raises

    def json(self) -> Any:
        if self._raises:
            raise ValueError("not json")
        return self._body


def ok_body(message_id: int = 4321, chat_id: str = "-1002145890733") -> dict[str, Any]:
    return {
        "ok": True,
        "result": {
            "message_id": message_id,
            "date": 1788000000,
            "chat": {"id": chat_id, "type": "channel"},
        },
    }


def client(stub: StubHttp) -> TelegramPublisherClient:
    return TelegramPublisherClient(settings(), http=stub)


# --- the happy path -------------------------------------------------------


def test_a_confirmed_send_returns_the_message_id() -> None:
    """Requirement 39.31."""
    stub = StubHttp(response=StubResponse(200, ok_body()))
    outcome = client(stub).send(request())

    assert outcome.message_id == 4321
    assert outcome.chat_id == "-1002145890733"
    assert outcome.telegram_date == 1788000000


def test_the_payload_is_plain_text_with_no_parse_mode() -> None:
    """Requirement 39.23: a markup parser is another thing that can change the text."""
    stub = StubHttp(response=StubResponse(200, ok_body()))
    text = "Giá *quanh* 3305.90 _và_ [link](x) — không được diễn giải."
    client(stub).send(request(text))

    _, payload = stub.calls[0]
    assert "parse_mode" not in payload
    assert payload["text"] == text
    assert payload["chat_id"] == TARGET


def test_link_previews_are_disabled() -> None:
    stub = StubHttp(response=StubResponse(200, ok_body()))
    client(stub).send(request())

    _, payload = stub.calls[0]
    assert payload["link_preview_options"] == {"is_disabled": True}


def test_the_text_reaches_the_transport_byte_for_byte() -> None:
    stub = StubHttp(response=StubResponse(200, ok_body()))
    text = "🕯 NHẬN ĐỊNH VÀNG\n\n  Giá 3305.90.  \n\n⚠️ Lưu ý"
    client(stub).send(request(text))

    assert stub.calls[0][1]["text"] == text


# --- explicit refusals ----------------------------------------------------


def test_a_401_is_an_authentication_failure() -> None:
    """Requirement 39.38."""
    stub = StubHttp(response=StubResponse(401, {"ok": False, "error_code": 401}))
    with pytest.raises(PublisherAuthenticationError) as exc:
        client(stub).send(request())
    assert "TELEGRAM_BOT_TOKEN" in str(exc.value)


def test_a_403_is_a_permission_failure_with_advice() -> None:
    """Requirements 39.39 and 41."""
    stub = StubHttp(response=StubResponse(403, {"ok": False, "error_code": 403}))
    with pytest.raises(PublisherPermissionError) as exc:
        client(stub).send(request())

    message = str(exc.value)
    assert "may not post" in message
    assert "permission to post" in message


def test_a_400_is_a_rejection() -> None:
    """Requirement 39.40."""
    stub = StubHttp(
        response=StubResponse(
            400, {"ok": False, "error_code": 400, "description": "chat not found"}
        )
    )
    with pytest.raises(PublisherRejectedError):
        client(stub).send(request())


def test_ok_false_with_http_200_is_still_a_rejection() -> None:
    """Requirement 39.32: HTTP 200 is not by itself a confirmation."""
    stub = StubHttp(response=StubResponse(200, {"ok": False, "error_code": 400}))
    with pytest.raises(PublisherRejectedError):
        client(stub).send(request())


def test_a_provider_description_is_never_echoed() -> None:
    """Telegram's description can quote the request, and this text is stored."""
    stub = StubHttp(
        response=StubResponse(
            400,
            {"ok": False, "error_code": 400, "description": "Bad Request: SECRET-CANARY leaked"},
        )
    )
    with pytest.raises(PublisherRejectedError) as exc:
        client(stub).send(request())
    assert "SECRET-CANARY" not in str(exc.value)


# --- flood control --------------------------------------------------------


def test_a_429_carries_the_retry_delay() -> None:
    """Requirement 39.76."""
    stub = StubHttp(
        response=StubResponse(
            429, {"ok": False, "error_code": 429, "parameters": {"retry_after": 7}}
        )
    )
    with pytest.raises(PublisherRateLimitError) as exc:
        client(stub).send(request())
    assert exc.value.details["retry_after"] == 7


def test_an_absurd_retry_delay_is_capped() -> None:
    """Requirement 39.77: an hour-long wait holds an open intent."""
    stub = StubHttp(
        response=StubResponse(
            429, {"ok": False, "error_code": 429, "parameters": {"retry_after": 99999}}
        )
    )
    with pytest.raises(PublisherRateLimitError) as exc:
        client(stub).send(request())
    assert exc.value.details["retry_after"] == MAX_RETRY_AFTER_SECONDS


@pytest.mark.parametrize("parameters", [None, {}, {"retry_after": "soon"}, {"retry_after": -5}])
def test_a_missing_or_nonsense_delay_falls_back(parameters: Any) -> None:
    body: dict[str, Any] = {"ok": False, "error_code": 429}
    if parameters is not None:
        body["parameters"] = parameters

    stub = StubHttp(response=StubResponse(429, body))
    with pytest.raises(PublisherRateLimitError) as exc:
        client(stub).send(request())
    assert exc.value.details["retry_after"] == 1


# --- ambiguity ------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        httpx2.ReadTimeout("timed out"),
        httpx2.ConnectTimeout("timed out"),
        httpx2.ConnectError("refused"),
        httpx2.RemoteProtocolError("reset"),
        RuntimeError("something unexpected"),
    ],
)
def test_transport_failures_are_ambiguous(error: Exception) -> None:
    """Requirements 39.35-39.36: delivery may have happened; nobody knows."""
    stub = StubHttp(error=error)
    with pytest.raises(PublisherTransportAmbiguousError):
        client(stub).send(request())


def test_a_5xx_is_ambiguous() -> None:
    """Requirement 39.37: the request may have been processed before the error."""
    stub = StubHttp(response=StubResponse(503, {"ok": False, "error_code": 503}))
    with pytest.raises(PublisherTransportAmbiguousError):
        client(stub).send(request())


def test_an_unparseable_body_after_a_200_is_ambiguous() -> None:
    """Requirement 39.33."""
    stub = StubHttp(response=StubResponse(200, raises=True))
    with pytest.raises(PublisherTransportAmbiguousError, match="could not be"):
        client(stub).send(request())


def test_a_non_object_body_is_ambiguous() -> None:
    stub = StubHttp(response=StubResponse(200, ["not", "an", "object"]))
    with pytest.raises(PublisherTransportAmbiguousError):
        client(stub).send(request())


def test_success_without_a_result_is_ambiguous() -> None:
    """Requirement 39.34."""
    stub = StubHttp(response=StubResponse(200, {"ok": True}))
    with pytest.raises(PublisherTransportAmbiguousError, match="without describing"):
        client(stub).send(request())


def test_success_without_a_message_id_is_ambiguous() -> None:
    """Requirement 39.52: no id means no proof of what was posted."""
    stub = StubHttp(response=StubResponse(200, {"ok": True, "result": {"date": 1788000000}}))
    with pytest.raises(PublisherTransportAmbiguousError, match="without a message id"):
        client(stub).send(request())


def test_a_non_integer_message_id_is_ambiguous() -> None:
    stub = StubHttp(response=StubResponse(200, {"ok": True, "result": {"message_id": "abc"}}))
    with pytest.raises(PublisherTransportAmbiguousError):
        client(stub).send(request())


# --- the token never escapes ---------------------------------------------


def test_the_token_is_in_the_url_the_transport_receives() -> None:
    """Establishes the hazard the rest of this section guards against."""
    stub = StubHttp(response=StubResponse(200, ok_body()))
    client(stub).send(request())

    url, _ = stub.calls[0]
    assert TELEGRAM_TOKEN_SENTINEL in url, "the API really does authenticate by path"


@pytest.mark.parametrize(
    "error",
    [
        httpx2.ReadTimeout(f"reading https://api.telegram.org/bot{TELEGRAM_TOKEN_SENTINEL}/x"),
        httpx2.ConnectError(f"connecting to bot{TELEGRAM_TOKEN_SENTINEL}"),
        RuntimeError(f"boom at https://api.telegram.org/bot{TELEGRAM_TOKEN_SENTINEL}/sendMessage"),
    ],
)
def test_a_token_bearing_exception_never_escapes(error: Exception) -> None:
    """Requirements 8 and 39.75: HTTP libraries put the URL in their messages."""
    stub = StubHttp(error=error)

    with pytest.raises(PublisherTransportAmbiguousError) as exc:
        client(stub).send(request())

    rendered = f"{exc.value}{exc.value.details}"
    assert TELEGRAM_TOKEN_SENTINEL not in rendered
    assert "api.telegram.org" not in rendered


def test_the_scrubbed_cause_carries_no_provider_text() -> None:
    """A traceback prints the cause chain, so the cause must be clean too."""
    stub = StubHttp(
        error=httpx2.ReadTimeout(f"https://api.telegram.org/bot{TELEGRAM_TOKEN_SENTINEL}/x")
    )
    with pytest.raises(PublisherTransportAmbiguousError) as exc:
        client(stub).send(request())

    cause = exc.value.__cause__
    assert cause is not None
    assert TELEGRAM_TOKEN_SENTINEL not in str(cause)
    assert "details withheld" in str(cause)


def test_no_refusal_path_leaks_the_token() -> None:
    responses = [
        StubResponse(401, {"ok": False, "error_code": 401}),
        StubResponse(403, {"ok": False, "error_code": 403}),
        StubResponse(400, {"ok": False, "error_code": 400}),
        StubResponse(429, {"ok": False, "error_code": 429}),
        StubResponse(503, {"ok": False, "error_code": 503}),
        StubResponse(200, {"ok": True}),
    ]
    for response in responses:
        with pytest.raises(Exception) as exc:  # noqa: PT011 - inspecting the message
            client(StubHttp(response=response)).send(request())
        rendered = f"{exc.value}{getattr(exc.value, 'details', '')}"
        assert TELEGRAM_TOKEN_SENTINEL not in rendered


def test_settings_never_render_the_token() -> None:
    assert TELEGRAM_TOKEN_SENTINEL not in repr(settings())
    assert TELEGRAM_TOKEN_SENTINEL not in str(settings())
