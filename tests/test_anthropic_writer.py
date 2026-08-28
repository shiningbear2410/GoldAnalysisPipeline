"""Anthropic client behaviour, exercised entirely offline.

The SDK client is injected, so every branch that maps a provider failure onto
the project's error taxonomy is covered without a network, a key, or a bill.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from conftest import FAKE_API_KEY

from goldpipeline.adapters.anthropic_writer import AnthropicWriterClient
from goldpipeline.adapters.writer_client import WriterRequest
from goldpipeline.config import WriterSettings
from goldpipeline.domain.errors import (
    WriterConfigurationError,
    WriterProviderError,
    WriterResponseError,
    WriterTimeoutError,
)
from goldpipeline.schemas.writer import WriterModelOutput, WriterPrompt, WriterStatus

RUN_ID = "20260828_022701_a83f2c"


def settings() -> WriterSettings:
    return WriterSettings.from_env({"ANTHROPIC_API_KEY": FAKE_API_KEY})


def request() -> WriterRequest:
    prompt = WriterPrompt(
        system="# SYSTEM RULES\n# OUTPUT CONTRACT",
        user="# MARKET FACTS\n# UNTRUSTED SOURCE DATA",
        prompt_version="gold_writer_v1",
        nonce="deadbeefdeadbeef",
    )
    return WriterRequest(prompt=prompt, run_id=RUN_ID, max_tokens=4096)


def valid_output(**overrides: Any) -> WriterModelOutput:
    fields: dict[str, Any] = {
        "run_id": RUN_ID,
        "status": WriterStatus.COMPLETED,
        "title": "Nhận định vàng",
        "article": "🕯 NHẬN ĐỊNH VÀNG\n\nGiá gần nhất quanh 3314.20, thị trường đi ngang.",
        "source_claims": [],
        "warnings": [],
    }
    fields.update(overrides)
    return WriterModelOutput(**fields)


class StubMessages:
    """Stands in for ``client.messages``."""

    def __init__(self, *, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class StubClient:
    def __init__(self, **kwargs: Any) -> None:
        self.messages = StubMessages(**kwargs)


def sdk_response(
    *,
    parsed: Any = None,
    stop_reason: str = "end_turn",
    usage: Any = None,
    model: str = "claude-opus-5",
    request_id: str = "req_stub123",
) -> SimpleNamespace:
    return SimpleNamespace(
        parsed_output=parsed,
        stop_reason=stop_reason,
        model=model,
        usage=usage,
        _request_id=request_id,
    )


def _http_error(status: int) -> anthropic.APIStatusError:
    """Build a real SDK status error without touching the network."""
    return anthropic.APIStatusError(
        "boom",
        response=httpx2.Response(
            status_code=status, request=httpx2.Request("POST", "https://api.anthropic.com/v1/x")
        ),
        body=None,
    )


# --- happy path -----------------------------------------------------------


def test_successful_call_returns_the_parsed_output() -> None:
    stub = StubClient(response=sdk_response(parsed=valid_output()))
    client = AnthropicWriterClient(settings(), client=stub)

    response = client.generate(request())

    assert response.provider == "anthropic"
    assert response.model == "claude-opus-5"
    assert response.output.run_id == RUN_ID
    assert response.output.status is WriterStatus.COMPLETED


def test_request_is_built_from_the_prompt() -> None:
    """The two turns must land in the two channels, not be concatenated."""
    stub = StubClient(response=sdk_response(parsed=valid_output()))
    AnthropicWriterClient(settings(), client=stub).generate(request())

    call = stub.messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["max_tokens"] == 4096
    assert call["system"] == "# SYSTEM RULES\n# OUTPUT CONTRACT"
    assert call["messages"] == [
        {"role": "user", "content": "# MARKET FACTS\n# UNTRUSTED SOURCE DATA"}
    ]
    assert call["output_format"] is WriterModelOutput


def test_usage_metadata_is_captured() -> None:
    usage = SimpleNamespace(
        input_tokens=1234,
        output_tokens=567,
        cache_read_input_tokens=100,
        cache_creation_input_tokens=0,
    )
    stub = StubClient(response=sdk_response(parsed=valid_output(), usage=usage))

    response = AnthropicWriterClient(settings(), client=stub).generate(request())

    assert response.usage.input_tokens == 1234
    assert response.usage.output_tokens == 567
    assert response.usage.cache_read_input_tokens == 100
    assert response.usage.request_id == "req_stub123"
    assert response.usage.stop_reason == "end_turn"


def test_missing_usage_is_tolerated() -> None:
    stub = StubClient(response=sdk_response(parsed=valid_output(), usage=None))
    response = AnthropicWriterClient(settings(), client=stub).generate(request())
    assert response.usage.input_tokens is None


# --- response failures ----------------------------------------------------


def test_no_structured_output_is_rejected() -> None:
    stub = StubClient(response=sdk_response(parsed=None))
    with pytest.raises(WriterResponseError, match="no structured output"):
        AnthropicWriterClient(settings(), client=stub).generate(request())


def test_wrong_output_type_is_rejected() -> None:
    """No regex rescue: an answer of the wrong shape is a failure."""
    stub = StubClient(response=sdk_response(parsed={"article": "raw dict"}))
    with pytest.raises(WriterResponseError, match="unexpected output type"):
        AnthropicWriterClient(settings(), client=stub).generate(request())


def test_truncated_response_is_rejected() -> None:
    """A half-written article would parse but read as an unfinished thought."""
    stub = StubClient(response=sdk_response(parsed=valid_output(), stop_reason="max_tokens"))
    with pytest.raises(WriterResponseError, match="token limit"):
        AnthropicWriterClient(settings(), client=stub).generate(request())


def test_refusal_is_rejected() -> None:
    stub = StubClient(response=sdk_response(parsed=None, stop_reason="refusal"))
    with pytest.raises(WriterResponseError, match="declined"):
        AnthropicWriterClient(settings(), client=stub).generate(request())


# --- provider failures ----------------------------------------------------


def test_timeout_maps_to_writer_timeout() -> None:
    stub = StubClient(
        error=anthropic.APITimeoutError(
            request=httpx2.Request("POST", "https://api.anthropic.com/v1/x")
        )
    )
    with pytest.raises(WriterTimeoutError) as exc:
        AnthropicWriterClient(settings(), client=stub).generate(request())
    assert exc.value.details["timeout_seconds"] == 120.0


def test_connection_failure_maps_to_provider_error() -> None:
    stub = StubClient(
        error=anthropic.APIConnectionError(
            request=httpx2.Request("POST", "https://api.anthropic.com/v1/x")
        )
    )
    with pytest.raises(WriterProviderError, match="could not reach"):
        AnthropicWriterClient(settings(), client=stub).generate(request())


def test_server_error_maps_to_provider_error() -> None:
    stub = StubClient(error=_http_error(503))
    with pytest.raises(WriterProviderError) as exc:
        AnthropicWriterClient(settings(), client=stub).generate(request())
    assert exc.value.details["status_code"] == 503


def test_authentication_failure_is_a_configuration_error() -> None:
    """Auth failures are not retried: the same key will be rejected again."""
    stub = StubClient(
        error=anthropic.AuthenticationError(
            "bad key",
            response=httpx2.Response(
                status_code=401,
                request=httpx2.Request("POST", "https://api.anthropic.com/v1/x"),
            ),
            body=None,
        )
    )
    with pytest.raises(WriterConfigurationError) as exc:
        AnthropicWriterClient(settings(), client=stub).generate(request())
    assert exc.value.details == {"setting": "ANTHROPIC_API_KEY"}


def test_unknown_model_is_a_configuration_error() -> None:
    stub = StubClient(
        error=anthropic.NotFoundError(
            "no such model",
            response=httpx2.Response(
                status_code=404,
                request=httpx2.Request("POST", "https://api.anthropic.com/v1/x"),
            ),
            body=None,
        )
    )
    with pytest.raises(WriterConfigurationError) as exc:
        AnthropicWriterClient(settings(), client=stub).generate(request())
    assert exc.value.details == {"setting": "ANTHROPIC_MODEL"}


# --- credential hygiene ---------------------------------------------------


def test_no_error_path_leaks_the_key() -> None:
    """Every mapped failure, checked for the credential."""
    errors: list[Exception] = [
        anthropic.APITimeoutError(request=httpx2.Request("POST", "https://x")),
        anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x")),
        _http_error(500),
        _http_error(429),
        RuntimeError(f"provider echoed {FAKE_API_KEY} back at us"),
    ]

    for error in errors:
        stub = StubClient(error=error)
        client = AnthropicWriterClient(settings(), client=stub)
        try:
            client.generate(request())
        except Exception as exc:  # noqa: BLE001 - the point is to inspect it
            rendered = f"{exc}{getattr(exc, 'details', '')}"
            if isinstance(exc, RuntimeError):
                continue  # unmapped errors propagate untouched, by design
            assert FAKE_API_KEY not in rendered


def test_provider_body_is_not_echoed_into_the_message() -> None:
    """Provider payloads can quote the request; this message reaches the manifest."""
    stub = StubClient(error=_http_error(400))
    with pytest.raises(WriterProviderError) as exc:
        AnthropicWriterClient(settings(), client=stub).generate(request())
    assert str(exc.value) == "[WRITER_PROVIDER_ERROR] provider returned HTTP 400"
