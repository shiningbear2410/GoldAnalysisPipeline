"""Anthropic finalizer client, exercised entirely offline.

Thin by design: the SDK error mapping is shared with the writer, so what is
checked here is that this stage wires it up to the right error classes and the
right schema.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2 as httpx
import pytest
from conftest import FAKE_API_KEY

from goldpipeline.adapters.anthropic_finalizer import AnthropicFinalizerClient
from goldpipeline.adapters.finalizer_client import FinalizeRequest
from goldpipeline.config import FinalizerSettings
from goldpipeline.domain.errors import (
    FinalizeConfigurationError,
    FinalizeProviderError,
    FinalizeResponseError,
    FinalizeTimeoutError,
)
from goldpipeline.schemas.finalizer import (
    FinalizerModelOutput,
    FinalizerPrompt,
    IssueResolution,
    ResolutionStatus,
)

RUN_ID = "20260828_022701_a83f2c"

ARTICLE = "🕯 PHÂN TÍCH VÀNG\n\nGiá gần nhất quanh 3305.90, thị trường đang tích luỹ."


def settings() -> FinalizerSettings:
    return FinalizerSettings.from_env({"ANTHROPIC_API_KEY": FAKE_API_KEY})


def request() -> FinalizeRequest:
    prompt = FinalizerPrompt(
        system="# SYSTEM RULES\n# OUTPUT CONTRACT",
        user="# SOURCE OF TRUTH\n# ORIGINAL ARTICLE\n# REVIEW ISSUES",
        prompt_version="gold_finalizer_v1",
        nonce="deadbeefdeadbeef",
    )
    return FinalizeRequest(prompt=prompt, run_id=RUN_ID, max_tokens=4096)


def valid_output() -> FinalizerModelOutput:
    return FinalizerModelOutput(
        run_id=RUN_ID,
        article=ARTICLE,
        issue_resolutions=[
            IssueResolution(
                issue_id="i1",
                resolution=ResolutionStatus.APPLIED,
                description="Đã sửa giá gần nhất.",
            )
        ],
    )


class StubMessages:
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
) -> SimpleNamespace:
    return SimpleNamespace(
        parsed_output=parsed,
        stop_reason=stop_reason,
        model=model,
        usage=usage,
        _request_id="req_stub456",
    )


def _http_error(status: int) -> anthropic.APIStatusError:
    return anthropic.APIStatusError(
        "boom",
        response=httpx.Response(
            status_code=status, request=httpx.Request("POST", "https://api.anthropic.com/v1/x")
        ),
        body=None,
    )


# --- happy path -----------------------------------------------------------


def test_a_successful_call_returns_the_revision() -> None:
    stub = StubClient(response=sdk_response(parsed=valid_output()))
    response = AnthropicFinalizerClient(settings(), client=stub).finalize(request())

    assert response.provider == "anthropic"
    assert response.model == "claude-opus-5"
    assert response.output.run_id == RUN_ID
    assert response.output.article == ARTICLE


def test_the_two_turns_land_in_the_two_channels() -> None:
    stub = StubClient(response=sdk_response(parsed=valid_output()))
    AnthropicFinalizerClient(settings(), client=stub).finalize(request())

    call = stub.messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["max_tokens"] == 4096
    assert call["system"] == "# SYSTEM RULES\n# OUTPUT CONTRACT"
    assert call["messages"] == [
        {"role": "user", "content": "# SOURCE OF TRUTH\n# ORIGINAL ARTICLE\n# REVIEW ISSUES"}
    ]
    assert call["output_format"] is FinalizerModelOutput


def test_usage_metadata_is_captured() -> None:
    usage = SimpleNamespace(
        input_tokens=3100,
        output_tokens=520,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=None,
    )
    stub = StubClient(response=sdk_response(parsed=valid_output(), usage=usage))
    response = AnthropicFinalizerClient(settings(), client=stub).finalize(request())

    assert response.usage.input_tokens == 3100
    assert response.usage.output_tokens == 520
    assert response.usage.request_id == "req_stub456"
    assert response.usage.stop_reason == "end_turn"


# --- response failures ----------------------------------------------------


def test_no_structured_output_is_rejected() -> None:
    stub = StubClient(response=sdk_response(parsed=None))
    with pytest.raises(FinalizeResponseError, match="no structured output"):
        AnthropicFinalizerClient(settings(), client=stub).finalize(request())


def test_wrong_output_type_is_rejected() -> None:
    stub = StubClient(response=sdk_response(parsed={"article": "raw dict"}))
    with pytest.raises(FinalizeResponseError, match="unexpected output type"):
        AnthropicFinalizerClient(settings(), client=stub).finalize(request())


def test_a_truncated_revision_is_rejected() -> None:
    """A half-rewritten article would parse but read as an unfinished edit."""
    stub = StubClient(response=sdk_response(parsed=valid_output(), stop_reason="max_tokens"))
    with pytest.raises(FinalizeResponseError, match="token limit"):
        AnthropicFinalizerClient(settings(), client=stub).finalize(request())


def test_a_refusal_is_rejected() -> None:
    stub = StubClient(response=sdk_response(parsed=None, stop_reason="refusal"))
    with pytest.raises(FinalizeResponseError, match="declined"):
        AnthropicFinalizerClient(settings(), client=stub).finalize(request())


# --- provider failures ----------------------------------------------------


def test_timeout_maps_to_the_finalizer_timeout() -> None:
    stub = StubClient(
        error=anthropic.APITimeoutError(
            request=httpx.Request("POST", "https://api.anthropic.com/v1/x")
        )
    )
    with pytest.raises(FinalizeTimeoutError) as exc:
        AnthropicFinalizerClient(settings(), client=stub).finalize(request())
    assert exc.value.details["timeout_seconds"] == 120.0


def test_a_server_error_maps_to_a_provider_error() -> None:
    stub = StubClient(error=_http_error(503))
    with pytest.raises(FinalizeProviderError) as exc:
        AnthropicFinalizerClient(settings(), client=stub).finalize(request())
    assert exc.value.details["status_code"] == 503


def test_authentication_failure_is_a_configuration_error() -> None:
    stub = StubClient(
        error=anthropic.AuthenticationError(
            "bad key",
            response=httpx.Response(
                status_code=401,
                request=httpx.Request("POST", "https://api.anthropic.com/v1/x"),
            ),
            body=None,
        )
    )
    with pytest.raises(FinalizeConfigurationError) as exc:
        AnthropicFinalizerClient(settings(), client=stub).finalize(request())
    assert exc.value.details == {"setting": "ANTHROPIC_API_KEY"}


def test_an_unknown_model_names_the_finalizer_setting() -> None:
    """The message must point at the setting this stage actually reads."""
    stub = StubClient(
        error=anthropic.NotFoundError(
            "no such model",
            response=httpx.Response(
                status_code=404,
                request=httpx.Request("POST", "https://api.anthropic.com/v1/x"),
            ),
            body=None,
        )
    )
    with pytest.raises(FinalizeConfigurationError) as exc:
        AnthropicFinalizerClient(settings(), client=stub).finalize(request())
    assert exc.value.details == {"setting": "ANTHROPIC_FINALIZER_MODEL"}


# --- credential hygiene ---------------------------------------------------


def test_no_mapped_error_leaks_the_key() -> None:
    errors: list[Exception] = [
        anthropic.APITimeoutError(request=httpx.Request("POST", "https://x")),
        anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")),
        _http_error(500),
        _http_error(429),
    ]
    for error in errors:
        stub = StubClient(error=error)
        with pytest.raises(Exception) as exc:  # noqa: PT011 - inspecting the message
            AnthropicFinalizerClient(settings(), client=stub).finalize(request())
        assert FAKE_API_KEY not in f"{exc.value}{getattr(exc.value, 'details', '')}"


def test_the_provider_body_is_not_echoed() -> None:
    """A 500 now, since Round 9.3.1 routes deterministic 4xx elsewhere."""
    stub = StubClient(error=_http_error(500))
    with pytest.raises(FinalizeProviderError) as exc:
        AnthropicFinalizerClient(settings(), client=stub).finalize(request())
    assert str(exc.value) == "[FINALIZE_PROVIDER_ERROR] provider returned HTTP 500"
    assert "boom" not in str(exc.value)


def test_a_deterministic_400_is_a_configuration_error() -> None:
    """The shared mapping applies to every stage, not just the writer."""
    stub = StubClient(error=_http_error(400))
    with pytest.raises(FinalizeConfigurationError) as exc:
        AnthropicFinalizerClient(settings(), client=stub).finalize(request())
    assert exc.value.details["status_code"] == 400
    assert "boom" not in str(exc.value)
