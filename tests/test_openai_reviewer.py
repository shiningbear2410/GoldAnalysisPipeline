"""OpenAI reviewer client, exercised entirely offline.

The SDK client is injected, so every branch that maps a provider failure onto
the project's error taxonomy runs without a network, a key, or a bill.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx2 as httpx
import openai
import pytest
from conftest import FAKE_OPENAI_KEY

from goldpipeline.adapters.openai_reviewer import OpenAIReviewerClient
from goldpipeline.adapters.reviewer_client import ReviewRequest
from goldpipeline.config import ReviewerSettings
from goldpipeline.domain.errors import (
    ReviewConfigurationError,
    ReviewProviderError,
    ReviewResponseError,
    ReviewTimeoutError,
)
from goldpipeline.schemas.review import ReviewerPrompt, ReviewModelOutput, ReviewStatus

RUN_ID = "20260828_022701_a83f2c"


def settings() -> ReviewerSettings:
    return ReviewerSettings.from_env({"OPENAI_API_KEY": FAKE_OPENAI_KEY})


def request() -> ReviewRequest:
    prompt = ReviewerPrompt(
        system="# SYSTEM RULES\n# REVIEW RUBRIC\n# OUTPUT CONTRACT",
        user="# SOURCE OF TRUTH\n# ARTICLE UNDER REVIEW\n# DETERMINISTIC PRECHECK",
        prompt_version="gold_reviewer_v1",
        nonce="deadbeefdeadbeef",
    )
    return ReviewRequest(prompt=prompt, run_id=RUN_ID, max_output_tokens=4096)


def valid_output(**overrides: Any) -> ReviewModelOutput:
    fields: dict[str, Any] = {
        "run_id": RUN_ID,
        "status": ReviewStatus.PASS,
        "score": 94,
        "summary": "Bài viết bám sát dữ liệu nguồn.",
        "issues": [],
        "revision_instructions": [],
    }
    fields.update(overrides)
    return ReviewModelOutput(**fields)


class StubResponses:
    """Stands in for ``client.responses``."""

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
        self.responses = StubResponses(**kwargs)


def sdk_response(
    *,
    parsed: Any = None,
    status: str = "completed",
    usage: Any = None,
    model: str = "gpt-5.1",
    response_id: str = "resp_stub123",
    incomplete_details: Any = None,
    error: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        output_parsed=parsed,
        status=status,
        model=model,
        usage=usage,
        id=response_id,
        incomplete_details=incomplete_details,
        error=error,
    )


def _http_error(status: int) -> openai.APIStatusError:
    """Build a real SDK status error without touching the network."""
    return openai.APIStatusError(
        "boom",
        response=httpx.Response(
            status_code=status, request=httpx.Request("POST", "https://api.openai.com/v1/x")
        ),
        body=None,
    )


# --- happy path -----------------------------------------------------------


def test_successful_call_returns_the_parsed_output() -> None:
    stub = StubClient(response=sdk_response(parsed=valid_output()))
    response = OpenAIReviewerClient(settings(), client=stub).review(request())

    assert response.provider == "openai"
    assert response.model == "gpt-5.1"
    assert response.output.run_id == RUN_ID
    assert response.output.status is ReviewStatus.PASS


def test_the_two_turns_land_in_the_two_channels() -> None:
    """Rules go to `instructions`, Run data to `input`. Never concatenated."""
    stub = StubClient(response=sdk_response(parsed=valid_output()))
    OpenAIReviewerClient(settings(), client=stub).review(request())

    call = stub.responses.calls[0]
    assert call["model"] == "gpt-5.1"
    assert call["max_output_tokens"] == 4096
    assert call["instructions"] == "# SYSTEM RULES\n# REVIEW RUBRIC\n# OUTPUT CONTRACT"
    assert call["input"] == "# SOURCE OF TRUTH\n# ARTICLE UNDER REVIEW\n# DETERMINISTIC PRECHECK"
    assert call["text_format"] is ReviewModelOutput


def test_usage_metadata_is_captured() -> None:
    usage = SimpleNamespace(input_tokens=2400, output_tokens=380, total_tokens=2780)
    stub = StubClient(response=sdk_response(parsed=valid_output(), usage=usage))

    response = OpenAIReviewerClient(settings(), client=stub).review(request())

    assert response.usage.input_tokens == 2400
    assert response.usage.output_tokens == 380
    assert response.usage.total_tokens == 2780
    assert response.usage.request_id == "resp_stub123"
    assert response.usage.response_status == "completed"


def test_missing_usage_is_tolerated() -> None:
    stub = StubClient(response=sdk_response(parsed=valid_output(), usage=None))
    response = OpenAIReviewerClient(settings(), client=stub).review(request())
    assert response.usage.input_tokens is None


# --- response failures ----------------------------------------------------


def test_no_structured_output_is_rejected() -> None:
    """Requirement 27.17."""
    stub = StubClient(response=sdk_response(parsed=None))
    with pytest.raises(ReviewResponseError, match="no structured output"):
        OpenAIReviewerClient(settings(), client=stub).review(request())


def test_wrong_output_type_is_rejected() -> None:
    """No regex rescue: an answer of the wrong shape is a failure."""
    stub = StubClient(response=sdk_response(parsed={"status": "PASS"}))
    with pytest.raises(ReviewResponseError, match="unexpected output type"):
        OpenAIReviewerClient(settings(), client=stub).review(request())


def test_an_incomplete_response_is_rejected() -> None:
    stub = StubClient(
        response=sdk_response(
            parsed=valid_output(),
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        )
    )
    with pytest.raises(ReviewResponseError, match="incomplete"):
        OpenAIReviewerClient(settings(), client=stub).review(request())


def test_a_response_carrying_an_error_is_rejected() -> None:
    stub = StubClient(
        response=sdk_response(parsed=valid_output(), error=SimpleNamespace(code="oops"))
    )
    with pytest.raises(ReviewResponseError, match="reported an error"):
        OpenAIReviewerClient(settings(), client=stub).review(request())


def test_a_length_stop_is_a_response_error() -> None:
    """A truncated review would parse but silently omit findings."""
    # The SDK only reads `.usage` off the completion when building the message,
    # so a stand-in is enough to exercise the mapping without a real API object.
    completion = SimpleNamespace(usage=SimpleNamespace(total_tokens=4096))
    stub = StubClient(
        error=openai.LengthFinishReasonError(completion=completion)  # type: ignore[arg-type]
    )
    with pytest.raises(ReviewResponseError, match="token limit"):
        OpenAIReviewerClient(settings(), client=stub).review(request())


def test_a_content_filter_stop_is_a_response_error() -> None:
    stub = StubClient(error=openai.ContentFilterFinishReasonError())
    with pytest.raises(ReviewResponseError, match="content filter"):
        OpenAIReviewerClient(settings(), client=stub).review(request())


# --- provider failures ----------------------------------------------------


def test_timeout_maps_to_review_timeout() -> None:
    stub = StubClient(
        error=openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/x"))
    )
    with pytest.raises(ReviewTimeoutError) as exc:
        OpenAIReviewerClient(settings(), client=stub).review(request())
    assert exc.value.details["timeout_seconds"] == 120.0


def test_connection_failure_maps_to_provider_error() -> None:
    stub = StubClient(
        error=openai.APIConnectionError(
            request=httpx.Request("POST", "https://api.openai.com/v1/x")
        )
    )
    with pytest.raises(ReviewProviderError, match="could not reach"):
        OpenAIReviewerClient(settings(), client=stub).review(request())


def test_server_error_maps_to_provider_error() -> None:
    stub = StubClient(error=_http_error(503))
    with pytest.raises(ReviewProviderError) as exc:
        OpenAIReviewerClient(settings(), client=stub).review(request())
    assert exc.value.details["status_code"] == 503


def test_rate_limit_maps_to_provider_error() -> None:
    stub = StubClient(
        error=openai.RateLimitError(
            "slow down",
            response=httpx.Response(
                status_code=429,
                request=httpx.Request("POST", "https://api.openai.com/v1/x"),
            ),
            body=None,
        )
    )
    with pytest.raises(ReviewProviderError) as exc:
        OpenAIReviewerClient(settings(), client=stub).review(request())
    assert exc.value.details["status_code"] == 429


def test_authentication_failure_is_a_configuration_error() -> None:
    """Not retried: the same key will be rejected again."""
    stub = StubClient(
        error=openai.AuthenticationError(
            "bad key",
            response=httpx.Response(
                status_code=401,
                request=httpx.Request("POST", "https://api.openai.com/v1/x"),
            ),
            body=None,
        )
    )
    with pytest.raises(ReviewConfigurationError) as exc:
        OpenAIReviewerClient(settings(), client=stub).review(request())
    assert exc.value.details == {"setting": "OPENAI_API_KEY"}


def test_unknown_model_is_a_configuration_error() -> None:
    stub = StubClient(
        error=openai.NotFoundError(
            "no such model",
            response=httpx.Response(
                status_code=404,
                request=httpx.Request("POST", "https://api.openai.com/v1/x"),
            ),
            body=None,
        )
    )
    with pytest.raises(ReviewConfigurationError) as exc:
        OpenAIReviewerClient(settings(), client=stub).review(request())
    assert exc.value.details == {"setting": "OPENAI_REVIEW_MODEL"}


# --- credential hygiene ---------------------------------------------------


def test_no_mapped_error_leaks_the_key() -> None:
    """Requirement 27.26, at the boundary that actually holds the credential."""
    errors: list[Exception] = [
        openai.APITimeoutError(request=httpx.Request("POST", "https://x")),
        openai.APIConnectionError(request=httpx.Request("POST", "https://x")),
        _http_error(500),
        _http_error(400),
    ]
    for error in errors:
        stub = StubClient(error=error)
        with pytest.raises(Exception) as exc:  # noqa: PT011 - inspecting the message
            OpenAIReviewerClient(settings(), client=stub).review(request())
        rendered = f"{exc.value}{getattr(exc.value, 'details', '')}"
        assert FAKE_OPENAI_KEY not in rendered


def test_provider_body_is_not_echoed_into_the_message() -> None:
    """Provider payloads can quote the request; this message reaches the manifest."""
    stub = StubClient(error=_http_error(400))
    with pytest.raises(ReviewProviderError) as exc:
        OpenAIReviewerClient(settings(), client=stub).review(request())
    assert str(exc.value) == "[REVIEW_PROVIDER_ERROR] provider returned HTTP 400"
