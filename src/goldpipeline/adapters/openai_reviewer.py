"""OpenAI implementation of :class:`ReviewerClient`.

The only module that imports the OpenAI SDK, and the only one that holds an
``OPENAI_API_KEY``.

Two deliberate choices:

* **The Responses API with a schema.** ``responses.parse`` takes the system
  rules as ``instructions`` and the Run's data as ``input`` - the same two-channel
  split the writer uses - and constrains the answer to
  :class:`ReviewModelOutput` via ``text_format``. There is no free-form JSON to
  fish out afterwards, and no regex rescue when the shape is wrong: a malformed
  answer is a failure, because a review assembled from fragments is worse than
  no review.
* **Nothing from the provider reaches the manifest.** SDK exceptions can carry
  request context, so each is caught and re-raised with a message written here.
"""

from __future__ import annotations

import logging
from typing import Any

from goldpipeline.adapters.reviewer_client import ReviewRequest, ReviewResponse
from goldpipeline.config import ReviewerSettings
from goldpipeline.domain.errors import (
    ReviewConfigurationError,
    ReviewProviderError,
    ReviewResponseError,
    ReviewTimeoutError,
)
from goldpipeline.schemas.review import ReviewModelOutput, ReviewUsage

logger = logging.getLogger(__name__)

OPENAI_PROVIDER = "openai"


class OpenAIReviewerClient:
    """Calls an OpenAI model to audit a draft."""

    def __init__(self, settings: ReviewerSettings, *, client: Any | None = None) -> None:
        """Build a client.

        Args:
            settings: Credentials and tuning. Never logged, never stored on a Run.
            client: A pre-built SDK client. Injected by tests so the parsing and
                error-mapping paths below can be exercised without a network.
        """
        self._settings = settings
        self._client = client if client is not None else _build_sdk_client(settings)

    @property
    def provider(self) -> str:
        return OPENAI_PROVIDER

    @property
    def model(self) -> str:
        return self._settings.model

    def review(self, request: ReviewRequest) -> ReviewResponse:
        """Call the model and return its parsed review."""
        logger.info(
            "reviewer.provider call model=%s max_output_tokens=%d timeout=%.0fs",
            self._settings.model,
            request.max_output_tokens,
            self._settings.timeout_seconds,
        )

        response = self._call(request)
        output = self._extract_output(response)

        return ReviewResponse(
            output=output,
            model=getattr(response, "model", self._settings.model) or self._settings.model,
            provider=self.provider,
            usage=_usage_from(response),
        )

    # -- provider call -----------------------------------------------------

    def _call(self, request: ReviewRequest) -> Any:
        """Issue the request, mapping SDK failures onto the project taxonomy."""
        import openai

        try:
            return self._client.responses.parse(
                model=self._settings.model,
                instructions=request.prompt.system,
                input=request.prompt.user,
                max_output_tokens=request.max_output_tokens,
                text_format=ReviewModelOutput,
            )
        except openai.APITimeoutError as exc:
            raise ReviewTimeoutError(
                f"provider did not respond within {self._settings.timeout_seconds:g}s",
                timeout_seconds=self._settings.timeout_seconds,
            ) from exc
        except openai.AuthenticationError as exc:
            # Not retried: a rejected credential will be rejected again.
            raise ReviewConfigurationError(
                "provider rejected the credentials; check OPENAI_API_KEY",
                setting="OPENAI_API_KEY",
            ) from exc
        except openai.PermissionDeniedError as exc:
            raise ReviewConfigurationError(
                "credentials lack permission for this model", setting="OPENAI_REVIEW_MODEL"
            ) from exc
        except openai.NotFoundError as exc:
            raise ReviewConfigurationError(
                f"provider does not recognise model {self._settings.model!r}",
                setting="OPENAI_REVIEW_MODEL",
            ) from exc
        except openai.LengthFinishReasonError as exc:
            raise ReviewResponseError(
                "response hit the output token limit before completing"
            ) from exc
        except openai.ContentFilterFinishReasonError as exc:
            raise ReviewResponseError("provider content filter stopped the review") from exc
        except openai.RateLimitError as exc:
            raise ReviewProviderError("provider rate limit reached", status_code=429) from exc
        except openai.APIStatusError as exc:
            # Deliberately not echoing the provider body: it can quote the
            # request back, and this message is stored in the manifest.
            raise ReviewProviderError(
                f"provider returned HTTP {exc.status_code}", status_code=exc.status_code
            ) from exc
        except openai.APIConnectionError as exc:
            raise ReviewProviderError("could not reach the provider") from exc

    # -- response handling -------------------------------------------------

    @staticmethod
    def _extract_output(response: Any) -> ReviewModelOutput:
        """Pull the validated model output out of an SDK response."""
        status = getattr(response, "status", None)
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", None) or "unknown"
            raise ReviewResponseError(
                f"provider returned an incomplete response ({reason})", reason=str(reason)
            )
        if getattr(response, "error", None):
            raise ReviewResponseError("provider reported an error on the response")

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise ReviewResponseError("provider returned no structured output")
        if not isinstance(parsed, ReviewModelOutput):
            raise ReviewResponseError(
                f"provider returned unexpected output type {type(parsed).__name__}"
            )
        return parsed


def _usage_from(response: Any) -> ReviewUsage:
    """Collect safe usage metadata. Never headers, never credentials."""
    usage = getattr(response, "usage", None)
    return ReviewUsage(
        input_tokens=_int_or_none(getattr(usage, "input_tokens", None)),
        output_tokens=_int_or_none(getattr(usage, "output_tokens", None)),
        total_tokens=_int_or_none(getattr(usage, "total_tokens", None)),
        request_id=_str_or_none(getattr(response, "id", None)),
        response_status=_str_or_none(getattr(response, "status", None)),
    )


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _str_or_none(value: Any) -> str | None:
    return value[:200] if isinstance(value, str) and value else None


def _build_sdk_client(settings: ReviewerSettings) -> Any:
    """Construct the real SDK client.

    Imported lazily so the rest of the pipeline - and the whole offline test
    suite - does not require the SDK to be loaded.
    """
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ReviewConfigurationError(
            "the openai package is not installed; install it to use the real reviewer",
            setting="openai",
        ) from exc

    return openai.OpenAI(
        api_key=settings.api_key,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )


__all__ = ["OPENAI_PROVIDER", "OpenAIReviewerClient"]
