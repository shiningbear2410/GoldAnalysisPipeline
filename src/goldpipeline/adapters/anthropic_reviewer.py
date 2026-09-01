"""Anthropic implementation of :class:`ReviewerClient`.

The production reviewer since Round 9.3.1. Shares the SDK error mapping with the
writer and the finalizer through :mod:`goldpipeline.adapters.anthropic_errors`,
so all three stages agree about what a 400 or a 429 means - which matters more
than usual here, because that shared mapping is what keeps a deterministic
rejection out of the automation layer's retry loop.

**Same vendor, still an independent judgement.** Writer and Reviewer now call
the same account, and it would be easy to mistake that for a reason to merge
them. It is not. The reviewer receives the immutable context, the finished draft
and the writer's metadata, and answers a *different* prompt with a *different*
schema in a *separate* request. Nothing about the writer's reasoning is in
scope; the review has to be able to disagree with it, and a model asked to
critique its own answer in one call reliably will not. The independence lives in
the request boundary, not in the billing account.

The artifact this stage produces is still ``gpt_review.json``. The name is
historical and deliberately unchanged: it is written into manifests and digest
chains of Runs that already exist, and renaming it would invalidate their
provenance to buy nothing but tidiness.
"""

from __future__ import annotations

import logging
from typing import Any

from goldpipeline.adapters.anthropic_errors import (
    AnthropicErrorMap,
    build_sdk_client,
    check_stop_reason,
    raise_mapped,
    usage_fields,
)
from goldpipeline.adapters.reviewer_client import ReviewRequest, ReviewResponse
from goldpipeline.config import API_KEY_ENV, REVIEWER_MODEL_ENV, ReviewerSettings
from goldpipeline.domain.errors import (
    ReviewConfigurationError,
    ReviewProviderError,
    ReviewResponseError,
    ReviewTimeoutError,
)
from goldpipeline.schemas.review import ReviewModelOutput, ReviewUsage

logger = logging.getLogger(__name__)

ANTHROPIC_PROVIDER = "anthropic"

REVIEWER_ERRORS = AnthropicErrorMap(
    timeout=ReviewTimeoutError,
    provider=ReviewProviderError,
    configuration=ReviewConfigurationError,
    response=ReviewResponseError,
    api_key_setting=API_KEY_ENV,
    model_setting=REVIEWER_MODEL_ENV,
)


class AnthropicReviewerClient:
    """Calls Claude to audit a draft."""

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
        return ANTHROPIC_PROVIDER

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
        import anthropic

        try:
            return self._client.messages.parse(
                model=self._settings.model,
                max_tokens=request.max_output_tokens,
                system=request.prompt.system,
                messages=[{"role": "user", "content": request.prompt.user}],
                output_format=ReviewModelOutput,
            )
        except anthropic.AnthropicError as exc:
            raise_mapped(
                exc,
                errors=REVIEWER_ERRORS,
                timeout_seconds=self._settings.timeout_seconds,
                model=self._settings.model,
            )

    # -- response handling -------------------------------------------------

    @staticmethod
    def _extract_output(response: Any) -> ReviewModelOutput:
        """Pull the validated model output out of an SDK response.

        A malformed answer is a failure rather than something to salvage: a
        review assembled from fragments is worse than no review, because it
        looks like a verdict.
        """
        check_stop_reason(response, response_error=ReviewResponseError)

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise ReviewResponseError("provider returned no structured output")
        if not isinstance(parsed, ReviewModelOutput):
            raise ReviewResponseError(
                f"provider returned unexpected output type {type(parsed).__name__}"
            )
        return parsed


def _usage_from(response: Any) -> ReviewUsage:
    """Collect safe usage metadata, in the shape the review artifact expects.

    Anthropic reports input and output tokens separately and has no total, so
    the total is derived here rather than left blank - the artifact has carried
    one since Round 3 and an operator reading two Runs side by side should not
    have to know which vendor answered.
    """
    fields = usage_fields(response)
    input_tokens = fields["input_tokens"]
    output_tokens = fields["output_tokens"]
    total = (
        input_tokens + output_tokens
        if isinstance(input_tokens, int) and isinstance(output_tokens, int)
        else None
    )
    return ReviewUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
        request_id=fields["request_id"],
        response_status=fields["stop_reason"],
    )


def _build_sdk_client(settings: ReviewerSettings) -> Any:
    """Construct the real SDK client."""
    return build_sdk_client(
        api_key=settings.api_key,
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
        configuration_error=ReviewConfigurationError,
    )


__all__ = ["ANTHROPIC_PROVIDER", "REVIEWER_ERRORS", "AnthropicReviewerClient"]
