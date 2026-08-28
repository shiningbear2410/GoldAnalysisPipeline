"""Anthropic implementation of :class:`FinalizerClient`.

Shares the SDK error mapping with the writer through
:mod:`goldpipeline.adapters.anthropic_errors`, so the two stages cannot come to
disagree about what a 429 or a refusal means. What differs is only the schema
requested and the error classes raised.
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
from goldpipeline.adapters.finalizer_client import FinalizeRequest, FinalizeResponse
from goldpipeline.config import API_KEY_ENV, FINALIZER_MODEL_ENV, FinalizerSettings
from goldpipeline.domain.errors import (
    FinalizeConfigurationError,
    FinalizeProviderError,
    FinalizeResponseError,
    FinalizeTimeoutError,
)
from goldpipeline.schemas.finalizer import FinalizerModelOutput, FinalizerUsage

logger = logging.getLogger(__name__)

ANTHROPIC_PROVIDER = "anthropic"

FINALIZER_ERRORS = AnthropicErrorMap(
    timeout=FinalizeTimeoutError,
    provider=FinalizeProviderError,
    configuration=FinalizeConfigurationError,
    response=FinalizeResponseError,
    api_key_setting=API_KEY_ENV,
    model_setting=FINALIZER_MODEL_ENV,
)


class AnthropicFinalizerClient:
    """Calls Claude to revise a draft."""

    def __init__(self, settings: FinalizerSettings, *, client: Any | None = None) -> None:
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

    def finalize(self, request: FinalizeRequest) -> FinalizeResponse:
        """Call the model and return its parsed revision."""
        logger.info(
            "finalizer.provider call model=%s max_tokens=%d timeout=%.0fs",
            self._settings.model,
            request.max_tokens,
            self._settings.timeout_seconds,
        )

        response = self._call(request)
        output = self._extract_output(response)

        return FinalizeResponse(
            output=output,
            model=getattr(response, "model", self._settings.model) or self._settings.model,
            provider=self.provider,
            usage=FinalizerUsage(**usage_fields(response)),
        )

    def _call(self, request: FinalizeRequest) -> Any:
        """Issue the request, mapping SDK failures onto the project taxonomy."""
        import anthropic

        try:
            return self._client.messages.parse(
                model=self._settings.model,
                max_tokens=request.max_tokens,
                system=request.prompt.system,
                messages=[{"role": "user", "content": request.prompt.user}],
                output_format=FinalizerModelOutput,
            )
        except anthropic.AnthropicError as exc:
            raise_mapped(
                exc,
                errors=FINALIZER_ERRORS,
                timeout_seconds=self._settings.timeout_seconds,
                model=self._settings.model,
            )

    @staticmethod
    def _extract_output(response: Any) -> FinalizerModelOutput:
        """Pull the validated model output out of an SDK response."""
        check_stop_reason(response, response_error=FinalizeResponseError)

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise FinalizeResponseError("provider returned no structured output")
        if not isinstance(parsed, FinalizerModelOutput):
            raise FinalizeResponseError(
                f"provider returned unexpected output type {type(parsed).__name__}"
            )
        return parsed


def _build_sdk_client(settings: FinalizerSettings) -> Any:
    """Construct the real SDK client."""
    return build_sdk_client(
        api_key=settings.api_key,
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
        configuration_error=FinalizeConfigurationError,
    )


__all__ = ["ANTHROPIC_PROVIDER", "FINALIZER_ERRORS", "AnthropicFinalizerClient"]
