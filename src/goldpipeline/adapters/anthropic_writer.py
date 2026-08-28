"""Anthropic implementation of :class:`WriterClient`.

Together with :mod:`goldpipeline.adapters.anthropic_finalizer`, one of only two
modules that reach Claude - and the SDK itself is touched only through
:mod:`goldpipeline.adapters.anthropic_errors`, so the two stages cannot drift
apart in how they interpret a failure.

Two things it is careful about:

* **Structured output.** The response is requested through ``messages.parse``
  with :class:`WriterModelOutput` as the schema, so the model is constrained to
  the contract rather than asked to "reply in JSON" and parsed hopefully
  afterwards. There is no regex rescue of a malformed answer: an answer that
  does not satisfy the schema is a failure, and a failed draft is better than a
  plausible-looking one assembled from fragments.
* **Not leaking the key.** SDK exceptions can carry request context. Every one
  is caught and re-raised as a project error whose message is written here, not
  copied from the provider.
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
from goldpipeline.adapters.writer_client import WriterRequest, WriterResponse
from goldpipeline.config import API_KEY_ENV, MODEL_ENV, WriterSettings
from goldpipeline.domain.errors import (
    WriterConfigurationError,
    WriterProviderError,
    WriterResponseError,
    WriterTimeoutError,
)
from goldpipeline.schemas.writer import WriterModelOutput, WriterUsage

logger = logging.getLogger(__name__)

ANTHROPIC_PROVIDER = "anthropic"

WRITER_ERRORS = AnthropicErrorMap(
    timeout=WriterTimeoutError,
    provider=WriterProviderError,
    configuration=WriterConfigurationError,
    response=WriterResponseError,
    api_key_setting=API_KEY_ENV,
    model_setting=MODEL_ENV,
)


class AnthropicWriterClient:
    """Calls Claude to produce a draft."""

    def __init__(self, settings: WriterSettings, *, client: Any | None = None) -> None:
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

    def generate(self, request: WriterRequest) -> WriterResponse:
        """Call the model and return its parsed draft."""
        logger.info(
            "writer.provider call model=%s max_tokens=%d timeout=%.0fs",
            self._settings.model,
            request.max_tokens,
            self._settings.timeout_seconds,
        )

        response = self._call(request)
        output = self._extract_output(response)

        return WriterResponse(
            output=output,
            model=getattr(response, "model", self._settings.model) or self._settings.model,
            provider=self.provider,
            usage=_usage_from(response),
        )

    # -- provider call -----------------------------------------------------

    def _call(self, request: WriterRequest) -> Any:
        """Issue the request, mapping SDK failures onto the project taxonomy."""
        import anthropic

        try:
            return self._client.messages.parse(
                model=self._settings.model,
                max_tokens=request.max_tokens,
                system=request.prompt.system,
                messages=[{"role": "user", "content": request.prompt.user}],
                output_format=WriterModelOutput,
            )
        except anthropic.AnthropicError as exc:
            raise_mapped(
                exc,
                errors=WRITER_ERRORS,
                timeout_seconds=self._settings.timeout_seconds,
                model=self._settings.model,
            )

    # -- response handling -------------------------------------------------

    @staticmethod
    def _extract_output(response: Any) -> WriterModelOutput:
        """Pull the validated model output out of an SDK response."""
        check_stop_reason(response, response_error=WriterResponseError)

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise WriterResponseError("provider returned no structured output")
        if not isinstance(parsed, WriterModelOutput):
            raise WriterResponseError(
                f"provider returned unexpected output type {type(parsed).__name__}"
            )
        return parsed


def _usage_from(response: Any) -> WriterUsage:
    """Collect safe usage metadata. Never headers, never credentials."""
    return WriterUsage(**usage_fields(response))


def _build_sdk_client(settings: WriterSettings) -> Any:
    """Construct the real SDK client."""
    return build_sdk_client(
        api_key=settings.api_key,
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
        configuration_error=WriterConfigurationError,
    )


__all__ = ["ANTHROPIC_PROVIDER", "WRITER_ERRORS", "AnthropicWriterClient"]
