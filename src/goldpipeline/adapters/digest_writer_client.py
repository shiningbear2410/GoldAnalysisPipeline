"""Asking a provider for digest editorial content.

Round 6.5b. A second *output schema*, not a second provider system.

The digest asks the same vendor, on the same account, with the same
:class:`~goldpipeline.config.WriterSettings` - the same model, timeout, retry
budget and credential the analysis writer uses, resolved through the same
:class:`~goldpipeline.adapters.secrets.SecretProvider` that Round 6.4e.1 wired.
What differs is the shape of the answer: an analysis returns an article, a
digest returns the two things a model should decide about a digest.

That is why this is a small adapter beside the writer rather than a parallel
stack. There is no new credential, no new configuration key and no new model
selection - only ``output_format``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from goldpipeline.adapters.anthropic_errors import (
    check_stop_reason,
    raise_mapped,
    usage_fields,
)
from goldpipeline.adapters.anthropic_writer import WRITER_ERRORS
from goldpipeline.config import WriterSettings
from goldpipeline.domain.errors import WriterResponseError
from goldpipeline.schemas.news_digest import DigestEditorial
from goldpipeline.schemas.writer import WriterPrompt, WriterUsage

logger = logging.getLogger(__name__)

# The writer's own error map, reused rather than restated. A second copy is how
# two adapters on one vendor come to disagree about which failure is a timeout.


@dataclass(frozen=True)
class DigestRequest:
    """One digest generation request."""

    prompt: WriterPrompt
    run_id: str
    max_tokens: int = 8000


@dataclass(frozen=True)
class DigestResponse:
    """A provider's answer, already parsed into the editorial schema."""

    output: DigestEditorial
    model: str
    provider: str
    usage: WriterUsage = field(default_factory=WriterUsage)
    selection_id: str | None = None


@runtime_checkable
class DigestWriterClient(Protocol):
    """Anything that can turn a digest prompt into editorial content."""

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    def generate(self, request: DigestRequest) -> DigestResponse:
        """Produce the editorial half of a digest.

        Raises:
            WriterTimeoutError: The provider did not answer in time.
            WriterProviderError: The provider refused or failed.
            WriterResponseError: The answer could not be parsed into the schema.
        """
        ...


class AnthropicDigestWriterClient:
    """Anthropic, constrained to :class:`DigestEditorial`."""

    def __init__(self, settings: WriterSettings, *, client: Any | None = None) -> None:
        """Build a client.

        Args:
            settings: Credentials and tuning - the writer's own, unchanged.
            client: A pre-built SDK client, injected by tests so the parsing and
                error-mapping paths are exercisable without a network.
        """
        self._settings = settings
        self._client = client if client is not None else _build_sdk_client(settings)

    @property
    def provider(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._settings.model

    def generate(self, request: DigestRequest) -> DigestResponse:
        """Call the model and return its parsed editorial content."""
        logger.info(
            "digest_writer.provider call model=%s max_tokens=%d timeout=%.0fs",
            self._settings.model,
            request.max_tokens,
            self._settings.timeout_seconds,
        )
        response = self._call(request)
        return DigestResponse(
            output=self._extract_output(response),
            model=self._settings.model,
            provider=self.provider,
            usage=WriterUsage(**usage_fields(response)),
        )

    def _call(self, request: DigestRequest) -> Any:
        """Issue the request, mapping SDK failures onto the project taxonomy."""
        import anthropic

        try:
            return self._client.messages.parse(
                model=self._settings.model,
                max_tokens=request.max_tokens,
                system=request.prompt.system,
                messages=[{"role": "user", "content": request.prompt.user}],
                output_format=DigestEditorial,
            )
        except anthropic.AnthropicError as exc:
            raise_mapped(
                exc,
                errors=WRITER_ERRORS,
                timeout_seconds=self._settings.timeout_seconds,
                model=self._settings.model,
            )

    @staticmethod
    def _extract_output(response: Any) -> DigestEditorial:
        """Pull the validated editorial content out of an SDK response."""
        check_stop_reason(response, response_error=WriterResponseError)

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise WriterResponseError("provider returned no structured output")
        if not isinstance(parsed, DigestEditorial):
            raise WriterResponseError(
                f"provider returned unexpected output type {type(parsed).__name__}"
            )
        return parsed


def _build_sdk_client(settings: WriterSettings) -> Any:
    """The vendor client, built from the writer's own settings."""
    import anthropic

    return anthropic.Anthropic(
        api_key=settings.api_key,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )


__all__ = [
    "AnthropicDigestWriterClient",
    "DigestRequest",
    "DigestResponse",
    "DigestWriterClient",
]
