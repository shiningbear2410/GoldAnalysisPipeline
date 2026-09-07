"""Asking a provider to repair digest editorial content.

Round 6.5c.3. A third *output schema* on the same vendor, and the last one this
product needs.

The pattern is the one Round 6.5b established for the digest writer: the same
:class:`~goldpipeline.config.FinalizerSettings` the analysis finalizer uses -
the same model, timeout, retry budget and credential, resolved through the same
:class:`SecretProvider` - with a different ``output_format``. No new credential,
no new configuration key, no parallel provider stack.

What the different output format buys is not convenience. The analysis finalizer
returns an article and must then be *checked* for having left the title, the
disclaimer and the prices alone; this one returns items, a balance and claims,
and there is no field it could have put a title in. The safety property is the
schema, and this adapter is what binds the provider to it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from goldpipeline.adapters.anthropic_errors import (
    check_stop_reason,
    raise_mapped,
    usage_fields,
)
from goldpipeline.adapters.anthropic_finalizer import FINALIZER_ERRORS
from goldpipeline.config import FinalizerSettings
from goldpipeline.domain.errors import FinalizeResponseError
from goldpipeline.schemas.digest_finalizer import DigestFinalizerModelOutput
from goldpipeline.schemas.finalizer import FinalizerPrompt, FinalizerUsage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DigestFinalizeRequest:
    """One digest revision request."""

    prompt: FinalizerPrompt
    run_id: str
    max_tokens: int = 8000


@dataclass(frozen=True)
class DigestFinalizeResponse:
    """A provider's answer, already parsed into the revision schema."""

    output: DigestFinalizerModelOutput
    model: str
    provider: str
    usage: FinalizerUsage = field(default_factory=FinalizerUsage)
    selection_id: str | None = None


@runtime_checkable
class DigestFinalizerClient(Protocol):
    """Anything that can repair digest editorial content."""

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    def finalize(self, request: DigestFinalizeRequest) -> DigestFinalizeResponse:
        """Repair the editorial content, once.

        Raises:
            FinalizeTimeoutError: The provider did not answer in time.
            FinalizeProviderError: The provider refused or failed.
            FinalizeResponseError: The answer could not be parsed into the schema.
        """
        ...


class AnthropicDigestFinalizerClient:
    """Anthropic, constrained to :class:`DigestFinalizerModelOutput`."""

    def __init__(self, settings: FinalizerSettings, *, client: Any | None = None) -> None:
        """Build a client.

        Args:
            settings: Credentials and tuning - the finalizer's own, unchanged.
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

    def finalize(self, request: DigestFinalizeRequest) -> DigestFinalizeResponse:
        """Call the model and return its parsed repair."""
        logger.info(
            "digest_finalizer.provider call model=%s max_tokens=%d timeout=%.0fs",
            self._settings.model,
            request.max_tokens,
            self._settings.timeout_seconds,
        )
        response = self._call(request)
        return DigestFinalizeResponse(
            output=self._extract_output(response),
            model=self._settings.model,
            provider=self.provider,
            usage=FinalizerUsage(**usage_fields(response)),
        )

    def _call(self, request: DigestFinalizeRequest) -> Any:
        """Issue the request, mapping SDK failures onto the project taxonomy."""
        import anthropic

        try:
            return self._client.messages.parse(
                model=self._settings.model,
                max_tokens=request.max_tokens,
                system=request.prompt.system,
                messages=[{"role": "user", "content": request.prompt.user}],
                output_format=DigestFinalizerModelOutput,
            )
        except anthropic.AnthropicError as exc:
            raise_mapped(
                exc,
                errors=FINALIZER_ERRORS,
                timeout_seconds=self._settings.timeout_seconds,
                model=self._settings.model,
            )

    @staticmethod
    def _extract_output(response: Any) -> DigestFinalizerModelOutput:
        """Pull the validated repair out of an SDK response."""
        check_stop_reason(response, response_error=FinalizeResponseError)

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise FinalizeResponseError("provider returned no structured output")
        if not isinstance(parsed, DigestFinalizerModelOutput):
            raise FinalizeResponseError(
                f"provider returned unexpected output type {type(parsed).__name__}"
            )
        return parsed


def _build_sdk_client(settings: FinalizerSettings) -> Any:
    """The vendor client, built from the finalizer's own settings."""
    import anthropic

    return anthropic.Anthropic(
        api_key=settings.api_key,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )


class LazyDigestFinalizerClient:
    """A :class:`DigestFinalizerClient` that builds its real client on first use.

    The same reasoning as
    :class:`~goldpipeline.adapters.finalizer_client.LazyFinalizerClient`, and it
    matters more here rather than less. A digest reaches a repair client only
    when its review asked for one; the common case - content PASS, style PASS -
    is a byte copy. Wrapping means an operator finishing a clean digest never
    needs an Anthropic key present, while a repair that genuinely needs one
    fails clearly and at the moment it matters.
    """

    def __init__(self, factory: Callable[[], DigestFinalizerClient]) -> None:
        """Wrap *factory*, which is called at most once."""
        self._factory = factory
        self._inner: DigestFinalizerClient | None = None

    def _client(self) -> DigestFinalizerClient:
        if self._inner is None:
            self._inner = self._factory()
        return self._inner

    @property
    def built(self) -> bool:
        """Whether the real client has been constructed yet."""
        return self._inner is not None

    @property
    def provider(self) -> str:
        return self._client().provider

    @property
    def model(self) -> str:
        return self._client().model

    def finalize(self, request: DigestFinalizeRequest) -> DigestFinalizeResponse:
        return self._client().finalize(request)


__all__ = [
    "AnthropicDigestFinalizerClient",
    "DigestFinalizeRequest",
    "DigestFinalizeResponse",
    "DigestFinalizerClient",
    "LazyDigestFinalizerClient",
]
