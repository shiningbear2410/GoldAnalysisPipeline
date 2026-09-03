"""DeepSeek implementation of :class:`FinalizerClient`.

The mirror of :mod:`goldpipeline.adapters.deepseek_writer`, and for the same
reason the Anthropic pair are two modules: the two stages have different output
schemas and different error classes, and sharing a class would mean one set of
messages describing both.

Everything that decides whether a revision is *acceptable* -
:mod:`goldpipeline.services.finalizer_policy`, the postchecks, the gate - is
provider-independent and untouched.

**No fallback, ever.** A DeepSeek finalizer failure fails the finalize stage. It
does not reach for Claude and it does not change the selection's mode.
"""

from __future__ import annotations

import logging

from goldpipeline.adapters.deepseek_client import (
    DEEPSEEK_PROVIDER,
    ChatResult,
    DeepSeekChatClient,
    DeepSeekErrorMap,
    build_payload,
    parse_content,
    schema_appendix,
    usage_fields,
)
from goldpipeline.adapters.finalizer_client import FinalizeRequest, FinalizeResponse
from goldpipeline.config import DEEPSEEK_API_KEY_ENV, DeepSeekSettings
from goldpipeline.domain.errors import (
    FinalizeConfigurationError,
    FinalizeProviderError,
    FinalizeResponseError,
    FinalizeTimeoutError,
)
from goldpipeline.schemas.finalizer import FinalizerModelOutput, FinalizerUsage
from goldpipeline.schemas.preferences import ModelSpec, Provider, resolve_model

logger = logging.getLogger(__name__)

FINALIZER_ERRORS = DeepSeekErrorMap(
    timeout=FinalizeTimeoutError,
    provider=FinalizeProviderError,
    configuration=FinalizeConfigurationError,
    response=FinalizeResponseError,
    api_key_setting=DEEPSEEK_API_KEY_ENV,
)


class DeepSeekFinalizerClient:
    """Calls DeepSeek to revise an article to order."""

    def __init__(
        self,
        settings: DeepSeekSettings,
        model: ModelSpec,
        *,
        transport: object | None = None,
    ) -> None:
        self._settings = settings
        self._model = model
        self._chat = DeepSeekChatClient(settings, errors=FINALIZER_ERRORS, transport=transport)

    @property
    def provider(self) -> str:
        return DEEPSEEK_PROVIDER

    @property
    def model(self) -> str:
        """The vendor id, which is what actually ran."""
        return self._model.api_model_id

    @property
    def selection_id(self) -> str:
        return self._model.selection_id

    def finalize(self, request: FinalizeRequest) -> FinalizeResponse:
        """Call the model and return its parsed revision."""
        result = self._chat.complete(
            build_payload(
                self._model,
                system=request.prompt.system,
                user=request.prompt.user + schema_appendix(FinalizerModelOutput),
                max_tokens=request.max_tokens,
            )
        )
        output = parse_content(
            result.content, model=FinalizerModelOutput, response_error=FinalizeResponseError
        )
        assert isinstance(output, FinalizerModelOutput)  # noqa: S101 - guaranteed by parse_content

        return FinalizeResponse(
            output=output,
            model=result.api_model or self._model.api_model_id,
            provider=self.provider,
            usage=_usage(result),
        )


def _usage(result: ChatResult) -> FinalizerUsage:
    return FinalizerUsage(
        **usage_fields(
            result.usage, finish_reason=result.finish_reason, request_id=result.request_id
        )
    )


def build_deepseek_finalizer(
    selection_id: str,
    *,
    settings: DeepSeekSettings | None = None,
    transport: object | None = None,
) -> DeepSeekFinalizerClient:
    """Build a finalizer for one catalog selection, resolving the key lazily.

    Raises:
        FinalizeConfigurationError: No key, or an unusable endpoint.
        ValueError: The catalog does not offer that selection.
    """
    model = resolve_model(Provider.DEEPSEEK, selection_id)
    resolved = settings or DeepSeekSettings.from_env(error=FinalizeConfigurationError)
    return DeepSeekFinalizerClient(resolved, model, transport=transport)


__all__ = [
    "FINALIZER_ERRORS",
    "DeepSeekFinalizerClient",
    "build_deepseek_finalizer",
]
