"""DeepSeek implementation of :class:`WriterClient`.

Same protocol, same output schema, same downstream checks. Everything that
decides whether a draft is *acceptable* - the run id echo, claim resolution,
news provenance, the prechecks, the gate - is provider-independent and untouched
by this module. What differs is the transport and how a response becomes a
:class:`WriterModelOutput`.

**No fallback, ever.** If this client fails, the writer stage fails. It does not
reach for Claude, it does not step a Pro selection down to Flash, and it does not
turn thinking off to try again. One selection means one runtime behaviour, and a
silent substitution would mean an operator reading ``provider: deepseek`` on an
artifact that Claude wrote.
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
from goldpipeline.adapters.writer_client import WriterRequest, WriterResponse
from goldpipeline.config import DEEPSEEK_API_KEY_ENV, DeepSeekSettings
from goldpipeline.domain.errors import (
    WriterConfigurationError,
    WriterProviderError,
    WriterResponseError,
    WriterTimeoutError,
)
from goldpipeline.schemas.preferences import ModelSpec, Provider, resolve_model
from goldpipeline.schemas.writer import WriterModelOutput, WriterUsage

logger = logging.getLogger(__name__)

WRITER_ERRORS = DeepSeekErrorMap(
    timeout=WriterTimeoutError,
    provider=WriterProviderError,
    configuration=WriterConfigurationError,
    response=WriterResponseError,
    api_key_setting=DEEPSEEK_API_KEY_ENV,
)


class DeepSeekWriterClient:
    """Calls DeepSeek to produce a draft."""

    def __init__(
        self,
        settings: DeepSeekSettings,
        model: ModelSpec,
        *,
        transport: object | None = None,
    ) -> None:
        """Build a client.

        Args:
            settings: Credential and bounds. Never logged, never stored on a Run.
            model: The catalog entry for the operator's selection. It carries
                both what the person picked and what goes on the wire.
            transport: A pre-built HTTP client, injected by tests so every path
                below runs without a network.
        """
        self._settings = settings
        self._model = model
        self._chat = DeepSeekChatClient(settings, errors=WRITER_ERRORS, transport=transport)

    @property
    def provider(self) -> str:
        return DEEPSEEK_PROVIDER

    @property
    def model(self) -> str:
        """What the artifact records as the model.

        The vendor id rather than the selection, because this field answers
        "what actually ran". The selection is recorded separately, so a Run
        written under ``DeepSeek Reasoner`` still says which model served it.
        """
        return self._model.api_model_id

    @property
    def selection_id(self) -> str:
        return self._model.selection_id

    def generate(self, request: WriterRequest) -> WriterResponse:
        """Call the model and return its parsed draft."""
        result = self._chat.complete(
            build_payload(
                self._model,
                system=request.prompt.system,
                user=request.prompt.user + schema_appendix(WriterModelOutput),
                max_tokens=request.max_tokens,
            )
        )
        output = parse_content(
            result.content, model=WriterModelOutput, response_error=WriterResponseError
        )
        assert isinstance(output, WriterModelOutput)  # noqa: S101 - parse_content guarantees it

        return WriterResponse(
            output=output,
            model=result.api_model or self._model.api_model_id,
            provider=self.provider,
            usage=_usage(result),
        )


def _usage(result: ChatResult) -> WriterUsage:
    return WriterUsage(
        **usage_fields(
            result.usage, finish_reason=result.finish_reason, request_id=result.request_id
        )
    )


def build_deepseek_writer(
    selection_id: str,
    *,
    settings: DeepSeekSettings | None = None,
    transport: object | None = None,
) -> DeepSeekWriterClient:
    """Build a writer for one catalog selection, resolving the key lazily.

    The settings - and therefore the credential lookup - happen here, at the
    moment a DeepSeek call is actually about to be made. An operator who never
    selects DeepSeek never causes a DeepSeek key to be looked for.

    Raises:
        WriterConfigurationError: No key, or an unusable endpoint.
        ValueError: The catalog does not offer that selection.
    """
    model = resolve_model(Provider.DEEPSEEK, selection_id)
    resolved = settings or DeepSeekSettings.from_env(error=WriterConfigurationError)
    return DeepSeekWriterClient(resolved, model, transport=transport)


__all__ = [
    "WRITER_ERRORS",
    "DeepSeekWriterClient",
    "build_deepseek_writer",
]
