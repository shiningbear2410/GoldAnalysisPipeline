"""Anthropic implementation of :class:`TradeAnalystClient`.

Round 6.6h. The third module that reaches Claude, and the smallest: it sends two
strings and returns one.

**No structured output, deliberately.** The writer uses ``messages.parse`` with
a schema because an article has a rich shape worth constraining. A ranking is
four arrays of opaque ids, and its correctness is not a JSON-shape question - it
is "are these exactly the candidates we asked about", which no schema can
express. So the text comes back raw and the domain validator decides, which is
also what makes the offline fakes an honest rehearsal of the real thing.

**Two constraints on generation, and neither touches the validator.** The first
live call sent 238,000 input tokens describing 130 candidates and got back
16,000 output tokens of pure reasoning and no answer at all: ordering a list is
not a problem extended thinking helps with, and on a set this size it consumed
the entire budget before reaching the JSON. Thinking is therefore switched off,
and the same call then finished in 37 seconds using 1,919 output tokens.

The second is a prefilled ``{``. The prompt asks for bare JSON and the model
wrapped it in a markdown fence anyway; prefilling the opening brace makes the
answer JSON by construction, the way the writer's schema does for an article.
Both are transport-level: they shape what is generated, and every id in the
result is still checked against the deterministic candidate set afterwards.

**Same transport, same error map.** Built through
:mod:`~goldpipeline.adapters.anthropic_errors` like the writer and the finalizer,
so a timeout, a refusal and a bad key are classified identically wherever they
happen. There is no second HTTP stack here and no credential read in this file.
"""

from __future__ import annotations

import logging
from typing import Any

from goldpipeline.adapters.anthropic_errors import (
    AnthropicErrorMap,
    build_sdk_client,
    check_stop_reason,
    raise_mapped,
)
from goldpipeline.adapters.trade_analyst_client import (
    TradeAnalystRequest,
    TradeAnalystResponse,
)
from goldpipeline.config import API_KEY_ENV, MODEL_ENV, WriterSettings
from goldpipeline.domain.errors import (
    WriterConfigurationError,
    WriterProviderError,
    WriterResponseError,
    WriterTimeoutError,
)

logger = logging.getLogger(__name__)

ANTHROPIC_PROVIDER = "anthropic"

ANSWER_PREFILL = "{"
"""The opening brace of the ranking, put into the model's mouth.

A one-character assistant turn. It cannot change which ids come back - the
validator still checks every one against the deterministic set - and it
removes the one failure the prompt could not: a correct answer wrapped in a
markdown fence, which is not JSON.
"""

ANALYST_ERRORS = AnthropicErrorMap(
    timeout=WriterTimeoutError,
    provider=WriterProviderError,
    configuration=WriterConfigurationError,
    response=WriterResponseError,
    api_key_setting=API_KEY_ENV,
    model_setting=MODEL_ENV,
)
"""Reuses the writer's error taxonomy rather than inventing a parallel one.

The failures are the same failures - a socket that timed out, a key that is
wrong, a model that stopped early - and giving them new names would mean the
retry classifier had to learn a second vocabulary for the same events.
"""


class AnthropicTradeAnalystClient:
    """Calls Claude to rank candidates, and returns whatever it said."""

    def __init__(self, settings: WriterSettings, *, client: Any | None = None) -> None:
        """Build a client.

        Args:
            settings: Credentials and tuning. Never logged, never stored on a Run.
            client: A pre-built SDK client, injected by tests.
        """
        self._settings = settings
        self._client = client if client is not None else _build_sdk_client(settings)

    @property
    def provider(self) -> str:
        return ANTHROPIC_PROVIDER

    @property
    def model(self) -> str:
        return self._settings.model

    def rank(self, request: TradeAnalystRequest) -> TradeAnalystResponse:
        """Send the ranking prompt and return the raw answer.

        The answer is not trusted here in any way. It is text until the domain
        validator has checked every id against the deterministic candidate set.
        """
        logger.info(
            "trade_analyst.provider call model=%s max_tokens=%d timeout=%.0fs",
            self._settings.model,
            request.max_tokens,
            self._settings.timeout_seconds,
        )
        response = self._call(request)
        check_stop_reason(response, response_error=WriterResponseError)
        return TradeAnalystResponse(
            # The prefill is part of the answer and the provider does not echo
            # it, so it is put back before anything tries to parse the JSON.
            text=ANSWER_PREFILL + _text_of(response),
            model=getattr(response, "model", self._settings.model) or self._settings.model,
            provider=self.provider,
        )

    def _call(self, request: TradeAnalystRequest) -> Any:
        import anthropic

        try:
            return self._client.messages.create(
                model=self._settings.model,
                max_tokens=request.max_tokens,
                system=request.system,
                messages=[
                    {"role": "user", "content": request.user},
                    # Prefill. The answer continues from here, so it is JSON by
                    # construction rather than by the model's good manners.
                    {"role": "assistant", "content": ANSWER_PREFILL},
                ],
                thinking={"type": "disabled"},
            )
        except anthropic.AnthropicError as exc:
            raise_mapped(
                exc,
                errors=ANALYST_ERRORS,
                timeout_seconds=self._settings.timeout_seconds,
                model=self._settings.model,
            )


def _text_of(response: Any) -> str:
    """Join the response's text blocks, and refuse anything else.

    A tool-use block or an image in a ranking answer is not something to skip
    past quietly - it means the model did something other than what was asked,
    and the validator downstream would report a confusing JSON error instead of
    the real one.
    """
    blocks = getattr(response, "content", None)
    if not blocks:
        raise WriterResponseError("provider returned no content")

    parts: list[str] = []
    for block in blocks:
        kind = getattr(block, "type", None)
        if kind != "text":
            raise WriterResponseError(f"provider returned a {kind!r} block, not text")
        parts.append(str(getattr(block, "text", "")))

    joined = "".join(parts).strip()
    if not joined:
        raise WriterResponseError("provider returned empty text")
    return joined


def _build_sdk_client(settings: WriterSettings) -> Any:
    return build_sdk_client(
        api_key=settings.api_key,
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
        configuration_error=WriterConfigurationError,
    )


__all__ = [
    "ANALYST_ERRORS",
    "ANSWER_PREFILL",
    "ANTHROPIC_PROVIDER",
    "AnthropicTradeAnalystClient",
]
