"""The one place this pipeline talks to DeepSeek.

Shared by the writer and the finalizer for the same reason
:mod:`goldpipeline.adapters.anthropic_errors` is: two stages needing the same
twenty lines of "which failure means what" is two copies drifting apart, one
learning about a failure mode the other does not.

**Why raw HTTP and not an SDK.** The Anthropic adapters lean on
``messages.parse`` with a pydantic ``output_format`` - a Claude-specific feature
this project already depends on and that has no equivalent here. DeepSeek's
OpenAI-format endpoint is a single POST with a JSON body, ``httpx2`` is already a
dependency used by three other adapters, and the thinking control this has to
send is documented in that format. A second SDK would add a dependency to gain
nothing, and pointing the Anthropic SDK at the vendor's compatibility endpoint
would make structured output an assumption nobody here can check offline.

**Why no redirects.** The bearer token is on the request. A redirect to another
origin would re-send it there, handing the credential to whoever answered. So
redirects are refused and reported, exactly as the remote-intake transport does.

**Why the credential never appears in an error.** Every message raised here is
written here, not copied from the response body. A provider error page can
contain the request that produced it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from goldpipeline.config import DeepSeekSettings
from goldpipeline.domain.errors import PipelineError
from goldpipeline.schemas.preferences import ModelSpec, ThinkingMode

logger = logging.getLogger(__name__)

DEEPSEEK_PROVIDER = "deepseek"
"""Recorded on the artifact. Short, lower-case, like ``anthropic``."""

CHAT_COMPLETIONS_PATH = "/chat/completions"

RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
"""Statuses that describe a moment rather than a request.

A 5xx is the server's problem and may pass; ``408`` and ``429`` are the server
saying "later". Every other 4xx says *this request* is wrong, and waiting
changes nothing about it - the lesson a real scheduled Run taught the Anthropic
adapter when a 400 was retried three times to discover the same certainty.
"""

MAX_RETRY_AFTER_SECONDS = 20.0
"""The longest ``Retry-After`` this will honour.

A one-shot scheduled process has a tick budget. A vendor asking for two minutes
is asking for more than this process has, so the attempt is abandoned and the
work returns through the ordinary retry ladder rather than sleeping through the
window it was given.
"""

MAX_ATTEMPTS_CEILING = 4
"""Hard ceiling on attempts regardless of configuration.

Each attempt is a billable generation. A misconfigured ``max_retries`` should
cost a wasted minute, not a wasted afternoon's budget.
"""


@dataclass(frozen=True)
class DeepSeekErrorMap:
    """Which error class a stage wants for each family of failure."""

    timeout: type[PipelineError]
    provider: type[PipelineError]
    configuration: type[PipelineError]
    response: type[PipelineError]
    api_key_setting: str


@dataclass(frozen=True)
class ChatResult:
    """One completed call, reduced to what a stage may use."""

    content: str
    """The visible answer. Never the reasoning."""

    api_model: str
    """What the vendor said it ran, falling back to what was asked for."""

    usage: dict[str, Any]
    finish_reason: str | None

    request_id: str | None = None


def build_payload(
    model: ModelSpec,
    *,
    system: str,
    user: str,
    max_tokens: int,
    json_output: bool = True,
) -> dict[str, Any]:
    """The request body for one generation.

    Two decisions worth reading:

    **The wire model is never the selection.** ``model`` on the request is
    ``api_model_id``, so a retired alias a person still sees on a button - Chat,
    Reasoner - never reaches the vendor. A test asserts it.

    **Sampling controls are absent, deliberately.** The Anthropic path has no
    ``temperature`` either, but here their absence is load-bearing: the vendor
    documents ``temperature``, ``top_p``, ``presence_penalty`` and
    ``frequency_penalty`` as having no effect in thinking mode, and three of the
    four shipped selections think. A field that changes nothing is worse than a
    missing one - somebody eventually tunes it and cannot work out why the output
    does not move.
    """
    payload: dict[str, Any] = {
        "model": model.api_model_id,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    if model.thinking is not ThinkingMode.NOT_APPLICABLE:
        # Sent explicitly in both directions. Thinking is on by default, so
        # relying on that default would leave "DeepSeek Chat" - a user-visible
        # promise of a non-thinking answer - at the mercy of a vendor default
        # nobody here controls.
        payload["thinking"] = {
            "type": "enabled" if model.thinking is ThinkingMode.ENABLED else "disabled"
        }

    if json_output:
        payload["response_format"] = {"type": "json_object"}

    return payload


class DeepSeekChatClient:
    """Posts one chat completion and returns the visible content.

    Stateless between calls and holds no connection: a one-shot scheduled
    process makes one or two generations and exits, so a pooled client would be
    a lifetime to manage for no benefit.
    """

    def __init__(
        self,
        settings: DeepSeekSettings,
        *,
        errors: DeepSeekErrorMap,
        transport: Any | None = None,
    ) -> None:
        """Build a client.

        Args:
            settings: Credential, endpoint and bounds. Never logged.
            errors: The calling stage's error classes.
            transport: A pre-built ``httpx2``-shaped client, injected by tests so
                every path below runs without a network.
        """
        self._settings = settings
        self._errors = errors
        self._transport = transport

    @property
    def base_url(self) -> str:
        return self._settings.base_url

    def complete(self, payload: dict[str, Any]) -> ChatResult:
        """Send one request, retrying only what is safe to retry.

        Raises:
            PipelineError: One of the calling stage's four error classes. The
                message is written here; no part of a provider response body and
                no part of the credential ever reaches it.
        """
        attempts = min(self._settings.max_retries + 1, MAX_ATTEMPTS_CEILING)
        last: PipelineError | None = None

        for attempt in range(1, attempts + 1):
            try:
                return self._attempt(payload)
            except PipelineError as exc:
                if not _retryable(exc) or attempt == attempts:
                    raise
                last = exc
                logger.warning("deepseek.retry attempt=%d/%d code=%s", attempt, attempts, exc.code)

        raise last if last else self._errors.provider("deepseek call made no attempt")

    # -- one attempt -------------------------------------------------------

    def _attempt(self, payload: dict[str, Any]) -> ChatResult:
        response = self._post(payload)
        self._raise_for_status(response)
        return self._read(response, payload)

    def _post(self, payload: dict[str, Any]) -> Any:
        import httpx2

        url = f"{self._settings.base_url}{CHAT_COMPLETIONS_PATH}"
        headers = {
            # Read at call time and never stored on this object beyond settings,
            # so nothing that repr()s the client can print it.
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        logger.info(
            "deepseek.call model=%s thinking=%s max_tokens=%s timeout=%.0fs",
            payload.get("model"),
            (payload.get("thinking") or {}).get("type", "default"),
            payload.get("max_tokens"),
            self._settings.timeout_seconds,
        )

        try:
            if self._transport is not None:
                return self._transport.post(url, headers=headers, json=payload)
            with httpx2.Client(
                timeout=self._settings.timeout_seconds,
                follow_redirects=False,
                verify=True,
            ) as client:
                return client.post(url, headers=headers, json=payload)
        except httpx2.TimeoutException as exc:
            raise self._errors.timeout(
                f"deepseek did not respond within {self._settings.timeout_seconds:g}s",
                timeout_seconds=self._settings.timeout_seconds,
            ) from exc
        except httpx2.HTTPError as exc:
            raise self._errors.provider("could not reach deepseek") from exc

    def _raise_for_status(self, response: Any) -> None:
        """Turn a status code into the right kind of refusal."""
        status = int(getattr(response, "status_code", 0))
        if 200 <= status < 300:
            return

        if status in (301, 302, 303, 307, 308):
            # Never followed: the Authorization header would travel to whatever
            # origin answered, which is how a credential leaves the machine.
            raise self._errors.provider(
                "deepseek answered with a redirect, which is not followed because "
                "the Authorization header must not be re-sent to another origin",
                status_code=status,
            )

        if status in (401, 403):
            raise self._errors.configuration(
                f"deepseek rejected the credentials (HTTP {status}); "
                f"check {self._errors.api_key_setting}",
                status_code=status,
                setting=self._errors.api_key_setting,
            )

        if status in RETRYABLE_STATUS_CODES:
            raise self._errors.provider(
                f"deepseek returned HTTP {status}",
                status_code=status,
                retry_after=_retry_after(response),
            )

        raise self._errors.configuration(
            f"deepseek rejected the request (HTTP {status}); the same request would "
            "be rejected again, so it is not retried",
            status_code=status,
        )

    def _read(self, response: Any, payload: dict[str, Any]) -> ChatResult:
        """Extract the visible content, refusing anything unusable."""
        body = self._body(response)

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise self._errors.response("deepseek returned no choices")

        first: dict[str, Any] = choices[0] if isinstance(choices[0], dict) else {}
        raw_message = first.get("message")
        message: dict[str, Any] = raw_message if isinstance(raw_message, dict) else {}
        finish = first.get("finish_reason")
        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            # The failure mode this check exists for: a thinking response that
            # spends its whole budget reasoning and returns empty visible text.
            # Reasoning is not an answer - it is not the JSON contract, it was
            # never validated, and treating it as output would publish a model's
            # working. It is not read, not persisted and not reported.
            reasoned = bool(message.get("reasoning_content"))
            raise self._errors.response(
                "deepseek returned no visible content"
                + (
                    "; the response spent its budget on reasoning, which is not an answer"
                    if reasoned
                    else ""
                ),
                finish_reason=_safe_str(finish),
            )

        if finish == "length":
            raise self._errors.response(
                "deepseek hit the output token limit before completing",
                finish_reason="length",
            )

        return ChatResult(
            content=content,
            api_model=_safe_str(body.get("model")) or str(payload.get("model", "")),
            usage=_usage_object(body),
            finish_reason=_safe_str(finish),
            request_id=_safe_str(body.get("id")),
        )

    def _body(self, response: Any) -> dict[str, Any]:
        """Decode the response, refusing one that is too large or not JSON."""
        content = getattr(response, "content", b"")
        if isinstance(content, bytes) and len(content) > self._settings.max_response_bytes:
            raise self._errors.response(
                "deepseek response exceeds the size cap",
                limit_bytes=self._settings.max_response_bytes,
            )

        try:
            body = response.json()
        except (ValueError, TypeError, AttributeError) as exc:
            raise self._errors.response("deepseek response is not valid JSON") from exc

        if not isinstance(body, dict):
            raise self._errors.response("deepseek response is not a JSON object")
        return body


def parse_content(
    content: str,
    *,
    model: type[Any],
    response_error: type[PipelineError],
) -> Any:
    """Validate the visible content against a stage's output schema.

    JSON mode plus strict validation, never a regex rescue of a malformed
    answer. A response that does not satisfy the contract is a failure, and a
    failed draft is better than a plausible-looking one assembled from
    fragments - the same rule the Anthropic path gets from ``output_format``.

    Raises:
        PipelineError: The content is not JSON, or not this schema.
    """
    from pydantic import ValidationError

    try:
        return model.model_validate_json(content)
    except ValidationError as exc:
        problems = exc.errors()
        where = ".".join(str(p) for p in problems[0]["loc"]) if problems else "document"
        detail = problems[0]["msg"] if problems else "invalid"
        raise response_error(
            f"deepseek output does not satisfy {model.__name__} ({where}: {detail})"
        ) from exc
    except ValueError as exc:
        raise response_error(f"deepseek output is not valid JSON for {model.__name__}") from exc


def schema_appendix(model: type[Any]) -> str:
    """A section naming the exact JSON object the answer must be.

    The Anthropic path gets this for free: ``messages.parse`` constrains the
    model to the schema. DeepSeek's JSON mode guarantees only *valid JSON*, so
    the shape has to be stated - and stated from the schema itself, generated
    here, rather than written out by hand where it would drift the first time a
    field changed.

    Appended to the user turn by the adapter, never to the versioned prompt
    file: it describes what this transport needs, not what the product is.
    """
    return "\n".join(
        (
            "",
            "# OUTPUT SCHEMA",
            "",
            "Return one JSON object and nothing else - no prose, no code fence.",
            "It must validate against this JSON Schema:",
            "",
            "```json",
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2),
            "```",
        )
    )


def usage_fields(
    usage: dict[str, Any], *, finish_reason: str | None, request_id: str | None
) -> dict[str, Any]:
    """Map vendor usage onto the project's usage schema. Counts and ids only.

    Cache fields are left ``None`` rather than guessed: the two vendors count
    different things, and a number invented to fill a field is worse than an
    honest gap in a cost record.
    """
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    return {
        "input_tokens": _int_or_none(usage.get("prompt_tokens")),
        "output_tokens": _int_or_none(usage.get("completion_tokens")),
        "cache_read_input_tokens": _int_or_none(cached),
        "cache_creation_input_tokens": None,
        "request_id": request_id,
        "stop_reason": finish_reason,
    }


def _usage_object(body: dict[str, Any]) -> dict[str, Any]:
    """The vendor's usage block, or an empty one when it sent none."""
    usage = body.get("usage")
    return usage if isinstance(usage, dict) else {}


def _retryable(exc: PipelineError) -> bool:
    """Whether trying the same request again could plausibly answer differently.

    Only a provider-class failure with a retryable status, or a transport
    failure with no status at all. A configuration error is never retried - it
    would fail identically - and neither is a schema-invalid success: the model
    answered, and it answered wrongly.
    """
    status = exc.details.get("status_code")
    if status is None:
        return exc.code.endswith("_PROVIDER_ERROR") or exc.code in _TRANSPORT_CODES
    return isinstance(status, int) and status in RETRYABLE_STATUS_CODES


_TRANSPORT_CODES = frozenset({"WRITER_PROVIDER_ERROR", "FINALIZE_PROVIDER_ERROR"})


def _retry_after(response: Any) -> float | None:
    """The vendor's requested wait, when it is short enough to be worth honouring.

    Reported rather than slept through. This process does not block on a
    vendor's schedule; the automation layer already owns backoff, and it is the
    layer that knows how much of the tick budget is left.
    """
    headers: Any = getattr(response, "headers", None)
    if not hasattr(headers, "get"):
        return None
    try:
        seconds = float(headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None
    return seconds if 0 < seconds <= MAX_RETRY_AFTER_SECONDS else None


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _safe_str(value: Any) -> str | None:
    return value[:200] if isinstance(value, str) and value else None


__all__ = [
    "CHAT_COMPLETIONS_PATH",
    "DEEPSEEK_PROVIDER",
    "MAX_ATTEMPTS_CEILING",
    "MAX_RETRY_AFTER_SECONDS",
    "RETRYABLE_STATUS_CODES",
    "ChatResult",
    "DeepSeekChatClient",
    "DeepSeekErrorMap",
    "build_payload",
    "parse_content",
    "schema_appendix",
    "usage_fields",
]
