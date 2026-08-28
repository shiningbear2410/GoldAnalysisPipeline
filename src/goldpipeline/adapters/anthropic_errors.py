"""Mapping Anthropic SDK failures onto a stage's own error taxonomy.

The writer and the finalizer both call Claude and both need the same twelve
lines of "which SDK exception means what". Duplicating that would mean two
copies drifting apart - one learning about a new failure mode the other does
not - so the decisions live here once and each stage supplies its own error
classes.

The mapping itself encodes two judgements worth stating:

* **Authentication and model problems are configuration errors, not provider
  errors.** They will fail identically on retry, so they must not look
  transient to a caller deciding whether to try again.
* **Provider messages are never echoed.** An SDK exception can carry the
  request body back, and these messages end up in a Run's manifest. Every
  message below is written here, not copied from the provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn

from goldpipeline.domain.errors import PipelineError


@dataclass(frozen=True)
class AnthropicErrorMap:
    """Which error class a stage wants for each family of SDK failure."""

    timeout: type[PipelineError]
    provider: type[PipelineError]
    configuration: type[PipelineError]
    response: type[PipelineError]
    api_key_setting: str
    model_setting: str


def raise_mapped(
    exc: Exception,
    *,
    errors: AnthropicErrorMap,
    timeout_seconds: float,
    model: str,
) -> NoReturn:
    """Re-raise an Anthropic SDK exception as the stage's own error type.

    Args:
        exc: The exception the SDK raised.
        errors: The stage's error classes.
        timeout_seconds: Configured timeout, for the timeout message.
        model: Model id, for the unknown-model message.

    Raises:
        PipelineError: Always. The class depends on *exc*.
    """
    import anthropic

    if isinstance(exc, anthropic.APITimeoutError):
        raise errors.timeout(
            f"provider did not respond within {timeout_seconds:g}s",
            timeout_seconds=timeout_seconds,
        ) from exc

    if isinstance(exc, anthropic.AuthenticationError):
        raise errors.configuration(
            f"provider rejected the credentials; check {errors.api_key_setting}",
            setting=errors.api_key_setting,
        ) from exc

    if isinstance(exc, anthropic.PermissionDeniedError):
        raise errors.configuration(
            "credentials lack permission for this model", setting=errors.model_setting
        ) from exc

    if isinstance(exc, anthropic.NotFoundError):
        raise errors.configuration(
            f"provider does not recognise model {model!r}", setting=errors.model_setting
        ) from exc

    if isinstance(exc, anthropic.RateLimitError):
        raise errors.provider("provider rate limit reached", status_code=429) from exc

    if isinstance(exc, anthropic.APIStatusError):
        raise errors.provider(
            f"provider returned HTTP {exc.status_code}", status_code=exc.status_code
        ) from exc

    if isinstance(exc, anthropic.APIConnectionError):
        raise errors.provider("could not reach the provider") from exc

    raise exc


def check_stop_reason(response: Any, *, response_error: type[PipelineError]) -> None:
    """Reject a response that stopped for a reason that invalidates it.

    A refusal has no content to use. A ``max_tokens`` stop leaves text that
    parses but reads as an unfinished thought - worse than a failure, because it
    looks usable.

    Raises:
        PipelineError: If the response did not complete normally.
    """
    stop_reason = getattr(response, "stop_reason", None)

    if stop_reason == "refusal":
        raise response_error("model declined this request", stop_reason=stop_reason)
    if stop_reason == "max_tokens":
        raise response_error(
            "response hit the output token limit before completing",
            stop_reason=stop_reason,
        )


def usage_fields(response: Any) -> dict[str, Any]:
    """Extract safe usage metadata. Counts and opaque ids only.

    Never headers, never credentials, never an echo of the request body.
    """
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": _int_or_none(getattr(usage, "input_tokens", None)),
        "output_tokens": _int_or_none(getattr(usage, "output_tokens", None)),
        "cache_read_input_tokens": _int_or_none(getattr(usage, "cache_read_input_tokens", None)),
        "cache_creation_input_tokens": _int_or_none(
            getattr(usage, "cache_creation_input_tokens", None)
        ),
        "request_id": _str_or_none(getattr(response, "_request_id", None)),
        "stop_reason": _str_or_none(getattr(response, "stop_reason", None)),
    }


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _str_or_none(value: Any) -> str | None:
    return value[:200] if isinstance(value, str) and value else None


def build_sdk_client(
    *,
    api_key: str,
    timeout_seconds: float,
    max_retries: int,
    configuration_error: type[PipelineError],
) -> Any:
    """Construct the real SDK client.

    Imported lazily so the rest of the pipeline - and the whole offline test
    suite - does not require the SDK to be loaded.
    """
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise configuration_error(
            "the anthropic package is not installed; install it to reach the provider",
            setting="anthropic",
        ) from exc

    return anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds, max_retries=max_retries)


__all__ = [
    "AnthropicErrorMap",
    "build_sdk_client",
    "check_stop_reason",
    "raise_mapped",
    "usage_fields",
]
