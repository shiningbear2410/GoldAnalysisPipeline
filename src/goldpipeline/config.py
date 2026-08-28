"""Configuration read from the environment.

Secrets live here and nowhere else. Two rules this module exists to enforce:

* an API key is never written to a Run, a log line, or an exception message.
  :class:`WriterSettings` therefore stores the key behind ``api_key`` and
  overrides ``__repr__`` so an accidental ``print(settings)`` or a traceback
  frame dump cannot leak it;
* nothing that arrives through pipeline *data* can change configuration. Every
  value below comes from the process environment or an explicit argument -
  never from ``context.json`` or from the raw analysis text.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from goldpipeline.domain.errors import (
    FinalizeConfigurationError,
    ReviewConfigurationError,
    WriterConfigurationError,
)

DEFAULT_MODEL = "claude-opus-5"
"""Writer model used when ``ANTHROPIC_MODEL`` is not set."""

DEFAULT_REVIEW_MODEL = "gpt-5.1"
"""Reviewer model used when ``OPENAI_REVIEW_MODEL`` is not set."""

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_TOKENS = 8000

API_KEY_ENV = "ANTHROPIC_API_KEY"
MODEL_ENV = "ANTHROPIC_MODEL"
TIMEOUT_ENV = "GOLDPIPELINE_WRITER_TIMEOUT"
MAX_RETRIES_ENV = "GOLDPIPELINE_WRITER_MAX_RETRIES"
MAX_TOKENS_ENV = "GOLDPIPELINE_WRITER_MAX_TOKENS"

FINALIZER_MODEL_ENV = "ANTHROPIC_FINALIZER_MODEL"
FINALIZER_TIMEOUT_ENV = "GOLDPIPELINE_FINALIZER_TIMEOUT"
FINALIZER_MAX_RETRIES_ENV = "GOLDPIPELINE_FINALIZER_MAX_RETRIES"
FINALIZER_MAX_TOKENS_ENV = "GOLDPIPELINE_FINALIZER_MAX_TOKENS"

REVIEW_API_KEY_ENV = "OPENAI_API_KEY"
REVIEW_MODEL_ENV = "OPENAI_REVIEW_MODEL"
REVIEW_TIMEOUT_ENV = "GOLDPIPELINE_REVIEW_TIMEOUT"
REVIEW_MAX_RETRIES_ENV = "GOLDPIPELINE_REVIEW_MAX_RETRIES"
REVIEW_MAX_TOKENS_ENV = "GOLDPIPELINE_REVIEW_MAX_TOKENS"

_REDACTED = "***redacted***"


@dataclass(frozen=True)
class WriterSettings:
    """Everything the writer stage needs in order to call a provider."""

    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    max_tokens: int = DEFAULT_MAX_TOKENS

    def __repr__(self) -> str:
        """Render without the credential.

        ``field(repr=False)`` already hides it; this makes the redaction
        visible so a reader of a log line knows a key was present.
        """
        return (
            f"WriterSettings(api_key={_REDACTED}, model={self.model!r}, "
            f"timeout_seconds={self.timeout_seconds}, max_retries={self.max_retries}, "
            f"max_tokens={self.max_tokens})"
        )

    __str__ = __repr__

    @classmethod
    def from_env(
        cls,
        env: dict[str, str] | None = None,
        *,
        model_override: str | None = None,
    ) -> WriterSettings:
        """Build settings from the environment.

        Args:
            env: Mapping to read from. Defaults to ``os.environ``; injectable so
                tests never depend on the developer's real shell.
            model_override: Model id from the command line, which wins over the
                environment.

        Raises:
            WriterConfigurationError: If a required value is missing or unusable.
                The message names the setting, never its value.
        """
        source = os.environ if env is None else env

        api_key = (source.get(API_KEY_ENV) or "").strip()
        if not api_key:
            raise WriterConfigurationError(
                f"{API_KEY_ENV} is not set; the writer cannot reach the provider",
                setting=API_KEY_ENV,
            )

        model = (model_override or source.get(MODEL_ENV) or DEFAULT_MODEL).strip()
        if not model:
            raise WriterConfigurationError(
                f"{MODEL_ENV} is set but empty; unset it to use the default", setting=MODEL_ENV
            )

        return cls(
            api_key=api_key,
            model=model,
            timeout_seconds=_positive_float(source, TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS),
            max_retries=_non_negative_int(source, MAX_RETRIES_ENV, DEFAULT_MAX_RETRIES),
            max_tokens=_positive_int(source, MAX_TOKENS_ENV, DEFAULT_MAX_TOKENS),
        )


@dataclass(frozen=True)
class ReviewerSettings:
    """Everything the reviewer stage needs in order to call a provider.

    Deliberately a separate type from :class:`WriterSettings` rather than a
    shared one with a provider flag: the two stages use different vendors, and
    a single object holding both credentials would be one object too easy to
    log by accident.
    """

    api_key: str = field(repr=False)
    model: str = DEFAULT_REVIEW_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    max_output_tokens: int = DEFAULT_MAX_TOKENS

    def __repr__(self) -> str:
        """Render without the credential."""
        return (
            f"ReviewerSettings(api_key={_REDACTED}, model={self.model!r}, "
            f"timeout_seconds={self.timeout_seconds}, max_retries={self.max_retries}, "
            f"max_output_tokens={self.max_output_tokens})"
        )

    __str__ = __repr__

    @classmethod
    def from_env(
        cls,
        env: dict[str, str] | None = None,
        *,
        model_override: str | None = None,
    ) -> ReviewerSettings:
        """Build settings from the environment.

        Raises:
            ReviewConfigurationError: If a required value is missing or
                unusable. The message names the setting, never its value.
        """
        source = os.environ if env is None else env

        api_key = (source.get(REVIEW_API_KEY_ENV) or "").strip()
        if not api_key:
            raise ReviewConfigurationError(
                f"{REVIEW_API_KEY_ENV} is not set; the reviewer cannot reach the provider",
                setting=REVIEW_API_KEY_ENV,
            )

        model = (model_override or source.get(REVIEW_MODEL_ENV) or DEFAULT_REVIEW_MODEL).strip()
        if not model:
            raise ReviewConfigurationError(
                f"{REVIEW_MODEL_ENV} is set but empty; unset it to use the default",
                setting=REVIEW_MODEL_ENV,
            )

        return cls(
            api_key=api_key,
            model=model,
            timeout_seconds=_positive_float(
                source, REVIEW_TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS, ReviewConfigurationError
            ),
            max_retries=_non_negative_int(
                source, REVIEW_MAX_RETRIES_ENV, DEFAULT_MAX_RETRIES, ReviewConfigurationError
            ),
            max_output_tokens=_positive_int(
                source, REVIEW_MAX_TOKENS_ENV, DEFAULT_MAX_TOKENS, ReviewConfigurationError
            ),
        )


@dataclass(frozen=True)
class FinalizerSettings:
    """Everything the finalizer stage needs in order to call a provider.

    Shares ``ANTHROPIC_API_KEY`` with the writer - it is the same vendor and the
    same account - but resolves its model separately, so a cheaper or stricter
    model can be used for editing than for drafting without forcing the choice.
    ``ANTHROPIC_FINALIZER_MODEL`` wins; failing that ``ANTHROPIC_MODEL``; failing
    that the default.
    """

    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    max_tokens: int = DEFAULT_MAX_TOKENS

    def __repr__(self) -> str:
        """Render without the credential."""
        return (
            f"FinalizerSettings(api_key={_REDACTED}, model={self.model!r}, "
            f"timeout_seconds={self.timeout_seconds}, max_retries={self.max_retries}, "
            f"max_tokens={self.max_tokens})"
        )

    __str__ = __repr__

    @classmethod
    def from_env(
        cls,
        env: dict[str, str] | None = None,
        *,
        model_override: str | None = None,
    ) -> FinalizerSettings:
        """Build settings from the environment.

        Raises:
            FinalizeConfigurationError: If a required value is missing or
                unusable. The message names the setting, never its value.
        """
        source = os.environ if env is None else env

        api_key = (source.get(API_KEY_ENV) or "").strip()
        if not api_key:
            raise FinalizeConfigurationError(
                f"{API_KEY_ENV} is not set; the finalizer cannot reach the provider",
                setting=API_KEY_ENV,
            )

        model = (
            model_override
            or source.get(FINALIZER_MODEL_ENV)
            or source.get(MODEL_ENV)
            or DEFAULT_MODEL
        ).strip()
        if not model:
            raise FinalizeConfigurationError(
                f"{FINALIZER_MODEL_ENV} is set but empty; unset it to use the default",
                setting=FINALIZER_MODEL_ENV,
            )

        return cls(
            api_key=api_key,
            model=model,
            timeout_seconds=_positive_float(
                source, FINALIZER_TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS, FinalizeConfigurationError
            ),
            max_retries=_non_negative_int(
                source,
                FINALIZER_MAX_RETRIES_ENV,
                DEFAULT_MAX_RETRIES,
                FinalizeConfigurationError,
            ),
            max_tokens=_positive_int(
                source, FINALIZER_MAX_TOKENS_ENV, DEFAULT_MAX_TOKENS, FinalizeConfigurationError
            ),
        )


def _raw(source: dict[str, str] | os._Environ[str], name: str) -> str | None:
    value = source.get(name)
    return value.strip() if value and value.strip() else None


ConfigError = (
    type[WriterConfigurationError]
    | type[ReviewConfigurationError]
    | type[FinalizeConfigurationError]
)
"""Which configuration error a helper should raise.

The validation rules are identical for both stages; only the error class and
the setting name differ, so the caller supplies the class rather than the code
being duplicated.
"""


def _positive_float(
    source: dict[str, str] | os._Environ[str],
    name: str,
    default: float,
    error: ConfigError = WriterConfigurationError,
) -> float:
    raw = _raw(source, name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise error(f"{name} must be a number of seconds", setting=name) from None
    if value <= 0:
        raise error(f"{name} must be greater than zero", setting=name)
    return value


def _positive_int(
    source: dict[str, str] | os._Environ[str],
    name: str,
    default: int,
    error: ConfigError = WriterConfigurationError,
) -> int:
    value = _non_negative_int(source, name, default, error)
    if value <= 0:
        raise error(f"{name} must be greater than zero", setting=name)
    return value


def _non_negative_int(
    source: dict[str, str] | os._Environ[str],
    name: str,
    default: int,
    error: ConfigError = WriterConfigurationError,
) -> int:
    raw = _raw(source, name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise error(f"{name} must be a whole number", setting=name) from None
    if value < 0:
        raise error(f"{name} must not be negative", setting=name)
    return value


__all__ = [
    "API_KEY_ENV",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_REVIEW_MODEL",
    "DEFAULT_TIMEOUT_SECONDS",
    "FINALIZER_MODEL_ENV",
    "MODEL_ENV",
    "REVIEW_API_KEY_ENV",
    "REVIEW_MODEL_ENV",
    "FinalizerSettings",
    "ReviewerSettings",
    "WriterSettings",
]
