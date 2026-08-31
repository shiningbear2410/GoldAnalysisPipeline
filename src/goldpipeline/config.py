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
import re
from dataclasses import dataclass, field
from pathlib import Path

from goldpipeline.domain.errors import (
    FinalizeConfigurationError,
    MarketDataConfigurationError,
    PublisherConfigurationError,
    ReviewConfigurationError,
    WriterConfigurationError,
)
from goldpipeline.schemas.common import normalize_symbol

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

TELEGRAM_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
TELEGRAM_TARGET_ENV = "TELEGRAM_TARGET_CHAT_ID"
TELEGRAM_TIMEOUT_ENV = "GOLDPIPELINE_PUBLISH_TIMEOUT"

DEFAULT_PUBLISH_TIMEOUT_SECONDS = 30.0
"""Finite by construction. An unbounded publish call is how a process hangs
holding an uncommitted publish intent."""

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


_CHAT_USERNAME = re.compile(r"^@[A-Za-z][A-Za-z0-9_]{4,31}$")
_CHAT_NUMERIC = re.compile(r"^-?\d{1,19}$")


@dataclass(frozen=True)
class TelegramSettings:
    """Everything the publisher needs in order to post.

    The destination lives here and nowhere else. It comes from the environment,
    is validated before use, and is never taken from article text, the analyst's
    note, or any model output - a pipeline whose destination could be influenced
    by its own content would be one prompt away from posting to a stranger's
    channel.
    """

    bot_token: str = field(repr=False)
    target_chat: str = ""
    timeout_seconds: float = DEFAULT_PUBLISH_TIMEOUT_SECONDS

    def __repr__(self) -> str:
        """Render without the token.

        The Telegram API embeds the token in the request path, so a leaked
        settings object is a leaked credential.
        """
        return (
            f"TelegramSettings(bot_token={_REDACTED}, target_chat={self.target_chat!r}, "
            f"timeout_seconds={self.timeout_seconds})"
        )

    __str__ = __repr__

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> TelegramSettings:
        """Build settings from the environment.

        Raises:
            PublisherConfigurationError: If a required value is missing or the
                destination is not a shape Telegram accepts. The message names
                the setting, never its value.
        """
        source = os.environ if env is None else env

        token = (source.get(TELEGRAM_TOKEN_ENV) or "").strip()
        if not token:
            raise PublisherConfigurationError(
                f"{TELEGRAM_TOKEN_ENV} is not set; the publisher cannot reach Telegram",
                setting=TELEGRAM_TOKEN_ENV,
            )

        target = (source.get(TELEGRAM_TARGET_ENV) or "").strip()
        if not target:
            raise PublisherConfigurationError(
                f"{TELEGRAM_TARGET_ENV} is not set; the publisher has no destination",
                setting=TELEGRAM_TARGET_ENV,
            )

        return cls(
            bot_token=token,
            target_chat=validate_target_chat(target),
            timeout_seconds=_positive_float(
                source,
                TELEGRAM_TIMEOUT_ENV,
                DEFAULT_PUBLISH_TIMEOUT_SECONDS,
                PublisherConfigurationError,
            ),
        )


MT5_SYMBOL_ENV = "GOLDPIPELINE_MT5_SYMBOL"
CANONICAL_SYMBOL_ENV = "GOLDPIPELINE_CANONICAL_SYMBOL"
TIMEFRAME_ENV = "GOLDPIPELINE_OHLC_TIMEFRAME"
BARS_ENV = "GOLDPIPELINE_OHLC_BARS"
MAX_DATA_AGE_ENV = "GOLDPIPELINE_MAX_DATA_AGE_MINUTES"
INBOX_DIR_ENV = "GOLDPIPELINE_INBOX_DIR"

DEFAULT_MT5_SYMBOL = "XAUUSD"
"""Most brokers name gold this. Many do not - hence the setting."""

DEFAULT_TIMEFRAME = "M15"
DEFAULT_BAR_COUNT = 20
MIN_BAR_COUNT = 10
MAX_BAR_COUNT = 500
"""A ceiling, so no configuration mistake asks a terminal for a million candles."""

DEFAULT_MAX_DATA_AGE_MINUTES = 90
"""How old the latest closed candle may be before ingestion refuses.

Conservative rather than clever. Gold trades nearly around the clock on
weekdays, so on a working market this is generous; over a weekend it will refuse,
which is the right answer - an article quoting Friday's close on Sunday evening
is stale whether or not the market being shut explains it. Raise it deliberately
for a backfill; there is no market calendar here and there should not be one yet.
"""

SUPPORTED_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4")
"""Timeframes this pipeline will fetch.

A small whitelist on purpose. The provider's own timeframe constants are opaque
integers, and accepting an arbitrary one from configuration - let alone from a
payload - would mean fetching a resolution nothing downstream was written for.
"""


@dataclass(frozen=True)
class MarketDataSettings:
    """Which instrument to fetch, at what resolution, and how much of it.

    No credential lives here. Reading candles from an already-authenticated
    terminal needs no login, and this pipeline never performs one.

    ``provider_symbol`` and ``canonical_symbol`` are separate because brokers
    rename things: gold may be ``XAUUSDm``, ``GOLD`` or ``XAUUSD.a`` on the
    terminal while the article, the context and every downstream check still
    talk about ``XAUUSD``. Both are recorded, and neither is inferred from the
    other.
    """

    provider_symbol: str = DEFAULT_MT5_SYMBOL
    canonical_symbol: str = DEFAULT_MT5_SYMBOL
    timeframe: str = DEFAULT_TIMEFRAME
    bar_count: int = DEFAULT_BAR_COUNT
    max_data_age_minutes: int = DEFAULT_MAX_DATA_AGE_MINUTES

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> MarketDataSettings:
        """Build settings from the environment.

        Raises:
            MarketDataConfigurationError: If a value is missing or unusable.
        """
        source = os.environ if env is None else env

        provider_symbol = (source.get(MT5_SYMBOL_ENV) or DEFAULT_MT5_SYMBOL).strip()
        if not provider_symbol:
            raise MarketDataConfigurationError(
                f"{MT5_SYMBOL_ENV} is set but empty; unset it to use the default",
                setting=MT5_SYMBOL_ENV,
            )

        canonical = (source.get(CANONICAL_SYMBOL_ENV) or DEFAULT_MT5_SYMBOL).strip()
        try:
            canonical_symbol = normalize_symbol(canonical)
        except ValueError as exc:
            raise MarketDataConfigurationError(
                f"{CANONICAL_SYMBOL_ENV} is not a valid instrument symbol",
                setting=CANONICAL_SYMBOL_ENV,
            ) from exc

        timeframe = (source.get(TIMEFRAME_ENV) or DEFAULT_TIMEFRAME).strip().upper()
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise MarketDataConfigurationError(
                f"{TIMEFRAME_ENV} must be one of {', '.join(SUPPORTED_TIMEFRAMES)}",
                setting=TIMEFRAME_ENV,
                supported=list(SUPPORTED_TIMEFRAMES),
            )

        bar_count = _positive_int(source, BARS_ENV, DEFAULT_BAR_COUNT, MarketDataConfigurationError)
        if not MIN_BAR_COUNT <= bar_count <= MAX_BAR_COUNT:
            raise MarketDataConfigurationError(
                f"{BARS_ENV} must be between {MIN_BAR_COUNT} and {MAX_BAR_COUNT}",
                setting=BARS_ENV,
                minimum=MIN_BAR_COUNT,
                maximum=MAX_BAR_COUNT,
            )

        return cls(
            provider_symbol=provider_symbol,
            canonical_symbol=canonical_symbol,
            timeframe=timeframe,
            bar_count=bar_count,
            max_data_age_minutes=_positive_int(
                source,
                MAX_DATA_AGE_ENV,
                DEFAULT_MAX_DATA_AGE_MINUTES,
                MarketDataConfigurationError,
            ),
        )


def inbox_dir_from_env(env: dict[str, str] | None = None) -> Path:
    """Where the analysis inbox lives. Configuration only, never a payload."""
    source = os.environ if env is None else env
    return Path((source.get(INBOX_DIR_ENV) or "inbox").strip() or "inbox")


def validate_target_chat(target: str) -> str:
    """Check a destination is a shape Telegram recognises, and return it canonically.

    Two accepted forms: a public ``@username`` and a numeric chat id, which is
    negative for channels and supergroups. The numeric form is kept as a
    *string* throughout - a channel id like ``-1002145890733`` exceeds what a
    float represents exactly, and rounding a destination is a way to post
    somewhere unintended.

    Raises:
        PublisherConfigurationError: If the target is neither form.
    """
    cleaned = target.strip()
    if _CHAT_USERNAME.fullmatch(cleaned) or _CHAT_NUMERIC.fullmatch(cleaned):
        return cleaned
    raise PublisherConfigurationError(
        f"{TELEGRAM_TARGET_ENV} must be an @username or a numeric chat id",
        setting=TELEGRAM_TARGET_ENV,
    )


def _raw(source: dict[str, str] | os._Environ[str], name: str) -> str | None:
    value = source.get(name)
    return value.strip() if value and value.strip() else None


ConfigError = (
    type[WriterConfigurationError]
    | type[ReviewConfigurationError]
    | type[FinalizeConfigurationError]
    | type[PublisherConfigurationError]
    | type[MarketDataConfigurationError]
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
    "BARS_ENV",
    "CANONICAL_SYMBOL_ENV",
    "DEFAULT_BAR_COUNT",
    "DEFAULT_MAX_DATA_AGE_MINUTES",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_MT5_SYMBOL",
    "DEFAULT_PUBLISH_TIMEOUT_SECONDS",
    "DEFAULT_REVIEW_MODEL",
    "DEFAULT_TIMEFRAME",
    "DEFAULT_TIMEOUT_SECONDS",
    "FINALIZER_MODEL_ENV",
    "FinalizerSettings",
    "INBOX_DIR_ENV",
    "MAX_BAR_COUNT",
    "MAX_DATA_AGE_ENV",
    "MIN_BAR_COUNT",
    "MODEL_ENV",
    "MT5_SYMBOL_ENV",
    "MarketDataSettings",
    "REVIEW_API_KEY_ENV",
    "REVIEW_MODEL_ENV",
    "ReviewerSettings",
    "SUPPORTED_TIMEFRAMES",
    "TELEGRAM_TARGET_ENV",
    "TELEGRAM_TOKEN_ENV",
    "TIMEFRAME_ENV",
    "TelegramSettings",
    "WriterSettings",
    "inbox_dir_from_env",
    "validate_target_chat",
]
