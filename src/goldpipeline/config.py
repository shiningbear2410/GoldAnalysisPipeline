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
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from goldpipeline.adapters.secrets import SecretProvider, default_provider
from goldpipeline.domain.errors import (
    AutomationConfigurationError,
    FinalizeConfigurationError,
    MarketDataConfigurationError,
    PublisherConfigurationError,
    ReviewConfigurationError,
    WriterConfigurationError,
)
from goldpipeline.schemas.common import normalize_symbol
from goldpipeline.schemas.secrets import SecretName

DEFAULT_MODEL = "claude-opus-5"
"""Writer model used when ``ANTHROPIC_MODEL`` is not set."""

DEFAULT_REVIEWER_MODEL = DEFAULT_MODEL
"""Reviewer model used when ``ANTHROPIC_REVIEWER_MODEL`` is not set.

Same model as the Writer by default, and a separate setting on purpose: the two
stages are independent judgements, and an operator may well want the reviewer on
a different model without touching the writer.
"""

DEFAULT_REVIEW_MODEL = "gpt-5.1"
"""Legacy OpenAI reviewer default. Retained for the optional legacy adapter."""

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

REVIEW_ENABLED_ENV = "GOLDPIPELINE_TELEGRAM_REVIEW_ENABLED"
REVIEW_CHAT_ENV = "GOLDPIPELINE_TELEGRAM_REVIEW_CHAT_ID"
REVIEW_MAX_RUN_AGE_ENV = "GOLDPIPELINE_TELEGRAM_REVIEW_MAX_RUN_AGE_MINUTES"

DEFAULT_REVIEW_MAX_RUN_AGE_MINUTES = 60
"""How old an approved Run may be and still be worth reviewing.

Doubles as the backlog guard. Enabling review delivery on a machine that
already holds finished Runs must not post a week of old articles at once, and
an age limit expresses that without a separate activation marker to keep in
sync. It is also the right rule on its own terms: an article about candles from
this morning is not worth reading this evening.
"""

TELEGRAM_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
TELEGRAM_TARGET_ENV = "TELEGRAM_TARGET_CHAT_ID"
TELEGRAM_TIMEOUT_ENV = "GOLDPIPELINE_PUBLISH_TIMEOUT"

DEFAULT_PUBLISH_TIMEOUT_SECONDS = 30.0
"""Finite by construction. An unbounded publish call is how a process hangs
holding an uncommitted publish intent."""

REVIEWER_MODEL_ENV = "ANTHROPIC_REVIEWER_MODEL"
"""Which model reviews the draft. Non-secret, like every other model setting."""

REVIEW_API_KEY_ENV = "OPENAI_API_KEY"
"""Legacy. No longer read by production; kept so the legacy adapter's messages
can still name the setting an operator would have configured."""

REVIEW_MODEL_ENV = "OPENAI_REVIEW_MODEL"
"""Legacy, as above."""
REVIEW_TIMEOUT_ENV = "GOLDPIPELINE_REVIEW_TIMEOUT"
REVIEW_MAX_RETRIES_ENV = "GOLDPIPELINE_REVIEW_MAX_RETRIES"
REVIEW_MAX_TOKENS_ENV = "GOLDPIPELINE_REVIEW_MAX_TOKENS"

_REDACTED = "***redacted***"


def _secret(
    name: SecretName,
    secrets: SecretProvider | None,
    env: Mapping[str, str] | None,
) -> str | None:
    """Resolve one credential through the provider the caller supplied.

    Defaults to environment-only, which is what every round before 9.1 did.
    Reaching the operating system's credential store is something a caller
    opts into by passing a composite provider, so importing this module
    never touches a vault.
    """
    provider = secrets if secrets is not None else default_provider(env)
    return provider.get_secret(name)


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
        env: Mapping[str, str] | None = None,
        *,
        model_override: str | None = None,
        secrets: SecretProvider | None = None,
    ) -> WriterSettings:
        """Build settings from the environment and a credential provider.

        Args:
            env: Mapping for non-secret values. Defaults to ``os.environ``;
                injectable so tests never depend on the developer's real shell.
            model_override: Model id from the command line, which wins over the
                environment.
            secrets: Where the API key comes from. Defaults to the process
                environment alone; the CLI passes a provider that falls back to
                the operating system's credential store.

        Raises:
            WriterConfigurationError: If a required value is missing or unusable.
                The message names the setting, never its value.
        """
        source = os.environ if env is None else env

        api_key = _secret(SecretName.ANTHROPIC_API_KEY, secrets, env)
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

    Still a separate type from :class:`WriterSettings` even though both stages
    now call Anthropic on the same account. The reviewer has its own model, its
    own timeout and its own token budget, and merging the two would make it
    fractionally easier to accidentally review a draft with the writer's
    settings - which is the one thing the review stage exists to avoid.
    """

    api_key: str = field(repr=False)
    model: str = DEFAULT_REVIEWER_MODEL
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
        env: Mapping[str, str] | None = None,
        *,
        model_override: str | None = None,
        secrets: SecretProvider | None = None,
    ) -> ReviewerSettings:
        """Build settings from the environment and a credential provider.

        Raises:
            ReviewConfigurationError: If a required value is missing or
                unusable. The message names the setting, never its value.
        """
        source = os.environ if env is None else env

        # Anthropic since Round 9.3.1. The reviewer is still an entirely
        # separate request with its own prompt and its own schema; what changed
        # is only which account answers it, and that this pipeline no longer
        # obliges an operator to hold a second vendor's credential.
        api_key = _secret(SecretName.ANTHROPIC_API_KEY, secrets, env)
        if not api_key:
            raise ReviewConfigurationError(
                f"{API_KEY_ENV} is not set; the reviewer cannot reach the provider",
                setting=API_KEY_ENV,
            )

        model = (model_override or source.get(REVIEWER_MODEL_ENV) or DEFAULT_REVIEWER_MODEL).strip()
        if not model:
            raise ReviewConfigurationError(
                f"{REVIEWER_MODEL_ENV} is set but empty; unset it to use the default",
                setting=REVIEWER_MODEL_ENV,
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
        env: Mapping[str, str] | None = None,
        *,
        model_override: str | None = None,
        secrets: SecretProvider | None = None,
    ) -> FinalizerSettings:
        """Build settings from the environment and a credential provider.

        Shares the writer's credential - same vendor, same account - so it
        resolves the same secret name through the same provider.

        Raises:
            FinalizeConfigurationError: If a required value is missing or
                unusable. The message names the setting, never its value.
        """
        source = os.environ if env is None else env

        api_key = _secret(SecretName.ANTHROPIC_API_KEY, secrets, env)
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
class ReviewDeliverySettings:
    """Where an approved-but-unpublished article is sent for a human to read.

    Deliberately its own type rather than a flag on :class:`TelegramSettings`.
    Review delivery and publishing share a transport and nothing else: they have
    different destinations, different artifacts, different triggers, and only one
    of them changes a Run's status. A single settings object holding both
    destinations would make "post to the review chat" and "publish to the
    channel" one typo apart.

    The token is not here for the same reason it is not in the config file: it
    is a credential, and it comes from the credential store at the point of use.
    """

    enabled: bool = False
    chat_id: str = ""
    max_run_age_minutes: int = DEFAULT_REVIEW_MAX_RUN_AGE_MINUTES

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ReviewDeliverySettings:
        """Build settings from non-secret configuration.

        Raises:
            PublisherConfigurationError: Review delivery is on but the
                destination is missing or is not a shape Telegram accepts.
                Named, never valued.
        """
        source = os.environ if env is None else env
        enabled = _flag(source, REVIEW_ENABLED_ENV)
        raw = _raw(source, REVIEW_CHAT_ENV) or ""

        chat_id = raw.strip()
        if enabled and not chat_id:
            # Fail closed. A review delivery with no destination must never
            # borrow the publish target: those are different audiences, and
            # one of them is a public channel.
            raise PublisherConfigurationError(
                f"{REVIEW_ENABLED_ENV} is on but {REVIEW_CHAT_ENV} is not set; "
                "review delivery never falls back to the publish destination",
                setting=REVIEW_CHAT_ENV,
            )
        if chat_id:
            chat_id = validate_target_chat(chat_id)

        return cls(
            enabled=enabled,
            chat_id=chat_id,
            max_run_age_minutes=_positive_int(
                source,
                REVIEW_MAX_RUN_AGE_ENV,
                DEFAULT_REVIEW_MAX_RUN_AGE_MINUTES,
                PublisherConfigurationError,
            ),
        )


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
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        secrets: SecretProvider | None = None,
    ) -> TelegramSettings:
        """Build settings from the environment and a credential provider.

        The token is a credential and comes from *secrets*. The destination is
        not: knowing which channel the pipeline posts to grants nobody the
        ability to post there, and keeping it in the environment is what lets
        an operator read it back and check it against the allowlist.

        Raises:
            PublisherConfigurationError: If a required value is missing or the
                destination is not a shape Telegram accepts. The message names
                the setting, never its value.
        """
        source = os.environ if env is None else env

        token = _secret(SecretName.TELEGRAM_BOT_TOKEN, secrets, env)
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
    def from_env(cls, env: Mapping[str, str] | None = None) -> MarketDataSettings:
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


AUTOMATION_ENABLED_ENV = "GOLDPIPELINE_AUTOMATION_ENABLED"
AUTOMATION_DIR_ENV = "GOLDPIPELINE_AUTOMATION_DIR"
MAX_EVENTS_PER_TICK_ENV = "GOLDPIPELINE_AUTOMATION_MAX_EVENTS_PER_TICK"
MAX_TICK_MINUTES_ENV = "GOLDPIPELINE_AUTOMATION_MAX_TICK_MINUTES"
MAX_EVENT_AGE_ENV = "GOLDPIPELINE_MAX_ANALYSIS_EVENT_AGE_MINUTES"
DEFER_RETRY_ENV = "GOLDPIPELINE_DEFER_RETRY_MINUTES"
AUTOPUBLISH_ENABLED_ENV = "GOLDPIPELINE_AUTOPUBLISH_ENABLED"
AUTOPUBLISH_TARGET_ENV = "GOLDPIPELINE_AUTOPUBLISH_ALLOWED_TARGET"
AUTOPUBLISH_MAX_AGE_ENV = "GOLDPIPELINE_AUTOPUBLISH_MAX_RUN_AGE_MINUTES"

DEFAULT_MAX_EVENTS_PER_TICK = 3
MIN_EVENTS_PER_TICK = 1
MAX_EVENTS_PER_TICK = 20
"""A tick does a bounded amount of work and exits.

Draining an inbox of fifty events in one invocation would hold the worker lock
for an hour and spend an unbounded amount on providers before anyone could look
at the first result.
"""

DEFAULT_MAX_TICK_MINUTES = 10
DEFAULT_MAX_EVENT_AGE_MINUTES = 60
"""How old an analysis may be before it is too late to write about it.

Deliberately separate from - and shorter than - the market data limit. They
answer different questions: ``MAX_DATA_AGE`` asks whether the *candles* are
current, and this asks whether the *analyst's note* still describes the market
anyone is looking at. One hour is conservative for intraday M15 commentary; the
failure it exists to prevent is a Saturday note waiting in the queue and being
paired with Monday's opening bars.
"""

DEFAULT_DEFER_RETRY_MINUTES = 5
"""How long a deferred event waits before the worker looks at it again.

The scheduler fires every minute. Retrying a market that was closed sixty
seconds ago just burns terminal round-trips to learn the same thing.
"""

DEFAULT_AUTOPUBLISH_MAX_RUN_AGE_MINUTES = 30
"""How old an approved Run may be and still be published unattended.

This is the guard against the worst automation accident available here: someone
enables auto-publish and a backlog of last week's approved articles goes out at
once. An article older than this is left for a human, who can still publish it
deliberately with the `publish` command.
"""


def _flag(source: Mapping[str, str], name: str, default: bool = False) -> bool:
    """Read a boolean setting.

    Only an explicit affirmative turns something on. Anything unrecognised -
    including ``"yes please"``, an empty string, or a typo - reads as off,
    because every flag here defaults to the safer answer.
    """
    raw = _raw(source, name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AutomationSettings:
    """What the scheduled worker is allowed to do, and how much of it.

    Holds no credential. Every field here answers a question about *scope* -
    how much work, for how long, how old is too old, and whether publishing is
    authorised - and none of them can be influenced by a payload: they come from
    the process environment, which a producer cannot write to.
    """

    enabled: bool = False
    """Whether the *scheduled* entry point does anything.

    Defaults off so that registering the task is not the same act as switching
    the system on. `automation-run-once` ignores this; it is a person typing a
    command, which is its own authorisation.
    """

    automation_dir: Path = Path("automation")
    max_events_per_tick: int = DEFAULT_MAX_EVENTS_PER_TICK
    max_tick_minutes: int = DEFAULT_MAX_TICK_MINUTES
    max_event_age_minutes: int = DEFAULT_MAX_EVENT_AGE_MINUTES
    defer_retry_minutes: int = DEFAULT_DEFER_RETRY_MINUTES

    auto_publish_enabled: bool = False
    auto_publish_allowed_target: str | None = None
    auto_publish_max_run_age_minutes: int = DEFAULT_AUTOPUBLISH_MAX_RUN_AGE_MINUTES

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> AutomationSettings:
        """Build settings from the environment.

        Raises:
            AutomationConfigurationError: If a value is present but unusable.
        """
        source = os.environ if env is None else env
        error = AutomationConfigurationError

        events = _positive_int(source, MAX_EVENTS_PER_TICK_ENV, DEFAULT_MAX_EVENTS_PER_TICK, error)
        if not MIN_EVENTS_PER_TICK <= events <= MAX_EVENTS_PER_TICK:
            raise error(
                f"{MAX_EVENTS_PER_TICK_ENV} must be between {MIN_EVENTS_PER_TICK} and "
                f"{MAX_EVENTS_PER_TICK}",
                setting=MAX_EVENTS_PER_TICK_ENV,
            )

        target = _raw(source, AUTOPUBLISH_TARGET_ENV)
        if target is not None:
            # Validated the same way the publisher validates its destination, so
            # an allowlist entry that could never match anything fails loudly
            # here rather than silently blocking every tick.
            target = validate_target_chat(target)

        return cls(
            enabled=_flag(source, AUTOMATION_ENABLED_ENV),
            automation_dir=Path(_raw(source, AUTOMATION_DIR_ENV) or "automation"),
            max_events_per_tick=events,
            max_tick_minutes=_positive_int(
                source, MAX_TICK_MINUTES_ENV, DEFAULT_MAX_TICK_MINUTES, error
            ),
            max_event_age_minutes=_positive_int(
                source, MAX_EVENT_AGE_ENV, DEFAULT_MAX_EVENT_AGE_MINUTES, error
            ),
            defer_retry_minutes=_positive_int(
                source, DEFER_RETRY_ENV, DEFAULT_DEFER_RETRY_MINUTES, error
            ),
            auto_publish_enabled=_flag(source, AUTOPUBLISH_ENABLED_ENV),
            auto_publish_allowed_target=target,
            auto_publish_max_run_age_minutes=_positive_int(
                source,
                AUTOPUBLISH_MAX_AGE_ENV,
                DEFAULT_AUTOPUBLISH_MAX_RUN_AGE_MINUTES,
                error,
            ),
        )


def inbox_dir_from_env(env: Mapping[str, str] | None = None) -> Path:
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


def _raw(source: Mapping[str, str], name: str) -> str | None:
    value = source.get(name)
    return value.strip() if value and value.strip() else None


ConfigError = (
    type[WriterConfigurationError]
    | type[ReviewConfigurationError]
    | type[FinalizeConfigurationError]
    | type[PublisherConfigurationError]
    | type[MarketDataConfigurationError]
    | type[AutomationConfigurationError]
)
"""Which configuration error a helper should raise.

The validation rules are identical for both stages; only the error class and
the setting name differ, so the caller supplies the class rather than the code
being duplicated.
"""


def _positive_float(
    source: Mapping[str, str],
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
    source: Mapping[str, str],
    name: str,
    default: int,
    error: ConfigError = WriterConfigurationError,
) -> int:
    value = _non_negative_int(source, name, default, error)
    if value <= 0:
        raise error(f"{name} must be greater than zero", setting=name)
    return value


def _non_negative_int(
    source: Mapping[str, str],
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
    "AUTOMATION_DIR_ENV",
    "AUTOMATION_ENABLED_ENV",
    "AUTOPUBLISH_ENABLED_ENV",
    "AUTOPUBLISH_MAX_AGE_ENV",
    "AUTOPUBLISH_TARGET_ENV",
    "AutomationSettings",
    "BARS_ENV",
    "CANONICAL_SYMBOL_ENV",
    "DEFAULT_AUTOPUBLISH_MAX_RUN_AGE_MINUTES",
    "DEFAULT_BAR_COUNT",
    "DEFAULT_DEFER_RETRY_MINUTES",
    "DEFAULT_MAX_DATA_AGE_MINUTES",
    "DEFAULT_MAX_EVENTS_PER_TICK",
    "DEFAULT_MAX_EVENT_AGE_MINUTES",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_TICK_MINUTES",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_MT5_SYMBOL",
    "DEFAULT_PUBLISH_TIMEOUT_SECONDS",
    "DEFAULT_REVIEWER_MODEL",
    "DEFAULT_REVIEW_MAX_RUN_AGE_MINUTES",
    "REVIEW_CHAT_ENV",
    "REVIEW_ENABLED_ENV",
    "REVIEW_MAX_RUN_AGE_ENV",
    "ReviewDeliverySettings",
    "DEFAULT_REVIEW_MODEL",
    "REVIEWER_MODEL_ENV",
    "DEFAULT_TIMEFRAME",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFER_RETRY_ENV",
    "FINALIZER_MODEL_ENV",
    "FinalizerSettings",
    "INBOX_DIR_ENV",
    "MAX_BAR_COUNT",
    "MAX_DATA_AGE_ENV",
    "MAX_EVENTS_PER_TICK",
    "MAX_EVENTS_PER_TICK_ENV",
    "MAX_EVENT_AGE_ENV",
    "MAX_TICK_MINUTES_ENV",
    "MIN_BAR_COUNT",
    "MIN_EVENTS_PER_TICK",
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
