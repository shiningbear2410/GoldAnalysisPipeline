"""Persistent non-secret configuration.

The companion to Round 9.1's credential store, and deliberately a *separate*
concept. A scheduled task needs two different things to survive into a fresh
process, and they want opposite treatment:

* **credentials** must be unreadable by anything but the user, so they live in
  the operating system's encrypted store;
* **configuration** must be readable *by a person*, so an operator can check
  which channel is allowlisted, what symbol is being fetched, and whether
  automation is on. Hiding those in a vault would make them harder to audit
  without making anything safer.

So this store holds only non-secrets, and the three credential names are refused
by name rather than merely absent from the schema - the difference being that a
refusal says why.

**Why not User environment variables.** ``setx`` writes to the registry, where
the value is visible to every process the user runs and to any support
screenshot, and where a typo persists silently forever. A file under
``%LOCALAPPDATA%`` is inspectable, editable, atomically replaceable, and can be
deleted to reset the machine to defaults.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from goldpipeline.schemas.common import StrictModel

RUNTIME_CONFIG_SCHEMA_VERSION = "1.0.0"

APP_DIR_NAME = "GoldAnalysisPipeline"
CONFIG_FILENAME = "config.json"


class ConfigKey(StrEnum):
    """Every setting that may be persisted. A whitelist, not a suggestion.

    Named for the environment variable each corresponds to, so the persisted
    file, ``.env.example`` and a status line all use the same word for the same
    thing.
    """

    TELEGRAM_TARGET_CHAT_ID = "TELEGRAM_TARGET_CHAT_ID"
    MT5_SYMBOL = "GOLDPIPELINE_MT5_SYMBOL"
    CANONICAL_SYMBOL = "GOLDPIPELINE_CANONICAL_SYMBOL"
    OHLC_TIMEFRAME = "GOLDPIPELINE_OHLC_TIMEFRAME"
    OHLC_BARS = "GOLDPIPELINE_OHLC_BARS"
    MAX_DATA_AGE_MINUTES = "GOLDPIPELINE_MAX_DATA_AGE_MINUTES"
    MAX_ANALYSIS_EVENT_AGE_MINUTES = "GOLDPIPELINE_MAX_ANALYSIS_EVENT_AGE_MINUTES"
    DEFER_RETRY_MINUTES = "GOLDPIPELINE_DEFER_RETRY_MINUTES"
    AUTOMATION_MAX_EVENTS_PER_TICK = "GOLDPIPELINE_AUTOMATION_MAX_EVENTS_PER_TICK"
    AUTOMATION_MAX_TICK_MINUTES = "GOLDPIPELINE_AUTOMATION_MAX_TICK_MINUTES"
    AUTOMATION_ENABLED = "GOLDPIPELINE_AUTOMATION_ENABLED"
    AUTOPUBLISH_ENABLED = "GOLDPIPELINE_AUTOPUBLISH_ENABLED"
    AUTOPUBLISH_ALLOWED_TARGET = "GOLDPIPELINE_AUTOPUBLISH_ALLOWED_TARGET"
    AUTOPUBLISH_MAX_RUN_AGE_MINUTES = "GOLDPIPELINE_AUTOPUBLISH_MAX_RUN_AGE_MINUTES"
    TELEGRAM_REVIEW_ENABLED = "GOLDPIPELINE_TELEGRAM_REVIEW_ENABLED"
    TELEGRAM_REVIEW_CHAT_ID = "GOLDPIPELINE_TELEGRAM_REVIEW_CHAT_ID"
    TELEGRAM_REVIEW_MAX_RUN_AGE_MINUTES = "GOLDPIPELINE_TELEGRAM_REVIEW_MAX_RUN_AGE_MINUTES"


BOOLEAN_KEYS = frozenset(
    {
        ConfigKey.AUTOMATION_ENABLED,
        ConfigKey.AUTOPUBLISH_ENABLED,
        ConfigKey.TELEGRAM_REVIEW_ENABLED,
    }
)
"""Keys whose value must read as an unambiguous true or false.

Parsed strictly here, unlike the environment. A typo in a shell session is a
five-minute puzzle; a typo in a file that persists across reboots quietly means
"off" forever, and the operator who wrote ``GOLDPIPELINE_AUTOMATION_ENABLED=yes``
deserves to be told rather than ignored.
"""

FORBIDDEN_KEYS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "INGEST_TOKEN",
    }
)
"""Credentials, refused by name.

They are already absent from :class:`ConfigKey`, so the schema would reject them
anyway - but "unknown setting" is the wrong message. Someone trying to persist
an API key here needs to be told where it actually goes.
"""

TRUE_VALUES = frozenset({"true", "1", "yes", "on"})
FALSE_VALUES = frozenset({"false", "0", "no", "off"})


REQUIRED_PRODUCTION_KEYS = frozenset(ConfigKey)
"""Every approved setting, all of which a scheduled worker must find explicitly.

Deliberately derived from :class:`ConfigKey` rather than listed again. Two
reasons, and the second is the important one:

* a hand-written second list drifts from the first, and the drift shows up as a
  production setting nobody notices is missing;
* a key added in a later round becomes mandatory by default. That is the safe
  direction to be wrong in. Forgetting to add a key here would silently
  reintroduce exactly the defect this module exists to prevent, whereas an
  over-strict list fails loudly the first time a scheduled tick runs.

Note this is a *production* requirement, not a general one. An operator running
a command by hand still gets built-in defaults; see :class:`ConfigMode`.

**Why the remote-intake settings are not here yet.** ``GOLDPIPELINE_INGEST_*``
is read by :class:`~goldpipeline.config.IngestSettings` but is deliberately
absent from :class:`ConfigKey`, and therefore from this set. Adding a member
makes it mandatory *immediately* - which is the right default, and exactly the
problem: the running production file would become incomplete the moment the code
shipped, and the next scheduled tick would fail closed before anyone could
rewrite it. Remote intake is off by default and reads its settings by name, so
in ``STRICT_PERSISTENT`` the keys are simply absent and the feature stays off.

Turning it on is therefore a deliberate two-step migration, in this order:
add the members here, then rewrite the authoritative file to match, as one
operator action. Until then the feature is reachable only in ``LAYERED`` mode,
which is what a person testing it by hand actually wants.
"""


class ConfigMode(StrEnum):
    """How a process resolved its non-secret configuration.

    Recorded on every tick so an incident does not begin by guessing. The two
    modes answer different questions and are not interchangeable:

    * ``LAYERED`` - process environment, then the file, then built-in defaults.
      Right for a person at a keyboard, who may want to override one value for
      one command without editing the machine's settings.
    * ``STRICT_PERSISTENT`` - the file, in full, or nothing. Right for an
      unattended worker, where a missing value is not a request for a sensible
      default but evidence that something is wrong.
    """

    LAYERED = "LAYERED"
    STRICT_PERSISTENT = "STRICT_PERSISTENT"


class ProductionConfigStatus(StrEnum):
    """Health of the production configuration file, for diagnostics."""

    FOUND = "FOUND"
    MISSING = "MISSING"
    INVALID = "INVALID"


class ConfigSource(StrEnum):
    """Where a resolved setting came from.

    The same distinction the credential status draws, for the same reason: a
    value from this session will not survive into a scheduled task, and a value
    from the persisted file will.
    """

    PROCESS_ENV = "PROCESS_ENV"
    PERSISTENT = "PERSISTENT"
    DEFAULT = "DEFAULT"


class RuntimeConfig(StrictModel):
    """The persisted document.

    ``extra="forbid"`` and a typed mapping, so an unknown key is a refusal at
    load time rather than a value silently ignored at the moment it matters.
    """

    schema_version: str = Field(default=RUNTIME_CONFIG_SCHEMA_VERSION)
    values: dict[ConfigKey, str] = Field(default_factory=dict)

    def get(self, key: ConfigKey) -> str | None:
        return self.values.get(key)


class ProductionConfig(StrictModel):
    """A complete, validated production configuration and the file it came from.

    The fingerprint is what makes a tick record worth reading. "Exit 0" only
    says a process ran; ``sha256`` says *which configuration it ran on*, so a
    scheduler quietly reading a different file - or no file - is visible in the
    history rather than inferred months later.
    """

    path: str
    sha256: str = Field(description="SHA-256 of the file's exact bytes.")
    schema_version: str
    values: dict[ConfigKey, str]

    def as_mapping(self) -> dict[str, str]:
        """The settings as the loaders expect them, keyed by variable name."""
        return {key.value: value for key, value in self.values.items()}


class ProductionConfigReport(StrictModel):
    """Whether unattended operation would find a usable configuration.

    Every field is safe to print: this store holds no credentials, which is the
    reason it is separate from the vault.
    """

    status: ProductionConfigStatus
    path: str | None = None
    schema_version: str | None = None
    sha256: str | None = None
    error_code: str | None = None
    error: str | None = None
    missing_keys: list[str] = Field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.status is ProductionConfigStatus.FOUND


class ConfigEntry(StrictModel):
    """One resolved setting, and where it came from.

    Safe to print in full: nothing here is a credential, which is the entire
    reason this store exists separately from the vault.
    """

    key: ConfigKey
    value: str | None
    source: ConfigSource

    @property
    def summary(self) -> str:
        if self.value is None:
            return "(not set)"
        return f"{self.value}  [{_HUMAN[self.source]}]"


_HUMAN = {
    ConfigSource.PROCESS_ENV: "process environment",
    ConfigSource.PERSISTENT: "persistent app config",
    ConfigSource.DEFAULT: "built-in default",
}


__all__ = [
    "APP_DIR_NAME",
    "BOOLEAN_KEYS",
    "CONFIG_FILENAME",
    "FALSE_VALUES",
    "FORBIDDEN_KEYS",
    "REQUIRED_PRODUCTION_KEYS",
    "RUNTIME_CONFIG_SCHEMA_VERSION",
    "TRUE_VALUES",
    "ConfigEntry",
    "ConfigKey",
    "ConfigMode",
    "ConfigSource",
    "ProductionConfig",
    "ProductionConfigReport",
    "ProductionConfigStatus",
    "RuntimeConfig",
]
