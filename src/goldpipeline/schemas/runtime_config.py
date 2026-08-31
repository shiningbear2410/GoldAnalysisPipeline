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


BOOLEAN_KEYS = frozenset({ConfigKey.AUTOMATION_ENABLED, ConfigKey.AUTOPUBLISH_ENABLED})
"""Keys whose value must read as an unambiguous true or false.

Parsed strictly here, unlike the environment. A typo in a shell session is a
five-minute puzzle; a typo in a file that persists across reboots quietly means
"off" forever, and the operator who wrote ``GOLDPIPELINE_AUTOMATION_ENABLED=yes``
deserves to be told rather than ignored.
"""

FORBIDDEN_KEYS = frozenset({"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TELEGRAM_BOT_TOKEN"})
"""Credentials, refused by name.

They are already absent from :class:`ConfigKey`, so the schema would reject them
anyway - but "unknown setting" is the wrong message. Someone trying to persist
an API key here needs to be told where it actually goes.
"""

TRUE_VALUES = frozenset({"true", "1", "yes", "on"})
FALSE_VALUES = frozenset({"false", "0", "no", "off"})


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
    "RUNTIME_CONFIG_SCHEMA_VERSION",
    "TRUE_VALUES",
    "ConfigEntry",
    "ConfigKey",
    "ConfigSource",
    "RuntimeConfig",
]
