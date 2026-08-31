"""Persistent non-secret configuration.

The line this file keeps testing is the one between a *credential* and a
*setting*. Both must survive into a scheduled task's fresh process, and they
want opposite treatment: a credential goes into an encrypted store nobody can
read, a setting goes into a file a person can read and check. Confusing the two
in either direction is the failure this store exists to prevent.

Everything here is offline and writes only under ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import FAKE_API_KEY, FAKE_OPENAI_KEY, TELEGRAM_TOKEN_SENTINEL

from goldpipeline.adapters.config_store import (
    LayeredConfig,
    RuntimeConfigStore,
    default_config_path,
    parse_key,
    validate,
)
from goldpipeline.config import AutomationSettings, MarketDataSettings
from goldpipeline.domain.errors import RuntimeConfigError, SecretNotPersistableError
from goldpipeline.schemas.runtime_config import (
    FORBIDDEN_KEYS,
    ConfigKey,
    ConfigSource,
    RuntimeConfig,
)

TARGET = ConfigKey.TELEGRAM_TARGET_CHAT_ID


@pytest.fixture
def store(tmp_path: Path) -> RuntimeConfigStore:
    return RuntimeConfigStore(tmp_path / "appdata" / "config.json")


# --- writing and reading ---------------------------------------------------


def test_a_setting_survives_a_write_and_a_read(store: RuntimeConfigStore) -> None:
    """Requirements 1 and 2, and the whole point of the file."""
    store.set(TARGET, "@pcfxsn")

    assert RuntimeConfigStore(store.path).load().get(TARGET) == "@pcfxsn"


def test_the_file_is_valid_utf8_json(store: RuntimeConfigStore) -> None:
    """Requirement 11."""
    store.set(ConfigKey.MT5_SYMBOL, "XAUUSD")

    payload = json.loads(store.path.read_text(encoding="utf-8"))
    assert payload["values"]["GOLDPIPELINE_MT5_SYMBOL"] == "XAUUSD"
    assert payload["schema_version"] == "1.0.0"


def test_a_partial_write_never_becomes_visible(
    store: RuntimeConfigStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 1: the write is atomic.

    A half-written configuration is worse than none - the next scheduled tick
    would parse whatever survived and run on a partial picture of the machine.
    """
    import os

    store.set(TARGET, "@pcfxsn")
    before = store.path.read_bytes()

    def explode(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", explode)
    with pytest.raises(OSError, match="disk full"):
        store.set(ConfigKey.MT5_SYMBOL, "XAUUSD")

    assert store.path.read_bytes() == before
    assert list(store.path.parent.glob("*.tmp")) == []


def test_an_empty_store_reads_as_empty(store: RuntimeConfigStore) -> None:
    assert store.load().values == {}


def test_a_corrupt_file_is_refused_rather_than_ignored(store: RuntimeConfigStore) -> None:
    """Reading it as "empty" would silently start a task on every default.

    That includes a symbol and a destination nobody chose, which is exactly the
    kind of quiet wrong answer this project refuses everywhere else.
    """
    store.path.parent.mkdir(parents=True)
    store.path.write_text("{ not json", encoding="utf-8")

    with pytest.raises(RuntimeConfigError, match="could not be read"):
        store.load()


def test_deleting_falls_back_to_the_default(store: RuntimeConfigStore) -> None:
    store.set(ConfigKey.OHLC_BARS, "50")
    store.delete(ConfigKey.OHLC_BARS)

    assert store.load().get(ConfigKey.OHLC_BARS) is None
    assert MarketDataSettings.from_env(LayeredConfig({}, store.load())).bar_count == 20


def test_deleting_something_absent_is_calm(store: RuntimeConfigStore) -> None:
    store.delete(ConfigKey.OHLC_BARS)

    assert store.load().values == {}


def test_the_file_lives_in_the_user_profile_not_the_repository() -> None:
    """Requirement 7.

    A checkout can be deleted or re-cloned without losing the machine's
    settings, and no ``git add .`` can ever capture them.
    """
    path = default_config_path({"LOCALAPPDATA": r"C:\Users\someone\AppData\Local"})

    assert path == Path(r"C:\Users\someone\AppData\Local\GoldAnalysisPipeline\config.json")
    assert "GoldAnalysisPipeline" in str(path)


# --- credentials are refused -----------------------------------------------


@pytest.mark.parametrize("secret", sorted(FORBIDDEN_KEYS))
def test_a_credential_name_is_refused_by_name(secret: str) -> None:
    """Requirements 5, 6 and 7 of the test list.

    Refused by name rather than merely absent from the schema. "Unknown setting"
    would be the wrong message: the person needs to be told where an API key
    actually goes.
    """
    with pytest.raises(SecretNotPersistableError) as exc:
        parse_key(secret)

    assert secret in str(exc.value)
    assert "secrets-set" in str(exc.value)


@pytest.mark.parametrize("value", [FAKE_API_KEY, FAKE_OPENAI_KEY, TELEGRAM_TOKEN_SENTINEL])
def test_a_credential_value_has_nowhere_to_go(store: RuntimeConfigStore, value: str) -> None:
    """Even if someone tried, there is no key it could be stored under."""
    for secret in FORBIDDEN_KEYS:
        with pytest.raises(SecretNotPersistableError):
            parse_key(secret)

    assert value not in json.dumps({key.value: key.value for key in ConfigKey})


def test_an_unknown_setting_is_refused() -> None:
    with pytest.raises(RuntimeConfigError, match="not a setting"):
        parse_key("GOLDPIPELINE_MAKE_IT_FASTER")


def test_every_allowed_setting_can_be_persisted(store: RuntimeConfigStore) -> None:
    """Requirement 8: the whitelist from the round specification, in full."""
    values = {
        ConfigKey.TELEGRAM_TARGET_CHAT_ID: "@pcfxsn",
        ConfigKey.MT5_SYMBOL: "XAUUSD",
        ConfigKey.CANONICAL_SYMBOL: "XAUUSD",
        ConfigKey.OHLC_TIMEFRAME: "M15",
        ConfigKey.OHLC_BARS: "20",
        ConfigKey.MAX_DATA_AGE_MINUTES: "90",
        ConfigKey.MAX_ANALYSIS_EVENT_AGE_MINUTES: "60",
        ConfigKey.DEFER_RETRY_MINUTES: "5",
        ConfigKey.AUTOMATION_MAX_EVENTS_PER_TICK: "3",
        ConfigKey.AUTOMATION_MAX_TICK_MINUTES: "10",
        ConfigKey.AUTOMATION_ENABLED: "false",
        ConfigKey.AUTOPUBLISH_ENABLED: "false",
        ConfigKey.AUTOPUBLISH_ALLOWED_TARGET: "@pcfxsn",
        ConfigKey.AUTOPUBLISH_MAX_RUN_AGE_MINUTES: "30",
    }
    for key, value in values.items():
        store.set(key, value)

    assert store.load().values == values
    assert len(values) == len(ConfigKey)


# --- validation reuses the settings classes --------------------------------


@pytest.mark.parametrize("rejected", ["TRUE!", "", "maybe", "0.0", "enable", "2"])
def test_a_boolean_must_be_unambiguous(store: RuntimeConfigStore, rejected: str) -> None:
    """Requirement 9.

    Stricter than the environment on purpose. A typo in a shell session is a
    five-minute puzzle; a typo in a file that persists across reboots quietly
    means "off" forever.
    """
    with pytest.raises(RuntimeConfigError, match="must be true or false"):
        store.set(ConfigKey.AUTOMATION_ENABLED, rejected)

    assert store.load().values == {}


@pytest.mark.parametrize("accepted", ["true", "FALSE", "on", "off", "yes ", "no", "1", "0"])
def test_recognised_booleans_are_accepted(store: RuntimeConfigStore, accepted: str) -> None:
    store.set(ConfigKey.AUTOMATION_ENABLED, accepted)

    assert store.load().get(ConfigKey.AUTOMATION_ENABLED) == accepted.strip()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (ConfigKey.OHLC_BARS, "5"),
        (ConfigKey.OHLC_BARS, "5000"),
        (ConfigKey.OHLC_BARS, "not-a-number"),
        (ConfigKey.AUTOMATION_MAX_EVENTS_PER_TICK, "0"),
        (ConfigKey.AUTOMATION_MAX_EVENTS_PER_TICK, "999"),
        (ConfigKey.OHLC_TIMEFRAME, "H2"),
        (ConfigKey.TELEGRAM_TARGET_CHAT_ID, "not a chat"),
        (ConfigKey.AUTOPUBLISH_ALLOWED_TARGET, "nope"),
        (ConfigKey.CANONICAL_SYMBOL, "XAU USD!"),
    ],
)
def test_a_value_that_would_not_load_is_never_persisted(
    store: RuntimeConfigStore, key: ConfigKey, value: str
) -> None:
    """Requirement 10, and the reason validation reuses the real loaders.

    Every rule here - bar ranges, the timeframe whitelist, destination shape -
    is enforced in exactly one place in this codebase. A value that would break
    the next scheduled tick is refused now instead of at 3am.
    """
    with pytest.raises(RuntimeConfigError):
        store.set(key, value)

    assert store.load().values == {}


def test_validation_accepts_the_intended_production_configuration() -> None:
    """The values this installation will actually use."""
    validate(
        {
            ConfigKey.TELEGRAM_TARGET_CHAT_ID: "@pcfxsn",
            ConfigKey.MT5_SYMBOL: "XAUUSD",
            ConfigKey.OHLC_TIMEFRAME: "M15",
            ConfigKey.OHLC_BARS: "20",
            ConfigKey.AUTOMATION_ENABLED: "false",
            ConfigKey.AUTOPUBLISH_ENABLED: "false",
            ConfigKey.AUTOPUBLISH_ALLOWED_TARGET: "@pcfxsn",
        }
    )


# --- resolution order ------------------------------------------------------


def test_the_environment_overrides_the_persisted_file() -> None:
    """Requirement 3.

    A temporary override must not require editing a persisted file and
    remembering to change it back.
    """
    config = RuntimeConfig(values={ConfigKey.MT5_SYMBOL: "XAUUSD"})
    layered = LayeredConfig({"GOLDPIPELINE_MT5_SYMBOL": "XAUUSDm"}, config)

    entry = layered.resolve(ConfigKey.MT5_SYMBOL)
    assert entry.value == "XAUUSDm"
    assert entry.source is ConfigSource.PROCESS_ENV
    assert layered["GOLDPIPELINE_MT5_SYMBOL"] == "XAUUSDm"


def test_the_file_answers_when_the_environment_is_silent() -> None:
    """Requirement 2, and the Task Scheduler case: a process with no session."""
    config = RuntimeConfig(values={ConfigKey.MT5_SYMBOL: "XAUUSD"})
    layered = LayeredConfig({}, config)

    entry = layered.resolve(ConfigKey.MT5_SYMBOL)
    assert entry.value == "XAUUSD"
    assert entry.source is ConfigSource.PERSISTENT


def test_neither_layer_means_the_built_in_default() -> None:
    """Requirement 4.

    ``DEFAULT`` is a meaningfully different answer from a persisted value that
    happens to equal the default - one was chosen, the other was not.
    """
    layered = LayeredConfig({}, RuntimeConfig())

    entry = layered.resolve(ConfigKey.OHLC_BARS)
    assert entry.value is None
    assert entry.source is ConfigSource.DEFAULT
    assert MarketDataSettings.from_env(layered).bar_count == 20


def test_a_blank_environment_value_does_not_shadow_the_file() -> None:
    """An exported-but-empty variable is a common way to think you unset one."""
    config = RuntimeConfig(values={ConfigKey.MT5_SYMBOL: "XAUUSD"})
    layered = LayeredConfig({"GOLDPIPELINE_MT5_SYMBOL": "   "}, config)

    assert layered.resolve(ConfigKey.MT5_SYMBOL).source is ConfigSource.PERSISTENT


def test_the_persisted_file_is_never_written_back_from_the_environment(
    store: RuntimeConfigStore,
) -> None:
    """The same one-way rule the credential provider follows.

    Promoting a throwaway override into permanent storage is how a stale value
    outlives the reason for it.
    """
    LayeredConfig({"GOLDPIPELINE_MT5_SYMBOL": "XAUUSDm"}, store.load()).resolve(
        ConfigKey.MT5_SYMBOL
    )

    assert store.load().values == {}
    assert not hasattr(LayeredConfig, "promote")


# --- the settings classes read through it ----------------------------------


def test_the_settings_classes_read_the_persisted_file(store: RuntimeConfigStore) -> None:
    """Requirement 8 of the spec: existing loaders resolve from the store.

    This is what makes a scheduled task work: it inherits no session, so every
    value it needs has to come from the file.
    """
    store.set(ConfigKey.MT5_SYMBOL, "XAUUSDm")
    store.set(ConfigKey.CANONICAL_SYMBOL, "XAUUSD")
    store.set(ConfigKey.OHLC_TIMEFRAME, "M5")
    store.set(ConfigKey.OHLC_BARS, "40")
    store.set(ConfigKey.AUTOMATION_MAX_EVENTS_PER_TICK, "7")
    store.set(ConfigKey.AUTOPUBLISH_ALLOWED_TARGET, "@pcfxsn")

    layered = LayeredConfig({}, store.load())
    market = MarketDataSettings.from_env(layered)
    automation = AutomationSettings.from_env(layered)

    assert market.provider_symbol == "XAUUSDm"
    assert market.canonical_symbol == "XAUUSD"
    assert market.timeframe == "M5"
    assert market.bar_count == 40
    assert automation.max_events_per_tick == 7
    assert automation.auto_publish_allowed_target == "@pcfxsn"


def test_automation_flags_default_to_off_when_nothing_is_persisted() -> None:
    """The safe default survives the introduction of a config file."""
    automation = AutomationSettings.from_env(LayeredConfig({}, RuntimeConfig()))

    assert automation.enabled is False
    assert automation.auto_publish_enabled is False


def test_a_persisted_false_stays_false() -> None:
    config = RuntimeConfig(
        values={
            ConfigKey.AUTOMATION_ENABLED: "false",
            ConfigKey.AUTOPUBLISH_ENABLED: "false",
        }
    )
    automation = AutomationSettings.from_env(LayeredConfig({}, config))

    assert automation.enabled is False
    assert automation.auto_publish_enabled is False


def test_the_layered_mapping_still_exposes_unrelated_environment_values() -> None:
    """Settings loaders read many keys this store does not hold."""
    layered = LayeredConfig({"ANTHROPIC_MODEL": "claude-x"}, RuntimeConfig())

    assert layered.get("ANTHROPIC_MODEL") == "claude-x"
    assert layered.get("NOT_SET_ANYWHERE") is None


def test_unicode_survives_a_round_trip(tmp_path: Path) -> None:
    """Requirement 11: the path may contain non-ASCII too."""
    store = RuntimeConfigStore(tmp_path / "hồ sơ" / "config.json")
    store.set(ConfigKey.MT5_SYMBOL, "XAUUSD")

    assert store.path.is_file()
    assert RuntimeConfigStore(store.path).load().get(ConfigKey.MT5_SYMBOL) == "XAUUSD"
