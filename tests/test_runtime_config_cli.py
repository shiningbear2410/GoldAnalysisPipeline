"""The config commands, and how preflight reads what they persist.

Values here are not secret, so a command-line argument is fine - the whole point
of the split is that a destination chat and a symbol are things an operator
should be able to read back and check. What must never appear is a credential,
and several tests are about that boundary rather than about configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import FAKE_API_KEY, FAKE_OPENAI_KEY, TELEGRAM_TOKEN_SENTINEL

from goldpipeline.cli import EXIT_INVALID_DATA, EXIT_OK, main
from goldpipeline.schemas.runtime_config import ConfigKey

SENTINELS = (FAKE_API_KEY, FAKE_OPENAI_KEY, TELEGRAM_TOKEN_SENTINEL)


def invoke(args: list[str]) -> int:
    return main(args)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start from an environment that shadows nothing."""
    for key in ConfigKey:
        monkeypatch.delenv(key.value, raising=False)


# --- setting ---------------------------------------------------------------


def test_setting_a_value_persists_it(config_store: Any, capsys: pytest.CaptureFixture[str]) -> None:
    code = invoke(["config-set", "TELEGRAM_TARGET_CHAT_ID", "@pcfxsn"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "@pcfxsn" in out
    assert config_store.load().get(ConfigKey.TELEGRAM_TARGET_CHAT_ID) == "@pcfxsn"


def test_a_lowercase_name_is_accepted(
    config_store: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    invoke(["config-set", "goldpipeline_mt5_symbol", "XAUUSD"])
    capsys.readouterr()

    assert config_store.load().get(ConfigKey.MT5_SYMBOL) == "XAUUSD"


@pytest.mark.parametrize("secret", ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TELEGRAM_BOT_TOKEN"])
def test_a_credential_cannot_be_persisted(
    config_store: Any, capsys: pytest.CaptureFixture[str], secret: str
) -> None:
    """Requirements 5, 6 and 7.

    Refused by name, with a message that says where the value actually goes -
    "unknown setting" would leave someone guessing.
    """
    code = invoke(["config-set", secret, FAKE_API_KEY])
    err = capsys.readouterr().err

    assert code != EXIT_OK
    assert "secrets-set" in err
    assert config_store.load().values == {}
    assert FAKE_API_KEY not in err


def test_a_credential_value_never_reaches_the_file(
    config_store: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    invoke(["config-set", "TELEGRAM_BOT_TOKEN", TELEGRAM_TOKEN_SENTINEL])
    capsys.readouterr()

    assert not config_store.path.exists() or TELEGRAM_TOKEN_SENTINEL not in (
        config_store.path.read_text(encoding="utf-8")
    )


def test_an_unusable_value_is_refused(
    config_store: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    code = invoke(["config-set", "GOLDPIPELINE_OHLC_BARS", "9999"])
    err = capsys.readouterr().err

    assert code == EXIT_INVALID_DATA
    assert "would not load" in err
    assert config_store.load().values == {}


def test_a_bad_boolean_is_refused(config_store: Any, capsys: pytest.CaptureFixture[str]) -> None:
    code = invoke(["config-set", "GOLDPIPELINE_AUTOMATION_ENABLED", "maybe"])
    err = capsys.readouterr().err

    assert code == EXIT_INVALID_DATA
    assert "true or false" in err


# --- status ----------------------------------------------------------------


def test_status_shows_the_source_of_each_value(
    config_store: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 12.

    The distinction that matters for a scheduled task: a value from this session
    will not survive into it, and a persisted one will.
    """
    config_store.set(ConfigKey.MT5_SYMBOL, "XAUUSD")

    invoke(["config-status", "--json"])
    payload = json.loads(capsys.readouterr().out)
    sources = {entry["key"]: entry["source"] for entry in payload["settings"]}

    assert sources["GOLDPIPELINE_MT5_SYMBOL"] == "PERSISTENT"
    assert sources["GOLDPIPELINE_OHLC_BARS"] == "DEFAULT"


def test_the_environment_is_reported_as_winning(
    config_store: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 3, reported honestly rather than flattened to one value."""
    config_store.set(ConfigKey.MT5_SYMBOL, "XAUUSD")
    monkeypatch.setenv("GOLDPIPELINE_MT5_SYMBOL", "XAUUSDm")

    invoke(["config-status", "--json"])
    payload = json.loads(capsys.readouterr().out)
    entry = next(e for e in payload["settings"] if e["key"] == "GOLDPIPELINE_MT5_SYMBOL")

    assert entry["value"] == "XAUUSDm"
    assert entry["source"] == "PROCESS_ENV"


def test_setting_a_shadowed_value_says_so(
    config_store: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Writing succeeded, but this shell is showing something else.

    Without the note, an operator would check `config-status`, see the old
    value, and conclude the write failed.
    """
    monkeypatch.setenv("GOLDPIPELINE_MT5_SYMBOL", "XAUUSDm")

    invoke(["config-set", "GOLDPIPELINE_MT5_SYMBOL", "XAUUSD"])
    captured = capsys.readouterr()

    assert config_store.load().get(ConfigKey.MT5_SYMBOL) == "XAUUSD"
    assert "wins here" in captured.err


def test_status_lists_every_allowed_setting(
    config_store: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    invoke(["config-status", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert {entry["key"] for entry in payload["settings"]} == {key.value for key in ConfigKey}


def test_status_prints_no_credential(
    config_store: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TELEGRAM_TOKEN_SENTINEL)
    config_store.set(ConfigKey.TELEGRAM_TARGET_CHAT_ID, "@pcfxsn")

    invoke(["config-status"])
    captured = capsys.readouterr()

    for sentinel in SENTINELS:
        assert sentinel not in captured.out
        assert sentinel not in captured.err
    assert "@pcfxsn" in captured.out, "a destination is configuration, not a secret"


# --- deleting --------------------------------------------------------------


def test_deleting_falls_back_to_the_default(
    config_store: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    config_store.set(ConfigKey.OHLC_BARS, "50")

    code = invoke(["config-delete", "GOLDPIPELINE_OHLC_BARS"])
    capsys.readouterr()

    assert code == EXIT_OK
    assert config_store.load().get(ConfigKey.OHLC_BARS) is None


# --- preflight reads the persisted configuration ---------------------------


def preflight_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "automation-preflight",
        "--inbox-dir",
        str(tmp_path / "inbox"),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--automation-dir",
        str(tmp_path / "automation"),
        "--fake-mt5",
        *extra,
    ]


def test_preflight_reads_the_persisted_target(
    config_store: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirement 13, and what makes the allowlist check meaningful.

    Both settings must come from somewhere a scheduled task can see, or the
    comparison would pass by hand and fail unattended.
    """
    from conftest import FakeKeyringModule

    from goldpipeline import cli
    from goldpipeline.adapters.windows_credentials import (
        SERVICE_NAME,
        WindowsCredentialSecretProvider,
        inspect_backend,
    )

    module = FakeKeyringModule(
        {
            (SERVICE_NAME, "anthropic_api_key"): FAKE_API_KEY,
            (SERVICE_NAME, "openai_api_key"): FAKE_OPENAI_KEY,
            (SERVICE_NAME, "telegram_bot_token"): TELEGRAM_TOKEN_SENTINEL,
        }
    )
    monkeypatch.setattr(cli, "_credential_store", lambda: WindowsCredentialSecretProvider(module))
    monkeypatch.setattr(cli, "inspect_backend", lambda *a, **k: inspect_backend(module))

    config_store.set(ConfigKey.TELEGRAM_TARGET_CHAT_ID, "@pcfxsn")
    config_store.set(ConfigKey.AUTOPUBLISH_ALLOWED_TARGET, "@pcfxsn")
    config_store.set(ConfigKey.AUTOPUBLISH_ENABLED, "true")

    invoke(preflight_args(tmp_path, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert payload["allowed_target"] == "@pcfxsn"
    assert payload["configured_target"] == "@pcfxsn"
    assert payload["auto_publish_enabled"] is True
    assert not any("differ" in blocker for blocker in payload["blockers"])


def test_preflight_is_ready_on_persisted_config_and_stored_credentials(
    config_store: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """What a scheduled task actually needs: nothing session-bound anywhere."""
    from conftest import FakeKeyringModule

    from goldpipeline import cli
    from goldpipeline.adapters.windows_credentials import (
        SERVICE_NAME,
        WindowsCredentialSecretProvider,
        inspect_backend,
    )

    module = FakeKeyringModule(
        {
            (SERVICE_NAME, "anthropic_api_key"): FAKE_API_KEY,
            (SERVICE_NAME, "openai_api_key"): FAKE_OPENAI_KEY,
        }
    )
    monkeypatch.setattr(cli, "_credential_store", lambda: WindowsCredentialSecretProvider(module))
    monkeypatch.setattr(cli, "inspect_backend", lambda *a, **k: inspect_backend(module))
    config_store.set(ConfigKey.TELEGRAM_TARGET_CHAT_ID, "@pcfxsn")

    invoke(preflight_args(tmp_path, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert payload["task_readiness"] == "READY"
    assert payload["auto_publish_enabled"] is False
    assert payload["blockers"] == []


def test_the_worker_reads_persisted_settings(
    config_store: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The end of the chain: a tick with no session still knows the symbol.

    Nothing is exported here, so every value the tick uses came out of the file.
    """
    config_store.set(ConfigKey.MT5_SYMBOL, "XAUUSD")
    config_store.set(ConfigKey.AUTOMATION_MAX_EVENTS_PER_TICK, "1")

    code = invoke(
        [
            "automation-run-once",
            "--inbox-dir",
            str(tmp_path / "inbox"),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--automation-dir",
            str(tmp_path / "automation"),
            "--fake-mt5",
            "--fake-ai",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["auto_publish_enabled"] is False
    assert payload["mode"] == "READY_FOR_PUBLISH"


def test_enabling_automation_does_not_enable_publishing(
    config_store: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirements 18 and 27 of the spec, at the command boundary."""
    invoke(["config-set", "GOLDPIPELINE_AUTOMATION_ENABLED", "true"])
    capsys.readouterr()

    code = invoke(
        [
            "automation-worker-tick",
            "--inbox-dir",
            str(tmp_path / "inbox"),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--automation-dir",
            str(tmp_path / "automation"),
            "--fake-mt5",
            "--fake-ai",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["status"] == "OK", "the worker ran"
    assert payload["auto_publish_enabled"] is False
    assert payload["mode"] == "READY_FOR_PUBLISH"
