"""Writer configuration, and the rule that credentials never leak."""

from __future__ import annotations

import dataclasses

import pytest
from conftest import FAKE_API_KEY

from goldpipeline.config import (
    API_KEY_ENV,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    MODEL_ENV,
    WriterSettings,
)
from goldpipeline.domain.errors import WriterConfigurationError


def test_defaults_apply_when_only_a_key_is_set() -> None:
    settings = WriterSettings.from_env({API_KEY_ENV: FAKE_API_KEY})

    assert settings.api_key == FAKE_API_KEY
    assert settings.model == DEFAULT_MODEL
    assert settings.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert settings.max_retries == DEFAULT_MAX_RETRIES


def test_model_is_configurable_from_the_environment() -> None:
    settings = WriterSettings.from_env({API_KEY_ENV: FAKE_API_KEY, MODEL_ENV: "claude-sonnet-5"})
    assert settings.model == "claude-sonnet-5"


def test_command_line_model_beats_the_environment() -> None:
    settings = WriterSettings.from_env(
        {API_KEY_ENV: FAKE_API_KEY, MODEL_ENV: "claude-sonnet-5"},
        model_override="claude-opus-5",
    )
    assert settings.model == "claude-opus-5"


def test_missing_key_is_a_configuration_error() -> None:
    with pytest.raises(WriterConfigurationError) as exc:
        WriterSettings.from_env({})
    assert exc.value.details == {"setting": API_KEY_ENV}
    assert exc.value.code == "WRITER_CONFIGURATION_ERROR"


def test_blank_key_is_treated_as_missing() -> None:
    with pytest.raises(WriterConfigurationError):
        WriterSettings.from_env({API_KEY_ENV: "   "})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("GOLDPIPELINE_WRITER_TIMEOUT", "not-a-number"),
        ("GOLDPIPELINE_WRITER_TIMEOUT", "0"),
        ("GOLDPIPELINE_WRITER_TIMEOUT", "-5"),
        ("GOLDPIPELINE_WRITER_MAX_RETRIES", "-1"),
        ("GOLDPIPELINE_WRITER_MAX_TOKENS", "0"),
    ],
)
def test_unusable_tuning_values_are_rejected(name: str, value: str) -> None:
    with pytest.raises(WriterConfigurationError) as exc:
        WriterSettings.from_env({API_KEY_ENV: FAKE_API_KEY, name: value})
    assert exc.value.details == {"setting": name}


def test_tuning_values_are_read_when_valid() -> None:
    settings = WriterSettings.from_env(
        {
            API_KEY_ENV: FAKE_API_KEY,
            "GOLDPIPELINE_WRITER_TIMEOUT": "45.5",
            "GOLDPIPELINE_WRITER_MAX_RETRIES": "0",
            "GOLDPIPELINE_WRITER_MAX_TOKENS": "4096",
        }
    )
    assert settings.timeout_seconds == 45.5
    assert settings.max_retries == 0
    assert settings.max_tokens == 4096


# --- credential hygiene ---------------------------------------------------


def test_repr_and_str_never_contain_the_key() -> None:
    """A settings object reaches log lines and traceback frames."""
    settings = WriterSettings.from_env({API_KEY_ENV: FAKE_API_KEY})

    assert FAKE_API_KEY not in repr(settings)
    assert FAKE_API_KEY not in str(settings)
    assert FAKE_API_KEY not in f"{settings}"
    assert "redacted" in repr(settings)
    assert settings.model in repr(settings)


def test_configuration_errors_never_echo_the_value() -> None:
    """The message names the setting; it must not quote what was in it."""
    with pytest.raises(WriterConfigurationError) as exc:
        WriterSettings.from_env(
            {API_KEY_ENV: FAKE_API_KEY, "GOLDPIPELINE_WRITER_TIMEOUT": FAKE_API_KEY}
        )
    assert FAKE_API_KEY not in str(exc.value)
    assert FAKE_API_KEY not in repr(exc.value.details)


def test_settings_are_immutable() -> None:
    settings = WriterSettings.from_env({API_KEY_ENV: FAKE_API_KEY})
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.model = "something-else"  # type: ignore[misc]
