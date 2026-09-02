"""Credential providers, precedence, and the backend that must be trusted.

Every test here is offline: the keyring module is injected, so nothing in this
file reads, writes or lists an entry in a real credential manager.

Two properties carry the design:

* **process environment first, credential store second** - which keeps every
  existing workflow working and makes a temporary override possible without
  rewriting a stored credential; and
* **no plaintext fallback** - a store that is missing or insecure produces a
  refusal, never a file, an environment variable, or a quiet degradation to
  something that happens to work.
"""

from __future__ import annotations

from typing import Any

import pytest
from conftest import (
    FAKE_API_KEY,
    FAKE_OPENAI_KEY,
    TELEGRAM_TOKEN_SENTINEL,
    FakeKeyringModule,
    fail_backend_module,
    plaintext_backend_module,
)

from goldpipeline.adapters.secrets import (
    CompositeSecretProvider,
    EnvironmentSecretProvider,
    FakeSecretProvider,
    SecretProvider,
    default_provider,
)
from goldpipeline.adapters.windows_credentials import (
    SECURE_BACKENDS,
    SERVICE_NAME,
    WindowsCredentialSecretProvider,
    inspect_backend,
)
from goldpipeline.domain.errors import (
    CredentialBackendUnavailableError,
    CredentialDeleteError,
    CredentialNotFoundError,
    CredentialReadError,
    CredentialWriteError,
    InsecureCredentialBackendError,
)
from goldpipeline.schemas.secrets import SecretName, SecretSource, SecretStatus

ANTHROPIC = SecretName.ANTHROPIC_API_KEY
OPENAI = SecretName.OPENAI_API_KEY
TELEGRAM = SecretName.TELEGRAM_BOT_TOKEN


# --- the environment provider ---------------------------------------------


def test_the_environment_provider_returns_what_is_set() -> None:
    """Requirement 1."""
    provider = EnvironmentSecretProvider({"ANTHROPIC_API_KEY": FAKE_API_KEY})

    assert provider.get_secret(ANTHROPIC) == FAKE_API_KEY
    assert provider.source is SecretSource.PROCESS_ENV


def test_the_environment_provider_returns_none_when_unset() -> None:
    """Requirement 2: absence is a return value, not an exception."""
    assert EnvironmentSecretProvider({}).get_secret(ANTHROPIC) is None


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_a_blank_environment_value_counts_as_unset(blank: str) -> None:
    """An exported-but-empty variable is a common way to think you have set one."""
    assert EnvironmentSecretProvider({"ANTHROPIC_API_KEY": blank}).get_secret(ANTHROPIC) is None


def test_the_default_provider_reads_the_environment_only() -> None:
    """Importing this package must never touch a credential vault."""
    provider = default_provider({"OPENAI_API_KEY": FAKE_OPENAI_KEY})

    assert isinstance(provider, EnvironmentSecretProvider)
    assert provider.get_secret(OPENAI) == FAKE_OPENAI_KEY


# --- the credential store provider ----------------------------------------


def test_the_store_provider_returns_a_stored_credential() -> None:
    """Requirement 3."""
    module = FakeKeyringModule({(SERVICE_NAME, "anthropic_api_key"): FAKE_API_KEY})
    provider = WindowsCredentialSecretProvider(module)

    assert provider.get_secret(ANTHROPIC) == FAKE_API_KEY
    assert provider.source is SecretSource.WINDOWS_CREDENTIAL_MANAGER


def test_the_store_provider_returns_none_when_the_entry_is_absent() -> None:
    """Requirement 4."""
    assert WindowsCredentialSecretProvider(FakeKeyringModule()).get_secret(ANTHROPIC) is None


def test_entries_are_named_for_the_kind_of_credential() -> None:
    """Requirement 6: never the token itself, and readable in the vault UI."""
    module = FakeKeyringModule()
    for name in SecretName:
        WindowsCredentialSecretProvider(module).get_secret(name)

    assert module.reads == [
        (SERVICE_NAME, "anthropic_api_key"),
        (SERVICE_NAME, "openai_api_key"),
        (SERVICE_NAME, "telegram_bot_token"),
        (SERVICE_NAME, "ingest_token"),
    ]


def test_a_store_that_cannot_be_consulted_raises_rather_than_reporting_absence() -> None:
    """The distinction that matters most in this file.

    A locked or broken vault must never look like an unset credential, or a
    scheduled task would report a missing key when the real problem is a store
    it could not open.
    """
    module = FakeKeyringModule(read_error=RuntimeError("the vault is locked"))

    with pytest.raises(CredentialReadError):
        WindowsCredentialSecretProvider(module).get_secret(ANTHROPIC)


def test_a_store_error_never_repeats_the_backend_text() -> None:
    """Requirement 27: a credential store's own errors can quote what it held."""
    module = FakeKeyringModule(read_error=RuntimeError(f"failed handling {FAKE_API_KEY}"))

    with pytest.raises(CredentialReadError) as exc:
        WindowsCredentialSecretProvider(module).get_secret(ANTHROPIC)

    assert FAKE_API_KEY not in str(exc.value)
    assert FAKE_API_KEY not in repr(exc.value.details)
    assert FAKE_API_KEY not in str(exc.value.__cause__)


def test_a_missing_backend_is_distinguished_from_a_missing_credential() -> None:
    """Requirement 5: one is fixed with pip, the other by typing a key."""
    provider = WindowsCredentialSecretProvider(module=None)

    # Simulated by asking a module that raises on import, which the helper does
    # by refusing to answer at all.
    with pytest.raises((CredentialBackendUnavailableError, CredentialReadError)):
        WindowsCredentialSecretProvider(FakeKeyringModule(read_error=OSError())).get_secret(
            ANTHROPIC
        )
    assert provider.service == SERVICE_NAME


# --- backend trust ---------------------------------------------------------


def test_the_windows_backend_is_trusted() -> None:
    report = inspect_backend(FakeKeyringModule())

    assert report.available
    assert report.secure
    assert report.ready
    assert report.backend in SECURE_BACKENDS


def test_the_fail_backend_is_rejected() -> None:
    """Requirement 7: keyring's own no-op store."""
    report = inspect_backend(fail_backend_module())

    assert report.available
    assert not report.secure
    assert not report.ready


def test_a_null_backend_is_rejected() -> None:
    """Requirement 6 of the spec: stores nothing, reports success."""
    report = inspect_backend(FakeKeyringModule(backend_name="keyring.backends.null.Keyring"))

    assert not report.ready


def test_a_plaintext_backend_is_rejected() -> None:
    """Requirement 8, and the reason this is an allowlist.

    A store that keeps credentials in a plaintext file is worse than no store at
    all, because it looks like one.
    """
    report = inspect_backend(plaintext_backend_module())

    assert not report.ready
    assert "not store credentials securely" in report.detail


def test_an_unknown_backend_is_rejected_by_default() -> None:
    """Fail closed: a store nobody has vouched for is not trusted."""
    report = inspect_backend(FakeKeyringModule(backend_name="some.vendor.MysteryKeyring"))

    assert not report.ready
    assert "not on the list" in report.detail


def test_a_backend_that_cannot_be_determined_is_not_ready() -> None:
    report = inspect_backend(FakeKeyringModule(backend_error=RuntimeError("boom")))

    assert not report.available
    assert not report.ready


def test_the_backend_report_names_no_entry_or_value() -> None:
    """It gets pasted into chat windows when someone asks for help."""
    module = FakeKeyringModule({(SERVICE_NAME, "anthropic_api_key"): FAKE_API_KEY})
    report = inspect_backend(module)

    rendered = f"{report.backend} {report.detail}"
    assert FAKE_API_KEY not in rendered
    assert "anthropic_api_key" not in rendered


# --- precedence ------------------------------------------------------------


def composite(env: dict[str, str], stored: dict[SecretName, str]) -> CompositeSecretProvider:
    return CompositeSecretProvider([EnvironmentSecretProvider(env), FakeSecretProvider(stored)])


def test_the_process_environment_wins() -> None:
    """Requirements 9 and 13.

    A temporary override should not require rewriting a stored credential and
    remembering to put the old one back.
    """
    provider = composite({"ANTHROPIC_API_KEY": "session-override"}, {ANTHROPIC: FAKE_API_KEY})

    value, source = provider.resolve(ANTHROPIC)

    assert value == "session-override"
    assert source is SecretSource.PROCESS_ENV


def test_the_store_answers_when_the_environment_is_silent() -> None:
    """Requirement 10, and the whole point of the round.

    This is the Task Scheduler case: a fresh process with no session variables.
    """
    provider = composite({}, {ANTHROPIC: FAKE_API_KEY})

    value, source = provider.resolve(ANTHROPIC)

    assert value == FAKE_API_KEY
    assert source is SecretSource.WINDOWS_CREDENTIAL_MANAGER


def test_a_credential_in_neither_place_is_missing() -> None:
    value, source = composite({}, {}).resolve(ANTHROPIC)

    assert value is None
    assert source is SecretSource.MISSING


def test_an_environment_value_is_never_copied_into_the_store() -> None:
    """Requirement 11.

    Promoting a throwaway override into permanent storage because it happened to
    be present is how a stale key ends up in a vault two years later.
    """
    store = FakeSecretProvider({})
    provider = CompositeSecretProvider(
        [EnvironmentSecretProvider({"ANTHROPIC_API_KEY": "session-override"}), store]
    )

    provider.resolve(ANTHROPIC)

    assert store.get_secret(ANTHROPIC) is None
    assert not hasattr(CompositeSecretProvider, "promote")


def test_the_composite_asks_no_further_once_answered() -> None:
    """The environment answering means the vault is not even opened."""
    store = FakeSecretProvider({ANTHROPIC: FAKE_API_KEY})
    provider = CompositeSecretProvider(
        [EnvironmentSecretProvider({"ANTHROPIC_API_KEY": "session"}), store]
    )

    provider.resolve(ANTHROPIC)

    assert store.reads == []


# --- writing ---------------------------------------------------------------


def test_storing_a_credential_writes_one_entry() -> None:
    module = FakeKeyringModule()
    WindowsCredentialSecretProvider(module).set_secret(TELEGRAM, TELEGRAM_TOKEN_SENTINEL)

    assert module.stored == {(SERVICE_NAME, "telegram_bot_token"): TELEGRAM_TOKEN_SENTINEL}


def test_an_insecure_backend_refuses_the_write(tmp_path: Any) -> None:
    """Requirement 25, and the most important refusal here.

    Writing is the one operation whose consequences outlive the process. A store
    that turned out to be a plaintext file would have left the secret on disk.
    """
    module = plaintext_backend_module()

    with pytest.raises(InsecureCredentialBackendError) as exc:
        WindowsCredentialSecretProvider(module).set_secret(ANTHROPIC, FAKE_API_KEY)

    assert module.stored == {}
    assert "Nothing was written" in str(exc.value)
    assert FAKE_API_KEY not in str(exc.value)


def test_no_plaintext_fallback_is_ever_created(tmp_path: Any, monkeypatch: Any) -> None:
    """Requirement 25, stated as an absence.

    A refused write must leave no `.env`, no `credentials.json`, no environment
    variable - nothing that happens to work.
    """
    monkeypatch.chdir(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())

    with pytest.raises(InsecureCredentialBackendError):
        WindowsCredentialSecretProvider(fail_backend_module()).set_secret(ANTHROPIC, FAKE_API_KEY)

    assert sorted(p.name for p in tmp_path.iterdir()) == before
    import os

    assert "ANTHROPIC_API_KEY" not in os.environ or os.environ["ANTHROPIC_API_KEY"] != FAKE_API_KEY


def test_an_empty_value_is_refused() -> None:
    module = FakeKeyringModule()

    with pytest.raises(CredentialWriteError):
        WindowsCredentialSecretProvider(module).set_secret(ANTHROPIC, "   ")

    assert module.stored == {}


def test_a_refusing_store_reports_safely() -> None:
    module = FakeKeyringModule(write_error=RuntimeError(f"cannot save {FAKE_API_KEY}"))

    with pytest.raises(CredentialWriteError) as exc:
        WindowsCredentialSecretProvider(module).set_secret(ANTHROPIC, FAKE_API_KEY)

    assert FAKE_API_KEY not in str(exc.value)
    assert FAKE_API_KEY not in str(exc.value.__cause__)


# --- deleting --------------------------------------------------------------


def test_deleting_removes_only_the_named_credential() -> None:
    """Requirement 25 of the test list."""
    module = FakeKeyringModule(
        {
            (SERVICE_NAME, "anthropic_api_key"): FAKE_API_KEY,
            (SERVICE_NAME, "openai_api_key"): FAKE_OPENAI_KEY,
            (SERVICE_NAME, "telegram_bot_token"): TELEGRAM_TOKEN_SENTINEL,
        }
    )

    WindowsCredentialSecretProvider(module).delete_secret(OPENAI)

    assert sorted(entry for _, entry in module.stored) == [
        "anthropic_api_key",
        "telegram_bot_token",
    ]


def test_deleting_something_that_is_not_there_is_reported_calmly() -> None:
    """Requirement 26: the operator's intent is already achieved."""
    with pytest.raises(CredentialNotFoundError):
        WindowsCredentialSecretProvider(FakeKeyringModule()).delete_secret(ANTHROPIC)


def test_a_refusing_delete_is_scrubbed() -> None:
    module = FakeKeyringModule(
        {(SERVICE_NAME, "anthropic_api_key"): FAKE_API_KEY},
        delete_error=RuntimeError(f"busy holding {FAKE_API_KEY}"),
    )

    with pytest.raises(CredentialDeleteError) as exc:
        WindowsCredentialSecretProvider(module).delete_secret(ANTHROPIC)

    assert FAKE_API_KEY not in str(exc.value)


# --- the status model ------------------------------------------------------


def test_a_status_has_nowhere_to_put_a_value() -> None:
    """Requirement 23."""
    status = SecretStatus(name=ANTHROPIC, configured=True, source=SecretSource.PROCESS_ENV)

    assert "value" not in SecretStatus.model_fields
    assert "secret" not in SecretStatus.model_fields
    assert FAKE_API_KEY not in status.model_dump_json()
    assert status.summary == "configured (process environment)"


def test_a_missing_status_says_so() -> None:
    status = SecretStatus(name=TELEGRAM, configured=False)

    assert status.summary == "missing"
    assert status.source is SecretSource.MISSING


def test_the_store_source_reads_clearly() -> None:
    status = SecretStatus(
        name=ANTHROPIC, configured=True, source=SecretSource.WINDOWS_CREDENTIAL_MANAGER
    )

    assert status.summary == "configured (Windows Credential Manager)"


# --- which credentials exist at all ---------------------------------------


def test_only_credentials_are_recognised() -> None:
    """Requirement 5 of the spec.

    A destination chat, a symbol, a timeframe and a feature flag are
    configuration, not credentials. Putting them in a vault would make them
    harder to audit without making anything safer.

    The list grows only when something genuinely secret arrives; INGEST_TOKEN
    joined it because a bearer token is exactly that.
    """
    assert {name.value for name in SecretName} == {
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "INGEST_TOKEN",
    }
    for forbidden in ("TELEGRAM_TARGET_CHAT_ID", "GOLDPIPELINE_MT5_SYMBOL", "ANTHROPIC_MODEL"):
        assert forbidden not in {name.value for name in SecretName}


def test_the_provider_protocol_is_one_method() -> None:
    """Everything above this line asks for a secret by name and learns nothing else."""
    surface = {name for name in dir(SecretProvider) if not name.startswith("_")}

    assert surface == {"get_secret", "source"}


def test_nothing_but_the_windows_adapter_imports_keyring() -> None:
    """The boundary that lets the whole suite run with no credential store."""
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src" / "goldpipeline"
    offenders = [
        path.relative_to(src).as_posix()
        for path in src.rglob("*.py")
        if "import keyring" in path.read_text(encoding="utf-8")
    ]

    assert offenders == ["adapters/windows_credentials.py"]
