"""The credential commands, and the leak surface around them.

Nothing here touches a real credential manager: the store is substituted at one
seam, and the hidden prompt is answered by a stub. Most of the file is negative
space - what these commands must *not* accept, print, or write.

The flag that does not exist is the point of several tests. A secret passed as
``--value`` lands in shell history, in every process listing on the machine, and
in the terminal transcript an operator later pastes into a support thread.
"""

from __future__ import annotations

import json
from pathlib import Path
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

from goldpipeline import cli
from goldpipeline.adapters.windows_credentials import (
    SERVICE_NAME,
    WindowsCredentialSecretProvider,
)
from goldpipeline.cli import EXIT_ERROR, EXIT_INVALID_DATA, EXIT_OK, main

SENTINELS = (FAKE_API_KEY, FAKE_OPENAI_KEY, TELEGRAM_TOKEN_SENTINEL)


def invoke(args: list[str]) -> int:
    return main(args)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeKeyringModule:
    """Substitute an offline credential store at the CLI's single seam."""
    module = FakeKeyringModule()
    monkeypatch.setattr(cli, "_credential_store", lambda: WindowsCredentialSecretProvider(module))
    monkeypatch.setattr(cli, "inspect_backend", lambda *a, **k: _report(module))
    return module


def _report(module: FakeKeyringModule) -> Any:
    from goldpipeline.adapters.windows_credentials import inspect_backend

    return inspect_backend(module)


@pytest.fixture(autouse=True)
def no_ambient_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from an environment with no credentials in it."""
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TELEGRAM_BOT_TOKEN", "INGEST_TOKEN"):
        monkeypatch.delenv(name, raising=False)


# --- status ----------------------------------------------------------------


def test_status_reports_missing_credentials(
    store: FakeKeyringModule, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 36: the source is MISSING when nothing has one."""
    code = invoke(["secrets-status", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert {entry["name"]: entry["source"] for entry in payload["secrets"]} == {
        "ANTHROPIC_API_KEY": "MISSING",
        "DEEPSEEK_API_KEY": "MISSING",
        "OPENAI_API_KEY": "MISSING",
        "TELEGRAM_BOT_TOKEN": "MISSING",
        "INGEST_TOKEN": "MISSING",
    }


def test_status_names_the_process_environment_as_the_source(
    store: FakeKeyringModule, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 34.

    The distinction that matters for a scheduled task: this works now, but will
    not survive into a fresh process.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)

    invoke(["secrets-status", "--json"])
    payload = json.loads(capsys.readouterr().out)
    sources = {entry["name"]: entry["source"] for entry in payload["secrets"]}

    assert sources["ANTHROPIC_API_KEY"] == "PROCESS_ENV"


def test_status_names_the_credential_store_as_the_source(
    store: FakeKeyringModule, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 35, and the state a scheduled task actually needs."""
    store.stored[(SERVICE_NAME, "telegram_bot_token")] = TELEGRAM_TOKEN_SENTINEL

    invoke(["secrets-status", "--json"])
    payload = json.loads(capsys.readouterr().out)
    sources = {entry["name"]: entry["source"] for entry in payload["secrets"]}

    assert sources["TELEGRAM_BOT_TOKEN"] == "WINDOWS_CREDENTIAL_MANAGER"


def test_the_environment_wins_in_the_status_too(
    store: FakeKeyringModule, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 9, reported honestly rather than flattened to 'configured'."""
    store.stored[(SERVICE_NAME, "anthropic_api_key")] = FAKE_API_KEY
    monkeypatch.setenv("ANTHROPIC_API_KEY", "session-override")

    invoke(["secrets-status", "--json"])
    payload = json.loads(capsys.readouterr().out)
    sources = {entry["name"]: entry["source"] for entry in payload["secrets"]}

    assert sources["ANTHROPIC_API_KEY"] == "PROCESS_ENV"


def test_status_never_prints_a_value(
    store: FakeKeyringModule, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirements 21, 28 and 29."""
    store.stored[(SERVICE_NAME, "anthropic_api_key")] = FAKE_API_KEY
    store.stored[(SERVICE_NAME, "telegram_bot_token")] = TELEGRAM_TOKEN_SENTINEL
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI_KEY)

    invoke(["secrets-status"])
    captured = capsys.readouterr()

    for sentinel in SENTINELS:
        assert sentinel not in captured.out
        assert sentinel not in captured.err
    assert "configured" in captured.out


def test_status_json_never_carries_a_value(
    store: FakeKeyringModule, capsys: pytest.CaptureFixture[str]
) -> None:
    store.stored[(SERVICE_NAME, "anthropic_api_key")] = FAKE_API_KEY

    invoke(["secrets-status", "--json"])
    out = capsys.readouterr().out

    assert FAKE_API_KEY not in out
    assert json.loads(out)["secrets"][0]["configured"] is True


# --- setting ---------------------------------------------------------------


def test_setting_a_credential_prompts_invisibly(
    store: FakeKeyringModule, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 22, and the reason there is no flag for this.

    ``getpass`` reads from the terminal without echoing, so the value never
    reaches the transcript, the scrollback, or shell history.
    """
    import getpass

    asked: list[str] = []

    def prompt(message: str = "") -> str:
        asked.append(message)
        return TELEGRAM_TOKEN_SENTINEL

    monkeypatch.setattr(getpass, "getpass", prompt)

    code = invoke(["secrets-set", "telegram"])
    captured = capsys.readouterr()

    assert code == EXIT_OK
    assert asked, "the value must be prompted for, not read from arguments"
    assert store.stored == {(SERVICE_NAME, "telegram_bot_token"): TELEGRAM_TOKEN_SENTINEL}
    assert TELEGRAM_TOKEN_SENTINEL not in captured.out
    assert TELEGRAM_TOKEN_SENTINEL not in captured.err


@pytest.mark.parametrize(
    ("short", "entry"),
    [
        ("anthropic", "anthropic_api_key"),
        ("openai", "openai_api_key"),
        ("telegram", "telegram_bot_token"),
    ],
)
def test_each_credential_can_be_stored(
    store: FakeKeyringModule, monkeypatch: pytest.MonkeyPatch, short: str, entry: str
) -> None:
    import getpass

    monkeypatch.setattr(getpass, "getpass", lambda *a, **k: "a-value")

    assert invoke(["secrets-set", short]) == EXIT_OK
    assert (SERVICE_NAME, entry) in store.stored


@pytest.mark.parametrize("rejected", ["--value", "--token", "--api-key", "--secret", "--key"])
def test_there_is_no_way_to_pass_a_secret_as_an_argument(
    capsys: pytest.CaptureFixture[str], rejected: str
) -> None:
    """Requirements 23 and 24.

    A secret on a command line lands in shell history and in every process
    listing on the machine, where it outlives the terminal it was typed into.
    """
    from goldpipeline.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["secrets-set", "telegram", rejected, "a-value"])
    capsys.readouterr()


def test_an_insecure_backend_refuses_to_store_anything(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 25 of the spec, at the command boundary."""
    import getpass

    module = plaintext_backend_module()
    monkeypatch.setattr(cli, "_credential_store", lambda: WindowsCredentialSecretProvider(module))
    monkeypatch.setattr(cli, "inspect_backend", lambda *a, **k: _report(module))
    monkeypatch.setattr(getpass, "getpass", lambda *a, **k: FAKE_API_KEY)

    code = invoke(["secrets-set", "anthropic"])
    captured = capsys.readouterr()

    assert code == EXIT_ERROR
    assert module.stored == {}
    assert "Nothing was written" in captured.err
    assert FAKE_API_KEY not in captured.err


def test_a_missing_backend_refuses_and_creates_no_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 25: no `.env`, no environment variable, nothing that 'works'."""
    import getpass

    module = fail_backend_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_credential_store", lambda: WindowsCredentialSecretProvider(module))
    monkeypatch.setattr(cli, "inspect_backend", lambda *a, **k: _report(module))
    monkeypatch.setattr(getpass, "getpass", lambda *a, **k: FAKE_API_KEY)

    assert invoke(["secrets-set", "anthropic"]) == EXIT_ERROR
    capsys.readouterr()

    assert list(tmp_path.iterdir()) == []


def test_an_empty_answer_stores_nothing(
    store: FakeKeyringModule, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import getpass

    monkeypatch.setattr(getpass, "getpass", lambda *a, **k: "   ")

    code = invoke(["secrets-set", "anthropic"])
    capsys.readouterr()

    assert code == EXIT_INVALID_DATA
    assert store.stored == {}


def test_cancelling_the_prompt_stores_nothing(
    store: FakeKeyringModule, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import getpass

    def cancel(*args: Any, **kwargs: Any) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(getpass, "getpass", cancel)

    code = invoke(["secrets-set", "openai"])
    err = capsys.readouterr().err

    assert code == EXIT_ERROR
    assert "Cancelled" in err
    assert store.stored == {}


# --- deleting --------------------------------------------------------------


def test_deleting_removes_only_that_credential(
    store: FakeKeyringModule, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 25 of the test list."""
    store.stored[(SERVICE_NAME, "anthropic_api_key")] = FAKE_API_KEY
    store.stored[(SERVICE_NAME, "openai_api_key")] = FAKE_OPENAI_KEY

    code = invoke(["secrets-delete", "openai"])
    capsys.readouterr()

    assert code == EXIT_OK
    assert list(store.stored) == [(SERVICE_NAME, "anthropic_api_key")]


def test_deleting_something_absent_is_calm(
    store: FakeKeyringModule, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 26: the operator's intent is already achieved."""
    code = invoke(["secrets-delete", "telegram"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "Nothing to remove" in out


# --- preflight -------------------------------------------------------------


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


def test_preflight_reports_where_each_credential_came_from(
    store: FakeKeyringModule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirement 33 and requirement 16 of the spec."""
    store.stored[(SERVICE_NAME, "anthropic_api_key")] = FAKE_API_KEY
    store.stored[(SERVICE_NAME, "openai_api_key")] = FAKE_OPENAI_KEY
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TELEGRAM_TOKEN_SENTINEL)

    invoke(preflight_args(tmp_path, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert payload["anthropic"] == "configured (Windows Credential Manager)"
    assert payload["openai"] == "configured (Windows Credential Manager)"
    assert payload["telegram"] == "configured (process environment)"


def test_preflight_warns_that_a_session_credential_will_not_survive(
    store: FakeKeyringModule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The trap this whole round exists to close.

    Variables set with ``$env:`` live in one process tree. Task Scheduler starts
    a new one, so a preflight that called this READY would be lying.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI_KEY)

    invoke(preflight_args(tmp_path, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert payload["task_readiness"] == "NOT_READY"
    assert any("this session only" in blocker for blocker in payload["blockers"])


def test_preflight_is_ready_when_the_store_holds_everything(
    store: FakeKeyringModule,
    production_config: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """What a scheduled task actually needs: nothing session-bound.

    Round 9.2.1 added the production configuration to that list, so readiness
    now requires the file the worker will read as well as the credentials it
    will resolve.
    """
    store.stored[(SERVICE_NAME, "anthropic_api_key")] = FAKE_API_KEY
    store.stored[(SERVICE_NAME, "openai_api_key")] = FAKE_OPENAI_KEY
    production_config()

    invoke(preflight_args(tmp_path, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert payload["credential_backend_secure"] is True
    assert payload["task_readiness"] == "READY"
    assert payload["blockers"] == []


def test_preflight_never_prints_a_value(
    store: FakeKeyringModule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirement 53 of Round 9, still true now that sources are reported."""
    store.stored[(SERVICE_NAME, "anthropic_api_key")] = FAKE_API_KEY
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TELEGRAM_TOKEN_SENTINEL)

    invoke(preflight_args(tmp_path))
    captured = capsys.readouterr()

    for sentinel in SENTINELS:
        assert sentinel not in captured.out
        assert sentinel not in captured.err


def test_preflight_flags_an_absent_credential_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A scheduled task with no vault has nowhere to get credentials from."""
    module = fail_backend_module()
    monkeypatch.setattr(cli, "inspect_backend", lambda *a, **k: _report(module))

    invoke(preflight_args(tmp_path, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert payload["credential_backend_secure"] is False
    assert any("no secure credential store" in blocker for blocker in payload["blockers"])


# --- nothing leaks into artifacts -----------------------------------------


def test_the_task_definition_is_still_secret_free(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirements 17 and 31.

    The whole reason credentials moved into the credential store is so this file
    never needs one. Re-checked with credentials actually present.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI_KEY)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TELEGRAM_TOKEN_SENTINEL)

    invoke(["automation-task-plan", "--xml"])
    xml = capsys.readouterr().out

    for sentinel in SENTINELS:
        assert sentinel not in xml
    for forbidden in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TELEGRAM_BOT_TOKEN"):
        assert forbidden not in xml
    assert "InteractiveToken" in xml
    assert "SYSTEM" not in xml


def test_automation_state_and_history_hold_no_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 32."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TELEGRAM_TOKEN_SENTINEL)
    automation = tmp_path / "automation"

    invoke(
        [
            "automation-run-once",
            "--inbox-dir",
            str(tmp_path / "inbox"),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--automation-dir",
            str(automation),
            "--fake-mt5",
            "--fake-ai",
        ]
    )
    capsys.readouterr()

    written = "".join(p.read_text(encoding="utf-8") for p in automation.rglob("*.json"))
    for sentinel in SENTINELS:
        assert sentinel not in written


def test_the_credential_commands_open_no_socket(
    store: FakeKeyringModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 40."""
    import socket

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("credential handling must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", explode)
    assert invoke(["secrets-status"]) == EXIT_OK
