"""Round 9.2.1: the scheduled worker fails closed on its configuration.

The defect these tests pin down was found in production, not in review. The
persisted configuration file was unreachable, the layered loader read that as
"no settings", every value came from a built-in default - including
``AUTOMATION_ENABLED=false`` - and the worker reported ``exit 0`` every minute
for seven hours. Task Scheduler's history was green throughout. Nothing was
wrong with the code that ran; what was wrong is that *nothing ran*, and the
evidence for "switched off deliberately" and "configuration gone" was identical.

So the properties worth defending are:

* an unattended worker never fills a production setting from a default;
* a missing or unusable configuration exits non-zero, having touched nothing;
* an explicit ``false`` is visibly different from an absent file;
* every tick records which configuration it read, by fingerprint, so the
  question "was the scheduler reading this file?" has an answer on disk.

The two kill switches get their own tests because their default is ``false``,
which is also what an absent file looks like. They are the settings where a
silent default is indistinguishable from a decision.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from conftest import COMPLETE_PRODUCTION_CONFIG, make_event_payload, write_json

from goldpipeline import cli
from goldpipeline.adapters.production_config import (
    inspect_production_config,
    load_production_config,
    production_config_path,
)
from goldpipeline.cli import EXIT_BLOCKED, EXIT_ERROR, EXIT_OK, main
from goldpipeline.domain.errors import (
    ConfigPathUnavailableError,
    PersistentConfigIncompleteError,
    PersistentConfigInvalidJsonError,
    PersistentConfigSecretKeyError,
    PersistentConfigUnknownKeyError,
    ProductionConfigError,
)
from goldpipeline.schemas.runtime_config import (
    REQUIRED_PRODUCTION_KEYS,
    ConfigKey,
    ConfigMode,
    ProductionConfigStatus,
)


def invoke(args: list[str]) -> int:
    return main(args)


@pytest.fixture
def dirs(tmp_path: Path) -> dict[str, Path]:
    return {
        "inbox": tmp_path / "inbox",
        "runs": tmp_path / "runs",
        "automation": tmp_path / "automation",
    }


def worker_args(dirs: dict[str, Path], *extra: str) -> list[str]:
    return [
        "automation-worker-tick",
        "--inbox-dir",
        str(dirs["inbox"]),
        "--runs-dir",
        str(dirs["runs"]),
        "--automation-dir",
        str(dirs["automation"]),
        "--fake-mt5",
        "--fake-ai",
        *extra,
    ]


def submit(dirs: dict[str, Path], tmp_path: Path) -> None:
    """Put one fresh event in the inbox through the ordinary command."""
    from datetime import UTC, datetime

    payload = make_event_payload(created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"))
    path = write_json(tmp_path / "event.json", payload)
    invoke(["inbox-submit", "--file", str(path), "--inbox-dir", str(dirs["inbox"])])


def config_path(tmp_path: Path) -> Path:
    """The file both readers are redirected to by the autouse guard."""
    return tmp_path / "appdata" / "config.json"


# --- the worker runs, or explicitly declines ------------------------------


def test_a_valid_configuration_lets_the_scheduled_worker_run(
    dirs: dict[str, Path],
    tmp_path: Path,
    production_config: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirement 22.1."""
    submit(dirs, tmp_path)
    production_config(GOLDPIPELINE_AUTOMATION_ENABLED="true")
    capsys.readouterr()

    code = invoke(worker_args(dirs, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["status"] == "OK"
    assert payload["config_mode"] == "STRICT_PERSISTENT"
    assert len(payload["processed_events"]) == 1


def test_an_explicit_false_is_a_healthy_disabled_tick(
    dirs: dict[str, Path],
    tmp_path: Path,
    production_config: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirements 22.2 and 11.

    Zero exit, and the output says the file was read. That last part is the
    whole point of the round: "off" must be evidence of a decision.
    """
    submit(dirs, tmp_path)
    production_config(GOLDPIPELINE_AUTOMATION_ENABLED="false")
    capsys.readouterr()

    code = invoke(worker_args(dirs, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["status"] == "DISABLED"
    assert payload["config_sha256"], "a healthy disabled tick still names its configuration"
    assert payload["config_mode"] == "STRICT_PERSISTENT"


def test_a_missing_configuration_fails_closed(
    dirs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirements 22.3 and 12.

    Non-zero, because ``Last Result: 0`` beside "nothing was done" is the exact
    signature the original defect wore for seven hours.
    """
    submit(dirs, tmp_path)
    capsys.readouterr()

    code = invoke(worker_args(dirs))
    captured = capsys.readouterr()

    assert code == EXIT_ERROR
    assert "PERSISTENT_CONFIG_NOT_FOUND" in captured.err
    assert "No pipeline work was attempted." in captured.err


def test_a_missing_configuration_is_not_read_as_automation_off(
    dirs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 22.4 - the regression itself.

    The old behaviour produced the word "disabled" and a zero exit. Both are
    forbidden here: an absent file must never be reported as a configured off.
    """
    submit(dirs, tmp_path)
    capsys.readouterr()

    code = invoke(worker_args(dirs))
    captured = capsys.readouterr()

    assert code != EXIT_OK
    assert "disabled" not in (captured.out + captured.err).lower()


# --- a failed configuration check touches nothing -------------------------


def test_a_missing_configuration_reaches_no_provider_and_claims_no_event(
    dirs: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirements 22.5 to 22.8, asserted at the one place they all pass.

    MT5, both AI clients, the publisher and the inbox claim are all reached
    through the worker context. Refusing before it is built is what makes "no
    MT5 call, no AI call, no Telegram call, no claim" a single structural fact
    rather than four separate hopes - so the assertion is that the context is
    never constructed, plus the observable consequence in the inbox.
    """
    import socket

    submit(dirs, tmp_path)
    pending_before = sorted(p.name for p in (dirs["inbox"] / "incoming").glob("*.json"))
    assert pending_before, "the event is waiting before the tick"

    def refuse_context(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a failed configuration check must build no worker context")

    def refuse_socket(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a failed configuration check must open no socket")

    monkeypatch.setattr(cli, "_worker_context", refuse_context)
    monkeypatch.setattr(socket.socket, "connect", refuse_socket)
    capsys.readouterr()

    assert invoke(worker_args(dirs)) == EXIT_ERROR

    pending_after = sorted(p.name for p in (dirs["inbox"] / "incoming").glob("*.json"))
    assert pending_after == pending_before, "the event was not claimed"
    assert not dirs["runs"].exists(), "no Run was created"


# --- every way a configuration can be unusable ----------------------------


def test_invalid_json_fails_closed(
    dirs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 22.9."""
    path = config_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schema_version": "1.0.0", "values": {', encoding="utf-8")
    capsys.readouterr()

    code = invoke(worker_args(dirs))

    assert code == EXIT_ERROR
    assert "PERSISTENT_CONFIG_INVALID_JSON" in capsys.readouterr().err


def test_an_unreadable_configuration_fails_closed(
    tmp_path: Path, production_config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 22.10.

    Simulated rather than produced with an ACL: the property under test is that
    an ``OSError`` on read becomes a refusal, not that Windows permissions work.
    """
    production_config()
    path = config_path(tmp_path)

    original = Path.read_bytes

    def explode(self: Path) -> bytes:
        if self == path:
            raise PermissionError(13, "denied")
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", explode)

    with pytest.raises(ProductionConfigError) as caught:
        load_production_config(path)
    assert caught.value.code == "PERSISTENT_CONFIG_UNREADABLE"


@pytest.mark.parametrize("missing", sorted(key.value for key in REQUIRED_PRODUCTION_KEYS))
def test_every_required_key_is_required(
    missing: str, tmp_path: Path, production_config: Any
) -> None:
    """Requirements 22.11 to 22.13, once per approved setting.

    Parametrised over the whole set rather than a chosen sample, so a key added
    later cannot quietly become optional. The two kill switches are in here by
    construction, which is the case that matters most.
    """
    production_config(**{missing: None})

    with pytest.raises(PersistentConfigIncompleteError) as caught:
        load_production_config(config_path(tmp_path))
    assert missing in caught.value.details["missing"]


def test_an_empty_value_counts_as_missing(tmp_path: Path, production_config: Any) -> None:
    """Present-but-blank is absent. A key with no value configures nothing."""
    production_config(GOLDPIPELINE_AUTOMATION_ENABLED="   ")

    with pytest.raises(PersistentConfigIncompleteError):
        load_production_config(config_path(tmp_path))


def test_an_unparseable_kill_switch_is_refused(tmp_path: Path, production_config: Any) -> None:
    """A switch that cannot be read is not a switch."""
    production_config(GOLDPIPELINE_AUTOMATION_ENABLED="maybe")

    with pytest.raises(PersistentConfigIncompleteError):
        load_production_config(config_path(tmp_path))


def test_an_unknown_key_is_rejected(tmp_path: Path) -> None:
    """Requirement 22.14.

    Refused rather than ignored: an unknown key is usually a misspelt known one,
    and dropping it silently leaves the setting the operator meant to configure
    sitting at a default they never chose.
    """
    values = dict(COMPLETE_PRODUCTION_CONFIG)
    values["GOLDPIPELINE_NOT_A_SETTING"] = "1"
    path = config_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": "1.0.0", "values": values}), encoding="utf-8")

    with pytest.raises(PersistentConfigUnknownKeyError):
        load_production_config(path)


@pytest.mark.parametrize("secret", ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TELEGRAM_BOT_TOKEN"])
def test_a_credential_name_is_refused(secret: str, tmp_path: Path) -> None:
    """Requirement 22.15 and requirement 8.

    Refused by name, with a message that says where credentials actually live.
    """
    values = dict(COMPLETE_PRODUCTION_CONFIG)
    values[secret] = "should-never-be-here"
    path = config_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": "1.0.0", "values": values}), encoding="utf-8")

    with pytest.raises(PersistentConfigSecretKeyError) as caught:
        load_production_config(path)
    assert "should-never-be-here" not in caught.value.message, "the value is never echoed"
    assert "secrets-set" in caught.value.message


def test_an_unexpected_schema_version_is_refused(tmp_path: Path) -> None:
    """A document this build does not understand is not read optimistically."""
    path = config_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": "9.9.9", "values": dict(COMPLETE_PRODUCTION_CONFIG)}),
        encoding="utf-8",
    )

    with pytest.raises(ProductionConfigError) as caught:
        load_production_config(path)
    assert caught.value.code == "PERSISTENT_CONFIG_SCHEMA_MISMATCH"


# --- path resolution ------------------------------------------------------


def test_windows_without_localappdata_is_an_error() -> None:
    """Requirement 22.16."""
    with pytest.raises(ConfigPathUnavailableError):
        production_config_path({}, windows=True)


def test_windows_never_falls_back_to_the_home_config_directory() -> None:
    """Requirement 22.17 - the specific fallback that hid the defect.

    ``~/.config`` on Windows is a path nobody configures and nobody inspects, so
    a worker reading it finds nothing and says so in the quietest possible way.
    """
    with pytest.raises(ConfigPathUnavailableError):
        production_config_path({"XDG_CONFIG_HOME": r"C:\somewhere"}, windows=True)


def test_localappdata_decides_the_path_on_windows() -> None:
    resolved = production_config_path({"LOCALAPPDATA": r"C:\Users\x\AppData\Local"}, windows=True)
    assert resolved.parts[-2:] == ("GoldAnalysisPipeline", "config.json")


def test_other_platforms_may_use_the_xdg_location(tmp_path: Path) -> None:
    """Not a production path; it exists so the suite runs off Windows."""
    resolved = production_config_path({"XDG_CONFIG_HOME": str(tmp_path)}, windows=False)
    assert resolved == tmp_path / "GoldAnalysisPipeline" / "config.json"


# --- operator mode is deliberately unchanged ------------------------------


def test_operator_commands_still_get_built_in_defaults(
    dirs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 22.18.

    ``automation-run-once`` is a person's decision, and a person investigating a
    machine with no configuration should not be stopped by the rule that exists
    to protect an unattended one.
    """
    submit(dirs, tmp_path)
    capsys.readouterr()

    code = invoke(
        [
            "automation-run-once",
            "--inbox-dir",
            str(dirs["inbox"]),
            "--runs-dir",
            str(dirs["runs"]),
            "--automation-dir",
            str(dirs["automation"]),
            "--fake-mt5",
            "--fake-ai",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["config_mode"] == "LAYERED"
    assert len(payload["processed_events"]) == 1


def test_the_environment_still_overrides_configuration_in_operator_mode(
    tmp_path: Path, production_config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 22.19 - the layered contract, unchanged."""
    from goldpipeline.adapters.config_store import LayeredConfig, RuntimeConfigStore

    production_config(GOLDPIPELINE_MT5_SYMBOL="XAUUSD")
    layered = LayeredConfig(
        {"GOLDPIPELINE_MT5_SYMBOL": "XAGUSD"}, RuntimeConfigStore(config_path(tmp_path)).load()
    )
    entry = layered.resolve(ConfigKey.MT5_SYMBOL)

    assert entry.value == "XAGUSD"
    assert entry.source.value == "PROCESS_ENV"


def test_the_environment_cannot_override_configuration_in_strict_mode(
    tmp_path: Path, production_config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half, and the reason the fingerprint means anything.

    If a session variable could change what a tick did, ``config_sha256`` would
    describe a file rather than the settings actually used.
    """
    production_config(GOLDPIPELINE_MT5_SYMBOL="XAUUSD")
    monkeypatch.setenv("GOLDPIPELINE_MT5_SYMBOL", "XAGUSD")

    config = load_production_config(config_path(tmp_path))

    assert config.values[ConfigKey.MT5_SYMBOL] == "XAUUSD"


# --- the fingerprint ------------------------------------------------------


def test_the_fingerprint_is_stable_across_reads(tmp_path: Path, production_config: Any) -> None:
    """Requirement 22.20."""
    production_config()
    first = load_production_config(config_path(tmp_path))
    second = load_production_config(config_path(tmp_path))

    assert first.sha256 == second.sha256


def test_the_fingerprint_is_the_digest_of_the_file(tmp_path: Path, production_config: Any) -> None:
    """Verifiable by anyone with the file and a hashing tool."""
    production_config()
    path = config_path(tmp_path)

    assert load_production_config(path).sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_changing_a_setting_changes_the_fingerprint(tmp_path: Path, production_config: Any) -> None:
    """Requirement 22.21."""
    production_config(GOLDPIPELINE_OHLC_BARS="20")
    before = load_production_config(config_path(tmp_path)).sha256

    production_config(GOLDPIPELINE_OHLC_BARS="40")
    after = load_production_config(config_path(tmp_path)).sha256

    assert before != after


def test_the_file_is_read_as_utf8(tmp_path: Path) -> None:
    """Requirement 22.31, at the only boundary where it can be shown.

    No approved setting legitimately holds non-ASCII text - symbols, timeframes,
    minute counts and an ``@username`` are all ASCII by their own validators - so
    a "value survives a round trip" test would be asserting something the schema
    forbids. What is worth pinning down is that the *reader* is UTF-8: a file
    containing multi-byte characters must be decoded correctly and then judged
    on its contents, rather than failing as an encoding error or arriving as
    mojibake and being refused for the wrong reason.
    """
    path = config_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = dict(COMPLETE_PRODUCTION_CONFIG)
    values["TELEGRAM_TARGET_CHAT_ID"] = "@kênh_vàng"
    path.write_bytes(
        json.dumps({"schema_version": "1.0.0", "values": values}, ensure_ascii=False).encode(
            "utf-8"
        )
    )

    with pytest.raises(ProductionConfigError) as caught:
        load_production_config(path)

    # Refused by the destination rule, not by the decoder - which is only
    # possible if the multi-byte characters were read correctly.
    assert caught.value.code == "PERSISTENT_CONFIG_INCOMPLETE"
    assert "TELEGRAM_TARGET_CHAT_ID" in caught.value.message


def test_invalid_utf8_is_reported_as_such(tmp_path: Path) -> None:
    """The neighbouring case, so the two failures stay distinguishable."""
    path = config_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'{"schema_version": "1.0.0", "values": {"\xff\xfe": "x"}}')

    with pytest.raises(PersistentConfigInvalidJsonError):
        load_production_config(path)


# --- tick provenance ------------------------------------------------------


def latest_history(dirs: dict[str, Path]) -> dict[str, Any]:
    records = sorted((dirs["automation"] / "history").glob("*.json"))
    assert records, "the tick recorded no history"
    parsed: dict[str, Any] = json.loads(records[-1].read_text(encoding="utf-8"))
    return parsed


def test_tick_history_records_the_configuration_it_read(
    dirs: dict[str, Path],
    tmp_path: Path,
    production_config: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirements 22.22, 22.23 and 9 - the evidence that was missing.

    With this on disk, "was the scheduler reading this file?" is a comparison
    rather than an investigation.
    """
    submit(dirs, tmp_path)
    production_config(GOLDPIPELINE_AUTOMATION_ENABLED="true")
    capsys.readouterr()

    invoke(worker_args(dirs))
    record = latest_history(dirs)

    assert record["config_sha256"] == load_production_config(config_path(tmp_path)).sha256
    assert record["config_path"] == str(config_path(tmp_path))
    assert record["config_mode"] == ConfigMode.STRICT_PERSISTENT.value
    assert record["config_schema_version"] == "1.0.0"
    assert record["automation_enabled"] is True
    assert record["code_version"], "the record names the code as well as the configuration"


def test_tick_history_holds_no_settings_and_no_credentials(
    dirs: dict[str, Path],
    tmp_path: Path,
    production_config: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirements 22.24 and 22.25.

    A fingerprint, not a copy. The record is written every minute and read
    during an incident; both argue for it staying small, and neither argues for
    duplicating settings that are already readable in the file it names.
    """
    from conftest import FAKE_API_KEY, FAKE_OPENAI_KEY, TELEGRAM_TOKEN_SENTINEL

    submit(dirs, tmp_path)
    production_config(GOLDPIPELINE_AUTOMATION_ENABLED="true", TELEGRAM_TARGET_CHAT_ID="@secretish")
    capsys.readouterr()

    invoke(worker_args(dirs))
    raw = json.dumps(latest_history(dirs))

    for sentinel in (FAKE_API_KEY, FAKE_OPENAI_KEY, TELEGRAM_TOKEN_SENTINEL):
        assert sentinel not in raw
    assert "@secretish" not in raw, "settings are named by digest, never copied"
    assert "XAUUSD" not in raw


# --- diagnostics ----------------------------------------------------------


def test_automation_status_reports_a_healthy_configuration(
    dirs: dict[str, Path],
    tmp_path: Path,
    production_config: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirements 22.26 and 13."""
    production_config()
    capsys.readouterr()

    invoke(
        [
            "automation-status",
            "--inbox-dir",
            str(dirs["inbox"]),
            "--runs-dir",
            str(dirs["runs"]),
            "--automation-dir",
            str(dirs["automation"]),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["persistent_config"]["status"] == ProductionConfigStatus.FOUND.value
    assert payload["persistent_config"]["sha256"]
    assert payload["scheduled_strict_mode"] == "READY"


def test_automation_status_reports_a_missing_configuration(
    dirs: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 22.27.

    The status command survives what the worker refuses. Describing a broken
    machine and running on one are opposite requirements.
    """
    capsys.readouterr()

    code = invoke(
        [
            "automation-status",
            "--inbox-dir",
            str(dirs["inbox"]),
            "--runs-dir",
            str(dirs["runs"]),
            "--automation-dir",
            str(dirs["automation"]),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["persistent_config"]["status"] == ProductionConfigStatus.MISSING.value
    assert payload["scheduled_strict_mode"] == "NOT_READY"


def test_preflight_blocks_on_a_missing_configuration(
    dirs: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirements 22.28 and 14 - the check that would have caught this on day one."""
    capsys.readouterr()

    code = invoke(
        [
            "automation-preflight",
            "--inbox-dir",
            str(dirs["inbox"]),
            "--runs-dir",
            str(dirs["runs"]),
            "--automation-dir",
            str(dirs["automation"]),
            "--fake-mt5",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_BLOCKED
    assert payload["scheduled_config"] == "NOT_READY"
    assert payload["task_readiness"] == "NOT_READY"
    assert any("PERSISTENT_CONFIG_NOT_FOUND" in blocker for blocker in payload["blockers"])


def test_task_status_compares_the_configuration_with_the_last_tick(
    dirs: dict[str, Path],
    tmp_path: Path,
    production_config: Any,
    task_scheduler: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirements 22.29 and 15.

    A drifted scheduler is the failure this reports, and it is the one that took
    an evening to diagnose by hand.
    """
    submit(dirs, tmp_path)
    production_config(GOLDPIPELINE_AUTOMATION_ENABLED="true")
    invoke(worker_args(dirs))
    capsys.readouterr()

    invoke(["automation-task-status", "--automation-dir", str(dirs["automation"]), "--json"])
    matched = json.loads(capsys.readouterr().out)
    assert matched["config_match"] == "YES"
    assert matched["last_tick_config_sha256"] == matched["current_config_sha256"]

    # The operator edits the machine; the last tick now describes something else.
    production_config(GOLDPIPELINE_AUTOMATION_ENABLED="true", GOLDPIPELINE_OHLC_BARS="40")
    invoke(["automation-task-status", "--automation-dir", str(dirs["automation"]), "--json"])
    drifted = json.loads(capsys.readouterr().out)

    assert drifted["config_match"] == "NO"


def test_task_status_says_so_when_nothing_has_ticked_yet(
    dirs: dict[str, Path],
    production_config: Any,
    task_scheduler: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Absent evidence is reported as absent, never as a mismatch."""
    production_config()
    capsys.readouterr()

    invoke(["automation-task-status", "--automation-dir", str(dirs["automation"]), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["config_match"] == "NO_TICK_YET"


# --- the report, as a whole -----------------------------------------------


def test_inspection_never_raises_on_a_broken_machine(tmp_path: Path) -> None:
    """Every diagnostic path returns a report rather than an exception."""
    report = inspect_production_config(tmp_path / "nothing" / "here.json")

    assert report.status is ProductionConfigStatus.MISSING
    assert report.ready is False
    assert report.error_code == "PERSISTENT_CONFIG_NOT_FOUND"


def test_a_healthy_report_carries_no_setting_values(tmp_path: Path, production_config: Any) -> None:
    """Safe to print anywhere, including a support ticket."""
    production_config(TELEGRAM_TARGET_CHAT_ID="@somewhere")

    raw = inspect_production_config(config_path(tmp_path)).model_dump_json()

    assert "@somewhere" not in raw
    assert ProductionConfigStatus.FOUND.value in raw


def test_the_production_fixture_keeps_auto_publish_off() -> None:
    """Requirement 22.33.

    A fixture that quietly enabled publishing would make every other test in
    this file a much more expensive experiment.
    """
    assert COMPLETE_PRODUCTION_CONFIG["GOLDPIPELINE_AUTOPUBLISH_ENABLED"] == "false"
    assert COMPLETE_PRODUCTION_CONFIG["GOLDPIPELINE_AUTOMATION_ENABLED"] == "false"


def test_the_fixture_covers_every_required_key() -> None:
    """The fixture and the requirement cannot drift apart unnoticed."""
    assert set(COMPLETE_PRODUCTION_CONFIG) == {key.value for key in REQUIRED_PRODUCTION_KEYS}
