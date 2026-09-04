"""Generation clients resolve their credential through the project's provider.

Round 6.4e.1, and the bug it fixes is worth stating plainly because the shape
recurs. ``build_writer_client`` was introduced as the seam that turns a Run's
frozen selection into a client. It was given the selection and nothing else -
no configuration mapping and no credential provider - so it fell through to the
settings loader's default, which is the process environment alone.

The branch beside it, for Runs with no selection, passed both. So the two paths
agreed on any machine where a developer had exported ``ANTHROPIC_API_KEY``, and
disagreed only under the scheduled task, which inherits no session and reads its
credential from Windows Credential Manager. Every test passed. Every manual run
worked. The first unattended Run carrying a selection would have failed.

So the tests here are deliberately built the awkward way round: the process
environment is *emptied* of the credential and the secret is placed only in an
offline credential store. That is the production shape, and it is the only shape
in which the defect is visible.

Every test is offline. No provider is contacted, no real credential manager is
read, and no HTTP client is constructed with a live transport.
"""

from __future__ import annotations

import argparse
import inspect as inspect_module
import os
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeKeyringModule

from goldpipeline import cli
from goldpipeline.adapters.finalizer_client import FinalizerClient
from goldpipeline.adapters.secrets import (
    CompositeSecretProvider,
    EnvironmentSecretProvider,
    FakeSecretProvider,
)
from goldpipeline.adapters.windows_credentials import (
    SERVICE_NAME,
    WindowsCredentialSecretProvider,
    inspect_backend,
)
from goldpipeline.adapters.writer_client import WriterClient
from goldpipeline.config import WriterSettings
from goldpipeline.domain.errors import FinalizeConfigurationError, WriterConfigurationError
from goldpipeline.schemas.generation import GenerationSelection
from goldpipeline.schemas.preferences import PreferencesSource, Provider, ThinkingMode
from goldpipeline.schemas.secrets import SecretName
from goldpipeline.services import generation
from goldpipeline.services.generation import build_finalizer_client, build_writer_client

STORED_KEY = "sk-ant-offline-fake-key-for-tests"
"""Not a credential. Never sent anywhere: no test here builds a transport."""

ENV_KEY = "sk-ant-offline-env-override"
"""A different fake, so precedence is provable rather than assumed."""

SELECTION_ID = "claude-opus-5"

ANTHROPIC_CREDENTIAL = (SERVICE_NAME, "anthropic_api_key")


# --------------------------------------------------------------------------
# offline stand-ins
# --------------------------------------------------------------------------


def anthropic_only(value: str = STORED_KEY) -> FakeSecretProvider:
    """A provider holding the Anthropic key and nothing else."""
    return FakeSecretProvider({SecretName.ANTHROPIC_API_KEY: value})


def empty_provider() -> FakeSecretProvider:
    """A store that exists and holds nothing - not a broken store."""
    return FakeSecretProvider({})


def claude_selection(selection_id: str = SELECTION_ID) -> GenerationSelection:
    """The snapshot a Run created under Claude preferences carries."""
    return GenerationSelection(
        provider=Provider.CLAUDE,
        selection_id=selection_id,
        api_model_id=selection_id,
        thinking=ThinkingMode.NOT_APPLICABLE,
        preference_source=PreferencesSource.FILE,
    )


@pytest.fixture(autouse=True)
def no_generation_key_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The production shape: the scheduled task has no session to inherit.

    Autouse, because a test in this module that accidentally left the variable
    set would pass for the wrong reason - which is exactly how the defect
    survived the round that introduced it.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)


# --------------------------------------------------------------------------
# writer: the credential comes from the supplied provider
# --------------------------------------------------------------------------


@pytest.mark.parametrize("selection_id", ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"])
def test_a_claude_writer_builds_from_the_supplied_provider(selection_id: str) -> None:
    """The property the round exists for, at the generation seam."""
    provider = anthropic_only()

    client = build_writer_client(Provider.CLAUDE, selection_id, secrets=provider)

    assert client.provider == "anthropic"
    assert client.model == selection_id
    assert SecretName.ANTHROPIC_API_KEY in provider.reads


def test_the_writer_keeps_the_snapshotted_model_when_the_key_comes_from_a_store() -> None:
    """Fixing where the key comes from must not move which model is used."""
    selection = claude_selection()

    client = build_writer_client(
        selection.provider, selection.selection_id, secrets=anthropic_only()
    )

    assert client.model == selection.api_model_id == SELECTION_ID


def test_a_claude_writer_without_the_environment_variable_still_builds() -> None:
    """Stated as its own test because it is the whole production failure."""
    client = build_writer_client(Provider.CLAUDE, SELECTION_ID, secrets=anthropic_only())

    assert client.provider == "anthropic"


def test_the_environment_wins_when_both_hold_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Project policy: process environment first, credential store second.

    Documented in :mod:`goldpipeline.adapters.secrets` and load-bearing - it is
    what makes a ten-minute override possible without rewriting a stored
    credential. This hotfix must not quietly invert it, so the precedence is
    asserted through a composite built in that order rather than trusted.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", ENV_KEY)
    composite = CompositeSecretProvider([EnvironmentSecretProvider(), anthropic_only(STORED_KEY)])

    resolved, _ = composite.resolve(SecretName.ANTHROPIC_API_KEY)

    assert resolved == ENV_KEY


def test_a_missing_anthropic_credential_fails_explicitly() -> None:
    """No fallback, no retry, no other provider. The stage names its setting."""
    with pytest.raises(WriterConfigurationError) as caught:
        build_writer_client(Provider.CLAUDE, SELECTION_ID, secrets=empty_provider())

    assert caught.value.details["setting"] == "ANTHROPIC_API_KEY"
    assert "deepseek" not in caught.value.message.lower()


def test_a_missing_credential_costs_no_provider_call() -> None:
    """A configuration failure must happen before anything is dialled."""
    provider = empty_provider()

    with pytest.raises(WriterConfigurationError):
        build_writer_client(Provider.CLAUDE, SELECTION_ID, secrets=provider)

    assert provider.reads == [SecretName.ANTHROPIC_API_KEY]


# --------------------------------------------------------------------------
# finalizer: the same, because failing here is worse
# --------------------------------------------------------------------------


def test_a_claude_finalizer_builds_from_the_supplied_provider() -> None:
    provider = anthropic_only()

    client = build_finalizer_client(Provider.CLAUDE, SELECTION_ID, secrets=provider)

    assert client.provider == "anthropic"
    assert client.model == SELECTION_ID
    assert SecretName.ANTHROPIC_API_KEY in provider.reads


def test_the_finalizer_uses_the_same_snapshot_the_writer_did() -> None:
    """One Run, one model, whichever stage is asking."""
    selection = claude_selection()

    writer = build_writer_client(
        selection.provider, selection.selection_id, secrets=anthropic_only()
    )
    finalizer = build_finalizer_client(
        selection.provider, selection.selection_id, secrets=anthropic_only()
    )

    assert writer.model == finalizer.model == selection.api_model_id


def test_a_missing_finalizer_credential_fails_explicitly() -> None:
    with pytest.raises(FinalizeConfigurationError) as caught:
        build_finalizer_client(Provider.CLAUDE, SELECTION_ID, secrets=empty_provider())

    assert caught.value.details["setting"] == "ANTHROPIC_API_KEY"


# --------------------------------------------------------------------------
# the exact production construction route
# --------------------------------------------------------------------------


def worker_args(tmp_path: Path) -> argparse.Namespace:
    """The namespace the scheduled worker's tick actually carries.

    Built from the same attribute names ``_pipeline_clients`` reads, so this
    exercises the production route rather than a convenient stand-in. Every
    ``fake`` flag is off: a fake short-circuits before any credential is read,
    which would make the test pass without proving anything.
    """
    return argparse.Namespace(
        fake_ai=False,
        fake_writer=False,
        fake_finalizer=False,
        fake_reviewer=False,
        fake_publisher=True,
        runs_dir=tmp_path / "runs",
    )


def production_writer(args: argparse.Namespace) -> WriterClient:
    """Build a writer the way the worker's tick does.

    The factory fields are optional on :class:`PipelineClients`, so the
    assertion narrows the type rather than asserting behaviour - but a ``None``
    here would mean the CLI stopped wiring the stage at all, which is worth
    failing on either way.
    """
    factory = cli._pipeline_clients(args).writer
    assert factory is not None
    return factory(claude_selection())


def production_finalizer(args: argparse.Namespace) -> FinalizerClient:
    """Build a finalizer the way a revision does."""
    factory = cli._pipeline_clients(args).finalizer
    assert factory is not None
    return factory(claude_selection())


@pytest.fixture
def credential_store_holding_the_key(monkeypatch: pytest.MonkeyPatch) -> FakeKeyringModule:
    """An offline Windows Credential Manager, wired into the CLI's seam.

    This is the closest offline thing to the production machine: the CLI's own
    ``_secret_provider`` composite, with a real
    :class:`WindowsCredentialSecretProvider` over a fake keyring module. The
    autouse guard in ``conftest`` reports no backend, so ``inspect_backend`` is
    substituted too - otherwise the composite would drop the store and the test
    would prove only that the environment is empty.
    """
    module = FakeKeyringModule({ANTHROPIC_CREDENTIAL: STORED_KEY})
    monkeypatch.setattr(cli, "_credential_store", lambda: WindowsCredentialSecretProvider(module))
    monkeypatch.setattr(cli, "inspect_backend", lambda *a, **k: inspect_backend(module))
    return module


def test_the_production_route_builds_a_writer_with_no_environment_key(
    tmp_path: Path, credential_store_holding_the_key: FakeKeyringModule
) -> None:
    """The regression test for the defect, on the real route.

    ``_pipeline_clients`` is what the worker's tick context is built from, and
    its ``writer`` factory is what the orchestrator calls when a Run reaches the
    draft stage. Nothing is stubbed between here and ``WriterSettings``.
    """
    client = production_writer(worker_args(tmp_path))

    assert client.provider == "anthropic"
    assert client.model == SELECTION_ID
    assert ANTHROPIC_CREDENTIAL in credential_store_holding_the_key.reads


def test_the_production_route_builds_a_finalizer_with_no_environment_key(
    tmp_path: Path, credential_store_holding_the_key: FakeKeyringModule
) -> None:
    """Offline proof for the finalizer; it is never called live in this round."""
    client = production_finalizer(worker_args(tmp_path))

    assert client.provider == "anthropic"
    assert client.model == SELECTION_ID


def test_the_production_route_fails_explicitly_when_the_store_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing means missing: an explicit configuration error, not a fallback."""
    module = FakeKeyringModule({})
    monkeypatch.setattr(cli, "_credential_store", lambda: WindowsCredentialSecretProvider(module))
    monkeypatch.setattr(cli, "inspect_backend", lambda *a, **k: inspect_backend(module))
    with pytest.raises(WriterConfigurationError) as caught:
        production_writer(worker_args(tmp_path))

    assert caught.value.details["setting"] == "ANTHROPIC_API_KEY"


def test_the_production_route_still_honours_a_fake_writer(tmp_path: Path) -> None:
    """``--fake-ai`` must short-circuit before any credential is looked for."""
    args = worker_args(tmp_path)
    args.fake_ai = True

    client = production_writer(args)

    assert client.provider == "fake"


# --------------------------------------------------------------------------
# what must not have moved
# --------------------------------------------------------------------------


def test_the_legacy_path_without_a_selection_is_unchanged(
    tmp_path: Path, credential_store_holding_the_key: FakeKeyringModule
) -> None:
    """A Run predating preferences keeps its documented resolution.

    It already passed the composite provider - that is not new here - so the
    assertion is that this round did not alter it, and that ``None`` still
    means "resolve the model the way it always did".
    """
    factory = cli._pipeline_clients(worker_args(tmp_path)).writer
    assert factory is not None

    client = factory(None)

    assert client.provider == "anthropic"
    assert client.model == "claude-opus-5"


def test_the_reviewer_takes_no_selection() -> None:
    """Reviewer independence, asserted by signature rather than intention."""
    signature = inspect_module.signature(cli._reviewer_client)

    assert "selection" not in signature.parameters


def test_generation_still_cannot_build_a_reviewer() -> None:
    """The absence is the guarantee; a later round must not add one."""
    assert not hasattr(generation, "build_reviewer_client")
    assert "reviewer" not in " ".join(generation.__all__)


def test_a_claude_selection_never_looks_for_a_deepseek_key() -> None:
    """Lazy per vendor, still."""
    provider = anthropic_only()

    build_writer_client(Provider.CLAUDE, SELECTION_ID, secrets=provider)

    assert SecretName.DEEPSEEK_API_KEY not in provider.reads


def test_a_deepseek_selection_never_looks_for_an_anthropic_key() -> None:
    provider = FakeSecretProvider({SecretName.DEEPSEEK_API_KEY: "ds-offline-fake"})

    build_writer_client(Provider.DEEPSEEK, "deepseek-v4-pro", secrets=provider)

    assert SecretName.ANTHROPIC_API_KEY not in provider.reads


@pytest.mark.parametrize(
    ("selection_id", "api_model"),
    [
        ("deepseek-v4-pro", "deepseek-v4-pro"),
        ("deepseek-v4-flash", "deepseek-v4-flash"),
        ("deepseek-chat", "deepseek-v4-flash"),
        ("deepseek-reasoner", "deepseek-v4-flash"),
    ],
)
def test_the_deepseek_catalog_mapping_is_untouched(selection_id: str, api_model: str) -> None:
    """Regression only. This round changed where a key comes from, nothing else."""
    provider = FakeSecretProvider({SecretName.DEEPSEEK_API_KEY: "ds-offline-fake"})

    writer = build_writer_client(Provider.DEEPSEEK, selection_id, secrets=provider)
    finalizer = build_finalizer_client(Provider.DEEPSEEK, selection_id, secrets=provider)

    assert (writer.provider, writer.model) == ("deepseek", api_model)
    assert (finalizer.provider, finalizer.model) == ("deepseek", api_model)


def test_deepseek_defaults_to_the_environment_exactly_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supplying no provider must behave identically to the previous round."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-offline-fake")

    client = build_writer_client(Provider.DEEPSEEK, "deepseek-v4-pro")

    assert client.model == "deepseek-v4-pro"


def test_claude_defaults_to_the_environment_exactly_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new parameters are additive: omitting them changes nothing."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", ENV_KEY)

    client = build_writer_client(Provider.CLAUDE, SELECTION_ID)

    assert client.model == SELECTION_ID


# --------------------------------------------------------------------------
# the credential must not escape
# --------------------------------------------------------------------------


def test_the_key_is_absent_from_the_settings_repr() -> None:
    settings = WriterSettings.from_env({}, model_override=SELECTION_ID, secrets=anthropic_only())

    assert STORED_KEY not in repr(settings)
    assert STORED_KEY not in str(settings)
    assert "redacted" in repr(settings)


def test_the_key_is_absent_from_the_failure() -> None:
    """A configuration error names the setting and never its value."""
    with pytest.raises(WriterConfigurationError) as caught:
        build_writer_client(Provider.CLAUDE, SELECTION_ID, secrets=empty_provider())

    rendered = f"{caught.value} {caught.value.details}"
    assert STORED_KEY not in rendered
    assert ENV_KEY not in rendered


def test_building_a_client_logs_no_credential(caplog: pytest.LogCaptureFixture) -> None:
    """The seam logs provider and selection. Never more than that."""
    caplog.set_level("DEBUG")

    build_writer_client(Provider.CLAUDE, SELECTION_ID, secrets=anthropic_only())

    assert STORED_KEY not in caplog.text


def test_building_a_client_mutates_no_environment(
    tmp_path: Path, credential_store_holding_the_key: FakeKeyringModule
) -> None:
    """A resolved credential is never promoted into the process environment.

    The one-way rule from :mod:`goldpipeline.adapters.secrets`: a key found in a
    store stays in that store. Copying it into ``os.environ`` would leak it to
    every subprocess the worker ever spawns.
    """
    before = dict(os.environ)

    production_writer(worker_args(tmp_path))

    assert dict(os.environ) == before
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_the_selection_snapshot_carries_no_credential() -> None:
    """Nothing secret reaches an artifact, because the snapshot has no field for it."""
    payload: Any = claude_selection().model_dump()

    assert "api_key" not in payload
    assert STORED_KEY not in str(payload)
