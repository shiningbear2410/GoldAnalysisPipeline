"""Operator preferences: catalog, defaults, storage, mutation and the safe view.

Every test is offline and writes only into ``tmp_path``. No provider is called,
no key is read, no Credential Manager is touched, and nothing here ever writes
into the real ``automation/`` directory.

Two properties carry most of the weight:

* **a damaged file is never silently replaced** - an operator's stored choice is
  either read back exactly or reported as damaged, and never quietly swapped for
  a default that looks like a decision somebody made;
* **nothing free-text reaches the file** - a future callback payload can select
  among declared choices and cannot become a key, a path, or a model the catalog
  does not offer.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from goldpipeline.domain.errors import PreferencesUnavailableError
from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.news import MAX_LOOKBACK, MIN_LOOKBACK
from goldpipeline.schemas.preferences import (
    CATALOG,
    DEFAULT_PREFERENCES,
    DEFAULT_SELECTION_ID,
    PREFERENCES_FILENAME,
    PREFERENCES_SCHEMA_VERSION,
    PreferencesHealth,
    PreferencesSource,
    Provider,
    RuntimeReadiness,
    UserPreferences,
    provider_spec,
    resolve_model,
)
from goldpipeline.schemas.secrets import SecretName
from goldpipeline.services.article_routing import SPECS
from goldpipeline.services.preferences import PreferencesStore

CLAUDE_MODELS = ("claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5")
DEEPSEEK_MODELS = (
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "deepseek-chat",
    "deepseek-reasoner",
)


@pytest.fixture
def store(tmp_path: Path) -> PreferencesStore:
    return PreferencesStore(tmp_path / "automation")


def written(store: PreferencesStore) -> dict[str, object]:
    return json.loads(store.path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# defaults
# --------------------------------------------------------------------------


def test_a_missing_file_resolves_to_defaults(store: PreferencesStore) -> None:
    result = store.read()
    assert result.source is PreferencesSource.DEFAULT
    assert result.health is PreferencesHealth.OK
    assert result.preferences == DEFAULT_PREFERENCES
    assert not store.path.exists(), "reading must not create the file"


def test_the_default_window_is_24h() -> None:
    assert DEFAULT_PREFERENCES.news_lookback == timedelta(hours=24)


def test_the_default_article_type_is_analysis() -> None:
    assert DEFAULT_PREFERENCES.article_type is ArticleType.ANALYSIS


def test_the_default_model_is_what_production_already_uses() -> None:
    """The pin that stops a preferences schema from changing production.

    Both the writer and the finalizer resolve their model from
    ``config.DEFAULT_MODEL``. If someone changes one of these two constants
    without the other, an operator who never expressed a preference would
    silently start generating with a different model.
    """
    from goldpipeline.config import DEFAULT_MODEL

    assert DEFAULT_SELECTION_ID == DEFAULT_MODEL
    assert DEFAULT_PREFERENCES.provider is Provider.CLAUDE
    assert DEFAULT_PREFERENCES.selection_id == DEFAULT_MODEL


def test_defaults_are_a_valid_selection() -> None:
    resolve_model(DEFAULT_PREFERENCES.provider, DEFAULT_PREFERENCES.selection_id)


# --------------------------------------------------------------------------
# the catalog
# --------------------------------------------------------------------------


def test_every_provider_appears_once() -> None:
    listed = [spec.provider for spec in CATALOG]
    assert listed == sorted(set(listed), key=listed.index)
    assert set(listed) == set(Provider)


def test_the_claude_models_are_the_agreed_three() -> None:
    assert tuple(m.selection_id for m in provider_spec(Provider.CLAUDE).models) == CLAUDE_MODELS


def test_the_deepseek_models_are_the_agreed_four() -> None:
    assert tuple(m.selection_id for m in provider_spec(Provider.DEEPSEEK).models) == DEEPSEEK_MODELS


@pytest.mark.parametrize(
    ("selection_id", "label"),
    [
        ("claude-haiku-4-5", "Haiku 4.5"),
        ("claude-sonnet-5", "Sonnet 5"),
        ("claude-opus-5", "Opus 5"),
        ("deepseek-v4-pro", "DeepSeek-V4 Pro"),
        ("deepseek-v4-flash", "DeepSeek-V4 Flash"),
        ("deepseek-chat", "DeepSeek Chat"),
        ("deepseek-reasoner", "DeepSeek Reasoner"),
    ],
)
def test_display_labels_are_stable(selection_id: str, label: str) -> None:
    """A label is what a person clicked. Changing one silently is a UI bug."""
    found = next(m for spec in CATALOG for m in spec.models if m.selection_id == selection_id)
    assert found.label == label


def test_catalog_iteration_is_deterministic() -> None:
    once = [(spec.provider, tuple(m.selection_id for m in spec.models)) for spec in CATALOG]
    twice = [(spec.provider, tuple(m.selection_id for m in spec.models)) for spec in CATALOG]
    assert once == twice


def test_provider_labels_are_present() -> None:
    assert [spec.label for spec in CATALOG] == ["Claude API", "DeepSeek API"]


@pytest.mark.parametrize("selection_id", DEEPSEEK_MODELS)
def test_claude_does_not_offer_deepseek_models(selection_id: str) -> None:
    with pytest.raises(ValueError, match="does not offer"):
        resolve_model(Provider.CLAUDE, selection_id)


@pytest.mark.parametrize("selection_id", CLAUDE_MODELS)
def test_deepseek_does_not_offer_claude_models(selection_id: str) -> None:
    with pytest.raises(ValueError, match="does not offer"):
        resolve_model(Provider.DEEPSEEK, selection_id)


@pytest.mark.parametrize(
    "selection_id",
    ["", "gpt-4", "claude-opus-4", "claude-opus-5 ", "CLAUDE-OPUS-5", "../../etc/passwd"],
)
def test_an_unknown_model_is_refused(selection_id: str) -> None:
    with pytest.raises(ValueError):
        resolve_model(Provider.CLAUDE, selection_id)


@pytest.mark.parametrize("provider", ["claude", "CLAUDE_API", "openai", "", "anthropic"])
def test_an_unknown_provider_is_refused(provider: str) -> None:
    with pytest.raises(ValidationError):
        UserPreferences(provider=provider, selection_id="claude-opus-5")  # type: ignore[arg-type]


def test_provider_is_never_inferred_from_a_model_prefix() -> None:
    """``claude-`` is a naming convention, not authority."""
    with pytest.raises(ValidationError):
        UserPreferences(provider=Provider.DEEPSEEK, selection_id="claude-opus-5")


# --------------------------------------------------------------------------
# runtime readiness
# --------------------------------------------------------------------------


def test_both_providers_are_implemented() -> None:
    """DeepSeek stopped being a placeholder this round."""
    assert provider_spec(Provider.CLAUDE).implemented is True
    assert provider_spec(Provider.DEEPSEEK).implemented is True
    assert provider_spec(Provider.DEEPSEEK).requires, "say what it still needs"


def test_implemented_is_not_available_until_a_credential_is_found() -> None:
    """Code existing is not the same as the thing working.

    ``IMPLEMENTED`` is the honest answer when nobody looked at the credential
    store, and only a real probe may upgrade it. A status that showed green for
    a provider whose key is absent would promise a call nobody can make.
    """
    spec = provider_spec(Provider.DEEPSEEK)
    assert spec.readiness(secret_present=None) is RuntimeReadiness.IMPLEMENTED
    assert spec.readiness(secret_present=False) is RuntimeReadiness.IMPLEMENTED_NOT_CONFIGURED
    assert spec.readiness(secret_present=True) is RuntimeReadiness.AVAILABLE


def test_an_unimplemented_provider_stays_unimplemented_whatever_the_key() -> None:
    from dataclasses import replace

    spec = replace(provider_spec(Provider.DEEPSEEK), implemented=False)
    for present in (None, True, False):
        assert spec.readiness(secret_present=present) is RuntimeReadiness.NOT_IMPLEMENTED


def test_status_does_not_claim_configured_without_a_probe(store: PreferencesStore) -> None:
    """This round reads no credential store, so nothing may look green."""
    store.set_provider_model(Provider.DEEPSEEK, "deepseek-reasoner")
    status = store.status()

    assert status.provider is Provider.DEEPSEEK
    assert status.model_label == "DeepSeek Reasoner"
    assert status.provider_runtime is RuntimeReadiness.IMPLEMENTED
    assert status.generation_ready is False


def test_a_probe_can_report_a_configured_provider(store: PreferencesStore) -> None:
    """What a future bot does, with a fake store rather than the real one."""
    store.set_provider_model(Provider.DEEPSEEK, "deepseek-v4-pro")
    stored: set[SecretName] = {SecretName.DEEPSEEK_API_KEY}

    status = store.status(secret_present=lambda name: name in stored)
    assert status.provider_runtime is RuntimeReadiness.AVAILABLE
    assert status.generation_ready is True

    status = store.status(secret_present=lambda name: name in set())
    assert status.provider_runtime is RuntimeReadiness.IMPLEMENTED_NOT_CONFIGURED
    assert status.generation_ready is False


def test_the_probe_is_asked_only_about_the_selected_provider(
    store: PreferencesStore,
) -> None:
    """Selecting Claude must never cause a DeepSeek key to be looked for."""
    asked: list[SecretName] = []
    store.set_provider_model(Provider.CLAUDE, "claude-opus-5")
    store.status(secret_present=lambda name: bool(asked.append(name)) or True)
    assert asked == [SecretName.ANTHROPIC_API_KEY]


# --------------------------------------------------------------------------
# article type
# --------------------------------------------------------------------------


@pytest.mark.parametrize("article_type", list(ArticleType))
def test_any_article_type_may_be_stored(article_type: ArticleType, store: PreferencesStore) -> None:
    stored = store.set_article_type(article_type)
    assert stored.article_type is article_type
    assert store.read().preferences is not None
    assert store.read().preferences.article_type is article_type  # type: ignore[union-attr]


@pytest.mark.parametrize("article_type", [ArticleType.TRADE_PLAN, ArticleType.NEWS_DIGEST])
def test_storing_an_unfinished_type_does_not_make_it_ready(
    article_type: ArticleType, store: PreferencesStore
) -> None:
    """A preference records a wish. Readiness is a fact about the code."""
    store.set_article_type(article_type)
    status = store.status()

    assert status.article_type is article_type
    assert status.article_type_ready is False
    assert status.article_type_requires
    assert SPECS[article_type].ready is False
    assert status.generation_ready is False


def test_readiness_is_read_from_routing_not_stored(store: PreferencesStore) -> None:
    """No copy of readiness exists in the document, so none can go stale."""
    store.set_article_type(ArticleType.TRADE_PLAN)
    document = written(store)
    assert "ready" not in json.dumps(document)
    assert set(document) == {
        "schema_version",
        "provider",
        "selection_id",
        "article_type",
        "news_lookback_seconds",
    }


def test_analysis_remains_ready(store: PreferencesStore) -> None:
    store.set_article_type(ArticleType.ANALYSIS)
    assert store.status().article_type_ready is True


# --------------------------------------------------------------------------
# news lookback
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hours", [6, 12, 24, 48, 72, 168])
def test_the_offered_presets_are_accepted(hours: int, store: PreferencesStore) -> None:
    stored = store.set_news_lookback(timedelta(hours=hours))
    assert stored.news_lookback == timedelta(hours=hours)


@pytest.mark.parametrize("minutes", [61, 90, 200, 1000])
def test_a_bounded_custom_duration_is_accepted(minutes: int, store: PreferencesStore) -> None:
    assert store.set_news_lookback(timedelta(minutes=minutes)).news_lookback_seconds == minutes * 60


@pytest.mark.parametrize(
    "lookback",
    [timedelta(0), timedelta(minutes=59), timedelta(days=8), timedelta(days=30)],
)
def test_a_window_outside_the_bounds_is_refused(
    lookback: timedelta, store: PreferencesStore
) -> None:
    with pytest.raises(ValueError):
        store.set_news_lookback(lookback)
    assert not store.path.exists(), "a refused change must write nothing"


def test_the_bounds_are_the_collector_s_own(store: PreferencesStore) -> None:
    """One authority, so a preference cannot be stored that the collector clamps."""
    from goldpipeline.schemas.producer import MAX_NEWS_LOOKBACK, MIN_NEWS_LOOKBACK
    from goldpipeline.services.news_collector import MAX_LOOKBACK as COLLECTOR_MAX
    from goldpipeline.services.news_collector import MIN_LOOKBACK as COLLECTOR_MIN

    assert (MIN_NEWS_LOOKBACK, MAX_NEWS_LOOKBACK) == (MIN_LOOKBACK, MAX_LOOKBACK)
    assert (COLLECTOR_MIN, COLLECTOR_MAX) == (MIN_LOOKBACK, MAX_LOOKBACK)

    store.set_news_lookback(MIN_LOOKBACK)
    store.set_news_lookback(MAX_LOOKBACK)


def test_a_stored_window_is_accepted_by_a_producer_request(store: PreferencesStore) -> None:
    """The eventual handoff, proven now rather than assumed later."""
    from goldpipeline.schemas.common import utc_now
    from goldpipeline.schemas.producer import ProducerRequest

    stored = store.set_news_lookback(timedelta(hours=72))
    request = ProducerRequest(
        request_id="pref-000001",
        requested_at=utc_now(),
        news_lookback_seconds=stored.news_lookback_seconds,
    )
    assert request.news_lookback == timedelta(hours=72)


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------


def test_a_valid_file_is_read_back(store: PreferencesStore) -> None:
    store.set_provider_model(Provider.CLAUDE, "claude-haiku-4-5")
    result = store.read()

    assert result.source is PreferencesSource.FILE
    assert result.health is PreferencesHealth.OK
    assert result.preferences is not None
    assert result.preferences.selection_id == "claude-haiku-4-5"


@pytest.mark.parametrize(
    "content",
    ["", "{", "not json at all", "[]", '"a string"', "null", "123"],
)
def test_malformed_content_is_reported_not_repaired(content: str, store: PreferencesStore) -> None:
    store.root.mkdir(parents=True)
    store.path.write_text(content, encoding="utf-8")
    result = store.read()

    assert result.preferences is None
    assert result.health is PreferencesHealth.UNREADABLE
    assert result.source is PreferencesSource.FILE
    assert result.detail
    assert store.path.read_text(encoding="utf-8") == content, "a read must not rewrite the file"


def test_an_unsupported_version_is_reported(store: PreferencesStore) -> None:
    store.root.mkdir(parents=True)
    store.path.write_text(json.dumps({"schema_version": "2", "provider": "CLAUDE"}), "utf-8")
    result = store.read()

    assert result.preferences is None
    assert result.health is PreferencesHealth.UNSUPPORTED_VERSION
    assert "2" in (result.detail or "")


@pytest.mark.parametrize(
    "document",
    [
        {"schema_version": "1", "provider": "DEEPSEEK", "selection_id": "claude-opus-5"},
        {"schema_version": "1", "provider": "OPENAI", "selection_id": "gpt-4"},
        {"schema_version": "1", "selection_id": "not-a-model"},
        {"schema_version": "1", "news_lookback_seconds": 10},
        {"schema_version": "1", "news_lookback_seconds": 9_999_999},
        {"schema_version": "1", "article_type": "SOMETHING_ELSE"},
    ],
)
def test_an_invalid_selection_fails_closed(
    document: dict[str, object], store: PreferencesStore
) -> None:
    store.root.mkdir(parents=True)
    store.path.write_text(json.dumps(document), encoding="utf-8")
    result = store.read()

    assert result.preferences is None
    assert result.health is PreferencesHealth.INVALID
    assert result.detail


def test_extra_fields_are_refused(store: PreferencesStore) -> None:
    """The whitelist is the point: a new key cannot arrive by being written."""
    store.root.mkdir(parents=True)
    store.path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "provider": "CLAUDE",
                "selection_id": "claude-opus-5",
                "reviewer_model": "claude-opus-5",
            }
        ),
        encoding="utf-8",
    )
    assert store.read().health is PreferencesHealth.INVALID


def test_the_serialization_is_canonical_and_stable(store: PreferencesStore) -> None:
    store.set_provider_model(Provider.CLAUDE, "claude-sonnet-5")
    first = store.path.read_bytes()
    store.write(store.read().usable)
    assert store.path.read_bytes() == first


def test_the_write_creates_the_directory(tmp_path: Path) -> None:
    store = PreferencesStore(tmp_path / "nested" / "automation")
    store.set_article_type(ArticleType.ANALYSIS)
    assert store.path.is_file()


def test_a_failed_write_leaves_the_original_intact(
    store: PreferencesStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The atomic-replace promise, exercised rather than asserted in prose."""
    store.set_provider_model(Provider.CLAUDE, "claude-sonnet-5")
    original = store.path.read_bytes()

    import goldpipeline.services.preferences as module

    def explode(*_: object, **__: object) -> str:
        raise OSError("disk full")

    monkeypatch.setattr(module, "atomic_write_bytes", explode)
    with pytest.raises(OSError, match="disk full"):
        store.set_provider_model(Provider.CLAUDE, "claude-haiku-4-5")

    assert store.path.read_bytes() == original
    assert store.read().preferences.selection_id == "claude-sonnet-5"  # type: ignore[union-attr]


def test_no_temporary_files_are_left_behind(store: PreferencesStore) -> None:
    store.set_article_type(ArticleType.ANALYSIS)
    store.set_news_lookback(timedelta(hours=12))
    assert [p.name for p in store.root.iterdir()] == [PREFERENCES_FILENAME]


def test_the_last_writer_wins_on_the_whole_object(store: PreferencesStore) -> None:
    """The documented concurrency contract: never corrupt, possibly superseded."""
    store.write(UserPreferences(provider=Provider.CLAUDE, selection_id="claude-haiku-4-5"))
    store.write(UserPreferences(provider=Provider.CLAUDE, selection_id="claude-opus-5"))
    result = store.read()

    assert result.health is PreferencesHealth.OK
    assert result.preferences is not None
    assert result.preferences.selection_id == "claude-opus-5"


# --------------------------------------------------------------------------
# mutation
# --------------------------------------------------------------------------


def test_provider_and_model_change_together(store: PreferencesStore) -> None:
    updated = store.set_provider_model(Provider.DEEPSEEK, "deepseek-v4-flash")
    assert (updated.provider, updated.selection_id) == (Provider.DEEPSEEK, "deepseek-v4-flash")


def test_an_invalid_pairing_writes_nothing(store: PreferencesStore) -> None:
    with pytest.raises(ValueError):
        store.set_provider_model(Provider.DEEPSEEK, "claude-opus-5")
    assert not store.path.exists()


def test_one_change_preserves_the_others(store: PreferencesStore) -> None:
    store.set_provider_model(Provider.CLAUDE, "claude-haiku-4-5")
    store.set_news_lookback(timedelta(hours=48))
    updated = store.set_article_type(ArticleType.NEWS_DIGEST)

    assert updated.provider is Provider.CLAUDE
    assert updated.selection_id == "claude-haiku-4-5"
    assert updated.news_lookback == timedelta(hours=48)
    assert updated.article_type is ArticleType.NEWS_DIGEST


def test_a_corrupt_file_is_not_repaired_by_a_mutation(store: PreferencesStore) -> None:
    """The failure mode this module exists to avoid, tested directly."""
    store.root.mkdir(parents=True)
    store.path.write_text("{ truncated", encoding="utf-8")

    with pytest.raises(PreferencesUnavailableError) as caught:
        store.set_article_type(ArticleType.ANALYSIS)

    assert caught.value.code == "PREFERENCES_UNAVAILABLE"
    assert store.path.read_text(encoding="utf-8") == "{ truncated"


def test_a_corrupt_file_still_refuses_after_several_attempts(store: PreferencesStore) -> None:
    store.root.mkdir(parents=True)
    store.path.write_text("nonsense", encoding="utf-8")
    for _ in range(3):
        with pytest.raises(PreferencesUnavailableError):
            store.set_news_lookback(timedelta(hours=6))
    assert store.path.read_text(encoding="utf-8") == "nonsense"


def test_the_store_exposes_no_keyed_setter() -> None:
    """A callback payload must not be able to name a field."""
    exposed = {name for name in dir(PreferencesStore) if not name.startswith("_")}
    assert exposed == {
        "path",
        "read",
        "set_article_type",
        "set_news_lookback",
        "set_provider_model",
        "status",
        "write",
    }


# --------------------------------------------------------------------------
# the status read model
# --------------------------------------------------------------------------


def test_status_shows_labels_a_person_can_read(store: PreferencesStore) -> None:
    store.set_provider_model(Provider.CLAUDE, "claude-sonnet-5")
    status = store.status()

    assert status.provider_label == "Claude API"
    assert status.model_label == "Sonnet 5"
    assert status.selection_id == "claude-sonnet-5"
    assert status.news_lookback_seconds == 86_400


def test_status_distinguishes_default_from_stored(store: PreferencesStore) -> None:
    assert store.status().source is PreferencesSource.DEFAULT
    store.set_article_type(ArticleType.ANALYSIS)
    assert store.status().source is PreferencesSource.FILE


def test_status_reports_a_damaged_file_without_inventing_selections(
    store: PreferencesStore,
) -> None:
    store.root.mkdir(parents=True)
    store.path.write_text("{ broken", encoding="utf-8")
    status = store.status()

    assert status.health is PreferencesHealth.UNREADABLE
    assert status.usable is False
    assert status.provider is None
    assert status.model_label is None
    assert status.article_type is None
    assert status.detail


def test_status_carries_nothing_secret(store: PreferencesStore) -> None:
    store.set_provider_model(Provider.CLAUDE, "claude-opus-5")
    dumped = json.loads(store.status().model_dump_json())

    for forbidden in (
        "api_key",
        "anthropic_api_key",
        "deepseek_api_key",
        "token",
        "chat_id",
        "target",
        "publish",
        "path",
        "config",
        "reviewer",
        "secret",
    ):
        assert not any(forbidden in key for key in dumped), forbidden


def test_status_has_no_reviewer_field() -> None:
    """The review must be able to disagree with the writer.

    A single setting that moved both would remove the disagreement, quietly, and
    the whole reason a separate reviewer exists is that nobody notices when it
    stops being independent.
    """
    from goldpipeline.schemas.preferences import PreferencesStatus

    for model in (UserPreferences, PreferencesStatus):
        assert not any("review" in name for name in model.model_fields)


# --------------------------------------------------------------------------
# security
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hostile",
    [
        "runs_dir",
        "../../config.json",
        "ANTHROPIC_API_KEY",
        "TELEGRAM_TARGET_CHAT_ID",
        "auto_publish",
        "prompt_id",
    ],
)
def test_an_arbitrary_string_cannot_become_a_key(hostile: str, store: PreferencesStore) -> None:
    store.root.mkdir(parents=True)
    store.path.write_text(
        json.dumps({"schema_version": "1", hostile: "anything"}), encoding="utf-8"
    )
    assert store.read().health is PreferencesHealth.INVALID


@pytest.mark.parametrize(
    "hostile",
    ["../../../etc/passwd", "C:\\Windows\\System32", "/dev/null", "file:///etc/shadow"],
)
def test_a_model_id_cannot_name_a_path(hostile: str) -> None:
    with pytest.raises(ValueError):
        UserPreferences(provider=Provider.CLAUDE, selection_id=hostile)


def test_preferences_are_not_production_config_keys() -> None:
    """Adding them to ConfigKey would make every one of them mandatory.

    ``REQUIRED_PRODUCTION_KEYS = frozenset(ConfigKey)``, so a preference key
    there would fail a scheduled worker closed until an operator who never
    wanted an opinion supplied one.
    """
    from goldpipeline.schemas.runtime_config import REQUIRED_PRODUCTION_KEYS, ConfigKey

    names = {key.name for key in ConfigKey} | {key.value for key in ConfigKey}
    for field in UserPreferences.model_fields:
        assert field.upper() not in names
    assert set(REQUIRED_PRODUCTION_KEYS) == set(ConfigKey)


def test_the_preferences_file_lives_beside_the_other_mutable_state(
    store: PreferencesStore,
) -> None:
    from goldpipeline.services.automation_state import STATE_FILENAME

    store.set_article_type(ArticleType.ANALYSIS)
    assert store.path.name == PREFERENCES_FILENAME
    assert store.path.parent == store.root
    assert store.path.name != STATE_FILENAME


def test_the_document_records_its_schema_version(store: PreferencesStore) -> None:
    store.set_article_type(ArticleType.ANALYSIS)
    assert written(store)["schema_version"] == PREFERENCES_SCHEMA_VERSION


# --------------------------------------------------------------------------
# production is untouched
# --------------------------------------------------------------------------


def test_no_live_preferences_file_exists() -> None:
    """This round adds the machinery and creates no state.

    Nothing in production has expressed a preference, so production behaviour is
    whatever it was: the writer and finalizer still read their model from
    configuration, and no code path consults this store yet.
    """
    assert not (Path("automation") / PREFERENCES_FILENAME).exists()


def test_nothing_reads_preferences_in_the_pipeline_yet() -> None:
    """The mapping is documented and deliberately not wired.

    Preferences will drive the writer and the finalizer, and the producer's
    article type and window. None of that happens this round, and a grep is the
    cheapest way to notice the day somebody wires it without meaning to.
    """
    import ast

    for name in (
        "writer.py",
        "finalizer.py",
        "producer.py",
        "reviewer.py",
        "orchestrator.py",
        "automation.py",
    ):
        source = Path("src/goldpipeline/services") / name
        modules = {
            node.module or ""
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
            if isinstance(node, ast.ImportFrom)
        }
        assert "goldpipeline.services.preferences" not in modules, name
        assert "goldpipeline.schemas.preferences" not in modules, name
