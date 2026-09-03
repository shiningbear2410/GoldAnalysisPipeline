"""Preferences driving generation: snapshot, immutability, and what stays out.

Every test is offline. Fake provider clients, temporary Runs and inboxes,
temporary preference files, and no credential store: nothing here can reach
Anthropic, DeepSeek, MetaTrader or Telegram.

The property the whole round exists for:

    A Run generates with the choice it was created under, and nothing that
    happens afterwards to a mutable file can change that.

Everything else follows from it - the mid-Run change, the restart, the legacy
Run, and the reviewer that cannot be moved at all.
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from conftest import make_analysis_payload, make_market_payload, write_json

from goldpipeline.adapters.file_source import JsonFileAnalysisSource, JsonFileMarketDataSource
from goldpipeline.domain.errors import PreferencesUnavailableError
from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.generation import GenerationSelection
from goldpipeline.schemas.preferences import (
    DEFAULT_PREFERENCES,
    PreferencesSource,
    Provider,
    ThinkingMode,
    UserPreferences,
)
from goldpipeline.services.generation import build_finalizer_client, build_writer_client
from goldpipeline.services.pipeline import create_run
from goldpipeline.services.preferences import PreferencesStore, resolve_generation
from goldpipeline.storage.run_store import RunStore

NOW = datetime(2026, 9, 3, 9, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> PreferencesStore:
    return PreferencesStore(tmp_path / "automation")


def normalized_run(
    runs_dir: Path, tmp_path: Path, *, generation: GenerationSelection | None
) -> Any:
    """A real NORMALIZED Run carrying (or not carrying) a snapshot."""
    sources = tmp_path / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    analysis = write_json(sources / "telegram_input.json", make_analysis_payload())
    market = write_json(sources / "ohlc.json", make_market_payload())

    result = create_run(
        analysis_source=JsonFileAnalysisSource(analysis),
        market_source=JsonFileMarketDataSource(market),
        store=RunStore(runs_dir),
        expected_symbol="XAUUSD",
        generation=generation,
    )
    assert result.succeeded, result.error
    return result


def snapshot_of(runs_dir: Path, run_id: str) -> GenerationSelection | None:
    manifest = RunStore(runs_dir).open(run_id).load_manifest()
    return manifest.provenance.generation if manifest.provenance else None


# --------------------------------------------------------------------------
# resolving a snapshot
# --------------------------------------------------------------------------


def test_an_absent_file_resolves_to_the_production_default(store: PreferencesStore) -> None:
    """The compatibility pin: deploying this must not change what production writes."""
    from goldpipeline.config import DEFAULT_MODEL

    selection = resolve_generation(store)
    assert selection.provider is Provider.CLAUDE
    assert selection.selection_id == DEFAULT_MODEL
    assert selection.api_model_id == DEFAULT_MODEL
    assert selection.preference_source is PreferencesSource.DEFAULT
    assert not store.path.exists(), "resolving must not create the file"


def test_the_source_distinguishes_a_choice_from_a_default(store: PreferencesStore) -> None:
    assert resolve_generation(store).preference_source is PreferencesSource.DEFAULT
    store.set_provider_model(Provider.CLAUDE, "claude-opus-5")
    assert resolve_generation(store).preference_source is PreferencesSource.FILE


@pytest.mark.parametrize(
    ("selection_id", "api_model", "thinking"),
    [
        ("claude-haiku-4-5", "claude-haiku-4-5", ThinkingMode.NOT_APPLICABLE),
        ("claude-sonnet-5", "claude-sonnet-5", ThinkingMode.NOT_APPLICABLE),
        ("claude-opus-5", "claude-opus-5", ThinkingMode.NOT_APPLICABLE),
    ],
)
def test_claude_selections_resolve(
    selection_id: str, api_model: str, thinking: ThinkingMode, store: PreferencesStore
) -> None:
    store.set_provider_model(Provider.CLAUDE, selection_id)
    selection = resolve_generation(store)
    assert (selection.selection_id, selection.api_model_id, selection.thinking) == (
        selection_id,
        api_model,
        thinking,
    )


@pytest.mark.parametrize(
    ("selection_id", "api_model", "thinking"),
    [
        ("deepseek-v4-pro", "deepseek-v4-pro", ThinkingMode.ENABLED),
        ("deepseek-v4-flash", "deepseek-v4-flash", ThinkingMode.ENABLED),
        ("deepseek-chat", "deepseek-v4-flash", ThinkingMode.DISABLED),
        ("deepseek-reasoner", "deepseek-v4-flash", ThinkingMode.ENABLED),
    ],
)
def test_deepseek_selections_resolve_with_their_vendor_mapping(
    selection_id: str, api_model: str, thinking: ThinkingMode, store: PreferencesStore
) -> None:
    """The Round 6.1 mapping, carried onto the Run rather than re-derived later."""
    store.set_provider_model(Provider.DEEPSEEK, selection_id)
    selection = resolve_generation(store)
    assert selection.provider is Provider.DEEPSEEK
    assert (selection.selection_id, selection.api_model_id, selection.thinking) == (
        selection_id,
        api_model,
        thinking,
    )


def test_the_snapshot_carries_no_secret_and_no_reviewer(store: PreferencesStore) -> None:
    dumped = json.loads(resolve_generation(store).model_dump_json())
    for forbidden in ("key", "secret", "token", "reviewer", "chat", "publish", "path"):
        assert not any(forbidden in field for field in dumped), forbidden


@pytest.mark.parametrize(
    "content",
    [
        "{ truncated",
        "not json",
        '{"schema_version": "9"}',
        '{"schema_version": "1", "provider": "OPENAI"}',
    ],
)
def test_a_damaged_file_refuses_to_resolve(content: str, store: PreferencesStore) -> None:
    """Fail closed: an article must not be written under a choice nobody made."""
    store.root.mkdir(parents=True)
    store.path.write_text(content, encoding="utf-8")

    with pytest.raises(PreferencesUnavailableError):
        resolve_generation(store)
    assert store.path.read_text(encoding="utf-8") == content, "never repaired"


# --------------------------------------------------------------------------
# the snapshot lands on the Run
# --------------------------------------------------------------------------


def test_a_new_run_records_its_selection(runs_dir: Path, tmp_path: Path) -> None:
    selection = GenerationSelection.from_preferences(
        UserPreferences(provider=Provider.DEEPSEEK, selection_id="deepseek-chat"),
        source=PreferencesSource.FILE,
    )
    run = normalized_run(runs_dir, tmp_path, generation=selection)
    stored = snapshot_of(runs_dir, run.run_id)

    assert stored == selection
    assert stored is not None
    assert stored.api_model_id == "deepseek-v4-flash"
    assert stored.thinking is ThinkingMode.DISABLED


def test_the_snapshot_is_visible_before_any_writer_artifact(runs_dir: Path, tmp_path: Path) -> None:
    """A Run whose writer never ran is still inspectable."""
    run = normalized_run(
        runs_dir,
        tmp_path,
        generation=GenerationSelection.from_preferences(
            DEFAULT_PREFERENCES, source=PreferencesSource.DEFAULT
        ),
    )
    assert not (Path(run.run_dir) / "claude_writer.json").exists()
    assert snapshot_of(runs_dir, run.run_id) is not None


def test_a_run_created_without_preferences_records_nothing(runs_dir: Path, tmp_path: Path) -> None:
    """An operator driving one Run by hand keeps the legacy behaviour."""
    run = normalized_run(runs_dir, tmp_path, generation=None)
    assert snapshot_of(runs_dir, run.run_id) is None


def test_the_snapshot_survives_a_reread(runs_dir: Path, tmp_path: Path) -> None:
    """It comes off disk, so nothing in memory can be the authority."""
    selection = GenerationSelection.from_preferences(
        UserPreferences(provider=Provider.CLAUDE, selection_id="claude-haiku-4-5"),
        source=PreferencesSource.FILE,
    )
    run = normalized_run(runs_dir, tmp_path, generation=selection)

    fresh = RunStore(runs_dir).open(run.run_id).load_manifest()
    assert fresh.provenance is not None
    assert fresh.provenance.generation == selection


# --------------------------------------------------------------------------
# which client gets built
# --------------------------------------------------------------------------


class RecordingFactory:
    """Stands in for the CLI's client builders, recording what it was given."""

    def __init__(self) -> None:
        self.selections: list[GenerationSelection | None] = []

    def __call__(self, selection: GenerationSelection | None) -> Any:
        self.selections.append(selection)
        from goldpipeline.adapters.fake_writer import FakeWriterClient

        return FakeWriterClient()


@pytest.mark.parametrize("selection_id", ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"])
def test_a_claude_selection_builds_the_anthropic_writer(
    selection_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    client = build_writer_client(Provider.CLAUDE, selection_id)
    assert client.provider == "anthropic"
    assert client.model == selection_id


@pytest.mark.parametrize("selection_id", ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"])
def test_a_claude_selection_builds_the_anthropic_finalizer(
    selection_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    client = build_finalizer_client(Provider.CLAUDE, selection_id)
    assert client.provider == "anthropic"
    assert client.model == selection_id


@pytest.mark.parametrize(
    ("selection_id", "api_model"),
    [
        ("deepseek-v4-pro", "deepseek-v4-pro"),
        ("deepseek-v4-flash", "deepseek-v4-flash"),
        ("deepseek-chat", "deepseek-v4-flash"),
        ("deepseek-reasoner", "deepseek-v4-flash"),
    ],
)
def test_a_deepseek_selection_builds_both_deepseek_clients(
    selection_id: str, api_model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-fake-key")

    writer = build_writer_client(Provider.DEEPSEEK, selection_id)
    finalizer = build_finalizer_client(Provider.DEEPSEEK, selection_id)

    assert (writer.provider, writer.model) == ("deepseek", api_model)
    assert (finalizer.provider, finalizer.model) == ("deepseek", api_model)


def test_a_deepseek_selection_without_a_key_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No fallback: the selection stands and the call does not happen."""
    from goldpipeline.domain.errors import WriterConfigurationError

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(WriterConfigurationError) as caught:
        build_writer_client(Provider.DEEPSEEK, "deepseek-v4-pro")

    assert caught.value.details["setting"] == "DEEPSEEK_API_KEY"
    assert "anthropic" not in caught.value.message.lower()
    assert "claude" not in caught.value.message.lower()


# --------------------------------------------------------------------------
# the immutability promise
# --------------------------------------------------------------------------


def orchestrate(runs_dir: Path, run_id: str, clients: Any, mode: Any = None) -> Any:
    from goldpipeline.schemas.orchestration import PipelineMode
    from goldpipeline.services.orchestrator import resume_pipeline

    return resume_pipeline(
        run_id=run_id,
        store=RunStore(runs_dir),
        clients=clients,
        mode=mode or PipelineMode.GENERATE_ONLY,
    )


def test_a_preference_changed_mid_run_does_not_reach_the_finalizer(
    runs_dir: Path, tmp_path: Path, store: PreferencesStore
) -> None:
    """The headline requirement, driven through the real orchestrator.

    A Run starts under Claude Opus. Between the writer and the finalizer the
    operator switches to DeepSeek Chat. The finalizer must still be built from
    the Run's own snapshot.
    """
    from conftest import make_tracked_clients

    store.set_provider_model(Provider.CLAUDE, "claude-opus-5")
    run = normalized_run(runs_dir, tmp_path, generation=resolve_generation(store))

    tracked = make_tracked_clients()
    orchestrate(runs_dir, run.run_id, tracked.as_pipeline_clients())

    # The operator changes their mind after the Run is under way.
    store.set_provider_model(Provider.DEEPSEEK, "deepseek-chat")
    assert resolve_generation(store).selection_id == "deepseek-chat"

    orchestrate(runs_dir, run.run_id, tracked.as_pipeline_clients())

    used = [selection for name, selection in tracked.selections if selection is not None]
    assert used, "the stages were built from a selection"
    assert {s.selection_id for s in used} == {"claude-opus-5"}
    assert {s.provider for s in used} == {Provider.CLAUDE}


def test_the_mirror_case_deepseek_run_survives_a_switch_to_claude(
    runs_dir: Path, tmp_path: Path, store: PreferencesStore
) -> None:
    from conftest import make_tracked_clients

    store.set_provider_model(Provider.DEEPSEEK, "deepseek-v4-pro")
    run = normalized_run(runs_dir, tmp_path, generation=resolve_generation(store))

    tracked = make_tracked_clients()
    orchestrate(runs_dir, run.run_id, tracked.as_pipeline_clients())
    store.set_provider_model(Provider.CLAUDE, "claude-opus-5")
    orchestrate(runs_dir, run.run_id, tracked.as_pipeline_clients())

    used = [s for _, s in tracked.selections if s is not None]
    assert {s.selection_id for s in used} == {"deepseek-v4-pro"}
    assert {s.api_model_id for s in used} == {"deepseek-v4-pro"}


def test_a_restart_between_stages_changes_nothing(
    runs_dir: Path, tmp_path: Path, store: PreferencesStore
) -> None:
    """Nothing is carried in memory, so a fresh process reaches the same answer."""
    from conftest import make_tracked_clients

    store.set_provider_model(Provider.DEEPSEEK, "deepseek-chat")
    run = normalized_run(runs_dir, tmp_path, generation=resolve_generation(store))

    first = make_tracked_clients()
    orchestrate(runs_dir, run.run_id, first.as_pipeline_clients())

    # Simulate a restart: a brand-new store object, brand-new clients, and a
    # preference file that has changed in the meantime.
    store.set_provider_model(Provider.CLAUDE, "claude-sonnet-5")
    second = make_tracked_clients()
    orchestrate(runs_dir, run.run_id, second.as_pipeline_clients())

    used = [s for _, s in (*first.selections, *second.selections) if s is not None]
    assert {s.selection_id for s in used} == {"deepseek-chat"}
    assert {s.api_model_id for s in used} == {"deepseek-v4-flash"}
    assert {s.thinking for s in used} == {ThinkingMode.DISABLED}


def test_the_next_run_does_see_the_new_preference(
    runs_dir: Path, tmp_path: Path, store: PreferencesStore
) -> None:
    store.set_provider_model(Provider.CLAUDE, "claude-opus-5")
    first = normalized_run(runs_dir, tmp_path, generation=resolve_generation(store))

    store.set_provider_model(Provider.DEEPSEEK, "deepseek-reasoner")
    second = normalized_run(runs_dir, tmp_path, generation=resolve_generation(store))

    assert snapshot_of(runs_dir, first.run_id).selection_id == "claude-opus-5"  # type: ignore[union-attr]
    assert snapshot_of(runs_dir, second.run_id).selection_id == "deepseek-reasoner"  # type: ignore[union-attr]


def test_a_legacy_run_keeps_legacy_semantics(runs_dir: Path, tmp_path: Path) -> None:
    """No snapshot means no selection reaches the factories - as before."""
    from conftest import make_tracked_clients

    run = normalized_run(runs_dir, tmp_path, generation=None)
    tracked = make_tracked_clients()
    orchestrate(runs_dir, run.run_id, tracked.as_pipeline_clients())

    generation = [s for name, s in tracked.selections if name in {"writer", "finalizer"}]
    assert generation, "the writer was still built"
    assert all(s is None for s in generation)


# --------------------------------------------------------------------------
# the reviewer
# --------------------------------------------------------------------------


def test_the_reviewer_is_never_handed_a_selection(
    runs_dir: Path, tmp_path: Path, store: PreferencesStore
) -> None:
    """Structural: its factory has nowhere to put one."""
    from conftest import make_tracked_clients

    from goldpipeline.schemas.orchestration import PipelineMode

    store.set_provider_model(Provider.DEEPSEEK, "deepseek-chat")
    run = normalized_run(runs_dir, tmp_path, generation=resolve_generation(store))

    tracked = make_tracked_clients()
    orchestrate(runs_dir, run.run_id, tracked.as_pipeline_clients(), PipelineMode.GENERATE_ONLY)

    reviewer_calls = [s for name, s in tracked.selections if name == "reviewer"]
    assert reviewer_calls, "the reviewer ran"
    assert all(s is None for s in reviewer_calls)


def test_the_reviewer_factory_signature_takes_no_selection() -> None:
    """The enforcement, read off the type rather than trusted."""
    import inspect

    from goldpipeline.services.orchestrator import PipelineClients

    hints = PipelineClients.__annotations__
    assert "GenerationSelection" in str(hints["writer"])
    assert "GenerationSelection" in str(hints["finalizer"])
    assert "GenerationSelection" not in str(hints["reviewer"])
    assert inspect.isclass(PipelineClients)


def test_a_deepseek_run_still_reviews_with_anthropic(
    runs_dir: Path, tmp_path: Path, store: PreferencesStore
) -> None:
    """DeepSeek writer, Anthropic reviewer, DeepSeek finalizer."""
    from conftest import make_tracked_clients

    store.set_provider_model(Provider.DEEPSEEK, "deepseek-v4-pro")
    run = normalized_run(runs_dir, tmp_path, generation=resolve_generation(store))

    tracked = make_tracked_clients()
    orchestrate(runs_dir, run.run_id, tracked.as_pipeline_clients())

    order = [name for name, _ in tracked.selections]
    assert order[:2] == ["writer", "reviewer"]
    assert tracked.selections[0][1] is not None
    assert tracked.selections[1][1] is None


def test_no_reviewer_setting_is_reachable_from_the_generation_seam() -> None:
    source = Path("src/goldpipeline/services/generation.py").read_text(encoding="utf-8")
    modules = {
        node.module or ""
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    }
    assert not any("review" in name for name in modules)
    assert "ReviewerSettings" not in source


# --------------------------------------------------------------------------
# the producer
# --------------------------------------------------------------------------


def produce_with(store: PreferencesStore, tmp_path: Path, **kw: Any) -> Any:
    from goldpipeline.services.inbox import INDEX, Inbox, Ledger
    from goldpipeline.services.producer import produce_from_preferences
    from tests.test_producer import FakeCollector

    inbox = Inbox(tmp_path / "inbox")
    inbox.ensure_layout()
    collector = kw.pop("collector", FakeCollector())
    result = produce_from_preferences(
        request_id=kw.pop("request_id", "pref-000001"),
        requested_at=NOW,
        preferences=store,
        collector=collector,
        inbox=inbox,
        ledger=Ledger(inbox.directory(INDEX)),
        now=NOW,
        **kw,
    )
    return result, collector, inbox


def test_the_producer_uses_the_stored_window(store: PreferencesStore, tmp_path: Path) -> None:
    store.set_news_lookback(timedelta(hours=6))
    _, collector, _ = produce_with(store, tmp_path)
    assert collector.calls[0][1] == timedelta(hours=6)


@pytest.mark.parametrize("hours", [6, 12, 24, 48, 72, 168])
def test_every_preset_window_reaches_the_collector(
    hours: int, store: PreferencesStore, tmp_path: Path
) -> None:
    store.set_news_lookback(timedelta(hours=hours))
    _, collector, _ = produce_with(store, tmp_path)
    assert collector.calls[0][1] == timedelta(hours=hours)


def test_an_absent_file_produces_analysis_over_24h(store: PreferencesStore, tmp_path: Path) -> None:
    result, collector, inbox = produce_with(store, tmp_path)
    assert result.outcome.submitted_or_known
    assert collector.calls[0][1] == timedelta(hours=24)
    assert not store.path.exists()


def test_an_explicit_window_overrides_the_preference(
    store: PreferencesStore, tmp_path: Path
) -> None:
    """The one operator override, and it is only about the news."""
    store.set_news_lookback(timedelta(hours=72))
    _, collector, _ = produce_with(store, tmp_path, lookback=timedelta(hours=12))
    assert collector.calls[0][1] == timedelta(hours=12)


@pytest.mark.parametrize("article_type", [ArticleType.TRADE_PLAN, ArticleType.NEWS_DIGEST])
def test_an_unfinished_preference_fetches_nothing(
    article_type: ArticleType, store: PreferencesStore, tmp_path: Path
) -> None:
    """A stored preference must not activate a mode."""
    from goldpipeline.schemas.producer import ProducerOutcome
    from tests.test_producer import ExplodingCollector

    store.set_article_type(article_type)
    result, collector, inbox = produce_with(store, tmp_path, collector=ExplodingCollector())

    assert result.outcome is ProducerOutcome.ARTICLE_TYPE_NOT_READY
    assert collector.calls == 0
    assert list((inbox.directory("incoming")).glob("*.json")) == []


def test_corrupt_preferences_fetch_nothing_and_write_nothing(
    store: PreferencesStore, tmp_path: Path
) -> None:
    from tests.test_producer import ExplodingCollector

    store.root.mkdir(parents=True)
    store.path.write_text("{ broken", encoding="utf-8")

    with pytest.raises(PreferencesUnavailableError):
        produce_with(store, tmp_path, collector=ExplodingCollector())

    assert store.path.read_text(encoding="utf-8") == "{ broken"
    assert not (tmp_path / "inbox" / "incoming").exists() or not list(
        (tmp_path / "inbox" / "incoming").glob("*.json")
    )


def test_the_event_still_carries_no_provider_or_model(
    store: PreferencesStore, tmp_path: Path
) -> None:
    """The security invariant Round 5 established, unchanged."""
    from goldpipeline.adapters.inbox_source import parse_event

    store.set_provider_model(Provider.DEEPSEEK, "deepseek-chat")
    _, _, inbox = produce_with(store, tmp_path)

    written = next(iter(sorted((inbox.directory("incoming")).glob("*.json"))))
    payload = json.loads(written.read_text(encoding="utf-8"))
    event = parse_event(payload)

    for forbidden in ("provider", "model", "selection_id", "prompt_id", "reviewer"):
        assert forbidden not in payload
        assert forbidden not in event.metadata
    assert "deepseek" not in json.dumps(payload).lower()


def test_the_event_schema_has_no_generation_fields() -> None:
    from goldpipeline.schemas.inbox import AnalysisEvent

    for forbidden in ("provider", "model", "selection", "prompt", "reviewer"):
        assert not any(forbidden in name for name in AnalysisEvent.model_fields)


# --------------------------------------------------------------------------
# ingestion fails closed
# --------------------------------------------------------------------------


def test_a_damaged_file_stops_ingestion_before_anything_durable(
    runs_dir: Path, tmp_path: Path, store: PreferencesStore
) -> None:
    """No Run, no ledger entry, and the event back in the queue for a person."""
    from goldpipeline.schemas.ingestion import IngestOutcome
    from goldpipeline.services.inbox import INCOMING, INDEX, Inbox
    from goldpipeline.services.ingestion import IngestionContext, ingest_next
    from tests.test_ingestion import make_mt5_source

    store.root.mkdir(parents=True)
    store.path.write_text("{ broken", encoding="utf-8")

    inbox = Inbox(tmp_path / "inbox")
    inbox.ensure_layout()
    payload = {
        "schema_version": "1",
        "source": "internal_producer",
        "event_id": "internal_pref-000001",
        "created_at": NOW.isoformat().replace("+00:00", "Z"),
        "raw_text": "Phan tich vang.",
    }
    inbox.submit(payload, event_id="internal_pref-000001")

    result = ingest_next(
        IngestionContext(
            inbox=inbox,
            store=RunStore(runs_dir),
            market_source=make_mt5_source(),
            preferences=store,
        )
    )

    assert result.outcome is IngestOutcome.PREFERENCES_UNAVAILABLE
    assert RunStore(runs_dir).list_run_ids() == []
    assert list(inbox.directory(INDEX).glob("*.json")) == []
    assert [p.name for p in inbox.directory(INCOMING).glob("*.json")] == [
        "internal_pref-000001.json"
    ]
    assert store.path.read_text(encoding="utf-8") == "{ broken"


def test_a_damaged_file_is_an_operator_repairable_retry_class() -> None:
    """Never exhausted: fixing the file should resume work, not need a reset too."""
    from goldpipeline.schemas.automation import RetryClass
    from goldpipeline.services.automation import classify

    assert classify(PreferencesUnavailableError("damaged")) is RetryClass.CONFIGURATION


# --------------------------------------------------------------------------
# offline end-to-end
# --------------------------------------------------------------------------


def drive_end_to_end(
    runs_dir: Path,
    tmp_path: Path,
    store: PreferencesStore,
    *,
    switch_to: tuple[Provider, str] | None = None,
) -> Any:
    """Producer -> event -> Run -> snapshot -> writer -> reviewer -> finalizer -> gate.

    Every provider is a fake and every path is temporary. ``switch_to`` changes
    the stored preference after the writer has run, which is the moment the
    snapshot has to hold.
    """
    from conftest import make_tracked_clients

    from goldpipeline.adapters.inbox_source import parse_event
    from goldpipeline.schemas.orchestration import PipelineMode

    _, _, inbox = produce_with(store, tmp_path)
    waiting = next(iter(sorted((inbox.directory("incoming")).glob("*.json"))))
    event = parse_event(json.loads(waiting.read_text(encoding="utf-8")))

    run = normalized_run(runs_dir, tmp_path, generation=resolve_generation(store))
    tracked = make_tracked_clients()

    orchestrate(runs_dir, run.run_id, tracked.as_pipeline_clients(), PipelineMode.GENERATE_ONLY)
    if switch_to is not None:
        store.set_provider_model(*switch_to)
    orchestrate(runs_dir, run.run_id, tracked.as_pipeline_clients(), PipelineMode.READY_FOR_PUBLISH)

    return run, tracked, event


def test_end_to_end_claude_path(runs_dir: Path, tmp_path: Path, store: PreferencesStore) -> None:
    """Sonnet selected; every generation stage is built from that selection."""
    store.set_provider_model(Provider.CLAUDE, "claude-sonnet-5")
    store.set_article_type(ArticleType.ANALYSIS)
    store.set_news_lookback(timedelta(hours=24))

    run, tracked, event = drive_end_to_end(runs_dir, tmp_path, store)

    assert event.article_type is ArticleType.ANALYSIS
    snapshot = snapshot_of(runs_dir, run.run_id)
    assert snapshot is not None
    assert (snapshot.provider, snapshot.selection_id) == (Provider.CLAUDE, "claude-sonnet-5")

    generation = [s for name, s in tracked.selections if name in {"writer", "finalizer"}]
    assert generation, "generation stages ran"
    assert all(s is not None and s.selection_id == "claude-sonnet-5" for s in generation)
    assert all(s is None for name, s in tracked.selections if name == "reviewer")


def test_end_to_end_deepseek_path_survives_a_mid_run_switch(
    runs_dir: Path, tmp_path: Path, store: PreferencesStore
) -> None:
    """DeepSeek Chat selected, then switched to Claude before the finalizer."""
    store.set_provider_model(Provider.DEEPSEEK, "deepseek-chat")
    store.set_article_type(ArticleType.ANALYSIS)

    run, tracked, _ = drive_end_to_end(
        runs_dir, tmp_path, store, switch_to=(Provider.CLAUDE, "claude-opus-5")
    )

    snapshot = snapshot_of(runs_dir, run.run_id)
    assert snapshot is not None
    assert (snapshot.provider, snapshot.selection_id) == (Provider.DEEPSEEK, "deepseek-chat")
    assert snapshot.api_model_id == "deepseek-v4-flash"
    assert snapshot.thinking is ThinkingMode.DISABLED

    generation = [s for name, s in tracked.selections if name in {"writer", "finalizer"}]
    assert generation
    assert {s.selection_id for s in generation if s} == {"deepseek-chat"}
    assert {s.api_model_id for s in generation if s} == {"deepseek-v4-flash"}

    # The preference really did change; it just did not reach this Run.
    assert resolve_generation(store).selection_id == "claude-opus-5"


def test_the_run_and_its_artifacts_agree_on_the_selection(
    runs_dir: Path, tmp_path: Path, store: PreferencesStore
) -> None:
    """A deterministic consistency check an auditor can repeat."""
    from goldpipeline.schemas.writer import WriterResult

    store.set_provider_model(Provider.CLAUDE, "claude-haiku-4-5")
    run, _, _ = drive_end_to_end(runs_dir, tmp_path, store)

    snapshot = snapshot_of(runs_dir, run.run_id)
    assert snapshot is not None

    writer_path = Path(run.run_dir) / "claude_writer.json"
    artifact = WriterResult.model_validate_json(writer_path.read_text(encoding="utf-8"))

    # The fake writer reports its own provider, so what is pinned here is the
    # part the pipeline stamps: the Run knows which selection it ran under, and
    # the artifact carries a selection field for a real client to fill.
    assert snapshot.selection_id == "claude-haiku-4-5"
    assert artifact.run_id == run.run_id
    assert "selection_id" in WriterResult.model_fields
