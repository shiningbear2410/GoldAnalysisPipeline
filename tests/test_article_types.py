"""Article types: declared everywhere, executable only where implemented.

The risk this round introduces is not that ``TRADE_PLAN`` fails - it is that it
quietly *succeeds*, by falling back to the analysis prompt and producing a
market commentary under a heading that promises entries and stops. So most of
what follows checks that an unimplemented mode stops, stops early, and stops
without substituting anything.

The second risk is regression: ANALYSIS must behave exactly as it did before
routing existed. Those tests pin the prompt id rather than trusting that nothing
moved.

Offline throughout: no MT5, no model, no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from goldpipeline.domain.errors import ArticleTypeNotReadyError
from goldpipeline.prompts import DEFAULT_WRITER_PROMPT
from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.inbox import AnalysisEvent
from goldpipeline.services.article_routing import (
    READY_TYPES,
    REMOTE_ALLOWED_TYPES,
    SPECS,
    require_ready,
    spec_for,
    writer_prompt_for,
)

BASE_EVENT = {
    "schema_version": "1",
    "source": "gold_analysis_bot",
    "event_id": "article_type_test_0001",
    "created_at": "2026-09-02T10:00:00Z",
    "raw_text": "phan tich",
}


# ------------------------------------------------------------------- schema
class TestSchema:
    def test_an_event_without_article_type_is_analysis(self) -> None:
        """Backward compatibility: silence means what it has always meant."""
        assert AnalysisEvent.model_validate(BASE_EVENT).article_type is ArticleType.ANALYSIS

    @pytest.mark.parametrize("value", ["ANALYSIS", "TRADE_PLAN", "NEWS_DIGEST"])
    def test_every_declared_type_is_accepted(self, value: str) -> None:
        event = AnalysisEvent.model_validate({**BASE_EVENT, "article_type": value})
        assert event.article_type is ArticleType(value)

    @pytest.mark.parametrize(
        "value",
        ["analysis", "TRADEPLAN", "SOMETHING_ELSE", "", "gold_writer_v2", 1, None],
    )
    def test_anything_else_is_refused(self, value: Any) -> None:
        with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
            AnalysisEvent.model_validate({**BASE_EVENT, "article_type": value})

    def test_metadata_cannot_smuggle_an_article_type(self) -> None:
        """Routing reads the declared field; metadata is inert data."""
        event = AnalysisEvent.model_validate(
            {**BASE_EVENT, "metadata": {"article_type": "TRADE_PLAN"}}
        )
        assert event.article_type is ArticleType.ANALYSIS

    def test_no_prompt_model_or_destination_field_exists(self) -> None:
        """The narrowed invariant, asserted rather than trusted."""
        forbidden = {
            "prompt",
            "prompt_id",
            "prompt_version",
            "model",
            "provider",
            "reviewer",
            "finalizer",
            "target_chat",
            "chat_target",
            "publish",
            "runs_dir",
            "path",
        }
        assert forbidden.isdisjoint(set(AnalysisEvent.model_fields))

    def test_unknown_keys_are_still_refused(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            AnalysisEvent.model_validate({**BASE_EVENT, "prompt_id": "gold_writer_v2"})


# ------------------------------------------------------------------ routing
class TestRouting:
    def test_every_enum_member_has_a_spec(self) -> None:
        """A mode with no entry would fall through the lookup."""
        assert set(SPECS) == set(ArticleType)

    def test_analysis_uses_the_current_production_prompt(self) -> None:
        """Routing moved; prose did not."""
        assert writer_prompt_for(ArticleType.ANALYSIS) == DEFAULT_WRITER_PROMPT
        assert DEFAULT_WRITER_PROMPT == "gold_writer_v2"

    def test_analysis_is_the_only_ready_type(self) -> None:
        assert {ArticleType.ANALYSIS} == READY_TYPES

    @pytest.mark.parametrize("kind", [ArticleType.TRADE_PLAN, ArticleType.NEWS_DIGEST])
    def test_unimplemented_types_refuse(self, kind: ArticleType) -> None:
        with pytest.raises(ArticleTypeNotReadyError) as caught:
            require_ready(kind)
        assert caught.value.code == "ARTICLE_TYPE_NOT_READY"
        assert spec_for(kind).requires, "a refusal must say what is missing"

    @pytest.mark.parametrize("kind", [ArticleType.TRADE_PLAN, ArticleType.NEWS_DIGEST])
    def test_unimplemented_types_never_borrow_the_analysis_prompt(self, kind: ArticleType) -> None:
        """The failure mode that matters: silent substitution."""
        assert spec_for(kind).prompt_id is None
        with pytest.raises(ArticleTypeNotReadyError):
            writer_prompt_for(kind)

    def test_no_placeholder_prompt_files_were_created(self) -> None:
        """Prefer no prompt over a fake usable one."""
        prompts = Path("src/goldpipeline/prompts")
        names = {p.stem for p in prompts.glob("*.md")}
        assert "gold_trade_plan_v1" not in names
        assert "gold_news_digest_v1" not in names


# ----------------------------------------------------------- remote policy
class TestRemotePolicy:
    def test_only_analysis_is_allowed_remotely(self) -> None:
        assert {ArticleType.ANALYSIS} == REMOTE_ALLOWED_TYPES

    def test_remote_policy_is_narrower_than_readiness(self) -> None:
        """Being implemented is not the same as being allowed from a network."""
        assert REMOTE_ALLOWED_TYPES <= READY_TYPES


# -------------------------------------------------------- run auditability
class TestRunProvenance:
    def test_provenance_defaults_to_analysis(self) -> None:
        from goldpipeline.schemas.manifest import RunProvenance

        provenance = RunProvenance(analysis_origin="file", market_origin="file")
        assert provenance.article_type is ArticleType.ANALYSIS

    def test_provenance_records_the_requested_type(self) -> None:
        from goldpipeline.schemas.manifest import RunProvenance

        provenance = RunProvenance(
            article_type=ArticleType.TRADE_PLAN,
            analysis_origin="inbox",
            market_origin="mt5",
        )
        assert provenance.article_type is ArticleType.TRADE_PLAN

    def test_historical_manifests_still_load(self) -> None:
        """Real Runs on disk predate the field entirely."""
        from goldpipeline.schemas.manifest import RunManifest

        seen = 0
        for run in sorted(Path("runs").iterdir()):
            manifest_file = run / "manifest.json"
            if not manifest_file.is_file():
                continue
            manifest = RunManifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))
            seen += 1
            if manifest.provenance is not None:
                assert manifest.provenance.article_type is ArticleType.ANALYSIS
        assert seen, "expected at least one historical Run to check"


# ------------------------------------------------------------ idempotency
class TestPayloadIdentity:
    """Article type is part of the payload, so identity follows from the bytes."""

    def encoded(self, payload: dict[str, Any]) -> str:
        from goldpipeline.storage.atomic import encode_json, sha256_bytes

        return sha256_bytes(encode_json(payload))

    def test_same_type_same_bytes_same_digest(self) -> None:
        one = {**BASE_EVENT, "article_type": "ANALYSIS"}
        two = {**BASE_EVENT, "article_type": "ANALYSIS"}
        assert self.encoded(one) == self.encoded(two)

    def test_different_type_is_a_different_payload(self) -> None:
        """Same id with a different mode must collide as a conflict, not pass."""
        analysis = {**BASE_EVENT, "article_type": "ANALYSIS"}
        trade = {**BASE_EVENT, "article_type": "TRADE_PLAN"}
        assert self.encoded(analysis) != self.encoded(trade)

    def test_absent_and_explicit_analysis_are_different_bytes(self) -> None:
        """An honest consequence: adding the key changes the digest.

        Not a defect - the ledger compares the bytes a producer wrote, and a
        producer that starts sending the field is sending different bytes. It
        matters only if the same event_id is re-sent across that change, which
        is a conflict worth seeing rather than hiding.
        """
        assert self.encoded(BASE_EVENT) != self.encoded({**BASE_EVENT, "article_type": "ANALYSIS"})


# ------------------------------------------------- fixtures on disk still load
class TestHistoricalEvents:
    def test_processed_inbox_events_still_parse(self) -> None:
        from goldpipeline.adapters.inbox_source import parse_event

        seen = 0
        for path in sorted(Path("inbox/processed").glob("*.json")):
            event = parse_event(json.loads(path.read_text(encoding="utf-8")))
            assert event.article_type is ArticleType.ANALYSIS
            seen += 1
        assert seen, "expected historical events to check"
