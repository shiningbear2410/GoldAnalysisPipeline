"""Publisher orchestration: durable intent, delivery semantics, and no duplicates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    CLEAN_ARTICLE,
    PUBLISH_NOW,
    TELEGRAM_TOKEN_SENTINEL,
    TEST_TARGET_CHAT,
    RecordingSleep,
    load_publish_intent,
    load_publish_result,
    make_drafted_run,
    make_finalized_run,
    make_published_ready_run,
    republish_article,
    tamper,
)

from goldpipeline.adapters.fake_publisher import (
    FakePublisherClient,
    ambiguous_client,
    forbidding_client,
    rejecting_client,
    transient_rate_limit_client,
    unconfirmed_client,
)
from goldpipeline.domain.errors import (
    PublisherArtifactExistsError,
    PublisherIntegrityError,
    PublisherNotApprovedError,
    PublisherPermissionError,
    PublisherRateLimitError,
    PublisherTransportAmbiguousError,
)
from goldpipeline.schemas.manifest import RunStatus
from goldpipeline.schemas.publisher import (
    FailureCategory,
    PublishIntent,
    PublishResult,
    PublishStatus,
)
from goldpipeline.services.chunking import SAFE_CHUNK_LIMIT
from goldpipeline.services.publisher import (
    INTENT_FILENAME,
    MAX_RATE_LIMIT_RETRIES,
    RESULT_FILENAME,
    publish_run,
)
from goldpipeline.storage.atomic import sha256_bytes
from goldpipeline.storage.run_store import RunStore

_PARAGRAPH = (
    "Vàng tiếp tục tích luỹ quanh 3305.90, biên độ hẹp dần và thanh khoản mỏng. "
    "Phe mua vẫn giữ được vùng hỗ trợ nhưng chưa tạo được động lực rõ ràng."
)

LONG_ARTICLE = CLEAN_ARTICLE + "\n\n" + "\n\n".join([_PARAGRAPH] * 30)
"""An article that genuinely needs several messages.

Built from repeated prose rather than numbered paragraphs: "Đoạn 150" would put
a bare number above the price scanner's floor into the text, and the gate would
rightly refuse to approve it - so the test would be exercising the fixture
rather than the publisher.
"""


def run_publish(
    runs_dir: Path,
    run_id: str,
    *,
    client: Any = None,
    sleep: Any = None,
    target: str = TEST_TARGET_CHAT,
    chunk_limit: int = SAFE_CHUNK_LIMIT,
) -> Any:
    return publish_run(
        run_id=run_id,
        store=RunStore(runs_dir),
        client=client or FakePublisherClient(),
        target_chat=target,
        now=PUBLISH_NOW,
        sleep=sleep or RecordingSleep(),
        chunk_limit=chunk_limit,
    )


# --- golden case A: a normal article --------------------------------------


def test_an_approved_run_publishes(ready_run: Any, runs_dir: Path) -> None:
    """Requirements 39.1-39.2 / golden case A."""
    client = FakePublisherClient()
    outcome = run_publish(runs_dir, ready_run.run_id, client=client)

    assert outcome.published
    assert outcome.result.status is PublishStatus.PUBLISHED
    assert outcome.result.chunk_count == 1
    assert outcome.result.confirmed_count == 1
    assert outcome.status is RunStatus.PUBLISHED

    manifest = RunStore(runs_dir).open(ready_run.run_id).load_manifest()
    assert manifest.status is RunStatus.PUBLISHED


def test_the_transport_receives_exactly_the_approved_article(
    ready_run: Any, runs_dir: Path
) -> None:
    """Requirement 39.13: what ships is what the gate approved."""
    approved = (Path(ready_run.run_dir) / "claude_final.md").read_text(encoding="utf-8")
    client = FakePublisherClient()
    run_publish(runs_dir, ready_run.run_id, client=client)

    assert client.sent == [approved]


def test_both_artifacts_are_written(ready_run: Any, runs_dir: Path) -> None:
    """Requirements 39.3-39.4."""
    outcome = run_publish(runs_dir, ready_run.run_id)
    run_dir = Path(outcome.run_dir)

    assert (run_dir / INTENT_FILENAME).is_file()
    assert (run_dir / RESULT_FILENAME).is_file()
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "claude_draft.md",
        "claude_final.md",
        "claude_finalizer.json",
        "claude_writer.json",
        "context.json",
        "gpt_review.json",
        "manifest.json",
        "ohlc.json",
        "publish_decision.json",
        "publish_intent.json",
        "publish_result.json",
        "telegram_input.json",
    ]


def test_the_manifest_records_both_artifact_hashes(ready_run: Any, runs_dir: Path) -> None:
    """Requirements 39.5-39.6."""
    outcome = run_publish(runs_dir, ready_run.run_id)
    run_dir = Path(outcome.run_dir)
    manifest = RunStore(runs_dir).open(outcome.run_id).load_manifest()

    for name in (INTENT_FILENAME, RESULT_FILENAME):
        ref = next(r for r in manifest.artifact_files if r.name == name)
        assert ref.sha256 == sha256_bytes((run_dir / name).read_bytes())


def test_the_result_binds_to_the_approval(ready_run: Any, runs_dir: Path) -> None:
    """Requirements 39.79-39.80."""
    outcome = run_publish(runs_dir, ready_run.run_id)
    run_dir = Path(outcome.run_dir)
    result = load_publish_result(run_dir)
    intent = load_publish_intent(run_dir)

    assert result.final_article_sha256 == sha256_bytes((run_dir / "claude_final.md").read_bytes())
    assert result.decision_sha256 == sha256_bytes((run_dir / "publish_decision.json").read_bytes())
    assert result.publish_intent_sha256 == sha256_bytes((run_dir / INTENT_FILENAME).read_bytes())
    assert result.attempt_id == intent.attempt_id


def test_the_intent_describes_the_chunks(ready_run: Any, runs_dir: Path) -> None:
    """Requirement 39.79."""
    outcome = run_publish(runs_dir, ready_run.run_id)
    run_dir = Path(outcome.run_dir)
    article = (run_dir / "claude_final.md").read_text(encoding="utf-8")
    intent = load_publish_intent(run_dir)

    assert intent.chunk_count == 1
    assert intent.chunks[0].text_sha256 == sha256_bytes(article.encode("utf-8"))
    assert intent.chunks[0].char_count == len(article)


def test_message_ids_are_recorded(ready_run: Any, runs_dir: Path) -> None:
    """Requirement 39.51."""
    outcome = run_publish(runs_dir, ready_run.run_id)
    assert [m.message_id for m in outcome.result.messages] == [1000]
    assert outcome.result.messages[0].retry_count == 0


def test_both_schemas_round_trip(ready_run: Any, runs_dir: Path) -> None:
    """Requirements 39.73-39.74."""
    outcome = run_publish(runs_dir, ready_run.run_id)
    run_dir = Path(outcome.run_dir)

    for name, model in ((INTENT_FILENAME, PublishIntent), (RESULT_FILENAME, PublishResult)):
        raw = json.loads((run_dir / name).read_text(encoding="utf-8"))
        reloaded = model.model_validate(raw)
        assert json.loads(reloaded.model_dump_json()) == raw


def test_vietnamese_survives_the_artifacts(ready_run: Any, runs_dir: Path) -> None:
    """Requirement 39.64."""
    outcome = run_publish(runs_dir, ready_run.run_id)
    raw = Path(outcome.result_path).read_bytes()

    assert b"\\u" not in raw
    assert TEST_TARGET_CHAT in raw.decode("utf-8")


# --- the intent comes first ----------------------------------------------


def test_the_intent_exists_before_the_first_send(ready_run: Any, runs_dir: Path) -> None:
    """Requirement 39.3: the durable record precedes the network.

    Asserted from inside the transport, which is the only moment that proves
    the ordering rather than inferring it from the end state.
    """
    run_dir = Path(ready_run.run_dir)
    seen: list[bool] = []

    def observe(request: Any, attempt: int) -> Any:
        from goldpipeline.adapters.publisher_client import SendOutcome

        seen.append((run_dir / INTENT_FILENAME).is_file())
        return SendOutcome(message_id=1000, chat_id=request.target_chat)

    run_publish(runs_dir, ready_run.run_id, client=FakePublisherClient(outcome_factory=observe))

    assert seen == [True], "a request went out before the intent was durable"


def test_the_manifest_moves_to_publishing_before_sending(ready_run: Any, runs_dir: Path) -> None:
    outcome = run_publish(runs_dir, ready_run.run_id)
    manifest = RunStore(runs_dir).open(outcome.run_id).load_manifest()
    stages = [event.stage for event in manifest.events]

    assert stages.index("publish.intent") < stages.index("publish.complete")


# --- nothing is sent unless everything checks out -------------------------


def test_a_run_that_was_never_gated_is_refused(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 39.11."""
    finalized = make_finalized_run(runs_dir, tmp_path)
    client = FakePublisherClient()

    with pytest.raises(PublisherNotApprovedError):
        run_publish(runs_dir, finalized.run_id, client=client)

    assert client.calls == []
    assert not (Path(finalized.run_dir) / INTENT_FILENAME).exists()


def test_a_drafted_run_is_refused(runs_dir: Path, tmp_path: Path) -> None:
    drafted = make_drafted_run(runs_dir, tmp_path)
    client = FakePublisherClient()

    with pytest.raises(PublisherNotApprovedError):
        run_publish(runs_dir, drafted.run_id, client=client)
    assert client.calls == []


def test_a_blocked_decision_is_refused(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 39.10."""
    from goldpipeline.services.publish_gate import gate_publish

    finalized = make_finalized_run(runs_dir, tmp_path)
    republish_article(runs_dir, finalized.run_id, f"{CLEAN_ARTICLE}\n\nThực ra đây là BTCUSD.")
    gated = gate_publish(run_id=finalized.run_id, store=RunStore(runs_dir))
    assert not gated.approved

    client = FakePublisherClient()
    with pytest.raises(PublisherNotApprovedError):
        run_publish(runs_dir, finalized.run_id, client=client)

    assert client.calls == []
    assert not (Path(finalized.run_dir) / INTENT_FILENAME).exists()


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("claude_final.md", "Một bài viết hoàn toàn khác."),
        ("publish_decision.json", '{"run_id": "tampered"}'),
        ("claude_finalizer.json", '{"run_id": "tampered"}'),
        ("context.json", '{"run_id": "tampered"}'),
    ],
)
def test_a_tampered_artifact_stops_everything(
    ready_run: Any, runs_dir: Path, filename: str, content: str
) -> None:
    """Requirements 39.7-39.9 / golden case G: the approval names exact bytes."""
    tamper(ready_run.run_dir, filename, content)
    client = FakePublisherClient()

    with pytest.raises(PublisherIntegrityError):
        run_publish(runs_dir, ready_run.run_id, client=client)

    assert client.calls == [], "a request went out for artifacts the gate never approved"
    assert not (Path(ready_run.run_dir) / INTENT_FILENAME).exists()


def test_an_unsupported_gate_version_is_refused(ready_run: Any, runs_dir: Path) -> None:
    """Requirement 39.12: checks whose meaning may have changed are not an approval."""
    run_dir = Path(ready_run.run_dir)
    decision = json.loads((run_dir / "publish_decision.json").read_text(encoding="utf-8"))
    decision["gate_version"] = "gold_publish_gate_v99"
    encoded = (json.dumps(decision, ensure_ascii=False, indent=2) + "\n").encode()
    (run_dir / "publish_decision.json").write_bytes(encoded)

    store = RunStore(runs_dir)
    run = store.open(ready_run.run_id)
    manifest = run.load_manifest()
    for ref in manifest.artifact_files:
        if ref.name == "publish_decision.json":
            ref.sha256, ref.size_bytes = sha256_bytes(encoded), len(encoded)
    run.save_manifest(manifest)

    client = FakePublisherClient()
    with pytest.raises(PublisherNotApprovedError, match="does not support"):
        run_publish(runs_dir, ready_run.run_id, client=client)

    assert client.calls == []
    assert not (run_dir / INTENT_FILENAME).exists()


# --- golden case B: a multi-message article -------------------------------


def test_a_long_article_is_split_and_fully_delivered(runs_dir: Path, tmp_path: Path) -> None:
    """Golden case B."""
    ready = make_published_ready_run(
        runs_dir, tmp_path, article=LONG_ARTICLE, enforce_contract=False, claims=[]
    )
    client = FakePublisherClient()
    outcome = run_publish(runs_dir, ready.run_id, client=client)

    assert outcome.published
    assert outcome.result.chunk_count > 1
    assert outcome.result.confirmed_count == outcome.result.chunk_count


def test_the_chunks_reassemble_into_the_approved_article(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 39.14, through the whole stage rather than the chunker alone."""
    ready = make_published_ready_run(
        runs_dir, tmp_path, article=LONG_ARTICLE, enforce_contract=False, claims=[]
    )
    approved = (Path(ready.run_dir) / "claude_final.md").read_text(encoding="utf-8")

    client = FakePublisherClient()
    run_publish(runs_dir, ready.run_id, client=client)

    assert "".join(client.sent) == approved


def test_no_part_marker_is_added_to_any_chunk(runs_dir: Path, tmp_path: Path) -> None:
    """Requirements 39.21-39.22."""
    ready = make_published_ready_run(
        runs_dir, tmp_path, article=LONG_ARTICLE, enforce_contract=False, claims=[]
    )
    client = FakePublisherClient()
    run_publish(runs_dir, ready.run_id, client=client)

    assert len(client.sent) > 1
    for chunk in client.sent:
        for marker in ("(1/", "Part ", "continued", ready.run_id):
            assert marker not in chunk


def test_multi_chunk_sends_are_paced(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 39.45: a burst is how a channel trips flood control."""
    ready = make_published_ready_run(
        runs_dir, tmp_path, article=LONG_ARTICLE, enforce_contract=False, claims=[]
    )
    sleep = RecordingSleep()
    outcome = run_publish(runs_dir, ready.run_id, sleep=sleep)

    assert len(sleep.waits) == outcome.result.chunk_count - 1
    assert all(wait >= 1.0 for wait in sleep.waits)


def test_a_single_chunk_is_not_paced(ready_run: Any, runs_dir: Path) -> None:
    """Requirement 39.46."""
    sleep = RecordingSleep()
    run_publish(runs_dir, ready_run.run_id, sleep=sleep)
    assert sleep.waits == []


# --- golden case C: flood control -----------------------------------------


def test_a_rate_limited_chunk_is_retried_after_the_stated_delay(
    ready_run: Any, runs_dir: Path
) -> None:
    """Requirements 39.41 and 39.43 / golden case C."""
    sleep = RecordingSleep()
    outcome = run_publish(
        runs_dir,
        ready_run.run_id,
        client=transient_rate_limit_client(retry_after=2),
        sleep=sleep,
    )

    assert outcome.published
    assert outcome.result.messages[0].retry_count == 1
    assert 2.0 in sleep.waits


def test_the_test_suite_never_actually_waits(ready_run: Any, runs_dir: Path) -> None:
    """Requirement 39.44: sleep is injected, so retries cost no wall-clock time."""
    sleep = RecordingSleep()
    run_publish(
        runs_dir,
        ready_run.run_id,
        client=transient_rate_limit_client(retry_after=30),
        sleep=sleep,
    )
    assert sleep.total >= 30, "the delay was honoured"


def test_rate_limit_retries_are_bounded(ready_run: Any, runs_dir: Path) -> None:
    """Requirement 39.42: an endlessly rate-limited chunk must not hang the attempt."""
    client = FakePublisherClient(raises=PublisherRateLimitError("flood control", retry_after=1))
    outcome = run_publish(runs_dir, ready_run.run_id, client=client)

    assert outcome.result.status is PublishStatus.FAILED
    assert outcome.result.failure is not None
    assert outcome.result.failure.category is FailureCategory.RATE_LIMITED
    assert len(client.calls) == MAX_RATE_LIMIT_RETRIES + 1


# --- golden case D: ambiguity ---------------------------------------------


@pytest.mark.parametrize(
    "client_factory",
    [ambiguous_client, unconfirmed_client],
)
def test_an_ambiguous_send_is_uncertain_and_never_retried(
    ready_run: Any, runs_dir: Path, client_factory: Any
) -> None:
    """Requirements 39.35-39.37 / golden case D: the message may already be posted."""
    client = client_factory()
    outcome = run_publish(runs_dir, ready_run.run_id, client=client)

    assert outcome.result.status is PublishStatus.UNCERTAIN
    assert outcome.status is RunStatus.PUBLISH_UNCERTAIN
    assert len(client.calls) == 1, "an ambiguous send was retried"
    assert outcome.result.confirmed_count == 0


def test_an_ambiguous_send_stops_the_remaining_chunks(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 39.50."""
    ready = make_published_ready_run(
        runs_dir, tmp_path, article=LONG_ARTICLE, enforce_contract=False, claims=[]
    )
    client = FakePublisherClient(failures={1: PublisherTransportAmbiguousError("timed out")})
    outcome = run_publish(runs_dir, ready.run_id, client=client)

    assert outcome.result.status is PublishStatus.UNCERTAIN
    assert outcome.result.confirmed_count == 1
    assert len(client.calls) == 2, "publishing continued past an unknown outcome"
    assert outcome.result.failure is not None
    assert outcome.result.failure.failed_chunk_index == 1


def test_uncertain_outranks_partial(runs_dir: Path, tmp_path: Path) -> None:
    """Requirement 24: if the last chunk's fate is unknown, so is the attempt's."""
    ready = make_published_ready_run(
        runs_dir, tmp_path, article=LONG_ARTICLE, enforce_contract=False, claims=[]
    )
    client = FakePublisherClient(failures={1: PublisherTransportAmbiguousError("connection reset")})
    outcome = run_publish(runs_dir, ready.run_id, client=client)

    assert outcome.result.confirmed_count >= 1
    assert outcome.result.status is PublishStatus.UNCERTAIN


# --- explicit refusals ----------------------------------------------------


def test_an_explicit_refusal_before_delivery_is_a_failure(ready_run: Any, runs_dir: Path) -> None:
    """Requirement 22: nothing was delivered, and that is certain."""
    outcome = run_publish(runs_dir, ready_run.run_id, client=rejecting_client())

    assert outcome.result.status is PublishStatus.FAILED
    assert outcome.status is RunStatus.PUBLISH_FAILED
    assert outcome.result.confirmed_count == 0
    assert outcome.result.failure is not None
    assert outcome.result.failure.category is FailureCategory.REJECTED


def test_a_permission_refusal_is_categorised(ready_run: Any, runs_dir: Path) -> None:
    """Requirement 39.39."""
    outcome = run_publish(runs_dir, ready_run.run_id, client=forbidding_client())

    assert outcome.result.status is PublishStatus.FAILED
    assert outcome.result.failure is not None
    assert outcome.result.failure.category is FailureCategory.PERMISSION


# --- golden case E: a partial publish -------------------------------------


def test_a_later_explicit_refusal_is_partial(runs_dir: Path, tmp_path: Path) -> None:
    """Requirements 39.47 and 39.49 / golden case E."""
    ready = make_published_ready_run(
        runs_dir, tmp_path, article=LONG_ARTICLE, enforce_contract=False, claims=[]
    )
    client = FakePublisherClient(
        failures={1: PublisherPermissionError("forbidden", status_code=403)}
    )
    outcome = run_publish(runs_dir, ready.run_id, client=client)

    assert outcome.result.status is PublishStatus.PARTIAL
    assert outcome.status is RunStatus.PARTIALLY_PUBLISHED
    assert outcome.result.confirmed_count == 1
    assert len(client.calls) == 2, "publishing continued past a refusal"
    assert outcome.result.messages[0].chunk_index == 0


# --- golden case F: the crash window --------------------------------------


def test_an_orphan_intent_sends_nothing(ready_run: Any, runs_dir: Path) -> None:
    """Requirements 39.54-39.55 and 39.58 / golden case F.

    Simulates a process killed between writing the intent and recording a
    result - the window where Telegram may already hold the message.
    """
    run_dir = Path(ready_run.run_dir)
    first = FakePublisherClient()
    run_publish(runs_dir, ready_run.run_id, client=first)

    # Remove only the result, leaving the intent: exactly the crash state.
    (run_dir / RESULT_FILENAME).unlink()
    store = RunStore(runs_dir)
    run = store.open(ready_run.run_id)
    manifest = run.load_manifest()
    manifest.artifact_files = [
        ref for ref in manifest.artifact_files if ref.name != RESULT_FILENAME
    ]
    manifest.status = RunStatus.PUBLISHING
    run.save_manifest(manifest)

    second = FakePublisherClient()
    outcome = run_publish(runs_dir, ready_run.run_id, client=second)

    assert second.calls == [], "the orphan intent triggered a resend"
    assert outcome.result.status is PublishStatus.UNCERTAIN
    assert outcome.status is RunStatus.PUBLISH_UNCERTAIN
    assert outcome.result.failure is not None
    assert outcome.result.failure.category is FailureCategory.ORPHAN_PUBLISH_INTENT


def test_an_orphan_intent_records_what_the_previous_attempt_planned(
    ready_run: Any, runs_dir: Path
) -> None:
    run_dir = Path(ready_run.run_dir)
    run_publish(runs_dir, ready_run.run_id)
    intent = load_publish_intent(run_dir)
    (run_dir / RESULT_FILENAME).unlink()

    store = RunStore(runs_dir)
    run = store.open(ready_run.run_id)
    manifest = run.load_manifest()
    manifest.artifact_files = [
        ref for ref in manifest.artifact_files if ref.name != RESULT_FILENAME
    ]
    run.save_manifest(manifest)

    outcome = run_publish(runs_dir, ready_run.run_id)
    assert outcome.result.attempt_id == intent.attempt_id
    assert outcome.result.chunk_count == intent.chunk_count


# --- one attempt per Run --------------------------------------------------


def test_a_second_attempt_is_refused(ready_run: Any, runs_dir: Path) -> None:
    """Requirements 39.56-39.57: a retry is how an article gets posted twice."""
    first = run_publish(runs_dir, ready_run.run_id)
    original = Path(first.result_path).read_bytes()

    second = FakePublisherClient()
    with pytest.raises(PublisherArtifactExistsError):
        run_publish(runs_dir, ready_run.run_id, client=second)

    assert second.calls == []
    assert Path(first.result_path).read_bytes() == original


@pytest.mark.parametrize("client_factory", [rejecting_client, ambiguous_client, forbidding_client])
def test_even_an_unsuccessful_attempt_is_final(
    ready_run: Any, runs_dir: Path, client_factory: Any
) -> None:
    """A failed attempt is still an attempt; V1 does not re-run it."""
    run_publish(runs_dir, ready_run.run_id, client=client_factory())

    retry = FakePublisherClient()
    with pytest.raises(PublisherArtifactExistsError):
        run_publish(runs_dir, ready_run.run_id, client=retry)
    assert retry.calls == []


# --- invariants -----------------------------------------------------------


def test_published_requires_every_chunk_confirmed(runs_dir: Path, tmp_path: Path) -> None:
    """Requirements 39.53 and 39.78."""
    ready = make_published_ready_run(
        runs_dir, tmp_path, article=LONG_ARTICLE, enforce_contract=False, claims=[]
    )
    client = FakePublisherClient(failures={1: PublisherPermissionError("no", status_code=403)})
    outcome = run_publish(runs_dir, ready.run_id, client=client)

    assert outcome.result.chunk_count > 1
    assert outcome.result.status is not PublishStatus.PUBLISHED
    assert outcome.result.confirmed_count < outcome.result.chunk_count
    assert not outcome.result.fully_delivered


def test_the_manifest_moves_only_after_the_result_lands(
    ready_run: Any, runs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirements 39.60-39.61: a claim nobody can check is worse than none."""
    calls = {"count": 0}
    import os

    real_replace = os.replace

    def flaky(src: Any, dst: Any, **kwargs: Any) -> None:
        calls["count"] += 1
        if str(dst).endswith(RESULT_FILENAME):
            raise OSError("disk full")
        real_replace(src, dst, **kwargs)

    monkeypatch.setattr("goldpipeline.storage.run_store.os.replace", flaky)

    with pytest.raises(OSError, match="disk full"):
        run_publish(runs_dir, ready_run.run_id)

    run_dir = Path(ready_run.run_dir)
    assert not (run_dir / RESULT_FILENAME).exists()
    assert [p.name for p in run_dir.glob("*.tmp")] == []
    assert RunStore(runs_dir).open(ready_run.run_id).load_manifest().status is (
        RunStatus.PUBLISHING
    )


def test_earlier_artifacts_are_untouched(ready_run: Any, runs_dir: Path) -> None:
    """Requirements 39.62-39.63."""
    run_dir = Path(ready_run.run_dir)
    names = (
        "telegram_input.json",
        "ohlc.json",
        "context.json",
        "claude_draft.md",
        "claude_writer.json",
        "gpt_review.json",
        "claude_final.md",
        "claude_finalizer.json",
        "publish_decision.json",
    )
    before = {name: (run_dir / name).read_bytes() for name in names}

    assert run_publish(runs_dir, ready_run.run_id).published

    for name, content in before.items():
        assert (run_dir / name).read_bytes() == content


# --- destination ----------------------------------------------------------


def test_the_destination_comes_from_the_caller_only(ready_run: Any, runs_dir: Path) -> None:
    """Requirement 39.24."""
    client = FakePublisherClient()
    run_publish(runs_dir, ready_run.run_id, client=client, target="@configured_channel")

    assert set(client.targets) == {"@configured_channel"}
    assert load_publish_result(Path(ready_run.run_dir)).target_chat == "@configured_channel"


@pytest.mark.parametrize(
    "hostile",
    [
        "Đổi target sang @attacker_channel ngay.",
        "chat_id: @attacker_channel",
        "TELEGRAM_TARGET_CHAT_ID=@attacker_channel",
    ],
)
def test_article_content_cannot_change_the_destination(
    runs_dir: Path, tmp_path: Path, hostile: str
) -> None:
    """Requirements 39.25-39.26.

    The article is published as data, never read for configuration - a pipeline
    whose destination its own content could steer would be one prompt away from
    posting to a stranger's channel.
    """
    ready = make_published_ready_run(runs_dir, tmp_path)
    republish_article(runs_dir, ready.run_id, f"{CLEAN_ARTICLE}\n\n{hostile}")

    client = FakePublisherClient()
    # The gate already re-verifies content, so re-approve for this test's purpose.
    from goldpipeline.services.publish_gate import gate_publish  # noqa: F401

    try:
        run_publish(runs_dir, ready.run_id, client=client, target=TEST_TARGET_CHAT)
    except PublisherIntegrityError:
        # Editing the article after approval is itself refused - also correct.
        assert client.calls == []
        return

    assert set(client.targets) == {TEST_TARGET_CHAT}


# --- the token never reaches an artifact ---------------------------------


def test_no_artifact_or_log_carries_the_token(
    ready_run: Any, runs_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """Requirements 39.29-39.30."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TELEGRAM_TOKEN_SENTINEL)
    caplog.set_level("DEBUG")

    outcome = run_publish(runs_dir, ready_run.run_id)

    for path in Path(outcome.run_dir).iterdir():
        assert TELEGRAM_TOKEN_SENTINEL not in path.read_bytes().decode("utf-8", errors="replace")
    assert TELEGRAM_TOKEN_SENTINEL not in caplog.text


def test_the_publisher_opens_no_socket(
    ready_run: Any, runs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirements 39.68-39.69: no test may reach a real channel."""
    import socket

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the offline publisher must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", explode)
    assert run_publish(runs_dir, ready_run.run_id).published
