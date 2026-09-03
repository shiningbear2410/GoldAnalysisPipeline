"""The ``producer-generate`` operator seam.

An entry point for testing and for manual use until a bot exists. It is not a
second producer: it parses a window, generates an id when none was given, and
hands both to :func:`goldpipeline.services.producer.produce`.

Every test is offline. The live collector is replaced, so no page is fetched and
no channel is contacted; the inbox is a temporary directory. Nothing here runs
against production.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from goldpipeline import cli
from goldpipeline.schemas.news import CollectionOutcome
from goldpipeline.services.inbox import INCOMING
from tests.test_producer import FakeCollector, make_collection, make_item


def stamp(minutes_ago: int) -> str:
    """A past instant, as the CLI would be given one.

    Derived from the real clock rather than hard-coded: the producer refuses a
    window that ends in the future, so a fixed literal would start failing on
    its own the day it went stale.
    """
    moment = datetime.now(UTC).replace(microsecond=0) - timedelta(minutes=minutes_ago)
    return moment.isoformat().replace("+00:00", "Z")


@pytest.fixture
def inbox_root(tmp_path: Path) -> Path:
    return tmp_path / "inbox"


@pytest.fixture(autouse=True)
def offline_collector(monkeypatch: pytest.MonkeyPatch) -> FakeCollector:
    """Replace the live collector so the CLI never reaches the network."""
    collector = FakeCollector()
    monkeypatch.setattr(cli, "LiveNewsCollector", lambda: collector)
    return collector


def run(*argv: str) -> int:
    return cli.main(list(argv))


def waiting(root: Path) -> list[Path]:
    return sorted((root / INCOMING).glob("*.json"))


def test_generate_submits_one_event(inbox_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = run("producer-generate", "--inbox-dir", str(inbox_root), "--request-id", "cli-000001")
    assert code == 0
    assert [p.name for p in waiting(inbox_root)] == ["internal_cli-000001.json"]
    assert "SUBMITTED" in capsys.readouterr().out


def test_generated_request_id_is_printed(
    inbox_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An operator whose terminal dies must be able to retry the same request."""
    assert run("producer-generate", "--inbox-dir", str(inbox_root)) == 0
    printed = capsys.readouterr().out
    submitted = waiting(inbox_root)[0].stem.removeprefix("internal_")
    assert submitted in printed


def test_generated_request_ids_differ(inbox_root: Path) -> None:
    run("producer-generate", "--inbox-dir", str(inbox_root))
    run("producer-generate", "--inbox-dir", str(inbox_root))
    assert len(waiting(inbox_root)) == 2


def test_repeating_the_whole_request_creates_no_second_event(
    inbox_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Identity is the id *and* the instant, and both are printed for this reason."""
    args = (
        "producer-generate",
        "--inbox-dir",
        str(inbox_root),
        "--request-id",
        "cli-000002",
        "--requested-at",
        stamp(30),
    )
    assert run(*args) == 0
    capsys.readouterr()
    assert run(*args) == 0
    assert "ALREADY_SUBMITTED" in capsys.readouterr().out
    assert len(waiting(inbox_root)) == 1


def test_repeating_only_the_id_is_a_conflict_not_a_second_event(
    inbox_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A moved window is a different question, and gets a refusal, not a Run."""
    base = ("producer-generate", "--inbox-dir", str(inbox_root), "--request-id", "cli-000009")
    assert run(*base, "--requested-at", stamp(120)) == 0
    capsys.readouterr()
    assert run(*base, "--requested-at", stamp(60)) == cli.EXIT_BLOCKED
    assert "CONFLICT" in capsys.readouterr().out
    assert len(waiting(inbox_root)) == 1


def test_the_requested_instant_is_printed(
    inbox_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    moment = stamp(45)
    run("producer-generate", "--inbox-dir", str(inbox_root), "--requested-at", moment)
    assert moment in capsys.readouterr().out


def test_a_naive_requested_at_is_refused(inbox_root: Path) -> None:
    with pytest.raises(SystemExit):
        run("producer-generate", "--inbox-dir", str(inbox_root), "--requested-at", "2026-09-03")


def test_conflicting_repeat_is_blocked_not_written(
    inbox_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = ("producer-generate", "--inbox-dir", str(inbox_root), "--request-id", "cli-000003")
    assert run(*args) == 0
    before = waiting(inbox_root)[0].read_bytes()

    monkeypatch.setattr(
        cli,
        "LiveNewsCollector",
        lambda: FakeCollector(collection=make_collection(items=[make_item(text="edited")])),
    )
    capsys.readouterr()
    assert run(*args) == cli.EXIT_BLOCKED
    assert "CONFLICT" in capsys.readouterr().out
    assert len(waiting(inbox_root)) == 1
    assert waiting(inbox_root)[0].read_bytes() == before


@pytest.mark.parametrize("lookback", ["6h", "12h", "24h", "48h", "72h", "7d"])
def test_supported_lookbacks(
    lookback: str, inbox_root: Path, offline_collector: FakeCollector
) -> None:
    code = run(
        "producer-generate",
        "--inbox-dir",
        str(inbox_root),
        "--lookback",
        lookback,
        "--request-id",
        f"cli-{lookback}-00001",
    )
    assert code == 0
    unit = {"h": 3600, "d": 86_400}[lookback[-1]]
    assert offline_collector.calls[0][1] == timedelta(seconds=int(lookback[:-1]) * unit)


@pytest.mark.parametrize("bad", ["24", "day", "24 hours", "-6h", "24y", ""])
def test_unparseable_lookback_is_refused(bad: str, inbox_root: Path) -> None:
    """A typo must not quietly collect a different window than the one asked for."""
    assert run("producer-generate", "--inbox-dir", str(inbox_root), "--lookback", bad) == (
        cli.EXIT_ERROR
    )
    assert not (inbox_root / INCOMING).exists() or waiting(inbox_root) == []


def test_out_of_range_lookback_is_refused(inbox_root: Path) -> None:
    assert run("producer-generate", "--inbox-dir", str(inbox_root), "--lookback", "30d") == (
        cli.EXIT_INVALID_DATA
    )
    assert waiting(inbox_root) == []


def test_invalid_request_id_is_refused(inbox_root: Path) -> None:
    assert (
        run("producer-generate", "--inbox-dir", str(inbox_root), "--request-id", "no")
        == cli.EXIT_INVALID_DATA
    )
    assert waiting(inbox_root) == []


def test_failed_collection_writes_nothing(
    inbox_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "LiveNewsCollector",
        lambda: FakeCollector(
            collection=make_collection(outcome=CollectionOutcome.FAILED, items=[])
        ),
    )
    assert run("producer-generate", "--inbox-dir", str(inbox_root)) == cli.EXIT_BLOCKED
    assert "NEWS_COLLECTION_FAILED" in capsys.readouterr().out
    assert waiting(inbox_root) == []


def test_json_output_carries_no_news_text(
    inbox_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secretish = "Fed vua cong bo mot dieu rat cu the va rieng biet"
    monkeypatch.setattr(
        cli,
        "LiveNewsCollector",
        lambda: FakeCollector(collection=make_collection(items=[make_item(text=secretish)])),
    )
    assert run("producer-generate", "--inbox-dir", str(inbox_root), "--json") == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["outcome"] == "SUBMITTED"
    assert payload["item_count"] == 1
    assert secretish not in out


def test_the_command_offers_no_provider_or_publish_options() -> None:
    """The seam generates events. It does not run the pipeline or send anything."""
    parser = cli.build_parser()
    for forbidden in (
        "--model",
        "--provider",
        "--publish",
        "--chat-id",
        "--target",
        "--fake-writer",
        "--auto-publish",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(["producer-generate", forbidden, "x"])


def test_the_command_creates_no_run(inbox_root: Path, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    run("producer-generate", "--inbox-dir", str(inbox_root))
    assert list(runs.iterdir()) == []
    assert list((inbox_root / "processed").iterdir()) == []


def test_producer_is_not_wired_into_the_worker_tick() -> None:
    """The producer is request-driven. The worker only consumes what is already there."""
    import ast

    tree = ast.parse(Path("src/goldpipeline/services/automation.py").read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    modules = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "produce" not in names
    assert "goldpipeline.services.producer" not in modules


def test_inbox_layout_is_created_before_submitting(inbox_root: Path) -> None:
    assert not inbox_root.exists()
    run("producer-generate", "--inbox-dir", str(inbox_root))
    for name in ("incoming", "processing", "processed", "failed", "index"):
        assert (inbox_root / name).is_dir()
