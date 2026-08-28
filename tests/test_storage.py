"""Run directory creation, immutability and atomic writes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from goldpipeline.domain.errors import ArtifactAlreadyExistsError, RunAlreadyExistsError
from goldpipeline.schemas.manifest import RunManifest
from goldpipeline.storage.atomic import atomic_write_json, encode_json, sha256_bytes
from goldpipeline.storage.run_store import RunStore

RUN_ID = "20260828_022701_a83f2c"


def test_run_directory_is_created_correctly(runs_dir: Path) -> None:
    """Requirement 14.11."""
    store = RunStore(runs_dir)
    run = store.create(run_id=RUN_ID)

    assert run.run_id == RUN_ID
    assert run.path == runs_dir / RUN_ID
    assert run.path.is_dir()
    assert list(run.path.iterdir()) == []


def test_runs_root_is_created_on_demand(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "nested" / "runs")
    run = store.create()
    assert run.path.is_dir()


def test_existing_run_is_never_reused(runs_dir: Path) -> None:
    """Requirement 14.12: an explicit id that already exists is an error."""
    store = RunStore(runs_dir)
    store.create(run_id=RUN_ID)
    with pytest.raises(RunAlreadyExistsError):
        store.create(run_id=RUN_ID)


def test_generated_ids_route_around_collisions(runs_dir: Path) -> None:
    store = RunStore(runs_dir)
    ids = {store.create().run_id for _ in range(25)}
    assert len(ids) == 25
    assert len(list(runs_dir.iterdir())) == 25


def test_source_files_are_write_once(runs_dir: Path) -> None:
    """Requirement 14.12/14.15: a Run's inputs are immutable."""
    run = RunStore(runs_dir).create(run_id=RUN_ID)
    manifest = RunManifest(run_id=RUN_ID)

    run.write_source("ohlc.json", {"symbol": "XAUUSD"}, manifest)
    with pytest.raises(ArtifactAlreadyExistsError):
        run.write_source("ohlc.json", {"symbol": "TAMPERED"}, manifest)

    stored = json.loads((run.path / "ohlc.json").read_text(encoding="utf-8"))
    assert stored == {"symbol": "XAUUSD"}


def test_artifacts_are_write_once(runs_dir: Path) -> None:
    run = RunStore(runs_dir).create(run_id=RUN_ID)
    manifest = RunManifest(run_id=RUN_ID)
    run.write_artifact("context.json", {"run_id": RUN_ID}, manifest)
    with pytest.raises(ArtifactAlreadyExistsError):
        run.write_artifact("context.json", {"run_id": "other"}, manifest)


def test_manifest_is_the_only_rewritable_file(runs_dir: Path) -> None:
    run = RunStore(runs_dir).create(run_id=RUN_ID)
    manifest = RunManifest(run_id=RUN_ID)

    run.save_manifest(manifest)
    manifest.record_event("normalize", "OK")
    run.save_manifest(manifest)

    assert len(run.load_manifest().events) == 1


def test_artifact_names_cannot_escape_the_run_directory(runs_dir: Path) -> None:
    run = RunStore(runs_dir).create(run_id=RUN_ID)
    manifest = RunManifest(run_id=RUN_ID)
    for name in ["../escape.json", "sub/dir.json", ".."]:
        with pytest.raises(ValueError):
            run.write_artifact(name, {}, manifest)


def test_manifest_records_digest_and_size(runs_dir: Path) -> None:
    run = RunStore(runs_dir).create(run_id=RUN_ID)
    manifest = RunManifest(run_id=RUN_ID)
    payload = {"symbol": "XAUUSD"}

    ref = run.write_source("ohlc.json", payload, manifest)
    on_disk = (run.path / "ohlc.json").read_bytes()

    assert ref.sha256 == sha256_bytes(on_disk)
    assert ref.size_bytes == len(on_disk)
    assert manifest.source_files == [ref]


def test_json_is_utf8_without_ascii_escapes(tmp_path: Path) -> None:
    """Requirement 14.7: Vietnamese must be readable directly in the file."""
    target = tmp_path / "out.json"
    atomic_write_json(target, {"text": "Nhận định vàng phiên Á"})

    raw = target.read_bytes()
    assert b"\\u" not in raw
    assert "Nhận định vàng phiên Á" in raw.decode("utf-8")


def test_write_is_atomic_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A serialization or IO failure must leave no partial file behind."""
    target = tmp_path / "context.json"
    atomic_write_json(target, {"generation": 1})
    original = target.read_bytes()

    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("goldpipeline.storage.atomic.os.replace", explode)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_json(target, {"generation": 2})

    assert target.read_bytes() == original
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_encode_json_is_stable() -> None:
    payload = {"b": 1, "a": "vàng"}
    assert encode_json(payload) == encode_json(payload)
    assert encode_json(payload).endswith(b"\n")


def test_open_rejects_unknown_and_malformed_ids(runs_dir: Path) -> None:
    store = RunStore(runs_dir)
    with pytest.raises(ValueError):
        store.open("../../etc")
    with pytest.raises(FileNotFoundError):
        store.open(RUN_ID)


def test_list_run_ids_is_sorted(runs_dir: Path) -> None:
    store = RunStore(runs_dir)
    store.create(run_id="20260828_030000_bbbbbb")
    store.create(run_id="20260828_010000_aaaaaa")
    (runs_dir / "not-a-run").mkdir()
    assert store.list_run_ids() == ["20260828_010000_aaaaaa", "20260828_030000_bbbbbb"]
