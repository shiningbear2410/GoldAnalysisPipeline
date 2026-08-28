"""Run directory lifecycle.

A Run directory is append-only::

    runs/20260828_022701_a83f2c/
        manifest.json        <- ledger, rewritten on each stage transition
        telegram_input.json  <- write-once source
        ohlc.json            <- write-once source
        context.json         <- write-once artifact

Later rounds add ``claude_draft.md``, ``gpt_review.json`` and friends to the
same directory. They must go through :meth:`RunDirectory.write_artifact`, which
refuses to clobber an existing file.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from goldpipeline.domain.errors import ArtifactAlreadyExistsError, RunAlreadyExistsError
from goldpipeline.domain.run_id import generate_run_id, is_valid_run_id
from goldpipeline.schemas.manifest import ArtifactRef, RunManifest, RunStatus
from goldpipeline.storage.atomic import (
    atomic_write_json,
    encode_json,
    encode_text,
    read_json,
    sha256_bytes,
)

MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class PreparedArtifact:
    """An artifact serialized in memory, ready to be committed.

    Serialization happens before anything touches the filesystem, so a
    serialization failure cannot leave a half-written Run behind.
    """

    name: str
    payload: bytes

    @classmethod
    def from_json(cls, name: str, data: Any) -> PreparedArtifact:
        """Serialize *data* as UTF-8 JSON."""
        return cls(name=name, payload=encode_json(data))

    @classmethod
    def from_text(cls, name: str, text: str) -> PreparedArtifact:
        """Serialize *text* as UTF-8 with a single trailing newline."""
        return cls(name=name, payload=encode_text(text))

    @classmethod
    def from_bytes(cls, name: str, payload: bytes) -> PreparedArtifact:
        """Carry bytes through untouched.

        For copying one artifact to another verbatim. :meth:`from_text`
        normalizes the trailing newline, which is right when serializing a
        string but wrong when the requirement is that two files be identical
        byte for byte.
        """
        return cls(name=name, payload=payload)

    @property
    def sha256(self) -> str:
        """Digest of the bytes that will be written."""
        return sha256_bytes(self.payload)


@dataclass(frozen=True)
class RunDirectory:
    """Handle to one Run on disk."""

    run_id: str
    path: Path

    # -- artifacts ---------------------------------------------------------

    def _target(self, filename: str) -> Path:
        if "/" in filename or "\\" in filename or filename in {".", ".."}:
            raise ValueError(f"artifact name must be a plain file name: {filename!r}")
        return self.path / filename

    def write_source(self, filename: str, data: Any, manifest: RunManifest) -> ArtifactRef:
        """Persist an immutable *source* file and register it in *manifest*."""
        ref = self._write_once(filename, data)
        manifest.source_files.append(ref)
        return ref

    def write_artifact(self, filename: str, data: Any, manifest: RunManifest) -> ArtifactRef:
        """Persist an immutable *derived* file and register it in *manifest*."""
        ref = self._write_once(filename, data)
        manifest.artifact_files.append(ref)
        return ref

    def _write_once(self, filename: str, data: Any) -> ArtifactRef:
        target = self._target(filename)
        if target.exists():
            raise ArtifactAlreadyExistsError(
                f"refusing to overwrite existing artifact {filename!r} in run {self.run_id}",
                run_id=self.run_id,
                filename=filename,
            )
        digest, size = atomic_write_json(target, data)
        return ArtifactRef(name=filename, sha256=digest, size_bytes=size)

    # -- multi-file commit -------------------------------------------------

    def commit_artifacts(
        self, artifacts: list[PreparedArtifact], manifest: RunManifest
    ) -> list[ArtifactRef]:
        """Write several artifacts as one all-or-nothing unit.

        A stage that produces two files - a draft and its metadata - must not be
        able to leave one of them behind. ``os.replace`` is atomic per file but
        says nothing about a pair, so this method adds the missing guarantee:

        1. refuse if any target already exists;
        2. write every file to a temporary sibling and fsync it;
        3. rename them into place;
        4. if any rename fails, remove the temporaries **and** the ones already
           renamed, restoring the directory to its previous contents.

        The caller updates the manifest only after this returns, so a manifest
        that says a stage completed always describes files that exist.

        Raises:
            ArtifactAlreadyExistsError: If any target is already present.
        """
        targets = [(artifact, self._target(artifact.name)) for artifact in artifacts]

        existing = [artifact.name for artifact, path in targets if path.exists()]
        if existing:
            raise ArtifactAlreadyExistsError(
                f"refusing to overwrite existing artifacts in run {self.run_id}: {existing}",
                run_id=self.run_id,
                filenames=existing,
            )

        staged: list[tuple[Path, Path]] = []
        committed: list[Path] = []
        try:
            for artifact, target in targets:
                staged.append((self._stage(artifact, target), target))
            for temporary, target in staged:
                os.replace(temporary, target)
                committed.append(target)
        except BaseException:
            for temporary, _ in staged:
                temporary.unlink(missing_ok=True)
            for target in committed:
                target.unlink(missing_ok=True)
            raise

        refs = [
            ArtifactRef(
                name=artifact.name,
                sha256=sha256_bytes(artifact.payload),
                size_bytes=len(artifact.payload),
            )
            for artifact, _ in targets
        ]
        manifest.artifact_files.extend(refs)
        return refs

    @staticmethod
    def _stage(artifact: PreparedArtifact, target: Path) -> Path:
        """Write *artifact* to a durable temporary file beside *target*."""
        handle, name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        temporary = Path(name)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(artifact.payload)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return temporary

    # -- reading -----------------------------------------------------------

    def read_artifact_bytes(self, filename: str) -> bytes:
        """Read one artifact's raw bytes."""
        return self._target(filename).read_bytes()

    def has_artifact(self, filename: str) -> bool:
        """Whether *filename* exists in this Run."""
        return self._target(filename).is_file()

    # -- manifest ----------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        """Path of this Run's manifest."""
        return self.path / MANIFEST_FILENAME

    def save_manifest(self, manifest: RunManifest) -> None:
        """Write the manifest.

        This is the one file in a Run that is legitimately rewritten - it is the
        ledger recording how the Run progressed.
        """
        atomic_write_json(self.manifest_path, manifest)

    def load_manifest(self) -> RunManifest:
        """Read the manifest back from disk."""
        return RunManifest.model_validate(read_json(self.manifest_path))

    def artifact_path(self, filename: str) -> Path:
        """Absolute path of *filename* inside this Run."""
        return self._target(filename)


class RunStore:
    """Creates and locates Run directories under a root."""

    def __init__(self, root: Path | str = Path("runs")) -> None:
        self.root = Path(root)

    def create(self, *, run_id: str | None = None, attempts: int = 5) -> RunDirectory:
        """Create a fresh Run directory.

        A pre-existing directory is never reused. When *run_id* is generated and
        collides (same second, same random suffix), a new id is drawn; when
        *run_id* was supplied explicitly, the collision is an error.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        explicit = run_id is not None

        for _ in range(attempts):
            candidate = run_id or generate_run_id()
            if not is_valid_run_id(candidate):
                raise ValueError(f"invalid run id: {candidate!r}")
            target = self.root / candidate
            try:
                target.mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                if explicit:
                    raise RunAlreadyExistsError(
                        f"run directory already exists: {target}", run_id=candidate
                    ) from None
                continue
            return RunDirectory(run_id=candidate, path=target)

        raise RunAlreadyExistsError(
            f"could not allocate a unique run id after {attempts} attempts",
            attempts=attempts,
        )

    def open(self, run_id: str) -> RunDirectory:
        """Return a handle to an existing Run.

        Raises:
            FileNotFoundError: If the Run directory does not exist.
        """
        if not is_valid_run_id(run_id):
            raise ValueError(f"invalid run id: {run_id!r}")
        target = self.root / run_id
        if not target.is_dir():
            raise FileNotFoundError(f"no such run: {target}")
        return RunDirectory(run_id=run_id, path=target)

    def list_run_ids(self) -> list[str]:
        """All Run ids under the root, oldest first."""
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and is_valid_run_id(p.name))


__all__ = [
    "MANIFEST_FILENAME",
    "PreparedArtifact",
    "RunDirectory",
    "RunStatus",
    "RunStore",
]
