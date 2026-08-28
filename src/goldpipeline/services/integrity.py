"""Proving that a Run's artifacts are the ones the pipeline wrote.

Every stage after the first works from files an earlier stage produced. The
manifest records a SHA-256 for each, so "we read the real inputs" can be a
checked fact rather than an assumption.

This matters most at the reviewer: an auditor working from an article someone
edited by hand would produce a verdict about a document that never existed.
Verification happens before any provider is contacted, so a tampered Run costs
nothing and fails loudly.
"""

from __future__ import annotations

from dataclasses import dataclass

from goldpipeline.domain.errors import ArtifactIntegrityError
from goldpipeline.schemas.manifest import RunManifest
from goldpipeline.storage.atomic import sha256_bytes
from goldpipeline.storage.run_store import RunDirectory


@dataclass(frozen=True)
class VerifiedArtifact:
    """An artifact whose bytes match the digest recorded for it."""

    name: str
    payload: bytes
    sha256: str

    @property
    def text(self) -> str:
        """The artifact decoded as UTF-8."""
        return self.payload.decode("utf-8")


def verify_artifact(run: RunDirectory, manifest: RunManifest, filename: str) -> VerifiedArtifact:
    """Read *filename* and check it against the manifest.

    Raises:
        ArtifactIntegrityError: If the file is missing from the Run, missing
            from the manifest, or its bytes no longer match the recorded digest.
    """
    if not run.has_artifact(filename):
        raise ArtifactIntegrityError(
            f"run {run.run_id} has no {filename}", run_id=run.run_id, artifact=filename
        )

    payload = run.read_artifact_bytes(filename)
    digest = sha256_bytes(payload)

    recorded = next(
        (ref for ref in (*manifest.artifact_files, *manifest.source_files) if ref.name == filename),
        None,
    )
    if recorded is None:
        raise ArtifactIntegrityError(
            f"manifest for run {run.run_id} does not record {filename}",
            run_id=run.run_id,
            artifact=filename,
        )
    if recorded.sha256 != digest:
        raise ArtifactIntegrityError(
            f"{filename} has changed since it was written; refusing to use it",
            run_id=run.run_id,
            artifact=filename,
            expected_sha256=recorded.sha256,
            actual_sha256=digest,
        )
    return VerifiedArtifact(name=filename, payload=payload, sha256=digest)


def require_digest_match(
    *, label: str, expected: str, actual: str, run_id: str, **details: object
) -> None:
    """Assert two digests agree, or raise with both.

    Used for the cross-references one artifact holds about another - a writer
    result naming the digest of the draft it produced, for instance. Those pairs
    only stay meaningful if something checks them.
    """
    if expected != actual:
        raise ArtifactIntegrityError(
            f"{label} does not match; the two artifacts disagree",
            run_id=run_id,
            expected_sha256=expected,
            actual_sha256=actual,
            **details,
        )


__all__ = ["VerifiedArtifact", "require_digest_match", "verify_artifact"]
