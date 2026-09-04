"""The digest facts a Run owns, written once and never recomputed.

Round 6.5c.1. Round 6.5b built the deterministic facts in memory and handed
them straight to a writer; nothing survived the process. This is where they
become an artifact of the Run.

**Why a snapshot rather than a recomputation.** Every input a digest is built
from can move. A venue may revise a candle hours later. The clock crosses
midnight and a title dated "today" becomes a different date. The news window is
recoverable from the event, but only while the event is still to hand. Each of
those makes a resumed Run describe something subtly different from the one that
was reviewed, and the difference is invisible - the article still looks right.

So the facts are captured once, at the moment the Run is created, and every
later stage reads them. A resumed Run rebuilds nothing: not the window, not the
price arithmetic, not the rendered lines.

**Fail closed, never regenerate.** A snapshot that is missing where one is
required, or that does not match the digest the manifest recorded, stops the
Run. Silently rebuilding it would produce an article about a slightly different
six hours than the one a reviewer approved, which is exactly the failure the
artifact exists to prevent.
"""

from __future__ import annotations

import logging

from goldpipeline.domain.errors import ArtifactIntegrityError
from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.manifest import RunManifest
from goldpipeline.services.digest_context import DigestFacts
from goldpipeline.services.integrity import verify_artifact
from goldpipeline.storage.run_store import PreparedArtifact, RunDirectory

logger = logging.getLogger(__name__)

DIGEST_CONTEXT_FILENAME = "digest_context.json"
"""The Run's own copy of every deterministic digest fact.

Named beside ``context.json`` deliberately: it is the same kind of thing, for
the half of a digest that ``AnalysisContext`` has no room for.
"""


def requires_snapshot(article_type: ArticleType) -> bool:
    """Whether a Run of this type must carry a digest snapshot.

    Only ``NEWS_DIGEST``. An ANALYSIS Run has no digest window and no price
    reaction, and demanding an empty artifact of it would make every historical
    Run retroactively invalid.
    """
    return article_type is ArticleType.NEWS_DIGEST


def write_digest_snapshot(
    run: RunDirectory, manifest: RunManifest, facts: DigestFacts
) -> PreparedArtifact:
    """Commit the digest facts as an immutable Run artifact.

    Raises:
        ArtifactIntegrityError: The Run already has one. Runs are immutable and
            a second snapshot would mean two answers to "which six hours is
            this?" - so the write is refused rather than allowed to win.
    """
    if run.has_artifact(DIGEST_CONTEXT_FILENAME):
        raise ArtifactIntegrityError(
            f"run {run.run_id} already has {DIGEST_CONTEXT_FILENAME}; "
            "digest facts are captured once and never replaced",
            run_id=run.run_id,
            artifact=DIGEST_CONTEXT_FILENAME,
        )

    artifact = PreparedArtifact.from_json(DIGEST_CONTEXT_FILENAME, facts)
    run.commit_artifacts([artifact], manifest)
    logger.info(
        "digest.snapshot written run=%s window=%s..%s activity=%s items=%d sha=%s",
        run.run_id,
        facts.window.start.isoformat(),
        facts.window.end.isoformat(),
        facts.price_reaction.market_activity,
        len(facts.news_items),
        artifact.sha256[:12],
    )
    return artifact


def load_digest_snapshot(run: RunDirectory, manifest: RunManifest) -> DigestFacts:
    """Read the Run's digest facts back, proving they are the ones recorded.

    The digest check is the whole point of going through
    :func:`~goldpipeline.services.integrity.verify_artifact` rather than reading
    the file: a snapshot that has been edited since it was committed describes a
    window nobody reviewed.

    Raises:
        ArtifactIntegrityError: Missing, tampered with, or unreadable as digest
            facts. Every one of those stops the Run; none of them regenerates it.
    """
    artifact = verify_artifact(run, manifest, DIGEST_CONTEXT_FILENAME)

    try:
        facts = DigestFacts.model_validate_json(
            run.read_artifact_bytes(DIGEST_CONTEXT_FILENAME).decode("utf-8")
        )
    except Exception as exc:  # noqa: BLE001 - any parse fault is one integrity failure
        raise ArtifactIntegrityError(
            f"run {run.run_id}: {DIGEST_CONTEXT_FILENAME} is not readable as digest facts",
            run_id=run.run_id,
            artifact=DIGEST_CONTEXT_FILENAME,
        ) from exc

    logger.info(
        "digest.snapshot loaded run=%s window=%s..%s sha=%s",
        run.run_id,
        facts.window.start.isoformat(),
        facts.window.end.isoformat(),
        artifact.sha256[:12],
    )
    return facts


def require_digest_snapshot(
    run: RunDirectory, manifest: RunManifest, article_type: ArticleType
) -> DigestFacts | None:
    """The snapshot this Run must have, or ``None`` when it needs none.

    The article type decides. A ``NEWS_DIGEST`` Run without one cannot be
    continued: the window, the price arithmetic and the rendered lines are all
    in it, and reconstructing them would mean asking the clock and the provider
    again for answers that have since moved.

    Raises:
        ArtifactIntegrityError: A digest Run whose snapshot is absent or
            unusable.
    """
    if not requires_snapshot(article_type):
        return None
    return load_digest_snapshot(run, manifest)


__all__ = [
    "DIGEST_CONTEXT_FILENAME",
    "load_digest_snapshot",
    "require_digest_snapshot",
    "requires_snapshot",
    "write_digest_snapshot",
]
