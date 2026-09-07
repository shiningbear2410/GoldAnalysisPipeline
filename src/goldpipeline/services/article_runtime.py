"""Which runtime an article type is written and reviewed by. One table.

Round 6.5c.2. Two article types now reach a provider, and they do different
things at the same two stages: an analysis asks a model for prose and audits it
afterwards; a digest computes its facts, asks a model for editorial content, and
assembles the article in code. Something has to choose between them.

**The point of this module is that it is the only place that chooses.** The
alternative - ``if article_type is NEWS_DIGEST`` wherever a difference happens -
starts as two branches and becomes a dozen scattered across the writer, the
reviewer, the finalizer and the gate, at which point nobody can answer "what
does a digest actually do differently?" without reading all of them.
:data:`RUNTIMES` answers it in one screen.

**Readiness lives here too, and it is two different questions.**
:mod:`~goldpipeline.services.article_routing` says whether a type has a *prompt*
- a product-registry fact. This says whether the pipeline can *run* one. Round
6.5c.1 found those had silently diverged: ``NEWS_DIGEST`` was registry-ready
while nothing dispatched it, and the orchestrator had to fail closed to stop a
digest prompt reaching the analysis writer. Keeping both facts in the same table
is what stops that gap reopening.

``revision_available`` is the third fact and the one this round deliberately
leaves ``False`` for a digest. A content ``NEEDS_REVISION`` on a digest has
nowhere to go - ``gold_finalizer_v2`` repairs an analysis, and pointing it at a
digest would rewrite a deterministic shell that no model is allowed to touch. So
the Run stops, visibly, until Round 6.5c.3 gives the digest a repair path of its
own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from goldpipeline.domain.errors import ArticleTypeNotReadyError
from goldpipeline.schemas.article import ArticleType


class WriteRuntime(StrEnum):
    """Which write implementation a Run uses."""

    ANALYSIS = "ANALYSIS"
    """`services.writer.write_draft`: one model call returning an article."""

    NEWS_DIGEST = "NEWS_DIGEST"
    """`services.digest_stage`: snapshot, editorial call, deterministic render."""


@dataclass(frozen=True)
class ArticleRuntime:
    """What the pipeline can do with one article type, end to end."""

    write: WriteRuntime
    dispatchable: bool
    """Whether the orchestrator can run this type at all.

    Distinct from being registry-ready. A type may have a prompt and a contract
    and still have no stage that knows how to use them, which is exactly the
    state ``NEWS_DIGEST`` was in between Rounds 6.5b and 6.5c.2.
    """

    revision_available: bool
    """Whether a content ``NEEDS_REVISION`` has a repair path.

    ``False`` means the Run stops rather than being handed to a finalizer built
    for a different product.
    """


RUNTIMES: Mapping[ArticleType, ArticleRuntime] = {
    ArticleType.ANALYSIS: ArticleRuntime(
        write=WriteRuntime.ANALYSIS,
        dispatchable=True,
        revision_available=True,
    ),
    ArticleType.NEWS_DIGEST: ArticleRuntime(
        write=WriteRuntime.NEWS_DIGEST,
        dispatchable=True,
        revision_available=False,
    ),
    ArticleType.TRADE_PLAN: ArticleRuntime(
        write=WriteRuntime.ANALYSIS,
        dispatchable=False,
        revision_available=False,
    ),
}
"""Every article type, and what the pipeline can do with it.

``TRADE_PLAN`` names a write runtime it will never use - the field is not
optional, and ``dispatchable=False`` is what actually governs. A rendered trade
plan has no writer at all; when it arrives it will bring its own runtime and
this row changes with it.
"""


def runtime_for(article_type: ArticleType) -> ArticleRuntime:
    """The runtime for *article_type*.

    Raises:
        ArticleTypeNotReadyError: The type cannot be dispatched. Raised rather
            than returning a default, because every default here is "write some
            other product", which is a silent wrong answer where a loud refusal
            belongs.
    """
    runtime = RUNTIMES.get(article_type)
    if runtime is None or not runtime.dispatchable:
        raise ArticleTypeNotReadyError(
            f"{article_type} has no runtime in this pipeline yet",
            article_type=str(article_type),
        )
    return runtime


def is_dispatchable(article_type: ArticleType) -> bool:
    """Whether the orchestrator can run this type. Never raises."""
    runtime = RUNTIMES.get(article_type)
    return runtime is not None and runtime.dispatchable


def revision_available(article_type: ArticleType) -> bool:
    """Whether a content revision can be performed for this type."""
    runtime = RUNTIMES.get(article_type)
    return runtime is not None and runtime.revision_available


__all__ = [
    "RUNTIMES",
    "ArticleRuntime",
    "WriteRuntime",
    "is_dispatchable",
    "revision_available",
    "runtime_for",
]
