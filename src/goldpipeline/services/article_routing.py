"""Which article type runs, and with which prompt.

One table, consulted everywhere. The alternative - ``if article_type ==`` spread
through the writer, the orchestrator and the CLI - is how a mode ends up
executable in one path and refused in another, which for an unfinished mode
means it runs.

**The prompt is chosen here, never carried by a payload.** A producer names a
product mode from a closed enum; this module turns that into a prompt id from
application code. No event field reaches ``load_prompt``, so no producer can
select a template, a path, or anything that is not on this table.

**Declared before executable, on purpose.** ``TRADE_PLAN`` and ``NEWS_DIGEST``
have entries and are marked not ready. They fail closed at the boundary and
never reach a provider. That way the day they arrive is a single change to this
table plus the code it points at, rather than a schema migration performed under
pressure.
"""

from __future__ import annotations

from dataclasses import dataclass

from goldpipeline.domain.errors import ArticleTypeNotReadyError
from goldpipeline.prompts import DEFAULT_WRITER_PROMPT, GOLD_NEWS_DIGEST_WRITER_V1
from goldpipeline.schemas.article import ArticleType


@dataclass(frozen=True)
class ArticleTypeSpec:
    """How one article type is produced, or why it cannot be."""

    article_type: ArticleType
    ready: bool
    prompt_id: str | None
    """The writer prompt for this mode, or ``None`` while it has none.

    Deliberately optional rather than a placeholder file. A stub prompt that
    exists but is not fit to use is indistinguishable from a real one at the
    call site, and the first thing that would happen is somebody wiring it up.
    """

    requires: str
    """What is still missing, in words an operator can act on."""


SPECS: dict[ArticleType, ArticleTypeSpec] = {
    ArticleType.ANALYSIS: ArticleTypeSpec(
        article_type=ArticleType.ANALYSIS,
        ready=True,
        # The prompt production has been running on. Deliberately not a new
        # `gold_analysis_v1`: this round moves routing, not prose, and changing
        # the article the pipeline writes while changing how it is selected
        # would make a regression impossible to attribute.
        prompt_id=DEFAULT_WRITER_PROMPT,
        requires="",
    ),
    ArticleType.TRADE_PLAN: ArticleTypeSpec(
        article_type=ArticleType.TRADE_PLAN,
        ready=False,
        prompt_id=None,
        requires=(
            "a deterministic engine for entry, invalidation and targets. "
            "context.levels holds candidate technical zones, which is not a trade plan"
        ),
    ),
    ArticleType.NEWS_DIGEST: ArticleTypeSpec(
        article_type=ArticleType.NEWS_DIGEST,
        ready=True,
        # Its own prompt, not the analysis writer's. The two ask for different
        # shapes of answer: an article, versus the editorial judgements a digest
        # needs around facts the pipeline computes itself.
        prompt_id=GOLD_NEWS_DIGEST_WRITER_V1,
        requires="",
    ),
}
"""Every article type, including the ones that will not run.

Complete by construction - a test asserts the table covers the enum - so a mode
added to the enum without an entry here fails loudly rather than falling through
to whatever the lookup happens to return.
"""


def spec_for(article_type: ArticleType) -> ArticleTypeSpec:
    """The routing entry for *article_type*."""
    return SPECS[article_type]


def require_ready(article_type: ArticleType) -> ArticleTypeSpec:
    """Return the spec, or refuse if the mode is not implemented.

    Raises:
        ArticleTypeNotReadyError: The type is valid but has no implementation.
            Never a fallback to ``ANALYSIS`` - writing a different kind of
            article than the one requested is a silent substitution, and the
            whole point of refusing is that somebody notices.
    """
    spec = spec_for(article_type)
    if not spec.ready or spec.prompt_id is None:
        raise ArticleTypeNotReadyError(
            f"article type {article_type} is not implemented yet; it requires {spec.requires}",
            article_type=str(article_type),
        )
    return spec


def writer_prompt_for(article_type: ArticleType) -> str:
    """The prompt id for a runnable article type."""
    spec = require_ready(article_type)
    assert spec.prompt_id is not None  # noqa: S101 - guaranteed by require_ready
    return spec.prompt_id


READY_TYPES = frozenset(t for t, spec in SPECS.items() if spec.ready)
"""Types that can actually be produced today."""

REMOTE_ALLOWED_TYPES = frozenset({ArticleType.ANALYSIS})
"""What a *remote* producer may ask for.

Narrower than :data:`READY_TYPES` and narrower on purpose. A local producer is
this machine; a remote one is a machine on the other end of a network, and the
modes it may select should be the smallest set that is useful. Enforced at the
intake boundary, which is the only layer that knows the transport.
"""


__all__ = [
    "READY_TYPES",
    "REMOTE_ALLOWED_TYPES",
    "SPECS",
    "ArticleTypeSpec",
    "require_ready",
    "spec_for",
    "writer_prompt_for",
]
