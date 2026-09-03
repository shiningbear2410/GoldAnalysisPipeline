"""What was checked about an article's news statements, written down.

The vocabulary lives here rather than beside the verifier because two layers
need it: the service that decides, and the publish decision that records what
was decided. Schemas is the layer both may depend on.

Everything here is an audit record. It answers, for an operator reading a
blocked - or an approved - decision months later: which sentence claimed a news
source, which item it named, and whether that held up. No news text is copied
into it beyond the excerpts the writer itself chose, and no secret can reach it.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from goldpipeline.schemas.common import StrictModel

PROVENANCE_SCHEMA_VERSION = "1.0.0"

NEWS_PROVENANCE_VERSION = "news_provenance_v1"
"""Which ruleset judged a Run's news claims.

Versioned for the same reason the gate is: an approval recorded last month was
made under the rules of last month, and a reader has to be able to tell.
"""

MAX_AUDIT_EXCERPT_CHARS = 400


class ProvenanceState(StrEnum):
    """Whether a Run has a news source at all."""

    AVAILABLE = "AVAILABLE"
    """The input is an authentic, fully parsed producer brief."""

    NOT_PRODUCER = "NOT_PRODUCER"
    """An ordinary analyst note. No news provenance exists, and none is claimed."""

    UNPARSEABLE_BRIEF = "UNPARSEABLE_BRIEF"
    """Submitted under the producer's name, but not a brief this code can read.

    Fails closed: a half-understood document would hand the verifier item
    records of unknown origin, every one usable to justify a statement.
    """


class ClaimVerdict(StrEnum):
    """What became of one declared news claim."""

    SUPPORTED = "SUPPORTED"

    NO_PROVENANCE = "NO_PROVENANCE"
    """Claimed on a Run with no authentic brief behind it."""

    MALFORMED_ID = "MALFORMED_ID"
    """Not a ``<channel>:<message_id>`` id - a URL, say, or a bare channel name."""

    UNKNOWN_ITEM = "UNKNOWN_ITEM"
    """A well-formed id naming no item in this brief."""

    EVIDENCE_NOT_IN_ITEM = "EVIDENCE_NOT_IN_ITEM"
    """The quoted evidence appears in none of the cited items."""

    STATEMENT_NOT_IN_ARTICLE = "STATEMENT_NOT_IN_ARTICLE"
    """The quoted statement is not in the final article.

    The ordinary cause is a finalizer rewrite, and it is why verification runs
    against the published bytes rather than the draft: provenance describes a
    sentence, and a sentence that changed no longer has it.
    """


class NewsClaimAudit(StrictModel):
    """One claim as judged, for a human reading the decision."""

    statement: str = Field(max_length=MAX_AUDIT_EXCERPT_CHARS)
    news_item_ids: list[str] = Field(default_factory=list)
    verdict: ClaimVerdict
    supporting_item_id: str | None = Field(
        default=None, description="The cited item whose text actually carried the evidence."
    )
    article_spans: int = Field(
        default=0, ge=0, description="How many places in the final article this statement covers."
    )
    detail: str | None = Field(
        default=None,
        max_length=400,
        description="Why it was refused, in words an operator can act on.",
    )


class NewsProvenanceReport(StrictModel):
    """The whole news-provenance finding for one Run."""

    schema_version: str = Field(default=PROVENANCE_SCHEMA_VERSION)
    version: str = Field(default=NEWS_PROVENANCE_VERSION)
    state: ProvenanceState
    brief_version: str | None = None
    item_count: int = Field(default=0, ge=0, description="Items the brief offered to cite.")
    claims: list[NewsClaimAudit] = Field(default_factory=list)

    @property
    def supported_count(self) -> int:
        return sum(1 for claim in self.claims if claim.verdict is ClaimVerdict.SUPPORTED)


__all__ = [
    "MAX_AUDIT_EXCERPT_CHARS",
    "NEWS_PROVENANCE_VERSION",
    "PROVENANCE_SCHEMA_VERSION",
    "ClaimVerdict",
    "NewsClaimAudit",
    "NewsProvenanceReport",
    "ProvenanceState",
]
