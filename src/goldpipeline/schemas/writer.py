"""Writer stage contracts.

Two distinct models, and the split matters:

* :class:`WriterModelOutput` is what the *model* is allowed to author - the
  article and its supporting claims. Nothing else.
* :class:`WriterResult` is the artifact this pipeline *stamps* - run id, model,
  timestamps, usage, digests. A model can never influence those fields, because
  it never produces them.

The one field the model does echo is ``run_id``, and it is echoed precisely so
it can be checked. A mismatch means the response does not belong to this Run and
is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import Field, field_validator

from goldpipeline.schemas.common import StrictModel, UtcDatetime, utc_now

WRITER_SCHEMA_VERSION = "1.0.0"
"""Version of the writer artifact contract."""

MAX_TITLE_CHARS = 200
MAX_ARTICLE_CHARS = 20_000
MAX_CLAIMS = 60
MAX_WARNINGS = 40


@dataclass(frozen=True)
class WriterPrompt:
    """A rendered prompt: a system turn of rules and a user turn of data.

    Lives here rather than beside the builder because both the adapter layer
    (which sends it) and the service layer (which builds it) need the type, and
    schemas is the one layer both may depend on.

    The split between the two fields is the security boundary: ``system`` is
    built only from a versioned template on disk, ``user`` is the only place
    source data ever appears.
    """

    system: str
    user: str
    prompt_version: str
    nonce: str

    @property
    def sections(self) -> tuple[str, ...]:
        """Upper-case headings across both turns, for assertions and debugging."""
        combined = "\n".join((self.system, self.user))
        return tuple(
            line for line in combined.splitlines() if line.startswith("# ") and line.isupper()
        )


class WriterStatus(StrEnum):
    """Outcome the model reports for its own attempt."""

    COMPLETED = "COMPLETED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ClaimType(StrEnum):
    """What kind of statement a claim backs.

    The distinction the reviewer in Round 3 needs: a ``PRICE`` is checkable
    against the OHLC series, a ``SOURCE_OPINION`` is only attributable to the
    human analyst, and a ``DERIVED`` observation is arithmetic over the bars.
    """

    PRICE = "PRICE"
    TIME = "TIME"
    MARKET_META = "MARKET_META"
    DERIVED = "DERIVED"
    SOURCE_OPINION = "SOURCE_OPINION"


class WarningCode(StrEnum):
    """Closed set of writer-stage warnings.

    Closed on purpose: a downstream stage should be able to branch on these, and
    a free-text code from a language model is not something you can branch on.
    """

    SOURCE_PRICE_OUT_OF_RANGE = "SOURCE_PRICE_OUT_OF_RANGE"
    SOURCE_CONTRADICTS_MARKET = "SOURCE_CONTRADICTS_MARKET"
    SOURCE_CONTAINS_INSTRUCTIONS = "SOURCE_CONTAINS_INSTRUCTIONS"
    MISSING_DATA_OMITTED = "MISSING_DATA_OMITTED"
    DEGRADED_INPUT_QUALITY = "DEGRADED_INPUT_QUALITY"
    SOURCE_TOO_THIN = "SOURCE_TOO_THIN"


class SourceClaim(StrictModel):
    """One factual statement the article makes, and where it came from.

    Round 3 uses these to answer a single question: which number did the writer
    use, and from which field of ``context.json`` did it come?
    """

    type: ClaimType
    value: str = Field(
        min_length=1,
        max_length=400,
        description="The value as used in the article, e.g. '3314.20'.",
    )
    source: str = Field(
        min_length=1,
        max_length=200,
        description="Dotted path into the context, e.g. 'context.price.latest_close'.",
    )
    note: str | None = Field(default=None, max_length=400, description="Optional clarification.")


MAX_NEWS_CLAIMS = 20
MAX_EXCERPT_CHARS = 400


class NewsClaim(StrictModel):
    """One external-news statement in the article, and the item that supports it.

    **Deliberately not a** :class:`SourceClaim`. Those address a dotted path into
    ``context.json`` - a value this pipeline computed from closed candles, which
    :mod:`goldpipeline.services.claim_resolver` resolves and compares exactly.
    A news claim addresses a message somebody else published, recovered from the
    producer brief, and the strongest thing that can be said about it is that the
    article did not invent it. Two different kinds of evidence with two different
    strengths; merging them into one list would let the weaker sort borrow the
    authority of the stronger.

    All three fields are excerpts rather than summaries, because every check a
    later stage makes is a substring check. A paraphrase would have to be judged,
    and nothing deterministic can judge a paraphrase.
    """

    statement: str = Field(
        min_length=1,
        max_length=MAX_EXCERPT_CHARS,
        description="The article's own words asserting the fact, copied exactly from it.",
    )
    evidence: str = Field(
        min_length=1,
        max_length=MAX_EXCERPT_CHARS,
        description="The words from the cited news item that support it, copied exactly.",
    )
    news_item_ids: list[str] = Field(
        min_length=1,
        max_length=8,
        description="Ids from the producer brief, '<channel>:<message_id>'. Never a URL.",
    )


class WriterWarning(StrictModel):
    """Something the writer noticed and chose not to paper over."""

    code: WarningCode
    message: str = Field(min_length=1, max_length=600)


class WriterModelOutput(StrictModel):
    """The structured response a writer model must return.

    Deliberately small. Every field here is content the model is entitled to
    author; identity and provenance fields are stamped by the pipeline.
    """

    run_id: str = Field(
        min_length=1,
        max_length=64,
        description="Echo of the run id from the context, checked against the real one.",
    )
    status: WriterStatus
    title: str = Field(max_length=MAX_TITLE_CHARS)
    article: str = Field(max_length=MAX_ARTICLE_CHARS)
    source_claims: list[SourceClaim] = Field(default_factory=list, max_length=MAX_CLAIMS)
    news_claims: list[NewsClaim] = Field(
        default_factory=list,
        max_length=MAX_NEWS_CLAIMS,
        description=(
            "External-news statements and the producer-brief items behind them. "
            "Empty unless the Run was fed by the internal producer."
        ),
    )
    warnings: list[WriterWarning] = Field(default_factory=list, max_length=MAX_WARNINGS)

    @field_validator("title", "article")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        """An article of whitespace is an empty article, not a short one."""
        if not value.strip():
            raise ValueError("must not be empty or whitespace only")
        return value.strip()


class WriterUsage(StrictModel):
    """Provider usage metadata, kept for cost accounting.

    Only counts and opaque identifiers. No headers, no credentials, no echo of
    the request body.
    """

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    request_id: str | None = Field(default=None, max_length=200)
    stop_reason: str | None = Field(default=None, max_length=64)


class WriterResult(StrictModel):
    """The ``claude_writer.json`` artifact.

    ``article`` itself is **not** stored here - it lives in ``claude_draft.md``.
    Instead this carries ``article_sha256`` and ``article_chars``, which bind the
    two artifacts together: if either file is altered, the pair stops agreeing
    and the tampering is detectable without re-reading the whole article.
    """

    schema_version: str = Field(default=WRITER_SCHEMA_VERSION)
    run_id: str
    status: WriterStatus
    stage: str = Field(default="claude_writer")
    title: str
    model: str = Field(description="Model id that produced the draft.")
    provider: str = Field(description="Which client produced it, e.g. 'anthropic', 'fake'.")
    prompt_version: str = Field(description="Version of the prompt template used.")
    context_sha256: str = Field(
        min_length=64,
        max_length=64,
        description="Digest of the context.json this draft was written from.",
    )
    draft_file: str = Field(description="Name of the markdown artifact holding the article.")
    article_sha256: str = Field(min_length=64, max_length=64)
    article_chars: int = Field(ge=1)
    created_at: UtcDatetime = Field(default_factory=utc_now)
    source_claims: list[SourceClaim] = Field(default_factory=list)
    news_claims: list[NewsClaim] = Field(default_factory=list)
    warnings: list[WriterWarning] = Field(default_factory=list)
    usage: WriterUsage = Field(default_factory=WriterUsage)


__all__ = [
    "MAX_ARTICLE_CHARS",
    "MAX_EXCERPT_CHARS",
    "MAX_NEWS_CLAIMS",
    "MAX_TITLE_CHARS",
    "WRITER_SCHEMA_VERSION",
    "ClaimType",
    "NewsClaim",
    "SourceClaim",
    "WarningCode",
    "WriterModelOutput",
    "WriterPrompt",
    "WriterResult",
    "WriterStatus",
    "WriterUsage",
    "WriterWarning",
]
