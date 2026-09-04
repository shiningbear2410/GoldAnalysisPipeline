"""What a model may author in a news digest, and nothing else.

Round 6.5b. The digest is assembled by code from two halves.

The **deterministic half** - the title, the exact window, the price arithmetic,
the disclaimer - was built in Round 6.5a and is rendered from typed facts. The
model never sees a field it could write those into.

The **editorial half** is here: which of the collected items mattered, how each
one reads in a line, which way it leans for gold, and one short paragraph on the
balance of the whole window. Those are judgements, and a model is the right
thing to make them.

**The split is the safety property.** Round 6.4e asked a model to copy a
deterministic date into an article and then checked that it had; that works, but
it spends a check on a problem that need not exist. Here the model is not given
the opportunity: there is no ``article`` field, no ``title`` field and no place
to put a price. A digest whose numbers are wrong cannot be produced by an
editorial mistake, only by a bug in arithmetic that is tested directly.

**Items are chosen from a closed list.** ``news_item_id`` must name an item the
prompt actually offered. The timestamp a reader sees is looked up from that
item rather than repeated by the model, so a digest cannot misdate an event
even if the model would have.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, field_validator

from goldpipeline.schemas.common import StrictModel, UtcDatetime
from goldpipeline.schemas.writer import NewsClaim, WriterStatus, WriterWarning

NEWS_DIGEST_SCHEMA_VERSION = "1.0.0"

MIN_DIGEST_ITEMS = 1
MAX_DIGEST_ITEMS = 6
"""One to six.

Six is the reading limit for a digest meant to take a minute on a phone. One is
the floor rather than three because a quiet window with a single material story
should report one story - padding to three means inventing importance, which is
the failure this product can least afford.
"""

MAX_HEADLINE_CHARS = 160
MAX_NOTE_CHARS = 200
MAX_BALANCE_CHARS = 400
MAX_DIGEST_CLAIMS = 30


class ImpactMarker(StrEnum):
    """Which way one news item leans for gold.

    A closed vocabulary of three, and closed is the point: an open one gives a
    model somewhere to invent a fourth category and a new emoji, and the reader
    then has to learn a symbol that means whatever that day's model thought.

    This is an *editorial reading of the news*, not an observation of price.
    A dovish Fed line is ``SUPPORTS_GOLD`` whether or not gold rose afterwards,
    and the digest reports both without connecting them - see
    :mod:`goldpipeline.services.digest_render` for why the price block has no
    vocabulary for causes.
    """

    SUPPORTS_GOLD = "SUPPORTS_GOLD"
    PRESSURES_GOLD = "PRESSURES_GOLD"
    MIXED_OR_UNCLEAR = "MIXED_OR_UNCLEAR"
    """Genuinely two-sided, or too early to read. Not a place to put laziness."""


IMPACT_LABELS: dict[ImpactMarker, str] = {
    ImpactMarker.SUPPORTS_GOLD: "🟢 Hỗ trợ vàng",
    ImpactMarker.PRESSURES_GOLD: "🔴 Gây áp lực lên vàng",
    ImpactMarker.MIXED_OR_UNCLEAR: "🟠 Hai chiều / chưa rõ",
}
"""The exact published wording for each marker.

Held here, rendered by code, and never written by the model. Three fixed
phrases mean a regular reader learns them once; three phrasings a day means
they read the words every time instead of the colour.
"""


class DigestSourceItem(StrictModel):
    """One curated item the writer may choose from.

    Supplied *to* the model and read back *by* the renderer, which is what
    makes the timestamp trustworthy: the model returns an id, and the time a
    reader sees is looked up from the item the producer actually collected.
    """

    item_id: str = Field(min_length=1, max_length=200)
    published_at: UtcDatetime = Field(description="When the item was posted, UTC.")
    text: str = Field(min_length=1, description="Normalized item text. UNTRUSTED.")


class DigestItem(StrictModel):
    """One item the writer chose, as it will appear.

    Note what is absent: a timestamp, a date, any price, and any field the
    renderer needs for structure. The model supplies the *judgement* - this one
    mattered, here is how it reads, here is which way it leans - and the
    pipeline supplies everything checkable.
    """

    news_item_id: str = Field(
        min_length=1,
        max_length=200,
        description="Must name an item the prompt offered. Never a URL, never invented.",
    )
    headline: str = Field(
        min_length=1,
        max_length=MAX_HEADLINE_CHARS,
        description="One line: what happened. Compressed from the item, never beyond it.",
    )
    note: str | None = Field(
        default=None,
        max_length=MAX_NOTE_CHARS,
        description="Optional second line: why it matters. Omitted when it would restate.",
    )
    impact: ImpactMarker

    @field_validator("headline")
    @classmethod
    def _headline_is_one_line(cls, value: str) -> str:
        """A headline that wraps is a paragraph, and the renderer owns layout."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("a digest item needs a headline")
        if "\n" in cleaned:
            raise ValueError("a headline is a single line")
        return cleaned

    @field_validator("note")
    @classmethod
    def _note_is_prose_or_absent(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class DigestEditorial(StrictModel):
    """The structured response a news-digest writer must return.

    No ``article``, no ``title``, no ``window``, no price. The digest's facts
    are not this model's to state, and the schema is where that is enforced -
    ``extra="forbid"`` means a model that invents an ``article`` field has its
    whole answer rejected rather than quietly overriding the renderer.
    """

    run_id: str = Field(min_length=1, max_length=64, description="Echo of the run id.")
    status: WriterStatus
    items: list[DigestItem] = Field(
        min_length=MIN_DIGEST_ITEMS,
        max_length=MAX_DIGEST_ITEMS,
        description="The material items, in the order they should be read.",
    )
    balance: str = Field(
        min_length=1,
        max_length=MAX_BALANCE_CHARS,
        description="🧭 Cán cân: where the news of this window leans, in a sentence or three.",
    )
    news_claims: list[NewsClaim] = Field(default_factory=list, max_length=MAX_DIGEST_CLAIMS)
    warnings: list[WriterWarning] = Field(default_factory=list, max_length=MAX_DIGEST_CLAIMS)

    @field_validator("balance")
    @classmethod
    def _balance_says_something(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("the balance section must say something")
        return cleaned

    @field_validator("items")
    @classmethod
    def _one_line_per_story(cls, value: list[DigestItem]) -> list[DigestItem]:
        """The same item may not be selected twice.

        Two entries citing one id is how a single Fed headline becomes three
        bullet points that look like three events.
        """
        seen = [item.news_item_id for item in value]
        duplicates = sorted({item for item in seen if seen.count(item) > 1})
        if duplicates:
            raise ValueError(f"each item may be selected once; repeated: {duplicates}")
        return value


__all__ = [
    "IMPACT_LABELS",
    "MAX_BALANCE_CHARS",
    "MAX_DIGEST_ITEMS",
    "MAX_HEADLINE_CHARS",
    "MAX_NOTE_CHARS",
    "MIN_DIGEST_ITEMS",
    "NEWS_DIGEST_SCHEMA_VERSION",
    "DigestEditorial",
    "DigestItem",
    "DigestSourceItem",
    "ImpactMarker",
]
