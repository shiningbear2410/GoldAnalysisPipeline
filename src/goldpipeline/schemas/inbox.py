"""The handoff contract between the analysis producer and this pipeline.

**Why a file drop and not a Telegram reader.** The Telegram Bot API does not
deliver messages authored by other bots. A second bot watching the channel where
the existing Gold Analysis Bot posts would receive nothing - not intermittently,
not with the right permissions, but never. So the producer hands its output over
directly, by writing one JSON file into a directory this pipeline watches.

**The payload is a whitelist, and it is data.** ``extra="forbid"`` is doing real
work here: it means a producer cannot smuggle a ``runs_dir``, a ``model``, a
``target_chat`` or a ``publish`` flag into the pipeline by adding a key.
``raw_text`` in particular is untrusted content and travels as such all the way
to the writer's fenced prompt.

**One field does influence behaviour, and it is worth being exact about it.**
Until article types existed, nothing in this schema changed what the pipeline
*did*. ``article_type`` changes that, so the invariant is now narrower rather
than quietly false:

    A producer may select one value from a closed, code-defined enum of product
    modes. It may not select anything else.

Concretely, no field here can name a prompt, a prompt id, a model, a provider,
a reviewer, a finalizer, a Telegram destination, a publish behaviour, a
filesystem path, or a pipeline stage. ``article_type`` chooses among three
modes whose implementations are chosen by
:mod:`goldpipeline.services.article_routing`, in application code, from a table
a producer cannot reach. The worst a hostile producer can do with it is ask for
a mode that is refused.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import Field, field_validator

from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.common import StrictModel, UtcDatetime
from goldpipeline.schemas.telegram import MAX_RAW_TEXT_CHARS

INBOX_SCHEMA_VERSION = "1"
"""Version of the payload contract a producer writes."""

EVENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$")
"""What an ``event_id`` may look like.

Deliberately narrow, because this value becomes a file name in the ingestion
ledger. No separators, no ``:`` (invalid on Windows), no leading dot, nothing
that could climb out of the directory it is written into. A producer that wants
structure in its ids can use dots or dashes.
"""


class AnalysisEvent(StrictModel):
    """One analysis handed over by the producing bot.

    Only ``source``, ``event_id``, ``created_at`` and ``raw_text`` are required.
    Everything else is optional and, when absent, stays absent - Round 1's rule
    that missing metadata is never invented applies here too.
    """

    schema_version: Literal["1"] = Field(
        default="1",
        description="Payload contract version. An unknown version is refused, not guessed.",
    )
    source: str = Field(description="Which producer wrote this, e.g. 'gold_analysis_bot'.")
    event_id: str = Field(
        description="Stable, unique id for this analysis. The pipeline dedupes on it."
    )
    created_at: UtcDatetime = Field(description="When the producer created this event.")
    message_date: UtcDatetime | None = Field(
        default=None, description="When the underlying message was posted, if different."
    )
    raw_text: str = Field(description="Verbatim analysis text. UNTRUSTED user content.")
    article_type: ArticleType = Field(
        default=ArticleType.ANALYSIS,
        description=(
            "Which product mode to produce. Closed enum; an unknown value is refused. "
            "Defaults to ANALYSIS so events written before this field existed still load."
        ),
    )
    chat_id: int | str | None = None
    message_id: int | None = None
    author: str | None = Field(
        default=None, description="Free-text author label, if the producer knows one."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Producer-specific extras, carried through as data. Nothing here is "
            "ever read as configuration."
        ),
    )

    @field_validator("event_id")
    @classmethod
    def _usable_as_a_filename(cls, value: str) -> str:
        cleaned = value.strip()
        if not EVENT_ID_PATTERN.fullmatch(cleaned):
            raise ValueError(
                "event_id must be 8-128 characters of letters, digits, dot, dash or "
                "underscore, starting with a letter or digit"
            )
        return cleaned

    @field_validator("source")
    @classmethod
    def _non_empty_source(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned:
            raise ValueError("source must not be empty")
        return cleaned

    @field_validator("raw_text")
    @classmethod
    def _usable_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("raw_text must not be empty")
        if len(value) > MAX_RAW_TEXT_CHARS:
            raise ValueError(f"raw_text exceeds {MAX_RAW_TEXT_CHARS} characters")
        return value


__all__ = ["EVENT_ID_PATTERN", "INBOX_SCHEMA_VERSION", "AnalysisEvent"]
