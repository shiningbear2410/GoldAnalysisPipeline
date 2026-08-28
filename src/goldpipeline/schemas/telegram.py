"""Schema for the raw analysis input (Telegram or any other text source).

Security note for later rounds: ``raw_text`` is **untrusted data**. It is
carried through the pipeline verbatim (modulo control-character sanitisation)
and must never be interpreted as configuration, as a command, or as part of a
system prompt. The :attr:`TelegramAnalysisInput.trust_level` field exists so the
downstream prompt builder cannot forget this.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from goldpipeline.schemas.common import LenientModel, UtcDatetime, utc_now

MAX_RAW_TEXT_CHARS = 200_000
"""Hard ceiling on analysis text. Beyond this the payload is rejected."""

RAW_TEXT_WARN_CHARS = 20_000
"""Soft ceiling; above it a quality warning is recorded but the Run proceeds."""


class Author(LenientModel):
    """Who wrote the analysis. Every field is optional - Telegram may omit all."""

    id: int | str | None = None
    username: str | None = None
    display_name: str | None = None


class TelegramAnalysisInput(LenientModel):
    """A single raw analysis message plus the metadata needed to audit it.

    Only ``raw_text`` is genuinely required. Missing optional metadata is
    represented as ``null`` (or an empty object for ``metadata``) and reported
    as a quality warning - it is never invented.
    """

    source: str = Field(
        default="telegram",
        description="Origin of the analysis, e.g. 'telegram', 'manual', 'fixture'.",
    )
    chat_id: int | str | None = Field(
        default=None, description="Telegram chat identifier, if known."
    )
    message_id: int | None = Field(
        default=None, description="Telegram message identifier, if known."
    )
    message_date: UtcDatetime | None = Field(
        default=None, description="When the message was posted, as reported by the source."
    )
    received_at: UtcDatetime = Field(
        default_factory=utc_now, description="When this pipeline ingested the message."
    )
    raw_text: str = Field(description="Verbatim analysis text. UNTRUSTED user content.")
    author: Author | None = Field(default=None, description="Message author, if known.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific extras carried through untouched.",
    )
    trust_level: Literal["UNTRUSTED"] = Field(
        default="UNTRUSTED",
        description="Constant marker: this content is data, never instructions.",
    )

    @field_validator("message_date", "received_at")
    @classmethod
    def _require_aware(cls, value: object) -> object:
        from datetime import UTC, datetime

        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise ValueError("datetime must carry an explicit timezone offset")
            return value.astimezone(UTC)
        return value

    @field_validator("source")
    @classmethod
    def _non_empty_source(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned:
            raise ValueError("source must not be empty")
        return cleaned


__all__ = ["MAX_RAW_TEXT_CHARS", "RAW_TEXT_WARN_CHARS", "Author", "TelegramAnalysisInput"]
