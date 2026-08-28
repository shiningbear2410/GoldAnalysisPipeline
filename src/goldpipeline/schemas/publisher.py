"""Publisher contracts.

Two artifacts, written at two different moments, and the gap between them is the
whole point.

``publish_intent.json`` is committed **before the first network call**. It
records what is about to be sent and to where. If the process dies mid-send, the
intent is the only evidence that a request may have reached Telegram - and it is
what stops the next run from sending the same article again.

``publish_result.json`` is committed **after** the attempt ends, however it
ended. It records what was confirmed, what was not, and why.

Neither artifact ever contains the bot token. The Telegram URL embeds the token
in its path, so a stray URL in a log or an error is a leaked credential; nothing
here carries one.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from goldpipeline.schemas.common import StrictModel, UtcDatetime, utc_now

PUBLISHER_SCHEMA_VERSION = "1.0.0"
"""Version of the publisher artifact contract."""

PUBLISHER_VERSION = "telegram_publisher_v1"
"""Which publisher produced an attempt."""

SUPPORTED_GATE_VERSIONS = frozenset({"gold_publish_gate_v1"})
"""Gate versions this publisher will act on.

An approval from a gate this publisher does not recognise is not an approval it
can reason about. Publishing on one would mean trusting checks whose meaning has
changed.
"""


class PublishStatus(StrEnum):
    """How an attempt ended.

    The distinction that matters is between *knowing* and *not knowing*:

    * ``PUBLISHED`` - every chunk confirmed by Telegram.
    * ``FAILED`` - nothing was delivered, and Telegram said so explicitly. An
      explicit refusal is good news: the outcome is certain.
    * ``PARTIAL`` - some chunks confirmed, then an explicit refusal. Still
      certain, just incomplete.
    * ``UNCERTAIN`` - a timeout, a reset, a 5xx, an unparseable reply. Telegram
      may or may not have the message. **Never retried automatically.**

    ``UNCERTAIN`` outranks ``PARTIAL``: if the last chunk's fate is unknown, the
    attempt as a whole is unknown.
    """

    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    UNCERTAIN = "UNCERTAIN"


class FailureCategory(StrEnum):
    """Why an attempt did not fully succeed."""

    CONFIGURATION = "CONFIGURATION"
    AUTHENTICATION = "AUTHENTICATION"
    PERMISSION = "PERMISSION"
    REJECTED = "REJECTED"
    RATE_LIMITED = "RATE_LIMITED"
    TRANSPORT_AMBIGUOUS = "TRANSPORT_AMBIGUOUS"
    RESPONSE_INVALID = "RESPONSE_INVALID"
    ORPHAN_PUBLISH_INTENT = "ORPHAN_PUBLISH_INTENT"


class ChunkPlan(StrictModel):
    """One message the article was split into.

    The text itself is not stored - the article already lives in
    ``claude_final.md`` and duplicating it would create a second copy to keep in
    sync. The digest is enough to prove which bytes were planned.
    """

    index: int = Field(ge=0)
    text_sha256: str = Field(min_length=64, max_length=64)
    char_count: int = Field(ge=1, description="Code points, as Python counts them.")
    utf16_units: int = Field(ge=1, description="UTF-16 code units, as Telegram counts them.")


class PublishIntent(StrictModel):
    """The ``publish_intent.json`` artifact, written before any request.

    Its existence without a matching result is what marks a Run as having an
    unknown fate. That is the mechanism preventing duplicate posts.
    """

    schema_version: str = Field(default=PUBLISHER_SCHEMA_VERSION)
    publisher_version: str = Field(default=PUBLISHER_VERSION)
    run_id: str
    attempt_id: str = Field(
        min_length=1, max_length=64, description="Identifies this attempt in logs and the result."
    )
    created_at: UtcDatetime = Field(default_factory=utc_now)

    provider: str = Field(description="Which client will send, e.g. 'telegram' or 'fake'.")
    target_chat: str = Field(
        min_length=1,
        max_length=200,
        description="Destination, from configuration only. Never from article content.",
    )

    gate_version: str
    decision_sha256: str = Field(min_length=64, max_length=64)
    final_article_sha256: str = Field(min_length=64, max_length=64)

    chunk_count: int = Field(ge=1)
    chunks: list[ChunkPlan] = Field(min_length=1)


class DeliveredMessage(StrictModel):
    """One chunk Telegram confirmed."""

    chunk_index: int = Field(ge=0)
    chunk_sha256: str = Field(min_length=64, max_length=64)
    message_id: int = Field(description="Telegram's identifier for the posted message.")
    telegram_date: int | None = Field(
        default=None, ge=0, description="Unix timestamp Telegram reported."
    )
    retry_count: int = Field(default=0, ge=0, description="429 retries before this succeeded.")


class PublishFailure(StrictModel):
    """Why an attempt stopped, in terms safe to store and print."""

    category: FailureCategory
    safe_code: str = Field(min_length=1, max_length=64)
    safe_message: str = Field(
        min_length=1,
        max_length=600,
        description="Written by this pipeline. Never a provider string, never a URL.",
    )
    failed_chunk_index: int | None = Field(default=None, ge=0)


class PublishResult(StrictModel):
    """The ``publish_result.json`` artifact."""

    schema_version: str = Field(default=PUBLISHER_SCHEMA_VERSION)
    publisher_version: str = Field(default=PUBLISHER_VERSION)
    run_id: str
    attempt_id: str
    stage: str = Field(default="telegram_publisher")

    status: PublishStatus
    provider: str
    target_chat: str

    started_at: UtcDatetime
    completed_at: UtcDatetime

    decision_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    final_article_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    publish_intent_sha256: str | None = Field(default=None, min_length=64, max_length=64)

    chunk_count: int = Field(ge=0)
    confirmed_count: int = Field(ge=0)
    messages: list[DeliveredMessage] = Field(default_factory=list)

    failure: PublishFailure | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def fully_delivered(self) -> bool:
        """Whether every planned chunk was confirmed.

        The only condition under which ``status`` may be ``PUBLISHED``.
        """
        return self.chunk_count > 0 and self.confirmed_count == self.chunk_count


__all__ = [
    "PUBLISHER_SCHEMA_VERSION",
    "PUBLISHER_VERSION",
    "SUPPORTED_GATE_VERSIONS",
    "ChunkPlan",
    "DeliveredMessage",
    "FailureCategory",
    "PublishFailure",
    "PublishIntent",
    "PublishResult",
    "PublishStatus",
]
