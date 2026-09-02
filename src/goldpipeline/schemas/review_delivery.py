"""Contracts for delivering an approved article to a human for review.

**Review delivery is not publishing.** They share a transport and nothing else:

===================  ==============================  =========================
                     review delivery                 publishing
===================  ==============================  =========================
destination          the operator's own chat         the audience's channel
trigger              automatic, once approved        an explicit human act
Run status after     ``READY_TO_PUBLISH``            ``PUBLISHED``
artifacts            ``review_delivery_*.json``      ``publish_*.json``
===================  ==============================  =========================

Keeping the artifacts separate is what keeps that distinction true on disk. If
review delivery wrote ``publish_intent.json`` it would consume the publisher's
one-attempt budget, and a Run a human had merely *read* would look, forever
afterwards, like a Run that had been posted to the channel.

The safety model is copied deliberately from the publisher, because the hazard
is identical: Telegram's ``sendMessage`` has no idempotency key, so an intent is
committed before the first request and a delivery whose outcome is unknown is
never retried. A duplicate review message is less damaging than a duplicate
publication, but a scheduler that resends every sixty seconds would be
unusable - and the fix for that is the same fix.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from goldpipeline.schemas.common import StrictModel, UtcDatetime
from goldpipeline.schemas.publisher import ChunkPlan, DeliveredMessage, PublishFailure

REVIEW_DELIVERY_SCHEMA_VERSION = "1.0.0"

INTENT_FILENAME = "review_delivery_intent.json"
RESULT_FILENAME = "review_delivery_result.json"


class ReviewDeliveryStatus(StrEnum):
    """How one review-delivery attempt ended.

    ``UNCERTAIN`` exists for the same reason it does in publishing: an attempt
    that may or may not have arrived must never be repeated automatically. The
    Run stays approved and unpublished either way; only the operator's inbox is
    affected, and a human can look.
    """

    DELIVERED = "DELIVERED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    UNCERTAIN = "UNCERTAIN"
    SKIPPED = "SKIPPED"
    """Not eligible - too old, not approved, or already delivered."""


class ReviewDeliveryIntent(StrictModel):
    """Committed before the first request, so a crash is distinguishable.

    Without it, "never sent" and "sent, acknowledgement lost" look identical to
    the next tick, and the next tick happens in sixty seconds.
    """

    schema_version: str = Field(default=REVIEW_DELIVERY_SCHEMA_VERSION)
    run_id: str
    attempt_id: str
    created_at: UtcDatetime
    provider: str
    target_chat: str = Field(description="The review destination. Never the publish target.")
    decision_sha256: str | None = None
    final_article_sha256: str | None = None
    chunk_count: int = Field(ge=0)
    chunks: list[ChunkPlan] = Field(default_factory=list)


class ReviewDeliveryResult(StrictModel):
    """The durable record that this Run has been shown to a human.

    Its existence is the idempotency key: a Run carrying one is never delivered
    again, whatever the outcome was.
    """

    schema_version: str = Field(default=REVIEW_DELIVERY_SCHEMA_VERSION)
    run_id: str
    attempt_id: str
    status: ReviewDeliveryStatus
    provider: str
    target_chat: str
    started_at: UtcDatetime
    completed_at: UtcDatetime
    final_article_sha256: str | None = None
    review_intent_sha256: str | None = None
    chunk_count: int = Field(default=0, ge=0)
    confirmed_count: int = Field(default=0, ge=0)
    messages: list[DeliveredMessage] = Field(default_factory=list)
    failure: PublishFailure | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def delivered(self) -> bool:
        return self.status is ReviewDeliveryStatus.DELIVERED


__all__ = [
    "INTENT_FILENAME",
    "REVIEW_DELIVERY_SCHEMA_VERSION",
    "RESULT_FILENAME",
    "ReviewDeliveryIntent",
    "ReviewDeliveryResult",
    "ReviewDeliveryStatus",
]
