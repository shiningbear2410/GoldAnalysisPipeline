"""The publisher transport interface.

The service layer knows only this protocol, so the orchestration - durable
intent, chunk pacing, the refusal to retry an ambiguous send - is testable
without a network and identical for every transport.

The contract is narrow: send one message, return a confirmation, or raise. What
matters is *which* exception, because the service maps them to outcomes that
differ in whether the message might already be posted:

* :class:`PublisherRejectedError` and friends - Telegram said no. Nothing was
  delivered, and that is certain.
* :class:`PublisherRateLimitError` - Telegram said "not now", with a delay. The
  only condition safe to retry.
* :class:`PublisherTransportAmbiguousError` - nobody knows. Never retried.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SendRequest:
    """One message to deliver."""

    target_chat: str
    text: str
    chunk_index: int


@dataclass(frozen=True)
class SendOutcome:
    """A confirmed delivery.

    Only ever constructed from a reply that positively identified the posted
    message. An HTTP 200 alone is not a confirmation.
    """

    message_id: int
    chat_id: str | None = None
    telegram_date: int | None = None


@runtime_checkable
class PublisherClient(Protocol):
    """Anything that can deliver one message."""

    @property
    def provider(self) -> str:
        """Short provider label recorded on the artifacts, e.g. ``telegram``."""
        ...

    def send(self, request: SendRequest) -> SendOutcome:
        """Deliver one message.

        Raises:
            PublisherAuthenticationError: Credentials rejected.
            PublisherPermissionError: The bot may not post to this target.
            PublisherRejectedError: The request was explicitly refused.
            PublisherRateLimitError: Flood control; carries ``retry_after``.
            PublisherTransportAmbiguousError: Delivery cannot be determined.
            PublisherResponseError: The reply did not confirm a delivery.
        """
        ...


__all__ = ["PublisherClient", "SendOutcome", "SendRequest"]
