"""Offline publisher.

Every test in this repository runs against this, and so does ``--fake-publisher``.
Nothing here opens a socket, so no test can post to a real channel by accident -
which for this stage is not a convenience but a safety requirement.

It records the exact text of every send, so tests can assert the strongest
property the publisher has: what reached the transport is byte-for-byte what the
gate approved.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from goldpipeline.adapters.publisher_client import SendOutcome, SendRequest
from goldpipeline.domain.errors import (
    PublisherError,
    PublisherPermissionError,
    PublisherRateLimitError,
    PublisherRejectedError,
    PublisherResponseError,
    PublisherTransportAmbiguousError,
)

FAKE_PROVIDER = "fake"
FIRST_MESSAGE_ID = 1000


@dataclass
class FakePublisherClient:
    """Deterministic, offline implementation of :class:`PublisherClient`.

    Configure a behaviour per chunk index with *failures*, or a single behaviour
    for every send with *raises*. Anything not configured succeeds.

    ``sent`` holds the exact text of each accepted send, in order.
    """

    raises: PublisherError | None = None
    failures: dict[int, PublisherError] = field(default_factory=dict)
    outcome_factory: Callable[[SendRequest, int], SendOutcome] | None = None

    calls: list[SendRequest] = field(default_factory=list)
    """Every request seen, including ones that then raised."""

    sent: list[str] = field(default_factory=list)
    """Text of every *successful* send, in order."""

    targets: list[str] = field(default_factory=list)
    """Destination of every request, so tests can prove it never varies."""

    _attempts: dict[int, int] = field(default_factory=dict)

    @property
    def provider(self) -> str:
        return FAKE_PROVIDER

    def send(self, request: SendRequest) -> SendOutcome:
        """Record the request, then behave as configured."""
        self.calls.append(request)
        self.targets.append(request.target_chat)

        attempt = self._attempts.get(request.chunk_index, 0)
        self._attempts[request.chunk_index] = attempt + 1

        error = self.raises or self.failures.get(request.chunk_index)
        if error is not None:
            raise error

        if self.outcome_factory is not None:
            outcome = self.outcome_factory(request, attempt)
        else:
            outcome = SendOutcome(
                message_id=FIRST_MESSAGE_ID + request.chunk_index,
                chat_id=request.target_chat,
                telegram_date=1788000000 + request.chunk_index,
            )

        self.sent.append(request.text)
        return outcome


def transient_rate_limit_client(retry_after: int = 2, fail_times: int = 1) -> FakePublisherClient:
    """Rate-limits the first *fail_times* attempts of each chunk, then succeeds.

    Models the one condition this pipeline retries: Telegram stating plainly
    that it did not accept the request, and when to come back.
    """
    state: dict[int, int] = {}

    def outcome(request: SendRequest, attempt: int) -> SendOutcome:
        seen = state.get(request.chunk_index, 0)
        state[request.chunk_index] = seen + 1
        if seen < fail_times:
            raise PublisherRateLimitError("provider applied flood control", retry_after=retry_after)
        return SendOutcome(
            message_id=FIRST_MESSAGE_ID + request.chunk_index,
            chat_id=request.target_chat,
            telegram_date=1788000000,
        )

    return FakePublisherClient(outcome_factory=outcome)


def rejecting_client(message: str = "provider refused the request") -> FakePublisherClient:
    """Always refuses explicitly - nothing is delivered, and that is certain."""
    return FakePublisherClient(raises=PublisherRejectedError(message, status_code=400))


def forbidding_client() -> FakePublisherClient:
    """Always refuses on permission grounds."""
    return FakePublisherClient(
        raises=PublisherPermissionError(
            "the bot may not post to the configured target", status_code=403
        )
    )


def ambiguous_client(reason: str = "the request timed out") -> FakePublisherClient:
    """Always leaves delivery unknown - the case that must never be retried."""
    return FakePublisherClient(raises=PublisherTransportAmbiguousError(reason))


def unconfirmed_client() -> FakePublisherClient:
    """Answers without confirming a message id."""
    return FakePublisherClient(
        raises=PublisherResponseError("provider reply did not identify a posted message")
    )


__all__ = [
    "FAKE_PROVIDER",
    "FIRST_MESSAGE_ID",
    "FakePublisherClient",
    "ambiguous_client",
    "forbidding_client",
    "rejecting_client",
    "transient_rate_limit_client",
    "unconfirmed_client",
]
