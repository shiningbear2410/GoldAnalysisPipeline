"""The reviewer provider interface.

Mirrors :mod:`goldpipeline.adapters.writer_client`: the service layer knows only
this protocol, so swapping OpenAI for another vendor - or for the offline fake -
changes nothing above this line.

A client's job stops at "return a schema-valid answer or raise". Whether the
answer is *acceptable* - the right Run, internally consistent, compatible with
what the deterministic checks found - is decided in
:mod:`goldpipeline.services.review_policy`, so those rules apply to every client
including the fake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from goldpipeline.schemas.review import ReviewerPrompt, ReviewModelOutput, ReviewUsage


@dataclass(frozen=True)
class ReviewRequest:
    """One review request."""

    prompt: ReviewerPrompt
    run_id: str
    max_output_tokens: int = 8000


@dataclass(frozen=True)
class ReviewResponse:
    """A provider's answer, already parsed into the output schema."""

    output: ReviewModelOutput
    model: str
    provider: str
    usage: ReviewUsage = field(default_factory=ReviewUsage)


@runtime_checkable
class ReviewerClient(Protocol):
    """Anything that can audit an article."""

    @property
    def provider(self) -> str:
        """Short provider label recorded on the artifact, e.g. ``openai``."""
        ...

    @property
    def model(self) -> str:
        """Model id this client will use."""
        ...

    def review(self, request: ReviewRequest) -> ReviewResponse:
        """Produce a review.

        Raises:
            ReviewTimeoutError: The provider did not answer in time.
            ReviewProviderError: The provider refused or failed.
            ReviewResponseError: The answer could not be parsed into the schema.
        """
        ...


__all__ = ["ReviewRequest", "ReviewResponse", "ReviewerClient"]
