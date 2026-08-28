"""The finalizer provider interface.

Mirrors the writer and reviewer protocols: the service layer knows only this,
so swapping the vendor - or the offline fake - changes nothing above this line.

A client returns a schema-valid revision or raises. Whether the revision is
*acceptable* - complete resolutions, severe issues actually fixed, no new
deterministic problems - is decided in
:mod:`goldpipeline.services.finalizer_policy`, so those rules apply to every
client including the fake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from goldpipeline.schemas.finalizer import (
    FinalizerModelOutput,
    FinalizerPrompt,
    FinalizerUsage,
)


@dataclass(frozen=True)
class FinalizeRequest:
    """One revision request."""

    prompt: FinalizerPrompt
    run_id: str
    max_tokens: int = 8000


@dataclass(frozen=True)
class FinalizeResponse:
    """A provider's answer, already parsed into the output schema."""

    output: FinalizerModelOutput
    model: str
    provider: str
    usage: FinalizerUsage = field(default_factory=FinalizerUsage)


@runtime_checkable
class FinalizerClient(Protocol):
    """Anything that can revise an article to order."""

    @property
    def provider(self) -> str:
        """Short provider label recorded on the artifact, e.g. ``anthropic``."""
        ...

    @property
    def model(self) -> str:
        """Model id this client will use."""
        ...

    def finalize(self, request: FinalizeRequest) -> FinalizeResponse:
        """Produce a revised article.

        Raises:
            FinalizeTimeoutError: The provider did not answer in time.
            FinalizeProviderError: The provider refused or failed.
            FinalizeResponseError: The answer could not be parsed into the schema.
        """
        ...


__all__ = ["FinalizeRequest", "FinalizeResponse", "FinalizerClient"]
