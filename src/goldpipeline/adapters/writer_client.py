"""The writer provider interface.

The service layer knows only this protocol. Swapping Anthropic for another
provider, or for the offline fake, changes nothing above this line.

A client's contract is narrow on purpose: given a rendered prompt, return a
schema-valid :class:`WriterModelOutput` or raise a
:class:`~goldpipeline.domain.errors.WriterError`. Deciding whether the output is
*acceptable for this Run* - run id, warning codes, article length - is the
service's job, so that the same checks apply to every client including fakes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from goldpipeline.schemas.writer import WriterModelOutput, WriterPrompt, WriterUsage


@dataclass(frozen=True)
class WriterRequest:
    """One generation request."""

    prompt: WriterPrompt
    run_id: str
    max_tokens: int = 8000


@dataclass(frozen=True)
class WriterResponse:
    """A provider's answer, already parsed into the output schema."""

    output: WriterModelOutput
    model: str
    """The vendor model that actually served the request."""

    provider: str
    usage: WriterUsage = field(default_factory=WriterUsage)
    selection_id: str | None = None
    """The catalog choice behind `model`, when the two differ.

    ``None`` whenever the selection *is* the vendor id - every Claude model, and
    the offline fake. Set by a provider whose choices no longer map one to one
    onto vendor models, so an artifact can still say what was asked for.
    """


@runtime_checkable
class WriterClient(Protocol):
    """Anything that can turn a prompt into a draft."""

    @property
    def provider(self) -> str:
        """Short provider label recorded on the artifact, e.g. ``anthropic``."""
        ...

    @property
    def model(self) -> str:
        """Model id this client will use."""
        ...

    def generate(self, request: WriterRequest) -> WriterResponse:
        """Produce a draft.

        Raises:
            WriterTimeoutError: The provider did not answer in time.
            WriterProviderError: The provider refused or failed.
            WriterResponseError: The answer could not be parsed into the schema.
        """
        ...


__all__ = ["WriterClient", "WriterRequest", "WriterResponse"]
