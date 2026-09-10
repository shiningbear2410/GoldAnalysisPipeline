"""The trade analyst provider interface.

The narrowest possible seam: a rendered system and user turn go in, text comes
back. It deliberately does **not** know what a candidate is, what a ranking is,
or how to validate one - all of that lives in the domain, where it can be tested
without a provider and where a provider cannot influence it.

**Why a sibling protocol rather than reusing ``WriterClient``.** The existing
generation seam is the right architecture and the wrong type: ``WriterClient``
returns a ``WriterModelOutput``, which is the ANALYSIS article schema and has
nowhere to put four lists of candidate ids. Bending it to fit would either widen
the article schema - touching the two shipped products - or smuggle a ranking
through a prose field. So the pattern is copied and the type is not: same
Protocol shape, same request/response dataclasses, same offline fake, and no
second HTTP client stack anywhere.

**This module reaches no vendor.** There is no client construction here and no
credential lookup. Choosing a provider and a model for the analyst is 6.6h
wiring; until then the only implementations are the offline fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class TradeAnalystRequest:
    """One ranking request, already rendered.

    Text in both directions, which is the whole point of putting the boundary
    here: whatever the domain builds, a provider sees a string, and whatever a
    provider says, the domain sees a string it has to validate.
    """

    system: str
    user: str
    max_tokens: int = 2000


@dataclass(frozen=True)
class TradeAnalystResponse:
    """A provider's answer, unparsed and untrusted.

    ``text`` is not a ranking. It is a claim that something is a ranking, and it
    stays a claim until the domain validator has checked every id against the
    deterministic candidate set.
    """

    text: str
    model: str
    provider: str


@runtime_checkable
class TradeAnalystClient(Protocol):
    """Anything that can turn a ranking prompt into text."""

    @property
    def provider(self) -> str:
        """Short provider label, e.g. ``fake``."""
        ...

    @property
    def model(self) -> str:
        """Model id this client will use."""
        ...

    def rank(self, request: TradeAnalystRequest) -> TradeAnalystResponse:
        """Produce a candidate ranking as text."""
        ...


__all__ = ["TradeAnalystClient", "TradeAnalystRequest", "TradeAnalystResponse"]
