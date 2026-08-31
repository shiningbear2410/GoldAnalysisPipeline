"""Source interfaces.

The domain must not know whether analysis text arrived over the Telegram Bot
API, from a local fixture, or from a paste buffer - only that it arrived and
parses. Round 1 ships the fixture/file implementation; a real Telegram client
in a later round implements the same protocol and nothing downstream changes.

Each loader returns both the parsed model *and* the raw payload it parsed. The
raw payload is what gets persisted into the Run, so an audit reads exactly what
the provider sent, not a re-serialization of our interpretation of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from goldpipeline.schemas.market import MarketDataInput
from goldpipeline.schemas.telegram import TelegramAnalysisInput


@dataclass(frozen=True)
class LoadedSource[ModelT]:
    """A parsed source payload alongside the bytes-level data it came from."""

    model: ModelT
    raw_payload: dict[str, Any]
    origin: str
    """Human-readable description of where this came from, for logs and audit."""

    provenance: dict[str, Any] = field(default_factory=dict)
    """Adapter-specific facts about *this fetch*, recorded in the Run manifest.

    Where ``raw_payload`` is what the provider said, this is what the adapter
    knows about the act of asking: which event id, which broker symbol, when the
    request went out and when the answer came back. It never carries a
    credential - an adapter that needs a key records the setting's name, not its
    value.

    Optional, and empty for the file adapters, whose origin path already says
    everything there is to know.
    """


class AnalysisSource(Protocol):
    """Supplies the raw human analysis the pipeline works from."""

    def load(self) -> LoadedSource[TelegramAnalysisInput]:
        """Fetch and parse one analysis message.

        Raises:
            InputValidationError: If the payload does not satisfy the schema.
        """
        ...


class MarketDataSource(Protocol):
    """Supplies OHLC market data."""

    def load(self) -> LoadedSource[MarketDataInput]:
        """Fetch and parse one market data payload.

        Raises:
            InputValidationError: If the payload does not satisfy the schema.
        """
        ...


__all__ = ["AnalysisSource", "LoadedSource", "MarketDataSource"]
