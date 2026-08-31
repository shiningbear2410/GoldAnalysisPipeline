"""Turns an inbox event into the analysis input Round 1 already understands.

The mapping is deliberately small and total: every field of
:class:`~goldpipeline.schemas.inbox.AnalysisEvent` either becomes a field of
:class:`~goldpipeline.schemas.telegram.TelegramAnalysisInput` or is recorded as
provenance. Nothing is dropped silently, and nothing is invented.

What this adapter does *not* do is as important. It reads no configuration from
the event, resolves no paths from it, and passes no part of it to anything that
executes. ``raw_text`` arrives here untrusted and leaves here untrusted, with
``trust_level`` set so no later stage can forget.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError as PydanticValidationError

from goldpipeline.adapters.base import LoadedSource
from goldpipeline.domain.errors import InboxPayloadError
from goldpipeline.schemas.inbox import AnalysisEvent
from goldpipeline.schemas.telegram import Author, TelegramAnalysisInput
from goldpipeline.storage.atomic import sha256_bytes


def parse_event(payload: dict[str, Any]) -> AnalysisEvent:
    """Validate one inbox payload.

    Raises:
        InboxPayloadError: The payload is not a shape this pipeline accepts.
            Unknown keys are a failure, not a curiosity: the schema is the list
            of things a producer is allowed to influence.
    """
    try:
        return AnalysisEvent.model_validate(payload)
    except PydanticValidationError as exc:
        raise InboxPayloadError(
            "inbox event failed schema validation",
            errors=[
                {"field": ".".join(str(part) for part in error["loc"]), "error": error["msg"]}
                for error in exc.errors()
            ],
        ) from exc


class InboxAnalysisSource:
    """Supplies an :class:`AnalysisSource` from an already-validated event.

    The event is passed in rather than read here, because by the time this runs
    the ingestion service has already claimed the file, hashed its bytes and
    checked the ledger. Re-reading it would risk working from different content
    than the one that was hashed.
    """

    def __init__(self, event: AnalysisEvent, *, raw: bytes, origin: str) -> None:
        """Wrap *event*.

        Args:
            event: The validated payload.
            raw: The exact bytes the producer wrote, for the provenance digest.
            origin: Where it came from, for logs and the manifest.
        """
        self._event = event
        self._raw = raw
        self._origin = origin

    @property
    def event(self) -> AnalysisEvent:
        return self._event

    def load(self) -> LoadedSource[TelegramAnalysisInput]:
        """Map the event onto Round 1's analysis input."""
        event = self._event
        model = TelegramAnalysisInput(
            source=event.source,
            chat_id=event.chat_id,
            message_id=event.message_id,
            message_date=event.message_date or event.created_at,
            # Falling back to created_at is not inventing metadata: for a bot
            # that generates its own analysis the two are the same instant, and
            # both are recorded separately in provenance either way.
            raw_text=event.raw_text,
            author=Author(display_name=event.author) if event.author else None,
            metadata=dict(event.metadata),
        )
        # The stored source file is the pipeline's own view of the analysis, in
        # the schema every later stage reads. The producer's original bytes are
        # not lost - their digest is in the manifest and in the ledger.
        payload = model.model_dump(mode="json")
        return LoadedSource(
            model=model,
            raw_payload=payload,
            origin=self._origin,
            provenance={
                "kind": "inbox",
                "event_id": event.event_id,
                "event_source": event.source,
                "event_created_at": event.created_at.isoformat().replace("+00:00", "Z"),
                "payload_sha256": sha256_bytes(self._raw),
                "raw_text_chars": len(event.raw_text),
            },
        )


__all__ = ["InboxAnalysisSource", "parse_event"]
