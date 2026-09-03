"""The generation choice a single Run was created under.

**Why a snapshot and not a lookup.** Preferences are mutable operator state; a
Run is immutable evidence. If the writer and the finalizer each read the
preferences file when their turn came, an operator switching model between the
two stages would get an article drafted by one vendor and edited by another -
and the artifact would record both, truthfully, with nothing saying why. Worse,
the same Run resumed after a scheduler restart could come back on a different
model again.

So the choice is resolved **once**, when the Run is created, and written into
the Run's own provenance. Every later stage reads it from there. Changing a
preference affects the next Run; it cannot reach into one already under way.

**What is deliberately absent.** No API key and no hint of whether one exists;
no reviewer model, because the review is independent of the draft it judges and
a field here would be the first step towards coupling them; no Telegram target
and no publish behaviour, because a generation choice is about *what to write*,
never about where it goes.

``api_model_id`` is resolved from the catalog at snapshot time rather than
stored as a second source of truth: it is recorded because two DeepSeek choices
share one vendor model, so the selection alone no longer says what actually ran.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from goldpipeline.schemas.common import StrictModel
from goldpipeline.schemas.preferences import (
    PREFERENCES_SCHEMA_VERSION,
    PreferencesSource,
    Provider,
    ThinkingMode,
    UserPreferences,
    resolve_model,
)

GENERATION_SCHEMA_VERSION = "1"


class GenerationSelection(StrictModel):
    """Which provider and model this Run generates with, fixed at creation."""

    schema_version: Literal["1"] = "1"

    provider: Provider
    selection_id: str = Field(description="What the operator picked. Stable across vendor changes.")
    api_model_id: str = Field(description="What actually goes to the vendor for this selection.")
    thinking: ThinkingMode = Field(
        default=ThinkingMode.NOT_APPLICABLE,
        description="Whether this selection asks the vendor to think. Claude has no such control.",
    )

    preferences_schema_version: str = Field(
        default=PREFERENCES_SCHEMA_VERSION,
        description="Which preferences contract the choice was read under.",
    )
    preference_source: PreferencesSource = Field(
        description=(
            "DEFAULT when nobody had expressed a preference, FILE when one was stored. "
            "The difference matters to an audit: one is a decision, the other is not."
        )
    )

    @classmethod
    def from_preferences(
        cls, preferences: UserPreferences, *, source: PreferencesSource
    ) -> GenerationSelection:
        """Freeze one set of preferences into a Run's generation choice.

        The catalog is consulted here, once, so the vendor mapping recorded on
        the Run is the one that was current when the Run began - and so a later
        catalog change cannot silently reinterpret an old Run's snapshot.

        Raises:
            ValueError: The catalog does not offer this pairing. Unreachable
                through a validated ``UserPreferences``, and checked anyway:
                this is the last point before the choice becomes durable.
        """
        model = resolve_model(preferences.provider, preferences.selection_id)
        return cls(
            provider=preferences.provider,
            selection_id=model.selection_id,
            api_model_id=model.api_model_id,
            thinking=model.thinking,
            preferences_schema_version=preferences.schema_version,
            preference_source=source,
        )


__all__ = ["GENERATION_SCHEMA_VERSION", "GenerationSelection"]
