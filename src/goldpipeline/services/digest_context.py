"""Everything a future digest writer is *given* rather than asked to work out.

Round 6.5a. The seam, and no writer behind it yet.

**What the split is for.** A ``NEWS_DIGEST`` has two kinds of content. Which
three or four items out of eleven actually mattered, and what they mean
together, is editorial judgement that only a model can do. The date range, the
window boundaries and the price arithmetic are facts, and a model asked for
them produces something plausible instead. So this object carries the second
kind, already rendered, and Round 6.5b's job is to hand them over and require
that they come back unchanged.

**The window is found, never recomputed.** The authority already exists: a
producer request records ``requested_at`` as the end of its news window and the
lookback it used, and both are carried into the Run - the instant as the
event's ``created_at``, the lookback in the metadata the inbox source passes
through verbatim. This module reads them. It does not add a second place for
the window to live, because two places is how a Run resumed an hour later ends
up describing a different day than the news items it carries.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import timedelta

from pydantic import Field

from goldpipeline.schemas.common import StrictModel, Timeframe
from goldpipeline.schemas.digest import DigestWindow, PriceReaction
from goldpipeline.schemas.inbox import AnalysisEvent
from goldpipeline.schemas.news import MAX_LOOKBACK, MIN_LOOKBACK
from goldpipeline.schemas.news_digest import DigestSourceItem
from goldpipeline.services.digest_render import (
    digest_title,
    digest_window_line,
    render_price_reaction,
)

logger = logging.getLogger(__name__)

NEWS_WINDOW_METADATA_KEY = "news_window_seconds"
"""Where the producer records its lookback, and the only key read for it.

Named once here rather than spelled out at each use. The producer writes it in
``services.producer``; this is the only reader, and a test pins that the two
agree - an untyped dictionary key is exactly the kind of contract that drifts
silently when the two ends are written months apart.
"""


class DigestFacts(StrictModel):
    """The deterministic half of a news digest, ready to be copied.

    Both the typed facts and their rendered form are carried. The rendering is
    what gets published and what a later round will require the model to
    reproduce byte for byte; the facts behind it are what makes the rendering
    auditable, and what a checker compares against when the model is suspected
    of having edited a line it was told to copy.
    """

    window: DigestWindow
    title: str = Field(description="The 📰 headline, dated in Vietnam time.")
    window_line: str = Field(description="The 🕐 span, to the minute, in Vietnam time.")
    price_reaction: PriceReaction
    price_reaction_block: str = Field(description="The 📈 section, exactly as published.")
    symbol: str
    timeframe: Timeframe

    news_items: tuple[DigestSourceItem, ...] = Field(
        default=(),
        description=(
            "The curated items available to choose from. Selection is editorial "
            "and belongs to the writer; this is the closed list it may choose "
            "within, and never a ranking. Each carries its own timestamp, which "
            "is what the renderer prints - so a digest reports the time the "
            "producer collected rather than the time a model remembered."
        ),
    )

    @property
    def news_item_ids(self) -> tuple[str, ...]:
        """Just the ids, for the closed-vocabulary check."""
        return tuple(item.item_id for item in self.news_items)

    @property
    def sources_by_id(self) -> dict[str, DigestSourceItem]:
        """The items the renderer looks timestamps up in."""
        return {item.item_id: item for item in self.news_items}

    @property
    def deterministic_lines(self) -> tuple[str, ...]:
        """The lines a writer must reproduce unchanged.

        Named as a group because Round 6.5b will enforce them as a group: the
        title, the window and the price block are the three places where an
        edit would replace a checked fact with a plausible one.
        """
        return (self.title, self.window_line, self.price_reaction_block)


def digest_window_from_event(event: AnalysisEvent) -> DigestWindow | None:
    """The window an event was produced under, or ``None`` when it records none.

    Args:
        event: The producer's own payload, as accepted.

    Returns:
        The window, built from ``created_at`` as the end and the recorded
        lookback. ``None`` when the event carries no lookback at all - every
        event written before the producer existed, and every hand-submitted
        one. Absence is not an error here: those events are ANALYSIS, which has
        no window, and inventing a default would hand a digest a span nobody
        chose.

    Raises:
        ValueError: The lookback is present but unusable - not a whole number,
            or outside the collector's own bounds. A malformed value is a
            different thing from a missing one, and guessing past it would mean
            describing a window the news was never gathered for.
    """
    raw = event.metadata.get(NEWS_WINDOW_METADATA_KEY)
    if raw is None:
        return None

    if isinstance(raw, bool) or not isinstance(raw, int | float | str):
        raise ValueError(f"{NEWS_WINDOW_METADATA_KEY} must be a number of seconds, got {type(raw)}")
    try:
        seconds = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{NEWS_WINDOW_METADATA_KEY} is not a whole number of seconds") from exc

    low = int(MIN_LOOKBACK.total_seconds())
    high = int(MAX_LOOKBACK.total_seconds())
    if not low <= seconds <= high:
        raise ValueError(f"{NEWS_WINDOW_METADATA_KEY}={seconds} is outside [{low}, {high}]")

    return DigestWindow.ending_at(event.created_at, timedelta(seconds=seconds))


def build_digest_facts(
    *,
    window: DigestWindow,
    price_reaction: PriceReaction,
    symbol: str,
    timeframe: Timeframe,
    news_items: Sequence[DigestSourceItem] = (),
) -> DigestFacts:
    """Assemble the deterministic half, rendering each line once.

    Rendered here rather than at use so that the artifact and the article carry
    the same characters. A renderer called twice in two places is a renderer
    that will one day be called with two different arguments.
    """
    if price_reaction.window != window:
        raise ValueError("the price reaction describes a different window than the digest")

    facts = DigestFacts(
        window=window,
        title=digest_title(window),
        window_line=digest_window_line(window),
        price_reaction=price_reaction,
        price_reaction_block=render_price_reaction(price_reaction),
        symbol=symbol,
        timeframe=timeframe,
        news_items=tuple(news_items),
    )
    logger.info(
        "digest.facts symbol=%s timeframe=%s window=%s..%s activity=%s items=%d",
        symbol,
        timeframe,
        window.start.isoformat(),
        window.end.isoformat(),
        price_reaction.market_activity,
        len(facts.news_item_ids),
    )
    return facts


__all__ = [
    "NEWS_WINDOW_METADATA_KEY",
    "DigestFacts",
    "build_digest_facts",
    "digest_window_from_event",
]
