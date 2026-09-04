"""Deterministic facts a news digest is built from.

Round 6.5a. ``NEWS_DIGEST`` answers *what happened*, and one of its sections -
"📈 Giá phản ứng" - is arithmetic over candles. This module holds the typed
facts that section will be rendered from, and nothing else: no prose is
authoritative here, no model is consulted, and no news item is interpreted.

**Why the facts come before the writer.** An article that says gold rose 105
USD is making a claim the pipeline can check only if something computed 105
first. Letting a model read a candle table and do the subtraction produces a
number nobody can trace, arriving in the one section whose whole value is that
it is not an opinion. So the arithmetic is done here, the model is later handed
the answer, and the deterministic line is copied rather than composed.

**Nothing here explains anything.** A :class:`PriceReaction` records that price
moved and by how much. It has no field for *why*, and it never will: the news
items sit in the same digest, and a structure that carried both an event and a
price move would be read as connecting them. Interpretation belongs to the
writer, under the causality rules Round 6.4e already fixed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from goldpipeline.schemas.common import (
    Price,
    StrictModel,
    Timeframe,
    UtcDatetime,
)
from goldpipeline.schemas.news import MAX_LOOKBACK, MIN_LOOKBACK

DIGEST_SCHEMA_VERSION = "1.0.0"

DIGEST_TARGET_MIN_CHARS = 900
DIGEST_TARGET_MAX_CHARS = 1500
DIGEST_HARD_CAP_CHARS = 1900
"""The locked length contract, recorded but not yet enforced.

``NEWS_DIGEST`` is not producible, so wiring a cap into a stage that never runs
would be a rule nobody could test against a real article. The numbers live here
so the activation round adopts the agreed contract rather than re-deciding it.
"""


class MarketActivity(StrEnum):
    """What the candles say about the window, as a fact rather than a guess.

    Four states, and each one is something the bars can actually establish. The
    distinction that matters is between *no bar closed* and *no bar was even
    open*: a six-minute window inside one M15 candle has data and simply has
    nothing finished in it, while a Saturday window has no candle overlapping it
    at all. Those call for different sentences, and only the second is what a
    reader would call "the market was shut".

    What is deliberately absent is a state meaning "the market was closed".
    Telling a closed market from a hole in the data needs a session calendar,
    which this round is told not to build - and a state whose meaning the code
    cannot establish is worse than one state fewer, because it invites a
    confident sentence about something nobody checked.
    """

    NORMAL = "NORMAL"
    """At least one candle closed inside the window. Every figure is available."""

    NO_NEW_CLOSED_BAR = "NO_NEW_CLOSED_BAR"
    """Candles overlap the window but none closed in it.

    A window shorter than one bar, or one that ends mid-bar. There is a known
    price at the start boundary and nothing new to compare it against yet.
    """

    NO_MARKET_ACTIVITY = "NO_MARKET_ACTIVITY"
    """No candle's interval overlaps the window at all.

    A weekend, a holiday, or a gap in what the provider returned - the code
    records that the window is empty and does not say which of those it was.
    """

    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    """No closed candle at or before the window start.

    The start reference cannot be established, so no change can be stated. The
    first available candle is *not* used as a stand-in: that would report the
    move since an arbitrary point as though it were the move since the boundary.
    """

    @property
    def has_change(self) -> bool:
        """Whether a start-to-end change was actually measured."""
        return self is MarketActivity.NORMAL


class DigestWindow(StrictModel):
    """The span of time a digest describes. Fixed when the request was accepted.

    Immutable, and that is the point rather than a detail. The window belongs to
    the *request*: a Run resumed an hour after it was created must still
    describe the same hours, and one recomputed from ``now`` at writing time
    would silently describe a different day than the news items it carries. So
    the end is the producer's own observation instant, carried through, and
    nothing downstream is permitted to recompute it.

    Both bounds are UTC. Vietnam is a presentation concern and lives in the
    renderer; storing local time here would make the authority depend on where
    the reader is.
    """

    start: UtcDatetime = Field(description="Inclusive lower bound, UTC.")
    end: UtcDatetime = Field(description="The producer's observation instant, UTC.")
    lookback_seconds: int = Field(
        ge=int(MIN_LOOKBACK.total_seconds()),
        le=int(MAX_LOOKBACK.total_seconds()),
        description="How far back the window reaches. Bounded by the news collector's own limits.",
    )

    @field_validator("start", "end")
    @classmethod
    def _aware_and_normalized(cls, value: datetime) -> datetime:
        """Require an offset, then store the instant as UTC.

        ``UtcDatetime`` promises the serialization, not the coercion - a value
        built from ``13:00+07:00`` keeps that offset unless a validator moves
        it. The instant would be right either way, but the stored authority
        would read as local time, and the whole point of holding the window in
        UTC is that two Runs constructed on different machines produce the same
        bytes. The producer request already does this for the same reason.

        Naive is refused rather than assumed: there is no context at this layer
        to interpret one with, and reading it as this machine's local time puts
        a Vietnamese window seven hours from where the caller meant it.
        """
        if value.tzinfo is None:
            raise ValueError("a digest window boundary must carry an explicit UTC offset")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _consistent(self) -> DigestWindow:
        """The three fields must agree; two of them decide the third.

        Stored rather than derived because an artifact a person reads months
        later should not require arithmetic to answer "what period is this?" -
        and validated here so the redundancy can never become a disagreement.
        """
        if self.end <= self.start:
            raise ValueError("a digest window must end after it starts")
        actual = int((self.end - self.start).total_seconds())
        if actual != self.lookback_seconds:
            raise ValueError(
                f"window spans {actual}s but declares a {self.lookback_seconds}s lookback"
            )
        return self

    @classmethod
    def ending_at(cls, end: datetime, lookback: timedelta) -> DigestWindow:
        """The window that reaches *lookback* back from *end*."""
        return cls(
            start=end - lookback,
            end=end,
            lookback_seconds=int(lookback.total_seconds()),
        )

    @property
    def lookback(self) -> timedelta:
        return timedelta(seconds=self.lookback_seconds)

    def covers(self, moment: datetime) -> bool:
        """Whether *moment* falls in ``(start, end]``.

        Half-open at the start on purpose: an instant exactly at the boundary
        belongs to the window before this one, and counting it in both would
        double-report an item or a candle on consecutive digests.
        """
        return self.start < moment <= self.end


class PriceReference(StrictModel):
    """One boundary price, with the candle it came from.

    The timestamps are what make the number auditable. "The price at the start
    of the window was 4323" is unfalsifiable on its own; "the M5 candle opening
    at 09:35 and closing at 09:40 closed at 4323" can be checked against the
    series.
    """

    candle_open_at: UtcDatetime = Field(description="Open time of the reference candle.")
    candle_close_at: UtcDatetime = Field(description="When that candle closed.")
    close: Price = Field(description="That candle's close. The boundary price.")


class PriceReaction(StrictModel):
    """What price did across a digest window, computed rather than described.

    Every derived figure is optional, and absent means *not measured* rather
    than zero. A weekend window has no change to report, and reporting ``0``
    would be a claim that price held steady - which nobody observed.
    """

    schema_version: str = Field(default=DIGEST_SCHEMA_VERSION)

    window: DigestWindow
    symbol: str = Field(description="Canonical instrument symbol.")
    timeframe: Timeframe = Field(description="Which series the arithmetic ran over.")
    provider: str | None = Field(
        default=None,
        description=(
            "Which provider supplied the candles. Audit metadata only: the same "
            "bars from any provider must produce the same figures, and no public "
            "rendering ever names it."
        ),
    )

    market_activity: MarketActivity

    start_reference: PriceReference | None = Field(
        default=None, description="The last known closed price at or before the window start."
    )
    end_reference: PriceReference | None = Field(
        default=None, description="The last closed price at or before the window end."
    )

    window_high: Price | None = Field(
        default=None, description="Highest high among candles overlapping the window."
    )
    window_low: Price | None = Field(
        default=None, description="Lowest low among candles overlapping the window."
    )

    net_change: Decimal | None = Field(
        default=None,
        description="end close - start close. The start-to-end move. NEVER the range.",
    )
    price_range: Decimal | None = Field(
        default=None,
        description="window high - window low. The biên độ. NEVER the start-to-end move.",
    )
    percent_change: Decimal | None = Field(
        default=None, description="net_change / start close * 100. Unrounded."
    )

    closed_bars_in_window: int = Field(
        default=0, ge=0, description="How many candles closed inside the window."
    )
    overlapping_bars: int = Field(
        default=0, ge=0, description="How many candles' intervals touch the window."
    )

    @model_validator(mode="after")
    def _figures_match_the_state(self) -> PriceReaction:
        """A measured change requires both boundaries, and vice versa.

        The pairing is what stops a partially-filled object from being read as
        a complete one - an ``end_reference`` with no ``net_change`` would look
        like a move of zero to anything that only checked for a price.
        """
        measured = self.net_change is not None
        if measured and (self.start_reference is None or self.end_reference is None):
            raise ValueError("a net change requires both boundary references")
        if self.market_activity is MarketActivity.INSUFFICIENT_HISTORY and self.start_reference:
            raise ValueError("insufficient history cannot carry a start reference")
        if measured and self.market_activity is not MarketActivity.NORMAL:
            raise ValueError(
                f"{self.market_activity} states no measurable move, but a net change is present"
            )
        if (self.window_high is None) != (self.window_low is None):
            raise ValueError("a window range needs both a high and a low")
        return self

    @property
    def direction(self) -> int:
        """``1`` up, ``-1`` down, ``0`` flat or not measured."""
        if self.net_change is None:
            return 0
        if self.net_change > 0:
            return 1
        return -1 if self.net_change < 0 else 0


__all__ = [
    "DIGEST_HARD_CAP_CHARS",
    "DIGEST_SCHEMA_VERSION",
    "DIGEST_TARGET_MAX_CHARS",
    "DIGEST_TARGET_MIN_CHARS",
    "DigestWindow",
    "MarketActivity",
    "PriceReaction",
    "PriceReference",
]
