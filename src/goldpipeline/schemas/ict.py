"""One ICT observation of the market, across several timeframes at once.

Round 6.6a, and the first piece of the ``TRADE_PLAN`` branch.

**Why a snapshot type rather than a bag of series.** Every later stage of this
branch - structure, liquidity, candidate zones, the plan itself - reasons about
what several timeframes were doing *at the same moment*. H4 fetched at 14:01 and
M5 fetched at 14:06 describe two different markets, and code handed them
separately has no way to notice. So the observation instant belongs to the
snapshot, not to each series, and every bar in it is proved closed against that
one instant.

**Closed bars only, by arithmetic.** A bar is in a snapshot when
``open_time + timeframe.duration <= observed_at``. That is the same rule the
digest uses, for the same reason: a provider's "completed" flag is a claim, and
this is a fact. A forming candle cannot become a swing, fill a gap, or move an
ATR, because it never enters the model.

**Nothing here means anything yet.** A snapshot holds candles and their
provenance. It has no bias, no levels, no zones and no opinion; the primitives
in :mod:`goldpipeline.services.ict_primitives` compute observations from it,
and even those stop short of interpretation. A swing high is a shape, not
buy-side liquidity; a fair value gap is three candles in a relationship, not a
place to trade. Those readings need structure, and structure is a later round.

**H4 is stored as the venue reported it.** TradingView's XAUUSD H4 candles open
at 01/05/09/13/17/21 UTC, and this model records those timestamps unchanged.
Shifting them to 00/04/08 would be inventing a bar boundary the venue never
traded, and the closed-bar arithmetic above works on whatever grid the provider
actually uses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import Field, model_validator

from goldpipeline.schemas.common import StrictModel, Timeframe, UtcDatetime
from goldpipeline.schemas.market import OHLCBar

ICT_SCHEMA_VERSION = "1.0.0"

ICT_TIMEFRAMES: tuple[Timeframe, ...] = (
    Timeframe.H4,
    Timeframe.H1,
    Timeframe.M15,
    Timeframe.M5,
    Timeframe.M1,
)
"""The five series the eventual engine reads, highest first.

Their intended roles, recorded here as documentation and **not** implemented as
behaviour anywhere in this round: H4 and H1 carry higher-timeframe context, M15
is where setup structure is read, M5 refines it, and M1 is entry-level
refinement. No code branches on these roles yet, and none should until the
structure engine gives them something to decide.
"""


class IctTimeframeSnapshot(StrictModel):
    """One timeframe's closed candles, as of the snapshot's instant.

    The three derived fields are stored rather than recomputed on access
    because this is an artifact type: somebody reading it months later should
    not have to scan a list to answer "how much history is this, and when does
    it end?". They are cross-validated below so the redundancy can never become
    a disagreement.
    """

    timeframe: Timeframe
    bars: tuple[OHLCBar, ...] = Field(
        min_length=1, description="Closed candles, ascending by open time."
    )
    first_closed_at: UtcDatetime = Field(description="Close time of the oldest bar held.")
    latest_closed_at: UtcDatetime = Field(description="Close time of the newest bar held.")
    bar_count: int = Field(ge=1)

    @model_validator(mode="after")
    def _series_is_well_formed(self) -> Self:
        """Ascending, unique, and consistent with what the fields claim.

        Refused rather than repaired, following the market-data contract this
        project already uses: a series arriving out of order or with a repeated
        open time is not a series with a formatting problem, it is one whose
        provider disagrees with itself about what happened. Sorting it would
        hide that and hand every later primitive a silent guess.
        """
        for entry in self.bars:
            if not entry.is_utc:
                # `OHLCBar` tolerates a naive timestamp at input; the snapshot
                # layer is where the project has always required awareness, and
                # this follows `MarketDataSnapshot` rather than inventing a
                # second convention. A naive bar compared against an aware
                # observation would raise a TypeError deep in the arithmetic
                # instead of being refused here, by name.
                raise ValueError(f"bar {entry.timestamp} is not timezone-aware UTC")

        times = [bar.timestamp for bar in self.bars]

        if any(later <= earlier for earlier, later in zip(times, times[1:], strict=False)):
            raise ValueError(
                f"{self.timeframe} bars must ascend strictly by open time; "
                "a repeated or out-of-order timestamp is refused rather than sorted"
            )

        if self.bar_count != len(self.bars):
            raise ValueError(
                f"{self.timeframe} declares {self.bar_count} bars but holds {len(self.bars)}"
            )

        duration = self.timeframe.duration
        if duration is None:
            raise ValueError(
                f"{self.timeframe} has no fixed duration; an ICT snapshot cannot prove "
                "its bars are closed"
            )

        if self.first_closed_at != times[0] + duration:
            raise ValueError("first_closed_at does not match the oldest bar's close")
        if self.latest_closed_at != times[-1] + duration:
            raise ValueError("latest_closed_at does not match the newest bar's close")

        return self

    @property
    def closes_at(self) -> tuple[datetime, ...]:
        """Every bar's close time, by arithmetic over its open."""
        duration = self.timeframe.duration
        assert duration is not None  # guaranteed by the validator above
        return tuple(bar.timestamp + duration for bar in self.bars)


class IctMarketSnapshot(StrictModel):
    """Several timeframes of one instrument, all as of one instant.

    Immutable, like every other authority in this pipeline. A later stage that
    wanted "one more bar" has to take a new observation rather than growing
    this one, because a series that can grow is one whose primitives can change
    answer without anything visibly happening.
    """

    schema_version: str = Field(default=ICT_SCHEMA_VERSION)

    observed_at: UtcDatetime = Field(
        description=(
            "The single instant every series is current to. Every bar in every "
            "timeframe closed at or before this."
        )
    )
    symbol: str = Field(description="Canonical instrument symbol, e.g. 'XAUUSD'.")
    provider: str = Field(description="Which source answered, e.g. 'tradingview'.")
    provider_symbol: str | None = Field(
        default=None,
        description=(
            "The venue's own name for the instrument, e.g. 'OANDA:XAUUSD'. "
            "Provenance only: the same candles from any provider must produce "
            "identical primitives, and no primitive reads this field."
        ),
    )
    timeframes: tuple[IctTimeframeSnapshot, ...] = Field(
        min_length=1, description="One entry per timeframe, in no required order."
    )

    @model_validator(mode="after")
    def _one_instant_one_market(self) -> Self:
        """Each timeframe appears once, and nothing in it is still forming.

        The closed-bar proof is the point. It is done here rather than trusted
        from the adapter because this model is what every later stage reads: a
        primitive holding a snapshot needs no further evidence that the candle
        it is looking at had finished.
        """
        seen = [entry.timeframe for entry in self.timeframes]
        duplicates = sorted({str(tf) for tf in seen if seen.count(tf) > 1})
        if duplicates:
            raise ValueError(f"a timeframe may appear once: {', '.join(duplicates)}")

        for entry in self.timeframes:
            if entry.latest_closed_at > self.observed_at:
                raise ValueError(
                    f"{entry.timeframe} holds a bar closing at {entry.latest_closed_at}, "
                    f"after the observation at {self.observed_at}; a forming candle is "
                    "not part of an observation"
                )
        return self

    def series(self, timeframe: Timeframe) -> IctTimeframeSnapshot | None:
        """The snapshot for *timeframe*, or ``None`` when it was not observed."""
        return next((entry for entry in self.timeframes if entry.timeframe is timeframe), None)

    def require(self, timeframe: Timeframe) -> IctTimeframeSnapshot:
        """The snapshot for *timeframe*.

        Raises:
            KeyError: It was not observed. Raised rather than returning an empty
                series, because "no bars" and "never looked" are different
                facts and a primitive given the first would compute a confident
                nothing.
        """
        found = self.series(timeframe)
        if found is None:
            raise KeyError(f"{timeframe} is not in this snapshot")
        return found

    @property
    def observed_timeframes(self) -> tuple[Timeframe, ...]:
        return tuple(entry.timeframe for entry in self.timeframes)


def build_timeframe_snapshot(
    *,
    timeframe: Timeframe,
    bars: tuple[OHLCBar, ...],
    observed_at: datetime,
) -> IctTimeframeSnapshot:
    """Keep the bars of *bars* that had closed by *observed_at*.

    The one place a series is filtered, so "which bars count" has a single
    answer. Bars are dropped, never adjusted: a candle that had not finished is
    excluded rather than truncated to the observation instant, because a
    truncated candle is a price that never traded.

    Raises:
        ValueError: Nothing had closed yet. An empty series would let a
            primitive report "no swings" when the truth is "no data".
    """
    duration = timeframe.duration
    if duration is None:
        raise ValueError(f"{timeframe} has no fixed duration; its bars cannot be proved closed")

    closed = tuple(bar for bar in bars if bar.timestamp + duration <= observed_at)
    if not closed:
        raise ValueError(f"{timeframe} has no bar closed at or before {observed_at.isoformat()}")

    return IctTimeframeSnapshot(
        timeframe=timeframe,
        bars=closed,
        first_closed_at=closed[0].timestamp + duration,
        latest_closed_at=closed[-1].timestamp + duration,
        bar_count=len(closed),
    )


__all__ = [
    "ICT_SCHEMA_VERSION",
    "ICT_TIMEFRAMES",
    "IctMarketSnapshot",
    "IctTimeframeSnapshot",
    "build_timeframe_snapshot",
]
