"""What price did across a digest window. Arithmetic, not commentary.

Round 6.5a. One function, no model, no provider.

**Two different questions, and the whole module exists to keep them apart.**
"How much did gold move?" is ``end - start``. "How far did it swing?" is
``high - low``. They are both distances in price units, they are both correct
answers to *a* question, and a piece that reports one under the other's name is
wrong in a way no reader can catch. Round 6.4's design named that failure; this
is where the two are computed separately and typed separately so a later stage
cannot confuse them by accident.

**Closed candles only.** A forming candle has a close that is not yet a fact,
and reading one gives an article a price that changes after publication. A
candle counts as closed when ``open_time + duration <= end`` - the same
arithmetic rule Round 6.4b established for TradingView, restated here in terms
of the normalized series rather than any provider's own "completed" flag.

**Boundaries are found, never invented.** A window rarely begins on a candle
close. The rule is to take the last price that was actually known at the
boundary, and to say so when there was none - never to reach forward to the
first candle after the boundary and present its open as though it were the
price at the boundary, which is a different number about a different instant.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.digest import (
    DigestWindow,
    MarketActivity,
    PriceReaction,
    PriceReference,
)
from goldpipeline.schemas.market import OHLCBar

logger = logging.getLogger(__name__)

PREFERRED_DIGEST_TIMEFRAME = Timeframe.M5
"""The timeframe a digest would ask for, when a later round wires a fetch.

Not used by anything here - the calculator is timeframe-neutral and takes
whatever series it is handed. It is recorded because the choice has a reason
worth keeping: at five minutes a one-hour window still has twelve bars to say
something about, and seven days is roughly 2016 - comfortably under the
provider's own ceiling. M15, which ANALYSIS uses, would leave a one-hour digest
with four bars. Nothing about ANALYSIS's timeframe is evidence about this one.
"""


PROVIDER_BAR_CEILING = 5000
"""The hard limit a request may never exceed, whatever the arithmetic says."""

BOUNDARY_HISTORY_BARS = 2
"""Extra bars before the window, so a start reference always exists.

One would do when the window begins exactly on a close. Two covers the ordinary
case where it begins mid-bar, and costs nothing.
"""

GAP_MARGIN_BARS = 12
"""Slack for the forming bar and for holes in the series.

An hour of M5. Not a guess about how long a market is shut - a weekend is far
longer than this, and a digest spanning one correctly reports
``NO_MARKET_ACTIVITY`` rather than being rescued by a bigger request. This
covers the two ordinary cases: the bar still running at the window end, and a
handful of bars a provider did not return.
"""


def digest_bar_count(window: DigestWindow, timeframe: Timeframe) -> int:
    """How many bars to ask a provider for, to describe *window*.

    ``ceil(window / bar) + boundary history + margin``, capped at the provider
    ceiling. Bounded on both ends and derived from the window rather than fixed:
    asking for 5000 bars to describe one hour wastes a request that a provider
    may rate-limit, and asking for a fixed 300 silently truncates a seven-day
    window into a digest describing the wrong period.

    For the extremes: a one-hour M5 window asks for 26, and a seven-day one for
    2030 - comfortably inside the ceiling, which is therefore a guard rather
    than a routine clamp.

    Raises:
        ValueError: The timeframe has no fixed duration, so "how many bars is
            this window" has no answer.
    """
    duration = timeframe.duration
    if duration is None:
        raise ValueError(f"{timeframe} has no fixed duration; a bar count cannot be derived")

    span = window.end - window.start
    bars = -(-int(span.total_seconds()) // int(duration.total_seconds()))  # ceil
    return min(bars + BOUNDARY_HISTORY_BARS + GAP_MARGIN_BARS, PROVIDER_BAR_CEILING)


class InsufficientHistoryError(ValueError):
    """Raised only by callers that require a measurable window.

    The calculator itself does not raise. A weekend is an ordinary state of the
    world, not a failure, and forcing every caller to wrap an expected outcome
    in ``try`` is how expected outcomes end up being swallowed. This exists for
    a future caller that genuinely cannot proceed without a change.
    """


def close_time(bar: OHLCBar, timeframe: Timeframe) -> datetime:
    """When *bar* finished, by arithmetic over its open time.

    Raises:
        ValueError: The timeframe has no fixed duration. Calendar-based
            timeframes - a month is not a fixed number of seconds - cannot have
            a close derived this way, and guessing one would put a candle's
            close on the wrong side of a boundary.
    """
    duration = timeframe.duration
    if duration is None:
        raise ValueError(f"{timeframe} has no fixed duration; a close time cannot be derived")
    closes: datetime = bar.timestamp + duration
    return closes


def calculate_price_reaction(
    bars: Sequence[OHLCBar],
    *,
    timeframe: Timeframe,
    window: DigestWindow,
    symbol: str,
    provider: str | None = None,
) -> PriceReaction:
    """Everything deterministic that can be said about price across *window*.

    Args:
        bars: A normalized series, ascending by open time. Any provider's, as
            long as it is the pipeline's own :class:`OHLCBar`.
        timeframe: Which series these are. Decides when each candle closed.
        window: The immutable span being described.
        symbol: Canonical instrument symbol, for the record.
        provider: Who supplied the candles. Audit metadata; never rendered.

    Returns:
        A :class:`PriceReaction`. An unmeasurable window is an ordinary result
        with a state explaining why, not an exception - see
        :class:`InsufficientHistoryError`.
    """
    ordered = sorted(bars, key=lambda bar: bar.timestamp)
    closed = [bar for bar in ordered if close_time(bar, timeframe) <= window.end]

    start_ref = _reference_at(closed, timeframe, boundary=window.start)
    end_ref = _reference_at(closed, timeframe, boundary=window.end)

    overlapping = [bar for bar in closed if _overlaps(bar, timeframe, window)]
    in_window = [bar for bar in closed if window.covers(close_time(bar, timeframe))]

    # Every figure below is computed from *closed* bars. This one signal reads
    # the whole series, and only to tell two silences apart: a window that had
    # a bar running in it and finished none, versus one where no bar was open
    # at all. Without it the two collapse, and a digest would tell a reader the
    # market was shut when it was merely mid-candle.
    any_bar_overlaps = any(_overlaps(bar, timeframe, window) for bar in ordered)

    common = {
        "window": window,
        "symbol": symbol,
        "timeframe": timeframe,
        "provider": provider,
        "closed_bars_in_window": len(in_window),
        "overlapping_bars": len(overlapping),
    }

    if start_ref is None:
        # No price was known at the start boundary. Anything reported as a
        # change would be a change from a moment nobody asked about.
        logger.info(
            "price_reaction state=INSUFFICIENT_HISTORY symbol=%s timeframe=%s bars=%d",
            symbol,
            timeframe,
            len(ordered),
        )
        return PriceReaction(market_activity=MarketActivity.INSUFFICIENT_HISTORY, **common)

    high, low = _extremes(overlapping)

    if not in_window:
        # A start price exists and nothing has finished since. The distinction
        # below is between "a bar is running" and "no bar was open at all", and
        # both are reported without a number rather than as a move of zero.
        state = (
            MarketActivity.NO_NEW_CLOSED_BAR
            if any_bar_overlaps
            else MarketActivity.NO_MARKET_ACTIVITY
        )
        logger.info(
            "price_reaction state=%s symbol=%s overlapping=%d", state, symbol, len(overlapping)
        )
        return PriceReaction(
            market_activity=state,
            start_reference=start_ref,
            end_reference=end_ref,
            window_high=high,
            window_low=low,
            **common,
        )

    assert end_ref is not None  # noqa: S101 - a closed in-window bar guarantees one
    net = end_ref.close - start_ref.close
    percent = (net / start_ref.close) * Decimal(100) if start_ref.close else None

    return PriceReaction(
        market_activity=MarketActivity.NORMAL,
        start_reference=start_ref,
        end_reference=end_ref,
        window_high=high,
        window_low=low,
        net_change=net,
        price_range=(high - low) if high is not None and low is not None else None,
        percent_change=percent,
        **common,
    )


def _reference_at(
    closed: Sequence[OHLCBar], timeframe: Timeframe, *, boundary: datetime
) -> PriceReference | None:
    """The last candle to have closed at or before *boundary*.

    "At or before", not "the nearest": a candle closing exactly on the boundary
    is the price at that boundary, and one closing a second later is not yet
    known there.
    """
    best: OHLCBar | None = None
    best_close: datetime | None = None
    for bar in closed:
        closes = close_time(bar, timeframe)
        if closes <= boundary and (best_close is None or closes > best_close):
            best, best_close = bar, closes
    if best is None:
        return None
    return PriceReference(
        candle_open_at=best.timestamp,
        candle_close_at=close_time(best, timeframe),
        close=best.close,
    )


def _overlaps(bar: OHLCBar, timeframe: Timeframe, window: DigestWindow) -> bool:
    """Whether *bar*'s trading interval touches the window.

    The rule, stated once and relied on everywhere: a bar occupies
    ``[open_time, close_time)`` and belongs to the window when that interval
    intersects ``(window.start, window.end]``. Concretely::

        close_time > window.start  and  open_time < window.end

    A bar that closed exactly at the start boundary is excluded: it is entirely
    before the window, and it is usually the very bar that supplied the start
    reference, which must not then also contribute its high and low to the
    window's range.

    This predicate is applied to *closed* bars for every figure, so a bar still
    running at the end boundary contributes nothing to the range - its high and
    low so far are real but not final, and a range that could still widen after
    publication is not a fact. The one place the predicate is applied to the
    whole series is the state decision in
    :func:`calculate_price_reaction`, which needs to know a bar was open
    without reading anything off it.
    """
    return close_time(bar, timeframe) > window.start and bar.timestamp < window.end


def _extremes(bars: Sequence[OHLCBar]) -> tuple[Decimal | None, Decimal | None]:
    """Highest high and lowest low, or a pair of ``None`` for an empty window."""
    if not bars:
        return None, None
    return max(bar.high for bar in bars), min(bar.low for bar in bars)


__all__ = [
    "BOUNDARY_HISTORY_BARS",
    "GAP_MARGIN_BARS",
    "PREFERRED_DIGEST_TIMEFRAME",
    "PROVIDER_BAR_CEILING",
    "InsufficientHistoryError",
    "calculate_price_reaction",
    "close_time",
    "digest_bar_count",
]
