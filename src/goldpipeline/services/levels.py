"""Deterministic technical measurements over closed bars.

Everything here is arithmetic. No model is consulted, no configuration reaches
it from a payload, and no number is ever estimated: given the same bars and the
same parameters, this module returns the same values, and an auditor holding
``context.json`` can recompute every one of them by hand.

**Why that matters more than the algorithms.** The writer is forbidden to invent
price levels, and the claim/precheck machinery enforces that by resolving every
cited path against the context. Levels the model may cite therefore have to
*exist* in the context first - so the useful question is not "which indicator is
best" but "which numbers can be defended". That biases every choice below toward
the reproducible and away from the clever.

Three consequences worth naming:

* **Closed bars only.** The market-data adapter already refuses the forming
  candle; nothing here re-derives that guarantee, it simply never looks beyond
  the bars it was handed. A candle still forming cannot move an ATR, a pivot, a
  zone, a structure label or a window extreme, because it is not in the list.
* **A simple mean, not Wilder.** See :func:`average_true_range`.
* **Absent beats approximate.** Too few bars yields ``None`` and an empty list,
  never a value computed from a shorter window and quietly presented as though
  it were the requested one.

Nothing here decides anything about trading. There are no entries, stops,
targets or recommendations, and a "zone" is a place where pivots clustered - not
advice about it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

from goldpipeline.schemas.context import (
    ContextLevels,
    MarketStructure,
    PriceZone,
    SwingPoint,
    ZoneKind,
)
from goldpipeline.schemas.market import OHLCBar

logger = logging.getLogger(__name__)

DEFAULT_ATR_PERIOD = 14
"""Bars averaged for the ATR.

The conventional value, and it fits: the pipeline's default window is 20 bars,
which yields 19 true ranges - enough for a 14-period average with room to spare.
"""

DEFAULT_PIVOT_WINDOW = 2
"""Bars required either side of a confirmed pivot.

Two is the smallest window that ignores single-bar noise while still finding
pivots inside a 20-bar context. Larger windows are supported and find fewer,
stronger pivots; on a short window they find none at all, which is why this is
bounded rather than free.
"""

MIN_PIVOT_WINDOW = 1
MAX_PIVOT_WINDOW = 5

DEFAULT_CLUSTER_ATR_FRACTION = Decimal("0.5")
"""How close two pivots must be to belong to one zone, as a fraction of ATR.

Expressed in ATR rather than points because a fixed distance is wrong for every
instrument except the one it was tuned on. When ATR is unavailable there is no
defensible threshold, so no zones are produced at all.
"""

DEFAULT_MAX_ZONES_PER_SIDE = 4
"""Bound on how many candidate zones each side may report.

The nearest to the latest close are kept: a zone thirty ATR away is arithmetic,
not information, and an unbounded list would be padding for a prompt that has to
stay small.
"""

MAX_SWING_POINTS = 12
"""Bound on reported pivots per side, most recent first."""


@dataclass(frozen=True)
class LevelSettings:
    """Parameters for one computation. Bounded, and never read from a payload."""

    atr_period: int = DEFAULT_ATR_PERIOD
    pivot_window: int = DEFAULT_PIVOT_WINDOW
    cluster_atr_fraction: Decimal = DEFAULT_CLUSTER_ATR_FRACTION
    max_zones_per_side: int = DEFAULT_MAX_ZONES_PER_SIDE

    def validated(self) -> LevelSettings:
        """Clamp the pivot window into its supported range."""
        window = max(MIN_PIVOT_WINDOW, min(MAX_PIVOT_WINDOW, self.pivot_window))
        if window == self.pivot_window:
            return self
        return LevelSettings(
            atr_period=self.atr_period,
            pivot_window=window,
            cluster_atr_fraction=self.cluster_atr_fraction,
            max_zones_per_side=self.max_zones_per_side,
        )


# --------------------------------------------------------------------------
# ATR
# --------------------------------------------------------------------------


def true_ranges(bars: list[OHLCBar]) -> list[Decimal]:
    """True range for every bar that has a predecessor.

    ``TR = max(high - low, |high - prev_close|, |low - prev_close|)``

    The first bar has no previous close and therefore no true range - it is
    omitted rather than approximated by ``high - low``, which would silently
    understate a gap.
    """
    ranges: list[Decimal] = []
    for previous, current in zip(bars, bars[1:], strict=False):
        prev_close = previous.close
        ranges.append(
            max(
                current.high - current.low,
                abs(current.high - prev_close),
                abs(current.low - prev_close),
            )
        )
    return ranges


def average_true_range(bars: list[OHLCBar], period: int = DEFAULT_ATR_PERIOD) -> Decimal | None:
    """Mean of the last *period* true ranges, or ``None`` if there are too few.

    **A simple mean, not Wilder's smoothing, and the reason is auditability.**
    Wilder's ATR is recursive: today's value depends on yesterday's, back to a
    seed outside this window. Two people with the same ``context.json`` would get
    different answers depending on how much history each happened to hold, and
    nobody could check the number from the Run alone. A mean over the last
    *period* true ranges is fully determined by the bars in the context, so a
    claim citing ``context.levels.atr`` is verifiable by re-running the sum.

    Exact throughout: ``Decimal`` in, ``Decimal`` out, and the division is the
    only rounding, applied once at the end.
    """
    if period < 1:
        return None
    ranges = true_ranges(bars)
    if len(ranges) < period:
        # Deliberately not "average what we have". A 6-bar mean labelled as a
        # 14-period ATR is a wrong number wearing a right name.
        return None
    window = ranges[-period:]
    return sum(window, Decimal(0)) / Decimal(period)


# --------------------------------------------------------------------------
# pivots
# --------------------------------------------------------------------------


def swing_highs(bars: list[OHLCBar], window: int = DEFAULT_PIVOT_WINDOW) -> list[SwingPoint]:
    """Confirmed fractal highs, oldest first.

    A bar at index ``i`` is a swing high when its high is **strictly greater**
    than every high in the *window* bars before it, and **greater than or equal
    to** every high in the *window* bars after it.

    The asymmetry is how ties are resolved. Two adjacent equal highs would
    otherwise both qualify or both fail; with strict-left and non-strict-right
    exactly the earlier one is a pivot, always, on every platform. Determinism
    here is not cosmetic - a pivot set that varies would move zones, which would
    move the levels a writer is allowed to cite.

    Only indices ``[window, len(bars) - 1 - window]`` are eligible, so the last
    *window* bars can never produce a pivot. That is what "confirmed" means: no
    future bar can revoke one of these.
    """
    return _pivots(bars, window, high=True)


def swing_lows(bars: list[OHLCBar], window: int = DEFAULT_PIVOT_WINDOW) -> list[SwingPoint]:
    """Confirmed fractal lows, oldest first. Mirror of :func:`swing_highs`."""
    return _pivots(bars, window, high=False)


def _pivots(bars: list[OHLCBar], window: int, *, high: bool) -> list[SwingPoint]:
    if window < 1 or len(bars) < 2 * window + 1:
        return []

    found: list[SwingPoint] = []
    for index in range(window, len(bars) - window):
        bar = bars[index]
        value = bar.high if high else bar.low
        left = [(b.high if high else b.low) for b in bars[index - window : index]]
        right = [(b.high if high else b.low) for b in bars[index + 1 : index + 1 + window]]

        if high:
            confirmed = all(value > other for other in left) and all(
                value >= other for other in right
            )
        else:
            confirmed = all(value < other for other in left) and all(
                value <= other for other in right
            )

        if confirmed:
            found.append(SwingPoint(bar_index=index, timestamp=bar.timestamp, price=value))
    return found


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------


def classify_structure(highs: list[SwingPoint], lows: list[SwingPoint]) -> MarketStructure:
    """Label the trend from the two most recent confirmed pivots on each side.

    The whole rule, stated so it can be argued with:

    * fewer than two highs **or** fewer than two lows - ``INSUFFICIENT_DATA``;
    * last high > previous high **and** last low > previous low - ``BULLISH``;
    * last high < previous high **and** last low < previous low - ``BEARISH``;
    * anything else, including any equality - ``RANGE``.

    Both sides must agree. A market making higher highs while also making lower
    lows is broadening, not trending, and calling that bullish because one half
    of the pattern fits is the kind of confident wrongness this layer exists to
    avoid. Equality falls to ``RANGE`` for the same reason: an equal high is not
    a higher high.
    """
    if len(highs) < 2 or len(lows) < 2:
        return MarketStructure.INSUFFICIENT_DATA

    higher_high = highs[-1].price > highs[-2].price
    higher_low = lows[-1].price > lows[-2].price
    lower_high = highs[-1].price < highs[-2].price
    lower_low = lows[-1].price < lows[-2].price

    if higher_high and higher_low:
        return MarketStructure.BULLISH
    if lower_high and lower_low:
        return MarketStructure.BEARISH
    return MarketStructure.RANGE


# --------------------------------------------------------------------------
# candidate zones
# --------------------------------------------------------------------------


def cluster_zones(
    pivots: list[SwingPoint],
    *,
    kind: ZoneKind,
    threshold: Decimal,
) -> list[PriceZone]:
    """Group pivots whose prices sit within *threshold* of each other.

    Single-linkage over prices sorted ascending: walk in order and start a new
    cluster whenever the gap from the previous price exceeds the threshold. That
    yields clusters which cannot overlap - each price belongs to exactly one -
    so no normalisation pass is needed to remove duplicates or intersections.

    Bounds are observed pivot prices. Nothing is widened to ATR, padded, or
    rounded to look tidy: a zone that claims more territory than the market
    printed is an invented level with extra steps.
    """
    if not pivots or threshold < 0:
        return []

    ordered = sorted(pivots, key=lambda p: (p.price, p.timestamp))
    clusters: list[list[SwingPoint]] = [[ordered[0]]]
    for pivot in ordered[1:]:
        if pivot.price - clusters[-1][-1].price <= threshold:
            clusters[-1].append(pivot)
        else:
            clusters.append([pivot])

    zones: list[PriceZone] = []
    for cluster in clusters:
        prices = [p.price for p in cluster]
        stamps = sorted(p.timestamp for p in cluster)
        lower, upper = min(prices), max(prices)
        zones.append(
            PriceZone(
                kind=kind,
                lower=lower,
                upper=upper,
                width=upper - lower,
                pivot_count=len(cluster),
                first_timestamp=stamps[0],
                last_timestamp=stamps[-1],
            )
        )
    return zones


def _nearest(zones: list[PriceZone], reference: Decimal, limit: int) -> list[PriceZone]:
    """Keep the *limit* zones closest to *reference*, then sort by price.

    Two passes on purpose: proximity decides *which* survive, price decides the
    order they are reported in. Reporting them in proximity order would make the
    list read as a ranking, which it is not.
    """

    def distance(zone: PriceZone) -> Decimal:
        if zone.lower <= reference <= zone.upper:
            return Decimal(0)
        return min(abs(reference - zone.lower), abs(reference - zone.upper))

    kept = sorted(zones, key=lambda z: (distance(z), z.lower))[:limit]
    return sorted(kept, key=lambda z: z.lower)


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def build_levels(
    bars: list[OHLCBar],
    settings: LevelSettings | None = None,
) -> ContextLevels:
    """Compute every deterministic measurement for *bars*.

    Never raises on thin or degenerate input. A window too short for an ATR
    still produces a valid block - with ``atr=None``, empty pivot lists and
    ``structure=INSUFFICIENT_DATA`` - because one optional measurement being
    unavailable is not a reason to fail a Run whose article may not need it.
    Malformed bars cannot reach here: ``OHLCBar`` enforces its own invariants at
    construction, and the normalizer has already rejected duplicate or
    out-of-order timestamps.
    """
    resolved = (settings or LevelSettings()).validated()

    if not bars:
        return ContextLevels(
            bars_considered=0,
            atr_period=resolved.atr_period,
            pivot_window=resolved.pivot_window,
        )

    atr = average_true_range(bars, resolved.atr_period)
    highs = swing_highs(bars, resolved.pivot_window)
    lows = swing_lows(bars, resolved.pivot_window)

    support: list[PriceZone] = []
    resistance: list[PriceZone] = []
    if atr is not None and atr > 0:
        threshold = atr * resolved.cluster_atr_fraction
        latest_close = bars[-1].close
        resistance = _nearest(
            cluster_zones(highs, kind=ZoneKind.RESISTANCE, threshold=threshold),
            latest_close,
            resolved.max_zones_per_side,
        )
        support = _nearest(
            cluster_zones(lows, kind=ZoneKind.SUPPORT, threshold=threshold),
            latest_close,
            resolved.max_zones_per_side,
        )

    high_bar = max(bars, key=lambda b: (b.high, -b.timestamp.timestamp()))
    low_bar = min(bars, key=lambda b: (b.low, b.timestamp.timestamp()))

    return ContextLevels(
        bars_considered=len(bars),
        atr_period=resolved.atr_period,
        atr=atr,
        pivot_window=resolved.pivot_window,
        swing_highs=highs[-MAX_SWING_POINTS:],
        swing_lows=lows[-MAX_SWING_POINTS:],
        structure=classify_structure(highs, lows),
        support_zones=support,
        resistance_zones=resistance,
        window_high=high_bar.high,
        window_high_at=high_bar.timestamp,
        window_low=low_bar.low,
        window_low_at=low_bar.timestamp,
    )


__all__ = [
    "DEFAULT_ATR_PERIOD",
    "DEFAULT_CLUSTER_ATR_FRACTION",
    "DEFAULT_MAX_ZONES_PER_SIDE",
    "DEFAULT_PIVOT_WINDOW",
    "MAX_PIVOT_WINDOW",
    "MAX_SWING_POINTS",
    "MIN_PIVOT_WINDOW",
    "LevelSettings",
    "average_true_range",
    "build_levels",
    "classify_structure",
    "cluster_zones",
    "swing_highs",
    "swing_lows",
    "true_ranges",
]
