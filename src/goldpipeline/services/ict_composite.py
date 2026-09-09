"""One market state, computed once, so nothing downstream can disagree with itself.

Round 6.6e.1. This module adds no market interpretation whatsoever. Every
number in its output comes from an authority that already existed and is
unchanged; what is new is that they are wired together so each authority runs
**once per timeframe** and every consumer sees the same answer.

**Why that matters more than it sounds.** Each engine on this branch may derive
what it needs: structure derives confirmed swings, liquidity derives them too,
protected structure can derive structure, ranges and order blocks can derive
both, and the order-block lifecycle can derive formation. Every one of those
paths is individually correct, and a future candidate engine calling all of
them independently would still get correct answers - but it would compute
"the swings" several times over and hold several objects that merely happen to
be equal today. The moment any of them gained a tie-break, a tolerance or a
cache, they could stop being equal, and nothing would have told anyone. Running
each authority once and threading the result removes the possibility rather
than testing for it.

**Orchestration, not semantics.** There is no pivot detector here, no ATR
formula, no gap definition, no break classifier, no anchor selector, no range
extender, no source-candle rule and no mitigation rule. Guards in
``tests/test_ict_composite_guards.py`` read this module's identifiers and fail
if one appears.

**Policy is stated, never inherited.** :class:`IctCompositeConfig` has no
defaults, even where the primitives underneath it do. A pivot width of 2 is a
perfectly good default for someone exploring a chart; it is not a good thing for
a published trade plan to have acquired without anyone choosing it. The config
travels with the result, so a consumer can always answer "under which policy?"
without inferring it from the geometry.

**Five independent readings, and no opinion about them.** The composite holds
one analysis per timeframe and says nothing about how they relate. There is no
overall bias, no agreement count, no confluence score and no higher-timeframe
override, because deciding when H4 governs H1 is a real question with real
content and it is not a geometry module's to answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import (
    ICT_TIMEFRAMES,
    IctMarketSnapshot,
    IctTimeframeSnapshot,
    build_timeframe_snapshot,
)
from goldpipeline.services.ict_fvg import FvgLifecycleAnalysis, analyse_fvg_lifecycle
from goldpipeline.services.ict_liquidity import (
    LiquidityAnalysis,
    LiquidityConfig,
    analyse_liquidity,
)
from goldpipeline.services.ict_order_block import (
    OrderBlockAnalysis,
    OrderBlockConfig,
    OrderBlockZoneBasis,
    analyse_order_blocks,
)
from goldpipeline.services.ict_order_block_lifecycle import (
    OrderBlockLifecycleAnalysis,
    OrderBlockLifecycleConfig,
    OrderBlockMitigationRule,
    analyse_order_block_lifecycle,
)
from goldpipeline.services.ict_primitives import (
    AtrPoint,
    FairValueGap,
    SwingPoint,
    confirmed_swings,
    fair_value_gaps,
    latest_atr,
)
from goldpipeline.services.ict_protected import (
    ProtectedStructureAnalysis,
    analyse_protected_structure,
)
from goldpipeline.services.ict_range import DealingRangeAnalysis, analyse_dealing_ranges
from goldpipeline.services.ict_structure import StructureAnalysis, analyse_structure

logger = logging.getLogger(__name__)

COMPOSITE_METHOD_VERSION = "1.0.0"
"""Stamped on every composite and per-timeframe analysis."""


class IctCompositeError(ValueError):
    """The orchestration could not honestly answer, so it refused to guess."""


@dataclass(frozen=True)
class IctCompositeConfig:
    """Every policy the ICT stack takes, named in one place.

    No defaults anywhere, deliberately. Three of these already had no default at
    their own layer - the liquidity tolerance, the order-block zone basis and
    the mitigation rule - and three did: the pivot window and the ATR period.
    Letting the latter three keep their defaults here would mean a future trade
    plan inherited a pivot width nobody argued about, sitting in the same object
    as choices that were argued about at length. So all six are stated.

    The enums are the existing ones, imported rather than redefined: a second
    ``FULL_CANDLE`` would be a second definition waiting to drift.
    """

    swing_left_bars: int
    swing_right_bars: int
    atr_period: int
    liquidity_price_tolerance: Decimal
    order_block_zone_basis: OrderBlockZoneBasis
    order_block_mitigation_rule: OrderBlockMitigationRule

    def __post_init__(self) -> None:
        if self.swing_left_bars < 1 or self.swing_right_bars < 1:
            raise IctCompositeError(
                f"pivot windows must be positive, got left={self.swing_left_bars} "
                f"right={self.swing_right_bars}"
            )
        if self.atr_period < 1:
            raise IctCompositeError(f"ATR period must be positive, got {self.atr_period}")
        if not isinstance(self.liquidity_price_tolerance, Decimal):
            raise IctCompositeError(
                "liquidity tolerance must be a Decimal; a float tolerance would make pool "
                "membership depend on binary rounding"
            )
        if self.liquidity_price_tolerance < 0:
            raise IctCompositeError(
                f"liquidity tolerance cannot be negative, got {self.liquidity_price_tolerance}"
            )

    @property
    def liquidity(self) -> LiquidityConfig:
        return LiquidityConfig(price_tolerance=self.liquidity_price_tolerance)

    @property
    def order_block(self) -> OrderBlockConfig:
        return OrderBlockConfig(zone_basis=self.order_block_zone_basis)

    @property
    def order_block_lifecycle(self) -> OrderBlockLifecycleConfig:
        return OrderBlockLifecycleConfig(mitigation_rule=self.order_block_mitigation_rule)


@dataclass(frozen=True)
class IctTimeframeAnalysis:
    """One timeframe's complete deterministic ICT state, at one instant.

    The nested analyses are the authorities' own immutable results, held rather
    than copied: flattening their fields into this object would create a second
    place for the same number to live, which is the exact failure this round
    exists to prevent.

    No candidate fields, no scores, no prose.
    """

    method_version: str
    timeframe: Timeframe
    symbol: str
    observed_at: datetime

    series: IctTimeframeSnapshot
    """The closed candles every nested analysis was computed from."""

    atr: AtrPoint | None
    """``None`` when the series is shorter than the ATR period needs, which is a
    real state and not an error."""

    swings: tuple[SwingPoint, ...]
    gaps: tuple[FairValueGap, ...]

    fvg_lifecycle: FvgLifecycleAnalysis
    structure: StructureAnalysis
    liquidity: LiquidityAnalysis
    protected: ProtectedStructureAnalysis
    dealing_ranges: DealingRangeAnalysis
    order_blocks: OrderBlockAnalysis
    order_block_lifecycle: OrderBlockLifecycleAnalysis


@dataclass(frozen=True)
class IctCompositeAnalysis:
    """One snapshot's timeframes, each analysed independently under one policy.

    Deliberately not a synthesis. There is no ``overall_bias``, no
    ``higher_timeframe_bias``, no ``confluence_score``, no ``agreement_count``
    and no ``trend_alignment`` - see the module docstring.
    """

    method_version: str
    observed_at: datetime
    symbol: str
    provider: str
    provider_symbol: str | None

    config: IctCompositeConfig
    timeframes: tuple[IctTimeframeAnalysis, ...]

    def timeframe(self, wanted: Timeframe) -> IctTimeframeAnalysis | None:
        return next((entry for entry in self.timeframes if entry.timeframe is wanted), None)

    def require(self, wanted: Timeframe) -> IctTimeframeAnalysis:
        found = self.timeframe(wanted)
        if found is None:
            raise IctCompositeError(f"this composite holds no {wanted.value} analysis")
        return found


def analyse_ict_timeframe(
    series: IctTimeframeSnapshot,
    *,
    config: IctCompositeConfig,
    symbol: str = "",
    as_of: datetime | None = None,
) -> IctTimeframeAnalysis:
    """Run every ICT authority once over one timeframe and keep all the results.

    The order is the dependency order, and each result is threaded into the
    stages that need it rather than being rederived:

    1. confirmed swings - once, into structure and liquidity;
    2. fair value gaps - once, into the gap lifecycle;
    3. Wilder ATR - once;
    4. structure, from those swings;
    5. liquidity, from those swings and the explicit tolerance;
    6. protected structure, from that structure;
    7. dealing ranges, from that structure and protected state;
    8. order blocks, from the same two plus the explicit zone basis;
    9. order-block lifecycle, from those exact blocks plus the explicit rule.

    Step 5 takes swings and **not** structure. Liquidity is structurally
    independent by design - Round 6.6c.1 settled that a pool is not consumed by
    a break of structure - and handing it a ``StructureAnalysis`` merely because
    this function has one would quietly couple them.

    Args:
        series: Closed candles for one timeframe.
        config: Every policy, stated. There is no default.
        symbol: Instrument name. Provenance, and part of several identities.
        as_of: Reconstruct the state as it stood at this instant.

    Raises:
        ValueError: *as_of* precedes every bar's close.
    """
    working = series
    if as_of is not None:
        working = build_timeframe_snapshot(
            timeframe=series.timeframe, bars=series.bars, observed_at=as_of
        )
    observed_at = as_of if as_of is not None else series.latest_closed_at

    # (1)-(3): the three raw authorities, once each.
    swings = tuple(
        confirmed_swings(
            working, left_bars=config.swing_left_bars, right_bars=config.swing_right_bars
        )
    )
    gaps = tuple(fair_value_gaps(working))
    atr = latest_atr(working, config.atr_period)

    # (4)-(9): each derived authority once, fed the results above.
    structure = analyse_structure(
        working,
        swings=swings,
        symbol=symbol,
        as_of=observed_at,
        left_bars=config.swing_left_bars,
        right_bars=config.swing_right_bars,
    )
    liquidity = analyse_liquidity(
        working,
        config=config.liquidity,
        swings=swings,
        symbol=symbol,
        as_of=observed_at,
        left_bars=config.swing_left_bars,
        right_bars=config.swing_right_bars,
    )
    protected = analyse_protected_structure(
        working,
        structure=structure,
        symbol=symbol,
        as_of=observed_at,
        left_bars=config.swing_left_bars,
        right_bars=config.swing_right_bars,
    )
    dealing_ranges = analyse_dealing_ranges(
        working,
        structure=structure,
        protected=protected,
        symbol=symbol,
        as_of=observed_at,
        left_bars=config.swing_left_bars,
        right_bars=config.swing_right_bars,
    )
    order_blocks = analyse_order_blocks(
        working,
        config=config.order_block,
        structure=structure,
        protected=protected,
        symbol=symbol,
        as_of=observed_at,
        left_bars=config.swing_left_bars,
        right_bars=config.swing_right_bars,
    )
    order_block_lifecycle = analyse_order_block_lifecycle(
        working,
        config=config.order_block_lifecycle,
        order_blocks=order_blocks,
        symbol=symbol,
        as_of=observed_at,
        left_bars=config.swing_left_bars,
        right_bars=config.swing_right_bars,
    )
    fvg_lifecycle = analyse_fvg_lifecycle(working, gaps=gaps, symbol=symbol, as_of=observed_at)

    return IctTimeframeAnalysis(
        method_version=COMPOSITE_METHOD_VERSION,
        timeframe=working.timeframe,
        symbol=symbol,
        observed_at=observed_at,
        series=working,
        atr=atr,
        swings=swings,
        gaps=gaps,
        fvg_lifecycle=fvg_lifecycle,
        structure=structure,
        liquidity=liquidity,
        protected=protected,
        dealing_ranges=dealing_ranges,
        order_blocks=order_blocks,
        order_block_lifecycle=order_block_lifecycle,
    )


def analyse_ict_composite(
    snapshot: IctMarketSnapshot, *, config: IctCompositeConfig
) -> IctCompositeAnalysis:
    """Analyse every timeframe the snapshot carries, under one policy.

    All timeframes are analysed as of ``snapshot.observed_at`` exactly - not
    each series' own last close, which can differ between timeframes and would
    silently date the five readings differently. No stage reads a clock and no
    stage chooses its own instant.

    Timeframes are analysed in :data:`ICT_TIMEFRAMES` order, filtered to those
    actually present. A snapshot carrying a subset is a legitimate input and is
    analysed as it stands: nothing here resamples, back-fills or invents a
    timeframe the provider did not answer with.

    Raises:
        IctCompositeError: A nested analysis reports a different instant or
            symbol from the snapshot's.
    """
    present = {series.timeframe: series for series in snapshot.timeframes}
    ordered = tuple(present[tf] for tf in ICT_TIMEFRAMES if tf in present)

    analyses = tuple(
        analyse_ict_timeframe(
            series, config=config, symbol=snapshot.symbol, as_of=snapshot.observed_at
        )
        for series in ordered
    )
    for entry in analyses:
        _require_consistent(entry, snapshot)

    return IctCompositeAnalysis(
        method_version=COMPOSITE_METHOD_VERSION,
        observed_at=snapshot.observed_at,
        symbol=snapshot.symbol,
        provider=snapshot.provider,
        provider_symbol=snapshot.provider_symbol,
        config=config,
        timeframes=analyses,
    )


def _require_consistent(entry: IctTimeframeAnalysis, snapshot: IctMarketSnapshot) -> None:
    """Every nested analysis must describe the snapshot's instant and symbol.

    Checked rather than assumed. The threading above should make it impossible,
    which is exactly why a silent disagreement would be so hard to notice: the
    numbers would all look plausible and only their dates would be wrong.
    """
    nested: tuple[tuple[str, datetime, str], ...] = (
        ("fvg lifecycle", entry.fvg_lifecycle.observed_at, entry.fvg_lifecycle.symbol),
        ("structure", entry.structure.observed_at, entry.structure.symbol),
        ("liquidity", entry.liquidity.observed_at, entry.liquidity.symbol),
        ("protected structure", entry.protected.observed_at, entry.protected.symbol),
        ("dealing ranges", entry.dealing_ranges.observed_at, entry.dealing_ranges.symbol),
        ("order blocks", entry.order_blocks.observed_at, entry.order_blocks.symbol),
        (
            "order block lifecycle",
            entry.order_block_lifecycle.observed_at,
            entry.order_block_lifecycle.symbol,
        ),
    )
    for what, observed_at, symbol in (
        (entry.timeframe.value, entry.observed_at, entry.symbol),
        *nested,
    ):
        if observed_at != snapshot.observed_at:
            raise IctCompositeError(
                f"{entry.timeframe.value} {what} is dated {observed_at.isoformat()} but the "
                f"snapshot is {snapshot.observed_at.isoformat()}"
            )
        if symbol != snapshot.symbol:
            raise IctCompositeError(
                f"{entry.timeframe.value} {what} is for {symbol!r}, not {snapshot.symbol!r}"
            )


__all__ = [
    "COMPOSITE_METHOD_VERSION",
    "IctCompositeAnalysis",
    "IctCompositeConfig",
    "IctCompositeError",
    "IctTimeframeAnalysis",
    "analyse_ict_composite",
    "analyse_ict_timeframe",
]
