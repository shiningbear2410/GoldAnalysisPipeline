"""One price the whole market is measured against, and what that rules out.

Round 6.6e.2b, the second candidate-layer round. It answers three questions and
refuses everything after them: what is *the* current price, what role could each
projected source play, and which sources are ruled out by a named deterministic
fact.

**One reference price, not five.** Every timeframe has a last closed candle, and
using each timeframe's own would give five incompatible meanings of "current
price" - an H4 candidate measured against a four-hour-old close while an M1
candidate is measured against a one-minute-old one. So the authority is global:
the latest close time across every supplied timeframe, and the price observed
there. If two timeframes both close at that instant with different prices the
resolution **fails closed**, because one symbol cannot truthfully have two
closing prices at one instant, and quietly preferring the finer timeframe would
bury a data fault under a plausible number.

**Terminal truth is not configurable.** A caller chooses which *non-terminal*
statuses count as eligible - that is a real product decision and it has no
default here. What no caller may do is configure ``INVALIDATED`` or ``FILLED``
back in: those are deterministic facts the lifecycle engines established, and a
config that could override them would make policy outrank evidence.

**Reference price is context, not lifecycle authority.** The global price may be
newer than an H4 series' last close. That does not let this module rewrite one
H4 order block's status. Lifecycle belongs to the engines that computed it;
this layer may only decide where a source sits relative to price and whether
that makes it usable.

**Liquidity is a level, not an entry.** Buy-side liquidity sitting above price
is not a sell entry, and calling it ``SEO`` because of which side it rests on
would be a category error. Pools become ``UPPER_REFERENCE`` or
``LOWER_REFERENCE`` - somewhere price may be drawn to - and only order blocks
and gaps become entry zones with a ``BAI``/``SEO`` side.

**Every source gets a decision.** Nothing is dropped: a terminal source receives
an ineligible decision naming exactly why. Reasons accumulate rather than
short-circuiting, so an invalidated block that is also on the wrong side of
price says both. That is the audit boundary Round 6.6e.2a's projection was built
to preserve, honoured here rather than spent.

**Still no consolidation and no ranking.** Two blocks sharing a zone remain two
decisions. Distance to reference is reported because it is geometry; nothing is
sorted by it, scored, merged or selected.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from goldpipeline.schemas.common import Timeframe
from goldpipeline.services.ict_candidate_source import (
    CandidateSource,
    CandidateSourceKind,
    CandidateSourceProjection,
    FairValueGapEvidence,
    LiquidityPoolEvidence,
    OrderBlockEvidence,
    project_candidate_sources,
)
from goldpipeline.services.ict_composite import IctCompositeAnalysis, IctCompositeConfig
from goldpipeline.services.ict_fvg import FvgStatus
from goldpipeline.services.ict_liquidity import LiquiditySide, PoolStatus
from goldpipeline.services.ict_order_block_lifecycle import OrderBlockStatus
from goldpipeline.services.ict_primitives import GapDirection
from goldpipeline.services.ict_range import DealingRange
from goldpipeline.services.ict_structure import BreakDirection, StructureBias

logger = logging.getLogger(__name__)

CANDIDATE_ELIGIBILITY_METHOD_VERSION = "1.0.0"
"""Stamped on every decision and analysis, and part of every candidate identity."""


class CandidateEligibilityError(ValueError):
    """The layer could not honestly answer, so it refused to guess."""


# --------------------------------------------------------------------------
# the global reference price
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferencePriceWitness:
    """One timeframe's bar that closed at the selected instant."""

    timeframe: Timeframe
    bar_open_time: datetime
    bar_close_time: datetime
    close: Decimal


@dataclass(frozen=True)
class ReferencePrice:
    """The one price every candidate on every timeframe is measured against.

    ``bar_close_time`` may be well before ``observed_at`` - a weekend, a market
    pause, a partial fixture. That is a real state and is reported as it stands:
    nothing here fabricates a newer price, interpolates one, or refuses a stale
    one. Whether a price is too old to act on is a freshness question, and this
    round has no freshness policy.
    """

    symbol: str
    observed_at: datetime
    bar_close_time: datetime
    price: Decimal
    witnesses: tuple[ReferencePriceWitness, ...]
    """Every timeframe that closed at that instant, in branch order.

    More than one is normal and all of them are kept - throwing provenance away
    because two sources agreed would make the agreement unverifiable later.
    """


def resolve_reference_price(composite: IctCompositeAnalysis) -> ReferencePrice:
    """The latest closed price across every timeframe the composite holds.

    Derived entirely from already-normalized closed series: no clock, no ticker,
    no provider call, and no forming candle can reach it.

    Raises:
        CandidateEligibilityError: Two timeframes closed at the selected instant
            with different prices, or the composite holds no timeframe.
    """
    if not composite.timeframes:
        raise CandidateEligibilityError("a composite with no timeframe has no reference price")

    latest = max(entry.series.latest_closed_at for entry in composite.timeframes)
    witnesses = tuple(
        ReferencePriceWitness(
            timeframe=entry.timeframe,
            bar_open_time=entry.series.bars[-1].timestamp,
            bar_close_time=entry.series.latest_closed_at,
            close=entry.series.bars[-1].close,
        )
        for entry in composite.timeframes
        if entry.series.latest_closed_at == latest
    )

    prices = {witness.close for witness in witnesses}
    if len(prices) != 1:
        # Fail closed. Preferring the finer timeframe, averaging, or taking the
        # first would each turn a contradiction into a plausible number, and
        # every candidate downstream would inherit it silently.
        detail = ", ".join(f"{witness.timeframe.value}={witness.close}" for witness in witnesses)
        raise CandidateEligibilityError(
            f"timeframes disagree about the close at {latest.isoformat()}: {detail}; "
            "one symbol cannot have two closing prices at one instant"
        )

    return ReferencePrice(
        symbol=composite.symbol,
        observed_at=composite.observed_at,
        bar_close_time=latest,
        price=witnesses[0].close,
        witnesses=witnesses,
    )


# --------------------------------------------------------------------------
# roles, sides and relations
# --------------------------------------------------------------------------


class CandidateRole(StrEnum):
    """What a source could be used as, if anything.

    ``ENTRY_ZONE`` is an interval price might be traded from. The two reference
    roles are levels price might be drawn *to*; they are not entries, which is
    why liquidity never receives an entry side.
    """

    ENTRY_ZONE = "ENTRY_ZONE"
    UPPER_REFERENCE = "UPPER_REFERENCE"
    LOWER_REFERENCE = "LOWER_REFERENCE"


class EntrySide(StrEnum):
    """The side an entry zone would be taken on.

    ``BAI`` is the project's word for a long-side entry and ``SEO`` for a
    short-side one. Only order blocks and gaps get one, and it follows the
    source's own structural direction rather than where price happens to be.
    """

    BAI = "BAI"
    SEO = "SEO"


class MarketRelation(StrEnum):
    """Where a zone or level sits relative to the reference price.

    ``OVERLAPS`` covers a price strictly inside a zone *and* a price resting
    exactly on either edge, with no tolerance. A single level equal to the
    reference price also overlaps it.
    """

    BELOW = "BELOW"
    OVERLAPS = "OVERLAPS"
    ABOVE = "ABOVE"


class EligibilityReason(StrEnum):
    """Why a source did not proceed. Never collapsed into one word.

    A reviewer has to be able to see the difference between "this block was
    invalidated by price" and "this block is fine but your policy excludes
    mitigated blocks" - those are entirely different conversations, and
    ``REJECTED`` would hide both.
    """

    ORDER_BLOCK_INVALIDATED = "ORDER_BLOCK_INVALIDATED"
    ORDER_BLOCK_STATUS_NOT_ALLOWED = "ORDER_BLOCK_STATUS_NOT_ALLOWED"
    FVG_FILLED = "FVG_FILLED"
    FVG_STATUS_NOT_ALLOWED = "FVG_STATUS_NOT_ALLOWED"
    LIQUIDITY_TERMINAL = "LIQUIDITY_TERMINAL"
    ENTRY_ZONE_WRONG_SIDE = "ENTRY_ZONE_WRONG_SIDE"
    REFERENCE_NOT_AHEAD = "REFERENCE_NOT_AHEAD"


REASON_ORDER: tuple[EligibilityReason, ...] = tuple(EligibilityReason)
"""The one order reasons are reported in - the enum's own declaration order."""

TERMINAL_ORDER_BLOCK_STATUSES: frozenset[OrderBlockStatus] = frozenset(
    {OrderBlockStatus.INVALIDATED}
)
TERMINAL_FVG_STATUSES: frozenset[FvgStatus] = frozenset({FvgStatus.FILLED})
TERMINAL_POOL_STATUSES: frozenset[PoolStatus] = frozenset(
    {PoolStatus.SWEPT, PoolStatus.CLOSED_THROUGH}
)
"""Deterministic terminal truth. No config may reach past these."""


@dataclass(frozen=True)
class CandidateEligibilityConfig:
    """Which non-terminal statuses a caller is willing to consider.

    No defaults. Whether a touched order block is still worth trading, and
    whether a mitigated one is, are genuine disagreements between practitioners;
    picking one here would ship an unargued answer wearing the authority of
    geometry. An empty set is a legitimate policy meaning "none of that kind".

    Liquidity has no entry here on purpose: only ``ACTIVE`` pools can be a
    reference, so there is nothing left to choose.
    """

    allowed_order_block_statuses: frozenset[OrderBlockStatus]
    allowed_fvg_statuses: frozenset[FvgStatus]

    def __post_init__(self) -> None:
        forbidden_blocks = self.allowed_order_block_statuses & TERMINAL_ORDER_BLOCK_STATUSES
        if forbidden_blocks:
            raise CandidateEligibilityError(
                f"terminal order-block statuses cannot be made eligible: "
                f"{sorted(status.value for status in forbidden_blocks)}"
            )
        forbidden_gaps = self.allowed_fvg_statuses & TERMINAL_FVG_STATUSES
        if forbidden_gaps:
            raise CandidateEligibilityError(
                f"terminal gap statuses cannot be made eligible: "
                f"{sorted(status.value for status in forbidden_gaps)}"
            )


def relation_of_zone(lower: Decimal, upper: Decimal, price: Decimal) -> MarketRelation:
    """Where ``[lower, upper]`` sits relative to *price*, with no tolerance."""
    if upper < price:
        return MarketRelation.BELOW
    if lower > price:
        return MarketRelation.ABOVE
    return MarketRelation.OVERLAPS


def relation_of_level(level: Decimal, price: Decimal) -> MarketRelation:
    """Where a single level sits relative to *price*. Equality overlaps."""
    if level < price:
        return MarketRelation.BELOW
    if level > price:
        return MarketRelation.ABOVE
    return MarketRelation.OVERLAPS


def distance_of_zone(lower: Decimal, upper: Decimal, price: Decimal) -> Decimal:
    """How far price is from the nearest edge of the zone, or zero inside it."""
    if upper < price:
        return price - upper
    if lower > price:
        return lower - price
    return Decimal(0)


def entry_side_for(direction: BreakDirection | GapDirection) -> EntrySide:
    """The side a bullish or bearish entry source would be taken on."""
    return EntrySide.BAI if direction.value == "BULLISH" else EntrySide.SEO


def role_for(source: CandidateSource) -> CandidateRole:
    """What role this source's kind and side give it.

    A buy-side pool becomes an ``UPPER_REFERENCE`` because that is where it
    rests, not because anybody should sell into it.
    """
    if source.kind is not CandidateSourceKind.LIQUIDITY_POOL:
        return CandidateRole.ENTRY_ZONE
    evidence = source.evidence
    assert isinstance(evidence, LiquidityPoolEvidence)
    return (
        CandidateRole.UPPER_REFERENCE
        if evidence.pool.side is LiquiditySide.BUY_SIDE
        else CandidateRole.LOWER_REFERENCE
    )


def reference_level_for(source: CandidateSource) -> Decimal:
    """The exact boundary a pool would be referenced at.

    The far edge in the direction the pool rests: a buy-side pool's ``upper``,
    a sell-side pool's ``lower``. Not the midpoint and not an average - those
    would invent a price between two swings rather than name the one they made.
    An exact-equality pool has one price, and it is that one.
    """
    evidence = source.evidence
    assert isinstance(evidence, LiquidityPoolEvidence)
    return (
        evidence.pool.upper if evidence.pool.side is LiquiditySide.BUY_SIDE else evidence.pool.lower
    )


# --------------------------------------------------------------------------
# the decision
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateEligibilityDecision:
    """One projected source, and whether it could be used.

    Geometry follows the role. An ``ENTRY_ZONE`` carries the source's exact zone
    and no reference level; a reference role carries the level and leaves the
    zone fields ``None`` - its band is still reachable through ``source``, but
    a band is not candidate geometry and giving it those fields would invite a
    later reader to trade it as a zone.
    """

    candidate_id: str
    method_version: str

    candidate_source_id: str
    source_id: str
    timeframe: Timeframe
    symbol: str

    role: CandidateRole
    entry_side: EntrySide | None
    """``None`` for both reference roles. Liquidity is never BAI or SEO."""

    reference_price: Decimal
    market_relation: MarketRelation
    distance_to_reference: Decimal
    """Non-negative and exact. Geometry, not a score - nothing sorts by it."""

    lower: Decimal | None
    upper: Decimal | None
    midpoint: Decimal | None
    width: Decimal | None

    reference_level: Decimal | None

    eligible: bool
    reasons: tuple[EligibilityReason, ...]
    """Empty exactly when eligible. Ordered by :data:`REASON_ORDER`."""

    source: CandidateSource

    @property
    def is_entry(self) -> bool:
        return self.role is CandidateRole.ENTRY_ZONE


@dataclass(frozen=True)
class CandidateEligibilityTimeframe:
    """One timeframe's decisions, plus the context they were taken in."""

    method_version: str
    timeframe: Timeframe
    symbol: str
    observed_at: datetime

    decisions: tuple[CandidateEligibilityDecision, ...]

    structure_bias: StructureBias
    active_dealing_range: DealingRange | None
    """Metadata, carried from the projection. Never consulted: no source is
    ruled out for sitting in premium, in discount, or outside the range."""

    def decision(self, identity: str) -> CandidateEligibilityDecision | None:
        return next((entry for entry in self.decisions if entry.candidate_id == identity), None)

    @property
    def eligible(self) -> tuple[CandidateEligibilityDecision, ...]:
        """A view of :attr:`decisions`, never a separately computed collection."""
        return tuple(entry for entry in self.decisions if entry.eligible)


@dataclass(frozen=True)
class CandidateEligibilityAnalysis:
    """Every timeframe's decisions under one reference price and one policy."""

    method_version: str
    observed_at: datetime
    symbol: str
    provider: str
    provider_symbol: str | None

    composite_config: IctCompositeConfig
    eligibility_config: CandidateEligibilityConfig

    reference_price: ReferencePrice
    timeframes: tuple[CandidateEligibilityTimeframe, ...]

    def timeframe(self, wanted: Timeframe) -> CandidateEligibilityTimeframe | None:
        return next((entry for entry in self.timeframes if entry.timeframe is wanted), None)

    def require(self, wanted: Timeframe) -> CandidateEligibilityTimeframe:
        found = self.timeframe(wanted)
        if found is None:
            raise CandidateEligibilityError(f"this analysis holds no {wanted.value} decisions")
        return found


def _candidate_id(
    *, candidate_source_id: str, role: CandidateRole, entry_side: EntrySide | None
) -> str:
    """Identity from what a candidate *is*, never from how it was judged.

    No price, no status, no distance and no reason enters the preimage, so the
    same market source keeps the same identity under two different eligibility
    policies. The decision changes; the candidate does not.
    """
    preimage = "|".join(
        (
            CANDIDATE_ELIGIBILITY_METHOD_VERSION,
            candidate_source_id,
            role.value,
            "" if entry_side is None else entry_side.value,
        )
    )
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]


def _entry_reasons(
    source: CandidateSource,
    *,
    side: EntrySide,
    relation: MarketRelation,
    config: CandidateEligibilityConfig,
) -> list[EligibilityReason]:
    """Every reason an entry zone does not proceed, accumulated not short-circuited."""
    found: list[EligibilityReason] = []
    evidence = source.evidence

    if isinstance(evidence, OrderBlockEvidence):
        if evidence.status in TERMINAL_ORDER_BLOCK_STATUSES:
            found.append(EligibilityReason.ORDER_BLOCK_INVALIDATED)
        elif evidence.status not in config.allowed_order_block_statuses:
            found.append(EligibilityReason.ORDER_BLOCK_STATUS_NOT_ALLOWED)
    elif isinstance(evidence, FairValueGapEvidence):
        if evidence.status in TERMINAL_FVG_STATUSES:
            found.append(EligibilityReason.FVG_FILLED)
        elif evidence.status not in config.allowed_fvg_statuses:
            found.append(EligibilityReason.FVG_STATUS_NOT_ALLOWED)

    # A long-side zone entirely above price, or a short-side zone entirely
    # below it, is on the wrong side. It is not re-read as the other side: its
    # source direction is a fact, and flipping it would invent a second reading
    # of one candle.
    wrong_side = (side is EntrySide.BAI and relation is MarketRelation.ABOVE) or (
        side is EntrySide.SEO and relation is MarketRelation.BELOW
    )
    if wrong_side:
        found.append(EligibilityReason.ENTRY_ZONE_WRONG_SIDE)

    return found


def _reference_reasons(
    source: CandidateSource, *, role: CandidateRole, level: Decimal, price: Decimal
) -> list[EligibilityReason]:
    """Every reason a liquidity reference does not proceed."""
    found: list[EligibilityReason] = []
    evidence = source.evidence
    assert isinstance(evidence, LiquidityPoolEvidence)

    if evidence.status in TERMINAL_POOL_STATUSES:
        found.append(EligibilityReason.LIQUIDITY_TERMINAL)

    # A reference level is somewhere price has not been yet. Exact equality is
    # not ahead of anything.
    ahead = level > price if role is CandidateRole.UPPER_REFERENCE else level < price
    if not ahead:
        found.append(EligibilityReason.REFERENCE_NOT_AHEAD)

    return found


def _decide(
    source: CandidateSource, *, price: Decimal, config: CandidateEligibilityConfig
) -> CandidateEligibilityDecision:
    role = role_for(source)
    side: EntrySide | None

    if role is CandidateRole.ENTRY_ZONE:
        evidence = source.evidence
        if isinstance(evidence, OrderBlockEvidence):
            side = entry_side_for(evidence.order_block.direction)
        else:
            assert isinstance(evidence, FairValueGapEvidence)
            side = entry_side_for(evidence.gap.direction)
        relation = relation_of_zone(source.lower, source.upper, price)
        reasons = _entry_reasons(source, side=side, relation=relation, config=config)
        level = None
        bounds: tuple[Decimal | None, ...] = (
            source.lower,
            source.upper,
            source.midpoint,
            source.width,
        )
        distance = distance_of_zone(source.lower, source.upper, price)
    else:
        side = None
        level = reference_level_for(source)
        relation = relation_of_level(level, price)
        reasons = _reference_reasons(source, role=role, level=level, price=price)
        bounds = (None, None, None, None)
        distance = abs(level - price)

    ordered = tuple(reason for reason in REASON_ORDER if reason in reasons)

    return CandidateEligibilityDecision(
        candidate_id=_candidate_id(
            candidate_source_id=source.candidate_source_id, role=role, entry_side=side
        ),
        method_version=CANDIDATE_ELIGIBILITY_METHOD_VERSION,
        candidate_source_id=source.candidate_source_id,
        source_id=source.source_id,
        timeframe=source.timeframe,
        symbol=source.symbol,
        role=role,
        entry_side=side,
        reference_price=price,
        market_relation=relation,
        distance_to_reference=distance,
        lower=bounds[0],
        upper=bounds[1],
        midpoint=bounds[2],
        width=bounds[3],
        reference_level=level,
        eligible=not ordered,
        reasons=ordered,
        source=source,
    )


def analyse_candidate_eligibility(
    composite: IctCompositeAnalysis,
    *,
    config: CandidateEligibilityConfig,
    projection: CandidateSourceProjection | None = None,
) -> CandidateEligibilityAnalysis:
    """Decide, for every projected source, what role it could play and whether it can.

    Exactly one decision per source, in the projection's own order. Nothing is
    filtered, merged, scored or sorted: a source ruled out keeps its decision and
    its reasons, which is what lets an auditor tell "never existed" from "existed
    and was excluded because of this".

    Args:
        composite: The finished composite analysis. Its closed series decide the
            reference price and its config travels through as provenance.
        config: Which non-terminal statuses are eligible. Required.
        projection: A projection a caller already has. Omit and one is computed
            here, exactly once. A supplied projection must describe this same
            composite - instant, symbol, provider and policy - and is refused
            rather than trimmed if it does not.

    Raises:
        CandidateEligibilityError: The reference price cannot be resolved, or a
            supplied projection describes a different composite.
    """
    reference = resolve_reference_price(composite)

    if projection is None:
        projection = project_candidate_sources(composite)
    else:
        _require_same_composite(projection, composite)

    timeframes = tuple(
        CandidateEligibilityTimeframe(
            method_version=CANDIDATE_ELIGIBILITY_METHOD_VERSION,
            timeframe=entry.timeframe,
            symbol=entry.symbol,
            observed_at=entry.observed_at,
            decisions=tuple(
                _decide(source, price=reference.price, config=config) for source in entry.sources
            ),
            structure_bias=entry.structure_bias,
            active_dealing_range=entry.active_dealing_range,
        )
        for entry in projection.timeframes
    )

    return CandidateEligibilityAnalysis(
        method_version=CANDIDATE_ELIGIBILITY_METHOD_VERSION,
        observed_at=composite.observed_at,
        symbol=composite.symbol,
        provider=composite.provider,
        provider_symbol=composite.provider_symbol,
        composite_config=composite.config,
        eligibility_config=config,
        reference_price=reference,
        timeframes=timeframes,
    )


def _require_same_composite(
    projection: CandidateSourceProjection, composite: IctCompositeAnalysis
) -> None:
    """A supplied projection must describe this exact composite.

    Refused rather than trimmed, for the reason every seam on this branch is:
    a projection carries an instant and a policy of its own, and quietly using
    one built from a later composite would date every decision wrongly while
    every individual number still looked real.
    """
    if projection.observed_at != composite.observed_at:
        raise CandidateEligibilityError(
            f"projection describes {projection.observed_at.isoformat()} but this composite is "
            f"{composite.observed_at.isoformat()}; supply one built from it"
        )
    if projection.symbol != composite.symbol:
        raise CandidateEligibilityError(
            f"projection is for {projection.symbol!r}, not {composite.symbol!r}"
        )
    if projection.provider != composite.provider:
        raise CandidateEligibilityError(
            f"projection is from {projection.provider!r}, not {composite.provider!r}"
        )
    if projection.config != composite.config:
        raise CandidateEligibilityError(
            "projection was built under a different composite policy; supply one built from "
            "this composite"
        )


__all__ = [
    "CANDIDATE_ELIGIBILITY_METHOD_VERSION",
    "REASON_ORDER",
    "TERMINAL_FVG_STATUSES",
    "TERMINAL_ORDER_BLOCK_STATUSES",
    "TERMINAL_POOL_STATUSES",
    "CandidateEligibilityAnalysis",
    "CandidateEligibilityConfig",
    "CandidateEligibilityDecision",
    "CandidateEligibilityError",
    "CandidateEligibilityTimeframe",
    "CandidateRole",
    "EligibilityReason",
    "EntrySide",
    "MarketRelation",
    "ReferencePrice",
    "ReferencePriceWitness",
    "analyse_candidate_eligibility",
    "distance_of_zone",
    "entry_side_for",
    "reference_level_for",
    "relation_of_level",
    "relation_of_zone",
    "resolve_reference_price",
    "role_for",
]
