"""Which deterministic market objects could later become candidates, and nothing more.

Round 6.6e.2a, the first candidate-layer round. It answers exactly one question -
*what source facts exist?* - and deliberately refuses the eight that follow it:
whether a source is still eligible, where it sits relative to price, whether it
is redundant, whether it is strong, whether AI should rank it, and whether it
becomes a buy or a sell zone.

**A projection, not an analysis.** The input is a finished
:class:`~goldpipeline.services.ict_composite.IctCompositeAnalysis` and nothing
else. No lower authority is called here - not one - which is what makes the
composite the single input seam the architecture depends on. Reopening
``analyse_structure`` from this layer would recreate the several-copies problem
Round 6.6e.1 removed, one level further up.

**Lifecycle status is evidence, never a filter.** An invalidated order block, a
filled gap and a swept pool all appear in this projection. That is the whole
point: the next round has to be able to prove it removed a source *because of a
named deterministic fact*, and it cannot prove that against something this layer
quietly deleted. Dropping them here would erase the boundary between "the source
never existed" and "the source existed and was ruled out".

**Three status systems, kept apart.** An order block is ``INVALIDATED``, a gap
is ``FILLED``, a pool is ``SWEPT``. Those words are not synonyms and there is no
exact mapping between them, so this module keeps three typed evidence payloads
rather than flattening them into one ``LIVE``/``DEAD`` enum that would read as
though a decision had been made. The evidence also carries each source's own
direction - bullish/bearish for blocks and gaps, buy-side/sell-side for pools -
because those already exist in the source domain. They are facts about which way
a move went, not instructions about which way to trade.

**One observation, one fact.** Two order blocks sharing a source candle stay two
facts. Two gaps with identical bounds stay two facts. A pool overlapping a block
stays separate. No merging, no overlap counting, no confluence: consolidation is
a decision with content, and preserving the raw observations is what lets the
round that makes it show its working.

**Dealing ranges are context, not sources.** Whether a range's origin,
equilibrium or terminal should become an entry zone, a farther reference level
or nothing at all is undecided. Minting candidate identities for them now would
settle that by accident, so the active range is attached as context and no range
gets a source id.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from goldpipeline.schemas.common import Timeframe
from goldpipeline.services.ict_composite import (
    IctCompositeAnalysis,
    IctCompositeConfig,
    IctTimeframeAnalysis,
)
from goldpipeline.services.ict_fvg import FvgState, FvgStatus
from goldpipeline.services.ict_liquidity import LiquidityPool, PoolStatus
from goldpipeline.services.ict_order_block import OrderBlock
from goldpipeline.services.ict_order_block_lifecycle import (
    OrderBlockMitigationRule,
    OrderBlockState,
    OrderBlockStatus,
)
from goldpipeline.services.ict_primitives import FairValueGap
from goldpipeline.services.ict_range import DealingRange
from goldpipeline.services.ict_structure import StructureBias

logger = logging.getLogger(__name__)

CANDIDATE_SOURCE_METHOD_VERSION = "1.0.0"
"""Stamped on every projected fact, and part of every projected identity."""


class CandidateSourceError(ValueError):
    """The projection could not honestly answer, so it refused to guess."""


class CandidateSourceKind(StrEnum):
    """What kind of market object a projected fact came from.

    Three, and no more this round. ``DEALING_RANGE``, ``PROTECTED_SWING``,
    ``STRUCTURAL_LEG``, ``SWING`` and ``ATR`` are absent because none of them has
    a decided role yet - see the module docstring. They remain context.
    """

    ORDER_BLOCK = "ORDER_BLOCK"
    FAIR_VALUE_GAP = "FAIR_VALUE_GAP"
    LIQUIDITY_POOL = "LIQUIDITY_POOL"


class CandidateGeometryKind(StrEnum):
    """What shape a source occupies in price.

    ``ZONE`` is an interval a candle or a gap actually drew. ``BAND`` is the
    span a set of near-equal swings occupies within a stated tolerance, which is
    a different kind of object: it may legitimately be a single price, and
    whether it should later collapse to one boundary level is undecided. Keeping
    the two apart is what stops that decision being made here by accident.
    """

    ZONE = "ZONE"
    BAND = "BAND"


GEOMETRY_OF: dict[CandidateSourceKind, CandidateGeometryKind] = {
    CandidateSourceKind.ORDER_BLOCK: CandidateGeometryKind.ZONE,
    CandidateSourceKind.FAIR_VALUE_GAP: CandidateGeometryKind.ZONE,
    CandidateSourceKind.LIQUIDITY_POOL: CandidateGeometryKind.BAND,
}
"""The mapping, in one place, so no branch can disagree with another."""


@dataclass(frozen=True)
class OrderBlockEvidence:
    """Everything the order-block engines said about this source.

    The formation record and the lifecycle state are both held whole. ``status``
    and ``mitigation_rule`` are lifted out because a reader will want them
    constantly; everything else stays reachable through the two nested objects
    rather than being copied into a second home.
    """

    order_block: OrderBlock
    state: OrderBlockState
    status: OrderBlockStatus
    mitigation_rule: OrderBlockMitigationRule


@dataclass(frozen=True)
class FairValueGapEvidence:
    """Everything the gap engines said about this source."""

    gap: FairValueGap
    state: FvgState
    status: FvgStatus


@dataclass(frozen=True)
class LiquidityPoolEvidence:
    """Everything the liquidity engine said about this source.

    A pool carries its own status, unlike blocks and gaps whose status lives in
    a separate lifecycle analysis. That asymmetry is real and is preserved
    rather than smoothed over.
    """

    pool: LiquidityPool
    status: PoolStatus
    tolerance: Decimal
    terminal_event_id: str | None


CandidateEvidence = OrderBlockEvidence | FairValueGapEvidence | LiquidityPoolEvidence
"""A typed union, so a reader that handles one kind cannot silently mishandle another."""


@dataclass(frozen=True)
class CandidateSource:
    """One market object, projected as a fact that could later be considered.

    No eligibility, no side, no score, no distance from price. ``source_id``
    stays alongside ``candidate_source_id`` so an audit can always walk back
    from a projected fact to the exact original object by identity.
    """

    candidate_source_id: str
    method_version: str

    timeframe: Timeframe
    symbol: str

    kind: CandidateSourceKind
    geometry: CandidateGeometryKind

    source_id: str
    """The original object's own identity - ``order_block_id``, ``fvg_id`` or
    ``pool_id``. Never replaced by the projected identity."""

    formed_at: datetime

    lower: Decimal
    upper: Decimal
    midpoint: Decimal
    width: Decimal
    """``upper - lower``. Zero is legitimate for a ``BAND`` and impossible for a
    ``ZONE`` - see :func:`_check_geometry`."""

    evidence: CandidateEvidence
    """The source-specific facts, including its direction or side."""


@dataclass(frozen=True)
class CandidateSourceTimeframe:
    """One timeframe's projected sources, plus the context they sit in."""

    method_version: str
    timeframe: Timeframe
    symbol: str
    observed_at: datetime

    sources: tuple[CandidateSource, ...]

    structure_bias: StructureBias
    """Context. Nothing here derives a role, a side or a score from it."""

    active_dealing_range: DealingRange | None
    """The composite's own active range, or ``None``.

    Context only, and explicitly not a filter: a source outside this range, or a
    bullish block in its premium half, projects exactly the same as any other.
    A directional bias with no active range is a legitimate state Round 6.6c.3b
    established, and every source still projects normally.
    """

    def source(self, identity: str) -> CandidateSource | None:
        return next(
            (entry for entry in self.sources if entry.candidate_source_id == identity), None
        )

    def of_kind(self, kind: CandidateSourceKind) -> tuple[CandidateSource, ...]:
        return tuple(entry for entry in self.sources if entry.kind is kind)


@dataclass(frozen=True)
class CandidateSourceProjection:
    """One composite analysis, projected into per-timeframe source facts.

    Deliberately not a synthesis: no cross-timeframe merging, no confluence, no
    promotion of one timeframe over another. Five independent sets of facts,
    each carrying the timeframe it came from.
    """

    method_version: str
    observed_at: datetime
    symbol: str
    provider: str
    provider_symbol: str | None

    config: IctCompositeConfig
    """The composite's own six-field policy, carried through. This round adds no
    seventh policy of its own."""

    timeframes: tuple[CandidateSourceTimeframe, ...]

    def timeframe(self, wanted: Timeframe) -> CandidateSourceTimeframe | None:
        return next((entry for entry in self.timeframes if entry.timeframe is wanted), None)

    def require(self, wanted: Timeframe) -> CandidateSourceTimeframe:
        found = self.timeframe(wanted)
        if found is None:
            raise CandidateSourceError(f"this projection holds no {wanted.value} sources")
        return found


def _candidate_source_id(*, timeframe: Timeframe, kind: CandidateSourceKind, source_id: str) -> str:
    """Identity from the projection's own facts, none of which is a price.

    The source's identity already owns whatever geometry policy produced it - an
    order-block id encodes its zone basis, a pool id its tolerance - so no price
    enters this preimage and no fourth Decimal-canonicalisation site appears.
    ``kind`` is present so two different source types cannot collide even if
    their own ids coincided.
    """
    preimage = "|".join((CANDIDATE_SOURCE_METHOD_VERSION, timeframe.value, kind.value, source_id))
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]


def _check_geometry(
    kind: CandidateSourceKind, lower: Decimal, upper: Decimal, midpoint: Decimal, source_id: str
) -> None:
    """The common geometry contract, checked rather than assumed.

    ``lower <= midpoint <= upper`` always. A ``ZONE`` additionally requires
    ``lower < upper``, because a block or a gap with no width would be a level
    wearing a zone's name. A ``BAND`` does not: two swings at exactly the same
    price form a real pool of zero width, and widening it to look tidier would
    invent a tick the market never printed.
    """
    if upper < lower:
        raise CandidateSourceError(f"source {source_id} has upper {upper} below lower {lower}")
    if not lower <= midpoint <= upper:
        raise CandidateSourceError(
            f"source {source_id} has midpoint {midpoint} outside {lower}..{upper}"
        )
    if GEOMETRY_OF[kind] is CandidateGeometryKind.ZONE and upper <= lower:
        raise CandidateSourceError(
            f"{kind.value} source {source_id} has no width: {lower}..{upper}"
        )


def _source(
    *,
    timeframe: Timeframe,
    symbol: str,
    kind: CandidateSourceKind,
    source_id: str,
    formed_at: datetime,
    lower: Decimal,
    upper: Decimal,
    midpoint: Decimal,
    evidence: CandidateEvidence,
) -> CandidateSource:
    _check_geometry(kind, lower, upper, midpoint, source_id)
    return CandidateSource(
        candidate_source_id=_candidate_source_id(
            timeframe=timeframe, kind=kind, source_id=source_id
        ),
        method_version=CANDIDATE_SOURCE_METHOD_VERSION,
        timeframe=timeframe,
        symbol=symbol,
        kind=kind,
        geometry=GEOMETRY_OF[kind],
        source_id=source_id,
        formed_at=formed_at,
        lower=lower,
        upper=upper,
        midpoint=midpoint,
        width=upper - lower,
        evidence=evidence,
    )


def project_timeframe_sources(entry: IctTimeframeAnalysis) -> CandidateSourceTimeframe:
    """Project one timeframe's completed analysis into source facts.

    Exactly one fact per lifecycle state and per pool, in every case, with no
    filtering of any kind. The counts are therefore an invariant rather than an
    outcome, and ``tests/test_ict_candidate_source.py`` asserts them as one.

    Args:
        entry: One timeframe of a composite analysis. Nothing is recomputed
            from it and no lower authority is called.

    Raises:
        CandidateSourceError: A lifecycle state names an order block the
            analysis does not hold, or a source's geometry contradicts itself.
    """
    sources: list[CandidateSource] = []

    for state in entry.order_block_lifecycle.states:
        block = entry.order_blocks.order_block(state.order_block_id)
        if block is None:
            # Joined by identity, never by price or bounds. A lifecycle state
            # whose block is absent means the two analyses disagree, which is a
            # contradiction rather than a source with a missing field.
            raise CandidateSourceError(
                f"lifecycle state {state.order_block_id} names an order block this analysis "
                "does not hold"
            )
        sources.append(
            _source(
                timeframe=entry.timeframe,
                symbol=entry.symbol,
                kind=CandidateSourceKind.ORDER_BLOCK,
                source_id=block.order_block_id,
                formed_at=block.formed_at,
                lower=block.lower,
                upper=block.upper,
                midpoint=block.midpoint,
                evidence=OrderBlockEvidence(
                    order_block=block,
                    state=state,
                    status=state.status,
                    mitigation_rule=state.mitigation_rule,
                ),
            )
        )

    for gap_state in entry.fvg_lifecycle.states:
        gap = gap_state.gap
        sources.append(
            _source(
                timeframe=entry.timeframe,
                symbol=entry.symbol,
                kind=CandidateSourceKind.FAIR_VALUE_GAP,
                source_id=gap_state.fvg_id,
                formed_at=gap.formed_at,
                lower=gap.lower,
                upper=gap.upper,
                midpoint=gap.midpoint,
                evidence=FairValueGapEvidence(gap=gap, state=gap_state, status=gap_state.status),
            )
        )

    for pool in entry.liquidity.pools:
        sources.append(
            _source(
                timeframe=entry.timeframe,
                symbol=entry.symbol,
                kind=CandidateSourceKind.LIQUIDITY_POOL,
                source_id=pool.pool_id,
                formed_at=pool.formed_at,
                lower=pool.lower,
                upper=pool.upper,
                midpoint=pool.midpoint,
                evidence=LiquidityPoolEvidence(
                    pool=pool,
                    status=pool.status,
                    tolerance=pool.tolerance,
                    terminal_event_id=pool.terminal_event_id,
                ),
            )
        )

    # Formation order first, then kind, then the source's own identity - a total
    # order that needs no reference price. Ordering by distance from price is a
    # real question and it belongs to the round that chooses a price authority.
    sources.sort(key=lambda found: (found.formed_at, found.kind.value, found.source_id))

    return CandidateSourceTimeframe(
        method_version=CANDIDATE_SOURCE_METHOD_VERSION,
        timeframe=entry.timeframe,
        symbol=entry.symbol,
        observed_at=entry.observed_at,
        sources=tuple(sources),
        structure_bias=entry.dealing_ranges.structure_bias,
        active_dealing_range=entry.dealing_ranges.active_range,
    )


def project_candidate_sources(composite: IctCompositeAnalysis) -> CandidateSourceProjection:
    """Project a whole composite analysis, timeframe by timeframe.

    Pure: the observation instant, the source set, every status and the range
    context are exactly those the composite already holds. There is no ``as_of``
    parameter because there is nothing here to reconstruct - reconstructing an
    earlier state means handing in an earlier composite.

    Timeframes keep the composite's order, which is the branch's own
    ``ICT_TIMEFRAMES`` order. That is ordering, not importance: nothing here
    treats H4 as governing M15.
    """
    return CandidateSourceProjection(
        method_version=CANDIDATE_SOURCE_METHOD_VERSION,
        observed_at=composite.observed_at,
        symbol=composite.symbol,
        provider=composite.provider,
        provider_symbol=composite.provider_symbol,
        config=composite.config,
        timeframes=tuple(project_timeframe_sources(entry) for entry in composite.timeframes),
    )


__all__ = [
    "CANDIDATE_SOURCE_METHOD_VERSION",
    "GEOMETRY_OF",
    "CandidateEvidence",
    "CandidateGeometryKind",
    "CandidateSource",
    "CandidateSourceError",
    "CandidateSourceKind",
    "CandidateSourceProjection",
    "CandidateSourceTimeframe",
    "FairValueGapEvidence",
    "LiquidityPoolEvidence",
    "OrderBlockEvidence",
    "project_candidate_sources",
    "project_timeframe_sources",
]
