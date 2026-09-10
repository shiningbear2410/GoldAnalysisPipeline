"""Facts about consolidated candidates, arranged so something else can judge them.

Round 6.6f. This is the last deterministic layer before an AI is allowed to have
an opinion, and its entire job is to lay out what is true without deciding what
matters. Every number here is copied or subtracted; none is weighted, scaled or
compared to a threshold.

**Why a feature graph rather than a score.** The stages below this one refused
to rank for a good reason: a score is a policy nobody argued about, hidden
inside an integer. That refusal does not change because a model is next in the
pipeline. What changes is that the model needs the facts *arranged* - ages
computed, ranges classified, overlaps named - and arranging facts is not the
same as ordering them. So this module computes ``support_count`` and
``age_seconds`` and ``OVERLAPPING``, and it does not contain the word "best".

**Overlap is observed, never acted on.** Round 6.6e.2c1 consolidated only exact
duplicates, precisely because merging overlaps invents geometry and chains by
transitivity. That reasoning is unchanged. This module *names* the overlap
between two candidates and creates no third candidate from it: A overlapping B
and B overlapping C produces two relations and zero new zones.

**Age is a fact, not a freshness verdict.** ``age_seconds`` is
``observed_at - formed_at`` and nothing else. There is no maximum age, no expiry
and no staleness flag, because how much a four-hour-old order block is worth is
exactly the sort of question this project has refused to answer with a constant.
A model may weigh it. The engine reports it.

**Range context is metadata here, and still not a filter.** Round 6.6e.2b
carried the active dealing range through eligibility without reading it, so that
no candidate was ever rejected for sitting in premium. That remains true: this
module classifies where a candidate sits and rejects nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from goldpipeline.schemas.common import Timeframe
from goldpipeline.services.ict_candidate_consolidation import (
    CandidateConsolidationAnalysis,
    ConsolidatedCandidate,
)
from goldpipeline.services.ict_candidate_eligibility import (
    CandidateEligibilityConfig,
    CandidateEligibilityDecision,
    CandidateRole,
    EntrySide,
    MarketRelation,
    ReferencePrice,
)
from goldpipeline.services.ict_candidate_source import (
    CandidateSourceKind,
    FairValueGapEvidence,
    LiquidityPoolEvidence,
    OrderBlockEvidence,
)
from goldpipeline.services.ict_composite import IctCompositeConfig
from goldpipeline.services.ict_fvg import FvgStatus
from goldpipeline.services.ict_liquidity import LiquiditySide, PoolStatus
from goldpipeline.services.ict_order_block import OrderBlockZoneBasis
from goldpipeline.services.ict_order_block_lifecycle import (
    OrderBlockMitigationRule,
    OrderBlockStatus,
)
from goldpipeline.services.ict_range import PriceLocation, locate
from goldpipeline.services.ict_structure import BreakDirection, StructureBias

logger = logging.getLogger(__name__)

CANDIDATE_FEATURE_METHOD_VERSION = "1.0.0"
"""Stamped on the analysis and carried into the analyst request."""

FEATURE_TIMEFRAME_ORDER: tuple[Timeframe, ...] = (
    Timeframe.H4,
    Timeframe.H1,
    Timeframe.M15,
    Timeframe.M5,
    Timeframe.M1,
)
"""Slowest to fastest. A display order, and deliberately not a ranking.

H4 appearing before M1 says nothing about H4 mattering more; the list has to be
in *some* order and an arbitrary one would be worse than a conventional one.
"""


class CandidateFeatureError(ValueError):
    """The feature graph could not answer honestly, so it refused to guess."""


class ZoneRelation(StrEnum):
    """How two entry zones sit against each other. A closed, geometric answer.

    ``EQUAL`` survives consolidation only when role or side differs - two zones
    with identical geometry and identical semantics are already one candidate.
    Two zones with identical geometry and *opposite sides* are two different
    propositions about the same prices, and merging them would be a claim about
    market meaning that geometry cannot support.
    """

    EQUAL = "EQUAL"
    OVERLAPPING = "OVERLAPPING"
    TOUCHING = "TOUCHING"
    DISJOINT = "DISJOINT"


# --------------------------------------------------------------------------
# support facts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SupportFact:
    """One supporting decision, flattened to the facts a ranker can use.

    The three lifecycle fields are three fields on purpose. An order block is
    ACTIVE/TOUCHED/MITIGATED/INVALIDATED, a gap is OPEN/TOUCHED/FILLED and a
    pool is ACTIVE/SWEPT/CLOSED_THROUGH; these are three vocabularies that
    happen to share two words, and collapsing them into one ``status: str``
    would invent a shared lifecycle that no engine implements. Exactly one is
    populated, decided by ``source_kind``.
    """

    candidate_id: str
    candidate_source_id: str
    source_id: str

    source_kind: CandidateSourceKind
    timeframe: Timeframe

    formed_at: datetime
    age_seconds: int
    """``observed_at - formed_at``, whole seconds, never negative. A fact."""

    order_block_status: OrderBlockStatus | None
    order_block_zone_basis: OrderBlockZoneBasis | None
    order_block_mitigation_rule: OrderBlockMitigationRule | None

    fvg_status: FvgStatus | None

    liquidity_side: LiquiditySide | None
    liquidity_pool_status: PoolStatus | None
    liquidity_tolerance: Decimal | None


# --------------------------------------------------------------------------
# range and structure context
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RangeContext:
    """Where one candidate sits inside one timeframe's active dealing range.

    Every location field is ``None`` when that timeframe has no active range.
    That is a real state - a timeframe whose structure has not yet produced a
    protected range - and it is reported rather than filled in with a guess.
    """

    timeframe: Timeframe

    active_range_id: str | None
    range_direction: BreakDirection | None
    range_lower: Decimal | None
    range_upper: Decimal | None
    range_equilibrium: Decimal | None

    lower_location: PriceLocation | None
    midpoint_location: PriceLocation | None
    upper_location: PriceLocation | None
    """The three edges of an entry zone, each classified exactly. All ``None``
    for a reference candidate, which has no zone to classify."""

    reference_location: PriceLocation | None
    """The single level of a reference candidate. ``None`` for an entry zone."""


@dataclass(frozen=True)
class TimeframeContext:
    """What one timeframe looked like, independent of any candidate."""

    timeframe: Timeframe
    structure_bias: StructureBias

    active_range_id: str | None
    range_direction: BreakDirection | None
    range_lower: Decimal | None
    range_upper: Decimal | None
    range_equilibrium: Decimal | None
    decision_count: int
    eligible_count: int


# --------------------------------------------------------------------------
# the candidate feature
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateFeature:
    """Everything factual about one consolidated candidate.

    Geometry is copied from the consolidated candidate, never recomputed - the
    same rule that kept a merged zone from acquiring a midpoint no source drew.
    """

    candidate_id: str
    method_version: str
    symbol: str

    role: CandidateRole
    entry_side: EntrySide | None

    lower: Decimal | None
    upper: Decimal | None
    midpoint: Decimal | None
    width: Decimal | None
    reference_level: Decimal | None

    market_relation: MarketRelation
    distance_to_reference: Decimal

    support_count: int
    timeframe_count: int
    source_kind_count: int
    """Three counts. Not summed, not weighted, not combined into a fourth."""

    source_kinds: tuple[CandidateSourceKind, ...]
    timeframes: tuple[Timeframe, ...]

    oldest_source_formed_at: datetime
    newest_source_formed_at: datetime
    oldest_source_age_seconds: int
    newest_source_age_seconds: int

    support_facts: tuple[SupportFact, ...]
    range_contexts: tuple[RangeContext, ...]

    @property
    def is_entry_zone(self) -> bool:
        return self.role is CandidateRole.ENTRY_ZONE

    def range_context(self, timeframe: Timeframe) -> RangeContext | None:
        return next((entry for entry in self.range_contexts if entry.timeframe is timeframe), None)


@dataclass(frozen=True)
class CandidatePairRelation:
    """How two entry-zone candidates sit against each other, exactly.

    Undirected: the pair is keyed by its two ids in lexicographic order, so
    (A, B) and (B, A) are the same relation stored once. Only entry zones get
    relations - a reference is a single level, and asking whether a level
    "overlaps" a zone would be inventing the very zone §12 refuses to give it.
    """

    first_candidate_id: str
    second_candidate_id: str
    relation: ZoneRelation

    intersection_lower: Decimal | None
    intersection_upper: Decimal | None
    intersection_width: Decimal | None
    """Present for EQUAL, OVERLAPPING and TOUCHING; ``None`` for DISJOINT.
    Zero width for TOUCHING, where the intersection is a single price."""

    gap_distance: Decimal | None
    """Present only for DISJOINT: how far apart the two zones are. ``None``
    elsewhere, because a gap that does not exist is absent, not zero."""


@dataclass(frozen=True)
class CandidateFeatureAnalysis:
    """Every candidate's facts at one instant, over one consolidation."""

    method_version: str
    observed_at: datetime
    symbol: str
    provider: str
    provider_symbol: str | None

    composite_config: IctCompositeConfig
    eligibility_config: CandidateEligibilityConfig
    reference_price: ReferencePrice

    candidates: tuple[CandidateFeature, ...]
    pair_relations: tuple[CandidatePairRelation, ...]
    timeframe_contexts: tuple[TimeframeContext, ...]

    consolidation: CandidateConsolidationAnalysis
    """The reading this is a view of, retained whole.

    Which is what keeps the authoritative geometry one dereference away: a
    ranked id resolves back through here to a ``ConsolidatedCandidate``, and
    nothing downstream ever needs to trust a price that travelled through a
    prompt.
    """

    def feature(self, candidate_id: str) -> CandidateFeature | None:
        return next(
            (entry for entry in self.candidates if entry.candidate_id == candidate_id), None
        )

    def of_role(self, role: CandidateRole) -> tuple[CandidateFeature, ...]:
        return tuple(entry for entry in self.candidates if entry.role is role)

    def entry_zones(self, side: EntrySide) -> tuple[CandidateFeature, ...]:
        return tuple(
            entry
            for entry in self.candidates
            if entry.role is CandidateRole.ENTRY_ZONE and entry.entry_side is side
        )

    def relation(self, first: str, second: str) -> CandidatePairRelation | None:
        low, high = sorted((first, second))
        return next(
            (
                entry
                for entry in self.pair_relations
                if entry.first_candidate_id == low and entry.second_candidate_id == high
            ),
            None,
        )

    def timeframe_context(self, timeframe: Timeframe) -> TimeframeContext | None:
        return next(
            (entry for entry in self.timeframe_contexts if entry.timeframe is timeframe), None
        )

    def candidate(self, candidate_id: str) -> ConsolidatedCandidate | None:
        """The authoritative geometry behind a feature. The only price source."""
        return self.consolidation.candidate(candidate_id)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def _age_seconds(observed_at: datetime, formed_at: datetime) -> int:
    """Whole seconds between formation and observation, refusing the future.

    A negative age would mean a source formed after the instant that observed
    it, which is either a broken snapshot or a lookahead leak. Both are worth
    stopping for; neither is worth clamping to zero.
    """
    delta = observed_at - formed_at
    seconds = int(delta.total_seconds())
    if seconds < 0:
        raise CandidateFeatureError(
            f"source formed at {formed_at.isoformat()} is after the observation "
            f"at {observed_at.isoformat()}"
        )
    return seconds


def _support_fact(decision: CandidateEligibilityDecision, *, observed_at: datetime) -> SupportFact:
    evidence = decision.source.evidence

    block_status: OrderBlockStatus | None = None
    zone_basis: OrderBlockZoneBasis | None = None
    mitigation_rule: OrderBlockMitigationRule | None = None
    gap_status: FvgStatus | None = None
    side: LiquiditySide | None = None
    pool_status: PoolStatus | None = None
    tolerance: Decimal | None = None

    if isinstance(evidence, OrderBlockEvidence):
        block_status = evidence.status
        zone_basis = evidence.order_block.zone_basis
        mitigation_rule = evidence.mitigation_rule
    elif isinstance(evidence, FairValueGapEvidence):
        gap_status = evidence.status
    elif isinstance(evidence, LiquidityPoolEvidence):
        side = evidence.pool.side
        pool_status = evidence.status
        tolerance = evidence.tolerance
    else:  # pragma: no cover - the union is closed and mypy proves it
        raise CandidateFeatureError(f"unknown evidence payload on {decision.source_id}")

    return SupportFact(
        candidate_id=decision.candidate_id,
        candidate_source_id=decision.candidate_source_id,
        source_id=decision.source_id,
        source_kind=decision.source.kind,
        timeframe=decision.timeframe,
        formed_at=decision.source.formed_at,
        age_seconds=_age_seconds(observed_at, decision.source.formed_at),
        order_block_status=block_status,
        order_block_zone_basis=zone_basis,
        order_block_mitigation_rule=mitigation_rule,
        fvg_status=gap_status,
        liquidity_side=side,
        liquidity_pool_status=pool_status,
        liquidity_tolerance=tolerance,
    )


def _range_context(
    candidate: ConsolidatedCandidate, *, timeframe: Timeframe, context: TimeframeContext
) -> RangeContext:
    """Classify one candidate against one timeframe's active range.

    Uses the range engine's own :func:`locate`, which is a pure classifier over
    two edges rather than an analysis. Reusing it is the point: a second
    premium/discount rule written here would drift from the one the range
    engine publishes, and then two parts of the system would disagree about
    where equilibrium is.
    """
    empty = RangeContext(
        timeframe=timeframe,
        active_range_id=None,
        range_direction=None,
        range_lower=None,
        range_upper=None,
        range_equilibrium=None,
        lower_location=None,
        midpoint_location=None,
        upper_location=None,
        reference_location=None,
    )
    if (
        context.active_range_id is None
        or context.range_lower is None
        or context.range_upper is None
    ):
        return empty

    edges = {"lower": context.range_lower, "upper": context.range_upper}

    def where(price: Decimal | None) -> PriceLocation | None:
        if price is None:
            return None
        return locate(price, lower=edges["lower"], upper=edges["upper"])

    if candidate.role is CandidateRole.ENTRY_ZONE:
        lower_location = where(candidate.lower)
        midpoint_location = where(candidate.midpoint)
        upper_location = where(candidate.upper)
        reference_location = None
    else:
        lower_location = midpoint_location = upper_location = None
        reference_location = where(candidate.reference_level)

    return RangeContext(
        timeframe=timeframe,
        active_range_id=context.active_range_id,
        range_direction=context.range_direction,
        range_lower=context.range_lower,
        range_upper=context.range_upper,
        range_equilibrium=context.range_equilibrium,
        lower_location=lower_location,
        midpoint_location=midpoint_location,
        upper_location=upper_location,
        reference_location=reference_location,
    )


def _timeframe_contexts(
    consolidation: CandidateConsolidationAnalysis,
) -> tuple[TimeframeContext, ...]:
    """One record per timeframe present in the eligibility reading.

    Ordered slowest to fastest rather than by whatever order the branch
    happened to assemble, so a snapshot with four timeframes and one with five
    read the same way.
    """
    present = {entry.timeframe: entry for entry in consolidation.eligibility.timeframes}
    ordered = [tf for tf in FEATURE_TIMEFRAME_ORDER if tf in present]
    unknown = [tf for tf in present if tf not in FEATURE_TIMEFRAME_ORDER]
    if unknown:
        raise CandidateFeatureError(
            f"eligibility carries timeframes outside the feature order: "
            f"{[tf.value for tf in unknown]}"
        )

    contexts: list[TimeframeContext] = []
    for timeframe in ordered:
        entry = present[timeframe]
        found = entry.active_dealing_range
        contexts.append(
            TimeframeContext(
                timeframe=timeframe,
                structure_bias=entry.structure_bias,
                active_range_id=None if found is None else found.range_id,
                range_direction=None if found is None else found.direction,
                range_lower=None if found is None else found.lower,
                range_upper=None if found is None else found.upper,
                range_equilibrium=None if found is None else found.equilibrium,
                decision_count=len(entry.decisions),
                eligible_count=sum(1 for decision in entry.decisions if decision.eligible),
            )
        )
    return tuple(contexts)


def _feature(
    candidate: ConsolidatedCandidate,
    *,
    observed_at: datetime,
    contexts: tuple[TimeframeContext, ...],
) -> CandidateFeature:
    facts = tuple(
        _support_fact(decision, observed_at=observed_at)
        for decision in candidate.supporting_decisions
    )
    if not facts:  # pragma: no cover - consolidation never emits an empty group
        raise CandidateFeatureError(
            f"candidate {candidate.consolidated_candidate_id} has no support"
        )

    formed = [fact.formed_at for fact in facts]
    ages = [fact.age_seconds for fact in facts]

    return CandidateFeature(
        candidate_id=candidate.consolidated_candidate_id,
        method_version=CANDIDATE_FEATURE_METHOD_VERSION,
        symbol=candidate.symbol,
        role=candidate.role,
        entry_side=candidate.entry_side,
        lower=candidate.lower,
        upper=candidate.upper,
        midpoint=candidate.midpoint,
        width=candidate.width,
        reference_level=candidate.reference_level,
        market_relation=candidate.market_relation,
        distance_to_reference=candidate.distance_to_reference,
        support_count=candidate.support_count,
        timeframe_count=candidate.timeframe_count,
        source_kind_count=len(candidate.source_kinds),
        source_kinds=candidate.source_kinds,
        timeframes=candidate.timeframes,
        oldest_source_formed_at=min(formed),
        newest_source_formed_at=max(formed),
        oldest_source_age_seconds=max(ages),
        newest_source_age_seconds=min(ages),
        support_facts=facts,
        range_contexts=tuple(
            _range_context(candidate, timeframe=context.timeframe, context=context)
            for context in contexts
        ),
    )


def relate_zones(
    *, first_lower: Decimal, first_upper: Decimal, second_lower: Decimal, second_upper: Decimal
) -> tuple[ZoneRelation, Decimal | None, Decimal | None, Decimal | None]:
    """Classify two closed intervals and give the exact intersection or gap.

    Closed intervals, matching the order-block convention: sharing a single
    boundary price is contact, so it is ``TOUCHING`` rather than ``DISJOINT``.
    No tolerance anywhere - 4010 and 4010.01 are apart, by exactly 0.01.

    Returns:
        The relation, then intersection lower, intersection upper and gap
        distance. Intersection is ``None`` for ``DISJOINT``; gap is ``None``
        for everything else.
    """
    if first_lower == second_lower and first_upper == second_upper:
        return ZoneRelation.EQUAL, first_lower, first_upper, None

    overlap_lower = max(first_lower, second_lower)
    overlap_upper = min(first_upper, second_upper)

    if overlap_lower > overlap_upper:
        return ZoneRelation.DISJOINT, None, None, overlap_lower - overlap_upper
    if overlap_lower == overlap_upper:
        return ZoneRelation.TOUCHING, overlap_lower, overlap_upper, None
    return ZoneRelation.OVERLAPPING, overlap_lower, overlap_upper, None


def _pair_relations(features: tuple[CandidateFeature, ...]) -> tuple[CandidatePairRelation, ...]:
    """Every entry-zone pair, once, keyed by its two ids in sorted order.

    Every pair is recorded, including ``DISJOINT`` ones. Recording only the
    interesting pairs would make absence ambiguous: a missing relation would
    mean either "these do not overlap" or "nobody looked".
    """
    zones = [entry for entry in features if entry.is_entry_zone]
    relations: list[CandidatePairRelation] = []

    for index, first in enumerate(zones):
        for second in zones[index + 1 :]:
            if first.lower is None or first.upper is None:
                raise CandidateFeatureError(f"entry zone {first.candidate_id} has no bounds")
            if second.lower is None or second.upper is None:
                raise CandidateFeatureError(f"entry zone {second.candidate_id} has no bounds")

            low, high = sorted((first.candidate_id, second.candidate_id))
            left = first if first.candidate_id == low else second
            right = second if first.candidate_id == low else first

            relation, lower, upper, gap = relate_zones(
                first_lower=left.lower,  # type: ignore[arg-type]
                first_upper=left.upper,  # type: ignore[arg-type]
                second_lower=right.lower,  # type: ignore[arg-type]
                second_upper=right.upper,  # type: ignore[arg-type]
            )
            relations.append(
                CandidatePairRelation(
                    first_candidate_id=low,
                    second_candidate_id=high,
                    relation=relation,
                    intersection_lower=lower,
                    intersection_upper=upper,
                    intersection_width=None if lower is None or upper is None else upper - lower,
                    gap_distance=gap,
                )
            )

    relations.sort(key=lambda found: (found.first_candidate_id, found.second_candidate_id))
    return tuple(relations)


def build_candidate_features(
    consolidation: CandidateConsolidationAnalysis,
) -> CandidateFeatureAnalysis:
    """Arrange the facts about every consolidated candidate.

    Args:
        consolidation: A finished consolidation reading. Nothing is recomputed
            from it: no composite, no projection, no eligibility, no reference
            price and no lower ICT authority is called.

    Raises:
        CandidateFeatureError: A source formed after the observation instant, a
            timeframe falls outside the known order, or an entry zone reaches
            here without bounds.
    """
    contexts = _timeframe_contexts(consolidation)
    features = tuple(
        _feature(candidate, observed_at=consolidation.observed_at, contexts=contexts)
        for candidate in consolidation.candidates
    )
    # Consolidation order, kept exactly. Re-sorting here by distance, support or
    # age would be a ranking wearing a display order's clothes, and the model
    # would read the order as an opinion the engine did not mean to express.

    return CandidateFeatureAnalysis(
        method_version=CANDIDATE_FEATURE_METHOD_VERSION,
        observed_at=consolidation.observed_at,
        symbol=consolidation.symbol,
        provider=consolidation.provider,
        provider_symbol=consolidation.provider_symbol,
        composite_config=consolidation.composite_config,
        eligibility_config=consolidation.eligibility_config,
        reference_price=consolidation.reference_price,
        candidates=features,
        pair_relations=_pair_relations(features),
        timeframe_contexts=contexts,
        consolidation=consolidation,
    )


__all__ = [
    "CANDIDATE_FEATURE_METHOD_VERSION",
    "FEATURE_TIMEFRAME_ORDER",
    "CandidateFeature",
    "CandidateFeatureAnalysis",
    "CandidateFeatureError",
    "CandidatePairRelation",
    "RangeContext",
    "SupportFact",
    "TimeframeContext",
    "ZoneRelation",
    "build_candidate_features",
    "relate_zones",
]
