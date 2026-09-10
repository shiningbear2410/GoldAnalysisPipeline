"""Candidates that are the same candidate, and nothing that merely looks close.

Round 6.6e.2c1. Two eligible decisions become one consolidated candidate only
when their final candidate geometry is **exactly** identical - same role, same
side, same prices, compared as Decimal values. Everything softer is deferred.

**Why exactness is the whole content.** A zone of 3990-4010 and a zone of
4000-4020 overlap, and merging them would produce 3990-4020: a band neither
source ever drew, wider than both, and now indistinguishable from a real one.
Worse, overlap chains: 3990-4010, 4000-4020 and 4010-4030 would collapse into
3990-4030 by transitivity, which is not consolidation but erasure. Exact
equality has neither problem - the consolidated geometry is precisely the
geometry every member already had, so nothing is invented and nothing can chain.

**Support is a count, not a verdict.** Two sources landing on one price is a
fact worth recording and it is recorded as ``support_count``. It is not a score,
and nothing here treats two supporters as better than one. Whether agreement
means anything is a question for whatever eventually ranks candidates.

**Consolidation is a view, not a history.** A group exists at an instant
because at least one eligible decision supports it. Supporters can be lost as
well as gained - a block gets invalidated, a policy narrows - and a group can
disappear and reappear. Only the underlying source histories are monotonic;
imposing that on groups would be a false invariant.

**Eligible decisions only, and every one of them.** Ineligible decisions get no
group, are not deleted, and stay reachable through the eligibility analysis this
module is handed. Every eligible decision belongs to exactly one group.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from goldpipeline.schemas.common import Timeframe
from goldpipeline.services.ict_candidate_eligibility import (
    CandidateEligibilityAnalysis,
    CandidateEligibilityConfig,
    CandidateEligibilityDecision,
    CandidateRole,
    EntrySide,
    MarketRelation,
    ReferencePrice,
)
from goldpipeline.services.ict_candidate_source import CandidateSourceKind
from goldpipeline.services.ict_composite import IctCompositeConfig

logger = logging.getLogger(__name__)

CANDIDATE_CONSOLIDATION_METHOD_VERSION = "1.0.0"
"""Stamped on every candidate and analysis, and part of every group identity."""


class CandidateConsolidationError(ValueError):
    """The grouping could not honestly answer, so it refused to guess."""


def canonical_price(price: Decimal) -> str:
    """A price as it appears in a group identity, trailing zeros removed.

    ``Decimal("4000")``, ``Decimal("4000.0")`` and ``Decimal("4000.00")`` are
    one price written at three precisions, and providers disagree about that
    routinely. Two decisions carrying the same price at different precision
    describe the same candidate and must reach the same identity.

    Deliberately a **new** helper rather than an extraction of
    :func:`~goldpipeline.services.ict_fvg._canonical`, which is identical in
    behaviour. Extracting that one would mean editing the module that mints
    every historical fair-value-gap identity, and no tidiness is worth the risk
    of moving an id that already exists in stored analyses. The two are pinned
    equal across a wide range of values in
    ``tests/test_ict_candidate_consolidation.py``; unifying them is a backlog
    item with a byte-identical-or-not-at-all condition attached.

    Note that ``Decimal("-0.0")`` canonicalises to ``"-0"`` rather than ``"0"``.
    No price on this branch is negative, so the two never meet; it is recorded
    here rather than guarded against, because a guard would imply the case is
    reachable.
    """
    return format(price.normalize(), "f")


@dataclass(frozen=True)
class ConsolidatedCandidate:
    """One exact candidate, and every eligible decision that supports it.

    Geometry follows the role exactly as it does on a decision: an entry zone
    carries the zone and no level, a reference carries the level and no zone.
    Neither is derived - both are copied from members that already agreed.
    """

    consolidated_candidate_id: str
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
    """Shared by every member, because they share geometry, role and the one
    global reference price. Verified rather than picked from the first member."""

    supporting_decisions: tuple[CandidateEligibilityDecision, ...]
    """The decisions themselves, in eligibility order.

    Held whole rather than as bare ids so a later feature stage can walk
    candidate → decision → source → the original block, gap or pool without a
    single geometry-based lookup.
    """

    source_kinds: tuple[CandidateSourceKind, ...]
    timeframes: tuple[Timeframe, ...]

    @property
    def supporting_candidate_ids(self) -> tuple[str, ...]:
        """One per supporting decision.

        A decision has no identity separate from its ``candidate_id``, so this
        is also the supporting-decision id list - exposed once rather than
        twice under two names for one thing.
        """
        return tuple(decision.candidate_id for decision in self.supporting_decisions)

    @property
    def supporting_candidate_source_ids(self) -> tuple[str, ...]:
        return tuple(decision.candidate_source_id for decision in self.supporting_decisions)

    @property
    def supporting_source_ids(self) -> tuple[str, ...]:
        """The original market objects' own ids - an order block's, a gap's, a pool's."""
        return tuple(decision.source_id for decision in self.supporting_decisions)

    @property
    def support_count(self) -> int:
        """How many eligible decisions landed on this exact candidate. A count."""
        return len(self.supporting_decisions)

    @property
    def timeframe_count(self) -> int:
        """How many distinct timeframes support it. Also a count, and H4 is not
        worth more than M1 here."""
        return len(self.timeframes)


@dataclass(frozen=True)
class CandidateConsolidationAnalysis:
    """Every exact candidate at one instant, over one eligibility reading."""

    method_version: str
    observed_at: datetime
    symbol: str
    provider: str
    provider_symbol: str | None

    composite_config: IctCompositeConfig
    eligibility_config: CandidateEligibilityConfig
    reference_price: ReferencePrice

    eligibility: CandidateEligibilityAnalysis
    """The reading this consolidation is a view of.

    Retained whole, which is what keeps ineligible decisions and their reasons
    reachable: they have no group, and they have not been thrown away.
    """

    candidates: tuple[ConsolidatedCandidate, ...]

    def candidate(self, identity: str) -> ConsolidatedCandidate | None:
        return next(
            (entry for entry in self.candidates if entry.consolidated_candidate_id == identity),
            None,
        )

    def of_role(self, role: CandidateRole) -> tuple[ConsolidatedCandidate, ...]:
        return tuple(entry for entry in self.candidates if entry.role is role)


_ROLE_ORDER: dict[CandidateRole, int] = {role: index for index, role in enumerate(CandidateRole)}
_SIDE_ORDER: dict[EntrySide | None, int] = {None: 0, EntrySide.BAI: 1, EntrySide.SEO: 2}
"""Fixed orders, so group ordering never depends on enum hashing."""


def _group_key(decision: CandidateEligibilityDecision) -> tuple[str, ...]:
    """What makes two decisions the same candidate. Exact values only.

    An entry zone is keyed on both edges and its side; a reference on its single
    level. Role is in both, so an entry zone whose boundary happens to equal a
    liquidity level never joins it - they are different kinds of thing that
    share a number.
    """
    if decision.role is CandidateRole.ENTRY_ZONE:
        if decision.entry_side is None or decision.lower is None or decision.upper is None:
            raise CandidateConsolidationError(
                f"entry zone {decision.candidate_id} is missing a side or an edge"
            )
        return (
            decision.symbol,
            decision.role.value,
            decision.entry_side.value,
            canonical_price(decision.lower),
            canonical_price(decision.upper),
        )

    if decision.entry_side is not None:
        raise CandidateConsolidationError(
            f"reference {decision.candidate_id} carries entry side {decision.entry_side.value}"
        )
    if decision.reference_level is None:
        raise CandidateConsolidationError(
            f"reference {decision.candidate_id} has no reference level"
        )
    return (
        decision.symbol,
        decision.role.value,
        canonical_price(decision.reference_level),
    )


def _consolidated_id(key: tuple[str, ...]) -> str:
    """Identity from role, side and exact geometry - and nothing else.

    Not the supporters, so a later identical source joining an existing
    candidate does not rename it. Not the reference price, the relation or the
    distance, so a price move that leaves the candidate eligible leaves its
    identity alone. Not the policy, so widening an allowed-status set changes
    who supports a candidate and never which candidate it is.
    """
    preimage = "|".join((CANDIDATE_CONSOLIDATION_METHOD_VERSION, *key))
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]


def _require_agreement(members: list[CandidateEligibilityDecision], key: tuple[str, ...]) -> None:
    """Members of one exact group must agree on everything derived from it.

    They share geometry, role and the single global reference price, so their
    relation and distance are forced to match. Checked rather than assumed: a
    disagreement would mean two decisions with identical geometry had been
    measured against different prices, and picking one would hide that.
    """
    first = members[0]
    for other in members[1:]:
        if other.market_relation is not first.market_relation:
            raise CandidateConsolidationError(
                f"exact candidate {key} has members disagreeing about market relation: "
                f"{first.market_relation.value} and {other.market_relation.value}"
            )
        if other.distance_to_reference != first.distance_to_reference:
            raise CandidateConsolidationError(
                f"exact candidate {key} has members disagreeing about distance: "
                f"{first.distance_to_reference} and {other.distance_to_reference}"
            )
        if other.entry_side is not first.entry_side:
            raise CandidateConsolidationError(
                f"exact candidate {key} has members disagreeing about entry side"
            )
        if (other.lower, other.upper, other.midpoint, other.width) != (
            first.lower,
            first.upper,
            first.midpoint,
            first.width,
        ):
            raise CandidateConsolidationError(
                f"exact candidate {key} has members disagreeing about zone geometry"
            )
        if other.reference_level != first.reference_level:
            raise CandidateConsolidationError(
                f"exact candidate {key} has members disagreeing about reference level"
            )


def _unique(values: list[str]) -> tuple[str, ...]:
    """First-seen order, without a set - so nothing depends on hashing."""
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return tuple(seen)


def _candidate(
    key: tuple[str, ...], members: list[CandidateEligibilityDecision]
) -> ConsolidatedCandidate:
    _require_agreement(members, key)
    first = members[0]

    kind_order = _unique([decision.source.kind.value for decision in members])
    timeframe_order = _unique([decision.timeframe.value for decision in members])

    return ConsolidatedCandidate(
        consolidated_candidate_id=_consolidated_id(key),
        method_version=CANDIDATE_CONSOLIDATION_METHOD_VERSION,
        symbol=first.symbol,
        role=first.role,
        entry_side=first.entry_side,
        lower=first.lower,
        upper=first.upper,
        midpoint=first.midpoint,
        width=first.width,
        reference_level=first.reference_level,
        market_relation=first.market_relation,
        distance_to_reference=first.distance_to_reference,
        supporting_decisions=tuple(members),
        source_kinds=tuple(CandidateSourceKind(value) for value in kind_order),
        timeframes=tuple(Timeframe(value) for value in timeframe_order),
    )


def consolidate_candidates(
    eligibility: CandidateEligibilityAnalysis,
) -> CandidateConsolidationAnalysis:
    """Group every eligible decision by exact candidate geometry.

    Walks the eligibility analysis in its own order - branch timeframe order,
    then decision order - so supporters keep that order inside each group and
    nothing depends on dictionary or set iteration.

    Args:
        eligibility: A finished eligibility reading. Nothing is recomputed from
            it: no reference price is resolved, no source is projected, and no
            lower ICT authority is called.

    Raises:
        CandidateConsolidationError: A decision contradicts its own role, or two
            decisions with identical geometry disagree about relation, distance,
            side or bounds.
    """
    groups: dict[tuple[str, ...], list[CandidateEligibilityDecision]] = {}
    order: list[tuple[str, ...]] = []

    for entry in eligibility.timeframes:
        for decision in entry.decisions:
            if not decision.eligible:
                # Not dropped - it stays reachable through `eligibility`, with
                # the reasons that ruled it out. It simply supports nothing.
                continue
            key = _group_key(decision)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(decision)

    candidates = [_candidate(key, groups[key]) for key in order]
    # Role, then side, then identity: a total order that needs no reference
    # price and expresses no preference. Sorting by distance or support count
    # would be ranking, which this round does not do.
    candidates.sort(
        key=lambda found: (
            _ROLE_ORDER[found.role],
            _SIDE_ORDER[found.entry_side],
            found.consolidated_candidate_id,
        )
    )

    return CandidateConsolidationAnalysis(
        method_version=CANDIDATE_CONSOLIDATION_METHOD_VERSION,
        observed_at=eligibility.observed_at,
        symbol=eligibility.symbol,
        provider=eligibility.provider,
        provider_symbol=eligibility.provider_symbol,
        composite_config=eligibility.composite_config,
        eligibility_config=eligibility.eligibility_config,
        reference_price=eligibility.reference_price,
        eligibility=eligibility,
        candidates=tuple(candidates),
    )


__all__ = [
    "CANDIDATE_CONSOLIDATION_METHOD_VERSION",
    "CandidateConsolidationAnalysis",
    "CandidateConsolidationError",
    "ConsolidatedCandidate",
    "canonical_price",
    "consolidate_candidates",
]
