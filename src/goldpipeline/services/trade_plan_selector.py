"""Which ranked candidates get published, decided by code rather than by a model.

Round 6.6g. The analyst has already done its only job: it ordered every
candidate in its bucket. Nothing here re-judges that order. What happens here is
the part that was never the model's to decide - how many zones a reader sees,
which redundant one disappears, and which farther level is worth naming.

**Ranking is complete; selection is deterministic.** The model was forbidden to
omit a candidate precisely so that omission would land here, in code that can be
read, replayed and argued with. A model that quietly dropped a zone would be
making a product decision inside a prompt.

**Only containment is redundancy.** Two zones that partially overlap are two
different propositions about two different price bands, and publishing both is
honest. A zone wholly inside another says nothing the outer one does not already
say, so the lower-ranked of the pair is suppressed - never merged, never
widened, never averaged. Touching at a single boundary is not containment.
There is no tolerance, no ATR multiple and no overlap percentage anywhere in
this module, because each of those is a threshold nobody chose.

**Five is a cap, not a target.** Three to five zones is the shape the product
wants when that many honest, non-redundant zones exist. When two exist, two are
published. Nothing is backfilled from the suppressed pile to reach a number, and
nothing is invented.

**Rank decides survival; price decides display.** Which candidate wins a
containment pair, which one carries ``VUNG_CHINH`` and which farther reference
is preferred all follow the analyst's order. What a reader sees is ordered by
price, nearest to the market first, because a published list ordered by a
model's confidence would be a recommendation this project has not agreed to make.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from goldpipeline.services.ict_candidate_consolidation import ConsolidatedCandidate
from goldpipeline.services.ict_candidate_eligibility import (
    CandidateRole,
    EntrySide,
    ReferencePrice,
)
from goldpipeline.services.ict_candidate_features import (
    CandidateFeature,
    CandidateFeatureAnalysis,
)
from goldpipeline.services.trade_analyst import (
    BAI_KEY,
    LOWER_KEY,
    RANKING_KEYS,
    SEO_KEY,
    UPPER_KEY,
    TradeAnalystRanking,
    expected_buckets,
)

logger = logging.getLogger(__name__)

TRADE_PLAN_SELECTION_METHOD_VERSION = "1.0.0"
"""Stamped on the selection, and separate from the ranking's own versions."""

MAX_ENTRY_ZONES_PER_SIDE = 6
"""A ceiling, and deliberately not a quota.

Round 6.7 raised it from five to six; nothing else about selection changed.
Nothing here counts *up* to a minimum: a side with two non-redundant zones
publishes two, and a side with none publishes none.
"""


class TradePlanSelectionError(ValueError):
    """The ranking did not describe the candidate set it was handed with."""


class ZoneLabel(StrEnum):
    """The only two labels this round is willing to attach.

    ``SCALP``, ``CANH_CA_TUAN`` and ``HOLD`` are deliberately absent. Each would
    have to be inferred from a timeframe, and a timeframe is not evidence for
    how long a reader should hold anything - the branch has no deterministic
    basis for those semantics yet, so it says nothing rather than guessing.
    """

    VUNG_CHINH = "VUNG_CHINH"
    SAU_HON = "SAU_HON"


class SelectionOutcome(StrEnum):
    """Why a ranked candidate did or did not reach the page."""

    SELECTED = "SELECTED"
    SUPPRESSED_CONTAINMENT = "SUPPRESSED_CONTAINMENT"
    BELOW_MAX_COUNT = "BELOW_MAX_COUNT"
    NOT_FARTHER = "NOT_FARTHER"
    """A reference level that is not strictly beyond every selected zone on its
    side, so publishing it as "sâu hơn" would be false."""

    NO_ENTRY_ZONE = "NO_ENTRY_ZONE"
    """A reference on a side that published no zone. A farther level alone is
    not a trade plan, so it is not published alone."""

    REFERENCE_ALREADY_SELECTED = "REFERENCE_ALREADY_SELECTED"
    """One farther reference per side; the rest are recorded, not published."""


@dataclass(frozen=True)
class SelectionDecision:
    """One ranked candidate and what became of it. The audit trail.

    Every candidate in the ranking gets exactly one of these, so "why is that
    zone not on the page" is answerable without re-running anything.
    """

    candidate_id: str
    bucket: str
    ai_rank: int
    outcome: SelectionOutcome
    suppressed_by_candidate_id: str | None = None


@dataclass(frozen=True)
class SelectedEntryZone:
    """One published zone, with the deterministic candidate behind it."""

    candidate_id: str
    side: EntrySide
    lower: Decimal
    upper: Decimal
    midpoint: Decimal
    label: ZoneLabel | None
    ai_rank: int
    candidate: ConsolidatedCandidate
    """Held whole, so a renderer never has to trust a price that travelled
    through a prompt: the three Decimals above are copies of this object's."""


@dataclass(frozen=True)
class SelectedReference:
    """One published farther level. A single price, and never a zone."""

    candidate_id: str
    role: CandidateRole
    level: Decimal
    label: ZoneLabel
    ai_rank: int
    candidate: ConsolidatedCandidate


@dataclass(frozen=True)
class TradePlanSelection:
    """Everything the renderer is allowed to see. No prose anywhere on it."""

    method_version: str
    observed_at: datetime
    symbol: str
    reference_price: ReferencePrice

    seo_entries: tuple[SelectedEntryZone, ...]
    bai_entries: tuple[SelectedEntryZone, ...]
    """In *display* order - nearest the market first - not in rank order."""

    seo_reference: SelectedReference | None
    bai_reference: SelectedReference | None

    decisions: tuple[SelectionDecision, ...]

    features: CandidateFeatureAnalysis
    ranking: TradeAnalystRanking

    def entries(self, side: EntrySide) -> tuple[SelectedEntryZone, ...]:
        return self.seo_entries if side is EntrySide.SEO else self.bai_entries

    def reference(self, side: EntrySide) -> SelectedReference | None:
        return self.seo_reference if side is EntrySide.SEO else self.bai_reference

    def main_zone(self, side: EntrySide) -> SelectedEntryZone | None:
        return next(
            (zone for zone in self.entries(side) if zone.label is ZoneLabel.VUNG_CHINH), None
        )

    def decision(self, candidate_id: str) -> SelectionDecision | None:
        return next((entry for entry in self.decisions if entry.candidate_id == candidate_id), None)

    @property
    def selected_candidate_ids(self) -> tuple[str, ...]:
        return tuple(zone.candidate_id for zone in (*self.seo_entries, *self.bai_entries)) + tuple(
            reference.candidate_id
            for reference in (self.seo_reference, self.bai_reference)
            if reference is not None
        )


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def contains(
    *, outer_lower: Decimal, outer_upper: Decimal, inner_lower: Decimal, inner_upper: Decimal
) -> bool:
    """Whether the first zone wholly contains the second. Closed, exact.

    Equal bounds count as containment in both directions, which is why §7 fails
    closed on a same-side duplicate rather than relying on this to hide one:
    two identical zones on one side mean consolidation did not do its job, and
    quietly suppressing one would destroy the evidence.
    """
    return outer_lower <= inner_lower and outer_upper >= inner_upper


def _redundant_against(
    zone: CandidateFeature, selected: list[SelectedEntryZone]
) -> SelectedEntryZone | None:
    """The already-selected zone that makes *zone* redundant, if any.

    Containment in *either* direction. A later, larger zone that swallows an
    earlier one says nothing new either - the earlier one is already on the
    page and is the higher-ranked of the two, so the newcomer is the one that
    goes.
    """
    if zone.lower is None or zone.upper is None:  # pragma: no cover - guarded upstream
        raise TradePlanSelectionError(f"entry zone {zone.candidate_id} has no bounds")

    for chosen in selected:
        if contains(
            outer_lower=chosen.lower,
            outer_upper=chosen.upper,
            inner_lower=zone.lower,
            inner_upper=zone.upper,
        ) or contains(
            outer_lower=zone.lower,
            outer_upper=zone.upper,
            inner_lower=chosen.lower,
            inner_upper=chosen.upper,
        ):
            return chosen
    return None


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


def _require_matching_ranking(
    features: CandidateFeatureAnalysis, ranking: TradeAnalystRanking
) -> None:
    """The ranking must be *this* feature analysis's ranking.

    A ranking is only a list of opaque ids, so one taken over a different
    instant, a different symbol or a different candidate set would resolve
    partially and publish a plan built from two different readings of the
    market. Checked field by field rather than trusted.
    """
    if ranking.symbol != features.symbol:
        raise TradePlanSelectionError(
            f"ranking is for {ranking.symbol}, features are for {features.symbol}"
        )
    if ranking.observed_at != features.observed_at:
        raise TradePlanSelectionError(
            f"ranking observed {ranking.observed_at.isoformat()}, "
            f"features observed {features.observed_at.isoformat()}"
        )
    if ranking.candidate_feature_method_version != features.method_version:
        raise TradePlanSelectionError(
            f"ranking was built against feature method version "
            f"{ranking.candidate_feature_method_version}, these features are "
            f"{features.method_version}"
        )

    expected = expected_buckets(features)
    for key in RANKING_KEYS:
        returned = ranking.buckets[key]
        if sorted(returned) != sorted(expected[key]):
            raise TradePlanSelectionError(
                f"{key} does not describe this candidate set: "
                f"expected {sorted(expected[key])}, got {sorted(returned)}"
            )


def _require_no_same_side_duplicate(features: CandidateFeatureAnalysis, side: EntrySide) -> None:
    """§7. Identical geometry on one side is a contradiction, not a duplicate.

    Round 6.6e.2c1 consolidates exact matches, so two candidates with the same
    role, side and bounds cannot both exist. If they do, something upstream is
    wrong, and deduplicating here would paper over it.
    """
    seen: dict[tuple[str, str], str] = {}
    for zone in features.entry_zones(side):
        if zone.lower is None or zone.upper is None:
            raise TradePlanSelectionError(f"entry zone {zone.candidate_id} has no bounds")
        key = (str(zone.lower), str(zone.upper))
        if key in seen:
            raise TradePlanSelectionError(
                f"{side.value} has two candidates with identical geometry "
                f"{key[0]}-{key[1]}: {seen[key]} and {zone.candidate_id}; "
                "exact duplicates must have been consolidated already"
            )
        seen[key] = zone.candidate_id


def _require_role(zone: CandidateFeature, *, side: EntrySide, bucket: str) -> None:
    if zone.role is not CandidateRole.ENTRY_ZONE:
        raise TradePlanSelectionError(
            f"{bucket} contains {zone.candidate_id}, which is a {zone.role.value}"
        )
    if zone.entry_side is not side:
        raise TradePlanSelectionError(
            f"{bucket} contains {zone.candidate_id}, whose side is "
            f"{zone.entry_side.value if zone.entry_side else 'none'}"
        )


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def _select_side(
    features: CandidateFeatureAnalysis,
    ranking: TradeAnalystRanking,
    *,
    side: EntrySide,
    bucket: str,
    decisions: list[SelectionDecision],
) -> list[SelectedEntryZone]:
    """Walk one bucket in rank order and keep what survives.

    Suppressed candidates are never reconsidered. Backfilling one to reach three
    would republish exactly the zone that was found to say nothing new.
    """
    _require_no_same_side_duplicate(features, side)
    selected: list[SelectedEntryZone] = []

    for position, candidate_id in enumerate(ranking.buckets[bucket], start=1):
        zone = features.feature(candidate_id)
        if zone is None:  # pragma: no cover - the provenance check catches this first
            raise TradePlanSelectionError(f"{bucket} names unknown candidate {candidate_id}")
        _require_role(zone, side=side, bucket=bucket)

        if len(selected) >= MAX_ENTRY_ZONES_PER_SIDE:
            decisions.append(
                SelectionDecision(
                    candidate_id=candidate_id,
                    bucket=bucket,
                    ai_rank=position,
                    outcome=SelectionOutcome.BELOW_MAX_COUNT,
                )
            )
            continue

        blocking = _redundant_against(zone, selected)
        if blocking is not None:
            decisions.append(
                SelectionDecision(
                    candidate_id=candidate_id,
                    bucket=bucket,
                    ai_rank=position,
                    outcome=SelectionOutcome.SUPPRESSED_CONTAINMENT,
                    suppressed_by_candidate_id=blocking.candidate_id,
                )
            )
            continue

        candidate = features.candidate(candidate_id)
        if candidate is None:  # pragma: no cover - features always carry it
            raise TradePlanSelectionError(f"{candidate_id} has no deterministic candidate")
        if candidate.lower is None or candidate.upper is None or candidate.midpoint is None:
            raise TradePlanSelectionError(f"candidate {candidate_id} has no zone geometry")

        selected.append(
            SelectedEntryZone(
                candidate_id=candidate_id,
                side=side,
                # Copied from the consolidated candidate, never from the feature
                # and never recomputed: this is the only place a published price
                # comes from.
                lower=candidate.lower,
                upper=candidate.upper,
                midpoint=candidate.midpoint,
                label=None,
                ai_rank=position,
                candidate=candidate,
            )
        )
        decisions.append(
            SelectionDecision(
                candidate_id=candidate_id,
                bucket=bucket,
                ai_rank=position,
                outcome=SelectionOutcome.SELECTED,
            )
        )

    return selected


def _label_main(zones: list[SelectedEntryZone]) -> list[SelectedEntryZone]:
    """``VUNG_CHINH`` goes to the first survivor in the analyst's order.

    Not to the nearest, not to the widest, and not to the one with the most
    support - deciding that again here would be a second ranking sitting quietly
    behind the first.
    """
    if not zones:
        return zones
    main = min(zones, key=lambda zone: zone.ai_rank)
    return [
        SelectedEntryZone(
            candidate_id=zone.candidate_id,
            side=zone.side,
            lower=zone.lower,
            upper=zone.upper,
            midpoint=zone.midpoint,
            label=ZoneLabel.VUNG_CHINH if zone is main else None,
            ai_rank=zone.ai_rank,
            candidate=zone.candidate,
        )
        for zone in zones
    ]


def _display_order(
    zones: list[SelectedEntryZone], *, side: EntrySide
) -> tuple[SelectedEntryZone, ...]:
    """Price order, nearest the market first.

    SEO climbs away from the market and BAI descends from it, so the secondary
    keys mirror each other. The candidate id is the final tie-break, present so
    two zones that agree on every price still order the same way twice.
    """
    if side is EntrySide.SEO:
        return tuple(
            sorted(
                zones,
                key=lambda zone: (
                    _distance(zone),
                    zone.lower,
                    zone.upper,
                    zone.candidate_id,
                ),
            )
        )
    return tuple(
        sorted(
            zones,
            key=lambda zone: (
                _distance(zone),
                -zone.upper,
                -zone.lower,
                zone.candidate_id,
            ),
        )
    )


def _distance(zone: SelectedEntryZone) -> Decimal:
    return zone.candidate.distance_to_reference


def _select_reference(
    features: CandidateFeatureAnalysis,
    ranking: TradeAnalystRanking,
    *,
    bucket: str,
    entries: tuple[SelectedEntryZone, ...],
    decisions: list[SelectionDecision],
) -> SelectedReference | None:
    """The first ranked level strictly beyond every published zone on that side.

    "Beyond" is exact and directional: an upper reference must sit above the
    highest published SEO edge, a lower one below the lowest published BAI edge.
    A level inside or on the boundary of the published zones is not farther
    than them, and calling it "sâu hơn" would simply be untrue.
    """
    upper_side = bucket == UPPER_KEY
    boundary: Decimal | None = None
    if entries:
        boundary = (
            max(zone.upper for zone in entries)
            if upper_side
            else min(zone.lower for zone in entries)
        )

    chosen: SelectedReference | None = None
    for position, candidate_id in enumerate(ranking.buckets[bucket], start=1):
        feature = features.feature(candidate_id)
        if feature is None:  # pragma: no cover - provenance check catches this
            raise TradePlanSelectionError(f"{bucket} names unknown candidate {candidate_id}")

        if not entries:
            decisions.append(
                SelectionDecision(
                    candidate_id=candidate_id,
                    bucket=bucket,
                    ai_rank=position,
                    outcome=SelectionOutcome.NO_ENTRY_ZONE,
                )
            )
            continue
        if chosen is not None:
            decisions.append(
                SelectionDecision(
                    candidate_id=candidate_id,
                    bucket=bucket,
                    ai_rank=position,
                    outcome=SelectionOutcome.REFERENCE_ALREADY_SELECTED,
                )
            )
            continue

        candidate = features.candidate(candidate_id)
        if candidate is None or candidate.reference_level is None:
            raise TradePlanSelectionError(f"reference {candidate_id} has no level")
        level = candidate.reference_level

        assert boundary is not None
        farther = level > boundary if upper_side else level < boundary
        if not farther:
            decisions.append(
                SelectionDecision(
                    candidate_id=candidate_id,
                    bucket=bucket,
                    ai_rank=position,
                    outcome=SelectionOutcome.NOT_FARTHER,
                )
            )
            continue

        chosen = SelectedReference(
            candidate_id=candidate_id,
            role=candidate.role,
            level=level,
            label=ZoneLabel.SAU_HON,
            ai_rank=position,
            candidate=candidate,
        )
        decisions.append(
            SelectionDecision(
                candidate_id=candidate_id,
                bucket=bucket,
                ai_rank=position,
                outcome=SelectionOutcome.SELECTED,
            )
        )

    return chosen


def select_trade_plan(
    features: CandidateFeatureAnalysis, ranking: TradeAnalystRanking
) -> TradePlanSelection:
    """Turn a complete ranking into the zones and levels a reader will see.

    Args:
        features: The deterministic feature analysis. The only source of
            geometry; nothing is recomputed from it and no engine below it is
            called.
        ranking: A validated ranking over *these* features. Its provenance and
            its candidate set are re-checked here rather than assumed.

    Raises:
        TradePlanSelectionError: The ranking describes a different reading, a
            bucket contains a candidate of the wrong role or side, or one side
            carries two candidates with identical geometry.
    """
    _require_matching_ranking(features, ranking)

    decisions: list[SelectionDecision] = []
    seo = _label_main(
        _select_side(features, ranking, side=EntrySide.SEO, bucket=SEO_KEY, decisions=decisions)
    )
    bai = _label_main(
        _select_side(features, ranking, side=EntrySide.BAI, bucket=BAI_KEY, decisions=decisions)
    )
    seo_entries = _display_order(seo, side=EntrySide.SEO)
    bai_entries = _display_order(bai, side=EntrySide.BAI)

    seo_reference = _select_reference(
        features, ranking, bucket=UPPER_KEY, entries=seo_entries, decisions=decisions
    )
    bai_reference = _select_reference(
        features, ranking, bucket=LOWER_KEY, entries=bai_entries, decisions=decisions
    )

    logger.info(
        "trade_plan.select seo=%d bai=%d seo_reference=%s bai_reference=%s",
        len(seo_entries),
        len(bai_entries),
        seo_reference is not None,
        bai_reference is not None,
    )

    return TradePlanSelection(
        method_version=TRADE_PLAN_SELECTION_METHOD_VERSION,
        observed_at=features.observed_at,
        symbol=features.symbol,
        reference_price=features.reference_price,
        seo_entries=seo_entries,
        bai_entries=bai_entries,
        seo_reference=seo_reference,
        bai_reference=bai_reference,
        decisions=tuple(decisions),
        features=features,
        ranking=ranking,
    )


__all__ = [
    "MAX_ENTRY_ZONES_PER_SIDE",
    "TRADE_PLAN_SELECTION_METHOD_VERSION",
    "SelectedEntryZone",
    "SelectedReference",
    "SelectionDecision",
    "SelectionOutcome",
    "TradePlanSelection",
    "TradePlanSelectionError",
    "ZoneLabel",
    "contains",
    "select_trade_plan",
]
