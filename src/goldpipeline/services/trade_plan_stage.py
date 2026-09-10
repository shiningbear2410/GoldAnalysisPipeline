"""The TRADE_PLAN stage: live candles in, a published page and its evidence out.

Round 6.6h. Every engine this calls was built and proved in an earlier round;
nothing here reimplements a market rule. What this module owns is the order the
engines run in, the single instant they all share, and the audit trail that
makes a published price traceable back to the candle it came from.

**One question this Run must always be able to answer.** Given a price on the
page: which selected zone, which consolidated candidate, which eligibility
decision, which projected source, which order block or gap or pool, and which
candle. Every artifact below exists to keep one link of that chain, and the
chain is checked by the gate rather than assumed.

**The model is called once, in the middle.** Before it: deterministic analysis.
After it: deterministic selection and rendering. Its answer is text until the
validator has proved it names exactly the candidates it was given, and its only
contribution to the page is the order two zones appear in and which one is
called *vùng chính*.

**Failure is refusal, never a smaller plan.** A timeframe that will not fetch, a
reference price two feeds disagree about, a ranking that invents an id - each
stops the Run. A trade plan built from four timeframes, or from a repaired
ranking, would look exactly like a good one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from goldpipeline.adapters.mtf_market import (
    MultiTimeframeError,
    MultiTimeframeObservation,
)
from goldpipeline.adapters.trade_analyst_client import TradeAnalystClient
from goldpipeline.domain.errors import PipelineError, RunNotReadyError
from goldpipeline.schemas.manifest import RunManifest, RunStatus
from goldpipeline.schemas.news import CuratedNews
from goldpipeline.services.ict_candidate_consolidation import (
    CandidateConsolidationAnalysis,
    canonical_price,
    consolidate_candidates,
)
from goldpipeline.services.ict_candidate_eligibility import (
    CandidateEligibilityAnalysis,
    CandidateEligibilityError,
    analyse_candidate_eligibility,
)
from goldpipeline.services.ict_candidate_features import (
    CandidateFeatureAnalysis,
    build_candidate_features,
)
from goldpipeline.services.ict_composite import IctCompositeAnalysis, analyse_ict_composite
from goldpipeline.services.trade_analyst import (
    TradeAnalystError,
    build_trade_analyst_input,
    build_trade_analyst_prompt,
    expected_buckets,
    parse_ranking,
)
from goldpipeline.services.trade_plan_policy import (
    PRODUCTION_POLICY_V1,
    TradePlanProductionPolicyV1,
)
from goldpipeline.services.trade_plan_render import render_and_validate
from goldpipeline.services.trade_plan_selector import (
    TradePlanSelection,
    TradePlanSelectionError,
    select_trade_plan,
)
from goldpipeline.storage.run_store import PreparedArtifact, RunDirectory, RunStore

logger = logging.getLogger(__name__)

TRADE_PLAN_STAGE_VERSION = "trade_plan_stage_v1"

POLICY_FILENAME = "trade_plan_policy.json"
MARKET_FILENAME = "trade_plan_market.json"
CANDIDATES_FILENAME = "trade_plan_candidates.json"
ANALYST_REQUEST_FILENAME = "trade_plan_analyst_request.json"
ANALYST_RESPONSE_FILENAME = "trade_plan_analyst_response.json"
RANKING_FILENAME = "trade_plan_ranking.json"
SELECTION_FILENAME = "trade_plan_selection.json"
FINAL_ARTICLE_FILENAME = "claude_final.md"
"""The historical name, reused deliberately.

Review delivery reads this file and nothing else, and running a writer purely to
justify the filename would be inventing an author for a document that has none.
The name is a location in the Run layout, not a claim about who wrote it.
"""

TRADE_PLAN_ARTIFACTS = (
    POLICY_FILENAME,
    MARKET_FILENAME,
    CANDIDATES_FILENAME,
    ANALYST_REQUEST_FILENAME,
    ANALYST_RESPONSE_FILENAME,
    RANKING_FILENAME,
    SELECTION_FILENAME,
    FINAL_ARTICLE_FILENAME,
)


class TradePlanStageError(PipelineError):
    """The trade plan could not be built honestly, so it was not built."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message, **context)


@dataclass(frozen=True)
class TradePlanStageResult:
    """Outcome of one attempt at the stage."""

    run_id: str
    run_dir: Path
    status: RunStatus
    plan_chars: int | None = None
    provider: str | None = None
    model: str | None = None
    reference_price: Decimal | None = None
    candidate_count: int | None = None
    selected_count: int | None = None
    error: PipelineError | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is RunStatus.FINALIZED and self.plan_chars is not None


# --------------------------------------------------------------------------
# preconditions
# --------------------------------------------------------------------------


def _require_ready(run: RunDirectory, manifest: RunManifest) -> None:
    """Refuse unless the Run is normalized and has no plan already.

    Runs are immutable. A second pass over a Run that already has a plan would
    either duplicate an artifact or overwrite evidence, and both are worse than
    refusing.
    """
    existing = [name for name in TRADE_PLAN_ARTIFACTS if run.has_artifact(name)]
    if existing:
        raise TradePlanStageError(
            f"run {run.run_id} already has trade plan artifacts: {existing}. "
            "Runs are immutable; create a new Run instead.",
            run_id=run.run_id,
            artifacts=existing,
        )
    if manifest.status is not RunStatus.NORMALIZED:
        raise RunNotReadyError(
            f"run {run.run_id} is {manifest.status}, the trade plan stage needs "
            f"{RunStatus.NORMALIZED}",
            run_id=run.run_id,
            status=str(manifest.status),
        )


# --------------------------------------------------------------------------
# artifact bodies
# --------------------------------------------------------------------------


def _range_document(found: Any) -> dict[str, Any] | None:
    """The active dealing range, or the honest absence of one."""
    if found is None:
        return None
    return {
        "range_id": found.range_id,
        "direction": found.direction.value,
        "lower": canonical_price(found.lower),
        "upper": canonical_price(found.upper),
        "equilibrium": canonical_price(found.equilibrium),
    }


def _market_document(
    observation: MultiTimeframeObservation, composite: IctCompositeAnalysis
) -> dict[str, Any]:
    """How the candles were obtained, and what the composite made of them.

    A summary rather than the whole composite: the analyses themselves are
    reachable by replaying this Run's policy over these candles, and copying
    several megabytes of swings into every Run would make the audit trail harder
    to read rather than easier.
    """
    return {
        "stage_version": TRADE_PLAN_STAGE_VERSION,
        "provenance": observation.provenance(),
        "composite": [
            {
                "timeframe": entry.timeframe.value,
                "bar_count": entry.series.bar_count,
                "first_closed_at": entry.series.first_closed_at.isoformat(),
                "latest_closed_at": entry.series.latest_closed_at.isoformat(),
                "structure_bias": entry.structure.current_bias.value,
                "swing_count": len(entry.swings),
                "structure_break_count": len(entry.structure.breaks),
                "order_block_count": len(entry.order_blocks.order_blocks),
                "fair_value_gap_count": len(entry.gaps),
                "liquidity_pool_count": len(entry.liquidity.pools),
                "active_dealing_range": _range_document(entry.dealing_ranges.active_range),
            }
            for entry in composite.timeframes
        ],
    }


def _candidates_document(
    eligibility: CandidateEligibilityAnalysis, consolidation: CandidateConsolidationAnalysis
) -> dict[str, Any]:
    """Every decision and every consolidated candidate.

    Ineligible decisions are kept with their reasons. "Why is that zone not on
    the page" has to be answerable for a zone that never became a candidate at
    all, not only for one the selector suppressed.
    """
    return {
        "stage_version": TRADE_PLAN_STAGE_VERSION,
        "observed_at": eligibility.observed_at.isoformat(),
        "reference_price": {
            "price": canonical_price(eligibility.reference_price.price),
            "bar_close_time": eligibility.reference_price.bar_close_time.isoformat(),
            "witnesses": [
                {
                    "timeframe": witness.timeframe.value,
                    "bar_open_time": witness.bar_open_time.isoformat(),
                    "bar_close_time": witness.bar_close_time.isoformat(),
                    "close": canonical_price(witness.close),
                }
                for witness in eligibility.reference_price.witnesses
            ],
        },
        "decisions": [
            {
                "candidate_id": decision.candidate_id,
                "candidate_source_id": decision.candidate_source_id,
                "source_id": decision.source_id,
                "source_kind": decision.source.kind.value,
                "timeframe": decision.timeframe.value,
                "role": decision.role.value,
                "entry_side": None if decision.entry_side is None else decision.entry_side.value,
                "lower": None if decision.lower is None else canonical_price(decision.lower),
                "upper": None if decision.upper is None else canonical_price(decision.upper),
                "reference_level": (
                    None
                    if decision.reference_level is None
                    else canonical_price(decision.reference_level)
                ),
                "market_relation": decision.market_relation.value,
                "distance_to_reference": canonical_price(decision.distance_to_reference),
                "eligible": decision.eligible,
                "reasons": [reason.value for reason in decision.reasons],
                "source_formed_at": decision.source.formed_at.isoformat(),
            }
            for entry in eligibility.timeframes
            for decision in entry.decisions
        ],
        "consolidated": [
            {
                "consolidated_candidate_id": candidate.consolidated_candidate_id,
                "role": candidate.role.value,
                "entry_side": None if candidate.entry_side is None else candidate.entry_side.value,
                "lower": None if candidate.lower is None else canonical_price(candidate.lower),
                "upper": None if candidate.upper is None else canonical_price(candidate.upper),
                "midpoint": (
                    None if candidate.midpoint is None else canonical_price(candidate.midpoint)
                ),
                "reference_level": (
                    None
                    if candidate.reference_level is None
                    else canonical_price(candidate.reference_level)
                ),
                "support_count": candidate.support_count,
                "supporting_candidate_ids": list(candidate.supporting_candidate_ids),
                "supporting_source_ids": list(candidate.supporting_source_ids),
                "source_kinds": [kind.value for kind in candidate.source_kinds],
                "timeframes": [timeframe.value for timeframe in candidate.timeframes],
            }
            for candidate in consolidation.candidates
        ],
    }


def _selection_document(
    selection: TradePlanSelection, *, plan: str, news: CuratedNews | None
) -> dict[str, Any]:
    """What was published, what was suppressed, and why."""

    def zone(entry: Any) -> dict[str, Any]:
        return {
            "candidate_id": entry.candidate_id,
            "side": entry.side.value,
            "lower": canonical_price(entry.lower),
            "upper": canonical_price(entry.upper),
            "midpoint": canonical_price(entry.midpoint),
            "label": None if entry.label is None else entry.label.value,
            "ai_rank": entry.ai_rank,
        }

    def reference(entry: Any) -> dict[str, Any] | None:
        if entry is None:
            return None
        return {
            "candidate_id": entry.candidate_id,
            "role": entry.role.value,
            "level": canonical_price(entry.level),
            "label": entry.label.value,
            "ai_rank": entry.ai_rank,
        }

    return {
        "stage_version": TRADE_PLAN_STAGE_VERSION,
        "selection_method_version": selection.method_version,
        "observed_at": selection.observed_at.isoformat(),
        "symbol": selection.symbol,
        "reference_price": canonical_price(selection.reference_price.price),
        "news_context": (
            None
            if news is None
            else {
                "item_count": len(news.items),
                "omitted_count": news.omitted_count,
                "truncated_count": news.truncated_count,
                "channels": sorted({item.channel for item in news.items}),
                "trust_level": "UNTRUSTED",
            }
        ),
        "seo_entries": [zone(entry) for entry in selection.seo_entries],
        "bai_entries": [zone(entry) for entry in selection.bai_entries],
        "seo_reference": reference(selection.seo_reference),
        "bai_reference": reference(selection.bai_reference),
        "decisions": [
            {
                "candidate_id": decision.candidate_id,
                "bucket": decision.bucket,
                "ai_rank": decision.ai_rank,
                "outcome": decision.outcome.value,
                "suppressed_by_candidate_id": decision.suppressed_by_candidate_id,
            }
            for decision in selection.decisions
        ],
        "final_text": plan,
        "final_chars": len(plan),
    }


# --------------------------------------------------------------------------
# the stage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Built:
    artifacts: list[PreparedArtifact]
    plan: str
    provider: str
    model: str
    reference_price: Decimal
    candidate_count: int
    selected_count: int


def build_trade_plan(
    *,
    observation: MultiTimeframeObservation,
    analyst: TradeAnalystClient,
    policy: TradePlanProductionPolicyV1,
    news: CuratedNews | None,
    max_tokens: int,
) -> _Built:
    """Run the whole deterministic chain, call the analyst once, render the page.

    Pure over its inputs: it touches no Run and writes no file, so the offline
    tests can drive the entire product without a store.
    """
    composite = analyse_ict_composite(observation.snapshot, config=policy.composite_config())
    eligibility = analyse_candidate_eligibility(composite, config=policy.eligibility_config())
    consolidation = consolidate_candidates(eligibility)
    features: CandidateFeatureAnalysis = build_candidate_features(consolidation)

    request = build_trade_analyst_input(features, news=news)
    prompt = build_trade_analyst_prompt(request)
    payload = prompt.user

    response = analyst.rank(_request(system=prompt.system, user=payload, max_tokens=max_tokens))
    ranking = parse_ranking(response.text, features=features)
    selection = select_trade_plan(features, ranking)
    plan = render_and_validate(selection)

    buckets = expected_buckets(features)
    artifacts = [
        PreparedArtifact.from_json(POLICY_FILENAME, policy.snapshot()),
        PreparedArtifact.from_json(MARKET_FILENAME, _market_document(observation, composite)),
        PreparedArtifact.from_json(
            CANDIDATES_FILENAME, _candidates_document(eligibility, consolidation)
        ),
        PreparedArtifact.from_text(ANALYST_REQUEST_FILENAME, payload),
        PreparedArtifact.from_json(
            ANALYST_RESPONSE_FILENAME,
            {
                "provider": response.provider,
                "model": response.model,
                "text": response.text,
            },
        ),
        PreparedArtifact.from_json(
            RANKING_FILENAME,
            {
                "prompt_version": ranking.prompt_version,
                "method_version": ranking.method_version,
                "candidate_feature_method_version": ranking.candidate_feature_method_version,
                "observed_at": ranking.observed_at.isoformat(),
                "symbol": ranking.symbol,
                "offered": {key: list(values) for key, values in buckets.items()},
                "ranked": {key: list(values) for key, values in ranking.buckets.items()},
            },
        ),
        PreparedArtifact.from_json(
            SELECTION_FILENAME, _selection_document(selection, plan=plan, news=news)
        ),
        PreparedArtifact.from_text(FINAL_ARTICLE_FILENAME, plan),
    ]

    return _Built(
        artifacts=artifacts,
        plan=plan,
        provider=response.provider,
        model=response.model,
        reference_price=features.reference_price.price,
        candidate_count=len(features.candidates),
        selected_count=len(selection.seo_entries) + len(selection.bai_entries),
    )


def _request(*, system: str, user: str, max_tokens: int) -> Any:
    from goldpipeline.adapters.trade_analyst_client import TradeAnalystRequest

    return TradeAnalystRequest(system=system, user=user, max_tokens=max_tokens)


def write_trade_plan(
    *,
    run_id: str,
    store: RunStore,
    observe: Any,
    analyst: TradeAnalystClient,
    news: CuratedNews | None = None,
    policy: TradePlanProductionPolicyV1 = PRODUCTION_POLICY_V1,
    max_tokens: int = 2000,
    now: datetime | None = None,
) -> TradePlanStageResult:
    """Build and persist a trade plan for an existing normalized Run.

    Args:
        run_id: The Run to build for. Must be ``NORMALIZED``.
        store: Where Runs live.
        observe: Zero-argument callable returning a
            :class:`MultiTimeframeObservation`. A callable rather than a source
            so a Run that is not due for this stage never opens a socket.
        analyst: Any client satisfying the analyst protocol.
        news: Optional untrusted context, threaded verbatim.
        policy: The production policy. Persisted in full on the Run.
        max_tokens: Ceiling for the ranking call.
        now: Injection point for tests.

    Returns:
        A :class:`TradePlanStageResult`. The Run reaches ``FINALIZED`` on
        success - it skips ``DRAFTED`` and ``REVIEWED`` because no draft and no
        review happened, and recording either would be a false statement about
        what this Run did.
    """
    run = store.open(run_id)
    manifest = run.load_manifest()

    try:
        _require_ready(run, manifest)

        manifest.record_event(
            "trade_plan.start",
            "OK",
            f"policy={policy.version} provider={analyst.provider} model={analyst.model}",
        )
        run.save_manifest(manifest)

        observation = observe()
        built = build_trade_plan(
            observation=observation,
            analyst=analyst,
            policy=policy,
            news=news,
            max_tokens=max_tokens,
        )
    except (
        MultiTimeframeError,
        CandidateEligibilityError,
        TradeAnalystError,
        TradePlanSelectionError,
        ValueError,
    ) as exc:
        error = (
            exc
            if isinstance(exc, PipelineError)
            else TradePlanStageError(str(exc), run_id=run_id, kind=type(exc).__name__)
        )
        logger.warning("run=%s stage=trade_plan status=FAILED %s", run_id, exc)
        manifest.record_event("trade_plan", "FAILED", str(exc)[:400])
        run.save_manifest(manifest)
        return TradePlanStageResult(
            run_id=run_id, run_dir=run.path, status=manifest.status, error=error
        )
    except PipelineError as exc:
        logger.warning("run=%s stage=trade_plan status=FAILED %s", run_id, exc)
        manifest.record_event("trade_plan", "FAILED", str(exc)[:400])
        run.save_manifest(manifest)
        return TradePlanStageResult(
            run_id=run_id, run_dir=run.path, status=manifest.status, error=exc
        )

    # Every artifact lands together or none of them does. A Run holding a
    # rendered page with no evidence behind it would be exactly the thing this
    # branch exists to prevent.
    run.commit_artifacts(built.artifacts, manifest)

    manifest.status = RunStatus.FINALIZED
    manifest.record_event(
        "trade_plan",
        "COMPLETED",
        f"{built.selected_count} zone(s), {len(built.plan)} chars",
    )
    run.save_manifest(manifest)

    logger.info(
        "run=%s stage=trade_plan status=COMPLETED candidates=%d selected=%d chars=%d",
        run_id,
        built.candidate_count,
        built.selected_count,
        len(built.plan),
    )
    return TradePlanStageResult(
        run_id=run_id,
        run_dir=run.path,
        status=RunStatus.FINALIZED,
        plan_chars=len(built.plan),
        provider=built.provider,
        model=built.model,
        reference_price=built.reference_price,
        candidate_count=built.candidate_count,
        selected_count=built.selected_count,
        error=None,
    )


__all__ = [
    "ANALYST_REQUEST_FILENAME",
    "ANALYST_RESPONSE_FILENAME",
    "CANDIDATES_FILENAME",
    "FINAL_ARTICLE_FILENAME",
    "MARKET_FILENAME",
    "POLICY_FILENAME",
    "RANKING_FILENAME",
    "SELECTION_FILENAME",
    "TRADE_PLAN_ARTIFACTS",
    "TRADE_PLAN_STAGE_VERSION",
    "TradePlanStageError",
    "TradePlanStageResult",
    "build_trade_plan",
    "write_trade_plan",
]
