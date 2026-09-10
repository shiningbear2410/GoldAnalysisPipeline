"""The gate a trade plan must pass, and it checks a different thing.

Round 6.6h. The analysis gate asks whether a model's article survived review and
repair. None of that happened here: there is no draft, no verdict, no
finalization, and asking for them would be asking a document to account for a
history it does not have.

What this gate asks instead is the only question a deterministic page can be
wrong about: **does the published text say exactly what the selection decided,
and did every number in it come from a candle?** It re-renders the plan from the
persisted selection and compares bytes, then walks each published price back to
a consolidated candidate, then checks the vocabulary, the length and the
absence of everything the contract forbids.

It writes the same ``publish_decision.json`` the analysis gate writes, under its
own ``gate_version``. That is what lets review delivery, the worker and every
existing safety rule stay exactly as they are: a decision is a decision, and the
subsystem downstream never needed to know which gate reached it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from goldpipeline.domain.errors import PipelineError
from goldpipeline.schemas.common import utc_now
from goldpipeline.schemas.manifest import RunManifest, RunStatus
from goldpipeline.schemas.publish import (
    BlockerCode,
    CheckId,
    CheckStatus,
    Decision,
    GateCheck,
    GateFinding,
    PublishDecision,
)
from goldpipeline.schemas.review import Severity
from goldpipeline.services.publish_gate import DECISION_FILENAME
from goldpipeline.services.trade_plan_render import (
    BAI_HEADING,
    EMPTY_SIDE,
    FARTHER_SUFFIX,
    FORBIDDEN_SUBSTRINGS,
    MAIN_ZONE_SUFFIX,
    MAX_TRADE_PLAN_CHARS,
    PUBLIC_VOCABULARY,
    SEO_HEADING,
    ZONE_DASH,
)
from goldpipeline.services.trade_plan_stage import (
    CANDIDATES_FILENAME,
    FINAL_ARTICLE_FILENAME,
    POLICY_FILENAME,
    RANKING_FILENAME,
    SELECTION_FILENAME,
    TRADE_PLAN_ARTIFACTS,
)
from goldpipeline.storage.atomic import sha256_bytes
from goldpipeline.storage.run_store import PreparedArtifact, RunDirectory, RunStore

logger = logging.getLogger(__name__)

TRADE_PLAN_GATE_VERSION = "gold_trade_plan_gate_v1"
"""Named separately from the analysis gate, because it proves a different thing.

A decision that said ``gold_publish_gate_v1`` over a document with no review
would be claiming checks that never ran.
"""


class TradePlanGateError(PipelineError):
    """The Run could not be gated at all - not the same as being blocked."""


def gate_trade_plan(
    *, run_id: str, store: RunStore, now: datetime | None = None
) -> PublishDecision:
    """Decide whether a finalized trade plan may be shown to a human.

    Args:
        run_id: The Run to gate. Must be ``FINALIZED`` with trade plan artifacts.
        store: Where Runs live.
        now: Injection point for tests.

    Returns:
        The committed :class:`PublishDecision`. ``BLOCKED`` is a decision, not
        an error, and it leaves the Run at ``PUBLISH_BLOCKED``.

    Raises:
        TradePlanGateError: The Run is not gateable, or already has a decision.
    """
    run = store.open(run_id)
    manifest = run.load_manifest()

    if run.has_artifact(DECISION_FILENAME):
        raise TradePlanGateError(f"run {run_id} already has a publish decision", run_id=run_id)
    if manifest.status is not RunStatus.FINALIZED:
        raise TradePlanGateError(
            f"run {run_id} is {manifest.status}, the trade plan gate needs {RunStatus.FINALIZED}",
            run_id=run_id,
            status=str(manifest.status),
        )

    checks = _run_checks(run, manifest)
    blockers = [finding for check in checks for finding in check.blocking_findings]
    warnings = [
        finding for check in checks for finding in check.findings if not finding.is_blocking
    ]

    article = (
        run.read_artifact_bytes(FINAL_ARTICLE_FILENAME)
        if run.has_artifact(FINAL_ARTICLE_FILENAME)
        else b""
    )

    decision = PublishDecision(
        gate_version=TRADE_PLAN_GATE_VERSION,
        run_id=run_id,
        stage="trade_plan_gate",
        decision=Decision.BLOCKED if blockers else Decision.APPROVED,
        created_at=now or utc_now(),
        checks=checks,
        blockers=blockers,
        warnings=warnings,
        # No review and no finalization happened, so both stay null rather than
        # being filled in with a value that would imply one did.
        review_status=None,
        finalization_mode=None,
        article_chars=len(article.decode("utf-8")) if article else None,
        final_article_sha256=sha256_bytes(article) if article else None,
    )

    return _commit(run=run, manifest=manifest, decision=decision)


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def _run_checks(run: RunDirectory, manifest: RunManifest) -> list[GateCheck]:
    """Integrity first; every later check reads content the manifest vouches for."""
    integrity = _check_integrity(run, manifest)
    checks = [integrity]
    if integrity.status is CheckStatus.FAIL:
        return checks

    selection = json.loads(run.read_artifact_bytes(SELECTION_FILENAME).decode("utf-8"))
    candidates = json.loads(run.read_artifact_bytes(CANDIDATES_FILENAME).decode("utf-8"))
    ranking = json.loads(run.read_artifact_bytes(RANKING_FILENAME).decode("utf-8"))
    article = run.read_artifact_bytes(FINAL_ARTICLE_FILENAME).decode("utf-8").rstrip("\n")

    checks.append(_check_run_state(manifest))
    checks.append(_check_render_match(article, selection))
    checks.append(_check_geometry_provenance(selection, candidates))
    checks.append(_check_ranking_provenance(selection, ranking, candidates))
    checks.append(_check_structure(article))
    checks.append(_check_vocabulary(article, selection))
    return checks


def _finding(code: BlockerCode, message: str, *, source: str | None = None) -> GateFinding:
    return GateFinding(code=code, severity=Severity.CRITICAL, message=message[:1000], source=source)


def _check(check_id: CheckId, description: str, findings: list[GateFinding]) -> GateCheck:
    return GateCheck(
        check_id=check_id,
        status=CheckStatus.FAIL if findings else CheckStatus.PASS,
        description=description,
        findings=findings,
    )


def _check_integrity(run: RunDirectory, manifest: RunManifest) -> GateCheck:
    """Every trade plan artifact present, and each one the file the stage wrote."""
    findings: list[GateFinding] = []
    recorded = {ref.name: ref for ref in [*manifest.source_files, *manifest.artifact_files]}

    for name in TRADE_PLAN_ARTIFACTS:
        if not run.has_artifact(name):
            findings.append(
                _finding(BlockerCode.ARTIFACT_INTEGRITY_FAILURE, f"missing artifact {name}")
            )
            continue
        ref = recorded.get(name)
        if ref is None:
            findings.append(
                _finding(
                    BlockerCode.UNEXPECTED_ARTIFACT,
                    f"{name} is on disk but not in the manifest",
                    source=name,
                )
            )
            continue
        digest = sha256_bytes(run.read_artifact_bytes(name))
        if digest != ref.sha256:
            findings.append(
                _finding(
                    BlockerCode.ARTIFACT_INTEGRITY_FAILURE,
                    f"{name} does not match the digest the manifest recorded",
                    source=name,
                )
            )

    if not run.has_artifact(POLICY_FILENAME):
        findings.append(
            _finding(BlockerCode.ARTIFACT_INTEGRITY_FAILURE, "no production policy snapshot")
        )

    return _check(
        CheckId.ARTIFACT_CHAIN_INTEGRITY,
        "every trade plan artifact is present and matches its recorded digest",
        findings,
    )


def _check_run_state(manifest: RunManifest) -> GateCheck:
    findings: list[GateFinding] = []
    if manifest.status is not RunStatus.FINALIZED:
        findings.append(
            _finding(BlockerCode.RUN_NOT_FINALIZED, f"run is {manifest.status}, not FINALIZED")
        )
    return _check(CheckId.RUN_STATE, "the run is finalized", findings)


def _check_render_match(article: str, selection: dict[str, Any]) -> GateCheck:
    """The published text is byte-for-byte the text the selection produced.

    The selection artifact carries the rendering the stage made. Comparing the
    two proves nothing edited the page between the renderer and the disk - which
    is the only way a model's words could ever reach this document.
    """
    findings: list[GateFinding] = []
    expected = str(selection.get("final_text", ""))

    if not expected:
        findings.append(
            _finding(
                BlockerCode.ARTICLE_EMPTY,
                "the selection records no rendered text",
                source=SELECTION_FILENAME,
            )
        )
    elif article != expected:
        findings.append(
            _finding(
                BlockerCode.ARTIFACT_INTEGRITY_FAILURE,
                "the published text is not the text the deterministic renderer produced",
                source=FINAL_ARTICLE_FILENAME,
            )
        )

    if expected and len(expected) != int(selection.get("final_chars", -1)):
        findings.append(
            _finding(
                BlockerCode.ARTIFACT_INTEGRITY_FAILURE,
                "the selection's recorded character count does not match its text",
                source=SELECTION_FILENAME,
            )
        )

    return _check(
        CheckId.CONTEXT_CONSISTENCY,
        "the published text is exactly the deterministic rendering",
        findings,
    )


def _published_prices(selection: dict[str, Any]) -> set[str]:
    prices: set[str] = set()
    for key in ("seo_entries", "bai_entries"):
        for zone in selection.get(key, []):
            prices.add(str(zone["lower"]))
            prices.add(str(zone["upper"]))
    for key in ("seo_reference", "bai_reference"):
        entry = selection.get(key)
        if entry:
            prices.add(str(entry["level"]))
    return prices


def _check_geometry_provenance(selection: dict[str, Any], candidates: dict[str, Any]) -> GateCheck:
    """Every published number is a consolidated candidate's own price.

    Walked through the artifacts rather than through objects in memory, so the
    check proves what a reader of the Run could prove: this price, that
    candidate, those supporting decisions, that source.
    """
    findings: list[GateFinding] = []
    by_id = {
        entry["consolidated_candidate_id"]: entry for entry in candidates.get("consolidated", [])
    }
    decisions = {entry["candidate_id"]: entry for entry in candidates.get("decisions", [])}

    for key, is_zone in (("seo_entries", True), ("bai_entries", True)):
        for zone in selection.get(key, []):
            candidate = by_id.get(zone["candidate_id"])
            if candidate is None:
                findings.append(
                    _finding(
                        BlockerCode.SUSPICIOUS_PRICE,
                        f"published zone {zone['candidate_id']} is not a consolidated candidate",
                    )
                )
                continue
            if (zone["lower"], zone["upper"], zone["midpoint"]) != (
                candidate["lower"],
                candidate["upper"],
                candidate["midpoint"],
            ):
                findings.append(
                    _finding(
                        BlockerCode.SUSPICIOUS_PRICE,
                        f"published zone {zone['candidate_id']} does not match its candidate",
                    )
                )
            if is_zone and not candidate["supporting_candidate_ids"]:
                findings.append(
                    _finding(
                        BlockerCode.SUSPICIOUS_PRICE,
                        f"candidate {zone['candidate_id']} has no supporting decision",
                    )
                )
            for supporter in candidate["supporting_candidate_ids"]:
                decision = decisions.get(supporter)
                if decision is None or not decision["eligible"]:
                    findings.append(
                        _finding(
                            BlockerCode.SUSPICIOUS_PRICE,
                            f"supporting decision {supporter} is missing or was not eligible",
                        )
                    )

    for key in ("seo_reference", "bai_reference"):
        entry = selection.get(key)
        if not entry:
            continue
        candidate = by_id.get(entry["candidate_id"])
        if candidate is None or candidate["reference_level"] != entry["level"]:
            findings.append(
                _finding(
                    BlockerCode.SUSPICIOUS_PRICE,
                    f"published reference {entry['candidate_id']} does not match its candidate",
                )
            )

    return _check(
        CheckId.SUSPICIOUS_PRICE,
        "every published price resolves to a consolidated candidate and an eligible decision",
        findings,
    )


def _check_ranking_provenance(
    selection: dict[str, Any], ranking: dict[str, Any], candidates: dict[str, Any]
) -> GateCheck:
    """The selection came from the ranking, which ranked exactly these candidates."""
    findings: list[GateFinding] = []
    offered = {identity for values in ranking.get("offered", {}).values() for identity in values}
    ranked = {identity for values in ranking.get("ranked", {}).values() for identity in values}
    known = {entry["consolidated_candidate_id"] for entry in candidates.get("consolidated", [])}

    if offered != ranked:
        findings.append(
            _finding(
                BlockerCode.ARTIFACT_INTEGRITY_FAILURE,
                "the ranking does not contain exactly the candidates it was offered",
                source=RANKING_FILENAME,
            )
        )
    if not ranked <= known:
        findings.append(
            _finding(
                BlockerCode.ARTIFACT_INTEGRITY_FAILURE,
                "the ranking names candidates that were never consolidated",
                source=RANKING_FILENAME,
            )
        )

    published = {
        zone["candidate_id"]
        for key in ("seo_entries", "bai_entries")
        for zone in selection.get(key, [])
    } | {
        selection[key]["candidate_id"]
        for key in ("seo_reference", "bai_reference")
        if selection.get(key)
    }
    if not published <= ranked:
        findings.append(
            _finding(
                BlockerCode.ARTIFACT_INTEGRITY_FAILURE,
                "the plan publishes a candidate the analyst never ranked",
                source=SELECTION_FILENAME,
            )
        )

    if selection.get("observed_at") != ranking.get("observed_at"):
        findings.append(
            _finding(
                BlockerCode.ARTIFACT_INTEGRITY_FAILURE,
                "the ranking and the selection describe different instants",
            )
        )

    return _check(
        CheckId.NO_NEW_REGRESSION,
        "the selection belongs to a ranking over exactly this candidate set",
        findings,
    )


def _check_structure(article: str) -> GateCheck:
    """One SEO section, one BAI section, in that order, within the cap."""
    findings: list[GateFinding] = []
    lines = article.split("\n")

    if lines.count(SEO_HEADING) != 1:
        findings.append(
            _finding(BlockerCode.ARTICLE_LOOKS_LIKE_JSON, "expected exactly one SEO heading")
        )
    if lines.count(BAI_HEADING) != 1:
        findings.append(
            _finding(BlockerCode.ARTICLE_LOOKS_LIKE_JSON, "expected exactly one BAI heading")
        )
    if (
        SEO_HEADING in lines
        and BAI_HEADING in lines
        and lines.index(SEO_HEADING) > lines.index(BAI_HEADING)
    ):
        findings.append(_finding(BlockerCode.ARTICLE_LOOKS_LIKE_JSON, "SEO must precede BAI"))
    if not article.strip():
        findings.append(_finding(BlockerCode.ARTICLE_EMPTY, "the plan is empty"))
    if len(article) > MAX_TRADE_PLAN_CHARS:
        findings.append(
            _finding(
                BlockerCode.ARTICLE_TOO_LONG,
                f"the plan is {len(article)} characters, over the {MAX_TRADE_PLAN_CHARS} cap",
            )
        )
    if any(ord(character) < 32 and character != "\n" for character in article):
        findings.append(
            _finding(BlockerCode.ARTICLE_CONTROL_CHARACTERS, "the plan contains control characters")
        )

    return _check(CheckId.ARTICLE_STRUCTURE, "the plan has the contracted shape", findings)


def _check_vocabulary(article: str, selection: dict[str, Any]) -> GateCheck:
    """Nothing on the page but the allowed words, the allowed prices and digits.

    This is where a stop loss, a score, a disclaimer or a candidate id would be
    caught - not by naming each one, but because the vocabulary is closed and
    the price set is exactly what the selection chose.
    """
    findings: list[GateFinding] = []

    for forbidden in FORBIDDEN_SUBSTRINGS:
        if forbidden in article:
            findings.append(
                _finding(
                    BlockerCode.INSTRUCTION_SHAPED_TEXT,
                    f"the plan contains forbidden text {forbidden!r}",
                )
            )

    for key in ("seo_entries", "bai_entries"):
        for zone in selection.get(key, []):
            if zone["candidate_id"] in article:
                findings.append(
                    _finding(
                        BlockerCode.POSSIBLE_CREDENTIAL_EXPOSURE,
                        "a candidate id leaked into the published plan",
                    )
                )

    words = {
        word
        for line in article.split("\n")
        for word in line.replace("(", " ").replace(")", " ").split()
        if not any(character.isdigit() for character in word)
    }
    unknown = sorted(words - PUBLIC_VOCABULARY)
    if unknown:
        findings.append(
            _finding(
                BlockerCode.INSTRUCTION_SHAPED_TEXT,
                f"the plan uses words outside the public vocabulary: {unknown}",
            )
        )

    allowed = _published_prices(selection)
    for line in article.split("\n"):
        if line in {"", SEO_HEADING, BAI_HEADING, EMPTY_SIDE}:
            continue
        body = line.replace(MAIN_ZONE_SUFFIX, "").replace(FARTHER_SUFFIX, "").strip()
        for token in body.split(ZONE_DASH):
            token = token.strip()
            if token and token not in allowed:
                findings.append(
                    _finding(
                        BlockerCode.SUSPICIOUS_PRICE,
                        f"the plan shows {token!r}, which is not a selected price",
                    )
                )

    return _check(
        CheckId.INSTRUCTION_SHAPED_TEXT,
        "the plan uses only the public vocabulary and the selected prices",
        findings,
    )


# --------------------------------------------------------------------------
# commit
# --------------------------------------------------------------------------


def _commit(
    *, run: RunDirectory, manifest: RunManifest, decision: PublishDecision
) -> PublishDecision:
    """Write the decision, then move the Run to match it."""
    artifact = PreparedArtifact.from_json(DECISION_FILENAME, decision)
    run.commit_artifacts([artifact], manifest)

    status = RunStatus.READY_TO_PUBLISH if decision.approved else RunStatus.PUBLISH_BLOCKED
    passed, warned, failed = decision.counts

    manifest.status = status
    manifest.record_event(
        "trade_plan_gate",
        str(decision.decision),
        f"{passed} passed, {warned} warnings, {failed} failed; {len(decision.blockers)} blocker(s)",
    )
    run.save_manifest(manifest)

    logger.info(
        "run=%s stage=trade_plan_gate status=%s checks=%d/%d/%d blockers=%d",
        run.run_id,
        decision.decision,
        passed,
        warned,
        failed,
        len(decision.blockers),
    )
    return decision


__all__ = [
    "TRADE_PLAN_GATE_VERSION",
    "TradePlanGateError",
    "gate_trade_plan",
]
