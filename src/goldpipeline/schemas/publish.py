"""Publish gate contracts.

The last automated boundary before anything leaves the machine. Everything here
is decided in code: no model is consulted, so there is nothing to be persuaded
of and nothing to hallucinate.

The gate answers exactly one question - *is it safe to publish this article
automatically?* - and it answers ``APPROVED`` or ``BLOCKED``. It never edits,
sanitises, or retries. An article that cannot be approved needs a human or a new
Run, and quietly cleaning it up would defeat the point of having a boundary.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from goldpipeline.schemas.common import StrictModel, UtcDatetime, utc_now
from goldpipeline.schemas.finalizer import FinalizationMode
from goldpipeline.schemas.review import ReviewStatus, Severity

PUBLISH_SCHEMA_VERSION = "1.0.0"
"""Version of the publish decision contract."""

GATE_VERSION = "gold_publish_gate_v1"
"""Which gate produced a decision.

Recorded on every artifact so Round 6 can tell whether an approval came from a
gate it still trusts. There is no prompt to version here - the rules are code.
"""


class Decision(StrEnum):
    """The verdict. Two values, and no third.

    ``BLOCKED`` is a legitimate, fully-formed outcome - not a failure. The gate
    ran, reached a conclusion, and wrote it down. That is different from the
    stage crashing, and callers must be able to tell the two apart.
    """

    APPROVED = "APPROVED"
    BLOCKED = "BLOCKED"


class CheckStatus(StrEnum):
    """Outcome of one gate check."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class CheckId(StrEnum):
    """The checks this gate runs, in the order it runs them.

    A closed set so Round 6 - and a human reading a blocked decision - can
    branch on what actually failed rather than parse prose.
    """

    ARTIFACT_CHAIN_INTEGRITY = "ARTIFACT_CHAIN_INTEGRITY"
    RUN_STATE = "RUN_STATE"
    REVIEW_VERDICT_STATE = "REVIEW_VERDICT_STATE"
    REVIEW_ISSUE_CLOSURE = "REVIEW_ISSUE_CLOSURE"
    CORRECTION_CLOSURE = "CORRECTION_CLOSURE"
    ARTICLE_STRUCTURE = "ARTICLE_STRUCTURE"
    TELEGRAM_COMPATIBILITY = "TELEGRAM_COMPATIBILITY"
    INSTRUCTION_SHAPED_TEXT = "INSTRUCTION_SHAPED_TEXT"
    CREDENTIAL_EXPOSURE = "CREDENTIAL_EXPOSURE"
    FOREIGN_SYMBOL = "FOREIGN_SYMBOL"
    UNSUPPORTED_INDICATOR = "UNSUPPORTED_INDICATOR"
    SUSPICIOUS_PRICE = "SUSPICIOUS_PRICE"
    RISK_LANGUAGE = "RISK_LANGUAGE"
    EXTERNAL_FACT_WITHOUT_SOURCE = "EXTERNAL_FACT_WITHOUT_SOURCE"
    CONTEXT_CONSISTENCY = "CONTEXT_CONSISTENCY"
    NO_NEW_REGRESSION = "NO_NEW_REGRESSION"


class BlockerCode(StrEnum):
    """Why a check failed.

    Finer-grained than :class:`CheckId`: one check can fail for more than one
    reason, and the reason is what a human needs in order to act.
    """

    ARTIFACT_INTEGRITY_FAILURE = "ARTIFACT_INTEGRITY_FAILURE"
    RUN_NOT_FINALIZED = "RUN_NOT_FINALIZED"
    IMPOSSIBLE_REVIEW_STATE = "IMPOSSIBLE_REVIEW_STATE"
    UNRESOLVED_REVIEW_ISSUE = "UNRESOLVED_REVIEW_ISSUE"
    CORRECTION_NOT_APPLIED = "CORRECTION_NOT_APPLIED"

    ARTICLE_EMPTY = "ARTICLE_EMPTY"
    ARTICLE_TOO_SHORT = "ARTICLE_TOO_SHORT"
    ARTICLE_TOO_LONG = "ARTICLE_TOO_LONG"
    ARTICLE_CONTROL_CHARACTERS = "ARTICLE_CONTROL_CHARACTERS"
    ARTICLE_LOOKS_LIKE_JSON = "ARTICLE_LOOKS_LIKE_JSON"
    ARTICLE_CONTAINS_TRACEBACK = "ARTICLE_CONTAINS_TRACEBACK"
    ARTICLE_CONTAINS_CODE_BLOCK = "ARTICLE_CONTAINS_CODE_BLOCK"
    ARTICLE_NOT_UTF8 = "ARTICLE_NOT_UTF8"

    INSTRUCTION_SHAPED_TEXT = "INSTRUCTION_SHAPED_TEXT"
    POSSIBLE_CREDENTIAL_EXPOSURE = "POSSIBLE_CREDENTIAL_EXPOSURE"
    FOREIGN_SYMBOL_MENTIONED = "FOREIGN_SYMBOL_MENTIONED"
    UNSUPPORTED_INDICATOR_MENTIONED = "UNSUPPORTED_INDICATOR_MENTIONED"
    SUSPICIOUS_PRICE = "SUSPICIOUS_PRICE"
    ABSOLUTE_RISK_LANGUAGE = "ABSOLUTE_RISK_LANGUAGE"
    EXTERNAL_FACT_WITHOUT_SOURCE = "EXTERNAL_FACT_WITHOUT_SOURCE"
    SYMBOL_CONTRADICTS_CONTEXT = "SYMBOL_CONTRADICTS_CONTEXT"
    TIMEFRAME_CONTRADICTS_CONTEXT = "TIMEFRAME_CONTRADICTS_CONTEXT"
    NEW_REGRESSION_SINCE_DRAFT = "NEW_REGRESSION_SINCE_DRAFT"

    UNEXPECTED_ARTIFACT = "UNEXPECTED_ARTIFACT"


BLOCKING_SEVERITIES = frozenset({Severity.HIGH, Severity.CRITICAL})
"""Severities that turn a finding into a blocker.

The gate fails closed: anything at HIGH or above stops publication. LOW and
MEDIUM are recorded as warnings, because a style nit is not a reason to hold an
otherwise correct article.
"""


class GateFinding(StrictModel):
    """One thing a check noticed.

    ``evidence`` is a short, human-readable excerpt - and never a credential. A
    finding about a leaked token carries its *shape* and a redaction, so the
    decision artifact can be read, logged and shared without becoming a second
    copy of the secret.
    """

    code: BlockerCode
    severity: Severity
    message: str = Field(min_length=1, max_length=1000)
    evidence: str | None = Field(
        default=None,
        max_length=400,
        description="Redacted excerpt. Never the raw value of a detected secret.",
    )
    source: str | None = Field(
        default=None,
        max_length=200,
        description="Where it came from, e.g. 'claude_final.md' or a context path.",
    )
    position: int | None = Field(
        default=None, ge=0, description="Character offset in the article, when applicable."
    )

    @property
    def is_blocking(self) -> bool:
        """Whether this finding alone stops publication."""
        return self.severity in BLOCKING_SEVERITIES


class GateCheck(StrictModel):
    """The result of one deterministic check."""

    check_id: CheckId
    status: CheckStatus
    description: str = Field(min_length=1, max_length=400)
    findings: list[GateFinding] = Field(default_factory=list)

    @property
    def blocking_findings(self) -> list[GateFinding]:
        """Findings severe enough to stop publication."""
        return [finding for finding in self.findings if finding.is_blocking]


class PublishDecision(StrictModel):
    """The ``publish_decision.json`` artifact.

    Round 6 must treat this as the sole authority: it publishes only when
    ``decision`` is ``APPROVED`` **and** the digests below still match the
    artifacts on disk. An approval describes the exact bytes it was given.
    """

    schema_version: str = Field(default=PUBLISH_SCHEMA_VERSION)
    gate_version: str = Field(default=GATE_VERSION)
    run_id: str
    stage: str = Field(default="publish_gate")

    decision: Decision
    created_at: UtcDatetime = Field(default_factory=utc_now)

    checks: list[GateCheck] = Field(default_factory=list)
    blockers: list[GateFinding] = Field(
        default_factory=list, description="Every blocking finding, flattened."
    )
    warnings: list[GateFinding] = Field(
        default_factory=list, description="Non-blocking findings, recorded but not fatal."
    )

    review_status: ReviewStatus | None = Field(
        default=None, description="Null when the review could not be read."
    )
    finalization_mode: FinalizationMode | None = Field(
        default=None, description="Null when the finalization could not be read."
    )
    article_chars: int | None = Field(default=None, ge=0)

    context_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    draft_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    writer_metadata_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    review_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    final_article_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    finalizer_metadata_sha256: str | None = Field(default=None, min_length=64, max_length=64)

    @property
    def approved(self) -> bool:
        """Whether Round 6 may publish."""
        return self.decision is Decision.APPROVED

    @property
    def counts(self) -> tuple[int, int, int]:
        """``(passed, warned, failed)`` across the checks that ran."""
        passed = sum(1 for check in self.checks if check.status is CheckStatus.PASS)
        warned = sum(1 for check in self.checks if check.status is CheckStatus.WARN)
        failed = sum(1 for check in self.checks if check.status is CheckStatus.FAIL)
        return passed, warned, failed


__all__ = [
    "BLOCKING_SEVERITIES",
    "GATE_VERSION",
    "PUBLISH_SCHEMA_VERSION",
    "BlockerCode",
    "CheckId",
    "CheckStatus",
    "Decision",
    "GateCheck",
    "GateFinding",
    "PublishDecision",
]
