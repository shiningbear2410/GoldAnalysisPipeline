"""Holding a written ANALYSIS to its shape, before it becomes a draft.

Round 6.4a built the contract vocabulary and the deterministic checks and left
them unwired, so they could be reviewed before they could refuse anything. This
module is the narrow piece that wires them - for ``ANALYSIS`` only, at one seam,
enforcing only what a machine can decide without pretending to judge prose.

**What blocks, and why only this.** Four things, each one a fact about the
document rather than an opinion about the writing:

* the article exceeds the type's hard character cap;
* a required section is missing, or a section belonging to another product is
  present;
* the disclaimer count is wrong for this type;
* a sentence links a news reference to a price move with an unqualified causal
  connector - the one wording the prompt explicitly forbids, and the one whose
  absence no reviewer can be relied upon to notice.

Everything else is reported and not enforced. The style symptoms in particular
are counted, attached to the failure detail when something else already blocked,
and otherwise logged - because a connective count is evidence about prose, not a
verdict on it, and the round that gives style a verdict is the next one.

**Not a global gate.** The check runs when the Run's article type is
``ANALYSIS`` and the type's contract says a model wrote it. ``NEWS_DIGEST`` and
``TRADE_PLAN`` are not ready, have no writer, and are not silently brought under
a rule written for a product they are not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.article_contract import (
    ArticleContract,
    GenerationMode,
    contract_for,
)
from goldpipeline.schemas.output_findings import OutputFinding, OutputFindingCode
from goldpipeline.services.article_contract_checks import check_contract, length_report
from goldpipeline.services.causality_language import find_causal_claims
from goldpipeline.services.style_symptoms import find_style_symptoms

logger = logging.getLogger(__name__)

ENFORCED_TYPES = frozenset({ArticleType.ANALYSIS})
"""Article types whose shape is enforced today.

One member. The others are declared but not runnable, and bringing them under a
contract they have never been written against would turn their activation round
into a debugging session.
"""

BLOCKING_CODES = frozenset(
    {
        OutputFindingCode.HARD_CAP_EXCEEDED,
        OutputFindingCode.MISSING_REQUIRED_SECTION,
        OutputFindingCode.FORBIDDEN_SECTION_PRESENT,
        OutputFindingCode.FORBIDDEN_STRUCTURE_PRESENT,
        OutputFindingCode.DISCLAIMER_COUNT_MISMATCH,
        OutputFindingCode.FORBIDDEN_TERMINOLOGY,
        OutputFindingCode.UNATTRIBUTED_CAUSAL_CLAIM,
    }
)
"""Findings a draft may not carry.

Deliberately excludes every style symptom. A piece with three soft connectives
is worse writing; a piece with two disclaimers or an invented causal link is a
different document from the one the contract describes.
"""

MAX_REPORTED_FINDINGS = 8
"""How many findings an error message names. Enough to fix, bounded for a manifest."""


@dataclass(frozen=True)
class AnalysisContractReport:
    """Everything the deterministic checks saw in one article."""

    article_type: ArticleType
    chars: int
    within_target: bool
    over_hard_cap: bool
    contract_findings: tuple[OutputFinding, ...] = ()
    causality_findings: tuple[OutputFinding, ...] = ()
    style_findings: tuple[OutputFinding, ...] = field(default=())

    @property
    def blocking(self) -> tuple[OutputFinding, ...]:
        """Findings that must stop the draft."""
        return tuple(
            finding
            for finding in (*self.contract_findings, *self.causality_findings)
            if finding.code in BLOCKING_CODES
        )

    @property
    def observed(self) -> tuple[OutputFinding, ...]:
        """Findings recorded but not enforced. Style lives here."""
        return tuple(
            finding
            for finding in (*self.contract_findings, *self.causality_findings, *self.style_findings)
            if finding.code not in BLOCKING_CODES
        )


def is_enforced(article_type: ArticleType) -> bool:
    """Whether this type's written shape is checked deterministically today."""
    if article_type not in ENFORCED_TYPES:
        return False
    return contract_for(article_type).generation_mode is GenerationMode.LLM


def inspect_article(article: str, article_type: ArticleType) -> AnalysisContractReport:
    """Run every deterministic check, enforcing nothing.

    Separated from the decision so a caller - a test, a diagnostic, a future
    reviewer - can see what the checks found without a refusal being the only
    way to learn it.
    """
    contract: ArticleContract = contract_for(article_type)
    report = length_report(article, contract)
    return AnalysisContractReport(
        article_type=article_type,
        chars=report.chars,
        within_target=report.within_target,
        over_hard_cap=report.over_hard_cap,
        contract_findings=tuple(check_contract(article, contract)),
        causality_findings=tuple(find_causal_claims(article)),
        style_findings=tuple(find_style_symptoms(article, contract=contract)),
    )


def describe(findings: tuple[OutputFinding, ...]) -> str:
    """A short, safe rendering of findings for an error message.

    Codes and messages only. Excerpts carry article text, which belongs in the
    artifact rather than duplicated into an exception that travels into a
    manifest and a log line.
    """
    shown = findings[:MAX_REPORTED_FINDINGS]
    rendered = "; ".join(f"{finding.code}: {finding.message}" for finding in shown)
    if len(findings) > len(shown):
        rendered += f"; and {len(findings) - len(shown)} more"
    return rendered


def missing_article_date(article: str, expected: str) -> bool:
    """Whether the article failed to carry the date it was handed.

    The date is computed in Python and given to the model as data; this confirms
    it arrived unchanged. A model that reformats ``04.09.2026`` into ``4/9/2026``
    has invented a rendering, and one that writes a different date has invented a
    day - both from a value it was told to copy.
    """
    return expected.strip() not in article


__all__ = [
    "BLOCKING_CODES",
    "ENFORCED_TYPES",
    "AnalysisContractReport",
    "describe",
    "inspect_article",
    "is_enforced",
    "missing_article_date",
]
