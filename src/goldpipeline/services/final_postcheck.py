"""What the pipeline demands of a *finished* article, after the one revision.

Round 6.4g. Until now the finalizer's own postcheck compared deterministic
findings before and after and refused a regression. That catches a model that
invents an RSI reading while removing a wrong price. It does not catch a model
that quietly rewrites the whole piece, drops the disclaimer, changes the date in
the title, or replaces a supported figure with a plausible one - because none of
those are *new precheck findings*, they are contract failures the finalizer path
never looked for.

This module is the seam that looks. It runs once, after the single model call
and before the article is accepted, and it decides only one thing: keep or stop.

**There is no repair path here.** Every failure below is terminal for the Run.
The round's hard invariant is one automatic model call, so "ask it again with
the failure attached" is not an option that exists - and quietly accepting a
broken article would be worse than stopping. A Run that stops here waits for a
person, with the reasons on the artifact ledger.

**It never judges prose.** Deterministic code cannot decide whether an article
now sounds human, and pretending otherwise would put a machine in charge of the
one judgement this pipeline deliberately gives to a reviewer. Style symptoms are
recounted for observability and recorded; they are not a verdict, and one more
LOW symptom is not a failure.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from goldpipeline.schemas.article import ArticleType
from goldpipeline.schemas.article_contract import SectionKey, contract_for
from goldpipeline.schemas.context import AnalysisContext
from goldpipeline.schemas.output_findings import OutputFinding
from goldpipeline.schemas.review import HumanStyleFinding
from goldpipeline.schemas.writer import WriterResult
from goldpipeline.services import analysis_contract
from goldpipeline.services.article_contract_checks import detect_sections
from goldpipeline.services.claim_resolver import verify_claims
from goldpipeline.services.numeric_mentions import (
    AuthorisedFact,
    FactProvenance,
    NumericResolution,
    ResolutionStatus,
    extract_numeric_mentions,
    resolve_mention,
)
from goldpipeline.services.numeric_semantics import SemanticType
from goldpipeline.services.precheck import known_numbers
from goldpipeline.services.style_symptoms import find_style_symptoms

logger = logging.getLogger(__name__)

UNRESOLVED_BLOCKING_SEMANTICS = frozenset(
    {
        SemanticType.ABSOLUTE_PRICE,
        SemanticType.UNKNOWN_PRICE_LIKE,
    }
)
"""Mention types whose non-resolution stops the Run.

A number the article presents as a price, or one nothing has vouched for that
reads like a price, is a factual assertion. If no authorised fact carries that
value, the article states a price that does not exist.

Deliberately *not* every semantic type. A count, a percentage or a mass may
legitimately come from a news item whose provenance lives outside the market
context, and blocking every unresolved one of those here would re-litigate news
provenance in the wrong module and reject correct articles. Those are covered by
their own machinery; see :func:`_news_facts`.
"""


@dataclass
class FinalPostcheckReport:
    """Everything the check established about the finished article.

    Note the shape of the failure fields: each one is what the revision
    **introduced**, not simply what the final article contains. The finalizer is
    answerable for what it changed, and blaming it for a defect it was handed
    would both be unjust and, worse, useless - a Run whose draft was already
    malformed would fail here every time with no edit able to clear it.

    In production the distinction is invisible, because the draft is already
    contract-valid: Round 6.4e enforces the output contract at the writer, and
    the review and prechecks have both passed over the numbers by the time this
    runs. There "introduced" and "present" are the same set, and this check is
    the absolute invariant the round asked for. Outside production - a fixture,
    a Run drafted before the contract existed - it degrades to the honest claim
    rather than a false one.
    """

    contract: analysis_contract.AnalysisContractReport | None = None
    """The contract report on the finished article. ``None`` when unenforced."""

    contract_regressions: tuple[OutputFinding, ...] = ()
    """Blocking contract findings the draft did not have."""

    missing_date: bool = False
    """The draft carried the authoritative date and the revision lost it."""

    numeric: list[NumericResolution] = field(default_factory=list)
    unsupported_numbers: list[NumericResolution] = field(default_factory=list)
    """Price-like numbers the revision introduced that nothing vouches for."""

    mistyped_numbers: list[NumericResolution] = field(default_factory=list)
    """Numbers the revision introduced that a fact carries under another meaning."""

    unexpected_sections: list[SectionKey] = field(default_factory=list)
    changed_sections: list[SectionKey] = field(default_factory=list)
    symptoms_before: int = 0
    symptoms_after: int = 0

    @property
    def contract_blocking(self) -> tuple[OutputFinding, ...]:
        """Every blocking contract finding on the final article, regression or not."""
        return () if self.contract is None else self.contract.blocking

    @property
    def ok(self) -> bool:
        """Whether the finished article may be kept."""
        return not (
            self.contract_regressions
            or self.missing_date
            or self.unsupported_numbers
            or self.mistyped_numbers
            or self.unexpected_sections
        )

    @property
    def symptoms_worse_by(self) -> int:
        """How many more countable symptoms the revision has. Never a failure."""
        return max(0, self.symptoms_after - self.symptoms_before)


def authorised_facts(context: AnalysisContext, writer_result: WriterResult) -> list[AuthorisedFact]:
    """Every value the finished article is entitled to state, typed.

    Built on :func:`~goldpipeline.services.precheck.known_numbers` rather than
    beside it. That function already knows the four sources - candle values,
    verified source claims, the derived-formula catalog, analyst levels near the
    market - and already types each one. A second vocabulary here would be a
    second answer to "what is this number", and the two would disagree the first
    time either changed.

    What this adds is provenance: which provider vouched for the value. Nothing
    reads it yet to make a decision, and that is correct - the semantic type is
    the meaning and the provenance is only the audit trail. It exists so a
    market-data migration can be traced through an artifact rather than guessed.
    """
    bars = context.ohlc.bars
    low = min(bar.low for bar in bars)
    high = max(bar.high for bar in bars)
    resolved = verify_claims(context, list(writer_result.source_claims))

    market = FactProvenance(
        source=str(context.market.provider),
        symbol=context.market.symbol,
        timeframe=str(context.market.timeframe),
    )

    facts = [
        AuthorisedFact(
            value=known.value,
            semantic=known.semantic,
            origin=known.origin,
            provenance=market,
        )
        for known in known_numbers(context, resolved, low, high)
    ]
    facts.extend(_news_facts(writer_result))
    return facts


def _news_facts(writer_result: WriterResult) -> list[AuthorisedFact]:
    """Numbers a curated news item vouches for.

    Round 5.1's provenance rule is the authority here and is not re-derived: a
    ``news_claim`` was already checked to be a substring of the article and
    backed by a cited item, so any number inside the quoted statement is
    evidence-backed text rather than a market figure.

    They enter as :attr:`SemanticType.NON_MARKET_NUMBER`, which is the honest
    type: the pipeline knows a source vouched for the value and does not know
    whether it is tonnes, a percentage or a dollar amount. Under
    :func:`~goldpipeline.services.numeric_mentions.compatible` a coarse fact
    like this vouches for an untyped mention and never for one the article
    typed as a price - so a news figure can never launder itself into a gold
    quote, which is exactly the confusion Round 6.4's design warned about.
    """
    facts: list[AuthorisedFact] = []
    provenance = FactProvenance(source="news")
    for claim in writer_result.news_claims:
        for mention in extract_numeric_mentions(claim.statement):
            facts.append(
                AuthorisedFact(
                    value=mention.value,
                    semantic=SemanticType.NON_MARKET_NUMBER,
                    origin=f"news_claim:{claim.statement[:40]}",
                    provenance=provenance,
                )
            )
    return facts


def check_final_article(
    *,
    article: str,
    draft: str,
    context: AnalysisContext,
    writer_result: WriterResult,
    article_type: ArticleType,
    expected_date: str | None = None,
    style_findings: tuple[HumanStyleFinding, ...] = (),
) -> FinalPostcheckReport:
    """Everything deterministic that can be said about the finished article.

    Args:
        article: The revision, as it would be published.
        draft: What the writer produced, for comparison.
        context: The Run's source of truth.
        writer_result: The writer's own record, for its claims.
        article_type: What the Run is writing.
        expected_date: The authoritative article date, if one applies.
        style_findings: What the revision was asked to repair, which decides
            whether section preservation can be enforced.

    Returns:
        A report. Nothing raises here - the caller decides, and the caller is
        the one place that knows a failure is terminal.
    """
    report = FinalPostcheckReport()

    if analysis_contract.is_enforced(article_type):
        report.contract = analysis_contract.inspect_article(article, article_type)
        before = {f.code for f in analysis_contract.inspect_article(draft, article_type).blocking}
        report.contract_regressions = tuple(
            f for f in report.contract.blocking if f.code not in before
        )
        if expected_date is not None:
            draft_had_it = not analysis_contract.missing_article_date(draft, expected_date)
            report.missing_date = draft_had_it and analysis_contract.missing_article_date(
                article, expected_date
            )

    facts = authorised_facts(context, writer_result)
    report.numeric = _resolve_numbers(article, facts)
    unchanged = {(r.mention.literal, r.status) for r in _resolve_numbers(draft, facts)}
    report.unsupported_numbers = [
        r
        for r in report.numeric
        if r.status is ResolutionStatus.UNRESOLVED
        and r.mention.semantic in UNRESOLVED_BLOCKING_SEMANTICS
        and (r.mention.literal, r.status) not in unchanged
    ]
    report.mistyped_numbers = [
        r
        for r in report.numeric
        if r.status is ResolutionStatus.TYPE_MISMATCH
        and (r.mention.literal, r.status) not in unchanged
    ]

    report.changed_sections, report.unexpected_sections = _section_changes(
        draft=draft, article=article, article_type=article_type, style_findings=style_findings
    )

    contract = contract_for(article_type)
    report.symptoms_before = len(find_style_symptoms(draft, contract=contract))
    report.symptoms_after = len(find_style_symptoms(article, contract=contract))

    return report


def _resolve_numbers(text: str, facts: Sequence[AuthorisedFact]) -> list[NumericResolution]:
    """Every numeric mention in *text*, against the authorised facts."""
    return [resolve_mention(mention, facts) for mention in extract_numeric_mentions(text)]


# --------------------------------------------------------------------------
# section preservation
# --------------------------------------------------------------------------


GLOBAL_CATEGORIES = frozenset({"AI_VOICE", "NEWS_DESK_VOICE", "NO_POSITION", "REPETITIVE_RHYTHM"})
"""Style categories whose repair legitimately touches the whole article.

A piece that reads as a research note does not read that way in one paragraph,
and demanding that its repair leave five of six sections byte-identical would
be a rule that only a refusal can satisfy. Held as category *names* rather than
enum members so that a category added later does not silently inherit
whole-article licence by being absent from a set nobody updated - an unknown
name is treated as scoped, which is the stricter reading.
"""


def _section_bodies(article: str, article_type: ArticleType) -> dict[SectionKey, str] | None:
    """The article sliced at its section boundaries, or ``None`` if it cannot be.

    Returns ``None`` rather than guessing when the parser does not find the
    contract's required sections. A preservation rule built on a mis-segmented
    article would reject correct revisions, and a brittle protection is worse
    than an absent one because people switch it off.
    """
    contract = contract_for(article_type)
    if not contract.required_sections:
        return None

    found = detect_sections(article)
    if not set(contract.required_sections) <= set(found):
        return None

    ordered = sorted(found.items(), key=lambda item: item[1])
    bodies: dict[SectionKey, str] = {}
    for index, (key, start) in enumerate(ordered):
        end = ordered[index + 1][1] if index + 1 < len(ordered) else len(article)
        bodies[key] = article[start:end].strip()
    return bodies


def _section_changes(
    *,
    draft: str,
    article: str,
    article_type: ArticleType,
    style_findings: tuple[HumanStyleFinding, ...],
) -> tuple[list[SectionKey], list[SectionKey]]:
    """Which sections changed, and which of those had no licence to.

    The second list is the "smoother is a regression" protection with teeth. A
    revision asked to trim ``PRICE_READ`` has no business rewriting the verdict,
    and a model that does it anyway is polishing rather than repairing - which
    is the single most likely way this stage damages a good article.

    Licence comes from three places, and if none applies the section must be
    byte-identical:

    * a finding scoped to that section;
    * any finding whose category is inherently whole-article;
    * the absence of any style findings at all, which means this was a content
      revision and content corrections may land anywhere.
    """
    before = _section_bodies(draft, article_type)
    after = _section_bodies(article, article_type)
    if before is None or after is None:
        # Not segmentable: report nothing rather than assert something false.
        return [], []

    changed = [key for key, text in after.items() if before.get(key) != text]
    changed.extend(key for key in before if key not in after)

    if not style_findings:
        return sorted(set(changed), key=lambda k: k.value), []

    if any(str(f.category) in GLOBAL_CATEGORIES for f in style_findings):
        return sorted(set(changed), key=lambda k: k.value), []

    licensed = {f.section for f in style_findings if f.section is not None}
    unexpected = [key for key in set(changed) if key not in licensed]
    return (
        sorted(set(changed), key=lambda k: k.value),
        sorted(unexpected, key=lambda k: k.value),
    )


def describe(report: FinalPostcheckReport) -> dict[str, object]:
    """The failure, as structured detail for an error. Never article text."""
    detail: dict[str, object] = {}
    if report.contract_regressions:
        detail["contract_broken_by_the_revision"] = [
            {"code": str(f.code), "severity": str(f.severity)} for f in report.contract_regressions
        ]
    if report.missing_date:
        detail["article_date"] = "missing or not the authoritative date"
    if report.unsupported_numbers:
        detail["unsupported_numbers"] = [
            {"literal": r.mention.literal, "semantic": str(r.mention.semantic)}
            for r in report.unsupported_numbers
        ]
    if report.mistyped_numbers:
        detail["mistyped_numbers"] = [
            {
                "literal": r.mention.literal,
                "stated_as": str(r.mention.semantic),
                "fact_is": str(r.fact.semantic) if r.fact else None,
            }
            for r in report.mistyped_numbers
        ]
    if report.unexpected_sections:
        detail["rewritten_without_a_finding"] = [str(k) for k in report.unexpected_sections]
    return detail


__all__ = [
    "GLOBAL_CATEGORIES",
    "UNRESOLVED_BLOCKING_SEMANTICS",
    "FinalPostcheckReport",
    "authorised_facts",
    "check_final_article",
    "describe",
]
