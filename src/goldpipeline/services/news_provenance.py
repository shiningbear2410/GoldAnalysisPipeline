"""Deciding, in code, whether an article's news statements are actually sourced.

The problem this exists for. The gate blocks an article that asserts a named
economic event happened, because until the producer existed the pipeline
collected no news and any such sentence was invented. The producer now supplies
real news - and the gate had no way to tell a faithfully relayed item from a
hallucination, so it blocked both. The fix is not to trust the article more; it
is to give the gate something checkable.

**Three independent conditions, all deterministic, all conservative.**

1. *Authenticity.* The Run's input must be a producer brief: submitted under the
   internal producer's source, and parsing completely as a brief this code
   understands. Either condition failing yields no provenance at all.
2. *The cited item must exist.* Ids are resolved against the parsed brief. An id
   that is not there - a typo, an invention, a URL, a channel with no message
   number - supports nothing. The model cannot mint authority.
3. *The words must line up.* The claimed evidence must appear in the cited
   item's text, and the claimed statement must appear in the **final** article.
   Both are substring checks after a fold that ignores case and diacritics.

**Why substrings and not meaning.** Judging whether an item "supports" a
paraphrase is exactly the job a language model would do, and a verifier that
consults a model is a verifier that can be argued with. A substring check cannot
be persuaded. It costs false negatives - a writer who rewords its evidence gets
told its claim is unsupported - and that is the trade this round chooses
deliberately: a rejected true statement costs an article, an accepted false one
publishes a fabricated fact under a real channel's name.

**Coverage is by span, never by article.** A verified claim suppresses the
external-fact finding only where its statement actually sits in the article, and
only when the whole assertion - the entity and the past-tense marker - lies
inside it. An article with one sourced Fed sentence and one invented one is
still blocked, and citing two real items does not license a third conclusion
drawn from neither.

Nothing in a news item can change any of this. The rules are here, in code; the
item supplies text that is either quoted or not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from goldpipeline.schemas.context import AnalysisContext
from goldpipeline.schemas.producer import PRODUCER_SOURCE
from goldpipeline.schemas.provenance import (
    NEWS_PROVENANCE_VERSION,
    ClaimVerdict,
    NewsClaimAudit,
    NewsProvenanceReport,
    ProvenanceState,
)
from goldpipeline.schemas.writer import NewsClaim
from goldpipeline.services.content_safety import fold
from goldpipeline.services.producer_brief import (
    ITEM_ID_PATTERN,
    ParsedBrief,
    ParsedNewsItem,
    parse_brief,
)

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class VerifiedClaim:
    """One claim, judged, with the spans it covers in the final article."""

    statement: str
    news_item_ids: tuple[str, ...]
    verdict: ClaimVerdict
    supporting_item_id: str | None = None
    spans: tuple[tuple[int, int], ...] = ()
    detail: str | None = None

    @property
    def supported(self) -> bool:
        return self.verdict is ClaimVerdict.SUPPORTED


@dataclass(frozen=True)
class NewsProvenance:
    """Everything the gate needs to judge external-news findings for one Run."""

    state: ProvenanceState
    version: str = NEWS_PROVENANCE_VERSION
    brief_version: str | None = None
    item_count: int = 0
    claims: tuple[VerifiedClaim, ...] = ()

    @property
    def available(self) -> bool:
        return self.state is ProvenanceState.AVAILABLE

    @property
    def covered_spans(self) -> tuple[tuple[int, int], ...]:
        """Article ranges a supported claim vouches for."""
        return tuple(span for claim in self.claims if claim.supported for span in claim.spans)

    def covers(self, start: int, end: int) -> bool:
        """Whether some supported statement wholly contains ``start:end``.

        Containment, not overlap. A claim that clips the first half of an
        assertion has not sourced it, and treating a partial overlap as coverage
        would let a writer quote three innocuous words next to a fabricated
        clause and pass the whole thing off as sourced.
        """
        return any(low <= start and end <= high for low, high in self.covered_spans)

    def report(self) -> NewsProvenanceReport:
        """The audit record for the publish decision.

        Spans become a count rather than offsets: an offset into an article is
        meaningful only against the exact bytes it was computed from, and a
        decision artifact that recorded them would invite someone to trust them
        against a different copy.
        """
        return NewsProvenanceReport(
            version=self.version,
            state=self.state,
            brief_version=self.brief_version,
            item_count=self.item_count,
            claims=[
                NewsClaimAudit(
                    statement=claim.statement,
                    news_item_ids=list(claim.news_item_ids),
                    verdict=claim.verdict,
                    supporting_item_id=claim.supporting_item_id,
                    article_spans=len(claim.spans),
                    detail=claim.detail,
                )
                for claim in self.claims
            ],
        )


# --------------------------------------------------------------------------
# authenticity
# --------------------------------------------------------------------------


def authenticate(context: AnalysisContext) -> tuple[ProvenanceState, ParsedBrief | None]:
    """Decide whether this Run's input is a producer brief worth reading.

    Two gates, and the order is the cheap one first. ``source`` is a field the
    submitting producer set, recorded in the immutable context - it says which
    producer this pipeline believes wrote the event. It is a *local inbox* trust
    boundary, not a signature: anything that can write into the inbox can set it,
    and anything that can write into the inbox can already do worse than claim a
    news item.

    The boundary that carries real weight is the second one. The threat this
    module is built against is scraped text, which cannot choose ``source`` and
    cannot forge a record inside a length-framed brief. So a document that does
    not parse exactly is refused rather than read leniently, and a Run that is
    not the producer's gets no provenance and behaves precisely as it did before
    any of this existed.
    """
    if context.raw_analysis.source != PRODUCER_SOURCE:
        return ProvenanceState.NOT_PRODUCER, None

    parsed = parse_brief(context.raw_analysis.text)
    if parsed is None:
        return ProvenanceState.UNPARSEABLE_BRIEF, None
    return ProvenanceState.AVAILABLE, parsed


def eligible_item_ids(context: AnalysisContext) -> tuple[str, ...]:
    """Ids a writer may cite for this Run, in brief order.

    Built here, from application code, for the same reason the ``VALID SOURCE
    PATHS`` catalog is: a model offered a closed list copies from it, and a model
    offered nothing invents. Empty whenever no authentic brief exists, which is
    the honest answer - there is nothing to cite.
    """
    _, parsed = authenticate(context)
    return tuple(item.item_id for item in parsed.items) if parsed else ()


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


def _squeeze(text: str) -> str:
    """Fold, then flatten whitespace, for comparisons that need no offsets.

    Used only where a span is not wanted. A writer quoting an item across a line
    break has quoted the item; insisting the newlines match too would reject a
    correct citation for a typographic reason.
    """
    return _WHITESPACE.sub(" ", fold(text)).strip()


def _spans_of(needle: str, haystack: str) -> tuple[tuple[int, int], ...]:
    """Every place ``needle`` occurs in ``haystack``, as offsets into the original.

    Both sides are folded with the character-preserving fold, so the offsets are
    valid against the unfolded article a person reads.
    """
    target, source = fold(needle).strip(), fold(haystack)
    if not target:
        return ()

    found: list[tuple[int, int]] = []
    start = source.find(target)
    while start >= 0:
        found.append((start, start + len(target)))
        start = source.find(target, start + 1)
    return tuple(found)


def verify(
    context: AnalysisContext,
    claims: list[NewsClaim],
    article: str,
) -> NewsProvenance:
    """Judge every declared news claim against the brief and the final article.

    ``article`` is the text about to be published, not the draft. Verification
    of a draft would answer a question nobody is asking by the time the gate
    runs.
    """
    state, parsed = authenticate(context)

    if parsed is None:
        return NewsProvenance(
            state=state,
            claims=tuple(
                VerifiedClaim(
                    statement=claim.statement,
                    news_item_ids=tuple(claim.news_item_ids),
                    verdict=ClaimVerdict.NO_PROVENANCE,
                    detail=(
                        "the Run's input is not an authentic producer brief"
                        if state is ProvenanceState.UNPARSEABLE_BRIEF
                        else "the Run was not fed by the internal producer"
                    ),
                )
                for claim in claims
            ),
        )

    by_id = parsed.by_id
    return NewsProvenance(
        state=state,
        brief_version=parsed.brief_version,
        item_count=len(parsed.items),
        claims=tuple(_verify_one(claim, by_id, article) for claim in claims),
    )


def _verify_one(
    claim: NewsClaim,
    by_id: dict[str, ParsedNewsItem],
    article: str,
) -> VerifiedClaim:
    """Judge one claim. Every failure names itself."""
    ids = tuple(claim.news_item_ids)

    def refused(verdict: ClaimVerdict, detail: str) -> VerifiedClaim:
        return VerifiedClaim(
            statement=claim.statement, news_item_ids=ids, verdict=verdict, detail=detail
        )

    malformed = sorted(i for i in ids if not ITEM_ID_PATTERN.match(i))
    if malformed:
        return refused(
            ClaimVerdict.MALFORMED_ID,
            f"not a '<channel>:<message_id>' id: {', '.join(malformed[:3])}",
        )

    unknown = sorted(i for i in ids if i not in by_id)
    if unknown:
        # Every id must resolve, not merely one of them. A claim that cites a
        # real item beside an invented one is a claim whose author did not know
        # which item supported it.
        return refused(
            ClaimVerdict.UNKNOWN_ITEM,
            f"no such item in this brief: {', '.join(unknown[:3])}",
        )

    evidence = _squeeze(claim.evidence)
    supporting = next((i for i in ids if evidence and evidence in _squeeze(by_id[i].text)), None)
    if supporting is None:
        return refused(
            ClaimVerdict.EVIDENCE_NOT_IN_ITEM,
            "the quoted evidence appears in none of the cited items",
        )

    spans = _spans_of(claim.statement, article)
    if not spans:
        return refused(
            ClaimVerdict.STATEMENT_NOT_IN_ARTICLE,
            "the quoted statement is not in the final article; it may have been rewritten",
        )

    return VerifiedClaim(
        statement=claim.statement,
        news_item_ids=ids,
        verdict=ClaimVerdict.SUPPORTED,
        supporting_item_id=supporting,
        spans=spans,
    )


__all__ = [
    "NewsProvenance",
    "VerifiedClaim",
    "authenticate",
    "eligible_item_ids",
    "verify",
]
