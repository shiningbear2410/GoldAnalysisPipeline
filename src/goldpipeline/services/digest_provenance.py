"""Keeping the balance paragraph inside the evidence, like everything else.

Round 6.5c.1, and it exists because of a hole the audit found.

News provenance checks two things independently: the quoted ``evidence`` must
appear in a cited item, and the quoted ``statement`` must appear in the final
article. Neither check asks whether the statement *follows from* the evidence -
that is a judgement about meaning, and this layer deliberately makes none.

For a bullet that is survivable: the headline compresses one item, and the
reviewer reads both. For 🧭 Cán cân it is not. The balance is free prose
synthesising several items, and a model summarising "mua ròng 9.98 tấn" as "mua
ròng gần 10 tấn" produces a statement that is in the article, citing evidence
that is in the item, and a quantity that is in neither. The live Round 6.5b
digest did exactly that, and every existing check passed it.

**The fix removes the temptation rather than loosening the matching.** A
quantity in the balance must be one the digest already holds - stated by a
collected item, or computed by the deterministic shell. Rounding, approximating
and re-scaling are all refused, not because ``gần 10`` is far from ``9.98``, but
because deciding how near is near enough is exactly the judgement this layer
must never start making.

**Quantities are compared as values, not as text.** A substring search for
``10`` finds it inside ``2010``, which is how a check like this quietly stops
checking. The extractor already parses each literal and classifies it by its
unit, so the comparison runs on what the numbers *are*.

Signs are dropped on both sides. Vietnamese prose carries direction in words -
"giảm 50.24" against a computed ``-50.24`` - and the sign of the price move is
the price block's business, which is copied verbatim and postchecked. What is
being asked here is narrower: does this quantity exist in the evidence at all.

The balance keeps every synthesis it is for. "Tin nghiêng tích cực vì USD yếu và
ETF mua ròng" needs no number at all, and that sentence is the product.

**Who owns what, Round 6.5c.1a.** Three layers, and the split is deliberate
because only the first two can be decided without judging meaning:

1. *Claim locality* - :func:`verify_claims`. A declared claim's evidence must
   occur in an item it cites, and its statement must occur in the article. Two
   substring tests against two different texts. This is the same rule
   :mod:`~goldpipeline.services.news_provenance` applies to an analysis, and it
   is applied here separately because that module authenticates a *producer
   brief* a digest Run does not have - its collected items are the authority
   directly.
2. *Numeric equivalence* - :func:`unsupported_balance_numbers`. Typed, exact,
   sign-insensitive.
3. *Everything else* - the Reviewer. A balance clause asserting something no
   item says, or supplying a motive ("ETF mua **vì lo lạm phát**") the evidence
   does not carry, is an entailment question. Nothing here pretends to answer
   it: a substring machine cannot, and an approximate one would be worse than
   none, because it would licence the failures it could not see.

The gap between 1 and 3 is real and is not papered over. Locality checks the
claims a writer *declared*; it cannot know that an undeclared sentence needed
one. Detecting which clauses are factual is exactly the judgement layer 3 owns,
and the reviewer is handed the items and the article to make it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal

from goldpipeline.schemas.digest import DigestWindow
from goldpipeline.schemas.news_digest import DigestSourceItem
from goldpipeline.schemas.provenance import ClaimVerdict
from goldpipeline.schemas.writer import NewsClaim
from goldpipeline.services.news_provenance import squeeze
from goldpipeline.services.numeric_mentions import extract_numeric_mentions
from goldpipeline.services.numeric_semantics import SemanticType
from goldpipeline.services.producer_brief import ITEM_ID_PATTERN

logger = logging.getLogger(__name__)

SECONDS_PER_HOUR = 3600

EXEMPT_SEMANTICS = frozenset({SemanticType.DATE_TIME})
"""Mentions that assert no quantity, and so need nothing to vouch for them.

A clock time or a calendar date says *when*, not *how much*. The window line
already fixes when the digest is talking about, deterministically and in the
article, so a date in the balance is at worst redundant. Every other kind of
number claims a magnitude, and a claimed magnitude needs a source.

Exempt on both sides: a date inside an item authorises nothing either, or a
bulletin mentioning "2026" would license a bare ``2026`` anywhere in the prose.
"""


def _quantities(text: str) -> set[Decimal]:
    """The magnitudes *text* states, ignoring sign and formatting."""
    return {
        abs(mention.value)
        for mention in extract_numeric_mentions(text)
        if mention.semantic not in EXEMPT_SEMANTICS
    }


def authorised_quantities(
    sources: Sequence[DigestSourceItem],
    window: DigestWindow,
    deterministic_lines: Iterable[str] = (),
) -> set[Decimal]:
    """Every magnitude the digest can already prove, from all three holders.

    Args:
        sources: The closed list of items the model was offered - all of them,
            not only the ones it gave a bullet to. A synthesis may rest on an
            item it chose not to headline.
        window: The span being described. Its length in hours is authorised, so
            "trong 6 giờ qua" is a restatement of a fact the Run owns rather
            than an unsourced number.
        deterministic_lines: The title, window line and price block as
            published. Anything the pipeline itself computed and printed is by
            construction available to restate - and a *wrong* restatement still
            fails, because the value has to match.
    """
    authorised: set[Decimal] = set()
    for item in sources:
        authorised |= _quantities(item.text)
    for line in deterministic_lines:
        authorised |= _quantities(line)

    hours = Decimal(window.lookback_seconds) / SECONDS_PER_HOUR
    authorised.add(abs(hours.normalize()))
    return authorised


def unsupported_balance_numbers(
    balance: str,
    sources: Sequence[DigestSourceItem],
    window: DigestWindow,
    deterministic_lines: Iterable[str] = (),
) -> list[str]:
    """Literals in *balance* stating a magnitude nothing in the digest holds.

    Returns:
        The offending literals as written, in the order they appear. Empty means
        every quantity in the balance traces to an item or to the pipeline's own
        arithmetic.
    """
    if not balance.strip():
        return []

    authorised = authorised_quantities(sources, window, deterministic_lines)
    offending = [
        mention.literal
        for mention in extract_numeric_mentions(balance)
        if mention.semantic not in EXEMPT_SEMANTICS and abs(mention.value) not in authorised
    ]

    if offending:
        logger.info("digest.balance unsupported_quantities=%s", offending)
    return offending


# --------------------------------------------------------------------------
# claim locality
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckedClaim:
    """One declared claim, judged against the items and the article."""

    statement: str
    news_item_ids: tuple[str, ...]
    verdict: ClaimVerdict
    detail: str = ""
    supporting_item_id: str | None = None

    @property
    def supported(self) -> bool:
        return self.verdict is ClaimVerdict.SUPPORTED


def verify_claims(
    claims: Sequence[NewsClaim],
    sources: Sequence[DigestSourceItem],
    article: str | None = None,
) -> tuple[CheckedClaim, ...]:
    """Judge each declared claim by locality: evidence in an item, statement in the article.

    The digest's collected items are the authority directly, so there is no
    ``NO_PROVENANCE`` outcome here. That verdict exists for an analysis, whose
    news arrives inside a producer brief that must first be proved authentic; a
    digest is handed its items as typed values by the pipeline that collected
    them, and a brief it never had cannot fail to authenticate.

    Args:
        claims: What the writer declared. An empty list is not a failure - it
            means nothing was claimed, which is a different question and one
            this function does not answer.
        sources: The closed list of collected items.
        article: The assembled digest, when it exists. Before assembly the
            statement half is simply not run, rather than failed: a statement
            cannot be missing from an article nobody has built yet.

    Returns:
        One :class:`CheckedClaim` per input claim, in order.
    """
    by_id = {item.item_id: item for item in sources}
    return tuple(_check_one(claim, by_id, article) for claim in claims)


def _check_one(
    claim: NewsClaim,
    by_id: dict[str, DigestSourceItem],
    article: str | None,
) -> CheckedClaim:
    """Judge one claim. Every failure names itself."""
    ids = tuple(claim.news_item_ids)

    def refused(verdict: ClaimVerdict, detail: str) -> CheckedClaim:
        return CheckedClaim(
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
        # Every id must resolve, not merely one of them: a claim citing a real
        # item beside an invented one is a claim whose author did not know which
        # item supported it.
        return refused(
            ClaimVerdict.UNKNOWN_ITEM,
            f"no such collected item: {', '.join(unknown[:3])}",
        )

    evidence = squeeze(claim.evidence)
    supporting = next((i for i in ids if evidence and evidence in squeeze(by_id[i].text)), None)
    if supporting is None:
        return refused(
            ClaimVerdict.EVIDENCE_NOT_IN_ITEM,
            "the quoted evidence appears in none of the cited items",
        )

    if article is not None and squeeze(claim.statement) not in squeeze(article):
        return refused(
            ClaimVerdict.STATEMENT_NOT_IN_ARTICLE,
            "the quoted statement is not in the assembled digest",
        )

    return CheckedClaim(
        statement=claim.statement,
        news_item_ids=ids,
        verdict=ClaimVerdict.SUPPORTED,
        supporting_item_id=supporting,
    )


__all__ = [
    "EXEMPT_SEMANTICS",
    "SECONDS_PER_HOUR",
    "CheckedClaim",
    "authorised_quantities",
    "unsupported_balance_numbers",
    "verify_claims",
]
