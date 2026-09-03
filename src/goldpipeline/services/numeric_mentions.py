"""Every number an article mentions, with what its own words say it is.

The production scanner asks "is this literal a price the context supports?"
and, to keep the question answerable, ignores anything below
:data:`~goldpipeline.services.source_guard.MIN_PLAUSIBLE_QUOTE`. That floor is
why ``9.98 tấn`` and ``0.21%`` have never been mistaken for gold - not because
anything read the unit, but because they are small. The first news figure over
a hundred (``141 triệu USD``) will not be small, and the scanner will call it
an unexplained price.

This module reads the unit. A mention is classified from the words around it -
``%``, ``tấn``, ``triệu USD``, a clock time - into the refined
:class:`~goldpipeline.services.numeric_semantics.SemanticType` vocabulary,
before anyone asks whether a market fact vouches for it. A bare number with no
unit stays unknown; classification is never inferred from the value.

**The resolution seam.** :func:`resolve_mention` is the future rule, stated as
code: every numeric claim in a final article must resolve to an authorised
fact *of a compatible type*. A ``9.98`` that the article calls tonnes is not
resolved by a ``9.98`` the market data calls a price. This is what lets a
later round remove or correct a wrong number instead of preserving it - the
draft's literal is nothing; the fact is the authority.

**Provenance is orthogonal.** An :class:`AuthorisedFact` may say which provider
produced it. Its semantic type does not depend on the answer, and a test holds
that a fact from ``mt5`` and the same fact from ``tradingview`` resolve a
mention identically.

**Not wired.** No production stage calls this module.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from goldpipeline.services.content_safety import fold
from goldpipeline.services.numeric_semantics import KnownNumber, SemanticType, rendered_as
from goldpipeline.services.source_guard import MIN_PLAUSIBLE_QUOTE

_DATE_TIME_RE = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?\b"  # 14:07, 09:25:01
    r"|\b\d{4}-\d{2}-\d{2}\b"  # 2026-09-03
    r"|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b"  # 03/09, 03/09/2026
    r"|\b\d{1,2}\.\d{1,2}\.\d{2,4}\b"  # 03.09.2026 - the dotted form needs a year,
    # or 9.98 tonnes would be a date
    r"|\b(?:19|20)\d{2}\b"  # a bare year
)
_NUMBER_RE = re.compile(
    r"(?<![\w.,])[+-]?(?P<int>\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,](?P<frac>\d{1,4}))?(?![\d])"
)
_AFTER_WINDOW = 18
_BEFORE_WINDOW = 4

_PERCENT_RE = re.compile(r"^\s*(?:%|phan tram)")
_TONNES_RE = re.compile(r"^\s*tan\b")
_QUANTITY_UNIT_RE = re.compile(r"^\s*(?:ounce|oz|kg|lot|lots)\b")
_SCALE_RE = re.compile(r"^\s*(?:trieu|ty|nghin|ngan|million|billion|thousand|bn|mn)\b")
_CURRENCY_RE = re.compile(r"^\s*(?:usd|\$|do la|dong|vnd|eur|jpy|gbp)\b|^\s*\$")
_SCALE_THEN_CURRENCY_RE = re.compile(
    r"^\s*(?:trieu|ty|nghin|ngan|million|billion|thousand|bn|mn)\s*(?:usd|\$|do la|dong|vnd|eur)"
)
_DISTANCE_RE = re.compile(r"^\s*(?:diem|pip|pips|point|points)\b")
_COUNT_RE = re.compile(
    r"^\s*(?:gio|phut|giay|ngay|tuan|thang|nam|nen|phien|lan|kich ban|bar|candle|bars|candles)\b"
)


@dataclass(frozen=True)
class NumericMention:
    """One number in the article, and what its surroundings say it is."""

    literal: str
    value: Decimal
    start: int
    end: int
    semantic: SemanticType
    unit: str
    """The folded unit text that decided the type, or ``""`` for a bare number."""


def extract_numeric_mentions(text: str) -> list[NumericMention]:
    """Every number in *text*, classified by its unit. Never by its size alone."""
    folded = fold(text)
    mentions: list[NumericMention] = []
    covered: list[tuple[int, int]] = []

    for match in _DATE_TIME_RE.finditer(text):
        covered.append((match.start(), match.end()))
        mentions.append(
            NumericMention(
                literal=match.group(0),
                value=Decimal(0),
                start=match.start(),
                end=match.end(),
                semantic=SemanticType.DATE_TIME,
                unit="date/time",
            )
        )

    for match in _NUMBER_RE.finditer(text):
        if any(start <= match.start() < end for start, end in covered):
            continue
        value = _parse(match.group("int"), match.group("frac"))
        if value is None:
            continue
        semantic, unit = _classify(folded, match.start(), match.end(), value)
        mentions.append(
            NumericMention(
                literal=match.group(0),
                value=value,
                start=match.start(),
                end=match.end(),
                semantic=semantic,
                unit=unit,
            )
        )

    mentions.sort(key=lambda m: m.start)
    return mentions


def _parse(integer: str, fraction: str | None) -> Decimal | None:
    digits = integer.replace(".", "").replace(",", "")
    text = f"{digits}.{fraction}" if fraction else digits
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _classify(folded: str, start: int, end: int, value: Decimal) -> tuple[SemanticType, str]:
    after = folded[end : end + _AFTER_WINDOW]
    before = folded[max(0, start - _BEFORE_WINDOW) : start]

    rules: tuple[tuple[re.Pattern[str], SemanticType], ...] = (
        (_PERCENT_RE, SemanticType.PERCENT_CHANGE),
        (_TONNES_RE, SemanticType.MASS_TONNES),
        (_SCALE_THEN_CURRENCY_RE, SemanticType.MONETARY_NON_PRICE),
        (_SCALE_RE, SemanticType.QUANTITY),
        (_QUANTITY_UNIT_RE, SemanticType.QUANTITY),
        (_DISTANCE_RE, SemanticType.MAGNITUDE),
        (_COUNT_RE, SemanticType.COUNT),
    )
    for pattern, semantic in rules:
        match = pattern.match(after)
        if match is not None:
            return semantic, match.group(0).strip()

    currency = _CURRENCY_RE.match(after)
    if currency is not None or before.rstrip().endswith("$"):
        # Money on the instrument's own scale. Still nothing has vouched for
        # it as a market price - that takes a fact, not a unit.
        unit = currency.group(0).strip() if currency is not None else "$"
        return SemanticType.UNKNOWN_PRICE_LIKE, unit

    if value >= MIN_PLAUSIBLE_QUOTE:
        return SemanticType.UNKNOWN_PRICE_LIKE, ""
    return SemanticType.UNKNOWN, ""


# --------------------------------------------------------------------------
# resolution seam
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FactProvenance:
    """Where an authorised value came from. Orthogonal to what it means."""

    source: str
    """A provider name: ``mt5``, ``tradingview``, ``news``, ``derived``."""

    symbol: str | None = None
    timeframe: str | None = None


@dataclass(frozen=True)
class AuthorisedFact:
    """A value the article is entitled to state, typed, with its origin.

    The same shape as :class:`KnownNumber` plus provenance, which the scanner
    never needed and a market-data migration will. ``semantic`` is the whole
    meaning; ``provenance`` is only the audit trail.
    """

    value: Decimal
    semantic: SemanticType
    origin: str
    provenance: FactProvenance | None = None

    def as_known(self) -> KnownNumber:
        """The scanner's view of this fact, provenance dropped."""
        return KnownNumber(value=self.value, semantic=self.semantic, origin=self.origin)


class ResolutionStatus(StrEnum):
    """What happened when a mention was checked against the authorised facts."""

    RESOLVED = "RESOLVED"
    """A fact of compatible type has this value. The claim may stand."""

    TYPE_MISMATCH = "TYPE_MISMATCH"
    """A fact has this value, but the article calls it something it is not."""

    UNRESOLVED = "UNRESOLVED"
    """No fact has this value. Remove the claim or replace it with the fact."""

    NOT_A_FACT_CLAIM = "NOT_A_FACT_CLAIM"
    """Dates and times are not numeric claims about the market."""


@dataclass(frozen=True)
class NumericResolution:
    """One mention, and the fact that vouches for it, if any."""

    mention: NumericMention
    status: ResolutionStatus
    fact: AuthorisedFact | None = None


def compatible(mention: SemanticType, fact: SemanticType) -> bool:
    """Whether a fact of type *fact* may vouch for a mention read as *mention*.

    An unknown mention asserted nothing, so any fact may explain it - the
    fact's type then becomes the claim's meaning. A mention typed coarsely
    (``MAGNITUDE``, from a unit like ``điểm``) accepts any fact in that family.
    A mention typed finely accepts only that exact meaning: a net change is
    not explained by a range, and ``MASS_TONNES`` is never explained by a
    price.
    """
    if mention in (SemanticType.UNKNOWN, SemanticType.UNKNOWN_PRICE_LIKE):
        return True
    if mention.family is mention:
        return fact.family is mention
    return fact is mention


def resolve_mention(mention: NumericMention, facts: Iterable[AuthorisedFact]) -> NumericResolution:
    """Find the authorised fact that vouches for *mention*.

    Value matching is by faithful rendering - the same rule the scanner uses -
    so ``66.14`` resolves to a fact of ``66.140`` and not to ``66.15``.
    """
    if mention.semantic is SemanticType.DATE_TIME:
        return NumericResolution(mention, ResolutionStatus.NOT_A_FACT_CLAIM)

    mismatched: AuthorisedFact | None = None
    for fact in facts:
        if fact.value != mention.value and not rendered_as(fact.value, mention.literal):
            continue
        if compatible(mention.semantic, fact.semantic):
            return NumericResolution(mention, ResolutionStatus.RESOLVED, fact)
        mismatched = mismatched or fact
    if mismatched is not None:
        return NumericResolution(mention, ResolutionStatus.TYPE_MISMATCH, mismatched)
    return NumericResolution(mention, ResolutionStatus.UNRESOLVED)


__all__ = [
    "AuthorisedFact",
    "FactProvenance",
    "NumericMention",
    "NumericResolution",
    "ResolutionStatus",
    "compatible",
    "extract_numeric_mentions",
    "resolve_mention",
]
