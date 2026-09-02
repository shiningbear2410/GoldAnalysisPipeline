"""Deciding whether a news item could move gold, deterministically.

No model is consulted. The same text scores the same number today and in a
year, which is what makes a threshold meaningful and a regression visible.

**A taxonomy, not a keyword list.** Each category owns its terms and one weight,
so a term can be added, moved or re-weighted in one place, and each category can
be tested on its own. A single flat list of two hundred words is untestable in
practice and therefore, eventually, unmaintained.

**Categories score once.** An item's score is the sum of the weights of the
categories it matched, not of the terms. A post repeating "gold gold gold" is
about gold exactly as much as a post that says it once, and counting terms would
let a headline generator outrank a central bank.

**Matching is on words, not substrings.** ``fed`` must not fire inside
"federal", "feeding" or "confederate", and it does not: text is folded to
diacritic-free lowercase and terms match at word boundaries. Vietnamese survives
that folding, which is why it is done rather than merely lowercasing - ``lãi
suất`` and ``lai suat`` are the same term to a reader and should be to a scanner.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from goldpipeline.schemas.news import NewsCategory

DEFAULT_RELEVANCE_THRESHOLD = 2.0
"""Below this an item is not worth a reader's attention, or a prompt's budget.

Set so that a single weak signal - a passing mention of the dollar - is not
enough on its own, while any gold mention or any policy story clears it.
"""


@dataclass(frozen=True)
class CategorySpec:
    """One reason an item might matter, and how much it counts."""

    category: NewsCategory
    weight: float
    terms: tuple[str, ...]
    """Words and phrases, already diacritic-free and lowercase.

    Written in folded form so the table reads the way matching works. A term
    with a space is matched as a phrase.
    """


TAXONOMY: tuple[CategorySpec, ...] = (
    CategorySpec(
        category=NewsCategory.GOLD,
        weight=5.0,
        terms=(
            "gold",
            "xau",
            "xauusd",
            "xau usd",
            "bullion",
            "vang",
            "gia vang",
            "vang mieng",
            "vang the gioi",
        ),
    ),
    CategorySpec(
        category=NewsCategory.MONETARY_POLICY,
        weight=3.0,
        terms=(
            "fed",
            "fomc",
            "powell",
            "rate cut",
            "rate hike",
            "rate decision",
            "hawkish",
            "dovish",
            "monetary policy",
            "quantitative tightening",
            "lai suat",
            "ha lai suat",
            "tang lai suat",
            "chinh sach tien te",
        ),
    ),
    CategorySpec(
        category=NewsCategory.INFLATION,
        weight=3.0,
        terms=(
            "cpi",
            "pce",
            "inflation",
            "core inflation",
            "deflation",
            "lam phat",
            "chi so gia tieu dung",
        ),
    ),
    CategorySpec(
        category=NewsCategory.USD_DXY,
        weight=2.5,
        terms=(
            "dxy",
            "dollar index",
            "us dollar",
            "greenback",
            "dong usd",
            "dong do la",
            "chi so dollar",
            "ty gia usd",
        ),
    ),
    CategorySpec(
        category=NewsCategory.TREASURY_YIELDS,
        weight=2.5,
        terms=(
            "treasury",
            "treasuries",
            "bond yield",
            "yields",
            "yield curve",
            "10 year yield",
            "loi suat",
            "loi suat trai phieu",
            "trai phieu kho bac",
        ),
    ),
    CategorySpec(
        category=NewsCategory.ETF_FLOWS,
        weight=2.0,
        terms=(
            "etf",
            "gld",
            "spdr",
            "etf holdings",
            "etf inflows",
            "etf outflows",
            "dong von",
            "quy etf",
        ),
    ),
    CategorySpec(
        category=NewsCategory.GEOPOLITICS_RISK,
        weight=2.0,
        terms=(
            "war",
            "conflict",
            "sanctions",
            "geopolitical",
            "tensions",
            "military strike",
            "ceasefire",
            "chien tranh",
            "xung dot",
            "trung phat",
            "cang thang dia chinh tri",
        ),
    ),
    CategorySpec(
        category=NewsCategory.US_MACRO,
        weight=1.5,
        terms=(
            "nonfarm",
            "non farm",
            "payrolls",
            "jobless claims",
            "unemployment",
            "gdp",
            "pmi",
            "retail sales",
            "consumer confidence",
            "that nghiep",
            "bang luong",
        ),
    ),
    CategorySpec(
        category=NewsCategory.CENTRAL_BANKS,
        weight=1.5,
        terms=(
            "ecb",
            "boj",
            "pboc",
            "boe",
            "central bank",
            "rba",
            "ngan hang trung uong",
        ),
    ),
)
"""Every category, in descending weight. A test asserts it covers the enum."""


@dataclass(frozen=True)
class RelevanceResult:
    """What a scoring pass concluded about one item."""

    score: float
    categories: tuple[NewsCategory, ...]

    @property
    def relevant(self) -> bool:
        return self.score > 0


_LETTER_SUBSTITUTIONS = str.maketrans({"đ": "d", "Đ": "d", "ð": "d", "ø": "o", "ł": "l"})
"""Letters that decomposition does not touch.

Vietnamese ``đ`` is its own letter, not ``d`` plus a mark, so NFD leaves it
whole and ``xung đột`` folds to ``xung đot`` - which matches no term anybody
would write. Every Vietnamese phrase containing ``đ`` silently scored zero until
this table existed: ``xung đột``, ``đồng USD``, ``địa chính trị``. A few other
stroked letters are included for the same reason, cheaply.
"""


def fold(text: str) -> str:
    """Lowercase, strip diacritics, and collapse whitespace.

    Explicit rather than incidental: NFD decomposition then dropping combining
    marks turns ``lãi suất`` into ``lai suat`` and ``Fed's`` into ``fed's``, so
    one written form of a term matches every way a channel might spell it.
    Stroked letters are substituted first, because decomposition cannot help
    with them. Punctuation is kept - it is what word boundaries are made of.
    """
    substituted = text.lower().translate(_LETTER_SUBSTITUTIONS)
    decomposed = unicodedata.normalize("NFD", substituted)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", stripped).strip()


def _pattern_for(term: str) -> re.Pattern[str]:
    """A word-boundary pattern for one term or phrase.

    Interior whitespace becomes ``\\s+`` so "rate cut" matches across a line
    break, and the whole thing is bounded so "fed" cannot match "federal".
    """
    parts = [re.escape(word) for word in term.split()]
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b")


_COMPILED: tuple[tuple[CategorySpec, tuple[re.Pattern[str], ...]], ...] = tuple(
    (spec, tuple(_pattern_for(term) for term in spec.terms)) for spec in TAXONOMY
)


def score_text(text: str) -> RelevanceResult:
    """Score already-normalized or raw text against the taxonomy.

    Folding happens here, so callers never have to remember to do it.
    """
    folded = fold(text)
    matched: list[NewsCategory] = []
    total = 0.0

    for spec, patterns in _COMPILED:
        if any(pattern.search(folded) for pattern in patterns):
            matched.append(spec.category)
            total += spec.weight

    return RelevanceResult(score=round(total, 3), categories=tuple(matched))


def is_relevant(result: RelevanceResult, threshold: float = DEFAULT_RELEVANCE_THRESHOLD) -> bool:
    """Whether an item clears the bar for inclusion."""
    return result.score >= threshold


__all__ = [
    "DEFAULT_RELEVANCE_THRESHOLD",
    "TAXONOMY",
    "CategorySpec",
    "RelevanceResult",
    "fold",
    "is_relevant",
    "score_text",
]
