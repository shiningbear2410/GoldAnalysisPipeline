"""A narrow detector for causal claims about price that nothing can support.

This pipeline has candles and time-stamped news. It has no counterfactual, so
it can never know that a headline *moved* the price - only that the price
moved after the headline appeared. The writer prompt will say so in words; this
module says so in a check, for the one sentence shape where the claim is
unmistakable: a news reference, a price movement and a causal connector in the
same sentence, with nobody named as the source of the link.

    tin Fed khiến vàng tăng            -> finding
    do USD yếu nên vàng bật lên        -> finding
    giá tăng sau thời điểm tin ra      -> nothing: temporal, not causal
    nhịp tăng trùng thời điểm ETF mua  -> nothing
    theo Reuters, vàng tăng do USD yếu -> nothing: attributed, a reported claim

The last case is the one real exception. When a source is named, the article
is reporting that *the source* asserted the cause - a fact the provenance
verifier already knows how to check - rather than asserting it in its own
voice.

**Deliberately narrow.** All three ingredients must be present in one sentence
before anything is reported. A sentence with a connector and no price, or a
price and no news entity, is somebody else's problem. This is a rule about one
sentence shape, not a model of causation, and it is meant to stay that way.

**Not wired.** The finding carries a severity as data; nothing reads it yet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from goldpipeline.schemas.news import NewsCategory
from goldpipeline.schemas.output_findings import (
    MAX_FINDING_EXCERPT_CHARS,
    OutputFinding,
    OutputFindingCode,
)
from goldpipeline.schemas.review import Severity
from goldpipeline.services.news_taxonomy import TAXONOMY
from goldpipeline.services.prose import Span, fold, sentences

EXTRA_NEWS_TERMS: tuple[str, ...] = (
    "tin",
    "tin tuc",
    "thong tin",
    "du lieu",
    "so lieu",
    "bao cao",
    "phat bieu",
    "bien ban",
    "nfp",
    "thue quan",
    "bau cu",
    "usd",
    "dollar",
    "do la",
    "chung khoan",
    "dau tho",
)
"""News references the taxonomy has no category for, in folded form."""

NEWS_TERMS: tuple[str, ...] = (
    tuple(
        term for spec in TAXONOMY if spec.category is not NewsCategory.GOLD for term in spec.terms
    )
    + EXTRA_NEWS_TERMS
)
"""What counts as a news or entity reference.

Reuses the collector's taxonomy so a term that makes an item relevant is the
same term that makes a sentence about it. Gold's own category is excluded: in
this check ``vàng`` is the thing being moved, not the thing doing the moving.
"""

PRICE_SUBJECTS: tuple[str, ...] = ("vang", "gia", "xau", "xauusd", "gold", "kim loai quy")
MOVE_TERMS: tuple[str, ...] = (
    "tang",
    "giam",
    "bat",
    "roi",
    "lao doc",
    "sut",
    "tut",
    "truot",
    "phuc hoi",
    "hoi phuc",
    "dieu chinh",
    "bung no",
    "di len",
    "di xuong",
    "len",
    "xuong",
    "leo",
    "nhay vot",
)

ATTRIBUTION_MARKERS: tuple[str, ...] = (
    "cho biet",
    "cho rang",
    "cho hay",
    "nhan dinh rang",
    "danh gia rang",
    "noi rang",
    "dan loi",
    "trich dan",
    "theo nguon",
)
"""Words that make the sentence reported speech rather than the article's claim."""


@dataclass(frozen=True)
class Connector:
    """One causal wording, and the folded pattern that finds it."""

    label: str
    pattern: re.Pattern[str]


CAUSAL_CONNECTORS: tuple[Connector, ...] = (
    Connector("khiến", re.compile(r"\bkhien\b")),
    Connector("làm cho", re.compile(r"\blam cho\b")),
    Connector("làm giá/vàng", re.compile(r"\blam (?:gia|vang)\b")),
    Connector("gây ra", re.compile(r"\bgay ra\b")),
    Connector("dẫn đến", re.compile(r"\bdan den\b")),
    Connector("nguyên nhân", re.compile(r"\bnguyen nhan\b")),
    Connector("bởi vì", re.compile(r"\bboi vi\b")),
    Connector("do … nên", re.compile(r"\bdo\b(?!\s+do\b).{1,120}?\bnen\b")),
    Connector("nhờ … mà/nên", re.compile(r"\bnho\b.{1,120}?\b(?:ma|nen)\b")),
    Connector("đẩy giá/vàng", re.compile(r"\bday (?:gia|vang)\b")),
    Connector("kéo giá/vàng", re.compile(r"\bkeo (?:gia|vang)\b")),
)
"""Article-voice causal wordings. ``do đó`` is a connective, not a cause, and
is excluded from the ``do … nên`` pattern by the lookahead."""

# "theo" alone is attribution ("theo Reuters") but also "theo dõi" (follow),
# "theo sát", "theo hướng" - so it counts only when not one of those.
_THEO_RE = re.compile(r"\btheo\b(?!\s+(?:doi|sat|huong|chieu|kip|xu|da|do)\b)")


def _any_re(terms: tuple[str, ...]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.compile(r"(?<![a-z0-9])(?:" + alternatives + r")(?![a-z0-9])")


_NEWS_RE = _any_re(NEWS_TERMS)
_PRICE_RE = _any_re(PRICE_SUBJECTS)
_MOVE_RE = _any_re(MOVE_TERMS)
_ATTRIBUTION_RE = _any_re(ATTRIBUTION_MARKERS)

# "tin" is a news word; "tín hiệu" folds to "tin hieu" and is not.
_TIN_HIEU_RE = re.compile(r"\btin hieu\b")


def find_causal_claims(article: str) -> list[OutputFinding]:
    """Sentences that assert, in the article's own voice, that news moved price."""
    findings: list[OutputFinding] = []
    for sentence in sentences(article):
        finding = _examine(article, sentence)
        if finding is not None:
            findings.append(finding)
    return findings


def _examine(article: str, sentence: Span) -> OutputFinding | None:
    if sentence.text.rstrip().endswith("?"):
        return None  # a question asks; it does not assert
    folded = _TIN_HIEU_RE.sub("signal", fold(sentence.text))

    if _PRICE_RE.search(folded) is None or _MOVE_RE.search(folded) is None:
        return None
    if _NEWS_RE.search(folded) is None:
        return None
    if _ATTRIBUTION_RE.search(folded) is not None or _THEO_RE.search(folded) is not None:
        return None

    for connector in CAUSAL_CONNECTORS:
        match = connector.pattern.search(folded)
        if match is None:
            continue
        return OutputFinding(
            code=OutputFindingCode.UNATTRIBUTED_CAUSAL_CLAIM,
            severity=Severity.HIGH,
            message=(
                f"'{connector.label}' links a news reference to a price move with no "
                f"source named. The data can show timing; it cannot show cause."
            ),
            count=1,
            threshold=1,
            position=sentence.start + match.start(),
            excerpt=sentence.text[:MAX_FINDING_EXCERPT_CHARS],
        )
    return None


__all__ = [
    "ATTRIBUTION_MARKERS",
    "CAUSAL_CONNECTORS",
    "MOVE_TERMS",
    "NEWS_TERMS",
    "PRICE_SUBJECTS",
    "find_causal_claims",
]
