"""Holding an article to its type's contract, deterministically.

Length against the hard cap, sections present and absent, structures that
belong to a different product, the disclaimer count for *this* type, and the
plan's terminology. Every check reads the contract from
:mod:`goldpipeline.schemas.article_contract` and nothing else, so a contract
change is the only way a check changes.

**Cues are provisional.** A section is found by a short heading cue - the
folded start of a line - and the cues below follow the shapes the operator
locked in the design round. The prompt round owns the final wording; when it
settles, this table is updated to match, and the contract vocabulary it maps
onto does not move.

**Short is not a finding.** The contract carries target lengths as guidance.
:func:`length_report` reports them; :func:`check_length` reports only the hard
cap. Nothing here ever asks for more words.

**Not wired.** No production stage calls these checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from goldpipeline.schemas.article_contract import (
    ArticleContract,
    SectionKey,
    StructureKind,
)
from goldpipeline.schemas.output_findings import (
    MAX_FINDING_EXCERPT_CHARS,
    OutputFinding,
    OutputFindingCode,
)
from goldpipeline.schemas.review import Severity
from goldpipeline.services.prose import Span, fold, lines, sentences, tokens

SECTION_CUES: dict[SectionKey, tuple[str, ...]] = {
    SectionKey.ANALYSIS_TITLE: ("phan tich vang",),
    SectionKey.VERDICT: ("chot:", "chot lai:", "chot nhanh"),
    SectionKey.DRIVERS_UP: ("day len",),
    SectionKey.DRIVERS_DOWN: ("keo xuong",),
    SectionKey.PRICE_READ: ("gia dang noi gi",),
    SectionKey.WATCHING: ("minh dang cho",),
    SectionKey.DIGEST_TITLE: ("tin vang",),
    SectionKey.DIGEST_ITEMS: ("tin dang chu y",),
    SectionKey.PRICE_REACTION: ("gia phan ung",),
    SectionKey.BALANCE: ("can can",),
    SectionKey.PLAN_TITLE: ("chien luoc vang",),
}
"""Folded line-start cues per section. Emoji and punctuation before them are ignored.

``SEO``, ``BAI``, the digest window and the disclaimer are recognised by shape
rather than by cue, below.
"""

_LEADING_NOISE_RE = re.compile(r"^[^a-z]+")
_DIGEST_WINDOW_RE = re.compile(r"\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}.*(?:→|->|–|—|-).*\d{1,2}:\d{2}")
_NEWS_ITEM_LINE_RE = re.compile(r"^\s*[^\w\s]*\s*\d{1,2}:\d{2}\s*[—–-]")
_ZONE_LINE_RE = re.compile(r"^\s*[^\w\s]*\s*\d{3,6}(?:[.,]\d+)?\s*[–—-]\s*\d{3,6}(?:[.,]\d+)?")
_SCENARIO_RE = re.compile(r"\bneu\b.{1,120}?(?:\bthi\b|→|->)")
_DATE_LINE_RE = re.compile(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}|\d{4}-\d{2}-\d{2}")
_RISK_PARAMETER_RE = re.compile(
    r"\bsl\b|\btp\b|\br\s*:\s*r\b|\brr\b|stop loss|take profit|chot loi|cat lo|dung lo"
)
_SIDE_LABEL_RE = re.compile(r"^(seo|bai)\b[\s:]*$")
_FORBIDDEN_SIDE_RE = re.compile(r"\b(sell|buy)\b")
_DISCLAIMER_CUES = ("khong phai loi khuyen dau tu", "nhan dinh ca nhan")

MIN_STRUCTURE_LINES = 2
"""A structure is a pattern, and one line is not a pattern."""

EXPLANATORY_PROSE_MIN_WORDS = 12
"""A plan line longer than this is explaining something."""


@dataclass(frozen=True)
class LengthReport:
    """Where an article sits against its contract's lengths. Observation only."""

    chars: int
    target_min: int
    target_max: int
    hard_max: int

    @property
    def within_target(self) -> bool:
        return self.target_min <= self.chars <= self.target_max

    @property
    def over_hard_cap(self) -> bool:
        return self.chars > self.hard_max


def length_report(article: str, contract: ArticleContract) -> LengthReport:
    """Character count against the contract. Counts code points, as Telegram does."""
    return LengthReport(
        chars=len(article),
        target_min=contract.target_min_chars,
        target_max=contract.target_max_chars,
        hard_max=contract.hard_max_chars,
    )


def check_length(article: str, contract: ArticleContract) -> list[OutputFinding]:
    """The hard cap, and only the hard cap."""
    report = length_report(article, contract)
    if not report.over_hard_cap:
        return []
    return [
        OutputFinding(
            code=OutputFindingCode.HARD_CAP_EXCEEDED,
            severity=Severity.HIGH,
            message=(
                f"{report.chars} characters against a hard cap of {report.hard_max} "
                f"for {contract.article_type}."
            ),
            count=report.chars,
            threshold=report.hard_max,
        )
    ]


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------


def _heading_text(line: Span) -> str:
    return _LEADING_NOISE_RE.sub("", fold(line.text)).strip()


def _is_disclaimer_line(line: Span) -> bool:
    folded = fold(line.text)
    return any(cue in folded for cue in _DISCLAIMER_CUES)


def detect_sections(article: str) -> dict[SectionKey, int]:
    """Every section found, with the offset of the first line that shows it."""
    found: dict[SectionKey, int] = {}
    for line in lines(article):
        heading = _heading_text(line)
        for key, cues in SECTION_CUES.items():
            if key not in found and any(heading.startswith(cue) for cue in cues):
                found[key] = line.start
        side = _SIDE_LABEL_RE.match(heading)
        if side is not None:
            key = SectionKey.SEO if side.group(1) == "seo" else SectionKey.BAI
            found.setdefault(key, line.start)
        if SectionKey.DIGEST_WINDOW not in found and _DIGEST_WINDOW_RE.search(line.text):
            found[SectionKey.DIGEST_WINDOW] = line.start
        if SectionKey.DISCLAIMER not in found and _is_disclaimer_line(line):
            found[SectionKey.DISCLAIMER] = line.start
    structures = detect_structures(article)
    if SectionKey.DIGEST_ITEMS not in found and StructureKind.NEWS_ITEM_LIST in structures:
        found[SectionKey.DIGEST_ITEMS] = structures[StructureKind.NEWS_ITEM_LIST]
    return found


def check_sections(article: str, contract: ArticleContract) -> list[OutputFinding]:
    """Required sections missing, forbidden sections present."""
    found = detect_sections(article)
    findings: list[OutputFinding] = []
    for key in contract.required_sections:
        if key not in found:
            findings.append(
                OutputFinding(
                    code=OutputFindingCode.MISSING_REQUIRED_SECTION,
                    severity=Severity.HIGH,
                    message=f"{contract.article_type} requires a {key} section; none found.",
                )
            )
    for key in sorted(contract.forbidden_sections):
        if key in found:
            findings.append(
                OutputFinding(
                    code=OutputFindingCode.FORBIDDEN_SECTION_PRESENT,
                    severity=Severity.HIGH,
                    message=f"{contract.article_type} must not carry a {key} section.",
                    position=found[key],
                    excerpt=_excerpt(article, found[key]),
                )
            )
    return findings


# --------------------------------------------------------------------------
# structures
# --------------------------------------------------------------------------


def detect_structures(article: str) -> dict[StructureKind, int]:
    """Every forbidden-able shape found, with the offset of its first evidence."""
    found: dict[StructureKind, int] = {}
    article_lines = lines(article)

    item_lines = [ln for ln in article_lines if _NEWS_ITEM_LINE_RE.match(ln.text)]
    if len(item_lines) >= MIN_STRUCTURE_LINES:
        found[StructureKind.NEWS_ITEM_LIST] = item_lines[0].start

    zone_lines = [ln for ln in article_lines if _ZONE_LINE_RE.match(ln.text)]
    if len(zone_lines) >= MIN_STRUCTURE_LINES:
        found[StructureKind.TRADE_ZONE_LIST] = zone_lines[0].start

    scenario = [s for s in sentences(article) if _SCENARIO_RE.search(fold(s.text))]
    if len(scenario) >= MIN_STRUCTURE_LINES:
        found[StructureKind.SCENARIO_ESSAY] = scenario[0].start

    for line in article_lines:
        if len(tokens(line.text)) >= EXPLANATORY_PROSE_MIN_WORDS:
            found.setdefault(StructureKind.EXPLANATORY_PROSE, line.start)
        if _DATE_LINE_RE.search(line.text):
            found.setdefault(StructureKind.DATE_LINE, line.start)
        if _RISK_PARAMETER_RE.search(fold(line.text)):
            found.setdefault(StructureKind.RISK_PARAMETERS, line.start)
    return found


def check_structures(article: str, contract: ArticleContract) -> list[OutputFinding]:
    """Shapes that belong to another product."""
    found = detect_structures(article)
    return [
        OutputFinding(
            code=OutputFindingCode.FORBIDDEN_STRUCTURE_PRESENT,
            severity=Severity.HIGH,
            message=f"{contract.article_type} must not contain a {kind}.",
            position=found[kind],
            excerpt=_excerpt(article, found[kind]),
        )
        for kind in sorted(contract.forbidden_structures)
        if kind in found
    ]


# --------------------------------------------------------------------------
# disclaimer and terminology
# --------------------------------------------------------------------------


def count_disclaimers(article: str) -> int:
    """Lines that read as a disclaimer, exact wording or not."""
    return sum(1 for line in lines(article) if _is_disclaimer_line(line))


def check_disclaimer(article: str, contract: ArticleContract) -> list[OutputFinding]:
    """The count this type expects - one, or none - and the exact text when one."""
    expected = contract.disclaimer.expected_count
    actual = count_disclaimers(article)
    if actual != expected:
        return [
            OutputFinding(
                code=OutputFindingCode.DISCLAIMER_COUNT_MISMATCH,
                severity=Severity.MEDIUM,
                message=(
                    f"{contract.article_type} expects {expected} disclaimer(s); found {actual}."
                ),
                count=actual,
                threshold=expected,
            )
        ]
    if expected and contract.disclaimer.text not in article:
        return [
            OutputFinding(
                code=OutputFindingCode.DISCLAIMER_TEXT_MISMATCH,
                severity=Severity.LOW,
                message="The disclaimer is present but not in the contracted wording.",
                count=actual,
                threshold=expected,
            )
        ]
    return []


def check_terminology(article: str, contract: ArticleContract) -> list[OutputFinding]:
    """``SELL``/``BUY`` where the contract asks for ``SEO``/``BAI``."""
    if SectionKey.SEO not in contract.required_sections:
        return []
    findings: list[OutputFinding] = []
    for line in lines(article):
        match = _FORBIDDEN_SIDE_RE.search(fold(line.text))
        if match is None:
            continue
        findings.append(
            OutputFinding(
                code=OutputFindingCode.FORBIDDEN_TERMINOLOGY,
                severity=Severity.HIGH,
                message=(
                    f"{match.group(1).upper()!r} is not this product's vocabulary. "
                    f"The sides are SEO and BAI, literally."
                ),
                position=line.start + match.start(),
                excerpt=_excerpt(article, line.start),
            )
        )
    return findings


def check_contract(article: str, contract: ArticleContract) -> list[OutputFinding]:
    """Every contract check, in one list."""
    findings: list[OutputFinding] = []
    findings.extend(check_length(article, contract))
    findings.extend(check_sections(article, contract))
    findings.extend(check_structures(article, contract))
    findings.extend(check_disclaimer(article, contract))
    findings.extend(check_terminology(article, contract))
    return findings


def _excerpt(text: str, start: int, window: int = 80) -> str:
    snippet = text[start : start + window].split("\n", 1)[0].strip()
    return snippet[:MAX_FINDING_EXCERPT_CHARS]


__all__ = [
    "SECTION_CUES",
    "LengthReport",
    "check_contract",
    "check_disclaimer",
    "check_length",
    "check_sections",
    "check_structures",
    "check_terminology",
    "count_disclaimers",
    "detect_sections",
    "detect_structures",
    "length_report",
]
