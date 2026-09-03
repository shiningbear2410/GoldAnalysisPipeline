"""Countable symptoms of machine-written prose. Symptoms, not a diagnosis.

Everything here is a number: how many soft connectives, how many bullets with
the same skeleton, how similar the sentence lengths are. A finding says "this
text has a property that machine-written Vietnamese usually has and a trader's
note usually does not". It never says the text was written by a machine - no
count can say that, and the module's name is chosen so nobody mistakes it for
one. Whether a piece *reads* like a person is the reviewer's question.

**Thresholds, not bans.** One ``Tuy nhiên`` is Vietnamese. Three connectives
in a 900-character note is a register. The operator's amendment is explicit
that phrases like ``Đáng chú ý là`` are allowed once, so every list below is
counted against a threshold rather than matched as forbidden - with the single
exception of openers that have no natural use in a short trading note at all.

**Conservative on purpose.** A false positive here teaches the finalizer to
"fix" a sentence that was fine, and that edit is the machine register arriving
by another door. Where a rule could over-trigger on the digest's own
contracted shape - time-stamped item lines are uniform by design - the rule
excludes that shape rather than guessing.

**Scope.** Only types whose contract carries the human-style flag are checked.
A rendered ``TRADE_PLAN`` has no author and no voice; its uniform bullets are
correct, and this module returns nothing for it.

**Not wired.** Findings carry a severity as data. Nothing reads it yet.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Sequence

from goldpipeline.schemas.article_contract import ArticleContract
from goldpipeline.schemas.output_findings import (
    MAX_FINDING_EXCERPT_CHARS,
    OutputFinding,
    OutputFindingCode,
)
from goldpipeline.schemas.review import Severity
from goldpipeline.services.prose import Span, fold, lines, paragraphs, sentences, tokens

# --------------------------------------------------------------------------
# vocabulary, in folded form so the tables read the way matching works
# --------------------------------------------------------------------------

THROAT_CLEARING_PHRASES: tuple[str, ...] = (
    "trong bai viet nay",
    "co the thay rang",
    "dieu nay cho thay",
    "nhu da de cap",
    "noi tom lai",
    "ve co ban",
    "khong the phu nhan",
    "can luu y rang",
)
"""Openers with no natural use in a short trading note. One is a finding."""

SOFT_CONNECTIVES: tuple[str, ...] = (
    "tuy nhien",
    "vi vay",
    "do do",
    "ben canh do",
    "hon nua",
    "ngoai ra",
    "dang chu y la",
    "nhin chung",
    "tom lai",
    "nhu vay",
    "mat khac",
    "dong thoi",
    "them vao do",
    "trong boi canh",
)
"""Ordinary Vietnamese, one at a time. Counted, never banned."""

CONNECTIVE_THRESHOLD = 3
"""Total soft connectives at which the density becomes a finding."""

REPEATED_CONSTRUCTION_THRESHOLD = 2
"""``không chỉ … mà còn`` twice is a template, not a sentence."""

MIN_SENTENCES_FOR_RHYTHM = 5
SENTENCE_MIN_WORDS = 3
SENTENCE_VARIATION_FLOOR = 0.25
"""Coefficient of variation below which sentence lengths count as uniform."""

MIN_UNIFORM_BULLETS = 4
BULLET_MIN_WORDS = 6
"""Shorter bullets are labels or markers - contracted shapes, not prose."""

MIN_UNIFORM_PARAGRAPHS = 3
PARAGRAPH_MIN_CHARS = 60
PARAGRAPH_TOLERANCE = 0.10

DUPLICATE_MIN_TOKENS = 6
DUPLICATE_OVERLAP = 0.6
"""Jaccard overlap of folded tokens at which two sentences say the same thing."""

NUMBER_RESTATED_THRESHOLD = 3
OPENER_TOKENS = 6

_BULLET_MARKER_RE = re.compile(r"^(?P<marker>[^\w\s]+|\d{1,2}[.)])\s+(?P<body>.+)$")
_TIMESTAMPED_ITEM_RE = re.compile(r"^\d{1,2}:\d{2}\s*[—–-]")
_HEADING_LINE_RE = re.compile(r"^(?:[^\w\s]+\s*[^\n]{0,48}|[^\n]{0,48}[:?])$")
"""A short line led by a marker, or ending in a colon or question mark."""
_SKELETON_PUNCTUATION_RE = re.compile(r"[—–\-:;,()]")
_CONSTRUCTION_RE = re.compile(r"\bkhong chi\b.{1,80}?\bma con\b")
_NUMBER_RE = re.compile(r"(?<![\d:/.,])\d[\d.,]*\d(?![\d:/])|(?<![\d:/.,])\d(?![\d:/.,])")


def _phrase_re(phrase: str) -> re.Pattern[str]:
    return re.compile(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])")


_THROAT_CLEARING_RES = tuple(_phrase_re(p) for p in THROAT_CLEARING_PHRASES)
_CONNECTIVE_RES = tuple(_phrase_re(p) for p in SOFT_CONNECTIVES)


def find_style_symptoms(
    article: str,
    *,
    contract: ArticleContract,
    prior_openings: Sequence[str] = (),
) -> list[OutputFinding]:
    """Every countable symptom in *article*, under *contract*.

    Args:
        article: The prose to examine.
        contract: Decides whether the human-style contract applies at all.
        prior_openings: The opening text of recently published articles of
            the same type, so an opener repeated day after day can be seen.
            The caller decides how many; this module only compares.
    """
    if not contract.human_style:
        return []

    folded = fold(article)
    findings: list[OutputFinding] = []
    findings.extend(_throat_clearing(article, folded))
    findings.extend(_connective_density(article, folded))
    findings.extend(_repeated_construction(article, folded))
    findings.extend(_sentence_rhythm(article))
    findings.extend(_bullet_shapes(article))
    findings.extend(_paragraph_rhythm(article))
    findings.extend(_duplicate_statements(article))
    findings.extend(_number_restatement(article))
    findings.extend(_repeated_opener(article, prior_openings))
    return findings


# --------------------------------------------------------------------------
# phrase counts
# --------------------------------------------------------------------------


def _throat_clearing(article: str, folded: str) -> list[OutputFinding]:
    out: list[OutputFinding] = []
    for pattern in _THROAT_CLEARING_RES:
        match = pattern.search(folded)
        if match is None:
            continue
        out.append(
            OutputFinding(
                code=OutputFindingCode.THROAT_CLEARING,
                severity=Severity.MEDIUM,
                message=(
                    f"Opener {article[match.start() : match.end()]!r} narrates instead of saying."
                ),
                count=1,
                threshold=1,
                position=match.start(),
                excerpt=_excerpt(article, match.start(), match.end()),
            )
        )
    return out


def _connective_density(article: str, folded: str) -> list[OutputFinding]:
    hits = sorted(
        (m.start(), m.end()) for pattern in _CONNECTIVE_RES for m in pattern.finditer(folded)
    )
    if len(hits) < CONNECTIVE_THRESHOLD:
        return []
    first = hits[0]
    return [
        OutputFinding(
            code=OutputFindingCode.CONNECTIVE_DENSITY,
            severity=Severity.MEDIUM,
            message=(
                f"{len(hits)} soft connectives. One is Vietnamese; this many is a register "
                f"most of which can be deleted with no loss."
            ),
            count=len(hits),
            threshold=CONNECTIVE_THRESHOLD,
            position=first[0],
            excerpt=_excerpt(article, first[0], first[1]),
        )
    ]


def _repeated_construction(article: str, folded: str) -> list[OutputFinding]:
    hits = list(_CONSTRUCTION_RE.finditer(folded))
    if len(hits) < REPEATED_CONSTRUCTION_THRESHOLD:
        return []
    return [
        OutputFinding(
            code=OutputFindingCode.REPEATED_CONSTRUCTION,
            severity=Severity.LOW,
            message=f"'không chỉ … mà còn' used {len(hits)} times.",
            count=len(hits),
            threshold=REPEATED_CONSTRUCTION_THRESHOLD,
            position=hits[0].start(),
            excerpt=_excerpt(article, hits[0].start(), hits[0].end()),
        )
    ]


# --------------------------------------------------------------------------
# rhythm
# --------------------------------------------------------------------------


def _sentence_rhythm(article: str) -> list[OutputFinding]:
    lengths = [len(tokens(s.text)) for s in sentences(article)]
    lengths = [n for n in lengths if n >= SENTENCE_MIN_WORDS]
    if len(lengths) < MIN_SENTENCES_FOR_RHYTHM:
        return []
    mean = statistics.fmean(lengths)
    variation = statistics.pstdev(lengths) / mean if mean else 0.0
    if variation >= SENTENCE_VARIATION_FLOOR:
        return []
    return [
        OutputFinding(
            code=OutputFindingCode.SENTENCE_LENGTH_UNIFORM,
            severity=Severity.LOW,
            message=(
                f"{len(lengths)} sentences of nearly equal length (mean {mean:.1f} words, "
                f"variation {variation:.2f}). A person varies the rhythm."
            ),
            count=len(lengths),
            threshold=MIN_SENTENCES_FOR_RHYTHM,
        )
    ]


def _bullet_skeleton(line: Span) -> tuple[str, str, int] | None:
    match = _BULLET_MARKER_RE.match(line.text)
    if match is None:
        return None
    body = match.group("body")
    if _TIMESTAMPED_ITEM_RE.match(body):
        return None  # a digest item line: uniform by contract, not by habit
    words = len(tokens(body))
    if words < BULLET_MIN_WORDS:
        return None
    punctuation = "".join(_SKELETON_PUNCTUATION_RE.findall(body))
    return match.group("marker"), punctuation, words // 4


def _bullet_shapes(article: str) -> list[OutputFinding]:
    skeletons: dict[tuple[str, str, int], list[Span]] = {}
    for line in lines(article):
        skeleton = _bullet_skeleton(line)
        if skeleton is not None:
            skeletons.setdefault(skeleton, []).append(line)
    for members in skeletons.values():
        if len(members) >= MIN_UNIFORM_BULLETS:
            first = members[0]
            return [
                OutputFinding(
                    code=OutputFindingCode.BULLET_SHAPE_UNIFORM,
                    severity=Severity.LOW,
                    message=(
                        f"{len(members)} bullets share one marker, punctuation skeleton and "
                        f"length. Identical rows read as a template."
                    ),
                    count=len(members),
                    threshold=MIN_UNIFORM_BULLETS,
                    position=first.start,
                    excerpt=_excerpt(article, first.start, first.end),
                )
            ]
    return []


def _prose_body(block: Span) -> str:
    """The paragraph without a section heading on its first line.

    The locked shapes put a heading over each block; measuring the heading
    would make every contracted section look the same length, which is the
    template's doing and not the writer's.
    """
    first, _, rest = block.text.partition("\n")
    if rest and _HEADING_LINE_RE.match(first):
        return rest.strip()
    return block.text


def _paragraph_rhythm(article: str) -> list[OutputFinding]:
    blocks = paragraphs(article)
    sizes_by_block = [len(_prose_body(p)) for p in blocks]
    for index in range(len(blocks) - MIN_UNIFORM_PARAGRAPHS + 1):
        window = blocks[index : index + MIN_UNIFORM_PARAGRAPHS]
        sizes = sizes_by_block[index : index + MIN_UNIFORM_PARAGRAPHS]
        if min(sizes) < PARAGRAPH_MIN_CHARS:
            continue  # a short block breaks the run; it is variation
        mean = statistics.fmean(sizes)
        if all(abs(size - mean) <= PARAGRAPH_TOLERANCE * mean for size in sizes):
            return [
                OutputFinding(
                    code=OutputFindingCode.PARAGRAPH_LENGTH_UNIFORM,
                    severity=Severity.LOW,
                    message=(
                        f"{MIN_UNIFORM_PARAGRAPHS} consecutive paragraphs within "
                        f"±{int(PARAGRAPH_TOLERANCE * 100)}% of one another in length."
                    ),
                    count=MIN_UNIFORM_PARAGRAPHS,
                    threshold=MIN_UNIFORM_PARAGRAPHS,
                    position=window[0].start,
                    excerpt=_excerpt(article, window[0].start, window[0].end),
                )
            ]
    return []


# --------------------------------------------------------------------------
# repetition
# --------------------------------------------------------------------------


def _duplicate_statements(article: str) -> list[OutputFinding]:
    candidates = [
        (span, frozenset(words))
        for span in sentences(article)
        if len(words := tokens(span.text)) >= DUPLICATE_MIN_TOKENS
    ]
    for i, (_first, a) in enumerate(candidates):
        for second, b in candidates[i + 1 :]:
            overlap = len(a & b) / len(a | b)
            if overlap >= DUPLICATE_OVERLAP:
                return [
                    OutputFinding(
                        code=OutputFindingCode.DUPLICATE_STATEMENT,
                        severity=Severity.MEDIUM,
                        message=(
                            f"Two sentences share {int(overlap * 100)}% of their words. "
                            f"One conclusion is enough."
                        ),
                        count=2,
                        threshold=2,
                        position=second.start,
                        excerpt=_excerpt(article, second.start, second.end),
                    )
                ]
    return []


def _number_restatement(article: str) -> list[OutputFinding]:
    seen: Counter[str] = Counter()
    first_at: dict[str, tuple[int, int]] = {}
    for match in _NUMBER_RE.finditer(article):
        literal = match.group(0)
        if sum(ch.isdigit() for ch in literal) < 2:
            continue
        key = literal.replace(",", ".")
        seen[key] += 1
        first_at.setdefault(key, (match.start(), match.end()))
    for key, count in seen.most_common():
        if count < NUMBER_RESTATED_THRESHOLD:
            break
        start, end = first_at[key]
        return [
            OutputFinding(
                code=OutputFindingCode.NUMBER_RESTATED,
                severity=Severity.LOW,
                message=f"The number {article[start:end]!r} is stated {count} times.",
                count=count,
                threshold=NUMBER_RESTATED_THRESHOLD,
                position=start,
                excerpt=_excerpt(article, start, end),
            )
        ]
    return []


def opening_tokens(article: str) -> tuple[str, ...]:
    """The first words of the body, skipping a title line if there is one."""
    body_lines = lines(article)
    if len(body_lines) > 1:
        body_lines = body_lines[1:]
    words: list[str] = []
    for line in body_lines:
        words.extend(tokens(line.text))
        if len(words) >= OPENER_TOKENS:
            break
    return tuple(words[:OPENER_TOKENS])


def _repeated_opener(article: str, prior_openings: Sequence[str]) -> list[OutputFinding]:
    current = opening_tokens(article)
    if len(current) < OPENER_TOKENS:
        return []
    repeats = sum(1 for prior in prior_openings if opening_tokens(prior) == current)
    if not repeats:
        return []
    return [
        OutputFinding(
            code=OutputFindingCode.REPEATED_OPENER,
            severity=Severity.MEDIUM,
            message=(
                f"Opens with the same {OPENER_TOKENS} words as {repeats} recent article(s). "
                f"A catchphrase is a template read over days."
            ),
            count=repeats,
            threshold=1,
        )
    ]


def _excerpt(text: str, start: int, end: int, window: int = 60) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right].replace("\n", " ").strip()
    framed = f"...{snippet}..." if left > 0 or right < len(text) else snippet
    return framed[:MAX_FINDING_EXCERPT_CHARS]


__all__ = [
    "CONNECTIVE_THRESHOLD",
    "SOFT_CONNECTIVES",
    "THROAT_CLEARING_PHRASES",
    "find_style_symptoms",
    "opening_tokens",
]
