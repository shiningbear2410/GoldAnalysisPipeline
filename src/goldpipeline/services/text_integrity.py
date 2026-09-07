"""Did an edit quietly stop being Vietnamese?

Round 6.5c.3a. Narrow on purpose, and the narrowness is the design.

This module answers exactly one question about a model's edit: **is the text it
returned the same prose with its diacritics removed?** That is a mechanical
transformation with a mechanical signature, and it is detectable without
knowing a word of Vietnamese.

**It is not a spellchecker, and must never become one.** It does not judge
grammar, naturalness, or whether every word carries the accent a dictionary
would want. `Fed`, `SPDR`, `USD`, `CPI` and `ETF` are correct Vietnamese
financial prose, and a check that wanted accents on those would block the
product for being right. Nothing here counts accents against an expectation;
it compares an edit against *what it was an edit of*.

**Why a comparison rather than an absolute rule.** "This text should contain
Vietnamese diacritics" sounds simpler and is worse. A digest built from
accent-free source bulletins - which is a thing that happens - would fail it
while being an honest report of what those bulletins said. What is genuinely
wrong is an edit that takes accented prose and hands back the same sentences
without accents, and that is a two-sided fact.

**Conservative by construction.** Four signals must agree before anything is
reported, and every threshold is set so that a false positive is harder than a
missed isolated accent. A check that blocked a correct repair would be worse
than the defect it is looking for: the repair path allows exactly one model
call, so a false positive does not cost a retry - it costs the Run.
"""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher

from goldpipeline.services.content_safety import fold

logger = logging.getLogger(__name__)

REPLACEMENT_CHAR = "�"
"""What a decoder writes when bytes did not mean what it was told they meant."""

MIN_MARKS_BEFORE = 6
"""How accented the original must be before its loss is meaningful.

Six is low enough to catch a single short sentence and high enough that a
mostly-English line - "Fed CPI USD" plus a word or two - never qualifies as
"was accented" in the first place.
"""

MAX_MARKS_AFTER = 1
"""How few marks the edit must retain to look stripped.

One rather than zero because a model that strips accents sometimes leaves a
single word untouched, and a check that demanded a perfect zero would be
defeated by the least interesting possible accident.
"""

MIN_LETTERS_AFTER = 40
"""Below this an edit is too short to judge.

A repair that cuts a sentence to four words has changed the text, not
transliterated it, and the similarity signal below is noise at that length.
"""

SIMILARITY_THRESHOLD = 0.60
"""How alike the two texts must be, once both are stripped, to be "the same prose".

The decisive signal. A genuine rewrite says something different and scores far
below this even when it happens to use fewer accents; a transliteration is the
same words and scores near 1.0. Set at 0.60 rather than higher because a
stripping model often also trims a clause, and rather than lower because two
unrelated Vietnamese sentences share a surprising amount of short-word
structure.
"""

VIETNAMESE_LETTERS = frozenset("ăâđêôơưĂÂĐÊÔƠƯ")
"""Letters Vietnamese has and ASCII does not, independent of tone marks.

``đ`` and the vowel variants are *letters*, not decorated ones, so no amount of
Unicode decomposition finds a combining mark on them. They are counted here so
that "Đà Nẵng" reads as accented even before its tones are considered.
"""


@dataclass(frozen=True)
class TextIntegrityFinding:
    """One field whose edit does not look like Vietnamese any more."""

    field: str
    reason: str
    marks_before: int
    marks_after: int
    similarity: float

    def describe(self) -> str:
        return (
            f"{self.field}: {self.reason} "
            f"(diacritics {self.marks_before} -> {self.marks_after}, "
            f"similarity {self.similarity:.2f})"
        )


def diacritic_count(text: str) -> int:
    """How many characters carry Vietnamese orthography beyond plain ASCII.

    Two kinds, and both count: a letter with a combining tone or vowel mark,
    and the letters Vietnamese simply has that English does not.
    """
    total = 0
    for char in unicodedata.normalize("NFC", text):
        if char in VIETNAMESE_LETTERS:
            total += 1
            continue
        decomposed = unicodedata.normalize("NFD", char)
        if len(decomposed) > 1 and any(unicodedata.combining(part) for part in decomposed[1:]):
            base = decomposed[0]
            if base.isalpha() and base.isascii():
                total += 1
    return total


def _letters(text: str) -> int:
    return sum(1 for char in text if char.isalpha())


def _similarity(before: str, after: str) -> float:
    """How alike two texts are once accents, case and spacing are gone.

    Uses :func:`~goldpipeline.services.content_safety.fold` - the project's one
    diacritic-stripping function - rather than a second copy of the same three
    lines. A separate implementation here would eventually disagree with the one
    provenance matching uses, and then two checks would mean different things by
    "the same text".
    """
    left = " ".join(fold(before).split())
    right = " ".join(fold(after).split())
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def check_edit(field: str, before: str, after: str) -> TextIntegrityFinding | None:
    """Whether *after* is *before* with the Vietnamese taken out of it.

    Args:
        field: Where this text lives, for the failure message.
        before: The text the model was asked to repair.
        after: What it returned.

    Returns:
        A finding, or ``None`` when the edit is acceptable - which includes
        every ordinary rewrite, every deletion, and every text that was never
        accented to begin with.
    """
    if REPLACEMENT_CHAR in after:
        return TextIntegrityFinding(
            field=field,
            reason="contains the Unicode replacement character; the text was decoded wrongly",
            marks_before=diacritic_count(before),
            marks_after=diacritic_count(after),
            similarity=_similarity(before, after),
        )

    marks_before = diacritic_count(before)
    marks_after = diacritic_count(after)

    # Four signals, all required. Any one of them alone would be an opinion.
    if marks_before < MIN_MARKS_BEFORE:
        return None  # it was not accented prose; there was nothing to strip
    if marks_after > MAX_MARKS_AFTER:
        return None  # it still reads as Vietnamese
    if _letters(after) < MIN_LETTERS_AFTER:
        return None  # too short to distinguish a rewrite from a transliteration

    similarity = _similarity(before, after)
    if similarity < SIMILARITY_THRESHOLD:
        return None  # different prose, not the same prose without its accents

    return TextIntegrityFinding(
        field=field,
        reason="reads as the same Vietnamese with its diacritics removed",
        marks_before=marks_before,
        marks_after=marks_after,
        similarity=similarity,
    )


def check_new_text(field: str, text: str) -> TextIntegrityFinding | None:
    """Whether text with no prior version is decodable at all.

    Only the corruption half applies here: with nothing to compare against,
    there is no honest way to say a sentence "should" have been accented, and
    guessing would be the spellchecker this module refuses to be.
    """
    if REPLACEMENT_CHAR not in text:
        return None
    return TextIntegrityFinding(
        field=field,
        reason="contains the Unicode replacement character; the text was decoded wrongly",
        marks_before=0,
        marks_after=diacritic_count(text),
        similarity=0.0,
    )


def check_edits(pairs: Iterable[tuple[str, str, str]]) -> list[TextIntegrityFinding]:
    """Check many ``(field, before, after)`` edits at once, in order."""
    findings = [
        finding
        for field, before, after in pairs
        if (finding := check_edit(field, before, after)) is not None
    ]
    if findings:
        logger.info("text_integrity failures=%s", [finding.describe() for finding in findings])
    return findings


__all__ = [
    "MAX_MARKS_AFTER",
    "MIN_LETTERS_AFTER",
    "MIN_MARKS_BEFORE",
    "REPLACEMENT_CHAR",
    "SIMILARITY_THRESHOLD",
    "TextIntegrityFinding",
    "check_edit",
    "check_edits",
    "check_new_text",
    "diacritic_count",
]
