"""Splitting an article into the units the output checks count.

Three checkers need the same three things - lines, paragraphs, sentences - and
the same folding. Kept here so a sentence means one thing across all of them:
a rhythm check and a causality check that disagree about where a sentence ends
would report on different texts.

Deliberately simple. A sentence ends at ``.``, ``!``, ``?`` or ``…`` followed
by whitespace, or at a line break; a decimal point inside ``4323.5`` is not
followed by whitespace, so it does not end anything. There is no abbreviation
table and no attempt at more, because the checks built on this are counting
symptoms, and a splitter nobody can predict would make the counts unreadable.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from goldpipeline.services.content_safety import fold

_TERMINATORS = frozenset(".!?…")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Span:
    """A slice of the article, with where it came from."""

    text: str
    start: int
    end: int


def lines(text: str) -> list[Span]:
    """Non-empty lines, stripped, with their original offsets."""
    out: list[Span] = []
    position = 0
    for raw in text.split("\n"):
        stripped = raw.strip()
        if stripped:
            start = position + raw.index(stripped)
            out.append(Span(stripped, start, start + len(stripped)))
        position += len(raw) + 1
    return out


def paragraphs(text: str) -> list[Span]:
    """Blocks separated by at least one blank line."""
    out: list[Span] = []
    for match in re.finditer(r"(?:[^\n]+\n?)+", text):
        block = match.group(0).strip()
        if block:
            start = match.start() + match.group(0).index(block)
            out.append(Span(block, start, start + len(block)))
    return out


def sentences(text: str) -> list[Span]:
    """Sentence-shaped runs of text, as described in the module docstring."""
    return [span for span in _sentence_spans(text) if span.text]


def _sentence_spans(text: str) -> Iterator[Span]:
    start = 0
    length = len(text)
    index = 0
    while index < length:
        char = text[index]
        boundary = char == "\n" or (
            char in _TERMINATORS and (index + 1 >= length or text[index + 1].isspace())
        )
        if boundary:
            end = index + 1 if char != "\n" else index
            yield _trimmed(text, start, end)
            start = index + 1
        index += 1
    if start < length:
        yield _trimmed(text, start, length)


def _trimmed(text: str, start: int, end: int) -> Span:
    chunk = text[start:end]
    stripped = chunk.strip()
    if not stripped:
        return Span("", start, start)
    offset = start + chunk.index(stripped)
    return Span(stripped, offset, offset + len(stripped))


def tokens(text: str) -> list[str]:
    """Folded alphanumeric words. Punctuation and emoji contribute nothing."""
    return _TOKEN_RE.findall(fold(text))


__all__ = ["Span", "fold", "lines", "paragraphs", "sentences", "tokens"]
