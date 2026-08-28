"""Splitting an approved article into Telegram-sized messages.

One invariant governs everything here:

    "".join(plan_chunks(article)) == article

Exactly. Not "modulo whitespace", not "after normalisation". The article was
approved as specific bytes by the publish gate, and the publisher's job is to
deliver those bytes - so the chunker may only decide *where* to cut, never what
the text says. Nothing is stripped, normalised, or annotated; in particular
there is no ``(1/2)`` marker, because that would be content nobody approved.

Telegram measures message length in UTF-16 code units, not characters, so an
emoji outside the Basic Multilingual Plane costs two. This module counts the way
Telegram counts.
"""

from __future__ import annotations

import re

TELEGRAM_HARD_LIMIT = 4096
"""Telegram's documented ``sendMessage`` limit, in UTF-16 code units."""

SAFE_CHUNK_LIMIT = 3900
"""What this pipeline actually targets.

Deliberately under the hard limit. The margin costs nothing - almost every
article fits in one message anyway - and it means a miscount at the boundary
produces a slightly short message rather than a rejected one.
"""

_PARAGRAPH_BREAK = "\n\n"
_SENTENCE_END = re.compile(r"[.!?…]['\"”’)]?\s")


def utf16_length(text: str) -> int:
    """Length as Telegram counts it: UTF-16 code units.

    ``len(text)`` counts code points, which under-counts every emoji above the
    BMP. An article full of 🕯 and ⚡ can therefore pass a naive check and still
    be refused by the API.
    """
    return len(text.encode("utf-16-le")) // 2


def _max_prefix(text: str, limit: int) -> int:
    """The largest index whose prefix still fits in *limit* UTF-16 units.

    Walks code points, so a cut is never placed inside one. Surrogate pairs are
    a UTF-16 encoding detail; Python strings are sequences of code points, and
    slicing them cannot split a character.
    """
    used = 0
    for index, char in enumerate(text):
        cost = 2 if ord(char) > 0xFFFF else 1
        if used + cost > limit:
            return index
        used += cost
    return len(text)


def _find_cut(text: str, limit: int) -> int:
    """Choose where to split, preferring the most natural boundary available.

    Tried in order: a paragraph break, a line break, a sentence end, any
    whitespace, and finally a hard cut. Each candidate must leave a non-empty
    first chunk, or the loop would not make progress.

    The separator stays at the end of the chunk before it. That is what keeps
    the join exact - removing it, or re-adding it to the next chunk, would
    change the text.
    """
    ceiling = _max_prefix(text, limit)
    if ceiling >= len(text):
        return len(text)

    window = text[:ceiling]

    paragraph = window.rfind(_PARAGRAPH_BREAK)
    if paragraph > 0:
        return paragraph + len(_PARAGRAPH_BREAK)

    newline = window.rfind("\n")
    if newline > 0:
        return newline + 1

    sentence = None
    for match in _SENTENCE_END.finditer(window):
        sentence = match.end()
    if sentence and sentence > 0:
        return sentence

    for index in range(len(window) - 1, 0, -1):
        if window[index].isspace():
            return index + 1

    return ceiling


def plan_chunks(article: str, limit: int = SAFE_CHUNK_LIMIT) -> list[str]:
    """Split *article* into messages that each fit within *limit*.

    Args:
        article: The approved text, delivered verbatim.
        limit: Maximum UTF-16 code units per chunk.

    Returns:
        Chunks whose concatenation is exactly *article*. An article that already
        fits comes back as a single chunk - no artificial splitting.

    Raises:
        ValueError: If *limit* is too small to make progress.
    """
    if limit < 16:
        raise ValueError(f"chunk limit {limit} is too small to be usable")

    if not article:
        return [article]
    if utf16_length(article) <= limit:
        return [article]

    chunks: list[str] = []
    remaining = article

    while remaining:
        if utf16_length(remaining) <= limit:
            chunks.append(remaining)
            break
        cut = _find_cut(remaining, limit)
        if cut <= 0:  # pragma: no cover - _find_cut guarantees progress
            raise ValueError("chunker failed to make progress")
        chunks.append(remaining[:cut])
        remaining = remaining[cut:]

    return chunks


def verify_plan(article: str, chunks: list[str], limit: int = SAFE_CHUNK_LIMIT) -> None:
    """Assert a plan is safe to send.

    Called by the publisher before anything reaches the network, so a chunker
    bug becomes a refusal to publish rather than a mangled post.

    Raises:
        ValueError: If the chunks do not reassemble into *article*, if any chunk
            is empty, or if any chunk exceeds *limit*.
    """
    if not chunks:
        raise ValueError("chunk plan is empty")

    rejoined = "".join(chunks)
    if rejoined != article:
        raise ValueError(
            "chunk plan does not reassemble into the approved article "
            f"({len(rejoined)} characters rejoined vs {len(article)} approved)"
        )

    for index, chunk in enumerate(chunks):
        if not chunk:
            raise ValueError(f"chunk {index} is empty")
        size = utf16_length(chunk)
        if size > limit:
            raise ValueError(f"chunk {index} is {size} UTF-16 units, above the {limit} limit")


__all__ = [
    "SAFE_CHUNK_LIMIT",
    "TELEGRAM_HARD_LIMIT",
    "plan_chunks",
    "utf16_length",
    "verify_plan",
]
