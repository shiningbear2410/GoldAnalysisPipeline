"""Merging the same story told by several channels.

Four channels carrying one Reuters headline is one fact, not four, and a prompt
given all four spends its budget saying the same thing. But two genuinely
different reports that happen to share vocabulary are two facts, and merging
them loses one.

**Transparent, not learned.** Similarity is Jaccard overlap of normalized token
sets - a number anyone can recompute by hand from the two texts. No embedding
model, no threshold nobody can explain, and no dependence on a service being up.

**Deterministic regardless of input order.** Items are sorted canonically before
grouping, so shuffling the input cannot change which item becomes the
representative or which becomes a corroboration. Greedy grouping over an
unsorted list is the classic way to get results that differ between runs and
are maddening to debug.

**The earliest copy wins.** Whoever published first is the representative; the
others become corroborating channels. That keeps the timestamp honest - a story
is as old as its first telling, not as its latest repost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from goldpipeline.schemas.news import NewsItem
from goldpipeline.services.news_taxonomy import fold

DEFAULT_SIMILARITY = 0.75
"""Token overlap above which two texts are the same story.

High enough that two reports sharing only financial vocabulary stay separate,
low enough that a repost with an added emoji, hashtag or source credit merges.
"""

MIN_TOKENS_FOR_SIMILARITY = 4
"""Below this, only an exact match counts.

Short texts overlap by accident: "Gold up" and "Gold down" share half their
tokens and mean opposite things. Rather than tune a threshold for them, they are
required to be identical.
"""

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> frozenset[str]:
    """Folded alphanumeric tokens, as a set.

    A set rather than a sequence: word order varies between reposts far more
    than word choice does, and the question here is "the same story?", not "the
    same sentence?".
    """
    return frozenset(_TOKEN_RE.findall(fold(text)))


def similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard overlap: shared tokens over total distinct tokens."""
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


@dataclass(frozen=True)
class _Group:
    representative: NewsItem
    tokens: frozenset[str]
    members: list[NewsItem]


def deduplicate(
    items: list[NewsItem],
    *,
    threshold: float = DEFAULT_SIMILARITY,
) -> list[NewsItem]:
    """Merge near-identical items, keeping the earliest of each story.

    Returns one item per distinct story, carrying the channels that corroborated
    it and how many copies were folded in. Input is not mutated.
    """
    if not items:
        return []

    # Canonical order first: earliest, then channel, then id. Grouping is greedy,
    # so the order decides the outcome - and it must not be the caller's order.
    ordered = sorted(items, key=lambda item: (item.published_at, item.channel, item.message_id))

    groups: list[_Group] = []
    for item in ordered:
        tokens = tokenize(item.text)
        match = _find_group(groups, tokens, threshold)
        if match is None:
            groups.append(_Group(representative=item, tokens=tokens, members=[item]))
        else:
            match.members.append(item)

    merged: list[NewsItem] = []
    for group in groups:
        others = [m for m in group.members if m is not group.representative]
        # Distinct channels only. Counting messages would let one channel
        # reposting itself look like independent confirmation.
        corroborating = sorted(
            {m.channel for m in others if m.channel != group.representative.channel}
        )
        merged.append(
            group.representative.model_copy(
                update={
                    "corroborating_channels": corroborating,
                    "duplicate_count": len(others),
                }
            )
        )
    return merged


def _find_group(groups: list[_Group], tokens: frozenset[str], threshold: float) -> _Group | None:
    """The first existing group this text belongs to, if any."""
    for group in groups:
        if len(tokens) < MIN_TOKENS_FOR_SIMILARITY or len(group.tokens) < MIN_TOKENS_FOR_SIMILARITY:
            if tokens == group.tokens:
                return group
            continue
        if similarity(tokens, group.tokens) >= threshold:
            return group
    return None


__all__ = [
    "DEFAULT_SIMILARITY",
    "MIN_TOKENS_FOR_SIMILARITY",
    "deduplicate",
    "similarity",
    "tokenize",
]
