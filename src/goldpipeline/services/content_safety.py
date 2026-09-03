"""Deterministic content scanners for the publish gate.

These run only on ``claude_final.md`` - the bytes about to leave the machine.
They are deliberately separate from the Round 3 prechecks: those judge a draft
mid-pipeline and their thresholds are tuned for that, while these are the last
boundary and are tuned to fail closed.

Four concerns, all pattern-based and all with tests:

* **instruction-shaped text.** Round 4 exposed a real gap: a finalizer told to
  make the minimum necessary edit will leave "Ignore all previous instructions"
  in the prose, because no review issue named it. Nothing downstream should ever
  publish a sentence that reads as an attempt to steer a model.
* **credential-shaped values.** Not variable *names* - actual token shapes. A
  detection is reported redacted, so the decision artifact never becomes a
  second copy of the secret.
* **external factual claims.** The context carries no news. An article stating
  that the Fed *did* something has invented it.
* **structure and transport sanity.** Empty, truncated, absurdly long, a JSON
  dump, a traceback, control characters.

Every pattern list below is small, explicit and covered by both positive and
negative tests. There is no NLP here, and there is deliberately no attempt at
cleverness: a scanner nobody can predict is a scanner nobody can trust.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MIN_ARTICLE_CHARS = 100
"""Below this an article is a stub, not a publishable piece.

The shipped fixtures run 400-700 characters; a real commentary is longer still.
"""

MAX_ARTICLE_CHARS = 8000
"""Above this something has gone wrong - a dump, a loop, a pasted transcript.

Also comfortably under Telegram's 4096-character *message* limit times two, so
Round 6 can split a long piece rather than being handed something unpublishable.
"""

TELEGRAM_MESSAGE_LIMIT = 4096
"""Telegram's per-message character limit. Longer articles need splitting."""


# --------------------------------------------------------------------------
# instruction-shaped text
# --------------------------------------------------------------------------

_INSTRUCTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"ignore\s+(?:all\s+|any\s+)?(?:the\s+)?previous\s+instructions?",
        "ignore previous instructions",
    ),
    (
        r"ignore\s+(?:all\s+|any\s+)?(?:prior|earlier|above)\s+instructions?",
        "ignore prior instructions",
    ),
    (r"ignore\s+all\s+instructions?", "ignore all instructions"),
    (
        r"disregard\s+(?:all\s+|any\s+)?(?:previous|prior|earlier)\s+instructions?",
        "disregard instructions",
    ),
    (r"system\s+prompt", "system prompt"),
    (r"developer\s+(?:message|prompt|instructions?)", "developer message"),
    (
        r"you\s+are\s+(?:now\s+)?"
        r"(?:chatgpt|claude|gpt-?\d|an?\s+(?:ai|assistant|language\s+model))",
        "role reassignment",
    ),
    (
        r"(?:print|reveal|show|output|display|dump)\s+(?:me\s+)?(?:your\s+|the\s+)?"
        r"(?:api[\s_-]?key|secret|token|credentials?|system\s+prompt)",
        "reveal a secret",
    ),
    (
        r"\b(?:ANTHROPIC|OPENAI|TELEGRAM)_[A-Z_]*(?:API_KEY|TOKEN|SECRET)\b",
        "credential variable name",
    ),
    (r"mark\s+(?:this|it|the)\s*(?:\w+\s+)?(?:as\s+)?pass\b", "mark this pass"),
    (r"change\s+(?:the\s+)?symbol\s+to\b", "change the symbol"),
    (r"follow\s+these\s+instructions\b", "follow these instructions"),
    (r"</?(?:system|assistant|user|instructions?)[_\s]*(?:instructions?)?>", "role tag"),
    (r"^\s*SYSTEM\s*:", "system-role line"),
)
"""English model-control phrases.

Each entry is ``(regex, label)``; the label is what a reader sees in the
decision artifact, so no raw article text is required to explain a block.
"""

_INSTRUCTION_PATTERNS_VI: tuple[tuple[str, str], ...] = (
    (
        r"bo\s+qua\s+(?:moi\s+|tat\s+ca\s+|cac\s+)?(?:chi\s+dan|huong\s+dan|quy\s+tac|lenh)",
        "bỏ qua chỉ dẫn",
    ),
    (r"(?:in|hien\s+thi|xuat)\s+(?:ra\s+)?api\s*key", "in API key"),
    (r"danh\s+dau\s+(?:la\s+)?pass\b", "đánh dấu PASS"),
    (r"doi\s+symbol\s+(?:thanh|sang)\b", "đổi symbol"),
    (r"ban\s+(?:khong\s+con|bay\s+gio\s+la)\s+", "role reassignment"),
)
"""Vietnamese equivalents, matched against diacritic-folded text.

Folding is what makes this tractable: ``bỏ qua``, ``bo qua`` and ``BỎ QUA`` all
reduce to the same string, so one pattern covers how people actually type.
"""


# --------------------------------------------------------------------------
# credential shapes
# --------------------------------------------------------------------------

_CREDENTIAL_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"sk-ant-api\d{2}-[A-Za-z0-9_-]{20,}", "anthropic_api_key"),
    (r"sk-ant-[A-Za-z0-9_-]{24,}", "anthropic_key"),
    (r"sk-proj-[A-Za-z0-9_-]{20,}", "openai_project_key"),
    (r"sk-[A-Za-z0-9]{32,}", "openai_key"),
    (r"gh[pousr]_[A-Za-z0-9]{36,}", "github_token"),
    (r"github_pat_[A-Za-z0-9_]{30,}", "github_pat"),
    (r"xox[baprs]-[A-Za-z0-9-]{10,}", "slack_token"),
    (r"\b\d{8,10}:AA[A-Za-z0-9_-]{33}\b", "telegram_bot_token"),
    (r"AKIA[0-9A-Z]{16}", "aws_access_key"),
    (r"Bearer\s+[A-Za-z0-9._~+/-]{24,}=*", "bearer_token"),
    (r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}", "jwt"),
)
"""Token shapes worth blocking on sight.

Conservative on purpose: each requires a vendor prefix *and* a substantial body,
so ordinary prose cannot trip them. This is not an enterprise secret scanner -
it is a last line of defence against the specific shapes this pipeline could
plausibly leak.
"""

REDACTION_PREFIX_CHARS = 6
REDACTION_SUFFIX_CHARS = 4
"""How much of a detected token survives into the decision artifact.

The prefix identifies the vendor and the last four identify *which* key, so an
operator knows what to rotate - the same convention a card statement uses. The
body, which is the part that would let someone use it, never leaves this module.
"""


def redact(value: str) -> str:
    """Render a detected secret safely.

    Short values are masked entirely: with little to hide behind, a prefix and a
    suffix would be most of the string.
    """
    if len(value) <= REDACTION_PREFIX_CHARS + REDACTION_SUFFIX_CHARS + 4:
        return f"<redacted:{len(value)} chars>"
    head = value[:REDACTION_PREFIX_CHARS]
    tail = value[-REDACTION_SUFFIX_CHARS:]
    return f"{head}…{tail} (<redacted:{len(value)} chars>)"


# --------------------------------------------------------------------------
# external factual claims
# --------------------------------------------------------------------------

_NEWS_ENTITIES: tuple[str, ...] = (
    "fed",
    "fomc",
    "powell",
    "ecb",
    "boj",
    "cpi",
    "ppi",
    "nfp",
    "pce",
    "non-farm",
    "nonfarm",
    "lagarde",
    "yellen",
)
"""Economic actors and releases. The context carries data about none of them."""

_PAST_OCCURRENCE_MARKERS: tuple[str, ...] = (
    "vua",
    "da cong bo",
    "da phat bieu",
    "vua cong bo",
    "vua phat bieu",
    "just released",
    "just announced",
    "just said",
    "announced",
    "released",
    "reported",
)
"""Markers that turn a mention into a claim that something *happened*.

The distinction matters. "Tin PCE tối nay có thể tạo biến động" is a
forward-looking remark an analyst may legitimately make; "PCE vừa công bố" is an
assertion about the world that this pipeline has no data to support.
"""

NEWS_PROXIMITY_CHARS = 50
"""How close an entity and a past-tense marker must be to count as one claim."""


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------

_TRACEBACK_MARKERS: tuple[str, ...] = (
    "traceback (most recent call last)",
    'file "<stdin>"',
    "modulenotfounderror",
    "typeerror:",
    "valueerror:",
    "keyerror:",
    "attributeerror:",
    "pydantic_core._pydantic_core",
)

_ALLOWED_CONTROL_CHARS = frozenset({"\n", "\t"})


@dataclass(frozen=True)
class TextMatch:
    """One pattern hit in the article."""

    label: str
    matched: str
    position: int


def fold(text: str) -> str:
    """Strip diacritics and case, so Vietnamese matches however it is typed.

    **Character-preserving**, and callers depend on it: an offset in the folded
    string is the same offset in the original, which is what lets a match found
    here be reported - and later located - against the article a human reads.
    Decomposition followed by dropping combining marks restores one base
    character per precomposed one; the two explicit replacements handle ``đ``,
    which is its own letter rather than ``d`` plus a mark.
    """
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D").casefold()


_fold = fold
"""Internal alias, so the private call sites in this module read unchanged."""


def excerpt(text: str, start: int, end: int, window: int = 50) -> str:
    """A readable slice around a match, for the decision artifact."""
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right].replace("\n", " ").strip()
    return f"...{snippet}..." if left > 0 or right < len(text) else snippet


def find_instruction_text(article: str) -> list[TextMatch]:
    """Find prose that reads as an attempt to steer a model.

    Runs the English patterns against the raw text and the Vietnamese ones
    against a diacritic-folded copy. Positions are reported against the raw
    text in both cases, since the folding is character-preserving.
    """
    matches: list[TextMatch] = []
    seen: set[str] = set()

    for pattern, label in _INSTRUCTION_PATTERNS:
        for hit in re.finditer(pattern, article, re.IGNORECASE | re.MULTILINE):
            if label in seen:
                break
            seen.add(label)
            matches.append(
                TextMatch(
                    label=label,
                    matched=excerpt(article, hit.start(), hit.end()),
                    position=hit.start(),
                )
            )

    folded = _fold(article)
    for pattern, label in _INSTRUCTION_PATTERNS_VI:
        for hit in re.finditer(pattern, folded):
            if label in seen:
                break
            seen.add(label)
            matches.append(
                TextMatch(
                    label=label,
                    matched=excerpt(article, hit.start(), hit.end()),
                    position=hit.start(),
                )
            )

    return sorted(matches, key=lambda match: match.position)


def find_credentials(article: str) -> list[TextMatch]:
    """Find credential-shaped values.

    ``matched`` is always redacted - this function never returns a usable
    secret, so no caller can accidentally write one into an artifact or a log.
    """
    matches: list[TextMatch] = []
    claimed: list[tuple[int, int]] = []

    # Patterns run most-specific first, and one secret must produce one finding:
    # `sk-ant-api03-...` matches both the specific and the general Anthropic
    # shape, and reporting it twice would overstate how much leaked.
    for pattern, kind in _CREDENTIAL_PATTERNS:
        for hit in re.finditer(pattern, article):
            if any(start <= hit.start() < end for start, end in claimed):
                continue
            claimed.append((hit.start(), hit.end()))
            matches.append(
                TextMatch(label=kind, matched=redact(hit.group(0)), position=hit.start())
            )

    return sorted(matches, key=lambda match: match.position)


@dataclass(frozen=True)
class ClaimOccurrence:
    """One place an article asserts that a named economic event happened.

    Carries the whole span - from the entity through the end of the past-tense
    marker - rather than only where it starts. A caller deciding whether some
    piece of provenance covers this assertion has to know how far the assertion
    reaches: "Fed" alone is not the claim, "Fed vừa công bố" is, and provenance
    that covers only the first word covers nothing worth covering.
    """

    entity: str
    marker: str
    start: int
    end: int

    @property
    def label(self) -> str:
        return f"{self.entity} + '{self.marker}'"


def external_claim_occurrences(article: str) -> list[ClaimOccurrence]:
    """Every entity-plus-past-marker assertion in the article, in order.

    All of them, not one per entity: an article may support its first mention of
    the Fed and invent the second, and a caller that only ever saw the first
    would approve the invention.
    """
    folded = fold(article)
    found: list[ClaimOccurrence] = []

    for entity in _NEWS_ENTITIES:
        for hit in re.finditer(rf"\b{re.escape(entity)}\b", folded):
            window = folded[hit.end() : hit.end() + NEWS_PROXIMITY_CHARS]
            # First marker in table order, matching what this scanner has always
            # reported. Which marker is named changes the label a human reads,
            # so it is not a detail to quietly re-decide here.
            marker = next((m for m in _PAST_OCCURRENCE_MARKERS if m in window), None)
            if marker is None:
                continue
            offset = window.index(marker)
            found.append(
                ClaimOccurrence(
                    entity=entity,
                    marker=marker,
                    start=hit.start(),
                    end=hit.end() + offset + len(marker),
                )
            )

    return sorted(found, key=lambda o: (o.start, o.end))


def find_external_claims(article: str) -> list[TextMatch]:
    """Find claims that a named economic event has occurred.

    An entity alone is not enough - "vàng phản ứng với dữ liệu Mỹ" says nothing
    checkable. What is blocked is an entity paired with a past-tense marker
    close by, which asserts an event this pipeline has no source for.

    One finding per entity, at its first assertion: three sentences about the
    Fed are one problem to fix, not three. Callers that need every occurrence -
    because they are deciding coverage rather than reporting - use
    :func:`external_claim_occurrences` instead.
    """
    seen: set[str] = set()
    matches: list[TextMatch] = []

    for occurrence in external_claim_occurrences(article):
        if occurrence.entity in seen:
            continue
        seen.add(occurrence.entity)
        matches.append(
            TextMatch(
                label=occurrence.label,
                matched=excerpt(
                    article, occurrence.start, occurrence.start + len(occurrence.entity)
                ),
                position=occurrence.start,
            )
        )

    return sorted(matches, key=lambda match: match.position)


def find_control_characters(article: str) -> list[TextMatch]:
    """Find control characters that have no business in a published article."""
    matches: list[TextMatch] = []
    for index, char in enumerate(article):
        if char in _ALLOWED_CONTROL_CHARS:
            continue
        if unicodedata.category(char) == "Cc":
            matches.append(
                TextMatch(
                    label=f"U+{ord(char):04X}",
                    matched="<control character>",
                    position=index,
                )
            )
    return matches


def looks_like_json(article: str) -> bool:
    """Whether the article is a serialized object rather than prose."""
    stripped = article.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return False
    return bool(re.search(r'"\w+"\s*:', stripped))


def find_tracebacks(article: str) -> list[TextMatch]:
    """Find pasted error output."""
    folded = article.casefold()
    matches: list[TextMatch] = []
    for marker in _TRACEBACK_MARKERS:
        index = folded.find(marker)
        if index != -1:
            matches.append(
                TextMatch(
                    label=marker,
                    matched=excerpt(article, index, index + len(marker)),
                    position=index,
                )
            )
    return matches


def find_code_blocks(article: str) -> list[TextMatch]:
    """Find fenced code blocks.

    A market commentary has no reason to carry one, and a fence is how a dumped
    JSON payload or a leaked prompt usually arrives.
    """
    return [
        TextMatch(label="fenced code block", matched="```", position=hit.start())
        for hit in re.finditer(r"^```", article, re.MULTILINE)
    ]


__all__ = [
    "MAX_ARTICLE_CHARS",
    "MIN_ARTICLE_CHARS",
    "NEWS_PROXIMITY_CHARS",
    "TELEGRAM_MESSAGE_LIMIT",
    "TextMatch",
    "excerpt",
    "find_code_blocks",
    "find_control_characters",
    "find_credentials",
    "ClaimOccurrence",
    "external_claim_occurrences",
    "find_external_claims",
    "fold",
    "find_instruction_text",
    "find_tracebacks",
    "looks_like_json",
    "redact",
]
