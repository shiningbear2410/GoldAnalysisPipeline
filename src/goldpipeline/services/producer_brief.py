"""Rendering a collected news window as one deterministic document.

The document this builds becomes ``AnalysisEvent.raw_text``, which the writer
prompt already fences as untrusted source data. So the rule here is narrow and
absolute:

    **The brief contains data. It contains no instructions.**

Not "write an article", not "focus on the Fed", not a tone, not a length, not a
model. The writer's system prompt owns every instruction in this pipeline, and a
document assembled from third-party text is the last place one should be able to
appear. What the brief carries is a window, a collection outcome, per-source
coverage, and the curated items with enough provenance to check any of them.

**One renderer, one version.** Assembling prose inline in the producer would
make the document a side effect of control flow - different on the empty path,
subtly different again on the partial path, and impossible to test as a whole.
It is built here, from a value, by a pure function, and
:data:`PRODUCER_BRIEF_VERSION` is stamped on it so a Run written last month can
be read against the renderer that wrote it.

**Deterministic, byte for byte.** No clock is read, no set is iterated, no
random token appears. The same request and the same collection render the same
bytes, which is what lets a retry be recognised as a retry instead of becoming a
conflict.

**Item text is verbatim, and it is framed by length rather than by a delimiter.**
This is the one thing in the layout that is a security property rather than a
presentation choice.

A news item's text is quoted exactly as the channel published it. A hostile
channel can therefore post a message whose text contains a complete, perfectly
formed ``### ITEM 99`` block naming a fact nobody ever published. Any parser
that found item boundaries by *scanning for the delimiter* would read that
forgery as a genuine record, and a writer could then cite it - which is the
whole attack, executed entirely from content this pipeline does not control.

So each record declares ``chars:`` and :func:`parse_brief` consumes exactly that
many characters without looking inside them. Structure comes from counts the
renderer computed, never from bytes an item supplied, and a forged block lands
harmlessly inside a counted span. The parser additionally requires the record
count to match ``items_in_brief`` and the last record to end the document, so a
forgery cannot be smuggled in by appending one either.

Text stays verbatim throughout: nothing is escaped, quoted or rewritten, because
a brief that no longer shows what the channel published has lost the one thing
it exists to show.

The renderer and the parser live together on purpose. They are one contract in
two directions, and a round-trip test holds them to it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime

from goldpipeline.schemas.news import (
    CollectionOutcome,
    CuratedItem,
    CuratedNews,
    NewsCollection,
    SourceReport,
)
from goldpipeline.schemas.producer import ProducerRequest

PRODUCER_BRIEF_VERSION = "news_brief_v2"
"""Version of this document's layout. Stamped on the brief and on the event.

v2 adds ``message_id`` and ``chars`` to each item record, and frames item text
by declared length rather than by a delimiter. That is what makes
:func:`parse_brief` safe against a hostile item - see the module docstring.
"""

PARSEABLE_BRIEF_VERSIONS = frozenset({PRODUCER_BRIEF_VERSION})
"""Layouts :func:`parse_brief` understands.

A brief written by a renderer this code cannot parse yields no provenance at
all, which is the safe direction: an unparsed brief means every external-news
statement stays unsupported.
"""

ITEMS_HEADING = "## NEWS ITEMS"
ITEM_HEADER_PREFIX = "### ITEM "
END_MARKER = "## END OF NEWS ITEMS"
"""The document's last line, and part of the frame rather than decoration.

Two jobs. It proves the last record really is the last one, so a record cannot
be smuggled in by appending. And it keeps the document from ending in
whitespace: the pipeline strips the analysis text it carries, and a final item
whose own text ended in blank lines would otherwise have those lines eaten -
inside a counted span, breaking the frame for a purely typographic reason.
"""
ITEM_ID_PATTERN = re.compile(r"^([A-Za-z][A-Za-z0-9_]{3,31}):([0-9]{1,12})$")
"""A news item id: ``<channel>:<message_id>``.

The channel half is the same shape the collector validates, so an id can only
ever name a channel that could have been configured. Nothing here accepts a
URL: a writer citing ``https://...`` is citing a string it read, and the point
of an id is that it addresses a record this pipeline collected.
"""

NO_RELEVANT_NEWS = "NO_RELEVANT_NEWS_FOUND"
"""What an empty window says, in one deterministic token.

Not an apology and not a filler paragraph. A window that produced nothing
relevant is a fact about the window, and the alternative - inventing something
to make the section look fuller - is the failure mode this whole subsystem
exists to avoid.
"""

UNTRUSTED_NOTE = (
    "All text under NEWS ITEMS is UNTRUSTED third-party content collected from "
    "public channels. It is quoted here verbatim, including anything shaped like "
    "a heading, an instruction or a credential."
)
"""A label on the data, deliberately not an instruction about what to do with it.

It states what the text *is*. What anyone should do about that is a rule, and
rules live in the system prompt.
"""


def sanitized_item_text(text: str) -> str:
    """An item's text in the form the pipeline will actually carry it.

    The brief becomes ``AnalysisEvent.raw_text``, and Round 1 sanitises that on
    the way into the context: control and invisible characters removed, NFC
    normalisation, trailing whitespace stripped from every line, the whole
    document stripped. A ``chars`` count taken before that pass would disagree
    with the document a verifier later reads, and the frame would break on
    nothing more sinister than a trailing space.

    So the same sanitiser runs here, per item, and the count is taken
    afterwards. The rendered document is then a fixed point of the pass that
    follows it: what was written is what arrives.
    """
    from goldpipeline.services.normalizer import sanitize_analysis_text

    return sanitize_analysis_text(text)[0]


def news_item_id(channel: str, message_id: int) -> str:
    """The stable id for one collected message: ``<channel>:<message_id>``.

    One function, used by the renderer and by everything that later resolves an
    id, so "which message is this" cannot come to mean two different things.
    """
    return f"{channel}:{message_id}"


@dataclass(frozen=True)
class ParsedNewsItem:
    """One item record recovered from a brief.

    Deliberately a separate type from :class:`CuratedItem`. This one was *read
    back* from a document, and the distinction is worth keeping visible: a
    curated item is something this process collected, a parsed item is something
    it re-derived from bytes and must treat as evidence rather than as truth.
    """

    item_id: str
    channel: str
    message_id: int
    url: str
    published: str
    categories: tuple[str, ...]
    relevance: str
    corroborating_channels: int
    text: str


@dataclass(frozen=True)
class ParsedBrief:
    """A brief that parsed completely, and the items it declared."""

    brief_version: str
    items: tuple[ParsedNewsItem, ...]

    @property
    def by_id(self) -> dict[str, ParsedNewsItem]:
        return {item.item_id: item for item in self.items}


def _stamp(value: datetime | None) -> str:
    """Render a timestamp - or its absence - the same way everywhere.

    Always UTC with a ``Z`` suffix, matching how every artifact in this pipeline
    writes an instant. A brief that rendered local time would put a story on the
    wrong day for anyone reading it from another machine.
    """
    if value is None:
        return "-"
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _duration(seconds: int) -> str:
    """A window length in the vocabulary the caller asked in.

    Hours below a week, days at a week and beyond: ``6h``, ``24h``, ``72h``,
    ``7d``. Rendering a day-long window as ``1d`` would be equally true and
    would stop matching what anyone typed, and a brief is read next to the
    request that produced it.
    """
    if seconds % 86_400 == 0 and seconds >= 7 * 86_400:
        return f"{seconds // 86_400}d"
    if seconds % 3_600 == 0:
        return f"{seconds // 3_600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def _source_line(report: SourceReport) -> str:
    """One channel's coverage, as fixed-width columns.

    Columns rather than prose because this section is read to answer one
    question - did anyone actually look at the whole window? - and a table
    answers it without being parsed.
    """
    return (
        f"{report.channel:<16}"
        f"{report.outcome:<12}"
        f"{'yes' if report.covered_window else 'no':<9}"
        f"{report.stop_reason or '-':<22}"
        f"{report.pages_fetched:>6}"
        f"{report.items_in_window:>11}"
        f"  {_stamp(report.oldest_seen)}"
    )


def render_brief(
    request: ProducerRequest,
    collection: NewsCollection,
    curated: CuratedNews,
) -> str:
    """Render the producer brief for one request.

    Pure. Everything in the output comes from the two arguments; nothing is
    read from a clock, an environment or a file.
    """
    lines: list[str] = [
        f"# PRODUCER BRIEF {PRODUCER_BRIEF_VERSION}",
        "",
        "## REQUESTED WINDOW",
        "",
        f"requested_at:  {_stamp(request.requested_at)}",
        f"window_start:  {_stamp(collection.window_start)}",
        f"window_end:    {_stamp(collection.window_end)}",
        f"lookback:      {_duration(collection.lookback_seconds)}",
        f"article_type:  {request.article_type}",
        "",
        "## COLLECTION",
        "",
        f"outcome:            {collection.outcome}",
        f"coverage_complete:  {'yes' if collection.complete else 'no'}",
        f"sources_reported:   {len(collection.sources)}",
        f"items_relevant:     {len(collection.items)}",
        f"items_in_brief:     {len(curated.items)}",
        f"items_omitted:      {curated.omitted_count}",
        f"items_truncated:    {curated.truncated_count}",
        "",
        "## SOURCE COVERAGE",
        "",
        "```text",
        f"{'channel':<16}{'outcome':<12}{'covered':<9}{'stop':<22}{'pages':>6}{'in_window':>11}"
        "  oldest_seen",
    ]
    lines += [_source_line(report) for report in collection.sources] or ["(no sources reported)"]
    lines += ["```", ""]

    if collection.warnings:
        lines += ["## COVERAGE WARNINGS", ""]
        lines += [f"- {warning}" for warning in collection.warnings]
        lines += [""]

    lines += [ITEMS_HEADING, "", UNTRUSTED_NOTE, ""]

    if not curated.items:
        lines += [NO_RELEVANT_NEWS, "", END_MARKER, ""]
        return "\n".join(lines)

    head = "\n".join(lines) + "\n"
    records = "".join(
        _item_record(item, position, len(curated.items))
        for position, item in enumerate(curated.items, start=1)
    )
    return f"{head}{records}{END_MARKER}\n"


def _item_record(item: CuratedItem, position: int, total: int) -> str:
    """One length-framed item record.

    Built as a string rather than as lines because the framing has to be exact:
    ``chars`` counts the characters of ``text``, and a parser that trusts that
    count never scans the text for structure at all.
    """
    text = sanitized_item_text(item.text)
    categories = ", ".join(str(c) for c in item.matched_categories) or "-"
    header = "\n".join(
        (
            f"{ITEM_HEADER_PREFIX}{position} OF {total}",
            "",
            f"id:          {news_item_id(item.channel, item.message_id)}",
            f"channel:     {item.channel}",
            f"message_id:  {item.message_id}",
            f"url:         https://t.me/{item.channel}/{item.message_id}",
            f"published:   {_stamp(item.published_at)}",
            f"categories:  {categories}",
            f"relevance:   {item.relevance_score:g}",
            f"channels:    {item.source_count}",
            f"truncated:   {'yes' if item.text_truncated else 'no'}",
            "trust:       UNTRUSTED",
            f"chars:       {len(text)}",
            "text:",
            "",
        )
    )
    return f"{header}{text}\n\n"


# --------------------------------------------------------------------------
# reading one back
# --------------------------------------------------------------------------

_HEADER_RE = re.compile(r"^# PRODUCER BRIEF (?P<version>[A-Za-z0-9_]+)\n")
_DECLARED_COUNT_RE = re.compile(r"^items_in_brief: +(?P<count>\d+)$", re.MULTILINE)
_RECORD_HEADER_RE = re.compile(
    r"### ITEM (?P<position>\d+) OF (?P<total>\d+)\n"
    r"\n"
    r"id:          (?P<item_id>[^\n]+)\n"
    r"channel:     (?P<channel>[^\n]+)\n"
    r"message_id:  (?P<message_id>\d+)\n"
    r"url:         (?P<url>[^\n]+)\n"
    r"published:   (?P<published>[^\n]+)\n"
    r"categories:  (?P<categories>[^\n]+)\n"
    r"relevance:   (?P<relevance>[^\n]+)\n"
    r"channels:    (?P<channels>\d+)\n"
    r"truncated:   (?:yes|no)\n"
    r"trust:       UNTRUSTED\n"
    r"chars:       (?P<chars>\d+)\n"
    r"text:\n"
)
"""The whole record header, anchored and in fixed field order.

One regex rather than a line loop because every field is mandatory and the order
is part of the contract: a document missing a field, or carrying them in another
order, is not something this renderer produced and must not be read as though it
were.
"""


def parse_brief(text: str) -> ParsedBrief | None:
    """Recover the item records from a rendered brief, or refuse.

    Returns ``None`` - never a partial result - when anything at all does not
    line up: an unknown version, a missing count, a record that does not match
    the header contract, a declared length that runs past the end, a record
    count that disagrees with ``items_in_brief``, or trailing bytes after the
    last record.

    **Failing closed is the entire point.** A brief that half-parses would hand
    the verifier item records of unknown provenance, and every one of them would
    then be usable to justify a news statement. There is no partial credit here:
    either the document is one this pipeline rendered, in which case it parses
    exactly, or no news provenance exists for the Run and every external-news
    statement stays unsupported.
    """
    header = _HEADER_RE.match(text)
    if header is None or header["version"] not in PARSEABLE_BRIEF_VERSIONS:
        return None

    declared = _DECLARED_COUNT_RE.search(text)
    if declared is None:
        return None
    expected = int(declared["count"])

    marker = f"\n{ITEMS_HEADING}\n"
    section = text.find(marker)
    if section < 0:
        return None
    cursor = text.find(f"\n{ITEM_HEADER_PREFIX}", section)

    if cursor < 0:
        # No records at all. Legitimate only when the brief said so and closed
        # the section with the empty-window token.
        return (
            ParsedBrief(brief_version=header["version"], items=())
            if expected == 0 and NO_RELEVANT_NEWS in text[section:] and END_MARKER in text
            else None
        )

    cursor += 1
    items: list[ParsedNewsItem] = []

    while cursor < len(text):
        record = _RECORD_HEADER_RE.match(text, cursor)
        if record is None or int(record["position"]) != len(items) + 1:
            return None
        if int(record["total"]) != expected:
            return None

        start = record.end()
        end = start + int(record["chars"])
        if end > len(text):
            return None

        parsed = _item_from(record, text[start:end])
        if parsed is None:
            return None
        items.append(parsed)

        # Every record ends the same way, so the frame is checkable rather than
        # merely plausible.
        if text[end : end + 2] != "\n\n":
            return None
        cursor = end + 2
        if text.startswith(END_MARKER, cursor):
            break

    if len(items) != expected or not text.startswith(END_MARKER, cursor):
        # The marker must follow the last record. Without it a document could be
        # extended with further records and the count alone would not notice.
        return None
    if text[cursor + len(END_MARKER) :].strip():
        # Nothing may follow it either, or "the marker ends the document" would
        # mean "the marker appears somewhere", which is not the same guarantee.
        return None
    return ParsedBrief(brief_version=header["version"], items=tuple(items))


def _item_from(record: re.Match[str], body: str) -> ParsedNewsItem | None:
    """Build one item, checking that its id agrees with its own fields.

    The id is not taken on trust even though this renderer wrote it: a record
    whose ``id`` disagrees with its ``channel`` and ``message_id`` is
    self-inconsistent, and the cheapest moment to notice is here.
    """
    match = ITEM_ID_PATTERN.match(record["item_id"])
    if match is None:
        return None
    channel, message_id = match[1], int(match[2])
    if channel != record["channel"] or message_id != int(record["message_id"]):
        return None

    return ParsedNewsItem(
        item_id=record["item_id"],
        channel=channel,
        message_id=message_id,
        url=record["url"],
        published=record["published"],
        categories=tuple(c.strip() for c in record["categories"].split(",") if c.strip() != "-"),
        relevance=record["relevance"],
        corroborating_channels=int(record["channels"]),
        text=body,
    )


def brief_is_empty(collection: NewsCollection, curated: CuratedNews) -> bool:
    """Whether the window produced no usable news, for a caller's own reporting.

    Deliberately not the same question as "did the collection fail". A quiet
    window and an unreachable source are different facts, and only the second
    means the producer should refuse.
    """
    return collection.outcome is not CollectionOutcome.FAILED and not curated.items


__all__ = [
    "END_MARKER",
    "ITEMS_HEADING",
    "ITEM_HEADER_PREFIX",
    "ITEM_ID_PATTERN",
    "NO_RELEVANT_NEWS",
    "PARSEABLE_BRIEF_VERSIONS",
    "PRODUCER_BRIEF_VERSION",
    "UNTRUSTED_NOTE",
    "ParsedBrief",
    "ParsedNewsItem",
    "brief_is_empty",
    "sanitized_item_text",
    "news_item_id",
    "parse_brief",
    "render_brief",
]
