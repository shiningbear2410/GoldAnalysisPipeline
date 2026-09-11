"""The TRADE_PLAN page, version two: readable prose around prices no model chose.

Round 6.7. Version one published two headings and a handful of numbers, and it
was honest but unreadable. This version adds a title, a market view, short notes
and a news section - and every one of those additions is **words**. Prices are
still copied from the deterministic selection, written by the same canonical
helper that minted the candidates' identities, and nothing a model returns can
put a digit on the page.

**The page is a pure function of three artifacts.** The selection (prices and
labels), the validated copy (the words) and the curated news (the headlines).
:func:`document_from_artifacts` reads exactly those three documents, and the
stage and the gate both call it: the stage over what it is about to write, the
gate over what is on disk. A page that differs from that rendering by one byte
is not the page the pipeline decided on.

**Prose is checked twice, by the same rules.** Once when the copywriter's answer
is parsed, and again here when the finished page is validated. The rule set is
defined once, in this module, so the two checks cannot drift apart.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from goldpipeline.schemas.news import CuratedNews

logger = logging.getLogger(__name__)

PRESENTATION_VERSION = "gold_trade_plan_presentation_v2"

VIETNAM = ZoneInfo("Asia/Ho_Chi_Minh")
"""The reader's calendar. The date on the page is the market's instant, here."""

TITLE_PREFIX = "🎯 KẾ HOẠCH VÀNG — "
SEO_HEADING = "🔴 SEO"
BAI_HEADING = "🟢 BAI"
HEADING_NOTE_SEPARATOR = " · "
ZONE_MARKER = "▸ "
ZONE_SEPARATOR = " – "
"""Space, en dash, space, between the two edges of a zone. Chosen once, pinned."""

NOTE_GAP = "  "
DEEP_LABEL = "Vùng sâu chờ sẵn: "
MAIN_ZONE_PHRASE = "vùng chủ đạo"
EMPTY_SIDE = "—"
NEWS_HEADING = "📰 Tin cần chú ý"
NEWS_BULLET = "- "
NO_NEWS_LINE = "- Chưa có tin đáng chú ý đủ độ tin cậy."
DISCLAIMER = (
    "🔴 Trên đây là nhận định cá nhân của mình, thông tin mang tính tham khảo. "
    "Không phải lời khuyên đầu tư, tư vấn tài chính"
)
CLOSING = "👉 Chúc mọi người một phiên giao dịch kỷ luật."

MAX_PLAN_CHARS = 3500
"""A hard ceiling that fails closed. Twelve hundred to two thousand six hundred is
the target, and nothing is ever truncated to meet either number."""

MAX_ZONES_PER_SIDE = 6
MAX_NEWS_ITEMS = 3
MAX_NEWS_DISPLAY_CHARS = 160

TIMEFRAME_TOKEN = re.compile(r"(?<![A-Za-z0-9])(?:M15|M1|M5|H1|H4)(?![A-Za-z0-9])")
"""The only tokens with a digit in them that prose may contain."""

FORBIDDEN_PROSE: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in (
        ("stop loss", r"\bSL\b"),
        ("take profit", r"\bTP\b"),
        ("risk-reward", r"R\s*:\s*R|\bRR\b"),
        ("stop loss", r"stop\s*-?\s*loss"),
        ("take profit", r"take\s*-?\s*profit"),
        ("stop loss", r"cắt\s+lỗ|dừng\s+lỗ"),
        ("take profit", r"chốt\s+lời"),
        ("indicator", r"\b(?:EMA|SMA|RSI|MACD)\b"),
        ("the retired label", r"vùng\s+chính"),
        ("a link", r"https?://|www\."),
        ("markup", r"`|[{}<>]"),
    )
)
"""Everything the copywriter may not say, by name.

Named so a rejection says *why*: "take profit" is a clearer failure than "the
text matched a pattern". Indicators are refused because the data carries none,
and a sentence about an RSI would be a sentence about nothing.
"""

_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

ZONE_LINE = re.compile(
    r"^▸ Vùng (?P<position>[1-6]) · (?P<lower>[0-9]+(?:\.[0-9]+)?) – "
    r"(?P<upper>[0-9]+(?:\.[0-9]+)?)(?:  \((?P<note>[^\n]+)\))?$"
)
DEEP_LINE = re.compile(
    r"^▸ Vùng sâu chờ sẵn: (?P<level>[0-9]+(?:\.[0-9]+)?)(?:  \((?P<note>[^\n]+)\))?$"
)


class PlanPresentationError(ValueError):
    """The page could not be presented honestly, so it was not presented."""


# --------------------------------------------------------------------------
# prose rules, shared with the copywriter's parser
# --------------------------------------------------------------------------


def normalise_prose(text: str) -> str:
    """NFC, so "vùng chủ đạo" is one spelling however the model encoded it."""
    return unicodedata.normalize("NFC", text).strip()


def has_number(text: str) -> bool:
    """Whether *text* contains a digit outside a timeframe name.

    ``isdigit`` rather than ``0-9``: a full-width or Arabic-Indic digit is still
    a number on the page.
    """
    return any(character.isdigit() for character in TIMEFRAME_TOKEN.sub(" ", text))


def prose_problem(text: str) -> str | None:
    """Why *text* may not be published as commentary, or ``None``."""
    if has_number(text):
        return "a number"
    if any(unicodedata.category(character).startswith("C") for character in text):
        return "a control character"
    for label, pattern in FORBIDDEN_PROSE:
        if pattern.search(text):
            return label
    return None


# --------------------------------------------------------------------------
# news
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanNewsItem:
    """One curated item, with the identity a model cites and the line a reader sees."""

    news_item_id: str
    channel: str
    message_id: int
    published_at: datetime
    display: str
    text: str
    matched_categories: tuple[str, ...]


def news_item_id(channel: str, message_id: int) -> str:
    """The item's own coordinates. Nothing is minted that the source did not name."""
    return f"{channel}/{message_id}"


def news_display(text: str) -> str:
    """The headline shown for an item: its first non-empty line, without links.

    Cut at a word boundary with an ellipsis when long, so a number inside a
    headline is shown whole or not at all.
    """
    for raw in text.splitlines():
        line = " ".join(_URL.sub(" ", raw).split())
        if line:
            break
    else:
        return ""

    if len(line) <= MAX_NEWS_DISPLAY_CHARS:
        return line
    head, _, _ = line[: MAX_NEWS_DISPLAY_CHARS - 1].rpartition(" ")
    return (head or line[: MAX_NEWS_DISPLAY_CHARS - 1]).rstrip(" ,.;:–-") + "…"


def plan_news_items(news: CuratedNews | None) -> tuple[PlanNewsItem, ...]:
    """The items a copywriter may cite, in curation order. Empty without news."""
    if news is None:
        return ()
    items: list[PlanNewsItem] = []
    seen: set[str] = set()
    for item in news.items:
        identity = news_item_id(item.channel, item.message_id)
        display = news_display(item.text)
        if not display or identity in seen:
            continue
        seen.add(identity)
        items.append(
            PlanNewsItem(
                news_item_id=identity,
                channel=item.channel,
                message_id=item.message_id,
                published_at=item.published_at,
                display=display,
                text=item.text,
                matched_categories=tuple(category.value for category in item.matched_categories),
            )
        )
    return tuple(items)


def news_document(
    news: CuratedNews | None, items: Iterable[PlanNewsItem], *, lookback_seconds: int
) -> dict[str, Any]:
    """What was offered to the copywriter, persisted so the gate can re-render."""
    return {
        "presentation_version": PRESENTATION_VERSION,
        "lookback_seconds": lookback_seconds,
        "collected": news is not None,
        "trust_level": "UNTRUSTED",
        "items": [
            {
                "news_item_id": item.news_item_id,
                "channel": item.channel,
                "message_id": item.message_id,
                "published_at": item.published_at.isoformat(),
                "display": item.display,
                "text": item.text,
                "matched_categories": list(item.matched_categories),
            }
            for item in items
        ],
    }


# --------------------------------------------------------------------------
# the document
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ZoneLine:
    lower: str
    upper: str
    main: bool
    note: str | None


@dataclass(frozen=True)
class PlanSide:
    heading_note: str | None
    zones: tuple[ZoneLine, ...]
    deep_level: str | None
    deep_note: str | None


@dataclass(frozen=True)
class PlanDocument:
    """Everything the page says, already decided. Prices are canonical strings."""

    date: str
    market_view: str
    seo: PlanSide
    bai: PlanSide
    news: tuple[str, ...]


def plan_date(observed_at: datetime) -> str:
    """``DD.MM.YYYY`` of the market's instant in Vietnam. Never the wall clock."""
    if observed_at.tzinfo is None:
        raise PlanPresentationError("the market instant has no time zone")
    return observed_at.astimezone(VIETNAM).strftime("%d.%m.%Y")


def _optional(value: object) -> str | None:
    if value is None:
        return None
    text = normalise_prose(str(value))
    return text or None


def selected_prices(selection: Mapping[str, Any]) -> frozenset[str]:
    """Every price the selection published, as canonical strings."""
    prices: set[str] = set()
    for key in ("seo_entries", "bai_entries"):
        for zone in selection.get(key, []):
            prices.update((str(zone["lower"]), str(zone["upper"])))
    for key in ("seo_reference", "bai_reference"):
        reference = selection.get(key)
        if reference:
            prices.add(str(reference["level"]))
    return frozenset(prices)


def document_from_artifacts(
    selection: Mapping[str, Any], copy: Mapping[str, Any], news: Mapping[str, Any]
) -> PlanDocument:
    """Assemble the page's content from the three persisted documents.

    The stage and the gate both call this, which is what makes "the page on
    disk is the page the pipeline decided" a byte comparison rather than a hope.

    Raises:
        PlanPresentationError: The copy cites a news item that was not offered.
    """
    zone_notes: Mapping[str, Any] = copy.get("zone_notes") or {}
    reference_notes: Mapping[str, Any] = copy.get("reference_notes") or {}

    def side(entries_key: str, reference_key: str, heading_key: str) -> PlanSide:
        zones = tuple(
            ZoneLine(
                lower=str(zone["lower"]),
                upper=str(zone["upper"]),
                main=zone.get("label") == "VUNG_CHINH",
                note=_optional(zone_notes.get(zone["candidate_id"])),
            )
            for zone in selection.get(entries_key, [])
        )
        if not zones:
            # A farther level with nothing to be farther than is not a plan, and
            # a note about an empty side has nothing to describe.
            return PlanSide(heading_note=None, zones=(), deep_level=None, deep_note=None)
        reference = selection.get(reference_key)
        return PlanSide(
            heading_note=_optional(copy.get(heading_key)),
            zones=zones,
            deep_level=None if not reference else str(reference["level"]),
            deep_note=(
                None if not reference else _optional(reference_notes.get(reference["candidate_id"]))
            ),
        )

    offered = {str(item["news_item_id"]): str(item["display"]) for item in news.get("items", [])}
    cited: list[str] = []
    for identity in copy.get("news_item_ids") or []:
        if identity not in offered:
            raise PlanPresentationError(f"the copy cites news item {identity!r}, never offered")
        cited.append(offered[identity])

    observed = datetime.fromisoformat(str(selection["observed_at"]))
    return PlanDocument(
        date=plan_date(observed),
        market_view=normalise_prose(str(copy.get("market_view", ""))),
        seo=side("seo_entries", "seo_reference", "seo_heading_note"),
        bai=side("bai_entries", "bai_reference", "bai_heading_note"),
        news=tuple(cited),
    )


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def _parenthetical(zone: ZoneLine) -> str | None:
    """The zone's bracket: the main label, the note, or both - never the label twice."""
    if not zone.main:
        return zone.note
    if zone.note is None:
        return MAIN_ZONE_PHRASE
    if MAIN_ZONE_PHRASE in zone.note.casefold():
        return zone.note
    return f"{MAIN_ZONE_PHRASE}, {zone.note}"


def _with_note(body: str, note: str | None) -> str:
    return f"{body}{NOTE_GAP}({note})" if note else body


def _side_lines(heading: str, side: PlanSide) -> list[str]:
    if not side.zones:
        return [heading, EMPTY_SIDE]
    title = (
        f"{heading}{HEADING_NOTE_SEPARATOR}{side.heading_note}" if side.heading_note else heading
    )
    lines = [title]
    for position, zone in enumerate(side.zones, start=1):
        body = f"{ZONE_MARKER}Vùng {position} · {zone.lower}{ZONE_SEPARATOR}{zone.upper}"
        lines.append(_with_note(body, _parenthetical(zone)))
    if side.deep_level is not None:
        lines.append(_with_note(f"{ZONE_MARKER}{DEEP_LABEL}{side.deep_level}", side.deep_note))
    return lines


def render_plan(document: PlanDocument) -> str:
    """The published text, in full. Plain text: no markup, no code fence, no table.

    Raises:
        PlanPresentationError: The page is over the hard cap. Nothing is cut.
    """
    news = [f"{NEWS_BULLET}{display}" for display in document.news] or [NO_NEWS_LINE]
    lines = [
        f"{TITLE_PREFIX}{document.date}",
        document.market_view,
        "",
        *_side_lines(SEO_HEADING, document.seo),
        "",
        *_side_lines(BAI_HEADING, document.bai),
        "",
        NEWS_HEADING,
        *news,
        "",
        DISCLAIMER,
        CLOSING,
    ]
    text = "\n".join(lines)
    if len(text) > MAX_PLAN_CHARS:
        raise PlanPresentationError(
            f"the plan is {len(text)} characters, over the {MAX_PLAN_CHARS} cap; "
            "refusing to cut a line"
        )
    logger.info(
        "trade_plan.present chars=%d seo=%d bai=%d news=%d",
        len(text),
        len(document.seo.zones),
        len(document.bai.zones),
        len(document.news),
    )
    return text


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def _prose_fragments(document: PlanDocument) -> list[tuple[str, str]]:
    fragments = [("market view", document.market_view)]
    for name, side in (("SEO", document.seo), ("BAI", document.bai)):
        if side.heading_note:
            fragments.append((f"{name} heading note", side.heading_note))
        fragments.extend((f"{name} zone note", zone.note) for zone in side.zones if zone.note)
        if side.deep_note:
            fragments.append((f"{name} deep note", side.deep_note))
    return fragments


def _side_block(lines: list[str], heading: str) -> list[str]:
    start = next(index for index, line in enumerate(lines) if line.startswith(heading))
    block: list[str] = []
    for line in lines[start:]:
        if not line:
            break
        block.append(line)
    return block


def _check_side(block: list[str], heading: str, allowed: frozenset[str], shown: set[str]) -> None:
    body = block[1:]
    if body == [EMPTY_SIDE]:
        return
    zones = [ZONE_LINE.fullmatch(line) for line in body]
    deep = [DEEP_LINE.fullmatch(line) for line in body]

    positions: list[int] = []
    prices: tuple[str, ...]
    for line, zone, level in zip(body, zones, deep, strict=True):
        if zone is None and level is None:
            raise PlanPresentationError(f"{heading}: unexpected line {line!r}")
        if zone is not None:
            if level is not None or positions and deep[len(positions)] is not None:
                raise PlanPresentationError(f"{heading}: a zone follows the deep level")
            positions.append(int(zone["position"]))
            prices = (zone["lower"], zone["upper"])
        else:
            assert level is not None
            prices = (level["level"],)
        for price in prices:
            if price not in allowed:
                raise PlanPresentationError(f"{heading}: {price} is not a selected price")
            shown.add(price)

    if not positions:
        raise PlanPresentationError(f"{heading}: a deep level with no entry zone")
    if positions != list(range(1, len(positions) + 1)) or len(positions) > MAX_ZONES_PER_SIDE:
        raise PlanPresentationError(f"{heading}: zones are numbered {positions}")
    if sum(1 for match in deep if match is not None) > 1:
        raise PlanPresentationError(f"{heading}: more than one deep level")
    if deep[-1] is None and any(match is not None for match in deep):
        raise PlanPresentationError(f"{heading}: the deep level must come last")
    if sum(line.casefold().count(MAIN_ZONE_PHRASE) for line in body) > 1:
        raise PlanPresentationError(f"{heading}: more than one {MAIN_ZONE_PHRASE}")


def validate_plan(
    text: str,
    document: PlanDocument,
    *,
    allowed_prices: frozenset[str],
    candidate_ids: frozenset[str],
    news_displays: frozenset[str],
) -> None:
    """Prove the page says exactly what was decided, and nothing a model invented.

    Independent of the renderer except for one comparison: every structural
    rule below is re-derived from the text, so a renderer bug that produced a
    self-consistent but wrong page would still be caught here.

    Raises:
        PlanPresentationError: The first rule the text breaks.
    """
    if not text.strip():
        raise PlanPresentationError("the plan is empty")
    if len(text) > MAX_PLAN_CHARS:
        raise PlanPresentationError(f"the plan is {len(text)} characters")
    if any(unicodedata.category(character) == "Cc" and character != "\n" for character in text):
        raise PlanPresentationError("the plan contains control characters")
    if "`" in text or text.lstrip().startswith(("{", "[")):
        raise PlanPresentationError("the plan looks like code or JSON")
    if text != render_plan(document):
        raise PlanPresentationError("the plan is not the rendering of its document")

    lines = text.split("\n")
    if lines[0] != f"{TITLE_PREFIX}{document.date}" or text.count(TITLE_PREFIX) != 1:
        raise PlanPresentationError("the title must appear once, first, with the market's date")

    for heading in (SEO_HEADING, BAI_HEADING):
        if sum(1 for line in lines if line.startswith(heading)) != 1:
            raise PlanPresentationError(f"expected exactly one {heading} heading")
    seo_at = next(i for i, line in enumerate(lines) if line.startswith(SEO_HEADING))
    bai_at = next(i for i, line in enumerate(lines) if line.startswith(BAI_HEADING))
    if seo_at > bai_at:
        raise PlanPresentationError("SEO must precede BAI")

    shown: set[str] = set()
    for heading in (SEO_HEADING, BAI_HEADING):
        _check_side(_side_block(lines, heading), heading, allowed_prices, shown)
    if shown != set(allowed_prices):
        raise PlanPresentationError(
            f"selected prices missing from the plan: {sorted(set(allowed_prices) - shown)}"
        )

    if lines.count(NEWS_HEADING) != 1:
        raise PlanPresentationError("expected exactly one news heading")
    bullets = _side_block(lines, NEWS_HEADING)[1:]
    if bullets != [NO_NEWS_LINE]:
        if len(bullets) > MAX_NEWS_ITEMS:
            raise PlanPresentationError(f"{len(bullets)} news items, over {MAX_NEWS_ITEMS}")
        for bullet in bullets:
            if (
                not bullet.startswith(NEWS_BULLET)
                or bullet[len(NEWS_BULLET) :] not in news_displays
            ):
                raise PlanPresentationError(f"news line {bullet!r} is not an offered item")

    if text.count(DISCLAIMER) != 1 or lines.count(DISCLAIMER) != 1:
        raise PlanPresentationError("the disclaimer must appear exactly once")
    if text.count(CLOSING) != 1 or lines[-1] != CLOSING:
        raise PlanPresentationError("the closing line must appear exactly once, last")

    for name, fragment in _prose_fragments(document):
        problem = prose_problem(fragment)
        if problem is not None:
            raise PlanPresentationError(f"the {name} contains {problem}")

    folded = text.casefold()
    for identity in candidate_ids:
        if identity.casefold() in folded:
            raise PlanPresentationError("a candidate id leaked into the plan")


__all__ = [
    "BAI_HEADING",
    "CLOSING",
    "DEEP_LABEL",
    "DISCLAIMER",
    "EMPTY_SIDE",
    "MAIN_ZONE_PHRASE",
    "MAX_NEWS_ITEMS",
    "MAX_PLAN_CHARS",
    "MAX_ZONES_PER_SIDE",
    "NEWS_HEADING",
    "NO_NEWS_LINE",
    "PRESENTATION_VERSION",
    "SEO_HEADING",
    "TITLE_PREFIX",
    "VIETNAM",
    "ZONE_SEPARATOR",
    "PlanDocument",
    "PlanNewsItem",
    "PlanPresentationError",
    "PlanSide",
    "ZoneLine",
    "document_from_artifacts",
    "has_number",
    "news_display",
    "news_document",
    "news_item_id",
    "normalise_prose",
    "plan_date",
    "plan_news_items",
    "prose_problem",
    "render_plan",
    "selected_prices",
    "validate_plan",
]
