"""The digest lines a model is never allowed to compose.

Round 6.5a. Three renderers - the title, the window line, the price-reaction
block - and each turns typed facts into the exact characters that will be
published.

**Why these are rendered rather than described.** A model handed a window and
asked for a title writes ``TIN VÀNG hôm nay`` on one day and ``TIN VÀNG
03-04/09`` on the next, and neither is checkable against anything. A model
handed a net change writes the range instead roughly one time in ten, because
both are numbers of the same size in the same units. Rendering here means the
later writer *copies* a line it cannot get wrong, and the round that activates
the digest can enforce that copy byte for byte.

**Presentation only.** Nothing here changes a Decimal, and nothing here decides
what a price is. Domain values keep every digit the provider gave them; these
functions choose how many of those digits a reader sees, which is a different
question with a different answer.

**No causality, ever.** The price block reports what price did. It has no
access to the news items and no vocabulary for connecting them - "vàng tăng do"
is not a sentence any function here can produce, because the observation and the
explanation are owned by different stages on purpose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

from goldpipeline.schemas.common import resolve_timezone
from goldpipeline.schemas.digest import DigestWindow, MarketActivity, PriceReaction
from goldpipeline.schemas.news_digest import IMPACT_LABELS, DigestItem, DigestSourceItem
from goldpipeline.services.market_facts import ARTICLE_TIMEZONE

DIGEST_TITLE_PREFIX = "📰 TIN VÀNG"
WINDOW_LINE_PREFIX = "🕐"
WINDOW_LINE_SUFFIX = "(giờ VN)"
PRICE_REACTION_HEADING = "📈 Giá phản ứng"

MOVE_DECIMALS = 2
"""How many decimals a *move* may show before trailing zeros are dropped.

A move is a distance, not a price, and the two want different treatment. A
price is padded rather than rounded, because the gate matches article prices
against the context exactly and a rounded one appears nowhere in the data. A
move appears in prose - "tăng khoảng 105 USD" - where ``105.00000000`` is
noise. Two places is enough to keep a genuinely small move visible: a move of a
quarter of a dollar still reads as 0.25 rather than collapsing to nothing.
"""

PERCENT_DECIMALS = 2

CURRENCY_LABEL = "USD"
RANGE_LABEL = "biên độ"
"""The one word for high-low, so nothing else in the product borrows it."""


def digest_title(window: DigestWindow) -> str:
    """The digest headline, dated by the reader's own calendar.

    Three shapes, chosen by what actually differs::

        📰 TIN VÀNG 04.09.2026              one Vietnam calendar day
        📰 TIN VÀNG 03.09 → 04.09.2026      two days, same year
        📰 TIN VÀNG 31.12.2025 → 01.01.2026 across a year boundary

    The year is written once when both ends share it and twice when they do not,
    because "31.12 → 01.01" is ambiguous in exactly the case where being wrong
    matters most. Vietnam local decides the calendar day: a window ending at
    00:30 UTC on the 4th ended at 07:30 on the 4th for the reader, and dating it
    the 3rd would put yesterday on this morning's digest.
    """
    start = _local(window.start)
    end = _local(window.end)

    if (start.year, start.month, start.day) == (end.year, end.month, end.day):
        return f"{DIGEST_TITLE_PREFIX} {end:%d.%m.%Y}"
    if start.year == end.year:
        return f"{DIGEST_TITLE_PREFIX} {start:%d.%m} → {end:%d.%m.%Y}"
    return f"{DIGEST_TITLE_PREFIX} {start:%d.%m.%Y} → {end:%d.%m.%Y}"


def digest_window_line(window: DigestWindow) -> str:
    """The exact span, to the minute, in Vietnam time.

    ``🕐 03/09 14:07 → 04/09 14:07 (giờ VN)``

    Minute precision, and no rounding: a window that ran to 14:07 did not run to
    14:00, and a reader comparing the digest against a news item's timestamp
    should find the two agree. The model may later write "24 giờ qua" in its
    prose - that is a description - but this line is the record, and it is
    copied rather than rewritten.
    """
    start = _local(window.start)
    end = _local(window.end)
    return f"{WINDOW_LINE_PREFIX} {start:%d/%m %H:%M} → {end:%d/%m %H:%M} {WINDOW_LINE_SUFFIX}"


def format_move(value: Decimal) -> str:
    """Render a price *distance* for prose: ``105``, ``105.3``, ``0.25``.

    Rounded to :data:`MOVE_DECIMALS` and then stripped of trailing zeros, so a
    whole-number move reads as a whole number. The sign is never included -
    direction is carried by the verb, and "tăng -105" is not a sentence.
    """
    quantized = abs(value).quantize(Decimal(1).scaleb(-MOVE_DECIMALS), rounding=ROUND_HALF_UP)
    normalized = quantized.normalize()
    if normalized == normalized.to_integral_value():
        normalized = normalized.to_integral_value()
    return f"{normalized:f}"


def format_percent(value: Decimal) -> str:
    """Render a percentage with an explicit sign: ``+0.98%``, ``-1.20%``."""
    quantized = value.quantize(Decimal(1).scaleb(-PERCENT_DECIMALS), rounding=ROUND_HALF_UP)
    sign = "+" if quantized > 0 else ""
    return f"{sign}{quantized:f}%"


def render_price_reaction(reaction: PriceReaction) -> str:
    """The 📈 Giá phản ứng block, exactly as it would be published.

    One sentence for the move, one clause for the range, and nothing else. Every
    branch below reports an observation; none of them offers a reason, and none
    names the provider that supplied the candles.
    """
    body = _reaction_body(reaction)
    return f"{PRICE_REACTION_HEADING}\n{body}"


def _reaction_body(reaction: PriceReaction) -> str:
    """The sentence under the heading, chosen by what was actually measured."""
    if reaction.market_activity is MarketActivity.INSUFFICIENT_HISTORY:
        return "Chưa đủ dữ liệu giá để đối chiếu trong khung thời gian này."

    if reaction.market_activity is MarketActivity.NO_MARKET_ACTIVITY:
        return "Không có phiên giao dịch nào trong khung thời gian này."

    if reaction.market_activity is MarketActivity.NO_NEW_CLOSED_BAR:
        return "Chưa có nến nào đóng trong khung thời gian này."

    assert reaction.net_change is not None  # noqa: S101 - NORMAL implies a measured move
    move = format_move(reaction.net_change)
    percent = (
        f" ({format_percent(reaction.percent_change)})"
        if reaction.percent_change is not None
        else ""
    )

    if reaction.net_change > 0:
        sentence = f"Vàng tăng khoảng {move} {CURRENCY_LABEL}{percent} từ đầu đến cuối cửa sổ."
    elif reaction.net_change < 0:
        sentence = f"Vàng giảm khoảng {move} {CURRENCY_LABEL}{percent} từ đầu đến cuối cửa sổ."
    else:
        # Zero is a real observation here - both boundaries were measured and
        # they matched - and it is worded as such rather than as an absence.
        sentence = "Vàng gần như không đổi từ đầu đến cuối cửa sổ."

    if reaction.price_range is not None:
        span = format_move(reaction.price_range)
        sentence += f" {RANGE_LABEL.capitalize()} khoảng {span} {CURRENCY_LABEL}."
    return sentence


# --------------------------------------------------------------------------
# assembling the whole digest
# --------------------------------------------------------------------------

SEPARATOR = "━━━━━━━━━━━━━━━"
NEWS_HEADING = "📌 Tin đáng chú ý"
BALANCE_HEADING = "🧭 Cán cân"
ITEM_BULLET = "🔹"
IMPACT_ARROW = "→"


def render_digest(
    *,
    title: str,
    window_line: str,
    items: Sequence[DigestItem],
    sources: Mapping[str, DigestSourceItem],
    price_reaction_block: str,
    balance: str,
    disclaimer: str,
) -> str:
    """Assemble the published digest. The one place this happens.

    Every structural decision is made here rather than by the model: the order
    of the sections, the separators, the bullet, the timestamp beside each
    item, the exact impact wording, and the disclaimer. The model contributed
    the headline, the optional note, the impact and the balance paragraph, and
    could not have contributed anything else - :class:`DigestEditorial` has no
    field for it.

    The timestamp is looked up from *sources* by id, so a digest reports the
    time the producer collected rather than the time the model remembered.

    Raises:
        KeyError: An item names a source the caller did not supply. That is a
            plumbing bug or a fabricated id, and both should stop rather than
            publish a bullet whose provenance nobody can find.
    """
    blocks: list[str] = [
        title,
        "",
        window_line,
        "",
        SEPARATOR,
        "",
        NEWS_HEADING,
        "",
        *_render_items(items, sources),
        SEPARATOR,
        "",
        price_reaction_block,
        "",
        f"{BALANCE_HEADING}\n{balance.strip()}",
        "",
        disclaimer,
    ]
    return "\n".join(blocks).strip()


def _render_items(
    items: Sequence[DigestItem], sources: Mapping[str, DigestSourceItem]
) -> list[str]:
    """One block per selected item, each followed by a blank line.

    The note is omitted rather than replaced by an empty line when the writer
    had nothing to add - an item that needs one line gets one line, which is
    what stops every entry looking like a form someone filled in.
    """
    blocks: list[str] = []
    for item in items:
        source = sources[item.news_item_id]
        when = _local(source.published_at).strftime("%H:%M")
        lines = [f"{ITEM_BULLET} {when} — {item.headline}"]
        if item.note:
            lines.append(item.note)
        lines.append(f"{IMPACT_ARROW} {IMPACT_LABELS[item.impact]}")
        blocks.append("\n".join(lines))
        blocks.append("")
    return blocks


def _local(moment: datetime) -> datetime:
    """The same instant, in the reader's timezone."""
    return moment.astimezone(resolve_timezone(ARTICLE_TIMEZONE))


__all__ = [
    "BALANCE_HEADING",
    "CURRENCY_LABEL",
    "DIGEST_TITLE_PREFIX",
    "IMPACT_ARROW",
    "ITEM_BULLET",
    "MOVE_DECIMALS",
    "NEWS_HEADING",
    "PERCENT_DECIMALS",
    "PRICE_REACTION_HEADING",
    "RANGE_LABEL",
    "SEPARATOR",
    "WINDOW_LINE_PREFIX",
    "WINDOW_LINE_SUFFIX",
    "digest_title",
    "digest_window_line",
    "format_move",
    "format_percent",
    "render_digest",
    "render_price_reaction",
]
