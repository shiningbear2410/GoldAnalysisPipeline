"""The TRADE_PLAN text, composed by code because no model may compose it.

Round 6.6g. This is the last stage, and the shortest: two section labels, a
handful of numbers, two Vietnamese phrases and nothing else.

**No model, ever.** The analyst's contribution ended when it returned an
ordering of ids. Nothing here calls a writer, a reviewer, a finalizer or a
vendor, and a guard proves it by parsing this module rather than trusting the
absence. A rendered trade plan is the one document in this project that a
language model has never touched, which is exactly why it can carry prices.

**Every character is accounted for.** The published vocabulary is ``SEO``,
``BAI``, ``vùng chính``, ``sâu hơn``, ``—`` and digits. No disclaimer, no date,
no stop loss, no target, no ratio, no explanation, no emoji, no markdown. Not
because those are hard, but because each is a claim, and this stage has no
evidence for any of them.

**Numbers pass through untouched.** ``canonical_price`` - the same helper that
mints candidate identities - strips trailing zeros and changes no value:
4000, 4000.0 and 4000.00 all render as ``4000``, and 4050.30 as ``4050.3``.
Nothing is rounded to two decimals, padded to three, or given a thousands
separator - the reader sees the price the engine measured.

**Fail closed on length.** The caps upstream make 650 characters unreachable in
practice. If some unexpected numeric representation reached it anyway, this
module raises rather than truncating a price or dropping a line, because a
silently shortened trade plan is worse than none.
"""

from __future__ import annotations

import logging

from goldpipeline.services.ict_candidate_consolidation import canonical_price
from goldpipeline.services.ict_candidate_eligibility import EntrySide
from goldpipeline.services.trade_plan_selector import (
    SelectedEntryZone,
    SelectedReference,
    TradePlanSelection,
    ZoneLabel,
)

logger = logging.getLogger(__name__)

TRADE_PLAN_RENDER_METHOD_VERSION = "1.0.0"

SEO_HEADING = "SEO"
BAI_HEADING = "BAI"

ZONE_DASH = "–"
"""En dash, between the two edges of a zone. One character, chosen once."""

EMPTY_SIDE = "—"
"""Em dash, the whole body of a side with nothing to publish.

Deliberately different from :data:`ZONE_DASH`: one means "from here to there"
and the other means "nothing", and a reader should not have to tell them apart
by width.
"""

MAIN_ZONE_SUFFIX = "(vùng chính)"
FARTHER_SUFFIX = "(sâu hơn)"

MAX_TRADE_PLAN_CHARS = 650
"""A hard ceiling that fails closed. 200-450 is guidance, and never padding."""

PUBLIC_VOCABULARY: frozenset[str] = frozenset(
    {SEO_HEADING, BAI_HEADING, "vùng", "chính", "sâu", "hơn", EMPTY_SIDE}
)
"""Every non-numeric word this stage may publish. Checked by the validator."""

FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "BUY",
    "SELL",
    "ENTRY",
    "SL",
    "TP",
    "R:R",
    "OB",
    "FVG",
    "liquidity",
    "candidate",
    "score",
    "rank",
    "confidence",
    "disclaimer",
    "http",
)
"""Named so a regression is legible, not because the vocabulary check is weak.

``PUBLIC_VOCABULARY`` already refuses everything not on it; this list makes the
specific failure - a stop loss leaking into a published plan - say so by name.
"""


class TradePlanRenderError(ValueError):
    """The plan could not be rendered honestly, so it was not rendered."""


def render_price(value: object) -> str:
    """One price, exactly as measured.

    Reuses the identity helper rather than a second formatter, so a price on the
    page and the same price inside a candidate id are written the same way. That
    is not a coincidence worth risking to a duplicate implementation.
    """
    from decimal import Decimal

    if not isinstance(value, Decimal):  # pragma: no cover - typed callers only
        raise TradePlanRenderError(f"a price must be a Decimal, got {type(value).__name__}")
    return canonical_price(value)


def render_zone(zone: SelectedEntryZone) -> str:
    """``lower–upper``, plus ``(vùng chính)`` when it carries the label."""
    body = f"{render_price(zone.lower)}{ZONE_DASH}{render_price(zone.upper)}"
    if zone.label is ZoneLabel.VUNG_CHINH:
        return f"{body} {MAIN_ZONE_SUFFIX}"
    return body


def render_reference(reference: SelectedReference) -> str:
    """``price (sâu hơn)``. A single level, and never widened into a zone."""
    return f"{render_price(reference.level)} {FARTHER_SUFFIX}"


def _side_lines(
    entries: tuple[SelectedEntryZone, ...], reference: SelectedReference | None
) -> list[str]:
    if not entries:
        # A farther level with nothing to be farther *than* is not published.
        return [EMPTY_SIDE]
    lines = [render_zone(zone) for zone in entries]
    if reference is not None:
        # After the zones, because it is explicitly not one of them.
        lines.append(render_reference(reference))
    return lines


def render_trade_plan(selection: TradePlanSelection) -> str:
    """The published text, in full.

    Both headings always appear, so a reader can tell "no zones on this side"
    from "this section is missing".

    Raises:
        TradePlanRenderError: The result exceeds the hard character cap.
    """
    lines = [
        SEO_HEADING,
        *_side_lines(selection.seo_entries, selection.seo_reference),
        "",
        BAI_HEADING,
        *_side_lines(selection.bai_entries, selection.bai_reference),
    ]
    text = "\n".join(lines)

    if len(text) > MAX_TRADE_PLAN_CHARS:
        raise TradePlanRenderError(
            f"rendered trade plan is {len(text)} characters, over the "
            f"{MAX_TRADE_PLAN_CHARS} cap; refusing to truncate a price"
        )

    logger.info(
        "trade_plan.render chars=%d seo=%d bai=%d",
        len(text),
        len(selection.seo_entries),
        len(selection.bai_entries),
    )
    return text


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def _expected_prices(selection: TradePlanSelection) -> set[str]:
    prices: set[str] = set()
    for zone in (*selection.seo_entries, *selection.bai_entries):
        prices.add(render_price(zone.lower))
        prices.add(render_price(zone.upper))
    for reference in (selection.seo_reference, selection.bai_reference):
        if reference is not None:
            prices.add(render_price(reference.level))
    return prices


def validate_trade_plan(text: str, selection: TradePlanSelection) -> None:
    """Prove the text says only what the selection decided.

    Checked against the structured selection rather than by parsing prose back
    into prices: the selection is the authority, and re-deriving prices from the
    page would make the page the authority instead.

    Raises:
        TradePlanRenderError: The text is not a faithful rendering.
    """
    lines = text.split("\n")

    if lines.count(SEO_HEADING) != 1 or lines.count(BAI_HEADING) != 1:
        raise TradePlanRenderError(
            f"expected exactly one {SEO_HEADING} and one {BAI_HEADING} heading"
        )
    if lines.index(SEO_HEADING) > lines.index(BAI_HEADING):
        raise TradePlanRenderError("SEO must precede BAI")

    if len(text) > MAX_TRADE_PLAN_CHARS:
        raise TradePlanRenderError(f"rendered trade plan is {len(text)} characters")

    for forbidden in FORBIDDEN_SUBSTRINGS:
        if forbidden in text:
            raise TradePlanRenderError(f"forbidden text in trade plan: {forbidden!r}")

    for candidate_id in (
        *(zone.candidate_id for zone in (*selection.seo_entries, *selection.bai_entries)),
        *(
            reference.candidate_id
            for reference in (selection.seo_reference, selection.bai_reference)
            if reference is not None
        ),
    ):
        if candidate_id in text:
            raise TradePlanRenderError(f"candidate id {candidate_id} leaked into the plan")

    words = {
        word
        for line in lines
        for word in line.replace("(", " ").replace(")", " ").split()
        if not any(character.isdigit() for character in word)
    }
    unknown = sorted(words - PUBLIC_VOCABULARY)
    if unknown:
        raise TradePlanRenderError(f"words outside the public vocabulary: {unknown}")

    expected = _expected_prices(selection)
    found: set[str] = set()
    for line in lines:
        if line in {"", SEO_HEADING, BAI_HEADING, EMPTY_SIDE}:
            continue
        body = line.replace(MAIN_ZONE_SUFFIX, "").replace(FARTHER_SUFFIX, "").strip()
        for token in body.split(ZONE_DASH):
            token = token.strip()
            if not token:
                raise TradePlanRenderError(f"empty price in line {line!r}")
            if token not in expected:
                raise TradePlanRenderError(f"price {token!r} is not a selected candidate's price")
            found.add(token)

    missing = sorted(expected - found)
    if missing:
        raise TradePlanRenderError(f"selected prices missing from the plan: {missing}")

    labels = text.count(MAIN_ZONE_SUFFIX)
    expected_labels = sum(
        1 for side in (EntrySide.SEO, EntrySide.BAI) if selection.main_zone(side) is not None
    )
    if labels != expected_labels:
        raise TradePlanRenderError(f"expected {expected_labels} main-zone labels, found {labels}")


def render_and_validate(selection: TradePlanSelection) -> str:
    """Render, then prove the rendering. The only entry point callers need."""
    text = render_trade_plan(selection)
    validate_trade_plan(text, selection)
    return text


__all__ = [
    "BAI_HEADING",
    "EMPTY_SIDE",
    "FARTHER_SUFFIX",
    "FORBIDDEN_SUBSTRINGS",
    "MAIN_ZONE_SUFFIX",
    "MAX_TRADE_PLAN_CHARS",
    "PUBLIC_VOCABULARY",
    "SEO_HEADING",
    "TRADE_PLAN_RENDER_METHOD_VERSION",
    "ZONE_DASH",
    "TradePlanRenderError",
    "render_and_validate",
    "render_price",
    "render_reference",
    "render_trade_plan",
    "render_zone",
    "validate_trade_plan",
]
