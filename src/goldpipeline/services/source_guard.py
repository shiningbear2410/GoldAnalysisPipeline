"""First line of defence against publishing a number the market data denies.

The scenario this exists for: the raw analysis says "giá hiện tại 3400" while
``context.price.latest_close`` is 3315. A writer asked only to "rephrase the
analysis" would happily carry 3400 into the article as a present-tense fact.

So before the model is asked to write anything, the prices mentioned in the raw
text are extracted here, in Python, and compared against the range the candles
actually cover. Anything outside that range becomes:

* a :class:`~goldpipeline.schemas.writer.WriterWarning` recorded on the artifact, and
* an explicit caution injected into the prompt naming the offending numbers.

This is a *detector*, not a censor. It never edits the source text - Round 1's
inputs stay immutable - and it never decides what the article says. Round 3 will
check the finished article properly; this is the cheap check that stops the most
obvious contradiction from ever being drafted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from goldpipeline.schemas.context import AnalysisContext
from goldpipeline.schemas.writer import WarningCode, WriterWarning
from goldpipeline.services.market_facts import format_price

_NUMBER_RE = re.compile(r"(?<![\w.,])(\d{1,3}(?:[.,]\d{3})+|\d{3,6})(?:[.,](\d{1,3}))?(?![\d])")
"""Numbers that could plausibly be a gold quote.

Vietnamese analysis writes three thousand three hundred and fifteen point two as
``3.315,2`` or ``3315.2`` or ``3,315.2`` - all three appear in real messages, so
the separator handling below is deliberately tolerant.
"""

RANGE_TOLERANCE = Decimal("0.25")
"""Share of the observed candle range tolerated beyond it."""

PRICE_TOLERANCE = Decimal("0.005")
"""Share of the price level tolerated beyond the candle range.

The band has to be relative to the *price*, not only to the range, or the check
becomes useless noise. A quiet Asian session covers maybe 17 points on a 3315
instrument; a band of a few percent of that range is under half a point, so an
analyst naming a resistance two points above the session high gets flagged - and
a warning that fires on every ordinary message is a warning nobody reads.

0.5% of price (~16 points on gold) comfortably covers the levels an analyst
actually names around the current session, while a claim like "giá hiện tại
3400" against a 3305-3322 window is 2.4% away and still caught. The wider of the
two bands wins, so an unusually volatile window is handled too.
"""

MIN_PLAUSIBLE_QUOTE = Decimal("100")
"""Below this a number is a lot size, a percentage or a date fragment, not gold."""


@dataclass(frozen=True)
class NumberMatch:
    """One number found in free text, with the literal that produced it."""

    literal: str
    value: Decimal
    start: int
    end: int


def extract_numbers(text: str, *, minimum: Decimal | None = None) -> list[NumberMatch]:
    """Find every number in *text* that could plausibly be a gold quote.

    Vietnamese analysis writes three thousand three hundred and fifteen point two
    as ``3.315,2``, ``3315.2`` or ``3,315.2`` - all three appear in real
    messages, so separator handling is deliberately tolerant.

    Args:
        text: Free text to scan.
        minimum: Values below this are skipped. Defaults to
            :data:`MIN_PLAUSIBLE_QUOTE`, which drops lot sizes, percentages and
            small counts.
    """
    floor = MIN_PLAUSIBLE_QUOTE if minimum is None else minimum
    found: list[NumberMatch] = []
    for match in _NUMBER_RE.finditer(text):
        value = _parse_number(match.group(1), match.group(2))
        if value is None or value < floor:
            continue
        found.append(
            NumberMatch(literal=match.group(0), value=value, start=match.start(), end=match.end())
        )
    return found


@dataclass(frozen=True)
class SourcePriceFinding:
    """One price mentioned in the raw analysis that the candles do not support."""

    literal: str
    value: Decimal
    position: int


@dataclass(frozen=True)
class SourceGuardReport:
    """Outcome of screening the raw analysis against the market data."""

    window_low: Decimal
    window_high: Decimal
    out_of_range: list[SourcePriceFinding]

    @property
    def has_findings(self) -> bool:
        """Whether anything needs to be flagged to the writer."""
        return bool(self.out_of_range)


def _parse_number(whole: str, fraction: str | None) -> Decimal | None:
    """Interpret a matched number, resolving thousands-vs-decimal separators."""
    digits = whole.replace(".", "").replace(",", "")
    if not digits.isdigit():
        return None
    text = f"{digits}.{fraction}" if fraction else digits
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def screen_source_prices(context: AnalysisContext) -> SourceGuardReport:
    """Find prices in the raw analysis that fall outside the candle range.

    Args:
        context: The Run's context. Both the text and the candles come from it.
    """
    bars = context.ohlc.bars
    low = min(bar.low for bar in bars)
    high = max(bar.high for bar in bars)

    margin = max((high - low) * RANGE_TOLERANCE, high * PRICE_TOLERANCE)
    lower_bound, upper_bound = low - margin, high + margin

    findings: list[SourcePriceFinding] = []
    seen: set[Decimal] = set()

    for match in extract_numbers(context.raw_analysis.text):
        if lower_bound <= match.value <= upper_bound or match.value in seen:
            continue
        seen.add(match.value)
        findings.append(
            SourcePriceFinding(literal=match.literal, value=match.value, position=match.start)
        )

    return SourceGuardReport(window_low=low, window_high=high, out_of_range=findings)


def build_guard_warnings(report: SourceGuardReport) -> list[WriterWarning]:
    """Turn a report into warnings for the writer artifact."""
    if not report.has_findings:
        return []
    listed = ", ".join(finding.literal for finding in report.out_of_range[:10])
    return [
        WriterWarning(
            code=WarningCode.SOURCE_PRICE_OUT_OF_RANGE,
            message=(
                f"Raw analysis mentions price levels outside the candle range "
                f"{format_price(report.window_low)}-{format_price(report.window_high)}: {listed}. "
                "They must not be presented as the current market price."
            ),
        )
    ]


def build_guard_notice(report: SourceGuardReport) -> str | None:
    """Render the caution injected into the prompt, or ``None`` when clean."""
    if not report.has_findings:
        return None
    listed = ", ".join(finding.literal for finding in report.out_of_range[:10])
    return (
        f"The source text mentions these numbers, which lie outside the candle range "
        f"{format_price(report.window_low)}-{format_price(report.window_high)}: {listed}. "
        "Market facts win. Do not state any of them as the current price. You may "
        "refer to such a number as a level the analyst mentioned, but only if the "
        "source clearly means it as a target or a level rather than as the price now."
    )


__all__ = [
    "MIN_PLAUSIBLE_QUOTE",
    "PRICE_TOLERANCE",
    "RANGE_TOLERANCE",
    "NumberMatch",
    "extract_numbers",
    "SourceGuardReport",
    "SourcePriceFinding",
    "build_guard_notice",
    "build_guard_warnings",
    "screen_source_prices",
]
