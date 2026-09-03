"""Deterministic presentation and arithmetic over an AnalysisContext.

Everything a writer model would otherwise have to compute or format itself is
computed here instead, in Python, exactly once:

* **price formatting** - a minimum 2-decimal convention, so the article never
  shows ``3315.1`` next to ``3312.45``, and the model is never the thing that
  decides how a number is written. Digits are padded, never dropped: a broker
  quoting gold to three decimals keeps all three;
* **simple arithmetic** - net change over the window, session high/low, the
  length of the current run of up or down closes.

This is *not* a technical-analysis engine. There are no indicators here, no
bias, no levels - only arithmetic a spreadsheet would do, so that the model can
describe price behaviour without doing sums in its head.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from goldpipeline.schemas.common import resolve_timezone
from goldpipeline.schemas.context import AnalysisContext
from goldpipeline.schemas.market import OHLCBar

MIN_PRICE_DECIMALS = 2
"""Fewest decimals a rendered gold price ever shows.

``3315.1`` shows as ``3315.10`` and ``3315`` as ``3315.00``. Padding a trailing
zero does not change the value, and a single convention is what stops one
paragraph reading ``3315.1`` and the next ``3315.10``.
"""


ARTICLE_TIMEZONE = "Asia/Ho_Chi_Minh"
"""Whose calendar day the article is dated by.

The reader is in Vietnam, and an article published at 07:00 local on the 4th is
dated the 4th even though UTC still says the 3rd. Dating it by UTC would put
yesterday's date on this morning's piece for the seven hours that matter most.
"""

ARTICLE_DATE_FORMAT = "%d.%m.%Y"


def article_date(moment: datetime) -> str:
    """The article's headline date, in the reader's own calendar.

    Computed here and handed to the model as data, never left for it to work
    out: a model asked for "today" answers from its training cutoff or from
    whatever the harness implies, and neither is this Run's date.
    """
    return moment.astimezone(resolve_timezone(ARTICLE_TIMEZONE)).strftime(ARTICLE_DATE_FORMAT)


def format_price(value: Decimal) -> str:
    """Render *value* for display, without changing it.

    Digits are only ever *added*. A broker that quotes gold to three decimals -
    and some do, ``4451.824`` - keeps all three, because the downstream gate
    matches article prices against the context exactly. Rounding here to a house
    convention of two would produce ``4451.82``: a number that appears nowhere in
    the data, that the gate would then refuse as unsupported, and that is in any
    case not the price. The formatting layer pads; it does not decide what a
    price is.
    """
    exponent = value.as_tuple().exponent
    natural = -exponent if isinstance(exponent, int) else MIN_PRICE_DECIMALS
    # Never fewer places than the value already has, so quantize can only pad.
    places = max(MIN_PRICE_DECIMALS, natural)
    return str(value.quantize(Decimal(1).scaleb(-places)))


def format_signed_price(value: Decimal) -> str:
    """Render a price *delta*, always carrying an explicit sign."""
    rendered = format_price(abs(value))
    return f"-{rendered}" if value < 0 else f"+{rendered}"


def _closing_run(bars: list[OHLCBar]) -> tuple[str, int]:
    """Length of the unbroken run of rising or falling closes at the end.

    Returns ``("up" | "down" | "flat", count)``. A flat close ends a run - the
    intent is to describe momentum honestly, not to stretch a streak.
    """
    if len(bars) < 2:
        return "flat", 0

    direction = ""
    count = 0
    for older, newer in zip(reversed(bars[:-1]), reversed(bars[1:]), strict=True):
        if newer.close > older.close:
            step = "up"
        elif newer.close < older.close:
            step = "down"
        else:
            break
        if direction and step != direction:
            break
        direction = step
        count += 1
    return direction or "flat", count


@dataclass(frozen=True)
class MarketFacts:
    """Formatted, pre-computed facts handed to the writer.

    Every value is derived from ``context`` alone. Nothing here is an opinion,
    and nothing is fetched.
    """

    symbol: str
    timeframe: str
    provider: str
    timezone: str
    bar_count: int
    data_from: str
    data_to: str
    latest_candle_at: str
    latest_open: str
    latest_high: str
    latest_low: str
    latest_close: str
    window_open: str
    window_high: str
    window_low: str
    net_change: str
    net_change_percent: str
    closing_run_direction: str
    closing_run_length: int
    data_quality_status: str
    missing_fields: list[str]
    quality_warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Plain mapping, ready to be serialized into the prompt."""
        return {
            "instrument": {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "provider": self.provider,
                "timezone": self.timezone,
            },
            "window": {
                "bar_count": self.bar_count,
                "data_from": self.data_from,
                "data_to": self.data_to,
                "latest_candle_at": self.latest_candle_at,
            },
            "latest_candle": {
                "open": self.latest_open,
                "high": self.latest_high,
                "low": self.latest_low,
                "close": self.latest_close,
            },
            "window_summary": {
                "first_open": self.window_open,
                "highest_high": self.window_high,
                "lowest_low": self.window_low,
                "net_change": self.net_change,
                "net_change_percent": self.net_change_percent,
                "closing_run_direction": self.closing_run_direction,
                "closing_run_length": self.closing_run_length,
            },
            "data_quality": {
                "status": self.data_quality_status,
                "missing_fields": self.missing_fields,
                "warnings": self.quality_warnings,
            },
        }


@dataclass(frozen=True)
class DerivedValue:
    """One deterministic arithmetic fact about the window.

    Typed, so the numeric scanner can tell a distance from a price without
    inspecting the number. Computed here and nowhere else: the prompt's fact
    sheet and the scanner must agree about what the net change *is*, and two
    implementations of one formula is how they stop agreeing.
    """

    kind: str
    value: Decimal
    semantic: str
    inputs: tuple[str, ...]


def derived_values(context: AnalysisContext) -> list[DerivedValue]:
    """The closed set of derived numbers the article may state uncited.

    Three formulas, each reproducible from ``context.json`` by hand:

    * ``NET_CHANGE`` - last close minus first open, a signed distance;
    * ``NET_CHANGE_PERCENT`` - that change over the first open, as a percentage;
    * ``WINDOW_RANGE`` - highest high minus lowest low, a distance.

    Deliberately *not* here: the distance between two arbitrary declared prices.
    Eighty candle values admit thousands of pairwise differences, and blessing
    all of them would let almost any number through - the exact failure mode
    this catalog exists to avoid. A distance that matters can be cited, or
    declared as its own field.
    """
    bars = context.ohlc.bars
    if not bars:
        return []

    first, latest = bars[0], bars[-1]
    net_change = latest.close - first.open
    high = max(bar.high for bar in bars)
    low = min(bar.low for bar in bars)

    values = [
        DerivedValue(
            kind="NET_CHANGE",
            value=net_change,
            semantic="MAGNITUDE",
            inputs=("context.ohlc.bars[-1].close", "context.ohlc.bars[0].open"),
        ),
        DerivedValue(
            kind="WINDOW_RANGE",
            value=high - low,
            semantic="MAGNITUDE",
            inputs=("window high", "window low"),
        ),
    ]
    if first.open:
        values.append(
            DerivedValue(
                kind="NET_CHANGE_PERCENT",
                value=net_change / first.open * Decimal(100),
                semantic="PERCENTAGE",
                inputs=("NET_CHANGE", "context.ohlc.bars[0].open"),
            )
        )
    return values


def build_market_facts(context: AnalysisContext) -> MarketFacts:
    """Derive the formatted fact sheet for *context*."""
    bars = context.ohlc.bars
    first = bars[0]

    derived = {item.kind: item.value for item in derived_values(context)}
    net_change = derived["NET_CHANGE"]
    percent = derived.get("NET_CHANGE_PERCENT", Decimal(0))
    direction, run_length = _closing_run(bars)

    return MarketFacts(
        symbol=context.market.symbol,
        timeframe=str(context.market.timeframe),
        provider=context.market.provider,
        timezone=context.market.timezone,
        bar_count=context.ohlc.bar_count,
        data_from=_iso(context.timing.data_from),
        data_to=_iso(context.timing.data_to),
        latest_candle_at=_iso(context.timing.latest_candle_at),
        latest_open=format_price(context.price.latest_open),
        latest_high=format_price(context.price.latest_high),
        latest_low=format_price(context.price.latest_low),
        latest_close=format_price(context.price.latest_close),
        window_open=format_price(first.open),
        window_high=format_price(max(bar.high for bar in bars)),
        window_low=format_price(min(bar.low for bar in bars)),
        net_change=format_signed_price(net_change),
        net_change_percent=f"{percent.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):+}%",
        closing_run_direction=direction,
        closing_run_length=run_length,
        data_quality_status=str(context.data_quality.status),
        missing_fields=list(context.data_quality.missing_fields),
        quality_warnings=[str(warning.code) for warning in context.data_quality.warnings],
    )


def format_recent_bars(context: AnalysisContext, limit: int = 12) -> list[dict[str, str]]:
    """The last *limit* candles, formatted for the prompt.

    The full series stays in ``context.json``; the prompt carries a tail. A
    writer describing short-term behaviour needs recent candles, and sending
    hundreds of bars would push the interesting ones away from the instruction.

    **Each candle carries its own address.** Trimming re-indexes the list: the
    eighth candle here was the sixteenth in a twenty-bar series, so a writer
    citing "index 7" would name a real path pointing at the wrong candle. That
    is a quieter version of the failure this whole contract exists to prevent -
    a claim that resolves, to something else - so the absolute path travels with
    the row rather than being left for the model to work out.
    """
    bars = context.ohlc.bars
    offset = max(len(bars) - limit, 0) if limit > 0 else 0
    recent = bars[-limit:] if limit > 0 else bars
    return [
        {
            "path": f"context.ohlc.bars[{offset + position}]",
            "t": _iso(bar.timestamp),
            "o": format_price(bar.open),
            "h": format_price(bar.high),
            "l": format_price(bar.low),
            "c": format_price(bar.close),
        }
        for position, bar in enumerate(recent)
    ]


def _iso(value: Any) -> str:
    return str(value.isoformat().replace("+00:00", "Z"))


__all__ = [
    "ARTICLE_DATE_FORMAT",
    "ARTICLE_TIMEZONE",
    "MIN_PRICE_DECIMALS",
    "article_date",
    "MarketFacts",
    "build_market_facts",
    "format_price",
    "format_recent_bars",
    "format_signed_price",
]
