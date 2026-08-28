"""Normalization: turn loose provider payloads into guaranteed-consistent data.

This module owns every policy decision about input data. They are stated here
once, explicitly, so that reviewers do not have to infer them from behaviour:

===========================  ===========================================
Situation                    Policy
===========================  ===========================================
Duplicate bar timestamp      FAIL. Never silently de-duplicated - the
                             pipeline cannot know which bar is correct.
Bars out of order            Sorted ascending + WARNING. Ordering is
                             recoverable without guessing.
Naive timestamp, tz known    Interpreted in the declared source timezone.
Naive timestamp, no tz       FAIL.
Symbol disagreement          FAIL. Covers per-bar symbols and the
                             symbol the caller expected.
Provider latest_bar          Cross-checked against bars[-1]; FAIL on
                             mismatch. Never trusted as a source.
Declared range vs bars       Bars win; declared range adjusted + WARNING.
Missing volume               WARNING + missing_fields entry.
Gaps in the series           WARNING (weekends/holidays are normal).
Latest candle in the future  WARNING. Signals a provider clock or
                             timezone bug; prices must not be quoted
                             from a candle that has not closed.
Latest candle very old       WARNING.
Empty analysis text          FAIL.
Control chars in text        Stripped + WARNING. Text otherwise verbatim.
===========================  ===========================================
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from goldpipeline.domain.errors import (
    AnalysisTextTooLargeError,
    DuplicateTimestampError,
    EmptyAnalysisTextError,
    EmptyBarsError,
    LatestBarMismatchError,
    NaiveTimestampError,
    SymbolMismatchError,
    UnknownTimezoneError,
)
from goldpipeline.schemas.common import Timeframe, normalize_symbol, resolve_timezone, utc_now
from goldpipeline.schemas.market import MarketDataInput, MarketDataSnapshot, OHLCBar
from goldpipeline.schemas.quality import QualityWarning, WarningCode
from goldpipeline.schemas.telegram import (
    MAX_RAW_TEXT_CHARS,
    RAW_TEXT_WARN_CHARS,
    TelegramAnalysisInput,
)

LOW_BAR_COUNT_THRESHOLD = 10
"""Below this many candles, downstream analysis is thin enough to flag."""

STALE_DATA_MULTIPLIER = 12
"""Latest candle older than this many timeframe durations is flagged as stale."""

_ALLOWED_CONTROL_CHARS = frozenset({"\n", "\t"})

_INVISIBLE_CODEPOINTS = (
    0x200B,  # ZERO WIDTH SPACE
    0x200C,  # ZERO WIDTH NON-JOINER
    0x200D,  # ZERO WIDTH JOINER
    0x2060,  # WORD JOINER
    0xFEFF,  # ZERO WIDTH NO-BREAK SPACE / BOM
    0x202A,  # LEFT-TO-RIGHT EMBEDDING
    0x202B,  # RIGHT-TO-LEFT EMBEDDING
    0x202C,  # POP DIRECTIONAL FORMATTING
    0x202D,  # LEFT-TO-RIGHT OVERRIDE
    0x202E,  # RIGHT-TO-LEFT OVERRIDE
    0x2066,  # LEFT-TO-RIGHT ISOLATE
    0x2067,  # RIGHT-TO-LEFT ISOLATE
    0x2068,  # FIRST STRONG ISOLATE
    0x2069,  # POP DIRECTIONAL ISOLATE
)

_INVISIBLE_CHARS = frozenset(chr(code) for code in _INVISIBLE_CODEPOINTS)
"""Zero-width and bidirectional-override characters.

Stripped because they are invisible to a human reviewer but not to a model - a
classic vector for hiding text inside otherwise innocent-looking prose.
"""


@dataclass
class NormalizedMarketData:
    """Result of normalizing a market data payload."""

    snapshot: MarketDataSnapshot
    warnings: list[QualityWarning] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)


@dataclass
class NormalizedAnalysis:
    """Result of normalizing a raw analysis payload."""

    analysis: TelegramAnalysisInput
    warnings: list[QualityWarning] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# market data
# --------------------------------------------------------------------------


def normalize_market_data(
    raw: MarketDataInput,
    *,
    expected_symbol: str | None = None,
    now: datetime | None = None,
) -> NormalizedMarketData:
    """Validate and normalize *raw* into a self-consistent snapshot.

    Args:
        raw: Payload as parsed from a provider or fixture.
        expected_symbol: Instrument the caller believes it asked for. When it
            disagrees with the payload the call fails rather than guessing.
        now: Injection point for tests; defaults to the current UTC time.

    Raises:
        NormalizationError: On any fatal inconsistency (see module docstring).
    """
    moment = now or utc_now()
    warnings: list[QualityWarning] = []
    missing: list[str] = []

    if not raw.bars:
        raise EmptyBarsError("market data payload contains no bars", provider=raw.provider)

    symbol = _resolve_symbol(raw, expected_symbol)
    tzinfo = _resolve_source_timezone(raw.timezone)

    bars = [_to_utc_bar(bar, tzinfo) for bar in raw.bars]
    _reject_symbol_conflicts(bars, symbol)
    _reject_duplicate_timestamps(bars)

    timestamps = [bar.timestamp for bar in bars]
    if timestamps != sorted(timestamps):
        bars.sort(key=lambda bar: bar.timestamp)
        warnings.append(
            QualityWarning(
                code=WarningCode.BARS_REORDERED,
                message="Bars were not sorted ascending by timestamp; the pipeline sorted them.",
                details={"bar_count": len(bars)},
            )
        )

    latest = bars[-1]
    _reject_latest_bar_mismatch(raw.latest_bar, latest, tzinfo)

    data_from, data_to = bars[0].timestamp, latest.timestamp
    warnings.extend(_check_declared_range(raw, tzinfo, data_from, data_to))

    requested_at = raw.requested_at
    if requested_at is None:
        requested_at = moment
        warnings.append(
            QualityWarning(
                code=WarningCode.REQUESTED_AT_DEFAULTED,
                message="Payload did not declare requested_at; ingestion time was used.",
                details={"requested_at": requested_at.isoformat()},
            )
        )
        missing.append("market.requested_at")
    else:
        requested_at = _ensure_utc(requested_at, tzinfo, "requested_at")

    bars_without_volume = sum(1 for bar in bars if bar.volume is None)
    if bars_without_volume:
        missing.append("ohlc.volume")
        warnings.append(
            QualityWarning(
                code=WarningCode.MISSING_VOLUME,
                message="Provider did not supply volume for every bar.",
                details={"bars_without_volume": bars_without_volume, "bar_count": len(bars)},
            )
        )

    warnings.extend(_check_gaps(bars, raw.timeframe))
    warnings.extend(_check_bar_count(bars))
    warnings.extend(_check_recency(latest.timestamp, raw.timeframe, moment))

    snapshot = MarketDataSnapshot(
        symbol=symbol,
        provider=raw.provider,
        timeframe=raw.timeframe,
        timezone="UTC",
        source_timezone=raw.timezone,
        requested_at=requested_at,
        data_from=data_from,
        data_to=data_to,
        bars=bars,
        latest_bar=latest,
    )
    return NormalizedMarketData(snapshot=snapshot, warnings=warnings, missing_fields=missing)


def _resolve_symbol(raw: MarketDataInput, expected_symbol: str | None) -> str:
    symbol = raw.symbol
    if expected_symbol is None:
        return symbol
    expected = normalize_symbol(expected_symbol)
    if expected != symbol:
        raise SymbolMismatchError(
            f"caller expected symbol {expected!r} but market data declares {symbol!r}",
            expected=expected,
            actual=symbol,
        )
    return symbol


def _resolve_source_timezone(name: str | None) -> Any:
    """Resolve the declared timezone, or None when the payload declares none."""
    if name is None or not name.strip():
        return None
    try:
        return resolve_timezone(name)
    except ValueError as exc:
        raise UnknownTimezoneError(str(exc), timezone=name) from exc


def _ensure_utc(value: datetime, tzinfo: Any, label: str) -> datetime:
    """Coerce *value* to UTC, using the declared source timezone when naive.

    A naive timestamp with no declared timezone is fatal rather than assumed to
    be UTC: guessing here would silently shift every candle by the provider's
    offset, and every price claim built on them would be attached to the wrong
    moment.
    """
    if value.tzinfo is not None:
        return value.astimezone(UTC)
    if tzinfo is None:
        raise NaiveTimestampError(
            f"{label} has no timezone offset and the payload declares no source timezone",
            field=label,
        )
    return value.replace(tzinfo=tzinfo).astimezone(UTC)


def _to_utc_bar(bar: OHLCBar, tzinfo: Any) -> OHLCBar:
    if bar.timestamp.tzinfo is not None:
        if bar.timestamp.utcoffset() == timedelta(0):
            return bar
        return bar.model_copy(update={"timestamp": bar.timestamp.astimezone(UTC)})
    localized = _ensure_utc(bar.timestamp, tzinfo, "bar.timestamp")
    return bar.model_copy(update={"timestamp": localized})


def _reject_symbol_conflicts(bars: list[OHLCBar], symbol: str) -> None:
    conflicting = {bar.symbol for bar in bars if bar.symbol is not None and bar.symbol != symbol}
    if conflicting:
        raise SymbolMismatchError(
            f"bars declare symbols {sorted(conflicting)} but the snapshot declares {symbol!r}",
            snapshot_symbol=symbol,
            bar_symbols=sorted(conflicting),
        )


def _reject_duplicate_timestamps(bars: list[OHLCBar]) -> None:
    seen: set[datetime] = set()
    duplicates: set[str] = set()
    for bar in bars:
        if bar.timestamp in seen:
            duplicates.add(bar.timestamp.isoformat())
        seen.add(bar.timestamp)
    if duplicates:
        raise DuplicateTimestampError(
            "market data contains duplicate bar timestamps; "
            "refusing to guess which bar is authoritative",
            duplicates=sorted(duplicates),
        )


def _reject_latest_bar_mismatch(declared: OHLCBar | None, actual: OHLCBar, tzinfo: Any) -> None:
    if declared is None:
        return
    normalized = _to_utc_bar(declared, tzinfo)
    same_time = normalized.timestamp == actual.timestamp
    same_prices = (normalized.open, normalized.high, normalized.low, normalized.close) == (
        actual.open,
        actual.high,
        actual.low,
        actual.close,
    )
    if not (same_time and same_prices):
        raise LatestBarMismatchError(
            "payload latest_bar disagrees with the last bar of the series",
            declared={
                "timestamp": normalized.timestamp.isoformat(),
                "close": str(normalized.close),
            },
            actual={"timestamp": actual.timestamp.isoformat(), "close": str(actual.close)},
        )


def _check_declared_range(
    raw: MarketDataInput, tzinfo: Any, data_from: datetime, data_to: datetime
) -> list[QualityWarning]:
    adjustments: dict[str, str] = {}
    for label, declared, actual in (
        ("data_from", raw.data_from, data_from),
        ("data_to", raw.data_to, data_to),
    ):
        if declared is None:
            continue
        normalized = _ensure_utc(declared, tzinfo, label)
        if normalized != actual:
            adjustments[label] = f"declared {normalized.isoformat()}, actual {actual.isoformat()}"
    if not adjustments:
        return []
    return [
        QualityWarning(
            code=WarningCode.DECLARED_RANGE_ADJUSTED,
            message="Declared coverage range did not match the bars; bar coverage was used.",
            details=adjustments,
        )
    ]


def _check_gaps(bars: list[OHLCBar], timeframe: Timeframe) -> list[QualityWarning]:
    duration = timeframe.duration
    if duration is None or len(bars) < 2:
        return []
    gaps = [
        {
            "after": bars[index - 1].timestamp.isoformat(),
            "before": bars[index].timestamp.isoformat(),
            "missing_bars": int((bars[index].timestamp - bars[index - 1].timestamp) / duration) - 1,
        }
        for index in range(1, len(bars))
        if bars[index].timestamp - bars[index - 1].timestamp > duration
    ]
    if not gaps:
        return []
    return [
        QualityWarning(
            code=WarningCode.BAR_GAPS,
            message=(
                "Series is not contiguous at the declared timeframe. "
                "Weekends and session breaks are expected causes."
            ),
            details={"gap_count": len(gaps), "gaps": gaps[:10]},
        )
    ]


def _check_bar_count(bars: list[OHLCBar]) -> list[QualityWarning]:
    if len(bars) >= LOW_BAR_COUNT_THRESHOLD:
        return []
    return [
        QualityWarning(
            code=WarningCode.LOW_BAR_COUNT,
            message="Few candles available; downstream context will be thin.",
            details={"bar_count": len(bars), "threshold": LOW_BAR_COUNT_THRESHOLD},
        )
    ]


def _check_recency(
    latest_at: datetime, timeframe: Timeframe, moment: datetime
) -> list[QualityWarning]:
    """Flag market data that is too old, or timestamped in the future.

    Future-dated candles almost always mean a timezone or clock-skew bug on the
    provider side. Left unflagged they are actively dangerous: a writer agent
    would quote a price from a candle that has not closed yet. This is a warning
    rather than a failure because the pipeline cannot tell a slightly fast clock
    from a genuinely wrong timezone - it reports the fact and lets the caller
    decide.
    """
    duration = timeframe.duration
    if duration is None:
        return []

    age = moment - latest_at
    if age < -duration:
        return [
            QualityWarning(
                code=WarningCode.FUTURE_DATA,
                message=(
                    "Latest candle is timestamped in the future; check the provider "
                    "timezone and clock before quoting these prices."
                ),
                details={
                    "latest_candle_at": latest_at.isoformat(),
                    "now": moment.isoformat(),
                    "ahead_seconds": int(-age.total_seconds()),
                },
            )
        ]

    if age <= duration * STALE_DATA_MULTIPLIER:
        return []
    return [
        QualityWarning(
            code=WarningCode.STALE_DATA,
            message="Latest candle is old relative to the timeframe.",
            details={
                "latest_candle_at": latest_at.isoformat(),
                "now": moment.isoformat(),
                "age_seconds": int(age.total_seconds()),
            },
        )
    ]


# --------------------------------------------------------------------------
# analysis text
# --------------------------------------------------------------------------


def sanitize_analysis_text(text: str) -> tuple[str, dict[str, int]]:
    """Strip characters that are invisible or meaningless in analysis prose.

    Removes control characters (except newline and tab) and zero-width / bidi
    override characters, and normalizes to NFC so Vietnamese diacritics compare
    and render consistently. Nothing else is altered: wording, punctuation and
    line structure are preserved verbatim.

    Returns:
        The cleaned text and a count of what was removed, by category.
    """
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))

    removed = {"control": 0, "invisible": 0}
    kept: list[str] = []
    for char in normalized:
        if char in _INVISIBLE_CHARS:
            removed["invisible"] += 1
            continue
        if char not in _ALLOWED_CONTROL_CHARS and unicodedata.category(char) == "Cc":
            removed["control"] += 1
            continue
        kept.append(char)

    cleaned = "\n".join(line.rstrip() for line in "".join(kept).split("\n")).strip()
    return cleaned, {key: value for key, value in removed.items() if value}


def normalize_analysis(raw: TelegramAnalysisInput) -> NormalizedAnalysis:
    """Validate and sanitize a raw analysis message.

    Raises:
        EmptyAnalysisTextError: If nothing usable remains after sanitisation.
        AnalysisTextTooLargeError: If the text exceeds the hard size limit.
    """
    warnings: list[QualityWarning] = []
    missing: list[str] = []

    if len(raw.raw_text) > MAX_RAW_TEXT_CHARS:
        raise AnalysisTextTooLargeError(
            f"analysis text is {len(raw.raw_text)} characters, limit is {MAX_RAW_TEXT_CHARS}",
            length=len(raw.raw_text),
            limit=MAX_RAW_TEXT_CHARS,
        )

    cleaned, removed = sanitize_analysis_text(raw.raw_text)
    if not cleaned:
        raise EmptyAnalysisTextError("analysis text is empty after sanitisation")

    if removed:
        warnings.append(
            QualityWarning(
                code=WarningCode.RAW_TEXT_SANITIZED,
                message="Invisible or control characters were removed from the analysis text.",
                details=dict(removed),
            )
        )

    if len(cleaned) > RAW_TEXT_WARN_CHARS:
        warnings.append(
            QualityWarning(
                code=WarningCode.RAW_TEXT_VERY_LONG,
                message="Analysis text is unusually long.",
                details={"length": len(cleaned), "soft_limit": RAW_TEXT_WARN_CHARS},
            )
        )

    absent = [
        f"raw_analysis.{name}"
        for name, value in (
            ("chat_id", raw.chat_id),
            ("message_id", raw.message_id),
            ("message_date", raw.message_date),
            ("author", raw.author),
        )
        if value is None
    ]
    if absent:
        missing.extend(absent)
        warnings.append(
            QualityWarning(
                code=WarningCode.MISSING_TELEGRAM_METADATA,
                message="Source did not provide some optional message metadata.",
                details={"fields": absent},
            )
        )

    analysis = raw.model_copy(update={"raw_text": cleaned})
    return NormalizedAnalysis(analysis=analysis, warnings=warnings, missing_fields=missing)


__all__ = [
    "LOW_BAR_COUNT_THRESHOLD",
    "STALE_DATA_MULTIPLIER",
    "NormalizedAnalysis",
    "NormalizedMarketData",
    "normalize_analysis",
    "normalize_market_data",
    "sanitize_analysis_text",
]
