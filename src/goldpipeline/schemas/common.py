"""Shared field types and helpers.

Two decisions worth stating explicitly, because everything downstream depends
on them:

* **Prices are :class:`~decimal.Decimal`.** They are serialized to JSON as
  strings (``"3312.45"``), which round-trips exactly. A float would introduce
  representation noise into a document an LLM is expected to quote verbatim.
* **Timestamps are timezone-aware UTC.** They serialize as ``2026-08-28T02:20:00Z``.
  The provider's own timezone is recorded separately as metadata, never used to
  store values.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer

MAX_PRICE_EXPONENT = 8
"""Maximum number of decimal places accepted for a price."""


def _serialize_utc(value: datetime) -> str:
    """Render an aware datetime as a ``Z``-suffixed ISO-8601 string."""
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _coerce_utc(value: datetime) -> datetime:
    """Normalize an aware datetime to UTC.

    Naive datetimes are rejected here on purpose: at this layer there is no
    timezone context to interpret them with. The normalizer resolves naive
    market timestamps earlier, using the payload's declared source timezone.
    """
    if value.tzinfo is None:
        raise ValueError("naive datetime is not allowed; supply an explicit offset")
    return value.astimezone(UTC)


UtcDatetime = Annotated[
    datetime,
    PlainSerializer(_serialize_utc, return_type=str, when_used="json"),
]
"""An aware datetime, serialized as ``...Z``. Coercion is applied by validators."""

Price = Annotated[Decimal, Field(gt=0)]
"""A strictly positive price. Serialized by pydantic as a JSON string."""

Volume = Annotated[Decimal, Field(ge=0)]
"""A non-negative traded volume."""


class Timeframe(StrEnum):
    """Chart timeframes the pipeline understands."""

    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN1 = "MN1"

    @property
    def duration(self) -> timedelta | None:
        """Nominal bar duration, or ``None`` for calendar-based timeframes."""
        return _TIMEFRAME_DURATIONS.get(self)


_TIMEFRAME_DURATIONS: dict[Timeframe, timedelta] = {
    Timeframe.M1: timedelta(minutes=1),
    Timeframe.M5: timedelta(minutes=5),
    Timeframe.M15: timedelta(minutes=15),
    Timeframe.M30: timedelta(minutes=30),
    Timeframe.H1: timedelta(hours=1),
    Timeframe.H4: timedelta(hours=4),
    Timeframe.D1: timedelta(days=1),
    Timeframe.W1: timedelta(weeks=1),
}

_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9._+-]{1,23}$")
_FIXED_OFFSET_RE = re.compile(r"^(?:UTC|GMT)?(?P<sign>[+-])(?P<h>\d{1,2})(?::?(?P<m>\d{2}))?$")


def normalize_symbol(value: str) -> str:
    """Canonicalize an instrument symbol to a comparable form.

    ``" xauusd "`` and ``"XAUUSD"`` are the same instrument; ``"XAU/USD"`` is
    normalized to ``"XAUUSD"`` so that providers using different separators do
    not trip the mismatch check.
    """
    cleaned = value.strip().upper().replace("/", "").replace(" ", "")
    if not _SYMBOL_RE.fullmatch(cleaned):
        raise ValueError(f"invalid instrument symbol: {value!r}")
    return cleaned


def resolve_timezone(name: str) -> timezone | Any:
    """Resolve a timezone declaration to a tzinfo object.

    Accepts ``"UTC"``/``"Z"``, fixed offsets (``"+07:00"``, ``"UTC+7"``) and
    IANA names (``"Asia/Ho_Chi_Minh"``).

    Raises:
        ValueError: If the name cannot be resolved.
    """
    candidate = name.strip()
    if candidate.upper() in {"UTC", "Z", "GMT", "UTC+0", "UTC+00:00"}:
        return UTC

    if match := _FIXED_OFFSET_RE.fullmatch(candidate.upper()):
        sign = -1 if match["sign"] == "-" else 1
        offset = timedelta(hours=int(match["h"]), minutes=int(match["m"] or 0))
        if offset > timedelta(hours=14):
            raise ValueError(f"timezone offset out of range: {name!r}")
        return timezone(sign * offset)

    try:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    except ImportError as exc:  # pragma: no cover - stdlib always present on 3.12+
        raise ValueError(f"cannot resolve timezone {name!r}") from exc
    try:
        return ZoneInfo(candidate)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(
            f"unknown timezone {name!r} (IANA database may be missing; try `pip install tzdata`)"
        ) from exc


class StrictModel(BaseModel):
    """Base for artifact schemas: unknown fields are an error, instances frozen.

    ``extra="forbid"`` matters for a source-of-truth document. A typo in a
    fixture key would otherwise be silently dropped and the resulting context
    would quietly lack a field an agent expects.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
        validate_assignment=True,
    )


class LenientModel(BaseModel):
    """Base for *input* schemas parsed from third-party payloads.

    Unknown keys are still rejected, but the model is mutable during
    normalization and tolerates the loose typing providers produce.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def utc_now() -> datetime:
    """Current time as an aware UTC datetime."""
    return datetime.now(UTC)


def validate_price_exponent(value: Decimal, field_name: str) -> Decimal:
    """Reject prices with absurd precision or non-finite values."""
    if not value.is_finite():
        raise ValueError(f"{field_name} must be a finite number")
    exponent = value.as_tuple().exponent
    if not isinstance(exponent, int):  # 'n', 'N' or 'F' for NaN / Infinity
        raise ValueError(f"{field_name} must be a finite number")
    decimal_places = -exponent
    if decimal_places > MAX_PRICE_EXPONENT:
        raise ValueError(
            f"{field_name} has {decimal_places} decimal places, maximum is {MAX_PRICE_EXPONENT}"
        )
    return value


__all__ = [
    "LenientModel",
    "Price",
    "StrictModel",
    "Timeframe",
    "UtcDatetime",
    "Volume",
    "normalize_symbol",
    "resolve_timezone",
    "utc_now",
    "validate_price_exponent",
]
