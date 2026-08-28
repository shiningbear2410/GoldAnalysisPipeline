"""Data quality reporting.

A *quality warning* is not a *validation error*:

* validation error -> the data contradicts itself, the Run fails, no context
  is written (see :mod:`goldpipeline.domain.errors`);
* quality warning  -> the data is usable but incomplete or was adjusted during
  normalization. The Run completes and the warning is recorded here so a
  downstream agent (and a human auditor) can see exactly what was degraded.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from goldpipeline.schemas.common import StrictModel


class QualityStatus(StrEnum):
    """Overall verdict on a Run's input data."""

    OK = "OK"
    WARNING = "WARNING"
    FAIL = "FAIL"


class WarningCode(StrEnum):
    """Closed set of warning codes, so consumers can branch on them safely."""

    BARS_REORDERED = "BARS_REORDERED"
    BAR_GAPS = "BAR_GAPS"
    DECLARED_RANGE_ADJUSTED = "DECLARED_RANGE_ADJUSTED"
    FUTURE_DATA = "FUTURE_DATA"
    LOW_BAR_COUNT = "LOW_BAR_COUNT"
    MISSING_VOLUME = "MISSING_VOLUME"
    MISSING_TELEGRAM_METADATA = "MISSING_TELEGRAM_METADATA"
    RAW_TEXT_SANITIZED = "RAW_TEXT_SANITIZED"
    RAW_TEXT_VERY_LONG = "RAW_TEXT_VERY_LONG"
    REQUESTED_AT_DEFAULTED = "REQUESTED_AT_DEFAULTED"
    STALE_DATA = "STALE_DATA"


class QualityWarning(StrictModel):
    """A single non-fatal observation about the input data."""

    code: WarningCode
    message: str = Field(description="Human-readable explanation.")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Structured context for the warning."
    )


class DataQuality(StrictModel):
    """Quality report embedded in every :class:`~goldpipeline.schemas.context.AnalysisContext`."""

    bar_count: int = Field(ge=0, description="Number of candles carried in the context.")
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Dotted paths of fields the source did not provide, e.g. 'ohlc.volume'.",
    )
    warnings: list[QualityWarning] = Field(default_factory=list)
    status: QualityStatus = Field(
        default=QualityStatus.OK, description="OK when no warnings, WARNING otherwise."
    )

    @classmethod
    def build(
        cls,
        *,
        bar_count: int,
        missing_fields: list[str],
        warnings: list[QualityWarning],
    ) -> DataQuality:
        """Assemble a report, deriving ``status`` from the warning list.

        ``FAIL`` is never produced here: a fatal problem raises before a context
        is built. The value exists in the enum for downstream stages.
        """
        return cls(
            bar_count=bar_count,
            missing_fields=sorted(set(missing_fields)),
            warnings=warnings,
            status=QualityStatus.WARNING if warnings else QualityStatus.OK,
        )


__all__ = ["DataQuality", "QualityStatus", "QualityWarning", "WarningCode"]
