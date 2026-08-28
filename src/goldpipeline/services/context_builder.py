"""Assemble the AnalysisContext from normalized inputs.

This stage is pure: it takes already-validated data and rearranges it. It adds
no facts and derives nothing beyond lifting the latest candle into
``context.price``. Any interpretation - bias, levels, indicators - is out of
scope for Round 1 and belongs to the agents, not to this document.
"""

from __future__ import annotations

from goldpipeline.schemas.common import utc_now
from goldpipeline.schemas.context import (
    AnalysisContext,
    ContextMarket,
    ContextOHLC,
    ContextPrice,
    ContextRawAnalysis,
    ContextTiming,
)
from goldpipeline.schemas.quality import DataQuality
from goldpipeline.services.normalizer import NormalizedAnalysis, NormalizedMarketData


def build_context(
    *,
    run_id: str,
    market: NormalizedMarketData,
    analysis: NormalizedAnalysis,
    generated_at: object | None = None,
) -> AnalysisContext:
    """Combine normalized market data and analysis into a single context.

    Args:
        run_id: Identifier of the Run this context belongs to.
        market: Output of :func:`~goldpipeline.services.normalizer.normalize_market_data`.
        analysis: Output of :func:`~goldpipeline.services.normalizer.normalize_analysis`.
        generated_at: Injection point for tests; defaults to now (UTC).
    """
    snapshot = market.snapshot
    latest = snapshot.latest_bar
    message = analysis.analysis

    quality = DataQuality.build(
        bar_count=snapshot.bar_count,
        missing_fields=[*market.missing_fields, *analysis.missing_fields],
        warnings=[*market.warnings, *analysis.warnings],
    )

    return AnalysisContext(
        run_id=run_id,
        market=ContextMarket(
            symbol=snapshot.symbol,
            timeframe=snapshot.timeframe,
            provider=snapshot.provider,
            timezone=snapshot.timezone,
            source_timezone=snapshot.source_timezone,
        ),
        timing=ContextTiming(
            generated_at=generated_at or utc_now(),
            requested_at=snapshot.requested_at,
            data_from=snapshot.data_from,
            data_to=snapshot.data_to,
            latest_candle_at=latest.timestamp,
        ),
        price=ContextPrice(
            latest_open=latest.open,
            latest_high=latest.high,
            latest_low=latest.low,
            latest_close=latest.close,
        ),
        raw_analysis=ContextRawAnalysis(
            text=message.raw_text,
            source=message.source,
            chat_id=message.chat_id,
            message_id=message.message_id,
            message_date=message.message_date,
            author=message.author.model_dump() if message.author else None,
        ),
        ohlc=ContextOHLC(bar_count=snapshot.bar_count, bars=list(snapshot.bars)),
        data_quality=quality,
    )


__all__ = ["build_context"]
