"""Building a digest's deterministic facts, from an event to a rendered shell.

Round 6.5b. This is where the digest gets its own market data.

**Why not reuse the analysis series.** An ANALYSIS Run has already loaded M15
candles, and reaching for them would be free. It would also be wrong: at
fifteen minutes a one-hour digest window has four bars, and a digest is
routinely asked about one hour. The digest asks for M5 of its own, sized from
its own window - see
:func:`~goldpipeline.services.price_reaction.digest_bar_count`.

**Failure is explicit.** If the candles cannot be fetched the digest stops. It
does not fall back to another provider, does not quietly omit the price
section, and does not publish a digest whose one arithmetic section is missing.

A market that was *shut* is a different matter entirely: an empty weekend
window is a valid observation with its own state and its own sentence, and it
is not a failure of anything.
"""

from __future__ import annotations

import logging

from goldpipeline.adapters.base import MarketDataSource
from goldpipeline.domain.errors import MarketDataError, PipelineError
from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.digest import DigestWindow
from goldpipeline.schemas.news_digest import DigestSourceItem
from goldpipeline.services.digest_context import DigestFacts, build_digest_facts
from goldpipeline.services.normalizer import normalize_market_data
from goldpipeline.services.price_reaction import (
    PREFERRED_DIGEST_TIMEFRAME,
    calculate_price_reaction,
    digest_bar_count,
)

logger = logging.getLogger(__name__)


class DigestMarketDataError(MarketDataError):
    """The digest could not obtain candles it can describe a window with.

    Distinct from an empty window. ``NO_MARKET_ACTIVITY`` means the provider
    answered and there was nothing trading; this means the provider did not
    answer, or answered with something unusable. The first is publishable, the
    second stops the Run.
    """


def build_digest_facts_for_window(
    *,
    window: DigestWindow,
    market_source: MarketDataSource,
    symbol: str,
    news_items: tuple[DigestSourceItem, ...] = (),
    timeframe: Timeframe = PREFERRED_DIGEST_TIMEFRAME,
) -> DigestFacts:
    """Fetch the digest's own candles and turn them into deterministic facts.

    Args:
        window: The immutable span, already fixed by the producer's request.
        market_source: Any provider-neutral source. Production passes
            TradingView; tests pass an offline stand-in, and the arithmetic
            cannot tell the difference.
        symbol: Canonical instrument symbol.
        news_items: The curated items the writer may choose from.
        timeframe: Which series to describe the window with. Defaults to the
            digest's own preferred timeframe, never the analysis one.

    Raises:
        DigestMarketDataError: The provider failed, or returned nothing usable.
    """
    bars_wanted = digest_bar_count(window, timeframe)
    logger.info(
        "digest.market request symbol=%s timeframe=%s bars=%d window=%s..%s",
        symbol,
        timeframe,
        bars_wanted,
        window.start.isoformat(),
        window.end.isoformat(),
    )

    try:
        loaded = market_source.load()
    except MarketDataError:
        raise
    except Exception as exc:  # noqa: BLE001 - any provider fault is one failure here
        raise DigestMarketDataError(
            f"the digest could not fetch {timeframe} candles for {symbol}",
            symbol=symbol,
            timeframe=str(timeframe),
        ) from exc

    if not loaded.model.bars:
        # Not the same as a quiet market: a quiet market still returns candles
        # from before the window, which is what a start reference is made of.
        raise DigestMarketDataError(
            f"the digest received no {timeframe} candles for {symbol}",
            symbol=symbol,
            timeframe=str(timeframe),
        )

    # Normalized rather than read raw. A provider's timestamps may be naive, and
    # the arithmetic here compares them against an aware UTC window - the
    # normalizer is the one place that already knows how to make that safe, and
    # it checks the symbol and the ordering on the way past.
    try:
        normalized = normalize_market_data(loaded.model, expected_symbol=symbol, now=window.end)
    except PipelineError as exc:
        raise DigestMarketDataError(
            f"the {timeframe} candles for {symbol} could not be normalized",
            symbol=symbol,
            timeframe=str(timeframe),
        ) from exc

    snapshot = normalized.snapshot
    reaction = calculate_price_reaction(
        snapshot.bars,
        timeframe=timeframe,
        window=window,
        symbol=snapshot.symbol,
        provider=snapshot.provider,
    )
    logger.info(
        "digest.market ok symbol=%s bars=%d activity=%s",
        symbol,
        snapshot.bar_count,
        reaction.market_activity,
    )

    return build_digest_facts(
        window=window,
        price_reaction=reaction,
        symbol=snapshot.symbol,
        timeframe=timeframe,
        news_items=news_items,
    )


__all__ = ["DigestMarketDataError", "build_digest_facts_for_window"]
