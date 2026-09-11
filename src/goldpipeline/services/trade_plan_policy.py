"""Every production choice a trade plan depends on, written down once.

Round 6.6h. Six rounds of engines were built with no defaults on purpose: every
config in the ICT branch requires its caller to choose, so that nobody could
ship a policy by forgetting to pass one. This module is where the choosing
finally happens, and it happens in version control rather than in a config file.

**Why not configuration.** A settings file can be edited between a Run and the
Run beside it, and then two trade plans built from the same market disagree for
a reason no artifact records. It is also the wrong shape: these are not
operator knobs like a timeout, they are the definition of what the product
*means* by an order block. Changing one changes every candidate id in the
system, so it deserves a version number and a commit, not a text field.

**Why not new ``ConfigKey`` members.** ``REQUIRED_PRODUCTION_KEYS`` is
``frozenset(ConfigKey)``, so a new member becomes mandatory for every existing
installation the moment it is added - and the scheduled worker would start
refusing to run until someone filled in a value it could have read from here.

**Versioned, not mutable.** ``V1`` is a name, not a default. A future policy is
``V2`` beside this one, and Runs written under V1 keep meaning what they meant,
because every Run persists the effective policy it used rather than a pointer to
whatever the current one happens to be.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import Any

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import ICT_TIMEFRAMES
from goldpipeline.services.ict_candidate_consolidation import canonical_price
from goldpipeline.services.ict_candidate_eligibility import CandidateEligibilityConfig
from goldpipeline.services.ict_composite import IctCompositeConfig
from goldpipeline.services.ict_fvg import FvgStatus
from goldpipeline.services.ict_order_block import OrderBlockZoneBasis
from goldpipeline.services.ict_order_block_lifecycle import (
    OrderBlockMitigationRule,
    OrderBlockStatus,
)

TRADE_PLAN_POLICY_VERSION = "trade_plan_production_policy_v1"
"""Stamped on every Run. A change here is a new version, not an edit."""

TRADE_PLAN_PROVIDER = "anthropic"
TRADE_PLAN_MODEL = "claude-sonnet-5"
"""The analyst's vendor and model. No DeepSeek, no OpenAI, no fallback.

A fallback would mean two different judges could rank the same candidates and
nothing on the Run would say which one did.
"""

TRADE_PLAN_PROVIDER_SYMBOL = "OANDA:XAUUSD"
"""TradingView's own name for the instrument. The single market authority."""


@dataclass(frozen=True)
class TradePlanProductionPolicyV1:
    """The complete production definition of a trade plan.

    Every field is explicit and none has a default, for the same reason the
    engines below have none: a value nobody typed is a value nobody chose.
    """

    version: str

    swing_left_bars: int
    swing_right_bars: int
    atr_period: int
    liquidity_price_tolerance: Decimal
    order_block_zone_basis: OrderBlockZoneBasis
    order_block_mitigation_rule: OrderBlockMitigationRule

    allowed_order_block_statuses: frozenset[OrderBlockStatus]
    allowed_fvg_statuses: frozenset[FvgStatus]

    provider_symbol: str
    bars_per_timeframe: int
    timeframes: tuple[Timeframe, ...]

    analyst_provider: str
    analyst_model: str

    def composite_config(self) -> IctCompositeConfig:
        """The ICT orchestration config, constructed explicitly.

        Never a default constructor - the composite has none, and this is the
        one place in production that supplies its six values.
        """
        return IctCompositeConfig(
            swing_left_bars=self.swing_left_bars,
            swing_right_bars=self.swing_right_bars,
            atr_period=self.atr_period,
            liquidity_price_tolerance=self.liquidity_price_tolerance,
            order_block_zone_basis=self.order_block_zone_basis,
            order_block_mitigation_rule=self.order_block_mitigation_rule,
        )

    def eligibility_config(self) -> CandidateEligibilityConfig:
        """The status policy, constructed explicitly.

        Terminal states are not expressible here: the config refuses
        ``INVALIDATED`` and ``FILLED`` at construction, and a swept pool is
        excluded by the eligibility engine itself. Those rules belong to the
        engines and are not policy this module may soften.
        """
        return CandidateEligibilityConfig(
            allowed_order_block_statuses=self.allowed_order_block_statuses,
            allowed_fvg_statuses=self.allowed_fvg_statuses,
        )

    def snapshot(self) -> dict[str, Any]:
        """The policy as a JSON-safe document, for the Run's audit trail.

        Written on every Run rather than referenced by version, so a plan
        remains readable after this file changes. Sets are rendered sorted so
        the document is byte-stable across processes.
        """
        return {
            "version": self.version,
            "swing_left_bars": self.swing_left_bars,
            "swing_right_bars": self.swing_right_bars,
            "atr_period": self.atr_period,
            "liquidity_price_tolerance": canonical_price(self.liquidity_price_tolerance),
            "order_block_zone_basis": self.order_block_zone_basis.value,
            "order_block_mitigation_rule": self.order_block_mitigation_rule.value,
            "allowed_order_block_statuses": sorted(
                status.value for status in self.allowed_order_block_statuses
            ),
            "allowed_fvg_statuses": sorted(status.value for status in self.allowed_fvg_statuses),
            "provider_symbol": self.provider_symbol,
            "bars_per_timeframe": self.bars_per_timeframe,
            "timeframes": [timeframe.value for timeframe in self.timeframes],
            "analyst_provider": self.analyst_provider,
            "analyst_model": self.analyst_model,
        }


PRODUCTION_POLICY_V1 = TradePlanProductionPolicyV1(
    version=TRADE_PLAN_POLICY_VERSION,
    swing_left_bars=2,
    swing_right_bars=2,
    atr_period=14,
    liquidity_price_tolerance=Decimal("0.50"),
    order_block_zone_basis=OrderBlockZoneBasis.FULL_CANDLE,
    order_block_mitigation_rule=OrderBlockMitigationRule.MIDPOINT,
    allowed_order_block_statuses=frozenset(
        {OrderBlockStatus.ACTIVE, OrderBlockStatus.TOUCHED, OrderBlockStatus.MITIGATED}
    ),
    allowed_fvg_statuses=frozenset({FvgStatus.OPEN, FvgStatus.TOUCHED}),
    provider_symbol=TRADE_PLAN_PROVIDER_SYMBOL,
    bars_per_timeframe=500,
    timeframes=ICT_TIMEFRAMES,
    analyst_provider=TRADE_PLAN_PROVIDER,
    analyst_model=TRADE_PLAN_MODEL,
)
"""The one policy production uses. Frozen, versioned, and persisted per Run."""


PLAN_NEWS_LOOKBACK = timedelta(hours=24)
"""How far back a plan's curated news reaches. Round 6.7.

Beside the policy rather than inside it: V1 describes how candidates are found,
and its snapshot on every Run must keep meaning exactly that. The window is
recorded on each Run separately, in the news document the page is rendered from.
"""


__all__ = [
    "PLAN_NEWS_LOOKBACK",
    "PRODUCTION_POLICY_V1",
    "TRADE_PLAN_MODEL",
    "TRADE_PLAN_POLICY_VERSION",
    "TRADE_PLAN_PROVIDER",
    "TRADE_PLAN_PROVIDER_SYMBOL",
    "TradePlanProductionPolicyV1",
]
