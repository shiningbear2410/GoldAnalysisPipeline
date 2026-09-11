"""The provider seam for the Plan Copywriter.

Round 6.7. The same shape as the Trade Analyst's seam - two strings in, one
string out - and deliberately the same request and response types, because the
transport really is identical. What differs is the method name, so that a
ranking client can never be passed where a copywriter is expected by accident.
"""

from __future__ import annotations

from typing import Protocol

from goldpipeline.adapters.trade_analyst_client import (
    TradeAnalystRequest,
    TradeAnalystResponse,
)


class PlanCopyClient(Protocol):
    """Anything that can turn the copy prompt into text."""

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    def write(self, request: TradeAnalystRequest) -> TradeAnalystResponse: ...


__all__ = ["PlanCopyClient"]
