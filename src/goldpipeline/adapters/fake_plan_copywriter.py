"""Offline Plan Copywriters.

:class:`EchoPlanCopywriter` reads the fenced payload it was sent, exactly as a
real model has to, and answers with a valid, digit-free commentary about the
zones it found there. :class:`ScriptedPlanCopywriter` returns one exact string,
for the rejection matrix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from goldpipeline.adapters.trade_analyst_client import (
    TradeAnalystRequest,
    TradeAnalystResponse,
)

MARKET_VIEW = (
    "Cấu trúc khung lớn vẫn nghiêng về phe mua, còn khung nhỏ đang giằng co quanh "
    "vùng cân bằng nên chưa có lợi thế rõ cho việc đuổi giá. Ưu tiên kiên nhẫn chờ "
    "phản ứng tại các vùng đã đánh dấu, quan sát cách nến đóng trên H1 trước khi "
    "hành động và giữ kỷ luật với kế hoạch."
)


def plan_payload_of(request: TradeAnalystRequest) -> dict[str, Any]:
    """Read the fenced JSON document back out of the rendered user turn."""
    opening = request.user.index("<PLAN_DATA>") + len("<PLAN_DATA>")
    closing = request.user.index("</PLAN_DATA>")
    parsed: dict[str, Any] = json.loads(request.user[opening:closing])
    return parsed


@dataclass(frozen=True)
class EchoPlanCopywriter:
    """A valid answer about whatever zones the payload carried."""

    news_count: int = 2
    provider: str = "fake"
    model: str = "echo-copy"

    def write(self, request: TradeAnalystRequest) -> TradeAnalystResponse:
        payload = plan_payload_of(request)
        zones = [*payload["seo_zones"], *payload["bai_zones"]]
        answer = {
            "market_view": MARKET_VIEW,
            "seo_heading_note": "chờ hồi lên mới tính" if payload["seo_zones"] else "",
            "bai_heading_note": "ưu tiên canh mua" if payload["bai_zones"] else "",
            "zone_notes": {
                zone["candidate_id"]: (
                    "hợp lưu nhiều khung" if zone["is_vung_chu_dao"] else "vùng phản ứng phụ"
                )
                for zone in zones
            },
            "reference_notes": {
                reference["candidate_id"]: "chỉ để tham chiếu"
                for reference in payload["references"]
            },
            "news_item_ids": [item["news_item_id"] for item in payload["news_items"]][
                : self.news_count
            ],
        }
        return TradeAnalystResponse(
            text=json.dumps(answer, ensure_ascii=False), model=self.model, provider=self.provider
        )


@dataclass(frozen=True)
class ScriptedPlanCopywriter:
    """Returns one exact string, whatever it is."""

    text: str
    provider: str = "fake"
    model: str = "scripted-copy"

    def write(self, request: TradeAnalystRequest) -> TradeAnalystResponse:
        return TradeAnalystResponse(text=self.text, model=self.model, provider=self.provider)


__all__ = ["MARKET_VIEW", "EchoPlanCopywriter", "ScriptedPlanCopywriter", "plan_payload_of"]
