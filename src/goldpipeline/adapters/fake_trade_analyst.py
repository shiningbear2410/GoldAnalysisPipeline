"""Offline trade analysts, for proving the contract without a vendor.

Three shapes matter, and all three are here: one that answers correctly, one
that invents a candidate, and one that does not return JSON at all. A seam whose
only implementation is the happy path has not been tested; it has been described.

:class:`EchoTradeAnalyst` is the interesting one. It does not receive the
candidate set out of band - it *parses the request it was given*, exactly as a
real model would have to. That makes it a check on the serialiser as well as on
the validator: if the payload ever stopped carrying candidate ids, this fake
would start failing rather than quietly continuing to pass from a side channel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from goldpipeline.adapters.trade_analyst_client import (
    TradeAnalystRequest,
    TradeAnalystResponse,
)

BUCKET_KEYS = (
    "bai_entry_candidate_ids",
    "seo_entry_candidate_ids",
    "upper_reference_candidate_ids",
    "lower_reference_candidate_ids",
)


def _payload_of(request: TradeAnalystRequest) -> dict[str, Any]:
    """Read the fenced JSON document back out of the rendered user turn."""
    opening = request.user.index("<CANDIDATE_DATA>") + len("<CANDIDATE_DATA>")
    closing = request.user.index("</CANDIDATE_DATA>")
    parsed: dict[str, Any] = json.loads(request.user[opening:closing])
    return parsed


@dataclass(frozen=True)
class EchoTradeAnalyst:
    """Returns the deterministic buckets, optionally reversed.

    Reversing is how a test says "the model had an opinion": the answer is a
    genuinely different order from the one the payload listed, while still being
    a complete and valid permutation.
    """

    reverse: bool = False
    provider: str = "fake"
    model: str = "echo"

    def rank(self, request: TradeAnalystRequest) -> TradeAnalystResponse:
        buckets = _payload_of(request)["buckets"]
        answer = {
            key: list(reversed(buckets[key])) if self.reverse else list(buckets[key])
            for key in BUCKET_KEYS
        }
        return TradeAnalystResponse(
            text=json.dumps(answer), model=self.model, provider=self.provider
        )


@dataclass(frozen=True)
class HallucinatingTradeAnalyst:
    """Adds a candidate that does not exist, in the bucket of the caller's choice."""

    invented_id: str = "deadbeefdeadbeef"
    bucket: str = "bai_entry_candidate_ids"
    provider: str = "fake"
    model: str = "hallucinating"

    def rank(self, request: TradeAnalystRequest) -> TradeAnalystResponse:
        buckets = _payload_of(request)["buckets"]
        answer = {key: list(buckets[key]) for key in BUCKET_KEYS}
        answer[self.bucket] = [self.invented_id, *answer[self.bucket]]
        return TradeAnalystResponse(
            text=json.dumps(answer), model=self.model, provider=self.provider
        )


@dataclass(frozen=True)
class MalformedTradeAnalyst:
    """Returns something that is not the contract at all.

    The default is the failure a real model actually produces: a correct-looking
    JSON object wrapped in prose and a code fence, which is not JSON.
    """

    text: str = 'Here is my ranking:\n```json\n{"bai_entry_candidate_ids": []}\n```'
    provider: str = "fake"
    model: str = "malformed"

    def rank(self, request: TradeAnalystRequest) -> TradeAnalystResponse:
        return TradeAnalystResponse(text=self.text, model=self.model, provider=self.provider)


@dataclass(frozen=True)
class ScriptedTradeAnalyst:
    """Returns one exact string, whatever it is. For the rejection matrix."""

    text: str
    provider: str = "fake"
    model: str = "scripted"

    def rank(self, request: TradeAnalystRequest) -> TradeAnalystResponse:
        return TradeAnalystResponse(text=self.text, model=self.model, provider=self.provider)


__all__ = [
    "BUCKET_KEYS",
    "EchoTradeAnalyst",
    "HallucinatingTradeAnalyst",
    "MalformedTradeAnalyst",
    "ScriptedTradeAnalyst",
]
