"""An offline digest writer, for tests and dry runs.

Round 6.5c.2. The sibling of :mod:`~goldpipeline.adapters.fake_writer`, and it
exists for the same reason: every stage this pipeline dispatches must be
drivable end to end without a credential, a socket, or a bill.

**It answers from the prompt, not from a script.** The default behaviour reads
the collected items out of the fenced block it was given and selects the first
few, citing them exactly. That matters more here than it does for the analysis
fake: a digest's whole contract is that the editorial answer names only items
the prompt offered, and a fake that returned fixed ids would pass tests that
production would fail the moment the item ids changed.

The balance it writes carries no numbers at all, which is the shape the prompt
actually asks for and the shape the provenance rules leave open.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from goldpipeline.adapters.digest_writer_client import DigestRequest, DigestResponse
from goldpipeline.domain.errors import WriterError
from goldpipeline.schemas.news_digest import (
    MAX_DIGEST_ITEMS,
    DigestEditorial,
    DigestItem,
    ImpactMarker,
)
from goldpipeline.schemas.writer import NewsClaim, WriterStatus, WriterUsage

FAKE_PROVIDER = "fake"
FAKE_MODEL = "fake-digest-writer-v1"

DEFAULT_BALANCE = "Tin trong cửa sổ nghiêng tích cực, nhưng giá chưa xác nhận."
"""A view, not a count. The one sentence the section exists to produce."""

_ITEM_RE = re.compile(r'\{\s*"news_item_id":.*?\}', re.DOTALL)


@dataclass
class FakeDigestWriterClient:
    """Deterministic, offline implementation of ``DigestWriterClient``.

    Configure exactly one behaviour:

    * default - select the first collected items and cite them exactly;
    * ``raises`` - raise that error instead of answering;
    * ``output`` - return a specific :class:`DigestEditorial`, for contract
      violations such as a wrong run id or an uncollected item;
    * ``output_factory`` - compute the answer from the request.
    """

    output: DigestEditorial | None = None
    output_factory: Callable[[DigestRequest], DigestEditorial] | None = None
    raises: WriterError | None = None
    balance: str = DEFAULT_BALANCE
    items_to_select: int = 3
    usage: WriterUsage = field(
        default_factory=lambda: WriterUsage(input_tokens=900, output_tokens=260)
    )
    model_name: str = FAKE_MODEL
    calls: list[DigestRequest] = field(default_factory=list)
    """Every request seen, so tests can assert on what was actually sent."""

    @property
    def provider(self) -> str:
        return FAKE_PROVIDER

    @property
    def model(self) -> str:
        return self.model_name

    def generate(self, request: DigestRequest) -> DigestResponse:
        self.calls.append(request)
        if self.raises is not None:
            raise self.raises

        if self.output is not None:
            editorial = self.output
        elif self.output_factory is not None:
            editorial = self.output_factory(request)
        else:
            editorial = self._from_prompt(request)

        return DigestResponse(
            output=editorial,
            model=self.model_name,
            provider=FAKE_PROVIDER,
            usage=self.usage,
        )

    def _from_prompt(self, request: DigestRequest) -> DigestEditorial:
        """Select real items out of the prompt and quote them exactly."""
        offered = _collected_items(request.prompt.user)
        chosen = offered[: max(1, min(self.items_to_select, MAX_DIGEST_ITEMS))]

        items = tuple(
            DigestItem(
                news_item_id=item["news_item_id"],
                headline=_headline(item["text"]),
                impact=ImpactMarker.SUPPORTS_GOLD,
            )
            for item in chosen
        )
        claims = tuple(
            NewsClaim(
                statement=_headline(item["text"]),
                evidence=_headline(item["text"]),
                news_item_ids=[item["news_item_id"]],
            )
            for item in chosen
        )
        return DigestEditorial(
            run_id=request.run_id,
            status=WriterStatus.COMPLETED,
            items=items,
            balance=self.balance,
            news_claims=claims,
        )


def _collected_items(user_turn: str) -> list[dict[str, str]]:
    """Recover the offered items from the fenced block in the prompt.

    Parsed rather than assumed: a fake that invented ids would satisfy tests
    that production fails, because citing an item the prompt never offered is
    the exact failure `validate_editorial` exists to catch.
    """
    found: list[dict[str, str]] = []
    for match in _ITEM_RE.finditer(user_turn):
        try:
            record = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if "news_item_id" in record and "text" in record:
            found.append({"news_item_id": record["news_item_id"], "text": record["text"]})
    if not found:
        raise AssertionError(
            "the fake digest writer found no collected items in its prompt; it "
            "answers from what it was offered and has nothing to select from"
        )
    return found


def _headline(text: str) -> str:
    """One line, cut at a sentence boundary so the claim quotes the item exactly."""
    stripped = text.strip()
    stop = stripped.find(". ")
    return stripped if stop == -1 else stripped[: stop + 1]


__all__ = [
    "DEFAULT_BALANCE",
    "FAKE_MODEL",
    "FAKE_PROVIDER",
    "FakeDigestWriterClient",
]
