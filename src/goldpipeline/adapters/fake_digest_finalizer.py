"""An offline digest finalizer, for tests and dry runs.

Round 6.5c.3. The last of the four offline clients, and the one whose default
behaviour needed the most care.

**It repairs from the prompt, not from a script.** The default reads the current
editorial out of the prompt it was given and answers every finding it was shown:
content issues get an `APPLIED` resolution, style findings get `RESOLVED`, and
the editorial comes back with the balance rewritten to a short, numberless view.
A fake that returned a fixed answer would pass tests that production fails the
moment an item id changes - which is exactly the failure the digest writer's
fake was built to avoid.

The default repair is deliberately conservative: it touches the balance and
leaves the items alone. That is what a well-behaved repair looks like for the
common style finding, and a test that wants something else configures it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from goldpipeline.adapters.digest_finalizer_client import (
    DigestFinalizeRequest,
    DigestFinalizeResponse,
)
from goldpipeline.domain.errors import FinalizeError
from goldpipeline.schemas.digest_finalizer import (
    DigestEditorialRevision,
    DigestFinalizerModelOutput,
)
from goldpipeline.schemas.finalizer import (
    FinalizerUsage,
    IssueResolution,
    ResolutionStatus,
    StyleResolution,
    StyleResolutionStatus,
)
from goldpipeline.schemas.news_digest import DigestItem, ImpactMarker
from goldpipeline.schemas.writer import NewsClaim, WriterStatus

FAKE_PROVIDER = "fake"
FAKE_MODEL = "fake-digest-finalizer-v1"

DEFAULT_BALANCE = "Tin nghiêng tích cực, nhưng giá chưa xác nhận."
"""Short, a view rather than a count, and carrying no number to be checked."""

_BLOCK_RE = re.compile(r"```json\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


@dataclass
class FakeDigestFinalizerClient:
    """Deterministic, offline implementation of ``DigestFinalizerClient``.

    Configure exactly one behaviour:

    * default - answer every supplied finding and compress the balance;
    * ``raises`` - raise that error instead of answering;
    * ``output`` - return a specific :class:`DigestFinalizerModelOutput`, for
      the failure matrix: an unknown item id, a re-quantified figure, an
      unresolved finding;
    * ``output_factory`` - compute the answer from the request.
    """

    output: DigestFinalizerModelOutput | None = None
    output_factory: Callable[[DigestFinalizeRequest], DigestFinalizerModelOutput] | None = None
    raises: FinalizeError | None = None
    balance: str = DEFAULT_BALANCE
    style_status: StyleResolutionStatus = StyleResolutionStatus.RESOLVED
    issue_status: ResolutionStatus = ResolutionStatus.APPLIED
    usage: FinalizerUsage = field(
        default_factory=lambda: FinalizerUsage(input_tokens=1400, output_tokens=320)
    )
    model_name: str = FAKE_MODEL
    calls: list[DigestFinalizeRequest] = field(default_factory=list)
    """Every request seen. The one-call invariant is asserted against its length."""

    @property
    def provider(self) -> str:
        return FAKE_PROVIDER

    @property
    def model(self) -> str:
        return self.model_name

    def finalize(self, request: DigestFinalizeRequest) -> DigestFinalizeResponse:
        self.calls.append(request)
        if self.raises is not None:
            raise self.raises

        if self.output is not None:
            output = self.output
        elif self.output_factory is not None:
            output = self.output_factory(request)
        else:
            output = self._from_prompt(request)

        return DigestFinalizeResponse(
            output=output,
            model=self.model_name,
            provider=FAKE_PROVIDER,
            usage=self.usage,
        )

    def _from_prompt(self, request: DigestFinalizeRequest) -> DigestFinalizerModelOutput:
        """Answer the findings in the prompt, keeping the items as they were."""
        user = request.prompt.user
        editorial = _first_object(user)
        issues = _issue_ids(user)
        findings = _finding_ids(user)

        items = tuple(
            DigestItem(
                news_item_id=entry["news_item_id"],
                headline=entry["headline"],
                note=entry.get("note"),
                impact=ImpactMarker(entry["impact"]),
            )
            for entry in editorial["items"]
        )
        claims = tuple(
            NewsClaim(
                statement=claim["statement"],
                evidence=claim["evidence"],
                news_item_ids=list(claim["news_item_ids"]),
            )
            for claim in editorial.get("news_claims", [])
        )

        return DigestFinalizerModelOutput(
            run_id=request.run_id,
            status=WriterStatus.COMPLETED,
            editorial=DigestEditorialRevision(
                items=items, balance=self.balance, news_claims=claims
            ),
            issue_resolutions=[
                IssueResolution(
                    issue_id=issue_id,
                    resolution=self.issue_status,
                    description="Đã sửa theo yêu cầu của người kiểm duyệt.",
                )
                for issue_id in issues
            ],
            style_resolutions=[
                StyleResolution(
                    finding_id=finding_id,
                    status=self.style_status,
                    note="Đã rút gọn phần cán cân.",
                )
                for finding_id in findings
            ],
        )


def _first_object(user_turn: str) -> dict[str, Any]:
    """The editorial block, parsed out of the prompt's first JSON fence."""
    for match in _BLOCK_RE.finditer(user_turn):
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "items" in parsed and "balance" in parsed:
            return parsed
    raise AssertionError(
        "the fake digest finalizer found no editorial content in its prompt; it "
        "repairs what it was given and has nothing to work from"
    )


def _issue_ids(user_turn: str) -> list[str]:
    return _ids_from_blocks(user_turn, "issue_id")


def _finding_ids(user_turn: str) -> list[str]:
    return _ids_from_blocks(user_turn, "finding_id")


def _ids_from_blocks(user_turn: str, key: str) -> list[str]:
    """Every ``key`` in the prompt's JSON arrays, in order.

    Read from the prompt rather than accepted as a constructor argument, so a
    test that changes the findings does not also have to remember to change the
    fake's answer - and so a resolution list can never accidentally be complete
    for findings that were never sent.
    """
    found: list[str] = []
    for match in _BLOCK_RE.finditer(user_turn):
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            found.extend(entry[key] for entry in parsed if isinstance(entry, dict) and key in entry)
    return found


__all__ = [
    "DEFAULT_BALANCE",
    "FAKE_MODEL",
    "FAKE_PROVIDER",
    "FakeDigestFinalizerClient",
]
