"""Offline writer client.

Every test in this repository runs against this. It exists so the suite never
needs a network, a key, or a budget, and so failure modes that are awkward to
provoke against a real provider - a timeout, a mismatched run id, an empty
article - can be triggered exactly and deterministically.

It is also what ``--fake-writer`` uses, which makes the CLI path itself
smoke-testable end to end.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from goldpipeline.adapters.writer_client import WriterRequest, WriterResponse
from goldpipeline.domain.errors import (
    WriterError,
    WriterProviderError,
    WriterResponseError,
    WriterTimeoutError,
)
from goldpipeline.schemas.writer import (
    ClaimType,
    SourceClaim,
    WriterModelOutput,
    WriterStatus,
    WriterUsage,
)

FAKE_PROVIDER = "fake"
FAKE_MODEL = "fake-writer-v1"


NO_DRIVER = "Chưa thấy gì đáng kể."
"""What the contract requires when a side has no material driver."""

DISCLAIMER = "🔴 Nhận định cá nhân, không phải lời khuyên đầu tư."


def _default_article(request: WriterRequest) -> str:
    """A plausible Vietnamese draft built only from what the prompt carried.

    Deliberately assembled from the prompt rather than hardcoded, so a smoke run
    shows real numbers from the Run rather than a fixed sample that would hide a
    plumbing mistake.

    It emits the ``ANALYSIS`` shape the deterministic contract enforces, for the
    same reason `fake_mt5` models a real terminal's awkward parts: a double that
    produces something production would refuse is a double that lets a broken
    pipeline pass its own tests. The bearish side is deliberately the
    placeholder - an offline draft has no drivers to report, and inventing one
    to fill the section is exactly what the contract forbids.
    """
    facts = _extract_facts(request)
    close = facts.get("close", "n/a")
    date = facts.get("article_date", "")

    return "\n".join(
        [
            f"🕯 PHÂN TÍCH VÀNG — {date}",
            "",
            "⚡ Chốt: bản nháp offline, chưa có hướng nào rõ để nghiêng theo.",
            "",
            "🟢 Đẩy lên:",
            NO_DRIVER,
            "",
            "🔴 Kéo xuống:",
            NO_DRIVER,
            "",
            "📈 Giá đang nói gì?",
            f"Giá gần nhất quanh {close}. Chưa đủ để nói bên nào đang thắng.",
            "",
            "🧭 Mình đang chờ:",
            "Thêm dữ liệu thật. Đây là bản nháp do writer offline dựng ra để kiểm thử.",
            "",
            DISCLAIMER,
        ]
    )


def _extract_facts(request: WriterRequest) -> dict[str, str]:
    """Pull a few display values back out of the rendered prompt.

    Parsing our own prompt is fine here - this is a test double, and reading the
    values it was actually given is what makes a fake smoke run meaningful.
    """
    import json
    import re

    match = re.search(r"```json\n(\{.*?\})\n```", request.prompt.user, re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}

    candle = payload.get("latest_candle", {})
    instrument = payload.get("instrument", {})
    return {
        "close": str(candle.get("close", "")),
        "high": str(candle.get("high", "")),
        "low": str(candle.get("low", "")),
        "symbol": str(instrument.get("symbol", "")),
        "timeframe": str(instrument.get("timeframe", "")),
        # Copied, never computed - the same rule the real writer is held to.
        "article_date": str(payload.get("article_date", "")),
    }


@dataclass
class FakeWriterClient:
    """Deterministic, offline implementation of :class:`WriterClient`.

    Configure exactly one behaviour:

    * default - build a coherent draft from the prompt;
    * ``raises`` - raise that error instead of answering, for provider,
      timeout and configuration failures;
    * ``output`` - return a specific :class:`WriterModelOutput`, for contract
      violations such as a wrong run id or an empty article;
    * ``output_factory`` - compute the output from the request.
    """

    output: WriterModelOutput | None = None
    output_factory: Callable[[WriterRequest], WriterModelOutput] | None = None
    raises: WriterError | None = None
    usage: WriterUsage = field(
        default_factory=lambda: WriterUsage(input_tokens=1200, output_tokens=480)
    )
    model_name: str = FAKE_MODEL
    calls: list[WriterRequest] = field(default_factory=list)
    """Every request seen, so tests can assert on what was actually sent."""

    @property
    def provider(self) -> str:
        return FAKE_PROVIDER

    @property
    def model(self) -> str:
        return self.model_name

    def generate(self, request: WriterRequest) -> WriterResponse:
        """Return the configured response, or a generated draft."""
        self.calls.append(request)

        if self.raises is not None:
            raise self.raises

        if self.output is not None:
            output = self.output
        elif self.output_factory is not None:
            output = self.output_factory(request)
        else:
            output = self._build_default(request)

        return WriterResponse(
            output=output, model=self.model, provider=self.provider, usage=self.usage
        )

    def _build_default(self, request: WriterRequest) -> WriterModelOutput:
        facts = _extract_facts(request)
        claims = [
            SourceClaim(
                type=ClaimType.PRICE,
                value=facts.get("close", "unknown"),
                source="context.price.latest_close",
            ),
            SourceClaim(
                type=ClaimType.MARKET_META,
                value=facts.get("symbol", "XAUUSD"),
                source="context.market.symbol",
            ),
        ]
        return WriterModelOutput(
            run_id=request.run_id,
            status=WriterStatus.COMPLETED,
            title="Phân tích vàng - bản nháp offline",
            article=_default_article(request),
            source_claims=claims,
            warnings=[],
        )


def failing_client(error: WriterError) -> FakeWriterClient:
    """A client that always raises *error*."""
    return FakeWriterClient(raises=error)


def timing_out_client(seconds: float = 120.0) -> FakeWriterClient:
    """A client that always times out."""
    return failing_client(
        WriterTimeoutError(f"provider did not respond within {seconds:g}s", timeout_seconds=seconds)
    )


def erroring_client(message: str = "provider returned HTTP 500") -> FakeWriterClient:
    """A client that always reports a provider failure."""
    return failing_client(WriterProviderError(message, status_code=500))


def malformed_client(message: str = "response was not valid JSON") -> FakeWriterClient:
    """A client that always reports an unparseable answer."""
    return failing_client(WriterResponseError(message))


__all__ = [
    "DISCLAIMER",
    "FAKE_MODEL",
    "FAKE_PROVIDER",
    "NO_DRIVER",
    "FakeWriterClient",
    "erroring_client",
    "failing_client",
    "malformed_client",
    "timing_out_client",
]
