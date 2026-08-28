"""Deterministic checks run before the reviewer model is consulted.

Everything here is computed in Python from the Run's own artifacts. It costs
nothing, it cannot hallucinate, and it does not depend on a model noticing
anything - so it runs first, its findings go into the prompt, and they are
recorded on the review artifact whether or not the reviewer echoes them.

What it checks:

* **claims** - every ``source_claims`` entry the writer recorded is resolved
  against the context and compared;
* **numbers** - price-like numbers in the article that no context value,
  verified claim, or in-range analyst level accounts for;
* **instrument** - a foreign symbol appearing in an XAUUSD article;
* **indicators** - RSI, MACD and friends, which the context never carries, so
  any value stated for them was invented;
* **risk language** - certainty claims a market commentary must not make.

These are heuristics with a stated bias: when uncertain, warn rather than
condemn. Only findings that are unambiguous - a claim that resolves to a
different number, a foreign instrument, an invented indicator - reach a severity
that can block a PASS.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal

from goldpipeline.schemas.context import AnalysisContext
from goldpipeline.schemas.review import (
    BLOCKING_SEVERITIES,
    FindingCode,
    PrecheckFinding,
    Severity,
)
from goldpipeline.schemas.writer import WriterResult
from goldpipeline.services.claim_resolver import ResolvedClaim, verify_claims
from goldpipeline.services.market_facts import format_price
from goldpipeline.services.source_guard import (
    PRICE_TOLERANCE,
    RANGE_TOLERANCE,
    NumberMatch,
    extract_numbers,
)

OUTSIDE_RANGE_TOLERANCE = Decimal("0.05")
"""How far outside the candle range a number may sit before it is called wrong.

Wider than the writer's own guard (0.5% of price): by this point the number is
in a published article, and the question is no longer "did the analyst mention
an odd level" but "is this number recognisably about this market at all". 5% of
gold is ~165 points - a typo like 3325 for 3315 stays inside and is reported as
merely unexplained, while 9999 lands far outside and is called out.
"""

_YEAR_RANGE = range(1900, 2101)
"""Bare four-digit integers in this range are read as years, not prices.

A genuine XAUUSD quote inside it would be ambiguous to a human reader too.
"""

_UNIT_WORDS = (
    "%",
    "gio",
    "phut",
    "giay",
    "ngay",
    "tuan",
    "thang",
    "nam",
    "nen",
    "phien",
    "lot",
    "lan",
    "kich ban",
    "diem",
    "pip",
    "point",
    "bar",
    "candle",
)

_UNIT_SUFFIX_RE = re.compile(
    r"^\s*(" + "|".join(re.escape(word) for word in _UNIT_WORDS) + ")",
    re.IGNORECASE,
)
"""Words that make a preceding number a duration or a count, not a price."""

_INDICATORS = (
    "RSI",
    "MACD",
    "EMA",
    "SMA",
    "WMA",
    "Bollinger",
    "Fibonacci",
    "Stochastic",
    "Ichimoku",
    "ATR",
    "ADX",
    "CCI",
    "OBV",
    "VWAP",
)
"""Indicators the context never carries, so any value stated for one is invented."""

_KNOWN_SYMBOLS = (
    "BTCUSD",
    "ETHUSD",
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "XAGUSD",
    "NAS100",
    "US30",
    "SPX500",
    "DXY",
)
"""Instruments that have no business appearing in an XAUUSD commentary."""

_ABSOLUTE_PHRASES = (
    "chac chan thang",
    "chac chan tang",
    "chac chan giam",
    "chac chan se",
    "khong the tang",
    "khong the giam",
    "khong the thua",
    "khong bao gio giam",
    "khong bao gio tang",
    "cam ket loi nhuan",
    "dam bao loi nhuan",
    "dam bao thang",
    "100% tang",
    "100% giam",
    "chac thang",
    "guaranteed",
)
"""Certainty claims. Market commentary states scenarios, not promises."""


@dataclass
class PrecheckReport:
    """Everything the deterministic pass established."""

    findings: list[PrecheckFinding] = field(default_factory=list)
    resolved_claims: list[ResolvedClaim] = field(default_factory=list)
    market_low: Decimal | None = None
    market_high: Decimal | None = None

    @property
    def blocking(self) -> list[PrecheckFinding]:
        """Findings severe enough that a PASS verdict cannot stand."""
        return [finding for finding in self.findings if finding.is_blocking]

    @property
    def has_blocking(self) -> bool:
        """Whether any finding rules out a PASS."""
        return bool(self.blocking)

    @property
    def worst_severity(self) -> Severity | None:
        """The most severe finding, or ``None`` when the pass was clean."""
        order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        present = [f.severity for f in self.findings]
        return max(present, key=order.index) if present else None


def _fold(text: str) -> str:
    """Strip diacritics and case, so Vietnamese phrases match however typed."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D").casefold()


def run_prechecks(
    *,
    context: AnalysisContext,
    writer_result: WriterResult,
    article: str,
    check_claims: bool = True,
) -> PrecheckReport:
    """Run the deterministic checks over an article.

    Args:
        context: The Run's source of truth.
        writer_result: The writer's own record, including its ``source_claims``.
        article: The text to check.
        check_claims: Whether to verify ``source_claims`` against the context.

    ``check_claims=False`` exists for the finalizer. ``source_claims`` describe
    what the *writer* used and live in an immutable artifact the finalizer may
    not rewrite, so a claim mismatch survives a correct revision by design.
    Re-running that check over a fixed article would report a flaw that is no
    longer in the text, and no revision could ever clear it. The article-derived
    checks below are the ones that say something about the article itself.
    """
    bars = context.ohlc.bars
    low = min(bar.low for bar in bars)
    high = max(bar.high for bar in bars)

    report = PrecheckReport(market_low=low, market_high=high)

    if check_claims:
        report.resolved_claims = verify_claims(context, list(writer_result.source_claims))
        report.findings.extend(_check_claims(report.resolved_claims))

    report.findings.extend(_check_numbers(context, writer_result, article, low, high))
    report.findings.extend(_check_symbols(context, article))
    report.findings.extend(_check_indicators(article))
    report.findings.extend(_check_risk_language(article))
    return report


# --------------------------------------------------------------------------
# claims
# --------------------------------------------------------------------------


def _check_claims(resolved: list[ResolvedClaim]) -> list[PrecheckFinding]:
    if not resolved:
        return [
            PrecheckFinding(
                code=FindingCode.NO_SOURCE_CLAIMS,
                severity=Severity.MEDIUM,
                message=(
                    "The draft recorded no source_claims, so none of its numbers can be "
                    "traced back to the context automatically."
                ),
            )
        ]

    findings: list[PrecheckFinding] = []
    for item in resolved:
        if item.error is not None:
            findings.append(
                PrecheckFinding(
                    code=FindingCode.CLAIM_SOURCE_NOT_FOUND,
                    severity=Severity.HIGH,
                    message=(
                        f"Claim cites {item.claim.source!r}, which does not resolve in the "
                        f"context: {item.error}"
                    ),
                    source_path=item.claim.source,
                    actual=item.claim.value,
                )
            )
        elif not item.matches:
            findings.append(
                PrecheckFinding(
                    code=FindingCode.CLAIM_VALUE_MISMATCH,
                    severity=Severity.HIGH,
                    message=(
                        f"Claim states {item.claim.value!r} for {item.claim.source}, but the "
                        f"context holds {item.resolved!r}."
                    ),
                    source_path=item.claim.source,
                    expected=item.resolved,
                    actual=item.claim.value,
                )
            )
    return findings


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------


def _allowed_values(
    context: AnalysisContext,
    writer_result: WriterResult,
    low: Decimal,
    high: Decimal,
) -> set[Decimal]:
    """Every number the article is entitled to state.

    Analyst levels count only when they sit near the market. An out-of-range
    number in the source is exactly what Round 2 warned about, and repeating it
    in the article is worth flagging rather than waving through.
    """
    allowed: set[Decimal] = set()
    for bar in context.ohlc.bars:
        allowed.update({bar.open, bar.high, bar.low, bar.close})

    for claim in writer_result.source_claims:
        try:
            allowed.add(Decimal(claim.value.replace(",", "")))
        except (ArithmeticError, ValueError):
            continue

    margin = max((high - low) * RANGE_TOLERANCE, high * PRICE_TOLERANCE)
    for match in extract_numbers(context.raw_analysis.text):
        if low - margin <= match.value <= high + margin:
            allowed.add(match.value)

    return allowed


def _check_numbers(
    context: AnalysisContext,
    writer_result: WriterResult,
    article: str,
    low: Decimal,
    high: Decimal,
) -> list[PrecheckFinding]:
    allowed = _allowed_values(context, writer_result, low, high)
    outer = max((high - low) * RANGE_TOLERANCE, high * OUTSIDE_RANGE_TOLERANCE)

    findings: list[PrecheckFinding] = []
    reported: set[Decimal] = set()

    for match in extract_numbers(article):
        if match.value in allowed or match.value in reported:
            continue
        if _is_year(match) or _has_unit_suffix(article, match.end):
            continue

        reported.add(match.value)
        far = not (low - outer <= match.value <= high + outer)
        findings.append(
            PrecheckFinding(
                code=(
                    FindingCode.NUMBER_OUTSIDE_MARKET_RANGE
                    if far
                    else FindingCode.UNKNOWN_PRICE_LIKE_NUMBER
                ),
                severity=Severity.HIGH if far else Severity.MEDIUM,
                message=(
                    f"The article states {match.literal!r}, which is not any value in the "
                    f"context and is far outside the candle range "
                    f"{format_price(low)}-{format_price(high)}."
                    if far
                    else f"The article states {match.literal!r}, which does not appear in the "
                    f"context, in the recorded claims, or in the analyst's note."
                ),
                actual=match.literal,
                excerpt=_excerpt(article, match.start, match.end),
            )
        )
    return findings


def _is_year(match: NumberMatch) -> bool:
    """Whether a bare four-digit integer should be read as a year."""
    if "." in match.literal or "," in match.literal:
        return False
    return match.value == match.value.to_integral_value() and int(match.value) in _YEAR_RANGE


def _has_unit_suffix(text: str, end: int) -> bool:
    """Whether the number is followed by a unit that makes it a count."""
    return _UNIT_SUFFIX_RE.match(_fold(text[end : end + 24])) is not None


def _excerpt(text: str, start: int, end: int, window: int = 60) -> str:
    """A readable slice of the article around a match."""
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right].replace("\n", " ").strip()
    return f"...{snippet}..." if left > 0 or right < len(text) else snippet


# --------------------------------------------------------------------------
# instrument, indicators, risk language
# --------------------------------------------------------------------------


def _check_symbols(context: AnalysisContext, article: str) -> list[PrecheckFinding]:
    folded = _fold(article)
    symbol = context.market.symbol

    findings: list[PrecheckFinding] = []
    for foreign in _KNOWN_SYMBOLS:
        if foreign == symbol:
            continue
        if _fold(foreign) in folded:
            findings.append(
                PrecheckFinding(
                    code=FindingCode.FOREIGN_SYMBOL_MENTIONED,
                    severity=Severity.CRITICAL,
                    message=(
                        f"The article mentions {foreign}, but this Run is about {symbol}. "
                        "The instrument must not change."
                    ),
                    source_path="context.market.symbol",
                    expected=symbol,
                    actual=foreign,
                )
            )

    mentions_instrument = _fold(symbol) in folded or any(
        word in folded for word in ("vang", "gold", "xau")
    )
    if not mentions_instrument:
        findings.append(
            PrecheckFinding(
                code=FindingCode.SYMBOL_NOT_MENTIONED,
                severity=Severity.LOW,
                message=f"The article never names the instrument ({symbol}).",
                source_path="context.market.symbol",
                expected=symbol,
            )
        )
    return findings


def _check_indicators(article: str) -> list[PrecheckFinding]:
    """Flag indicators stated with a value. The context carries none of them.

    The trailing ``\\d*`` matters. Traders write ``EMA200``, ``RSI14``,
    ``SMA50`` - and a plain word boundary after the name matches none of them,
    because a digit is a word character. That is the notation an invented
    indicator most often arrives in, so without it the check missed the common
    case entirely. It stays safe against ordinary words: ``EMAIL`` and
    ``SCHEMA`` have a letter where the boundary must be.
    """
    findings: list[PrecheckFinding] = []
    for name in _INDICATORS:
        for match in re.finditer(rf"\b{re.escape(name)}\d*\b", article, re.IGNORECASE):
            findings.append(
                PrecheckFinding(
                    code=FindingCode.UNSUPPORTED_INDICATOR_MENTIONED,
                    severity=Severity.HIGH,
                    message=(
                        f"The article refers to {name}. The context contains no indicator "
                        "data, so any value or reading stated for it was invented."
                    ),
                    actual=name,
                    excerpt=_excerpt(article, match.start(), match.end()),
                )
            )
            break  # one finding per indicator is enough
    return findings


def _check_risk_language(article: str) -> list[PrecheckFinding]:
    folded = _fold(article)
    findings: list[PrecheckFinding] = []
    for phrase in _ABSOLUTE_PHRASES:
        index = folded.find(phrase)
        if index == -1:
            continue
        findings.append(
            PrecheckFinding(
                code=FindingCode.ABSOLUTE_RISK_LANGUAGE,
                severity=Severity.HIGH,
                message=(
                    f"The article contains an absolute claim ({phrase!r}). Commentary must "
                    "describe scenarios and leanings, never certainties."
                ),
                actual=phrase,
                excerpt=_excerpt(article, index, index + len(phrase)),
            )
        )
    return findings


def render_findings(report: PrecheckReport) -> str:
    """Render the findings for the prompt.

    The reviewer is shown these and told not to ignore them; they are facts it
    does not have to rediscover.
    """
    if not report.findings:
        return "No deterministic problems were found. Review the article on its own merits."

    lines: list[str] = []
    for index, finding in enumerate(report.findings, start=1):
        parts = [f"{index}. [{finding.severity}] {finding.code}: {finding.message}"]
        if finding.source_path:
            parts.append(f"   source: {finding.source_path}")
        if finding.expected is not None or finding.actual is not None:
            parts.append(f"   expected: {finding.expected!r}  actual: {finding.actual!r}")
        if finding.excerpt:
            parts.append(f"   excerpt: {finding.excerpt}")
        lines.append("\n".join(parts))
    return "\n".join(lines)


__all__ = [
    "BLOCKING_SEVERITIES",
    "OUTSIDE_RANGE_TOLERANCE",
    "PrecheckReport",
    "render_findings",
    "run_prechecks",
]
