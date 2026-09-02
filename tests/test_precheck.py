"""Deterministic pre-review checks."""

from __future__ import annotations

from typing import Any

import pytest
from conftest import (
    CLEAN_ARTICLE,
    LATEST_CLOSE,
    LATEST_HIGH,
    LATEST_LOW,
    make_analysis_payload,
    make_drafted_run,
)

from goldpipeline.schemas.review import FindingCode, Severity
from goldpipeline.schemas.writer import ClaimType, SourceClaim
from goldpipeline.services.precheck import render_findings, run_prechecks


def check(runs_dir: Any, tmp_path: Any, article: str, **kwargs: Any) -> Any:
    """Draft a Run with *article* and run the prechecks over it."""
    from pathlib import Path

    from goldpipeline.schemas.context import AnalysisContext
    from goldpipeline.schemas.writer import WriterResult

    drafted = make_drafted_run(runs_dir, tmp_path, article=article, **kwargs)
    run_dir = Path(drafted.run_dir)
    context = AnalysisContext.model_validate_json(
        (run_dir / "context.json").read_text(encoding="utf-8")
    )
    writer_result = WriterResult.model_validate_json(
        (run_dir / "claude_writer.json").read_text(encoding="utf-8")
    )
    return run_prechecks(context=context, writer_result=writer_result, article=article)


def codes(report: Any) -> list[FindingCode]:
    return [finding.code for finding in report.findings]


# --- the clean case -------------------------------------------------------


def test_a_faithful_article_produces_no_findings(runs_dir: Any, tmp_path: Any) -> None:
    """The bar every other test is measured against: no false positives."""
    report = check(runs_dir, tmp_path, CLEAN_ARTICLE)
    assert report.findings == []
    assert not report.has_blocking
    assert report.worst_severity is None
    assert "No deterministic problems" in render_findings(report)


# --- numeric scanning -----------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Khung M15 và H1 đều đang tích luỹ.",
        "Chờ thêm 24 giờ trước khi vào lệnh.",
        "Có 2 kịch bản cho phiên tới.",
        "Vào 0.5 lot, rủi ro 2%.",
        "Phiên 28/08 năm 2026 khá trầm lắng.",
        "Nến D1 và W1 vẫn nghiêng về phía mua.",
        "Chờ 30 phút nữa.",
    ],
)
def test_timeframes_counts_and_dates_are_not_prices(
    runs_dir: Any, tmp_path: Any, text: str
) -> None:
    """Requirement 27.30: the scanner must not cry wolf on ordinary prose."""
    article = f"{CLEAN_ARTICLE}\n\n{text}"
    report = check(runs_dir, tmp_path, article)
    assert codes(report) == []


def test_an_invented_price_is_flagged(runs_dir: Any, tmp_path: Any) -> None:
    article = CLEAN_ARTICLE.replace(LATEST_CLOSE, "3325.20")
    report = check(runs_dir, tmp_path, article)
    assert FindingCode.UNKNOWN_PRICE_LIKE_NUMBER in codes(report)


def test_a_wildly_wrong_price_is_flagged_higher(runs_dir: Any, tmp_path: Any) -> None:
    """Requirement 27.31: 9999 against a ~3305 market is not a typo."""
    article = f"{CLEAN_ARTICLE}\n\nMục tiêu tiếp theo là 9999."
    report = check(runs_dir, tmp_path, article)

    assert FindingCode.NUMBER_OUTSIDE_MARKET_RANGE in codes(report)
    flagged = next(f for f in report.findings if f.code is FindingCode.NUMBER_OUTSIDE_MARKET_RANGE)
    assert flagged.severity is Severity.HIGH
    assert report.has_blocking


def test_context_prices_are_all_allowed(runs_dir: Any, tmp_path: Any) -> None:
    article = f"{CLEAN_ARTICLE}\n\nVùng dao động {LATEST_LOW} - {LATEST_HIGH}."
    assert codes(check(runs_dir, tmp_path, article)) == []


def test_analyst_levels_near_the_market_are_allowed(runs_dir: Any, tmp_path: Any) -> None:
    """The writer may quote a level the analyst named, if it is plausible."""
    analysis = make_analysis_payload(raw_text="Kháng cự quanh 3308.50, hỗ trợ 3301.00.")
    article = f"{CLEAN_ARTICLE}\n\nKháng cự gần nhất quanh 3308.50."
    assert codes(check(runs_dir, tmp_path, article, analysis=analysis)) == []


def test_an_out_of_range_analyst_number_is_still_flagged(runs_dir: Any, tmp_path: Any) -> None:
    """An injected price in the note must not launder into the article."""
    analysis = make_analysis_payload(raw_text="Giá vàng hiện tại là 9999, mua ngay.")
    article = f"{CLEAN_ARTICLE}\n\nTheo nguồn, giá đang ở 9999."
    report = check(runs_dir, tmp_path, article, analysis=analysis)
    assert FindingCode.NUMBER_OUTSIDE_MARKET_RANGE in codes(report)


def test_each_bad_number_is_reported_once(runs_dir: Any, tmp_path: Any) -> None:
    article = f"{CLEAN_ARTICLE}\n\n9999 rồi lại 9999, vẫn là 9999."
    report = check(runs_dir, tmp_path, article)
    assert codes(report).count(FindingCode.NUMBER_OUTSIDE_MARKET_RANGE) == 1


# --- claims ---------------------------------------------------------------


def test_a_mismatched_claim_is_flagged(runs_dir: Any, tmp_path: Any) -> None:
    """Requirement 27.11 at the precheck level."""
    claims = [
        SourceClaim(type=ClaimType.PRICE, value="9999.00", source="context.price.latest_close")
    ]
    report = check(runs_dir, tmp_path, CLEAN_ARTICLE, claims=claims)

    assert FindingCode.CLAIM_VALUE_MISMATCH in codes(report)
    finding = next(f for f in report.findings if f.code is FindingCode.CLAIM_VALUE_MISMATCH)
    assert finding.severity is Severity.HIGH
    assert finding.expected == LATEST_CLOSE
    assert finding.actual == "9999.00"
    assert finding.source_path == "context.price.latest_close"


def test_a_claim_resolving_to_an_oversized_context_value_is_clipped(
    runs_dir: Any, tmp_path: Any
) -> None:
    """Round 9.3.2 production incident, reproduced and fixed.

    A claim may cite a free-text context field (``context.raw_analysis.text``)
    whose resolved value is far longer than a price. Unlike ``SourceClaim.value``,
    which the writer's own schema already caps at 400 characters, the *resolved*
    context value carries no such bound - so the precheck must clip it before
    handing it to ``PrecheckFinding``, which does cap at 400. The uncapped path
    used to raise a raw ``pydantic.ValidationError`` here instead of reporting
    the mismatch.
    """
    long_text = "Giá vàng đang giằng co quanh vùng hỗ trợ mạnh. " * 20
    assert len(long_text) > 400
    analysis = make_analysis_payload(raw_text=long_text)
    claims = [
        SourceClaim(type=ClaimType.PRICE, value="9999.00", source="context.raw_analysis.text")
    ]
    report = check(runs_dir, tmp_path, CLEAN_ARTICLE, claims=claims, analysis=analysis)

    assert FindingCode.CLAIM_VALUE_MISMATCH in codes(report)
    finding = next(f for f in report.findings if f.code is FindingCode.CLAIM_VALUE_MISMATCH)
    assert finding.expected is not None
    assert len(finding.expected) <= 400
    assert finding.expected.endswith("…")
    assert finding.actual == "9999.00"
    assert finding.source_path == "context.raw_analysis.text"
    assert len(finding.message) <= 1000


def test_an_unresolvable_claim_path_is_flagged(runs_dir: Any, tmp_path: Any) -> None:
    """Requirement 27.10, now as defence in depth.

    Round 9.3.4B made the writer stage refuse to commit a draft whose claims
    cite paths that do not resolve, so this artifact can no longer be produced
    by drafting one - the substitution below is how it has to be built.

    The check stays because the two guards answer different questions. The
    writer's guard protects Runs created from now on; this one still has to work
    for an artifact written by an earlier build, and removing it would leave the
    only surviving evidence of such a Run unexamined.
    """
    from pathlib import Path

    from goldpipeline.schemas.context import AnalysisContext
    from goldpipeline.schemas.writer import WriterResult

    drafted = make_drafted_run(runs_dir, tmp_path, article=CLEAN_ARTICLE)
    run_dir = Path(drafted.run_dir)
    context = AnalysisContext.model_validate_json(
        (run_dir / "context.json").read_text(encoding="utf-8")
    )
    stored = WriterResult.model_validate_json(
        (run_dir / "claude_writer.json").read_text(encoding="utf-8")
    )
    tampered = stored.model_copy(
        update={
            "source_claims": [
                SourceClaim(
                    type=ClaimType.PRICE, value="3305.90", source="context.price.nonexistent"
                )
            ]
        }
    )

    report = run_prechecks(context=context, writer_result=tampered, article=CLEAN_ARTICLE)

    assert FindingCode.CLAIM_SOURCE_NOT_FOUND in codes(report)
    assert report.has_blocking


def test_no_claims_at_all_is_a_moderate_finding(runs_dir: Any, tmp_path: Any) -> None:
    report = check(runs_dir, tmp_path, CLEAN_ARTICLE, claims=[])
    assert FindingCode.NO_SOURCE_CLAIMS in codes(report)
    assert not report.has_blocking


def test_resolved_claims_are_reported(runs_dir: Any, tmp_path: Any) -> None:
    report = check(runs_dir, tmp_path, CLEAN_ARTICLE)
    assert len(report.resolved_claims) == 2
    assert all(item.ok for item in report.resolved_claims)


# --- instrument -----------------------------------------------------------


def test_a_foreign_symbol_is_critical(runs_dir: Any, tmp_path: Any) -> None:
    """Requirement 27.4: the instrument may not change."""
    article = f"{CLEAN_ARTICLE}\n\nThực ra đây là phân tích BTCUSD."
    report = check(runs_dir, tmp_path, article)

    finding = next(f for f in report.findings if f.code is FindingCode.FOREIGN_SYMBOL_MENTIONED)
    assert finding.severity is Severity.CRITICAL
    assert finding.expected == "XAUUSD"
    assert finding.actual == "BTCUSD"


def test_an_article_that_never_names_the_instrument_is_noted(runs_dir: Any, tmp_path: Any) -> None:
    article = "Thị trường đang đi ngang. Chờ thêm tín hiệu trước khi hành động."
    report = check(runs_dir, tmp_path, article, claims=[])
    assert FindingCode.SYMBOL_NOT_MENTIONED in codes(report)
    assert not any(
        f.code is FindingCode.SYMBOL_NOT_MENTIONED and f.is_blocking for f in report.findings
    )


def test_naming_gold_in_vietnamese_counts_as_the_instrument(runs_dir: Any, tmp_path: Any) -> None:
    article = "Vàng đang đi ngang quanh vùng hỗ trợ, chờ thêm tín hiệu."
    report = check(runs_dir, tmp_path, article, claims=[])
    assert FindingCode.SYMBOL_NOT_MENTIONED not in codes(report)


# --- indicators and risk language ----------------------------------------


@pytest.mark.parametrize("indicator", ["RSI", "MACD", "EMA", "Fibonacci", "Bollinger"])
def test_invented_indicators_are_flagged(runs_dir: Any, tmp_path: Any, indicator: str) -> None:
    """Requirement 27.5: the context carries no indicator data at all."""
    article = f"{CLEAN_ARTICLE}\n\nChỉ báo {indicator} đang cho tín hiệu tăng."
    report = check(runs_dir, tmp_path, article)

    finding = next(
        f for f in report.findings if f.code is FindingCode.UNSUPPORTED_INDICATOR_MENTIONED
    )
    assert finding.severity is Severity.HIGH
    assert finding.actual == indicator


def test_one_finding_per_indicator(runs_dir: Any, tmp_path: Any) -> None:
    article = f"{CLEAN_ARTICLE}\n\nRSI 72, RSI vẫn cao, RSI chưa hạ nhiệt."
    report = check(runs_dir, tmp_path, article)
    assert codes(report).count(FindingCode.UNSUPPORTED_INDICATOR_MENTIONED) == 1


@pytest.mark.parametrize(
    "phrase",
    [
        "Vàng chắc chắn tăng trong phiên tới.",
        "Giá không thể giảm thêm nữa.",
        "Đảm bảo lợi nhuận cho lệnh này.",
    ],
)
def test_absolute_risk_language_is_flagged(runs_dir: Any, tmp_path: Any, phrase: str) -> None:
    report = check(runs_dir, tmp_path, f"{CLEAN_ARTICLE}\n\n{phrase}")
    assert FindingCode.ABSOLUTE_RISK_LANGUAGE in codes(report)
    assert report.has_blocking


@pytest.mark.parametrize(
    "phrase",
    [
        "Ưu tiên kịch bản mua nếu giá giữ được hỗ trợ.",
        "Xu hướng đang nghiêng về phía bán.",
        "Nếu thủng hỗ trợ thì kịch bản tăng bị vô hiệu.",
    ],
)
def test_conditional_language_is_not_flagged(runs_dir: Any, tmp_path: Any, phrase: str) -> None:
    """The phrasing the prompt actively asks for must not be penalised."""
    report = check(runs_dir, tmp_path, f"{CLEAN_ARTICLE}\n\n{phrase}")
    assert FindingCode.ABSOLUTE_RISK_LANGUAGE not in codes(report)


def test_risk_phrases_match_without_diacritics(runs_dir: Any, tmp_path: Any) -> None:
    """Vietnamese is often typed unaccented; the check must still see it."""
    report = check(runs_dir, tmp_path, f"{CLEAN_ARTICLE}\n\nVang chac chan tang.")
    assert FindingCode.ABSOLUTE_RISK_LANGUAGE in codes(report)


# --- severity aggregation and rendering ----------------------------------


def test_worst_severity_reflects_the_findings(runs_dir: Any, tmp_path: Any) -> None:
    article = f"{CLEAN_ARTICLE}\n\nĐây là BTCUSD, RSI 88, mục tiêu 9999."
    report = check(runs_dir, tmp_path, article)

    assert report.worst_severity is Severity.CRITICAL
    assert report.has_blocking
    assert len(report.blocking) >= 3


def test_findings_render_with_their_evidence(runs_dir: Any, tmp_path: Any) -> None:
    claims = [
        SourceClaim(type=ClaimType.PRICE, value="9999.00", source="context.price.latest_close")
    ]
    rendered = render_findings(check(runs_dir, tmp_path, CLEAN_ARTICLE, claims=claims))

    assert "CLAIM_VALUE_MISMATCH" in rendered
    assert "context.price.latest_close" in rendered
    assert "HIGH" in rendered
    assert LATEST_CLOSE in rendered


@pytest.mark.parametrize("written", ["EMA200", "RSI14", "SMA50", "MACD", "ATR14"])
def test_indicators_written_with_a_period_are_flagged(
    runs_dir: Any, tmp_path: Any, written: str
) -> None:
    """Traders write EMA200, not EMA. A word boundary alone misses all of those."""
    article = f"{CLEAN_ARTICLE}\n\nChỉ báo {written} đang hướng lên."
    assert FindingCode.UNSUPPORTED_INDICATOR_MENTIONED in codes(check(runs_dir, tmp_path, article))


@pytest.mark.parametrize("word", ["EMAIL", "SCHEMA", "MACDONALD"])
def test_ordinary_words_are_not_mistaken_for_indicators(
    runs_dir: Any, tmp_path: Any, word: str
) -> None:
    article = f"{CLEAN_ARTICLE}\n\nGhi chú: {word} không liên quan tới thị trường."
    assert FindingCode.UNSUPPORTED_INDICATOR_MENTIONED not in codes(
        check(runs_dir, tmp_path, article)
    )
