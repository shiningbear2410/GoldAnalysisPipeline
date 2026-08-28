"""The deterministic check that stops a contradicted price reaching a draft."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from conftest import make_analysis_payload, make_bar, make_market_payload, make_normalized_run

from goldpipeline.schemas.writer import WarningCode
from goldpipeline.services.source_guard import (
    build_guard_notice,
    build_guard_warnings,
    screen_source_prices,
)

WINDOW_BARS = [
    make_bar(
        minute_offset=index * 15,
        open_="3312.00",
        high="3322.00",
        low="3305.00",
        close="3315.00",
    )
    for index in range(4)
]
"""A tight 3305-3322 window, like a quiet Asian session."""


def _screen(runs_dir: Any, tmp_path: Any, text: str) -> Any:
    result = make_normalized_run(
        runs_dir,
        tmp_path,
        analysis=make_analysis_payload(raw_text=text),
        market=make_market_payload(bars=WINDOW_BARS),
    )
    assert result.context is not None
    return screen_source_prices(result.context)


def test_a_contradicted_current_price_is_caught(runs_dir: Any, tmp_path: Any) -> None:
    """The motivating case: the note claims a price the candles deny."""
    report = _screen(runs_dir, tmp_path, "Giá vàng hiện tại là 3400, tiếp tục nắm giữ vị thế mua.")

    assert report.has_findings
    assert [finding.value for finding in report.out_of_range] == [Decimal("3400")]
    assert report.window_low == Decimal("3305.00")
    assert report.window_high == Decimal("3322.00")


def test_levels_near_the_session_are_not_flagged(runs_dir: Any, tmp_path: Any) -> None:
    """A resistance a couple of points above the high is ordinary analysis.

    This is the false-positive case that decides whether the warning is worth
    reading at all.
    """
    report = _screen(
        runs_dir,
        tmp_path,
        "Vàng tích luỹ quanh 3315. Kháng cự 3.320 - 3.324, hỗ trợ 3.305. Mục tiêu 3.330.",
    )
    assert not report.has_findings


def test_vietnamese_thousand_separators_are_understood(runs_dir: Any, tmp_path: Any) -> None:
    """``3.400`` means three thousand four hundred in a Vietnamese note."""
    report = _screen(runs_dir, tmp_path, "Giá vàng hiện tại là 3.400 usd/oz.")
    assert [finding.value for finding in report.out_of_range] == [Decimal("3400")]


def test_comma_separators_are_understood(runs_dir: Any, tmp_path: Any) -> None:
    report = _screen(runs_dir, tmp_path, "Gold is now at 3,400.50 per ounce.")
    assert [finding.value for finding in report.out_of_range] == [Decimal("3400.50")]


def test_small_numbers_are_not_treated_as_quotes(runs_dir: Any, tmp_path: Any) -> None:
    """Lot sizes, percentages and counts are not gold prices."""
    report = _screen(
        runs_dir, tmp_path, "Vào lệnh 0.5 lot, rủi ro 2%, chốt sau 30 phút. Giá quanh 3315."
    )
    assert not report.has_findings


def test_each_offending_value_is_reported_once(runs_dir: Any, tmp_path: Any) -> None:
    report = _screen(runs_dir, tmp_path, "Lên 3400. Vẫn là 3400. Nhắc lại: 3400.")
    assert len(report.out_of_range) == 1


def test_a_finding_becomes_a_warning_and_a_prompt_notice(runs_dir: Any, tmp_path: Any) -> None:
    report = _screen(runs_dir, tmp_path, "Giá vàng hiện tại là 3400.")

    warnings = build_guard_warnings(report)
    assert [w.code for w in warnings] == [WarningCode.SOURCE_PRICE_OUT_OF_RANGE]
    assert "3400" in warnings[0].message
    assert "3305.00-3322.00" in warnings[0].message

    notice = build_guard_notice(report)
    assert notice is not None
    assert "Market facts win" in notice
    assert "3400" in notice


def test_a_clean_note_produces_neither(runs_dir: Any, tmp_path: Any) -> None:
    report = _screen(runs_dir, tmp_path, "Vàng đi ngang quanh 3315, chờ tín hiệu rõ hơn.")
    assert build_guard_warnings(report) == []
    assert build_guard_notice(report) is None


def test_the_source_text_is_never_modified(runs_dir: Any, tmp_path: Any) -> None:
    """The guard detects; it must never edit Round 1's immutable input."""
    text = "Giá vàng hiện tại là 3400, mua ngay."
    result = make_normalized_run(
        runs_dir,
        tmp_path,
        analysis=make_analysis_payload(raw_text=text),
        market=make_market_payload(bars=WINDOW_BARS),
    )
    assert result.context is not None

    before = result.context.raw_analysis.text
    screen_source_prices(result.context)
    assert result.context.raw_analysis.text == before == text


@pytest.mark.parametrize("value", ["3305", "3322", "3300", "3330"])
def test_values_inside_the_tolerance_band_pass(runs_dir: Any, tmp_path: Any, value: str) -> None:
    report = _screen(runs_dir, tmp_path, f"Vùng quan tâm quanh {value}.")
    assert not report.has_findings
