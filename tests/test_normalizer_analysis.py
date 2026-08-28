"""Raw analysis text handling: encoding, sanitisation, optional metadata."""

from __future__ import annotations

from typing import Any

import pytest
from conftest import VIETNAMESE_TEXT, make_analysis_payload

from goldpipeline.domain.errors import AnalysisTextTooLargeError, EmptyAnalysisTextError
from goldpipeline.schemas.quality import WarningCode
from goldpipeline.schemas.telegram import MAX_RAW_TEXT_CHARS, TelegramAnalysisInput
from goldpipeline.services.normalizer import (
    NormalizedAnalysis,
    normalize_analysis,
    sanitize_analysis_text,
)


def normalize(payload: dict[str, Any]) -> NormalizedAnalysis:
    return normalize_analysis(TelegramAnalysisInput.model_validate(payload))


def test_vietnamese_text_is_preserved_exactly() -> None:
    """Requirement 14.8: no mojibake, no stripped diacritics."""
    result = normalize(make_analysis_payload())
    assert result.analysis.raw_text == VIETNAMESE_TEXT
    assert "giằng co" in result.analysis.raw_text
    assert "khuyến nghị đầu tư" in result.analysis.raw_text


def test_missing_optional_metadata_does_not_crash() -> None:
    """Requirement 14.9: only raw_text is mandatory."""
    result = normalize(make_analysis_payload(include_metadata=False))
    message = result.analysis
    assert message.chat_id is None
    assert message.message_id is None
    assert message.message_date is None
    assert message.author is None
    assert message.metadata == {}
    assert WarningCode.MISSING_TELEGRAM_METADATA in {w.code for w in result.warnings}
    assert "raw_analysis.message_id" in result.missing_fields


def test_metadata_is_carried_through_untouched() -> None:
    payload = make_analysis_payload(metadata={"chat_title": "Gold Signals VN", "views": 1204})
    result = normalize(payload)
    assert result.analysis.metadata == {"chat_title": "Gold Signals VN", "views": 1204}


def test_empty_text_is_rejected() -> None:
    with pytest.raises(EmptyAnalysisTextError):
        normalize(make_analysis_payload(raw_text="   \n\t  "))


def test_oversized_text_is_rejected() -> None:
    with pytest.raises(AnalysisTextTooLargeError):
        normalize(make_analysis_payload(raw_text="x" * (MAX_RAW_TEXT_CHARS + 1)))


def test_trust_level_is_fixed_and_cannot_be_downgraded() -> None:
    """A payload must not be able to declare itself trusted."""
    from pydantic import ValidationError

    result = normalize(make_analysis_payload())
    assert result.analysis.trust_level == "UNTRUSTED"
    with pytest.raises(ValidationError):
        TelegramAnalysisInput.model_validate(make_analysis_payload(trust_level="TRUSTED"))


# --- sanitisation ---------------------------------------------------------


def test_zero_width_and_bidi_characters_are_stripped() -> None:
    """Hidden characters are invisible to a reviewer but not to a model."""
    hidden = f"Vàng tăng{chr(0x200B)} nhẹ{chr(0x202E)} hôm nay"
    cleaned, removed = sanitize_analysis_text(hidden)
    assert cleaned == "Vàng tăng nhẹ hôm nay"
    assert removed == {"invisible": 2}


def test_control_characters_are_stripped_but_newlines_survive() -> None:
    cleaned, removed = sanitize_analysis_text("dòng 1\x00\ndòng 2\x07\n\tthụt lề")
    assert cleaned == "dòng 1\ndòng 2\n\tthụt lề"
    assert removed == {"control": 2}


def test_sanitisation_is_reported_as_a_warning() -> None:
    result = normalize(make_analysis_payload(raw_text=f"Vàng{chr(0x200B)} tăng"))
    assert WarningCode.RAW_TEXT_SANITIZED in {w.code for w in result.warnings}


def test_windows_line_endings_are_normalized() -> None:
    cleaned, removed = sanitize_analysis_text("dòng 1\r\ndòng 2\rdòng 3")
    assert cleaned == "dòng 1\ndòng 2\ndòng 3"
    assert removed == {}


def test_clean_text_produces_no_sanitisation_warning() -> None:
    result = normalize(make_analysis_payload())
    assert WarningCode.RAW_TEXT_SANITIZED not in {w.code for w in result.warnings}


def test_wording_is_never_altered() -> None:
    """Sanitisation removes invisible characters only - never rewrites prose."""
    text = "Mua tại 3.309 - 3.311, mục tiêu 3.322. Cắt lỗ dưới 3.305!"
    cleaned, removed = sanitize_analysis_text(text)
    assert cleaned == text
    assert removed == {}
