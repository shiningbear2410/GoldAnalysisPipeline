"""Catching an edit that stopped being Vietnamese, and nothing else.

Round 6.5c.3a. The check this file covers exists because a digest whose entire
editorial prose was accent-free reached the Publish Gate and was approved. That
particular text was accent-free on the way *in* - a smoke fixture written in
ASCII, not a model stripping anything - but the Run proved the gap: nothing
between a repair and the gate ever asks whether the Vietnamese is still
Vietnamese.

**The false-positive corpus matters more than the positive one.** A repair path
allows exactly one model call, so a check that wrongly refuses a good repair
does not cost a retry - it costs the Run, and an operator's afternoon. Every
case in the second half below is a legitimate rewrite that uses fewer accented
characters than the text it replaced, and every one of them must pass.
"""

from __future__ import annotations

import pytest

from goldpipeline.services.text_integrity import (
    MIN_MARKS_BEFORE,
    TextIntegrityFinding,
    check_edit,
    check_edits,
    check_new_text,
    diacritic_count,
)

# --------------------------------------------------------------------------
# §8: the exact text the round was opened about
# --------------------------------------------------------------------------

OBSERVED_BEFORE = (
    "Cả ba tin trong cửa sổ đều nghiêng về phía hỗ trợ vàng. "
    "Giá thì không đi theo. "
    "Nên tin thuận chiều mà phe mua vẫn chưa tận dụng được — "
    "đó mới là phần đáng lưu ý."
)
OBSERVED_AFTER_BAD = (
    "Ca ba tin trong cua so deu nghieng ve phia ho tro vang. "
    "Gia thi khong di theo. "
    "Nen tin thuan chieu ma phe mua van chua tan dung duoc — "
    "do moi la phan dang luu y."
)


def test_1_the_observed_whole_section_stripping_is_refused() -> None:
    """The fixture this round exists to pin."""
    finding = check_edit("balance", OBSERVED_BEFORE, OBSERVED_AFTER_BAD)

    assert finding is not None
    assert finding.field == "balance"
    assert "diacritics removed" in finding.reason
    assert finding.marks_before > MIN_MARKS_BEFORE
    assert finding.marks_after <= 1
    assert finding.similarity > 0.9, "the same prose, so near-identical once folded"


def test_2_severe_partial_stripping_is_refused() -> None:
    """Most accents gone, one word left. A perfect zero is not required."""
    after = (
        "Ca ba tin trong cua so deu nghieng ve phia ho tro vàng. "
        "Gia thi khong di theo. "
        "Nen tin thuan chieu ma phe mua van chua tan dung duoc — "
        "do moi la phan dang luu y."
    )
    assert check_edit("balance", OBSERVED_BEFORE, after) is not None


def test_the_finding_reads_usefully_to_an_operator() -> None:
    finding = check_edit("balance", OBSERVED_BEFORE, OBSERVED_AFTER_BAD)

    assert finding is not None
    described = finding.describe()
    assert "balance" in described
    assert "diacritics" in described
    assert "similarity" in described


# --------------------------------------------------------------------------
# §9 + §17: legitimate rewrites, every one of which must pass
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "before", "after"),
    [
        (
            "a shorter, plainer restatement",
            "Fed giữ giọng trung lập.",
            "Fed chưa cho thấy tín hiệu cứng rắn hơn.",
        ),
        (
            "a figure dropped rather than approximated",
            "SPDR Gold Trust mua ròng 9.98 tấn.",
            "SPDR tiếp tục mua ròng.",
        ),
        (
            "a percentage replaced by a description",
            "USD giảm 0.21%.",
            "USD yếu thêm trong phiên.",
        ),
        (
            "a scheduled event reframed",
            "CPI Mỹ công bố tối nay.",
            "CPI Mỹ là tin đáng chờ nhất.",
        ),
        (
            "the balance genuinely rewritten, still accented",
            OBSERVED_BEFORE,
            "Tin nghiêng tích cực nhưng giá chưa xác nhận. Đó mới là phần đáng chú ý.",
        ),
        (
            "an English-heavy finance line",
            "SPDR ETF mua ròng, USD yếu, CPI sắp công bố.",
            "SPDR ETF mua ròng trong khi USD yếu; CPI vẫn ở phía trước.",
        ),
        (
            "unchanged text",
            OBSERVED_BEFORE,
            OBSERVED_BEFORE,
        ),
    ],
)
def test_legitimate_rewrites_are_not_blocked(label: str, before: str, after: str) -> None:
    assert check_edit("balance", before, after) is None, label


def test_an_already_ascii_source_is_never_flagged() -> None:
    """The case that actually occurred, and it is not this check's business.

    A digest built from accent-free bulletins is an honest report of what those
    bulletins said. There was nothing to strip, so there is nothing to find -
    and a check that demanded accents here would block correct output for being
    correct.
    """
    before = "Fed Williams noi loi suat tang gan day khong phan anh ky vong lam phat cao hon."
    after = "Fed Williams chua cho thay tin hieu cung ran hon trong phien nay."

    assert diacritic_count(before) == 0
    assert check_edit("items.x.headline", before, after) is None


def test_an_acronym_and_number_line_is_not_judged() -> None:
    assert check_edit("items.x.headline", "SPDR 9.98 USD CPI", "SPDR 9.98 USD ETF") is None


def test_a_short_edit_is_too_short_to_judge() -> None:
    """Below the length floor a rewrite and a transliteration look identical."""
    assert check_edit("balance", "Giá vàng tăng mạnh trong phiên.", "Gia vang tang.") is None


def test_removing_a_numeric_fact_is_a_provenance_question_not_this_one() -> None:
    """§17 case 7. Text integrity and provenance are orthogonal on purpose."""
    before = "SPDR Gold Trust mua ròng 9.98 tấn trong phiên gần nhất, mức cao nhất tuần."
    after = "SPDR Gold Trust tiếp tục mua ròng trong phiên gần nhất, theo số liệu quỹ."

    assert check_edit("balance", before, after) is None


# --------------------------------------------------------------------------
# encoding corruption
# --------------------------------------------------------------------------


def test_a_replacement_character_is_refused_whatever_else_is_true() -> None:
    after = "Tin nghi�ng t�ch c�c, nh�ng gi� ch�a x�c nh�n."

    finding = check_edit("balance", OBSERVED_BEFORE, after)

    assert finding is not None
    assert "replacement character" in finding.reason


def test_corruption_is_caught_even_in_text_with_no_previous_version() -> None:
    """A newly selected item has no "before"; the decoder is still questioned."""
    assert check_new_text("items.new.headline", "Gi� v�ng t�ng") is not None
    assert check_new_text("items.new.headline", "Giá vàng tăng") is None


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------


def test_the_same_text_in_nfd_and_nfc_compares_equal() -> None:
    """§17 case 9. A decomposed encoding is the same Vietnamese, not a defect."""
    import unicodedata

    nfc = unicodedata.normalize("NFC", OBSERVED_BEFORE)
    nfd = unicodedata.normalize("NFD", OBSERVED_BEFORE)

    assert nfc != nfd, "the two encodings really are different byte sequences"
    assert diacritic_count(nfd) == diacritic_count(nfc)
    assert check_edit("balance", nfc, nfd) is None
    assert check_edit("balance", nfd, nfc) is None


def test_the_stripped_form_of_nfd_input_is_still_caught() -> None:
    """Decomposing first must not become a way to smuggle stripping past."""
    import unicodedata

    nfd_before = unicodedata.normalize("NFD", OBSERVED_BEFORE)

    assert check_edit("balance", nfd_before, OBSERVED_AFTER_BAD) is not None


def test_vietnamese_only_letters_count_as_diacritics() -> None:
    """`đ` and `ơ` carry no combining mark and are still not ASCII."""
    assert diacritic_count("đường") > 0
    assert diacritic_count("Đà Nẵng") > 0
    assert diacritic_count("Fed USD CPI") == 0


# --------------------------------------------------------------------------
# batch behaviour
# --------------------------------------------------------------------------


def test_every_offending_field_is_reported_not_only_the_first() -> None:
    findings = check_edits(
        [
            ("balance", OBSERVED_BEFORE, OBSERVED_AFTER_BAD),
            ("items.a.headline", "Fed giữ giọng trung lập.", "Fed giữ giọng trung lập."),
            ("items.b.note", OBSERVED_BEFORE, OBSERVED_AFTER_BAD),
        ]
    )

    assert [f.field for f in findings] == ["balance", "items.b.note"]
    assert all(isinstance(f, TextIntegrityFinding) for f in findings)


def test_a_clean_batch_reports_nothing() -> None:
    assert check_edits([("balance", OBSERVED_BEFORE, OBSERVED_BEFORE)]) == []


def test_the_check_reaches_no_clock_network_or_dictionary() -> None:
    """It compares two strings. Nothing else is available to it."""
    import inspect

    from goldpipeline.services import text_integrity

    body = inspect.getsource(text_integrity)
    for forbidden in ("datetime.now", "utcnow", "requests", "httpx", "open(", "urllib"):
        assert forbidden not in body, forbidden
