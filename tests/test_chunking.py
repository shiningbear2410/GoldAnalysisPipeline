"""Chunking an approved article.

The governing property is exactness: the publisher delivers the bytes the gate
approved, so the chunker may only choose *where* to cut. Most tests here are
some form of that one assertion.
"""

from __future__ import annotations

import pytest

from goldpipeline.services.chunking import (
    SAFE_CHUNK_LIMIT,
    TELEGRAM_HARD_LIMIT,
    plan_chunks,
    utf16_length,
    verify_plan,
)

ARTICLE = (
    "🕯 NHẬN ĐỊNH VÀNG\n\n"
    "⚡ Chốt nhanh\n"
    "Giá gần nhất trong dữ liệu quanh 3305.90, thị trường đang tích luỹ.\n\n"
    "⚠️ Lưu ý\n"
    "Đây là quan điểm cá nhân, không phải khuyến nghị đầu tư."
)


def paragraphs(count: int) -> str:
    return "\n\n".join(
        f"Đoạn {i}: vàng tích luỹ quanh 3305.90, chờ tín hiệu rõ hơn từ phiên Mỹ."
        for i in range(count)
    )


# --- UTF-16 counting ------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("abc", 3), ("", 0), ("🕯", 2), ("🕯🕯", 4), ("Chốt nhanh", 10), ("⚡", 1)],
)
def test_length_is_counted_the_way_telegram_counts(text: str, expected: int) -> None:
    """An emoji above the BMP costs two units, and `len()` would under-count it."""
    assert utf16_length(text) == expected


def test_emoji_make_the_two_counts_differ() -> None:
    text = "🕯🕯🕯 vàng"
    assert len(text) < utf16_length(text)


# --- the exactness invariant ---------------------------------------------


@pytest.mark.parametrize(
    "article",
    [
        "",
        "Ngắn.",
        ARTICLE,
        paragraphs(4),
        paragraphs(200),
        "Vàng tích luỹ. " * 900,
        "🕯⚡🎯⚠️ Nhận định vàng phiên Á hôm nay. " * 300,
        "x" * 12000,
        "Vàng.   \n\n   " * 700,
        "\n\n\n\n" * 2000,
        "một dòng rất dài không có khoảng trắng" + "abcdefghij" * 900,
    ],
)
def test_chunks_always_reassemble_exactly(article: str) -> None:
    """Requirement 39.14: the single most important property in Round 6."""
    chunks = plan_chunks(article)
    assert "".join(chunks) == article


@pytest.mark.parametrize("article", [ARTICLE, paragraphs(200), "Vàng. " * 2000])
def test_every_chunk_fits_the_limit(article: str) -> None:
    """Requirement 39.19."""
    for chunk in plan_chunks(article):
        assert utf16_length(chunk) <= SAFE_CHUNK_LIMIT


def test_the_safe_limit_leaves_headroom() -> None:
    assert SAFE_CHUNK_LIMIT < TELEGRAM_HARD_LIMIT


def test_nothing_is_normalised() -> None:
    """Requirement 39.15: whitespace the gate approved is whitespace that ships."""
    article = "  Dòng đầu.  \n\n\n   Dòng sau.   \n\n" + ("Vàng đi ngang.  \n\n" * 400)
    chunks = plan_chunks(article)

    assert "".join(chunks) == article
    assert chunks[0].startswith("  Dòng đầu.")
    assert article.endswith(chunks[-1])


def test_no_part_markers_are_added() -> None:
    """Requirements 39.21-39.22: a marker is content nobody approved."""
    chunks = plan_chunks(paragraphs(300))

    assert len(chunks) > 1
    for chunk in chunks:
        for marker in ("(1/", "Part ", "continued", "…tiếp", "/2)"):
            assert marker not in chunk


def test_a_short_article_stays_one_chunk() -> None:
    """Requirement 39.20: no artificial splitting."""
    assert plan_chunks(ARTICLE) == [ARTICLE]
    assert len(plan_chunks("Vàng.")) == 1


def test_an_article_exactly_at_the_limit_stays_one_chunk() -> None:
    article = "a" * SAFE_CHUNK_LIMIT
    assert plan_chunks(article) == [article]


def test_one_over_the_limit_splits() -> None:
    article = "a" * (SAFE_CHUNK_LIMIT + 1)
    chunks = plan_chunks(article)
    assert len(chunks) == 2
    assert "".join(chunks) == article


# --- boundary preference --------------------------------------------------


def test_a_paragraph_boundary_is_preferred() -> None:
    """Requirement 39.16."""
    chunks = plan_chunks(paragraphs(200))

    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert chunk.endswith("\n\n"), "a cut landed somewhere other than a paragraph break"


def test_a_line_break_is_used_when_there_is_no_paragraph() -> None:
    article = "\n".join(f"Dòng {i}: vàng tích luỹ quanh 3305.90 chờ tín hiệu." for i in range(200))
    chunks = plan_chunks(article)

    assert len(chunks) > 1
    assert chunks[0].endswith("\n")
    assert "".join(chunks) == article


def test_a_long_paragraph_falls_back_to_whitespace() -> None:
    """Requirement 39.17: no paragraph or line break to use."""
    article = "vàng " * 3000
    chunks = plan_chunks(article)

    assert len(chunks) > 1
    assert "".join(chunks) == article
    assert chunks[0].endswith(" ")


def test_text_with_no_whitespace_still_splits() -> None:
    article = "x" * 9000
    chunks = plan_chunks(article)

    assert len(chunks) == 3
    assert "".join(chunks) == article


# --- Unicode safety -------------------------------------------------------


def test_a_cut_never_lands_inside_a_character() -> None:
    """Requirement 39.18: every chunk must be encodable on its own."""
    article = "🕯⚡🎯" * 4000
    chunks = plan_chunks(article)

    assert "".join(chunks) == article
    for chunk in chunks:
        chunk.encode("utf-8")
        chunk.encode("utf-16-le")
        assert "�" not in chunk


def test_emoji_are_counted_for_the_limit_not_just_characters() -> None:
    """An article of surrogate pairs needs twice as many chunks as `len` suggests."""
    article = "🕯" * 4000
    chunks = plan_chunks(article)

    assert utf16_length(article) == 8000
    assert len(chunks) >= 3
    assert all(utf16_length(chunk) <= SAFE_CHUNK_LIMIT for chunk in chunks)


# --- verify_plan ----------------------------------------------------------


def test_verify_accepts_a_good_plan() -> None:
    verify_plan(ARTICLE, plan_chunks(ARTICLE))
    long_article = paragraphs(200)
    verify_plan(long_article, plan_chunks(long_article))


def test_verify_rejects_a_plan_that_loses_text() -> None:
    """A chunker bug must stop the publish, not reach the channel."""
    with pytest.raises(ValueError, match="does not reassemble"):
        verify_plan(ARTICLE, [ARTICLE[:-5]])


def test_verify_rejects_a_plan_that_adds_text() -> None:
    with pytest.raises(ValueError, match="does not reassemble"):
        verify_plan(ARTICLE, [ARTICLE, " (1/1)"])


def test_verify_rejects_an_empty_chunk() -> None:
    with pytest.raises(ValueError, match="is empty"):
        verify_plan(ARTICLE, [ARTICLE, ""])


def test_verify_rejects_an_oversized_chunk() -> None:
    article = "a" * 5000
    with pytest.raises(ValueError, match="above the"):
        verify_plan(article, [article])


def test_verify_rejects_an_empty_plan() -> None:
    with pytest.raises(ValueError, match="plan is empty"):
        verify_plan(ARTICLE, [])


def test_an_unusable_limit_is_refused() -> None:
    with pytest.raises(ValueError, match="too small"):
        plan_chunks(ARTICLE, limit=4)


# --- determinism ----------------------------------------------------------


def test_planning_is_deterministic() -> None:
    article = paragraphs(200)
    assert plan_chunks(article) == plan_chunks(article)
