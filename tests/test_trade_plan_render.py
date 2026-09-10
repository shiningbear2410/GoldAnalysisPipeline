"""The published text, character by character.

Round 6.6g §19-§25. The renderer has no judgement to test - it has a shape, a
vocabulary and a cap, and each of those is either exactly right or a defect a
reader would see.

Validation is asserted against the structured selection rather than by parsing
prose back into prices. The selection is the authority; re-deriving prices from
the page would quietly make the page the authority instead.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from goldpipeline.services.ict_primitives import GapDirection
from goldpipeline.services.trade_plan_render import (
    BAI_HEADING,
    EMPTY_SIDE,
    FARTHER_SUFFIX,
    FORBIDDEN_SUBSTRINGS,
    MAIN_ZONE_SUFFIX,
    MAX_TRADE_PLAN_CHARS,
    PUBLIC_VOCABULARY,
    SEO_HEADING,
    ZONE_DASH,
    TradePlanRenderError,
    render_and_validate,
    render_price,
    render_trade_plan,
    validate_trade_plan,
)
from goldpipeline.services.trade_plan_selector import (
    TradePlanSelection,
    select_trade_plan,
)
from tests.test_ict_candidate_consolidation import decide
from tests.test_ict_candidate_eligibility import gap_source
from tests.test_trade_plan_selector import (
    SEVEN_BAI,
    bai_zones,
    by_bounds,
    empty_features,
    features_of,
    natural,
    ranked,
    seo_zones,
    with_references,
)


def plan(features: object, **order: list[str]) -> tuple[TradePlanSelection, str]:
    selection = select_trade_plan(features, ranked(features, **order))  # type: ignore[arg-type]
    return selection, render_and_validate(selection)


# --------------------------------------------------------------------------
# §19: numbers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("written", "rendered"),
    [
        ("4000", "4000"),
        ("4000.0", "4000"),
        ("4000.00", "4000"),
        ("4050.30", "4050.3"),
        ("4050.25", "4050.25"),
        ("3987.5", "3987.5"),
        ("0.01", "0.01"),
        ("1E+3", "1000"),
        ("4043.125", "4043.125"),
    ],
)
def test_a_price_renders_consistently_and_is_never_changed(written: str, rendered: str) -> None:
    """§19. Trailing zeros go; digits never do, and nothing is rounded."""
    assert render_price(Decimal(written)) == rendered
    assert Decimal(render_price(Decimal(written))) == Decimal(written)


def test_the_renderer_uses_the_identity_helper_rather_than_a_second_formatter() -> None:
    """§19. One price representation in the project, not two that can drift."""
    from goldpipeline.services.ict_candidate_consolidation import canonical_price

    for written in ("4000", "4000.00", "4050.30", "1E+3", "0.5"):
        assert render_price(Decimal(written)) == canonical_price(Decimal(written))


def test_no_thousands_separator_and_no_forced_decimals() -> None:
    """§19."""
    assert render_price(Decimal("14000")) == "14000"
    assert render_price(Decimal("4000")) == "4000"
    assert "," not in render_price(Decimal("14000"))


def test_a_non_decimal_price_is_refused() -> None:
    """§19. No float, at any point."""
    with pytest.raises(TradePlanRenderError, match="must be a Decimal"):
        render_price(4000.0)


def test_the_two_dashes_are_different_characters() -> None:
    """§19, §21. "from here to there" and "nothing" must not look alike."""
    assert ZONE_DASH == "–"
    assert EMPTY_SIDE == "—"
    assert ZONE_DASH != EMPTY_SIDE


# --------------------------------------------------------------------------
# §20: the shape
# --------------------------------------------------------------------------


def test_the_plan_has_both_headings_in_order() -> None:
    """§20, §21."""
    _, text = plan(bai_zones(("3900", "3910")))
    lines = text.split("\n")

    assert lines[0] == SEO_HEADING
    assert BAI_HEADING in lines
    assert lines.index(SEO_HEADING) < lines.index(BAI_HEADING)


def test_a_zone_renders_as_lower_dash_upper() -> None:
    """§19, §20."""
    _, text = plan(bai_zones(("3900", "3910")))

    assert f"3900{ZONE_DASH}3910" in text


def test_the_main_zone_carries_its_suffix() -> None:
    """§20."""
    _, text = plan(bai_zones(("3900", "3910"), ("3920", "3930")))

    assert text.count(MAIN_ZONE_SUFFIX) == 1


def test_a_farther_reference_renders_after_the_zones_with_its_suffix() -> None:
    """§18, §20. It is not a zone, so it does not sit among them."""
    features = with_references(upper=("4060",))
    selection, text = plan(features)
    lines = text.split("\n")

    assert selection.seo_reference is not None
    reference_line = f"4060 {FARTHER_SUFFIX}"
    assert reference_line in lines
    seo_body = lines[lines.index(SEO_HEADING) + 1 : lines.index("")]
    assert seo_body[-1] == reference_line
    assert len(seo_body) == 2


def test_an_empty_side_renders_one_em_dash() -> None:
    """§21. Both headings always appear; a fabricated price never does."""
    features = bai_zones(("3900", "3910"))
    _, text = plan(features)
    lines = text.split("\n")

    assert lines[lines.index(SEO_HEADING) + 1] == EMPTY_SIDE
    assert lines[lines.index(BAI_HEADING) + 1] == f"3900{ZONE_DASH}3910 {MAIN_ZONE_SUFFIX}"


def test_two_empty_sides_still_render_both_headings() -> None:
    """§21."""
    _, text = plan(empty_features())

    assert text == f"{SEO_HEADING}\n{EMPTY_SIDE}\n\n{BAI_HEADING}\n{EMPTY_SIDE}"
    assert len(text) == 12


def test_no_markdown_emoji_or_bullet_appears() -> None:
    """§20. No heading symbols, no bullets, no emoji."""
    features = with_references(upper=("4060",), lower=("3900",))
    _, text = plan(features)

    for forbidden in ("#", "*", "-", "•", "📈", "📰", "`", "_"):
        assert forbidden not in text, forbidden


def test_no_date_disclaimer_or_explanation_appears() -> None:
    """§20, §22. The final contract, checked as an absence."""
    features = with_references(upper=("4060",), lower=("3900",))
    _, text = plan(features)

    for forbidden in ("2026", ":", "%", "R:R", "rủi ro", "lưu ý", "phân tích"):
        assert forbidden not in text, forbidden


# --------------------------------------------------------------------------
# §22: the vocabulary
# --------------------------------------------------------------------------


def test_only_the_allowed_words_appear() -> None:
    """§22. Everything non-numeric on the page is on a five-word list."""
    features = with_references(upper=("4060",), lower=("3900",))
    _, text = plan(features)

    words = {
        word
        for line in text.split("\n")
        for word in line.replace("(", " ").replace(")", " ").split()
        if not any(character.isdigit() for character in word)
    }
    assert words <= PUBLIC_VOCABULARY
    assert words == {"SEO", "BAI", "vùng", "chính", "sâu", "hơn"}


@pytest.mark.parametrize("forbidden", FORBIDDEN_SUBSTRINGS)
def test_no_forbidden_word_appears(forbidden: str) -> None:
    """§22."""
    features = with_references(upper=("4060",), lower=("3900",))
    _, text = plan(features)

    assert forbidden not in text


def test_no_candidate_id_leaks() -> None:
    """§25."""
    features = with_references(upper=("4060",), lower=("3900",))
    selection, text = plan(features)

    for feature in features.candidates:
        assert feature.candidate_id not in text
    assert selection.selected_candidate_ids


# --------------------------------------------------------------------------
# §24: the cap
# --------------------------------------------------------------------------


def test_a_full_plan_stays_well_under_the_cap() -> None:
    """§24. Five zones and a reference on both sides is the worst honest case."""
    features = features_of(
        *(
            decide(gap_source(lower, upper, GapDirection.BULLISH), price="4200", tag="-bai")
            for lower, upper in SEVEN_BAI
        ),
        *(
            decide(gap_source(lower, upper, GapDirection.BEARISH), price="4200", tag="-seo")
            for lower, upper in (
                ("4300", "4310"),
                ("4320", "4330"),
                ("4340", "4350"),
                ("4360", "4370"),
                ("4380", "4390"),
            )
        ),
        price="4200",
    )
    selection, text = plan(features)

    assert len(selection.seo_entries) == 5
    assert len(selection.bai_entries) == 5
    assert len(text) <= MAX_TRADE_PLAN_CHARS
    assert len(text) < 200, f"ten zones of four digits fit easily: {len(text)}"


def test_the_cap_fails_closed_rather_than_truncating() -> None:
    """§24. A shortened trade plan is worse than a refused one."""
    from dataclasses import replace

    features = bai_zones(("3900", "3910"))
    selection = select_trade_plan(features, natural(features))
    zone = selection.bai_entries[0]
    huge = replace(
        selection,
        bai_entries=tuple(
            replace(zone, candidate_id=f"{zone.candidate_id}{index}") for index in range(60)
        ),
    )

    with pytest.raises(TradePlanRenderError, match="over the 650 cap"):
        render_trade_plan(huge)


def test_the_cap_is_650() -> None:
    assert MAX_TRADE_PLAN_CHARS == 650


# --------------------------------------------------------------------------
# §25: the validator
# --------------------------------------------------------------------------


def test_a_faithful_rendering_validates() -> None:
    features = with_references(upper=("4060",), lower=("3900",))
    selection, text = plan(features)

    validate_trade_plan(text, selection)


def test_an_unselected_price_is_refused() -> None:
    """§25. The sharpest failure the validator exists to catch."""
    features = bai_zones(("3900", "3910"))
    selection = select_trade_plan(features, natural(features))
    text = render_trade_plan(selection).replace("3910", "3915")

    with pytest.raises(TradePlanRenderError, match="not a selected candidate's price"):
        validate_trade_plan(text, selection)


def test_a_dropped_price_is_refused() -> None:
    """§25. Silently losing a line is as bad as inventing one."""
    features = bai_zones(("3900", "3910"), ("3920", "3930"))
    selection = select_trade_plan(features, natural(features))
    lines = render_trade_plan(selection).split("\n")
    text = "\n".join(lines[:-1])

    with pytest.raises(TradePlanRenderError, match="missing from the plan"):
        validate_trade_plan(text, selection)


def test_a_missing_heading_is_refused() -> None:
    """§25."""
    features = bai_zones(("3900", "3910"))
    selection = select_trade_plan(features, natural(features))
    text = render_trade_plan(selection).replace(f"{SEO_HEADING}\n", "")

    with pytest.raises(TradePlanRenderError, match="exactly one"):
        validate_trade_plan(text, selection)


def test_a_duplicated_heading_is_refused() -> None:
    """§25."""
    features = bai_zones(("3900", "3910"))
    selection = select_trade_plan(features, natural(features))
    text = render_trade_plan(selection) + f"\n{BAI_HEADING}"

    with pytest.raises(TradePlanRenderError, match="exactly one"):
        validate_trade_plan(text, selection)


def test_sections_out_of_order_are_refused() -> None:
    """§25."""
    features = bai_zones(("3900", "3910"))
    selection = select_trade_plan(features, natural(features))
    text = f"{BAI_HEADING}\n3900{ZONE_DASH}3910 {MAIN_ZONE_SUFFIX}\n\n{SEO_HEADING}\n{EMPTY_SIDE}"

    with pytest.raises(TradePlanRenderError, match="SEO must precede BAI"):
        validate_trade_plan(text, selection)


@pytest.mark.parametrize(
    "injected",
    [
        "SL 3890",
        "TP 4100",
        "BUY 3900",
        "candidate 3900",
        "score 3900",
        "http 3900",
    ],
)
def test_injected_forbidden_text_is_refused(injected: str) -> None:
    """§25."""
    features = bai_zones(("3900", "3910"))
    selection = select_trade_plan(features, natural(features))
    text = render_trade_plan(selection) + f"\n{injected}"

    with pytest.raises(TradePlanRenderError):
        validate_trade_plan(text, selection)


def test_an_unknown_word_is_refused() -> None:
    """§25, §22. The vocabulary is closed, so nothing needs enumerating."""
    features = bai_zones(("3900", "3910"))
    selection = select_trade_plan(features, natural(features))
    text = render_trade_plan(selection) + "\nvùng đẹp"

    with pytest.raises(TradePlanRenderError, match="outside the public vocabulary"):
        validate_trade_plan(text, selection)


def test_a_leaked_candidate_id_is_refused() -> None:
    """§25."""
    features = bai_zones(("3900", "3910"))
    selection = select_trade_plan(features, natural(features))
    text = render_trade_plan(selection) + f"\n{selection.bai_entries[0].candidate_id}"

    with pytest.raises(TradePlanRenderError, match="leaked into the plan"):
        validate_trade_plan(text, selection)


def test_an_extra_main_zone_label_is_refused() -> None:
    """§25, §9. One label per side, proved on the rendered text as well."""
    features = bai_zones(("3900", "3910"), ("3920", "3930"))
    selection = select_trade_plan(features, natural(features))
    text = render_trade_plan(selection).replace(
        f"3920{ZONE_DASH}3930", f"3920{ZONE_DASH}3930 {MAIN_ZONE_SUFFIX}"
    )

    with pytest.raises(TradePlanRenderError, match="main-zone labels"):
        validate_trade_plan(text, selection)


def test_an_over_long_text_is_refused_by_the_validator_too() -> None:
    """§25."""
    features = bai_zones(("3900", "3910"))
    selection = select_trade_plan(features, natural(features))
    text = render_trade_plan(selection) + "\n" + "3900" * 200

    with pytest.raises(TradePlanRenderError, match="characters"):
        validate_trade_plan(text, selection)


# --------------------------------------------------------------------------
# §16-§18: the rendered order
# --------------------------------------------------------------------------


def test_bai_lines_descend_from_the_market() -> None:
    """§17."""
    features = bai_zones(("3900", "3910"), ("3920", "3930"), ("3940", "3950"), price="4000")
    _, text = plan(features)
    body = text.split(f"{BAI_HEADING}\n")[1].split("\n")

    assert [line.split(ZONE_DASH)[0] for line in body] == ["3940", "3920", "3900"]


def test_seo_lines_climb_from_the_market() -> None:
    """§16."""
    features = seo_zones(("4000", "4010"), ("4020", "4030"), ("4040", "4050"), price="3900")
    _, text = plan(features)
    body = text.split(f"{SEO_HEADING}\n")[1].split("\n\n")[0].split("\n")

    assert [line.split(ZONE_DASH)[0] for line in body] == ["4000", "4020", "4040"]


def test_the_rendering_is_deterministic_for_one_selection() -> None:
    features = with_references(upper=("4060",), lower=("3900",))
    selection = select_trade_plan(features, natural(features))

    assert render_and_validate(selection) == render_and_validate(selection)


def test_the_ranking_moves_the_label_and_never_the_numbers() -> None:
    """§32. Same candidates, two rankings, same digits."""
    features = bai_zones(("3900", "3910"), ("3920", "3930"), ("3940", "3950"), price="4000")
    order = [
        by_bounds(features, "3900", "3910"),
        by_bounds(features, "3920", "3930"),
        by_bounds(features, "3940", "3950"),
    ]
    _, first = plan(features, bai_entry_candidate_ids=order)
    _, second = plan(features, bai_entry_candidate_ids=list(reversed(order)))

    assert first != second
    assert first.replace(f" {MAIN_ZONE_SUFFIX}", "") == second.replace(f" {MAIN_ZONE_SUFFIX}", "")


def test_a_selection_with_one_side_full_and_one_empty_renders_both() -> None:
    """§21."""
    features = seo_zones(("4000", "4010"), ("4020", "4030"), price="3900")
    _, text = plan(features)

    assert EMPTY_SIDE in text
    assert text.count(EMPTY_SIDE) == 1
    assert f"{BAI_HEADING}\n{EMPTY_SIDE}" in text


def test_the_length_of_the_realistic_shape_is_in_the_guidance_band_or_below() -> None:
    """§24. 200-450 is guidance, never a floor to pad up to."""
    features = with_references(upper=("4060",), lower=("3900",))
    _, text = plan(features)

    assert len(text) <= MAX_TRADE_PLAN_CHARS
    assert len(text) < 200, "a two-zone plan is short, and padding it would be dishonest"
