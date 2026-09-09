"""How a range opens, how far it runs, and where a price sits in it.

Round 6.6c.3b §44-§48. Small hand-checkable fixtures; the long realistic path is
in ``test_ict_range_fixture.py``.

The bar vocabulary is Round 6.6b's, because these fixtures must produce real
structure events and real protected anchors before there is a range to talk
about. Prices are gold-shaped and mean nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from decimal import Decimal

import pytest

from goldpipeline.services.ict_protected import analyse_protected_structure
from goldpipeline.services.ict_range import (
    RANGE_METHOD_VERSION,
    DealingRangeAnalysis,
    DealingRangeError,
    PriceLocation,
    RangeStatus,
    analyse_dealing_ranges,
    locate,
)
from goldpipeline.services.ict_structure import (
    BreakDirection,
    StructureBias,
    analyse_structure,
)
from tests.test_ict_protected import closing_bar, wicked
from tests.test_ict_structure import FLAT, START, Row, peak, series, trough

HOUR = timedelta(hours=1)


def ranges(rows: Sequence[Row], **kwargs: object) -> DealingRangeAnalysis:
    return analyse_dealing_ranges(series(rows), symbol="XAUUSD", **kwargs)  # type: ignore[arg-type]


def only(rows: Sequence[Row], **kwargs: object) -> object:
    result = ranges(rows, **kwargs)
    assert len(result.ranges) == 1, f"expected one range, got {len(result.ranges)}"
    return result.ranges[0]


# One bullish break with an opposite low to anchor on: origin 3970, and the
# break bar's high of 4055 is the initial terminal.
BULL: list[Row] = [
    FLAT, FLAT, peak("4030"), FLAT, FLAT, trough("3970"), FLAT, FLAT,
    wicked("4035", "4055", "3990"), wicked("4035", "4055", "3990"), FLAT, FLAT,
]  # fmt: skip

# The mirror: origin 4030, initial terminal 3945.
BEAR: list[Row] = [
    FLAT, FLAT, trough("3970"), FLAT, FLAT, peak("4030"), FLAT, FLAT,
    wicked("3965", "4010", "3945"), wicked("3965", "4010", "3945"), FLAT, FLAT,
]  # fmt: skip


def wick_high(high: str) -> Row:
    """A candle reaching *high* whose close moves nothing structurally."""
    return wicked("4000", high, "3990")


def wick_low(low: str) -> Row:
    return wicked("4000", "4010", low)


# --------------------------------------------------------------------------
# §6-§7, §44.1-2, §45.1-2: what a range looks like when it opens
# --------------------------------------------------------------------------


def test_a_bullish_range_opens_from_the_protected_low_to_the_break_bar_high() -> None:
    entry = only(BULL)

    assert entry.direction is BreakDirection.BULLISH  # type: ignore[attr-defined]
    assert entry.origin_price == Decimal("3970")  # type: ignore[attr-defined]
    assert entry.initial_terminal_price == Decimal("4055")  # type: ignore[attr-defined]
    assert entry.lower == Decimal("3970")  # type: ignore[attr-defined]
    assert entry.upper == Decimal("4055")  # type: ignore[attr-defined]
    assert entry.status is RangeStatus.ACTIVE  # type: ignore[attr-defined]


def test_a_bearish_range_opens_from_the_protected_high_to_the_break_bar_low() -> None:
    entry = only(BEAR)

    assert entry.direction is BreakDirection.BEARISH  # type: ignore[attr-defined]
    assert entry.origin_price == Decimal("4030")  # type: ignore[attr-defined]
    assert entry.initial_terminal_price == Decimal("3945")  # type: ignore[attr-defined]
    assert entry.lower == Decimal("3945")  # type: ignore[attr-defined]
    assert entry.upper == Decimal("4030")  # type: ignore[attr-defined]


def test_a_range_matches_the_leg_it_came_from() -> None:
    """§2. The leg is the authority; the range does not re-derive it."""
    protected = analyse_protected_structure(series(BULL), symbol="XAUUSD")
    (leg,) = protected.legs
    entry = only(BULL)

    assert entry.structural_leg_id == leg.leg_id  # type: ignore[attr-defined]
    assert entry.protected_assignment_id == leg.protected_assignment_id  # type: ignore[attr-defined]
    assert entry.establishing_event_id == leg.event_id  # type: ignore[attr-defined]
    assert entry.formed_at == leg.formed_at  # type: ignore[attr-defined]
    assert entry.origin_price == leg.origin_price  # type: ignore[attr-defined]
    assert entry.initial_terminal_price == leg.terminal_price  # type: ignore[attr-defined]


def test_the_formation_bar_is_the_first_terminal_witness() -> None:
    """§15."""
    entry = only(BULL)

    assert closing_bar(entry.terminal_bar_close_time) == 8  # type: ignore[attr-defined]
    assert entry.terminal_price == entry.initial_terminal_price  # type: ignore[attr-defined]


def test_a_range_carries_its_equilibrium_and_width() -> None:
    entry = only(BULL)

    assert entry.equilibrium == Decimal("4012.5")  # type: ignore[attr-defined]
    assert entry.width == Decimal("85")  # type: ignore[attr-defined]
    assert entry.equilibrium == (entry.lower + entry.upper) / Decimal(2)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# §8, §44: bullish extension
# --------------------------------------------------------------------------


def test_a_later_higher_high_extends_the_upper_edge() -> None:
    entry = only([*BULL, wick_high("4060")])

    assert entry.upper == Decimal("4060")  # type: ignore[attr-defined]
    assert entry.initial_terminal_price == Decimal("4055"), "the leg is unchanged"  # type: ignore[attr-defined]
    assert closing_bar(entry.terminal_bar_close_time) == 12  # type: ignore[attr-defined]


def test_a_later_lower_high_does_not_shrink_the_range() -> None:
    entry = only([*BULL, wick_high("4060"), wick_high("4020")])

    assert entry.upper == Decimal("4060")  # type: ignore[attr-defined]
    assert closing_bar(entry.terminal_bar_close_time) == 12  # type: ignore[attr-defined]


def test_an_equal_high_keeps_the_first_witness() -> None:
    """§14. Price revisiting a level is not price establishing it."""
    entry = only([*BULL, wick_high("4060"), wick_high("4060")])

    assert entry.upper == Decimal("4060")  # type: ignore[attr-defined]
    assert closing_bar(entry.terminal_bar_close_time) == 12, "the first bar to reach it"  # type: ignore[attr-defined]


def test_a_later_deep_low_does_not_move_the_fixed_origin() -> None:
    """§8, §10. Only a new structural event may create a different range."""
    entry = only([*BULL, wick_low("3950")])

    assert entry.lower == Decimal("3970")  # type: ignore[attr-defined]
    assert entry.origin_price == Decimal("3970")  # type: ignore[attr-defined]


def test_a_wick_below_the_origin_leaves_the_range_current_and_outside() -> None:
    """§10, §27. Price may trade outside a range that has not been replaced."""
    result = ranges([*BULL, wick_low("3950")])
    entry = result.active_range

    assert entry is not None
    assert entry.locate(Decimal("3950")) is PriceLocation.BELOW_RANGE
    assert entry.status is RangeStatus.ACTIVE


def test_equilibrium_and_width_move_with_the_extending_edge() -> None:
    before = only(BULL)
    after = only([*BULL, wick_high("4070")])

    assert before.equilibrium == Decimal("4012.5")  # type: ignore[attr-defined]
    assert after.equilibrium == Decimal("4020")  # type: ignore[attr-defined]
    assert before.width == Decimal("85")  # type: ignore[attr-defined]
    assert after.width == Decimal("100")  # type: ignore[attr-defined]


def test_the_range_id_is_unchanged_across_extension() -> None:
    """§11. The same range having run further is still the same range."""
    before = only(BULL)
    after = only([*BULL, wick_high("4070"), wick_high("4080")])

    assert before.range_id == after.range_id  # type: ignore[attr-defined]
    assert before.upper != after.upper  # type: ignore[attr-defined]


def test_a_forming_bar_cannot_extend_a_range() -> None:
    """§44.12. Only closed candles participate, as everywhere on this branch."""
    rows = [*BULL, wick_high("4090")]

    assert only(rows, as_of=START + HOUR * 12).upper == Decimal("4055")  # type: ignore[attr-defined]
    assert only(rows, as_of=START + HOUR * 13).upper == Decimal("4090")  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# §9, §45: bearish extension
# --------------------------------------------------------------------------


def test_a_later_lower_low_extends_the_lower_edge() -> None:
    entry = only([*BEAR, wick_low("3930")])

    assert entry.lower == Decimal("3930")  # type: ignore[attr-defined]
    assert entry.initial_terminal_price == Decimal("3945")  # type: ignore[attr-defined]
    assert closing_bar(entry.terminal_bar_close_time) == 12  # type: ignore[attr-defined]


def test_a_later_higher_low_does_not_shrink_a_bearish_range() -> None:
    entry = only([*BEAR, wick_low("3930"), wick_low("3960")])

    assert entry.lower == Decimal("3930")  # type: ignore[attr-defined]
    assert closing_bar(entry.terminal_bar_close_time) == 12  # type: ignore[attr-defined]


def test_an_equal_low_keeps_the_first_witness() -> None:
    entry = only([*BEAR, wick_low("3930"), wick_low("3930")])

    assert closing_bar(entry.terminal_bar_close_time) == 12  # type: ignore[attr-defined]


def test_a_high_above_the_origin_does_not_move_a_bearish_upper_edge() -> None:
    entry = only([*BEAR, wick_high("4050")])

    assert entry.upper == Decimal("4030")  # type: ignore[attr-defined]
    assert entry.origin_price == Decimal("4030")  # type: ignore[attr-defined]
    assert entry.locate(Decimal("4050")) is PriceLocation.ABOVE_RANGE  # type: ignore[attr-defined]


def test_a_bearish_equilibrium_moves_with_its_lower_edge() -> None:
    before = only(BEAR)
    after = only([*BEAR, wick_low("3930")])

    assert before.equilibrium == Decimal("3987.5")  # type: ignore[attr-defined]
    assert after.equilibrium == Decimal("3980")  # type: ignore[attr-defined]
    assert after.width == Decimal("100")  # type: ignore[attr-defined]
    assert before.range_id == after.range_id  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# §27-§29, §47: premium, discount and outside
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("price", "expected"),
    [
        ("3999.99", PriceLocation.BELOW_RANGE),
        ("4000", PriceLocation.DISCOUNT),
        ("4025", PriceLocation.DISCOUNT),
        ("4049.99", PriceLocation.DISCOUNT),
        ("4050", PriceLocation.EQUILIBRIUM),
        ("4050.01", PriceLocation.PREMIUM),
        ("4075", PriceLocation.PREMIUM),
        ("4100", PriceLocation.PREMIUM),
        ("4100.01", PriceLocation.ABOVE_RANGE),
    ],
)
def test_the_premium_discount_matrix(price: str, expected: PriceLocation) -> None:
    """§47, on the exact range the brief names."""
    assert locate(Decimal(price), lower=Decimal("4000"), upper=Decimal("4100")) is expected


def test_the_exact_boundaries_are_decided_not_accidental() -> None:
    """§28. No tolerance anywhere, least of all around equilibrium."""
    lower, upper = Decimal("4000"), Decimal("4100")

    assert locate(lower, lower=lower, upper=upper) is PriceLocation.DISCOUNT
    assert locate(upper, lower=lower, upper=upper) is PriceLocation.PREMIUM
    assert locate(Decimal("4050"), lower=lower, upper=upper) is PriceLocation.EQUILIBRIUM
    assert locate(Decimal("4050.0000001"), lower=lower, upper=upper) is PriceLocation.PREMIUM


def test_the_geometry_does_not_flip_for_a_bearish_range() -> None:
    """§29. Premium is the upper half whichever way the market got there."""
    bullish = only(BULL)
    bearish = only(BEAR)

    for entry in (bullish, bearish):
        assert entry.locate(entry.lower) is PriceLocation.DISCOUNT  # type: ignore[attr-defined]
        assert entry.locate(entry.equilibrium) is PriceLocation.EQUILIBRIUM  # type: ignore[attr-defined]
        assert entry.locate(entry.upper) is PriceLocation.PREMIUM  # type: ignore[attr-defined]
        assert entry.locate(entry.lower - Decimal(1)) is PriceLocation.BELOW_RANGE  # type: ignore[attr-defined]
        assert entry.locate(entry.upper + Decimal(1)) is PriceLocation.ABOVE_RANGE  # type: ignore[attr-defined]

    assert bullish.direction is not bearish.direction  # type: ignore[attr-defined]


def test_a_bearish_origin_is_in_premium_and_its_terminal_in_discount() -> None:
    """The consequence, stated so nobody reads it backwards later.

    A bearish range's protected high is its *upper* edge, so the anchor sits in
    premium. That is geometry, not a recommendation about either end.
    """
    entry = only(BEAR)

    assert entry.locate(entry.origin_price) is PriceLocation.PREMIUM  # type: ignore[attr-defined]
    assert entry.locate(entry.initial_terminal_price) is PriceLocation.DISCOUNT  # type: ignore[attr-defined]


def test_the_analysis_locates_a_price_in_the_current_range() -> None:
    result = ranges(BULL)

    assert result.locate(Decimal("4012.5")) is PriceLocation.EQUILIBRIUM
    assert result.locate(Decimal("3000")) is PriceLocation.BELOW_RANGE


def test_locating_against_no_range_answers_none() -> None:
    rows = [FLAT, FLAT, peak("4030"), FLAT, FLAT, trough("3970"), FLAT, FLAT]
    result = ranges(rows)

    assert result.ranges == ()
    assert result.active_range is None
    assert result.locate(Decimal("4000")) is None


def test_a_zero_width_range_is_refused_by_the_classifier() -> None:
    with pytest.raises(DealingRangeError, match="upper > lower"):
        locate(Decimal("4000"), lower=Decimal("4000"), upper=Decimal("4000"))


def test_every_range_has_positive_width() -> None:
    """§26."""
    for rows in (BULL, BEAR):
        for entry in ranges(rows).ranges:
            assert entry.width > 0


# --------------------------------------------------------------------------
# §5, §48: no anchor, no range
# --------------------------------------------------------------------------

NO_ANCHOR_BULL: list[Row] = [
    FLAT, FLAT, peak("4030"), FLAT, FLAT, wicked("4035", "4055", "3990"),
    wicked("4035", "4055", "3990"), FLAT, FLAT,
]  # fmt: skip


def test_an_initial_break_with_no_anchor_makes_no_range() -> None:
    """§48. The event is valid; it simply has nothing to anchor a range to."""
    structure = analyse_structure(series(NO_ANCHOR_BULL), symbol="XAUUSD")
    result = ranges(NO_ANCHOR_BULL)

    assert len(structure.breaks) == 1
    assert result.ranges == ()
    assert result.active_range_id is None


def test_nothing_is_substituted_for_a_missing_anchor() -> None:
    """No broken level, no event-bar extreme, no nearest swing."""
    result = ranges(NO_ANCHOR_BULL)

    assert result.ranges == ()
    assert result.structure_bias is StructureBias.BULLISH, "structure still moved"


def test_a_later_anchored_event_still_opens_a_range_normally() -> None:
    """§48's last item: a missing anchor is not a permanent state."""
    rows = [
        *NO_ANCHOR_BULL, FLAT, trough("3960"), FLAT, FLAT, FLAT,
        peak("4065"), FLAT, FLAT,
        wicked("4070", "4090", "3990"), wicked("4070", "4090", "3990"), FLAT, FLAT,
    ]  # fmt: skip
    result = ranges(rows)

    assert len(result.ranges) == 1
    entry = result.ranges[0]
    assert entry.origin_price == Decimal("3960")
    assert entry.upper == Decimal("4090")


# --------------------------------------------------------------------------
# §11, §43: identity
# --------------------------------------------------------------------------


def test_identities_are_derived_and_stable() -> None:
    one = ranges(BULL)
    two = ranges(BULL)

    assert [r.range_id for r in one.ranges] == [r.range_id for r in two.ranges]
    assert all(len(r.range_id) == 16 for r in one.ranges)


def test_a_different_symbol_gives_a_different_identity() -> None:
    gold = analyse_dealing_ranges(series(BULL), symbol="XAUUSD")
    other = analyse_dealing_ranges(series(BULL), symbol="XAGUSD")

    assert gold.ranges[0].range_id != other.ranges[0].range_id
    assert gold.ranges[0].lower == other.ranges[0].lower


def test_no_price_enters_the_range_identity() -> None:
    """§43. So no fourth Decimal canonicalisation helper was needed.

    The preimage is the symbol, timeframe, method version, assignment id and leg
    id. Assignment and leg identities are themselves built from swing and event
    ids, none of which carries a number.
    """
    import inspect

    from goldpipeline.services import ict_range

    source = inspect.getsource(ict_range._range_id)
    assert "normalize" not in source
    for forbidden in ("price", "terminal", "lower", "upper"):
        assert forbidden not in source.split('"|".join(')[1]


def test_the_method_version_is_stamped_everywhere() -> None:
    result = ranges(BULL)

    assert result.method_version == RANGE_METHOD_VERSION
    assert all(r.method_version == RANGE_METHOD_VERSION for r in result.ranges)


def test_prices_are_exact_decimals() -> None:
    entry = only(BULL)

    for value in (
        entry.origin_price,  # type: ignore[attr-defined]
        entry.terminal_price,  # type: ignore[attr-defined]
        entry.lower,  # type: ignore[attr-defined]
        entry.upper,  # type: ignore[attr-defined]
        entry.equilibrium,  # type: ignore[attr-defined]
        entry.width,  # type: ignore[attr-defined]
    ):
        assert isinstance(value, Decimal)


# --------------------------------------------------------------------------
# §36: a supplied analysis must describe this request
# --------------------------------------------------------------------------


def test_a_structure_analysis_for_another_instant_is_refused() -> None:
    snapshot = series(BULL)
    full = analyse_structure(snapshot, symbol="XAUUSD")

    with pytest.raises(DealingRangeError, match="same instant"):
        analyse_dealing_ranges(snapshot, structure=full, symbol="XAUUSD", as_of=START + HOUR * 10)


def test_a_protected_analysis_for_another_symbol_is_refused() -> None:
    snapshot = series(BULL)
    protected = analyse_protected_structure(snapshot, symbol="XAGUSD")

    with pytest.raises(DealingRangeError, match="not 'XAUUSD'"):
        analyse_dealing_ranges(snapshot, protected=protected, symbol="XAUUSD")


def test_matching_analyses_are_accepted_and_give_the_same_answer() -> None:
    """A composite stage computes structure and anchors once, then reuses them."""
    snapshot = series(BULL)
    structure = analyse_structure(snapshot, symbol="XAUUSD")
    protected = analyse_protected_structure(snapshot, structure=structure, symbol="XAUUSD")

    supplied = analyse_dealing_ranges(
        snapshot, structure=structure, protected=protected, symbol="XAUUSD"
    )
    derived = analyse_dealing_ranges(snapshot, symbol="XAUUSD")

    assert supplied == derived
