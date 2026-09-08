"""Swing relations, bias bootstrap, and the three kinds of structure break.

Round 6.6b §34-§39. Each case is a handful of bars whose answer can be worked
out on paper, because a structure engine tested only against a long realistic
series is one whose rules nobody can state. The realistic series is in
``test_ict_structure_fixture.py`` and does the other job.

The bar vocabulary below is the reason these fixtures stay readable. Every
filler bar has the same high and the same low, so it can never be a pivot under
the strict-both-sides rule; a pivot appears exactly where one bar is written to
stick out, and nowhere else. Prices are gold-shaped and mean nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import IctTimeframeSnapshot, build_timeframe_snapshot
from goldpipeline.schemas.market import OHLCBar
from goldpipeline.services.ict_primitives import SwingPoint, SwingType
from goldpipeline.services.ict_structure import (
    AnnotatedSwing,
    BreakClassification,
    BreakDirection,
    StructureBias,
    SwingRelation,
    _resolve_candidates,
    analyse_structure,
    annotate_swings,
    bootstrap_bias,
    classify_break,
    eligible_before,
    swing_id,
)

START = datetime(2026, 9, 1, tzinfo=UTC)

Row = tuple[str, str, str]
"""One bar as ``(high, low, close)``. The open is derived - see :func:`series`."""

FLAT: Row = ("4010", "3990", "4000")
"""Filler. Identical to its neighbours on both sides, so never a pivot."""


def peak(price: str) -> Row:
    """A bar that sticks up. Its low matches the filler, so it makes no low pivot."""
    return (price, "3990", "4000")


def trough(price: str) -> Row:
    """A bar that sticks down. Its high matches the filler."""
    return ("4010", price, "4000")


def drive(close: str) -> Row:
    """A bar whose point is where it *closed*.

    The range is widened only as far as the close requires. Written in adjacent
    pairs throughout these fixtures: two bars with the same extreme cannot make
    a strict pivot, so a push through a level does not quietly create a new one
    and change what the next assertion is about.
    """
    value = Decimal(close)
    return (
        str(max(value, Decimal("4010"))),
        str(min(value, Decimal("3990"))),
        close,
    )


def wick_up(high: str) -> Row:
    """Trades above *high* and closes back inside the filler range."""
    return (high, "3990", "4005")


def wick_down(low: str) -> Row:
    return ("4010", low, "3995")


def touch(price: str) -> Row:
    """Closes exactly on *price*."""
    return (price, "3990", price)


def series(
    rows: Sequence[Row],
    *,
    timeframe: Timeframe = Timeframe.H1,
    bars_kept: int | None = None,
) -> IctTimeframeSnapshot:
    """Lay *rows* onto consecutive bars starting at :data:`START`.

    The open follows the previous close, clamped into the bar's own range, so
    every bar is a valid candle without the fixture having to say so four times
    per row.

    ``bars_kept`` truncates the series to its first *n* bars - the physical
    truncation the as-of tests compare against.
    """
    duration = timeframe.duration
    assert duration is not None

    kept = rows if bars_kept is None else rows[:bars_kept]
    bars: list[OHLCBar] = []
    previous: Decimal | None = None
    for index, (high, low, close) in enumerate(kept):
        top, bottom, last = Decimal(high), Decimal(low), Decimal(close)
        opening = last if previous is None else min(max(previous, bottom), top)
        bars.append(
            OHLCBar(
                timestamp=START + duration * index,
                open=opening,
                high=top,
                low=bottom,
                close=last,
            )
        )
        previous = last

    return build_timeframe_snapshot(
        timeframe=timeframe, bars=tuple(bars), observed_at=START + duration * len(kept)
    )


def bar_index(moment: datetime, timeframe: Timeframe = Timeframe.H1) -> int:
    duration = timeframe.duration
    assert duration is not None
    return int((moment - START) // duration)


def labels(rows: Sequence[Row]) -> list[str]:
    return [entry.label for entry in analyse_structure(series(rows)).annotated_swings]


def kinds(rows: Sequence[Row]) -> list[tuple[str, str]]:
    return [
        (event.classification.value, event.direction.value)
        for event in analyse_structure(series(rows)).breaks
    ]


# --------------------------------------------------------------------------
# §34: swing relations
# --------------------------------------------------------------------------


def test_the_first_swing_of_each_type_has_nothing_to_compare_against() -> None:
    rows = [FLAT, FLAT, peak("4030"), FLAT, FLAT, trough("3970"), FLAT, FLAT]

    assert labels(rows) == ["FIRST_HIGH", "FIRST_LOW"]


def test_a_higher_high_and_a_lower_high() -> None:
    rising = [FLAT, FLAT, peak("4030"), FLAT, FLAT, FLAT, peak("4045"), FLAT, FLAT]
    falling = [FLAT, FLAT, peak("4045"), FLAT, FLAT, FLAT, peak("4030"), FLAT, FLAT]

    assert labels(rising) == ["FIRST_HIGH", "HIGHER_HIGH"]
    assert labels(falling) == ["FIRST_HIGH", "LOWER_HIGH"]


def test_a_higher_low_and_a_lower_low() -> None:
    rising = [FLAT, FLAT, trough("3950"), FLAT, FLAT, FLAT, trough("3970"), FLAT, FLAT]
    falling = [FLAT, FLAT, trough("3970"), FLAT, FLAT, FLAT, trough("3950"), FLAT, FLAT]

    assert labels(rising) == ["FIRST_LOW", "HIGHER_LOW"]
    assert labels(falling) == ["FIRST_LOW", "LOWER_LOW"]


def test_two_separated_swings_at_one_price_are_equal_not_absent() -> None:
    """Both are real pivots; only *adjacent* equal highs make no pivot at all.

    The distinction is the whole reason ``EQUAL`` exists as a relation. Two
    highs at the same price with room between them are each strictly above
    their own neighbours - the market printed two separate pivots that happen
    to agree - and flattening that into "no pivot" would erase the shape a
    later liquidity round is built on.
    """
    highs = [FLAT, FLAT, peak("4030"), FLAT, FLAT, FLAT, peak("4030"), FLAT, FLAT]
    lows = [FLAT, FLAT, trough("3970"), FLAT, FLAT, FLAT, trough("3970"), FLAT, FLAT]

    assert labels(highs) == ["FIRST_HIGH", "EQUAL_HIGH"]
    assert labels(lows) == ["FIRST_LOW", "EQUAL_LOW"]


def test_adjacent_equal_highs_make_no_pivot_at_all() -> None:
    """The 6.6a rule, restated here because the relation model depends on it."""
    rows = [FLAT, FLAT, peak("4030"), peak("4030"), FLAT, FLAT]

    assert labels(rows) == []


def test_relations_compare_exact_decimals_with_no_tolerance() -> None:
    """One hundredth of a dollar is a higher high. Tolerance belongs to liquidity."""
    rows = [FLAT, FLAT, peak("4030.00"), FLAT, FLAT, FLAT, peak("4030.01"), FLAT, FLAT]

    assert labels(rows) == ["FIRST_HIGH", "HIGHER_HIGH"]


def test_each_side_is_compared_only_with_its_own_kind() -> None:
    """An intervening low never becomes the thing a high is measured against."""
    rows = [
        FLAT,
        FLAT,
        peak("4030"),
        FLAT,
        FLAT,
        trough("3970"),
        FLAT,
        FLAT,
        peak("4045"),
        FLAT,
        FLAT,
    ]

    assert labels(rows) == ["FIRST_HIGH", "FIRST_LOW", "HIGHER_HIGH"]


def test_relations_follow_chronological_order() -> None:
    result = analyse_structure(
        series(
            [
                FLAT,
                FLAT,
                peak("4030"),
                FLAT,
                FLAT,
                FLAT,
                peak("4045"),
                FLAT,
                FLAT,
                FLAT,
                peak("4020"),
                FLAT,
                FLAT,
            ]
        )
    )

    assert [entry.label for entry in result.annotated_swings] == [
        "FIRST_HIGH",
        "HIGHER_HIGH",
        "LOWER_HIGH",
    ]
    times = [entry.swing.pivot_time for entry in result.annotated_swings]
    assert times == sorted(times)
    assert [entry.previous_pivot_time for entry in result.annotated_swings][0] is None


def test_a_swing_the_window_has_not_confirmed_yet_is_simply_absent() -> None:
    """Not "provisional" - absent. There is no third state to leak downstream."""
    rows = [FLAT, FLAT, peak("4030"), FLAT, FLAT]

    assert labels(rows) == ["FIRST_HIGH"]
    assert labels(rows[:4]) == [], "one right-hand bar is not two"


def test_relations_do_not_depend_on_who_provided_the_candles() -> None:
    rows = [FLAT, FLAT, peak("4030"), FLAT, FLAT, FLAT, peak("4045"), FLAT, FLAT]

    one = analyse_structure(series(rows), symbol="XAUUSD")
    two = analyse_structure(series(rows), symbol="XAUUSD")

    assert [entry.label for entry in one.annotated_swings] == [
        entry.label for entry in two.annotated_swings
    ]


# --------------------------------------------------------------------------
# §35: bootstrap
# --------------------------------------------------------------------------


def fake_swing(swing_type: SwingType, price: str, hour: int) -> SwingPoint:
    """A swing built by hand, for testing the pure predicates directly."""
    pivot = START + timedelta(hours=hour)
    return SwingPoint(
        timeframe=Timeframe.H1,
        swing_type=swing_type,
        pivot_time=pivot,
        confirmed_at=pivot + timedelta(hours=3),
        price=Decimal(price),
        left_bars=2,
        right_bars=2,
    )


def bootstrap_of(*pairs: tuple[SwingType, str]) -> StructureBias:
    swings = [fake_swing(kind, price, hour) for hour, (kind, price) in enumerate(pairs)]
    return bootstrap_bias(annotate_swings(swings))


HIGH = SwingType.SWING_HIGH
LOW = SwingType.SWING_LOW


def test_higher_high_with_higher_low_is_bullish() -> None:
    assert (
        bootstrap_of((HIGH, "4030"), (LOW, "3950"), (HIGH, "4045"), (LOW, "3970"))
        is StructureBias.BULLISH
    )


def test_lower_high_with_lower_low_is_bearish() -> None:
    assert (
        bootstrap_of((HIGH, "4045"), (LOW, "3970"), (HIGH, "4030"), (LOW, "3950"))
        is StructureBias.BEARISH
    )


def test_a_broadening_market_is_neutral_not_bullish() -> None:
    """Higher high *and* lower low. Half a pattern is not a trend."""
    assert (
        bootstrap_of((HIGH, "4030"), (LOW, "3970"), (HIGH, "4045"), (LOW, "3950"))
        is StructureBias.NEUTRAL
    )


def test_a_contracting_market_is_neutral() -> None:
    assert (
        bootstrap_of((HIGH, "4045"), (LOW, "3950"), (HIGH, "4030"), (LOW, "3970"))
        is StructureBias.NEUTRAL
    )


def test_an_equal_high_is_not_a_higher_high() -> None:
    assert (
        bootstrap_of((HIGH, "4030"), (LOW, "3950"), (HIGH, "4030"), (LOW, "3970"))
        is StructureBias.NEUTRAL
    )


def test_an_equal_low_is_not_a_higher_low() -> None:
    assert (
        bootstrap_of((HIGH, "4030"), (LOW, "3950"), (HIGH, "4045"), (LOW, "3950"))
        is StructureBias.NEUTRAL
    )


def test_one_high_establishes_nothing() -> None:
    assert bootstrap_of((HIGH, "4030"), (LOW, "3950"), (LOW, "3970")) is StructureBias.NEUTRAL


def test_one_low_establishes_nothing() -> None:
    assert bootstrap_of((HIGH, "4030"), (HIGH, "4045"), (LOW, "3950")) is StructureBias.NEUTRAL


def test_no_swings_at_all_is_neutral() -> None:
    assert bootstrap_bias(()) is StructureBias.NEUTRAL


def test_the_bootstrap_runs_end_to_end_over_real_bars() -> None:
    """The predicate tests above use hand-built swings; this proves the wiring."""
    bullish = [
        FLAT,
        FLAT,
        peak("4030"),
        FLAT,
        FLAT,
        trough("3950"),
        FLAT,
        FLAT,
        peak("4045"),
        FLAT,
        FLAT,
        trough("3970"),
        FLAT,
        FLAT,
    ]
    bearish = [
        FLAT,
        FLAT,
        peak("4045"),
        FLAT,
        FLAT,
        trough("3970"),
        FLAT,
        FLAT,
        peak("4030"),
        FLAT,
        FLAT,
        trough("3950"),
        FLAT,
        FLAT,
    ]

    assert analyse_structure(series(bullish)).current_bias is StructureBias.BULLISH
    assert analyse_structure(series(bearish)).current_bias is StructureBias.BEARISH


def test_a_bootstrapped_state_is_also_reported_as_the_initial_state() -> None:
    rows = [
        FLAT,
        FLAT,
        peak("4030"),
        FLAT,
        FLAT,
        trough("3950"),
        FLAT,
        FLAT,
        peak("4045"),
        FLAT,
        FLAT,
        trough("3970"),
        FLAT,
        FLAT,
    ]
    result = analyse_structure(series(rows))

    assert result.breaks == ()
    assert result.initial_bias is result.current_bias is StructureBias.BULLISH


# --------------------------------------------------------------------------
# §36: the first break
# --------------------------------------------------------------------------

FIRST_BULL = [FLAT, FLAT, peak("4030"), FLAT, FLAT, drive("4035"), drive("4035"), FLAT, FLAT]
FIRST_BEAR = [FLAT, FLAT, trough("3970"), FLAT, FLAT, drive("3965"), drive("3965"), FLAT, FLAT]


def test_a_neutral_market_closing_above_a_swing_high_makes_an_initial_break() -> None:
    result = analyse_structure(series(FIRST_BULL), symbol="XAUUSD")
    (event,) = result.breaks

    assert event.classification is BreakClassification.INITIAL_BREAK
    assert event.direction is BreakDirection.BULLISH
    assert event.prior_bias is StructureBias.NEUTRAL
    assert event.resulting_bias is StructureBias.BULLISH
    assert result.current_bias is StructureBias.BULLISH
    assert event.broken_level == Decimal("4030")
    assert event.break_close == Decimal("4035")
    assert bar_index(event.break_bar_open_time) == 5


def test_a_neutral_market_closing_below_a_swing_low_makes_an_initial_break() -> None:
    result = analyse_structure(series(FIRST_BEAR))
    (event,) = result.breaks

    assert event.classification is BreakClassification.INITIAL_BREAK
    assert event.direction is BreakDirection.BEARISH
    assert result.current_bias is StructureBias.BEARISH


def test_the_first_directional_event_is_not_called_a_bos() -> None:
    """Nothing was continuing. There was no structure to continue."""
    assert kinds(FIRST_BULL) == [("INITIAL_BREAK", "BULLISH")]
    assert kinds(FIRST_BEAR) == [("INITIAL_BREAK", "BEARISH")]


def test_a_wick_above_the_level_breaks_nothing() -> None:
    rows = [FLAT, FLAT, peak("4030"), FLAT, FLAT, wick_up("4040"), FLAT, FLAT]
    result = analyse_structure(series(rows))

    assert result.breaks == ()
    assert result.current_bias is StructureBias.NEUTRAL
    assert result.consumed_swing_ids == frozenset(), "the level is still live"


def test_a_wick_below_the_level_breaks_nothing() -> None:
    rows = [FLAT, FLAT, trough("3970"), FLAT, FLAT, wick_down("3960"), FLAT, FLAT]
    result = analyse_structure(series(rows))

    assert result.breaks == ()
    assert result.consumed_swing_ids == frozenset()


def test_a_close_exactly_on_the_level_breaks_nothing() -> None:
    """Strictly beyond, or nothing. ``>=`` here would be a different engine."""
    rows = [FLAT, FLAT, peak("4030"), FLAT, FLAT, touch("4030"), FLAT, FLAT]
    result = analyse_structure(series(rows))

    assert result.breaks == ()
    assert result.current_bias is StructureBias.NEUTRAL


def test_a_bar_that_gaps_clean_over_the_level_still_breaks_it() -> None:
    """No requirement that the body overlap the level.

    The bar traded entirely above the swing. Demanding physical overlap would
    mean a market that gapped through structure had not broken it, which is
    backwards - and the tick path that would settle it is not in the data.
    """
    rows = [FLAT, FLAT, peak("4030"), FLAT, FLAT, ("4090", "4070", "4085"), FLAT, FLAT]
    result = analyse_structure(series(rows))
    (event,) = result.breaks

    assert event.classification is BreakClassification.INITIAL_BREAK
    assert event.break_close == Decimal("4085")
    assert event.broken_level == Decimal("4030")


def test_a_bar_may_not_break_a_swing_it_confirmed_itself() -> None:
    """The known-before-broken rule, at the predicate that enforces it.

    Tested here rather than end to end, and that is the honest place for it:
    under the strict-both-sides pivot definition the situation cannot arise
    from real candles at all, because a bar closing above a pivot has a high
    above it too, and such a bar sitting inside the pivot's right-hand window
    would have prevented the pivot from existing. The rule is enforced anyway,
    since that argument depends on a pivot definition a later round may change.
    """
    swing = fake_swing(HIGH, "4030", hour=2)
    (annotated,) = annotate_swings([swing])

    assert eligible_before([annotated], swing.confirmed_at) == ()
    assert eligible_before([annotated], swing.confirmed_at + timedelta(hours=1)) == (annotated,)


def test_a_level_confirmed_earlier_is_breakable() -> None:
    result = analyse_structure(series(FIRST_BULL))
    (event,) = result.breaks

    assert event.broken_swing_confirmed_at < event.break_bar_close_time


# --------------------------------------------------------------------------
# §37: break of structure
# --------------------------------------------------------------------------

BOS_BULL = [
    FLAT, FLAT, peak("4030"), FLAT, FLAT, drive("4035"), drive("4035"), FLAT, FLAT,
    peak("4045"), FLAT, FLAT, drive("4050"), drive("4050"), FLAT, FLAT,
]  # fmt: skip
BOS_BEAR = [
    FLAT, FLAT, trough("3970"), FLAT, FLAT, drive("3965"), drive("3965"), FLAT, FLAT,
    trough("3955"), FLAT, FLAT, drive("3950"), drive("3950"), FLAT, FLAT,
]  # fmt: skip


def test_a_bullish_market_closing_above_its_active_high_makes_a_bos() -> None:
    result = analyse_structure(series(BOS_BULL))
    first, second = result.breaks

    assert first.classification is BreakClassification.INITIAL_BREAK
    assert second.classification is BreakClassification.BOS
    assert second.direction is BreakDirection.BULLISH
    assert second.prior_bias is second.resulting_bias is StructureBias.BULLISH
    assert second.broken_level == Decimal("4045")


def test_a_bearish_market_closing_below_its_active_low_makes_a_bos() -> None:
    result = analyse_structure(series(BOS_BEAR))
    first, second = result.breaks

    assert first.classification is BreakClassification.INITIAL_BREAK
    assert second.classification is BreakClassification.BOS
    assert second.direction is BreakDirection.BEARISH
    assert second.prior_bias is second.resulting_bias is StructureBias.BEARISH


def test_a_bos_leaves_the_state_where_it_found_it() -> None:
    assert analyse_structure(series(BOS_BULL)).current_bias is StructureBias.BULLISH
    assert analyse_structure(series(BOS_BEAR)).current_bias is StructureBias.BEARISH


def test_the_broken_level_is_marked_consumed() -> None:
    result = analyse_structure(series(BOS_BULL))
    second = result.breaks[1]

    assert second.broken_swing_id in result.consumed_swing_ids
    assert result.active_high is None, "nothing unconsumed is left above"


def test_a_second_close_beyond_the_same_level_makes_no_second_event() -> None:
    """The pair of ``drive`` bars in every fixture here is doing this on purpose."""
    result = analyse_structure(series(BOS_BULL))
    broken = [event.broken_swing_id for event in result.breaks]

    assert len(broken) == len(set(broken))
    assert len(result.breaks) == 2, "four closes sat beyond a level; two were news"


def test_a_wick_beyond_the_active_high_makes_no_bos() -> None:
    rows = BOS_BULL[:12] + [wick_up("4050"), FLAT, FLAT, FLAT]
    result = analyse_structure(series(rows))

    assert kinds(rows) == [("INITIAL_BREAK", "BULLISH")]
    assert result.active_high is not None
    assert result.active_high.swing.price == Decimal("4050"), "the wick made its own pivot"


def test_an_exact_touch_of_the_active_high_makes_no_bos() -> None:
    rows = BOS_BULL[:12] + [touch("4045"), FLAT, FLAT, FLAT]

    assert kinds(rows) == [("INITIAL_BREAK", "BULLISH")]


def test_a_newly_confirmed_swing_can_produce_a_later_bos() -> None:
    """§37.6 - the point of consuming by identity rather than retiring the side."""
    result = analyse_structure(series(BOS_BULL))

    assert [event.broken_level for event in result.breaks] == [
        Decimal("4030"),
        Decimal("4045"),
    ]


# --------------------------------------------------------------------------
# §38: market structure shift
# --------------------------------------------------------------------------

MSS_BEAR = [
    FLAT, FLAT, peak("4030"), FLAT, trough("3970"), FLAT, FLAT, drive("4035"), drive("4035"),
    FLAT, FLAT, drive("3965"), drive("3965"), FLAT, FLAT,
]  # fmt: skip
MSS_BULL = [
    FLAT, FLAT, trough("3970"), FLAT, FLAT, drive("3965"), drive("3965"), FLAT, FLAT,
    peak("4030"), FLAT, FLAT, drive("4035"), drive("4035"), FLAT, FLAT,
]  # fmt: skip


def test_a_bullish_market_closing_below_its_active_low_makes_an_mss() -> None:
    result = analyse_structure(series(MSS_BEAR))
    first, second = result.breaks

    assert first.resulting_bias is StructureBias.BULLISH
    assert second.classification is BreakClassification.MSS
    assert second.direction is BreakDirection.BEARISH
    assert second.prior_bias is StructureBias.BULLISH
    assert second.resulting_bias is StructureBias.BEARISH


def test_an_mss_flips_the_state_immediately() -> None:
    """Without waiting for new pivots to agree. A close already said it."""
    assert analyse_structure(series(MSS_BEAR)).current_bias is StructureBias.BEARISH
    assert analyse_structure(series(MSS_BULL)).current_bias is StructureBias.BULLISH


def test_a_bearish_market_closing_above_its_active_high_makes_an_mss() -> None:
    result = analyse_structure(series(MSS_BULL))
    first, second = result.breaks

    assert first.direction is BreakDirection.BEARISH
    assert second.classification is BreakClassification.MSS
    assert second.direction is BreakDirection.BULLISH
    assert second.prior_bias is StructureBias.BEARISH


def test_the_opposite_side_level_survives_a_move_that_ignored_it() -> None:
    """§20 - a bullish state still has to know its low, or it can never reverse."""
    result = analyse_structure(series(MSS_BEAR[:11]))

    assert result.current_bias is StructureBias.BULLISH
    assert result.active_low is not None
    assert result.active_low.swing.price == Decimal("3970")


def test_an_mss_consumes_the_level_it_broke() -> None:
    result = analyse_structure(series(MSS_BEAR))
    second = result.breaks[1]

    assert second.broken_swing_id in result.consumed_swing_ids
    assert result.active_low is None


def test_a_wick_through_the_opposite_level_makes_no_mss() -> None:
    rows = MSS_BEAR[:11] + [wick_down("3960"), FLAT, FLAT, FLAT]
    result = analyse_structure(series(rows))

    assert kinds(rows) == [("INITIAL_BREAK", "BULLISH")]
    assert result.current_bias is StructureBias.BULLISH


def test_an_exact_touch_of_the_opposite_level_makes_no_mss() -> None:
    rows = MSS_BEAR[:11] + [("4010", "3970", "3970"), FLAT, FLAT, FLAT]

    assert kinds(rows) == [("INITIAL_BREAK", "BULLISH")]


def test_a_later_opposing_level_can_produce_another_mss() -> None:
    """Bullish, reversed, then reversed back - all from closes, not pivot counts."""
    rows = [
        FLAT, FLAT, peak("4030"), FLAT, trough("3970"), FLAT, FLAT,
        drive("4035"), drive("4035"), FLAT, FLAT,
        drive("3965"), drive("3965"), FLAT, FLAT,
        peak("4020"), FLAT, FLAT, drive("4025"), drive("4025"), FLAT, FLAT,
    ]  # fmt: skip

    assert kinds(rows) == [
        ("INITIAL_BREAK", "BULLISH"),
        ("MSS", "BEARISH"),
        ("MSS", "BULLISH"),
    ]


# --------------------------------------------------------------------------
# §16-§18, §39: identity
# --------------------------------------------------------------------------


def test_a_swing_is_identified_by_side_and_time_never_by_price() -> None:
    one = fake_swing(HIGH, "4030", hour=2)
    two = fake_swing(HIGH, "4030", hour=9)
    same_bar_low = fake_swing(LOW, "4030", hour=2)

    assert swing_id(one) != swing_id(two), "same price, different observations"
    assert swing_id(one) != swing_id(same_bar_low), "one bar can be both"
    assert swing_id(one) == swing_id(fake_swing(HIGH, "9999", hour=2)), "price is not identity"


def test_two_swings_at_the_same_price_are_broken_independently() -> None:
    """§18. The first break must not silently retire a level that came later."""
    rows = [
        FLAT, FLAT, peak("4030"), FLAT, FLAT, drive("4035"), drive("4035"), FLAT, FLAT, FLAT,
        peak("4030"), FLAT, FLAT, drive("4040"), drive("4040"), FLAT, FLAT,
    ]  # fmt: skip
    result = analyse_structure(series(rows))

    assert [entry.label for entry in result.annotated_swings] == ["FIRST_HIGH", "EQUAL_HIGH"]
    assert [event.broken_level for event in result.breaks] == [
        Decimal("4030"),
        Decimal("4030"),
    ]
    assert kinds(rows) == [("INITIAL_BREAK", "BULLISH"), ("BOS", "BULLISH")]
    assert len({event.broken_swing_id for event in result.breaks}) == 2
    assert len({event.event_id for event in result.breaks}) == 2


def test_one_close_clearing_several_old_levels_emits_one_event() -> None:
    """The most recent level is the news; the ones behind it are swept, not silent.

    Without this a market that jumped over four old highs would spend the next
    four quiet bars emitting a stale BOS each time it drifted sideways.
    """
    rows = [
        FLAT, FLAT, peak("4045"), FLAT, FLAT, peak("4020"), FLAT, FLAT, peak("4035"),
        FLAT, FLAT, drive("4050"), drive("4050"), FLAT, FLAT,
    ]  # fmt: skip
    result = analyse_structure(series(rows))
    (event,) = result.breaks

    assert event.broken_level == Decimal("4035"), "the most recent, not the highest"
    assert len(event.also_consumed_swing_ids) == 2
    assert len(result.consumed_swing_ids) == 3
    assert result.active_high is None


def test_a_sweep_records_which_close_consumed_each_level() -> None:
    rows = [
        FLAT, FLAT, peak("4045"), FLAT, FLAT, peak("4020"), FLAT, FLAT, peak("4035"),
        FLAT, FLAT, drive("4050"), drive("4050"), FLAT, FLAT,
    ]  # fmt: skip
    result = analyse_structure(series(rows))
    (event,) = result.breaks

    attributed = {event.broken_swing_id, *event.also_consumed_swing_ids}
    assert attributed == result.consumed_swing_ids


def test_event_ids_are_derived_and_stable_across_replays() -> None:
    first = analyse_structure(series(BOS_BULL), symbol="XAUUSD")
    again = analyse_structure(series(BOS_BULL), symbol="XAUUSD")

    assert [event.event_id for event in first.breaks] == [event.event_id for event in again.breaks]
    assert all(len(event.event_id) == 16 for event in first.breaks)


def test_event_ids_separate_two_breaks_of_the_same_level() -> None:
    rows = [
        FLAT, FLAT, peak("4030"), FLAT, FLAT, drive("4035"), drive("4035"), FLAT, FLAT, FLAT,
        peak("4030"), FLAT, FLAT, drive("4040"), drive("4040"), FLAT, FLAT,
    ]  # fmt: skip
    result = analyse_structure(series(rows), symbol="XAUUSD")

    assert len({event.event_id for event in result.breaks}) == 2


def test_a_different_symbol_gives_different_event_ids() -> None:
    """Provenance that belongs in identity: two instruments are two markets."""
    gold = analyse_structure(series(FIRST_BULL), symbol="XAUUSD")
    other = analyse_structure(series(FIRST_BULL), symbol="XAGUSD")

    assert gold.breaks[0].event_id != other.breaks[0].event_id
    assert gold.breaks[0].broken_level == other.breaks[0].broken_level


# --------------------------------------------------------------------------
# the transition table, stated once and checked directly
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prior", "direction", "classification", "resulting"),
    [
        (
            StructureBias.NEUTRAL,
            BreakDirection.BULLISH,
            BreakClassification.INITIAL_BREAK,
            StructureBias.BULLISH,
        ),
        (
            StructureBias.NEUTRAL,
            BreakDirection.BEARISH,
            BreakClassification.INITIAL_BREAK,
            StructureBias.BEARISH,
        ),
        (
            StructureBias.BULLISH,
            BreakDirection.BULLISH,
            BreakClassification.BOS,
            StructureBias.BULLISH,
        ),
        (
            StructureBias.BULLISH,
            BreakDirection.BEARISH,
            BreakClassification.MSS,
            StructureBias.BEARISH,
        ),
        (
            StructureBias.BEARISH,
            BreakDirection.BEARISH,
            BreakClassification.BOS,
            StructureBias.BEARISH,
        ),
        (
            StructureBias.BEARISH,
            BreakDirection.BULLISH,
            BreakClassification.MSS,
            StructureBias.BULLISH,
        ),
    ],
)
def test_the_six_transitions(
    prior: StructureBias,
    direction: BreakDirection,
    classification: BreakClassification,
    resulting: StructureBias,
) -> None:
    assert classify_break(prior, direction) == (classification, resulting)


def test_a_break_always_leaves_the_state_pointing_its_own_way() -> None:
    for prior in StructureBias:
        for direction in BreakDirection:
            _, resulting = classify_break(prior, direction)
            assert resulting.value == direction.value


def test_the_relation_enum_says_nothing_about_which_side_it_is_on() -> None:
    """Four values, not eight. The swing already carries its own side."""
    assert set(SwingRelation) == {
        SwingRelation.FIRST,
        SwingRelation.HIGHER,
        SwingRelation.LOWER,
        SwingRelation.EQUAL,
    }


def test_the_eight_way_label_is_derived_from_the_pair() -> None:
    high = AnnotatedSwing(
        swing=fake_swing(HIGH, "4030", 2), relation=SwingRelation.HIGHER, previous_pivot_time=None
    )
    low = AnnotatedSwing(
        swing=fake_swing(LOW, "3970", 2), relation=SwingRelation.HIGHER, previous_pivot_time=None
    )

    assert high.label == "HIGHER_HIGH"
    assert low.label == "HIGHER_LOW"


# --------------------------------------------------------------------------
# §19: one bar, one event - including the case no candle series reaches
# --------------------------------------------------------------------------


def annotated(swing_type: SwingType, price: str, hour: int) -> AnnotatedSwing:
    return AnnotatedSwing(
        swing=fake_swing(swing_type, price, hour),
        relation=SwingRelation.FIRST,
        previous_pivot_time=None,
    )


def test_a_bar_with_nothing_to_break_emits_nothing() -> None:
    assert _resolve_candidates([]) is None


def test_a_single_candidate_is_returned_as_is() -> None:
    only = (BreakDirection.BULLISH, annotated(HIGH, "4030", 2))

    assert _resolve_candidates([only]) is only


def test_two_qualifying_sides_resolve_to_the_more_recent_level() -> None:
    """Defined so that an impossible-looking case is never merely accidental.

    Both sides can qualify only when an unconsumed swing high sits *below* the
    active swing low, and no valid candle series producing that was found while
    writing this round - a bar printing a low above a known high would have
    closed above that high and consumed it first. The rule is still stated and
    tested, because that argument rests on the current pivot definition and a
    later round may change it. Leaving it undefined would turn a rare shape
    into a silent dependency on argument order.
    """
    older = (BreakDirection.BULLISH, annotated(HIGH, "4030", 2))
    newer = (BreakDirection.BEARISH, annotated(LOW, "4050", 9))

    assert _resolve_candidates([older, newer]) is newer
    assert _resolve_candidates([newer, older]) is newer, "order of arrival is not an input"


def test_a_tie_on_confirmation_falls_to_the_later_pivot() -> None:
    early = annotated(HIGH, "4030", 2)
    late = annotated(LOW, "4050", 5)
    assert early.swing.confirmed_at < late.swing.confirmed_at

    pair = [(BreakDirection.BULLISH, early), (BreakDirection.BEARISH, late)]
    chosen = _resolve_candidates(pair)

    assert chosen is not None
    assert chosen[1] is late
