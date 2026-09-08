"""What a later candle did to a gap, one rule at a time.

Round 6.6c.2 §37-§40. Small hand-checkable fixtures; the long realistic path is
in ``test_ict_fvg_fixture.py``.

Every fixture here builds one deliberate gap out of three candles and then adds
later candles to act on it. Those later candles form gaps of their own - three
consecutive bars almost always do - so assertions select the gap by the bar its
third candle closed on rather than assuming there is only one. That is honest
about what a real series looks like, and it exercises multi-gap independence
for free.

Prices are gold-shaped and mean nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import IctTimeframeSnapshot, build_timeframe_snapshot
from goldpipeline.schemas.market import OHLCBar
from goldpipeline.services.ict_fvg import (
    FVG_LIFECYCLE_METHOD_VERSION,
    FvgLifecycleAnalysis,
    FvgState,
    FvgStatus,
    analyse_fvg_lifecycle,
    covers,
    enters,
    fvg_id,
    touchable,
)
from goldpipeline.services.ict_primitives import FairValueGap, GapDirection, fair_value_gaps

START = datetime(2026, 9, 1, tzinfo=UTC)
HOUR = timedelta(hours=1)

Row = tuple[str, str]
"""One bar as ``(high, low)``. The close sits at the midpoint of the range."""


def bar(high: str, low: str) -> Row:
    return (high, low)


def series(
    rows: Sequence[Row],
    *,
    timeframe: Timeframe = Timeframe.H1,
    bars_kept: int | None = None,
) -> IctTimeframeSnapshot:
    """Lay *rows* onto consecutive bars starting at :data:`START`.

    The close is the midpoint of each range and the open follows the previous
    close, clamped inside. Neither participates in any rule this module tests -
    fair value gaps and their lifecycle are decided by highs and lows alone -
    so the fixtures only ever have to state the two numbers that matter.
    """
    duration = timeframe.duration
    assert duration is not None

    kept = rows if bars_kept is None else rows[:bars_kept]
    bars: list[OHLCBar] = []
    previous: Decimal | None = None
    for index, (high, low) in enumerate(kept):
        top, bottom = Decimal(high), Decimal(low)
        close = (top + bottom) / Decimal(2)
        opening = close if previous is None else min(max(previous, bottom), top)
        bars.append(
            OHLCBar(
                timestamp=START + duration * index,
                open=opening,
                high=top,
                low=bottom,
                close=close,
            )
        )
        previous = close

    return build_timeframe_snapshot(
        timeframe=timeframe, bars=tuple(bars), observed_at=START + duration * len(kept)
    )


def bar_index(moment: datetime, timeframe: Timeframe = Timeframe.H1) -> int:
    duration = timeframe.duration
    assert duration is not None
    return int((moment - START) // duration) - 1


def analyse(rows: Sequence[Row], **kwargs: object) -> FvgLifecycleAnalysis:
    return analyse_fvg_lifecycle(series(rows), symbol="XAUUSD", **kwargs)  # type: ignore[arg-type]


def formed_on(result: FvgLifecycleAnalysis, index: int) -> FvgState:
    """The one gap whose third candle closed on bar *index*."""
    matches = [s for s in result.states if bar_index(s.gap.formed_at) == index]
    assert len(matches) == 1, f"expected one gap formed on bar {index}, found {len(matches)}"
    return matches[0]


# A bullish gap: A.high = 4000, C.low = 4010, so the untraded band is
# [4000, 4010] and it is knowable once bar 2 closes.
BULL: list[Row] = [bar("4000", "3990"), bar("4008", "3998"), bar("4020", "4010")]

# A bearish gap over the same band: A.low = 4010, C.high = 4000.
BEAR: list[Row] = [bar("4020", "4010"), bar("4012", "4002"), bar("4000", "3990")]

CLEAR_ABOVE = bar("4030", "4020")
"""Sits entirely above ``[4000, 4010]`` without contacting it."""


def after(*later: Row) -> list[Row]:
    return [*BULL, CLEAR_ABOVE, *later]


# --------------------------------------------------------------------------
# §3: identity
# --------------------------------------------------------------------------


def sample_gap(**overrides: object) -> FairValueGap:
    defaults: dict[str, object] = {
        "timeframe": Timeframe.H1,
        "direction": GapDirection.BULLISH,
        "formed_at": START + HOUR * 3,
        "first_time": START,
        "middle_time": START + HOUR,
        "last_time": START + HOUR * 2,
        "lower": Decimal("4000"),
        "upper": Decimal("4010"),
    }
    return FairValueGap(**{**defaults, **overrides})  # type: ignore[arg-type]


def test_identity_is_derived_and_stable() -> None:
    assert fvg_id(sample_gap(), symbol="XAUUSD") == fvg_id(sample_gap(), symbol="XAUUSD")
    assert len(fvg_id(sample_gap())) == 16


def test_identity_separates_two_gaps_that_share_a_band() -> None:
    """§24. Same band, different formation, so two observations - not one."""
    early = sample_gap()
    late = sample_gap(
        first_time=START + HOUR * 10,
        middle_time=START + HOUR * 11,
        last_time=START + HOUR * 12,
        formed_at=START + HOUR * 13,
    )

    assert (early.lower, early.upper) == (late.lower, late.upper)
    assert fvg_id(early) != fvg_id(late)


def test_identity_separates_direction_symbol_and_band() -> None:
    base = fvg_id(sample_gap(), symbol="XAUUSD")

    assert fvg_id(sample_gap(direction=GapDirection.BEARISH), symbol="XAUUSD") != base
    assert fvg_id(sample_gap(), symbol="XAGUSD") != base
    assert fvg_id(sample_gap(upper=Decimal("4011")), symbol="XAUUSD") != base
    assert fvg_id(sample_gap(timeframe=Timeframe.M5), symbol="XAUUSD") != base


def test_the_same_band_written_two_ways_is_one_gap() -> None:
    """``4000`` and ``4000.00`` are the same price, so the same identity."""
    assert fvg_id(sample_gap(lower=Decimal("4000"))) == fvg_id(sample_gap(lower=Decimal("4000.00")))


def test_identity_is_the_only_thing_this_module_adds_to_a_gap() -> None:
    """§1. The 6.6a detector remains the sole authority for what a gap *is*."""
    result = analyse(BULL)
    state = formed_on(result, 2)
    (detected,) = [g for g in fair_value_gaps(series(BULL)) if g.formed_at == state.gap.formed_at]

    assert state.gap == detected
    assert state.gap.lower == Decimal("4000")
    assert state.gap.upper == Decimal("4010")
    assert state.gap.size == Decimal("10")
    assert state.gap.midpoint == Decimal("4005")


# --------------------------------------------------------------------------
# §4, §37: the status model
# --------------------------------------------------------------------------


def test_the_status_enum_has_exactly_three_values() -> None:
    assert {member.value for member in FvgStatus} == {"OPEN", "TOUCHED", "FILLED"}


def test_a_new_bullish_gap_starts_open() -> None:
    state = formed_on(analyse(BULL), 2)

    assert state.gap.direction is GapDirection.BULLISH
    assert state.status is FvgStatus.OPEN
    assert state.first_touched_at is None
    assert state.first_touch is None
    assert state.filled_at is None
    assert state.fill is None


def test_a_new_bearish_gap_starts_open() -> None:
    state = formed_on(analyse(BEAR), 2)

    assert state.gap.direction is GapDirection.BEARISH
    assert (state.gap.lower, state.gap.upper) == (Decimal("4000"), Decimal("4010"))
    assert state.status is FvgStatus.OPEN


def test_the_method_version_is_stamped_everywhere() -> None:
    result = analyse(BULL)

    assert result.method_version == FVG_LIFECYCLE_METHOD_VERSION
    assert all(state.method_version == FVG_LIFECYCLE_METHOD_VERSION for state in result.states)


# --------------------------------------------------------------------------
# §38: the gap-over / false-fill matrix, on [4000, 4010]
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "high", "low", "expected"),
    [
        ("A: entirely below, never traded inside", "3995", "3980", FvgStatus.OPEN),
        ("B: intersects the interior from below", "4005", "3995", FvgStatus.TOUCHED),
        ("C: covers, low strictly under the band", "4010", "3995", FvgStatus.FILLED),
        ("D: covers the band exactly", "4010", "4000", FvgStatus.FILLED),
        ("E: wholly inside the band", "4009", "4001", FvgStatus.TOUCHED),
        ("F: exact contact with the upper edge only", "4020", "4010", FvgStatus.OPEN),
        ("G: exact contact with the lower edge only", "4000", "3990", FvgStatus.OPEN),
    ],
)
def test_the_false_fill_matrix(case: str, high: str, low: str, expected: FvgStatus) -> None:
    """The single most important table in this round.

    Case A is why :func:`covers` requires both halves. A candle sitting entirely
    under the gap satisfies ``low <= lower`` while demonstrably never having
    traded inside it - the intrabar path is not in the data, and inventing one
    would retire a zone the market has not been to.
    """
    state = formed_on(analyse(after(bar(high, low))), 2)

    assert state.status is expected, case


def test_the_geometry_predicates_agree_with_the_matrix() -> None:
    """The same table, at the two predicates, so a failure says which rule broke."""
    gap = sample_gap()

    assert not enters(gap, Decimal("3980"), Decimal("3995"))
    assert enters(gap, Decimal("3995"), Decimal("4005"))
    assert enters(gap, Decimal("4001"), Decimal("4009"))
    assert not enters(gap, Decimal("4010"), Decimal("4020")), "upper edge is not the interior"
    assert not enters(gap, Decimal("3990"), Decimal("4000")), "lower edge is not the interior"

    assert not covers(gap, Decimal("3980"), Decimal("3995")), "the false fill"
    assert covers(gap, Decimal("3995"), Decimal("4010"))
    assert covers(gap, Decimal("4000"), Decimal("4010"))
    assert not covers(gap, Decimal("4001"), Decimal("4009"))


def test_a_candle_above_the_band_does_not_fill_it_either() -> None:
    """The mirror of case A: ``high >= upper`` alone is not evidence."""
    gap = sample_gap()

    assert not covers(gap, Decimal("4050"), Decimal("4060"))
    assert not enters(gap, Decimal("4050"), Decimal("4060"))


# --------------------------------------------------------------------------
# §12-§15: transitions
# --------------------------------------------------------------------------


def test_open_goes_straight_to_filled_in_one_candle() -> None:
    """No persisted intermediate TOUCHED is required."""
    result = analyse(after(bar("4012", "3995")))
    state = formed_on(result, 2)

    assert state.status is FvgStatus.FILLED
    assert bar_index(state.filled_at) == 4  # type: ignore[arg-type]


def test_a_direct_fill_also_records_the_first_touch() -> None:
    """§21's chosen policy: "when did price first enter this gap?" always answers."""
    state = formed_on(analyse(after(bar("4012", "3995"))), 2)

    assert state.first_touched_at == state.filled_at
    assert state.first_touch == state.fill


def test_open_then_touched_then_filled() -> None:
    result = analyse(after(bar("4005", "4001"), CLEAR_ABOVE, bar("4012", "3995")))
    state = formed_on(result, 2)

    assert state.status is FvgStatus.FILLED
    assert bar_index(state.first_touched_at) == 4  # type: ignore[arg-type]
    assert bar_index(state.filled_at) == 6  # type: ignore[arg-type]
    assert state.first_touch != state.fill


def test_a_touched_gap_never_returns_to_open() -> None:
    result = analyse(after(bar("4005", "4001"), CLEAR_ABOVE, CLEAR_ABOVE))
    state = formed_on(result, 2)

    assert state.status is FvgStatus.TOUCHED


def test_filled_is_terminal() -> None:
    """Twenty later candles through the same band change nothing."""
    later = [bar("4012", "3995"), *([bar("4005", "4001")] * 4), bar("4012", "3995")]
    result = analyse(after(*later))
    state = formed_on(result, 2)

    assert state.status is FvgStatus.FILLED
    assert bar_index(state.filled_at) == 4  # type: ignore[arg-type]


def test_the_first_touch_is_never_overwritten() -> None:
    result = analyse(after(bar("4005", "4001"), CLEAR_ABOVE, bar("4006", "4002")))
    state = formed_on(result, 2)

    assert state.status is FvgStatus.TOUCHED
    assert bar_index(state.first_touched_at) == 4  # type: ignore[arg-type]
    assert state.first_touch is not None
    assert (state.first_touch.high, state.first_touch.low) == (Decimal("4005"), Decimal("4001"))


def test_no_touch_count_is_kept() -> None:
    """§18. A reaction count is a quality signal; quality belongs to scoring."""
    state = formed_on(analyse(after(bar("4005", "4001"), CLEAR_ABOVE, bar("4006", "4002"))), 2)

    assert not hasattr(state, "touch_count")
    assert not hasattr(state, "touches")


# --------------------------------------------------------------------------
# §16-§17: provenance
# --------------------------------------------------------------------------


def test_the_first_touch_names_the_candle_that_supplied_it() -> None:
    result = analyse(after(bar("4005", "3995")))
    state = formed_on(result, 2)

    witness = state.first_touch
    assert witness is not None
    assert witness.low == Decimal("3995")
    assert witness.high == Decimal("4005")
    assert witness.bar_close_time == state.first_touched_at
    assert witness.bar_close_time == witness.bar_open_time + HOUR


def test_the_fill_names_the_candle_that_covered_the_band() -> None:
    result = analyse(after(bar("4012", "3995")))
    state = formed_on(result, 2)

    witness = state.fill
    assert witness is not None
    assert witness.low <= state.gap.lower
    assert witness.high >= state.gap.upper
    assert witness.bar_close_time == state.filled_at


# --------------------------------------------------------------------------
# §19-§20, §25: temporal rules
# --------------------------------------------------------------------------


def test_a_gap_does_not_exist_before_its_third_candle_closes() -> None:
    """§39 A-B. Two candles of a three-candle pattern are not a fact."""
    assert analyse(BULL[:2]).states == ()

    result = analyse(BULL)
    assert len(result.states) == 1
    assert formed_on(result, 2).status is FvgStatus.OPEN


def test_candle_c_cannot_interact_with_the_gap_it_created() -> None:
    """§19-§20, at the predicate that enforces it.

    Tested here rather than end to end, and that is the honest place for it.
    The only bar closing at ``formed_at`` is C, and C's range can never enter
    its own band: a bullish gap's upper edge *is* ``C.low``, so C has nothing
    strictly inside it, and the bearish case mirrors that. So the rule cannot
    bind under the 6.6a gap definition - it is enforced because that argument
    rests on a definition a later round may change.
    """
    formed = START + HOUR * 3

    assert touchable(FvgStatus.OPEN, formed, formed) is False
    assert touchable(FvgStatus.OPEN, formed, formed + HOUR) is True
    assert touchable(FvgStatus.TOUCHED, formed, formed + HOUR) is True
    assert touchable(FvgStatus.FILLED, formed, formed + HOUR * 50) is False


def test_candle_c_leaves_the_gap_open_end_to_end() -> None:
    result = analyse(BULL)

    assert formed_on(result, 2).status is FvgStatus.OPEN
    assert result.open_ids == (formed_on(result, 2).fvg_id,)


def test_the_next_candle_can_touch_it() -> None:
    """§39 C-D."""
    result = analyse([*BULL, bar("4005", "3995")])

    assert formed_on(result, 2).status is FvgStatus.TOUCHED
    assert bar_index(formed_on(result, 2).first_touched_at) == 3  # type: ignore[arg-type]


def test_price_action_before_formation_never_touches_a_gap() -> None:
    """§25. No retroactive classification, matching the 6.6c.1 pool policy.

    Bar 1 traded from 3998 to 4008, squarely inside the band the next bar went
    on to create. It is not a touch: the gap did not exist, and the candle that
    made it is part of the formation rather than a later interaction.
    """
    state = formed_on(analyse(BULL), 2)

    assert state.gap.lower < Decimal("4008") < state.gap.upper, "bar 1 really was inside"
    assert state.status is FvgStatus.OPEN


# --------------------------------------------------------------------------
# §22-§24: several gaps at once
# --------------------------------------------------------------------------


def test_one_candle_touches_two_gaps() -> None:
    """§40.1. No one-transition-per-bar rule; each gap evolves on its own."""
    result = analyse(after(bar("4009", "4001")))
    touched = [state for state in result.states if state.status is FvgStatus.TOUCHED]

    assert len(touched) == 2
    assert {bar_index(state.gap.formed_at) for state in touched} == {2, 3}
    assert all(bar_index(state.first_touched_at) == 4 for state in touched)  # type: ignore[arg-type]


def test_one_candle_fills_one_gap_while_touching_another() -> None:
    """§40.3."""
    result = analyse(after(bar("4010", "3995")))

    assert formed_on(result, 2).status is FvgStatus.FILLED
    assert formed_on(result, 3).status is FvgStatus.TOUCHED


def test_an_unaffected_gap_is_left_alone() -> None:
    """§40.5."""
    result = analyse(after(bar("4005", "3995")))

    assert formed_on(result, 2).status is FvgStatus.TOUCHED
    assert formed_on(result, 3).status is FvgStatus.OPEN


def test_overlapping_gaps_are_not_merged() -> None:
    """§23. Confluence is a candidate-scoring idea, not a geometric one."""
    result = analyse(after(bar("4009", "4001")))
    first, second = formed_on(result, 2), formed_on(result, 3)

    assert first.gap.upper > second.gap.lower, "the bands really do overlap"
    assert first.fvg_id != second.fvg_id
    assert len({state.fvg_id for state in result.states}) == len(result.states)


def test_a_bullish_and_a_bearish_gap_over_one_band_stay_separate() -> None:
    """§24 arising naturally: bar 4 recreates the same band from the other side."""
    result = analyse(after(bar("4000", "3990")))
    bands = [(state.gap.direction, state.gap.lower, state.gap.upper) for state in result.states]

    assert (GapDirection.BULLISH, Decimal("4000"), Decimal("4010")) in bands
    assert (GapDirection.BEARISH, Decimal("4000"), Decimal("4010")) in bands
    assert len({state.fvg_id for state in result.states}) == len(result.states)


def test_the_status_id_lists_partition_the_states() -> None:
    result = analyse(after(bar("4009", "4001"), CLEAR_ABOVE, bar("4012", "3995")))

    everything = [state.fvg_id for state in result.states]
    assert sorted(everything) == sorted([*result.open_ids, *result.touched_ids, *result.filled_ids])
    assert len(everything) == len(set(everything))


# --------------------------------------------------------------------------
# §27: ordering
# --------------------------------------------------------------------------


def test_states_are_reported_in_formation_order() -> None:
    result = analyse(after(bar("4009", "4001"), CLEAR_ABOVE, bar("4012", "3995")))
    formed = [state.gap.formed_at for state in result.states]

    assert formed == sorted(formed)


# --------------------------------------------------------------------------
# §2, §49: supplying gaps rather than detecting them
# --------------------------------------------------------------------------


def test_supplying_precomputed_gaps_gives_the_same_answer() -> None:
    """§49. A future composite stage detects once and hands the list around."""
    rows = after(bar("4009", "4001"), CLEAR_ABOVE, bar("4012", "3995"))
    snapshot = series(rows)

    detected = analyse_fvg_lifecycle(snapshot, symbol="XAUUSD")
    supplied = analyse_fvg_lifecycle(snapshot, gaps=fair_value_gaps(snapshot), symbol="XAUUSD")

    assert detected == supplied


def test_supplied_gaps_are_filtered_to_what_had_formed() -> None:
    """§29. Handing over a list from a longer series must not leak the future."""
    rows = after(bar("4009", "4001"), CLEAR_ABOVE, bar("4012", "3995"))
    full = series(rows)
    moment = START + HOUR * 4

    leaky = analyse_fvg_lifecycle(full, gaps=fair_value_gaps(full), symbol="XAUUSD", as_of=moment)
    honest = analyse_fvg_lifecycle(full, symbol="XAUUSD", as_of=moment)

    assert leaky == honest
    assert all(state.gap.formed_at <= moment for state in leaky.states)


def test_an_empty_gap_list_is_respected() -> None:
    """A caller saying "no gaps" is not second-guessed by re-detecting."""
    result = analyse_fvg_lifecycle(series(BULL), gaps=[], symbol="XAUUSD")

    assert result.states == ()
    assert result.bars_considered == 3
