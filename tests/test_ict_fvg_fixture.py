"""One synthetic path through every lifecycle state, and five timeframes that differ.

Round 6.6c.2 §42-§44. The matrices in ``test_ict_fvg.py`` check one rule at a
time; this file pins a 26-bar series containing every shape the engine can
produce, so a later round changing a definition has to change these numbers on
purpose.

Every number below was read out of the engine before it was written down.

Prices are gold-shaped and mean nothing. Nothing downstream may treat them as a
market assumption.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import (
    ICT_TIMEFRAMES,
    IctMarketSnapshot,
    IctTimeframeSnapshot,
    build_timeframe_snapshot,
)
from goldpipeline.schemas.market import OHLCBar
from goldpipeline.services.ict_fvg import (
    FvgStatus,
    analyse_fvg_lifecycle,
    analyse_snapshot_fvg_lifecycle,
)
from goldpipeline.services.ict_primitives import GapDirection
from tests.test_ict_fvg import Row, bar, bar_index, series

HOLD_HIGH: Row = bar("4025", "4012")
HOLD_TOP: Row = bar("4048", "4037")

JOURNEY: list[Row] = [
    bar("4000", "3990"),  # 0   A of the first gap
    bar("4008", "3998"),  # 1   B
    bar("4020", "4010"),  # 2   C -> bullish [4000, 4010]
    HOLD_HIGH,  # 3            and, with bars 1-2, bullish [4008, 4012]
    HOLD_HIGH,  # 4
    HOLD_HIGH,  # 5
    HOLD_HIGH,  # 6
    bar("4033", "4028"),  # 7   -> bullish [4025, 4028]
    bar("4045", "4035"),  # 8   -> bullish [4025, 4035], overlapping it
    HOLD_TOP,  # 9              -> bullish [4033, 4037], overlapping both
    HOLD_TOP,  # 10
    bar("4040", "4030"),  # 11  reaches into two bands at once
    HOLD_TOP,  # 12
    HOLD_TOP,  # 13
    HOLD_TOP,  # 14
    bar("4035", "4022"),  # 15  -> bearish [4035, 4037], never revisited
    bar("4020", "4012"),  # 16  -> bearish [4020, 4037]
    bar("4012", "3995"),  # 17  covers the first two gaps outright
    HOLD_HIGH,  # 18
    bar("4030", "4025"),  # 19
    bar("4060", "4050"),  # 20  entirely above [4035, 4037]: not a fill
    bar("4062", "4052"),  # 21
    bar("4062", "4052"),  # 22
    bar("4058", "4048"),  # 23
    bar("4045", "4038"),  # 24  -> bearish [4045, 4052], late enough for as-of work
    bar("4046", "4040"),  # 25  -> bearish [4046, 4048], left open at the end
]


def journey() -> IctTimeframeSnapshot:
    return series(JOURNEY)


def summary() -> list[tuple[str, str, str, int, str, int | None, int | None]]:
    result = analyse_fvg_lifecycle(journey(), symbol="XAUUSD")
    return [
        (
            state.gap.direction.value,
            str(state.gap.lower),
            str(state.gap.upper),
            bar_index(state.gap.formed_at),
            state.status.value,
            None if state.first_touched_at is None else bar_index(state.first_touched_at),
            None if state.filled_at is None else bar_index(state.filled_at),
        )
        for state in result.states
    ]


# --------------------------------------------------------------------------
# §42: what the journey contains
# --------------------------------------------------------------------------


def test_the_journey_pins_every_gap_and_its_fate() -> None:
    assert summary() == [
        ("BULLISH", "4000", "4010", 2, "FILLED", 17, 17),
        ("BULLISH", "4008", "4012", 3, "FILLED", 17, 17),
        ("BULLISH", "4025", "4028", 7, "FILLED", 15, 15),
        ("BULLISH", "4025", "4035", 8, "FILLED", 11, 15),
        ("BULLISH", "4033", "4037", 9, "FILLED", 11, 11),
        ("BEARISH", "4035", "4037", 15, "OPEN", None, None),
        ("BEARISH", "4020", "4037", 16, "TOUCHED", 18, None),
        ("BEARISH", "4012", "4022", 17, "FILLED", 18, 18),
        ("BULLISH", "4012", "4025", 19, "OPEN", None, None),
        ("BULLISH", "4025", "4050", 20, "TOUCHED", 23, None),
        ("BULLISH", "4030", "4052", 21, "TOUCHED", 23, None),
        ("BEARISH", "4045", "4052", 24, "TOUCHED", 25, None),
        ("BEARISH", "4046", "4048", 25, "OPEN", None, None),
    ]


def test_the_journey_reaches_every_status_in_both_directions() -> None:
    result = analyse_fvg_lifecycle(journey(), symbol="XAUUSD")

    assert {state.status for state in result.states} == set(FvgStatus)
    assert {state.gap.direction for state in result.states} == set(GapDirection)
    assert len(result.open_ids) == 3
    assert len(result.touched_ids) == 4
    assert len(result.filled_ids) == 6


def test_the_journey_contains_a_direct_open_to_filled() -> None:
    """Bar 17 covered the first band outright, with no earlier interaction."""
    result = analyse_fvg_lifecycle(journey(), symbol="XAUUSD")
    (state,) = [s for s in result.states if bar_index(s.gap.formed_at) == 2]

    assert state.status is FvgStatus.FILLED
    assert state.first_touched_at == state.filled_at
    assert state.first_touch == state.fill


def test_the_journey_contains_an_open_touched_filled_arc() -> None:
    result = analyse_fvg_lifecycle(journey(), symbol="XAUUSD")
    (state,) = [s for s in result.states if bar_index(s.gap.formed_at) == 8]

    assert state.status is FvgStatus.FILLED
    assert bar_index(state.first_touched_at) == 11  # type: ignore[arg-type]
    assert bar_index(state.filled_at) == 15  # type: ignore[arg-type]
    assert state.first_touch != state.fill


def test_the_journey_contains_a_gap_over_that_did_not_fill() -> None:
    """§38 A, in a realistic path.

    The bearish band ``[4035, 4037]`` is left ``OPEN``. Bar 17 passed far below
    it and bar 20 far above; neither traded inside, so neither counts - even
    though bar 20's high is well past the upper edge and bar 17's low well
    under the lower one.
    """
    result = analyse_fvg_lifecycle(journey(), symbol="XAUUSD")
    (state,) = [s for s in result.states if bar_index(s.gap.formed_at) == 15]

    assert state.status is FvgStatus.OPEN
    assert state.gap.lower == Decimal("4035")
    assert state.gap.upper == Decimal("4037")

    above = journey().bars[20]
    below = journey().bars[17]
    assert above.high > state.gap.upper and above.low > state.gap.upper
    assert below.low < state.gap.lower and below.high < state.gap.lower


def test_the_journey_has_one_bar_acting_on_several_gaps() -> None:
    """§40. Each gap moves on its own; there is no one-transition-per-bar rule."""
    result = analyse_fvg_lifecycle(journey(), symbol="XAUUSD")

    at_seventeen = [
        state
        for state in result.states
        if state.filled_at is not None and bar_index(state.filled_at) == 17
    ]
    assert len(at_seventeen) == 2

    at_fifteen = [
        state
        for state in result.states
        if state.filled_at is not None and bar_index(state.filled_at) == 15
    ]
    assert len(at_fifteen) == 2

    at_eleven = [
        state
        for state in result.states
        if state.first_touched_at is not None and bar_index(state.first_touched_at) == 11
    ]
    assert len(at_eleven) == 2, "one candle touched one band and filled another"


def test_the_journey_contains_overlapping_gaps_that_stay_separate() -> None:
    """§23. Three bullish bands overlap and each keeps its own life."""
    result = analyse_fvg_lifecycle(journey(), symbol="XAUUSD")
    trio = [s for s in result.states if bar_index(s.gap.formed_at) in {7, 8, 9}]

    assert len(trio) == 3
    assert trio[0].gap.upper > trio[1].gap.lower, "the bands really do overlap"
    assert trio[1].gap.upper > trio[2].gap.lower
    assert len({s.fvg_id for s in trio}) == 3
    assert [bar_index(s.filled_at) for s in trio] == [15, 15, 11]  # type: ignore[arg-type]


def test_every_identity_in_the_journey_is_distinct() -> None:
    result = analyse_fvg_lifecycle(journey(), symbol="XAUUSD")
    identities = [state.fvg_id for state in result.states]

    assert len(identities) == len(set(identities)) == 13


def test_the_journey_is_reported_in_formation_order() -> None:
    result = analyse_fvg_lifecycle(journey(), symbol="XAUUSD")
    formed = [state.gap.formed_at for state in result.states]

    assert formed == sorted(formed)


def test_the_journey_replays_identically() -> None:
    assert analyse_fvg_lifecycle(journey(), symbol="XAUUSD") == analyse_fvg_lifecycle(
        journey(), symbol="XAUUSD"
    )


def test_lifecycle_never_alters_the_gap_it_describes() -> None:
    """§37.16. The 6.6a band is reported back exactly as detected."""
    result = analyse_fvg_lifecycle(journey(), symbol="XAUUSD")

    for state in result.states:
        assert state.gap.size == state.gap.upper - state.gap.lower
        assert state.gap.midpoint == (state.gap.lower + state.gap.upper) / Decimal(2)
        assert isinstance(state.gap.lower, Decimal)
        assert isinstance(state.gap.upper, Decimal)


# --------------------------------------------------------------------------
# §43: five timeframes, five independent answers
# --------------------------------------------------------------------------

OBSERVED_AT = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)

NO_GAPS: list[Row] = [bar("4020", "4000")] * 6
UNTOUCHED: list[Row] = [
    bar("4000", "3990"),
    bar("4008", "3998"),
    bar("4020", "4010"),
    bar("4030", "4020"),
]
FILLED_ONCE: list[Row] = [
    # The last bar covers the first band outright and reaches into the second
    # without covering it, so this path ends one FILLED and one TOUCHED.
    bar("4000", "3990"),
    bar("4008", "3998"),
    bar("4020", "4010"),
    bar("4030", "4020"),
    bar("4012", "3995"),
]

PATHS: dict[Timeframe, list[Row]] = {
    Timeframe.H4: FILLED_ONCE,
    Timeframe.H1: JOURNEY,
    Timeframe.M15: NO_GAPS,
    Timeframe.M5: UNTOUCHED,
    Timeframe.M1: NO_GAPS,
}


def ending_at(
    timeframe: Timeframe, rows: Sequence[Row], *, end: datetime = OBSERVED_AT
) -> IctTimeframeSnapshot:
    """Lay *rows* onto *timeframe* so the last bar closes exactly at *end*."""
    duration = timeframe.duration
    assert duration is not None

    first_open = end - duration * len(rows)
    bars: list[OHLCBar] = []
    previous: Decimal | None = None
    for index, (high, low) in enumerate(rows):
        top, bottom = Decimal(high), Decimal(low)
        close = (top + bottom) / Decimal(2)
        opening = close if previous is None else min(max(previous, bottom), top)
        bars.append(
            OHLCBar(
                timestamp=first_open + duration * index,
                open=opening,
                high=top,
                low=bottom,
                close=close,
            )
        )
        previous = close

    return build_timeframe_snapshot(timeframe=timeframe, bars=tuple(bars), observed_at=end)


def divergent_snapshot() -> IctMarketSnapshot:
    return IctMarketSnapshot(
        observed_at=OBSERVED_AT,
        symbol="XAUUSD",
        provider="fixture",
        provider_symbol=None,
        timeframes=tuple(ending_at(tf, PATHS[tf]) for tf in ICT_TIMEFRAMES),
    )


def test_five_timeframes_reach_five_answers_of_their_own() -> None:
    shot = divergent_snapshot()
    counts = {
        tf: (
            len(analyse_snapshot_fvg_lifecycle(shot, tf).states),
            len(analyse_snapshot_fvg_lifecycle(shot, tf).filled_ids),
        )
        for tf in ICT_TIMEFRAMES
    }

    assert counts == {
        Timeframe.H4: (2, 1),
        Timeframe.H1: (13, 6),
        Timeframe.M15: (0, 0),
        Timeframe.M5: (2, 0),
        Timeframe.M1: (0, 0),
    }


def test_nothing_merges_gaps_across_timeframes() -> None:
    """§43. Confluence is a scoring rule with content; it is not assumed here."""
    shot = divergent_snapshot()

    h4 = analyse_snapshot_fvg_lifecycle(shot, Timeframe.H4)
    m5 = analyse_snapshot_fvg_lifecycle(shot, Timeframe.M5)

    assert {state.gap.timeframe for state in h4.states} == {Timeframe.H4}
    assert {state.gap.timeframe for state in m5.states} == {Timeframe.M5}
    assert {s.fvg_id for s in h4.states}.isdisjoint(s.fvg_id for s in m5.states)


def test_analysing_one_timeframe_does_not_touch_another() -> None:
    shot = divergent_snapshot()

    alone = analyse_fvg_lifecycle(shot.require(Timeframe.M5), symbol="XAUUSD")
    after_the_others = [analyse_snapshot_fvg_lifecycle(shot, tf) for tf in ICT_TIMEFRAMES]

    assert alone == next(r for r in after_the_others if r.timeframe is Timeframe.M5)


def test_the_same_path_on_two_timeframes_finds_the_same_shape() -> None:
    """Independence is not indifference: bar duration must not change geometry."""
    shot = divergent_snapshot()

    m15 = analyse_snapshot_fvg_lifecycle(shot, Timeframe.M15)
    m1 = analyse_snapshot_fvg_lifecycle(shot, Timeframe.M1)

    assert len(m15.states) == len(m1.states)
    assert [s.status for s in m15.states] == [s.status for s in m1.states]


# --------------------------------------------------------------------------
# §44: H4 policy
# --------------------------------------------------------------------------


def test_provider_native_h4_timestamps_are_analysed_where_they_sit() -> None:
    """No resampling, no 00/04/08 correction, no DST assumption.

    TradingView's XAUUSD H4 opens at 01/05/09/13/17/21 UTC. Close times here are
    ``open + duration`` and nothing asks what hour of the day that lands on, so
    a venue's own grid needs no correcting - and correcting it would invent bar
    boundaries the venue never traded.
    """
    native_end = datetime(2026, 9, 7, 13, 0, tzinfo=UTC)
    shifted_end = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)

    native_series = ending_at(Timeframe.H4, FILLED_ONCE, end=native_end)
    assert all(candle.timestamp.hour % 4 == 1 for candle in native_series.bars)

    native = analyse_fvg_lifecycle(native_series, symbol="XAUUSD")
    shifted = analyse_fvg_lifecycle(
        ending_at(Timeframe.H4, FILLED_ONCE, end=shifted_end), symbol="XAUUSD"
    )

    assert [(s.gap.lower, s.gap.upper, s.status) for s in native.states] == [
        (s.gap.lower, s.gap.upper, s.status) for s in shifted.states
    ]
