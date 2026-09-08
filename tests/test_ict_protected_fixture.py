"""One journey through every anchor case, and five timeframes that differ.

Round 6.6c.3a §46-§47. The matrices in ``test_ict_protected.py`` check one rule
at a time; this file pins a 54-bar path containing every way an anchor can be
established, so a later round changing a definition has to change these numbers
on purpose.

Every number below was read out of the engine before it was written down.

The path extends Round 6.6b's journey rather than replacing it: the same five
structure events in the same order, plus a tall early high that survives the
bullish phase and a newer low that confirms mid-trend. Both additions exist
because the 6.6b journey alone cannot show what this round is about - there,
every reversal happens with the opposite side already fully consumed, so three
of its five events correctly establish nothing at all.
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
from goldpipeline.services.ict_protected import (
    ProtectedStructureAnalysis,
    ProtectedSwingType,
    analyse_protected_structure,
    analyse_snapshot_protected_structure,
)
from goldpipeline.services.ict_structure import (
    BreakClassification,
    BreakDirection,
    analyse_structure,
)
from tests.test_ict_protected import closing_bar, wicked
from tests.test_ict_structure import FLAT, Row, peak, series, trough

JOURNEY: list[Row] = [
    FLAT,  # 0
    FLAT,  # 1
    peak("4200"),  # 2   H0, never broken, anchors both bearish legs
    FLAT,  # 3
    FLAT,  # 4
    peak("4030"),  # 5   H1
    FLAT,  # 6
    FLAT,  # 7
    trough("3970"),  # 8   L1
    FLAT,  # 9
    FLAT,  # 10
    wicked("4035", "4055", "3990"),  # 11  INITIAL_BREAK bullish
    wicked("4035", "4055", "3990"),  # 12
    FLAT,  # 13
    FLAT,  # 14
    peak("4045"),  # 15  H2
    FLAT,  # 16
    FLAT,  # 17
    wicked("4050", "4070", "3990"),  # 18  BOS bullish, reaffirming L1
    wicked("4050", "4070", "3990"),  # 19
    FLAT,  # 20
    FLAT,  # 21
    trough("3980"),  # 22  L2, active long before it is protected
    FLAT,  # 23
    FLAT,  # 24
    FLAT,  # 25
    FLAT,  # 26
    peak("4060"),  # 27  H3
    FLAT,  # 28
    FLAT,  # 29
    wicked("4065", "4085", "3990"),  # 30  BOS bullish, establishing L2
    wicked("4065", "4085", "3990"),  # 31
    FLAT,  # 32
    wicked("3975", "4010", "3960"),  # 33  MSS bearish, anchoring on H0
    wicked("3975", "4010", "3960"),  # 34
    FLAT,  # 35
    FLAT,  # 36
    trough("3950"),  # 37  L3
    FLAT,  # 38
    FLAT,  # 39
    wicked("3945", "4010", "3930"),  # 40  BOS bearish
    wicked("3945", "4010", "3930"),  # 41
    FLAT,  # 42
    FLAT,  # 43
    trough("3960"),  # 44  L4
    FLAT,  # 45
    FLAT,  # 46
    peak("4020"),  # 47  H4
    FLAT,  # 48
    FLAT,  # 49
    wicked("4025", "4045", "3990"),  # 50  MSS bullish, anchoring on L4
    wicked("4025", "4045", "3990"),  # 51
    FLAT,  # 52
    FLAT,  # 53
]


def journey() -> IctTimeframeSnapshot:
    return series(JOURNEY)


def result() -> ProtectedStructureAnalysis:
    return analyse_protected_structure(journey(), symbol="XAUUSD")


# --------------------------------------------------------------------------
# §46: what the journey contains
# --------------------------------------------------------------------------


def test_the_journey_keeps_the_six_six_b_event_sequence() -> None:
    """The structure this round anchors onto, unchanged in shape."""
    structure = analyse_structure(journey(), symbol="XAUUSD")

    assert [(e.classification.value, e.direction.value) for e in structure.breaks] == [
        ("INITIAL_BREAK", "BULLISH"),
        ("BOS", "BULLISH"),
        ("BOS", "BULLISH"),
        ("MSS", "BEARISH"),
        ("BOS", "BEARISH"),
        ("MSS", "BULLISH"),
    ]


def test_the_journey_pins_every_assignment() -> None:
    assert [
        (
            a.protected_type.value,
            str(a.swing_price),
            closing_bar(a.swing_confirmed_at),
            closing_bar(a.established_at),
            a.event_classification.value,
        )
        for a in result().assignments
    ] == [
        ("LOW", "3970", 10, 11, "INITIAL_BREAK"),
        ("LOW", "3970", 10, 18, "BOS"),
        ("LOW", "3980", 24, 30, "BOS"),
        ("HIGH", "4200", 4, 33, "MSS"),
        ("HIGH", "4200", 4, 40, "BOS"),
        ("LOW", "3960", 46, 50, "MSS"),
    ]


def test_the_journey_pins_every_leg() -> None:
    assert [
        (
            leg.direction.value,
            leg.event_classification.value,
            str(leg.origin_price),
            str(leg.terminal_price),
            closing_bar(leg.formed_at),
        )
        for leg in result().legs
    ] == [
        ("BULLISH", "INITIAL_BREAK", "3970", "4055", 11),
        ("BULLISH", "BOS", "3970", "4070", 18),
        ("BULLISH", "BOS", "3980", "4085", 30),
        ("BEARISH", "MSS", "4200", "3960", 33),
        ("BEARISH", "BOS", "4200", "3930", 40),
        ("BULLISH", "MSS", "3960", "4045", 50),
    ]


def test_every_classification_and_direction_is_exercised() -> None:
    legs = result().legs

    assert {leg.event_classification for leg in legs} == set(BreakClassification)
    assert {leg.direction for leg in legs} == set(BreakDirection)
    assert {a.protected_type for a in result().assignments} == set(ProtectedSwingType)


def test_the_first_two_bullish_events_share_one_anchor() -> None:
    """§46, and §10 in a realistic path."""
    first, second = result().assignments[:2]

    assert first.swing_id == second.swing_id
    assert first.assignment_id != second.assignment_id
    assert result().legs[0].leg_id != result().legs[1].leg_id
    assert result().legs[0].origin_price == result().legs[1].origin_price


def test_the_third_bullish_event_establishes_the_newer_low() -> None:
    """L2 confirmed at bar 24 and was active from then; it is protected from bar 30."""
    third = result().assignments[2]

    assert third.swing_price == Decimal("3980")
    assert closing_bar(third.swing_confirmed_at) == 24
    assert closing_bar(third.established_at) == 30
    assert third.swing_confirmed_at < third.established_at


def test_the_bearish_mss_anchors_on_the_opposite_high() -> None:
    """§25, §42.8. Not the low it broke."""
    structure = analyse_structure(journey(), symbol="XAUUSD")
    mss = next(e for e in structure.breaks if e.classification is BreakClassification.MSS)
    assignment = result().assignments[3]

    assert mss.direction is BreakDirection.BEARISH
    assert mss.broken_level == Decimal("3980"), "it broke the low"
    assert assignment.protected_type is ProtectedSwingType.HIGH
    assert assignment.swing_price == Decimal("4200"), "and anchored on the high"
    assert assignment.swing_price != mss.broken_level


def test_the_bullish_mss_anchors_on_the_opposite_low() -> None:
    """§42.9."""
    structure = analyse_structure(journey(), symbol="XAUUSD")
    mss = [e for e in structure.breaks if e.classification is BreakClassification.MSS][-1]
    assignment = result().assignments[-1]

    assert mss.direction is BreakDirection.BULLISH
    assert mss.broken_level == Decimal("4020")
    assert assignment.protected_type is ProtectedSwingType.LOW
    assert assignment.swing_price == Decimal("3960")


def test_every_terminal_differs_materially_from_its_close() -> None:
    """§43.3, across the whole journey rather than one hand-picked leg."""
    structure = analyse_structure(journey(), symbol="XAUUSD")
    closes = {e.event_id: e.break_close for e in structure.breaks}

    for leg in result().legs:
        assert leg.terminal_price != closes[leg.event_id]
        if leg.direction is BreakDirection.BULLISH:
            assert leg.terminal_price > closes[leg.event_id]
        else:
            assert leg.terminal_price < closes[leg.event_id]


def test_every_leg_runs_the_right_way() -> None:
    """§20, as an invariant over the fixture."""
    for leg in result().legs:
        if leg.direction is BreakDirection.BULLISH:
            assert leg.terminal_price > leg.origin_price
        else:
            assert leg.terminal_price < leg.origin_price
        assert leg.span > 0


def test_every_anchor_was_confirmed_before_the_event_that_established_it() -> None:
    """§33, as an invariant over the fixture."""
    for assignment in result().assignments:
        assert assignment.swing_confirmed_at < assignment.established_at


def test_each_leg_names_the_assignment_and_event_that_made_it() -> None:
    analysis = result()
    assignment_ids = {a.assignment_id for a in analysis.assignments}
    structure = analyse_structure(journey(), symbol="XAUUSD")
    event_ids = {e.event_id for e in structure.breaks}

    for leg in analysis.legs:
        assert leg.protected_assignment_id in assignment_ids
        assert leg.event_id in event_ids
    assert len({leg.leg_id for leg in analysis.legs}) == 6
    assert len({a.assignment_id for a in analysis.assignments}) == 6


def test_the_journey_ends_on_the_bullish_mss_anchor() -> None:
    analysis = result()
    current = analysis.current_assignment

    assert current is not None
    assert current is analysis.assignments[-1]
    assert current.protected_type is ProtectedSwingType.LOW
    assert current.swing_price == Decimal("3960")
    assert analysis.current_leg is not None
    assert analysis.current_leg.terminal_price == Decimal("4045")


def test_the_journey_replays_identically() -> None:
    assert result() == result()


# --------------------------------------------------------------------------
# §7 in the wild: the unextended 6.6b journey establishes nothing three times
# --------------------------------------------------------------------------


def test_the_six_six_b_journey_honestly_anchors_only_twice() -> None:
    """Why this round's fixture had to be extended, pinned rather than asserted.

    In Round 6.6b's own journey every reversal happens after the opposite side
    has been entirely consumed, so three of its five events have no eligible
    anchor at all. The engine records that honestly instead of reaching for the
    break bar's own extreme.
    """
    from tests.test_ict_structure_fixture import JOURNEY as SIX_SIX_B

    structure = analyse_structure(series(SIX_SIX_B), symbol="XAUUSD")
    analysis = analyse_protected_structure(series(SIX_SIX_B), symbol="XAUUSD")

    assert len(structure.breaks) == 5
    assert len(analysis.assignments) == 2
    assert len(analysis.legs) == 2
    assert [e.opposite_active_swing_id is None for e in structure.breaks] == [
        False,
        False,
        True,
        True,
        True,
    ]


# --------------------------------------------------------------------------
# §47: five timeframes, five independent answers
# --------------------------------------------------------------------------

OBSERVED_AT = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)

BULL_ANCHORED: list[Row] = [
    FLAT, FLAT, peak("4030"), FLAT, FLAT, trough("3970"), FLAT, FLAT,
    wicked("4035", "4055", "3990"), wicked("4035", "4055", "3990"), FLAT, FLAT,
]  # fmt: skip
BEAR_ANCHORED: list[Row] = [
    FLAT, FLAT, trough("3970"), FLAT, FLAT, peak("4030"), FLAT, FLAT,
    wicked("3965", "4010", "3945"), wicked("3965", "4010", "3945"), FLAT, FLAT,
]  # fmt: skip
QUIET: list[Row] = [FLAT, FLAT, peak("4030"), FLAT, FLAT, FLAT, FLAT, FLAT]

PATHS: dict[Timeframe, list[Row]] = {
    Timeframe.H4: BEAR_ANCHORED,
    Timeframe.H1: JOURNEY,
    Timeframe.M15: QUIET,
    Timeframe.M5: BULL_ANCHORED,
    Timeframe.M1: QUIET,
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
    for index, (high, low, close) in enumerate(rows):
        top, bottom, last = Decimal(high), Decimal(low), Decimal(close)
        opening = last if previous is None else min(max(previous, bottom), top)
        bars.append(
            OHLCBar(
                timestamp=first_open + duration * index,
                open=opening,
                high=top,
                low=bottom,
                close=last,
            )
        )
        previous = last

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
    summary = {
        tf: (
            len(analyse_snapshot_protected_structure(shot, tf).assignments),
            None
            if analyse_snapshot_protected_structure(shot, tf).current_assignment is None
            else analyse_snapshot_protected_structure(
                shot, tf
            ).current_assignment.protected_type.value,  # type: ignore[union-attr]
        )
        for tf in ICT_TIMEFRAMES
    }

    assert summary == {
        Timeframe.H4: (1, "HIGH"),
        Timeframe.H1: (6, "LOW"),
        Timeframe.M15: (0, None),
        Timeframe.M5: (1, "LOW"),
        Timeframe.M1: (0, None),
    }


def test_no_timeframe_overrides_another() -> None:
    """§47. An H4 protected high does not govern an M5 protected low."""
    shot = divergent_snapshot()

    h4 = analyse_snapshot_protected_structure(shot, Timeframe.H4)
    m5 = analyse_snapshot_protected_structure(shot, Timeframe.M5)

    assert {a.timeframe for a in h4.assignments} == {Timeframe.H4}
    assert {a.timeframe for a in m5.assignments} == {Timeframe.M5}
    assert {a.assignment_id for a in h4.assignments}.isdisjoint(
        a.assignment_id for a in m5.assignments
    )
    assert h4.current_assignment is not None
    assert m5.current_assignment is not None
    assert h4.current_assignment.protected_type is not m5.current_assignment.protected_type


def test_analysing_one_timeframe_does_not_touch_another() -> None:
    shot = divergent_snapshot()

    alone = analyse_protected_structure(shot.require(Timeframe.M5), symbol="XAUUSD")
    after_the_others = [analyse_snapshot_protected_structure(shot, tf) for tf in ICT_TIMEFRAMES]

    assert alone == next(r for r in after_the_others if r.timeframe is Timeframe.M5)


def test_the_same_path_on_two_timeframes_finds_the_same_shape() -> None:
    """Independence is not indifference: bar duration must not change geometry."""
    shot = divergent_snapshot()

    m15 = analyse_snapshot_protected_structure(shot, Timeframe.M15)
    m1 = analyse_snapshot_protected_structure(shot, Timeframe.M1)

    assert len(m15.assignments) == len(m1.assignments)
    assert len(m15.legs) == len(m1.legs)


def test_provider_native_h4_timestamps_are_analysed_where_they_sit() -> None:
    """No resampling, no 00/04/08 correction, no DST assumption."""
    native_end = datetime(2026, 9, 7, 13, 0, tzinfo=UTC)
    shifted_end = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)

    native_series = ending_at(Timeframe.H4, BEAR_ANCHORED, end=native_end)
    assert all(candle.timestamp.hour % 4 == 1 for candle in native_series.bars)

    native = analyse_protected_structure(native_series, symbol="XAUUSD")
    shifted = analyse_protected_structure(
        ending_at(Timeframe.H4, BEAR_ANCHORED, end=shifted_end), symbol="XAUUSD"
    )

    assert [(str(leg.origin_price), str(leg.terminal_price)) for leg in native.legs] == [
        (str(leg.origin_price), str(leg.terminal_price)) for leg in shifted.legs
    ]
