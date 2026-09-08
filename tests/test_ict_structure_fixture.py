"""One long synthetic journey, and five timeframes that disagree.

Round 6.6b §41-§42. The matrices in ``test_ict_structure.py`` check one rule at
a time on a handful of bars. This file does the other job twice over:

* a single 48-bar series that passes through every state and every event kind
  the engine can produce, pinned exactly, so a later round that changes a
  definition has to change these numbers on purpose;
* five timeframes carrying genuinely *different* price paths, which is what the
  6.6a fixture could not show. That one put the same shape on all five - useful
  for proving the primitives are duration-blind, useless for proving the engine
  does not let one timeframe's answer contaminate another's.

Prices are gold-shaped and mean nothing. Nothing downstream may read them as a
market assumption.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import (
    ICT_TIMEFRAMES,
    IctMarketSnapshot,
    IctTimeframeSnapshot,
    build_timeframe_snapshot,
)
from goldpipeline.schemas.market import OHLCBar
from goldpipeline.services.ict_structure import (
    BreakClassification,
    BreakDirection,
    StructureBias,
    analyse_snapshot_structure,
    analyse_structure,
)
from tests.test_ict_structure import (
    FLAT,
    START,
    Row,
    bar_index,
    drive,
    peak,
    series,
    touch,
    trough,
    wick_up,
)

# --------------------------------------------------------------------------
# §41: the journey
# --------------------------------------------------------------------------

JOURNEY: list[Row] = [
    FLAT,                # 0
    FLAT,                # 1
    peak("4030"),        # 2   swing high A
    FLAT,                # 3
    FLAT,                # 4
    trough("3970"),      # 5   swing low B - survives the whole bullish phase
    FLAT,                # 6
    FLAT,                # 7
    touch("4030"),       # 8   closes exactly on A: no break, and a second 4030 pivot
    FLAT,                # 9
    FLAT,                # 10
    wick_up("4045"),     # 11  trades through 4030 and closes back inside: no break
    FLAT,                # 12
    FLAT,                # 13
    drive("4050"),       # 14  INITIAL_BREAK bullish, sweeping both 4030s
    drive("4050"),       # 15
    FLAT,                # 16
    FLAT,                # 17
    peak("4060"),        # 18
    FLAT,                # 19
    FLAT,                # 20
    drive("4065"),       # 21  BOS bullish
    drive("4065"),       # 22
    FLAT,                # 23
    drive("3965"),       # 24  MSS bearish, through B
    drive("3965"),       # 25
    FLAT,                # 26
    FLAT,                # 27
    trough("3950"),      # 28
    FLAT,                # 29
    FLAT,                # 30
    drive("3945"),       # 31  BOS bearish
    drive("3945"),       # 32
    FLAT,                # 33
    FLAT,                # 34
    peak("4020"),        # 35
    FLAT,                # 36
    FLAT,                # 37
    drive("4025"),       # 38  MSS bullish
    drive("4025"),       # 39
    FLAT,                # 40
    FLAT,                # 41
    trough("3975"),      # 42  left unbroken, so the analysis ends with levels in play
    FLAT,                # 43
    FLAT,                # 44
    peak("4035"),        # 45  likewise
    FLAT,                # 46
    FLAT,                # 47
]  # fmt: skip


def journey() -> IctTimeframeSnapshot:
    return series(JOURNEY)


def test_the_journey_visits_every_state_and_every_event_kind() -> None:
    result = analyse_structure(journey(), symbol="XAUUSD")

    assert [(event.classification.value, event.direction.value) for event in result.breaks] == [
        ("INITIAL_BREAK", "BULLISH"),
        ("BOS", "BULLISH"),
        ("MSS", "BEARISH"),
        ("BOS", "BEARISH"),
        ("MSS", "BULLISH"),
    ]
    assert set(BreakClassification) == {event.classification for event in result.breaks}, (
        "every classification the engine can emit is exercised here"
    )
    assert set(BreakDirection) == {event.direction for event in result.breaks}


def test_the_journey_starts_neutral_and_ends_bullish() -> None:
    result = analyse_structure(journey())

    assert result.initial_bias is StructureBias.NEUTRAL
    assert result.current_bias is StructureBias.BULLISH


def test_the_state_moves_only_where_a_close_moved_it() -> None:
    result = analyse_structure(journey())

    assert [(event.prior_bias.value, event.resulting_bias.value) for event in result.breaks] == [
        ("NEUTRAL", "BULLISH"),
        ("BULLISH", "BULLISH"),
        ("BULLISH", "BEARISH"),
        ("BEARISH", "BEARISH"),
        ("BEARISH", "BULLISH"),
    ]


def test_the_journey_pins_which_bar_broke_which_level() -> None:
    result = analyse_structure(journey())

    assert [
        (bar_index(event.break_bar_open_time), str(event.broken_level), str(event.break_close))
        for event in result.breaks
    ] == [
        (14, "4045", "4050"),
        (21, "4060", "4065"),
        (24, "3970", "3965"),
        (31, "3950", "3945"),
        (38, "4020", "4025"),
    ]


def test_the_journey_pins_every_swing_it_found() -> None:
    result = analyse_structure(journey())

    assert [
        (bar_index(entry.swing.pivot_time), entry.label, str(entry.swing.price))
        for entry in result.annotated_swings
    ] == [
        (2, "FIRST_HIGH", "4030"),
        (5, "FIRST_LOW", "3970"),
        (8, "EQUAL_HIGH", "4030"),
        (11, "HIGHER_HIGH", "4045"),
        (18, "HIGHER_HIGH", "4060"),
        (28, "LOWER_LOW", "3950"),
        (35, "LOWER_HIGH", "4020"),
        (42, "HIGHER_LOW", "3975"),
        (45, "HIGHER_HIGH", "4035"),
    ]


def test_the_exact_touch_at_bar_eight_broke_nothing_and_became_its_own_swing() -> None:
    result = analyse_structure(journey())

    assert not any(bar_index(event.break_bar_open_time) == 8 for event in result.breaks)
    equal = next(
        entry for entry in result.annotated_swings if bar_index(entry.swing.pivot_time) == 8
    )
    assert equal.label == "EQUAL_HIGH"
    assert equal.swing.price == Decimal("4030")


def test_the_wick_at_bar_eleven_broke_nothing() -> None:
    result = analyse_structure(journey())

    assert not any(bar_index(event.break_bar_open_time) == 11 for event in result.breaks)


def test_one_close_swept_both_swings_that_shared_a_price() -> None:
    """The 4030 at bar 2 and the 4030 at bar 8 are two levels, cleared together.

    Both are named in the event that cleared them rather than only in the
    analysis-wide set, so each consumption is attributable to the close that
    caused it.
    """
    result = analyse_structure(journey())
    initial = result.breaks[0]

    assert initial.broken_level == Decimal("4045")
    assert len(initial.also_consumed_swing_ids) == 2
    assert all("SWING_HIGH" in identity for identity in initial.also_consumed_swing_ids)
    assert len({*initial.also_consumed_swing_ids}) == 2, "two identities, one price"


def test_the_low_from_bar_five_survived_the_entire_bullish_phase() -> None:
    """§20 - both sides tracked, or a reversal can never be seen."""
    result = analyse_structure(journey(), as_of=START + timedelta(hours=23))

    assert result.current_bias is StructureBias.BULLISH
    assert result.active_low is not None
    assert result.active_low.swing.price == Decimal("3970")
    assert bar_index(result.active_low.swing.pivot_time) == 5


def test_the_journey_ends_with_a_level_in_play_on_each_side() -> None:
    result = analyse_structure(journey())

    assert result.active_high is not None
    assert result.active_low is not None
    assert result.active_high.swing.price == Decimal("4035")
    assert result.active_low.swing.price == Decimal("3975")
    assert len(result.consumed_swing_ids) == 7


def test_every_consumed_level_is_attributable_to_exactly_one_event() -> None:
    result = analyse_structure(journey())

    attributed: list[str] = []
    for event in result.breaks:
        attributed.append(event.broken_swing_id)
        attributed.extend(event.also_consumed_swing_ids)

    assert len(attributed) == len(set(attributed)), "no level consumed twice"
    assert set(attributed) == result.consumed_swing_ids


def test_no_swing_generated_two_break_events() -> None:
    result = analyse_structure(journey())
    broken = [event.broken_swing_id for event in result.breaks]

    assert len(broken) == len(set(broken))


def test_the_journey_replays_identically() -> None:
    assert analyse_structure(journey(), symbol="XAUUSD") == analyse_structure(
        journey(), symbol="XAUUSD"
    )


# --------------------------------------------------------------------------
# §42: five timeframes, five independent answers
# --------------------------------------------------------------------------

OBSERVED_AT = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)

BULLISH_ONLY: list[Row] = [
    FLAT, FLAT, peak("4030"), FLAT, FLAT, drive("4035"), drive("4035"), FLAT, FLAT,
]  # fmt: skip
BEARISH_ONLY: list[Row] = [
    FLAT, FLAT, trough("3970"), FLAT, FLAT, drive("3965"), drive("3965"), FLAT, FLAT,
]  # fmt: skip
QUIET: list[Row] = [
    FLAT, FLAT, peak("4030"), FLAT, FLAT, wick_up("4040"), FLAT, FLAT,
]  # fmt: skip

PATHS: dict[Timeframe, list[Row]] = {
    Timeframe.H4: BEARISH_ONLY,
    Timeframe.H1: JOURNEY,
    Timeframe.M15: QUIET,
    Timeframe.M5: BULLISH_ONLY,
    Timeframe.M1: QUIET,
}


def ending_at(
    timeframe: Timeframe, rows: Sequence[Row], *, end: datetime = OBSERVED_AT
) -> IctTimeframeSnapshot:
    """Lay *rows* onto *timeframe* so the last bar closes exactly at *end*.

    Every series in a snapshot has to be current to one instant; each starts
    wherever its own duration puts it.
    """
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
    """The same code on five different series. Nothing is shared but the algorithm."""
    shot = divergent_snapshot()

    biases = {tf: analyse_snapshot_structure(shot, tf).current_bias for tf in ICT_TIMEFRAMES}

    assert biases == {
        Timeframe.H4: StructureBias.BEARISH,
        Timeframe.H1: StructureBias.BULLISH,
        Timeframe.M15: StructureBias.NEUTRAL,
        Timeframe.M5: StructureBias.BULLISH,
        Timeframe.M1: StructureBias.NEUTRAL,
    }


def test_a_disagreement_between_timeframes_is_not_an_error() -> None:
    """H4 bearish while H1 is bullish is a completely ordinary market.

    Whatever eventually reconciles them is a synthesis rule with real content -
    which timeframe wins, whether the higher one gates the lower at all - and it
    is deliberately not in this round. Nothing here should paper over the
    disagreement in the meantime.
    """
    shot = divergent_snapshot()

    h4 = analyse_snapshot_structure(shot, Timeframe.H4)
    h1 = analyse_snapshot_structure(shot, Timeframe.H1)

    assert h4.current_bias is not h1.current_bias
    assert h4.timeframe is Timeframe.H4
    assert h1.timeframe is Timeframe.H1


def test_analysing_one_timeframe_does_not_touch_another() -> None:
    shot = divergent_snapshot()

    alone = analyse_structure(shot.require(Timeframe.M5), symbol="XAUUSD")
    after_the_others = [analyse_snapshot_structure(shot, tf) for tf in ICT_TIMEFRAMES]

    assert alone == next(r for r in after_the_others if r.timeframe is Timeframe.M5)


def test_each_timeframe_stamps_its_own_events_with_its_own_name() -> None:
    shot = divergent_snapshot()

    h4 = analyse_snapshot_structure(shot, Timeframe.H4)
    m5 = analyse_snapshot_structure(shot, Timeframe.M5)

    assert all(event.timeframe is Timeframe.H4 for event in h4.breaks)
    assert all(event.timeframe is Timeframe.M5 for event in m5.breaks)
    assert all(identity.startswith("H4:") for identity in h4.consumed_swing_ids)
    assert all(identity.startswith("M5:") for identity in m5.consumed_swing_ids)


def test_the_same_path_on_two_timeframes_gives_the_same_shape() -> None:
    """Independence is not indifference: duration must not change the geometry.

    M15 and M1 carry the identical path here, so their answers must agree on
    everything except the timestamps and names that identify the series.
    """
    shot = divergent_snapshot()

    m15 = analyse_snapshot_structure(shot, Timeframe.M15)
    m1 = analyse_snapshot_structure(shot, Timeframe.M1)

    assert m15.current_bias is m1.current_bias
    assert [entry.label for entry in m15.annotated_swings] == [
        entry.label for entry in m1.annotated_swings
    ]
    assert [entry.swing.price for entry in m15.annotated_swings] == [
        entry.swing.price for entry in m1.annotated_swings
    ]
    assert len(m15.breaks) == len(m1.breaks)


def test_provider_native_h4_timestamps_are_analysed_where_they_sit() -> None:
    """§44. No resampling, no 00/04/08 correction, no DST assumption.

    TradingView's XAUUSD H4 opens at 01/05/09/13/17/21 UTC. The engine reads
    close times as ``open + duration`` and never asks what hour of the day that
    lands on, so a venue's own grid needs no correcting - and correcting it
    would invent bar boundaries that never traded.
    """
    native_end = datetime(2026, 9, 7, 13, 0, tzinfo=UTC)
    shifted_end = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)

    native = analyse_structure(ending_at(Timeframe.H4, BEARISH_ONLY, end=native_end))
    shifted = analyse_structure(ending_at(Timeframe.H4, BEARISH_ONLY, end=shifted_end))

    assert all(
        bar.timestamp.hour % 4 == 1
        for bar in ending_at(Timeframe.H4, BEARISH_ONLY, end=native_end).bars
    )
    assert native.current_bias is shifted.current_bias
    assert [event.broken_level for event in native.breaks] == [
        event.broken_level for event in shifted.breaks
    ]
