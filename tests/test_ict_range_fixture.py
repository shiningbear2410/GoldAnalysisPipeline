"""One journey through every range state, and five timeframes that differ.

Round 6.6c.3b §49-§52. The matrices in ``test_ict_range.py`` check one rule at a
time; this file pins the whole sequence, so a later round changing a definition
has to change these numbers on purpose.

Every number below was read out of the engine before it was written down.

The path extends Round 6.6c.3a's 54-bar fixture by exactly two bars, in place:
bars 23 and 24 become a pair of candles wicking to 4075. Adjacent equal highs
make no strict pivot, so the six structure events stay exactly where they were -
the point is to give a range something to extend on that is *not* a structure
event, which the 6.6c.3a path never does.
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
from goldpipeline.services.ict_range import (
    DealingRangeAnalysis,
    PriceLocation,
    RangeStatus,
    analyse_dealing_ranges,
    analyse_snapshot_dealing_ranges,
)
from goldpipeline.services.ict_structure import BreakDirection, analyse_structure
from tests.test_ict_protected import closing_bar, wicked
from tests.test_ict_protected_fixture import JOURNEY as ANCHORED
from tests.test_ict_range import BEAR, BULL
from tests.test_ict_structure import FLAT, START, Row, peak, series

HOUR = timedelta(hours=1)

JOURNEY: list[Row] = list(ANCHORED)
JOURNEY[23] = wicked("4000", "4075", "3990")
JOURNEY[24] = wicked("4000", "4075", "3990")
"""Two candles reaching 4075 where the 6.6c.3a path had filler.

Written as a pair because two adjacent bars with the same high cannot make a
strict pivot, so no swing appears and no structure event moves. What changes is
that the second range now extends on an ordinary bar rather than only on the
event bars that supersede it.
"""


def journey() -> IctTimeframeSnapshot:
    return series(JOURNEY)


def result(**kwargs: object) -> DealingRangeAnalysis:
    return analyse_dealing_ranges(journey(), symbol="XAUUSD", **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# the structure underneath is untouched
# --------------------------------------------------------------------------


def test_the_two_extra_wicks_moved_no_structure_event() -> None:
    """The fixture extends the range story without rewriting the anchor story."""
    original = analyse_structure(series(ANCHORED), symbol="XAUUSD")
    extended = analyse_structure(journey(), symbol="XAUUSD")

    assert [
        (closing_bar(e.break_bar_close_time), e.classification.value, e.direction.value)
        for e in extended.breaks
    ] == [
        (closing_bar(e.break_bar_close_time), e.classification.value, e.direction.value)
        for e in original.breaks
    ]
    assert [e.event_id for e in extended.breaks] == [e.event_id for e in original.breaks]


# --------------------------------------------------------------------------
# §50: the whole range journey, pinned
# --------------------------------------------------------------------------


def test_the_journey_pins_every_range() -> None:
    assert [
        (
            closing_bar(entry.formed_at),
            entry.direction.value,
            str(entry.origin_price),
            str(entry.initial_terminal_price),
            str(entry.terminal_price),
            entry.status.value,
            None if entry.superseded_at is None else closing_bar(entry.superseded_at),
            entry.superseded_by_range_id is not None,
        )
        for entry in result().ranges
    ] == [
        (11, "BULLISH", "3970", "4055", "4070", "SUPERSEDED", 18, True),
        (18, "BULLISH", "3970", "4070", "4085", "SUPERSEDED", 30, True),
        (30, "BULLISH", "3980", "4085", "4085", "SUPERSEDED", 33, True),
        (33, "BEARISH", "4200", "3960", "3930", "SUPERSEDED", 40, True),
        (40, "BEARISH", "4200", "3930", "3930", "SUPERSEDED", 50, True),
        (50, "BULLISH", "3960", "4045", "4045", "ACTIVE", None, False),
    ]


def test_the_journey_pins_every_equilibrium_and_width() -> None:
    assert [
        (str(entry.lower), str(entry.upper), str(entry.equilibrium), str(entry.width))
        for entry in result().ranges
    ] == [
        ("3970", "4070", "4020", "100"),
        ("3970", "4085", "4027.5", "115"),
        ("3980", "4085", "4032.5", "105"),
        ("3930", "4200", "4065", "270"),
        ("3930", "4200", "4065", "270"),
        ("3960", "4045", "4002.5", "85"),
    ]


def test_each_supersession_names_the_event_and_the_replacement() -> None:
    structure = analyse_structure(journey(), symbol="XAUUSD")
    events = {closing_bar(e.break_bar_close_time): e.event_id for e in structure.breaks}
    entries = result().ranges

    for entry, at in zip(entries[:-1], (18, 30, 33, 40, 50), strict=True):
        assert entry.superseded_by_event_id == events[at]
    assert entries[-1].superseded_by_event_id is None

    # Each replacement id is the next range in the sequence.
    for earlier, later in zip(entries, entries[1:], strict=False):
        assert earlier.superseded_by_range_id == later.range_id


def test_the_journey_ends_on_one_active_bullish_range() -> None:
    analysis = result()
    current = analysis.active_range

    assert current is not None
    assert current is analysis.ranges[-1]
    assert current.direction is BreakDirection.BULLISH
    assert analysis.structure_bias.value == "BULLISH"
    assert sum(1 for entry in analysis.ranges if entry.is_active) == 1


def test_every_range_identity_is_distinct() -> None:
    identities = [entry.range_id for entry in result().ranges]

    assert len(identities) == len(set(identities)) == 6


def test_ranges_are_ordered_by_formation() -> None:
    """§34. Supersession does not reorder history."""
    formed = [entry.formed_at for entry in result().ranges]

    assert formed == sorted(formed)


# --------------------------------------------------------------------------
# §49: the specific things this fixture had to prove
# --------------------------------------------------------------------------


def test_a_range_extends_on_a_bar_that_is_not_a_structure_event() -> None:
    """The reason for the two added wicks.

    At bar 23 the second range reaches 4075 - no structure event closed there,
    and none is needed. Extension is a property of observed price, not of events.
    """
    structure = analyse_structure(journey(), symbol="XAUUSD", as_of=START + HOUR * 24)
    at_23 = analyse_dealing_ranges(journey(), symbol="XAUUSD", as_of=START + HOUR * 24)
    current = at_23.active_range

    assert all(closing_bar(e.break_bar_close_time) != 23 for e in structure.breaks)
    assert current is not None
    assert current.terminal_price == Decimal("4075")
    assert closing_bar(current.terminal_bar_close_time) == 23
    assert current.equilibrium == Decimal("4022.5")


def test_the_same_protected_anchor_opens_a_second_distinct_range() -> None:
    """§21. Same origin price, same swing, two events, two ranges."""
    first, second = result().ranges[:2]

    assert first.origin_price == second.origin_price == Decimal("3970")
    assert first.protected_assignment_id != second.protected_assignment_id
    assert first.range_id != second.range_id
    assert first.status is RangeStatus.SUPERSEDED
    assert second.initial_terminal_price == Decimal("4070")


def test_a_newer_protected_low_rebases_the_range() -> None:
    """§22. The mandatory fixture, watched on both sides of the BOS."""
    before = analyse_dealing_ranges(journey(), symbol="XAUUSD", as_of=START + HOUR * 26)
    after = analyse_dealing_ranges(journey(), symbol="XAUUSD", as_of=START + HOUR * 31)

    early = before.active_range
    late = after.active_range
    assert early is not None
    assert late is not None
    assert early.origin_price == Decimal("3970"), "the newer low is active, not protected"
    assert late.origin_price == Decimal("3980"), "the BOS established it"
    assert early.range_id != late.range_id


def test_the_superseding_bar_extends_the_old_range_before_replacing_it() -> None:
    """§19, and the clearest case in the fixture.

    Bar 18 closes a BOS. During that bar the first range is still current, so
    its upper edge takes the bar's high of 4070 - and the very same bar is the
    new range's first terminal witness. Both are true, and both are recorded.
    """
    first, second = result().ranges[:2]

    assert first.terminal_price == Decimal("4070")
    assert closing_bar(first.terminal_bar_close_time) == 18
    assert closing_bar(first.superseded_at) == 18  # type: ignore[arg-type]
    assert second.initial_terminal_price == Decimal("4070")

    # Read at formation, not at the end: by the time the second range is
    # superseded its own witness has moved on to bar 30. What bar 18 is, is its
    # *first* witness.
    at_formation = analyse_dealing_ranges(journey(), symbol="XAUUSD", as_of=START + HOUR * 19)
    fresh = at_formation.active_range
    assert fresh is not None
    assert fresh.range_id == second.range_id
    assert closing_bar(fresh.terminal_bar_close_time) == 18


def test_an_equal_terminal_revisit_keeps_the_first_witness() -> None:
    """§14, arising naturally: bar 41 prints the same 3930 low as bar 40."""
    fourth, fifth = result().ranges[3], result().ranges[4]
    bars = {closing_bar(bar.timestamp + HOUR): bar for bar in journey().bars}

    assert bars[41].low == Decimal("3930") == bars[40].low
    assert fourth.terminal_price == Decimal("3930")
    assert closing_bar(fourth.terminal_bar_close_time) == 40
    assert closing_bar(fifth.terminal_bar_close_time) == 40


def test_price_traded_outside_a_range_that_was_still_current() -> None:
    """§10, §27. The excursion the outside labels exist for.

    The third range is anchored at 3980 and is still current when bar 33 prints
    a low of 3960. The range does not widen to swallow it.
    """
    at_32 = analyse_dealing_ranges(journey(), symbol="XAUUSD", as_of=START + HOUR * 33)
    current = at_32.active_range
    bar_33 = journey().bars[33]

    assert current is not None
    assert current.origin_price == Decimal("3980")
    assert bar_33.low == Decimal("3960")
    assert current.locate(bar_33.low) is PriceLocation.BELOW_RANGE
    assert current.status is RangeStatus.ACTIVE


def test_a_bearish_range_extends_downward() -> None:
    """§45 in the realistic path: 3960 at formation, 3930 by supersession."""
    fourth = result().ranges[3]

    assert fourth.direction is BreakDirection.BEARISH
    assert fourth.initial_terminal_price == Decimal("3960")
    assert fourth.terminal_price == Decimal("3930")
    assert fourth.origin_price == Decimal("4200"), "the fixed edge never moved"


def test_equilibrium_moved_as_the_second_range_extended() -> None:
    """4020 at formation, 4022.5 mid-life, 4027.5 by supersession."""
    at_formation = analyse_dealing_ranges(journey(), symbol="XAUUSD", as_of=START + HOUR * 19)
    mid_life = analyse_dealing_ranges(journey(), symbol="XAUUSD", as_of=START + HOUR * 24)
    final = result().ranges[1]

    assert at_formation.active_range is not None
    assert at_formation.active_range.equilibrium == Decimal("4020")
    assert mid_life.active_range is not None
    assert mid_life.active_range.equilibrium == Decimal("4022.5")
    assert final.equilibrium == Decimal("4027.5")
    assert at_formation.active_range.range_id == mid_life.active_range.range_id == final.range_id


# --------------------------------------------------------------------------
# §51: five timeframes, five independent answers
# --------------------------------------------------------------------------

OBSERVED_AT = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)

QUIET: list[Row] = [FLAT, FLAT, peak("4030"), FLAT, FLAT, FLAT, FLAT, FLAT]

PATHS: dict[Timeframe, list[Row]] = {
    Timeframe.H4: BEAR,
    Timeframe.H1: JOURNEY,
    Timeframe.M15: QUIET,
    Timeframe.M5: BULL,
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
    summary = {}
    for tf in ICT_TIMEFRAMES:
        analysis = analyse_snapshot_dealing_ranges(shot, tf)
        current = analysis.active_range
        summary[tf] = (
            len(analysis.ranges),
            None if current is None else current.direction.value,
        )

    assert summary == {
        Timeframe.H4: (1, "BEARISH"),
        Timeframe.H1: (6, "BULLISH"),
        Timeframe.M15: (0, None),
        Timeframe.M5: (1, "BULLISH"),
        Timeframe.M1: (0, None),
    }


def test_no_timeframe_range_overrides_another() -> None:
    """§51. An H4 range does not decide what premium means on M5."""
    shot = divergent_snapshot()

    h4 = analyse_snapshot_dealing_ranges(shot, Timeframe.H4)
    m5 = analyse_snapshot_dealing_ranges(shot, Timeframe.M5)

    assert {entry.timeframe for entry in h4.ranges} == {Timeframe.H4}
    assert {entry.timeframe for entry in m5.ranges} == {Timeframe.M5}
    assert {r.range_id for r in h4.ranges}.isdisjoint(r.range_id for r in m5.ranges)

    price = Decimal("4000")
    assert h4.locate(price) is not None
    assert m5.locate(price) is not None
    assert h4.locate(price) is not m5.locate(price), "the same price sits differently"


def test_analysing_one_timeframe_does_not_touch_another() -> None:
    shot = divergent_snapshot()

    alone = analyse_dealing_ranges(shot.require(Timeframe.M5), symbol="XAUUSD")
    after_the_others = [analyse_snapshot_dealing_ranges(shot, tf) for tf in ICT_TIMEFRAMES]

    assert alone == next(r for r in after_the_others if r.timeframe is Timeframe.M5)


def test_the_same_path_on_two_timeframes_finds_the_same_shape() -> None:
    shot = divergent_snapshot()

    m15 = analyse_snapshot_dealing_ranges(shot, Timeframe.M15)
    m1 = analyse_snapshot_dealing_ranges(shot, Timeframe.M1)

    assert len(m15.ranges) == len(m1.ranges)
    assert (m15.active_range_id is None) == (m1.active_range_id is None)


# --------------------------------------------------------------------------
# §52: H4 policy
# --------------------------------------------------------------------------


def test_provider_native_h4_timestamps_are_analysed_where_they_sit() -> None:
    """No resampling, no 00/04/08 correction, no DST assumption."""
    native_end = datetime(2026, 9, 7, 13, 0, tzinfo=UTC)
    shifted_end = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)

    native_series = ending_at(Timeframe.H4, BEAR, end=native_end)
    assert all(candle.timestamp.hour % 4 == 1 for candle in native_series.bars)

    native = analyse_dealing_ranges(native_series, symbol="XAUUSD")
    shifted = analyse_dealing_ranges(
        ending_at(Timeframe.H4, BEAR, end=shifted_end), symbol="XAUUSD"
    )

    assert [
        (str(r.lower), str(r.upper), str(r.equilibrium), r.status.value) for r in native.ranges
    ] == [(str(r.lower), str(r.upper), str(r.equilibrium), r.status.value) for r in shifted.ranges]
