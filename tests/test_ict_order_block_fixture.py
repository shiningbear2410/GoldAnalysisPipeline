"""One journey through every order-block formation case, and five timeframes.

Round 6.6d.1 §48-§50. The matrices in ``test_ict_order_block.py`` check one rule
at a time; this file pins the whole sequence, so a later round changing a
definition has to change these numbers on purpose.

Every number below was read out of the engine before it was written down.

The path is Round 6.6c.3b's 54-bar fixture with **two candle opens rewritten**
and nothing else touched. Structure detection reads highs, lows and closes; the
open is read by nothing in the ICT branch, which the first test here proves
rather than assumes. So the six structure events, their ids, their protected
assignments, their legs and every dealing range are bit-for-bit what Round
6.6c.3b pinned, and the only thing that changes is which candles have bodies.

What the two edits buy:

* bar 8, the protected low's own pivot candle, becomes bearish - so the first
  event selects its origin pivot (§7, §41.4), and that candle's wick runs
  3970-4010 against a body of 4000-4010, which is the materially-wider wick
  §48 asks for;
* bar 35 becomes a doji - so the bearish BOS at bar 40 has to walk back past
  five dojis and two same-direction candles to reach bar 30, which is the very
  candle the bearish MSS at bar 33 already selected (§44).
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
from goldpipeline.services.ict_order_block import (
    BodyDirection,
    OrderBlockAnalysis,
    OrderBlockZoneBasis,
    analyse_order_blocks,
    analyse_snapshot_order_blocks,
    body_direction,
)
from goldpipeline.services.ict_protected import analyse_protected_structure
from goldpipeline.services.ict_range import analyse_dealing_ranges
from goldpipeline.services.ict_structure import BreakDirection, analyse_structure
from tests.test_ict_order_block import BODY, FULL, reopen
from tests.test_ict_protected import closing_bar
from tests.test_ict_range_fixture import JOURNEY, PATHS, ending_at
from tests.test_ict_range_fixture import journey as range_journey
from tests.test_ict_structure import bar_index

HOUR = timedelta(hours=1)

OPENS = {8: "4010", 35: "4000"}
"""The two rewritten opens, and the entire difference from Round 6.6c.3b."""


def journey() -> IctTimeframeSnapshot:
    return reopen(range_journey(), OPENS)


def result(**kwargs: object) -> OrderBlockAnalysis:
    return analyse_order_blocks(journey(), config=FULL, symbol="XAUUSD", **kwargs)  # type: ignore[arg-type]


def summary(analysis: OrderBlockAnalysis) -> list[tuple[int, str, str, int]]:
    return [
        (
            closing_bar(entry.formed_at),
            entry.event_classification.value,
            entry.direction.value,
            bar_index(entry.source_bar_open_time),
        )
        for entry in analysis.order_blocks
    ]


# --------------------------------------------------------------------------
# §48: the fixture changed bodies and nothing else
# --------------------------------------------------------------------------


def test_rewriting_two_opens_moved_no_structure_protected_state_or_range() -> None:
    """The claim the whole fixture rests on, proved rather than asserted in prose.

    If a later round teaches structure, protected swings or ranges to read an
    open, this test fails first and says so - which is the point of writing it
    as a comparison rather than as a grep.
    """
    base, shaped = range_journey(), journey()

    assert [b.open for b in base.bars] != [b.open for b in shaped.bars], "something did change"
    assert [(b.timestamp, b.high, b.low, b.close) for b in base.bars] == [
        (b.timestamp, b.high, b.low, b.close) for b in shaped.bars
    ]

    for compute in (analyse_structure, analyse_protected_structure, analyse_dealing_ranges):
        assert compute(base, symbol="XAUUSD") == compute(shaped, symbol="XAUUSD")


def test_the_two_rewritten_candles_are_what_the_fixture_needs() -> None:
    shaped = journey()

    assert body_direction(shaped.bars[8]) is BodyDirection.BEARISH
    assert body_direction(shaped.bars[35]) is BodyDirection.NEUTRAL
    assert (
        bar_index(analyse_protected_structure(shaped, symbol="XAUUSD").legs[0].origin_pivot_time)
        == 8
    ), "bar 8 is the first leg's own origin pivot"


# --------------------------------------------------------------------------
# §49: the whole journey, pinned
# --------------------------------------------------------------------------


def test_the_journey_pins_every_order_block() -> None:
    assert summary(result()) == [
        (11, "INITIAL_BREAK", "BULLISH", 8),
        (18, "BOS", "BULLISH", 13),
        (33, "MSS", "BEARISH", 30),
        (40, "BOS", "BEARISH", 30),
    ]


def test_the_journey_pins_every_source_candle() -> None:
    assert [
        (
            bar_index(entry.source_bar_open_time),
            str(entry.source_open),
            str(entry.source_high),
            str(entry.source_low),
            str(entry.source_close),
            entry.source_body_direction.value,
        )
        for entry in result().order_blocks
    ] == [
        (8, "4010", "4010", "3970", "4000", "BEARISH"),
        (13, "4010", "4010", "3990", "4000", "BEARISH"),
        (30, "4000", "4085", "3990", "4065", "BULLISH"),
        (30, "4000", "4085", "3990", "4065", "BULLISH"),
    ]


def test_the_journey_pins_every_full_candle_zone() -> None:
    assert [
        (str(entry.lower), str(entry.upper), str(entry.midpoint), str(entry.width))
        for entry in result().order_blocks
    ] == [
        ("3970", "4010", "3990", "40"),
        ("3990", "4010", "4000", "20"),
        ("3990", "4085", "4037.5", "95"),
        ("3990", "4085", "4037.5", "95"),
    ]


def test_the_journey_pins_every_body_zone() -> None:
    body = analyse_order_blocks(journey(), config=BODY, symbol="XAUUSD")

    assert summary(body) == summary(result()), "the same candles, differently drawn"
    assert [
        (str(entry.lower), str(entry.upper), str(entry.midpoint), str(entry.width))
        for entry in body.order_blocks
    ] == [
        ("4000", "4010", "4005", "10"),
        ("4000", "4010", "4005", "10"),
        ("4000", "4065", "4032.5", "65"),
        ("4000", "4065", "4032.5", "65"),
    ]


def test_each_order_block_names_its_own_event_assignment_and_leg() -> None:
    structure = analyse_structure(journey(), symbol="XAUUSD")
    protected = analyse_protected_structure(journey(), symbol="XAUUSD")
    events = {closing_bar(e.break_bar_close_time): e for e in structure.breaks}
    legs = {closing_bar(leg.formed_at): leg for leg in protected.legs}

    for entry, at in zip(result().order_blocks, (11, 18, 33, 40), strict=True):
        assert entry.structure_event_id == events[at].event_id
        assert entry.structural_leg_id == legs[at].leg_id
        assert entry.protected_assignment_id == legs[at].protected_assignment_id
        assert entry.formed_at == legs[at].formed_at == events[at].break_bar_close_time
        assert entry.event_classification is events[at].classification
        assert entry.direction is legs[at].direction


def test_every_order_block_identity_is_distinct() -> None:
    identities = [entry.order_block_id for entry in result().order_blocks]

    assert len(identities) == len(set(identities)) == 4


def test_order_blocks_are_ordered_by_formation() -> None:
    """§38."""
    formed = [entry.formed_at for entry in result().order_blocks]

    assert formed == sorted(formed)


# --------------------------------------------------------------------------
# §48-§49: the specific things this fixture had to prove
# --------------------------------------------------------------------------


def test_the_first_event_selected_its_own_origin_pivot_candle() -> None:
    """§7, §41.4. The protected low's candle is inside the leg, so it is eligible."""
    protected = analyse_protected_structure(journey(), symbol="XAUUSD")
    first = result().order_blocks[0]

    assert bar_index(protected.legs[0].origin_pivot_time) == 8
    assert bar_index(first.source_bar_open_time) == 8
    assert first.source_low == protected.legs[0].origin_price == Decimal("3970")


def test_the_second_event_moved_on_to_a_newer_candle() -> None:
    """§45. Both events share the 3970 anchor; the later one does not go stale.

    Bar 8 is still inside the second leg's window and is still bearish. Bar 13
    is simply more recent, and recency is the whole rule.
    """
    shaped = journey()
    protected = analyse_protected_structure(shaped, symbol="XAUUSD")
    first, second = result().order_blocks[:2]

    assert protected.legs[0].origin_price == protected.legs[1].origin_price == Decimal("3970")
    assert bar_index(protected.legs[1].origin_pivot_time) == 8, "same window start"
    assert body_direction(shaped.bars[8]) is BodyDirection.BEARISH, "bar 8 is still eligible"
    assert bar_index(first.source_bar_open_time) == 8
    assert bar_index(second.source_bar_open_time) == 13
    assert first.order_block_id != second.order_block_id


def test_two_events_selected_the_same_source_candle() -> None:
    """§22, §44. One candle, two establishing events, two order blocks.

    The bearish MSS at bar 33 and the bearish BOS at bar 40 share the 4200
    anchor, and no bullish candle printed between them - bars 34-39 are dojis
    and bars 32-33 point the same way as the leg. So both point back at bar 30,
    and neither is deduplicated: they were established by different events.
    """
    third, fourth = result().order_blocks[2:]

    assert third.source_bar_open_time == fourth.source_bar_open_time
    assert (third.source_open, third.source_high, third.source_low, third.source_close) == (
        fourth.source_open,
        fourth.source_high,
        fourth.source_low,
        fourth.source_close,
    )
    assert third.lower == fourth.lower and third.upper == fourth.upper

    assert third.structure_event_id != fourth.structure_event_id
    assert third.structural_leg_id != fourth.structural_leg_id
    assert third.protected_assignment_id != fourth.protected_assignment_id
    assert third.order_block_id != fourth.order_block_id
    assert third.formed_at != fourth.formed_at


def test_the_bearish_bos_walked_back_past_dojis_and_same_direction_candles() -> None:
    """§10, §41.5, §41.7. What lies between bar 30 and bar 40, and why none of it won."""
    shaped = journey()
    bodies = {index: body_direction(shaped.bars[index]) for index in range(31, 40)}

    assert bodies[32] is BodyDirection.BEARISH, "same direction as the leg"
    assert bodies[33] is BodyDirection.BEARISH
    assert all(bodies[index] is BodyDirection.NEUTRAL for index in (31, 34, 35, 36, 37, 38, 39))
    assert bar_index(result().order_blocks[3].source_bar_open_time) == 30


def test_two_events_produced_no_order_block_at_all() -> None:
    """§9, §41.8, §41.9. Two different reasons, both legitimate."""
    shaped = journey()
    structure = analyse_structure(shaped, symbol="XAUUSD")
    protected = analyse_protected_structure(shaped, symbol="XAUUSD")
    analysis = result()

    without = [
        closing_bar(e.break_bar_close_time)
        for e in structure.breaks
        if analysis.for_event(e.event_id) is None
    ]
    assert without == [30, 50]

    # Bar 30's leg starts at the bar-22 pivot; the nearest bearish candles, at
    # bars 13 and 20, are both before it. Bar 50's leg holds nothing but dojis.
    window_30 = next(leg for leg in protected.legs if closing_bar(leg.formed_at) == 30)
    assert bar_index(window_30.origin_pivot_time) == 22
    assert body_direction(shaped.bars[20]) is BodyDirection.BEARISH, "eligible, but too early"
    assert all(
        body_direction(shaped.bars[index]) is BodyDirection.NEUTRAL for index in range(22, 30)
    )

    window_50 = next(leg for leg in protected.legs if closing_bar(leg.formed_at) == 50)
    assert bar_index(window_50.origin_pivot_time) == 44
    assert all(
        body_direction(shaped.bars[index]) is BodyDirection.NEUTRAL for index in range(44, 50)
    )


def test_the_journey_holds_both_directions() -> None:
    directions = {entry.direction for entry in result().order_blocks}

    assert directions == {BreakDirection.BULLISH, BreakDirection.BEARISH}


def test_a_source_wick_runs_materially_wider_than_its_body() -> None:
    """§48. If the two bases never differed, the zone policy would be decorative."""
    full = result().order_blocks
    body = analyse_order_blocks(journey(), config=BODY, symbol="XAUUSD").order_blocks

    assert full[0].width == Decimal("40") and body[0].width == Decimal("10")
    assert full[2].width == Decimal("95") and body[2].width == Decimal("65")
    for wide, narrow in zip(full, body, strict=True):
        assert wide.full_candle_bounds == (wide.source_low, wide.source_high)
        assert narrow.body_bounds == (narrow.lower, narrow.upper)


def test_a_bullish_break_candle_became_a_bearish_events_order_block() -> None:
    """Bar 30 has two honest roles, exactly as bar 18 had two in Round 6.6c.3b.

    It is the break bar of the bullish BOS at bar 30, and the source candle of
    the bearish MSS at bar 33. Nothing needs reconciling: they are different
    events looking at the same candle for different reasons.
    """
    structure = analyse_structure(journey(), symbol="XAUUSD")
    bos_30 = next(e for e in structure.breaks if closing_bar(e.break_bar_close_time) == 30)
    third = result().order_blocks[2]

    assert bar_index(bos_30.break_bar_open_time) == 30
    assert bar_index(third.source_bar_open_time) == 30
    assert third.structure_event_id != bos_30.event_id


# --------------------------------------------------------------------------
# §50: five timeframes, five independent answers
# --------------------------------------------------------------------------

OBSERVED_AT = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)

TF_OPENS: dict[Timeframe, dict[int, str]] = {
    Timeframe.H1: OPENS,
    Timeframe.M5: {7: "4010"},
    Timeframe.H4: {7: "3990"},
}
"""Bodies shaped per timeframe, so more than one of them has a candle to find."""


def shaped_series(timeframe: Timeframe, rows: Sequence[object]) -> IctTimeframeSnapshot:
    built = ending_at(timeframe, rows, end=OBSERVED_AT)  # type: ignore[arg-type]
    return reopen(built, TF_OPENS.get(timeframe, {}))


def divergent_snapshot() -> IctMarketSnapshot:
    return IctMarketSnapshot(
        observed_at=OBSERVED_AT,
        symbol="XAUUSD",
        provider="fixture",
        provider_symbol=None,
        timeframes=tuple(shaped_series(tf, PATHS[tf]) for tf in ICT_TIMEFRAMES),
    )


def test_five_timeframes_reach_five_answers_of_their_own() -> None:
    shot = divergent_snapshot()

    assert {
        tf: len(analyse_snapshot_order_blocks(shot, tf, config=FULL).order_blocks)
        for tf in ICT_TIMEFRAMES
    } == {
        Timeframe.H4: 1,
        Timeframe.H1: 4,
        Timeframe.M15: 0,
        Timeframe.M5: 1,
        Timeframe.M1: 0,
    }


def test_no_timeframe_order_block_overrides_another() -> None:
    """§50. An H4 zone does not replace an M5 zone."""
    shot = divergent_snapshot()

    h4 = analyse_snapshot_order_blocks(shot, Timeframe.H4, config=FULL)
    m5 = analyse_snapshot_order_blocks(shot, Timeframe.M5, config=FULL)

    assert {entry.timeframe for entry in h4.order_blocks} == {Timeframe.H4}
    assert {entry.timeframe for entry in m5.order_blocks} == {Timeframe.M5}
    assert {entry.order_block_id for entry in h4.order_blocks}.isdisjoint(
        entry.order_block_id for entry in m5.order_blocks
    )
    assert h4.order_blocks[0].direction is not m5.order_blocks[0].direction


def test_analysing_one_timeframe_does_not_touch_another() -> None:
    shot = divergent_snapshot()

    alone = analyse_order_blocks(shot.require(Timeframe.M5), config=FULL, symbol="XAUUSD")
    after_the_others = [
        analyse_snapshot_order_blocks(shot, tf, config=FULL) for tf in ICT_TIMEFRAMES
    ]

    assert alone == next(r for r in after_the_others if r.timeframe is Timeframe.M5)


def test_the_same_path_on_two_timeframes_finds_the_same_shape() -> None:
    shot = divergent_snapshot()

    m15 = analyse_snapshot_order_blocks(shot, Timeframe.M15, config=FULL)
    m1 = analyse_snapshot_order_blocks(shot, Timeframe.M1, config=FULL)

    assert len(m15.order_blocks) == len(m1.order_blocks) == 0


# --------------------------------------------------------------------------
# H4 policy: unchanged
# --------------------------------------------------------------------------


def native_h4(first_open: datetime) -> IctTimeframeSnapshot:
    rows = PATHS[Timeframe.H4]
    duration = Timeframe.H4.duration
    assert duration is not None

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

    built = build_timeframe_snapshot(
        timeframe=Timeframe.H4,
        bars=tuple(bars),
        observed_at=first_open + duration * len(rows),
    )
    return reopen(built, TF_OPENS[Timeframe.H4])


def test_the_h4_grid_a_provider_uses_changes_no_zone() -> None:
    """Provider-native timestamps are analysed where they sit, as since 6.6a."""
    tradingview = native_h4(datetime(2026, 9, 1, 1, tzinfo=UTC))
    broker = native_h4(datetime(2026, 9, 1, 0, tzinfo=UTC))

    one = analyse_order_blocks(tradingview, config=FULL, symbol="XAUUSD")
    two = analyse_order_blocks(broker, config=FULL, symbol="XAUUSD")

    assert len(one.order_blocks) == len(two.order_blocks) == 1
    for left, right in zip(one.order_blocks, two.order_blocks, strict=True):
        assert (left.lower, left.upper, left.midpoint, left.width) == (
            right.lower,
            right.upper,
            right.midpoint,
            right.width,
        )
        assert left.direction is right.direction
        assert left.source_bar_open_time != right.source_bar_open_time, "different grids"


def test_the_journey_is_the_range_journey_with_two_opens_moved() -> None:
    """A guard on the fixture itself, so a later edit to 6.6c.3b's path is noticed."""
    assert len(JOURNEY) == 54
    assert set(OPENS) == {8, 35}
    assert journey().bar_count == 54


def test_the_zone_basis_is_reported_on_the_analysis() -> None:
    assert result().zone_basis is OrderBlockZoneBasis.FULL_CANDLE
    assert (
        analyse_order_blocks(journey(), config=BODY, symbol="XAUUSD").zone_basis
        is OrderBlockZoneBasis.BODY
    )


def test_an_unknown_identity_simply_is_not_found() -> None:
    analysis = result()

    assert analysis.order_block("nope") is None
    assert analysis.for_event("nope") is None
    assert (
        analysis.order_block(analysis.order_blocks[0].order_block_id) is (analysis.order_blocks[0])
    )
