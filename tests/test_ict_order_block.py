"""Which candle a structural event points back to, one rule at a time.

Round 6.6d.1 §41-§42, §46. Each case here is a handful of bars whose answer can
be worked out on paper. The realistic path is in
``test_ict_order_block_fixture.py`` and does the other job.

The bar vocabulary is Round 6.6b's, with one addition. ``series()`` derives every
bar's open from the previous close, which makes almost every filler candle a
doji - useful for the doji rule and useless for everything else. :func:`reopen`
rewrites chosen bars' opens *after* the series is built, which gives a fixture
full control over bodies while changing no structure at all: nothing in the ICT
branch reads ``bar.open``, which ``test_ict_order_block_fixture.py`` proves
rather than assumes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import IctTimeframeSnapshot, build_timeframe_snapshot
from goldpipeline.schemas.market import OHLCBar
from goldpipeline.services.ict_order_block import (
    BodyDirection,
    OrderBlockConfig,
    OrderBlockError,
    OrderBlockZoneBasis,
    analyse_order_blocks,
    body_direction,
    source_body_for,
    zone_of,
)
from goldpipeline.services.ict_protected import analyse_protected_structure
from goldpipeline.services.ict_structure import (
    BreakClassification,
    BreakDirection,
    analyse_structure,
)
from tests.test_ict_protected import closing_bar
from tests.test_ict_structure import FLAT, START, Row, bar_index, drive, peak, series, trough

HOUR = timedelta(hours=1)

FULL = OrderBlockConfig(zone_basis=OrderBlockZoneBasis.FULL_CANDLE)
BODY = OrderBlockConfig(zone_basis=OrderBlockZoneBasis.BODY)


def reopen(snapshot: IctTimeframeSnapshot, opens: Mapping[int, str]) -> IctTimeframeSnapshot:
    """Rewrite the opens of the bars named by index, leaving everything else alone.

    The open must stay inside the bar's own high and low, so ``OHLCBar`` refuses
    an impossible candle rather than a fixture inventing one.
    """
    bars = tuple(
        bar.model_copy(update={"open": Decimal(opens[index])}) if index in opens else bar
        for index, bar in enumerate(snapshot.bars)
    )
    return build_timeframe_snapshot(
        timeframe=snapshot.timeframe, bars=bars, observed_at=snapshot.latest_closed_at
    )


def bodied(
    rows: Sequence[Row], opens: Mapping[int, str], *, bars_kept: int | None = None
) -> IctTimeframeSnapshot:
    return reopen(series(rows, bars_kept=bars_kept), opens)


def blocks(
    rows: Sequence[Row], opens: Mapping[int, str], *, config: OrderBlockConfig = FULL
) -> list[tuple[int, str, int]]:
    """``(formed bar, direction, source bar)`` for each order block found."""
    analysis = analyse_order_blocks(bodied(rows, opens), config=config, symbol="XAUUSD")
    return [
        (
            closing_bar(entry.formed_at),
            entry.direction.value,
            bar_index(entry.source_bar_open_time),
        )
        for entry in analysis.order_blocks
    ]


# A neutral market that puts in a low, rallies through the high above it, and
# thereby makes one bullish INITIAL_BREAK with a protected low behind it.
BULL_LEG: list[Row] = [
    FLAT,
    FLAT,
    peak("4030"),
    FLAT,
    FLAT,
    trough("3970"),
    FLAT,
    FLAT,
    FLAT,
    FLAT,
    drive("4040"),
    drive("4040"),
    FLAT,
]

# The mirror: a high, then a break down through the low beneath it.
BEAR_LEG: list[Row] = [
    FLAT,
    FLAT,
    trough("3970"),
    FLAT,
    FLAT,
    peak("4030"),
    FLAT,
    FLAT,
    FLAT,
    FLAT,
    drive("3960"),
    drive("3960"),
    FLAT,
]


def test_the_two_scaffolds_make_the_events_the_rest_of_this_file_assumes() -> None:
    """Read once here so no other test in this file has to re-derive it."""
    for rows, direction, pivot, break_bar in (
        (BULL_LEG, BreakDirection.BULLISH, 5, 10),
        (BEAR_LEG, BreakDirection.BEARISH, 5, 10),
    ):
        structure = analyse_structure(series(rows), symbol="XAUUSD")
        protected = analyse_protected_structure(series(rows), symbol="XAUUSD")

        assert len(structure.breaks) == 1
        assert structure.breaks[0].direction is direction
        assert structure.breaks[0].classification is BreakClassification.INITIAL_BREAK
        assert len(protected.legs) == 1
        assert bar_index(protected.legs[0].origin_pivot_time) == pivot
        assert bar_index(protected.legs[0].terminal_bar_open_time) == break_bar


# --------------------------------------------------------------------------
# §4-§5: body direction is arithmetic
# --------------------------------------------------------------------------


def bar(open_: str, high: str, low: str, close: str) -> OHLCBar:
    return OHLCBar(
        timestamp=START,
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
    )


def test_a_close_below_its_open_is_bearish() -> None:
    assert body_direction(bar("4010", "4015", "3995", "4000")) is BodyDirection.BEARISH


def test_a_close_above_its_open_is_bullish() -> None:
    assert body_direction(bar("4000", "4015", "3995", "4010")) is BodyDirection.BULLISH


def test_an_exact_doji_is_neutral() -> None:
    """§4. ``close == open`` is neither, and no tolerance makes it one."""
    assert body_direction(bar("4000", "4015", "3995", "4000")) is BodyDirection.NEUTRAL


def test_the_smallest_representable_body_is_still_a_body() -> None:
    """§4. No tolerance. A one-cent body is a direction, not a rounding error."""
    assert body_direction(bar("4000.00", "4015", "3995", "4000.01")) is BodyDirection.BULLISH
    assert body_direction(bar("4000.01", "4015", "3995", "4000.00")) is BodyDirection.BEARISH


def test_the_same_body_written_with_different_exponents_is_the_same_direction() -> None:
    """Decimal comparison, not string comparison: ``4000`` equals ``4000.00``."""
    assert body_direction(bar("4000.00", "4010", "3990", "4000")) is BodyDirection.NEUTRAL


def test_a_long_upper_wick_does_not_make_a_candle_bullish() -> None:
    """§5. Wick shape is not direction."""
    assert body_direction(bar("4010", "4100", "3990", "4000")) is BodyDirection.BEARISH


def test_direction_never_looks_at_the_neighbouring_candle() -> None:
    """§5. Close-versus-previous-close is a different measurement entirely."""
    first = bar("4000", "4060", "3990", "4050")
    second = bar("4000", "4010", "3990", "4000")

    assert second.close < first.close, "it closed below the bar before it"
    assert body_direction(second) is BodyDirection.NEUTRAL, "and its own body points nowhere"


def test_a_bullish_leg_wants_a_bearish_candle_and_the_reverse() -> None:
    assert source_body_for(BreakDirection.BULLISH) is BodyDirection.BEARISH
    assert source_body_for(BreakDirection.BEARISH) is BodyDirection.BULLISH


# --------------------------------------------------------------------------
# §41: the source-selection matrix
# --------------------------------------------------------------------------


def test_a_bullish_leg_selects_the_latest_bearish_candle() -> None:
    """§41.1."""
    assert blocks(BULL_LEG, {7: "4010"}) == [(10, "BULLISH", 7)]


def test_a_bearish_leg_selects_the_latest_bullish_candle() -> None:
    """§41.2."""
    assert blocks(BEAR_LEG, {7: "3990"}) == [(10, "BEARISH", 7)]


# A bullish break whose own body points down: the pair reaches 4060 and closes
# back at 4040, which is still above the 4030 level it broke. Written as a pair
# so the taller high makes no strict pivot.
BEARISH_BREAK_BAR: list[Row] = [
    *BULL_LEG[:10],
    ("4060", "3990", "4040"),
    ("4060", "3990", "4040"),
    FLAT,
]


def test_the_break_bar_itself_is_never_the_source() -> None:
    """§41.3, and the case the realistic fixture cannot make.

    Bar 10 closes at 4040 - a bullish break - while opening at 4050, so its own
    body is bearish. It is still not eligible: it is the event, not the candle
    the event expanded away from. With no other bearish candle in the leg there
    is simply no order block.
    """
    snapshot = bodied(BEARISH_BREAK_BAR, {10: "4050"})
    breaking = snapshot.bars[10]
    protected = analyse_protected_structure(snapshot, symbol="XAUUSD")

    assert body_direction(breaking) is BodyDirection.BEARISH
    assert breaking.close == Decimal("4040"), "and it really did break upward"
    assert bar_index(protected.legs[0].terminal_bar_open_time) == 10, "it is the break bar"
    assert blocks(BEARISH_BREAK_BAR, {10: "4050"}) == []


def test_the_origin_pivot_candle_may_be_the_source() -> None:
    """§7, §41.4. The protected swing's own candle is inside the leg."""
    assert blocks(BULL_LEG, {5: "4010"}) == [(10, "BULLISH", 5)]


def test_a_doji_immediately_before_the_break_is_skipped() -> None:
    """§10, §41.5. Bar 9 is a doji; the search steps back to bar 6."""
    snapshot = bodied(BULL_LEG, {6: "4010"})

    assert body_direction(snapshot.bars[9]) is BodyDirection.NEUTRAL
    assert blocks(BULL_LEG, {6: "4010"}) == [(10, "BULLISH", 6)]


def test_the_latest_of_several_opposite_candles_wins() -> None:
    """§8, §41.6. Three eligible candles, and recency is the only rule.

    Bar 6 has the largest body and bar 5 came first. Neither matters.
    """
    snapshot = bodied(BULL_LEG, {5: "4010", 6: "4010", 8: "4005"})
    bodies = [body_direction(snapshot.bars[i]) for i in (5, 6, 8)]

    assert bodies == [BodyDirection.BEARISH] * 3
    assert snapshot.bars[6].open - snapshot.bars[6].close > snapshot.bars[8].open - (
        snapshot.bars[8].close
    ), "the biggest body is not the one selected"
    assert blocks(BULL_LEG, {5: "4010", 6: "4010", 8: "4005"}) == [(10, "BULLISH", 8)]


def test_same_direction_candles_are_ignored() -> None:
    """§41.7. A bullish leg does not settle for a bullish candle."""
    snapshot = bodied(BULL_LEG, {8: "3995"})

    assert body_direction(snapshot.bars[8]) is BodyDirection.BULLISH
    assert blocks(BULL_LEG, {8: "3995"}) == []


def test_no_opposite_candle_at_all_means_no_order_block() -> None:
    """§9, §41.8. Absence is a result, not a gap to fill."""
    analysis = analyse_order_blocks(series(BULL_LEG), config=FULL, symbol="XAUUSD")
    protected = analyse_protected_structure(series(BULL_LEG), symbol="XAUUSD")

    assert len(protected.legs) == 1, "the leg exists"
    assert analysis.order_blocks == (), "and holds no eligible candle"


def test_an_opposite_candle_before_the_origin_pivot_is_ignored() -> None:
    """§6, §41.9. The scan does not run back past the protected swing."""
    snapshot = bodied(BULL_LEG, {3: "4010"})

    assert body_direction(snapshot.bars[3]) is BodyDirection.BEARISH
    assert (
        bar_index(analyse_protected_structure(snapshot, symbol="XAUUSD").legs[0].origin_pivot_time)
        == 5
    )
    assert blocks(BULL_LEG, {3: "4010"}) == []


def test_a_candle_beyond_the_break_bar_never_enters_the_window() -> None:
    """§37, §41.10, both halves of "only closed candles, and only inside the leg".

    Bar 12 is bearish and would be a perfectly good source if the window ran
    forward - it does not. And while it is still forming it is not in the series
    at all, which is the closed-bar semantics this branch has used since 6.6a.
    """
    snapshot = series(BULL_LEG)
    forming = build_timeframe_snapshot(
        timeframe=snapshot.timeframe, bars=snapshot.bars, observed_at=START + HOUR * 12
    )

    assert body_direction(snapshot.bars[12]) is BodyDirection.BEARISH
    assert len(forming.bars) == 12, "the twelfth bar has not closed"
    assert analyse_order_blocks(snapshot, config=FULL, symbol="XAUUSD").order_blocks == ()


def test_the_source_candle_is_recorded_exactly_as_it_printed() -> None:
    """§12, §41.11. Four prices, no reconstruction."""
    snapshot = bodied(BULL_LEG, {7: "4010"})
    found = analyse_order_blocks(snapshot, config=FULL, symbol="XAUUSD").order_blocks[0]
    original = snapshot.bars[7]

    assert (found.source_open, found.source_high, found.source_low, found.source_close) == (
        original.open,
        original.high,
        original.low,
        original.close,
    )
    assert found.source_bar_open_time == original.timestamp
    assert found.source_bar_close_time == original.timestamp + HOUR
    assert found.source_body_direction is BodyDirection.BEARISH


def test_prices_survive_as_decimals_with_their_own_exponents() -> None:
    """§41.12. Nothing is normalized, rounded or turned into a float."""
    snapshot = bodied(BULL_LEG, {7: "4010"})
    found = analyse_order_blocks(snapshot, config=FULL, symbol="XAUUSD").order_blocks[0]

    for value in (found.source_open, found.lower, found.upper, found.midpoint, found.width):
        assert isinstance(value, Decimal)


# --------------------------------------------------------------------------
# §42: every classification, both directions
# --------------------------------------------------------------------------


def test_an_initial_break_keeps_its_classification() -> None:
    """§28. Not relabelled BOS because it produced an order block."""
    found = analyse_order_blocks(
        bodied(BULL_LEG, {7: "4010"}), config=FULL, symbol="XAUUSD"
    ).order_blocks[0]

    assert found.event_classification is BreakClassification.INITIAL_BREAK
    assert found.direction is BreakDirection.BULLISH


def test_a_bearish_initial_break_keeps_its_classification() -> None:
    found = analyse_order_blocks(
        bodied(BEAR_LEG, {7: "3990"}), config=FULL, symbol="XAUUSD"
    ).order_blocks[0]

    assert found.event_classification is BreakClassification.INITIAL_BREAK
    assert found.direction is BreakDirection.BEARISH


def test_every_order_block_reads_opposite_to_its_own_direction() -> None:
    """§42, stated once as an invariant rather than case by case.

    Checked across both scaffolds and the realistic journey, under both bases.
    """
    from tests.test_ict_order_block_fixture import journey

    for snapshot in (
        bodied(BULL_LEG, {5: "4010", 7: "4010"}),
        bodied(BEAR_LEG, {5: "3990", 7: "3990"}),
        journey(),
    ):
        for config in (FULL, BODY):
            for entry in analyse_order_blocks(
                snapshot, config=config, symbol="XAUUSD"
            ).order_blocks:
                assert entry.source_body_direction is source_body_for(entry.direction)


# --------------------------------------------------------------------------
# §46: no anchor, no order block
# --------------------------------------------------------------------------


def test_a_break_with_no_protected_anchor_produces_nothing() -> None:
    """§46, using Round 6.6b's own anchorless cases.

    Two of that journey's events establish nothing, so they draw no leg. They
    are still events; they simply have no candle to point at, and no fallback
    invents one.
    """
    from tests.test_ict_structure_fixture import JOURNEY as SPARSE

    snapshot = series(SPARSE)
    structure = analyse_structure(snapshot, symbol="XAUUSD")
    protected = analyse_protected_structure(snapshot, symbol="XAUUSD")
    analysis = analyse_order_blocks(snapshot, config=FULL, symbol="XAUUSD")

    anchorless = [e for e in structure.breaks if e.opposite_active_swing_id is None]
    assert len(anchorless) == 3
    assert {e.classification for e in anchorless} == {
        BreakClassification.MSS,
        BreakClassification.BOS,
    }
    assert len(structure.breaks) == 5
    assert len(protected.legs) == 2, "only the anchored events drew legs"

    for event in anchorless:
        assert analysis.for_event(event.event_id) is None


def test_an_anchorless_event_gets_no_event_bar_fallback() -> None:
    """§46, §2. Not the break bar, not the event-bar extreme, not an N-bar low."""
    from tests.test_ict_structure_fixture import JOURNEY as SPARSE

    snapshot = series(SPARSE)
    structure = analyse_structure(snapshot, symbol="XAUUSD")
    analysis = analyse_order_blocks(snapshot, config=FULL, symbol="XAUUSD")

    anchorless = {
        e.break_bar_open_time for e in structure.breaks if e.opposite_active_swing_id is None
    }
    assert anchorless
    for entry in analysis.order_blocks:
        assert entry.source_bar_open_time not in anchorless


def test_a_later_anchored_event_still_produces_normally() -> None:
    """§46. One event without an anchor does not poison the next one."""
    from tests.test_ict_order_block_fixture import journey

    analysis = analyse_order_blocks(journey(), config=FULL, symbol="XAUUSD")
    structure = analyse_structure(journey(), symbol="XAUUSD")
    without = [e for e in structure.breaks if analysis.for_event(e.event_id) is None]

    assert without, "the journey does contain events with no order block"
    assert closing_bar(analysis.order_blocks[-1].formed_at) > closing_bar(
        without[0].break_bar_close_time
    ), "and a later event still produced one"


# --------------------------------------------------------------------------
# §31-§33: exact lookups and refusals
# --------------------------------------------------------------------------


def test_a_leg_whose_break_bar_is_missing_is_refused() -> None:
    """§31. No nearest-time search, and no silently widened scan."""
    snapshot = bodied(BULL_LEG, {7: "4010"})
    structure = analyse_structure(snapshot, symbol="XAUUSD")
    protected = analyse_protected_structure(snapshot, symbol="XAUUSD")

    thinned = build_timeframe_snapshot(
        timeframe=snapshot.timeframe,
        bars=tuple(b for i, b in enumerate(snapshot.bars) if i != 10),
        observed_at=snapshot.latest_closed_at,
    )
    with pytest.raises(OrderBlockError, match="break bar"):
        analyse_order_blocks(
            thinned, config=FULL, structure=structure, protected=protected, symbol="XAUUSD"
        )


def test_a_leg_whose_origin_pivot_is_missing_is_refused() -> None:
    """§31. The lower bound must be a real bar too."""
    snapshot = bodied(BULL_LEG, {7: "4010"})
    structure = analyse_structure(snapshot, symbol="XAUUSD")
    protected = analyse_protected_structure(snapshot, symbol="XAUUSD")

    thinned = build_timeframe_snapshot(
        timeframe=snapshot.timeframe,
        bars=tuple(b for i, b in enumerate(snapshot.bars) if i != 5),
        observed_at=snapshot.latest_closed_at,
    )
    with pytest.raises(OrderBlockError, match="pivot"):
        analyse_order_blocks(
            thinned, config=FULL, structure=structure, protected=protected, symbol="XAUUSD"
        )


def test_a_leg_paired_with_the_wrong_event_is_refused() -> None:
    """§32. Event E1 with leg L2 would describe a move that never happened."""
    from dataclasses import replace

    from tests.test_ict_order_block_fixture import journey

    snapshot = journey()
    structure = analyse_structure(snapshot, symbol="XAUUSD")
    protected = analyse_protected_structure(snapshot, symbol="XAUUSD")

    first, second = protected.legs[0], protected.legs[1]
    crossed = replace(
        protected, legs=(replace(first, event_id=second.event_id), *protected.legs[1:])
    )

    with pytest.raises(OrderBlockError):
        analyse_order_blocks(
            snapshot, config=FULL, structure=structure, protected=crossed, symbol="XAUUSD"
        )


def test_a_leg_naming_an_absent_event_is_refused() -> None:
    from dataclasses import replace

    from tests.test_ict_order_block_fixture import journey

    snapshot = journey()
    structure = analyse_structure(snapshot, symbol="XAUUSD")
    protected = analyse_protected_structure(snapshot, symbol="XAUUSD")
    broken = replace(
        protected, legs=(replace(protected.legs[0], event_id="nope"), *protected.legs[1:])
    )

    with pytest.raises(OrderBlockError, match="which this structure does not hold"):
        analyse_order_blocks(
            snapshot, config=FULL, structure=structure, protected=broken, symbol="XAUUSD"
        )


def test_a_leg_naming_an_absent_assignment_is_refused() -> None:
    from dataclasses import replace

    from tests.test_ict_order_block_fixture import journey

    snapshot = journey()
    structure = analyse_structure(snapshot, symbol="XAUUSD")
    protected = analyse_protected_structure(snapshot, symbol="XAUUSD")
    broken = replace(
        protected,
        legs=(replace(protected.legs[0], protected_assignment_id="nope"), *protected.legs[1:]),
    )

    with pytest.raises(OrderBlockError, match="which this analysis does not hold"):
        analyse_order_blocks(
            snapshot, config=FULL, structure=structure, protected=broken, symbol="XAUUSD"
        )


# --------------------------------------------------------------------------
# §33: supplied analyses must describe this request
# --------------------------------------------------------------------------


def test_a_structure_from_another_instant_is_refused_not_trimmed() -> None:
    """§33. A final analysis handed to a historical query would leak legs."""
    from tests.test_ict_order_block_fixture import journey

    snapshot = journey()
    final = analyse_structure(snapshot, symbol="XAUUSD")

    with pytest.raises(OrderBlockError, match="supply one computed for the same instant"):
        analyse_order_blocks(
            snapshot, config=FULL, structure=final, symbol="XAUUSD", as_of=START + HOUR * 20
        )


def test_a_protected_analysis_from_another_instant_is_refused() -> None:
    from tests.test_ict_order_block_fixture import journey

    snapshot = journey()
    moment = START + HOUR * 20
    structure = analyse_structure(snapshot, symbol="XAUUSD", as_of=moment)
    final = analyse_protected_structure(snapshot, symbol="XAUUSD")

    with pytest.raises(OrderBlockError, match="supply one computed for the same instant"):
        analyse_order_blocks(
            snapshot,
            config=FULL,
            structure=structure,
            protected=final,
            symbol="XAUUSD",
            as_of=moment,
        )


def test_a_structure_for_another_timeframe_is_refused() -> None:
    """The instant matches; the timeframe does not, and that is enough."""
    from dataclasses import replace

    snapshot = bodied(BULL_LEG, {7: "4010"})
    other = replace(analyse_structure(snapshot, symbol="XAUUSD"), timeframe=Timeframe.M15)

    with pytest.raises(OrderBlockError, match="is for M15"):
        analyse_order_blocks(snapshot, config=FULL, structure=other, symbol="XAUUSD")


def test_a_structure_for_another_symbol_is_refused() -> None:
    snapshot = bodied(BULL_LEG, {7: "4010"})
    other = analyse_structure(snapshot, symbol="EURUSD")

    with pytest.raises(OrderBlockError, match="is for 'EURUSD'"):
        analyse_order_blocks(snapshot, config=FULL, structure=other, symbol="XAUUSD")


def test_a_matching_analysis_is_accepted_and_changes_nothing() -> None:
    """The other half: supplying what a composite stage already has is free."""
    snapshot = bodied(BULL_LEG, {7: "4010"})
    structure = analyse_structure(snapshot, symbol="XAUUSD")
    protected = analyse_protected_structure(snapshot, structure=structure, symbol="XAUUSD")

    assert analyse_order_blocks(
        snapshot, config=FULL, structure=structure, protected=protected, symbol="XAUUSD"
    ) == analyse_order_blocks(snapshot, config=FULL, symbol="XAUUSD")


def test_an_as_of_before_any_bar_closed_is_refused() -> None:
    with pytest.raises(ValueError, match="no bar closed"):
        analyse_order_blocks(series(BULL_LEG), config=FULL, symbol="XAUUSD", as_of=START)


# --------------------------------------------------------------------------
# §19-§20: geometry
# --------------------------------------------------------------------------


def test_the_midpoint_is_exactly_half_way() -> None:
    """§19. No rounding, and a third decimal is allowed to appear."""
    assert zone_of(bar("4000", "4015", "3995", "4010"), OrderBlockZoneBasis.FULL_CANDLE) == (
        Decimal("3995"),
        Decimal("4015"),
    )
    found = analyse_order_blocks(
        bodied(BULL_LEG, {7: "4010"}), config=FULL, symbol="XAUUSD"
    ).order_blocks[0]

    assert found.midpoint == (found.lower + found.upper) / Decimal(2)
    assert found.width == found.upper - found.lower


def test_every_order_block_has_positive_width() -> None:
    """§20. A zero-width zone would be a level pretending to be a zone."""
    from tests.test_ict_order_block_fixture import journey

    for config in (FULL, BODY):
        found = analyse_order_blocks(journey(), config=config, symbol="XAUUSD").order_blocks
        assert found
        for entry in found:
            assert entry.width > 0
            assert entry.upper > entry.lower
