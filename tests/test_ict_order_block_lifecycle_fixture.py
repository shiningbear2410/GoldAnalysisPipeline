"""One journey through every lifecycle state, under all three mitigation rules.

Round 6.6d.2 §52-§53. The matrices in ``test_ict_order_block_lifecycle.py``
check one rule at a time; this file pins whole state tables, so a later round
changing a definition has to change these numbers on purpose.

Every number below was read out of the engine before it was written down.

The path is Round 6.6d.1's 54-bar order-block fixture with **nine bars appended
and nothing else touched**. Appending only after the last existing formation is
what keeps the older expectations honest: all six structure events keep their
ids, all four order blocks keep their source candles and zones, and the first
test here proves it rather than asserting it in prose.

What the tail buys, and why it was needed:

* a fifth order block, formed by a bearish MSS at bar 56 out of the bar-50
  source candle, which then sits **untouched** for four bars - the ACTIVE case
  the 54-bar path never produced, since every earlier block is met by the very
  next candle;
* a gap at bar 61 that opens above the zone and closes above it, retiring that
  fifth block **without ever touching it** - §21's direct
  ``ACTIVE -> INVALIDATED``, which needs a real gap and so could not be
  constructed from the original path either;
* and one accident worth keeping: bar 56's own range runs 3900-3990 against a
  zone of 3990-4045, so the formation bar genuinely overlaps the block it
  creates. §10 stops being hypothetical.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import (
    ICT_TIMEFRAMES,
    IctMarketSnapshot,
    IctTimeframeSnapshot,
)
from goldpipeline.services.ict_order_block import (
    OrderBlockZoneBasis,
    analyse_order_blocks,
)
from goldpipeline.services.ict_order_block_lifecycle import (
    OrderBlockLifecycleAnalysis,
    OrderBlockMitigationRule,
    OrderBlockStatus,
    analyse_order_block_lifecycle,
    analyse_snapshot_order_block_lifecycle,
    touches,
)
from goldpipeline.services.ict_protected import analyse_protected_structure
from goldpipeline.services.ict_structure import BreakDirection, analyse_structure
from tests.test_ict_order_block import BODY, FULL, reopen
from tests.test_ict_order_block_fixture import OPENS
from tests.test_ict_order_block_fixture import journey as formation_journey
from tests.test_ict_order_block_lifecycle import FULL_ZONE, MIDPOINT, RULES, TOUCH
from tests.test_ict_protected import closing_bar
from tests.test_ict_range_fixture import JOURNEY, PATHS, ending_at
from tests.test_ict_structure import FLAT, START, Row, bar_index, series

HOUR = timedelta(hours=1)

TAIL: list[Row] = [
    FLAT,
    FLAT,
    ("3990", "3900", "3910"),
    ("3960", "3900", "3910"),
    ("3960", "3900", "3910"),
    ("3960", "3900", "3910"),
    ("3960", "3900", "3910"),
    ("4070", "4050", "4060"),
    ("4070", "4050", "4060"),
]
"""Bars 54-62.

Bar 56 breaks below the 3960 protected low, which is a bearish MSS and forms the
fifth block. Bars 57-60 stay beneath 3960, so their highs never reach the new
zone. Bars 61-62 gap to 4050-4070: above 4045 they cannot touch that zone, and
closing at 4060 they retire it - while staying under 4085, so the two bearish
blocks from bars 33 and 40 are left alone.

Written as adjacent pairs throughout, so no new strict pivot appears.
"""

ROWS: list[Row] = [*JOURNEY, *TAIL]


def journey() -> IctTimeframeSnapshot:
    return reopen(series(ROWS), OPENS)


def life(
    config: object = TOUCH,
    *,
    basis: OrderBlockZoneBasis = OrderBlockZoneBasis.FULL_CANDLE,
    as_of: datetime | None = None,
) -> OrderBlockLifecycleAnalysis:
    return analyse_order_block_lifecycle(
        journey(),
        config=config,  # type: ignore[arg-type]
        formation=FULL if basis is OrderBlockZoneBasis.FULL_CANDLE else BODY,
        symbol="XAUUSD",
        as_of=as_of,
    )


def table(
    analysis: OrderBlockLifecycleAnalysis,
) -> list[tuple[int, str, int | None, int | None, int | None]]:
    return [
        (
            closing_bar(state.order_block.formed_at),
            state.status.value,
            None if state.first_touched_at is None else closing_bar(state.first_touched_at),
            None if state.mitigated_at is None else closing_bar(state.mitigated_at),
            None if state.invalidated_at is None else closing_bar(state.invalidated_at),
        )
        for state in analysis.states
    ]


# --------------------------------------------------------------------------
# §52: the tail appended bars and nothing else
# --------------------------------------------------------------------------


def test_the_tail_moved_no_earlier_event_leg_or_order_block() -> None:
    """The claim the whole fixture rests on, proved rather than asserted."""
    base, extended = formation_journey(), journey()

    assert extended.bar_count == base.bar_count + len(TAIL) == 63
    assert [(b.timestamp, b.open, b.high, b.low, b.close) for b in extended.bars[:54]] == [
        (b.timestamp, b.open, b.high, b.low, b.close) for b in base.bars
    ]

    old_events = analyse_structure(base, symbol="XAUUSD").breaks
    new_events = analyse_structure(extended, symbol="XAUUSD").breaks
    assert [e.event_id for e in new_events[: len(old_events)]] == [e.event_id for e in old_events]
    assert len(new_events) == len(old_events) + 1, "exactly one new event"

    old_legs = analyse_protected_structure(base, symbol="XAUUSD").legs
    new_legs = analyse_protected_structure(extended, symbol="XAUUSD").legs
    assert new_legs[: len(old_legs)] == old_legs

    for basis, config in (
        (OrderBlockZoneBasis.FULL_CANDLE, FULL),
        (OrderBlockZoneBasis.BODY, BODY),
    ):
        old = analyse_order_blocks(base, config=config, symbol="XAUUSD").order_blocks
        new = analyse_order_blocks(extended, config=config, symbol="XAUUSD").order_blocks
        assert new[: len(old)] == old, basis
        assert len(new) == len(old) + 1


def test_the_new_event_is_a_bearish_mss_with_its_own_source_candle() -> None:
    extended = journey()
    structure = analyse_structure(extended, symbol="XAUUSD")
    newest = analyse_order_blocks(extended, config=FULL, symbol="XAUUSD").order_blocks[-1]

    assert closing_bar(structure.breaks[-1].break_bar_close_time) == 56
    assert structure.breaks[-1].classification.value == "MSS"
    assert newest.direction is BreakDirection.BEARISH
    assert bar_index(newest.source_bar_open_time) == 50
    assert (newest.lower, newest.upper, newest.midpoint) == (
        Decimal("3990"),
        Decimal("4045"),
        Decimal("4017.5"),
    )


# --------------------------------------------------------------------------
# §52: the final state tables, one per mitigation rule
# --------------------------------------------------------------------------


def test_the_full_candle_touch_table() -> None:
    assert table(life(TOUCH)) == [
        (11, "INVALIDATED", 12, 12, 40),
        (18, "INVALIDATED", 19, 19, 33),
        (33, "MITIGATED", 34, 34, None),
        (40, "MITIGATED", 41, 41, None),
        (56, "INVALIDATED", None, None, 61),
    ]


def test_the_full_candle_midpoint_table() -> None:
    assert table(life(MIDPOINT)) == [
        (11, "INVALIDATED", 12, 12, 40),
        (18, "INVALIDATED", 19, 19, 33),
        (33, "MITIGATED", 34, 50, None),
        (40, "MITIGATED", 41, 50, None),
        (56, "INVALIDATED", None, None, 61),
    ]


def test_the_full_candle_full_zone_table() -> None:
    assert table(life(FULL_ZONE)) == [
        (11, "INVALIDATED", 12, 33, 40),
        (18, "INVALIDATED", 19, 19, 33),
        (33, "TOUCHED", 34, None, None),
        (40, "TOUCHED", 41, None, None),
        (56, "INVALIDATED", None, None, 61),
    ]


def test_the_body_touch_table() -> None:
    assert table(life(TOUCH, basis=OrderBlockZoneBasis.BODY)) == [
        (11, "INVALIDATED", 12, 12, 33),
        (18, "INVALIDATED", 19, 19, 33),
        (33, "MITIGATED", 34, 34, None),
        (40, "MITIGATED", 41, 41, None),
        (56, "INVALIDATED", None, None, 61),
    ]


def test_the_body_midpoint_table() -> None:
    assert table(life(MIDPOINT, basis=OrderBlockZoneBasis.BODY)) == [
        (11, "INVALIDATED", 12, 12, 33),
        (18, "INVALIDATED", 19, 19, 33),
        (33, "MITIGATED", 34, 50, None),
        (40, "MITIGATED", 41, 50, None),
        (56, "INVALIDATED", None, None, 61),
    ]


def test_the_body_full_zone_table() -> None:
    assert table(life(FULL_ZONE, basis=OrderBlockZoneBasis.BODY)) == [
        (11, "INVALIDATED", 12, 12, 33),
        (18, "INVALIDATED", 19, 19, 33),
        (33, "TOUCHED", 34, None, None),
        (40, "TOUCHED", 41, None, None),
        (56, "INVALIDATED", None, None, 61),
    ]


def test_the_id_groups_agree_with_the_states() -> None:
    """§34. The four lists are a view of the same evidence, not a second record."""
    for config in (TOUCH, MIDPOINT, FULL_ZONE):
        for basis in (OrderBlockZoneBasis.FULL_CANDLE, OrderBlockZoneBasis.BODY):
            analysis = life(config, basis=basis)
            for status, ids in (
                (OrderBlockStatus.ACTIVE, analysis.active_ids),
                (OrderBlockStatus.TOUCHED, analysis.touched_ids),
                (OrderBlockStatus.MITIGATED, analysis.mitigated_ids),
                (OrderBlockStatus.INVALIDATED, analysis.invalidated_ids),
            ):
                assert ids == tuple(s.order_block_id for s in analysis.states if s.status is status)
            total = (
                len(analysis.active_ids)
                + len(analysis.touched_ids)
                + len(analysis.mitigated_ids)
                + len(analysis.invalidated_ids)
            )
            assert total == len(analysis.states) == 5


# --------------------------------------------------------------------------
# §52: the states the tail exists for
# --------------------------------------------------------------------------


def test_the_fifth_block_stays_active_for_four_bars() -> None:
    """§52. An ACTIVE block that is genuinely still ACTIVE, bar after bar.

    Under every rule and both bases: formed at bar 56, untouched through bar 60.
    """
    for config in (TOUCH, MIDPOINT, FULL_ZONE):
        for basis in (OrderBlockZoneBasis.FULL_CANDLE, OrderBlockZoneBasis.BODY):
            for kept in range(57, 62):
                analysis = life(config, basis=basis, as_of=START + HOUR * kept)
                newest = analysis.states[-1]
                assert closing_bar(newest.order_block.formed_at) == 56
                assert newest.status is OrderBlockStatus.ACTIVE, (config, basis, kept)
                assert newest.first_touched_at is None
                assert analysis.active_ids == (newest.order_block_id,)


def test_the_formation_bar_overlaps_its_own_zone_and_is_still_not_a_touch() -> None:
    """§9, §10, and not a hypothetical this time.

    Bar 56 runs 3900-3990 and the zone it creates starts at 3990, so the bar's
    high rests exactly on the lower edge - a touch by every rule in this module,
    if it were allowed to be one. It is the break bar, so it is not.
    """
    extended = journey()
    newest = analyse_order_blocks(extended, config=FULL, symbol="XAUUSD").order_blocks[-1]
    breaking = extended.bars[56]

    assert breaking.high == Decimal("3990") == newest.lower
    assert touches(newest, breaking.low, breaking.high) is True, "geometrically it does"
    assert life(TOUCH).states[-1].first_touched_at is None, "and it still is not a touch"


def test_the_fifth_block_is_invalidated_without_ever_being_touched() -> None:
    """§21, §46. The gap at bar 61: above the zone throughout, closing above it."""
    extended = journey()
    gap = extended.bars[61]
    newest = analyse_order_blocks(extended, config=FULL, symbol="XAUUSD").order_blocks[-1]

    assert gap.low == Decimal("4050") > newest.upper == Decimal("4045")
    assert touches(newest, gap.low, gap.high) is False, "it never traded in the zone"
    assert gap.close == Decimal("4060") > newest.upper

    for config in (TOUCH, MIDPOINT, FULL_ZONE):
        state = life(config).states[-1]
        assert state.status is OrderBlockStatus.INVALIDATED
        assert state.first_touch_witness is None
        assert state.mitigation_witness is None
        assert state.invalidation_witness is not None
        assert closing_bar(state.invalidated_at) == 61  # type: ignore[arg-type]


def test_a_touched_block_waits_sixteen_bars_for_its_midpoint() -> None:
    """§52. TOUCHED but not MIDPOINT-mitigated, held across a long stretch."""
    for kept in range(35, 51):
        analysis = life(MIDPOINT, as_of=START + HOUR * kept)
        third = analysis.states[2]
        assert third.status is OrderBlockStatus.TOUCHED, kept
        assert third.first_touched_at is not None
        assert third.mitigated_at is None

    after = life(MIDPOINT, as_of=START + HOUR * 51)
    assert after.states[2].status is OrderBlockStatus.MITIGATED
    assert closing_bar(after.states[2].mitigated_at) == 50  # type: ignore[arg-type]


def test_a_full_zone_mitigation_lands_on_its_own_bar() -> None:
    """§52. The first block is covered edge to edge at bar 33, not at first touch."""
    state = life(FULL_ZONE).states[0]
    covering = journey().bars[33]
    block = state.order_block

    assert closing_bar(state.first_touched_at) == 12  # type: ignore[arg-type]
    assert closing_bar(state.mitigated_at) == 33  # type: ignore[arg-type]
    assert covering.low <= block.lower and covering.high >= block.upper


def test_two_blocks_sharing_one_source_candle_evolve_independently() -> None:
    """§29, §50. Same candle, same zone, different events - and separate states."""
    for config in (TOUCH, MIDPOINT, FULL_ZONE):
        analysis = life(config)
        third, fourth = analysis.states[2], analysis.states[3]

        assert third.order_block.source_bar_open_time == fourth.order_block.source_bar_open_time
        assert (third.order_block.lower, third.order_block.upper) == (
            fourth.order_block.lower,
            fourth.order_block.upper,
        )
        assert third.order_block_id != fourth.order_block_id
        assert third.first_touched_at != fourth.first_touched_at, "each has its own first touch"
        assert third.status is fourth.status


def test_one_bar_changes_two_blocks_at_once() -> None:
    """§50. Bar 33 covers the first block and retires the second, in one candle.

    The two were in different states going in - the first only touched, the
    second already covered at bar 19 - and one candle moved both.
    """
    before = life(FULL_ZONE, as_of=START + HOUR * 33)
    after = life(FULL_ZONE, as_of=START + HOUR * 34)

    assert [s.status.value for s in before.states[:2]] == ["TOUCHED", "MITIGATED"]
    assert [s.status.value for s in after.states[:2]] == ["MITIGATED", "INVALIDATED"]
    assert closing_bar(after.states[0].mitigated_at) == 33  # type: ignore[arg-type]
    assert closing_bar(after.states[1].invalidated_at) == 33  # type: ignore[arg-type]


def test_a_block_born_on_that_same_bar_is_untouched_by_it() -> None:
    """§50, §10. Three blocks, one candle, and it does not reach the newest.

    Bar 33 is the break bar of the third block, so for that block it is
    formation rather than interaction - while for the two older ones it is the
    candle that changed both of them.
    """
    after = life(FULL_ZONE, as_of=START + HOUR * 34)

    assert len(after.states) == 3
    newest = after.states[2]
    assert closing_bar(newest.order_block.formed_at) == 33
    assert newest.status is OrderBlockStatus.ACTIVE
    assert newest.first_touched_at is None
    assert after.active_ids == (newest.order_block_id,)


# --------------------------------------------------------------------------
# §31, §49: the zone basis is load-bearing
# --------------------------------------------------------------------------


def test_the_body_block_is_retired_while_the_full_candle_block_lives_on() -> None:
    """§31. The same formation, two policies, two different answers.

    The first block's BODY zone bottoms at 4000 and its FULL_CANDLE zone at
    3970. Bar 33 closes at 3975: below the body, above the wick. For seven bars
    one reading says the block is finished and the other says it is not, and
    both are right about their own zone.
    """
    for kept in range(34, 41):
        body = life(TOUCH, basis=OrderBlockZoneBasis.BODY, as_of=START + HOUR * kept).states[0]
        full = life(TOUCH, as_of=START + HOUR * kept).states[0]

        assert body.status is OrderBlockStatus.INVALIDATED, kept
        assert full.status is not OrderBlockStatus.INVALIDATED, kept

    assert journey().bars[33].close == Decimal("3975")


def test_the_two_readings_have_different_zones_and_the_same_events() -> None:
    """§3, §31. Lifecycle never changes an identity or a zone it was handed."""
    body = life(TOUCH, basis=OrderBlockZoneBasis.BODY)
    full = life(TOUCH)

    for narrow, wide in zip(body.states, full.states, strict=True):
        assert narrow.order_block.structure_event_id == wide.order_block.structure_event_id
        assert narrow.order_block_id != wide.order_block_id
        assert (narrow.order_block.lower, narrow.order_block.upper) != (
            wide.order_block.lower,
            wide.order_block.upper,
        )
        assert narrow.order_block_id == narrow.order_block.order_block_id


def test_a_midpoint_result_differs_between_the_two_bases() -> None:
    """§49. Different midpoints, so different mitigation instants are possible."""
    body = life(MIDPOINT, basis=OrderBlockZoneBasis.BODY).states[0]
    full = life(MIDPOINT).states[0]

    assert body.order_block.midpoint == Decimal("4005")
    assert full.order_block.midpoint == Decimal("3990")
    assert body.order_block.midpoint != full.order_block.midpoint


# --------------------------------------------------------------------------
# §35: ordering
# --------------------------------------------------------------------------


def test_states_follow_formation_order_whatever_their_status() -> None:
    """§35. Lifecycle never reorders, and a retired block keeps its place."""
    for config in (TOUCH, MIDPOINT, FULL_ZONE):
        analysis = life(config)
        formed = [s.order_block.formed_at for s in analysis.states]

        assert formed == sorted(formed)
        assert [s.order_block_id for s in analysis.states] == [
            b.order_block_id
            for b in analyse_order_blocks(journey(), config=FULL, symbol="XAUUSD").order_blocks
        ]


def test_every_state_is_reachable_by_its_identity() -> None:
    analysis = life(TOUCH)

    for state in analysis.states:
        assert analysis.state(state.order_block_id) is state
    assert analysis.state("nope") is None


# --------------------------------------------------------------------------
# §53: five timeframes, five independent answers
# --------------------------------------------------------------------------

OBSERVED_AT = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)

TF_OPENS: dict[Timeframe, dict[int, str]] = {
    Timeframe.H1: OPENS,
    Timeframe.M5: {7: "4010"},
    Timeframe.H4: {7: "3990"},
}


def divergent_snapshot() -> IctMarketSnapshot:
    return IctMarketSnapshot(
        observed_at=OBSERVED_AT,
        symbol="XAUUSD",
        provider="fixture",
        provider_symbol=None,
        timeframes=tuple(
            reopen(ending_at(tf, PATHS[tf], end=OBSERVED_AT), TF_OPENS.get(tf, {}))
            for tf in ICT_TIMEFRAMES
        ),
    )


@pytest.mark.parametrize("rule", RULES)
def test_five_timeframes_reach_their_own_answers(rule: OrderBlockMitigationRule) -> None:
    from goldpipeline.services.ict_order_block_lifecycle import OrderBlockLifecycleConfig

    shot = divergent_snapshot()
    counts = {
        tf: len(
            analyse_snapshot_order_block_lifecycle(
                shot, tf, config=OrderBlockLifecycleConfig(mitigation_rule=rule), formation=FULL
            ).states
        )
        for tf in ICT_TIMEFRAMES
    }

    assert counts == {
        Timeframe.H4: 1,
        Timeframe.H1: 4,
        Timeframe.M15: 0,
        Timeframe.M5: 1,
        Timeframe.M1: 0,
    }


def test_no_timeframe_lifecycle_touches_another() -> None:
    """§53. An H4 block's state says nothing about an M5 block."""
    shot = divergent_snapshot()

    h4 = analyse_snapshot_order_block_lifecycle(shot, Timeframe.H4, config=TOUCH, formation=FULL)
    m5 = analyse_snapshot_order_block_lifecycle(shot, Timeframe.M5, config=TOUCH, formation=FULL)

    assert {s.order_block.timeframe for s in h4.states} == {Timeframe.H4}
    assert {s.order_block.timeframe for s in m5.states} == {Timeframe.M5}
    assert {s.order_block_id for s in h4.states}.isdisjoint(s.order_block_id for s in m5.states)


def test_analysing_one_timeframe_does_not_touch_another() -> None:
    shot = divergent_snapshot()

    alone = analyse_order_block_lifecycle(
        shot.require(Timeframe.M5), config=TOUCH, formation=FULL, symbol="XAUUSD"
    )
    after_the_others = [
        analyse_snapshot_order_block_lifecycle(shot, tf, config=TOUCH, formation=FULL)
        for tf in ICT_TIMEFRAMES
    ]

    assert alone == next(r for r in after_the_others if r.timeframe is Timeframe.M5)


def test_a_timeframe_with_no_blocks_has_no_states() -> None:
    shot = divergent_snapshot()

    quiet = analyse_snapshot_order_block_lifecycle(
        shot, Timeframe.M15, config=TOUCH, formation=FULL
    )

    assert quiet.states == ()
    assert quiet.active_ids == ()
