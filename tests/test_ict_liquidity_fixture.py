"""One long synthetic path through every pool state, and five timeframes that differ.

Round 6.6c.1 §49-§51. The matrices in ``test_ict_liquidity.py`` check one rule
at a time. This file pins a single 54-bar series that contains every shape the
engine can produce, so a later round changing a definition has to change these
numbers deliberately.

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
from goldpipeline.services.ict_liquidity import (
    LiquidityEventType,
    LiquiditySide,
    PoolStatus,
    analyse_liquidity,
    analyse_snapshot_liquidity,
)
from tests.test_ict_liquidity import (
    FLAT,
    Row,
    analyse,
    bar,
    bar_index,
    config,
    peak,
    series,
    trough,
)

TOLERANCE = "0.50"

JOURNEY: list[Row] = [
    FLAT,  # 0
    FLAT,  # 1
    peak("4060"),  # 2   P1 founder
    FLAT,  # 3
    FLAT,  # 4
    FLAT,  # 5
    peak("4060"),  # 6   P1 - exact-equality buy-side pool
    FLAT,  # 7
    FLAT,  # 8
    FLAT,  # 9
    peak("4030.00"),  # 10  P2 founder
    FLAT,  # 11
    FLAT,  # 12
    FLAT,  # 13
    peak("4030.30"),  # 14  P2 - near-equality buy-side pool
    FLAT,  # 15
    FLAT,  # 16
    bar("4030.30", "3990", "4005"),  # 17  exact touch of P2's edge: no event
    bar("4030.30", "3990", "4005"),  # 18  paired, so neither bar is a pivot
    FLAT,  # 19
    bar("4040", "3990", "4035"),  # 20  CLOSE_THROUGH of P2
    FLAT,  # 21
    FLAT,  # 22
    FLAT,  # 23
    peak("4030.00"),  # 24  P3 founder, at the dead P2's price
    FLAT,  # 25
    FLAT,  # 26
    FLAT,  # 27
    peak("4030.00"),  # 28  P3 - a new pool, a new identity
    FLAT,  # 29
    FLAT,  # 30
    FLAT,  # 31
    trough("3950"),  # 32  S1 founder
    FLAT,  # 33
    FLAT,  # 34
    FLAT,  # 35
    trough("3950"),  # 36  S1 - exact-equality sell-side pool
    FLAT,  # 37
    FLAT,  # 38
    FLAT,  # 39
    trough("3975.00"),  # 40  S2 founder
    FLAT,  # 41
    FLAT,  # 42
    FLAT,  # 43
    trough("3974.70"),  # 44  S2 - near-equality sell-side pool
    FLAT,  # 45
    FLAT,  # 46
    FLAT,  # 47
    bar("4065", "3945", "4000"),  # 48  one bar clears P1, P3, S1 and S2
    FLAT,  # 49
    FLAT,  # 50
    peak("4100"),  # 51  never finds a partner
    FLAT,  # 52
    FLAT,  # 53
]
"""Built so the higher band always forms first.

That ordering is forced, not stylistic: a swing high's price is its own bar's
high, so any bar printing a new high above an existing buy-side pool sweeps it
on the way past. A fixture that built the low pool first could never get a
second, higher pool to coexist with it.
"""


def journey() -> IctTimeframeSnapshot:
    return series(JOURNEY)


def result() -> object:
    return analyse(JOURNEY, TOLERANCE)


# --------------------------------------------------------------------------
# §49: what the journey contains
# --------------------------------------------------------------------------


def test_the_journey_builds_five_pools_on_both_sides() -> None:
    pools = analyse(JOURNEY, TOLERANCE).pools

    assert [
        (pool.side.value, str(pool.lower), str(pool.upper), str(pool.midpoint)) for pool in pools
    ] == [
        ("BUY_SIDE", "4060", "4060", "4060"),
        ("BUY_SIDE", "4030.00", "4030.30", "4030.15"),
        ("BUY_SIDE", "4030.00", "4030.00", "4030.00"),
        ("SELL_SIDE", "3950", "3950", "3950"),
        ("SELL_SIDE", "3974.70", "3975.00", "3974.85"),
    ]


def test_the_journey_has_both_exact_and_near_equality_pools() -> None:
    pools = analyse(JOURNEY, TOLERANCE).pools
    spans = {str(pool.span) for pool in pools}

    assert "0" in spans, "at least one pool is exact equality"
    assert {"0.30"} <= spans, "and at least one is near equality"


def test_the_journey_pins_when_each_pool_formed() -> None:
    """The later founding confirmation, never a pivot time."""
    pools = analyse(JOURNEY, TOLERANCE).pools

    assert [bar_index(pool.formed_at) for pool in pools] == [8, 16, 30, 38, 46]


def test_the_journey_pins_every_event() -> None:
    events = analyse(JOURNEY, TOLERANCE).events

    assert [
        (
            bar_index(event.event_bar_close_time),
            event.event_type.value,
            event.side.value,
            str(event.boundary),
            str(event.close),
        )
        for event in events
    ] == [
        (20, "CLOSE_THROUGH", "BUY_SIDE", "4030.30", "4035"),
        (48, "WICK_SWEEP", "BUY_SIDE", "4030.00", "4000"),
        (48, "WICK_SWEEP", "BUY_SIDE", "4060", "4000"),
        (48, "WICK_SWEEP", "SELL_SIDE", "3950", "4000"),
        (48, "WICK_SWEEP", "SELL_SIDE", "3974.70", "4000"),
    ]


def test_the_exact_touch_at_bar_seventeen_produced_nothing() -> None:
    events = analyse(JOURNEY, TOLERANCE).events

    assert not any(bar_index(event.event_bar_close_time) in {17, 18} for event in events)


def test_one_bar_cleared_four_pools_on_both_sides() -> None:
    """§26. The structure engine's one-event-per-bar rule is wrong here.

    Structure asks what a close said about the trend, and that has one answer.
    Liquidity asks what a candle took, and a single wide bar genuinely runs
    several bands - here two above and two below, in one outside candle.
    """
    events = analyse(JOURNEY, TOLERANCE).events
    same_bar = [event for event in events if bar_index(event.event_bar_close_time) == 48]

    assert len(same_bar) == 4
    assert {event.side for event in same_bar} == {
        LiquiditySide.BUY_SIDE,
        LiquiditySide.SELL_SIDE,
    }
    assert len({event.pool_id for event in same_bar}) == 4


def test_events_within_one_bar_are_ordered_by_side_then_boundary() -> None:
    """§27. Any total order derived only from the data would do; this is the one."""
    events = analyse(JOURNEY, TOLERANCE).events
    same_bar = [event for event in events if bar_index(event.event_bar_close_time) == 48]

    keys = [(event.side.value, event.boundary) for event in same_bar]
    assert keys == sorted(keys)


def test_a_new_pool_formed_at_the_dead_pool_s_price() -> None:
    """§15. The band that was closed through is finished; this is fresh liquidity."""
    pools = analyse(JOURNEY, TOLERANCE).pools
    dead = pools[1]
    fresh = pools[2]

    assert dead.status is PoolStatus.CLOSED_THROUGH
    assert dead.lower == fresh.lower == Decimal("4030.00")
    assert dead.pool_id != fresh.pool_id
    assert set(dead.member_swing_ids).isdisjoint(fresh.member_swing_ids)


def test_the_journey_ends_with_every_pool_terminal_and_swings_still_waiting() -> None:
    analysis = analyse(JOURNEY, TOLERANCE)

    assert analysis.active_pool_ids == ()
    assert len(analysis.terminal_pool_ids) == 5
    assert len(analysis.unpaired_swing_ids) == 4


def test_every_status_the_engine_can_reach_appears() -> None:
    statuses = {pool.status for pool in analyse(JOURNEY, TOLERANCE).pools}
    kinds = {event.event_type for event in analyse(JOURNEY, TOLERANCE).events}

    assert statuses == {PoolStatus.SWEPT, PoolStatus.CLOSED_THROUGH}
    assert kinds == set(LiquidityEventType)


def test_each_terminal_pool_names_the_event_that_ended_it() -> None:
    analysis = analyse(JOURNEY, TOLERANCE)
    by_id = {event.pool_id: event.event_id for event in analysis.events}

    for pool in analysis.pools:
        assert pool.terminal_event_id == by_id[pool.pool_id]


def test_no_pool_in_the_journey_exceeds_the_tolerance() -> None:
    for pool in analyse(JOURNEY, TOLERANCE).pools:
        assert pool.span <= Decimal(TOLERANCE)


def test_every_swing_is_in_at_most_one_pool_or_waiting() -> None:
    analysis = analyse(JOURNEY, TOLERANCE)

    members: list[str] = []
    for pool in analysis.pools:
        members.extend(pool.member_swing_ids)

    assert len(members) == len(set(members))
    assert set(members).isdisjoint(analysis.unpaired_swing_ids)


# --------------------------------------------------------------------------
# §50: five timeframes, five independent answers
# --------------------------------------------------------------------------

OBSERVED_AT = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)

BUY_ONLY: list[Row] = [
    FLAT, FLAT, peak("4030"), FLAT, FLAT, FLAT, peak("4030"), FLAT, FLAT,
]  # fmt: skip
SELL_SWEPT: list[Row] = [
    FLAT, FLAT, trough("3970"), FLAT, FLAT, FLAT, trough("3970"), FLAT, FLAT,
    bar("4010", "3965", "3995"), FLAT,
]  # fmt: skip
NOTHING: list[Row] = [
    FLAT, FLAT, peak("4030"), FLAT, FLAT, FLAT, peak("4080"), FLAT, FLAT,
]  # fmt: skip

PATHS: dict[Timeframe, list[Row]] = {
    Timeframe.H4: SELL_SWEPT,
    Timeframe.H1: JOURNEY,
    Timeframe.M15: NOTHING,
    Timeframe.M5: BUY_ONLY,
    Timeframe.M1: NOTHING,
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
            len(analyse_snapshot_liquidity(shot, tf, config=config(TOLERANCE)).pools),
            len(analyse_snapshot_liquidity(shot, tf, config=config(TOLERANCE)).events),
        )
        for tf in ICT_TIMEFRAMES
    }

    assert summary == {
        Timeframe.H4: (1, 1),
        Timeframe.H1: (5, 5),
        Timeframe.M15: (0, 0),
        Timeframe.M5: (1, 0),
        Timeframe.M1: (0, 0),
    }


def test_nothing_merges_pools_across_timeframes() -> None:
    """§40. Confluence is a scoring rule with content; it is not assumed here."""
    shot = divergent_snapshot()

    h1 = analyse_snapshot_liquidity(shot, Timeframe.H1, config=config(TOLERANCE))
    m5 = analyse_snapshot_liquidity(shot, Timeframe.M5, config=config(TOLERANCE))

    assert {pool.timeframe for pool in h1.pools} == {Timeframe.H1}
    assert {pool.timeframe for pool in m5.pools} == {Timeframe.M5}
    assert set(h1.active_pool_ids).isdisjoint(m5.active_pool_ids)


def test_analysing_one_timeframe_does_not_touch_another() -> None:
    shot = divergent_snapshot()

    alone = analyse_liquidity(shot.require(Timeframe.M5), config=config(TOLERANCE), symbol="XAUUSD")
    after_the_others = [
        analyse_snapshot_liquidity(shot, tf, config=config(TOLERANCE)) for tf in ICT_TIMEFRAMES
    ]

    assert alone == next(r for r in after_the_others if r.timeframe is Timeframe.M5)


def test_the_same_path_on_two_timeframes_finds_the_same_shape() -> None:
    """Independence is not indifference: bar duration must not change geometry."""
    shot = divergent_snapshot()

    m15 = analyse_snapshot_liquidity(shot, Timeframe.M15, config=config(TOLERANCE))
    m1 = analyse_snapshot_liquidity(shot, Timeframe.M1, config=config(TOLERANCE))

    assert len(m15.pools) == len(m1.pools)
    assert len(m15.unpaired_swing_ids) == len(m1.unpaired_swing_ids)


def test_a_timeframe_stamps_its_own_name_on_its_swing_identities() -> None:
    shot = divergent_snapshot()

    h4 = analyse_snapshot_liquidity(shot, Timeframe.H4, config=config(TOLERANCE))

    for pool in h4.pools:
        assert all(identity.startswith("H4:") for identity in pool.member_swing_ids)


# --------------------------------------------------------------------------
# §51: H4 policy
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

    native_series = ending_at(Timeframe.H4, SELL_SWEPT, end=native_end)
    assert all(candle.timestamp.hour % 4 == 1 for candle in native_series.bars)

    native = analyse_liquidity(native_series, config=config(TOLERANCE), symbol="XAUUSD")
    shifted = analyse_liquidity(
        ending_at(Timeframe.H4, SELL_SWEPT, end=shifted_end),
        config=config(TOLERANCE),
        symbol="XAUUSD",
    )

    assert [(pool.lower, pool.upper, pool.status) for pool in native.pools] == [
        (pool.lower, pool.upper, pool.status) for pool in shifted.pools
    ]
    assert [event.event_type for event in native.events] == [
        event.event_type for event in shifted.events
    ]
