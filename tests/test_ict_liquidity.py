"""Pool formation, and what a candle does to a band.

Round 6.6c.1 §44-§45. Small hand-checkable fixtures, one rule each. The long
realistic path is in ``test_ict_liquidity_fixture.py``.

The bar vocabulary is the one Round 6.6b established: every filler bar shares a
high and a low with its neighbours, so it can never be a pivot under the
strict-both-sides rule, and a swing appears exactly where one bar is written to
stick out. Prices are gold-shaped and mean nothing.

One property of this engine shapes almost every fixture below and is worth
stating once: **a swing high's price is its own bar's high.** So a new high
above an existing buy-side pool's boundary does not join that pool - the bar
printing it swept the pool as it printed. A third member can only arrive from
*inside* the band. Lows mirror it.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import IctTimeframeSnapshot, build_timeframe_snapshot
from goldpipeline.schemas.market import OHLCBar
from goldpipeline.services.ict_liquidity import (
    LiquidityAnalysis,
    LiquidityConfig,
    LiquidityEventType,
    LiquidityPool,
    LiquiditySide,
    PoolStatus,
    _best_pool_for,
    _eligible_pools,
    _WorkingPool,
    analyse_liquidity,
    side_of,
    touchable,
)
from goldpipeline.services.ict_primitives import SwingPoint, SwingType

START = datetime(2026, 9, 1, tzinfo=UTC)
HOUR = timedelta(hours=1)

Row = tuple[str, str, str]
"""One bar as ``(high, low, close)``. The open is derived - see :func:`series`."""

FLAT: Row = ("4010", "3990", "4000")


def peak(price: str) -> Row:
    """A bar sticking up to *price*. Its low matches the filler."""
    return (price, "3990", "4000")


def trough(price: str) -> Row:
    return ("4010", price, "4000")


def bar(high: str, low: str, close: str) -> Row:
    return (high, low, close)


def series(
    rows: Sequence[Row],
    *,
    timeframe: Timeframe = Timeframe.H1,
    bars_kept: int | None = None,
    start: datetime = START,
) -> IctTimeframeSnapshot:
    """Lay *rows* onto consecutive bars. The open follows the previous close."""
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
                timestamp=start + duration * index,
                open=opening,
                high=top,
                low=bottom,
                close=last,
            )
        )
        previous = last

    return build_timeframe_snapshot(
        timeframe=timeframe, bars=tuple(bars), observed_at=start + duration * len(kept)
    )


def config(tolerance: str) -> LiquidityConfig:
    return LiquidityConfig(price_tolerance=Decimal(tolerance))


def analyse(rows: Sequence[Row], tolerance: str = "0.50", **kwargs: object) -> LiquidityAnalysis:
    return analyse_liquidity(
        series(rows),
        config=config(tolerance),
        symbol="XAUUSD",
        **kwargs,  # type: ignore[arg-type]
    )


def bands(rows: Sequence[Row], tolerance: str = "0.50") -> list[tuple[str, str, str]]:
    return [
        (pool.side.value, str(pool.lower), str(pool.upper))
        for pool in analyse(rows, tolerance).pools
    ]


def bar_index(moment: datetime, timeframe: Timeframe = Timeframe.H1) -> int:
    duration = timeframe.duration
    assert duration is not None
    return int((moment - START) // duration) - 1


# Two swings at one price, six bars apart. The workhorse pool for event tests.
POOL_HIGHS: list[Row] = [FLAT, FLAT, peak("4030"), FLAT, FLAT, FLAT, peak("4030"), FLAT, FLAT]
POOL_LOWS: list[Row] = [FLAT, FLAT, trough("3970"), FLAT, FLAT, FLAT, trough("3970"), FLAT, FLAT]


# --------------------------------------------------------------------------
# §4-§5: the tolerance policy
# --------------------------------------------------------------------------


def test_the_tolerance_has_no_default() -> None:
    """The engine cannot pick this number, so it does not try.

    A baked-in 0.5 for XAUUSD would be a tuning choice wearing the costume of a
    definition, and it would end up underneath a published price level with
    nobody able to say who chose it.
    """
    with pytest.raises(TypeError):
        LiquidityConfig()  # type: ignore[call-arg]


def test_zero_tolerance_is_valid() -> None:
    assert config("0").price_tolerance == Decimal(0)


def test_a_negative_tolerance_is_refused() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        config("-0.01")


def test_a_non_finite_tolerance_is_refused() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        LiquidityConfig(price_tolerance=Decimal("NaN"))
    with pytest.raises(ValueError, match="must be finite"):
        LiquidityConfig(price_tolerance=Decimal("Infinity"))


def test_a_binary_float_tolerance_is_refused_by_type() -> None:
    """0.1 + 0.2 != 0.3 is not an acceptable way to decide pool membership."""
    with pytest.raises(TypeError, match="must be a Decimal"):
        LiquidityConfig(price_tolerance=0.5)  # type: ignore[arg-type]


def test_the_same_policy_written_two_ways_is_one_policy() -> None:
    """``0.50`` and ``0.5`` must not produce two pools differing in trailing zeros."""
    assert config("0.50").canonical_tolerance == config("0.5").canonical_tolerance == "0.5"
    assert config("0").canonical_tolerance == "0"

    one = analyse(POOL_HIGHS, "0.50").pools[0].pool_id
    two = analyse(POOL_HIGHS, "0.5").pools[0].pool_id
    assert one == two


def test_no_price_in_the_analysis_is_a_float() -> None:
    result = analyse(POOL_HIGHS, "0")
    pool = result.pools[0]

    for value in (pool.lower, pool.upper, pool.midpoint, pool.tolerance, pool.span):
        assert isinstance(value, Decimal)


# --------------------------------------------------------------------------
# §3: sides
# --------------------------------------------------------------------------


def test_a_swing_high_carries_buy_side_liquidity() -> None:
    assert side_of(SwingType.SWING_HIGH) is LiquiditySide.BUY_SIDE
    assert side_of(SwingType.SWING_LOW) is LiquiditySide.SELL_SIDE


def test_the_side_enum_is_not_a_trade_instruction() -> None:
    """Buy-side liquidity is a place to sell into as often as not."""
    assert {member.value for member in LiquiditySide} == {"BUY_SIDE", "SELL_SIDE"}


# --------------------------------------------------------------------------
# §44: pool formation
# --------------------------------------------------------------------------


def test_exact_equal_highs_form_a_buy_side_pool() -> None:
    assert bands(POOL_HIGHS, "0") == [("BUY_SIDE", "4030", "4030")]


def test_exact_equal_lows_form_a_sell_side_pool() -> None:
    assert bands(POOL_LOWS, "0") == [("SELL_SIDE", "3970", "3970")]


def test_near_highs_inside_tolerance_form_a_pool() -> None:
    rows = [FLAT, FLAT, peak("4030.00"), FLAT, FLAT, FLAT, peak("4030.30"), FLAT, FLAT]

    assert bands(rows) == [("BUY_SIDE", "4030.00", "4030.30")]


def test_near_lows_inside_tolerance_form_a_pool() -> None:
    rows = [FLAT, FLAT, trough("3970.00"), FLAT, FLAT, FLAT, trough("3969.70"), FLAT, FLAT]

    assert bands(rows) == [("SELL_SIDE", "3969.70", "3970.00")]


def test_swings_outside_tolerance_form_nothing() -> None:
    rows = [FLAT, FLAT, peak("4030.00"), FLAT, FLAT, FLAT, peak("4030.80"), FLAT, FLAT]
    result = analyse(rows)

    assert result.pools == ()
    assert len(result.unpaired_swing_ids) == 2


def test_a_span_exactly_equal_to_the_tolerance_forms_a_pool() -> None:
    """``<=``, not ``<``. The boundary case is a decision, so it is pinned."""
    rows = [FLAT, FLAT, peak("4030.00"), FLAT, FLAT, FLAT, peak("4030.50"), FLAT, FLAT]

    assert bands(rows, "0.50") == [("BUY_SIDE", "4030.00", "4030.50")]


def test_zero_tolerance_accepts_only_exact_equality() -> None:
    exact = [FLAT, FLAT, peak("4030.00"), FLAT, FLAT, FLAT, peak("4030.00"), FLAT, FLAT]
    near = [FLAT, FLAT, peak("4030.00"), FLAT, FLAT, FLAT, peak("4030.01"), FLAT, FLAT]

    assert len(analyse(exact, "0").pools) == 1
    assert analyse(near, "0").pools == ()


def test_pool_members_need_not_be_consecutive_swings() -> None:
    """§14. An unrelated high in between says nothing about orders resting below it."""
    rows = [
        FLAT, FLAT, peak("4030.00"), FLAT, FLAT, FLAT, peak("4080"), FLAT, FLAT, FLAT,
        peak("4030.20"), FLAT, FLAT,
    ]  # fmt: skip
    result = analyse(rows)

    assert bands(rows) == [("BUY_SIDE", "4030.00", "4030.20")]
    assert len(result.unpaired_swing_ids) == 1, "the 4080 is still waiting"


def test_a_high_and_a_low_at_one_price_never_pair() -> None:
    rows = [FLAT, FLAT, peak("4000"), FLAT, FLAT, FLAT, trough("4000"), FLAT, FLAT]
    result = analyse(rows, "0")

    assert result.pools == ()


def test_a_third_member_joins_from_inside_the_band() -> None:
    rows = [
        FLAT, FLAT, peak("4030.00"), FLAT, FLAT, FLAT, peak("4030.40"), FLAT, FLAT, FLAT,
        peak("4030.20"), FLAT, FLAT,
    ]  # fmt: skip
    (pool,) = analyse(rows).pools

    assert len(pool.member_swing_ids) == 3
    assert (pool.lower, pool.upper) == (Decimal("4030.00"), Decimal("4030.40"))
    assert pool.status is PoolStatus.ACTIVE


def test_a_third_member_joins_a_sell_side_pool_from_inside() -> None:
    rows = [
        FLAT, FLAT, trough("3970.00"), FLAT, FLAT, FLAT, trough("3969.60"), FLAT, FLAT, FLAT,
        trough("3969.80"), FLAT, FLAT,
    ]  # fmt: skip
    (pool,) = analyse(rows).pools

    assert len(pool.member_swing_ids) == 3
    assert (pool.lower, pool.upper) == (Decimal("3969.60"), Decimal("3970.00"))


def test_a_third_member_is_refused_when_the_full_span_would_exceed_tolerance() -> None:
    """Rejected on span alone - the candidate sits below the boundary, so it swept nothing."""
    rows = [
        FLAT, FLAT, peak("4030.00"), FLAT, FLAT, FLAT, peak("4030.40"), FLAT, FLAT, FLAT,
        peak("4029.85"), FLAT, FLAT,
    ]  # fmt: skip
    result = analyse(rows)
    (pool,) = result.pools

    assert len(pool.member_swing_ids) == 2
    assert pool.status is PoolStatus.ACTIVE, "no sweep: 4029.85 is below the 4030.40 boundary"
    assert len(result.unpaired_swing_ids) == 1


def test_the_transitive_chain_trap() -> None:
    """§6. 4030.0 / 4030.4 / 4030.8 with tolerance 0.5 is not one 0.8-wide pool.

    Two consecutive gaps of 0.4 do not add up to a pool, and this fixture shows
    the engine refusing it twice over: the third high is outside the full span,
    *and* the bar that printed it swept the band on the way past. Either alone
    would be enough; what matters is that no 0.8-wide pool exists.
    """
    rows = [
        FLAT, FLAT, peak("4030.00"), FLAT, FLAT, FLAT, peak("4030.40"), FLAT, FLAT, FLAT,
        peak("4030.80"), FLAT, FLAT,
    ]  # fmt: skip
    result = analyse(rows)
    (pool,) = result.pools

    assert (pool.lower, pool.upper) == (Decimal("4030.00"), Decimal("4030.40"))
    assert pool.span == Decimal("0.40")
    assert all(entry.span <= Decimal("0.50") for entry in result.pools)
    assert pool.status is PoolStatus.SWEPT
    assert len(result.unpaired_swing_ids) == 1


def test_a_pool_records_its_band_and_midpoint() -> None:
    rows = [FLAT, FLAT, peak("4030.00"), FLAT, FLAT, FLAT, peak("4030.30"), FLAT, FLAT]
    (pool,) = analyse(rows).pools

    assert pool.lower == Decimal("4030.00")
    assert pool.upper == Decimal("4030.30")
    assert pool.midpoint == Decimal("4030.15")
    assert pool.boundary == pool.upper, "buy-side liquidity rests above the top"


def test_a_sell_side_pool_is_tested_against_its_lower_edge() -> None:
    (pool,) = analyse(POOL_LOWS, "0").pools

    assert pool.boundary == pool.lower


def test_a_pool_is_a_band_not_an_average() -> None:
    """The midpoint is reference only; nothing is ever tested against it."""
    rows = [FLAT, FLAT, peak("4030.00"), FLAT, FLAT, FLAT, peak("4030.30"), FLAT, FLAT]
    (pool,) = analyse(rows).pools

    assert pool.boundary != pool.midpoint


# --------------------------------------------------------------------------
# §10-§11: when a pool starts existing
# --------------------------------------------------------------------------


def test_a_pool_is_formed_at_the_later_founding_confirmation() -> None:
    """Not a pivot time, and not the second pivot's own bar."""
    (pool,) = analyse(POOL_HIGHS, "0").pools

    assert bar_index(pool.formed_at) == 8, "the second pivot is bar 6; it confirms at bar 8"
    assert pool.formed_at == START + HOUR * 9


def test_a_pool_does_not_exist_before_it_is_formed() -> None:
    early = analyse_liquidity(series(POOL_HIGHS, bars_kept=8), config=config("0"), symbol="XAUUSD")
    later = analyse_liquidity(series(POOL_HIGHS, bars_kept=9), config=config("0"), symbol="XAUUSD")

    assert early.pools == ()
    assert len(later.pools) == 1


# --------------------------------------------------------------------------
# §45: wick sweeps, close-through and exact touches
# --------------------------------------------------------------------------


def only_event(rows: Sequence[Row], tolerance: str = "0") -> LiquidityEventType | None:
    events = analyse(rows, tolerance).events
    return events[0].event_type if events else None


def test_a_buy_side_wick_sweep() -> None:
    rows = [*POOL_HIGHS, bar("4035", "3990", "4005"), FLAT]
    result = analyse(rows, "0")
    (event,) = result.events

    assert event.event_type is LiquidityEventType.WICK_SWEEP
    assert event.side is LiquiditySide.BUY_SIDE
    assert event.boundary == Decimal("4030")
    assert event.wick_extreme == Decimal("4035")
    assert event.close == Decimal("4005")
    assert result.pools[0].status is PoolStatus.SWEPT
    assert result.pools[0].terminal_event_id == event.event_id


def test_a_close_exactly_on_the_boundary_after_trading_above_is_a_sweep() -> None:
    rows = [*POOL_HIGHS, bar("4035", "3990", "4030"), FLAT]

    assert only_event(rows) is LiquidityEventType.WICK_SWEEP


def test_an_exact_high_touch_of_a_buy_side_pool_does_nothing() -> None:
    """Liquidity has to actually trade strictly beyond the edge."""
    rows = [*POOL_HIGHS, bar("4030", "3990", "4005"), FLAT]
    result = analyse(rows, "0")

    assert result.events == ()
    assert result.pools[0].status is PoolStatus.ACTIVE


def test_a_buy_side_close_through() -> None:
    rows = [*POOL_HIGHS, bar("4040", "3990", "4035"), FLAT]
    result = analyse(rows, "0")
    (event,) = result.events

    assert event.event_type is LiquidityEventType.CLOSE_THROUGH
    assert result.pools[0].status is PoolStatus.CLOSED_THROUGH


def test_a_sell_side_wick_sweep() -> None:
    rows = [*POOL_LOWS, bar("4010", "3965", "3995"), FLAT]
    result = analyse(rows, "0")
    (event,) = result.events

    assert event.event_type is LiquidityEventType.WICK_SWEEP
    assert event.side is LiquiditySide.SELL_SIDE
    assert event.boundary == Decimal("3970")
    assert event.wick_extreme == Decimal("3965")


def test_a_sell_side_close_exactly_on_the_boundary_is_a_sweep() -> None:
    rows = [*POOL_LOWS, bar("4010", "3965", "3970"), FLAT]

    assert only_event(rows) is LiquidityEventType.WICK_SWEEP


def test_an_exact_low_touch_of_a_sell_side_pool_does_nothing() -> None:
    rows = [*POOL_LOWS, bar("4010", "3970", "3995"), FLAT]

    assert analyse(rows, "0").events == ()


def test_a_sell_side_close_through() -> None:
    rows = [*POOL_LOWS, bar("4010", "3960", "3965"), FLAT]
    result = analyse(rows, "0")

    assert result.events[0].event_type is LiquidityEventType.CLOSE_THROUGH
    assert result.pools[0].status is PoolStatus.CLOSED_THROUGH


def test_close_through_takes_precedence_over_the_wick() -> None:
    """§22. One candle, one verdict per pool, and the close is the stronger fact."""
    rows = [*POOL_HIGHS, bar("4040", "3990", "4035"), FLAT]
    result = analyse(rows, "0")

    assert len(result.events) == 1
    assert result.events[0].event_type is LiquidityEventType.CLOSE_THROUGH
    assert result.events[0].wick_extreme == Decimal("4040"), "the wick is recorded, not classified"


def test_a_gap_clean_beyond_the_pool_still_closes_through() -> None:
    """§29. The path is not in the data, and demanding it would be inventing one."""
    rows = [*POOL_HIGHS, bar("4060", "4050", "4055"), FLAT]

    assert only_event(rows) is LiquidityEventType.CLOSE_THROUGH


def test_a_pool_takes_at_most_one_event_ever() -> None:
    """§25. Three consecutive candles beyond the same band; one is news."""
    rows = [
        *POOL_HIGHS,
        bar("4035", "3990", "4005"),
        bar("4036", "3990", "4006"),
        bar("4037", "3990", "4007"),
        FLAT,
    ]
    result = analyse(rows, "0")

    assert len(result.events) == 1
    assert bar_index(result.events[0].event_bar_close_time) == 9


def test_a_terminal_pool_never_accepts_a_new_member() -> None:
    """§15. Later swings at that price are new liquidity, not the old pool."""
    rows = [
        *POOL_HIGHS, bar("4035", "3990", "4005"), FLAT, FLAT,
        peak("4030"), FLAT, FLAT, FLAT, peak("4030"), FLAT, FLAT,
    ]  # fmt: skip
    result = analyse(rows, "0")
    first, second = result.pools

    assert first.status is PoolStatus.SWEPT
    assert len(first.member_swing_ids) == 2
    assert second.status is PoolStatus.ACTIVE
    assert first.pool_id != second.pool_id, "same price, different pool"
    assert (second.lower, second.upper) == (Decimal("4030"), Decimal("4030"))


def test_an_event_records_the_band_as_it_stood() -> None:
    rows = [*POOL_HIGHS, bar("4035", "3990", "4005"), FLAT]
    (event,) = analyse(rows, "0").events
    (pool,) = analyse(rows, "0").pools

    assert event.pool_lower == pool.lower
    assert event.pool_upper == pool.upper
    assert event.member_swing_ids_at_event == pool.member_swing_ids
    assert event.pool_id == pool.pool_id


# --------------------------------------------------------------------------
# §23: known before touched
# --------------------------------------------------------------------------


def fake_pool(
    *,
    formed_at: datetime,
    status: PoolStatus = PoolStatus.ACTIVE,
    side: LiquiditySide = LiquiditySide.BUY_SIDE,
    lower: str = "4030",
    upper: str = "4030",
) -> LiquidityPool:
    return LiquidityPool(
        pool_id="pool",
        method_version="1.0.0",
        timeframe=Timeframe.H1,
        symbol="XAUUSD",
        side=side,
        formed_at=formed_at,
        founding_swing_ids=("a", "b"),
        member_swing_ids=("a", "b"),
        lower=Decimal(lower),
        upper=Decimal(upper),
        midpoint=(Decimal(lower) + Decimal(upper)) / Decimal(2),
        tolerance=Decimal(0),
        status=status,
        terminal_event_id=None,
    )


def test_a_pool_cannot_be_taken_by_the_close_that_created_it() -> None:
    """§23, at the predicate that enforces it.

    Tested here rather than end to end, and that is the honest place for it.
    The bar closing at a pool's formation instant is always inside the second
    founding swing's right-hand window, so a wick beyond the pool's boundary
    there would have destroyed that swing before it could found anything - the
    same structural argument as Round 6.6b's known-before-broken rule. The rule
    is enforced regardless, because it rests on a pivot definition a later round
    may change.
    """
    formed = START + HOUR * 9

    assert touchable(PoolStatus.ACTIVE, formed, formed) is False
    assert touchable(PoolStatus.ACTIVE, formed, formed + HOUR) is True
    assert fake_pool(formed_at=formed).is_touchable_at(formed) is False
    assert fake_pool(formed_at=formed).is_touchable_at(formed + HOUR) is True


@pytest.mark.parametrize("status", [PoolStatus.SWEPT, PoolStatus.CLOSED_THROUGH])
def test_a_terminal_pool_is_never_touchable(status: PoolStatus) -> None:
    formed = START + HOUR * 9

    assert touchable(status, formed, formed + HOUR * 50) is False


def test_the_next_closed_candle_can_take_a_freshly_formed_pool() -> None:
    """The other half of §45: formation blocks its own bar, not the one after."""
    rows = [*POOL_HIGHS, bar("4035", "3990", "4005"), FLAT]
    (pool,) = analyse(rows, "0").pools
    (event,) = analyse(rows, "0").events

    assert event.event_bar_close_time == pool.formed_at + HOUR


# --------------------------------------------------------------------------
# §13: one swing, one pool
# --------------------------------------------------------------------------


def working_pool(
    *,
    pool_id: str,
    lower: str,
    upper: str,
    formed_at: datetime,
    side: LiquiditySide = LiquiditySide.BUY_SIDE,
    status: PoolStatus = PoolStatus.ACTIVE,
) -> _WorkingPool:
    pool = _WorkingPool(
        pool_id=pool_id,
        side=side,
        formed_at=formed_at,
        founding=(f"{pool_id}-a", f"{pool_id}-b"),
        members=[(f"{pool_id}-a", Decimal(lower)), (f"{pool_id}-b", Decimal(upper))],
    )
    pool.status = status
    return pool


def candidate(price: str, swing_type: SwingType = SwingType.SWING_HIGH) -> SwingPoint:
    return SwingPoint(
        timeframe=Timeframe.H1,
        swing_type=swing_type,
        pivot_time=START + HOUR * 20,
        confirmed_at=START + HOUR * 22,
        price=Decimal(price),
        left_bars=2,
        right_bars=2,
    )


def test_no_swing_appears_in_two_pools() -> None:
    result = analyse(
        [
            FLAT,
            FLAT,
            peak("4030.00"),
            FLAT,
            FLAT,
            FLAT,
            peak("4030.40"),
            FLAT,
            FLAT,
            FLAT,
            peak("4030.20"),
            FLAT,
            FLAT,
            FLAT,
            peak("4030.10"),
            FLAT,
            FLAT,
        ]  # fmt: skip
    )

    seen: list[str] = []
    for pool in result.pools:
        seen.extend(pool.member_swing_ids)
    assert len(seen) == len(set(seen))
    assert not set(seen) & set(result.unpaired_swing_ids)


def test_two_active_pools_can_never_compete_for_one_swing() -> None:
    """§13's invariant, asserted rather than resolved.

    A swing high's price is its own bar's high, so a high above a pool's upper
    swept that pool as it printed; a swing can therefore only join a pool whose
    upper is at or above it. Combined with join-before-pair precedence, that
    makes two eligible active pools unreachable - the swing that set the lower
    pool's upper would itself have joined the higher pool instead. Proved here
    by exhaustive replay over a grid of prices rather than by argument alone.
    """
    tolerance = Decimal("0.50")
    for first in ("4030.00", "4030.20", "4030.40"):
        for second in ("4029.80", "4030.10", "4030.60", "4031.00"):
            for third in ("4029.70", "4030.05", "4030.35", "4030.90"):
                rows = [
                    FLAT, FLAT, peak(first), FLAT, FLAT, FLAT, peak(second), FLAT, FLAT, FLAT,
                    peak(third), FLAT, FLAT,
                ]  # fmt: skip
                result = analyse(rows)
                members: list[str] = []
                for pool in result.pools:
                    members.extend(pool.member_swing_ids)
                    assert pool.span <= tolerance
                assert len(members) == len(set(members))


def test_the_tie_break_prefers_the_smallest_resulting_span() -> None:
    """Unreachable from real candles, so exercised on the ranking directly."""
    wide = working_pool(pool_id="wide", lower="4029.60", upper="4030.00", formed_at=START)
    tight = working_pool(pool_id="tight", lower="4029.90", upper="4030.00", formed_at=START)

    chosen = _best_pool_for(
        candidate("4029.95"), LiquiditySide.BUY_SIDE, [wide, tight], Decimal("0.50")
    )
    assert chosen is tight


def test_the_tie_break_falls_to_the_nearest_midpoint() -> None:
    near = working_pool(pool_id="near", lower="4029.90", upper="4030.00", formed_at=START)
    far = working_pool(pool_id="far", lower="4030.30", upper="4030.40", formed_at=START)

    # Both would span 0.10 more; the candidate sits closer to `near`'s midpoint.
    chosen = _best_pool_for(
        candidate("4029.95"), LiquiditySide.BUY_SIDE, [far, near], Decimal("0.50")
    )
    assert chosen is near


def test_the_tie_break_falls_to_the_most_recently_formed_pool() -> None:
    older = working_pool(pool_id="aaa", lower="4030.00", upper="4030.10", formed_at=START)
    newer = working_pool(
        pool_id="zzz", lower="4030.00", upper="4030.10", formed_at=START + HOUR * 5
    )

    chosen = _best_pool_for(
        candidate("4030.05"), LiquiditySide.BUY_SIDE, [older, newer], Decimal("0.50")
    )
    assert chosen is newer


def test_the_tie_break_finally_falls_to_the_pool_id() -> None:
    first = working_pool(pool_id="aaa", lower="4030.00", upper="4030.10", formed_at=START)
    second = working_pool(pool_id="bbb", lower="4030.00", upper="4030.10", formed_at=START)

    forwards = _best_pool_for(
        candidate("4030.05"), LiquiditySide.BUY_SIDE, [first, second], Decimal("0.50")
    )
    backwards = _best_pool_for(
        candidate("4030.05"), LiquiditySide.BUY_SIDE, [second, first], Decimal("0.50")
    )
    assert forwards is not None
    assert backwards is not None
    assert forwards.pool_id == backwards.pool_id == "aaa", "argument order is not an input"


def test_a_terminal_pool_is_not_eligible_for_a_new_member() -> None:
    dead = working_pool(
        pool_id="dead",
        lower="4030.00",
        upper="4030.10",
        formed_at=START,
        status=PoolStatus.SWEPT,
    )

    assert (
        _eligible_pools(Decimal("4030.05"), LiquiditySide.BUY_SIDE, [dead], Decimal("0.50")) == []
    )


def test_a_pool_on_the_other_side_is_not_eligible() -> None:
    sell = working_pool(
        pool_id="sell",
        lower="4030.00",
        upper="4030.10",
        formed_at=START,
        side=LiquiditySide.SELL_SIDE,
    )

    assert (
        _eligible_pools(Decimal("4030.05"), LiquiditySide.BUY_SIDE, [sell], Decimal("0.50")) == []
    )


# --------------------------------------------------------------------------
# §32: liquidity equality is not structure equality
# --------------------------------------------------------------------------


def test_a_relation_label_is_not_a_prerequisite() -> None:
    """A ``HIGHER_HIGH`` may still be buy-side liquidity with the high before it.

    The two engines answer different questions and are allowed to disagree about
    the word "equal". Structure needs exact Decimals so a break is unambiguous;
    liquidity needs a band because orders do not rest on a single tick.
    """
    from goldpipeline.services.ict_primitives import confirmed_swings
    from goldpipeline.services.ict_structure import annotate_swings

    rows = [FLAT, FLAT, peak("4030.00"), FLAT, FLAT, FLAT, peak("4030.30"), FLAT, FLAT]
    labels = [entry.label for entry in annotate_swings(confirmed_swings(series(rows)))]

    assert labels == ["FIRST_HIGH", "HIGHER_HIGH"], "structure calls the second one higher"
    assert bands(rows, "0.50") == [("BUY_SIDE", "4030.00", "4030.30")], "liquidity pools them"


def test_a_lower_low_may_still_be_sell_side_liquidity() -> None:
    from goldpipeline.services.ict_primitives import confirmed_swings
    from goldpipeline.services.ict_structure import annotate_swings

    rows = [FLAT, FLAT, trough("3970.00"), FLAT, FLAT, FLAT, trough("3969.70"), FLAT, FLAT]
    labels = [entry.label for entry in annotate_swings(confirmed_swings(series(rows)))]

    assert labels == ["FIRST_LOW", "LOWER_LOW"]
    assert bands(rows, "0.50") == [("SELL_SIDE", "3969.70", "3970.00")]


# --------------------------------------------------------------------------
# §31: liquidity does not consult structure
# --------------------------------------------------------------------------


def test_a_structurally_consumed_swing_still_forms_a_pool() -> None:
    """A structure break and a repeated-swing pool answer different questions.

    Here the first 4030 high is broken by a close at 4035 - the structure engine
    consumes it and will never emit another event on it. The liquidity engine
    knows nothing about that, and correctly still pools it with the later 4030
    that shares its price. Coupling the two would mean a BOS silently deleting
    liquidity the market can plainly still see.
    """
    from goldpipeline.services.ict_structure import analyse_structure

    rows = [
        FLAT, FLAT, peak("4030"), FLAT, FLAT, bar("4036", "3990", "4035"),
        bar("4036", "3990", "4035"), FLAT, FLAT, peak("4030"), FLAT, FLAT,
    ]  # fmt: skip
    structure = analyse_structure(series(rows), symbol="XAUUSD")
    liquidity = analyse(rows, "0")

    assert structure.breaks, "the first high really was broken"
    consumed = structure.consumed_swing_ids
    assert consumed

    (pool,) = liquidity.pools
    assert set(pool.member_swing_ids) & consumed, "a consumed swing is still liquidity"
    assert (pool.lower, pool.upper) == (Decimal("4030"), Decimal("4030"))
