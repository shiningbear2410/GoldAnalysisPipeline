"""Two ways to draw a candle, both stated out loud.

Round 6.6d.1 §13-§17, §19-§21, §23, §43. The interesting content here is not the
arithmetic - it is that neither reading is the default. A module that quietly
chose one would settle a real disagreement by omission, and every stored
identity would carry that choice without anyone having argued for it.
"""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from datetime import timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.services import ict_order_block
from goldpipeline.services.ict_order_block import (
    OrderBlockConfig,
    OrderBlockZoneBasis,
    analyse_order_blocks,
    zone_of,
)
from tests.test_ict_order_block import BEAR_LEG, BODY, BULL_LEG, FULL, bar, bodied, reopen
from tests.test_ict_structure import series

HOUR = timedelta(hours=1)

# §43's candle, exactly as the brief writes it.
SPEC_OPEN, SPEC_HIGH, SPEC_LOW, SPEC_CLOSE = "4010", "4015", "3995", "4000"


# --------------------------------------------------------------------------
# §13: the policy is the caller's, and there is no default
# --------------------------------------------------------------------------


def test_the_config_has_no_default_zone_basis() -> None:
    """§13. Constructing one without saying which reading you mean is an error."""
    with pytest.raises(TypeError):
        OrderBlockConfig()  # type: ignore[call-arg]


def test_the_entry_points_demand_a_config() -> None:
    for function in (
        ict_order_block.analyse_order_blocks,
        ict_order_block.analyse_snapshot_order_blocks,
    ):
        parameter = inspect.signature(function).parameters["config"]
        assert parameter.default is inspect.Parameter.empty
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_the_config_is_immutable() -> None:
    config = OrderBlockConfig(zone_basis=OrderBlockZoneBasis.BODY)

    with pytest.raises(FrozenInstanceError):
        config.zone_basis = OrderBlockZoneBasis.FULL_CANDLE  # type: ignore[misc]


def test_there_are_exactly_two_bases() -> None:
    """§17. No open-to-low, no 50% body, no refined block, not yet."""
    assert [member.value for member in OrderBlockZoneBasis] == ["FULL_CANDLE", "BODY"]


# --------------------------------------------------------------------------
# §14-§15, §43: the geometry of each
# --------------------------------------------------------------------------


def test_the_full_candle_zone_is_the_exact_wick_range() -> None:
    """§14, §43. 3995-4015, with no padding of any kind."""
    candle = bar(SPEC_OPEN, SPEC_HIGH, SPEC_LOW, SPEC_CLOSE)

    assert zone_of(candle, OrderBlockZoneBasis.FULL_CANDLE) == (
        Decimal("3995"),
        Decimal("4015"),
    )


def test_the_body_zone_is_the_exact_open_to_close_range() -> None:
    """§15, §43. 4000-4010, wicks excluded."""
    candle = bar(SPEC_OPEN, SPEC_HIGH, SPEC_LOW, SPEC_CLOSE)

    assert zone_of(candle, OrderBlockZoneBasis.BODY) == (Decimal("4000"), Decimal("4010"))


def test_the_body_zone_sorts_its_edges_whichever_way_the_candle_points() -> None:
    """A bullish-bodied source draws the same interval, read the other way round."""
    bullish = bar("4000", "4015", "3995", "4010")

    assert zone_of(bullish, OrderBlockZoneBasis.BODY) == (Decimal("4000"), Decimal("4010"))


def test_the_spec_candle_end_to_end_under_both_bases() -> None:
    """§43. Midpoints and widths exact, on a real bullish leg.

    Written into bars 6 and 7 as an identical pair, so the taller high makes no
    strict pivot and the leg the rest of the file relies on is unchanged.
    """
    rows = list(BULL_LEG)
    rows[6] = rows[7] = (SPEC_HIGH, SPEC_LOW, SPEC_CLOSE)
    opens = {7: SPEC_OPEN}

    full = analyse_order_blocks(bodied(rows, opens), config=FULL, symbol="XAUUSD").order_blocks[0]
    body = analyse_order_blocks(bodied(rows, opens), config=BODY, symbol="XAUUSD").order_blocks[0]

    assert (full.lower, full.upper, full.midpoint, full.width) == (
        Decimal("3995"),
        Decimal("4015"),
        Decimal("4005"),
        Decimal("20"),
    )
    assert (body.lower, body.upper, body.midpoint, body.width) == (
        Decimal("4000"),
        Decimal("4010"),
        Decimal("4005"),
        Decimal("10"),
    )
    assert full.source_bar_open_time == body.source_bar_open_time


def test_the_mirror_candle_on_a_bearish_leg() -> None:
    """§43's mirror: a bullish-bodied source for a bearish order block."""
    rows = list(BEAR_LEG)
    rows[6] = rows[7] = (SPEC_HIGH, SPEC_LOW, "4010")
    opens = {7: "4000"}

    full = analyse_order_blocks(bodied(rows, opens), config=FULL, symbol="XAUUSD").order_blocks[0]
    body = analyse_order_blocks(bodied(rows, opens), config=BODY, symbol="XAUUSD").order_blocks[0]

    assert full.direction.value == body.direction.value == "BEARISH"
    assert (full.lower, full.upper, full.width) == (
        Decimal("3995"),
        Decimal("4015"),
        Decimal("20"),
    )
    assert (body.lower, body.upper, body.width) == (
        Decimal("4000"),
        Decimal("4010"),
        Decimal("10"),
    )


def test_a_midpoint_may_carry_a_decimal_the_edges_do_not() -> None:
    """§19. Half of an odd width is not rounded away."""
    rows = list(BULL_LEG)
    rows[6] = rows[7] = ("4015", "3990", "4000")
    found = analyse_order_blocks(
        bodied(rows, {7: "4011"}), config=FULL, symbol="XAUUSD"
    ).order_blocks[0]

    assert found.width == Decimal("25")
    assert found.midpoint == Decimal("4002.5")


# --------------------------------------------------------------------------
# §16: both source geometries survive, but only one is canonical
# --------------------------------------------------------------------------


def test_both_readings_stay_inspectable_whichever_basis_was_chosen() -> None:
    """§16. A later round can measure the body without re-reading provider data."""
    rows = list(BULL_LEG)
    rows[6] = rows[7] = (SPEC_HIGH, SPEC_LOW, SPEC_CLOSE)
    full = analyse_order_blocks(
        bodied(rows, {7: SPEC_OPEN}), config=FULL, symbol="XAUUSD"
    ).order_blocks[0]

    assert full.full_candle_bounds == (Decimal("3995"), Decimal("4015"))
    assert full.body_bounds == (Decimal("4000"), Decimal("4010"))


def test_only_the_chosen_basis_is_the_zone() -> None:
    """§16. ``lower``/``upper`` follow the config and nothing else."""
    rows = list(BULL_LEG)
    rows[6] = rows[7] = (SPEC_HIGH, SPEC_LOW, SPEC_CLOSE)
    opens = {7: SPEC_OPEN}

    full = analyse_order_blocks(bodied(rows, opens), config=FULL, symbol="XAUUSD").order_blocks[0]
    body = analyse_order_blocks(bodied(rows, opens), config=BODY, symbol="XAUUSD").order_blocks[0]

    assert (full.lower, full.upper) == full.full_candle_bounds
    assert (body.lower, body.upper) == body.body_bounds
    assert (full.lower, full.upper) != (body.lower, body.upper)


# --------------------------------------------------------------------------
# §21, §23: the basis is part of the identity
# --------------------------------------------------------------------------


def test_changing_the_basis_changes_the_identity() -> None:
    """§21, §23. The canonical zone changed, so it is a different claim."""
    snapshot = bodied(BULL_LEG, {7: "4010"})
    full = analyse_order_blocks(snapshot, config=FULL, symbol="XAUUSD").order_blocks[0]
    body = analyse_order_blocks(snapshot, config=BODY, symbol="XAUUSD").order_blocks[0]

    assert full.order_block_id != body.order_block_id
    assert full.structure_event_id == body.structure_event_id
    assert full.structural_leg_id == body.structural_leg_id
    assert full.source_bar_open_time == body.source_bar_open_time


def test_one_analysis_reports_one_policy() -> None:
    """§23. Two policies means two analyses, asked for deliberately."""
    snapshot = bodied(BULL_LEG, {7: "4010"})
    full = analyse_order_blocks(snapshot, config=FULL, symbol="XAUUSD")

    assert full.zone_basis is OrderBlockZoneBasis.FULL_CANDLE
    assert {entry.zone_basis for entry in full.order_blocks} == {OrderBlockZoneBasis.FULL_CANDLE}


def test_the_identity_is_stable_for_the_same_facts() -> None:
    snapshot = bodied(BULL_LEG, {7: "4010"})

    assert (
        analyse_order_blocks(snapshot, config=FULL, symbol="XAUUSD").order_blocks[0].order_block_id
        == analyse_order_blocks(snapshot, config=FULL, symbol="XAUUSD")
        .order_blocks[0]
        .order_block_id
    )


def test_the_symbol_and_timeframe_reach_the_identity() -> None:
    """The same candles on another instrument or another grid are not the same block."""
    base = analyse_order_blocks(bodied(BULL_LEG, {7: "4010"}), config=FULL, symbol="XAUUSD")
    other_symbol = analyse_order_blocks(bodied(BULL_LEG, {7: "4010"}), config=FULL, symbol="EURUSD")
    on_m15 = analyse_order_blocks(
        reopen(series(BULL_LEG, timeframe=Timeframe.M15), {7: "4010"}),
        config=FULL,
        symbol="XAUUSD",
    )

    identity = base.order_blocks[0].order_block_id
    assert other_symbol.order_blocks[0].order_block_id != identity
    assert on_m15.order_blocks[0].order_block_id != identity
    assert on_m15.order_blocks[0].timeframe is Timeframe.M15


def test_no_price_enters_the_order_block_identity() -> None:
    """§21. So this round needed no fourth Decimal canonicalisation site.

    The preimage is symbol, timeframe, method version, event id, leg id, source
    open time and basis. Every one of those is either a name or a timestamp.
    """
    source = inspect.getsource(ict_order_block._order_block_id)

    assert "normalize" not in source
    joined = source.split('"|".join(')[1]
    for forbidden in ("price", "open=", "high", "low", "close", "lower", "upper", "midpoint"):
        assert forbidden not in joined


def test_the_existing_identities_were_not_disturbed() -> None:
    """The other half of §21: nothing older was refactored for tidiness."""
    from pathlib import Path

    from goldpipeline.services import ict_fvg, ict_range

    assert "normalize" in inspect.getsource(ict_fvg._canonical)
    liquidity = Path("src/goldpipeline/services/ict_liquidity.py").read_text(encoding="utf-8")
    assert "self.price_tolerance.normalize()" in liquidity
    assert "normalize" not in inspect.getsource(ict_range._range_id)
