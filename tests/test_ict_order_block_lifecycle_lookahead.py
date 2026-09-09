"""Whether tomorrow's candle can change what a block's state was today.

Round 6.6d.2 §37-§42, §51. The seventh time this file has been written on the
branch, and the trap is the sharpest yet: a lifecycle is *supposed* to change,
so an engine leaking the future would look exactly like one working correctly.
Only a bar-by-bar comparison against physically truncated history can tell the
two apart, and that is what the first test does - at every close, under every
mitigation rule, under both zone bases.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import (
    IctMarketSnapshot,
    IctTimeframeSnapshot,
    build_timeframe_snapshot,
)
from goldpipeline.services.ict_order_block import (
    OrderBlockZoneBasis,
    analyse_order_blocks,
)
from goldpipeline.services.ict_order_block_lifecycle import (
    OrderBlockLifecycleAnalysis,
    OrderBlockLifecycleConfig,
    OrderBlockMitigationRule,
    OrderBlockStatus,
    analyse_order_block_lifecycle,
    analyse_snapshot_order_block_lifecycle,
)
from tests.test_ict_order_block import BODY, FULL, reopen
from tests.test_ict_order_block_fixture import OPENS
from tests.test_ict_order_block_lifecycle import FULL_ZONE, MIDPOINT, RULES, TOUCH
from tests.test_ict_order_block_lifecycle_fixture import ROWS, journey, life
from tests.test_ict_protected import closing_bar
from tests.test_ict_structure import START, series

HOUR = timedelta(hours=1)

RANK = {
    OrderBlockStatus.ACTIVE: 0,
    OrderBlockStatus.TOUCHED: 1,
    OrderBlockStatus.MITIGATED: 2,
    OrderBlockStatus.INVALIDATED: 3,
}

CONFIGS = (TOUCH, MIDPOINT, FULL_ZONE)
BASES = (OrderBlockZoneBasis.FULL_CANDLE, OrderBlockZoneBasis.BODY)


def truncated(kept: int) -> IctTimeframeSnapshot:
    return reopen(series(ROWS, bars_kept=kept), {i: v for i, v in OPENS.items() if i < kept})


def at(
    kept: int, config: OrderBlockLifecycleConfig, basis: OrderBlockZoneBasis
) -> OrderBlockLifecycleAnalysis:
    return life(config, basis=basis, as_of=START + HOUR * kept)


# --------------------------------------------------------------------------
# §37: the central invariant
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("basis", BASES)
def test_as_of_equals_physical_truncation_at_every_bar(
    rule: OrderBlockMitigationRule, basis: OrderBlockZoneBasis
) -> None:
    """§37. Field for field at all 63 closes, for all six policy combinations.

    Identities, statuses, all three witnesses, the four id groups and the
    ordering. A future touch, mitigation or invalidation reaching an earlier
    answer would break one of them.
    """
    config = OrderBlockLifecycleConfig(mitigation_rule=rule)
    formation = FULL if basis is OrderBlockZoneBasis.FULL_CANDLE else BODY
    full = journey()

    for kept in range(1, len(ROWS) + 1):
        physical = analyse_order_block_lifecycle(
            truncated(kept), config=config, formation=formation, symbol="XAUUSD"
        )
        as_of = analyse_order_block_lifecycle(
            full, config=config, formation=formation, symbol="XAUUSD", as_of=START + HOUR * kept
        )

        assert as_of == physical, f"disagreement at bar {kept - 1}"


def test_an_as_of_query_reports_the_instant_it_was_asked_about() -> None:
    assert at(30, TOUCH, OrderBlockZoneBasis.FULL_CANDLE).observed_at == START + HOUR * 30


# --------------------------------------------------------------------------
# §38-§40: the future cannot rewrite the past
# --------------------------------------------------------------------------


def test_a_future_touch_does_not_reach_back() -> None:
    """§38. The fifth block is ACTIVE at bar 60 and stays ACTIVE in that reading."""
    before = at(58, TOUCH, OrderBlockZoneBasis.FULL_CANDLE).states[-1]

    assert before.status is OrderBlockStatus.ACTIVE
    assert before.first_touched_at is None
    assert before.first_touch_witness is None


def test_a_future_mitigation_does_not_reach_back() -> None:
    """§39. Under MIDPOINT the third block waits until bar 50; at bar 45 it has not."""
    before = at(45, MIDPOINT, OrderBlockZoneBasis.FULL_CANDLE).states[2]
    after = at(51, MIDPOINT, OrderBlockZoneBasis.FULL_CANDLE).states[2]

    assert before.status is OrderBlockStatus.TOUCHED
    assert before.mitigated_at is None
    assert after.status is OrderBlockStatus.MITIGATED
    assert closing_bar(after.mitigated_at) == 50  # type: ignore[arg-type]
    assert before.order_block_id == after.order_block_id


def test_a_future_invalidation_does_not_reach_back() -> None:
    """§40. The first block is retired at bar 40; at bar 39 it is not."""
    before = at(39, TOUCH, OrderBlockZoneBasis.FULL_CANDLE).states[0]
    after = at(41, TOUCH, OrderBlockZoneBasis.FULL_CANDLE).states[0]

    assert before.status is not OrderBlockStatus.INVALIDATED
    assert before.invalidated_at is None
    assert after.status is OrderBlockStatus.INVALIDATED
    assert closing_bar(after.invalidated_at) == 40  # type: ignore[arg-type]


@pytest.mark.parametrize("rule", RULES)
def test_a_witness_once_recorded_never_changes(rule: OrderBlockMitigationRule) -> None:
    """§11, §16, §22. Each first witness is written once and then left alone."""
    config = OrderBlockLifecycleConfig(mitigation_rule=rule)
    seen: dict[tuple[str, str], object] = {}

    for kept in range(1, len(ROWS) + 1):
        analysis = at(kept, config, OrderBlockZoneBasis.FULL_CANDLE)
        for state in analysis.states:
            for name, witness in (
                ("touch", state.first_touch_witness),
                ("mitigation", state.mitigation_witness),
                ("invalidation", state.invalidation_witness),
            ):
                if witness is None:
                    continue
                key = (state.order_block_id, name)
                if key in seen:
                    assert seen[key] == witness, f"{key} was rewritten"
                seen[key] = witness

    assert seen, "an empty sweep would prove nothing"


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("basis", BASES)
def test_no_status_ever_moves_backwards(
    rule: OrderBlockMitigationRule, basis: OrderBlockZoneBasis
) -> None:
    """§25. Monotone for every block, across the whole replay."""
    config = OrderBlockLifecycleConfig(mitigation_rule=rule)
    highest: dict[str, int] = {}

    for kept in range(1, len(ROWS) + 1):
        for state in at(kept, config, basis).states:
            rank = RANK[state.status]
            assert rank >= highest.get(state.order_block_id, 0), state.order_block_id
            highest[state.order_block_id] = rank

    assert len(highest) == 5


def test_state_history_only_grows() -> None:
    """§37. Blocks are appended, never removed, whatever their status."""
    seen: list[tuple[str, ...]] = []

    for kept in range(1, len(ROWS) + 1):
        analysis = at(kept, TOUCH, OrderBlockZoneBasis.FULL_CANDLE)
        seen.append(tuple(state.order_block_id for state in analysis.states))

    assert seen[-1], "a fixture with no blocks would prove nothing"
    for earlier, later in zip(seen, seen[1:], strict=False):
        assert later[: len(earlier)] == earlier


def test_an_invalidated_block_is_never_revived_by_later_price() -> None:
    """§22, §25. Terminal across every remaining bar of the replay."""
    retired: dict[str, object] = {}

    for kept in range(1, len(ROWS) + 1):
        for state in at(kept, TOUCH, OrderBlockZoneBasis.FULL_CANDLE).states:
            if state.order_block_id in retired:
                assert state.status is OrderBlockStatus.INVALIDATED
                assert retired[state.order_block_id] == state.invalidation_witness
            if state.status is OrderBlockStatus.INVALIDATED:
                retired[state.order_block_id] = state.invalidation_witness

    assert len(retired) == 3


# --------------------------------------------------------------------------
# §51: the temporal matrix
# --------------------------------------------------------------------------


def test_no_state_exists_before_its_block_does() -> None:
    """§51 A. The source candle at bar 50 is not a block until bar 56."""
    for kept in range(51, 57):
        analysis = at(kept, TOUCH, OrderBlockZoneBasis.FULL_CANDLE)
        assert all(closing_bar(s.order_block.formed_at) != 56 for s in analysis.states), kept

    assert any(
        closing_bar(s.order_block.formed_at) == 56
        for s in at(57, TOUCH, OrderBlockZoneBasis.FULL_CANDLE).states
    )


def test_a_block_is_active_at_the_instant_it_forms() -> None:
    """§51 B. It exists, and nothing has happened to it yet."""
    newest = at(57, TOUCH, OrderBlockZoneBasis.FULL_CANDLE).states[-1]

    assert closing_bar(newest.order_block.formed_at) == 56
    assert newest.status is OrderBlockStatus.ACTIVE


def test_the_state_at_formation_survives_the_rest_of_the_replay() -> None:
    """§51 G. Later bars add evidence; they never edit what was already there."""
    at_formation = at(57, TOUCH, OrderBlockZoneBasis.FULL_CANDLE).states[-1]
    at_the_end = life(TOUCH).states[-1]

    assert at_formation.order_block == at_the_end.order_block
    assert at_formation.order_block_id == at_the_end.order_block_id
    assert at_formation.status is OrderBlockStatus.ACTIVE
    assert at_the_end.status is OrderBlockStatus.INVALIDATED


# --------------------------------------------------------------------------
# §41: forming candles
# --------------------------------------------------------------------------


def test_a_forming_candle_is_not_in_the_series_at_all() -> None:
    """§41. The closed-bar semantics every engine on this branch shares."""
    full = journey()
    mid_bar = build_timeframe_snapshot(
        timeframe=full.timeframe,
        bars=full.bars,
        observed_at=START + HOUR * 40 + timedelta(minutes=30),
    )

    assert len(mid_bar.bars) == 40
    assert analyse_order_block_lifecycle(
        mid_bar, config=TOUCH, formation=FULL, symbol="XAUUSD"
    ) == at(40, TOUCH, OrderBlockZoneBasis.FULL_CANDLE)


def test_a_forming_candle_cannot_invalidate() -> None:
    """§41. Bar 40 retires the first block only once it has closed."""
    full = journey()
    part_way = analyse_order_block_lifecycle(
        full,
        config=TOUCH,
        formation=FULL,
        symbol="XAUUSD",
        as_of=START + HOUR * 40 + timedelta(minutes=59),
    )

    assert part_way.states[0].status is not OrderBlockStatus.INVALIDATED
    assert life(TOUCH, as_of=START + HOUR * 41).states[0].status is OrderBlockStatus.INVALIDATED


# --------------------------------------------------------------------------
# §42: replay determinism and provider independence
# --------------------------------------------------------------------------


def snapshot_with(provider: str, provider_symbol: str | None) -> IctMarketSnapshot:
    full = journey()
    return IctMarketSnapshot(
        observed_at=full.latest_closed_at,
        symbol="XAUUSD",
        provider=provider,
        provider_symbol=provider_symbol,
        timeframes=(full,),
    )


@pytest.mark.parametrize("rule", RULES)
def test_the_provider_changes_nothing(rule: OrderBlockMitigationRule) -> None:
    """§42. Provenance is recorded on the snapshot and never consulted."""
    config = OrderBlockLifecycleConfig(mitigation_rule=rule)
    one = analyse_snapshot_order_block_lifecycle(
        snapshot_with("tradingview", "OANDA:XAUUSD"), Timeframe.H1, config=config, formation=FULL
    )
    two = analyse_snapshot_order_block_lifecycle(
        snapshot_with("metatrader", "XAUUSD.pro"), Timeframe.H1, config=config, formation=FULL
    )
    three = analyse_snapshot_order_block_lifecycle(
        snapshot_with("fixture", None), Timeframe.H1, config=config, formation=FULL
    )

    assert one == two == three
    assert one.states, "an empty comparison would prove nothing"


def test_repeated_analysis_gives_an_identical_result() -> None:
    assert life(MIDPOINT) == life(MIDPOINT)


def test_the_three_rules_do_not_contaminate_each_other() -> None:
    """Asking for one convention leaves the others' answers untouched."""
    first = life(FULL_ZONE)
    life(TOUCH)
    life(MIDPOINT)
    second = life(FULL_ZONE)

    assert first == second
    assert life(TOUCH) != first


def test_supplied_blocks_give_the_same_answer_at_every_instant() -> None:
    """§36. The composite-stage path and the compute-it-here path agree."""
    full = journey()

    for kept in range(1, len(ROWS) + 1, 7):
        moment = START + HOUR * kept
        formed = analyse_order_blocks(full, config=FULL, symbol="XAUUSD", as_of=moment)
        supplied = analyse_order_block_lifecycle(
            full, config=MIDPOINT, order_blocks=formed, symbol="XAUUSD", as_of=moment
        )
        computed = analyse_order_block_lifecycle(
            full, config=MIDPOINT, formation=FULL, symbol="XAUUSD", as_of=moment
        )

        assert supplied == computed, f"disagreement at bar {kept - 1}"


REPLAY_PROGRAM = """
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from tests.test_ict_order_block_lifecycle_fixture import journey
from goldpipeline.services.ict_order_block import OrderBlockConfig, OrderBlockZoneBasis
from goldpipeline.services.ict_order_block_lifecycle import (
    OrderBlockLifecycleConfig,
    OrderBlockMitigationRule,
    analyse_order_block_lifecycle,
)

for basis in (OrderBlockZoneBasis.FULL_CANDLE, OrderBlockZoneBasis.BODY):
    for rule in OrderBlockMitigationRule:
        result = analyse_order_block_lifecycle(
            journey(),
            config=OrderBlockLifecycleConfig(mitigation_rule=rule),
            formation=OrderBlockConfig(zone_basis=basis),
            symbol="XAUUSD",
        )
        for state in result.states:
            print("STATE", basis.value, rule.value, state.order_block_id, state.status.value,
                  state.first_touched_at, state.mitigated_at, state.invalidated_at,
                  state.first_touch_witness, state.mitigation_witness,
                  state.invalidation_witness)
        print("GROUPS", basis.value, rule.value, result.active_ids, result.touched_ids,
              result.mitigated_ids, result.invalidated_ids)
"""


@pytest.mark.parametrize("seed", ["0", "1", "524287"])
def test_string_hash_ordering_cannot_reach_the_output(seed: str) -> None:
    """§42. Set and dict iteration order is not allowed to be an input."""
    baseline = subprocess.run(  # noqa: S603
        [sys.executable, "-c", REPLAY_PROGRAM],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": "0", "PATH": ""},
    ).stdout
    other = subprocess.run(  # noqa: S603
        [sys.executable, "-c", REPLAY_PROGRAM],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": seed, "PATH": ""},
    ).stdout

    assert baseline == other
    assert "STATE" in baseline, "a silent empty run would prove nothing"


def test_no_randomness_is_available_to_the_module() -> None:
    """§42. No UUID, no shuffle, no clock."""
    from pathlib import Path

    text = Path("src/goldpipeline/services/ict_order_block_lifecycle.py").read_text(
        encoding="utf-8"
    )

    for forbidden in ("uuid", "random", "secrets", "shuffle", "datetime.now", "utcnow"):
        assert forbidden not in text


def test_witness_prices_come_straight_off_the_bar() -> None:
    """Nothing is normalized, rounded or turned into a float on the way in."""
    state = life(TOUCH).states[0]
    witness = state.first_touch_witness
    bar = journey().bars[12]

    assert witness is not None
    assert (witness.open, witness.high, witness.low, witness.close) == (
        bar.open,
        bar.high,
        bar.low,
        bar.close,
    )
    assert witness.bar_open_time == bar.timestamp
    assert witness.bar_close_time == bar.timestamp + HOUR
    for value in (witness.open, witness.high, witness.low, witness.close):
        assert isinstance(value, Decimal)
