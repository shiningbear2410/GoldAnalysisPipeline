"""Whether tomorrow's break can turn yesterday's candle into an order block.

Round 6.6d.1 §34-§40, §47. The sixth time this file has been written on the
branch, and the trap is a new shape: the source candle genuinely already exists
before the event. An engine that dated the block by the candle rather than by
the break would look completely reasonable and would report an order block days
before anybody could have known it was one.
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
    BodyDirection,
    OrderBlockAnalysis,
    analyse_order_blocks,
    analyse_snapshot_order_blocks,
    body_direction,
)
from goldpipeline.services.ict_structure import analyse_structure
from tests.test_ict_order_block import BODY, FULL, reopen
from tests.test_ict_order_block_fixture import OPENS, journey
from tests.test_ict_protected import closing_bar
from tests.test_ict_range_fixture import JOURNEY
from tests.test_ict_structure import START, bar_index, series

HOUR = timedelta(hours=1)


def truncated(kept: int) -> IctTimeframeSnapshot:
    return reopen(series(JOURNEY, bars_kept=kept), {i: v for i, v in OPENS.items() if i < kept})


def at(hours: int) -> OrderBlockAnalysis:
    return analyse_order_blocks(journey(), config=FULL, symbol="XAUUSD", as_of=START + HOUR * hours)


# --------------------------------------------------------------------------
# §34: the central invariant
# --------------------------------------------------------------------------


@pytest.mark.parametrize("config_name", ["FULL", "BODY"])
def test_as_of_equals_physical_truncation_at_every_bar(config_name: str) -> None:
    """§34. Field-for-field equality at all 54 closes, under both policies.

    Identities, source candles, directions, zones, formation instants and
    ordering. A future break, a future candle or a future reversal reaching an
    earlier answer would break one of them.
    """
    config = FULL if config_name == "FULL" else BODY
    full = journey()

    for kept in range(1, len(JOURNEY) + 1):
        moment = START + HOUR * kept
        physical = analyse_order_blocks(truncated(kept), config=config, symbol="XAUUSD")
        as_of = analyse_order_blocks(full, config=config, symbol="XAUUSD", as_of=moment)

        assert as_of == physical, f"disagreement at bar {kept - 1}"


def test_an_as_of_query_reports_the_instant_it_was_asked_about() -> None:
    assert at(20).observed_at == START + HOUR * 20


# --------------------------------------------------------------------------
# §35: a candle is not an order block until its event closes
# --------------------------------------------------------------------------


def test_a_future_break_does_not_make_a_past_candle_an_order_block() -> None:
    """§35, and the whole reason ``formed_at`` is the event's close.

    Bar 8 is a bearish candle from the moment it closes. It is not an order
    block at bar 9, or 10 - it becomes one when the INITIAL_BREAK confirms at
    bar 11, and not one instant earlier.
    """
    shaped = journey()

    assert body_direction(shaped.bars[8]) is BodyDirection.BEARISH
    for kept in (9, 10, 11):
        assert at(kept).order_blocks == (), f"an order block appeared at bar {kept - 1}"

    appeared = at(12)
    assert len(appeared.order_blocks) == 1
    assert bar_index(appeared.order_blocks[0].source_bar_open_time) == 8
    assert closing_bar(appeared.order_blocks[0].formed_at) == 11


def test_formation_is_dated_by_the_event_not_by_the_candle() -> None:
    """§11. Three bars separate the source candle from the block's birthday."""
    first = at(12).order_blocks[0]

    assert bar_index(first.source_bar_open_time) == 8
    assert closing_bar(first.formed_at) == 11
    assert first.formed_at > first.source_bar_close_time


def test_the_same_candle_reused_later_appears_only_at_its_own_event() -> None:
    """§47 C. Bar 30 becomes a second order block at bar 40, not before."""
    before = at(40)
    after = at(41)

    assert [closing_bar(e.formed_at) for e in before.order_blocks] == [11, 18, 33]
    assert [closing_bar(e.formed_at) for e in after.order_blocks] == [11, 18, 33, 40]
    assert bar_index(after.order_blocks[-1].source_bar_open_time) == 30


# --------------------------------------------------------------------------
# §36: history is append-only
# --------------------------------------------------------------------------


def test_a_later_event_rewrites_no_earlier_order_block() -> None:
    """§36. Source candle, zone, identity and formation instant are all fixed."""
    full = journey()
    seen: dict[str, tuple[object, ...]] = {}

    for kept in range(1, len(JOURNEY) + 1):
        analysis = analyse_order_blocks(
            full, config=FULL, symbol="XAUUSD", as_of=START + HOUR * kept
        )
        for entry in analysis.order_blocks:
            shape = (
                entry.source_bar_open_time,
                entry.source_open,
                entry.source_high,
                entry.source_low,
                entry.source_close,
                entry.lower,
                entry.upper,
                entry.midpoint,
                entry.width,
                entry.formed_at,
                entry.direction,
                entry.event_classification,
            )
            if entry.order_block_id in seen:
                assert seen[entry.order_block_id] == shape, f"{entry.order_block_id} was rewritten"
            seen[entry.order_block_id] = shape

    assert len(seen) == 4


def test_order_block_history_only_grows() -> None:
    """§47 F."""
    full = journey()
    seen: list[tuple[str, ...]] = []

    for kept in range(1, len(JOURNEY) + 1):
        analysis = analyse_order_blocks(
            full, config=FULL, symbol="XAUUSD", as_of=START + HOUR * kept
        )
        seen.append(tuple(entry.order_block_id for entry in analysis.order_blocks))

    assert seen[-1], "a fixture with no order blocks would prove nothing"
    for earlier, later in zip(seen, seen[1:], strict=False):
        assert later[: len(earlier)] == earlier


def test_a_future_opposite_candle_cannot_become_the_source_of_an_old_block() -> None:
    """§47 D. Bar 32 is bearish and inside no earlier bullish leg's window."""
    shaped = journey()
    first_at_formation = at(12).order_blocks[0]
    first_at_the_end = analyse_order_blocks(shaped, config=FULL, symbol="XAUUSD").order_blocks[0]

    assert body_direction(shaped.bars[32]) is BodyDirection.BEARISH
    assert first_at_the_end == first_at_formation


def test_an_opposite_candle_with_no_event_yet_produces_nothing() -> None:
    """§47 A. Bar 13 is bearish from bar 13; nothing has broken since bar 11."""
    shaped = journey()
    at_17 = at(17)

    assert body_direction(shaped.bars[13]) is BodyDirection.BEARISH
    assert [closing_bar(e.formed_at) for e in at_17.order_blocks] == [11]
    assert bar_index(at_17.order_blocks[0].source_bar_open_time) == 8, "still bar 8's block"


def test_the_second_event_closes_and_the_second_block_appears() -> None:
    """§47 B."""
    after = at(19)

    assert [closing_bar(e.formed_at) for e in after.order_blocks] == [11, 18]
    assert bar_index(after.order_blocks[1].source_bar_open_time) == 13


# --------------------------------------------------------------------------
# §37: forming candles
# --------------------------------------------------------------------------


def test_a_forming_candle_is_not_in_the_series_at_all() -> None:
    """§37. The closed-bar semantics every engine on this branch shares."""
    shaped = journey()
    mid_bar = build_timeframe_snapshot(
        timeframe=shaped.timeframe,
        bars=shaped.bars,
        observed_at=START + HOUR * 12 + timedelta(minutes=30),
    )

    assert len(mid_bar.bars) == 12
    assert analyse_order_blocks(mid_bar, config=FULL, symbol="XAUUSD") == analyse_order_blocks(
        shaped, config=FULL, symbol="XAUUSD", as_of=START + HOUR * 12
    )


def test_a_forming_break_bar_has_not_created_its_order_block() -> None:
    """§37. Half way through bar 11 the event does not exist yet."""
    shaped = journey()
    part_way = analyse_order_blocks(
        shaped, config=FULL, symbol="XAUUSD", as_of=START + HOUR * 11 + timedelta(minutes=59)
    )

    assert part_way.order_blocks == ()


# --------------------------------------------------------------------------
# §38: ordering is a property of the output
# --------------------------------------------------------------------------


def test_ordering_is_stable_and_chronological_at_every_instant() -> None:
    full = journey()

    for kept in range(1, len(JOURNEY) + 1):
        analysis = analyse_order_blocks(
            full, config=FULL, symbol="XAUUSD", as_of=START + HOUR * kept
        )
        keys = [
            (entry.formed_at, entry.structure_event_id, entry.order_block_id)
            for entry in analysis.order_blocks
        ]
        assert keys == sorted(keys)


def test_two_blocks_sharing_a_source_candle_keep_their_own_places() -> None:
    """The pair from §44, ordered by their events rather than by their candle."""
    third, fourth = analyse_order_blocks(journey(), config=FULL, symbol="XAUUSD").order_blocks[2:]

    assert third.source_bar_open_time == fourth.source_bar_open_time
    assert third.formed_at < fourth.formed_at


# --------------------------------------------------------------------------
# §39-§40: replay determinism and provider independence
# --------------------------------------------------------------------------


def snapshot_with(provider: str, provider_symbol: str | None) -> IctMarketSnapshot:
    shaped = journey()
    return IctMarketSnapshot(
        observed_at=shaped.latest_closed_at,
        symbol="XAUUSD",
        provider=provider,
        provider_symbol=provider_symbol,
        timeframes=(shaped,),
    )


def test_the_provider_changes_nothing() -> None:
    """§40. Provenance is recorded on the snapshot and never consulted."""
    one = analyse_snapshot_order_blocks(
        snapshot_with("tradingview", "OANDA:XAUUSD"), Timeframe.H1, config=FULL
    )
    two = analyse_snapshot_order_blocks(
        snapshot_with("metatrader", "XAUUSD.pro"), Timeframe.H1, config=FULL
    )
    three = analyse_snapshot_order_blocks(snapshot_with("fixture", None), Timeframe.H1, config=FULL)

    assert one == two == three
    assert one.order_blocks, "an empty comparison would prove nothing"


def test_repeated_analysis_gives_an_identical_result() -> None:
    assert analyse_order_blocks(journey(), config=FULL, symbol="XAUUSD") == analyse_order_blocks(
        journey(), config=FULL, symbol="XAUUSD"
    )


def test_the_two_policies_do_not_contaminate_each_other() -> None:
    """Asking for one basis leaves the other's answer unchanged."""
    first_full = analyse_order_blocks(journey(), config=FULL, symbol="XAUUSD")
    body = analyse_order_blocks(journey(), config=BODY, symbol="XAUUSD")
    second_full = analyse_order_blocks(journey(), config=FULL, symbol="XAUUSD")

    assert first_full == second_full
    assert body != first_full


def test_no_structure_event_moved_under_the_reshaped_bodies() -> None:
    """A last check that this file's fixture is still 6.6c.3b's path."""
    from tests.test_ict_range_fixture import journey as range_journey

    assert [e.event_id for e in analyse_structure(journey(), symbol="XAUUSD").breaks] == [
        e.event_id for e in analyse_structure(range_journey(), symbol="XAUUSD").breaks
    ]


REPLAY_PROGRAM = """
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from tests.test_ict_order_block_fixture import journey
from goldpipeline.services.ict_order_block import (
    OrderBlockConfig,
    OrderBlockZoneBasis,
    analyse_order_blocks,
)

for basis in (OrderBlockZoneBasis.FULL_CANDLE, OrderBlockZoneBasis.BODY):
    result = analyse_order_blocks(
        journey(), config=OrderBlockConfig(zone_basis=basis), symbol="XAUUSD"
    )
    for entry in result.order_blocks:
        print("OB", basis.value, entry.order_block_id, entry.direction.value,
              entry.event_classification.value, entry.structure_event_id,
              entry.structural_leg_id, entry.protected_assignment_id,
              entry.formed_at.isoformat(), entry.source_bar_open_time.isoformat(),
              entry.source_open, entry.source_high, entry.source_low, entry.source_close,
              entry.source_body_direction.value, entry.lower, entry.upper,
              entry.midpoint, entry.width)
"""


@pytest.mark.parametrize("seed", ["0", "1", "524287"])
def test_string_hash_ordering_cannot_reach_the_output(seed: str) -> None:
    """§39. Set and dict iteration order is not allowed to be an input.

    This engine keys events and bars by timestamp in dicts, so the risk is real
    rather than theoretical.
    """
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
    assert "OB" in baseline, "a silent empty run would prove nothing"


def test_no_randomness_is_available_to_the_module() -> None:
    """§39. No UUID, no shuffle, no clock."""
    from pathlib import Path

    text = Path("src/goldpipeline/services/ict_order_block.py").read_text(encoding="utf-8")

    for forbidden in ("uuid", "random", "secrets", "shuffle", "datetime.now", "utcnow"):
        assert forbidden not in text


def test_a_zone_edge_is_never_recomputed_from_a_rounded_value() -> None:
    """Prices come straight off the bar, exponent and all."""
    shaped = journey()
    found = analyse_order_blocks(shaped, config=FULL, symbol="XAUUSD").order_blocks[0]
    original = shaped.bars[8]

    assert found.lower is original.low or found.lower == original.low
    assert found.upper == original.high
    assert found.width == Decimal("40")
