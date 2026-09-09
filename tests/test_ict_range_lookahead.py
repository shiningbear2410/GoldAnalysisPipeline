"""Whether tomorrow's candle can change what a range was today.

Round 6.6c.3b §37-§42. The fifth time this file has been written on the branch,
and the sharpest case yet: a range's edge *is meant to move*, so an engine that
let a future extension leak backwards would look entirely normal while quietly
reporting a range the market had not yet drawn.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import IctMarketSnapshot, build_timeframe_snapshot
from goldpipeline.services.ict_range import (
    RangeStatus,
    analyse_dealing_ranges,
    analyse_snapshot_dealing_ranges,
)
from tests.test_ict_protected import closing_bar
from tests.test_ict_range_fixture import JOURNEY, journey
from tests.test_ict_structure import START, series

HOUR = timedelta(hours=1)


def at(hours: int) -> object:
    return analyse_dealing_ranges(journey(), symbol="XAUUSD", as_of=START + HOUR * hours)


# --------------------------------------------------------------------------
# §37: the central invariant
# --------------------------------------------------------------------------


def test_as_of_equals_physical_truncation_at_every_bar() -> None:
    """Field-for-field equality at all 54 closes.

    Range ids, origins, terminals, witnesses, equilibria, widths, statuses,
    supersession metadata and the active reference. A future extension, a future
    anchor or a future reversal reaching an earlier answer would break one.
    """
    full = journey()

    for kept in range(1, len(JOURNEY) + 1):
        moment = START + HOUR * kept
        truncated = analyse_dealing_ranges(series(JOURNEY, bars_kept=kept), symbol="XAUUSD")
        as_of = analyse_dealing_ranges(full, symbol="XAUUSD", as_of=moment)

        assert as_of == truncated, f"disagreement at bar {kept - 1}"


def test_an_as_of_query_reports_the_instant_it_was_asked_about() -> None:
    result = analyse_dealing_ranges(journey(), symbol="XAUUSD", as_of=START + HOUR * 20)

    assert result.observed_at == START + HOUR * 20


def test_an_as_of_before_any_bar_closed_is_refused() -> None:
    with pytest.raises(ValueError, match="no bar closed"):
        analyse_dealing_ranges(journey(), symbol="XAUUSD", as_of=START)


# --------------------------------------------------------------------------
# §38: a future extension must not reach backwards
# --------------------------------------------------------------------------


def test_a_future_high_does_not_widen_an_earlier_range() -> None:
    """§38, and the brief's own example shape.

    At bar 12 the first range runs 3970-4055. Bar 18 later prints 4070. The
    reconstruction of bar 12 must still say 4055 - same range, smaller edge.
    """
    early = at(13).active_range  # type: ignore[attr-defined]
    late = analyse_dealing_ranges(journey(), symbol="XAUUSD", as_of=START + HOUR * 19)

    assert early is not None
    assert early.upper == Decimal("4055")
    assert early.equilibrium == Decimal("4012.5")

    retired = late.range_of(early.range_id)
    assert retired is not None
    assert retired.upper == Decimal("4070"), "it did grow, later"
    assert retired.range_id == early.range_id, "and it is the same range"


def test_a_terminal_witness_is_never_back_dated() -> None:
    """§14 through time: the witness only ever moves forward."""
    full = journey()
    seen: dict[str, tuple[str, int]] = {}

    for kept in range(1, len(JOURNEY) + 1):
        analysis = analyse_dealing_ranges(full, symbol="XAUUSD", as_of=START + HOUR * kept)
        for entry in analysis.ranges:
            witness = closing_bar(entry.terminal_bar_close_time)
            if entry.range_id in seen:
                _, previous = seen[entry.range_id]
                assert witness >= previous, f"{entry.range_id} moved its witness backwards"
            seen[entry.range_id] = (str(entry.terminal_price), witness)

    assert len(seen) == 6


def test_a_range_edge_only_ever_moves_outward() -> None:
    """The fixed edge never moves at all; the directional edge never retreats."""
    full = journey()
    origins: dict[str, Decimal] = {}
    widths: dict[str, Decimal] = {}

    for kept in range(1, len(JOURNEY) + 1):
        analysis = analyse_dealing_ranges(full, symbol="XAUUSD", as_of=START + HOUR * kept)
        for entry in analysis.ranges:
            if entry.range_id in origins:
                assert entry.origin_price == origins[entry.range_id], "the anchor moved"
                assert entry.width >= widths[entry.range_id], "the range shrank"
            origins[entry.range_id] = entry.origin_price
            widths[entry.range_id] = entry.width


# --------------------------------------------------------------------------
# §39-§40: a future event must not re-anchor or retire the past
# --------------------------------------------------------------------------


def test_a_future_break_does_not_re_anchor_an_earlier_range() -> None:
    """§39. The 3980 low is protected only from bar 30 onwards."""
    for kept in range(25, 31):
        analysis = analyse_dealing_ranges(journey(), symbol="XAUUSD", as_of=START + HOUR * kept)
        current = analysis.active_range
        assert current is not None
        assert current.origin_price == Decimal("3970"), f"re-anchored early at bar {kept - 1}"

    after = analyse_dealing_ranges(journey(), symbol="XAUUSD", as_of=START + HOUR * 31)
    assert after.active_range is not None
    assert after.active_range.origin_price == Decimal("3980")


def test_a_future_reversal_does_not_supersede_an_earlier_range() -> None:
    """§40. Before bar 33 the bullish range is ACTIVE, whatever happens later."""
    for kept in range(31, 34):
        analysis = analyse_dealing_ranges(journey(), symbol="XAUUSD", as_of=START + HOUR * kept)
        current = analysis.active_range
        assert current is not None
        assert current.status is RangeStatus.ACTIVE
        assert current.origin_price == Decimal("3980")

    after = analyse_dealing_ranges(journey(), symbol="XAUUSD", as_of=START + HOUR * 34)
    retired = after.range_of(analysis.active_range_id)  # type: ignore[arg-type]
    assert retired is not None
    assert retired.status is RangeStatus.SUPERSEDED


def test_range_history_only_grows() -> None:
    full = journey()
    seen: list[tuple[str, ...]] = []

    for kept in range(1, len(JOURNEY) + 1):
        analysis = analyse_dealing_ranges(full, symbol="XAUUSD", as_of=START + HOUR * kept)
        seen.append(tuple(entry.range_id for entry in analysis.ranges))

    assert seen[-1], "a fixture with no ranges would prove nothing"
    for earlier, later in zip(seen, seen[1:], strict=False):
        assert later[: len(earlier)] == earlier


def test_a_status_never_goes_back_to_active() -> None:
    full = journey()
    retired: set[str] = set()

    for kept in range(1, len(JOURNEY) + 1):
        analysis = analyse_dealing_ranges(full, symbol="XAUUSD", as_of=START + HOUR * kept)
        for entry in analysis.ranges:
            if entry.range_id in retired:
                assert entry.status is RangeStatus.SUPERSEDED
            if entry.status is RangeStatus.SUPERSEDED:
                retired.add(entry.range_id)

    assert len(retired) == 5


# --------------------------------------------------------------------------
# §41-§42: replay determinism
# --------------------------------------------------------------------------


def snapshot_with(provider: str, provider_symbol: str | None) -> IctMarketSnapshot:
    observed = START + HOUR * len(JOURNEY)
    return IctMarketSnapshot(
        observed_at=observed,
        symbol="XAUUSD",
        provider=provider,
        provider_symbol=provider_symbol,
        timeframes=(
            build_timeframe_snapshot(
                timeframe=Timeframe.H1, bars=journey().bars, observed_at=observed
            ),
        ),
    )


def test_the_provider_changes_nothing() -> None:
    """§42. Provenance is recorded on the snapshot and never consulted."""
    one = analyse_snapshot_dealing_ranges(
        snapshot_with("tradingview", "OANDA:XAUUSD"), Timeframe.H1
    )
    two = analyse_snapshot_dealing_ranges(snapshot_with("metatrader", "XAUUSD.pro"), Timeframe.H1)
    three = analyse_snapshot_dealing_ranges(snapshot_with("fixture", None), Timeframe.H1)

    assert one == two == three


def test_repeated_analysis_gives_an_identical_result() -> None:
    assert analyse_dealing_ranges(journey(), symbol="XAUUSD") == analyse_dealing_ranges(
        journey(), symbol="XAUUSD"
    )


REPLAY_PROGRAM = """
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from tests.test_ict_range_fixture import journey
from goldpipeline.services.ict_range import analyse_dealing_ranges

result = analyse_dealing_ranges(journey(), symbol="XAUUSD")
for entry in result.ranges:
    print("RANGE", entry.range_id, entry.direction.value, entry.origin_price,
          entry.initial_terminal_price, entry.terminal_price, entry.lower, entry.upper,
          entry.equilibrium, entry.width, entry.status.value,
          entry.terminal_bar_close_time.isoformat(), entry.superseded_at,
          entry.superseded_by_event_id, entry.superseded_by_range_id)
print("ACTIVE", result.active_range_id)
print("BIAS", result.structure_bias.value)
"""


@pytest.mark.parametrize("seed", ["0", "1", "524287"])
def test_string_hash_ordering_cannot_reach_the_output(seed: str) -> None:
    """§41. Set and dict iteration order is not allowed to be an input.

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
    assert "RANGE" in baseline, "a silent empty run would prove nothing"
