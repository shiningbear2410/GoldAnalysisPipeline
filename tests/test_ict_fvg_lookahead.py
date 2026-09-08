"""Whether a future candle can change what a gap used to be.

Round 6.6c.2 §28-§30, §41. The third time this file has been written on this
branch, and for the same reason each time: a zone that quietly appears earlier
than it could have been known, or a fill that back-dates itself, produces a
backtest that looks excellent and a live system that does not.

The central assertion is again the strongest available - the state
reconstructed as of T from the whole series must equal, field for field, the
state computed from a series physically cut at T.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import timedelta

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import IctMarketSnapshot, build_timeframe_snapshot
from goldpipeline.services.ict_fvg import (
    FvgStatus,
    analyse_fvg_lifecycle,
    analyse_snapshot_fvg_lifecycle,
)
from goldpipeline.services.ict_primitives import fair_value_gaps
from tests.test_ict_fvg import START, Row, bar, bar_index, series

HOUR = timedelta(hours=1)

PATH: list[Row] = [
    bar("4000", "3990"),  # 0  A
    bar("4008", "3998"),  # 1  B
    bar("4020", "4010"),  # 2  C -> bullish [4000, 4010]
    bar("4030", "4020"),  # 3
    bar("4005", "4001"),  # 4  touches it
    bar("4030", "4020"),  # 5
    bar("4012", "3995"),  # 6  fills it
    bar("4030", "4020"),  # 7
    bar("4045", "4035"),  # 8
    bar("4048", "4037"),  # 9
    bar("4040", "4030"),  # 10
    bar("4048", "4037"),  # 11
    bar("4020", "4012"),  # 12
    bar("4030", "4022"),  # 13
    bar("4048", "4037"),  # 14
]
"""A gap opened, touched, filled, and several more formed behind it. Enough
shape that a leak in any direction would show."""


# --------------------------------------------------------------------------
# §29: a future gap may not appear early
# --------------------------------------------------------------------------


def test_a_gap_whose_third_candle_has_not_closed_does_not_exist() -> None:
    full = series(PATH)

    early = analyse_fvg_lifecycle(full, symbol="XAUUSD", as_of=START + HOUR * 2)
    later = analyse_fvg_lifecycle(full, symbol="XAUUSD", as_of=START + HOUR * 3)

    assert early.states == ()
    assert len(later.states) == 1
    assert bar_index(later.states[0].gap.formed_at) == 2


def test_the_set_of_known_gaps_only_ever_grows() -> None:
    """§30. Identities accumulate; none is ever withdrawn or renamed."""
    full = series(PATH)
    seen: list[tuple[str, ...]] = []

    for kept in range(1, len(PATH) + 1):
        result = analyse_fvg_lifecycle(full, symbol="XAUUSD", as_of=START + HOUR * kept)
        seen.append(tuple(state.fvg_id for state in result.states))

    assert seen[-1], "a fixture with no gaps would prove nothing"
    for earlier, later in zip(seen, seen[1:], strict=False):
        assert later[: len(earlier)] == earlier


def test_a_full_gap_list_handed_to_an_as_of_query_is_still_filtered() -> None:
    """§29's other half: a caller's convenience must not become a leak."""
    full = series(PATH)
    moment = START + HOUR * 4
    everything = fair_value_gaps(full)

    supplied = analyse_fvg_lifecycle(full, gaps=everything, symbol="XAUUSD", as_of=moment)
    detected = analyse_fvg_lifecycle(full, symbol="XAUUSD", as_of=moment)

    assert len(everything) > len(supplied.states), "the full list really is longer"
    assert supplied == detected


# --------------------------------------------------------------------------
# §28, §30: as-of reconstruction
# --------------------------------------------------------------------------


def test_as_of_equals_physical_truncation_at_every_bar() -> None:
    """The whole no-lookahead guarantee, at every instant in the path.

    Field-for-field equality: gap identities, statuses, first-touch and fill
    provenance, the three id lists and their order. A future formation or a
    future fill reaching an earlier answer would break one of them.
    """
    full = series(PATH)

    for kept in range(1, len(PATH) + 1):
        moment = START + HOUR * kept
        truncated = analyse_fvg_lifecycle(series(PATH, bars_kept=kept), symbol="XAUUSD")
        as_of = analyse_fvg_lifecycle(full, symbol="XAUUSD", as_of=moment)

        assert as_of == truncated, f"disagreement at bar {kept - 1}"


def test_an_as_of_query_reports_the_instant_it_was_asked_about() -> None:
    result = analyse_fvg_lifecycle(series(PATH), symbol="XAUUSD", as_of=START + HOUR * 8)

    assert result.observed_at == START + HOUR * 8
    assert result.bars_considered == 8


def test_an_as_of_before_any_bar_closed_is_refused() -> None:
    """ "No data" and "no gaps" would otherwise be the same answer."""
    with pytest.raises(ValueError, match="no bar closed"):
        analyse_fvg_lifecycle(series(PATH), symbol="XAUUSD", as_of=START)


def test_a_future_fill_does_not_rewrite_an_earlier_state() -> None:
    """§30, and §39 F-H of the temporal matrix, on one gap through time."""
    full = series(PATH)

    def status_at(kept: int) -> FvgStatus:
        result = analyse_fvg_lifecycle(full, symbol="XAUUSD", as_of=START + HOUR * kept)
        (state,) = [s for s in result.states if bar_index(s.gap.formed_at) == 2]
        return state.status

    assert status_at(3) is FvgStatus.OPEN, "formed, and candle C did not touch it"
    assert status_at(4) is FvgStatus.OPEN, "the next bar stayed clear"
    assert status_at(5) is FvgStatus.TOUCHED
    assert status_at(6) is FvgStatus.TOUCHED
    assert status_at(7) is FvgStatus.FILLED
    assert status_at(len(PATH)) is FvgStatus.FILLED


def test_provenance_is_fixed_once_recorded() -> None:
    """The first touch stays the first touch however much history is added."""
    full = series(PATH)

    early = analyse_fvg_lifecycle(full, symbol="XAUUSD", as_of=START + HOUR * 5)
    late = analyse_fvg_lifecycle(full, symbol="XAUUSD")

    (before,) = [s for s in early.states if bar_index(s.gap.formed_at) == 2]
    (after,) = [s for s in late.states if bar_index(s.gap.formed_at) == 2]

    assert before.status is FvgStatus.TOUCHED
    assert after.status is FvgStatus.FILLED
    assert before.first_touched_at == after.first_touched_at
    assert before.first_touch == after.first_touch
    assert before.filled_at is None
    assert after.filled_at is not None


def test_statuses_only_ever_advance() -> None:
    """OPEN → TOUCHED → FILLED, monotonically, for every gap in the path."""
    full = series(PATH)
    rank = {FvgStatus.OPEN: 0, FvgStatus.TOUCHED: 1, FvgStatus.FILLED: 2}
    highest: dict[str, int] = {}

    for kept in range(1, len(PATH) + 1):
        result = analyse_fvg_lifecycle(full, symbol="XAUUSD", as_of=START + HOUR * kept)
        for state in result.states:
            seen = rank[state.status]
            assert seen >= highest.get(state.fvg_id, 0), f"{state.fvg_id} went backwards"
            highest[state.fvg_id] = seen

    assert max(highest.values()) == 2, "at least one gap reached FILLED"


# --------------------------------------------------------------------------
# §32, §41: replay determinism
# --------------------------------------------------------------------------


def snapshot_with(provider: str, provider_symbol: str | None) -> IctMarketSnapshot:
    observed = START + HOUR * len(PATH)
    return IctMarketSnapshot(
        observed_at=observed,
        symbol="XAUUSD",
        provider=provider,
        provider_symbol=provider_symbol,
        timeframes=(
            build_timeframe_snapshot(
                timeframe=Timeframe.H1, bars=series(PATH).bars, observed_at=observed
            ),
        ),
    )


def test_the_provider_changes_nothing() -> None:
    """Provenance is recorded on the snapshot and never consulted for meaning."""
    one = analyse_snapshot_fvg_lifecycle(snapshot_with("tradingview", "OANDA:XAUUSD"), Timeframe.H1)
    two = analyse_snapshot_fvg_lifecycle(snapshot_with("metatrader", "XAUUSD.pro"), Timeframe.H1)
    three = analyse_snapshot_fvg_lifecycle(snapshot_with("fixture", None), Timeframe.H1)

    assert one == two == three


def test_repeated_analysis_gives_an_identical_result() -> None:
    assert analyse_fvg_lifecycle(series(PATH), symbol="XAUUSD") == analyse_fvg_lifecycle(
        series(PATH), symbol="XAUUSD"
    )


def test_a_different_symbol_gives_different_identities_but_identical_geometry() -> None:
    gold = analyse_fvg_lifecycle(series(PATH), symbol="XAUUSD")
    other = analyse_fvg_lifecycle(series(PATH), symbol="XAGUSD")

    assert [s.fvg_id for s in gold.states] != [s.fvg_id for s in other.states]
    assert [(s.gap.lower, s.gap.upper, s.status) for s in gold.states] == [
        (s.gap.lower, s.gap.upper, s.status) for s in other.states
    ]


REPLAY_PROGRAM = """
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from tests.test_ict_fvg import series
from tests.test_ict_fvg_lookahead import PATH
from goldpipeline.services.ict_fvg import analyse_fvg_lifecycle

result = analyse_fvg_lifecycle(series(PATH), symbol="XAUUSD")
for state in result.states:
    print("FVG", state.fvg_id, state.gap.direction.value, state.gap.lower, state.gap.upper,
          state.gap.formed_at.isoformat(), state.status.value,
          state.first_touched_at, state.filled_at, state.first_touch, state.fill)
print("OPEN", result.open_ids)
print("TOUCHED", result.touched_ids)
print("FILLED", result.filled_ids)
"""


@pytest.mark.parametrize("seed", ["0", "1", "524287"])
def test_string_hash_ordering_cannot_reach_the_output(seed: str) -> None:
    """§41. Set and dict iteration order is not allowed to be an input.

    Run in a subprocess because ``PYTHONHASHSEED`` is read once at interpreter
    start, and Python randomises string hashing per process by default.
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
    assert "FVG" in baseline, "a silent empty run would prove nothing"
