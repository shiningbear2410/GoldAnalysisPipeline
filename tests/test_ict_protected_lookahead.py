"""Whether a later break can protect a swing retroactively.

Round 6.6c.3a §38-§41, §44. The fourth time this file has been written on the
branch, and the failure it guards against is sharper here than anywhere else: a
protected swing that back-dates itself to its confirmation would make every
anchor look like it had been known long before any event established it, which
is exactly the illusion this round exists to prevent.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import IctMarketSnapshot, build_timeframe_snapshot
from goldpipeline.services.ict_protected import (
    ProtectedSwingType,
    analyse_protected_structure,
    analyse_snapshot_protected_structure,
)
from goldpipeline.services.ict_structure import analyse_structure
from tests.test_ict_protected import closing_bar, wicked
from tests.test_ict_structure import FLAT, START, Row, peak, series, trough

HOUR = timedelta(hours=1)

PATH: list[Row] = [
    FLAT, FLAT, peak("4030"), FLAT, FLAT, trough("3970"), FLAT, FLAT,
    wicked("4035", "4055", "3990"), wicked("4035", "4055", "3990"), FLAT, FLAT,
    peak("4045"), FLAT, FLAT,
    wicked("4050", "4070", "3990"), wicked("4050", "4070", "3990"), FLAT, FLAT,
    FLAT, trough("3980"), FLAT, FLAT, FLAT, FLAT,
    peak("4060"), FLAT, FLAT,
    wicked("4065", "4085", "3990"), wicked("4065", "4085", "3990"), FLAT, FLAT,
]  # fmt: skip
"""An initial break, a BOS reaffirming the same low, a newer low confirming
mid-trend, and a BOS that finally establishes it. The interesting window is
between the newer low's confirmation and the break that protects it."""

REVERSAL: list[Row] = [
    FLAT, FLAT, peak("4200"), FLAT, FLAT,
    *PATH[2:],
    FLAT, wicked("3975", "4010", "3960"), wicked("3975", "4010", "3960"), FLAT, FLAT,
]  # fmt: skip
"""The same path with a tall early high, then a bearish MSS.

The 4200 high is never broken - every bullish close in the path is far below it
- and it is older than the highs being taken out, so it only becomes the active
high once they are all consumed. That is what gives the reversal an opposite
anchor to find; without it the MSS would correctly establish nothing, which is
a different case tested elsewhere."""


# --------------------------------------------------------------------------
# §44 A-C, §39: protection begins at the event
# --------------------------------------------------------------------------


def test_an_unconfirmed_swing_cannot_be_protected() -> None:
    """§44 A. It is not even a swing yet."""
    early = analyse_protected_structure(series(PATH, bars_kept=7), symbol="XAUUSD")

    assert analyse_structure(series(PATH, bars_kept=7), symbol="XAUUSD").breaks == ()
    assert early.assignments == ()
    assert early.legs == ()


def test_a_confirmed_swing_is_not_protected_merely_by_confirming() -> None:
    """§44 B. The low confirms at bar 7; nothing has established it."""
    full = series(PATH)
    structure = analyse_structure(full, symbol="XAUUSD", as_of=START + HOUR * 8)
    result = analyse_protected_structure(full, symbol="XAUUSD", as_of=START + HOUR * 8)

    assert structure.active_low is not None
    assert structure.active_low.swing.price == Decimal("3970"), "it is active"
    assert result.assignments == (), "and it is not protected"


def test_the_break_that_establishes_it_is_where_protection_starts() -> None:
    """§44 C."""
    full = series(PATH)

    before = analyse_protected_structure(full, symbol="XAUUSD", as_of=START + HOUR * 8)
    after = analyse_protected_structure(full, symbol="XAUUSD", as_of=START + HOUR * 9)

    assert before.assignments == ()
    assert len(after.assignments) == 1
    assert after.assignments[0].swing_price == Decimal("3970")
    assert closing_bar(after.assignments[0].established_at) == 8


def test_a_future_break_does_not_protect_a_swing_early() -> None:
    """§39, and the case the brief spells out.

    The 3980 low confirms at bar 22's close. The break that establishes it does
    not close until bar 28. Through every instant in between it is active and
    unprotected, and no amount of later history may change that.
    """
    full = series(PATH)

    for kept in range(23, 29):
        result = analyse_protected_structure(full, symbol="XAUUSD", as_of=START + HOUR * kept)
        assert all(a.swing_price != Decimal("3980") for a in result.assignments), (
            f"3980 was protected too early, at bar {kept - 1}"
        )

    established = analyse_protected_structure(full, symbol="XAUUSD", as_of=START + HOUR * 29)
    assert established.assignments[-1].swing_price == Decimal("3980")
    assert closing_bar(established.assignments[-1].established_at) == 28


def test_the_active_low_moved_long_before_the_protected_one_did() -> None:
    """The distinction, watched through time rather than asserted once."""
    full = series(PATH)

    at_25 = START + HOUR * 26
    structure = analyse_structure(full, symbol="XAUUSD", as_of=at_25)
    result = analyse_protected_structure(full, symbol="XAUUSD", as_of=at_25)
    current = result.current_assignment

    assert structure.active_low is not None
    assert structure.active_low.swing.price == Decimal("3980")
    assert current is not None
    assert current.swing_price == Decimal("3970")


# --------------------------------------------------------------------------
# §38, §44 F-H: as-of reconstruction
# --------------------------------------------------------------------------


def test_as_of_equals_physical_truncation_at_every_bar() -> None:
    """§44 G. The whole no-lookahead guarantee, at every instant in the path.

    Field-for-field equality: assignment and leg identities, origins, terminals,
    the current references and their order.
    """
    full = series(PATH)

    for kept in range(1, len(PATH) + 1):
        moment = START + HOUR * kept
        truncated = analyse_protected_structure(series(PATH, bars_kept=kept), symbol="XAUUSD")
        as_of = analyse_protected_structure(full, symbol="XAUUSD", as_of=moment)

        assert as_of == truncated, f"disagreement at bar {kept - 1}"


def test_an_as_of_query_reports_the_instant_it_was_asked_about() -> None:
    result = analyse_protected_structure(series(PATH), symbol="XAUUSD", as_of=START + HOUR * 12)

    assert result.observed_at == START + HOUR * 12


def test_an_as_of_before_any_bar_closed_is_refused() -> None:
    with pytest.raises(ValueError, match="no bar closed"):
        analyse_protected_structure(series(PATH), symbol="XAUUSD", as_of=START)


def test_assignment_and_leg_histories_only_grow() -> None:
    """§44 H. History accumulates; it is never edited."""
    full = series(PATH)
    assignments: list[tuple[str, ...]] = []
    legs: list[tuple[str, ...]] = []

    for kept in range(1, len(PATH) + 1):
        result = analyse_protected_structure(full, symbol="XAUUSD", as_of=START + HOUR * kept)
        assignments.append(tuple(a.assignment_id for a in result.assignments))
        legs.append(tuple(leg.leg_id for leg in result.legs))

    assert assignments[-1], "a fixture with no assignments would prove nothing"
    for earlier, later in zip(assignments, assignments[1:], strict=False):
        assert later[: len(earlier)] == earlier
    for earlier, later in zip(legs, legs[1:], strict=False):
        assert later[: len(earlier)] == earlier


def test_a_later_break_never_rewrites_an_earlier_leg() -> None:
    """§40. Origin, terminal and identity are all fixed at formation."""
    full = series(PATH)
    seen: dict[str, tuple[str, str, str]] = {}

    for kept in range(1, len(PATH) + 1):
        result = analyse_protected_structure(full, symbol="XAUUSD", as_of=START + HOUR * kept)
        for leg in result.legs:
            fingerprint = (str(leg.origin_price), str(leg.terminal_price), leg.origin_swing_id)
            if leg.leg_id in seen:
                assert seen[leg.leg_id] == fingerprint, f"leg {leg.leg_id} changed"
            seen[leg.leg_id] = fingerprint

    assert len(seen) == 3


def test_a_leg_terminal_does_not_grow_with_later_expansion() -> None:
    """§27. The first leg's terminal is its own bar's high, forever.

    Price goes on to print 4070 and then 4085 after this leg formed. The leg
    still ends at 4055, because it records one event rather than a trend.
    """
    full = series(PATH)

    at_formation = analyse_protected_structure(full, symbol="XAUUSD", as_of=START + HOUR * 10)
    at_end = analyse_protected_structure(full, symbol="XAUUSD")

    assert at_formation.legs[0].terminal_price == Decimal("4055")
    assert at_end.legs[0].terminal_price == Decimal("4055")
    assert max(bar.high for bar in full.bars) == Decimal("4085")


# --------------------------------------------------------------------------
# §41: replay determinism
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
    """§37. Provenance is recorded on the snapshot and never consulted."""
    one = analyse_snapshot_protected_structure(
        snapshot_with("tradingview", "OANDA:XAUUSD"), Timeframe.H1
    )
    two = analyse_snapshot_protected_structure(
        snapshot_with("metatrader", "XAUUSD.pro"), Timeframe.H1
    )
    three = analyse_snapshot_protected_structure(snapshot_with("fixture", None), Timeframe.H1)

    assert one == two == three


def test_repeated_analysis_gives_an_identical_result() -> None:
    assert analyse_protected_structure(
        series(PATH), symbol="XAUUSD"
    ) == analyse_protected_structure(series(PATH), symbol="XAUUSD")


REPLAY_PROGRAM = """
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from tests.test_ict_structure import series
from tests.test_ict_protected_lookahead import PATH
from goldpipeline.services.ict_protected import analyse_protected_structure

result = analyse_protected_structure(series(PATH), symbol="XAUUSD")
for entry in result.assignments:
    print("ASSIGN", entry.assignment_id, entry.protected_type.value, entry.swing_id,
          entry.swing_price, entry.established_by_event_id, entry.established_at.isoformat())
for leg in result.legs:
    print("LEG", leg.leg_id, leg.direction.value, leg.origin_price, leg.terminal_price,
          leg.origin_swing_id, leg.formed_at.isoformat())
print("CURRENT", result.current_protected_assignment_id, result.current_leg_id)
"""


@pytest.mark.parametrize("seed", ["0", "1", "524287"])
def test_string_hash_ordering_cannot_reach_the_output(seed: str) -> None:
    """§41. Set and dict iteration order is not allowed to be an input.

    This engine keys break bars by open time in a dict, so the risk is real
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
    assert "ASSIGN" in baseline and "LEG" in baseline, "a silent empty run proves nothing"


def test_the_current_anchor_follows_the_bias_through_a_reversal() -> None:
    """§14's edge case, walked rather than argued.

    A bullish anchor cannot survive as *current* once a bearish close takes its
    swing out, because that close is an MSS from a bullish state - and the MSS
    establishes a bearish assignment and flips the bias in the same instant.
    """
    full = series(REVERSAL)

    before = analyse_protected_structure(full, symbol="XAUUSD", as_of=START + HOUR * 36)
    after = analyse_protected_structure(full, symbol="XAUUSD")

    bullish = before.current_assignment
    assert bullish is not None
    assert bullish.protected_type is ProtectedSwingType.LOW

    bearish = after.current_assignment
    assert bearish is not None
    assert bearish.protected_type is ProtectedSwingType.HIGH
    assert bearish.assignment_id != bullish.assignment_id
    assert before.assignments == after.assignments[: len(before.assignments)]
