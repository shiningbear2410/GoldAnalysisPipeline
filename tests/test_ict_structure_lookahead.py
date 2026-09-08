"""What the engine knew, when it knew it, and whether it can be made to change its mind.

Round 6.6b §7, §9, §26, §27, §40. This is the file that matters most for
anything published later. Every other property here is about being right; these
are about being *honestly* right - an engine that quietly uses a bar from after
the moment it claims to describe will backtest beautifully and lose money, and
the failure is invisible unless something specifically looks for it.

The central assertion is the strongest one available: the state reconstructed
as of T from the full series must equal, field for field, the state computed
from a series that was physically cut at T. If any future bar leaked into an
earlier answer, those two cannot agree.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import IctMarketSnapshot, build_timeframe_snapshot
from goldpipeline.services.ict_structure import (
    StructureBias,
    analyse_snapshot_structure,
    analyse_structure,
)
from tests.test_ict_structure import (
    FLAT,
    START,
    Row,
    bar_index,
    drive,
    peak,
    series,
    trough,
)

DURATION = timedelta(hours=1)

JOURNEY: list[Row] = [
    FLAT, FLAT, peak("4030"), FLAT, FLAT, trough("3970"), FLAT, FLAT,
    drive("4035"), drive("4035"), FLAT, FLAT,
    drive("3965"), drive("3965"), FLAT, FLAT,
    peak("4020"), FLAT, FLAT, drive("4025"), drive("4025"), FLAT, FLAT,
]  # fmt: skip
"""Bullish initial break, bearish MSS, bullish MSS. Verified in the module below."""


def closes_of(rows: list[Row]) -> list[object]:
    return [START + DURATION * (index + 1) for index in range(len(rows))]


# --------------------------------------------------------------------------
# §40 A-C: a pivot is not a level until it is confirmed
# --------------------------------------------------------------------------


def test_a_pivot_whose_right_window_has_not_closed_is_not_a_level() -> None:
    """Geometrically obvious on the chart; not yet knowable to the engine.

    Bar 2 is the highest bar in the series from bar 4 onwards. It becomes a
    confirmed swing only when bar 4 closes, and until then it cannot be an
    active level and cannot be broken.
    """
    rows = [FLAT, FLAT, peak("4030"), FLAT, FLAT]

    early = analyse_structure(series(rows, bars_kept=4))
    assert early.annotated_swings == ()
    assert early.active_high is None

    late = analyse_structure(series(rows, bars_kept=5))
    assert late.active_high is not None
    assert late.active_high.swing.price == Decimal("4030")


def test_an_unconfirmed_pivot_cannot_be_broken() -> None:
    """A close beyond a shape that is not yet a level is not a structure break."""
    rows = [FLAT, FLAT, peak("4030"), FLAT, drive("4035"), drive("4035"), FLAT, FLAT]
    result = analyse_structure(series(rows))

    beyond = [entry for entry in result.annotated_swings if entry.swing.price == Decimal("4030")]
    assert beyond == [], "bar 4's high of 4035 removed bar 2's claim to being a pivot"
    assert result.breaks == ()


def test_the_bar_after_confirmation_can_break_the_level() -> None:
    rows = [FLAT, FLAT, peak("4030"), FLAT, FLAT, drive("4035"), drive("4035"), FLAT, FLAT]
    (event,) = analyse_structure(series(rows)).breaks

    assert bar_index(event.break_bar_open_time) == 5
    assert event.broken_swing_confirmed_at == START + DURATION * 5
    assert event.break_bar_close_time == START + DURATION * 6
    assert event.broken_swing_confirmed_at < event.break_bar_close_time


def test_every_event_ever_emitted_respects_the_known_before_broken_rule() -> None:
    """An invariant over the whole journey rather than one hand-picked event."""
    result = analyse_structure(series(JOURNEY))

    assert result.breaks, "a vacuous pass here would be worse than a failure"
    for event in result.breaks:
        assert event.broken_swing_confirmed_at < event.break_bar_close_time


# --------------------------------------------------------------------------
# §26, §40 D: as-of reconstruction
# --------------------------------------------------------------------------


def test_as_of_equals_physical_truncation_at_every_bar() -> None:
    """The whole no-lookahead guarantee, in one assertion, at every instant.

    Field-for-field equality of the two analyses. If a swing confirmed after T
    ever reached an answer dated T - as a relation, an active level, an event,
    a consumed id or the state itself - these would differ.
    """
    full = series(JOURNEY)

    for kept in range(1, len(JOURNEY) + 1):
        moment = START + DURATION * kept
        truncated = analyse_structure(series(JOURNEY, bars_kept=kept))
        as_of = analyse_structure(full, as_of=moment)

        assert as_of == truncated, f"disagreement at bar {kept - 1}"


def test_an_as_of_query_reports_the_instant_it_was_asked_about() -> None:
    full = series(JOURNEY)
    moment = START + DURATION * 10

    result = analyse_structure(full, as_of=moment)

    assert result.observed_at == moment
    assert result.bars_considered == 10


def test_an_as_of_before_any_bar_closed_is_refused_not_answered_neutral() -> None:
    """ "No data" and "no structure" would otherwise be the same answer."""
    with pytest.raises(ValueError, match="no bar closed"):
        analyse_structure(series(JOURNEY), as_of=START)


def test_a_later_mss_does_not_rewrite_an_earlier_bullish_state() -> None:
    """§40 E. The bearish MSS at bar 12 must not reach back to bar 10."""
    full = series(JOURNEY)

    before = analyse_structure(full, as_of=START + DURATION * 11)
    after = analyse_structure(full, as_of=START + DURATION * 13)

    assert before.current_bias is StructureBias.BULLISH
    assert after.current_bias is StructureBias.BEARISH
    assert [event.classification.value for event in before.breaks] == ["INITIAL_BREAK"]
    assert [event.classification.value for event in after.breaks] == ["INITIAL_BREAK", "MSS"]


def test_a_later_swing_does_not_become_an_earlier_active_level() -> None:
    """§40 F. The 4020 pivot at bar 16 is not in play at bar 11."""
    full = series(JOURNEY)

    early = analyse_structure(full, as_of=START + DURATION * 11)
    late = analyse_structure(full, as_of=START + DURATION * 19)

    assert early.active_high is None
    assert late.active_high is not None
    assert late.active_high.swing.price == Decimal("4020")


def test_the_events_known_at_t_are_a_prefix_of_the_events_known_later() -> None:
    """History accumulates. It never gets edited."""
    full = series(JOURNEY)
    seen: list[tuple[str, ...]] = []

    for kept in range(1, len(JOURNEY) + 1):
        result = analyse_structure(full, as_of=START + DURATION * kept)
        seen.append(tuple(event.event_id for event in result.breaks))

    for earlier, later in zip(seen, seen[1:], strict=False):
        assert later[: len(earlier)] == earlier


# --------------------------------------------------------------------------
# §27: replay determinism
# --------------------------------------------------------------------------


def snapshot_with(provider: str, provider_symbol: str | None) -> IctMarketSnapshot:
    observed = START + DURATION * len(JOURNEY)
    return IctMarketSnapshot(
        observed_at=observed,
        symbol="XAUUSD",
        provider=provider,
        provider_symbol=provider_symbol,
        timeframes=(
            build_timeframe_snapshot(
                timeframe=Timeframe.H1,
                bars=series(JOURNEY).bars,
                observed_at=observed,
            ),
        ),
    )


def test_the_provider_changes_nothing_about_the_structure() -> None:
    """Provenance is recorded, never consulted. The candles are the candles."""
    one = analyse_snapshot_structure(snapshot_with("tradingview", "OANDA:XAUUSD"), Timeframe.H1)
    two = analyse_snapshot_structure(snapshot_with("metatrader", "XAUUSD.pro"), Timeframe.H1)
    three = analyse_snapshot_structure(snapshot_with("fixture", None), Timeframe.H1)

    assert one == two == three


def test_repeated_analysis_gives_an_identical_result_object() -> None:
    first = analyse_structure(series(JOURNEY), symbol="XAUUSD")
    second = analyse_structure(series(JOURNEY), symbol="XAUUSD")

    assert first == second


REPLAY_PROGRAM = """
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from tests.test_ict_structure import series
from tests.test_ict_structure_lookahead import JOURNEY
from goldpipeline.services.ict_structure import analyse_structure

result = analyse_structure(series(JOURNEY), symbol="XAUUSD")
print(result.current_bias.value)
print(result.initial_bias.value)
for entry in result.annotated_swings:
    print("SWING", entry.swing_id, entry.label, entry.swing.price)
for event in result.breaks:
    print("EVENT", event.event_id, event.classification.value, event.direction.value,
          event.broken_swing_id, event.broken_level, event.also_consumed_swing_ids)
print("ACTIVE", result.active_high, result.active_low)
print("CONSUMED", sorted(result.consumed_swing_ids))
"""


@pytest.mark.parametrize("seed", ["0", "1", "524287"])
def test_string_hash_ordering_cannot_reach_the_output(seed: str) -> None:
    """§39. Set and dict iteration order is not allowed to be an input.

    Run in a subprocess because ``PYTHONHASHSEED`` is read once at interpreter
    start. Python randomises string hashing per process by default, so a set
    that leaked into the output would give different answers on different seeds
    - which is precisely the kind of defect that survives a thousand passing
    runs and then fails in production once.
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
    assert "EVENT" in baseline, "a fixture with no events would prove nothing"
