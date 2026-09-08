"""Whether a future candle can reach back and change what liquidity used to be.

Round 6.6c.1 §35-§37, §47-§48. The same job Round 6.6b's lookahead file did for
structure, and for the same reason: a pool that quietly appears earlier than it
could have known to, or a sweep that back-dates itself, produces a backtest that
looks excellent and a live system that does not.

The central assertion is again the strongest available - the state reconstructed
as of T from the whole series must equal, field for field, the state computed
from a series physically cut at T.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import IctMarketSnapshot, build_timeframe_snapshot
from goldpipeline.services.ict_liquidity import (
    PoolStatus,
    analyse_liquidity,
    analyse_snapshot_liquidity,
)
from tests.test_ict_liquidity import (
    FLAT,
    START,
    Row,
    analyse,
    bar,
    bar_index,
    config,
    peak,
    series,
    trough,
)

HOUR = timedelta(hours=1)
TOLERANCE = "0.50"

PATH: list[Row] = [
    FLAT, FLAT, peak("4030.00"), FLAT, FLAT, FLAT, peak("4030.20"), FLAT, FLAT,
    bar("4035", "3990", "4005"), FLAT, FLAT,
    trough("3970"), FLAT, FLAT, FLAT, trough("3970"), FLAT, FLAT,
    bar("4010", "3960", "3965"), FLAT, FLAT,
    peak("4030.00"), FLAT, FLAT, FLAT, peak("4030.10"), FLAT, FLAT,
]  # fmt: skip
"""A buy-side pool swept, a sell-side pool closed through, and a third pool left
active. Enough shape that a leak in any direction would show."""


# --------------------------------------------------------------------------
# §47 A-C: a pool is not liquidity until its second founder confirms
# --------------------------------------------------------------------------


def test_a_second_swing_that_exists_but_has_not_confirmed_makes_no_pool() -> None:
    """Obvious on a chart; not yet knowable to the engine.

    The second 4030 pivot is bar 6. It becomes a confirmed swing only when bar 8
    closes, and the pool cannot exist before that.
    """
    rows = [FLAT, FLAT, peak("4030"), FLAT, FLAT, FLAT, peak("4030"), FLAT, FLAT]

    for kept in range(1, 9):
        assert (
            analyse_liquidity(
                series(rows, bars_kept=kept), config=config("0"), symbol="XAUUSD"
            ).pools
            == ()
        ), f"a pool appeared at bar {kept - 1}"

    formed = analyse_liquidity(series(rows, bars_kept=9), config=config("0"), symbol="XAUUSD")
    assert len(formed.pools) == 1
    assert bar_index(formed.pools[0].formed_at) == 8


def test_the_bar_that_forms_a_pool_cannot_also_take_it() -> None:
    """§23 end to end, as far as it can be shown.

    At the instant the pool is formed it exists and is untouched. Whether a bar
    could ever *both* form and take a pool is settled at the predicate in
    ``test_ict_liquidity`` - with strict-both-sides pivots it cannot arise,
    because that bar sits inside the founding swing's right-hand window.
    """
    rows = [FLAT, FLAT, peak("4030"), FLAT, FLAT, FLAT, peak("4030"), FLAT, FLAT]
    formed = analyse_liquidity(series(rows, bars_kept=9), config=config("0"), symbol="XAUUSD")

    (pool,) = formed.pools
    assert pool.status is PoolStatus.ACTIVE
    assert formed.events == ()
    assert pool.formed_at == formed.observed_at


def test_the_next_closed_candle_can_take_it() -> None:
    rows = [FLAT, FLAT, peak("4030"), FLAT, FLAT, FLAT, peak("4030"), FLAT, FLAT,
            bar("4035", "3990", "4005")]  # fmt: skip
    later = analyse_liquidity(series(rows), config=config("0"), symbol="XAUUSD")

    (event,) = later.events
    assert later.pools[0].status is PoolStatus.SWEPT
    assert event.event_bar_close_time == later.pools[0].formed_at + HOUR


# --------------------------------------------------------------------------
# §35, §47 D-F: as-of reconstruction
# --------------------------------------------------------------------------


def test_as_of_equals_physical_truncation_at_every_bar() -> None:
    """The whole no-lookahead guarantee, at every instant in the path.

    Field-for-field equality of the two analyses: pool identities, memberships,
    bands, statuses, events and the swings still waiting. A future founder, a
    future third member or a future sweep leaking into an earlier answer would
    break one of them.
    """
    full = series(PATH)

    for kept in range(1, len(PATH) + 1):
        moment = START + HOUR * kept
        truncated = analyse_liquidity(
            series(PATH, bars_kept=kept), config=config(TOLERANCE), symbol="XAUUSD"
        )
        as_of = analyse_liquidity(full, config=config(TOLERANCE), symbol="XAUUSD", as_of=moment)

        assert as_of == truncated, f"disagreement at bar {kept - 1}"


def test_an_as_of_query_reports_the_instant_it_was_asked_about() -> None:
    result = analyse_liquidity(
        series(PATH), config=config(TOLERANCE), symbol="XAUUSD", as_of=START + HOUR * 10
    )

    assert result.observed_at == START + HOUR * 10
    assert result.bars_considered == 10


def test_an_as_of_before_any_bar_closed_is_refused() -> None:
    with pytest.raises(ValueError, match="no bar closed"):
        analyse_liquidity(series(PATH), config=config(TOLERANCE), symbol="XAUUSD", as_of=START)


def test_a_future_third_member_does_not_widen_an_earlier_band() -> None:
    """§47 D. The band as of T is the band the market had built by T."""
    rows = [
        FLAT, FLAT, peak("4030.00"), FLAT, FLAT, FLAT, peak("4030.40"), FLAT, FLAT, FLAT,
        peak("4030.20"), FLAT, FLAT,
    ]  # fmt: skip
    full = series(rows)

    early = analyse_liquidity(
        full, config=config(TOLERANCE), symbol="XAUUSD", as_of=START + HOUR * 10
    )
    late = analyse_liquidity(full, config=config(TOLERANCE), symbol="XAUUSD")

    assert len(early.pools[0].member_swing_ids) == 2
    assert len(late.pools[0].member_swing_ids) == 3
    assert early.pools[0].pool_id == late.pools[0].pool_id, "identity is fixed at formation"
    assert (early.pools[0].lower, early.pools[0].upper) == (
        late.pools[0].lower,
        late.pools[0].upper,
    )


def test_a_future_sweep_does_not_terminalise_an_earlier_pool() -> None:
    """§47 E."""
    full = series(PATH)

    before = analyse_liquidity(
        full, config=config(TOLERANCE), symbol="XAUUSD", as_of=START + HOUR * 9
    )
    after = analyse_liquidity(
        full, config=config(TOLERANCE), symbol="XAUUSD", as_of=START + HOUR * 11
    )

    assert before.pools[0].status is PoolStatus.ACTIVE
    assert before.pools[0].terminal_event_id is None
    assert before.events == ()
    assert after.pools[0].status is PoolStatus.SWEPT
    assert after.pools[0].terminal_event_id is not None


def test_event_history_is_append_only() -> None:
    """§47 G. History accumulates; it never gets edited."""
    full = series(PATH)
    seen: list[tuple[str, ...]] = []

    for kept in range(1, len(PATH) + 1):
        result = analyse_liquidity(
            full, config=config(TOLERANCE), symbol="XAUUSD", as_of=START + HOUR * kept
        )
        seen.append(tuple(event.event_id for event in result.events))

    assert seen[-1], "a fixture with no events would prove nothing"
    for earlier, later in zip(seen, seen[1:], strict=False):
        assert later[: len(earlier)] == earlier


def test_pool_identities_are_stable_as_history_grows() -> None:
    full = series(PATH)
    first_seen: dict[str, tuple[str, str]] = {}

    for kept in range(1, len(PATH) + 1):
        result = analyse_liquidity(
            full, config=config(TOLERANCE), symbol="XAUUSD", as_of=START + HOUR * kept
        )
        for pool in result.pools:
            band = (str(pool.lower), str(pool.upper))
            if pool.pool_id in first_seen:
                # A band may widen when a member joins, but never shrink, and
                # the identity may never be reused for a different founding pair.
                previous = first_seen[pool.pool_id]
                assert Decimal(band[0]) <= Decimal(previous[0])
                assert Decimal(band[1]) >= Decimal(previous[1])
            first_seen[pool.pool_id] = band


# --------------------------------------------------------------------------
# §36, §48: replay determinism
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
    one = analyse_snapshot_liquidity(
        snapshot_with("tradingview", "OANDA:XAUUSD"), Timeframe.H1, config=config(TOLERANCE)
    )
    two = analyse_snapshot_liquidity(
        snapshot_with("metatrader", "XAUUSD.pro"), Timeframe.H1, config=config(TOLERANCE)
    )
    three = analyse_snapshot_liquidity(
        snapshot_with("fixture", None), Timeframe.H1, config=config(TOLERANCE)
    )

    assert one == two == three


def test_repeated_analysis_gives_an_identical_result() -> None:
    assert analyse(PATH, TOLERANCE) == analyse(PATH, TOLERANCE)


def test_a_different_tolerance_gives_different_pool_identities() -> None:
    """The policy is part of what a pool *is*, so it is part of its identity."""
    tight = analyse(PATH, "0.10")
    loose = analyse(PATH, "0.50")

    assert {pool.pool_id for pool in tight.pools}.isdisjoint(pool.pool_id for pool in loose.pools)


def test_a_different_symbol_gives_different_identities() -> None:
    gold = analyse_liquidity(series(PATH), config=config(TOLERANCE), symbol="XAUUSD")
    other = analyse_liquidity(series(PATH), config=config(TOLERANCE), symbol="XAGUSD")

    assert [pool.pool_id for pool in gold.pools] != [pool.pool_id for pool in other.pools]
    assert [(pool.lower, pool.upper) for pool in gold.pools] == [
        (pool.lower, pool.upper) for pool in other.pools
    ]


REPLAY_PROGRAM = """
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from decimal import Decimal
from tests.test_ict_liquidity import series
from tests.test_ict_liquidity_lookahead import PATH
from goldpipeline.services.ict_liquidity import LiquidityConfig, analyse_liquidity

result = analyse_liquidity(
    series(PATH), config=LiquidityConfig(price_tolerance=Decimal("0.50")), symbol="XAUUSD"
)
for pool in result.pools:
    print("POOL", pool.pool_id, pool.side.value, pool.lower, pool.upper, pool.midpoint,
          pool.status.value, pool.member_swing_ids, pool.terminal_event_id)
for event in result.events:
    print("EVENT", event.event_id, event.event_type.value, event.side.value, event.boundary,
          event.event_bar_close_time.isoformat())
print("ACTIVE", result.active_pool_ids)
print("TERMINAL", result.terminal_pool_ids)
print("UNPAIRED", result.unpaired_swing_ids)
"""


@pytest.mark.parametrize("seed", ["0", "1", "524287"])
def test_string_hash_ordering_cannot_reach_the_output(seed: str) -> None:
    """§36. Set and dict iteration order is not allowed to be an input.

    This engine keeps pools in a list and groups swings by confirmation time in
    a dict, so the risk is real rather than theoretical. Run in a subprocess
    because ``PYTHONHASHSEED`` is read once at interpreter start.
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
    assert "EVENT" in baseline and "POOL" in baseline, "a silent empty run would prove nothing"
