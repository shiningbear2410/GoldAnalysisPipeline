"""The precomputed-swing seam this round added, and what it refuses.

Round 6.6e.1 §2, §4, §31, §32. Two engines derived their own confirmed swings -
structure and liquidity - which is correct in isolation and wasteful in a
composite. The seam lets a caller hand over one collection instead.

Two properties matter and are tested separately. **Additivity**: omitting the
new argument must give byte-identical results to before, or every earlier
round's fixtures were quietly rewritten. **Refusal**: supplying the wrong
collection must fail rather than be repaired, because a swing set from another
timeframe, another pivot width or a later instant is not a smaller version of
the right answer - it is a different one.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.services.ict_liquidity import LiquidityConfig, analyse_liquidity
from goldpipeline.services.ict_primitives import (
    confirmed_swings,
    require_swings_for,
    swings_known_at,
)
from goldpipeline.services.ict_structure import analyse_structure
from tests.test_ict_liquidity_fixture import JOURNEY as LIQ_JOURNEY
from tests.test_ict_liquidity_fixture import TOLERANCE
from tests.test_ict_order_block_lifecycle_fixture import ROWS as OB_ROWS
from tests.test_ict_order_block_lifecycle_fixture import journey as ob_journey
from tests.test_ict_structure import START, series
from tests.test_ict_structure_fixture import JOURNEY as SPARSE

HOUR = timedelta(hours=1)
LIQUIDITY = LiquidityConfig(price_tolerance=Decimal(TOLERANCE))

PATHS = {"sparse": SPARSE, "liquidity": LIQ_JOURNEY, "order blocks": OB_ROWS}


# --------------------------------------------------------------------------
# §2: the seam is additive
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(PATHS))
def test_structure_without_the_argument_is_unchanged(name: str) -> None:
    """§2. The existing call and the seam call agree, on three real paths."""
    snapshot = series(PATHS[name])
    swings = confirmed_swings(snapshot)

    assert analyse_structure(snapshot, symbol="XAUUSD") == analyse_structure(
        snapshot, swings=swings, symbol="XAUUSD"
    )


@pytest.mark.parametrize("name", list(PATHS))
def test_liquidity_without_the_argument_is_unchanged(name: str) -> None:
    snapshot = series(PATHS[name])
    swings = confirmed_swings(snapshot)

    assert analyse_liquidity(snapshot, config=LIQUIDITY, symbol="XAUUSD") == analyse_liquidity(
        snapshot, config=LIQUIDITY, swings=swings, symbol="XAUUSD"
    )


def test_the_seam_agrees_at_every_instant_of_a_long_path() -> None:
    """§31. Not just at the end: the two paths agree bar by bar."""
    full = ob_journey()

    for kept in range(1, len(OB_ROWS) + 1, 5):
        moment = START + HOUR * kept
        working = series(OB_ROWS, bars_kept=kept)
        swings = confirmed_swings(working)

        assert analyse_structure(full, symbol="XAUUSD", as_of=moment) == analyse_structure(
            full, swings=swings, symbol="XAUUSD", as_of=moment
        )
        assert analyse_liquidity(
            full, config=LIQUIDITY, symbol="XAUUSD", as_of=moment
        ) == analyse_liquidity(full, config=LIQUIDITY, swings=swings, symbol="XAUUSD", as_of=moment)


def test_a_non_default_pivot_window_travels_through_the_seam() -> None:
    """The window is part of what makes a swing collection the right one."""
    snapshot = series(OB_ROWS)
    swings = confirmed_swings(snapshot, left_bars=3, right_bars=3)

    assert analyse_structure(
        snapshot, symbol="XAUUSD", left_bars=3, right_bars=3
    ) == analyse_structure(snapshot, swings=swings, symbol="XAUUSD", left_bars=3, right_bars=3)


def test_the_snapshot_wrappers_forward_the_argument() -> None:
    from goldpipeline.services.ict_liquidity import analyse_snapshot_liquidity
    from goldpipeline.services.ict_structure import analyse_snapshot_structure
    from tests.test_ict_composite_fixture import divergent_snapshot

    shot = divergent_snapshot()
    one = shot.require(Timeframe.H1)
    swings = confirmed_swings(one)

    assert analyse_snapshot_structure(
        shot, Timeframe.H1, swings=swings
    ) == analyse_snapshot_structure(shot, Timeframe.H1)
    assert analyse_snapshot_liquidity(
        shot, Timeframe.H1, config=LIQUIDITY, swings=swings
    ) == analyse_snapshot_liquidity(shot, Timeframe.H1, config=LIQUIDITY)


# --------------------------------------------------------------------------
# §4: truncation and filtering agree, which is why refusing is a choice
# --------------------------------------------------------------------------


def test_truncating_the_series_equals_filtering_the_swings() -> None:
    """The invariant that makes filtering *sound*, recorded even though the seam
    refuses instead.

    A pivot is confirmed by bars entirely at or before its ``confirmed_at``, so
    removing later bars can never create, destroy or alter one. Filtering would
    therefore have been safe; the seam still refuses, because filtering cannot
    also repair a wrong timeframe or a wrong pivot width.
    """
    full = ob_journey()
    everything = confirmed_swings(full)

    for kept in range(1, len(OB_ROWS) + 1, 3):
        moment = START + HOUR * kept
        truncated = confirmed_swings(series(OB_ROWS, bars_kept=kept))

        assert truncated == swings_known_at(everything, moment), f"disagreement at bar {kept - 1}"


# --------------------------------------------------------------------------
# §32: refusal
# --------------------------------------------------------------------------


def test_a_swing_confirmed_after_the_instant_is_refused() -> None:
    """The leak this exists to prevent, refused rather than trimmed."""
    full = ob_journey()
    everything = confirmed_swings(full)

    with pytest.raises(ValueError, match="after this analysis' instant"):
        analyse_structure(full, swings=everything, symbol="XAUUSD", as_of=START + HOUR * 20)

    with pytest.raises(ValueError, match="after this analysis' instant"):
        analyse_liquidity(
            full, config=LIQUIDITY, swings=everything, symbol="XAUUSD", as_of=START + HOUR * 20
        )


def test_swings_from_another_timeframe_are_refused() -> None:
    """§32. No cross-timeframe reuse, however equal the prices happen to be."""
    hourly = series(OB_ROWS)
    quarterly = series(OB_ROWS, timeframe=Timeframe.M15)
    foreign = confirmed_swings(quarterly)

    assert foreign, "an empty collection would prove nothing"
    with pytest.raises(ValueError, match="is for M15, not H1"):
        analyse_structure(hourly, swings=foreign, symbol="XAUUSD")


def test_swings_from_another_pivot_window_are_refused() -> None:
    """A 3/3 swing is not a 2/2 swing that happens to be rarer."""
    snapshot = series(OB_ROWS)
    wider = confirmed_swings(snapshot, left_bars=3, right_bars=3)

    with pytest.raises(ValueError, match="3/3 pivot window, not 2/2"):
        analyse_structure(snapshot, swings=wider, symbol="XAUUSD")

    with pytest.raises(ValueError, match="2/2 pivot window, not 3/3"):
        analyse_structure(
            snapshot, swings=confirmed_swings(snapshot), symbol="XAUUSD", left_bars=3, right_bars=3
        )


def test_an_out_of_order_collection_is_refused() -> None:
    """Consumers walk it chronologically, so the order is part of the contract."""
    snapshot = series(OB_ROWS)
    shuffled = list(reversed(confirmed_swings(snapshot)))

    with pytest.raises(ValueError, match="must ascend by pivot time"):
        analyse_structure(snapshot, swings=shuffled, symbol="XAUUSD")


def test_the_validator_returns_an_immutable_collection() -> None:
    snapshot = series(OB_ROWS)
    swings = confirmed_swings(snapshot)
    accepted = require_swings_for(
        swings,
        timeframe=Timeframe.H1,
        observed_at=snapshot.latest_closed_at,
        left_bars=2,
        right_bars=2,
    )

    assert isinstance(accepted, tuple)
    assert list(accepted) == swings


def test_an_empty_collection_is_accepted_and_means_no_swings() -> None:
    """A path with no confirmed pivot is a real state, not a missing argument."""
    snapshot = series(OB_ROWS, bars_kept=3)

    assert confirmed_swings(snapshot) == []
    assert analyse_structure(snapshot, swings=[], symbol="XAUUSD") == analyse_structure(
        snapshot, symbol="XAUUSD"
    )


def test_the_supplied_collection_is_not_mutated() -> None:
    """§33. The caller's list comes back as it went in."""
    snapshot = series(OB_ROWS)
    swings = confirmed_swings(snapshot)
    before = list(swings)

    analyse_structure(snapshot, swings=swings, symbol="XAUUSD")
    analyse_liquidity(snapshot, config=LIQUIDITY, swings=swings, symbol="XAUUSD")

    assert swings == before
