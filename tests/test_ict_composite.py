"""The composite's own contract, on small snapshots.

Round 6.6e.1 §15-§20, §32, §37. The fixture file pins what the five-timeframe
snapshot says; this file checks the rules that hold whatever the market did -
that every nested reading is dated at the snapshot's instant, that the policy
travels with the result, that a mismatched symbol fails closed, and that a
subset snapshot is analysed as it stands.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import ICT_TIMEFRAMES, IctMarketSnapshot
from goldpipeline.services.ict_composite import (
    COMPOSITE_METHOD_VERSION,
    IctCompositeError,
    analyse_ict_composite,
    analyse_ict_timeframe,
)
from tests.test_ict_composite_fixture import (
    OBSERVED_AT,
    config,
    divergent_snapshot,
)
from tests.test_ict_order_block_lifecycle_fixture import journey as h1_journey
from tests.test_ict_structure import START

HOUR = timedelta(hours=1)


# --------------------------------------------------------------------------
# §15: the policy travels with the result
# --------------------------------------------------------------------------


def test_the_config_is_reachable_from_the_result() -> None:
    """§15. Six questions a consumer can answer without inspecting geometry."""
    result = analyse_ict_composite(divergent_snapshot(), config=config(left=3, right=4, atr=21))

    assert result.config.swing_left_bars == 3
    assert result.config.swing_right_bars == 4
    assert result.config.atr_period == 21
    assert result.config.liquidity_price_tolerance == Decimal("0.50")
    assert result.config.order_block_zone_basis.value == "FULL_CANDLE"
    assert result.config.order_block_mitigation_rule.value == "TOUCH"


def test_the_policy_actually_reached_the_engines() -> None:
    """§15. Not merely recorded - the ATR period and windows were used."""
    result = analyse_ict_composite(divergent_snapshot(), config=config(atr=5))
    other = analyse_ict_composite(divergent_snapshot(), config=config(atr=20))
    h1 = result.require(Timeframe.H1)

    assert h1.atr is not None
    assert h1.atr.period == 5
    assert other.require(Timeframe.H1).atr is not None
    assert other.require(Timeframe.H1).atr.period == 20  # type: ignore[union-attr]


def test_the_swing_window_reaches_the_swings_themselves() -> None:
    result = analyse_ict_composite(divergent_snapshot(), config=config(left=3, right=3))
    h1 = result.require(Timeframe.H1)

    assert h1.swings
    assert {(s.left_bars, s.right_bars) for s in h1.swings} == {(3, 3)}


def test_the_method_version_is_stamped_everywhere() -> None:
    result = analyse_ict_composite(divergent_snapshot(), config=config())

    assert result.method_version == COMPOSITE_METHOD_VERSION
    for entry in result.timeframes:
        assert entry.method_version == COMPOSITE_METHOD_VERSION


def test_provenance_is_carried_not_invented() -> None:
    shot = divergent_snapshot()
    result = analyse_ict_composite(shot, config=config())

    assert (result.symbol, result.provider, result.provider_symbol) == (
        shot.symbol,
        shot.provider,
        shot.provider_symbol,
    )
    assert result.observed_at == shot.observed_at


# --------------------------------------------------------------------------
# §19-§20: one instant, one symbol
# --------------------------------------------------------------------------


def test_the_per_timeframe_entry_point_dates_itself_at_its_last_close() -> None:
    """Called alone with no *as_of*, it uses the series' own newest close."""
    series = h1_journey()
    entry = analyse_ict_timeframe(series, config=config(), symbol="XAUUSD")

    assert entry.observed_at == series.latest_closed_at
    assert entry.structure.observed_at == series.latest_closed_at


def test_the_per_timeframe_entry_point_honours_an_explicit_instant() -> None:
    series = h1_journey()
    moment = START + HOUR * 30
    entry = analyse_ict_timeframe(series, config=config(), symbol="XAUUSD", as_of=moment)

    assert entry.observed_at == moment
    for nested in (
        entry.structure,
        entry.liquidity,
        entry.protected,
        entry.dealing_ranges,
        entry.order_blocks,
        entry.order_block_lifecycle,
        entry.fvg_lifecycle,
    ):
        assert nested.observed_at == moment


def test_the_series_held_is_the_truncated_one() -> None:
    """The bars every nested analysis actually saw, not the ones handed in."""
    series = h1_journey()
    moment = START + HOUR * 30
    entry = analyse_ict_timeframe(series, config=config(), symbol="XAUUSD", as_of=moment)

    assert entry.series.bar_count == 30
    assert entry.series.latest_closed_at == moment
    assert series.bar_count == 63, "the caller's snapshot is untouched"


def test_the_symbol_reaches_every_nested_analysis() -> None:
    entry = analyse_ict_timeframe(h1_journey(), config=config(), symbol="XAUUSD")

    assert entry.symbol == "XAUUSD"
    for nested in (
        entry.structure,
        entry.liquidity,
        entry.protected,
        entry.dealing_ranges,
        entry.order_blocks,
        entry.order_block_lifecycle,
        entry.fvg_lifecycle,
    ):
        assert nested.symbol == "XAUUSD"


def test_a_nested_analysis_dated_elsewhere_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§19. The threading should make this impossible, which is why it is checked.

    A silent disagreement would be almost invisible: every number would look
    plausible and only the dates would be wrong.

    The gap lifecycle is drifted rather than structure, because structure feeds
    three stages that each validate what they are handed - so a drifted
    structure never reaches the composite's own check. The gap lifecycle feeds
    nothing, which makes it the one place where only this check stands between a
    misdated reading and the caller.
    """
    from dataclasses import replace

    from goldpipeline.services import ict_composite

    # Reached through the module namespace on purpose: this is an import the
    # composite uses, not part of the surface it exports.
    original: Any = vars(ict_composite)["analyse_fvg_lifecycle"]

    def drifting(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        return replace(result, observed_at=result.observed_at + HOUR)

    monkeypatch.setattr(ict_composite, "analyse_fvg_lifecycle", drifting)

    with pytest.raises(IctCompositeError, match="but the snapshot is"):
        analyse_ict_composite(divergent_snapshot(), config=config())


def test_a_drifted_structure_is_caught_by_the_stage_that_consumes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§19, and the reason the composite's check is a backstop rather than the only one.

    Each threaded analysis is validated by whichever stage receives it, so a
    misdated structure is refused before it can reach a result at all.
    """
    from dataclasses import replace

    from goldpipeline.services import ict_composite
    from goldpipeline.services.ict_protected import ProtectedStructureError

    # Reached through the module namespace on purpose: this is an import the
    # composite uses, not part of the surface it exports.
    original: Any = vars(ict_composite)["analyse_structure"]

    def drifting(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        return replace(result, observed_at=result.observed_at + HOUR)

    monkeypatch.setattr(ict_composite, "analyse_structure", drifting)

    with pytest.raises(ProtectedStructureError, match="supply one computed for the same instant"):
        analyse_ict_composite(divergent_snapshot(), config=config())


def test_a_nested_analysis_for_another_symbol_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§20. Refused, never silently rewritten."""
    from dataclasses import replace

    from goldpipeline.services import ict_composite

    # Reached through the module namespace on purpose: this is an import the
    # composite uses, not part of the surface it exports.
    original: Any = vars(ict_composite)["analyse_liquidity"]

    def foreign(*args: Any, **kwargs: Any) -> Any:
        return replace(original(*args, **kwargs), symbol="EURUSD")

    monkeypatch.setattr(ict_composite, "analyse_liquidity", foreign)

    with pytest.raises(IctCompositeError, match="is for 'EURUSD', not 'XAUUSD'"):
        analyse_ict_composite(divergent_snapshot(), config=config())


# --------------------------------------------------------------------------
# §37: what a snapshot may and may not hold
# --------------------------------------------------------------------------


def test_a_single_timeframe_snapshot_is_a_valid_input() -> None:
    shot = divergent_snapshot()
    one = IctMarketSnapshot(
        observed_at=OBSERVED_AT,
        symbol="XAUUSD",
        provider="fixture",
        provider_symbol=None,
        timeframes=(shot.require(Timeframe.M15),),
    )
    result = analyse_ict_composite(one, config=config())

    assert len(result.timeframes) == 1
    assert result.timeframes[0].timeframe is Timeframe.M15
    assert len(result.timeframes[0].liquidity.pools) == 5


def test_a_snapshot_with_no_timeframes_is_refused_at_construction() -> None:
    """§37. Not the composite's to repair - the snapshot type already says no."""
    with pytest.raises(ValueError, match="at least 1 item|too_short|min_length"):
        IctMarketSnapshot(
            observed_at=OBSERVED_AT,
            symbol="XAUUSD",
            provider="fixture",
            provider_symbol=None,
            timeframes=(),
        )


def test_lookups_are_by_identity_not_by_position() -> None:
    result = analyse_ict_composite(divergent_snapshot(), config=config())

    for entry in result.timeframes:
        assert result.timeframe(entry.timeframe) is entry
        assert result.require(entry.timeframe) is entry


def test_a_missing_timeframe_returns_none_before_it_raises() -> None:
    shot = divergent_snapshot()
    one = IctMarketSnapshot(
        observed_at=OBSERVED_AT,
        symbol="XAUUSD",
        provider="fixture",
        provider_symbol=None,
        timeframes=(shot.require(Timeframe.H1),),
    )
    result = analyse_ict_composite(one, config=config())

    assert result.timeframe(Timeframe.H4) is None
    with pytest.raises(IctCompositeError, match="holds no H4 analysis"):
        result.require(Timeframe.H4)


# --------------------------------------------------------------------------
# §16: what the per-timeframe result holds
# --------------------------------------------------------------------------


def test_the_swings_and_gaps_held_are_the_ones_the_stages_used() -> None:
    """§3, §5. One collection, visible on the result and fed to the consumers."""
    from goldpipeline.services.ict_primitives import confirmed_swings, fair_value_gaps

    series = h1_journey()
    entry = analyse_ict_timeframe(series, config=config(), symbol="XAUUSD")

    assert list(entry.swings) == confirmed_swings(series, left_bars=2, right_bars=2)
    assert list(entry.gaps) == fair_value_gaps(series)
    assert [state.gap for state in entry.fvg_lifecycle.states] == list(entry.gaps)


def test_the_collections_are_immutable_tuples() -> None:
    entry = analyse_ict_timeframe(h1_journey(), config=config(), symbol="XAUUSD")

    assert isinstance(entry.swings, tuple)
    assert isinstance(entry.gaps, tuple)


def test_a_timeframe_too_short_for_the_atr_reports_none() -> None:
    """A real state, not an error and not a zero."""
    entry = analyse_ict_composite(divergent_snapshot(), config=config()).require(Timeframe.M5)

    assert entry.series.bar_count == 8
    assert entry.atr is None


def test_a_long_enough_timeframe_reports_an_atr_for_its_last_bar() -> None:
    entry = analyse_ict_composite(divergent_snapshot(), config=config()).require(Timeframe.H1)

    assert entry.atr is not None
    assert entry.atr.timeframe is Timeframe.H1
    assert entry.atr.bar_close_time == entry.series.latest_closed_at


def test_the_composite_holds_no_more_timeframes_than_the_snapshot_did() -> None:
    result = analyse_ict_composite(divergent_snapshot(), config=config())

    assert len(result.timeframes) == len(ICT_TIMEFRAMES)
    assert len({entry.timeframe for entry in result.timeframes}) == len(result.timeframes)


def test_a_snapshot_cannot_hold_a_bar_that_closes_after_its_observation() -> None:
    """The composite never has to check this: the snapshot type already refuses it.

    Worth pinning here anyway, because it is what makes ``analyse_ict_composite``
    safe to write without an instant check of its own on the *bars* - a snapshot
    that reached this module has already proved every candle in it had closed.
    """
    shot = divergent_snapshot()

    with pytest.raises(ValueError, match="after the observation"):
        IctMarketSnapshot(
            observed_at=datetime(2020, 1, 1, tzinfo=UTC),
            symbol="XAUUSD",
            provider="fixture",
            provider_symbol=None,
            timeframes=(shot.require(Timeframe.H1),),
        )


def test_an_observation_before_the_series_begins_is_refused() -> None:
    """The per-timeframe entry point, asked about an instant with no closed bar."""
    with pytest.raises(ValueError, match="no bar closed"):
        analyse_ict_timeframe(h1_journey(), config=config(), symbol="XAUUSD", as_of=START - HOUR)
