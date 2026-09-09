"""Five timeframes, five genuinely different markets, one policy bundle.

Round 6.6e.1 §23-§28, §36, §38, §41-§42. This file pins what the composite
reports over a snapshot assembled from the branch's own hardest paths, so a
later round changing an engine has to change these numbers on purpose.

Every number below was read out of the engine before it was written down.

Each timeframe carries a fixture that already earned its keep in an earlier
round, which is what makes the snapshot cover everything §41 asks for without
inventing a sixth market:

* **H4** - Round 6.6c.3b's bearish path. Bearish structure, one protected leg,
  one dealing range, one order block. Too short for a 14-period ATR, which is a
  real state and is pinned as ``None``.
* **H1** - Round 6.6d.2's 63-bar order-block journey. Seven structure events,
  five order blocks across three lifecycle statuses, three fair value gaps.
* **M15** - Round 6.6c.1's liquidity journey. Five pools, four swept and one
  closed through, five events - and no protected leg at all, because its single
  structure event anchored nothing.
* **M5** - the quiet path. ``NEUTRAL``, no events, no legs, no ranges, no
  blocks. A timeframe with nothing to say is a legitimate reading.
* **M1** - Round 6.6c.2's fair-value-gap journey, whose ``(high, low)`` rows are
  widened to ``(high, low, close)`` with the close at the range midpoint, which
  is exactly what that fixture's own builder does. Thirteen gaps across all
  three lifecycle statuses.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import ICT_TIMEFRAMES, IctMarketSnapshot
from goldpipeline.services.ict_composite import (
    IctCompositeAnalysis,
    IctCompositeConfig,
    IctTimeframeAnalysis,
    analyse_ict_composite,
)
from goldpipeline.services.ict_order_block import OrderBlockZoneBasis
from goldpipeline.services.ict_order_block_lifecycle import OrderBlockMitigationRule
from tests.test_ict_fvg_fixture import JOURNEY as FVG_ROWS
from tests.test_ict_liquidity_fixture import JOURNEY as LIQ_ROWS
from tests.test_ict_liquidity_fixture import TOLERANCE
from tests.test_ict_order_block import reopen
from tests.test_ict_order_block_fixture import OPENS
from tests.test_ict_order_block_lifecycle_fixture import ROWS as OB_ROWS
from tests.test_ict_range import BEAR
from tests.test_ict_range_fixture import QUIET, ending_at
from tests.test_ict_structure import Row

OBSERVED_AT = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


def widened(rows: list[tuple[str, str]]) -> list[Row]:
    """Round 6.6c.2's ``(high, low)`` rows, with the close at the midpoint.

    The same close that fixture's own builder derives, so the gap journey is
    reproduced rather than approximated - fair value gaps are decided by highs
    and lows alone, and the close only matters to the structure engine, which
    on this timeframe is free to say whatever the path says.
    """
    return [(high, low, str((Decimal(high) + Decimal(low)) / Decimal(2))) for high, low in rows]


PATHS: dict[Timeframe, list[Row]] = {
    Timeframe.H4: BEAR,
    Timeframe.H1: OB_ROWS,
    Timeframe.M15: LIQ_ROWS,
    Timeframe.M5: QUIET,
    Timeframe.M1: widened(FVG_ROWS),
}

TF_OPENS: dict[Timeframe, dict[int, str]] = {
    Timeframe.H1: OPENS,
    Timeframe.H4: {7: "3990"},
}
"""The body reshaping Rounds 6.6d.1 and 6.6d.2 needed, carried across."""


def divergent_snapshot() -> IctMarketSnapshot:
    return IctMarketSnapshot(
        observed_at=OBSERVED_AT,
        symbol="XAUUSD",
        provider="fixture",
        provider_symbol=None,
        timeframes=tuple(
            reopen(ending_at(tf, PATHS[tf], end=OBSERVED_AT), TF_OPENS.get(tf, {}))
            for tf in ICT_TIMEFRAMES
        ),
    )


def config(
    *,
    basis: OrderBlockZoneBasis = OrderBlockZoneBasis.FULL_CANDLE,
    rule: OrderBlockMitigationRule = OrderBlockMitigationRule.TOUCH,
    left: int = 2,
    right: int = 2,
    atr: int = 14,
    tolerance: str = TOLERANCE,
) -> IctCompositeConfig:
    """A fully-stated config. The defaults live here, in a test, never in ``src``."""
    return IctCompositeConfig(
        swing_left_bars=left,
        swing_right_bars=right,
        atr_period=atr,
        liquidity_price_tolerance=Decimal(tolerance),
        order_block_zone_basis=basis,
        order_block_mitigation_rule=rule,
    )


def composite(**kwargs: object) -> IctCompositeAnalysis:
    return analyse_ict_composite(divergent_snapshot(), config=config(**kwargs))  # type: ignore[arg-type]


RULES = tuple(OrderBlockMitigationRule)
BASES = tuple(OrderBlockZoneBasis)


# --------------------------------------------------------------------------
# §41: the whole snapshot, pinned
# --------------------------------------------------------------------------


def test_the_composite_holds_five_timeframes_in_branch_order() -> None:
    """§18. ``ICT_TIMEFRAMES`` order, not dict or hash order."""
    assert [entry.timeframe for entry in composite().timeframes] == list(ICT_TIMEFRAMES)


def test_the_journey_pins_every_structure_reading() -> None:
    assert [
        (
            entry.timeframe.value,
            entry.series.bar_count,
            len(entry.swings),
            entry.structure.current_bias.value,
            len(entry.structure.breaks),
        )
        for entry in composite().timeframes
    ] == [
        ("H4", 12, 2, "BEARISH", 1),
        ("H1", 63, 9, "BEARISH", 7),
        ("M15", 54, 14, "BULLISH", 1),
        ("M5", 8, 1, "NEUTRAL", 0),
        ("M1", 26, 2, "BEARISH", 1),
    ]


def test_the_journey_pins_protected_ranges_and_order_blocks() -> None:
    assert [
        (
            entry.timeframe.value,
            len(entry.protected.assignments),
            len(entry.protected.legs),
            len(entry.dealing_ranges.ranges),
            len(entry.order_blocks.order_blocks),
        )
        for entry in composite().timeframes
    ] == [
        ("H4", 1, 1, 1, 1),
        ("H1", 7, 7, 7, 5),
        ("M15", 0, 0, 0, 0),
        ("M5", 0, 0, 0, 0),
        ("M1", 0, 0, 0, 0),
    ]


def test_the_journey_pins_liquidity() -> None:
    """§41. M15 carries the pools; the others legitimately have none."""
    entries = {entry.timeframe: entry for entry in composite().timeframes}
    m15 = entries[Timeframe.M15]

    assert len(m15.liquidity.pools) == 5
    assert len(m15.liquidity.events) == 5
    assert dict(Counter(pool.status.value for pool in m15.liquidity.pools)) == {
        "SWEPT": 4,
        "CLOSED_THROUGH": 1,
    }
    assert all(not entries[tf].liquidity.pools for tf in ICT_TIMEFRAMES if tf is not Timeframe.M15)


def test_the_journey_pins_the_gap_lifecycle() -> None:
    """§41. M1 carries gaps in all three statuses."""
    entries = {entry.timeframe: entry for entry in composite().timeframes}

    assert [(tf.value, len(entries[tf].gaps)) for tf in ICT_TIMEFRAMES] == [
        ("H4", 0),
        ("H1", 3),
        ("M15", 0),
        ("M5", 0),
        ("M1", 13),
    ]
    assert dict(Counter(s.status.value for s in entries[Timeframe.M1].fvg_lifecycle.states)) == {
        "FILLED": 6,
        "TOUCHED": 4,
        "OPEN": 3,
    }
    assert dict(Counter(s.status.value for s in entries[Timeframe.H1].fvg_lifecycle.states)) == {
        "OPEN": 3
    }


def test_the_journey_pins_the_order_block_lifecycle() -> None:
    entries = {entry.timeframe: entry for entry in composite().timeframes}

    assert dict(
        Counter(s.status.value for s in entries[Timeframe.H1].order_block_lifecycle.states)
    ) == {"INVALIDATED": 3, "MITIGATED": 2}
    assert dict(
        Counter(s.status.value for s in entries[Timeframe.H4].order_block_lifecycle.states)
    ) == {"MITIGATED": 1}


def test_the_journey_pins_atr_including_where_there_is_none() -> None:
    """A series shorter than the period needs is a real state, reported as ``None``."""
    assert [
        (entry.timeframe.value, None if entry.atr is None else str(entry.atr.atr)[:7])
        for entry in composite().timeframes
    ] == [
        ("H4", None),
        ("H1", "52.3773"),
        ("M15", "34.3659"),
        ("M5", None),
        ("M1", "14.0233"),
    ]


def test_every_nested_analysis_is_dated_at_the_snapshot_instant() -> None:
    """§19. Not each series' own last close, which differs between timeframes."""
    result = composite()

    assert result.observed_at == OBSERVED_AT
    for entry in result.timeframes:
        assert entry.observed_at == OBSERVED_AT
        for nested in (
            entry.structure,
            entry.liquidity,
            entry.protected,
            entry.dealing_ranges,
            entry.order_blocks,
            entry.order_block_lifecycle,
            entry.fvg_lifecycle,
        ):
            assert nested.observed_at == OBSERVED_AT, entry.timeframe
            assert nested.symbol == "XAUUSD"


def test_at_least_one_timeframe_ends_before_the_observation() -> None:
    """Which is why §19 is a real constraint rather than a tautology here."""
    result = composite()
    lagging = [
        entry.timeframe.value
        for entry in result.timeframes
        if entry.series.latest_closed_at != OBSERVED_AT
    ]

    assert lagging == [], "all five happen to end on the instant"
    assert result.timeframes[0].series.latest_closed_at == OBSERVED_AT


# --------------------------------------------------------------------------
# §23: the central correctness proof
# --------------------------------------------------------------------------


@pytest.mark.parametrize("timeframe", ICT_TIMEFRAMES)
def test_every_nested_result_equals_the_standalone_call(timeframe: Timeframe) -> None:
    """§23. Threading changed the plumbing and nothing else.

    Each authority is called the way any caller would call it, with the same
    explicit policies, and compared field for field against what the composite
    holds.
    """
    from goldpipeline.services.ict_fvg import analyse_fvg_lifecycle
    from goldpipeline.services.ict_liquidity import analyse_liquidity
    from goldpipeline.services.ict_order_block import analyse_order_blocks
    from goldpipeline.services.ict_order_block_lifecycle import analyse_order_block_lifecycle
    from goldpipeline.services.ict_primitives import (
        confirmed_swings,
        fair_value_gaps,
        latest_atr,
    )
    from goldpipeline.services.ict_protected import analyse_protected_structure
    from goldpipeline.services.ict_range import analyse_dealing_ranges
    from goldpipeline.services.ict_structure import analyse_structure

    shot = divergent_snapshot()
    settings = config()
    series = shot.require(timeframe)
    entry = analyse_ict_composite(shot, config=settings).require(timeframe)

    # Written out rather than unpacked from a dict, so the arguments a caller
    # would actually type are the ones compared.
    assert list(entry.swings) == confirmed_swings(series, left_bars=2, right_bars=2)
    assert list(entry.gaps) == fair_value_gaps(series)
    assert entry.atr == latest_atr(series, 14)
    assert entry.structure == analyse_structure(
        series, symbol="XAUUSD", as_of=OBSERVED_AT, left_bars=2, right_bars=2
    )
    assert entry.liquidity == analyse_liquidity(
        series,
        config=settings.liquidity,
        symbol="XAUUSD",
        as_of=OBSERVED_AT,
        left_bars=2,
        right_bars=2,
    )
    assert entry.protected == analyse_protected_structure(
        series, symbol="XAUUSD", as_of=OBSERVED_AT, left_bars=2, right_bars=2
    )
    assert entry.dealing_ranges == analyse_dealing_ranges(
        series, symbol="XAUUSD", as_of=OBSERVED_AT, left_bars=2, right_bars=2
    )
    assert entry.order_blocks == analyse_order_blocks(
        series,
        config=settings.order_block,
        symbol="XAUUSD",
        as_of=OBSERVED_AT,
        left_bars=2,
        right_bars=2,
    )
    assert entry.order_block_lifecycle == analyse_order_block_lifecycle(
        series,
        config=settings.order_block_lifecycle,
        formation=settings.order_block,
        symbol="XAUUSD",
        as_of=OBSERVED_AT,
        left_bars=2,
        right_bars=2,
    )
    assert entry.fvg_lifecycle == analyse_fvg_lifecycle(series, symbol="XAUUSD", as_of=OBSERVED_AT)


# --------------------------------------------------------------------------
# §24-§28: policy blast radius
# --------------------------------------------------------------------------


def upstream(entry: IctTimeframeAnalysis) -> tuple[object, ...]:
    """Everything a policy change is being asked *not* to touch, by name."""
    return (
        entry.atr,
        entry.swings,
        entry.gaps,
        entry.fvg_lifecycle,
        entry.structure,
        entry.protected,
        entry.dealing_ranges,
    )


def test_changing_the_liquidity_tolerance_moves_only_liquidity() -> None:
    """§24."""
    narrow = composite(tolerance="0.50")
    wide = composite(tolerance="25.00")

    moved = False
    for left, right in zip(narrow.timeframes, wide.timeframes, strict=True):
        assert upstream(left) == upstream(right), left.timeframe
        assert left.order_blocks == right.order_blocks, left.timeframe
        assert left.order_block_lifecycle == right.order_block_lifecycle, left.timeframe
        moved = moved or left.liquidity != right.liquidity

    assert moved, "a tolerance change that moved nothing would prove nothing"


def test_changing_the_zone_basis_moves_only_order_blocks_and_their_lifecycle() -> None:
    """§25."""
    full = composite(basis=OrderBlockZoneBasis.FULL_CANDLE)
    body = composite(basis=OrderBlockZoneBasis.BODY)

    moved = False
    for left, right in zip(full.timeframes, body.timeframes, strict=True):
        assert upstream(left) == upstream(right), left.timeframe
        assert left.liquidity == right.liquidity, left.timeframe
        moved = moved or left.order_blocks != right.order_blocks

    assert moved


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (OrderBlockMitigationRule.TOUCH, OrderBlockMitigationRule.MIDPOINT),
        (OrderBlockMitigationRule.TOUCH, OrderBlockMitigationRule.FULL_ZONE),
        (OrderBlockMitigationRule.MIDPOINT, OrderBlockMitigationRule.FULL_ZONE),
    ],
)
def test_changing_the_mitigation_rule_moves_only_the_lifecycle(
    first: OrderBlockMitigationRule, second: OrderBlockMitigationRule
) -> None:
    """§26. All three pairwise. The blocks themselves must be identical."""
    left_side = composite(rule=first)
    right_side = composite(rule=second)

    for left, right in zip(left_side.timeframes, right_side.timeframes, strict=True):
        assert upstream(left) == upstream(right), left.timeframe
        assert left.liquidity == right.liquidity, left.timeframe
        assert left.order_blocks == right.order_blocks, "the blocks must not move"


def test_the_full_zone_rule_really_does_change_the_lifecycle() -> None:
    """The other half of §26: the pairwise test would pass vacuously otherwise."""
    touch = composite(rule=OrderBlockMitigationRule.TOUCH).require(Timeframe.H1)
    full_zone = composite(rule=OrderBlockMitigationRule.FULL_ZONE).require(Timeframe.H1)

    assert touch.order_block_lifecycle != full_zone.order_block_lifecycle
    assert dict(Counter(s.status.value for s in full_zone.order_block_lifecycle.states)) == {
        "INVALIDATED": 3,
        "TOUCHED": 2,
    }


def test_changing_the_atr_period_moves_only_atr() -> None:
    """§27. ATR is a measurement, not a filter on any ICT truth."""
    short = composite(atr=5)
    long = composite(atr=20)

    moved = False
    for left, right in zip(short.timeframes, long.timeframes, strict=True):
        assert left.swings == right.swings, left.timeframe
        assert left.gaps == right.gaps
        assert left.fvg_lifecycle == right.fvg_lifecycle
        assert left.structure == right.structure
        assert left.liquidity == right.liquidity
        assert left.protected == right.protected
        assert left.dealing_ranges == right.dealing_ranges
        assert left.order_blocks == right.order_blocks
        assert left.order_block_lifecycle == right.order_block_lifecycle
        moved = moved or left.atr != right.atr

    assert moved


def test_changing_the_swing_window_moves_the_swing_branch_and_nothing_else() -> None:
    """§28. The dependency boundary, stated as an assertion.

    Everything downstream of a pivot may legitimately move. The candles, the
    gaps, the gap lifecycle and the ATR may not: none of them has ever looked
    at a swing.
    """
    narrow = composite(left=2, right=2)
    wide = composite(left=4, right=4)

    moved = False
    for left, right in zip(narrow.timeframes, wide.timeframes, strict=True):
        assert left.series == right.series, "the candles are not a function of the window"
        assert left.gaps == right.gaps, left.timeframe
        assert left.fvg_lifecycle == right.fvg_lifecycle, left.timeframe
        assert left.atr == right.atr, left.timeframe
        moved = moved or left.swings != right.swings or left.structure != right.structure

    assert moved, "a window change that moved nothing would prove nothing"


# --------------------------------------------------------------------------
# §42: the six-policy grid
# --------------------------------------------------------------------------


def test_the_six_policy_combinations_share_every_upstream_fact() -> None:
    """§42. One market, six policies, one set of upstream facts."""
    grid = [composite(basis=basis, rule=rule) for basis in BASES for rule in RULES]

    assert len(grid) == 6
    reference = grid[0]
    for other in grid[1:]:
        assert other.config != reference.config
        for left, right in zip(reference.timeframes, other.timeframes, strict=True):
            assert upstream(left) == upstream(right), left.timeframe
            assert left.liquidity == right.liquidity


def test_the_six_policy_combinations_are_not_collapsed() -> None:
    """§42, and Round 6.6d.2's observation carried into the composite.

    All six H1 lifecycle readings are distinct objects - more than the four
    distinct *status tables* below, because the two zone bases also give the
    underlying blocks different ids and different zones. Two policies that
    happen to reach the same verdict are still two readings of two different
    sets of blocks, and collapsing them would lose that.
    """
    readings = {
        (basis.value, rule.value): composite(basis=basis, rule=rule)
        .require(Timeframe.H1)
        .order_block_lifecycle
        for basis in BASES
        for rule in RULES
    }

    assert len(readings) == 6
    assert len({repr(value) for value in readings.values()}) == 6, "no two are the same reading"


def test_within_one_basis_the_three_rules_give_two_status_tables() -> None:
    """§42. The 6.6d.2 observation itself: TOUCH and MIDPOINT agree here, FULL_ZONE does not."""
    for basis in BASES:
        tables = {
            rule.value: tuple(
                (state.order_block_id, state.status.value)
                for state in composite(basis=basis, rule=rule)
                .require(Timeframe.H1)
                .order_block_lifecycle.states
            )
            for rule in RULES
        }

        assert len(set(tables.values())) == 2, basis
        assert tables["TOUCH"] == tables["MIDPOINT"]
        assert tables["FULL_ZONE"] != tables["TOUCH"]


def test_the_two_bases_never_share_an_order_block_identity() -> None:
    """§42. Six combinations, and the two families of blocks stay separate."""
    full = {
        block.order_block_id
        for block in composite(basis=OrderBlockZoneBasis.FULL_CANDLE)
        .require(Timeframe.H1)
        .order_blocks.order_blocks
    }
    body = {
        block.order_block_id
        for block in composite(basis=OrderBlockZoneBasis.BODY)
        .require(Timeframe.H1)
        .order_blocks.order_blocks
    }

    assert len(full) == len(body) == 5
    assert full.isdisjoint(body)


def test_each_result_carries_the_config_that_produced_it() -> None:
    """§15. A consumer never has to infer the policy from the geometry."""
    for basis in BASES:
        for rule in RULES:
            result = composite(basis=basis, rule=rule)

            assert result.config.order_block_zone_basis is basis
            assert result.config.order_block_mitigation_rule is rule
            assert result.config.swing_left_bars == 2
            assert result.config.swing_right_bars == 2
            assert result.config.atr_period == 14
            assert result.config.liquidity_price_tolerance == Decimal(TOLERANCE)
            assert result.require(Timeframe.H1).order_blocks.zone_basis is basis
            assert result.require(Timeframe.H1).order_block_lifecycle.mitigation_rule is rule


# --------------------------------------------------------------------------
# §36: cross-timeframe isolation
# --------------------------------------------------------------------------


def altered(timeframe: Timeframe) -> IctMarketSnapshot:
    """The snapshot with one timeframe's path replaced by the quiet one."""
    changed = dict(PATHS)
    changed[timeframe] = [*QUIET, *QUIET]
    return IctMarketSnapshot(
        observed_at=OBSERVED_AT,
        symbol="XAUUSD",
        provider="fixture",
        provider_symbol=None,
        timeframes=tuple(
            reopen(ending_at(tf, changed[tf], end=OBSERVED_AT), TF_OPENS.get(tf, {}))
            for tf in ICT_TIMEFRAMES
        ),
    )


@pytest.mark.parametrize("changed", [Timeframe.M1, Timeframe.H4, Timeframe.M15])
def test_changing_one_timeframe_leaves_the_others_untouched(changed: Timeframe) -> None:
    """§36. No shared mutable state, proved by comparison rather than by review."""
    reference = composite()
    after = analyse_ict_composite(altered(changed), config=config())

    for tf in ICT_TIMEFRAMES:
        left = reference.require(tf)
        right = after.require(tf)
        if tf is changed:
            assert left != right, f"{tf.value} was supposed to change"
        else:
            assert left == right, f"{tf.value} moved when only {changed.value} did"


def test_a_subset_snapshot_is_analysed_as_it_stands() -> None:
    """§37. Nothing is resampled, back-filled or invented."""
    shot = divergent_snapshot()
    subset = IctMarketSnapshot(
        observed_at=OBSERVED_AT,
        symbol="XAUUSD",
        provider="fixture",
        provider_symbol=None,
        timeframes=(shot.require(Timeframe.H1), shot.require(Timeframe.M15)),
    )
    result = analyse_ict_composite(subset, config=config())

    assert [entry.timeframe for entry in result.timeframes] == [Timeframe.H1, Timeframe.M15]
    assert result.timeframe(Timeframe.M1) is None
    assert result.require(Timeframe.H1) == composite().require(Timeframe.H1)


def test_a_subset_keeps_branch_order_regardless_of_input_order() -> None:
    """§18. Ordering is a property of the output, not of how the snapshot was built."""
    shot = divergent_snapshot()
    reversed_input = IctMarketSnapshot(
        observed_at=OBSERVED_AT,
        symbol="XAUUSD",
        provider="fixture",
        provider_symbol=None,
        timeframes=(shot.require(Timeframe.M1), shot.require(Timeframe.H4)),
    )
    result = analyse_ict_composite(reversed_input, config=config())

    assert [entry.timeframe for entry in result.timeframes] == [Timeframe.H4, Timeframe.M1]


def test_asking_for_an_absent_timeframe_fails_closed() -> None:
    from goldpipeline.services.ict_composite import IctCompositeError

    shot = divergent_snapshot()
    subset = IctMarketSnapshot(
        observed_at=OBSERVED_AT,
        symbol="XAUUSD",
        provider="fixture",
        provider_symbol=None,
        timeframes=(shot.require(Timeframe.H1),),
    )
    result = analyse_ict_composite(subset, config=config())

    with pytest.raises(IctCompositeError, match="holds no M1 analysis"):
        result.require(Timeframe.M1)


# --------------------------------------------------------------------------
# §38: H4 policy
# --------------------------------------------------------------------------


def test_the_h4_grid_a_provider_uses_is_preserved_not_corrected() -> None:
    """§38. No resampling, no 00/04/08 correction, no DST transformation."""
    result = composite()
    h4 = result.require(Timeframe.H4)
    hours = {bar.timestamp.hour for bar in h4.series.bars}

    assert hours <= {0, 4, 8, 12, 16, 20}, "the fixture's own grid, whatever it is"
    assert h4.series.timeframe is Timeframe.H4
    assert all(
        later.timestamp - earlier.timestamp == Timeframe.H4.duration
        for earlier, later in zip(h4.series.bars, h4.series.bars[1:], strict=False)
    )


def test_the_composite_returns_the_snapshot_candles_unchanged() -> None:
    """§38, §22. Orchestration does not touch the bars it was given."""
    shot = divergent_snapshot()
    result = analyse_ict_composite(shot, config=config())

    for tf in ICT_TIMEFRAMES:
        assert result.require(tf).series.bars == shot.require(tf).bars
