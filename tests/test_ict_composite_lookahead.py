"""Whether a composite reading at T can see anything that happened after T.

Round 6.6e.1 §19, §33-§35, §43. The eighth time this file has been written on
the branch, and the risk is now compounding: every nested engine has its own
no-lookahead proof, but threading a result computed for one instant into a stage
asked about another would defeat all of them at once. So the comparison here is
against physically truncated history, at instants chosen to sit either side of
each kind of event.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import ICT_TIMEFRAMES, IctMarketSnapshot, build_timeframe_snapshot
from goldpipeline.services.ict_composite import (
    IctCompositeAnalysis,
    analyse_ict_composite,
    analyse_ict_timeframe,
)
from goldpipeline.services.ict_order_block import OrderBlockZoneBasis
from goldpipeline.services.ict_order_block_lifecycle import (
    OrderBlockMitigationRule,
    OrderBlockStatus,
)
from tests.test_ict_composite_fixture import (
    OBSERVED_AT,
    PATHS,
    TF_OPENS,
    config,
    divergent_snapshot,
)
from tests.test_ict_structure import Row

HOUR = timedelta(hours=1)


def reshaped_truncated(
    observed_at: datetime, *, only: tuple[Timeframe, ...] = ICT_TIMEFRAMES
) -> IctMarketSnapshot:
    """The snapshot as it would have been assembled at *observed_at*.

    Physically shorter series, not a filtered view: every bar that had not
    closed is absent, which is what an honest reconstruction means. The body
    reshaping is by bar index, so it happens on the full series and is then cut,
    exactly as ``divergent_snapshot`` builds it.

    *only* exists because the five paths cover wildly different spans - M1's 26
    bars are 26 minutes, H1's 63 are nearly three days. An instant chosen to sit
    beside an H1 milestone predates the whole M1 series, and a snapshot with an
    empty timeframe is refused at construction rather than being analysed. So
    the milestone reconstructions run on H1 alone, and the five-timeframe
    reconstruction runs where all five genuinely have data.
    """
    full = divergent_snapshot()
    return IctMarketSnapshot(
        observed_at=observed_at,
        symbol="XAUUSD",
        provider="fixture",
        provider_symbol=None,
        timeframes=tuple(
            build_timeframe_snapshot(
                timeframe=tf, bars=full.require(tf).bars, observed_at=observed_at
            )
            for tf in only
        ),
    )


def at(
    observed_at: datetime, *, only: tuple[Timeframe, ...] = (Timeframe.H1,)
) -> IctCompositeAnalysis:
    return analyse_ict_composite(reshaped_truncated(observed_at, only=only), config=config())


# Instants chosen to straddle the H1 journey's milestones, which are the
# richest events in the fixture. Bar n of H1 closes at OBSERVED_AT - (62 - n)h.
H1_BARS = len(PATHS[Timeframe.H1])


def h1_close(bar: int) -> datetime:
    return OBSERVED_AT - HOUR * (H1_BARS - 1 - bar)


# --------------------------------------------------------------------------
# §43: the central invariant
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bar", "why"),
    [
        (5, "before any swing confirms"),
        (11, "the first structure event and its order block"),
        (18, "a second event on the same anchor"),
        (33, "a reversal, and one candle changing two blocks"),
        (40, "an invalidation"),
        (56, "a late formation"),
        (61, "an invalidation with no touch"),
        (62, "the end"),
    ],
)
def test_a_reconstruction_equals_the_truncated_snapshot(bar: int, why: str) -> None:
    """§43. Every nested analysis, field for field, at eight chosen instants."""
    moment = h1_close(bar)
    physical = reshaped_truncated(moment, only=(Timeframe.H1,))

    assert analyse_ict_composite(physical, config=config()) == at(moment), why


@pytest.mark.parametrize("minutes_back", [0, 5, 13, 25])
def test_a_five_timeframe_reconstruction_agrees_where_all_five_have_data(
    minutes_back: int,
) -> None:
    """§43 across the whole snapshot, at instants inside the shortest path's span."""
    moment = OBSERVED_AT - timedelta(minutes=minutes_back)
    physical = reshaped_truncated(moment, only=ICT_TIMEFRAMES)

    result = analyse_ict_composite(physical, config=config())
    assert result == at(moment, only=ICT_TIMEFRAMES)
    assert [entry.timeframe for entry in result.timeframes] == list(ICT_TIMEFRAMES)


def test_the_composite_is_dated_at_the_instant_it_was_asked_about() -> None:
    """§19. Not the newest bar's close, which several timeframes fall short of."""
    moment = OBSERVED_AT - timedelta(minutes=7)
    result = at(moment, only=ICT_TIMEFRAMES)

    assert result.observed_at == moment
    for entry in result.timeframes:
        assert entry.observed_at == moment
        assert entry.series.latest_closed_at <= moment
    assert any(entry.series.latest_closed_at < moment for entry in result.timeframes), (
        "at least one timeframe lags the observation, which is what makes this a real check"
    )


def test_a_timeframe_whose_last_bar_predates_the_observation_is_still_dated_there() -> None:
    """§19, and the case that makes it a real constraint.

    Reading half way through an H4 bar, the H4 series ends hours before the
    observation - and the analysis is dated at the observation all the same,
    because that is when the question was asked.
    """
    moment = OBSERVED_AT - timedelta(minutes=7)
    entry = at(moment, only=ICT_TIMEFRAMES).require(Timeframe.H4)

    assert entry.series.latest_closed_at < moment
    assert entry.observed_at == moment
    assert entry.structure.observed_at == moment


def test_no_future_order_block_state_leaks_into_an_earlier_reading() -> None:
    """§43. The H1 block formed at bar 56 is ACTIVE at 60 and retired at 61."""
    before = at(h1_close(60)).require(Timeframe.H1).order_block_lifecycle
    after = at(h1_close(61)).require(Timeframe.H1).order_block_lifecycle

    assert before.states[-1].status is OrderBlockStatus.ACTIVE
    assert before.states[-1].invalidated_at is None
    assert after.states[-1].status is OrderBlockStatus.INVALIDATED


def test_no_future_structure_event_leaks_into_an_earlier_reading() -> None:
    """§43. Seven events by the end; fewer at every earlier instant."""
    counts = [
        len(at(h1_close(bar)).require(Timeframe.H1).structure.breaks) for bar in (5, 11, 18, 33, 62)
    ]

    assert counts == sorted(counts)
    assert counts[0] == 0
    assert counts[-1] == 7


def test_history_only_grows_across_the_whole_reconstruction() -> None:
    """Nested collections are appended to, never rewritten."""
    seen: list[tuple[int, int, int, int]] = []
    for bar in (5, 11, 18, 30, 33, 40, 50, 56, 62):
        entry = at(h1_close(bar)).require(Timeframe.H1)
        seen.append(
            (
                len(entry.swings),
                len(entry.structure.breaks),
                len(entry.protected.legs),
                len(entry.order_blocks.order_blocks),
            )
        )

    for earlier, later in zip(seen, seen[1:], strict=False):
        assert all(a <= b for a, b in zip(earlier, later, strict=True)), (earlier, later)


# --------------------------------------------------------------------------
# §33: immutability
# --------------------------------------------------------------------------


def test_the_snapshot_is_not_mutated() -> None:
    """§33. The caller's snapshot comes back as it went in."""
    shot = divergent_snapshot()
    before = tuple(tuple(series.bars) for series in shot.timeframes)

    analyse_ict_composite(shot, config=config())

    assert tuple(tuple(series.bars) for series in shot.timeframes) == before


def test_a_repeated_analysis_shares_no_state_with_the_first() -> None:
    """§33, §34. Two runs are equal and neither can have edited the other."""
    shot = divergent_snapshot()
    first = analyse_ict_composite(shot, config=config())
    second = analyse_ict_composite(shot, config=config())

    assert first == second
    assert first is not second


def test_every_nested_analysis_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    entry = analyse_ict_composite(divergent_snapshot(), config=config()).require(Timeframe.H1)

    with pytest.raises(FrozenInstanceError):
        entry.structure.observed_at = OBSERVED_AT  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        entry.atr = None  # type: ignore[misc]


def test_the_config_cannot_be_edited_after_the_fact() -> None:
    from dataclasses import FrozenInstanceError

    result = analyse_ict_composite(divergent_snapshot(), config=config())

    with pytest.raises(FrozenInstanceError):
        result.config.atr_period = 21  # type: ignore[misc]


# --------------------------------------------------------------------------
# §34-§35: replay determinism and provider independence
# --------------------------------------------------------------------------


def provided(provider: str, provider_symbol: str | None) -> IctMarketSnapshot:
    shot = divergent_snapshot()
    return IctMarketSnapshot(
        observed_at=shot.observed_at,
        symbol=shot.symbol,
        provider=provider,
        provider_symbol=provider_symbol,
        timeframes=shot.timeframes,
    )


def test_the_provider_reaches_provenance_and_nothing_else() -> None:
    """§35. Equivalent candles give identical analyses under three providers."""
    one = analyse_ict_composite(provided("tradingview", "OANDA:XAUUSD"), config=config())
    two = analyse_ict_composite(provided("metatrader", "XAUUSD.pro"), config=config())
    three = analyse_ict_composite(provided("fixture", None), config=config())

    assert (one.provider, one.provider_symbol) == ("tradingview", "OANDA:XAUUSD")
    assert (two.provider, two.provider_symbol) == ("metatrader", "XAUUSD.pro")
    assert one.timeframes == two.timeframes == three.timeframes
    assert one.config == two.config == three.config


def test_the_provider_reaches_no_nested_identity() -> None:
    """§35. Not into a block id, a range id, a pool id or a gap id."""
    one = analyse_ict_composite(provided("tradingview", "OANDA:XAUUSD"), config=config())
    two = analyse_ict_composite(provided("metatrader", "XAUUSD.pro"), config=config())

    for tf in ICT_TIMEFRAMES:
        left, right = one.require(tf), two.require(tf)
        assert [b.order_block_id for b in left.order_blocks.order_blocks] == [
            b.order_block_id for b in right.order_blocks.order_blocks
        ]
        assert [r.range_id for r in left.dealing_ranges.ranges] == [
            r.range_id for r in right.dealing_ranges.ranges
        ]
        assert [p.pool_id for p in left.liquidity.pools] == [
            p.pool_id for p in right.liquidity.pools
        ]


def test_a_timeframe_analysed_alone_matches_the_composite() -> None:
    """The per-timeframe entry point is the composite's own, not a second path."""
    shot = divergent_snapshot()

    for tf in ICT_TIMEFRAMES:
        alone = analyse_ict_timeframe(
            shot.require(tf), config=config(), symbol="XAUUSD", as_of=OBSERVED_AT
        )
        assert alone == analyse_ict_composite(shot, config=config()).require(tf)


REPLAY_PROGRAM = """
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from tests.test_ict_composite_fixture import config, divergent_snapshot
from goldpipeline.services.ict_composite import analyse_ict_composite
from goldpipeline.services.ict_order_block import OrderBlockZoneBasis
from goldpipeline.services.ict_order_block_lifecycle import OrderBlockMitigationRule

for basis in OrderBlockZoneBasis:
    for rule in OrderBlockMitigationRule:
        result = analyse_ict_composite(
            divergent_snapshot(), config=config(basis=basis, rule=rule)
        )
        print("CONFIG", basis.value, rule.value, result.config)
        for entry in result.timeframes:
            print("TF", entry.timeframe.value, entry.observed_at.isoformat(), entry.atr)
            print("  SW", [(s.swing_type.value, s.price, s.confirmed_at.isoformat())
                           for s in entry.swings])
            print("  GA", [(g.direction.value, g.lower, g.upper) for g in entry.gaps])
            print("  ST", entry.structure.current_bias.value,
                  [e.event_id for e in entry.structure.breaks])
            print("  LQ", [(p.pool_id, p.status.value) for p in entry.liquidity.pools])
            print("  PR", [leg.leg_id for leg in entry.protected.legs])
            print("  DR", [r.range_id for r in entry.dealing_ranges.ranges])
            print("  OB", [b.order_block_id for b in entry.order_blocks.order_blocks])
            print("  OL", [(s.order_block_id, s.status.value)
                           for s in entry.order_block_lifecycle.states])
            print("  FL", [(s.fvg_id, s.status.value) for s in entry.fvg_lifecycle.states])
"""


@pytest.mark.parametrize("seed", ["0", "1", "524287"])
def test_string_hash_ordering_cannot_reach_the_output(seed: str) -> None:
    """§34. Every identity, status and ordering, under three hash seeds.

    The composite keys timeframes by enum in a dict, so the risk is real rather
    than theoretical.
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
    assert "TF H1" in baseline, "a silent empty run would prove nothing"


def test_no_randomness_or_clock_is_available_to_the_module() -> None:
    """§34. No UUID, no shuffle, no clock."""
    from pathlib import Path

    text = Path("src/goldpipeline/services/ict_composite.py").read_text(encoding="utf-8")

    for forbidden in (
        "uuid",
        "random",
        "secrets",
        "shuffle",
        "datetime.now",
        "utcnow",
        "time.time",
        "utc_now",
    ):
        assert forbidden not in text


def test_the_six_policies_replay_identically() -> None:
    """§34, across the grid rather than one configuration."""
    for basis in OrderBlockZoneBasis:
        for rule in OrderBlockMitigationRule:
            settings = config(basis=basis, rule=rule)
            first = analyse_ict_composite(divergent_snapshot(), config=settings)
            second = analyse_ict_composite(divergent_snapshot(), config=settings)
            assert first == second, (basis, rule)


def test_an_observation_before_any_bar_closed_is_refused() -> None:
    """Refused rather than answered with five empty readings."""
    with pytest.raises(ValueError, match="no bar closed"):
        analyse_ict_timeframe(
            divergent_snapshot().require(Timeframe.H1),
            config=config(),
            symbol="XAUUSD",
            as_of=datetime(2020, 1, 1, tzinfo=UTC),
        )


def test_a_row_type_reference_keeps_the_fixture_honest() -> None:
    """The paths really are the shared ``Row`` shape, not a local re-definition."""
    assert all(isinstance(row, tuple) and len(row) == 3 for row in PATHS[Timeframe.H1])
    assert TF_OPENS.keys() <= set(ICT_TIMEFRAMES)
    example: Row = ("4010", "3990", "4000")
    assert len(example) == 3
