"""Whether a decision taken at T can see anything that happened after T.

Round 6.6e.2b §59-§62. The layer is pure over its composite, so its
no-lookahead property is inherited - but the reference price is new, and a
reference price is exactly the sort of thing a leak hides in: a future M1 close
becoming "the current price" of a historical reading would look entirely normal
and would silently move every location, distance and verdict in the snapshot.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import ICT_TIMEFRAMES, IctMarketSnapshot, build_timeframe_snapshot
from goldpipeline.services.ict_candidate_eligibility import (
    CandidateEligibilityAnalysis,
    CandidateEligibilityError,
    EligibilityReason,
    analyse_candidate_eligibility,
    resolve_reference_price,
)
from goldpipeline.services.ict_candidate_source import project_candidate_sources
from goldpipeline.services.ict_composite import analyse_ict_composite
from tests.test_ict_candidate_eligibility_fixture import (
    SNAP_AT,
    eligibility_config,
    staggered_snapshot,
)
from tests.test_ict_composite_fixture import PATHS, config

MINUTE = timedelta(minutes=1)
HOUR = timedelta(hours=1)
H1_BARS = len(PATHS[Timeframe.H1])


def h1_close(bar: int) -> datetime:
    """The instant H1's bar *bar* closed, counting back from its last one."""
    return SNAP_AT - MINUTE * 3 - HOUR * (H1_BARS - 1 - bar)


def truncated(
    observed_at: datetime, *, only: tuple[Timeframe, ...] = ICT_TIMEFRAMES
) -> IctMarketSnapshot:
    """The snapshot as it would have been assembled at *observed_at*."""
    full = staggered_snapshot()
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
) -> CandidateEligibilityAnalysis:
    return analyse_candidate_eligibility(
        analyse_ict_composite(truncated(observed_at, only=only), config=config()),
        config=eligibility_config(),
    )


# --------------------------------------------------------------------------
# §60: a decision is exactly its composite
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bar", "why"),
    [
        (5, "before any source exists"),
        (11, "the first order block"),
        (18, "a second block forms"),
        (33, "a reversal and a first invalidation"),
        (40, "another invalidation"),
        (56, "a late formation"),
        (61, "an invalidation with no touch"),
        (62, "the end"),
    ],
)
def test_a_decision_equals_the_one_taken_on_truncated_history(bar: int, why: str) -> None:
    """§60. Same composite in, same decisions out, at eight milestones."""
    moment = h1_close(bar)
    composite = analyse_ict_composite(truncated(moment, only=(Timeframe.H1,)), config=config())

    assert analyse_candidate_eligibility(composite, config=eligibility_config()) == at(moment), why


@pytest.mark.parametrize("minutes_back", [0, 1, 2, 4])
def test_a_five_timeframe_decision_agrees_where_all_five_have_data(minutes_back: int) -> None:
    """§60 across the whole snapshot, inside the shortest path's span.

    The instants avoid five-minute boundaries for the reason the next test
    documents.
    """
    moment = SNAP_AT - timedelta(minutes=minutes_back)
    composite = analyse_ict_composite(truncated(moment, only=ICT_TIMEFRAMES), config=config())

    assert analyse_candidate_eligibility(composite, config=eligibility_config()) == at(
        moment, only=ICT_TIMEFRAMES
    )


@pytest.mark.parametrize("minutes_back", [3, 8, 13, 18, 23])
def test_a_truncation_onto_a_shared_boundary_conflicts_and_says_so(minutes_back: int) -> None:
    """§4 through time, and a real property of this fixture worth recording.

    Truncating to a five-minute boundary makes M5 and M1 both close at that
    instant - and on a fifteen-minute one M15 joins them. Their paths, written
    independently for different engines, disagree about the price, so resolution
    refuses rather than preferring the finer timeframe, exactly as it does on
    the unstaggered snapshot.
    """
    moment = SNAP_AT - timedelta(minutes=minutes_back)
    composite = analyse_ict_composite(truncated(moment, only=ICT_TIMEFRAMES), config=config())
    sharing = [
        entry.timeframe.value
        for entry in composite.timeframes
        if entry.series.latest_closed_at == moment
    ]

    assert len(sharing) > 1, sharing
    assert sharing[-2:] == ["M5", "M1"]
    with pytest.raises(CandidateEligibilityError, match="cannot have two closing prices"):
        resolve_reference_price(composite)


def test_the_analysis_takes_no_as_of_of_its_own() -> None:
    """§60. Reconstructing means handing in an earlier composite."""
    import inspect

    parameters = inspect.signature(analyse_candidate_eligibility).parameters

    assert list(parameters) == ["composite", "config", "projection"]
    assert "as_of" not in parameters
    assert "observed_at" not in parameters


# --------------------------------------------------------------------------
# §61: the reference price cannot see the future
# --------------------------------------------------------------------------


def test_a_future_close_cannot_become_a_historical_reference_price() -> None:
    """§61, and the sharpest risk this round introduced.

    M1 owns the latest close, so its future bars are the ones most likely to
    leak. At each earlier instant the reference must be a bar closed by then.
    """
    for minutes_back in (0, 1, 2, 3, 10):
        moment = SNAP_AT - timedelta(minutes=minutes_back)
        composite = analyse_ict_composite(truncated(moment, only=(Timeframe.M1,)), config=config())
        reference = resolve_reference_price(composite)

        assert reference.bar_close_time <= moment
        assert reference.observed_at == moment
        for witness in reference.witnesses:
            assert witness.bar_close_time <= moment


def test_the_reference_price_moves_backwards_through_the_replay() -> None:
    """§61. Each earlier reading names an earlier bar, never a later one."""
    seen: list[datetime] = []
    for minutes_back in (0, 1, 2, 3):
        moment = SNAP_AT - timedelta(minutes=minutes_back)
        composite = analyse_ict_composite(truncated(moment, only=(Timeframe.M1,)), config=config())
        seen.append(resolve_reference_price(composite).bar_close_time)

    assert seen == sorted(seen, reverse=True)
    assert seen[0] == SNAP_AT


def test_an_earlier_reading_uses_an_earlier_price() -> None:
    """§61. Not merely an earlier timestamp - a genuinely different number."""
    now = analyse_ict_composite(truncated(SNAP_AT, only=(Timeframe.M1,)), config=config())
    before = analyse_ict_composite(
        truncated(SNAP_AT - MINUTE * 3, only=(Timeframe.M1,)), config=config()
    )

    assert resolve_reference_price(now).price == Decimal("4043")
    assert resolve_reference_price(before).price == Decimal("4057")


# --------------------------------------------------------------------------
# §60: a future source or status cannot appear early
# --------------------------------------------------------------------------


def test_a_future_source_has_no_decision_yet() -> None:
    """§60. The H1 block formed at bar 56 does not exist at bar 55."""
    before = at(h1_close(55)).require(Timeframe.H1)
    after = at(h1_close(56)).require(Timeframe.H1)

    assert len(before.decisions) == 4
    assert len(after.decisions) == 5
    assert {d.candidate_id for d in before.decisions} < {d.candidate_id for d in after.decisions}


def test_a_future_invalidation_does_not_reach_an_earlier_decision() -> None:
    """§60. A candidate present at both instants gains its terminal reason only later.

    Found by identity rather than by position: bar 61 also *adds* a decision, so
    the last element of each list is not the same candidate.
    """
    before = at(h1_close(60)).require(Timeframe.H1)
    after = at(h1_close(61)).require(Timeframe.H1)

    gained = [
        decision
        for decision in before.decisions
        if EligibilityReason.ORDER_BLOCK_INVALIDATED not in decision.reasons
        and EligibilityReason.ORDER_BLOCK_INVALIDATED
        in (after.decision(decision.candidate_id) or decision).reasons
    ]

    assert len(gained) == 1
    earlier = gained[0]
    later = after.decision(earlier.candidate_id)
    assert later is not None
    assert earlier.candidate_id == later.candidate_id
    assert earlier.eligible is True
    assert later.eligible is False


def test_the_decision_set_only_grows() -> None:
    """§30, through time."""
    seen: list[set[str]] = []
    for bar in (5, 11, 18, 30, 33, 40, 50, 56, 62):
        entry = at(h1_close(bar)).require(Timeframe.H1)
        seen.append({d.candidate_id for d in entry.decisions})

    assert seen[-1]
    for earlier, later in zip(seen, seen[1:], strict=False):
        assert earlier <= later


def test_a_candidate_identity_never_changes_once_it_exists() -> None:
    """§28, through time. Only the verdict moves."""
    fixed: dict[str, tuple[object, ...]] = {}

    for bar in range(11, H1_BARS, 3):
        for decision in at(h1_close(bar)).require(Timeframe.H1).decisions:
            shape = (
                decision.candidate_source_id,
                decision.source_id,
                decision.role,
                decision.entry_side,
                decision.lower,
                decision.upper,
            )
            if decision.candidate_id in fixed:
                assert fixed[decision.candidate_id] == shape, decision.candidate_id
            fixed[decision.candidate_id] = shape

    assert len(fixed) >= 5


# --------------------------------------------------------------------------
# §59: immutability
# --------------------------------------------------------------------------


def test_nothing_upstream_is_mutated() -> None:
    """§59. Composite and projection come back as they went in."""
    composite = analyse_ict_composite(staggered_snapshot(), config=config())
    projection = project_candidate_sources(composite)
    before = (repr(composite), repr(projection))

    analyse_candidate_eligibility(composite, config=eligibility_config(), projection=projection)

    assert (repr(composite), repr(projection)) == before


def test_every_new_model_is_frozen() -> None:
    """§59."""
    from dataclasses import FrozenInstanceError

    result = analyse_candidate_eligibility(
        analyse_ict_composite(staggered_snapshot(), config=config()),
        config=eligibility_config(),
    )
    entry = result.require(Timeframe.H1)

    for target, field in (
        (result, "observed_at"),
        (result.reference_price, "price"),
        (result.reference_price.witnesses[0], "close"),
        (entry, "structure_bias"),
        (entry.decisions[0], "eligible"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(target, field, None)


# --------------------------------------------------------------------------
# §62: replay determinism
# --------------------------------------------------------------------------


def test_repeated_analysis_gives_an_identical_result() -> None:
    composite = analyse_ict_composite(staggered_snapshot(), config=config())

    assert analyse_candidate_eligibility(
        composite, config=eligibility_config()
    ) == analyse_candidate_eligibility(composite, config=eligibility_config())


def test_the_provider_reaches_provenance_and_no_decision() -> None:
    """§62."""
    shot = staggered_snapshot()
    one = analyse_candidate_eligibility(
        analyse_ict_composite(
            IctMarketSnapshot(
                observed_at=shot.observed_at,
                symbol=shot.symbol,
                provider="tradingview",
                provider_symbol="OANDA:XAUUSD",
                timeframes=shot.timeframes,
            ),
            config=config(),
        ),
        config=eligibility_config(),
    )
    two = analyse_candidate_eligibility(
        analyse_ict_composite(
            IctMarketSnapshot(
                observed_at=shot.observed_at,
                symbol=shot.symbol,
                provider="metatrader",
                provider_symbol="XAUUSD.pro",
                timeframes=shot.timeframes,
            ),
            config=config(),
        ),
        config=eligibility_config(),
    )

    assert one.provider != two.provider
    assert one.timeframes == two.timeframes
    assert one.reference_price == two.reference_price


REPLAY_PROGRAM = """
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from tests.test_ict_candidate_eligibility_fixture import eligibility_config, staggered_snapshot
from tests.test_ict_composite_fixture import config
from goldpipeline.services.ict_composite import analyse_ict_composite
from goldpipeline.services.ict_candidate_eligibility import analyse_candidate_eligibility
from goldpipeline.services.ict_fvg import FvgStatus
from goldpipeline.services.ict_order_block_lifecycle import OrderBlockStatus

for blocks in (frozenset({OrderBlockStatus.ACTIVE}), frozenset({OrderBlockStatus.MITIGATED})):
    for gaps in (frozenset({FvgStatus.OPEN}), frozenset({FvgStatus.OPEN, FvgStatus.TOUCHED})):
        result = analyse_candidate_eligibility(
            analyse_ict_composite(staggered_snapshot(), config=config()),
            config=eligibility_config(blocks=blocks, gaps=gaps),
        )
        ref = result.reference_price
        print("REF", ref.price, ref.bar_close_time.isoformat(),
              [w.timeframe.value for w in ref.witnesses])
        for entry in result.timeframes:
            print(" TF", entry.timeframe.value, entry.structure_bias.value)
            for d in entry.decisions:
                print("   D", d.candidate_id, d.candidate_source_id, d.source_id,
                      d.role.value, d.entry_side.value if d.entry_side else "-",
                      d.market_relation.value, d.distance_to_reference,
                      d.lower, d.upper, d.reference_level, d.eligible,
                      [r.value for r in d.reasons])
"""


@pytest.mark.parametrize("seed", ["0", "1", "524287"])
def test_string_hash_ordering_cannot_reach_the_output(seed: str) -> None:
    """§62. The config holds frozensets, so iteration order is a real risk."""
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
    assert "REF" in baseline, "a silent empty run would prove nothing"


def test_no_randomness_or_clock_is_available_to_the_module() -> None:
    """§62, §68."""
    from pathlib import Path

    text = Path("src/goldpipeline/services/ict_candidate_eligibility.py").read_text(
        encoding="utf-8"
    )

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
