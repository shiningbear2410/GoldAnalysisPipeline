"""Whether a projection at T can contain anything that only existed after T.

Round 6.6e.2a §35-§37. The projection is pure, so its no-lookahead property is
inherited rather than earned - which is precisely why it is worth proving. A
projection that quietly reached past its composite would be invisible: every
identity would be real and every status would be one the engine really produced,
just not at the instant being asked about.

There is no ``as_of`` parameter here on purpose. Reconstructing an earlier state
means handing in an earlier composite, and these tests do exactly that.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import ICT_TIMEFRAMES, IctMarketSnapshot, build_timeframe_snapshot
from goldpipeline.services.ict_candidate_source import (
    CandidateSourceKind,
    CandidateSourceProjection,
    project_candidate_sources,
)
from goldpipeline.services.ict_composite import analyse_ict_composite
from goldpipeline.services.ict_order_block import OrderBlockZoneBasis
from goldpipeline.services.ict_order_block_lifecycle import OrderBlockMitigationRule
from tests.test_ict_composite_fixture import OBSERVED_AT, PATHS, config, divergent_snapshot

HOUR = timedelta(hours=1)
H1_BARS = len(PATHS[Timeframe.H1])


def h1_close(bar: int) -> datetime:
    return OBSERVED_AT - HOUR * (H1_BARS - 1 - bar)


def truncated(
    observed_at: datetime, *, only: tuple[Timeframe, ...] = ICT_TIMEFRAMES
) -> IctMarketSnapshot:
    """The snapshot as it would have been assembled at *observed_at*.

    *only* exists for the same reason it does in Round 6.6e.1's lookahead file:
    the five paths span wildly different durations, so an instant beside an H1
    milestone predates the entire M1 series and a snapshot with an empty
    timeframe is refused at construction.
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
    observed_at: datetime, *, only: tuple[Timeframe, ...] = (Timeframe.H1,), **kwargs: object
) -> CandidateSourceProjection:
    return project_candidate_sources(
        analyse_ict_composite(truncated(observed_at, only=only), config=config(**kwargs))  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# §35: the projection is exactly its composite
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bar", "why"),
    [
        (5, "before any source exists"),
        (11, "the first order block"),
        (18, "a second block on the same anchor"),
        (33, "a reversal and a first invalidation"),
        (40, "another invalidation"),
        (56, "a late formation"),
        (61, "an invalidation with no touch"),
        (62, "the end"),
    ],
)
def test_a_projection_is_exactly_the_composite_it_was_given(bar: int, why: str) -> None:
    """§35. Projecting a truncated composite equals projecting at that instant.

    Trivially true for a pure function and worth stating anyway: it is the
    property that would break the moment this layer reached for anything it was
    not handed.
    """
    moment = h1_close(bar)
    composite = analyse_ict_composite(truncated(moment, only=(Timeframe.H1,)), config=config())

    assert project_candidate_sources(composite) == at(moment), why


def test_the_projection_reports_its_composite_s_instant() -> None:
    """§35. No clock, no shifted instant, no as_of of its own."""
    for bar in (11, 33, 62):
        moment = h1_close(bar)
        result = at(moment)

        assert result.observed_at == moment
        for entry in result.timeframes:
            assert entry.observed_at == moment


def test_the_projection_takes_no_as_of_parameter() -> None:
    """§35. Reconstructing means handing in an earlier composite, not a flag."""
    import inspect

    from goldpipeline.services.ict_candidate_source import (
        project_candidate_sources as project,
    )
    from goldpipeline.services.ict_candidate_source import (
        project_timeframe_sources as project_one,
    )

    for function in (project, project_one):
        assert "as_of" not in inspect.signature(function).parameters
        assert "observed_at" not in inspect.signature(function).parameters


@pytest.mark.parametrize("minutes_back", [0, 5, 13, 25])
def test_a_five_timeframe_projection_agrees_where_all_five_have_data(
    minutes_back: int,
) -> None:
    """§35 across the whole snapshot, inside the shortest path's span."""
    moment = OBSERVED_AT - timedelta(minutes=minutes_back)
    composite = analyse_ict_composite(truncated(moment, only=ICT_TIMEFRAMES), config=config())

    result = project_candidate_sources(composite)
    assert result == at(moment, only=ICT_TIMEFRAMES)
    assert [entry.timeframe for entry in result.timeframes] == list(ICT_TIMEFRAMES)


# --------------------------------------------------------------------------
# §36: a future source cannot appear early
# --------------------------------------------------------------------------


def test_a_future_order_block_is_absent_until_its_event_closes() -> None:
    """§36. The H1 block formed at bar 56 does not exist at bar 55."""
    before = at(h1_close(55)).require(Timeframe.H1)
    after = at(h1_close(56)).require(Timeframe.H1)

    assert len(before.of_kind(CandidateSourceKind.ORDER_BLOCK)) == 4
    assert len(after.of_kind(CandidateSourceKind.ORDER_BLOCK)) == 5
    assert {s.source_id for s in before.sources} < {s.source_id for s in after.sources}


def test_a_future_lifecycle_transition_does_not_rewrite_an_earlier_projection() -> None:
    """§36. ACTIVE at bar 60, INVALIDATED at bar 61 - and the earlier fact stays."""
    before = at(h1_close(60)).require(Timeframe.H1)
    after = at(h1_close(61)).require(Timeframe.H1)

    newest_before = before.of_kind(CandidateSourceKind.ORDER_BLOCK)[-1]
    newest_after = after.of_kind(CandidateSourceKind.ORDER_BLOCK)[-1]

    assert newest_before.source_id == newest_after.source_id
    assert newest_before.candidate_source_id == newest_after.candidate_source_id
    assert newest_before.evidence.status.value == "ACTIVE"
    assert newest_after.evidence.status.value == "INVALIDATED"
    assert (newest_before.lower, newest_before.upper) == (newest_after.lower, newest_after.upper)


def test_the_source_set_only_grows() -> None:
    """§36. Sources are appended, never removed, whatever their status."""
    seen: list[tuple[str, ...]] = []
    for bar in (5, 11, 18, 30, 33, 40, 50, 56, 62):
        entry = at(h1_close(bar)).require(Timeframe.H1)
        seen.append(tuple(source.candidate_source_id for source in entry.sources))

    assert seen[-1], "an empty sweep would prove nothing"
    for earlier, later in zip(seen, seen[1:], strict=False):
        assert set(earlier) <= set(later), (earlier, later)


def test_a_projected_identity_never_changes_once_it_exists() -> None:
    """§36. Geometry and identity are fixed at formation; only evidence moves."""
    fixed: dict[str, tuple[object, ...]] = {}

    for bar in range(11, H1_BARS, 3):
        for source in at(h1_close(bar)).require(Timeframe.H1).sources:
            shape = (
                source.source_id,
                source.kind,
                source.geometry,
                source.lower,
                source.upper,
                source.midpoint,
                source.width,
                source.formed_at,
            )
            if source.candidate_source_id in fixed:
                assert fixed[source.candidate_source_id] == shape, source.candidate_source_id
            fixed[source.candidate_source_id] = shape

    assert len(fixed) >= 5


def test_no_source_exists_before_the_market_made_one() -> None:
    """§36. At bar 5 the H1 series has confirmed nothing at all."""
    entry = at(h1_close(5)).require(Timeframe.H1)

    assert entry.sources == ()
    assert entry.active_dealing_range is None


# --------------------------------------------------------------------------
# §37: replay determinism
# --------------------------------------------------------------------------


def test_repeated_projection_gives_an_identical_result() -> None:
    composite = analyse_ict_composite(divergent_snapshot(), config=config())

    assert project_candidate_sources(composite) == project_candidate_sources(composite)


def test_projecting_two_equal_composites_gives_equal_results() -> None:
    first = analyse_ict_composite(divergent_snapshot(), config=config())
    second = analyse_ict_composite(divergent_snapshot(), config=config())

    assert first == second
    assert project_candidate_sources(first) == project_candidate_sources(second)


def provided(provider: str, provider_symbol: str | None) -> IctMarketSnapshot:
    shot = divergent_snapshot()
    return IctMarketSnapshot(
        observed_at=shot.observed_at,
        symbol=shot.symbol,
        provider=provider,
        provider_symbol=provider_symbol,
        timeframes=shot.timeframes,
    )


def test_the_provider_reaches_provenance_and_no_source_fact() -> None:
    """§37. Top-level provenance may differ; not one identity may."""
    one = project_candidate_sources(
        analyse_ict_composite(provided("tradingview", "OANDA:XAUUSD"), config=config())
    )
    two = project_candidate_sources(
        analyse_ict_composite(provided("metatrader", "XAUUSD.pro"), config=config())
    )

    assert (one.provider, one.provider_symbol) == ("tradingview", "OANDA:XAUUSD")
    assert (two.provider, two.provider_symbol) == ("metatrader", "XAUUSD.pro")
    assert one.timeframes == two.timeframes


REPLAY_PROGRAM = """
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from tests.test_ict_composite_fixture import config, divergent_snapshot
from goldpipeline.services.ict_composite import analyse_ict_composite
from goldpipeline.services.ict_candidate_source import project_candidate_sources
from goldpipeline.services.ict_order_block import OrderBlockZoneBasis
from goldpipeline.services.ict_order_block_lifecycle import OrderBlockMitigationRule

for basis in OrderBlockZoneBasis:
    for rule in OrderBlockMitigationRule:
        result = project_candidate_sources(
            analyse_ict_composite(divergent_snapshot(), config=config(basis=basis, rule=rule))
        )
        print("GRID", basis.value, rule.value)
        for entry in result.timeframes:
            print(" TF", entry.timeframe.value, entry.structure_bias.value,
                  None if entry.active_dealing_range is None
                  else entry.active_dealing_range.range_id)
            for source in entry.sources:
                print("   S", source.candidate_source_id, source.kind.value,
                      source.geometry.value, source.source_id,
                      source.formed_at.isoformat(), source.lower, source.upper,
                      source.midpoint, source.width, source.evidence.status.value)
"""


@pytest.mark.parametrize("seed", ["0", "1", "524287"])
def test_string_hash_ordering_cannot_reach_the_output(seed: str) -> None:
    """§37. Every identity, status and ordering, under three hash seeds."""
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
    assert "GRID" in baseline, "a silent empty run would prove nothing"


def test_the_six_policies_each_replay_identically() -> None:
    """§37 across the grid rather than one configuration."""
    for basis in OrderBlockZoneBasis:
        for rule in OrderBlockMitigationRule:
            composite = analyse_ict_composite(
                divergent_snapshot(), config=config(basis=basis, rule=rule)
            )
            assert project_candidate_sources(composite) == project_candidate_sources(composite)


def test_no_randomness_or_clock_is_available_to_the_module() -> None:
    """§37. No UUID, no shuffle, no clock."""
    from pathlib import Path

    text = Path("src/goldpipeline/services/ict_candidate_source.py").read_text(encoding="utf-8")

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
