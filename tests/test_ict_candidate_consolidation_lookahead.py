"""Whether a consolidation taken at T can see anything that happened after T.

Round 6.6e.2c1 §51-§54. This layer resolves nothing and recomputes nothing, so
its no-lookahead property is inherited from the eligibility reading it is handed
- but "inherited" is a claim, and an inherited property is exactly the kind that
stops holding the moment someone adds a convenience lookup. So it is replayed
here against truncated history rather than argued for.

The identity risk is the new one. A group id is a hash over role, side and
canonical prices, and hashing means two failure modes worth pinning: a group
whose id moves when nothing about the candidate moved, and a group whose id
depends on the order Python happened to iterate a dict or a frozenset in.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import ICT_TIMEFRAMES, IctMarketSnapshot
from goldpipeline.services.ict_candidate_consolidation import (
    CANDIDATE_CONSOLIDATION_METHOD_VERSION,
    CandidateConsolidationAnalysis,
    canonical_price,
    consolidate_candidates,
)
from goldpipeline.services.ict_candidate_eligibility import analyse_candidate_eligibility
from goldpipeline.services.ict_composite import analyse_ict_composite
from tests.test_ict_candidate_eligibility_fixture import (
    SNAP_AT,
    eligibility_config,
    staggered_snapshot,
)
from tests.test_ict_candidate_eligibility_lookahead import H1_BARS, h1_close, truncated
from tests.test_ict_composite_fixture import config

MINUTE = timedelta(minutes=1)


def at(
    observed_at: datetime, *, only: tuple[Timeframe, ...] = (Timeframe.H1,)
) -> CandidateConsolidationAnalysis:
    """The consolidation as it would have read at *observed_at*."""
    return consolidate_candidates(
        analyse_candidate_eligibility(
            analyse_ict_composite(truncated(observed_at, only=only), config=config()),
            config=eligibility_config(),
        )
    )


def now() -> CandidateConsolidationAnalysis:
    return consolidate_candidates(
        analyse_candidate_eligibility(
            analyse_ict_composite(staggered_snapshot(), config=config()),
            config=eligibility_config(),
        )
    )


# --------------------------------------------------------------------------
# §51: the reading at T is the reading on history truncated at T
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
def test_a_consolidation_equals_the_one_taken_on_truncated_history(bar: int, why: str) -> None:
    """§51, at the same eight milestones the layer below uses."""
    moment = h1_close(bar)
    eligibility = analyse_candidate_eligibility(
        analyse_ict_composite(truncated(moment, only=(Timeframe.H1,)), config=config()),
        config=eligibility_config(),
    )

    assert consolidate_candidates(eligibility) == at(moment), why


@pytest.mark.parametrize("minutes_back", [0, 1, 2, 4])
def test_the_five_timeframe_consolidation_agrees_through_time(minutes_back: int) -> None:
    """§51 over the whole snapshot, off the five-minute boundaries that conflict."""
    moment = SNAP_AT - timedelta(minutes=minutes_back)
    eligibility = analyse_candidate_eligibility(
        analyse_ict_composite(truncated(moment, only=ICT_TIMEFRAMES), config=config()),
        config=eligibility_config(),
    )

    assert consolidate_candidates(eligibility) == at(moment, only=ICT_TIMEFRAMES)


def test_the_analysis_takes_no_as_of_and_no_snapshot() -> None:
    """§51. Reconstructing a past reading means handing in a past eligibility.

    There is no ``as_of`` to pass, and no snapshot either, so there is nowhere
    for a future bar to enter this layer.
    """
    import inspect

    parameters = inspect.signature(consolidate_candidates).parameters

    assert list(parameters) == ["eligibility"]
    for forbidden in ("as_of", "observed_at", "snapshot", "now", "composite"):
        assert forbidden not in parameters


def test_the_observed_instant_comes_from_the_reading_not_a_clock() -> None:
    """§51. Every timestamp on the result is the eligibility reading's own."""
    for bar in (11, 33, 62):
        eligibility = analyse_candidate_eligibility(
            analyse_ict_composite(truncated(h1_close(bar), only=(Timeframe.H1,)), config=config()),
            config=eligibility_config(),
        )
        result = consolidate_candidates(eligibility)

        assert result.observed_at == eligibility.observed_at == h1_close(bar)
        assert result.reference_price is eligibility.reference_price
        assert result.reference_price.bar_close_time <= result.observed_at


def test_a_future_candidate_has_no_group_yet() -> None:
    """§51. The H1 block formed at bar 56 supports nothing at bar 55."""
    before = at(h1_close(55))
    after = at(h1_close(56))

    assert sum(c.support_count for c in before.candidates) < sum(
        c.support_count for c in after.candidates
    )
    assert {
        identity for candidate in before.candidates for identity in candidate.supporting_source_ids
    } < {identity for candidate in after.candidates for identity in candidate.supporting_source_ids}


def test_a_group_identity_never_changes_once_it_has_appeared() -> None:
    """§28 and §51 together, and the sharper of the two identity risks.

    Walking the whole H1 replay: whenever a group id is seen again, the
    candidate it names must be the same candidate. Support may change freely.
    """
    fixed: dict[str, tuple[object, ...]] = {}
    supports: dict[str, set[int]] = {}

    for bar in range(5, H1_BARS):
        for candidate in at(h1_close(bar)).candidates:
            shape = (
                candidate.role,
                candidate.entry_side,
                candidate.lower,
                candidate.upper,
                candidate.midpoint,
                candidate.width,
                candidate.reference_level,
                candidate.symbol,
            )
            identity = candidate.consolidated_candidate_id
            if identity in fixed:
                assert fixed[identity] == shape, identity
            fixed[identity] = shape
            supports.setdefault(identity, set()).add(candidate.support_count)

    assert len(fixed) >= 4
    assert any(len(seen) > 1 for seen in supports.values()), (
        "at least one group's support must have moved, or this proves nothing"
    )


def test_a_group_may_disappear_and_is_not_append_only() -> None:
    """§27. Stated as a fact of the replay, not as an invariant to defend.

    Groups are a view. A block invalidates, its group loses its last supporter,
    and the group is simply not there any more - which is why no monotonic
    growth is asserted over consolidations the way it is over decisions.
    """
    seen = [
        {candidate.consolidated_candidate_id for candidate in at(h1_close(bar)).candidates}
        for bar in range(5, H1_BARS)
    ]

    lost = [
        (earlier - later) for earlier, later in zip(seen, seen[1:], strict=False) if earlier - later
    ]
    assert lost, "this replay must actually lose a group somewhere"


def test_the_eligible_decisions_of_the_moment_are_exactly_the_supporters() -> None:
    """§29, through time rather than at one instant."""
    for bar in (11, 20, 33, 44, 56, 62):
        result = at(h1_close(bar))
        eligible = sorted(
            decision.candidate_id
            for entry in result.eligibility.timeframes
            for decision in entry.decisions
            if decision.eligible
        )
        placed = sorted(
            identity
            for candidate in result.candidates
            for identity in candidate.supporting_candidate_ids
        )

        assert placed == eligible, bar


# --------------------------------------------------------------------------
# §53: provider provenance
# --------------------------------------------------------------------------


def rebranded(provider: str, provider_symbol: str | None) -> CandidateConsolidationAnalysis:
    shot = staggered_snapshot()
    return consolidate_candidates(
        analyse_candidate_eligibility(
            analyse_ict_composite(
                IctMarketSnapshot(
                    observed_at=shot.observed_at,
                    symbol=shot.symbol,
                    provider=provider,
                    provider_symbol=provider_symbol,
                    timeframes=shot.timeframes,
                ),
                config=config(),
            ),
            config=eligibility_config(),
        )
    )


def test_the_provider_reaches_provenance_and_no_identity() -> None:
    """§53. Same bars from two feeds are the same candidates."""
    one = rebranded("tradingview", "OANDA:XAUUSD")
    two = rebranded("metatrader", "XAUUSD.pro")

    assert (one.provider, one.provider_symbol) != (two.provider, two.provider_symbol)
    assert one.candidates == two.candidates
    assert [c.consolidated_candidate_id for c in one.candidates] == [
        c.consolidated_candidate_id for c in two.candidates
    ]


def test_the_symbol_does_reach_identity() -> None:
    """§53. The instrument is part of what a candidate *is*; the feed is not."""
    shot = staggered_snapshot()
    other = consolidate_candidates(
        analyse_candidate_eligibility(
            analyse_ict_composite(
                IctMarketSnapshot(
                    observed_at=shot.observed_at,
                    symbol="XAGUSD",
                    provider=shot.provider,
                    provider_symbol=shot.provider_symbol,
                    timeframes=shot.timeframes,
                ),
                config=config(),
            ),
            config=eligibility_config(),
        )
    )
    base = now()

    assert [c.consolidated_candidate_id for c in base.candidates] != [
        c.consolidated_candidate_id for c in other.candidates
    ]
    # Sorted, because the group order is keyed on the identities that just moved.
    assert sorted(
        (c.role.value, str(c.entry_side), str(c.lower), str(c.upper)) for c in base.candidates
    ) == sorted(
        (c.role.value, str(c.entry_side), str(c.lower), str(c.upper)) for c in other.candidates
    )


def test_the_method_version_is_stamped_and_is_part_of_identity() -> None:
    """§53. A future rule change gets new ids rather than silently reusing old ones."""
    import hashlib

    result = now()
    candidate = result.candidates[0]

    assert result.method_version == CANDIDATE_CONSOLIDATION_METHOD_VERSION
    assert candidate.method_version == CANDIDATE_CONSOLIDATION_METHOD_VERSION

    preimage = "|".join(
        (
            CANDIDATE_CONSOLIDATION_METHOD_VERSION,
            candidate.symbol,
            candidate.role.value,
            str(candidate.entry_side.value if candidate.entry_side else ""),
            canonical_price(candidate.lower) if candidate.lower is not None else "",
            canonical_price(candidate.upper) if candidate.upper is not None else "",
        )
    )
    assert (
        candidate.consolidated_candidate_id
        == hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]
    )


# --------------------------------------------------------------------------
# §54: immutability
# --------------------------------------------------------------------------


def test_nothing_upstream_is_mutated() -> None:
    """§54. The eligibility reading comes back exactly as it went in."""
    eligibility = analyse_candidate_eligibility(
        analyse_ict_composite(staggered_snapshot(), config=config()),
        config=eligibility_config(),
    )
    before = repr(eligibility)

    result = consolidate_candidates(eligibility)

    assert repr(eligibility) == before
    assert result.eligibility is eligibility


def test_every_new_model_is_frozen() -> None:
    """§54."""
    result = now()
    candidate = result.candidates[0]

    for target, field in (
        (result, "observed_at"),
        (result, "candidates"),
        (candidate, "consolidated_candidate_id"),
        (candidate, "lower"),
        (candidate, "supporting_decisions"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(target, field, None)


def test_the_collections_are_tuples_and_not_lists() -> None:
    """§54. Nothing a caller holds can be appended to."""
    result = now()

    assert isinstance(result.candidates, tuple)
    for candidate in result.candidates:
        assert isinstance(candidate.supporting_decisions, tuple)
        assert isinstance(candidate.source_kinds, tuple)
        assert isinstance(candidate.timeframes, tuple)
        assert isinstance(candidate.supporting_candidate_ids, tuple)
        assert isinstance(candidate.supporting_source_ids, tuple)


def test_a_supporter_is_the_decision_object_itself() -> None:
    """§54. Held by reference, not copied - so no second version can drift."""
    eligibility = analyse_candidate_eligibility(
        analyse_ict_composite(staggered_snapshot(), config=config()),
        config=eligibility_config(),
    )
    originals = {
        id(decision)
        for entry in eligibility.timeframes
        for decision in entry.decisions
        if decision.eligible
    }

    for candidate in consolidate_candidates(eligibility).candidates:
        for member in candidate.supporting_decisions:
            assert id(member) in originals


# --------------------------------------------------------------------------
# §52: replay determinism
# --------------------------------------------------------------------------


def test_repeated_consolidation_gives_an_identical_result() -> None:
    eligibility = analyse_candidate_eligibility(
        analyse_ict_composite(staggered_snapshot(), config=config()),
        config=eligibility_config(),
    )

    assert consolidate_candidates(eligibility) == consolidate_candidates(eligibility)


def test_two_independent_readings_agree() -> None:
    """§52. Not the same object twice - two full runs from the raw bars."""
    assert now() == now()


REPLAY_PROGRAM = """
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from tests.test_ict_candidate_eligibility_fixture import eligibility_config, staggered_snapshot
from tests.test_ict_composite_fixture import config
from goldpipeline.services.ict_composite import analyse_ict_composite
from goldpipeline.services.ict_candidate_eligibility import analyse_candidate_eligibility
from goldpipeline.services.ict_candidate_consolidation import consolidate_candidates
from goldpipeline.services.ict_fvg import FvgStatus
from goldpipeline.services.ict_order_block_lifecycle import OrderBlockStatus

for blocks in (
    frozenset({OrderBlockStatus.ACTIVE}),
    frozenset({OrderBlockStatus.ACTIVE, OrderBlockStatus.TOUCHED, OrderBlockStatus.MITIGATED}),
):
    for gaps in (frozenset({FvgStatus.OPEN}), frozenset({FvgStatus.OPEN, FvgStatus.TOUCHED})):
        result = consolidate_candidates(
            analyse_candidate_eligibility(
                analyse_ict_composite(staggered_snapshot(), config=config()),
                config=eligibility_config(blocks=blocks, gaps=gaps),
            )
        )
        print("REF", result.reference_price.price)
        for c in result.candidates:
            print(
                " C", c.consolidated_candidate_id, c.role.value,
                c.entry_side.value if c.entry_side else "-",
                c.lower, c.upper, c.midpoint, c.width, c.reference_level,
                c.market_relation.value, c.distance_to_reference,
                c.support_count, c.timeframe_count,
                [t.value for t in c.timeframes], [k.value for k in c.source_kinds],
                list(c.supporting_candidate_ids), list(c.supporting_source_ids),
            )
"""


@pytest.mark.parametrize("seed", ["0", "1", "524287"])
def test_string_hash_ordering_cannot_reach_the_output(seed: str) -> None:
    """§52. Grouping uses a dict keyed on strings, so this is a real risk.

    The keys are tuples of strings and the policy sets are frozensets; either
    could leak iteration order into group order, into supporter order, or into
    the first-seen order of ``source_kinds`` and ``timeframes``.
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
    assert baseline.count(" C ") >= 8, "a silent empty run would prove nothing"


def test_no_randomness_or_clock_is_available_to_the_module() -> None:
    """§52, §68."""
    text = Path("src/goldpipeline/services/ict_candidate_consolidation.py").read_text(
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


def test_the_identity_is_stable_across_processes() -> None:
    """§52. A hash truncated to sixteen hex digits, pinned to actual bytes.

    ``PYTHONHASHSEED`` does not touch ``hashlib``, but a preimage assembled out
    of a set would still move, so the value is pinned rather than reasoned about.
    """
    result = now()

    assert [c.consolidated_candidate_id for c in result.candidates] == [
        "097b4590617fa647",
        "a5cc2e8b2526914a",
        "b0ac49bdde522f4a",
        "edf4afe7ca543f85",
        "17d67732ea0614e7",
        "be94c8741bcb02a3",
    ]
    assert result.reference_price.price == Decimal("4043")
