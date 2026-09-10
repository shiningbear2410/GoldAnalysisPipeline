"""Whether a feature graph taken at T can see anything that happened after T.

Round 6.6f §44-§45. The layer is pure over a finished consolidation, so its
no-lookahead property is inherited - and, as in every previous round, inherited
properties are replayed rather than argued for.

Two things here are genuinely new and worth the replay. Age is a subtraction
against ``observed_at``, so a leaked future instant would show up as an age that
is too small rather than as a missing candidate - a much quieter failure. And
range context is now *read* for the first time since Round 6.6e.2b carried it,
so a range that had not formed yet must not classify anything.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import ICT_TIMEFRAMES
from goldpipeline.services.ict_candidate_consolidation import consolidate_candidates
from goldpipeline.services.ict_candidate_eligibility import analyse_candidate_eligibility
from goldpipeline.services.ict_candidate_features import (
    CandidateFeatureAnalysis,
    build_candidate_features,
)
from goldpipeline.services.ict_candidate_source import CandidateSourceKind
from goldpipeline.services.ict_composite import analyse_ict_composite
from goldpipeline.services.ict_fvg import FvgStatus
from goldpipeline.services.ict_order_block import OrderBlockZoneBasis
from goldpipeline.services.ict_order_block_lifecycle import (
    OrderBlockMitigationRule,
    OrderBlockStatus,
)
from tests.test_ict_candidate_eligibility_fixture import (
    SNAP_AT,
    analysis,
    eligibility_config,
    staggered_snapshot,
)
from tests.test_ict_candidate_eligibility_lookahead import H1_BARS, h1_close, truncated
from tests.test_ict_composite_fixture import PATHS, config

WIDE_BLOCKS = frozenset(
    {OrderBlockStatus.ACTIVE, OrderBlockStatus.TOUCHED, OrderBlockStatus.MITIGATED}
)


def graph(**kwargs: object) -> CandidateFeatureAnalysis:
    return build_candidate_features(consolidate_candidates(analysis(**kwargs)))  # type: ignore[arg-type]


def at(
    observed_at: datetime, *, only: tuple[Timeframe, ...] = (Timeframe.H1,)
) -> CandidateFeatureAnalysis:
    """The feature graph as it would have read at *observed_at*."""
    return build_candidate_features(
        consolidate_candidates(
            analyse_candidate_eligibility(
                analyse_ict_composite(truncated(observed_at, only=only), config=config()),
                config=eligibility_config(),
            )
        )
    )


def shape(result: CandidateFeatureAnalysis) -> list[tuple[object, ...]]:
    return [
        (
            feature.candidate_id,
            feature.role,
            feature.entry_side,
            feature.lower,
            feature.upper,
            feature.reference_level,
        )
        for feature in result.candidates
    ]


def by_id(result: CandidateFeatureAnalysis) -> dict[str, tuple[object, ...]]:
    return {row[0]: row[1:] for row in shape(result)}  # type: ignore[misc]


def identities(result: CandidateFeatureAnalysis) -> set[str]:
    return {feature.candidate_id for feature in result.candidates}


# --------------------------------------------------------------------------
# §45: the reading at T is the reading on history truncated at T
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
def test_a_feature_graph_equals_the_one_taken_on_truncated_history(bar: int, why: str) -> None:
    """§45, at the same eight milestones every layer below uses."""
    moment = h1_close(bar)
    consolidation = consolidate_candidates(
        analyse_candidate_eligibility(
            analyse_ict_composite(truncated(moment, only=(Timeframe.H1,)), config=config()),
            config=eligibility_config(),
        )
    )

    assert build_candidate_features(consolidation) == at(moment), why


@pytest.mark.parametrize("minutes_back", [0, 1, 2, 4])
def test_the_five_timeframe_graph_agrees_through_time(minutes_back: int) -> None:
    """§45, off the five-minute boundaries where the reference price conflicts."""
    moment = SNAP_AT - timedelta(minutes=minutes_back)
    consolidation = consolidate_candidates(
        analyse_candidate_eligibility(
            analyse_ict_composite(truncated(moment, only=ICT_TIMEFRAMES), config=config()),
            config=eligibility_config(),
        )
    )

    assert build_candidate_features(consolidation) == at(moment, only=ICT_TIMEFRAMES)


def test_the_builder_takes_no_as_of_and_no_snapshot() -> None:
    """§45. There is nowhere for a future bar to enter this layer."""
    import inspect

    parameters = inspect.signature(build_candidate_features).parameters

    assert list(parameters) == ["consolidation"]
    for forbidden in ("as_of", "observed_at", "snapshot", "now", "clock"):
        assert forbidden not in parameters


def test_no_age_can_be_measured_against_a_future_instant() -> None:
    """§45, §5. The sharpest new leak: an age that is quietly too small.

    At each earlier instant every supporter must have formed by then, and the
    age must equal the subtraction against that instant rather than the latest.
    """
    for bar in (20, 33, 44, 56, 62):
        moment = h1_close(bar)
        result = at(moment)
        assert result.observed_at == moment

        for feature in result.candidates:
            for fact in feature.support_facts:
                assert fact.formed_at <= moment
                assert fact.age_seconds == int((moment - fact.formed_at).total_seconds())
                assert fact.age_seconds >= 0


@pytest.mark.parametrize(
    ("candidate_id", "earlier_bar", "later_bar", "earlier_age", "later_age"),
    [
        ("097b4590617fa647", 61, 62, 0, 3600),
        ("30ddd3655a6cc914", 56, 57, 0, 3600),
        ("30ddd3655a6cc914", 57, 59, 3600, 10800),
        ("5a2c9942aa89b975", 57, 60, 0, 10800),
    ],
)
def test_an_earlier_reading_gives_a_younger_candidate(
    candidate_id: str, earlier_bar: int, later_bar: int, earlier_age: int, later_age: int
) -> None:
    """§45. The same candidate is genuinely younger earlier on, not merely equal.

    The bar pairs are chosen rather than swept, because most pairs of instants in
    this H1 replay share no candidate at all - the branch invalidates and
    re-forms often enough that a candidate surviving several bars is the
    exception. Four such survivals exist and all four are pinned.
    """
    earlier = at(h1_close(earlier_bar)).feature(candidate_id)
    later = at(h1_close(later_bar)).feature(candidate_id)

    assert earlier is not None and later is not None
    assert earlier.oldest_source_age_seconds == earlier_age
    assert later.oldest_source_age_seconds == later_age
    assert earlier.oldest_source_age_seconds < later.oldest_source_age_seconds
    assert (earlier.lower, earlier.upper) == (later.lower, later.upper)


def test_a_range_that_has_not_formed_yet_classifies_nothing() -> None:
    """§45, §6. Range context is read for the first time this round.

    Early in the H1 replay no dealing range exists, so every location is absent;
    later one does, and locations appear. Both states are asserted, so a leak
    that back-dated a range would break the first half.
    """
    early = at(h1_close(5))
    late = at(h1_close(62))

    for context in early.timeframe_contexts:
        assert context.active_range_id is None

    assert any(context.active_range_id is not None for context in late.timeframe_contexts)


def test_a_future_supporter_is_not_counted_early() -> None:
    """§45. Support facts follow the candidate set, which follows history."""
    before = at(h1_close(55))
    after = at(h1_close(56))

    assert sum(f.support_count for f in before.candidates) < sum(
        f.support_count for f in after.candidates
    )
    early_sources = {
        fact.source_id for feature in before.candidates for fact in feature.support_facts
    }
    late_sources = {
        fact.source_id for feature in after.candidates for fact in feature.support_facts
    }
    assert early_sources < late_sources


def test_a_pair_relation_exists_only_once_both_candidates_do() -> None:
    """§45, §8."""
    for bar in (5, 20, 40, 62):
        result = at(h1_close(bar))
        known = identities(result)
        for relation in result.pair_relations:
            assert relation.first_candidate_id in known
            assert relation.second_candidate_id in known


# --------------------------------------------------------------------------
# §44: policy blast radius
# --------------------------------------------------------------------------


def test_a_eligibility_policy_change_may_change_the_candidate_set() -> None:
    """§44 A. Stated as the fact it is, rather than defended against."""
    narrow = graph()
    wide = graph(blocks=WIDE_BLOCKS)

    assert identities(narrow) < identities(wide)
    assert len(wide.candidates) == len(narrow.candidates) + 1
    for identity in identities(narrow):
        assert by_id(narrow)[identity] == by_id(wide)[identity]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (OrderBlockMitigationRule.TOUCH, OrderBlockMitigationRule.MIDPOINT),
        (OrderBlockMitigationRule.TOUCH, OrderBlockMitigationRule.FULL_ZONE),
        (OrderBlockMitigationRule.MIDPOINT, OrderBlockMitigationRule.FULL_ZONE),
    ],
)
def test_b_the_mitigation_rule_may_move_block_support_and_never_geometry(
    first: OrderBlockMitigationRule, second: OrderBlockMitigationRule
) -> None:
    """§44 B. A surviving candidate keeps its identity and its prices."""
    left = graph(rule=first, blocks=WIDE_BLOCKS)
    right = graph(rule=second, blocks=WIDE_BLOCKS)

    for identity in identities(left) & identities(right):
        assert by_id(left)[identity] == by_id(right)[identity]

    for result in (left, right):
        for feature in result.candidates:
            for fact in feature.support_facts:
                if fact.order_block_mitigation_rule is not None:
                    assert fact.order_block_mitigation_rule is (first if result is left else second)


def test_c_the_zone_basis_may_move_block_candidate_geometry() -> None:
    """§44 C. A different zone is a different candidate, with a different id."""
    full = graph(basis=OrderBlockZoneBasis.FULL_CANDLE, blocks=WIDE_BLOCKS)
    body = graph(basis=OrderBlockZoneBasis.BODY, blocks=WIDE_BLOCKS)

    def blocks(result: CandidateFeatureAnalysis) -> set[str]:
        return {
            feature.candidate_id
            for feature in result.candidates
            if CandidateSourceKind.ORDER_BLOCK in feature.source_kinds
        }

    assert blocks(full) != blocks(body)
    for identity in identities(full) & identities(body):
        assert by_id(full)[identity] == by_id(body)[identity]


def test_d_the_atr_period_changes_only_the_retained_config() -> None:
    """§44 D. ATR is not a candidate feature, so it reaches no feature."""
    short = graph(atr=5)
    long = graph(atr=20)

    assert short.composite_config != long.composite_config
    assert short.candidates == long.candidates
    assert short.pair_relations == long.pair_relations
    assert short.timeframe_contexts == long.timeframe_contexts
    assert short != long, "the difference is the carried provenance, and it is carried"


def test_e_a_reference_price_move_leaves_surviving_identities_alone() -> None:
    """§44 E. The candidate set may change wholesale. A survivor may not.

    Replacing the M1 path moves the reference price from 4043 to 4000, and only
    one candidate survives it - the wide H1 band 3960-4050. That band contains
    both prices, so its relation stays OVERLAPS and its distance stays 0: this
    fixture proves identity stability across a price move, and the distance
    *moving* is proved beside it on a constructed reading, where both readings
    can be made to keep the same candidate.
    """
    from tests.test_ict_candidate_eligibility_fixture import staggered_snapshot
    from tests.test_ict_range_fixture import QUIET

    changed = dict(PATHS)
    changed[Timeframe.M1] = [*QUIET, *QUIET, *QUIET, *QUIET]
    moved = build_candidate_features(
        consolidate_candidates(
            analyse_candidate_eligibility(
                analyse_ict_composite(staggered_snapshot(changed), config=config()),
                config=eligibility_config(),
            )
        )
    )
    before = graph()

    assert (before.reference_price.price, moved.reference_price.price) == (
        Decimal("4043"),
        Decimal("4000"),
    )
    shared = identities(before) & identities(moved)
    assert shared == {"097b4590617fa647"}
    for identity in shared:
        assert by_id(before)[identity] == by_id(moved)[identity]

    survivor_before = before.feature("097b4590617fa647")
    survivor_after = moved.feature("097b4590617fa647")
    assert survivor_before is not None and survivor_after is not None
    assert survivor_before.lower is not None and survivor_before.upper is not None
    assert survivor_before.lower <= Decimal("4000") <= survivor_before.upper
    assert survivor_before.market_relation is survivor_after.market_relation


def test_e_distance_and_relation_do_move_with_the_price() -> None:
    """§44 E, the other half, on a reading where the candidate survives the move."""
    from tests.test_ict_candidate_consolidation import decide, eligibility
    from tests.test_ict_candidate_eligibility import gap_source

    near = build_candidate_features(
        consolidate_candidates(
            eligibility(decide(gap_source("3990", "4010"), price="4100"), price="4100")
        )
    )
    far = build_candidate_features(
        consolidate_candidates(
            eligibility(decide(gap_source("3990", "4010"), price="4200"), price="4200")
        )
    )

    assert near.candidates[0].candidate_id == far.candidates[0].candidate_id
    assert (near.candidates[0].lower, near.candidates[0].upper) == (
        far.candidates[0].lower,
        far.candidates[0].upper,
    )
    assert near.candidates[0].distance_to_reference == Decimal("90")
    assert far.candidates[0].distance_to_reference == Decimal("190")


def test_f_support_membership_moves_through_time_while_identity_does_not() -> None:
    """§44 F. H1 bar 62 gives 3960-4050 a second supporter and not a new id."""
    before = at(h1_close(61)).feature("097b4590617fa647")
    after = at(h1_close(62)).feature("097b4590617fa647")

    assert before is not None and after is not None
    assert before.support_count == 1
    assert after.support_count == 2
    assert (before.lower, before.upper, before.role, before.entry_side) == (
        after.lower,
        after.upper,
        after.role,
        after.entry_side,
    )
    assert before.timeframe_count == after.timeframe_count == 1
    assert before.source_kind_count == after.source_kind_count == 1
    assert before.oldest_source_age_seconds != after.oldest_source_age_seconds


def test_f_a_narrower_policy_removes_candidates_whole_in_this_fixture() -> None:
    """§44 F, recorded as the fact it is rather than assumed to be otherwise.

    Narrowing the gap policy on this snapshot does not thin a candidate's
    support - it removes candidates entirely, and the survivors keep every
    supporter they had. Worth stating, because "membership may change" is easy
    to read as "membership always changes gradually", and here it does not.
    """
    everything = graph()
    open_only = graph(gaps=frozenset({FvgStatus.OPEN}))

    shared = identities(everything) & identities(open_only)
    assert identities(open_only) < identities(everything)
    assert len(shared) == 3

    for identity in shared:
        wide = everything.feature(identity)
        narrow = open_only.feature(identity)
        assert wide is not None and narrow is not None
        assert by_id(everything)[identity] == by_id(open_only)[identity]
        assert wide.support_count == narrow.support_count
        assert wide.support_facts == narrow.support_facts


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_two_independent_readings_agree() -> None:
    assert graph() == graph()


REPLAY_PROGRAM = """
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from tests.test_ict_candidate_eligibility_fixture import analysis
from goldpipeline.services.ict_candidate_consolidation import consolidate_candidates
from goldpipeline.services.ict_candidate_features import build_candidate_features
from goldpipeline.services.ict_fvg import FvgStatus
from goldpipeline.services.ict_order_block_lifecycle import OrderBlockStatus

for blocks in (
    frozenset({OrderBlockStatus.ACTIVE}),
    frozenset({OrderBlockStatus.ACTIVE, OrderBlockStatus.TOUCHED, OrderBlockStatus.MITIGATED}),
):
    for gaps in (frozenset({FvgStatus.OPEN}), frozenset({FvgStatus.OPEN, FvgStatus.TOUCHED})):
        result = build_candidate_features(
            consolidate_candidates(analysis(blocks=blocks, gaps=gaps))
        )
        for t in result.timeframe_contexts:
            print("TF", t.timeframe.value, t.structure_bias.value, t.active_range_id,
                  t.range_lower, t.range_upper, t.range_equilibrium,
                  t.decision_count, t.eligible_count)
        for f in result.candidates:
            print(" C", f.candidate_id, f.role.value, f.entry_side.value if f.entry_side else "-",
                  f.lower, f.upper, f.midpoint, f.width, f.reference_level,
                  f.market_relation.value, f.distance_to_reference,
                  f.support_count, f.timeframe_count, f.source_kind_count,
                  [k.value for k in f.source_kinds], [t.value for t in f.timeframes],
                  f.oldest_source_age_seconds, f.newest_source_age_seconds)
            for s in f.support_facts:
                print("   S", s.candidate_source_id, s.source_id, s.source_kind.value,
                      s.timeframe.value, s.formed_at.isoformat(), s.age_seconds,
                      s.order_block_status, s.order_block_zone_basis,
                      s.order_block_mitigation_rule, s.fvg_status,
                      s.liquidity_side, s.liquidity_pool_status, s.liquidity_tolerance)
            for r in f.range_contexts:
                print("   R", r.timeframe.value, r.active_range_id, r.range_lower, r.range_upper,
                      r.lower_location, r.midpoint_location, r.upper_location,
                      r.reference_location)
        for p in result.pair_relations:
            print(" P", p.first_candidate_id, p.second_candidate_id, p.relation.value,
                  p.intersection_lower, p.intersection_upper, p.intersection_width,
                  p.gap_distance)
"""


@pytest.mark.parametrize("seed", ["0", "1", "524287"])
def test_string_hash_ordering_cannot_reach_the_output(seed: str) -> None:
    """§41-adjacent. Frozensets in the policy, dict lookups in the timeframe map."""
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
    from pathlib import Path

    text = Path("src/goldpipeline/services/ict_candidate_features.py").read_text(encoding="utf-8")

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


def test_the_feature_graph_is_stable_across_processes() -> None:
    """The six ids and their ages, pinned to actual values."""
    result = graph()

    assert [
        (feature.candidate_id, feature.oldest_source_age_seconds) for feature in result.candidates
    ] == [
        ("097b4590617fa647", 3780),
        ("a5cc2e8b2526914a", 360),
        ("b0ac49bdde522f4a", 240),
        ("edf4afe7ca543f85", 300),
        ("17d67732ea0614e7", 0),
        ("be94c8741bcb02a3", 60),
    ]
    assert result.reference_price.price == Decimal("4043")
    assert len(staggered_snapshot().timeframes) == 5
    assert H1_BARS > 0
