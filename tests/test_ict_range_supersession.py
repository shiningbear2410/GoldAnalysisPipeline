"""What ends a dealing range, and what deliberately does not.

Round 6.6c.3b §17-§23, §46. A range is never invalidated by price - only
replaced, by a later structural event. There are exactly two ways that happens,
and one important case where it does not.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from goldpipeline.services.ict_protected import analyse_protected_structure
from goldpipeline.services.ict_range import (
    RangeStatus,
    analyse_dealing_ranges,
)
from goldpipeline.services.ict_structure import (
    BreakClassification,
    BreakDirection,
    StructureBias,
    analyse_structure,
)
from tests.test_ict_protected import closing_bar
from tests.test_ict_range_fixture import JOURNEY, journey
from tests.test_ict_structure import START, series
from tests.test_ict_structure_fixture import JOURNEY as SPARSE

HOUR = timedelta(hours=1)


def at(hours: int) -> object:
    return analyse_dealing_ranges(journey(), symbol="XAUUSD", as_of=START + HOUR * hours)


# --------------------------------------------------------------------------
# §46 A-D: the bullish sequence
# --------------------------------------------------------------------------


def test_a_bullish_break_opens_an_active_range() -> None:
    """§46 A."""
    analysis = at(12)

    assert len(analysis.ranges) == 1  # type: ignore[attr-defined]
    assert analysis.ranges[0].status is RangeStatus.ACTIVE  # type: ignore[attr-defined]
    assert analysis.active_range_id == analysis.ranges[0].range_id  # type: ignore[attr-defined]


def test_a_newer_low_confirming_without_a_break_changes_nothing() -> None:
    """§46 B, §22. Becoming ``active_low`` is not becoming an anchor."""
    structure = analyse_structure(journey(), symbol="XAUUSD", as_of=START + HOUR * 26)
    analysis = at(26)
    current = analysis.active_range  # type: ignore[attr-defined]

    assert structure.active_low is not None
    assert structure.active_low.swing.price == Decimal("3980"), "the newer low is active"
    assert current is not None
    assert current.origin_price == Decimal("3970"), "the range is not re-anchored"
    assert current.status is RangeStatus.ACTIVE


def test_a_later_break_on_a_new_anchor_supersedes_and_replaces() -> None:
    """§46 C, §17 A."""
    before = at(26)
    after = at(31)

    old = before.active_range  # type: ignore[attr-defined]
    new = after.active_range  # type: ignore[attr-defined]
    assert old is not None
    assert new is not None
    assert new.range_id != old.range_id
    assert new.origin_price == Decimal("3980")

    retired = after.range_of(old.range_id)  # type: ignore[attr-defined]
    assert retired is not None
    assert retired.status is RangeStatus.SUPERSEDED
    assert closing_bar(retired.superseded_at) == 30
    assert retired.superseded_by_range_id == new.range_id


def test_a_reaffirming_break_also_supersedes_and_replaces() -> None:
    """§46 D, §21. The same anchor, a new event, a new range."""
    analysis = at(19)
    first, second = analysis.ranges  # type: ignore[attr-defined]

    assert first.origin_price == second.origin_price == Decimal("3970")
    assert first.status is RangeStatus.SUPERSEDED
    assert first.superseded_by_range_id == second.range_id
    assert second.status is RangeStatus.ACTIVE
    assert first.range_id != second.range_id


# --------------------------------------------------------------------------
# §46 E, §23: reversal with an anchor
# --------------------------------------------------------------------------


def test_a_bearish_mss_with_an_anchor_replaces_the_bullish_range() -> None:
    """§46 E."""
    before = at(33)
    after = at(34)

    bullish = before.active_range  # type: ignore[attr-defined]
    bearish = after.active_range  # type: ignore[attr-defined]
    assert bullish is not None
    assert bearish is not None
    assert bullish.direction is BreakDirection.BULLISH
    assert bearish.direction is BreakDirection.BEARISH
    assert bearish.origin_price == Decimal("4200")

    retired = after.range_of(bullish.range_id)  # type: ignore[attr-defined]
    assert retired is not None
    assert retired.status is RangeStatus.SUPERSEDED
    assert retired.superseded_by_range_id == bearish.range_id


def test_the_old_bullish_range_is_not_reused_in_a_bearish_state() -> None:
    """§23. A new direction gets a new range or none at all."""
    after = at(34)

    assert after.structure_bias is StructureBias.BEARISH  # type: ignore[attr-defined]
    actives = [entry for entry in after.ranges if entry.is_active]  # type: ignore[attr-defined]
    assert len(actives) == 1
    assert actives[0].direction is BreakDirection.BEARISH


# --------------------------------------------------------------------------
# §46 F, §17 B: reversal with no anchor
# --------------------------------------------------------------------------


def test_an_mss_with_no_anchor_supersedes_and_leaves_nothing() -> None:
    """§46 F, §17 B. Round 6.6b's own journey, where the opposite side is gone.

    The bearish MSS at bar 24 has no eligible high to anchor on. The bullish
    range still cannot speak for a market that has turned, so it ends - and
    nothing replaces it until a later event establishes an anchor.
    """
    snapshot = series(SPARSE)
    structure = analyse_structure(snapshot, symbol="XAUUSD")
    protected = analyse_protected_structure(snapshot, symbol="XAUUSD")
    analysis = analyse_dealing_ranges(snapshot, symbol="XAUUSD")

    mss = [e for e in structure.breaks if e.classification is BreakClassification.MSS][0]
    assert mss.opposite_active_swing_id is None
    assert len(protected.legs) == 2, "only the two bullish events anchored"

    assert len(analysis.ranges) == 2
    assert all(entry.status is RangeStatus.SUPERSEDED for entry in analysis.ranges)
    assert analysis.active_range_id is None

    last = analysis.ranges[-1]
    assert last.superseded_at is not None
    assert closing_bar(last.superseded_at) == 24
    assert last.superseded_by_event_id == mss.event_id
    assert last.superseded_by_range_id is None, "nothing took its place"


def test_a_bias_with_no_range_is_a_legitimate_state() -> None:
    """The audit finding, pinned.

    Round 6.6b's journey ends BULLISH, and ``current_assignment`` would happily
    return the bar-21 bullish anchor - but that anchor's range was superseded at
    bar 24 and must not come back to life. Range currency is decided by the
    range history's own supersession, never by re-deriving the latest matching
    anchor.
    """
    snapshot = series(SPARSE)
    protected = analyse_protected_structure(snapshot, symbol="XAUUSD")
    analysis = analyse_dealing_ranges(snapshot, symbol="XAUUSD")

    assert analysis.structure_bias is StructureBias.BULLISH
    assert protected.current_assignment is not None, "an anchor still matches the bias"
    assert analysis.active_range_id is None, "but its range is long dead"


# --------------------------------------------------------------------------
# §46 G, §18: a same-direction event with no anchor
# --------------------------------------------------------------------------


def test_a_same_direction_event_with_no_anchor_kills_nothing() -> None:
    """§46 G, §18.

    Round 6.6b's journey contains a bearish BOS at bar 31 that established no
    anchor. It creates no range and, just as importantly, destroys none - the
    branch simply leaves the current state alone.

    Reachable, but never *with an active range*: for a bullish range's protected
    low to have been consumed, a bearish close must have taken it out, and from
    a bullish state that close is an MSS, which supersedes the range and flips
    the bias. So by the time a same-direction event can find itself anchorless,
    there is no range for this branch to protect. Defined anyway, because that
    reasoning rests on Round 6.6b's classification rules.
    """
    snapshot = series(SPARSE)
    structure = analyse_structure(snapshot, symbol="XAUUSD")

    bos = [
        e
        for e in structure.breaks
        if e.classification is BreakClassification.BOS and e.opposite_active_swing_id is None
    ]
    assert len(bos) == 1
    assert closing_bar(bos[0].break_bar_close_time) == 31

    before = analyse_dealing_ranges(snapshot, symbol="XAUUSD", as_of=START + HOUR * 31)
    after = analyse_dealing_ranges(snapshot, symbol="XAUUSD", as_of=START + HOUR * 32)

    assert before.active_range_id is None
    assert after.active_range_id is None
    assert [entry.range_id for entry in before.ranges] == [entry.range_id for entry in after.ranges]
    assert [entry.status for entry in before.ranges] == [entry.status for entry in after.ranges]


# --------------------------------------------------------------------------
# §46 H-I, §19: the superseding bar belongs to both
# --------------------------------------------------------------------------


def test_the_superseding_bar_extends_the_old_edge_first() -> None:
    """§46 H. The event bar is a fully observed candle during the old range's life."""
    before = at(18)
    after = at(19)

    old_live = before.active_range  # type: ignore[attr-defined]
    assert old_live is not None
    assert old_live.terminal_price == Decimal("4055"), "before bar 18 closed"

    old_retired = after.range_of(old_live.range_id)  # type: ignore[attr-defined]
    assert old_retired is not None
    assert old_retired.terminal_price == Decimal("4070"), "bar 18's own high"
    assert closing_bar(old_retired.terminal_bar_close_time) == 18
    assert closing_bar(old_retired.superseded_at) == 18


def test_the_new_range_starts_from_the_same_bars_leg_terminal() -> None:
    """§46 I. One candle, two roles, both honest."""
    after = at(19)
    old, new = after.ranges  # type: ignore[attr-defined]

    assert old.terminal_price == new.initial_terminal_price == Decimal("4070")
    assert closing_bar(old.terminal_bar_close_time) == 18
    assert closing_bar(new.formed_at) == 18
    assert closing_bar(new.terminal_bar_close_time) == 18


def test_every_range_in_the_journey_is_superseded_by_a_structure_event() -> None:
    """No range ever ends for any other reason."""
    structure = analyse_structure(journey(), symbol="XAUUSD")
    event_ids = {e.event_id for e in structure.breaks}

    for entry in analyse_dealing_ranges(journey(), symbol="XAUUSD").ranges:
        if entry.status is RangeStatus.SUPERSEDED:
            assert entry.superseded_by_event_id in event_ids
            assert entry.superseded_at is not None
        else:
            assert entry.superseded_at is None
            assert entry.superseded_by_event_id is None
            assert entry.superseded_by_range_id is None


def test_at_most_one_range_is_ever_active() -> None:
    for kept in range(1, len(JOURNEY) + 1):
        analysis = analyse_dealing_ranges(journey(), symbol="XAUUSD", as_of=START + HOUR * kept)
        actives = [entry for entry in analysis.ranges if entry.is_active]
        assert len(actives) <= 1
        if actives:
            assert analysis.active_range_id == actives[0].range_id
        else:
            assert analysis.active_range_id is None


def test_an_active_range_always_agrees_with_the_structure_bias() -> None:
    """§24. Disagreement is an invariant failure, so it must never occur."""
    for kept in range(1, len(JOURNEY) + 1):
        analysis = analyse_dealing_ranges(journey(), symbol="XAUUSD", as_of=START + HOUR * kept)
        current = analysis.active_range
        if current is None:
            continue
        assert current.direction.value == analysis.structure_bias.value
