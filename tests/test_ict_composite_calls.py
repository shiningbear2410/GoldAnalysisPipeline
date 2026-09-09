"""How many times each authority actually runs, counted rather than assumed.

Round 6.6e.1 §3, §5-§12, §29-§30. This is the file the round exists for. Every
other test proves the composite gets the *right* answers; these prove it gets
them once, which is a different claim and the one that stops five equal-today
copies of "the swings" from drifting apart tomorrow.

Counting is done by wrapping each authority where its consumers import it, so a
stage that quietly rederives something is caught wherever it reaches for it.
Only domain entry points are counted - a pure helper called inside one authority
is that authority's business.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import ICT_TIMEFRAMES, IctMarketSnapshot
from goldpipeline.services import (
    ict_composite,
    ict_fvg,
    ict_liquidity,
    ict_order_block,
    ict_order_block_lifecycle,
    ict_primitives,
    ict_protected,
    ict_range,
    ict_structure,
)
from goldpipeline.services.ict_composite import analyse_ict_composite, analyse_ict_timeframe
from tests.test_ict_composite_fixture import config, divergent_snapshot

AUTHORITIES: dict[str, tuple[tuple[Any, str], ...]] = {
    # Each entry lists every module that reaches for this authority, so a call
    # from any consumer is counted, not only the composite's own.
    "confirmed_swings": (
        (ict_primitives, "confirmed_swings"),
        (ict_structure, "confirmed_swings"),
        (ict_liquidity, "confirmed_swings"),
        (ict_composite, "confirmed_swings"),
    ),
    "fair_value_gaps": (
        (ict_primitives, "fair_value_gaps"),
        (ict_fvg, "fair_value_gaps"),
        (ict_composite, "fair_value_gaps"),
    ),
    "latest_atr": ((ict_primitives, "latest_atr"), (ict_composite, "latest_atr")),
    "analyse_structure": (
        (ict_structure, "analyse_structure"),
        (ict_protected, "analyse_structure"),
        (ict_range, "analyse_structure"),
        (ict_order_block, "analyse_structure"),
        (ict_composite, "analyse_structure"),
    ),
    "analyse_liquidity": (
        (ict_liquidity, "analyse_liquidity"),
        (ict_composite, "analyse_liquidity"),
    ),
    "analyse_protected_structure": (
        (ict_protected, "analyse_protected_structure"),
        (ict_range, "analyse_protected_structure"),
        (ict_order_block, "analyse_protected_structure"),
        (ict_composite, "analyse_protected_structure"),
    ),
    "analyse_dealing_ranges": (
        (ict_range, "analyse_dealing_ranges"),
        (ict_composite, "analyse_dealing_ranges"),
    ),
    "analyse_order_blocks": (
        (ict_order_block, "analyse_order_blocks"),
        (ict_order_block_lifecycle, "analyse_order_blocks"),
        (ict_composite, "analyse_order_blocks"),
    ),
    "analyse_order_block_lifecycle": (
        (ict_order_block_lifecycle, "analyse_order_block_lifecycle"),
        (ict_composite, "analyse_order_block_lifecycle"),
    ),
    "analyse_fvg_lifecycle": (
        (ict_fvg, "analyse_fvg_lifecycle"),
        (ict_composite, "analyse_fvg_lifecycle"),
    ),
}


@contextmanager
def counted(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, int]]:
    """Count every call to every domain authority, wherever it is reached for."""
    tally: dict[str, int] = dict.fromkeys(AUTHORITIES, 0)

    for name, sites in AUTHORITIES.items():
        original = getattr(sites[0][0], name)

        def spy(
            *args: object, _name: str = name, _fn: Callable[..., Any] = original, **kwargs: object
        ) -> Any:
            tally[_name] += 1
            return _fn(*args, **kwargs)

        for module, attribute in sites:
            monkeypatch.setattr(module, attribute, spy)

    yield tally


def test_the_spy_harness_can_actually_see_a_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A counting test that cannot detect double work would prove nothing.

    Called the naive way - every authority from raw bars, each deriving its own
    prerequisites - one timeframe costs far more than one call apiece. That is
    the number the composite has to beat.
    """
    series = divergent_snapshot().require(Timeframe.H1)
    settings = config()

    with counted(monkeypatch) as tally:
        ict_structure.analyse_structure(series, symbol="XAUUSD")
        ict_liquidity.analyse_liquidity(series, config=settings.liquidity, symbol="XAUUSD")
        ict_range.analyse_dealing_ranges(series, symbol="XAUUSD")
        ict_order_block_lifecycle.analyse_order_block_lifecycle(
            series,
            config=settings.order_block_lifecycle,
            formation=settings.order_block,
            symbol="XAUUSD",
        )

    assert tally["confirmed_swings"] > 1, "the naive path really does recompute swings"
    assert tally["analyse_structure"] > 1
    assert tally["analyse_protected_structure"] > 1


# --------------------------------------------------------------------------
# §29: one timeframe
# --------------------------------------------------------------------------


def test_one_timeframe_runs_each_authority_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """§29. Ten authorities, ten calls."""
    series = divergent_snapshot().require(Timeframe.H1)

    with counted(monkeypatch) as tally:
        analyse_ict_timeframe(series, config=config(), symbol="XAUUSD")

    assert tally == dict.fromkeys(AUTHORITIES, 1)


def test_one_timeframe_computes_swings_once_for_two_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§3. Structure and liquidity share one collection, not two equal ones."""
    series = divergent_snapshot().require(Timeframe.H1)

    with counted(monkeypatch) as tally:
        entry = analyse_ict_timeframe(series, config=config(), symbol="XAUUSD")

    assert tally["confirmed_swings"] == 1
    assert tally["analyse_structure"] == 1
    assert tally["analyse_liquidity"] == 1
    assert entry.swings, "an empty collection would make the count meaningless"


def test_one_timeframe_detects_gaps_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """§5. The gap lifecycle is fed, not left to redetect."""
    series = divergent_snapshot().require(Timeframe.M1)

    with counted(monkeypatch) as tally:
        entry = analyse_ict_timeframe(series, config=config(), symbol="XAUUSD")

    assert tally["fair_value_gaps"] == 1
    assert tally["analyse_fvg_lifecycle"] == 1
    assert len(entry.gaps) == 13


def test_the_downstream_stages_never_rederive_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§7, §8. Protected, ranges and order blocks all take what they are given."""
    series = divergent_snapshot().require(Timeframe.H1)

    with counted(monkeypatch) as tally:
        analyse_ict_timeframe(series, config=config(), symbol="XAUUSD")

    assert tally["analyse_structure"] == 1, "four stages could have wanted one"
    assert tally["analyse_protected_structure"] == 1, "three stages could have wanted one"


def test_the_lifecycle_never_reruns_formation(monkeypatch: pytest.MonkeyPatch) -> None:
    """§12. The blocks are handed over, so formation runs once."""
    series = divergent_snapshot().require(Timeframe.H1)

    with counted(monkeypatch) as tally:
        entry = analyse_ict_timeframe(series, config=config(), symbol="XAUUSD")

    assert tally["analyse_order_blocks"] == 1
    assert tally["analyse_order_block_lifecycle"] == 1
    assert len(entry.order_blocks.order_blocks) == 5


def test_a_quiet_timeframe_still_runs_each_authority_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeframe with nothing to report is not a timeframe that skipped work."""
    series = divergent_snapshot().require(Timeframe.M5)

    with counted(monkeypatch) as tally:
        entry = analyse_ict_timeframe(series, config=config(), symbol="XAUUSD")

    assert tally == dict.fromkeys(AUTHORITIES, 1)
    assert entry.structure.breaks == ()


# --------------------------------------------------------------------------
# §30: five timeframes
# --------------------------------------------------------------------------


def test_five_timeframes_run_each_authority_exactly_five_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§30. Five, not ten, not fifteen, not twenty-five."""
    with counted(monkeypatch) as tally:
        analyse_ict_composite(divergent_snapshot(), config=config())

    assert tally == dict.fromkeys(AUTHORITIES, 5)


def test_swings_and_gaps_are_computed_once_per_timeframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§30's two named cases, stated on their own."""
    with counted(monkeypatch) as tally:
        analyse_ict_composite(divergent_snapshot(), config=config())

    assert tally["confirmed_swings"] == len(ICT_TIMEFRAMES) == 5
    assert tally["fair_value_gaps"] == 5


def test_a_subset_snapshot_costs_only_what_it_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§37. Two timeframes, two of each call."""
    shot = divergent_snapshot()
    subset = IctMarketSnapshot(
        observed_at=shot.observed_at,
        symbol=shot.symbol,
        provider=shot.provider,
        provider_symbol=shot.provider_symbol,
        timeframes=(shot.require(Timeframe.H1), shot.require(Timeframe.M15)),
    )

    with counted(monkeypatch) as tally:
        analyse_ict_composite(subset, config=config())

    assert tally == dict.fromkeys(AUTHORITIES, 2)


def test_the_counts_hold_under_every_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Threading is not a property of one lucky configuration."""
    from goldpipeline.services.ict_order_block import OrderBlockZoneBasis
    from goldpipeline.services.ict_order_block_lifecycle import OrderBlockMitigationRule

    for basis in OrderBlockZoneBasis:
        for rule in OrderBlockMitigationRule:
            with counted(monkeypatch) as tally:
                analyse_ict_composite(divergent_snapshot(), config=config(basis=basis, rule=rule))
            assert tally == dict.fromkeys(AUTHORITIES, 5), (basis, rule)


def test_the_composite_calls_nothing_the_registry_does_not_know_about() -> None:
    """The counting is only as honest as the list of authorities it watches.

    Every ICT entry point the composite module names must appear in
    :data:`AUTHORITIES`, so a stage added later cannot escape the count by
    being new.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path("src/goldpipeline/services/ict_composite.py").read_text(encoding="utf-8"))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    domain = {
        name
        for name in called
        if name.startswith(("analyse_", "confirmed_", "fair_", "latest_"))
        # The composite calling its own per-timeframe function is orchestration
        # calling orchestration, not a domain authority running twice.
        and name != "analyse_ict_timeframe"
    }

    assert domain == set(AUTHORITIES), domain ^ set(AUTHORITIES)
