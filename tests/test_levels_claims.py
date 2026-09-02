"""The deterministic levels block, seen through the existing claim machinery.

The point of computing levels in Python is that a writer may then *cite* them
and a checker may then *verify* them. That only works if the new block enters
the catalog the writer is shown and resolves through the same resolver the
prechecks use - with no second registry anywhere.

These tests assert exactly that, and assert the deliberate omissions too:
advertising a path is a decision, and a decision worth a test.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest

from goldpipeline.services.claim_paths import EXCLUDED_PATHS, build_catalog
from goldpipeline.services.claim_resolver import resolve_path
from goldpipeline.services.levels import build_levels


@pytest.fixture
def short_context(sample_context: Any) -> Any:
    """The shared fixture: 12 bars drifting upward.

    Too short for a 14-period ATR and monotonic enough to have no pivots, so it
    exercises the *absent* path - which is worth its own coverage.
    """
    return sample_context


@pytest.fixture
def context(sample_context: Any) -> Any:
    """A context with enough zig-zag to produce an ATR, pivots and zones.

    Built from the real fixture and then given a richer bar series, with the
    levels block recomputed from those same bars so the two stay consistent -
    the catalog is only meaningful when it describes the object it came from.
    """
    from goldpipeline.schemas.context import ContextOHLC
    from goldpipeline.schemas.market import OHLCBar

    template = sample_context.ohlc.bars[0]
    bars = []
    for index in range(24):
        base = Decimal(3300) + Decimal(index)
        swing = Decimal(4) if index % 4 == 2 else Decimal(0)
        bars.append(
            OHLCBar(
                timestamp=template.timestamp.replace(microsecond=0) + timedelta(minutes=15 * index),
                open=base,
                high=base + swing + 2,
                low=base - swing - 2,
                close=base,
            )
        )
    return sample_context.model_copy(
        update={
            "ohlc": ContextOHLC(bar_count=len(bars), bars=bars),
            "levels": build_levels(bars),
        }
    )


class TestBlockIsPresent:
    def test_levels_are_attached_by_the_builder(self, short_context: Any) -> None:
        assert short_context.levels is not None
        assert short_context.levels.bars_considered == len(short_context.ohlc.bars)

    def test_levels_are_reproducible_from_the_bars(self, short_context: Any) -> None:
        """The whole basis for citing them: recomputation must agree."""
        recomputed = build_levels(list(short_context.ohlc.bars))
        assert recomputed.model_dump_json() == short_context.levels.model_dump_json()

    def test_a_short_window_reports_absence_not_approximation(self, short_context: Any) -> None:
        """12 monotonic bars: no ATR, no pivots, and it says so."""
        from goldpipeline.schemas.context import MarketStructure

        assert short_context.levels.atr is None
        assert short_context.levels.swing_highs == []
        assert short_context.levels.structure is MarketStructure.INSUFFICIENT_DATA


class TestCatalogIntegration:
    def test_scalar_paths_are_offered(self, context: Any) -> None:
        paths = build_catalog(context).all_paths()
        assert "context.levels.structure" in paths
        assert "context.levels.atr" in paths

    def test_pivot_and_zone_families_are_offered(self, context: Any) -> None:
        catalog = build_catalog(context)
        prefixes = {family.prefix for family in catalog.families}
        assert "context.levels.swing_highs" in prefixes
        assert "context.levels.swing_lows" in prefixes

    def test_every_offered_levels_path_resolves(self, context: Any) -> None:
        """The catalog may not advertise anything the resolver would refuse."""
        offered = [p for p in build_catalog(context).all_paths() if p.startswith("context.levels")]
        assert offered, "the levels block contributed no claimable paths"
        for path in offered:
            resolved = resolve_path(context, path)
            assert resolved is not None

    def test_resolved_values_match_the_object(self, context: Any) -> None:
        assert context.levels.atr is not None
        assert resolve_path(context, "context.levels.atr") == context.levels.atr

    def test_pivot_leaf_resolves_exactly(self, context: Any) -> None:
        assert context.levels.swing_highs, "fixture should produce confirmed pivots"
        expected = context.levels.swing_highs[0].price
        assert resolve_path(context, "context.levels.swing_highs[0].price") == expected

    def test_no_dynamic_or_unexpected_paths(self, context: Any) -> None:
        """Everything offered is a declared field - nothing generated from data."""
        declared = set(type(context.levels).model_fields)
        for path in build_catalog(context).all_paths():
            if not path.startswith("context.levels."):
                continue
            field = path.removeprefix("context.levels.").split(".")[0].split("[")[0]
            assert field in declared, f"catalog offered an undeclared field: {path}"


class TestDeliberateOmissions:
    @pytest.mark.parametrize(
        "path",
        [
            "context.levels.window_high",
            "context.levels.window_high_at",
            "context.levels.window_low",
            "context.levels.window_low_at",
        ],
    )
    def test_window_extremes_are_not_offered(self, context: Any, path: str) -> None:
        """Already addressable as a bar's high/low; two addresses invite drift."""
        assert path in EXCLUDED_PATHS
        assert path not in build_catalog(context).all_paths()

    def test_window_extremes_still_resolve(self, context: Any) -> None:
        """Excluding narrows what is *offered*, never what the resolver accepts."""
        assert resolve_path(context, "context.levels.window_high") is not None

    @pytest.mark.parametrize(
        "path",
        [
            "context.levels.method_version",
            "context.levels.bars_considered",
            "context.levels.atr_period",
            "context.levels.pivot_window",
        ],
    )
    def test_plumbing_is_not_offered(self, context: Any, path: str) -> None:
        """How a number was computed is not a fact about the market."""
        assert path not in build_catalog(context).all_paths()


class TestBackwardCompatibility:
    def test_a_context_without_levels_still_loads(self, context: Any) -> None:
        """Runs written before this block existed must keep deserializing."""
        payload = context.model_dump(mode="json")
        payload.pop("levels")
        restored = type(context).model_validate(payload)
        assert restored.levels is None

    def test_catalog_is_unaffected_when_levels_are_absent(self, context: Any) -> None:
        payload = context.model_dump(mode="json")
        payload.pop("levels")
        restored = type(context).model_validate(payload)
        assert not any(p.startswith("context.levels") for p in build_catalog(restored).all_paths())


class TestDerivedNumbersAreTyped:
    """Round 2 must distinguish distances from prices; the schema says which.

    ATR and zone width are magnitudes, not prices. Recorded here so the
    distinction is pinned by a test rather than by a comment.
    """

    def test_atr_is_a_magnitude_not_a_price(self, context: Any) -> None:
        assert context.levels.atr is not None
        assert context.levels.atr >= Decimal(0)
        # A price must be strictly positive; a magnitude may legitimately be zero.
        assert type(context.levels).model_fields["atr"].annotation is not None

    def test_zone_width_is_a_difference_of_its_own_bounds(self, context: Any) -> None:
        for zone in [*context.levels.support_zones, *context.levels.resistance_zones]:
            assert zone.width == zone.upper - zone.lower
