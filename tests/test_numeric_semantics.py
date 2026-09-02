"""Telling a price from a distance, and refusing to guess.

Two failure modes are in tension here, and both are represented below:

* the **false positive** that started this round - ``66.140`` was a correct net
  change, reported as a fabricated price because nothing recorded what kind of
  number it was;
* the **false negative** that would be far worse - a hallucinated level slipping
  through because the scanner was taught to forgive numbers it cannot explain.

So every acceptance test here has a rejection test beside it using the *same
number in a different role*. If the value alone ever decides the outcome, one of
each pair fails.

Offline throughout: no MT5, no model, no network.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from goldpipeline.schemas.context import ContextOHLC
from goldpipeline.schemas.market import OHLCBar
from goldpipeline.schemas.review import FindingCode, Severity
from goldpipeline.schemas.writer import ClaimType, SourceClaim
from goldpipeline.services.levels import build_levels
from goldpipeline.services.market_facts import derived_values
from goldpipeline.services.numeric_semantics import (
    SemanticType,
    classify_path,
    rendered_as,
)
from goldpipeline.services.precheck import run_prechecks

START = datetime(2026, 9, 2, tzinfo=UTC)


def make_bars(count: int = 24) -> list[OHLCBar]:
    """A zig-zag around 4300 so pivots, an ATR and zones all exist."""
    bars = []
    for index in range(count):
        base = Decimal(4300) + Decimal(index) / 2
        swing = Decimal(4) if index % 4 == 2 else Decimal(0)
        bars.append(
            OHLCBar(
                timestamp=START + timedelta(minutes=15 * index),
                open=base,
                high=base + swing + 2,
                low=base - swing - 2,
                close=base,
            )
        )
    return bars


@pytest.fixture
def context(sample_context: Any) -> Any:
    bars = make_bars()
    return sample_context.model_copy(
        update={
            "ohlc": ContextOHLC(bar_count=len(bars), bars=bars),
            "levels": build_levels(bars),
        }
    )


def make_wide_bars(count: int = 24) -> list[OHLCBar]:
    """Swings big enough that ATR and differences exceed the 100 scan floor.

    ``extract_numbers`` ignores anything under 100 - below that a number is a
    lot size or a count, not a gold quote. Cases about "an ATR used as a price"
    are therefore only reachable with an ATR of at least 100, so the fixture
    provides one rather than asserting something the scanner never sees.
    """
    bars = []
    for index in range(count):
        base = Decimal(4000) + Decimal(index) * 40
        swing = Decimal(150) if index % 4 == 2 else Decimal(60)
        bars.append(
            OHLCBar(
                timestamp=START + timedelta(minutes=15 * index),
                open=base,
                high=base + swing,
                low=base - swing,
                close=base,
            )
        )
    return bars


@pytest.fixture
def wide_context(sample_context: Any) -> Any:
    bars = make_wide_bars()
    return sample_context.model_copy(
        update={
            "ohlc": ContextOHLC(bar_count=len(bars), bars=bars),
            "levels": build_levels(bars),
        }
    )


def claim(value: str, source: str) -> SourceClaim:
    return SourceClaim(type=ClaimType.PRICE, value=value, source=source)


def check(context: Any, article: str, claims: list[SourceClaim] | None = None):
    """Run the deterministic prechecks over *article*.

    Builds the artifact directly rather than drafting a Run: these tests are
    about the scanner, and a real Run would tie every case to one fixed bar
    series.
    """
    import hashlib

    from goldpipeline.schemas.writer import WriterResult, WriterStatus

    digest = hashlib.sha256(article.encode("utf-8")).hexdigest()
    result = WriterResult(
        run_id=context.run_id,
        status=WriterStatus.COMPLETED,
        title="test",
        model="test",
        provider="test",
        prompt_version="test",
        context_sha256="0" * 64,
        draft_file="claude_draft.md",
        article_sha256=digest,
        article_chars=len(article),
        source_claims=claims or [],
    )
    return run_prechecks(context=context, writer_result=result, article=article)


def numeric_findings(report: Any) -> list[Any]:
    return [
        f
        for f in report.findings
        if f.code
        in {FindingCode.NUMBER_OUTSIDE_MARKET_RANGE, FindingCode.UNKNOWN_PRICE_LIKE_NUMBER}
    ]


def flagged(report: Any, literal: str) -> bool:
    return any(literal in (f.actual or "") for f in numeric_findings(report))


# ------------------------------------------------------------- classification
class TestClassifyPath:
    def test_bar_price_is_an_absolute_price(self, context: Any) -> None:
        assert classify_path(context, "context.ohlc.bars[0].close") is SemanticType.ABSOLUTE_PRICE

    def test_latest_close_is_an_absolute_price(self, context: Any) -> None:
        assert classify_path(context, "context.price.latest_close") is SemanticType.ABSOLUTE_PRICE

    def test_atr_is_a_magnitude(self, context: Any) -> None:
        """The optional-union case: the marker lives inside `Magnitude | None`."""
        assert classify_path(context, "context.levels.atr") is SemanticType.MAGNITUDE

    def test_zone_width_is_a_magnitude(self, context: Any) -> None:
        if not context.levels.resistance_zones:
            pytest.skip("no zones in this fixture")
        assert (
            classify_path(context, "context.levels.resistance_zones[0].width")
            is SemanticType.MAGNITUDE
        )

    @pytest.mark.parametrize("leaf", ["lower", "upper"])
    def test_zone_bounds_stay_prices(self, context: Any, leaf: str) -> None:
        if not context.levels.resistance_zones:
            pytest.skip("no zones in this fixture")
        path = f"context.levels.resistance_zones[0].{leaf}"
        assert classify_path(context, path) is SemanticType.ABSOLUTE_PRICE

    def test_swing_price_is_a_price(self, context: Any) -> None:
        if not context.levels.swing_highs:
            pytest.skip("no pivots in this fixture")
        assert (
            classify_path(context, "context.levels.swing_highs[0].price")
            is SemanticType.ABSOLUTE_PRICE
        )

    def test_window_extremes_are_prices(self, context: Any) -> None:
        assert classify_path(context, "context.levels.window_high") is SemanticType.ABSOLUTE_PRICE

    def test_non_numeric_field_is_not_price_like(self, context: Any) -> None:
        assert classify_path(context, "context.levels.structure") is SemanticType.NON_MARKET_NUMBER

    @pytest.mark.parametrize(
        "path",
        [
            "context.does.not.exist",
            "context.ohlc.bars[999].close",
            "not_context.price.latest_close",
            "context.__class__",
        ],
    )
    def test_unresolvable_paths_are_never_classified(self, context: Any, path: str) -> None:
        """Unclassified is not a licence."""
        assert classify_path(context, path) is SemanticType.UNKNOWN_PRICE_LIKE


# -------------------------------------------------------------- rendering
class TestRenderedAs:
    @pytest.mark.parametrize("literal", ["66.140", "66.14", "66.1"])
    def test_a_value_matches_its_own_roundings(self, literal: str) -> None:
        assert rendered_as(Decimal("66.140"), literal)

    def test_but_not_its_integer_truncation(self) -> None:
        """ "66" is a different statement from "66.14", so it needs its own source."""
        assert not rendered_as(Decimal("66.140"), "66")

    def test_three_decimal_source_rendered_as_two(self) -> None:
        assert rendered_as(Decimal("4373.127"), "4373.13")

    def test_half_up_boundary(self) -> None:
        assert rendered_as(Decimal("4373.125"), "4373.13")
        assert not rendered_as(Decimal("4373.124"), "4373.13")

    @pytest.mark.parametrize("literal", ["4373.14", "4373.12", "4374", "66.15"])
    def test_a_neighbouring_value_is_not_blessed(self, literal: str) -> None:
        """Tolerance is one ulp of the printed precision, not a margin in points."""
        assert not rendered_as(Decimal("4373.127"), literal)

    def test_a_bare_integer_gets_no_latitude(self) -> None:
        """The rule that keeps a neighbour from blessing a fabricated level.

        Observed on a real draft: with integer rounding allowed, the genuine
        findings for '4326' and '4331' disappeared, because some candle value
        sat within half a point of each.
        """
        assert not rendered_as(Decimal("4325.70"), "4326")
        assert not rendered_as(Decimal("4326.30"), "4326")
        assert rendered_as(Decimal("4326"), "4326")

    def test_a_decimal_literal_still_rounds(self) -> None:
        assert rendered_as(Decimal("4325.704"), "4325.70")

    def test_percentage_sign_is_tolerated(self) -> None:
        assert rendered_as(Decimal("1.5357"), "1.54%")

    def test_garbage_is_not_a_match(self) -> None:
        assert not rendered_as(Decimal("1"), "abc")
        assert not rendered_as(Decimal("1"), "")


# ------------------------------------------------------- the old false positive
class TestNetChange:
    def net_change(self, context: Any) -> Decimal:
        return next(d.value for d in derived_values(context) if d.kind == "NET_CHANGE")

    def test_the_regression_case_no_longer_fires(self, context: Any) -> None:
        """The 66.140 shape: a correct net change, stated in ordinary prose."""
        value = self.net_change(context)
        report = check(context, f"Biến động ròng của cửa sổ là {value} USD.")
        assert not flagged(report, str(value))

    def test_rounded_net_change_is_accepted(self, context: Any) -> None:
        value = self.net_change(context).quantize(Decimal("0.01"))
        report = check(context, f"Cửa sổ tăng {value} USD.")
        assert not flagged(report, str(value))

    def test_a_wrong_net_change_is_still_reported(self, wide_context: Any) -> None:
        wrong = self.net_change(wide_context) + Decimal("777")
        report = check(wide_context, f"Cửa sổ tăng {wrong} USD.")
        assert flagged(report, str(wrong))

    def test_window_range_is_accepted(self, context: Any) -> None:
        value = next(d.value for d in derived_values(context) if d.kind == "WINDOW_RANGE")
        report = check(context, f"Biên độ cửa sổ {value} USD.")
        assert not flagged(report, str(value))

    def test_percentage_is_derived_and_typed(self, context: Any) -> None:
        """The percentage is computed once, in market_facts, and typed as one."""
        pct = next(d for d in derived_values(context) if d.kind == "NET_CHANGE_PERCENT")
        assert pct.semantic == "PERCENTAGE"
        expected = (
            (context.ohlc.bars[-1].close - context.ohlc.bars[0].open)
            / context.ohlc.bars[0].open
            * Decimal(100)
        )
        assert pct.value == expected

    def test_display_rounding_of_the_percentage_matches(self, context: Any) -> None:
        pct = next(d.value for d in derived_values(context) if d.kind == "NET_CHANGE_PERCENT")
        assert rendered_as(pct, str(pct.quantize(Decimal("0.01"))))

    def test_a_wrong_percentage_does_not_match_the_derived_one(self, context: Any) -> None:
        pct = next(d.value for d in derived_values(context) if d.kind == "NET_CHANGE_PERCENT")
        assert not rendered_as(pct, "87.65")

    def test_percent_suffixed_numbers_are_exempt_upstream(self, context: Any) -> None:
        """Pre-existing behaviour, recorded rather than changed.

        ``%`` is in the unit-suffix list, so a percentage never reaches the
        range check at all. Widening that here would create findings this round
        was not asked to create.
        """
        report = check(context, "Tương đương 87.65%.")
        assert not flagged(report, "87.65")


# ------------------------------------------------------------ magnitudes
class TestMagnitudeClaims:
    def test_claimed_atr_is_accepted(self, context: Any) -> None:
        atr = context.levels.atr
        assert atr is not None
        report = check(
            context,
            f"Biên độ trung bình gần đây khoảng {atr} USD.",
            [claim(str(atr), "context.levels.atr")],
        )
        assert not flagged(report, str(atr))

    def test_unclaimed_atr_is_still_reported(self, wide_context: Any) -> None:
        """ATR is real, but nothing said this literal was the ATR.

        Admitting it unconditionally is what would let ``entry <atr>`` pass, so
        the ATR is never in the known set on its own - only via a claim.
        """
        atr = wide_context.levels.atr
        assert atr is not None and atr >= 100, "fixture must clear the scan floor"
        shown = atr.quantize(Decimal("0.01"))  # as an article would print it
        report = check(wide_context, f"Vào lệnh tại {shown}.")
        assert flagged(report, str(shown))

    def test_claimed_zone_width_is_accepted(self, context: Any) -> None:
        zones = context.levels.resistance_zones or context.levels.support_zones
        if not zones or zones[0].width == 0:
            pytest.skip("no non-degenerate zone in this fixture")
        width = zones[0].width
        side = "resistance_zones" if context.levels.resistance_zones else "support_zones"
        report = check(
            context,
            f"Vùng rộng {width} USD.",
            [claim(str(width), f"context.levels.{side}[0].width")],
        )
        assert not flagged(report, str(width))


# --------------------------------------------------------- anti-bypass
class TestTheScannerStaysStrong:
    def test_invented_price_above_the_window_is_reported(self, context: Any) -> None:
        report = check(context, "Giá sẽ chạm 9999.99 trong phiên tới.")
        codes = {f.code for f in numeric_findings(report)}
        assert FindingCode.NUMBER_OUTSIDE_MARKET_RANGE in codes

    def test_invented_price_below_the_window_is_reported(self, context: Any) -> None:
        report = check(context, "Giá lùi về 1234.56.")
        assert flagged(report, "1234.56")

    def test_a_claim_that_does_not_resolve_cannot_bless_a_number(self, context: Any) -> None:
        """The hole this round closed: asserting a citation used to be enough."""
        report = check(
            context,
            "Mục tiêu 8888.88.",
            [claim("8888.88", "context.made.up.path")],
        )
        assert flagged(report, "8888.88")

    def test_a_mismatched_claim_cannot_bless_a_number(self, context: Any) -> None:
        """Path resolves, but to something else entirely."""
        report = check(
            context,
            "Mục tiêu 8888.88.",
            [claim("8888.88", "context.price.latest_close")],
        )
        assert flagged(report, "8888.88")

    def test_a_magnitude_claim_does_not_justify_an_absolute_price(self, context: Any) -> None:
        """Same number, wrong role: the ATR value used as a price target.

        The claim is truthful - the value really is the ATR - so this passes the
        claim check. What must not happen is the scanner treating the article's
        other, unexplained numbers as prices because an ATR happened to match.
        """
        atr = context.levels.atr
        assert atr is not None
        fake_price = Decimal("7777.77")
        report = check(
            context,
            f"ATR {atr}. Mục tiêu {fake_price}.",
            [claim(str(atr), "context.levels.atr")],
        )
        assert flagged(report, str(fake_price))
        assert not flagged(report, str(atr))

    def test_no_arithmetic_search_blesses_arbitrary_differences(self, wide_context: Any) -> None:
        """A difference of two real candle values is not automatically safe."""
        bars = wide_context.ohlc.bars
        difference = bars[5].high - bars[2].low
        assert difference >= 100, "fixture must clear the scan floor"
        report = check(wide_context, f"Khoảng cách {difference} USD.")
        assert flagged(report, str(difference))

    def test_derived_catalog_is_small(self, context: Any) -> None:
        """The exemption surface must stay bounded, not grow with bar count."""
        assert len(derived_values(context)) <= 3


# --------------------------------------------- future trade-plan semantics
class TestFutureTradePlanSemantics:
    """No trade-plan logic here - only that the types would behave correctly."""

    def test_entry_invalidation_target_must_be_prices(self, context: Any) -> None:
        zones = context.levels.support_zones or context.levels.resistance_zones
        if not zones:
            pytest.skip("no zones in this fixture")
        for leaf in ("lower", "upper"):
            side = "support_zones" if context.levels.support_zones else "resistance_zones"
            assert (
                classify_path(context, f"context.levels.{side}[0].{leaf}")
                is SemanticType.ABSOLUTE_PRICE
            )

    def test_stop_distance_would_be_a_magnitude(self, context: Any) -> None:
        assert classify_path(context, "context.levels.atr") is SemanticType.MAGNITUDE

    def test_a_ratio_is_not_a_price(self) -> None:
        """An R multiple has no price scale; it must never reach the range check."""
        assert SemanticType.PERCENTAGE is not SemanticType.ABSOLUTE_PRICE
        assert SemanticType.NON_MARKET_NUMBER is not SemanticType.ABSOLUTE_PRICE


# ------------------------------------------------------------- severity
class TestFindingShape:
    def test_far_numbers_are_high_and_near_ones_medium(self, context: Any) -> None:
        report = check(context, "Giá 9999.99 và 4310.55.")
        by_code = {f.code: f for f in numeric_findings(report)}
        assert by_code[FindingCode.NUMBER_OUTSIDE_MARKET_RANGE].severity is Severity.HIGH
        assert by_code[FindingCode.UNKNOWN_PRICE_LIKE_NUMBER].severity is Severity.MEDIUM

    def test_a_faithful_article_has_no_numeric_findings(self, context: Any) -> None:
        latest = context.ohlc.bars[-1]
        report = check(
            context,
            f"Nến gần nhất đóng tại {latest.close}, cao {latest.high}, thấp {latest.low}.",
        )
        assert numeric_findings(report) == []


class TestLevelsBlockAbsent:
    def test_a_context_without_levels_still_scans(self, sample_context: Any) -> None:
        """Historical Runs carry no levels block; the scanner must not care."""
        report = check(sample_context, "Giá 9999.99.")
        assert flagged(report, "9999.99")


class TestTheProductionRegression:
    """The exact shape that produced the false HIGH finding in production.

    Worth its own class because the mechanism is subtler than "a magnitude was
    mistaken for a price". The article wrote the net change as ``66.140``;
    Vietnamese notation makes a dot a thousands separator, so the extractor read
    **66140** - a number far outside the gold range, and one the context indeed
    does not contain. Type information alone would not have saved it, because
    the value under scrutiny was never 66.14.

    What fixes it is matching against the *printed literal* as well as the
    parsed value: read plainly, ``66.140`` is exactly the derived net change.
    """

    @pytest.fixture
    def net_change_context(self, sample_context: Any) -> Any:
        """A window whose net change is exactly 66.14."""
        bars = [
            OHLCBar(
                timestamp=START + timedelta(minutes=15 * i),
                open=Decimal("4306.987") if i == 0 else Decimal("4350"),
                high=Decimal("4400"),
                low=Decimal("4300"),
                close=Decimal("4373.127") if i == 11 else Decimal("4350"),
            )
            for i in range(12)
        ]
        return sample_context.model_copy(
            update={
                "ohlc": ContextOHLC(bar_count=len(bars), bars=bars),
                "levels": build_levels(bars),
            }
        )

    def test_the_literal_parses_as_a_thousands_separated_integer(self) -> None:
        """Pinning the parser behaviour this whole case rests on."""
        from goldpipeline.services.source_guard import extract_numbers

        matches = extract_numbers("tang 66.140 USD")
        assert [m.value for m in matches] == [Decimal("66140")]

    def test_the_derived_net_change_is_exactly_the_printed_value(
        self, net_change_context: Any
    ) -> None:
        value = next(d.value for d in derived_values(net_change_context) if d.kind == "NET_CHANGE")
        assert value == Decimal("66.140")

    def test_the_false_positive_is_gone(self, net_change_context: Any) -> None:
        report = check(net_change_context, "Biến động ròng của cửa sổ là 66.140 USD.")
        assert not flagged(report, "66.140")

    def test_a_neighbouring_literal_is_still_reported(self, net_change_context: Any) -> None:
        """66.150 is not the net change, and must not ride along."""
        report = check(net_change_context, "Biến động ròng của cửa sổ là 66.150 USD.")
        assert flagged(report, "66.150")

    def test_a_genuine_large_number_is_still_reported(self, net_change_context: Any) -> None:
        """A real 66,140 claim about price is still nonsense for gold."""
        report = check(net_change_context, "Giá vàng đạt 88.240 USD.")
        assert flagged(report, "88.240")
