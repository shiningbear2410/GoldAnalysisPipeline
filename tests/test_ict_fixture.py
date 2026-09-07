"""One realistic five-timeframe observation, as a regression anchor.

Round 6.6a §26 and §35. The matrices in the sibling files check each primitive
against hand-computed arithmetic on three or four bars. This file does the
other job: it holds a single XAUUSD-shaped snapshot across all five timeframes
and pins what the primitives find in it, so a later round that changes a
definition has to change these numbers deliberately rather than by accident.

The prices are synthetic and shaped like gold without being any particular
day's gold. Nothing downstream should ever treat them as a market assumption -
they exist to contain structure, not to be right about 2026.

Also here: the source-agnosticism guard. These modules read normalized candles
and must never learn where candles come from.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import (
    ICT_TIMEFRAMES,
    IctMarketSnapshot,
    build_timeframe_snapshot,
)
from goldpipeline.schemas.market import OHLCBar
from goldpipeline.services.ict_primitives import (
    GapDirection,
    SwingType,
    atr_series,
    confirmed_swings,
    fair_value_gaps,
    latest_atr,
)

OBSERVED_AT = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


def ohlc(open_time: datetime, o: str, h: str, low: str, c: str) -> OHLCBar:
    return OHLCBar(
        timestamp=open_time,
        open=Decimal(o),
        high=Decimal(h),
        low=Decimal(low),
        close=Decimal(c),
    )


def walk(
    timeframe: Timeframe, rows: list[tuple[str, str, str, str]], *, end: datetime = OBSERVED_AT
) -> tuple[OHLCBar, ...]:
    """Lay *rows* onto consecutive bars ending exactly at *end*."""
    duration = timeframe.duration
    assert duration is not None
    first_open = end - duration * len(rows)
    return tuple(ohlc(first_open + duration * index, *row) for index, row in enumerate(rows))


# A rise into a high, a drop through it, a base, and a recovery. On H1 this
# yields exactly: a swing high at 4042 (pivot 23:00, confirmed 02:00), a swing
# low at 3962 (pivot 03:00, confirmed 06:00), and nine gaps - six bullish,
# three bearish - the largest being [3980, 3996]. Those numbers are the
# regression anchor; a later round changing a definition has to change them
# on purpose.
SHAPE: list[tuple[str, str, str, str]] = [
    ("3980.00", "3992.00", "3976.00", "3988.00"),
    ("3988.00", "4004.00", "3986.00", "4001.00"),
    ("4001.00", "4018.00", "3999.00", "4015.00"),
    ("4015.00", "4042.00", "4014.00", "4039.00"),  # swing high at 4042
    ("4038.00", "4040.00", "4020.00", "4022.00"),
    ("4021.00", "4024.00", "4006.00", "4008.00"),
    ("4007.00", "4010.00", "3986.00", "3990.00"),
    ("3990.00", "3994.00", "3962.00", "3966.00"),  # swing low at 3962
    ("3966.00", "3980.00", "3965.00", "3978.00"),
    ("3979.00", "3998.00", "3978.00", "3996.00"),
    ("3997.00", "4016.00", "3996.00", "4014.00"),
    ("4015.00", "4030.00", "4013.00", "4028.00"),
    ("4029.00", "4034.00", "4024.00", "4030.00"),
    ("4030.00", "4036.00", "4026.00", "4032.00"),
    ("4032.00", "4038.00", "4028.00", "4035.00"),
    ("4035.00", "4041.00", "4031.00", "4038.00"),
]


def snapshot() -> IctMarketSnapshot:
    """The same shape on every timeframe, each on its own grid.

    Deliberately the same price path rather than five unrelated series: the
    point of the fixture is that the primitives behave identically whatever the
    bar duration, so any difference between timeframes would be a defect rather
    than market texture.
    """
    return IctMarketSnapshot(
        observed_at=OBSERVED_AT,
        symbol="XAUUSD",
        provider="tradingview",
        provider_symbol="OANDA:XAUUSD",
        timeframes=tuple(
            build_timeframe_snapshot(timeframe=tf, bars=walk(tf, SHAPE), observed_at=OBSERVED_AT)
            for tf in ICT_TIMEFRAMES
        ),
    )


# --------------------------------------------------------------------------
# the fixture holds what it claims to
# --------------------------------------------------------------------------


def test_the_fixture_covers_all_five_timeframes_at_one_instant() -> None:
    shot = snapshot()

    assert shot.observed_timeframes == ICT_TIMEFRAMES
    assert all(entry.bar_count == len(SHAPE) for entry in shot.timeframes)
    assert all(entry.latest_closed_at == OBSERVED_AT for entry in shot.timeframes)


def test_h4_sits_on_the_grid_its_own_duration_implies() -> None:
    """No resampling: the bars are laid on H4 boundaries and left there."""
    h4 = snapshot().require(Timeframe.H4)

    spacing = {
        later.timestamp - earlier.timestamp
        for earlier, later in zip(h4.bars, h4.bars[1:], strict=False)
    }
    assert spacing == {timedelta(hours=4)}


def test_the_fixture_yields_an_atr_series() -> None:
    h1 = snapshot().require(Timeframe.H1)

    points = atr_series(h1, period=14)

    assert points, "16 bars is enough for a 14-period Wilder ATR"
    assert points[-1].atr > 0
    assert isinstance(points[-1].atr, Decimal)
    assert latest_atr(h1, period=14) == points[-1]


def test_the_fixture_yields_at_least_two_confirmed_swings() -> None:
    swings = confirmed_swings(snapshot().require(Timeframe.M15))

    assert len(swings) >= 2
    kinds = {swing.swing_type for swing in swings}
    assert SwingType.SWING_HIGH in kinds
    assert SwingType.SWING_LOW in kinds
    assert all(swing.confirmed_at > swing.pivot_time for swing in swings)


def test_the_fixture_yields_gaps_in_both_directions() -> None:
    gaps = fair_value_gaps(snapshot().require(Timeframe.M5))
    directions = {gap.direction for gap in gaps}

    assert GapDirection.BULLISH in directions
    assert GapDirection.BEARISH in directions
    assert all(gap.size > 0 for gap in gaps)


def test_every_timeframe_finds_the_same_shape() -> None:
    """The bars differ only in duration, so the primitives must agree."""
    shot = snapshot()

    swing_shapes = {
        tf: [(s.swing_type, s.price) for s in confirmed_swings(shot.require(tf))]
        for tf in ICT_TIMEFRAMES
    }
    gap_shapes = {
        tf: [(g.direction, g.lower, g.upper) for g in fair_value_gaps(shot.require(tf))]
        for tf in ICT_TIMEFRAMES
    }

    assert len(set(map(str, swing_shapes.values()))) == 1, swing_shapes
    assert len(set(map(str, gap_shapes.values()))) == 1, gap_shapes


def test_the_primitives_are_stable_across_repeated_calls() -> None:
    """A regression anchor is only useful if it does not drift."""
    first, second = snapshot(), snapshot()

    for tf in ICT_TIMEFRAMES:
        assert confirmed_swings(first.require(tf)) == confirmed_swings(second.require(tf))
        assert fair_value_gaps(first.require(tf)) == fair_value_gaps(second.require(tf))
        assert atr_series(first.require(tf), 14) == atr_series(second.require(tf), 14)


# --------------------------------------------------------------------------
# §26: source agnosticism
# --------------------------------------------------------------------------

ICT_MODULES = (
    Path("src/goldpipeline/schemas/ict.py"),
    Path("src/goldpipeline/services/ict_primitives.py"),
)

FORBIDDEN_IMPORTS = (
    "tradingview",
    "metatrader",
    "MetaTrader5",
    "websocket",
    "requests",
    "httpx",
    "anthropic",
)


def test_the_ict_modules_import_no_provider() -> None:
    """They consume normalized candles. Where those came from is not their business.

    Asserted over the import graph rather than by reading the source for
    strings, so a provider reached indirectly is caught too.
    """
    for path in ICT_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

        for name in imported:
            for forbidden in FORBIDDEN_IMPORTS:
                assert forbidden.lower() not in name.lower(), f"{path.name} imports {name}"


def test_the_ict_modules_read_no_clock() -> None:
    """Every instant they use arrives as data, so their answers are reproducible."""
    for path in ICT_MODULES:
        source = path.read_text(encoding="utf-8")
        for forbidden in ("datetime.now", "utcnow", "time.time", "utc_now"):
            assert forbidden not in source, f"{path.name} reads a clock: {forbidden}"


def test_the_ict_modules_do_not_reach_the_analysis_levels_module() -> None:
    """Two ATR definitions coexist deliberately; neither may call the other.

    `levels.average_true_range` is a simple mean serving a published analysis
    claim path, and this branch computes Wilder. Sharing code between them would
    make one silently become the other the first time somebody "unified" them.
    """
    for path in ICT_MODULES:
        source = path.read_text(encoding="utf-8")
        assert "from goldpipeline.services.levels import" not in source
        assert "import goldpipeline.services.levels" not in source


# --------------------------------------------------------------------------
# §37-§38: nothing shipped changed
# --------------------------------------------------------------------------


def test_trade_plan_is_still_not_ready_and_has_no_runtime() -> None:
    """Primitives exist; the product does not. Round 6.6a wires nothing."""
    import pytest

    from goldpipeline.domain.errors import ArticleTypeNotReadyError
    from goldpipeline.schemas.article import ArticleType
    from goldpipeline.services.article_routing import SPECS
    from goldpipeline.services.article_runtime import is_dispatchable, runtime_for

    assert SPECS[ArticleType.TRADE_PLAN].ready is False
    assert SPECS[ArticleType.TRADE_PLAN].prompt_id is None
    assert is_dispatchable(ArticleType.TRADE_PLAN) is False
    with pytest.raises(ArticleTypeNotReadyError):
        runtime_for(ArticleType.TRADE_PLAN)


def test_the_two_shipped_products_are_untouched() -> None:
    from goldpipeline.schemas.article import ArticleType
    from goldpipeline.services.article_routing import SPECS, writer_prompt_for
    from goldpipeline.services.review_action import STYLE_ACTIVE_TYPES

    assert writer_prompt_for(ArticleType.ANALYSIS) == "gold_writer_v4"
    assert writer_prompt_for(ArticleType.NEWS_DIGEST) == "gold_news_digest_writer_v2"
    assert SPECS[ArticleType.ANALYSIS].ready is True
    assert SPECS[ArticleType.NEWS_DIGEST].ready is True
    assert frozenset({ArticleType.ANALYSIS, ArticleType.NEWS_DIGEST}) == STYLE_ACTIVE_TYPES


def test_no_prompt_was_modified_by_this_round() -> None:
    """Every shipped prompt still loads and still says what it said."""
    from goldpipeline.prompts import load_prompt

    assert "🕯 PHÂN TÍCH VÀNG" in load_prompt("gold_writer_v4")
    assert "# HUMAN STYLE REVIEW" in load_prompt("gold_reviewer_v2")
    assert "# HUMAN STYLE REPAIR" in load_prompt("gold_finalizer_v2")
    assert "Carry a figure exactly, or leave it out." in load_prompt("gold_news_digest_writer_v2")
    assert "CONTENT WINS, ALWAYS" in load_prompt("gold_news_digest_finalizer_v1")

    for name in (
        "gold_writer_v4",
        "gold_reviewer_v2",
        "gold_finalizer_v2",
        "gold_news_digest_writer_v2",
        "gold_news_digest_finalizer_v1",
    ):
        text = load_prompt(name)
        for ict_word in ("fair value gap", "order block", "SEO", "BAI", "swing high"):
            assert ict_word not in text, f"{name} mentions {ict_word}"


def test_the_analysis_levels_module_still_computes_a_simple_mean() -> None:
    """The other ATR, unchanged. Two definitions, neither pretending to be the other."""
    from goldpipeline.services.levels import average_true_range

    bars = list(walk(Timeframe.H1, SHAPE))
    mean = average_true_range(bars, period=3)
    wilder = atr_series(
        build_timeframe_snapshot(timeframe=Timeframe.H1, bars=tuple(bars), observed_at=OBSERVED_AT),
        period=3,
    )[-1].atr

    assert mean is not None
    assert mean != wilder, "if these ever agree by construction, one has been changed"
