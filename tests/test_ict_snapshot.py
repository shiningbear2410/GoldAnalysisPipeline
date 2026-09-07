"""One instant, five timeframes, and only candles that had finished.

Round 6.6a §31. The snapshot is the authority every later ICT stage reads, so
the properties worth pinning are the ones a later stage would otherwise have to
re-establish for itself: that all five series describe the same moment, that
nothing still forming got in, and that a provider disagreeing with itself is
refused rather than tidied up.

Offline throughout - no provider, no clock.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.ict import (
    ICT_TIMEFRAMES,
    IctMarketSnapshot,
    IctTimeframeSnapshot,
    build_timeframe_snapshot,
)
from goldpipeline.schemas.market import OHLCBar

OBSERVED_AT = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


def bar(
    open_time: datetime,
    *,
    open_: str = "4000",
    high: str = "4010",
    low: str = "3990",
    close: str = "4005",
) -> OHLCBar:
    return OHLCBar(
        timestamp=open_time,
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
    )


def ladder(timeframe: Timeframe, count: int, *, end: datetime = OBSERVED_AT) -> list[OHLCBar]:
    """*count* consecutive bars whose last one closes exactly at *end*."""
    duration = timeframe.duration
    assert duration is not None
    return [bar(end - duration * (count - index)) for index in range(count)]


def series(timeframe: Timeframe, count: int = 6) -> IctTimeframeSnapshot:
    return build_timeframe_snapshot(
        timeframe=timeframe, bars=tuple(ladder(timeframe, count)), observed_at=OBSERVED_AT
    )


def snapshot(**overrides: object) -> IctMarketSnapshot:
    defaults: dict[str, object] = {
        "observed_at": OBSERVED_AT,
        "symbol": "XAUUSD",
        "provider": "tradingview",
        "provider_symbol": "OANDA:XAUUSD",
        "timeframes": tuple(series(tf) for tf in ICT_TIMEFRAMES),
    }
    return IctMarketSnapshot(**{**defaults, **overrides})  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# the shape of a valid observation
# --------------------------------------------------------------------------


def test_a_five_timeframe_snapshot_is_valid() -> None:
    shot = snapshot()

    assert shot.observed_timeframes == ICT_TIMEFRAMES
    assert set(ICT_TIMEFRAMES) == {
        Timeframe.H4,
        Timeframe.H1,
        Timeframe.M15,
        Timeframe.M5,
        Timeframe.M1,
    }


def test_every_series_is_current_to_the_same_instant() -> None:
    """The reason this type exists at all."""
    shot = snapshot()

    for entry in shot.timeframes:
        assert entry.latest_closed_at <= shot.observed_at


def test_a_snapshot_is_immutable() -> None:
    shot = snapshot()

    with pytest.raises(ValidationError):
        shot.observed_at = OBSERVED_AT + timedelta(hours=1)  # type: ignore[misc]


def test_a_timeframe_may_appear_only_once() -> None:
    with pytest.raises(ValidationError, match="may appear once"):
        snapshot(timeframes=(series(Timeframe.M5), series(Timeframe.M5)))


def test_require_names_the_missing_timeframe_rather_than_returning_nothing() -> None:
    """ "No bars" and "never looked" are different facts."""
    shot = snapshot(timeframes=(series(Timeframe.M5),))

    assert shot.series(Timeframe.M5) is not None
    assert shot.series(Timeframe.H4) is None
    with pytest.raises(KeyError):
        shot.require(Timeframe.H4)


# --------------------------------------------------------------------------
# closed bars only
# --------------------------------------------------------------------------


def test_a_forming_bar_is_excluded_not_truncated() -> None:
    """A candle cut short at the observation is a price that never traded."""
    bars = tuple(ladder(Timeframe.M5, 5)) + (bar(OBSERVED_AT),)  # opens at the instant

    built = build_timeframe_snapshot(timeframe=Timeframe.M5, bars=bars, observed_at=OBSERVED_AT)

    assert built.bar_count == 5
    assert built.latest_closed_at == OBSERVED_AT
    assert all(b.timestamp < OBSERVED_AT for b in built.bars)


def test_a_bar_closing_exactly_at_the_observation_is_included() -> None:
    built = series(Timeframe.M15, 4)

    assert built.latest_closed_at == OBSERVED_AT


def test_a_future_bar_is_excluded() -> None:
    bars = tuple(ladder(Timeframe.M5, 4)) + (bar(OBSERVED_AT + timedelta(hours=1)),)

    built = build_timeframe_snapshot(timeframe=Timeframe.M5, bars=bars, observed_at=OBSERVED_AT)

    assert built.bar_count == 4


def test_a_series_with_nothing_closed_is_refused_not_emptied() -> None:
    """An empty series would let a primitive report a confident nothing."""
    with pytest.raises(ValueError, match="no bar closed"):
        build_timeframe_snapshot(
            timeframe=Timeframe.M5,
            bars=(bar(OBSERVED_AT + timedelta(minutes=30)),),
            observed_at=OBSERVED_AT,
        )


def test_a_snapshot_refuses_a_series_that_outruns_its_observation() -> None:
    """Belt and braces: the model proves it, not only the builder."""
    late = series(Timeframe.M5)

    with pytest.raises(ValidationError, match="after the observation"):
        snapshot(observed_at=OBSERVED_AT - timedelta(minutes=10), timeframes=(late,))


# --------------------------------------------------------------------------
# series invariants: refused, not repaired
# --------------------------------------------------------------------------


def test_unsorted_bars_are_refused_rather_than_sorted() -> None:
    """Sorting would hide a provider disagreeing with itself."""
    bars = ladder(Timeframe.M5, 4)
    scrambled = tuple([bars[0], bars[2], bars[1], bars[3]])

    with pytest.raises(ValidationError, match="ascend strictly"):
        IctTimeframeSnapshot(
            timeframe=Timeframe.M5,
            bars=scrambled,
            first_closed_at=scrambled[0].timestamp + timedelta(minutes=5),
            latest_closed_at=scrambled[-1].timestamp + timedelta(minutes=5),
            bar_count=4,
        )


def test_a_duplicate_open_time_is_refused() -> None:
    bars = ladder(Timeframe.M5, 3)
    duplicated = tuple([bars[0], bars[1], bars[1]])

    with pytest.raises(ValidationError, match="ascend strictly"):
        IctTimeframeSnapshot(
            timeframe=Timeframe.M5,
            bars=duplicated,
            first_closed_at=duplicated[0].timestamp + timedelta(minutes=5),
            latest_closed_at=duplicated[-1].timestamp + timedelta(minutes=5),
            bar_count=3,
        )


def test_an_invalid_candle_is_refused_by_the_bar_model_itself() -> None:
    with pytest.raises(ValidationError):
        OHLCBar(
            timestamp=OBSERVED_AT - timedelta(minutes=5),
            open=Decimal("4000"),
            high=Decimal("3990"),  # below the low
            low=Decimal("3995"),
            close=Decimal("3992"),
        )


def test_a_naive_timestamp_is_refused_at_the_snapshot_layer() -> None:
    """The existing convention, followed rather than re-decided.

    `OHLCBar` tolerates a naive timestamp at input and `MarketDataSnapshot` is
    where awareness has always been required. The ICT snapshot does the same,
    so a naive bar is refused by name instead of raising a TypeError deep in
    the closed-bar arithmetic.
    """
    naive = OHLCBar(
        timestamp=datetime(2026, 9, 7, 11, 0),  # noqa: DTZ001 - the point of the test
        open=Decimal("4000"),
        high=Decimal("4010"),
        low=Decimal("3990"),
        close=Decimal("4005"),
    )
    assert not naive.is_utc, "the bar model itself still accepts it"

    with pytest.raises(ValidationError, match="not timezone-aware"):
        IctTimeframeSnapshot(
            timeframe=Timeframe.H1,
            bars=(naive,),
            first_closed_at=OBSERVED_AT,
            latest_closed_at=OBSERVED_AT,
            bar_count=1,
        )


def test_the_declared_counts_must_match_the_bars_held() -> None:
    bars = tuple(ladder(Timeframe.M5, 3))

    with pytest.raises(ValidationError, match="declares"):
        IctTimeframeSnapshot(
            timeframe=Timeframe.M5,
            bars=bars,
            first_closed_at=bars[0].timestamp + timedelta(minutes=5),
            latest_closed_at=bars[-1].timestamp + timedelta(minutes=5),
            bar_count=99,
        )


def test_the_declared_close_times_must_match_the_bars_held() -> None:
    bars = tuple(ladder(Timeframe.M5, 3))

    with pytest.raises(ValidationError, match="latest_closed_at"):
        IctTimeframeSnapshot(
            timeframe=Timeframe.M5,
            bars=bars,
            first_closed_at=bars[0].timestamp + timedelta(minutes=5),
            latest_closed_at=OBSERVED_AT + timedelta(days=1),
            bar_count=3,
        )


def test_a_calendar_timeframe_cannot_prove_its_bars_closed() -> None:
    """A month is not a fixed number of seconds, so the arithmetic has no answer."""
    assert Timeframe.MN1.duration is None

    with pytest.raises(ValueError, match="no fixed duration"):
        build_timeframe_snapshot(
            timeframe=Timeframe.MN1,
            bars=(bar(datetime(2026, 8, 1, tzinfo=UTC)),),
            observed_at=OBSERVED_AT,
        )


# --------------------------------------------------------------------------
# provenance, and H4's native grid
# --------------------------------------------------------------------------


def test_provenance_is_retained_and_is_not_meaning() -> None:
    shot = snapshot()

    assert shot.symbol == "XAUUSD"
    assert shot.provider == "tradingview"
    assert shot.provider_symbol == "OANDA:XAUUSD"


def test_a_provider_symbol_is_optional() -> None:
    assert snapshot(provider_symbol=None).provider_symbol is None


def test_provider_native_h4_timestamps_are_accepted_unchanged() -> None:
    """TradingView's XAUUSD H4 opens at 01/05/09/13/17/21 UTC.

    Shifting them onto a 00/04/08 grid would invent a bar boundary the venue
    never traded. The closed-bar arithmetic works on whatever grid the provider
    actually uses, so there is nothing to correct.
    """
    observed = datetime(2026, 9, 7, 13, 0, tzinfo=UTC)
    opens = [
        datetime(2026, 9, 6, 21, 0, tzinfo=UTC),
        datetime(2026, 9, 7, 1, 0, tzinfo=UTC),
        datetime(2026, 9, 7, 5, 0, tzinfo=UTC),
        datetime(2026, 9, 7, 9, 0, tzinfo=UTC),
    ]

    built = build_timeframe_snapshot(
        timeframe=Timeframe.H4,
        bars=tuple(bar(open_time) for open_time in opens),
        observed_at=observed,
    )

    assert [b.timestamp for b in built.bars] == opens
    assert built.latest_closed_at == observed
    assert all(b.timestamp.hour % 4 == 1 for b in built.bars)


def test_closes_at_is_arithmetic_over_open_times() -> None:
    built = series(Timeframe.H1, 3)

    assert built.closes_at == tuple(b.timestamp + timedelta(hours=1) for b in built.bars)
