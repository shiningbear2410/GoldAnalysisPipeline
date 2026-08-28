"""Deterministic price formatting and window arithmetic."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from conftest import make_bar, make_market_payload, make_normalized_run

from goldpipeline.services.market_facts import (
    build_market_facts,
    format_price,
    format_recent_bars,
    format_signed_price,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3315.2", "3315.20"),
        ("3315.20", "3315.20"),
        ("3315", "3315.00"),
        ("3315.00", "3315.00"),
        ("3315.456", "3315.46"),
        ("3315.455", "3315.46"),
        ("3315.454", "3315.45"),
    ],
)
def test_prices_render_with_a_single_convention(raw: str, expected: str) -> None:
    """Two decimals always. Padding a zero does not change the value."""
    assert format_price(Decimal(raw)) == expected


def test_formatting_never_alters_the_value() -> None:
    for raw in ("3315.2", "3315", "3304.80", "1.5"):
        assert Decimal(format_price(Decimal(raw))) == Decimal(raw)


def test_half_up_rounding_matches_desk_convention() -> None:
    """Python rounds 0.5 to even by default; a trading desk rounds it up."""
    assert format_price(Decimal("3315.125")) == "3315.13"
    assert format_price(Decimal("3315.135")) == "3315.14"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("12.5", "+12.50"), ("-12.5", "-12.50"), ("0", "+0.00")],
)
def test_deltas_always_carry_a_sign(raw: str, expected: str) -> None:
    assert format_signed_price(Decimal(raw)) == expected


# --- window arithmetic ----------------------------------------------------


def _context(runs_dir: Any, tmp_path: Any, bars: list[dict[str, Any]]) -> Any:
    result = make_normalized_run(runs_dir, tmp_path, market=make_market_payload(bars=bars))
    assert result.context is not None
    return result.context


def test_facts_are_derived_from_the_series(runs_dir: Any, tmp_path: Any) -> None:
    bars = [
        make_bar(minute_offset=0, open_="3300.00", high="3310.00", low="3299.00", close="3305.00"),
        make_bar(minute_offset=15, open_="3305.00", high="3325.00", low="3304.00", close="3320.00"),
        make_bar(minute_offset=30, open_="3320.00", high="3322.00", low="3312.00", close="3315.00"),
    ]
    facts = build_market_facts(_context(runs_dir, tmp_path, bars))

    assert facts.window_open == "3300.00"
    assert facts.window_high == "3325.00"
    assert facts.window_low == "3299.00"
    assert facts.latest_close == "3315.00"
    assert facts.net_change == "+15.00"
    assert facts.bar_count == 3


def test_negative_net_change_is_signed(runs_dir: Any, tmp_path: Any) -> None:
    bars = [
        make_bar(minute_offset=0, open_="3320.00", high="3321.00", low="3319.00", close="3320.00"),
        make_bar(minute_offset=15, open_="3320.00", high="3321.00", low="3309.00", close="3310.00"),
    ]
    facts = build_market_facts(_context(runs_dir, tmp_path, bars))
    assert facts.net_change == "-10.00"
    assert facts.net_change_percent.startswith("-0.30")


def test_closing_run_counts_consecutive_falling_closes(runs_dir: Any, tmp_path: Any) -> None:
    closes = ["3320.00", "3318.00", "3316.00", "3314.00"]
    bars = [
        make_bar(
            minute_offset=index * 15,
            open_="3320.00",
            high="3325.00",
            low="3305.00",
            close=close,
        )
        for index, close in enumerate(closes)
    ]
    facts = build_market_facts(_context(runs_dir, tmp_path, bars))

    assert facts.closing_run_direction == "down"
    assert facts.closing_run_length == 3


def test_closing_run_stops_at_a_reversal(runs_dir: Any, tmp_path: Any) -> None:
    closes = ["3310.00", "3320.00", "3318.00", "3316.00"]
    bars = [
        make_bar(
            minute_offset=index * 15,
            open_="3315.00",
            high="3325.00",
            low="3305.00",
            close=close,
        )
        for index, close in enumerate(closes)
    ]
    facts = build_market_facts(_context(runs_dir, tmp_path, bars))
    assert facts.closing_run_direction == "down"
    assert facts.closing_run_length == 2


def test_flat_closes_end_a_run(runs_dir: Any, tmp_path: Any) -> None:
    """A streak should not be stretched across an unchanged close."""
    closes = ["3310.00", "3315.00", "3315.00", "3316.00"]
    bars = [
        make_bar(
            minute_offset=index * 15,
            open_="3315.00",
            high="3325.00",
            low="3305.00",
            close=close,
        )
        for index, close in enumerate(closes)
    ]
    facts = build_market_facts(_context(runs_dir, tmp_path, bars))
    assert facts.closing_run_length == 1


def test_recent_bars_are_the_tail_and_are_formatted(runs_dir: Any, tmp_path: Any) -> None:
    context = make_normalized_run(runs_dir, tmp_path).context
    recent = format_recent_bars(context, limit=4)

    assert len(recent) == 4
    assert recent[-1]["c"] == format_price(context.ohlc.bars[-1].close)
    assert all(len(bar["c"].split(".")[1]) == 2 for bar in recent)


def test_fact_sheet_carries_quality_state(runs_dir: Any, tmp_path: Any) -> None:
    """The writer must be able to see that its inputs were degraded."""
    bars = [make_bar(minute_offset=index * 15, volume=None) for index in range(3)]
    for index, bar in enumerate(bars):
        bar["close"] = f"{3310 + index}.00"
        bar["open"] = "3312.45"
        bar["high"] = "3320.00"
        bar["low"] = "3305.00"

    facts = build_market_facts(_context(runs_dir, tmp_path, bars))
    assert facts.data_quality_status == "WARNING"
    assert "ohlc.volume" in facts.missing_fields
    assert "MISSING_VOLUME" in facts.quality_warnings
