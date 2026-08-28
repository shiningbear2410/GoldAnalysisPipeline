"""Resolving source_claims safely, and comparing them honestly."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from conftest import LATEST_CLOSE, make_normalized_run

from goldpipeline.schemas.writer import ClaimType, SourceClaim
from goldpipeline.services.claim_resolver import (
    ClaimPathError,
    render_value,
    resolve_path,
    values_match,
    verify_claim,
    verify_claims,
)


@pytest.fixture
def context(runs_dir: Any, tmp_path: Any) -> Any:
    result = make_normalized_run(runs_dir, tmp_path)
    assert result.context is not None
    return result.context


def claim(value: str, source: str) -> SourceClaim:
    return SourceClaim(type=ClaimType.PRICE, value=value, source=source)


# --- resolution -----------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "context.price.latest_close",
        "context.price.latest_high",
        "context.market.symbol",
        "context.market.timeframe",
        "context.market.provider",
        "context.timing.latest_candle_at",
        "context.ohlc.bar_count",
        "context.ohlc.bars[-1].high",
        "context.ohlc.bars[0].open",
        "context.raw_analysis.text",
        "context.data_quality.status",
    ],
)
def test_documented_paths_resolve(context: Any, path: str) -> None:
    """Requirement 27.9: every path the prompt tells the writer to use works."""
    assert resolve_path(context, path) is not None


def test_resolved_values_match_the_context(context: Any) -> None:
    assert resolve_path(context, "context.market.symbol") == "XAUUSD"
    assert resolve_path(context, "context.price.latest_close") == context.price.latest_close
    assert resolve_path(context, "context.ohlc.bars[-1].high") == context.ohlc.bars[-1].high
    assert resolve_path(context, "context.ohlc.bars[0].open") == context.ohlc.bars[0].open


def test_negative_and_positive_indices_agree(context: Any) -> None:
    last = context.ohlc.bar_count - 1
    assert resolve_path(context, "context.ohlc.bars[-1].close") == resolve_path(
        context, f"context.ohlc.bars[{last}].close"
    )


# --- the path grammar is a whitelist, not eval ----------------------------


@pytest.mark.parametrize(
    "path",
    [
        "price.latest_close",  # no root
        "ctx.price.latest_close",  # wrong root
        "context.__class__",  # dunder
        "context._abc",  # private
        "context.market.model_dump",  # a method, not a value
        "context.market.__init__",
        "context.nope",  # unknown field
        "context.market.symbol.upper",  # attribute on a plain value
        "context.ohlc.bars[999].high",  # out of range
        "context.price[0]",  # indexing a model
        "context.ohlc.bars[abc].high",  # non-integer index
        "context",  # nothing selected beyond the root
        "",
        "   ",
        "__import__('os').system('echo pwned')",
        "context.a.b.c.d.e.f.g.h.i.j",  # too deep
    ],
)
def test_dangerous_or_malformed_paths_are_refused(context: Any, path: str) -> None:
    """Requirement 12: a model-authored string must not become code."""
    if path == "context":
        # A bare root resolves to the context itself, which is not a claim value
        # but is not dangerous either.
        assert resolve_path(context, path) is context
        return
    with pytest.raises(ClaimPathError):
        resolve_path(context, path)


def test_overlong_path_is_refused(context: Any) -> None:
    with pytest.raises(ClaimPathError):
        resolve_path(context, "context." + "a" * 300)


# --- value comparison -----------------------------------------------------


def test_display_precision_does_not_create_a_mismatch(context: Any) -> None:
    """3314.2 and 3314.20 are the same price; only the rendering differs."""
    close = context.price.latest_close
    assert values_match(str(close), close)
    assert values_match(render_value(close), close)
    assert values_match(f"{close:.1f}", close)


def test_thousands_separators_are_tolerated(context: Any) -> None:
    close = context.price.latest_close
    assert values_match(f"{close:,}", close)


def test_symbols_compare_case_insensitively(context: Any) -> None:
    assert values_match("xauusd", "XAUUSD")
    assert values_match("XAUUSD", "XAUUSD")
    assert not values_match("BTCUSD", "XAUUSD")


def test_a_genuinely_different_number_does_not_match(context: Any) -> None:
    assert not values_match("9999.00", context.price.latest_close)


def test_long_text_claims_compare_by_containment(context: Any) -> None:
    """A claim citing the analyst's note quotes a phrase, not the whole message."""
    text = context.raw_analysis.text
    assert len(text) > 200
    assert values_match(text[10:40], text)
    assert not values_match("điều này không có trong tin nhắn", text)


def test_timestamps_compare_across_formats(context: Any) -> None:
    when = context.timing.latest_candle_at
    rendered = render_value(when)
    assert values_match(rendered, when)
    assert values_match(rendered.rstrip("Z"), when)


# --- verification ---------------------------------------------------------


def test_a_correct_claim_verifies(context: Any) -> None:
    resolved = verify_claim(context, claim(LATEST_CLOSE, "context.price.latest_close"))
    assert resolved.ok
    assert resolved.error is None
    assert resolved.resolved == LATEST_CLOSE


def test_a_wrong_value_is_reported(context: Any) -> None:
    """Requirement 27.11."""
    resolved = verify_claim(context, claim("9999.00", "context.price.latest_close"))
    assert not resolved.ok
    assert resolved.error is None
    assert resolved.matches is False
    assert resolved.resolved == LATEST_CLOSE


def test_a_missing_path_is_reported_not_raised(context: Any) -> None:
    """Requirement 27.10: a bad path is a finding, not a crash."""
    resolved = verify_claim(context, claim(LATEST_CLOSE, "context.price.does_not_exist"))
    assert not resolved.ok
    assert resolved.error is not None
    assert resolved.resolved is None


def test_a_dangerous_path_is_reported_not_executed(context: Any) -> None:
    resolved = verify_claim(context, claim("x", "__import__('os').system('echo pwned')"))
    assert not resolved.ok
    assert resolved.error is not None


def test_verify_claims_preserves_order(context: Any) -> None:
    claims = [
        claim(LATEST_CLOSE, "context.price.latest_close"),
        claim("9999", "context.price.latest_close"),
        claim("x", "context.nope"),
    ]
    results = verify_claims(context, claims)
    assert [item.ok for item in results] == [True, False, False]
    assert [item.claim.value for item in results] == [LATEST_CLOSE, "9999", "x"]


def test_no_claims_yields_no_results(context: Any) -> None:
    assert verify_claims(context, []) == []


def test_render_value_formats_prices_consistently(context: Any) -> None:
    assert render_value(Decimal("3305.4")) == "3305.40"
    assert render_value(Decimal("3305")) == "3305.00"
