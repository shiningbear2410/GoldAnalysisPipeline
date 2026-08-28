"""Run identifier behaviour."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from goldpipeline.domain.run_id import (
    generate_run_id,
    is_valid_run_id,
    run_id_created_at,
)


def test_generated_ids_are_unique_within_the_same_second() -> None:
    """Requirement 14.1: run ids must not collide."""
    fixed = datetime(2026, 8, 28, 2, 27, 1, tzinfo=UTC)
    ids = {generate_run_id(now=fixed) for _ in range(2000)}
    assert len(ids) > 1990, "random suffix is not providing enough entropy"


def test_id_encodes_the_utc_moment() -> None:
    fixed = datetime(2026, 8, 28, 2, 27, 1, tzinfo=UTC)
    run_id = generate_run_id(now=fixed)
    assert run_id.startswith("20260828_022701_")
    assert run_id_created_at(run_id) == fixed


def test_local_time_is_converted_to_utc() -> None:
    """A caller in UTC+7 must still produce a UTC-stamped id."""
    local = datetime(2026, 8, 28, 9, 27, 1, tzinfo=timezone(timedelta(hours=7)))
    assert generate_run_id(now=local).startswith("20260828_022701_")


def test_ids_sort_chronologically() -> None:
    earlier = generate_run_id(now=datetime(2026, 8, 28, 2, 27, 1, tzinfo=UTC))
    later = generate_run_id(now=datetime(2026, 8, 28, 2, 27, 2, tzinfo=UTC))
    next_day = generate_run_id(now=datetime(2026, 8, 29, 0, 0, 0, tzinfo=UTC))
    assert sorted([next_day, later, earlier]) == [earlier, later, next_day]


@pytest.mark.parametrize(
    "candidate",
    [
        "20260828_022701_a83f2c",
        "19700101_000000_000000",
    ],
)
def test_valid_ids_accepted(candidate: str) -> None:
    assert is_valid_run_id(candidate)


@pytest.mark.parametrize(
    "candidate",
    [
        "",
        "20260828_022701",
        "20260828_022701_A83F2C",  # uppercase hex is not canonical
        "20260828_022701_a83f2",  # suffix too short
        "../../etc/passwd",
        "20260828_022701_a83f2c/../..",
    ],
)
def test_invalid_ids_rejected(candidate: str) -> None:
    """Also guards path traversal: run ids index directories."""
    assert not is_valid_run_id(candidate)


def test_created_at_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="not a valid run id"):
        run_id_created_at("nonsense")
