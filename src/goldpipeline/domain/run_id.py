"""Run identifier generation.

Format::

    20260828_022701_a83f2c
    |------| |----| |----|
    UTC date  time  random suffix (24 bits, lowercase hex)

Properties:

* **Sortable** - the lexicographic order of two ids matches the chronological
  order of their creation second.
* **Collision resistant** - within the same second, 16.7M suffixes are possible;
  the storage layer additionally refuses to reuse an existing directory.
* **Readable** - a human can tell when a Run happened without decoding anything.
"""

from __future__ import annotations

import re
import secrets
from datetime import UTC, datetime

RUN_ID_PATTERN = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})_(?P<suffix>[0-9a-f]{6})$")
"""Canonical shape of a run id; also used to reject path traversal."""

_SUFFIX_BYTES = 3


def generate_run_id(*, now: datetime | None = None) -> str:
    """Return a fresh run id.

    Args:
        now: Injection point for tests. Defaults to the current UTC time.
            Naive datetimes are assumed to already be UTC.
    """
    moment = now or datetime.now(UTC)
    moment = moment.astimezone(UTC) if moment.tzinfo else moment.replace(tzinfo=UTC)
    suffix = secrets.token_hex(_SUFFIX_BYTES)
    return f"{moment:%Y%m%d_%H%M%S}_{suffix}"


def is_valid_run_id(value: str) -> bool:
    """Return ``True`` when *value* is a syntactically valid run id."""
    return RUN_ID_PATTERN.fullmatch(value) is not None


def run_id_created_at(run_id: str) -> datetime:
    """Recover the UTC second encoded in *run_id*.

    Raises:
        ValueError: If *run_id* is not a valid run id.
    """
    match = RUN_ID_PATTERN.fullmatch(run_id)
    if match is None:
        raise ValueError(f"not a valid run id: {run_id!r}")
    return datetime.strptime(f"{match['date']}{match['time']}", "%Y%m%d%H%M%S").replace(tzinfo=UTC)


__all__ = ["RUN_ID_PATTERN", "generate_run_id", "is_valid_run_id", "run_id_created_at"]
