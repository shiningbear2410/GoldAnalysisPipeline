"""Resolving ``source_claims`` against the context, safely.

Round 2 asks the writer to record where each number came from. Round 3 checks
those references actually hold - which means turning a string like
``context.ohlc.bars[-1].high`` into a value.

The obvious implementation is ``eval``. It is also the one that turns a field
written by a language model into arbitrary code execution. So this module walks
the object graph by hand instead:

* the path must start at ``context``;
* each step is either a field name declared on a Pydantic model, or an integer
  index into a list;
* names beginning with an underscore are refused, as are callables;
* depth and length are capped.

Anything outside that grammar is rejected before a single attribute is touched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from pydantic import BaseModel

from goldpipeline.schemas.context import AnalysisContext
from goldpipeline.schemas.writer import SourceClaim
from goldpipeline.services.market_facts import format_price

ROOT = "context"
MAX_PATH_DEPTH = 8
MAX_PATH_CHARS = 200

_SEGMENT_RE = re.compile(r"^(?P<name>[a-z][a-z0-9_]*)(?P<indices>(?:\[-?\d+\])*)$")
_INDEX_RE = re.compile(r"\[(-?\d+)\]")


class ClaimPathError(ValueError):
    """The path is malformed, or does not point at anything in the context."""


@dataclass(frozen=True)
class ResolvedClaim:
    """The outcome of checking one claim against the context."""

    claim: SourceClaim
    resolved: str | None
    matches: bool
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the claim resolved and agreed with the context."""
        return self.error is None and self.matches


def resolve_path(context: AnalysisContext, path: str) -> Any:
    """Resolve a dotted path into *context*.

    Args:
        context: The Run's source of truth.
        path: e.g. ``context.price.latest_close`` or ``context.ohlc.bars[-1].high``.

    Raises:
        ClaimPathError: If the path is malformed or leads nowhere.
    """
    cleaned = path.strip()
    if not cleaned or len(cleaned) > MAX_PATH_CHARS:
        raise ClaimPathError(f"path is empty or too long: {path!r}")

    segments = cleaned.split(".")
    if segments[0] != ROOT:
        raise ClaimPathError(f"path must start with {ROOT!r}: {path!r}")
    if len(segments) > MAX_PATH_DEPTH:
        raise ClaimPathError(f"path is nested too deeply: {path!r}")

    current: Any = context
    for segment in segments[1:]:
        match = _SEGMENT_RE.match(segment)
        if match is None:
            raise ClaimPathError(f"invalid path segment {segment!r} in {path!r}")

        current = _read_field(current, match["name"], path)
        for raw_index in _INDEX_RE.findall(match["indices"] or ""):
            current = _read_index(current, int(raw_index), path)

    return current


def _read_field(target: Any, name: str, path: str) -> Any:
    """Read one declared field. Not a generic ``getattr``."""
    if name.startswith("_"):
        raise ClaimPathError(f"private names are not addressable: {path!r}")
    if not isinstance(target, BaseModel):
        raise ClaimPathError(f"{name!r} is not reachable in {path!r}")
    if name not in type(target).model_fields:
        raise ClaimPathError(f"no field {name!r} in {type(target).__name__} ({path!r})")

    value = getattr(target, name)
    if callable(value):
        raise ClaimPathError(f"{name!r} is not a value: {path!r}")
    return value


def _read_index(target: Any, index: int, path: str) -> Any:
    if not isinstance(target, list):
        raise ClaimPathError(f"cannot index a non-list in {path!r}")
    try:
        return target[index]
    except IndexError:
        raise ClaimPathError(f"index {index} is out of range in {path!r}") from None


def render_value(value: Any) -> str:
    """Render a resolved value the way a claim would state it."""
    if isinstance(value, Decimal):
        return format_price(value)
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def values_match(claimed: str, resolved: Any) -> bool:
    """Compare a claimed string against a resolved value.

    Numbers compare numerically, so ``3314.2`` and ``3314.20`` agree - the
    writer's display convention should not be able to manufacture a mismatch.
    Text compares case-insensitively on trimmed content, because a claim quoting
    a symbol as ``xauusd`` is not a factual error.

    Long text fields (the raw analysis) compare by containment: a claim citing
    ``context.raw_analysis.text`` is attributing a view, and quoting the whole
    message back would be absurd.
    """
    claimed_text = claimed.strip()

    if isinstance(resolved, Decimal):
        try:
            return Decimal(claimed_text.replace(",", "")) == resolved
        except InvalidOperation:
            return False

    rendered = render_value(resolved)

    if isinstance(resolved, str) and len(resolved) > 200:
        return bool(claimed_text) and claimed_text.casefold() in resolved.casefold()

    if claimed_text.casefold() == rendered.casefold():
        return True

    # A datetime claim may legitimately drop the trailing Z or use a space.
    if isinstance(resolved, datetime):
        normalized = claimed_text.replace(" ", "T").rstrip("Z")
        return normalized == rendered.rstrip("Z")

    try:
        return Decimal(claimed_text.replace(",", "")) == Decimal(rendered)
    except InvalidOperation:
        return False


def verify_claim(context: AnalysisContext, claim: SourceClaim) -> ResolvedClaim:
    """Check one claim against the context."""
    try:
        value = resolve_path(context, claim.source)
    except ClaimPathError as exc:
        return ResolvedClaim(claim=claim, resolved=None, matches=False, error=str(exc))

    return ResolvedClaim(
        claim=claim,
        resolved=render_value(value),
        matches=values_match(claim.value, value),
    )


def verify_claims(context: AnalysisContext, claims: list[SourceClaim]) -> list[ResolvedClaim]:
    """Check every claim a draft recorded."""
    return [verify_claim(context, claim) for claim in claims]


__all__ = [
    "MAX_PATH_DEPTH",
    "ROOT",
    "ClaimPathError",
    "ResolvedClaim",
    "render_value",
    "resolve_path",
    "values_match",
    "verify_claim",
    "verify_claims",
]
