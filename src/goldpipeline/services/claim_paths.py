"""The catalog of context paths a writer may cite in ``source_claims``.

**The defect this module exists to prevent.** A production Run recorded
seventeen source claims and sixteen of them cited paths that do not exist:
``context.instrument``, ``context.window.bar_count``,
``context.latest_candle.close``, ``context.window_summary.highest_high``,
``context.recent_candles[7].c``. Every deterministic claim check failed, the
reviewer had to re-verify each number by hand, and fourteen artificial HIGH
findings went to a finalizer that then repaired an article which was never
wrong.

The model was not hallucinating. The prompt hands it a **MARKET FACTS** JSON
document keyed ``instrument`` / ``window`` / ``latest_candle`` /
``window_summary`` / ``recent_candles``, and then asked for dotted paths "such
as ``context.price.latest_close``" - an open-ended example list. Shown one
concrete document and asked for paths, it cited that document. Those keys are a
display convenience computed by :mod:`goldpipeline.services.market_facts`; the
resolver reads ``context.json``, whose shape is entirely different.

So the fix is not a sterner instruction. It is to stop asking the model to guess
a vocabulary and hand it the real one:

* the catalog is **derived from the context object itself**, walking declared
  Pydantic fields, so it cannot drift from the schema the resolver reads;
* the same paths that appear in the prompt are the paths
  :func:`goldpipeline.services.claim_resolver.resolve_path` accepts, and a test
  asserts every advertised path actually resolves;
* nothing here is hand-maintained, so there is no second list to update.

**What is deliberately not claimable.** Two kinds of thing:

* *Paraphrase.* ``context.raw_analysis.text`` resolves, but an article
  attributing the analyst's view restates it in the writer's own words, and a
  ``SourceClaim`` is defined as the value *as used in the article*. Advertising
  it guarantees a recurring mismatch, which is the same disease in a different
  place. The resolver still accepts it - a verbatim quotation remains
  verifiable - it is simply not offered.
* *Arithmetic.* Net change, net change percent and closing-run length are
  computed for the prompt and exist nowhere in the context. They have no path,
  and inventing one for them would be the original defect wearing a hat. The
  writer may still state them in prose; it may not cite a source for them.

Window high and low need no new field: they are a specific bar's ``high`` and
``low``, and that bar has a real address.

**Security.** The catalog is generated from application code, never from
context content. Untrusted source text cannot introduce a path, and the walk
only descends declared model fields - no credential, configuration, environment
or filesystem value is reachable, because none of them is in
:class:`~goldpipeline.schemas.context.AnalysisContext` to begin with.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel

from goldpipeline.schemas.context import AnalysisContext
from goldpipeline.services.claim_resolver import ROOT

EXCLUDED_PATHS = frozenset(
    {
        f"{ROOT}.raw_analysis.text",
        f"{ROOT}.raw_analysis.handling",
        f"{ROOT}.raw_analysis.trust_level",
        f"{ROOT}.schema_version",
    }
)
"""Resolvable paths that are nonetheless not offered as claim sources.

``text`` is excluded because articles paraphrase it; the rest are constants and
plumbing that no factual claim should ever rest on. Excluding a path narrows
what the writer is *offered*; it does not change what the resolver accepts, so
no existing artifact or test is invalidated.
"""

EXCLUDED_PREFIXES = (f"{ROOT}.data_quality.warnings",)
"""Sub-trees omitted wholesale: structured diagnostics, not article facts."""

_SCALAR_TYPES = (str, int, float, bool, Decimal, datetime, Enum)


@dataclass(frozen=True)
class IndexedFamily:
    """A list of models, and the leaf fields addressable on each element."""

    prefix: str
    """e.g. ``context.ohlc.bars``."""

    length: int
    leaves: tuple[str, ...]

    def paths(self) -> list[str]:
        """Every concrete path in this family, positive indices only."""
        return [
            f"{self.prefix}[{index}].{leaf}" for index in range(self.length) for leaf in self.leaves
        ]

    def describe(self) -> list[str]:
        """Prompt lines: the exact index range, then the leaf fields."""
        last = self.length - 1
        return [
            f"- {self.prefix}[i].<field>   i = 0..{last}   "
            f"(-1 also means the latest, i.e. index {last})",
            f"    <field> is one of: {', '.join(self.leaves)}",
        ]


@dataclass(frozen=True)
class ClaimPathCatalog:
    """Every path a writer may cite for this particular context."""

    scalars: tuple[str, ...]
    families: tuple[IndexedFamily, ...]

    def all_paths(self) -> set[str]:
        """Concrete expansion, for validation and tests."""
        paths = set(self.scalars)
        for family in self.families:
            paths.update(family.paths())
        return paths

    def describe(self) -> list[str]:
        """The catalog as prompt lines, in a stable order."""
        lines = [f"- {path}" for path in self.scalars]
        for family in self.families:
            lines.extend(family.describe())
        return lines


def build_catalog(context: AnalysisContext) -> ClaimPathCatalog:
    """Derive the claimable paths for *context* from its own declared fields.

    Walks the object rather than a hand-written list, so a field added to the
    schema becomes claimable without anyone remembering to update this module,
    and a field removed stops being advertised on the same commit.
    """
    scalars: list[str] = []
    families: list[IndexedFamily] = []
    _walk(context, ROOT, scalars, families)
    return ClaimPathCatalog(scalars=tuple(scalars), families=tuple(families))


def _walk(
    model: BaseModel,
    prefix: str,
    scalars: list[str],
    families: list[IndexedFamily],
) -> None:
    """Collect claimable paths beneath *model*.

    Only declared fields are followed - the same rule
    :func:`goldpipeline.services.claim_resolver.resolve_path` enforces - so the
    catalog can never advertise something the resolver would refuse to read.
    """
    for name in type(model).model_fields:
        path = f"{prefix}.{name}"
        if path in EXCLUDED_PATHS or path.startswith(EXCLUDED_PREFIXES):
            continue

        value = getattr(model, name)
        if isinstance(value, BaseModel):
            _walk(value, path, scalars, families)
            continue

        if isinstance(value, list) and value and all(isinstance(x, BaseModel) for x in value):
            element = value[0]
            leaves = tuple(
                leaf for leaf in type(element).model_fields if _is_scalar(getattr(element, leaf))
            )
            if leaves:
                families.append(IndexedFamily(prefix=path, length=len(value), leaves=leaves))
            continue

        if _is_scalar(value):
            scalars.append(path)


def _is_scalar(value: object) -> bool:
    """Whether a resolved value is something a claim can state exactly.

    ``None`` is excluded deliberately: a path that resolves to nothing today
    would invite a claim whose value can only be the string "None".
    """
    return value is not None and isinstance(value, _SCALAR_TYPES)


__all__ = [
    "EXCLUDED_PATHS",
    "EXCLUDED_PREFIXES",
    "ClaimPathCatalog",
    "IndexedFamily",
    "build_catalog",
]
