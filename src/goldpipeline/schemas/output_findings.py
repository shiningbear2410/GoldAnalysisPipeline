"""Typed findings from the deterministic output checks.

Three checkers - contract shape, style symptoms, causality language - produce
one finding type, so the round that wires them into a verdict has a single
thing to consume and a single place to decide what blocks.

**Severity is data, not policy.** Every finding carries one, and nothing reads
it yet. The reviewer's verdict, the gate's decision and the finalizer's edit
scope are all unchanged by anything in this module; a later round decides
which of these codes may block, and until then they are observations.

**Symptoms, not judgments.** A style code here names something countable - a
phrase, a repetition, a uniform rhythm. None of them says an article sounds
like a machine, because no count can. That question stays with the reviewer.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from goldpipeline.schemas.common import StrictModel
from goldpipeline.schemas.review import Severity

MAX_FINDING_EXCERPT_CHARS = 240


class OutputFindingCode(StrEnum):
    """Every deterministic output finding, by the checker that produces it."""

    # --- contract shape -------------------------------------------------
    HARD_CAP_EXCEEDED = "HARD_CAP_EXCEEDED"
    MISSING_REQUIRED_SECTION = "MISSING_REQUIRED_SECTION"
    FORBIDDEN_SECTION_PRESENT = "FORBIDDEN_SECTION_PRESENT"
    FORBIDDEN_STRUCTURE_PRESENT = "FORBIDDEN_STRUCTURE_PRESENT"
    DISCLAIMER_COUNT_MISMATCH = "DISCLAIMER_COUNT_MISMATCH"
    DISCLAIMER_TEXT_MISMATCH = "DISCLAIMER_TEXT_MISMATCH"
    FORBIDDEN_TERMINOLOGY = "FORBIDDEN_TERMINOLOGY"
    """``SELL``/``BUY`` standing where ``SEO``/``BAI`` belong."""

    # --- style symptoms -------------------------------------------------
    THROAT_CLEARING = "THROAT_CLEARING"
    CONNECTIVE_DENSITY = "CONNECTIVE_DENSITY"
    REPEATED_CONSTRUCTION = "REPEATED_CONSTRUCTION"
    SENTENCE_LENGTH_UNIFORM = "SENTENCE_LENGTH_UNIFORM"
    BULLET_SHAPE_UNIFORM = "BULLET_SHAPE_UNIFORM"
    PARAGRAPH_LENGTH_UNIFORM = "PARAGRAPH_LENGTH_UNIFORM"
    DUPLICATE_STATEMENT = "DUPLICATE_STATEMENT"
    NUMBER_RESTATED = "NUMBER_RESTATED"
    REPEATED_OPENER = "REPEATED_OPENER"

    # --- causality language ---------------------------------------------
    UNATTRIBUTED_CAUSAL_CLAIM = "UNATTRIBUTED_CAUSAL_CLAIM"
    """A news reference, price movement and causal connector in one sentence,
    with no source attributed for the link."""


class OutputFinding(StrictModel):
    """One observation about an article, made in Python."""

    code: OutputFindingCode
    severity: Severity
    message: str = Field(min_length=1, max_length=600)
    count: int | None = Field(default=None, ge=0, description="What was counted, if anything.")
    threshold: int | None = Field(
        default=None, ge=0, description="The count at which this becomes a finding."
    )
    position: int | None = Field(default=None, ge=0, description="Character offset, if one.")
    excerpt: str | None = Field(default=None, max_length=MAX_FINDING_EXCERPT_CHARS)


__all__ = ["MAX_FINDING_EXCERPT_CHARS", "OutputFinding", "OutputFindingCode"]
