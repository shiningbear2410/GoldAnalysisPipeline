"""Versioned prompt templates.

Prompts are part of the system, not incidental strings: a change to one changes
what gets published. They therefore live in files with a version in the name,
are loaded by id, and are covered by a structural test - so an edit that drops a
required section fails the suite rather than reaching production quietly.

Adding a revision means adding ``gold_writer_v2.md`` and a new id. Existing Runs
keep recording the version they were written with.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent

GOLD_WRITER_V1 = "gold_writer_v1"
"""The original writer prompt. Retained so historical Runs keep their meaning.

Its `source_claims` rule offered dotted paths "such as" three examples - an open
vocabulary - while the user turn showed a differently-shaped MARKET FACTS
document. A production Run cited that document instead, and sixteen of its
seventeen claims addressed paths that do not exist.
"""

GOLD_WRITER_V2 = "gold_writer_v2"
"""Current writer prompt id. Closes the `source` vocabulary.

Identical to v1 except that `source` must be copied from the `VALID SOURCE
PATHS` catalog the user turn now carries, and derived figures and the analyst's
note are explicitly not claimable.
"""

GOLD_WRITER_V3 = "gold_writer_v3"
"""Current writer prompt id. Adds explicit news provenance.

Identical to v2 in voice, structure and every style rule - the article this
writes is the article v2 wrote. What it adds is a `news_claims` contract: an
assertion that a named economic actor *did* something must cite the collected
news item it came from, by an id copied from a closed list in the user turn.

A new file rather than an edit to v2, because seven Runs record having been
written by v2 and quietly changing what that name means would detach them from
the rules they were actually written under.
"""

GOLD_REVIEWER_V1 = "gold_reviewer_v1"
"""Current reviewer prompt id."""

GOLD_FINALIZER_V1 = "gold_finalizer_v1"
"""Current finalizer prompt id."""

DEFAULT_WRITER_PROMPT = GOLD_WRITER_V3
DEFAULT_REVIEWER_PROMPT = GOLD_REVIEWER_V1
DEFAULT_FINALIZER_PROMPT = GOLD_FINALIZER_V1

REQUIRED_SECTIONS = ("# SYSTEM RULES", "# OUTPUT CONTRACT")
"""Headings the system prompt must contain.

Asserted at load time and in tests. A prompt missing its output contract would
still call the API and still cost money - it would just produce something the
response schema rejects, much later and much less legibly.
"""


@lru_cache(maxsize=8)
def load_prompt(prompt_id: str = DEFAULT_WRITER_PROMPT) -> str:
    """Load a versioned prompt template by id.

    Raises:
        FileNotFoundError: If no template with that id exists.
        ValueError: If the template is missing a required section.
    """
    if "/" in prompt_id or "\\" in prompt_id or ".." in prompt_id:
        raise ValueError(f"invalid prompt id: {prompt_id!r}")

    path = PROMPTS_DIR / f"{prompt_id}.md"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt template named {prompt_id!r} in {PROMPTS_DIR}")

    text = path.read_text(encoding="utf-8")
    missing = [section for section in REQUIRED_SECTIONS if section not in text]
    if missing:
        raise ValueError(f"prompt {prompt_id!r} is missing required sections: {missing}")
    return text


__all__ = [
    "DEFAULT_FINALIZER_PROMPT",
    "DEFAULT_REVIEWER_PROMPT",
    "DEFAULT_WRITER_PROMPT",
    "GOLD_FINALIZER_V1",
    "GOLD_REVIEWER_V1",
    "GOLD_WRITER_V1",
    "GOLD_WRITER_V2",
    "GOLD_WRITER_V3",
    "PROMPTS_DIR",
    "REQUIRED_SECTIONS",
    "load_prompt",
]
