"""Versioned prompt templates.

Prompts are part of the system, not incidental strings: a change to one changes
what gets published. They therefore live in files with a version in the name,
are loaded by id, and are covered by a structural test - so an edit that drops a
required section fails the suite rather than reaching production quietly.

Adding a revision means adding ``gold_writer_v2.md`` and a new id. Existing Runs
keep recording the version they were written with.
"""

from __future__ import annotations

import re
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

GOLD_WRITER_V4 = "gold_writer_v4"
"""Current writer prompt id. New ANALYSIS shape and voice.

Same claim contract as v3 - `source_claims`, `news_claims` and every Round 5.1
provenance rule survive unchanged - but a different article: a fixed five-section
shape, a stated view, required asymmetry when one side has no driver, temporal
rather than causal wording about price, and a hard character ceiling instead of a
word-count suggestion.

A new file rather than an edit to v3, because Runs record the prompt they were
written under and quietly changing what `gold_writer_v3` means would detach them
from the rules they were actually written to.
"""

GOLD_HUMAN_STYLE_V1 = "gold_human_style_v1"
"""The shared voice contract, versioned on its own.

Included by prose article prompts rather than copied into them, so the day the
voice changes it changes in one place and every prompt that includes it records
which version it got. Intended for `NEWS_DIGEST` too when that arrives.

Never included by a `TRADE_PLAN` prompt, because there will not be one: that
document is rendered deterministically and has no voice to contract.
"""

GOLD_NEWS_DIGEST_WRITER_V1 = "gold_news_digest_writer_v1"
"""The NEWS_DIGEST writer prompt. Editorial content only.

A separate prompt rather than a mode of `gold_writer_v4`, because it asks for a
different *shape of answer*: the analysis writer returns an article, and this
one returns the two things a model should decide about a digest - which items
mattered and how the window balances. Everything else in a digest is arithmetic,
rendered by code, and the schema behind this prompt has nowhere to put it.

Includes `gold_human_style_v1` for the fourth time in the pipeline. The voice is
the product's, not one article type's.
"""

GOLD_REVIEWER_V1 = "gold_reviewer_v1"
"""The original reviewer prompt. Retained so historical reviews keep their meaning.

Its rubric asked for style as a content issue category at LOW or MEDIUM. Seven
production reviews were written under it, and their verdicts mean what that
rubric said they mean.
"""

GOLD_REVIEWER_V2 = "gold_reviewer_v2"
"""Current reviewer prompt id. Two independent axes.

Content integrity is unchanged in meaning - the same precedence of evidence, the
same categories, the same thresholds. What is new is a second judgement, human
style, carried in its own object with its own vocabulary and its own severities,
and explicitly forbidden from touching `status`, `score`, `issues` or
`revision_instructions`.

Includes `gold_human_style_v1` as an editorial rubric rather than restating it,
so the reviewer is judging against the same contract the writer was given. A
second copy would drift, and then the two stages would disagree about the voice
for a reason invisible in any diff.

A new file rather than an edit to v1, because reviews record the prompt they
were judged under.
"""

GOLD_FINALIZER_V1 = "gold_finalizer_v1"
"""The original finalizer prompt. Retained so historical revisions keep meaning.

It knew one kind of repair: apply the content review's corrections. Runs
finalized under it were edited to that rule and to no other.
"""

GOLD_FINALIZER_V2 = "gold_finalizer_v2"
"""Current finalizer prompt id. Repairs both axes in one pass.

Every content rule from v1 survives verbatim - minimum necessary revision, the
preserve list, the never-introduce list, and the rule that HIGH and CRITICAL
issues may not be declined. What is new is a human-style repair section that
treats each finding's `repair_instruction` as the entire mandate, an explicit
"a smoother article is not a better article" rule, and the ordering that content
correctness beats every style preference.

Includes `gold_human_style_v1` for the third time in the pipeline and for a
third purpose: the writer writes to it, the reviewer judges against it, and the
finalizer reads it only to understand what a repair is aiming at. One contract,
three readers, no copies.

A new file rather than an edit to v1, because finalizations record the prompt
they were made under.
"""

DEFAULT_WRITER_PROMPT = GOLD_WRITER_V4
DEFAULT_REVIEWER_PROMPT = GOLD_REVIEWER_V2
DEFAULT_FINALIZER_PROMPT = GOLD_FINALIZER_V2

INCLUDE_PATTERN = re.compile(r"^<!-- include: ([a-z0-9_]+) -->$", re.MULTILINE)
"""How a prompt pulls in a shared, separately versioned block.

One level only, and resolved at load time. The alternative - pasting the voice
contract into every prompt that needs it - means the copies drift, and then two
prompts claiming the same style produce different articles for a reason nobody
can see in a diff.
"""

STANDALONE_PROMPTS = frozenset({GOLD_HUMAN_STYLE_V1})
"""Fragments meant to be included, not loaded as a prompt in their own right.

They carry no `# SYSTEM RULES` and no `# OUTPUT CONTRACT`, so the section check
would refuse them - correctly, since sending one to a model on its own would
produce an article with no rules at all.
"""

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

    text = _resolve_includes(path.read_text(encoding="utf-8"), prompt_id)
    if prompt_id in STANDALONE_PROMPTS:
        return text

    missing = [section for section in REQUIRED_SECTIONS if section not in text]
    if missing:
        raise ValueError(f"prompt {prompt_id!r} is missing required sections: {missing}")
    return text


def _resolve_includes(text: str, prompt_id: str) -> str:
    """Substitute each `<!-- include: id -->` with that fragment's text.

    One level deep, deliberately: a fragment that could include another is a
    graph, and a prompt assembled from a graph is one nobody can read in full.
    Raises rather than leaving the marker in place - a prompt that silently
    shipped with its voice contract missing would still call the API, still cost
    money, and produce a differently-written article for no visible reason.
    """

    def substitute(match: re.Match[str]) -> str:
        included = match.group(1)
        if included == prompt_id:
            raise ValueError(f"prompt {prompt_id!r} includes itself")
        fragment = PROMPTS_DIR / f"{included}.md"
        if not fragment.is_file():
            raise FileNotFoundError(
                f"prompt {prompt_id!r} includes {included!r}, which does not exist"
            )
        body = fragment.read_text(encoding="utf-8")
        if INCLUDE_PATTERN.search(body):
            raise ValueError(f"included fragment {included!r} contains its own include")
        return body.strip()

    return INCLUDE_PATTERN.sub(substitute, text)


__all__ = [
    "DEFAULT_FINALIZER_PROMPT",
    "DEFAULT_REVIEWER_PROMPT",
    "DEFAULT_WRITER_PROMPT",
    "GOLD_FINALIZER_V1",
    "GOLD_FINALIZER_V2",
    "GOLD_HUMAN_STYLE_V1",
    "GOLD_NEWS_DIGEST_WRITER_V1",
    "GOLD_REVIEWER_V1",
    "GOLD_REVIEWER_V2",
    "GOLD_WRITER_V1",
    "GOLD_WRITER_V2",
    "GOLD_WRITER_V3",
    "GOLD_WRITER_V4",
    "INCLUDE_PATTERN",
    "PROMPTS_DIR",
    "STANDALONE_PROMPTS",
    "REQUIRED_SECTIONS",
    "load_prompt",
]
