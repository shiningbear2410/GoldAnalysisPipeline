"""What a credential is, and where it came from.

Three names and no more. The line this module draws - and the reason it exists
as a schema rather than a pair of strings - is between a **credential** and a
**configuration value**:

* a credential is something an attacker wants: an API key, a bot token. It goes
  in the operating system's credential store, never in a file this repository
  can see;
* a destination chat, a symbol, a timeframe, a feature flag are configuration.
  They are not secret, they belong in the environment where an operator can
  read them back, and putting them in a credential store would only make them
  harder to audit without making anything safer.

``TELEGRAM_TARGET_CHAT_ID`` is the one people most want to move across that
line. It stays configuration: knowing which channel the pipeline posts to grants
nobody the ability to post there.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from goldpipeline.schemas.common import StrictModel


class SecretName(StrEnum):
    """The credentials this pipeline holds.

    Named for the environment variable each corresponds to, so an operator
    reading a status line and an operator reading `.env.example` are looking at
    the same word.
    """

    ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
    DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"
    OPENAI_API_KEY = "OPENAI_API_KEY"
    TELEGRAM_BOT_TOKEN = "TELEGRAM_BOT_TOKEN"
    INGEST_TOKEN = "INGEST_TOKEN"

    @property
    def entry(self) -> str:
        """Name of the entry in the credential store.

        Lower-cased and readable, so a person browsing Windows Credential
        Manager can tell what they are looking at. Deliberately describes the
        *kind* of credential and never contains one.
        """
        return self.value.lower()


REQUIRED_SECRETS = frozenset({SecretName.ANTHROPIC_API_KEY})
"""Credentials the production pipeline cannot do its work without.

One entry, because the Reviewer always calls Anthropic and the Writer and
Finalizer do so by default. They remain three independent requests with three
different prompts; what they share is an account, not a conversation.

``DEEPSEEK_API_KEY`` is deliberately absent. Generation may be pointed at
DeepSeek, but nothing requires it, and a required secret is one whose absence
stops the pipeline - see :data:`CONDITIONAL_SECRETS`.
"""

CONDITIONAL_SECRETS = frozenset(
    {SecretName.TELEGRAM_BOT_TOKEN, SecretName.INGEST_TOKEN, SecretName.DEEPSEEK_API_KEY}
)
"""Needed only when the feature that uses them actually runs.

Absent is not a fault while unattended publishing is off, which is why readiness
has always been reported against the mode rather than the list. ``INGEST_TOKEN``
joined on the same terms: remote intake is off by default, and a credential for a
switched-off feature is not a missing credential.

``DEEPSEEK_API_KEY`` joins on those terms too, with one difference worth naming:
the others are gated by a config flag, this one by a selection. An operator who
never picks DeepSeek must never be asked for a DeepSeek key, so it is resolved at
the moment a DeepSeek call is about to be made - never at import, at start-up, or
while rendering a status.
"""

OPTIONAL_SECRETS = frozenset({SecretName.OPENAI_API_KEY})
"""Kept for the optional legacy adapter, and never required.

The Reviewer used OpenAI until Round 9.3.1. The adapter still exists and still
works if a caller wires it up deliberately, so the credential stays *nameable* -
but an operator who never wants it must never be asked to create one. Reported
as "not required" rather than "missing": those are different facts, and only one
of them is a problem.
"""


class SecretSource(StrEnum):
    """Where a resolved credential came from.

    Reported so an operator can tell a temporary session override from the
    durable store - which is the difference between "this works now" and "this
    will still work when Task Scheduler runs it at 3am".
    """

    PROCESS_ENV = "PROCESS_ENV"
    WINDOWS_CREDENTIAL_MANAGER = "WINDOWS_CREDENTIAL_MANAGER"
    MISSING = "MISSING"


class SecretStatus(StrictModel):
    """Whether one credential is available, and from where.

    Carries no value and has nowhere to put one. Everything about this type is
    designed to be safe to print, log, serialize and paste into a chat window
    while asking for help.
    """

    name: SecretName
    configured: bool
    source: SecretSource = SecretSource.MISSING
    detail: str | None = Field(
        default=None, description="Safe explanation - a backend name or a reason, never a value."
    )

    @property
    def summary(self) -> str:
        """One line an operator can read at a glance."""
        if self.configured:
            return f"configured ({_HUMAN[self.source]})"
        if self.name in OPTIONAL_SECRETS:
            # Not a gap. Saying "missing" here would send an operator hunting
            # for a credential this pipeline has no use for.
            return "not required"
        return "missing"


_HUMAN = {
    SecretSource.PROCESS_ENV: "process environment",
    SecretSource.WINDOWS_CREDENTIAL_MANAGER: "Windows Credential Manager",
    SecretSource.MISSING: "missing",
}


__all__ = [
    "CONDITIONAL_SECRETS",
    "OPTIONAL_SECRETS",
    "CONDITIONAL_SECRETS",
    "REQUIRED_SECRETS",
    "SecretName",
    "SecretSource",
    "SecretStatus",
]
