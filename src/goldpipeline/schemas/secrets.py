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
    OPENAI_API_KEY = "OPENAI_API_KEY"
    TELEGRAM_BOT_TOKEN = "TELEGRAM_BOT_TOKEN"

    @property
    def entry(self) -> str:
        """Name of the entry in the credential store.

        Lower-cased and readable, so a person browsing Windows Credential
        Manager can tell what they are looking at. Deliberately describes the
        *kind* of credential and never contains one.
        """
        return self.value.lower()


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
        if not self.configured:
            return "missing"
        return f"configured ({_HUMAN[self.source]})"


_HUMAN = {
    SecretSource.PROCESS_ENV: "process environment",
    SecretSource.WINDOWS_CREDENTIAL_MANAGER: "Windows Credential Manager",
    SecretSource.MISSING: "missing",
}


__all__ = ["SecretName", "SecretSource", "SecretStatus"]
