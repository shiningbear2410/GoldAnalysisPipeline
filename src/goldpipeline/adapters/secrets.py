"""Where credentials come from.

The protocol is one method, and that is the point: everything above this line -
config, services, the worker - asks *for a secret by name* and never learns
whether the answer came from an environment variable, an operating system
credential store, or a test double.

**Process environment first, credential store second.** That precedence is
deliberate and load-bearing in both directions:

* it keeps every existing workflow working. An operator who exports
  ``ANTHROPIC_API_KEY`` in a PowerShell session gets exactly the behaviour they
  had before this module existed, and every test written against that behaviour
  still passes;
* it makes a temporary override possible without touching the durable store.
  Trying a different key for ten minutes should not mean rewriting a credential
  and remembering to put the old one back.

The direction is one-way. A value found in the environment is **never** written
into the credential store: promoting a throwaway override into permanent storage
because it happened to be present is exactly the sort of helpfulness that leaves
a stale key in a vault two years later.

Nothing here imports ``keyring``. The one module that does is
:mod:`goldpipeline.adapters.windows_credentials`, so this file - and everything
that depends on it - works on a machine with no credential backend at all.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol, runtime_checkable

from goldpipeline.schemas.secrets import SecretName, SecretSource

logger = logging.getLogger(__name__)


@runtime_checkable
class SecretProvider(Protocol):
    """Anything that can answer "what is the value of this credential?"."""

    @property
    def source(self) -> SecretSource:
        """Which kind of store this is, for safe reporting."""
        ...

    def get_secret(self, name: SecretName) -> str | None:
        """Return the credential, or ``None`` when this store does not have it.

        Absence is a return value, not an exception: a store not holding a
        credential is an ordinary situation that the next store in line may
        resolve.

        Raises:
            CredentialError: The store exists but could not be consulted. That
                *is* exceptional - it must never be mistaken for "not set", or a
                broken vault would silently look like a missing key.
        """
        ...


class EnvironmentSecretProvider:
    """Reads credentials from this process's environment.

    Only ``os.environ`` - the variables this process actually has. It never
    reaches into the User or Machine registry hives: those are what a *future*
    process would inherit, not what this one has, and reporting them as
    available would promise a scheduled task something that may not be true.
    """

    def __init__(self, env: dict[str, str] | None = None) -> None:
        """Read from *env*, defaulting to the live process environment."""
        self._env = env

    @property
    def source(self) -> SecretSource:
        return SecretSource.PROCESS_ENV

    def get_secret(self, name: SecretName) -> str | None:
        source = os.environ if self._env is None else self._env
        value = (source.get(name.value) or "").strip()
        return value or None


class CompositeSecretProvider:
    """Asks each provider in turn and takes the first answer.

    Order is precedence, and the first provider wins outright: this does not
    merge, compare, or prefer the "better" answer. A credential present in two
    places means the operator put it in two places, and second-guessing which
    they meant would make the behaviour unpredictable exactly when it matters.
    """

    def __init__(self, providers: list[SecretProvider]) -> None:
        self._providers = list(providers)

    @property
    def source(self) -> SecretSource:
        """The composite has no single source; callers should use :meth:`resolve`."""
        return SecretSource.MISSING

    @property
    def providers(self) -> list[SecretProvider]:
        return list(self._providers)

    def get_secret(self, name: SecretName) -> str | None:
        value, _ = self.resolve(name)
        return value

    def resolve(self, name: SecretName) -> tuple[str | None, SecretSource]:
        """Return the credential and which store answered.

        The source is what makes a status line worth reading: "configured" alone
        does not tell an operator whether a scheduled task will still find it
        tomorrow, and "configured (process environment)" does.
        """
        for provider in self._providers:
            value = provider.get_secret(name)
            if value is not None:
                return value, provider.source
        return None, SecretSource.MISSING


class FakeSecretProvider:
    """Offline stand-in, for tests and for ``--fake-secrets``.

    Holds whatever it is given and nothing else. Configure ``raises`` to model a
    store that exists but cannot be consulted - the case that must never be
    mistaken for a missing credential.
    """

    def __init__(
        self,
        secrets: dict[SecretName, str] | None = None,
        *,
        source: SecretSource = SecretSource.WINDOWS_CREDENTIAL_MANAGER,
        raises: Exception | None = None,
    ) -> None:
        self._secrets = dict(secrets or {})
        self._source = source
        self._raises = raises
        self.reads: list[SecretName] = []
        """Every lookup, so a test can prove a stage asked for nothing."""

    @property
    def source(self) -> SecretSource:
        return self._source

    def get_secret(self, name: SecretName) -> str | None:
        self.reads.append(name)
        if self._raises is not None:
            raise self._raises
        return self._secrets.get(name)

    def set_secret(self, name: SecretName, value: str) -> None:
        self._secrets[name] = value

    def delete_secret(self, name: SecretName) -> None:
        self._secrets.pop(name, None)


def default_provider(env: dict[str, str] | None = None) -> EnvironmentSecretProvider:
    """The provider used when a caller supplies none.

    Environment-only, which is precisely the behaviour every round before this
    one had. Reaching the operating system's credential store is something a
    caller opts into, so importing this package never touches a vault.
    """
    return EnvironmentSecretProvider(env)


__all__ = [
    "CompositeSecretProvider",
    "EnvironmentSecretProvider",
    "FakeSecretProvider",
    "SecretProvider",
    "default_provider",
]
