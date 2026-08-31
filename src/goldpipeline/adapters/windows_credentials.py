"""Windows Credential Manager, via keyring.

The only module in this project that imports ``keyring``. Everything above it
talks to :class:`~goldpipeline.adapters.secrets.SecretProvider`, so the whole
package - and the whole test suite - runs on a machine with no credential
backend at all.

**Why a credential store rather than the environment.** A Task Scheduler task
runs in a fresh process that inherits neither a PowerShell session's variables
nor anything set with ``$env:``. The alternatives are all worse than they look:
``setx`` and the registry put the secret in a place any process on the machine
can read and any support screenshot can capture; a ``.env`` file puts it one
mistaken ``git add`` away from a public repository; the task XML puts it in a
file that gets exported and mailed around. Credential Manager stores it
encrypted against the user's login and hands it back only to that user.

**The backend is checked, not assumed.** ``import keyring`` succeeding proves
nothing: keyring will happily fall back to a backend that stores nothing, and
plugins exist that store credentials in a plaintext file. Both would look like
success. So before this is declared ready for unattended use, the *active*
backend is compared against a list of stores that are actually encrypted by the
operating system, and anything else fails closed.

**No plaintext fallback, ever.** If the secure store is unavailable, this module
raises. It does not write a file, does not set an environment variable, and does
not degrade quietly to something that happens to work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from goldpipeline.domain.errors import (
    CredentialBackendUnavailableError,
    CredentialDeleteError,
    CredentialNotFoundError,
    CredentialReadError,
    CredentialWriteError,
    InsecureCredentialBackendError,
)
from goldpipeline.schemas.secrets import SecretName, SecretSource

logger = logging.getLogger(__name__)

SERVICE_NAME = "GoldAnalysisPipeline"
"""Service namespace for every entry this project owns.

One namespace, so an operator can find them together in Credential Manager and
remove them together when the project is retired.
"""

SECURE_BACKENDS = frozenset(
    {
        "keyring.backends.Windows.WinVaultKeyring",
        "keyring.backends.macOS.Keyring",
        "keyring.backends.SecretService.Keyring",
        "keyring.backends.libsecret.Keyring",
    }
)
"""Backends whose storage is encrypted by the operating system.

An allowlist rather than a blocklist, because the failure mode of getting this
wrong is silent. A new plaintext plugin appearing on a machine would be rejected
by default; a new *secure* one would merely need adding here, which is a code
change someone reviews.
"""

INSECURE_BACKEND_HINTS = ("fail.", "null.", "chainer.", "plaintext", "PlaintextKeyring")
"""Names that are definitely not a secure store, for a clearer message.

``fail`` and ``null`` are keyring's own no-op backends. ``chainer`` is a
dispatcher, not a store - if it is the active backend then nothing underneath it
volunteered, which means there is nowhere secure to write.
"""


@dataclass(frozen=True)
class BackendReport:
    """What the credential backend is, and whether it may be trusted.

    Names the class and nothing else. A backend report is printed by a
    diagnostic command and pasted into chat windows, so it must never contain an
    entry, a service, or a value.
    """

    available: bool
    secure: bool
    backend: str
    detail: str

    @property
    def ready(self) -> bool:
        """Whether unattended execution can rely on this store."""
        return self.available and self.secure


def inspect_backend(module: Any = None) -> BackendReport:
    """Identify the active credential backend without reading anything.

    Reads no entry, lists no credential, and writes nothing. It asks keyring
    which backend it would use and compares the answer against
    :data:`SECURE_BACKENDS`.

    Args:
        module: The keyring module, or a stand-in. Injected by tests so the
            whole check runs with no credential store present.
    """
    try:
        keyring = module if module is not None else _load_keyring()
    except CredentialBackendUnavailableError as exc:
        return BackendReport(available=False, secure=False, backend="none", detail=exc.message)

    try:
        backend = keyring.get_keyring()
    except Exception:  # noqa: BLE001 - vendor errors here are undocumented
        return BackendReport(
            available=False,
            secure=False,
            backend="unknown",
            detail="the credential backend could not be determined",
        )

    name = f"{type(backend).__module__}.{type(backend).__name__}"

    if name in SECURE_BACKENDS:
        return BackendReport(
            available=True,
            secure=True,
            backend=name,
            detail="credentials are encrypted by the operating system",
        )

    if any(hint in name for hint in INSECURE_BACKEND_HINTS):
        return BackendReport(
            available=True,
            secure=False,
            backend=name,
            detail=(
                "this backend does not store credentials securely, or stores "
                "nothing at all. Unattended execution will not use it."
            ),
        )

    return BackendReport(
        available=True,
        secure=False,
        backend=name,
        detail=(
            "this backend is not on the list of stores known to be encrypted by "
            "the operating system, so it is not trusted for unattended use. "
            "Adding one is a deliberate code change."
        ),
    )


class WindowsCredentialSecretProvider:
    """Reads and writes credentials in the operating system's store.

    Named for its purpose rather than its platform: the same code works against
    the macOS and Secret Service backends, and the allowlist decides. On this
    project's target it is Windows Credential Manager.
    """

    def __init__(self, module: Any = None, *, service: str = SERVICE_NAME) -> None:
        """Build a provider.

        Args:
            module: The keyring module, or a stand-in. Injected by tests, so no
                test in this repository touches a real credential store.
            service: Namespace for this project's entries.
        """
        self._module = module
        self._service = service

    @property
    def source(self) -> SecretSource:
        return SecretSource.WINDOWS_CREDENTIAL_MANAGER

    @property
    def service(self) -> str:
        return self._service

    # -- reading -----------------------------------------------------------

    def get_secret(self, name: SecretName) -> str | None:
        """Read one credential.

        Returns:
            The value, or ``None`` when the store simply has no such entry.

        Raises:
            CredentialBackendUnavailableError: keyring is not installed.
            CredentialReadError: The store exists but refused. Deliberately not
                ``None``: a broken vault must never be indistinguishable from an
                unset credential, or a scheduled task would report a missing key
                when the real problem is a locked one.
        """
        keyring = self._keyring()
        try:
            value = keyring.get_password(self._service, name.entry)
        except Exception as exc:  # noqa: BLE001 - backend errors are undocumented
            raise CredentialReadError(
                f"the credential store could not be read for {name.value}",
                setting=name.value,
                service=self._service,
            ) from _Scrubbed(exc)

        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    # -- writing -----------------------------------------------------------

    def set_secret(self, name: SecretName, value: str) -> None:
        """Save one credential, refusing an insecure store.

        The backend is re-checked here rather than trusted from an earlier call:
        writing a credential is the one operation whose consequences outlive the
        process, and a store that turned out to be a plaintext file would have
        left the secret on disk.

        Raises:
            InsecureCredentialBackendError: The active backend is not trusted.
            CredentialWriteError: The store refused.
        """
        cleaned = value.strip()
        if not cleaned:
            raise CredentialWriteError(
                f"refusing to store an empty value for {name.value}", setting=name.value
            )

        report = inspect_backend(self._module)
        if not report.ready:
            raise InsecureCredentialBackendError(
                f"refusing to write {name.value} to {report.backend}: {report.detail} "
                "Nothing was written, and no file or environment variable was "
                "created as a fallback.",
                setting=name.value,
                backend=report.backend,
            )

        try:
            self._keyring().set_password(self._service, name.entry, cleaned)
        except Exception as exc:  # noqa: BLE001
            raise CredentialWriteError(
                f"the credential store refused to save {name.value}",
                setting=name.value,
                service=self._service,
            ) from _Scrubbed(exc)

        logger.info("credential stored service=%s entry=%s", self._service, name.entry)

    def delete_secret(self, name: SecretName) -> None:
        """Remove one credential.

        Raises:
            CredentialNotFoundError: There was no such entry.
            CredentialDeleteError: The store refused.
        """
        keyring = self._keyring()
        try:
            keyring.delete_password(self._service, name.entry)
        except Exception as exc:  # noqa: BLE001
            if _looks_missing(exc):
                raise CredentialNotFoundError(
                    f"there is no stored credential for {name.value}", setting=name.value
                ) from None
            raise CredentialDeleteError(
                f"the credential store refused to delete {name.value}",
                setting=name.value,
                service=self._service,
            ) from _Scrubbed(exc)

        logger.info("credential deleted service=%s entry=%s", self._service, name.entry)

    # -- internals ---------------------------------------------------------

    def _keyring(self) -> Any:
        return self._module if self._module is not None else _load_keyring()


def _load_keyring() -> Any:
    """Import keyring, or explain how to get it.

    Imported here rather than at module scope so the package works, and the
    suite runs, on a machine that has never had a credential backend.
    """
    try:
        import keyring
    except ImportError as exc:  # pragma: no cover - exercised by injecting a fake
        raise CredentialBackendUnavailableError(
            "the keyring package is not installed, so the secure credential "
            'store cannot be reached. Install it with `pip install -e ".[secrets]"`.',
            setting="keyring",
        ) from exc
    return keyring


def _looks_missing(exc: BaseException) -> bool:
    """Whether a delete failed because there was nothing to delete.

    keyring's backends signal this differently, and the distinction matters:
    removing a credential that was already gone is the operator's intent
    achieved, not a failure to report.
    """
    return type(exc).__name__ in {"PasswordDeleteError", "KeyringError"} and (
        "not found" in str(exc).lower() or "no such" in str(exc).lower()
    )


class _Scrubbed(Exception):
    """A stand-in cause carrying no backend text.

    ``raise ... from exc`` would attach the original, and a printed traceback
    would then show whatever the credential store put in its message - which,
    for a store whose whole job is handling secrets, is not a risk worth taking.
    """

    def __init__(self, original: BaseException) -> None:
        super().__init__(f"{type(original).__name__} (details withheld)")


__all__ = [
    "INSECURE_BACKEND_HINTS",
    "SECURE_BACKENDS",
    "SERVICE_NAME",
    "BackendReport",
    "WindowsCredentialSecretProvider",
    "inspect_backend",
]
