"""Telegram transport for the ``/plan`` command bot.

Round 6.7. A second bot, deliberately separate from the one that delivers review
copies: a different token, a different credential name, and nothing in this
module can read the other one. The command bot listens to its operator; the
review bot talks to them. Folding both into one identity would mean a leaked
review token could also start Runs.

**Polling, never a server.** ``getUpdates`` with a zero timeout, called once per
scheduler tick: a short request that returns immediately whether or not anything
is waiting. No webhook, no listening socket, no public endpoint.

**The token is in the URL**, exactly as for the publisher, so every exception
from the HTTP layer is replaced before it leaves this module. ``sendMessage``
goes through :class:`TelegramPublisherClient` itself rather than a copy of it,
so a reply is confirmed - or classified as uncertain - by the one piece of code
this project already trusts to do that.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from goldpipeline.adapters.publisher_client import SendOutcome, SendRequest
from goldpipeline.adapters.telegram_publisher import (
    API_BASE,
    TelegramPublisherClient,
    _build_http_client,
    _Scrubbed,
)
from goldpipeline.config import TelegramSettings
from goldpipeline.domain.errors import PipelineError

logger = logging.getLogger(__name__)

COMMAND_BOT_PROVIDER = "telegram_command_bot"
POLL_LIMIT = 20
POLL_TIMEOUT_SECONDS = 0
"""Short polling. The scheduler owns the clock; this request never waits."""

DEFAULT_TIMEOUT_SECONDS = 10.0
ALLOWED_UPDATES = ("message",)
"""Message updates only. Edits, channel posts and callbacks are never delivered."""


class CommandBotError(PipelineError):
    code = "COMMAND_BOT_ERROR"


class CommandBotTransportError(CommandBotError):
    """The request may or may not have reached Telegram."""

    code = "COMMAND_BOT_TRANSPORT"


class CommandBotRefusedError(CommandBotError):
    """Telegram answered, and the answer was no. The description is not echoed."""

    code = "COMMAND_BOT_REFUSED"


class CommandBotResponseError(CommandBotError):
    """Telegram answered with something this module cannot read."""

    code = "COMMAND_BOT_RESPONSE"


@dataclass(frozen=True)
class BotIdentity:
    bot_id: int
    username: str


@dataclass(frozen=True)
class BotCommand:
    command: str
    description: str


class CommandBotClient(Protocol):
    """The five Bot API calls the command flow uses, and no others."""

    @property
    def provider(self) -> str: ...

    def get_me(self) -> BotIdentity: ...

    def get_updates(
        self, *, offset: int | None, limit: int = POLL_LIMIT
    ) -> list[dict[str, Any]]: ...

    def send_message(self, *, chat_id: str, text: str) -> SendOutcome: ...

    def set_my_commands(self, commands: Sequence[BotCommand]) -> None: ...

    def get_my_commands(self) -> list[BotCommand]: ...


class TelegramCommandBotClient:
    """The command bot, over the Bot API."""

    def __init__(
        self,
        *,
        token: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        http: Any | None = None,
    ) -> None:
        # Reuses the publisher's settings type purely as a container, so the
        # token travels inside an object whose repr already redacts it.
        self._settings = TelegramSettings(bot_token=token, timeout_seconds=timeout_seconds)
        self._http = http if http is not None else _build_http_client(self._settings)

    def __repr__(self) -> str:
        return "TelegramCommandBotClient(token=<redacted>)"

    __str__ = __repr__

    @property
    def provider(self) -> str:
        return COMMAND_BOT_PROVIDER

    def get_me(self) -> BotIdentity:
        result = self._call("getMe", {})
        if not isinstance(result, dict):
            raise CommandBotResponseError("getMe returned no bot description", method="getMe")
        bot_id, username = result.get("id"), result.get("username")
        if not isinstance(bot_id, int) or not isinstance(username, str) or not username:
            raise CommandBotResponseError("getMe returned no bot identity", method="getMe")
        return BotIdentity(bot_id=bot_id, username=username)

    def get_updates(self, *, offset: int | None, limit: int = POLL_LIMIT) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "limit": limit,
            "timeout": POLL_TIMEOUT_SECONDS,
            "allowed_updates": list(ALLOWED_UPDATES),
        }
        if offset is not None:
            payload["offset"] = offset
        result = self._call("getUpdates", payload)
        if not isinstance(result, list) or not all(isinstance(entry, dict) for entry in result):
            raise CommandBotResponseError("getUpdates returned no update list", method="getUpdates")
        return result

    def send_message(self, *, chat_id: str, text: str) -> SendOutcome:
        """One plain-text message, confirmed by the publisher's own logic."""
        client = TelegramPublisherClient(self._settings, http=self._http)
        return client.send(SendRequest(target_chat=chat_id, text=text, chunk_index=0))

    def set_my_commands(self, commands: Sequence[BotCommand]) -> None:
        payload = {
            "commands": [
                {"command": command.command, "description": command.description}
                for command in commands
            ]
        }
        if self._call("setMyCommands", payload) is not True:
            raise CommandBotResponseError("setMyCommands did not confirm", method="setMyCommands")

    def get_my_commands(self) -> list[BotCommand]:
        result = self._call("getMyCommands", {})
        if not isinstance(result, list):
            raise CommandBotResponseError("getMyCommands returned no list", method="getMyCommands")
        commands: list[BotCommand] = []
        for entry in result:
            if not isinstance(entry, dict):
                raise CommandBotResponseError("getMyCommands entry is not an object")
            commands.append(
                BotCommand(
                    command=str(entry.get("command", "")),
                    description=str(entry.get("description", "")),
                )
            )
        return commands

    def _call(self, method: str, payload: dict[str, Any]) -> Any:
        """One Bot API call. Nothing that leaves here has seen the URL."""
        logger.info("command_bot.call method=%s", method)
        url = f"{API_BASE}/bot{self._settings.bot_token}/{method}"
        try:
            response = self._http.post(url, json=payload)
        except Exception as exc:  # noqa: BLE001 - anything here may carry the URL
            raise CommandBotTransportError(
                f"the {method} request did not complete", method=method
            ) from _Scrubbed(exc)

        status = int(getattr(response, "status_code", 0) or 0)
        try:
            body = response.json()
        except Exception:  # noqa: BLE001 - a body that will not parse tells us nothing
            raise CommandBotResponseError(
                f"the {method} reply could not be parsed", method=method, status_code=status
            ) from None

        if not isinstance(body, dict) or body.get("ok") is not True:
            code = body.get("error_code") if isinstance(body, dict) else None
            raise CommandBotRefusedError(
                f"Telegram refused {method}",
                method=method,
                status_code=status,
                error_code=code if isinstance(code, int) else None,
            )
        return body.get("result")


__all__ = [
    "ALLOWED_UPDATES",
    "COMMAND_BOT_PROVIDER",
    "POLL_LIMIT",
    "POLL_TIMEOUT_SECONDS",
    "BotCommand",
    "BotIdentity",
    "CommandBotClient",
    "CommandBotError",
    "CommandBotRefusedError",
    "CommandBotResponseError",
    "CommandBotTransportError",
    "TelegramCommandBotClient",
]
