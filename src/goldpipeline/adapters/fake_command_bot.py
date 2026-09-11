"""An offline command bot. Records every call; reaches nothing.

Tests script the updates Telegram would return and read back what was sent. A
failure can be injected into polling or sending, which is how the uncertainty
and crash-replay paths are exercised without a network.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from goldpipeline.adapters.publisher_client import SendOutcome
from goldpipeline.adapters.telegram_command_bot import (
    POLL_LIMIT,
    BotCommand,
    BotIdentity,
)

FAKE_BOT_ID = 7000000001
FAKE_BOT_USERNAME = "pcplanbot"
FAKE_CHAT_ID = "7387726751"
MESSAGE_DATE = 1_789_000_000
"""2026-09-10T00:26:40Z, as Telegram would send it: epoch seconds."""


def plan_update(
    update_id: int,
    *,
    chat_id: str = FAKE_CHAT_ID,
    message_id: int = 100,
    text: str | None = "/plan",
    date: int = MESSAGE_DATE,
    chat_type: str = "private",
    from_bot: bool = False,
) -> dict[str, Any]:
    """One ``message`` update, shaped like the Bot API's."""
    message: dict[str, Any] = {
        "message_id": message_id,
        "date": date,
        "chat": {"id": int(chat_id), "type": chat_type},
        "from": {"id": int(chat_id), "is_bot": from_bot, "first_name": "Operator"},
    }
    if text is not None:
        message["text"] = text
    return {"update_id": update_id, "message": message}


@dataclass
class FakeCommandBot:
    identity: BotIdentity = field(
        default_factory=lambda: BotIdentity(bot_id=FAKE_BOT_ID, username=FAKE_BOT_USERNAME)
    )
    updates: list[dict[str, Any]] = field(default_factory=list)
    sent: list[tuple[str, str]] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    commands: list[BotCommand] = field(default_factory=list)
    send_failure: Exception | None = None
    poll_failure: Exception | None = None
    next_message_id: int = 900

    @property
    def provider(self) -> str:
        return "fake_command_bot"

    def get_me(self) -> BotIdentity:
        self.calls.append("getMe")
        return self.identity

    def get_updates(self, *, offset: int | None, limit: int = POLL_LIMIT) -> list[dict[str, Any]]:
        self.calls.append("getUpdates")
        if self.poll_failure is not None:
            raise self.poll_failure
        waiting = [u for u in self.updates if offset is None or u["update_id"] >= offset]
        return waiting[:limit]

    def send_message(self, *, chat_id: str, text: str) -> SendOutcome:
        self.calls.append("sendMessage")
        if self.send_failure is not None:
            raise self.send_failure
        self.sent.append((chat_id, text))
        self.next_message_id += 1
        return SendOutcome(message_id=self.next_message_id, chat_id=chat_id)

    def set_my_commands(self, commands: Sequence[BotCommand]) -> None:
        self.calls.append("setMyCommands")
        self.commands = list(commands)

    def get_my_commands(self) -> list[BotCommand]:
        self.calls.append("getMyCommands")
        return list(self.commands)


__all__ = [
    "FAKE_BOT_ID",
    "FAKE_BOT_USERNAME",
    "FAKE_CHAT_ID",
    "MESSAGE_DATE",
    "FakeCommandBot",
    "plan_update",
]
