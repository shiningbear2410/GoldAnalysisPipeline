"""Contracts for the ``/plan`` command: its settings, receipts and replies.

Round 6.7. The settings live in their own small file beside the automation
state, not in the production configuration. ``REQUIRED_PRODUCTION_KEYS`` is
every ``ConfigKey``, so a new member would become mandatory for the running
worker the moment it shipped; and the strict loader refuses keys it does not
know. A separate file whose absence means "off" is the smallest mechanism that
changes neither - the production configuration and its fingerprint stay exactly
as they are, and a worker that has never been told about ``/plan`` behaves
exactly as it did before.

Nothing here can hold a token. The bot's credential is resolved from the
credential store at the moment of use, by name.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from goldpipeline.schemas.common import StrictModel, UtcDatetime

PLAN_COMMAND_SCHEMA_VERSION = "1"

PLAN_COMMAND = "plan"
PLAN_COMMAND_DESCRIPTION = "Tạo kế hoạch vàng"
REQUEST_SOURCE = "telegram_plan"
EVENT_SOURCE = "telegram_plan_command"

BOT_USERNAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{1,29}[Bb][Oo][Tt]$")
CHAT_ID_PATTERN = re.compile(r"^-?[0-9]{1,19}$")
MAX_AUTHORISED_CHATS = 5


class PlanCommandSettings(StrictModel):
    """Whether the command bot listens, as whom, and to whom."""

    schema_version: Literal["1"] = "1"
    enabled: bool = False
    bot_id: int | None = Field(default=None, ge=1)
    bot_username: str | None = None
    authorised_chat_ids: list[str] = Field(default_factory=list, max_length=MAX_AUTHORISED_CHATS)
    activated_at: UtcDatetime | None = None

    @field_validator("bot_username")
    @classmethod
    def _a_bot_username(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().removeprefix("@")
        if not BOT_USERNAME_PATTERN.fullmatch(cleaned):
            raise ValueError("bot_username must be a Telegram bot username ending in 'bot'")
        return cleaned

    @field_validator("authorised_chat_ids")
    @classmethod
    def _numeric_chats(cls, value: list[str]) -> list[str]:
        cleaned = [entry.strip() for entry in value]
        for entry in cleaned:
            if not CHAT_ID_PATTERN.fullmatch(entry):
                raise ValueError("every authorised chat must be a numeric Telegram chat id")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("authorised chats must not repeat")
        return cleaned

    @model_validator(mode="after")
    def _complete_when_enabled(self) -> PlanCommandSettings:
        if self.enabled and (
            self.bot_id is None or self.bot_username is None or not self.authorised_chat_ids
        ):
            raise ValueError(
                "an enabled /plan command needs the bot's id, its username and at least "
                "one authorised chat"
            )
        return self


class DeliveryKind(StrEnum):
    ACKNOWLEDGEMENT = "ACKNOWLEDGEMENT"
    PLAN = "PLAN"
    FAILURE_NOTICE = "FAILURE_NOTICE"


class DeliveryStatus(StrEnum):
    """How one reply ended. ``UNCERTAIN`` is never retried."""

    DELIVERED = "DELIVERED"
    FAILED = "FAILED"
    UNCERTAIN = "UNCERTAIN"
    SKIPPED = "SKIPPED"


class IgnoreReason(StrEnum):
    NOT_A_MESSAGE = "NOT_A_MESSAGE"
    NOT_A_PLAN_COMMAND = "NOT_A_PLAN_COMMAND"
    UNAUTHORISED_CHAT = "UNAUTHORISED_CHAT"
    FROM_A_BOT = "FROM_A_BOT"
    MALFORMED = "MALFORMED"


class PlanReceipt(StrictModel):
    """One accepted ``/plan``. Written after the event is durably in the inbox."""

    schema_version: str = PLAN_COMMAND_SCHEMA_VERSION
    event_id: str
    update_id: int
    bot_id: int
    chat_id: str
    message_id: int
    message_date: UtcDatetime
    accepted_at: UtcDatetime


class IgnoredUpdate(StrictModel):
    """An update that was deliberately not acted on. Holds no text and no raw chat id."""

    schema_version: str = PLAN_COMMAND_SCHEMA_VERSION
    update_id: int
    reason: IgnoreReason
    chat_ref: str | None = Field(
        default=None, description="A digest of the chat id, so repeats are visible without it."
    )
    recorded_at: UtcDatetime


class CommandDeliveryIntent(StrictModel):
    """Committed before the send, so a crash is distinguishable from a failure."""

    schema_version: str = PLAN_COMMAND_SCHEMA_VERSION
    event_id: str
    kind: DeliveryKind
    attempt_id: str
    created_at: UtcDatetime
    chat_id: str
    run_id: str | None = None
    text_sha256: str
    char_count: int = Field(ge=1)


class CommandDeliveryResult(StrictModel):
    """The reply's outcome. Its existence means the reply is never sent again."""

    schema_version: str = PLAN_COMMAND_SCHEMA_VERSION
    event_id: str
    kind: DeliveryKind
    attempt_id: str
    status: DeliveryStatus
    chat_id: str
    run_id: str | None = None
    started_at: UtcDatetime
    completed_at: UtcDatetime
    message_id: int | None = None
    failure_code: str | None = None
    detail: str | None = None


__all__ = [
    "BOT_USERNAME_PATTERN",
    "CHAT_ID_PATTERN",
    "EVENT_SOURCE",
    "MAX_AUTHORISED_CHATS",
    "PLAN_COMMAND",
    "PLAN_COMMAND_DESCRIPTION",
    "PLAN_COMMAND_SCHEMA_VERSION",
    "REQUEST_SOURCE",
    "CommandDeliveryIntent",
    "CommandDeliveryResult",
    "DeliveryKind",
    "DeliveryStatus",
    "IgnoreReason",
    "IgnoredUpdate",
    "PlanCommandSettings",
    "PlanReceipt",
]
