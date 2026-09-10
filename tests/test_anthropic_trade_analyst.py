"""What the live ranking client sends, and what it refuses to accept back.

Round 6.6h. Three of this round's four defects were found by a real call and
none of them by a fixture, so the two constraints that made the call work are
pinned here against a stub SDK rather than left to the next live run to
rediscover.

Nothing in this file reaches a vendor. The SDK client is injected, which is the
seam the adapter was built with.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from goldpipeline.adapters.anthropic_trade_analyst import (
    ANTHROPIC_PROVIDER,
    FENCE,
    AnthropicTradeAnalystClient,
)
from goldpipeline.adapters.trade_analyst_client import TradeAnalystRequest
from goldpipeline.config import WriterSettings
from goldpipeline.domain.errors import WriterResponseError


@dataclass
class Block:
    type: str
    text: str = ""


@dataclass
class Reply:
    content: list[Block]
    stop_reason: str = "end_turn"
    model: str = "claude-sonnet-5"
    usage: Any = None


@dataclass
class Messages:
    reply: Reply
    calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> Reply:
        self.calls.append(kwargs)
        return self.reply


@dataclass
class StubClient:
    messages: Messages


def client(reply: Reply) -> tuple[AnthropicTradeAnalystClient, Messages]:
    messages = Messages(reply=reply)
    settings = WriterSettings(api_key="not-a-real-key", model="claude-sonnet-5")
    return AnthropicTradeAnalystClient(settings, client=StubClient(messages)), messages


def request(user: str = "payload") -> TradeAnalystRequest:
    return TradeAnalystRequest(system="rules", user=user, max_tokens=6752)


ANSWER = json.dumps(
    {
        "bai_entry_candidate_ids": ["a" * 16],
        "seo_entry_candidate_ids": [],
        "upper_reference_candidate_ids": [],
        "lower_reference_candidate_ids": [],
    }
)


# --------------------------------------------------------------------------
# what the client sends
# --------------------------------------------------------------------------


def test_thinking_is_switched_off() -> None:
    """The first live call spent 16,000 output tokens thinking and answered nothing.

    Ordering a list is not a problem extended thinking helps with, and on 130
    candidates it consumed the whole budget before reaching the JSON.
    """
    analyst, messages = client(Reply(content=[Block("text", ANSWER)]))

    analyst.rank(request())

    assert messages.calls[0]["thinking"] == {"type": "disabled"}


def test_the_conversation_ends_with_the_user_turn() -> None:
    """No assistant prefill: this model refuses one outright.

    Prefilling an opening brace would have been the tidier way to guarantee
    JSON, and the provider answers "the conversation must end with a user
    message" - so the fence is removed afterwards instead.
    """
    analyst, messages = client(Reply(content=[Block("text", ANSWER)]))

    analyst.rank(request())
    sent = messages.calls[0]["messages"]

    assert [entry["role"] for entry in sent] == ["user"]
    assert sent[0]["content"] == "payload"


def test_a_fenced_answer_is_unwrapped() -> None:
    """The prompt forbids a code fence and the model used one anyway."""
    fenced = FENCE + "json" + chr(10) + ANSWER + chr(10) + FENCE
    analyst, _ = client(Reply(content=[Block("text", fenced)]))

    response = analyst.rank(request())

    assert response.text == ANSWER
    assert json.loads(response.text)["bai_entry_candidate_ids"] == ["a" * 16]


def test_a_bare_answer_is_left_exactly_alone() -> None:
    analyst, _ = client(Reply(content=[Block("text", ANSWER)]))

    assert analyst.rank(request()).text == ANSWER


def test_unfencing_refuses_to_guess() -> None:
    """Narrow on purpose: only a fence on both ends, with content between."""
    from goldpipeline.adapters.anthropic_trade_analyst import _unfence

    for untouched in (ANSWER, FENCE + "json" + FENCE, "not fenced " + FENCE):
        assert _unfence(untouched) == untouched.strip()

    assert _unfence(FENCE + chr(10) + ANSWER + chr(10) + FENCE) == ANSWER


def test_the_system_and_the_ceiling_are_carried_through() -> None:
    analyst, messages = client(Reply(content=[Block("text", ANSWER)]))

    analyst.rank(request())

    assert messages.calls[0]["system"] == "rules"
    assert messages.calls[0]["max_tokens"] == 6752
    assert messages.calls[0]["model"] == "claude-sonnet-5"


def test_the_provider_and_model_are_reported() -> None:
    analyst, _ = client(Reply(content=[Block("text", ANSWER)], model="claude-sonnet-5"))

    response = analyst.rank(request())

    assert response.provider == ANTHROPIC_PROVIDER == "anthropic"
    assert response.model == "claude-sonnet-5"
    assert analyst.model == "claude-sonnet-5"


# --------------------------------------------------------------------------
# what it refuses
# --------------------------------------------------------------------------


def test_a_truncated_answer_is_refused() -> None:
    """§12. The failure the fixed ceiling used to produce, still caught."""
    analyst, _ = client(Reply(content=[Block("text", "{")], stop_reason="max_tokens"))

    with pytest.raises(WriterResponseError):
        analyst.rank(request())


def test_a_non_text_block_is_refused() -> None:
    """A tool call in a ranking answer means the model did something else."""
    analyst, _ = client(Reply(content=[Block("tool_use")]))

    with pytest.raises(WriterResponseError, match="not text"):
        analyst.rank(request())


def test_an_empty_response_is_refused() -> None:
    analyst, _ = client(Reply(content=[]))

    with pytest.raises(WriterResponseError, match="no content"):
        analyst.rank(request())


def test_empty_text_is_refused() -> None:
    """Sixteen thousand thinking tokens and no answer looked exactly like this."""
    analyst, _ = client(Reply(content=[Block("text", "   ")]))

    with pytest.raises(WriterResponseError, match="empty text"):
        analyst.rank(request())


# --------------------------------------------------------------------------
# what it never does
# --------------------------------------------------------------------------


def test_the_client_validates_nothing_about_the_ids() -> None:
    """The transport shapes generation; the domain decides correctness.

    A ranking naming candidates that do not exist comes back from here intact,
    and is refused by ``parse_ranking`` - which is the only place that knows
    what a candidate is.
    """
    nonsense = '{"bai_entry_candidate_ids": ["deadbeefdeadbeef"]}'
    analyst, _ = client(Reply(content=[Block("text", nonsense)]))

    response = analyst.rank(request())

    assert response.text == nonsense


def test_no_credential_is_read_or_logged_by_this_module() -> None:
    """The key arrives on the settings and goes straight to the SDK builder.

    This module resolves nothing, reads no environment and puts nothing about
    the credential in a log line - the one thing it logs is the model, the
    ceiling and the timeout.
    """
    import ast
    from pathlib import Path

    source = Path("src/goldpipeline/adapters/anthropic_trade_analyst.py").read_text(
        encoding="utf-8"
    )

    assert "from_env" not in source
    assert "os.environ" not in source
    assert "getenv" not in source

    logged = [
        ast.unparse(node)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"info", "warning", "error", "debug"}
    ]
    assert logged
    for call in logged:
        assert "api_key" not in call, call

    # The key is named exactly where it is handed over, and nowhere else.
    assert source.count("settings.api_key") == 1
