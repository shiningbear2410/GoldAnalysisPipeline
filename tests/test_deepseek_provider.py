"""DeepSeek as a generation provider: mapping, transport, failure and refusal.

Every test is offline. The HTTP layer is a fake object with a ``post`` method, no
credential store is consulted, no key is stored, and nothing here can reach
``api.deepseek.com``. A live smoke is a separate, explicitly authorised round.

Three properties carry the weight:

* **the wire never carries a retired alias** - four choices survive for the
  person, two ids exist for the vendor, and the translation happens in one place;
* **no silent fallback** - a DeepSeek failure fails the stage, and never
  constructs a Claude client or steps down to a cheaper model;
* **the reviewer is untouchable from here** - generation may move to another
  vendor; the judgement that disagrees with it may not follow.
"""

from __future__ import annotations

import ast
import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from goldpipeline.adapters.deepseek_client import (
    DEEPSEEK_PROVIDER,
    MAX_ATTEMPTS_CEILING,
    build_payload,
    schema_appendix,
)
from goldpipeline.adapters.deepseek_finalizer import build_deepseek_finalizer
from goldpipeline.adapters.deepseek_writer import build_deepseek_writer
from goldpipeline.adapters.finalizer_client import FinalizeRequest
from goldpipeline.adapters.writer_client import WriterRequest
from goldpipeline.config import DeepSeekSettings
from goldpipeline.domain.errors import (
    FinalizeConfigurationError,
    FinalizeResponseError,
    WriterConfigurationError,
    WriterProviderError,
    WriterResponseError,
    WriterTimeoutError,
)
from goldpipeline.schemas.finalizer import FinalizerPrompt
from goldpipeline.schemas.preferences import (
    Provider,
    ThinkingMode,
    provider_spec,
    resolve_model,
)
from goldpipeline.schemas.writer import WriterModelOutput, WriterPrompt
from goldpipeline.services.generation import build_finalizer_client, build_writer_client

FAKE_KEY = "ds-fake-key-never-real"  # noqa: S105 - fixture value, not a credential

VENDOR_MODELS = frozenset({"deepseek-v4-pro", "deepseek-v4-flash"})
"""The only ids the vendor still accepts."""

RETIRED_ALIASES = ("deepseek-chat", "deepseek-reasoner")
"""Selections a person still sees. Retired on the wire on 2026-07-24."""


def settings(**overrides: Any) -> DeepSeekSettings:
    return DeepSeekSettings(api_key=FAKE_KEY, **overrides)


# --------------------------------------------------------------------------
# doubles
# --------------------------------------------------------------------------


@dataclass
class FakeResponse:
    """Just enough of an httpx2 response for the client to read."""

    status_code: int = 200
    payload: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes | None = None

    @property
    def content(self) -> bytes:
        if self.body is not None:
            return self.body
        return json.dumps(self.payload).encode("utf-8")

    def json(self) -> Any:
        if self.payload is None:
            raise ValueError("not json")
        return self.payload


@dataclass
class FakeTransport:
    """Records every request and answers from a scripted queue."""

    responses: list[Any] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> Any:
        self.urls.append(url)
        self.headers.append(headers)
        self.requests.append(json)
        answer = self.responses.pop(0) if self.responses else FakeResponse()
        if isinstance(answer, Exception):
            raise answer
        return answer

    @property
    def calls(self) -> int:
        return len(self.requests)


def completion(content: str, **extra: Any) -> FakeResponse:
    """A well-formed vendor response carrying *content*."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    message.update(extra.pop("message", {}))
    return FakeResponse(
        payload={
            "id": "chatcmpl-fake",
            "model": extra.pop("model", "deepseek-v4-flash"),
            "choices": [
                {"index": 0, "message": message, "finish_reason": extra.pop("finish", "stop")}
            ],
            "usage": extra.pop("usage", {"prompt_tokens": 1200, "completion_tokens": 800}),
        }
    )


def writer_output(**overrides: Any) -> str:
    """A schema-valid writer answer, as JSON text."""
    document: dict[str, Any] = {
        "run_id": "20260903_090000_abcdef",
        "status": "COMPLETED",
        "title": "Nhận định vàng",
        "article": "Giá gần nhất quanh 3314.20. Thị trường đang cân bằng.",
        "source_claims": [
            {"type": "PRICE", "value": "3314.20", "source": "context.price.latest_close"}
        ],
        "news_claims": [],
        "warnings": [],
    }
    document.update(overrides)
    return json.dumps(document, ensure_ascii=False)


def finalizer_output(**overrides: Any) -> str:
    document: dict[str, Any] = {
        "run_id": "20260903_090000_abcdef",
        "article": "Giá gần nhất quanh 3314.20. Bản sửa.",
        "issue_resolutions": [],
        "warnings": [],
    }
    document.update(overrides)
    return json.dumps(document, ensure_ascii=False)


def prompt() -> WriterPrompt:
    return WriterPrompt(system="RULES", user="DATA", prompt_version="gold_writer_v3", nonce="n")


def writer_for(selection: str, transport: FakeTransport, **kw: Any) -> Any:
    return build_deepseek_writer(selection, settings=settings(**kw), transport=transport)


def generate(selection: str, transport: FakeTransport, **kw: Any) -> Any:
    return writer_for(selection, transport, **kw).generate(
        WriterRequest(prompt=prompt(), run_id="20260903_090000_abcdef")
    )


# --------------------------------------------------------------------------
# catalog and vendor mapping
# --------------------------------------------------------------------------


def test_the_four_choices_survive() -> None:
    spec = provider_spec(Provider.DEEPSEEK)
    assert [m.selection_id for m in spec.models] == [
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-chat",
        "deepseek-reasoner",
    ]
    assert [m.label for m in spec.models] == [
        "DeepSeek-V4 Pro",
        "DeepSeek-V4 Flash",
        "DeepSeek Chat",
        "DeepSeek Reasoner",
    ]


@pytest.mark.parametrize(
    ("selection", "api_model", "thinking"),
    [
        ("deepseek-v4-pro", "deepseek-v4-pro", ThinkingMode.ENABLED),
        ("deepseek-v4-flash", "deepseek-v4-flash", ThinkingMode.ENABLED),
        ("deepseek-chat", "deepseek-v4-flash", ThinkingMode.DISABLED),
        ("deepseek-reasoner", "deepseek-v4-flash", ThinkingMode.ENABLED),
    ],
)
def test_the_runtime_mapping(selection: str, api_model: str, thinking: ThinkingMode) -> None:
    """The whole point of separating selection from vendor id."""
    model = resolve_model(Provider.DEEPSEEK, selection)
    assert model.api_model_id == api_model
    assert model.thinking is thinking


def test_only_two_vendor_ids_are_used() -> None:
    spec = provider_spec(Provider.DEEPSEEK)
    assert {m.api_model_id for m in spec.models} == VENDOR_MODELS


@pytest.mark.parametrize("selection", ["deepseek-v4-pro", "deepseek-v4-flash", *RETIRED_ALIASES])
def test_a_retired_alias_never_reaches_the_wire(selection: str) -> None:
    """The rule this round exists to enforce."""
    transport = FakeTransport([completion(writer_output())])
    generate(selection, transport)

    sent = transport.requests[0]["model"]
    assert sent in VENDOR_MODELS
    assert sent not in RETIRED_ALIASES


def test_flash_and_reasoner_are_honestly_the_same_runtime() -> None:
    """A collision the vendor created, not disguised here."""
    flash = resolve_model(Provider.DEEPSEEK, "deepseek-v4-flash")
    reasoner = resolve_model(Provider.DEEPSEEK, "deepseek-reasoner")
    assert (flash.api_model_id, flash.thinking) == (reasoner.api_model_id, reasoner.thinking)
    assert flash.selection_id != reasoner.selection_id


# --------------------------------------------------------------------------
# thinking mode
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ("deepseek-v4-pro", "enabled"),
        ("deepseek-v4-flash", "enabled"),
        ("deepseek-reasoner", "enabled"),
        ("deepseek-chat", "disabled"),
    ],
)
def test_thinking_is_always_explicit(selection: str, expected: str) -> None:
    """Never left to a vendor default: Chat promises a non-thinking answer."""
    payload = build_payload(
        resolve_model(Provider.DEEPSEEK, selection), system="s", user="u", max_tokens=100
    )
    assert payload["thinking"] == {"type": expected}


@pytest.mark.parametrize(
    "control", ["temperature", "top_p", "presence_penalty", "frequency_penalty"]
)
def test_no_sampling_control_is_sent(control: str) -> None:
    """The vendor documents these as ineffective in thinking mode.

    A field that changes nothing is worse than a missing one: somebody
    eventually tunes it and cannot work out why the output does not move.
    """
    for selection in ("deepseek-v4-pro", "deepseek-chat"):
        payload = build_payload(
            resolve_model(Provider.DEEPSEEK, selection), system="s", user="u", max_tokens=100
        )
        assert control not in payload


def test_claude_selections_send_no_thinking_field() -> None:
    payload = build_payload(
        resolve_model(Provider.CLAUDE, "claude-opus-5"), system="s", user="u", max_tokens=10
    )
    assert "thinking" not in payload


def test_json_output_is_requested() -> None:
    payload = build_payload(
        resolve_model(Provider.DEEPSEEK, "deepseek-v4-pro"), system="s", user="u", max_tokens=10
    )
    assert payload["response_format"] == {"type": "json_object"}


def test_the_schema_appendix_names_the_contract() -> None:
    """JSON mode guarantees valid JSON, not the right JSON."""
    appendix = schema_appendix(WriterModelOutput)
    assert "# OUTPUT SCHEMA" in appendix
    assert "news_claims" in appendix
    assert "source_claims" in appendix


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------


def test_the_request_goes_to_the_documented_endpoint() -> None:
    transport = FakeTransport([completion(writer_output())])
    generate("deepseek-v4-pro", transport)

    assert transport.urls == ["https://api.deepseek.com/chat/completions"]
    assert transport.headers[0]["Content-Type"] == "application/json"
    assert transport.headers[0]["Authorization"] == f"Bearer {FAKE_KEY}"


def test_an_http_base_url_is_refused() -> None:
    """A bearer token must never go on the wire in clear."""
    with pytest.raises(WriterConfigurationError, match="https"):
        DeepSeekSettings.from_env(
            {"DEEPSEEK_API_KEY": FAKE_KEY, "DEEPSEEK_BASE_URL": "http://api.deepseek.com"}
        )


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_a_redirect_is_refused_rather_than_followed(status: int) -> None:
    """Following it would re-send the Authorization header to another origin."""
    transport = FakeTransport([FakeResponse(status_code=status, payload={})])
    with pytest.raises(WriterProviderError, match="redirect"):
        generate("deepseek-v4-pro", transport)
    assert transport.calls == 1


def test_a_timeout_is_reported_as_a_timeout() -> None:
    import httpx2

    transport = FakeTransport([httpx2.TimeoutException("slow")])
    with pytest.raises(WriterTimeoutError):
        generate("deepseek-v4-pro", transport)


def test_a_connection_failure_is_a_provider_error() -> None:
    import httpx2

    transport = FakeTransport([httpx2.ConnectError("no route")] * MAX_ATTEMPTS_CEILING)
    with pytest.raises(WriterProviderError, match="could not reach"):
        generate("deepseek-v4-pro", transport)


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_credential_is_a_configuration_error(status: int) -> None:
    transport = FakeTransport([FakeResponse(status_code=status, payload={})])
    with pytest.raises(WriterConfigurationError) as caught:
        generate("deepseek-v4-pro", transport)

    assert "DEEPSEEK_API_KEY" in caught.value.message
    assert FAKE_KEY not in caught.value.message
    assert transport.calls == 1, "a bad credential must not be retried"


@pytest.mark.parametrize("status", [400, 404, 422])
def test_a_deterministic_rejection_is_not_retried(status: int) -> None:
    transport = FakeTransport([FakeResponse(status_code=status, payload={})])
    with pytest.raises(WriterConfigurationError, match="not retried"):
        generate("deepseek-v4-pro", transport)
    assert transport.calls == 1


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_a_transient_status_is_retried_and_then_reported(status: int) -> None:
    transport = FakeTransport([FakeResponse(status_code=status, payload={})] * 8)
    with pytest.raises(WriterProviderError):
        generate("deepseek-v4-pro", transport, max_retries=2)
    assert transport.calls == 3


def test_retries_are_bounded_however_high_the_setting() -> None:
    """Each attempt is a billable generation."""
    transport = FakeTransport([FakeResponse(status_code=503, payload={})] * 50)
    with pytest.raises(WriterProviderError):
        generate("deepseek-v4-pro", transport, max_retries=99)
    assert transport.calls == MAX_ATTEMPTS_CEILING


def test_a_retry_succeeds_when_the_second_attempt_works() -> None:
    transport = FakeTransport(
        [FakeResponse(status_code=503, payload={}), completion(writer_output())]
    )
    response = generate("deepseek-v4-pro", transport, max_retries=2)
    assert response.output.status == "COMPLETED"
    assert transport.calls == 2


def test_a_long_retry_after_is_not_honoured() -> None:
    """This process does not sleep through a window it does not have."""
    from goldpipeline.adapters.deepseek_client import _retry_after

    assert _retry_after(FakeResponse(headers={"Retry-After": "5"})) == 5.0
    assert _retry_after(FakeResponse(headers={"Retry-After": "600"})) is None
    assert _retry_after(FakeResponse(headers={"Retry-After": "soon"})) is None
    assert _retry_after(FakeResponse()) is None


def test_an_oversized_response_is_refused() -> None:
    transport = FakeTransport([FakeResponse(payload={"choices": []}, body=b"x" * 2048)])
    with pytest.raises(WriterResponseError, match="size cap"):
        generate("deepseek-v4-pro", transport, max_response_bytes=1024)


def test_a_non_json_response_is_refused() -> None:
    transport = FakeTransport([FakeResponse(payload=None)])
    with pytest.raises(WriterResponseError, match="not valid JSON"):
        generate("deepseek-v4-pro", transport)


def test_a_response_with_no_choices_is_refused() -> None:
    transport = FakeTransport([FakeResponse(payload={"choices": []})])
    with pytest.raises(WriterResponseError, match="no choices"):
        generate("deepseek-v4-pro", transport)


# --------------------------------------------------------------------------
# empty and reasoning-only answers
# --------------------------------------------------------------------------


def test_empty_content_is_refused() -> None:
    transport = FakeTransport([completion("   ")])
    with pytest.raises(WriterResponseError, match="no visible content"):
        generate("deepseek-v4-pro", transport)


def test_reasoning_is_never_used_as_the_answer() -> None:
    """The failure GoldPlan hit: a thinking response that spends it all thinking.

    Reasoning is not an answer - it never satisfied the contract and was never
    validated - so it is refused, and the refusal says why without quoting it.
    """
    transport = FakeTransport(
        [
            completion(
                "",
                message={"reasoning_content": "Let me think about gold... " + writer_output()},
                finish="length",
            )
        ]
    )
    with pytest.raises(WriterResponseError) as caught:
        generate("deepseek-reasoner", transport)

    assert "reasoning" in caught.value.message
    assert "Let me think" not in caught.value.message


def test_a_length_stop_with_content_is_still_refused() -> None:
    """Text that parses but reads as an unfinished thought is worse than a failure."""
    transport = FakeTransport([completion(writer_output(), finish="length")])
    with pytest.raises(WriterResponseError, match="token limit"):
        generate("deepseek-v4-pro", transport)


def test_reasoning_content_never_reaches_the_response() -> None:
    transport = FakeTransport(
        [completion(writer_output(), message={"reasoning_content": "secret thoughts"})]
    )
    response = generate("deepseek-v4-pro", transport)
    assert "secret thoughts" not in json.dumps(response.output.model_dump(mode="json"))
    assert "secret thoughts" not in json.dumps(response.usage.model_dump(mode="json"))


# --------------------------------------------------------------------------
# structured output and news claims
# --------------------------------------------------------------------------


def test_a_valid_answer_becomes_a_writer_output() -> None:
    transport = FakeTransport([completion(writer_output())])
    response = generate("deepseek-v4-pro", transport)

    assert isinstance(response.output, WriterModelOutput)
    assert response.provider == DEEPSEEK_PROVIDER
    assert response.output.source_claims[0].source == "context.price.latest_close"


def test_news_claims_survive_the_deepseek_path() -> None:
    """The provenance verifier does not care which model wrote them."""
    claims = [
        {
            "statement": "Fed vừa công bố giữ nguyên lãi suất",
            "evidence": "Fed vua cong bo giu nguyen lai suat",
            "news_item_ids": ["tintucvnws:11"],
        }
    ]
    transport = FakeTransport([completion(writer_output(news_claims=claims))])
    response = generate("deepseek-v4-pro", transport)

    assert len(response.output.news_claims) == 1
    assert response.output.news_claims[0].news_item_ids == ["tintucvnws:11"]


def test_the_same_schema_governs_both_providers() -> None:
    """No reduced DeepSeek-specific writer contract exists."""
    from goldpipeline.adapters.anthropic_writer import AnthropicWriterClient

    source = Path("src/goldpipeline/adapters/deepseek_writer.py").read_text(encoding="utf-8")
    assert "WriterModelOutput" in source
    assert AnthropicWriterClient  # the two adapters, one schema


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "{",
        "[]",
        '{"run_id": "x"}',
        '{"run_id": "20260903_090000_abcdef", "status": "MADE_UP", "title": "t", "article": "a"}',
        json.dumps({"run_id": "20260903_090000_abcdef", "status": "COMPLETED", "title": "t"}),
    ],
)
def test_malformed_or_schema_invalid_output_is_refused(content: str) -> None:
    transport = FakeTransport([completion(content)])
    with pytest.raises(WriterResponseError):
        generate("deepseek-v4-pro", transport)


def test_a_schema_invalid_answer_is_not_retried() -> None:
    """The model answered. It answered wrongly, and it will again."""
    transport = FakeTransport([completion("not json")] * 5)
    with pytest.raises(WriterResponseError):
        generate("deepseek-v4-pro", transport, max_retries=3)
    assert transport.calls == 1


# --------------------------------------------------------------------------
# the finalizer
# --------------------------------------------------------------------------


def finalize(selection: str, transport: FakeTransport) -> Any:
    client = build_deepseek_finalizer(selection, settings=settings(), transport=transport)
    return client.finalize(
        FinalizeRequest(
            prompt=FinalizerPrompt(
                system="RULES", user="DATA", prompt_version="gold_finalizer_v1", nonce="n"
            ),
            run_id="20260903_090000_abcdef",
        )
    )


def test_the_finalizer_returns_a_revision() -> None:
    transport = FakeTransport([completion(finalizer_output())])
    response = finalize("deepseek-v4-pro", transport)

    assert response.provider == DEEPSEEK_PROVIDER
    assert response.output.article.endswith("Bản sửa.")
    assert transport.requests[0]["model"] == "deepseek-v4-pro"


def test_the_finalizer_uses_the_selected_runtime() -> None:
    transport = FakeTransport([completion(finalizer_output())])
    finalize("deepseek-chat", transport)

    assert transport.requests[0]["model"] == "deepseek-v4-flash"
    assert transport.requests[0]["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize("content", ["not json", "", '{"article": ""}'])
def test_finalizer_bad_output_is_refused(content: str) -> None:
    transport = FakeTransport([completion(content)])
    with pytest.raises(FinalizeResponseError):
        finalize("deepseek-v4-pro", transport)


def test_a_missing_key_fails_the_finalizer_with_its_own_error() -> None:
    """Stage errors stay distinguishable, so a caller knows what died."""
    with pytest.raises(FinalizeConfigurationError, match="DEEPSEEK_API_KEY"):
        build_deepseek_finalizer("deepseek-v4-pro", settings=None)


# --------------------------------------------------------------------------
# secrets
# --------------------------------------------------------------------------


def test_a_missing_key_is_an_explicit_configuration_error() -> None:
    with pytest.raises(WriterConfigurationError) as caught:
        DeepSeekSettings.from_env({})
    assert caught.value.details["setting"] == "DEEPSEEK_API_KEY"


def test_the_key_never_appears_in_a_repr() -> None:
    assert FAKE_KEY not in repr(settings())
    assert FAKE_KEY not in str(settings())
    assert "redacted" in repr(settings())


def test_the_key_is_forbidden_from_persistent_config() -> None:
    from goldpipeline.schemas.runtime_config import FORBIDDEN_KEYS, ConfigKey

    assert "DEEPSEEK_API_KEY" in FORBIDDEN_KEYS
    assert "DEEPSEEK_API_KEY" not in {key.value for key in ConfigKey}


def test_deepseek_is_not_a_required_secret() -> None:
    """An operator who never picks DeepSeek must never be asked for a key."""
    from goldpipeline.schemas.secrets import CONDITIONAL_SECRETS, REQUIRED_SECRETS, SecretName

    assert SecretName.DEEPSEEK_API_KEY not in REQUIRED_SECRETS
    assert SecretName.DEEPSEEK_API_KEY in CONDITIONAL_SECRETS


def test_building_a_claude_client_reads_no_deepseek_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lazy-resolution promise, exercised rather than asserted in prose."""
    import goldpipeline.config as config_module

    asked: list[str] = []
    original = config_module._secret

    def spy(name: Any, secrets: Any, env: Any) -> Any:
        asked.append(str(name))
        return "fake-anthropic-key" if "ANTHROPIC" in str(name) else None

    monkeypatch.setattr(config_module, "_secret", spy)
    with contextlib.suppress(Exception):
        # The SDK client is never built here; what matters is which credential
        # was reached for on the way, and that is recorded before it fails.
        build_writer_client(Provider.CLAUDE, "claude-opus-5")

    assert "SecretName.DEEPSEEK_API_KEY" not in asked
    assert original is not spy


def test_an_unknown_selection_costs_no_credential_lookup() -> None:
    """Validated against the catalog before anything is constructed."""
    with pytest.raises(ValueError, match="does not offer"):
        build_writer_client(Provider.DEEPSEEK, "deepseek-chat-turbo")


# --------------------------------------------------------------------------
# the factory seam
# --------------------------------------------------------------------------


@pytest.fixture
def fake_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key in the environment, so the real lazy path runs without a vault."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)


def test_the_factory_builds_a_deepseek_writer(fake_key: None) -> None:
    client = build_writer_client(Provider.DEEPSEEK, "deepseek-v4-pro", transport=FakeTransport())
    assert client.provider == DEEPSEEK_PROVIDER
    assert client.model == "deepseek-v4-pro"


def test_the_factory_builds_a_deepseek_finalizer(fake_key: None) -> None:
    client = build_finalizer_client(
        Provider.DEEPSEEK, "deepseek-reasoner", transport=FakeTransport()
    )
    assert client.provider == DEEPSEEK_PROVIDER
    assert client.model == "deepseek-v4-flash"


def test_the_factory_offers_no_reviewer() -> None:
    """The safety invariant: generation may move vendor, the judgement may not."""
    import goldpipeline.services.generation as module

    assert not [name for name in dir(module) if "review" in name.lower()]

    source = Path("src/goldpipeline/services/generation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any("review" in name for name in modules)


def test_no_reviewer_field_exists_in_preferences() -> None:
    from goldpipeline.schemas.preferences import PreferencesStatus, UserPreferences

    for model in (UserPreferences, PreferencesStatus):
        assert not any("review" in name for name in model.model_fields)


def test_the_reviewer_still_reads_its_own_anthropic_setting() -> None:
    from goldpipeline.config import DEFAULT_REVIEWER_MODEL, REVIEWER_MODEL_ENV, ReviewerSettings

    resolved = ReviewerSettings.from_env({"ANTHROPIC_API_KEY": "fake-anthropic-key"})
    assert resolved.model == DEFAULT_REVIEWER_MODEL
    assert REVIEWER_MODEL_ENV == "ANTHROPIC_REVIEWER_MODEL"


def test_deepseek_cannot_be_a_reviewer() -> None:
    """No DeepSeek reviewer adapter exists, and none is reachable."""
    assert not Path("src/goldpipeline/adapters/deepseek_reviewer.py").exists()

    from goldpipeline.adapters import anthropic_reviewer

    assert anthropic_reviewer.ANTHROPIC_PROVIDER == "anthropic"


# --------------------------------------------------------------------------
# no silent fallback
# --------------------------------------------------------------------------


DEEPSEEK_SOURCES = (
    Path("src/goldpipeline/adapters/deepseek_client.py"),
    Path("src/goldpipeline/adapters/deepseek_writer.py"),
    Path("src/goldpipeline/adapters/deepseek_finalizer.py"),
)


@pytest.mark.parametrize("path", DEEPSEEK_SOURCES)
def test_no_deepseek_module_can_reach_anthropic(path: Path) -> None:
    """A grep would be fooled by a docstring; this reads the imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    modules |= {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any("anthropic" in name for name in modules), sorted(modules)


@pytest.mark.parametrize(
    "failure",
    [
        FakeResponse(status_code=401, payload={}),
        FakeResponse(status_code=500, payload={}),
        FakeResponse(payload={"choices": []}),
        completion("not json"),
        completion(""),
    ],
)
def test_a_writer_failure_never_produces_a_draft(failure: Any) -> None:
    """Whatever goes wrong, nothing comes back - least of all from Claude."""
    transport = FakeTransport([failure] * MAX_ATTEMPTS_CEILING)
    with pytest.raises(Exception) as caught:  # noqa: PT011 - four classes, all acceptable
        generate("deepseek-v4-pro", transport)
    assert "anthropic" not in str(caught.value).lower()
    assert "claude" not in str(caught.value).lower()


def test_a_pro_failure_never_falls_back_to_flash() -> None:
    """One selection means one runtime behaviour."""
    transport = FakeTransport([FakeResponse(status_code=503, payload={})] * 8)
    with pytest.raises(WriterProviderError):
        generate("deepseek-v4-pro", transport, max_retries=2)

    assert {request["model"] for request in transport.requests} == {"deepseek-v4-pro"}


def test_a_chat_failure_never_turns_thinking_back_on() -> None:
    transport = FakeTransport([FakeResponse(status_code=503, payload={})] * 8)
    with pytest.raises(WriterProviderError):
        generate("deepseek-chat", transport, max_retries=2)

    assert all(r["thinking"] == {"type": "disabled"} for r in transport.requests)


# --------------------------------------------------------------------------
# artifacts and usage
# --------------------------------------------------------------------------


def test_the_response_records_provider_selection_and_vendor_model() -> None:
    transport = FakeTransport([completion(writer_output(), model="deepseek-v4-flash")])
    response = generate("deepseek-reasoner", transport)

    assert response.provider == "deepseek"
    assert response.model == "deepseek-v4-flash"
    assert transport.requests[0]["model"] == "deepseek-v4-flash"


def test_usage_is_carried_across() -> None:
    transport = FakeTransport(
        [
            completion(
                writer_output(),
                usage={
                    "prompt_tokens": 4321,
                    "completion_tokens": 765,
                    "prompt_tokens_details": {"cached_tokens": 100},
                },
            )
        ]
    )
    usage = generate("deepseek-v4-pro", transport).usage

    assert usage.input_tokens == 4321
    assert usage.output_tokens == 765
    assert usage.cache_read_input_tokens == 100
    assert usage.cache_creation_input_tokens is None, "not guessed to fill a field"
    assert usage.stop_reason == "stop"


def test_missing_usage_is_absent_rather_than_zero() -> None:
    transport = FakeTransport([completion(writer_output(), usage={})])
    usage = generate("deepseek-v4-pro", transport).usage
    assert usage.input_tokens is None
    assert usage.output_tokens is None


def test_the_writer_artifact_can_record_a_selection() -> None:
    from goldpipeline.schemas.writer import WriterResult

    result = WriterResult(
        run_id="20260903_090000_abcdef",
        status="COMPLETED",
        title="t",
        model="deepseek-v4-flash",
        provider="deepseek",
        selection_id="deepseek-reasoner",
        prompt_version="gold_writer_v3",
        context_sha256="a" * 64,
        draft_file="claude_draft.md",
        article_sha256="b" * 64,
        article_chars=42,
    )
    assert (result.provider, result.model, result.selection_id) == (
        "deepseek",
        "deepseek-v4-flash",
        "deepseek-reasoner",
    )


def test_an_artifact_without_a_selection_still_loads() -> None:
    """Every Run written before this round."""
    from goldpipeline.schemas.writer import WriterResult

    document = {
        "run_id": "20260828_182908_107496",
        "status": "COMPLETED",
        "title": "t",
        "model": "claude-opus-5",
        "provider": "anthropic",
        "prompt_version": "gold_writer_v2",
        "context_sha256": "a" * 64,
        "draft_file": "claude_draft.md",
        "article_sha256": "b" * 64,
        "article_chars": 42,
    }
    assert WriterResult.model_validate(document).selection_id is None
