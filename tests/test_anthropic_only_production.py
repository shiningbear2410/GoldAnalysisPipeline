"""Round 9.3.1: one AI vendor, and deterministic failures that stop failing twice.

Two corrections, and they arrived together for a reason.

**The classification defect, paid for in production.** A real scheduled Run met
``HTTP 400 invalid_request_error`` from Anthropic - an identity-linked API key
that required a workspace header the pipeline never sent. The SDK mapping had a
catch-all that turned every ``APIStatusError`` into a *provider* error, which the
automation layer classifies as transient, so it retried at one, two and five
minutes. Three real requests to discover the same certainty three times. A 4xx
describes the request; waiting changes nothing about it.

**The architecture correction.** The Reviewer ran on OpenAI, so production
demanded a second vendor's credential for a pipeline whose owner only has an
Anthropic account. Writer, Reviewer and Finalizer now all call Anthropic.

The thing worth guarding after that second change is the *independence* of the
review. Sharing a vendor is not sharing a judgement: the Reviewer still receives
the immutable context, the finished draft and the writer's metadata, and answers
a different prompt with a different schema in a separate request. A model asked
to critique its own answer inside one call does not reliably disagree with
itself, which is the whole reason this stage exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anthropic
import httpx2 as httpx
import pytest
from conftest import FAKE_API_KEY, FAKE_OPENAI_KEY, TELEGRAM_TOKEN_SENTINEL, make_event_payload

from goldpipeline.adapters.anthropic_errors import (
    RETRYABLE_STATUS_CODES,
    is_deterministic_status,
)
from goldpipeline.adapters.anthropic_reviewer import (
    ANTHROPIC_PROVIDER,
    AnthropicReviewerClient,
)
from goldpipeline.adapters.reviewer_client import ReviewerClient, ReviewRequest
from goldpipeline.cli import EXIT_OK, main
from goldpipeline.config import DEFAULT_REVIEWER_MODEL, REVIEWER_MODEL_ENV, ReviewerSettings
from goldpipeline.domain.errors import (
    ReviewConfigurationError,
    ReviewProviderError,
    ReviewTimeoutError,
)
from goldpipeline.schemas.review import (
    Evidence,
    IssueCategory,
    ReviewerPrompt,
    ReviewIssue,
    ReviewModelOutput,
    ReviewStatus,
    Severity,
)
from goldpipeline.schemas.secrets import OPTIONAL_SECRETS, REQUIRED_SECRETS, SecretName
from goldpipeline.services.automation import classify

RUN_ID = "20260828_022701_a83f2c"


def invoke(args: list[str]) -> int:
    return main(args)


# --- scaffolding ----------------------------------------------------------


def settings(**overrides: Any) -> ReviewerSettings:
    base: dict[str, Any] = {"api_key": FAKE_API_KEY, "model": DEFAULT_REVIEWER_MODEL}
    base.update(overrides)
    return ReviewerSettings(**base)


def request() -> ReviewRequest:
    prompt = ReviewerPrompt(
        system="# REVIEWER RULES\n# OUTPUT CONTRACT",
        user="# SOURCE OF TRUTH\n# WRITER METADATA\n# ARTICLE UNDER REVIEW",
        prompt_version="gold_reviewer_v1",
        nonce="feedfacefeedface",
    )
    return ReviewRequest(prompt=prompt, run_id=RUN_ID, max_output_tokens=4096)


def valid_output(status: ReviewStatus = ReviewStatus.PASS, **overrides: Any) -> ReviewModelOutput:
    base: dict[str, Any] = {
        "run_id": RUN_ID,
        "status": status,
        "score": 88,
        "summary": "Bài viết bám sát dữ liệu được cung cấp.",
        "issues": [],
        "revision_instructions": [],
    }
    base.update(overrides)
    return ReviewModelOutput(**base)


class StubMessages:
    def __init__(self, *, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class StubClient:
    def __init__(self, **kwargs: Any) -> None:
        self.messages = StubMessages(**kwargs)


def sdk_response(parsed: Any, *, stop_reason: str = "end_turn") -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        parsed_output=parsed,
        stop_reason=stop_reason,
        model=DEFAULT_REVIEWER_MODEL,
        usage=SimpleNamespace(input_tokens=1200, output_tokens=300),
        _request_id="req_stub789",
    )


def http_error(status: int) -> anthropic.APIStatusError:
    return anthropic.APIStatusError(
        "boom-should-never-be-echoed",
        response=httpx.Response(
            status_code=status, request=httpx.Request("POST", "https://api.anthropic.com/v1/x")
        ),
        body=None,
    )


# --- the production reviewer is Anthropic ---------------------------------


def test_the_production_factory_builds_an_anthropic_reviewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirement 12.1."""
    from goldpipeline import cli

    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = cli._reviewer_client(fake=False, model=None)

    assert isinstance(client, AnthropicReviewerClient)
    assert client.provider == ANTHROPIC_PROVIDER
    assert isinstance(client, ReviewerClient), "the orchestrator still sees only the protocol"


def test_the_reviewer_issues_its_own_independent_request() -> None:
    """Requirement 12.2, and the property that matters most after the switch.

    Same vendor, separate call: the reviewer's own prompt, its own schema, its
    own token budget. Nothing here is a continuation of the writer's request.
    """
    stub = StubClient(response=sdk_response(valid_output()))
    AnthropicReviewerClient(settings(), client=stub).review(request())

    (call,) = stub.messages.calls
    assert call["output_format"] is ReviewModelOutput, "the reviewer's schema, not the writer's"
    assert call["system"] == "# REVIEWER RULES\n# OUTPUT CONTRACT"
    assert call["max_tokens"] == 4096
    assert len(call["messages"]) == 1, "one user turn; no writer conversation is replayed"
    assert "ARTICLE UNDER REVIEW" in call["messages"][0]["content"]


def test_the_structured_output_contract_is_unchanged() -> None:
    """Requirement 12.3."""
    output = valid_output()
    stub = StubClient(response=sdk_response(output))

    response = AnthropicReviewerClient(settings(), client=stub).review(request())

    assert response.output is output
    assert response.provider == ANTHROPIC_PROVIDER
    assert response.model == DEFAULT_REVIEWER_MODEL
    assert response.usage.input_tokens == 1200
    assert response.usage.output_tokens == 300
    assert response.usage.total_tokens == 1500


@pytest.mark.parametrize(
    ("status", "issues", "instructions"),
    [
        (ReviewStatus.PASS, [], []),
        (
            ReviewStatus.NEEDS_REVISION,
            [
                ReviewIssue(
                    issue_id="i1",
                    category=IssueCategory.DATA_MISMATCH,
                    severity=Severity.MEDIUM,
                    message="Giá 9999 không có trong dữ liệu.",
                    claim="Giá chạm 9999.",
                    evidence=Evidence(
                        source_path="context.price.latest_close",
                        expected="4435.026",
                        actual="9999",
                    ),
                )
            ],
            ["Bỏ mức giá 9999."],
        ),
        (
            ReviewStatus.REJECT,
            [
                ReviewIssue(
                    issue_id="i1",
                    category=IssueCategory.UNSUPPORTED_CLAIM,
                    severity=Severity.HIGH,
                    message="Bài viết nhắc tin tức không có trong nguồn.",
                    claim="Fed sẽ hạ lãi suất.",
                    evidence=Evidence(
                        source_path="context.raw_analysis",
                        expected="(không có tin tức)",
                        actual="Fed sẽ hạ lãi suất.",
                    ),
                )
            ],
            [],
        ),
    ],
)
def test_every_verdict_survives_the_vendor_change(
    status: ReviewStatus, issues: list[ReviewIssue], instructions: list[str]
) -> None:
    """Requirements 12.4, 12.5 and 12.6.

    The three verdicts are the review contract. Changing which account answers
    must not change what an answer is allowed to be.
    """
    output = valid_output(status=status, issues=issues, revision_instructions=instructions)
    stub = StubClient(response=sdk_response(output))

    response = AnthropicReviewerClient(settings(), client=stub).review(request())

    assert response.output.status is status
    assert response.output.issues == issues
    assert response.output.revision_instructions == instructions


def test_the_reviewer_model_is_configurable_and_separate() -> None:
    """Requirement 4: model names stay non-secret configuration.

    A separate setting from the writer's, so the two stages can be moved
    independently - which is the point of having two stages.
    """
    resolved = ReviewerSettings.from_env(
        {"ANTHROPIC_API_KEY": FAKE_API_KEY, REVIEWER_MODEL_ENV: "claude-sonnet-5"}
    )
    assert resolved.model == "claude-sonnet-5"

    default = ReviewerSettings.from_env({"ANTHROPIC_API_KEY": FAKE_API_KEY})
    assert default.model == DEFAULT_REVIEWER_MODEL


# --- shared, hardened provider classification -----------------------------


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 405, 409, 418, 451])
def test_deterministic_statuses_are_configuration_failures(status: int) -> None:
    """Requirement 12.7, plus the whole of section 7.

    Every 4xx that is not 408 or 429 says "this request is wrong". Retrying it
    reproduces it, which is exactly what happened in Round 9.3.
    """
    assert is_deterministic_status(status)

    stub = StubClient(error=http_error(status))
    with pytest.raises(ReviewConfigurationError) as caught:
        AnthropicReviewerClient(settings(), client=stub).review(request())

    assert caught.value.details["status_code"] == status
    assert "boom-should-never-be-echoed" not in str(caught.value)


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_momentary_statuses_stay_transient(status: int) -> None:
    """Requirements 12.8 and 12.9.

    ``408`` and ``429`` are the server saying "later", and 5xx is its own
    problem. All of them may genuinely succeed on the next attempt, so they keep
    the bounded backoff.
    """
    assert not is_deterministic_status(status)

    stub = StubClient(error=http_error(status))
    with pytest.raises(ReviewProviderError) as caught:
        AnthropicReviewerClient(settings(), client=stub).review(request())
    assert caught.value.details["status_code"] == status


def test_the_retryable_set_is_exactly_the_two_wait_codes() -> None:
    assert frozenset({408, 429}) == RETRYABLE_STATUS_CODES
    assert not is_deterministic_status(500), "5xx is the server's problem, not the request's"


def test_a_deterministic_failure_never_enters_the_retry_loop() -> None:
    """The end-to-end statement of the Round 9.3 defect.

    Classification is what the automation layer acts on, so the assertion that
    matters is not which exception type is raised but which bucket it lands in.
    """
    stub = StubClient(error=http_error(400))
    with pytest.raises(ReviewConfigurationError) as caught:
        AnthropicReviewerClient(settings(), client=stub).review(request())

    assert classify(caught.value).value == "CONFIGURATION"


def test_a_timeout_is_still_a_timeout() -> None:
    stub = StubClient(error=anthropic.APITimeoutError(request=httpx.Request("POST", "https://x")))
    with pytest.raises(ReviewTimeoutError):
        AnthropicReviewerClient(settings(), client=stub).review(request())


def test_a_connection_failure_is_still_transient() -> None:
    stub = StubClient(
        error=anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
    )
    with pytest.raises(ReviewProviderError) as caught:
        AnthropicReviewerClient(settings(), client=stub).review(request())
    assert classify(caught.value).value == "TRANSIENT"


def test_all_three_stages_share_one_mapping() -> None:
    """Requirement 8: no divergent copies.

    Asserted structurally rather than behaviourally - three adapters importing
    the same function cannot drift, and three that merely happen to agree today
    can.
    """
    from goldpipeline.adapters import (
        anthropic_finalizer,
        anthropic_reviewer,
        anthropic_writer,
    )

    for module in (anthropic_writer, anthropic_reviewer, anthropic_finalizer):
        assert module.raise_mapped is not None
        assert module.raise_mapped.__module__ == "goldpipeline.adapters.anthropic_errors"


# --- OpenAI is not a production dependency --------------------------------


def test_openai_is_not_a_required_credential() -> None:
    """Requirements 1 and 5."""
    assert SecretName.ANTHROPIC_API_KEY in REQUIRED_SECRETS
    assert SecretName.OPENAI_API_KEY in OPTIONAL_SECRETS
    assert SecretName.OPENAI_API_KEY not in REQUIRED_SECRETS


def test_an_absent_openai_credential_reads_as_not_required() -> None:
    """Requirement 5: 'unused', not 'missing'. Those are different facts."""
    from goldpipeline.schemas.secrets import SecretStatus

    assert SecretStatus(name=SecretName.OPENAI_API_KEY, configured=False).summary == "not required"
    assert SecretStatus(name=SecretName.ANTHROPIC_API_KEY, configured=False).summary == "missing"


def preflight_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "automation-preflight",
        "--inbox-dir",
        str(tmp_path / "inbox"),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--automation-dir",
        str(tmp_path / "automation"),
        "--fake-mt5",
        *extra,
    ]


def test_preflight_is_ready_without_an_openai_credential(
    tmp_path: Path,
    production_config: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirements 12.11 and 12.12."""
    from conftest import FakeKeyringModule

    from goldpipeline import cli
    from goldpipeline.adapters.windows_credentials import (
        SERVICE_NAME,
        WindowsCredentialSecretProvider,
        inspect_backend,
    )

    module = FakeKeyringModule({(SERVICE_NAME, "anthropic_api_key"): FAKE_API_KEY})
    monkeypatch.setattr(cli, "_credential_store", lambda: WindowsCredentialSecretProvider(module))
    monkeypatch.setattr(cli, "inspect_backend", lambda *a, **k: inspect_backend(module))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    production_config()
    capsys.readouterr()

    code = invoke([*preflight_args(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["task_readiness"] == "READY"
    assert payload["blockers"] == []
    assert payload["openai"] == "not required"
    assert payload["reviewer_provider"] == ANTHROPIC_PROVIDER


def test_a_missing_anthropic_credential_still_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirements 12.13 and 12.14.

    One credential now gates the Writer *and* the Reviewer, so its absence has
    to be reported once and clearly.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    capsys.readouterr()

    invoke([*preflight_args(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["anthropic"] == "missing"
    assert any("Anthropic credential is missing" in b for b in payload["blockers"])
    assert not any("OpenAI" in b or "OPENAI" in b for b in payload["blockers"])


def test_the_reviewer_stage_reports_the_anthropic_setting_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirement 12.14, at the settings boundary."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ReviewConfigurationError) as caught:
        ReviewerSettings.from_env({})

    assert caught.value.details["setting"] == "ANTHROPIC_API_KEY"


# --- the whole pipeline, with no OpenAI anywhere --------------------------


def test_the_pipeline_reaches_ready_to_publish_with_no_openai_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirements 12.10, 12.16 and 12.17.

    The offline clients stand in for all three AI stages, so this proves the
    *wiring*: nothing on the path from event to ``READY_TO_PUBLISH`` consults an
    OpenAI credential, and with auto-publish off nothing reaches a publisher.
    """
    import socket

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the offline path must open no socket")

    monkeypatch.setattr(socket.socket, "connect", refuse)

    from datetime import UTC, datetime

    from conftest import write_json

    payload = make_event_payload(created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"))
    event = write_json(tmp_path / "event.json", payload)
    inbox = tmp_path / "inbox"
    runs = tmp_path / "runs"
    invoke(["inbox-submit", "--file", str(event), "--inbox-dir", str(inbox)])
    capsys.readouterr()

    code = invoke(
        [
            "automation-run-once",
            "--inbox-dir",
            str(inbox),
            "--runs-dir",
            str(runs),
            "--automation-dir",
            str(tmp_path / "automation"),
            "--fake-mt5",
            "--fake-ai",
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert result["auto_publish_enabled"] is False
    assert len(result["processed_events"]) == 1

    (run_dir,) = [p for p in runs.iterdir() if p.is_dir()]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "READY_TO_PUBLISH"
    assert not (run_dir / "publish_intent.json").exists()
    assert not (run_dir / "publish_result.json").exists()


def test_a_gate_resume_needs_no_ai_credential_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement 12.15.

    Unchanged from earlier rounds, and worth re-pinning: the client factories
    are lazy, so a Run resumed at the gate builds no client and reads no key.
    """
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TELEGRAM_BOT_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    from goldpipeline import cli

    def refuse_reviewer(**kwargs: Any) -> None:
        raise AssertionError("a gate-only resume must build no reviewer")

    monkeypatch.setattr(cli, "_reviewer_client", refuse_reviewer)

    # Nothing to resume, but the factories are wired the same way; the assertion
    # is that reaching this point required no credential.
    code = invoke(["list-runs", "--runs-dir", str(tmp_path / "runs")])
    capsys.readouterr()
    assert code == EXIT_OK


def test_no_sentinel_credential_leaks_through_the_reviewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reviewer holds a key; nothing it produces may contain one."""
    stub = StubClient(response=sdk_response(valid_output()))
    client = AnthropicReviewerClient(settings(), client=stub)

    rendered = repr(client._settings) + json.dumps(
        json.loads(client.review(request()).output.model_dump_json())
    )

    for sentinel in (FAKE_API_KEY, FAKE_OPENAI_KEY, TELEGRAM_TOKEN_SENTINEL):
        assert sentinel not in rendered
