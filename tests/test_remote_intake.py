"""Remote event intake: transport, admission and failure isolation.

Every test is offline. There is no HTTP server, no producer, no MT5, no model
and no Telegram anywhere in this file - the transport is a Protocol precisely so
that a list stands in for a network.

The invariant these tests exist to defend: **an at-least-once transport must
produce exactly-once admission.** A producer may offer the same event on every
poll forever, and exactly one Run may result.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from goldpipeline.adapters.event_transport import HttpOutboxTransport, _envelope_events
from goldpipeline.config import IngestSettings
from goldpipeline.domain.errors import (
    RemoteIntakeConfigurationError,
    RemoteIntakeResponseError,
    RemoteIntakeTransportError,
)
from goldpipeline.schemas.common import utc_now
from goldpipeline.schemas.ingestion import LedgerEntry, LedgerState
from goldpipeline.services.event_intake import intake
from goldpipeline.services.inbox import INCOMING, PROCESSED, Inbox, Ledger
from goldpipeline.storage.atomic import encode_json, sha256_bytes

TOKEN = "test-token-never-real"  # noqa: S105 - fixture value, not a credential


def event_payload(event_id: str = "remote_event_00000001", text: str = "phan tich") -> dict:
    return {
        "schema_version": "1",
        "source": "gold_analysis_bot",
        "event_id": event_id,
        "created_at": utc_now().isoformat().replace("+00:00", "Z"),
        "raw_text": text,
    }


class FakeTransport:
    """Returns a fixed batch, and counts how often it was asked."""

    def __init__(self, batches: list[list[dict]] | None = None, *, error: Exception | None = None):
        self._batches = batches or []
        self._error = error
        self.calls = 0

    def fetch_pending(self, *, limit: int) -> list[dict[str, Any]]:
        self.calls += 1
        if self._error is not None:
            raise self._error
        if not self._batches:
            return []
        return self._batches[min(self.calls - 1, len(self._batches) - 1)]


class FakeResponse:
    def __init__(self, status_code: int, body: bytes = b"", headers: dict | None = None):
        self.status_code = status_code
        self.content = body
        self.headers = headers or {}


class FakeHttpClient:
    """Stands in for httpx2.Client. Records the headers it was given."""

    def __init__(self, response: FakeResponse | Exception):
        self._response = response
        self.seen_headers: dict = {}
        self.seen_url = ""

    def get(self, url: str, headers: dict) -> FakeResponse:
        self.seen_url = url
        self.seen_headers = dict(headers)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


@pytest.fixture
def inbox(tmp_path: Path) -> Inbox:
    box = Inbox(tmp_path / "inbox")
    box.ensure_layout()
    return box


@pytest.fixture
def ledger(inbox: Inbox) -> Ledger:
    return Ledger(inbox.directory("index"))


def transport_for(body: dict, status: int = 200, headers: dict | None = None):
    client = FakeHttpClient(FakeResponse(status, json.dumps(body).encode("utf-8"), headers))
    return (
        HttpOutboxTransport(
            base_url="https://producer.example",
            token=TOKEN,
            timeout_seconds=5.0,
            max_bytes=1024 * 1024,
            client=client,
        ),
        client,
    )


# ---------------------------------------------------------------- settings
class TestIngestSettings:
    def test_disabled_by_default(self) -> None:
        settings = IngestSettings.from_env({})
        assert settings.enabled is False
        assert settings.url == ""

    def test_absent_keys_mean_off_in_strict_config(self) -> None:
        """A STRICT_PERSISTENT mapping has no INGEST_* keys at all."""
        strict = {"GOLDPIPELINE_AUTOMATION_ENABLED": "true", "TELEGRAM_TARGET_CHAT_ID": "123"}
        assert IngestSettings.from_env(strict).enabled is False

    def test_enabled_without_url_fails_closed(self) -> None:
        with pytest.raises(RemoteIntakeConfigurationError):
            IngestSettings.from_env({"GOLDPIPELINE_INGEST_ENABLED": "true"})

    def test_plain_http_url_is_refused(self) -> None:
        with pytest.raises(RemoteIntakeConfigurationError):
            IngestSettings.from_env(
                {
                    "GOLDPIPELINE_INGEST_ENABLED": "true",
                    "GOLDPIPELINE_INGEST_URL": "http://producer.example",
                }
            )

    def test_https_url_is_accepted(self) -> None:
        settings = IngestSettings.from_env(
            {
                "GOLDPIPELINE_INGEST_ENABLED": "true",
                "GOLDPIPELINE_INGEST_URL": "https://producer.example",
            }
        )
        assert settings.enabled is True
        assert settings.max_events > 0

    def test_max_events_is_bounded(self) -> None:
        with pytest.raises(RemoteIntakeConfigurationError):
            IngestSettings.from_env({"GOLDPIPELINE_INGEST_MAX_EVENTS": "10000"})

    def test_timeout_is_bounded(self) -> None:
        with pytest.raises(RemoteIntakeConfigurationError):
            IngestSettings.from_env({"GOLDPIPELINE_INGEST_TIMEOUT_SECONDS": "600"})


# --------------------------------------------------------------- transport
class TestHttpTransport:
    def test_sends_bearer_token_and_limit(self) -> None:
        transport, client = transport_for({"events": []})
        transport.fetch_pending(limit=7)
        assert client.seen_headers["Authorization"] == f"Bearer {TOKEN}"
        assert "limit=7" in client.seen_url
        assert client.seen_url.startswith("https://producer.example/outbox/pending")

    def test_empty_events_is_success(self) -> None:
        transport, _ = transport_for({"events": []})
        assert transport.fetch_pending(limit=10) == []

    def test_one_event_returned(self) -> None:
        transport, _ = transport_for({"events": [event_payload()]})
        assert len(transport.fetch_pending(limit=10)) == 1

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_failure_is_configuration(self, status: int) -> None:
        transport, _ = transport_for({}, status=status)
        with pytest.raises(RemoteIntakeConfigurationError):
            transport.fetch_pending(limit=10)

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 408])
    def test_retryable_statuses_are_transport(self, status: int) -> None:
        transport, _ = transport_for({}, status=status)
        with pytest.raises(RemoteIntakeTransportError):
            transport.fetch_pending(limit=10)

    @pytest.mark.parametrize("status", [301, 302, 307, 308])
    def test_redirects_are_refused_not_followed(self, status: int) -> None:
        """A redirect would re-send the Authorization header to another origin."""
        transport, _ = transport_for({}, status=status)
        with pytest.raises(RemoteIntakeResponseError):
            transport.fetch_pending(limit=10)

    def test_other_4xx_is_response_error(self) -> None:
        transport, _ = transport_for({}, status=418)
        with pytest.raises(RemoteIntakeResponseError):
            transport.fetch_pending(limit=10)

    def test_malformed_json_is_refused(self) -> None:
        client = FakeHttpClient(FakeResponse(200, b"{not json"))
        transport = HttpOutboxTransport(
            base_url="https://producer.example",
            token=TOKEN,
            timeout_seconds=5.0,
            max_bytes=1024,
            client=client,
        )
        with pytest.raises(RemoteIntakeResponseError):
            transport.fetch_pending(limit=10)

    def test_oversized_body_is_refused_by_declared_length(self) -> None:
        transport, _ = transport_for({"events": []}, headers={"content-length": "99999999"})
        with pytest.raises(RemoteIntakeResponseError):
            transport.fetch_pending(limit=10)

    def test_oversized_body_is_refused_by_actual_bytes(self) -> None:
        client = FakeHttpClient(FakeResponse(200, b"x" * 5000))
        transport = HttpOutboxTransport(
            base_url="https://producer.example",
            token=TOKEN,
            timeout_seconds=5.0,
            max_bytes=1000,
            client=client,
        )
        with pytest.raises(RemoteIntakeResponseError):
            transport.fetch_pending(limit=10)

    def test_too_many_events_is_refused(self) -> None:
        body = {"events": [event_payload(f"remote_event_{i:08d}") for i in range(5)]}
        transport, _ = transport_for(body)
        with pytest.raises(RemoteIntakeResponseError):
            transport.fetch_pending(limit=2)

    def test_transport_exception_is_scrubbed(self) -> None:
        """The cause must not carry a message that could hold the header."""
        boom = RuntimeError(f"connection to https://user:{TOKEN}@host failed")
        client = FakeHttpClient(boom)
        transport = HttpOutboxTransport(
            base_url="https://producer.example",
            token=TOKEN,
            timeout_seconds=5.0,
            max_bytes=1024,
            client=client,
        )
        with pytest.raises(RemoteIntakeTransportError) as caught:
            transport.fetch_pending(limit=10)
        assert TOKEN not in str(caught.value)
        assert TOKEN not in str(caught.value.__cause__)


class TestEnvelope:
    @pytest.mark.parametrize(
        "body",
        [[], "text", 5, None, {"other": []}, {"events": {}}, {"events": "no"}],
    )
    def test_bad_envelopes_are_refused_whole(self, body: Any) -> None:
        with pytest.raises(RemoteIntakeResponseError):
            _envelope_events(body, limit=10)

    def test_non_object_event_is_refused(self) -> None:
        with pytest.raises(RemoteIntakeResponseError):
            _envelope_events({"events": ["a string"]}, limit=10)


# ------------------------------------------------------------- admission
class TestAdmission:
    def run(self, transport: Any, inbox: Inbox, ledger: Ledger, limit: int = 10):
        return intake(transport=transport, inbox=inbox, ledger=ledger, limit=limit)

    def test_new_event_is_submitted(self, inbox: Inbox, ledger: Ledger) -> None:
        report = self.run(FakeTransport([[event_payload()]]), inbox, ledger)
        assert report.submitted == ["remote_event_00000001"]
        assert (inbox.directory(INCOMING) / "remote_event_00000001.json").is_file()

    def test_multiple_events_all_submitted(self, inbox: Inbox, ledger: Ledger) -> None:
        batch = [event_payload(f"remote_event_{i:08d}") for i in range(3)]
        report = self.run(FakeTransport([batch]), inbox, ledger)
        assert len(report.submitted) == 3

    def test_same_event_offered_repeatedly_is_admitted_once(
        self, inbox: Inbox, ledger: Ledger
    ) -> None:
        """The whole point: at-least-once transport, exactly-once admission."""
        payload = event_payload()
        transport = FakeTransport([[payload]])
        first = self.run(transport, inbox, ledger)
        assert first.submitted == ["remote_event_00000001"]

        for _ in range(30):
            later = self.run(transport, inbox, ledger)
            assert later.submitted == []
            assert later.duplicate == ["remote_event_00000001"]

        files = list((inbox.directory(INCOMING)).glob("*.json"))
        assert len(files) == 1

    def test_duplicate_already_processed_is_skipped(self, inbox: Inbox, ledger: Ledger) -> None:
        payload = event_payload()
        raw = encode_json(payload)
        (inbox.directory(PROCESSED) / "remote_event_00000001.json").write_bytes(raw)

        report = self.run(FakeTransport([[payload]]), inbox, ledger)
        assert report.duplicate == ["remote_event_00000001"]
        assert report.submitted == []
        assert not (inbox.directory(INCOMING) / "remote_event_00000001.json").exists()

    def test_duplicate_in_ledger_with_same_payload_is_skipped(
        self, inbox: Inbox, ledger: Ledger
    ) -> None:
        payload = event_payload()
        ledger.reserve(
            LedgerEntry(
                event_id="remote_event_00000001",
                source="gold_analysis_bot",
                payload_sha256=sha256_bytes(encode_json(payload)),
                run_id="20260902_000000_aaaaaa",
                state=LedgerState.RESERVED,
                reserved_at=utc_now(),
            )
        )
        report = self.run(FakeTransport([[payload]]), inbox, ledger)
        assert report.duplicate == ["remote_event_00000001"]
        assert report.submitted == []

    def test_same_id_different_payload_is_a_conflict(self, inbox: Inbox, ledger: Ledger) -> None:
        original = event_payload(text="ban goc")
        ledger.reserve(
            LedgerEntry(
                event_id="remote_event_00000001",
                source="gold_analysis_bot",
                payload_sha256=sha256_bytes(encode_json(original)),
                run_id="20260902_000000_aaaaaa",
                state=LedgerState.RESERVED,
                reserved_at=utc_now(),
            )
        )
        changed = dict(original)
        changed["raw_text"] = "noi dung khac han"

        report = self.run(FakeTransport([[changed]]), inbox, ledger)
        assert report.submitted == []
        assert len(report.conflicts) == 1
        conflict = report.conflicts[0]
        assert conflict.event_id == "remote_event_00000001"
        assert conflict.known_sha256 != conflict.offered_sha256
        assert conflict.where == "ledger"
        # Nothing was written and nothing was overwritten.
        assert not (inbox.directory(INCOMING) / "remote_event_00000001.json").exists()

    def test_conflict_against_a_waiting_file(self, inbox: Inbox, ledger: Ledger) -> None:
        original = event_payload(text="ban goc")
        inbox.submit(original, event_id="remote_event_00000001")
        before = (inbox.directory(INCOMING) / "remote_event_00000001.json").read_bytes()

        changed = dict(original)
        changed["raw_text"] = "noi dung khac han"
        report = self.run(FakeTransport([[changed]]), inbox, ledger)

        assert len(report.conflicts) == 1
        assert report.conflicts[0].where == INCOMING
        after = (inbox.directory(INCOMING) / "remote_event_00000001.json").read_bytes()
        assert after == before, "a conflicting remote event overwrote a waiting one"

    @pytest.mark.parametrize(
        "bad",
        [
            {},
            {"schema_version": "1"},
            {**event_payload(), "unexpected_key": "x"},
            {**event_payload(), "schema_version": "2"},
            {**event_payload(), "event_id": "short"},
            {**event_payload(), "event_id": "../escape/attempt"},
            {**event_payload(), "raw_text": "   "},
            {**event_payload(), "source": ""},
        ],
    )
    def test_invalid_events_are_counted_not_admitted(
        self, inbox: Inbox, ledger: Ledger, bad: dict
    ) -> None:
        report = self.run(FakeTransport([[bad]]), inbox, ledger)
        assert report.submitted == []
        assert report.invalid == 1
        assert list(inbox.directory(INCOMING).glob("*.json")) == []

    def test_one_bad_event_does_not_spoil_its_neighbours(
        self, inbox: Inbox, ledger: Ledger
    ) -> None:
        batch = [
            event_payload("remote_event_00000001"),
            {"schema_version": "1", "broken": True},
            event_payload("remote_event_00000003"),
        ]
        report = self.run(FakeTransport([batch]), inbox, ledger)
        assert sorted(report.submitted) == [
            "remote_event_00000001",
            "remote_event_00000003",
        ]
        assert report.invalid == 1

    def test_transport_errors_propagate_to_the_caller(self, inbox: Inbox, ledger: Ledger) -> None:
        transport = FakeTransport(error=RemoteIntakeTransportError("down", reason="timeout"))
        with pytest.raises(RemoteIntakeTransportError):
            self.run(transport, inbox, ledger)
        assert list(inbox.directory(INCOMING).glob("*.json")) == []

    def test_report_never_contains_analysis_text(self, inbox: Inbox, ledger: Ledger) -> None:
        secret = "noi dung phan tich rat rieng tu"
        report = self.run(FakeTransport([[event_payload(text=secret)]]), inbox, ledger)
        assert secret not in repr(report)


# ------------------------------------------------------ scheduler restart
class TestRestartSafety:
    def test_restart_does_not_readmit_processed_events(self, inbox: Inbox, ledger: Ledger) -> None:
        """A fresh process holds no memory; the ledger is the memory."""
        payload = event_payload()
        intake(transport=FakeTransport([[payload]]), inbox=inbox, ledger=ledger, limit=10)

        # Simulate the worker consuming it: file moves on, ledger records it.
        submitted = inbox.directory(INCOMING) / "remote_event_00000001.json"
        raw = submitted.read_bytes()
        submitted.rename(inbox.directory(PROCESSED) / "remote_event_00000001.json")
        ledger.reserve(
            LedgerEntry(
                event_id="remote_event_00000001",
                source="gold_analysis_bot",
                payload_sha256=sha256_bytes(raw),
                run_id="20260902_000000_aaaaaa",
                state=LedgerState.INGESTED,
                reserved_at=utc_now() - timedelta(minutes=5),
            )
        )

        # A brand-new Inbox/Ledger pair, as a restarted process would build.
        fresh_inbox = Inbox(inbox.root)
        fresh_ledger = Ledger(fresh_inbox.directory("index"))
        report = intake(
            transport=FakeTransport([[payload]]),
            inbox=fresh_inbox,
            ledger=fresh_ledger,
            limit=10,
        )
        assert report.submitted == []
        assert report.duplicate == ["remote_event_00000001"]
