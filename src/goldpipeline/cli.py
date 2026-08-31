"""Command-line entry point.

    python -m goldpipeline create-run --telegram fixtures/telegram_sample.json \\
                                      --ohlc fixtures/ohlc_sample.json

argparse from the stdlib is enough here; a CLI framework would be a dependency
without a job.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from goldpipeline import PIPELINE_VERSION
from goldpipeline.adapters.config_store import (
    LayeredConfig,
    RuntimeConfigStore,
    parse_key,
)
from goldpipeline.adapters.fake_finalizer import FakeFinalizerClient
from goldpipeline.adapters.fake_publisher import FakePublisherClient
from goldpipeline.adapters.fake_reviewer import FakeReviewerClient
from goldpipeline.adapters.fake_writer import FakeWriterClient
from goldpipeline.adapters.file_source import JsonFileAnalysisSource, JsonFileMarketDataSource
from goldpipeline.adapters.finalizer_client import FinalizerClient, LazyFinalizerClient
from goldpipeline.adapters.inbox_source import parse_event
from goldpipeline.adapters.publisher_client import PublisherClient
from goldpipeline.adapters.reviewer_client import ReviewerClient
from goldpipeline.adapters.secrets import (
    CompositeSecretProvider,
    EnvironmentSecretProvider,
    SecretProvider,
)
from goldpipeline.adapters.task_scheduler import (
    PowerShellTaskScheduler,
    TaskInfo,
    TaskSchedulerAdapter,
    compare,
)
from goldpipeline.adapters.windows_credentials import (
    WindowsCredentialSecretProvider,
    inspect_backend,
)
from goldpipeline.adapters.writer_client import WriterClient
from goldpipeline.config import (
    AUTOMATION_ENABLED_ENV,
    INBOX_DIR_ENV,
    AutomationSettings,
    FinalizerSettings,
    MarketDataSettings,
    ReviewerSettings,
    TelegramSettings,
    WriterSettings,
    inbox_dir_from_env,
)
from goldpipeline.domain.errors import (
    CredentialNotFoundError,
    FinalizationBlockedError,
    FinalizeConfigurationError,
    MarketDataError,
    PipelineError,
    PublisherConfigurationError,
    ReviewConfigurationError,
    RunLockedError,
    WriterConfigurationError,
)
from goldpipeline.logging_setup import configure_logging
from goldpipeline.schemas.automation import AutomationTickResult, TickStatus
from goldpipeline.schemas.common import utc_now
from goldpipeline.schemas.ingestion import IngestOutcome, IngestResult
from goldpipeline.schemas.orchestration import PipelineMode, PipelineStatus
from goldpipeline.schemas.quality import DataQuality
from goldpipeline.schemas.runtime_config import ConfigKey, ConfigSource
from goldpipeline.schemas.secrets import SecretName, SecretSource, SecretStatus
from goldpipeline.services.automation import (
    DEFERRED,
    EXPIRED,
    WorkerContext,
    may_resume,
    run_tick,
)
from goldpipeline.services.automation_state import AutomationStore, read_defer
from goldpipeline.services.finalizer import finalize_run
from goldpipeline.services.inbox import Inbox
from goldpipeline.services.ingestion import (
    IngestionContext,
    ingest_file,
    ingest_next,
    reconcile,
)
from goldpipeline.services.orchestrator import (
    DEFAULT_MODE,
    PipelineClients,
    PipelineRunResult,
    resume_pipeline,
    run_pipeline,
)
from goldpipeline.services.pipeline import create_run, validate_sources
from goldpipeline.services.publish_gate import gate_publish
from goldpipeline.services.publisher import publish_run
from goldpipeline.services.reviewer import review_draft
from goldpipeline.services.task_plan import (
    DEFAULT_INTERVAL_MINUTES,
    DEFAULT_TASK_NAME,
    build_plan,
)
from goldpipeline.services.writer import write_draft
from goldpipeline.storage.run_store import RunStore

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INVALID_DATA = 2
EXIT_BLOCKED = 3
"""A gate declined. Nothing went wrong; retrying will not help.

Used by both `finalize` (the review rejected the article) and `gate-publish`
(the article is not safe to publish). One code for one concept: a caller can
always tell *which* gate spoke from the command it ran, so a second number for
the same meaning would only invite mistakes.
"""

DEFAULT_RUNS_DIR = Path("runs")

DEFAULT_PIPELINE_MODE = "ready-for-publish"
"""What `pipeline-run` does when nobody says otherwise.

Every check runs, including the publish gate, and nothing is sent. Publishing is
a separate decision, and it stays that way even when the pipeline is driven by
one command.
"""


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="goldpipeline",
        description="XAUUSD analysis pipeline - Round 1 (normalize + immutable runs).",
    )
    parser.add_argument("--version", action="version", version=f"goldpipeline {PIPELINE_VERSION}")
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser(
        "create-run",
        help="Normalize an analysis + OHLC pair into a new immutable Run.",
    )
    create.add_argument(
        "--telegram",
        required=True,
        type=Path,
        metavar="PATH",
        help="JSON file holding the raw analysis message.",
    )
    create.add_argument(
        "--ohlc",
        required=True,
        type=Path,
        metavar="PATH",
        help="JSON file holding the OHLC payload.",
    )
    create.add_argument(
        "--symbol",
        default=None,
        help="Instrument you expect, e.g. XAUUSD. A mismatch fails the Run.",
    )
    create.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Root directory for Runs. Default: {DEFAULT_RUNS_DIR}.",
    )
    create.add_argument(
        "--now",
        type=_parse_now,
        default=None,
        metavar="ISO8601",
        help=(
            "Treat this instant as 'now', e.g. 2026-08-28T02:20:12Z. Must carry an "
            "explicit offset. Use it to re-create a Run over historical data without "
            "tripping the recency checks. Defaults to the current UTC time."
        ),
    )
    create.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the inputs and report, without creating a Run on disk.",
    )
    create.add_argument("--json", action="store_true", help="Emit the result as JSON.")

    draft = subparsers.add_parser(
        "write-draft",
        help="Run the Claude Writer over a normalized Run and store the draft.",
    )
    draft.add_argument("--run-id", required=True, help="Run to write for.")
    draft.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Root directory for Runs. Default: {DEFAULT_RUNS_DIR}.",
    )
    draft.add_argument(
        "--model",
        default=None,
        help="Model id, overriding ANTHROPIC_MODEL. Ignored with --fake-writer.",
    )
    draft.add_argument(
        "--fake-writer",
        action="store_true",
        help="Use the offline deterministic writer. No API call, no cost.",
    )
    draft.add_argument("--json", action="store_true", help="Emit the result as JSON.")

    review = subparsers.add_parser(
        "review-draft",
        help="Audit a drafted Run with the ChatGPT Reviewer and store the verdict.",
    )
    review.add_argument("--run-id", required=True, help="Run to review.")
    review.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Root directory for Runs. Default: {DEFAULT_RUNS_DIR}.",
    )
    review.add_argument(
        "--model",
        default=None,
        help="Model id, overriding OPENAI_REVIEW_MODEL. Ignored with --fake-reviewer.",
    )
    review.add_argument(
        "--fake-reviewer",
        action="store_true",
        help="Use the offline deterministic reviewer. No API call, no cost.",
    )
    review.add_argument("--json", action="store_true", help="Emit the result as JSON.")

    final = subparsers.add_parser(
        "finalize",
        help="Apply the review to a drafted Run and store the final article.",
    )
    final.add_argument("--run-id", required=True, help="Run to finalize.")
    final.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Root directory for Runs. Default: {DEFAULT_RUNS_DIR}.",
    )
    final.add_argument(
        "--model",
        default=None,
        help=(
            "Model id, overriding ANTHROPIC_FINALIZER_MODEL. Only used when the "
            "review asked for revisions."
        ),
    )
    final.add_argument(
        "--fake-finalizer",
        action="store_true",
        help="Use the offline deterministic finalizer. No API call, no cost.",
    )
    final.add_argument("--json", action="store_true", help="Emit the result as JSON.")

    gate = subparsers.add_parser(
        "gate-publish",
        help="Decide deterministically whether a finalized Run may be published.",
    )
    gate.add_argument("--run-id", required=True, help="Run to gate.")
    gate.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Root directory for Runs. Default: {DEFAULT_RUNS_DIR}.",
    )
    gate.add_argument("--json", action="store_true", help="Emit the result as JSON.")

    publish = subparsers.add_parser(
        "publish",
        help="Send an approved Run to Telegram. One attempt per Run.",
    )
    publish.add_argument("--run-id", required=True, help="Run to publish.")
    publish.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Root directory for Runs. Default: {DEFAULT_RUNS_DIR}.",
    )
    publish.add_argument(
        "--fake-publisher",
        action="store_true",
        help="Use the offline publisher. Nothing is sent anywhere.",
    )
    publish.add_argument("--json", action="store_true", help="Emit the result as JSON.")

    pipeline_run = subparsers.add_parser(
        "pipeline-run",
        help="Create a Run from JSON inputs and drive it through the whole pipeline.",
    )
    pipeline_run.add_argument(
        "--telegram",
        required=True,
        type=Path,
        metavar="PATH",
        help="JSON file holding the raw analysis message.",
    )
    pipeline_run.add_argument(
        "--ohlc",
        required=True,
        type=Path,
        metavar="PATH",
        help="JSON file holding the OHLC payload.",
    )
    pipeline_run.add_argument(
        "--symbol",
        default=None,
        help="Instrument you expect, e.g. XAUUSD. A mismatch fails the Run.",
    )
    pipeline_run.add_argument(
        "--now",
        type=_parse_now,
        default=None,
        metavar="ISO8601",
        help="Treat this instant as 'now' in every stage. Must carry an explicit offset.",
    )
    _add_pipeline_arguments(pipeline_run)

    pipeline_resume = subparsers.add_parser(
        "pipeline-resume",
        help="Continue an existing Run from whichever stage it is due for.",
    )
    pipeline_resume.add_argument("--run-id", required=True, help="Run to continue.")
    pipeline_resume.add_argument(
        "--now",
        type=_parse_now,
        default=None,
        metavar="ISO8601",
        help="Treat this instant as 'now' in every stage. Must carry an explicit offset.",
    )
    _add_pipeline_arguments(pipeline_resume)

    ingest = subparsers.add_parser(
        "pipeline-ingest",
        help="Submit one analysis payload and drive it through the whole pipeline.",
    )
    ingest.add_argument(
        "--analysis",
        required=True,
        type=Path,
        metavar="PATH",
        help="JSON file holding one inbox event.",
    )
    _add_ingest_arguments(ingest)
    _add_pipeline_arguments(ingest)

    process = subparsers.add_parser(
        "inbox-process-one",
        help="Claim the oldest waiting inbox event and drive it through the pipeline.",
    )
    _add_ingest_arguments(process)
    _add_pipeline_arguments(process)

    submit = subparsers.add_parser(
        "inbox-submit",
        help="Place one analysis payload in the inbox, atomically. Does not process it.",
    )
    submit.add_argument("--file", required=True, type=Path, metavar="PATH")
    submit.add_argument("--inbox-dir", type=Path, default=None)
    submit.add_argument("--json", action="store_true", help="Emit the result as JSON.")

    reconcile_parser = subparsers.add_parser(
        "inbox-reconcile",
        help="Report - and optionally resolve - events left behind by an interrupted run.",
    )
    reconcile_parser.add_argument("--inbox-dir", type=Path, default=None)
    reconcile_parser.add_argument(
        "--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help=f"Default: {DEFAULT_RUNS_DIR}."
    )
    reconcile_parser.add_argument(
        "--recover",
        action="store_true",
        help="Act on the report instead of only printing it. Never re-runs a stage.",
    )
    reconcile_parser.add_argument("--json", action="store_true")

    check = subparsers.add_parser(
        "mt5-check",
        help="Read-only MetaTrader 5 diagnostic: symbol, timeframe, latest closed candle.",
    )
    check.add_argument(
        "--fake-mt5",
        action="store_true",
        help="Run the diagnostic against the offline stand-in instead of a terminal.",
    )
    check.add_argument("--json", action="store_true", help="Emit the result as JSON.")

    once = subparsers.add_parser(
        "automation-run-once",
        help="Run one automation tick now. An operator action; ignores the kill switch.",
    )
    _add_automation_arguments(once)
    once.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the work this tick would do. Claims nothing and calls nothing.",
    )

    worker = subparsers.add_parser(
        "automation-worker-tick",
        help="What Task Scheduler runs. Does nothing unless automation is enabled.",
    )
    _add_automation_arguments(worker)

    status = subparsers.add_parser(
        "automation-status",
        help="Read-only summary of the queue, the Runs and the last tick.",
    )
    _add_automation_arguments(status)

    preflight = subparsers.add_parser(
        "automation-preflight",
        help="Report whether unattended operation is configured. Prints no secret values.",
    )
    _add_automation_arguments(preflight)

    plan = subparsers.add_parser(
        "automation-task-plan",
        help="Print the Windows Task Scheduler definition. Registers nothing.",
    )
    plan.add_argument("--task-name", default=None, help="Name to register under.")
    plan.add_argument(
        "--interval-minutes", type=int, default=None, help="How often to wake the worker."
    )
    plan.add_argument("--xml", action="store_true", help="Print the task XML.")
    plan.add_argument("--json", action="store_true", help="Emit the plan as JSON.")

    secrets_status = subparsers.add_parser(
        "secrets-status",
        help="Report which credentials are available and from where. Prints no values.",
    )
    secrets_status.add_argument("--json", action="store_true")

    secrets_set = subparsers.add_parser(
        "secrets-set",
        help="Store one credential in the OS credential manager. Prompts invisibly.",
    )
    secrets_set.add_argument("name", choices=sorted(_SECRET_CHOICES))
    # Deliberately no --value, --token or --api-key: a secret on a command line
    # lands in shell history and in every process listing on the machine.

    secrets_delete = subparsers.add_parser(
        "secrets-delete", help="Remove one credential from the OS credential manager."
    )
    secrets_delete.add_argument("name", choices=sorted(_SECRET_CHOICES))

    config_status = subparsers.add_parser(
        "config-status",
        help="Show every persistent setting and where its value came from.",
    )
    config_status.add_argument("--json", action="store_true")

    config_set = subparsers.add_parser(
        "config-set", help="Persist one non-secret setting. Credentials are refused."
    )
    config_set.add_argument("name", help="Setting name, e.g. TELEGRAM_TARGET_CHAT_ID.")
    config_set.add_argument("value", help="Value. Non-secret, so an argument is fine.")

    config_delete = subparsers.add_parser(
        "config-delete", help="Remove one persistent setting, falling back to the default."
    )
    config_delete.add_argument("name")

    task_install = subparsers.add_parser(
        "automation-task-install",
        help="Register the Windows scheduled task. Prints the plan unless --apply.",
    )
    task_install.add_argument("--task-name", default=None)
    task_install.add_argument("--interval-minutes", type=int, default=None)
    task_install.add_argument(
        "--apply",
        action="store_true",
        help="Actually register it. Without this, nothing on the machine changes.",
    )
    task_install.add_argument("--json", action="store_true")

    task_status = subparsers.add_parser(
        "automation-task-status", help="Read-only status of the registered task."
    )
    task_status.add_argument("--task-name", default=None)
    task_status.add_argument("--json", action="store_true")

    task_remove = subparsers.add_parser(
        "automation-task-remove",
        help="Unregister the scheduled task. Prints the plan unless --apply.",
    )
    task_remove.add_argument("--task-name", default=None)
    task_remove.add_argument("--apply", action="store_true")
    task_remove.add_argument("--json", action="store_true")

    show = subparsers.add_parser("show-run", help="Print a summary of an existing Run.")
    show.add_argument("run_id", help="Run identifier, e.g. 20260828_022701_a83f2c.")
    show.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)

    subparsers.add_parser("list-runs", help="List Run ids under the runs directory.").add_argument(
        "--runs-dir", type=Path, default=DEFAULT_RUNS_DIR
    )

    return parser


def _add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the flags shared by ``pipeline-run`` and ``pipeline-resume``.

    Note what is *not* here: there is no ``--all``, and no single flag that both
    runs the pipeline and sends the result. Reaching Telegram takes two flags
    that each say so.
    """
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Root directory for Runs. Default: {DEFAULT_RUNS_DIR}.",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value.lower().replace("_", "-") for mode in PipelineMode],
        default=DEFAULT_PIPELINE_MODE,
        help=(
            "How far to go: generate-only stops at FINALIZED, ready-for-publish "
            "stops at the gate's verdict, publish continues into the publisher. "
            f"Default: {DEFAULT_PIPELINE_MODE}."
        ),
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Shorthand for --mode publish. Nothing is sent without it.",
    )
    parser.add_argument(
        "--confirm-real-publish",
        action="store_true",
        help=(
            "Required alongside --publish when the transport is real. Two "
            "deliberate flags, because a published article cannot be unpublished."
        ),
    )
    parser.add_argument(
        "--fake-ai",
        action="store_true",
        help="Use the offline writer, reviewer and finalizer. No API calls, no cost.",
    )
    parser.add_argument("--fake-writer", action="store_true", help="Offline writer only.")
    parser.add_argument("--fake-reviewer", action="store_true", help="Offline reviewer only.")
    parser.add_argument("--fake-finalizer", action="store_true", help="Offline finalizer only.")
    parser.add_argument(
        "--fake-publisher",
        action="store_true",
        help="Offline publisher. Nothing is sent anywhere, even in publish mode.",
    )
    parser.add_argument("--writer-model", default=None, help="Overrides ANTHROPIC_MODEL.")
    parser.add_argument("--reviewer-model", default=None, help="Overrides OPENAI_REVIEW_MODEL.")
    parser.add_argument(
        "--finalizer-model", default=None, help="Overrides ANTHROPIC_FINALIZER_MODEL."
    )
    parser.add_argument("--json", action="store_true", help="Emit the execution result as JSON.")


def _add_ingest_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the flags shared by the ingestion commands."""
    parser.add_argument(
        "--inbox-dir",
        type=Path,
        default=None,
        help=f"Inbox root. Default: ${INBOX_DIR_ENV} or 'inbox'.",
    )
    parser.add_argument(
        "--market-source",
        choices=("mt5", "file"),
        default="mt5",
        help="Where candles come from. 'file' needs --ohlc. Default: mt5.",
    )
    parser.add_argument(
        "--ohlc",
        type=Path,
        default=None,
        metavar="PATH",
        help="JSON file of candles, for --market-source file.",
    )
    parser.add_argument(
        "--fake-mt5",
        action="store_true",
        help="Use the offline candle stand-in. No terminal, no market data.",
    )
    parser.add_argument(
        "--normalize-only",
        action="store_true",
        help="Stop once the Run exists. Nothing after ingestion runs.",
    )


def _add_automation_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the flags shared by the automation commands.

    Note the absence: there is no ``--publish``, no ``--auto-publish`` and no
    ``--target``. Unattended publishing is turned on in the environment, by
    someone with access to the machine, and no command line can do it.
    """
    parser.add_argument("--inbox-dir", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--automation-dir", type=Path, default=None)
    parser.add_argument(
        "--market-source", choices=("mt5", "file"), default="mt5", help="Default: mt5."
    )
    parser.add_argument("--ohlc", type=Path, default=None, metavar="PATH")
    parser.add_argument("--fake-mt5", action="store_true", help="Use the offline candle stand-in.")
    parser.add_argument("--fake-ai", action="store_true", help="Use the offline AI clients.")
    parser.add_argument(
        "--fake-publisher",
        action="store_true",
        help="Use the offline publisher. Nothing is sent, even with publishing enabled.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the result as JSON.")


def _parse_now(value: str) -> datetime:
    """Parse an ISO-8601 instant for ``--now``, rejecting naive input."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a valid ISO-8601 datetime: {value!r}") from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(f"--now needs an explicit timezone offset, got {value!r}")
    return parsed


def _cmd_create_run(args: argparse.Namespace) -> int:
    analysis_source = JsonFileAnalysisSource(args.telegram)
    market_source = JsonFileMarketDataSource(args.ohlc)

    if args.dry_run:
        try:
            context = validate_sources(
                analysis_source=analysis_source,
                market_source=market_source,
                expected_symbol=args.symbol,
                now=args.now,
            )
        except PipelineError as exc:
            return _report_failure(exc, as_json=args.json)

        quality = context.data_quality
        if args.json:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "valid": True,
                        "symbol": context.market.symbol,
                        "bar_count": quality.bar_count,
                        "quality_status": quality.status,
                        "warnings": [w.code for w in quality.warnings],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print("Dry run: inputs are valid, no Run was created.")
            print(f"Symbol: {context.market.symbol} {context.market.timeframe}")
            print(f"Bars: {quality.bar_count}")
            print(f"Data quality: {quality.status}")
            _print_warnings(quality)
        return EXIT_OK

    result = create_run(
        analysis_source=analysis_source,
        market_source=market_source,
        store=RunStore(args.runs_dir),
        expected_symbol=args.symbol,
        now=args.now,
    )

    if not result.succeeded:
        assert result.error is not None
        if args.json:
            print(
                json.dumps(
                    {
                        "run_id": result.run_id,
                        "status": result.status,
                        "run_dir": str(result.run_dir),
                        "error": result.error.to_dict(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(f"Run failed: {result.run_id}", file=sys.stderr)
            print(f"Status: {result.status}", file=sys.stderr)
            print(f"Directory: {result.run_dir}", file=sys.stderr)
            print(f"Error: {result.error}", file=sys.stderr)
            if result.error.details:
                print(
                    f"Details: {json.dumps(result.error.details, ensure_ascii=False)}",
                    file=sys.stderr,
                )
        return EXIT_INVALID_DATA

    assert result.context is not None
    quality = result.context.data_quality
    if args.json:
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "status": result.status,
                    "run_dir": str(result.run_dir),
                    "context": str(result.context_path),
                    "bar_count": quality.bar_count,
                    "quality_status": quality.status,
                    "warnings": [w.code for w in quality.warnings],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"Run created: {result.run_id}")
        print(f"Status: {result.status}")
        print(f"Context: {result.context_path}")
        print(f"Data quality: {quality.status} ({quality.bar_count} bars)")
        _print_warnings(quality)
    return EXIT_OK


def _print_warnings(quality: DataQuality) -> None:
    for warning in quality.warnings:
        print(f"  ! {warning.code}: {warning.message}")


class _UsageError(Exception):
    """A flag combination the parser cannot express on its own."""


def _classify_failure(exc: PipelineError) -> int:
    """Exit code for a failure: configuration problems are not data problems.

    A missing API key and a duplicated candle are fixed in different places -
    one in the environment, one in the input - so anything scripting these
    commands needs to tell them apart.
    """
    configuration = isinstance(
        exc,
        WriterConfigurationError
        | ReviewConfigurationError
        | FinalizeConfigurationError
        | PublisherConfigurationError,
    )
    return EXIT_ERROR if configuration else EXIT_INVALID_DATA


def _report_failure(exc: PipelineError, *, as_json: bool) -> int:
    """Print a failure and choose an exit code that says what kind it was."""
    code = _classify_failure(exc)
    configuration = code == EXIT_ERROR

    if as_json:
        print(json.dumps({"valid": False, "error": exc.to_dict()}, ensure_ascii=False, indent=2))
    else:
        label = "Configuration error" if configuration else "Validation failed"
        print(f"{label}: {exc}", file=sys.stderr)
        if exc.details:
            print(f"Details: {json.dumps(exc.details, ensure_ascii=False)}", file=sys.stderr)
    return code


def _writer_client(*, fake: bool, model: str | None) -> WriterClient:
    """Pick the writer client for this invocation.

    ``fake`` short-circuits before any credential is read, so a smoke run cannot
    accidentally reach the provider or need a key present.
    """
    if fake:
        return FakeWriterClient()
    settings = WriterSettings.from_env(
        _config_env(), model_override=model, secrets=_secret_provider()
    )
    from goldpipeline.adapters.anthropic_writer import AnthropicWriterClient

    return AnthropicWriterClient(settings)


def _cmd_write_draft(args: argparse.Namespace) -> int:
    client = _writer_client(fake=args.fake_writer, model=args.model)
    result = write_draft(
        run_id=args.run_id,
        store=RunStore(args.runs_dir),
        client=client,
    )

    if not result.succeeded:
        assert result.error is not None
        if args.json:
            print(
                json.dumps(
                    {
                        "run_id": result.run_id,
                        "status": result.status,
                        "writer_status": "FAILED",
                        "error": result.error.to_dict(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(f"Run: {result.run_id}", file=sys.stderr)
            print("Writer: FAILED", file=sys.stderr)
            print(f"Error: {result.error}", file=sys.stderr)
            if result.error.details:
                print(
                    f"Details: {json.dumps(result.error.details, ensure_ascii=False)}",
                    file=sys.stderr,
                )
        return EXIT_INVALID_DATA

    written = result.result
    assert written is not None
    if args.json:
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "status": result.status,
                    "writer_status": written.status,
                    "provider": written.provider,
                    "model": written.model,
                    "prompt_version": written.prompt_version,
                    "draft": str(result.draft_path),
                    "metadata": str(result.metadata_path),
                    "article_chars": written.article_chars,
                    "source_claims": len(written.source_claims),
                    "warnings": [w.code for w in written.warnings],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"Run: {result.run_id}")
        print(f"Writer: {written.provider} ({written.model})")
        print(f"Status: {written.status}")
        print(f"Draft: {result.draft_path}")
        print(f"Metadata: {result.metadata_path}")
        print(f"Article: {written.article_chars} chars, {len(written.source_claims)} claims")
        for warning in written.warnings:
            print(f"  ! {warning.code}: {warning.message}")
    return EXIT_OK


def _reviewer_client(*, fake: bool, model: str | None) -> ReviewerClient:
    """Pick the reviewer client for this invocation.

    ``fake`` short-circuits before any credential is read, so a smoke run cannot
    accidentally reach the provider or need a key present.
    """
    if fake:
        return FakeReviewerClient()
    settings = ReviewerSettings.from_env(
        _config_env(), model_override=model, secrets=_secret_provider()
    )
    from goldpipeline.adapters.openai_reviewer import OpenAIReviewerClient

    return OpenAIReviewerClient(settings)


def _cmd_review_draft(args: argparse.Namespace) -> int:
    client = _reviewer_client(fake=args.fake_reviewer, model=args.model)
    result = review_draft(run_id=args.run_id, store=RunStore(args.runs_dir), client=client)

    if not result.succeeded:
        assert result.error is not None
        if args.json:
            print(
                json.dumps(
                    {
                        "run_id": result.run_id,
                        "status": result.status,
                        "review_status": "FAILED",
                        "error": result.error.to_dict(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(f"Run: {result.run_id}", file=sys.stderr)
            print("Reviewer: FAILED", file=sys.stderr)
            print(f"Error: {result.error}", file=sys.stderr)
            if result.error.details:
                print(
                    f"Details: {json.dumps(result.error.details, ensure_ascii=False)}",
                    file=sys.stderr,
                )
        return EXIT_INVALID_DATA

    review = result.result
    assert review is not None
    if args.json:
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "status": result.status,
                    "verdict": review.status,
                    "model_verdict": review.model_status,
                    "verdict_source": review.verdict_source,
                    "score": review.score,
                    "issues": len(review.issues),
                    "blocking_issues": len(review.blocking_issues),
                    "provider": review.provider,
                    "model": review.model,
                    "prompt_version": review.prompt_version,
                    "review": str(result.review_path),
                    "deterministic_findings": [
                        str(finding.code) for finding in review.deterministic_findings
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"Run: {result.run_id}")
        print(f"Reviewer: {review.provider} ({review.model})")
        print(f"Verdict: {review.status}")
        print(f"Score: {review.score}")
        print(f"Issues: {len(review.issues)}")
        print(f"Review: {result.review_path}")
        for note in review.policy_notes:
            print(f"  * {note}")
        for issue in review.issues:
            print(f"  ! [{issue.severity}] {issue.category}: {issue.message}")
    return EXIT_OK


def _finalizer_client(*, fake: bool, model: str | None) -> FinalizerClient:
    """Build the finalizer client.

    Only ever called through :class:`LazyFinalizerClient`, so credentials are
    read lazily and only for the real revision path: a PASS is a byte copy and a
    REJECT is a refusal, and neither should demand an API key from an operator
    who is only trying to finish a Run.
    """
    if fake:
        return FakeFinalizerClient()

    from goldpipeline.adapters.anthropic_finalizer import AnthropicFinalizerClient

    return AnthropicFinalizerClient(
        FinalizerSettings.from_env(_config_env(), model_override=model, secrets=_secret_provider())
    )


def _cmd_finalize(args: argparse.Namespace) -> int:
    result = finalize_run(
        run_id=args.run_id,
        store=RunStore(args.runs_dir),
        client=LazyFinalizerClient(
            lambda: _finalizer_client(fake=args.fake_finalizer, model=args.model)
        ),
    )

    if not result.succeeded:
        assert result.error is not None
        blocked = result.blocked

        # A missing key surfaces from inside the stage, because credentials are
        # read lazily and only on the revision path. It is still a configuration
        # problem, so it must exit like one rather than like bad data.
        if isinstance(result.error, FinalizeConfigurationError):
            return _report_failure(result.error, as_json=args.json)

        if args.json:
            print(
                json.dumps(
                    {
                        "run_id": result.run_id,
                        "status": result.status,
                        "finalization": "BLOCKED" if blocked else "FAILED",
                        "error": result.error.to_dict(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif blocked:
            print(f"Run: {result.run_id}", file=sys.stderr)
            print("Review: REJECT", file=sys.stderr)
            print("Finalization blocked.", file=sys.stderr)
            print(f"Reason: {result.error.message}", file=sys.stderr)
        else:
            print(f"Run: {result.run_id}", file=sys.stderr)
            print("Finalization: FAILED", file=sys.stderr)
            print(f"Error: {result.error}", file=sys.stderr)
            if result.error.details:
                print(
                    f"Details: {json.dumps(result.error.details, ensure_ascii=False)}",
                    file=sys.stderr,
                )
        return EXIT_BLOCKED if blocked else EXIT_INVALID_DATA

    final = result.result
    assert final is not None
    if args.json:
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "status": result.status,
                    "review": final.review_status,
                    "finalization": final.finalization_mode,
                    "provider_called": final.provider_called,
                    "issues_applied": final.applied_count,
                    "issues_total": len(final.issue_resolutions),
                    "article_chars": final.article_chars,
                    "model": final.model,
                    "prompt_version": final.prompt_version,
                    "final": str(result.final_path),
                    "metadata": str(result.metadata_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"Run: {result.run_id}")
        print(f"Review: {final.review_status}")
        print(f"Finalization: {final.finalization_mode}")
        print(f"Provider called: {'Yes' if final.provider_called else 'No'}")
        if final.issue_resolutions:
            print(f"Issues resolved: {final.applied_count}/{len(final.issue_resolutions)}")
        if final.model:
            print(f"Finalizer: {final.provider} ({final.model})")
        print(f"Final: {result.final_path}")
        for resolution in final.issue_resolutions:
            print(f"  - [{resolution.resolution}] {resolution.issue_id}: {resolution.description}")
    return EXIT_OK


def _cmd_gate_publish(args: argparse.Namespace) -> int:
    """Run the final publish gate. No provider, no credentials, no network."""
    result = gate_publish(run_id=args.run_id, store=RunStore(args.runs_dir))
    decision = result.decision
    passed, warned, failed = decision.counts

    if args.json:
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "status": result.status,
                    "gate_version": decision.gate_version,
                    "decision": decision.decision,
                    "checks": {"passed": passed, "warnings": warned, "failed": failed},
                    "blockers": [
                        {"code": b.code, "severity": b.severity, "message": b.message}
                        for b in decision.blockers
                    ],
                    "warnings_detail": [
                        {"code": w.code, "severity": w.severity, "message": w.message}
                        for w in decision.warnings
                    ],
                    "artifact": str(result.decision_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return EXIT_OK if result.approved else EXIT_BLOCKED

    stream = sys.stdout if result.approved else sys.stderr
    print(f"Run: {result.run_id}", file=stream)
    print(f"Gate: {decision.gate_version}", file=stream)
    print(f"Decision: {decision.decision}", file=stream)
    print(f"Checks: {passed} passed, {warned} warnings, {failed} failed", file=stream)

    if decision.blockers:
        print(f"Blockers: {len(decision.blockers)}", file=stream)
        for blocker in decision.blockers:
            print(f"  - [{blocker.severity}] {blocker.code}: {blocker.message}", file=stream)
    for warning in decision.warnings:
        print(f"  ! {warning.code}: {warning.message}", file=stream)

    print(f"Artifact: {result.decision_path}", file=stream)
    return EXIT_OK if result.approved else EXIT_BLOCKED


FAKE_TARGET_CHAT = "@fake_offline_channel"
"""Destination recorded by `--fake-publisher`.

Obviously not a real channel, so a fake attempt can never be mistaken for one
that reached Telegram - and the offline path needs no credentials at all.
"""


def _publisher_client(*, fake: bool) -> tuple[PublisherClient, str]:
    """Pick the transport and the destination.

    The destination comes from configuration only. There is deliberately no
    `--chat-id`: a flag that redirects where an approved article is posted is
    one typo away from publishing to the wrong channel, and it would also give
    anything that can influence a command line control over the destination.
    """
    if fake:
        return FakePublisherClient(), FAKE_TARGET_CHAT

    settings = TelegramSettings.from_env(_config_env(), secrets=_secret_provider())
    from goldpipeline.adapters.telegram_publisher import TelegramPublisherClient

    return TelegramPublisherClient(settings), settings.target_chat


def _cmd_publish(args: argparse.Namespace) -> int:
    client, target = _publisher_client(fake=args.fake_publisher)
    outcome = publish_run(
        run_id=args.run_id,
        store=RunStore(args.runs_dir),
        client=client,
        target_chat=target,
    )
    result = outcome.result
    ok = outcome.published

    if args.json:
        print(
            json.dumps(
                {
                    "run_id": outcome.run_id,
                    "status": outcome.status,
                    "publish_status": result.status,
                    "provider": result.provider,
                    "target_chat": result.target_chat,
                    "attempt_id": result.attempt_id,
                    "chunk_count": result.chunk_count,
                    "confirmed_count": result.confirmed_count,
                    "message_ids": [m.message_id for m in result.messages],
                    "failure": (
                        {
                            "category": result.failure.category,
                            "code": result.failure.safe_code,
                            "message": result.failure.safe_message,
                        }
                        if result.failure
                        else None
                    ),
                    "artifact": str(outcome.result_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return EXIT_OK if ok else EXIT_BLOCKED

    stream = sys.stdout if ok else sys.stderr
    print(f"Run: {outcome.run_id}", file=stream)
    print(f"Publisher: {result.provider} -> {result.target_chat}", file=stream)
    print(f"Status: {result.status}", file=stream)
    print(f"Delivered: {result.confirmed_count}/{result.chunk_count} message(s)", file=stream)
    for message in result.messages:
        print(f"  chunk {message.chunk_index}: message_id={message.message_id}", file=stream)
    if result.failure is not None:
        print(f"Failure: [{result.failure.safe_code}] {result.failure.safe_message}", file=stream)
    for warning in result.warnings:
        print(f"  ! {warning}", file=stream)
    print(f"Artifact: {outcome.result_path}", file=stream)

    return EXIT_OK if ok else EXIT_BLOCKED


def _resolve_mode(args: argparse.Namespace) -> PipelineMode:
    """Turn the flags into a mode, refusing an unguarded real publish.

    ``--publish`` alone is enough for the offline transport, because nothing
    leaves the machine. For the real one it is not: ``--confirm-real-publish``
    has to be there too. Both flags name what they do, so no combination of them
    can post an article by accident.

    Passing ``--fake-publisher`` and ``--confirm-real-publish`` together is
    allowed and the fake wins. The alternative - rejecting the pair - would push
    an operator towards dropping ``--fake-publisher``, which is exactly the
    wrong flag to drop.
    """
    mode = (
        PipelineMode.PUBLISH if args.publish else PipelineMode(args.mode.upper().replace("-", "_"))
    )

    if mode is PipelineMode.PUBLISH and not args.fake_publisher and not args.confirm_real_publish:
        raise _UsageError(
            "publishing for real needs --confirm-real-publish alongside --publish. "
            "Use --fake-publisher to rehearse the whole pipeline offline instead."
        )
    return mode


def _pipeline_clients(args: argparse.Namespace) -> PipelineClients:
    """Wire up client factories. Nothing is built here.

    Each lambda runs at most once, and only if its stage is actually reached, so
    a Run that stops at the reviewer never needs a finalizer key and a Run
    resumed at the gate needs no keys at all.
    """
    # Read through `getattr`: the automation commands share these factories but
    # deliberately carry no per-model flags, because a scheduled worker should
    # take its model from configuration rather than from a command line nobody
    # re-reads after registering the task.
    fake_ai = args.fake_ai
    return PipelineClients(
        writer=lambda: _writer_client(
            fake=fake_ai or getattr(args, "fake_writer", False),
            model=getattr(args, "writer_model", None),
        ),
        reviewer=lambda: _reviewer_client(
            fake=fake_ai or getattr(args, "fake_reviewer", False),
            model=getattr(args, "reviewer_model", None),
        ),
        finalizer=lambda: _finalizer_client(
            fake=fake_ai or getattr(args, "fake_finalizer", False),
            model=getattr(args, "finalizer_model", None),
        ),
        publisher=lambda: _publisher_client(fake=args.fake_publisher),
    )


def _cmd_pipeline_run(args: argparse.Namespace) -> int:
    mode = _resolve_mode(args)
    outcome = run_pipeline(
        analysis_source=JsonFileAnalysisSource(args.telegram),
        market_source=JsonFileMarketDataSource(args.ohlc),
        store=RunStore(args.runs_dir),
        clients=_pipeline_clients(args),
        mode=mode,
        expected_symbol=args.symbol,
        now=args.now,
    )
    return _report_execution(outcome, as_json=args.json)


def _cmd_pipeline_resume(args: argparse.Namespace) -> int:
    mode = _resolve_mode(args)
    try:
        outcome = resume_pipeline(
            run_id=args.run_id,
            store=RunStore(args.runs_dir),
            clients=_pipeline_clients(args),
            mode=mode,
            now=args.now,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"{exc}", file=sys.stderr)
        return EXIT_ERROR
    return _report_execution(outcome, as_json=args.json)


_STAGE_LABELS = {
    "NORMALIZE": "NORMALIZE",
    "WRITE": "WRITER",
    "REVIEW": "REVIEW",
    "FINALIZE": "FINALIZE",
    "GATE": "GATE",
    "PUBLISH": "PUBLISH",
}


def _report_execution(outcome: PipelineRunResult, *, as_json: bool) -> int:
    """Print the execution and choose an exit code.

    The article itself is never printed. A pipeline summary is something an
    operator reads dozens of times a week; dumping the full text into it would
    make the one line that matters - the final status - the hardest to find.
    """
    result = outcome.result

    if as_json:
        print(result.model_dump_json(indent=2))
        return _pipeline_exit_code(outcome)

    stream = sys.stdout if outcome.succeeded else sys.stderr
    print(f"Run: {result.run_id}", file=stream)
    print(f"Mode: {result.mode}", file=stream)

    if result.stages:
        print(file=stream)
        for stage in result.stages:
            label = _STAGE_LABELS.get(str(stage.stage), str(stage.stage))
            print(f"{label:<10} {stage.detail or stage.outcome}", file=stream)

    print(file=stream)
    print(f"Final status: {result.run_status}", file=stream)
    print(f"Execution: {result.status}", file=stream)
    print(f"Run directory: {outcome.run_dir}", file=stream)

    if outcome.error is not None:
        print(f"Error: {outcome.error}", file=stream)
        if outcome.error.details:
            print(
                f"Details: {json.dumps(outcome.error.details, ensure_ascii=False)}",
                file=stream,
            )
    return _pipeline_exit_code(outcome)


def _pipeline_exit_code(outcome: PipelineRunResult) -> int:
    """Map an execution onto an exit code.

    A gate declining and a stage breaking are different events for whatever is
    scripting this, so they get different numbers - and a missing API key keeps
    exiting 1 the way it does from every other command here.
    """
    if outcome.succeeded:
        return EXIT_OK
    if outcome.result.status is PipelineStatus.FAILED and outcome.error is not None:
        return _classify_failure(outcome.error)
    return EXIT_BLOCKED


def _fake_mt5_module() -> Any:
    """The offline stand-in behind ``--fake-mt5``.

    It models one plausible broker - one that offers ``XAUUSD`` and a handful of
    neighbours - rather than agreeing with whatever symbol happens to be
    configured. That way a misconfigured symbol fails offline exactly as it
    would against a real terminal, which is most of what the flag is for.
    """
    from goldpipeline.adapters.fake_mt5 import FakeMt5Module

    return FakeMt5Module()


def _market_source(args: argparse.Namespace) -> Any:
    """Build the market data source this invocation asked for.

    Nothing is constructed until here, and the MetaTrader package is imported
    only inside the source it belongs to - so `--market-source file` and
    `--fake-mt5` work on a machine that has never had a terminal installed.
    """
    if args.market_source == "file":
        if args.ohlc is None:
            raise _UsageError("--market-source file needs --ohlc PATH")
        return JsonFileMarketDataSource(args.ohlc)

    settings = MarketDataSettings.from_env(_config_env())

    from goldpipeline.adapters.mt5_market import MetaTrader5MarketDataSource

    return MetaTrader5MarketDataSource(
        settings, module=_fake_mt5_module() if args.fake_mt5 else None
    )


def _ingestion_context(args: argparse.Namespace) -> IngestionContext:
    inbox_root = args.inbox_dir if args.inbox_dir is not None else inbox_dir_from_env(_config_env())
    inbox = Inbox(inbox_root)
    inbox.ensure_layout()
    return IngestionContext(
        inbox=inbox,
        store=RunStore(args.runs_dir),
        market_source=_market_source(args),
        expected_symbol=MarketDataSettings.from_env(_config_env()).canonical_symbol,
    )


def _cmd_pipeline_ingest(args: argparse.Namespace) -> int:
    # Resolved first, before anything is fetched or written: a refused flag
    # combination must cost nothing, and must not leave a Run behind.
    _resolve_mode(args)
    context = _ingestion_context(args)
    return _after_ingest(args, ingest_file(context, args.analysis))


def _cmd_inbox_process_one(args: argparse.Namespace) -> int:
    _resolve_mode(args)
    context = _ingestion_context(args)
    return _after_ingest(args, ingest_next(context))


def _after_ingest(args: argparse.Namespace, result: IngestResult) -> int:
    """Report an ingestion and, unless told otherwise, continue into the pipeline.

    A replay continues too. The producer's retry may be the first invocation to
    get past the writer, and refusing to touch an already-ingested Run would
    strand it at ``NORMALIZED`` forever.
    """
    if not result.succeeded or args.normalize_only:
        return _report_ingest(result, as_json=args.json)

    assert result.run_id is not None
    mode = _resolve_mode(args)
    try:
        outcome = resume_pipeline(
            run_id=result.run_id,
            store=RunStore(args.runs_dir),
            clients=_pipeline_clients(args),
            mode=mode,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"{exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.json:
        payload = json.loads(outcome.result.model_dump_json())
        payload["ingest"] = json.loads(result.model_dump_json())
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return _pipeline_exit_code(outcome)

    stream = sys.stdout if outcome.succeeded else sys.stderr
    print(f"Event: {result.event_id} ({result.outcome})", file=stream)
    return _report_execution(outcome, as_json=False)


def _report_ingest(result: IngestResult, *, as_json: bool) -> int:
    """Print an ingestion outcome and map it onto an exit code."""
    code = _ingest_exit_code(result)
    if as_json:
        print(result.model_dump_json(indent=2))
        return code

    # Keyed to the exit code, not to whether a Run appeared: an empty inbox is a
    # normal, successful answer and does not belong on stderr.
    stream = sys.stdout if code == EXIT_OK else sys.stderr
    print(f"Ingestion: {result.outcome}", file=stream)
    if result.event_id:
        print(f"Event: {result.event_id}", file=stream)
    if result.run_id:
        print(f"Run: {result.run_id}", file=stream)
    if result.payload_sha256:
        print(f"Payload SHA-256: {result.payload_sha256}", file=stream)
    if result.source_path:
        print(f"Event file: {result.source_path}", file=stream)
    if result.detail:
        print(f"Detail: {result.detail}", file=stream)
    return _ingest_exit_code(result)


def _ingest_exit_code(result: IngestResult) -> int:
    """A refusal is not a crash, and a replay is not a failure."""
    if result.succeeded or result.outcome is IngestOutcome.NOTHING_TO_DO:
        return EXIT_OK
    if result.outcome is IngestOutcome.INVALID_PAYLOAD:
        return EXIT_INVALID_DATA
    return EXIT_BLOCKED


def _cmd_inbox_submit(args: argparse.Namespace) -> int:
    """Place a payload in the inbox without processing it."""
    inbox_root = args.inbox_dir if args.inbox_dir is not None else inbox_dir_from_env(_config_env())
    inbox = Inbox(inbox_root)

    try:
        payload = json.loads(args.file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        print(f"payload could not be read: {exc}", file=sys.stderr)
        return EXIT_INVALID_DATA
    if not isinstance(payload, dict):
        print("payload must be a JSON object", file=sys.stderr)
        return EXIT_INVALID_DATA

    event = parse_event(payload)
    target = inbox.submit(payload, event_id=event.event_id)

    if args.json:
        print(
            json.dumps(
                {"event_id": event.event_id, "path": str(target)}, ensure_ascii=False, indent=2
            )
        )
    else:
        print(f"Submitted: {event.event_id}")
        print(f"Waiting at: {target}")
    return EXIT_OK


def _cmd_inbox_reconcile(args: argparse.Namespace) -> int:
    """Report what happened to events an interrupted run left behind."""
    inbox_root = args.inbox_dir if args.inbox_dir is not None else inbox_dir_from_env(_config_env())
    inbox = Inbox(inbox_root)
    inbox.ensure_layout()
    context = IngestionContext(
        inbox=inbox,
        store=RunStore(args.runs_dir),
        # Reconciliation reads a ledger and some manifests. It never fetches.
        market_source=_NoMarketSource(),
    )
    reports = reconcile(context, recover=args.recover)

    if args.json:
        print(
            json.dumps(
                {
                    "recovered": args.recover,
                    "orphans": [
                        {
                            "event_id": report.event_id,
                            "run_id": report.run_id,
                            "run_status": report.run_status,
                            "resolution": report.resolution,
                            "recovered_to": report.recovered_to,
                        }
                        for report in reports
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return EXIT_OK

    if not reports:
        print("No interrupted events. Nothing to reconcile.")
        return EXIT_OK

    print(f"Interrupted events: {len(reports)}")
    for report in reports:
        print(f"  {report.event_id}")
        print(f"    run:        {report.run_id or '(never reserved)'}")
        print(f"    resolution: {report.resolution}")
        if report.recovered_to:
            print(f"    moved to:   {report.recovered_to}")
    if not args.recover:
        print("\nRe-run with --recover to act on this report.")
    return EXIT_OK


class _NoMarketSource:
    """Stands in where a market source is structurally required but never used."""

    def load(self) -> Any:  # pragma: no cover - calling it is the bug
        raise AssertionError("this command does not fetch market data")


def _cmd_mt5_check(args: argparse.Namespace) -> int:
    """Read-only terminal diagnostic. Places no order and writes no Run."""
    settings = MarketDataSettings.from_env(_config_env())

    from goldpipeline.adapters.mt5_market import MetaTrader5MarketDataSource

    source = MetaTrader5MarketDataSource(
        settings, module=_fake_mt5_module() if args.fake_mt5 else None
    )

    report: dict[str, Any] = {
        "provider_symbol": settings.provider_symbol,
        "canonical_symbol": settings.canonical_symbol,
        "timeframe": settings.timeframe,
        "requested_bars": settings.bar_count,
        "max_data_age_minutes": settings.max_data_age_minutes,
        "connected": False,
    }

    try:
        loaded = source.load()
    except MarketDataError as exc:
        report["error"] = {"code": exc.code, "message": exc.message, "details": exc.details}
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        else:
            _print_mt5_report(report, stream=sys.stderr)
        return _classify_failure(exc)

    snapshot = loaded.model
    report.update(
        {
            "connected": True,
            "symbol_available": True,
            "returned_bars": len(snapshot.bars),
            "latest_closed_candle": snapshot.bars[-1].timestamp.isoformat().replace("+00:00", "Z"),
            "latest_close": str(snapshot.bars[-1].close),
            "provenance": loaded.provenance,
        }
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        _print_mt5_report(report, stream=sys.stdout)
    return EXIT_OK


def _print_mt5_report(report: dict[str, Any], *, stream: Any) -> None:
    print(
        f"MT5 terminal:      {'connected' if report['connected'] else 'not reachable'}", file=stream
    )
    print(f"Configured symbol: {report['provider_symbol']}", file=stream)
    print(f"Canonical symbol:  {report['canonical_symbol']}", file=stream)
    print(f"Timeframe:         {report['timeframe']}", file=stream)
    print(f"Requested bars:    {report['requested_bars']}", file=stream)
    if report["connected"]:
        print(f"Returned bars:     {report['returned_bars']}", file=stream)
        print(f"Latest closed:     {report['latest_closed_candle']}", file=stream)
        print(f"Latest close:      {report['latest_close']}", file=stream)
        print("Data quality:      OK (closed candle, within the age limit)", file=stream)
    if "error" in report:
        error = report["error"]
        print(f"Problem:           [{error['code']}] {error['message']}", file=stream)
        candidates = error["details"].get("candidates")
        if candidates:
            print(f"Symbols offered:   {', '.join(candidates)}", file=stream)
            print("                   (listed, never substituted)", file=stream)


def _worker_context(args: argparse.Namespace) -> WorkerContext:
    """Assemble a tick's dependencies. Builds no provider client.

    The publisher target is read here, from configuration, and only when
    unattended publishing is on - so a worker that is not publishing needs no
    Telegram credentials at all.
    """
    settings = AutomationSettings.from_env(_config_env())
    if args.automation_dir is not None:
        settings = replace(settings, automation_dir=args.automation_dir)

    inbox_root = args.inbox_dir if args.inbox_dir is not None else inbox_dir_from_env(_config_env())
    inbox = Inbox(inbox_root)
    inbox.ensure_layout()

    target: str | None = None
    if settings.auto_publish_enabled and not args.fake_publisher:
        target = TelegramSettings.from_env(_config_env(), secrets=_secret_provider()).target_chat
    elif settings.auto_publish_enabled:
        # The offline publisher has no destination of its own, so the allowlist
        # is checked against the stand-in's channel. Nothing leaves the machine.
        target = settings.auto_publish_allowed_target

    return WorkerContext(
        inbox=inbox,
        store=RunStore(args.runs_dir),
        automation=AutomationStore(settings.automation_dir),
        settings=settings,
        market_source=_market_source(args),
        clients=_pipeline_clients(args),
        expected_symbol=MarketDataSettings.from_env(_config_env()).canonical_symbol,
        publisher_target=target,
    )


def _cmd_automation_run_once(args: argparse.Namespace) -> int:
    """One tick, on purpose.

    Deliberately ignores ``GOLDPIPELINE_AUTOMATION_ENABLED``: a person typing
    this is the authorisation. The kill switch exists to stop the *scheduler*,
    not to stop an operator investigating.
    """
    if args.dry_run:
        return _report_dry_run(args)
    return _report_tick(run_tick(_worker_context(args)), as_json=args.json)


def _cmd_automation_worker_tick(args: argparse.Namespace) -> int:
    """What Task Scheduler runs, every minute.

    Honours the kill switch, so a registered task can be switched off without
    unregistering it - which matters because unregistering is the step people
    forget to undo.
    """
    settings = AutomationSettings.from_env(_config_env())
    if not settings.enabled:
        if args.json:
            print(json.dumps({"status": "DISABLED", "reason": AUTOMATION_ENABLED_ENV}, indent=2))
        else:
            print(f"Automation is disabled ({AUTOMATION_ENABLED_ENV}). Nothing was done.")
        return EXIT_OK
    return _report_tick(run_tick(_worker_context(args)), as_json=args.json)


def _report_tick(result: AutomationTickResult, *, as_json: bool) -> int:
    """Print a tick and choose an exit code a scheduler can live with."""
    code = _tick_exit_code(result)
    if as_json:
        print(result.model_dump_json(indent=2))
        return code

    stream = sys.stdout if code == EXIT_OK else sys.stderr
    print(f"Tick: {result.tick_id} ({result.status})", file=stream)
    print(f"Mode: {result.mode}", file=stream)
    print(f"Auto publish: {'ON' if result.auto_publish_enabled else 'OFF'}", file=stream)

    for label, items in (
        ("Reconciled", result.reconciled),
        ("Resumed", result.resumed_runs),
        ("Processed", result.processed_events),
        ("Deferred", result.deferred_events),
        ("Expired", result.expired_events),
        ("Blocked", result.blocked_runs),
    ):
        for item in items:
            suffix = f" (retry {item.next_attempt_at:%H:%M})" if item.next_attempt_at else ""
            print(
                f"  {label:<11} {item.identifier} {item.outcome}"
                f"{f' [{item.code}]' if item.code else ''}{suffix}",
                file=stream,
            )

    if not result.did_work:
        print("  nothing to do", file=stream)
    return code


def _tick_exit_code(result: AutomationTickResult) -> int:
    """Map a tick onto an exit code.

    A deferred event exits zero. It has to: with a minute schedule and a market
    that is shut two days a week, a non-zero exit for "the market is closed"
    would paint the Task Scheduler history red for the whole weekend and teach
    an operator to ignore it.
    """
    if result.status is TickStatus.FAILED:
        return EXIT_INVALID_DATA
    if result.status is TickStatus.BLOCKED:
        return EXIT_BLOCKED
    return EXIT_OK


def _report_dry_run(args: argparse.Namespace) -> int:
    """Report the work a tick would do, having done none of it.

    Reads directories and manifests. It claims no event, creates no Run, calls
    no provider and writes nothing at all - including the automation state.
    """
    context = _worker_context(args)
    settings = context.settings
    moment = utc_now()

    pending = [p.stem for p in context.inbox.pending()]
    deferred = _deferred_summary(context.inbox, moment)
    resumable: list[dict[str, str]] = []
    for run_id in context.store.list_run_ids():
        try:
            manifest = context.store.open(run_id).load_manifest()
        except (FileNotFoundError, ValueError, PipelineError):
            continue
        if may_resume(manifest.status, settings.auto_publish_enabled):
            resumable.append({"run_id": run_id, "status": str(manifest.status)})

    mode = PipelineMode.PUBLISH if settings.auto_publish_enabled else DEFAULT_MODE
    due = [str(item["event_id"]) for item in deferred if item["due"]]
    waiting = [str(item["event_id"]) for item in deferred if not item["due"]]

    if args.json:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "mode": str(mode),
                    "auto_publish_enabled": settings.auto_publish_enabled,
                    "would_resume": resumable,
                    "pending_events": pending[: settings.max_events_per_tick],
                    "pending_events_total": len(pending),
                    "deferred_due": due,
                    "deferred_waiting": waiting,
                    "max_events_per_tick": settings.max_events_per_tick,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return EXIT_OK

    print("Dry run: nothing was claimed, created, called or written.")
    print(f"Mode: {mode}")
    print(f"Auto publish: {'ON' if settings.auto_publish_enabled else 'OFF'}")
    print(f"Would resume: {len(resumable)} run(s)")
    for entry in resumable:
        print(f"  {entry['run_id']}  {entry['status']}")
    print(f"Pending events: {len(pending)} (cap {settings.max_events_per_tick} per tick)")
    for event_id in pending:
        print(f"  {event_id}")
    print(f"Deferred due now: {len(due)}")
    print(f"Deferred waiting: {len(waiting)}")
    return EXIT_OK


def _deferred_summary(inbox: Inbox, moment: datetime) -> list[dict[str, Any]]:
    """Which deferred events are due, without touching any of them."""
    directory = inbox.directory(DEFERRED)
    if not directory.is_dir():
        return []
    summary: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(".defer.json"):
            continue
        record = read_defer(path)
        summary.append(
            {
                "event_id": path.stem,
                "due": record is None or record.due(moment),
                "next_attempt_at": record.next_attempt_at.isoformat() if record else None,
                "reason_code": record.reason_code if record else None,
            }
        )
    return summary


def _cmd_automation_status(args: argparse.Namespace) -> int:
    """Read-only operational summary. Opens no socket and writes nothing."""
    settings = AutomationSettings.from_env(_config_env())
    if args.automation_dir is not None:
        settings = replace(settings, automation_dir=args.automation_dir)

    inbox_root = args.inbox_dir if args.inbox_dir is not None else inbox_dir_from_env(_config_env())
    inbox = Inbox(inbox_root)
    store = RunStore(args.runs_dir)
    state = AutomationStore(settings.automation_dir).read_state()
    moment = utc_now()

    counts: dict[str, int] = {}
    for run_id in store.list_run_ids():
        try:
            status = str(store.open(run_id).load_manifest().status)
        except (FileNotFoundError, ValueError, PipelineError):
            continue
        counts[status] = counts.get(status, 0) + 1

    deferred = _deferred_summary(inbox, moment)
    expired = inbox.directory(EXPIRED)
    report = {
        "automation_enabled": settings.enabled,
        "auto_publish_enabled": settings.auto_publish_enabled,
        "allowed_target": settings.auto_publish_allowed_target,
        "last_tick_id": state.last_tick_id,
        "last_tick_status": state.last_tick_status,
        "last_tick_completed_at": (
            state.last_tick_completed_at.isoformat() if state.last_tick_completed_at else None
        ),
        "pending_events": len(inbox.pending()),
        "deferred_events": len(deferred),
        "expired_events": len(list(expired.glob("*.json"))) if expired.is_dir() else 0,
        "run_status_counts": counts,
        "last_error_safe": state.last_error_safe,
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return EXIT_OK

    print(f"Automation:        {'enabled' if settings.enabled else 'disabled'}")
    print(f"Auto publish:      {'ON' if settings.auto_publish_enabled else 'OFF'}")
    print(f"Last tick:         {state.last_tick_id or 'never'} ({state.last_tick_status or '-'})")
    if state.last_tick_completed_at:
        print(f"Last completed:    {state.last_tick_completed_at.isoformat()}")
    print(f"Pending inbox:     {report['pending_events']}")
    print(f"Deferred:          {report['deferred_events']}")
    print(f"Expired:           {report['expired_events']}")
    print(f"READY_TO_PUBLISH:  {counts.get('READY_TO_PUBLISH', 0)}")
    print(f"PUBLISH_UNCERTAIN: {counts.get('PUBLISH_UNCERTAIN', 0)}")
    print(f"PUBLISH_BLOCKED:   {counts.get('PUBLISH_BLOCKED', 0)}")
    print(f"Last error:        {state.last_error_safe or 'none'}")
    return EXIT_OK


def _cmd_automation_preflight(args: argparse.Namespace) -> int:
    """Report whether unattended operation is configured, naming no values."""
    settings = AutomationSettings.from_env(_config_env())
    backend = inspect_backend()
    statuses = {status.name: status for status in _secret_statuses()}
    report: dict[str, Any] = {
        "automation_enabled": settings.enabled,
        "auto_publish_enabled": settings.auto_publish_enabled,
        "anthropic": statuses[SecretName.ANTHROPIC_API_KEY].summary,
        "openai": statuses[SecretName.OPENAI_API_KEY].summary,
        "telegram": statuses[SecretName.TELEGRAM_BOT_TOKEN].summary,
        "credential_backend": backend.backend,
        "credential_backend_secure": backend.secure,
        "mt5": _mt5_available(args),
        "allowed_target": settings.auto_publish_allowed_target,
        # The destination is configuration, not a credential, so it is reported
        # whether or not publishing is on. Reading it needs no Telegram token,
        # and an operator checking the allowlist should be able to see both
        # halves of the comparison before enabling anything.
        "configured_target": _config_env().resolve(ConfigKey.TELEGRAM_TARGET_CHAT_ID).value,
        "blockers": [],
    }

    blockers: list[str] = report["blockers"]
    if report["mt5"] != "available":
        blockers.append("MT5 is not reachable; new events will defer rather than run")
    if not statuses[SecretName.ANTHROPIC_API_KEY].configured or not (
        statuses[SecretName.OPENAI_API_KEY].configured
    ):
        # Only a blocker for *new* work. A Run resumed at the gate needs neither.
        blockers.append("writer/reviewer credentials are missing; new events cannot be drafted")

    if not backend.ready:
        # Not a blocker on its own - process-environment credentials still
        # work for a person at a keyboard. It *is* a blocker for a scheduled
        # task, which inherits no session.
        blockers.append(
            "no secure credential store is available, so a scheduled task would find no credentials"
        )

    session_only = [
        str(name) for name, status in statuses.items() if status.source is SecretSource.PROCESS_ENV
    ]
    if session_only:
        # The trap this whole round exists to close: variables set with
        # `$env:` live in one process tree, and Task Scheduler starts a new one.
        blockers.append(
            f"{', '.join(session_only)} come from this session only; "
            "a scheduled task will not see them"
        )

    if settings.auto_publish_enabled:
        if not statuses[SecretName.TELEGRAM_BOT_TOKEN].configured:
            blockers.append("unattended publishing is on but Telegram is not configured")
        else:
            configured = TelegramSettings.from_env(
                _config_env(), secrets=_secret_provider()
            ).target_chat
            report["configured_target"] = configured
            if settings.auto_publish_allowed_target != configured:  # noqa: SIM102
                blockers.append(
                    "the allowlisted target and the configured target differ; "
                    "nothing would be published"
                )

    report["task_readiness"] = "READY" if not blockers else "NOT_READY"

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return EXIT_OK if not blockers else EXIT_BLOCKED

    print(f"Credential store:  {backend.backend}")
    print(f"                   {'secure' if backend.secure else 'NOT trusted'}")
    print(f"MT5:               {report['mt5']}")
    print(f"Anthropic:         {report['anthropic']}")
    print(f"OpenAI:            {report['openai']}")
    print(f"Telegram:          {report['telegram']}")
    print(f"Automation:        {'enabled' if settings.enabled else 'disabled'}")
    print(f"Auto publish:      {'ON' if settings.auto_publish_enabled else 'OFF'}")
    print(f"Allowed target:    {settings.auto_publish_allowed_target or '(none)'}")
    print(f"Configured target: {report['configured_target'] or '(not read)'}")
    print(f"Task readiness:    {report['task_readiness']}")
    for blocker in blockers:
        print(f"  - {blocker}")
    print("\nNo credential values are printed by this command, only their presence.")
    return EXIT_OK if not blockers else EXIT_BLOCKED


def _configured(build: Any) -> str:
    """Whether a settings object can be built, without keeping what it holds."""
    try:
        build()
    except PipelineError:
        return "missing"
    return "configured"


def _mt5_available(args: argparse.Namespace) -> str:
    """Whether the market source answers, without creating anything."""
    try:
        _market_source(args).load()
    except PipelineError as exc:
        return f"unavailable ({exc.code})"
    except _UsageError:
        return "unavailable (not configured)"
    return "available"


def _cmd_automation_task_plan(args: argparse.Namespace) -> int:
    """Print the Task Scheduler definition. Registers nothing, ever.

    There is deliberately no ``--apply`` and no install command: registering a
    task that runs every minute is a decision to make while looking at the
    definition, not a side effect of printing it.
    """
    plan = build_plan(
        task_name=args.task_name or DEFAULT_TASK_NAME,
        interval_minutes=args.interval_minutes or DEFAULT_INTERVAL_MINUTES,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "task_name": plan.task_name,
                    "executable": str(plan.executable),
                    "arguments": plan.arguments,
                    "working_directory": str(plan.working_directory),
                    "interval_minutes": plan.interval_minutes,
                    "multiple_instances_policy": "IgnoreNew",
                    "principal": "InteractiveToken",
                    "executable_exists": plan.executable_exists,
                    "registered": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return EXIT_OK

    if args.xml:
        print(plan.to_xml())
        return EXIT_OK

    print(f"Task name:         {plan.task_name}")
    print(f"Executable:        {plan.executable}")
    print(f"Arguments:         {plan.arguments}")
    print(f"Working directory: {plan.working_directory}")
    print(f"Schedule:          every {plan.interval_minutes} minute(s)")
    print("Multiple instances: IgnoreNew")
    print("Principal:         the interactive user (never SYSTEM - MT5 needs a desktop session)")
    print(f"Interpreter found: {'yes' if plan.executable_exists else 'NO - check the path'}")
    print("\nThis task is NOT registered. Print the XML with --xml and import it")
    print("deliberately; no command here writes to Task Scheduler.")
    print("No credential appears in the definition - the task inherits the")
    print("environment it is registered in, which is a decision still to be made.")
    return EXIT_OK


_SECRET_CHOICES = {
    "anthropic": SecretName.ANTHROPIC_API_KEY,
    "openai": SecretName.OPENAI_API_KEY,
    "telegram": SecretName.TELEGRAM_BOT_TOKEN,
}
"""Short names an operator types, mapped to the credential each stands for."""


def _credential_store() -> WindowsCredentialSecretProvider:
    """The operating system's credential store.

    A single seam, so tests can substitute an offline store without any test in
    this repository touching a real credential manager.
    """
    return WindowsCredentialSecretProvider()


def _secret_provider() -> CompositeSecretProvider:
    """Process environment first, then the OS credential store.

    The credential store is only added when its backend is actually secure and
    reachable. On a machine without one this degrades to exactly the behaviour
    every earlier round had - environment variables - rather than raising on
    every lookup.
    """
    providers: list[SecretProvider] = [EnvironmentSecretProvider()]
    if inspect_backend().ready:
        providers.append(_credential_store())
    return CompositeSecretProvider(providers)


def _secret_statuses() -> list[SecretStatus]:
    """Resolve every credential's availability, keeping no value.

    The value returned by the provider is deliberately dropped on the same line
    it is produced: this function exists to answer "is it there?", and holding
    the answer any longer than that would be the beginning of a leak.
    """
    provider = _secret_provider()
    statuses: list[SecretStatus] = []
    for name in SecretName:
        try:
            value, source = provider.resolve(name)
        except PipelineError as exc:
            statuses.append(
                SecretStatus(name=name, configured=False, detail=f"[{exc.code}] {exc.message}")
            )
            continue
        statuses.append(SecretStatus(name=name, configured=value is not None, source=source))
    return statuses


def _cmd_secrets_status(args: argparse.Namespace) -> int:
    """Report credential availability. Never prints a value."""
    report = inspect_backend()
    statuses = _secret_statuses()

    if args.json:
        print(
            json.dumps(
                {
                    "backend": report.backend,
                    "backend_available": report.available,
                    "backend_secure": report.secure,
                    "backend_detail": report.detail,
                    "secrets": [json.loads(status.model_dump_json()) for status in statuses],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return EXIT_OK

    print(f"Credential backend: {report.backend}")
    print(f"                    {'secure' if report.secure else 'NOT trusted'} - {report.detail}")
    print()
    for status in statuses:
        print(f"{status.name:<20} {status.summary}")
        if status.detail:
            print(f"{'':<20} {status.detail}")
    print("\nValues are never printed by this command, only their presence and source.")
    return EXIT_OK


def _cmd_secrets_set(args: argparse.Namespace) -> int:
    """Store one credential, prompting invisibly.

    The secret is typed straight into this process and handed to the credential
    store. It never appears as an argument, so it cannot reach shell history, a
    process listing, or a terminal transcript.
    """
    import getpass

    name = _SECRET_CHOICES[args.name]
    report = inspect_backend()
    if not report.ready:
        print(f"Credential backend: {report.backend}", file=sys.stderr)
        print(f"Refusing to store {name}: {report.detail}", file=sys.stderr)
        print(
            "Nothing was written. No file or environment variable was created as a fallback.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    print(f"Storing {name} in {report.backend}.")
    print("The value is not echoed and is not saved to shell history.")
    try:
        value = getpass.getpass(f"{name}: ")
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled. Nothing was written.", file=sys.stderr)
        return EXIT_ERROR

    if not value.strip():
        print("Empty value. Nothing was written.", file=sys.stderr)
        return EXIT_INVALID_DATA

    _credential_store().set_secret(name, value)
    print(f"Stored. {name} is now available to a scheduled task running as this user.")
    return EXIT_OK


def _cmd_secrets_delete(args: argparse.Namespace) -> int:
    """Remove one credential, and only that one."""
    name = _SECRET_CHOICES[args.name]
    try:
        _credential_store().delete_secret(name)
    except CredentialNotFoundError:
        print(f"No stored credential for {name}. Nothing to remove.")
        return EXIT_OK
    print(f"Removed {name} from the credential store.")
    return EXIT_OK


def _config_store() -> RuntimeConfigStore:
    """The persisted settings file. A seam, so tests never touch the real one."""
    return RuntimeConfigStore()


def _config_env() -> LayeredConfig:
    """Non-secret configuration: process environment first, then the file.

    Every settings loader reads through this, so a scheduled task - which
    inherits no session - still finds the symbol, the timeframe and the flags
    that a person chose once.
    """
    return LayeredConfig(os.environ, _config_store().load())


def _cmd_config_status(args: argparse.Namespace) -> int:
    """Show every persistent setting and which layer supplied it."""
    layered = _config_env()
    entries = [layered.resolve(key) for key in ConfigKey]

    if args.json:
        print(
            json.dumps(
                {
                    "path": str(_config_store().path),
                    "settings": [json.loads(entry.model_dump_json()) for entry in entries],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return EXIT_OK

    print(f"Config file: {_config_store().path}")
    print()
    for entry in entries:
        print(f"{entry.key:<45} {entry.summary}")
    print("\nThis file holds no credentials. Those live in the OS credential store.")
    return EXIT_OK


def _cmd_config_set(args: argparse.Namespace) -> int:
    """Persist one non-secret setting."""
    key = parse_key(args.name)
    _config_store().set(key, args.value)
    entry = _config_env().resolve(key)
    print(f"Set {key} = {args.value}")
    if entry.source is ConfigSource.PROCESS_ENV:
        # Worth saying: the value was written, but something in this shell is
        # shadowing it, so what the operator just checked is not what a
        # scheduled task will read.
        print(
            f"Note: this session's environment also sets {key}, and that wins here. "
            "A scheduled task will use the persisted value.",
            file=sys.stderr,
        )
    return EXIT_OK


def _cmd_config_delete(args: argparse.Namespace) -> int:
    """Remove one persistent setting."""
    key = parse_key(args.name)
    _config_store().delete(key)
    print(f"Removed {key}. The built-in default applies unless the environment sets it.")
    return EXIT_OK


# --------------------------------------------------------------------------
# the scheduled task
# --------------------------------------------------------------------------


def _task_scheduler() -> TaskSchedulerAdapter:
    """The real Windows Task Scheduler. A seam, so tests stay offline."""
    return PowerShellTaskScheduler()


def _task_name(args: argparse.Namespace) -> str:
    return getattr(args, "task_name", None) or DEFAULT_TASK_NAME


def _cmd_automation_task_install(args: argparse.Namespace) -> int:
    """Register the scheduled task, but only when told to twice.

    Printing the plan is the default because registering something that runs
    every minute deserves to be read first. ``--apply`` is the second word.
    """
    plan = build_plan(
        task_name=_task_name(args),
        interval_minutes=args.interval_minutes or DEFAULT_INTERVAL_MINUTES,
    )
    scheduler = _task_scheduler()
    existing = scheduler.query(plan.task_name)

    if existing.installed:
        # Never silently replaced: it may be an older definition, or something a
        # person created by hand. A match is fine; a mismatch is a refusal.
        compare(
            existing,
            executable=str(plan.executable),
            arguments=plan.arguments,
            working_directory=str(plan.working_directory),
        )
        if not args.json:
            print(f"Already installed: {plan.task_name}")
            print("The registered definition matches. Nothing was changed.")
        return _report_task(scheduler.query(plan.task_name), plan, as_json=args.json)

    if not args.apply:
        print(f"Would register: {plan.task_name}")
        print(f"  Executable:        {plan.executable}")
        print(f"  Arguments:         {plan.arguments}")
        print(f"  Working directory: {plan.working_directory}")
        print(f"  Schedule:          every {plan.interval_minutes} minute(s)")
        print("  Multiple instances: IgnoreNew")
        print("  Principal:         the interactive user (never SYSTEM)")
        print("\nNothing was changed. Re-run with --apply to register it.")
        return EXIT_OK

    scheduler.install(plan.task_name, plan.to_xml())
    info = scheduler.query(plan.task_name)
    if info.runs_as_system:
        # Registered under the wrong principal: it could not see the MetaTrader
        # window and could not read credentials stored against the user's login.
        print(
            f"Registered as {info.task_user}, which is SYSTEM. That account cannot "
            "reach the interactive desktop or this user's credentials. Remove it "
            "with `automation-task-remove --apply` and investigate.",
            file=sys.stderr,
        )
        return EXIT_BLOCKED

    if not args.json:
        # `--json` means machine-readable, and a prose line in front of the
        # document is the difference between parsing and a support ticket.
        print(f"Registered: {plan.task_name}")
    return _report_task(info, plan, as_json=args.json)


def _cmd_automation_task_status(args: argparse.Namespace) -> int:
    """Read-only status of the registered task."""
    plan = build_plan(task_name=_task_name(args))
    info = _task_scheduler().query(plan.task_name)
    return _report_task(info, plan, as_json=args.json)


def _cmd_automation_task_remove(args: argparse.Namespace) -> int:
    """Unregister the task, when told to twice."""
    name = _task_name(args)
    scheduler = _task_scheduler()
    if not scheduler.query(name).installed:
        print(f"No task named {name} is registered. Nothing to remove.")
        return EXIT_OK

    if not args.apply:
        print(f"Would remove: {name}")
        print("Nothing was changed. Re-run with --apply to remove it.")
        return EXIT_OK

    scheduler.remove(name)
    print(f"Removed: {name}")
    return EXIT_OK


def _report_task(info: TaskInfo, plan: Any, *, as_json: bool) -> int:
    """Print a task's status. Contains no credential, by construction."""
    if as_json:
        print(
            json.dumps(
                {
                    "task_name": plan.task_name,
                    "installed": info.installed,
                    "enabled": info.enabled,
                    "state": info.state,
                    "task_user": info.task_user,
                    "logon_type": info.logon_type,
                    "multiple_instances_policy": info.multiple_instances_policy,
                    "executable": info.executable,
                    "arguments": info.arguments,
                    "working_directory": info.working_directory,
                    "last_run_time": info.last_run_time,
                    "last_result": info.last_result,
                    "next_run_time": info.next_run_time,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return EXIT_OK if info.installed else EXIT_BLOCKED

    if not info.installed:
        print(f"Task: {plan.task_name}")
        print("Installed: NO")
        print("\nRegister it with `automation-task-install --apply`.")
        return EXIT_BLOCKED

    print(f"Task:               {plan.task_name}")
    print("Installed:          YES")
    print(f"Enabled:            {'YES' if info.enabled else 'NO'} ({info.state})")
    print(f"Runs as:            {info.task_user} ({info.logon_type})")
    print(f"Multiple instances: {info.multiple_instances_policy}")
    print(f"Executable:         {info.executable}")
    print(f"Arguments:          {info.arguments}")
    print(f"Working directory:  {info.working_directory}")
    print(f"Last run:           {info.last_run_time or 'never'}")
    print(f"Last result:        {info.last_result if info.last_result is not None else '-'}")
    print(f"Next run:           {info.next_run_time or '-'}")
    if info.runs_as_system:
        print("\nWARNING: this task runs as SYSTEM. It cannot see the MetaTrader")
        print("window or read this user's credentials.", file=sys.stderr)
    return EXIT_OK


def _cmd_show_run(args: argparse.Namespace) -> int:
    store = RunStore(args.runs_dir)
    try:
        run = store.open(args.run_id)
    except (FileNotFoundError, ValueError) as exc:
        print(f"{exc}", file=sys.stderr)
        return EXIT_ERROR

    manifest = run.load_manifest()
    print(f"Run: {manifest.run_id}")
    print(f"Status: {manifest.status}")
    print(f"Created: {manifest.created_at.isoformat()}")
    print(f"Pipeline: {manifest.pipeline_version} / schema {manifest.schema_version}")
    print("Files:")
    for ref in [*manifest.source_files, *manifest.artifact_files]:
        print(f"  {ref.name:<22} {ref.size_bytes:>8} bytes  sha256={ref.sha256[:12]}...")
    if manifest.error:
        print(f"Error: [{manifest.error.code}] {manifest.error.message}")
    return EXIT_OK


def _cmd_list_runs(args: argparse.Namespace) -> int:
    run_ids = RunStore(args.runs_dir).list_run_ids()
    if not run_ids:
        print(f"No runs under {args.runs_dir}")
        return EXIT_OK
    for run_id in run_ids:
        print(run_id)
    return EXIT_OK


def _force_utf8_output() -> None:
    """Make stdout/stderr UTF-8 regardless of the console code page.

    A Windows console defaults to a legacy code page (cp1252 here), which cannot
    encode Vietnamese. Any path, warning or error message containing Vietnamese
    would then crash the CLI with a UnicodeEncodeError instead of printing -
    and this pipeline exists to handle Vietnamese text.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        # a detached or test-captured stream may refuse reconfiguration
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    _force_utf8_output()
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    handlers = {
        "create-run": _cmd_create_run,
        "write-draft": _cmd_write_draft,
        "review-draft": _cmd_review_draft,
        "finalize": _cmd_finalize,
        "gate-publish": _cmd_gate_publish,
        "publish": _cmd_publish,
        "pipeline-run": _cmd_pipeline_run,
        "pipeline-resume": _cmd_pipeline_resume,
        "pipeline-ingest": _cmd_pipeline_ingest,
        "inbox-process-one": _cmd_inbox_process_one,
        "inbox-submit": _cmd_inbox_submit,
        "inbox-reconcile": _cmd_inbox_reconcile,
        "mt5-check": _cmd_mt5_check,
        "automation-run-once": _cmd_automation_run_once,
        "automation-worker-tick": _cmd_automation_worker_tick,
        "automation-status": _cmd_automation_status,
        "automation-preflight": _cmd_automation_preflight,
        "automation-task-plan": _cmd_automation_task_plan,
        "secrets-status": _cmd_secrets_status,
        "secrets-set": _cmd_secrets_set,
        "secrets-delete": _cmd_secrets_delete,
        "config-status": _cmd_config_status,
        "config-set": _cmd_config_set,
        "config-delete": _cmd_config_delete,
        "automation-task-install": _cmd_automation_task_install,
        "automation-task-status": _cmd_automation_task_status,
        "automation-task-remove": _cmd_automation_task_remove,
        "show-run": _cmd_show_run,
        "list-runs": _cmd_list_runs,
    }
    try:
        return handlers[args.command](args)
    except _UsageError as exc:
        print(f"{exc}", file=sys.stderr)
        return EXIT_ERROR
    except RunLockedError as exc:
        # Not a data problem and not a gate declining: someone else is driving
        # this Run right now, or a crash left a lock behind for a human.
        print(f"{exc}", file=sys.stderr)
        print(f"Details: {json.dumps(exc.details, ensure_ascii=False)}", file=sys.stderr)
        return EXIT_ERROR
    except FinalizationBlockedError as exc:
        print(f"Finalization blocked: {exc}", file=sys.stderr)
        return EXIT_BLOCKED
    except PipelineError as exc:
        return _report_failure(exc, as_json=getattr(args, "json", False))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
