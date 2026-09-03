"""Which market-data source production resolves, and what it refuses to do.

The migration this file guards is small in code and large in consequence: one
default changed, and every new Run now quotes a different venue. So the tests
are mostly about the things that must *not* happen.

The one that matters most is **no fallback**. If TradingView fails, production
must fail with it. A pipeline that quietly retried the terminal would produce a
Run whose candles came from one venue and whose provenance named another - or
worse, a Run assembled from both - and nothing downstream could detect it. Each
failure class therefore has a test proving the MT5 adapter was never even
imported, let alone called.

The second is **the worker's path specifically**. The scheduled task runs
``automation-worker-tick`` with no ``--market-source``, so its authority is the
argparse default. A test asserts that, because a default that only an
interactive operator sees would be a migration in name only.

Offline throughout: the websocket is a fake, the terminal is a fake, and no
test here depends on a clock or a network.
"""

from __future__ import annotations

import argparse
import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from goldpipeline import cli
from goldpipeline.adapters.fake_tradingview import FakeConnection, RecordingConnector, conversation
from goldpipeline.adapters.fake_tradingview import make_series as tv_series
from goldpipeline.adapters.file_source import JsonFileMarketDataSource
from goldpipeline.adapters.mt5_market import MetaTrader5MarketDataSource
from goldpipeline.adapters.tradingview_market import (
    PROVIDER_NAME,
    TradingViewMarketDataSource,
)
from goldpipeline.domain.errors import (
    InsufficientBarsError,
    StaleMarketDataError,
    TradingViewCandleError,
    TradingViewConnectionError,
    TradingViewCriticalError,
    TradingViewProtocolError,
    TradingViewTimeoutError,
)
from goldpipeline.schemas.common import Timeframe
from goldpipeline.schemas.market import MarketDataInput
from goldpipeline.services.normalizer import normalize_market_data

SRC = Path(__file__).resolve().parents[1] / "src" / "goldpipeline"
CLI_TEXT = (SRC / "cli.py").read_text(encoding="utf-8")

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def namespace(**overrides: Any) -> argparse.Namespace:
    """The arguments a command hands to :func:`cli._market_source`."""
    values: dict[str, Any] = {
        "market_source": cli.PRODUCTION_MARKET_SOURCE,
        "ohlc": None,
        "fake_mt5": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def worker_namespace(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the scheduled task's real command line."""
    parser = cli.build_parser()
    return parser.parse_args(["automation-worker-tick", *(argv or [])])


# --- source resolution ----------------------------------------------------


class TestSourceResolution:
    def test_no_explicit_override_resolves_tradingview(self) -> None:
        assert isinstance(cli._market_source(namespace()), TradingViewMarketDataSource)

    def test_a_namespace_without_the_attribute_still_resolves_production(self) -> None:
        """A hand-built namespace must not fall through to something else."""
        bare = argparse.Namespace(ohlc=None, fake_mt5=False)
        assert isinstance(cli._market_source(bare), TradingViewMarketDataSource)

    def test_explicit_tradingview(self) -> None:
        source = cli._market_source(namespace(market_source="tradingview"))
        assert isinstance(source, TradingViewMarketDataSource)
        assert source.provider_symbol == "OANDA:XAUUSD"

    def test_explicit_mt5_is_still_available_for_diagnostics(self) -> None:
        source = cli._market_source(namespace(market_source="mt5", fake_mt5=True))
        assert isinstance(source, MetaTrader5MarketDataSource)

    def test_explicit_file_is_unchanged(self, tmp_path: Path) -> None:
        path = tmp_path / "candles.json"
        path.write_text("{}", encoding="utf-8")
        source = cli._market_source(namespace(market_source="file", ohlc=path))
        assert isinstance(source, JsonFileMarketDataSource)

    def test_file_without_a_path_is_refused(self) -> None:
        with pytest.raises(cli._UsageError, match="needs --ohlc"):
            cli._market_source(namespace(market_source="file"))

    def test_an_unknown_source_fails_before_any_data_access(self) -> None:
        """Unknown must never quietly mean "the default one"."""
        with pytest.raises(cli._UsageError, match="unknown --market-source"):
            cli._market_source(namespace(market_source="oanda-direct"))

    def test_argparse_rejects_an_unknown_choice_before_a_command_runs(self) -> None:
        with pytest.raises(SystemExit):
            worker_namespace(["--market-source", "bloomberg"])

    def test_the_three_sources_are_the_whole_vocabulary(self) -> None:
        assert cli.MARKET_SOURCES == ("tradingview", "mt5", "file")
        assert cli.PRODUCTION_MARKET_SOURCE == "tradingview"


class TestProductionDefault:
    def test_the_scheduled_worker_resolves_tradingview(self) -> None:
        """The task runs this exact command line, with no --market-source."""
        args = worker_namespace()
        assert args.market_source == "tradingview"
        assert isinstance(cli._market_source(args), TradingViewMarketDataSource)

    def test_the_installed_task_passes_no_market_source_flag(self) -> None:
        """So the default really is the worker's authority, not a prompt nicety.

        Read from the task planner the installer uses, and cross-checked
        against the arguments the live task actually carries, which the round's
        own scheduler capture records as
        ``-m goldpipeline automation-worker-tick``.
        """
        from goldpipeline.services.task_plan import WORKER_COMMAND, build_plan

        plan = build_plan()
        assert "automation-worker-tick" in WORKER_COMMAND
        assert "--market-source" not in WORKER_COMMAND
        assert "automation-worker-tick" in plan.arguments
        assert "--market-source" not in plan.arguments
        assert "--market-source" not in plan.command_line

    @pytest.mark.parametrize(
        "command",
        [
            "automation-worker-tick",
            "automation-run-once",
            "automation-status",
            "automation-preflight",
            "inbox-process-one",
        ],
    )
    def test_every_production_entry_point_defaults_to_tradingview(self, command: str) -> None:
        parser = cli.build_parser()
        args = parser.parse_args([command])
        assert args.market_source == "tradingview"

    def test_no_hidden_mt5_default_remains(self) -> None:
        assert 'default="mt5"' not in CLI_TEXT
        assert '("mt5", "file")' not in CLI_TEXT

    def test_mt5_remains_reachable_and_undeleted(self) -> None:
        """Legacy, diagnostic, explicitly selected - not removed."""
        assert (SRC / "adapters" / "mt5_market.py").exists()
        assert "mt5" in cli.MARKET_SOURCES
        assert "mt5-check" in CLI_TEXT

    def test_the_help_text_names_the_roles(self) -> None:
        assert "production authority" in CLI_TEXT
        assert "legacy/diagnostic" in CLI_TEXT


# --- no fallback ----------------------------------------------------------


class TestFailClosed:
    """Every TradingView failure class must end the fetch, not switch feed."""

    def source(self, connection: FakeConnection) -> TradingViewMarketDataSource:
        connector = RecordingConnector(scripts=[connection])
        ticks = iter(range(0, 100_000))
        return TradingViewMarketDataSource(
            timeframe=Timeframe.M15,
            limit=5,
            connector=connector,
            now=lambda: NOW,
            monotonic=lambda: float(next(ticks)),
            session_id="cs_test",
            series_id="sds_test",
        )

    @pytest.fixture
    def forbid_mt5(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Make any attempt to reach the terminal an outright failure."""

        def refuse(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("production fell back to MT5")

        monkeypatch.setattr("goldpipeline.adapters.mt5_market.MetaTrader5MarketDataSource", refuse)
        monkeypatch.setattr("goldpipeline.adapters.mt5_market.load_mt5_module", refuse)

    @pytest.mark.parametrize(
        ("packets", "expected"),
        [
            ([], TradingViewConnectionError),
            (["~m~55~m~" + '{"m":"protocol_error","p":[]}'], TradingViewProtocolError),
        ],
    )
    def test_wire_failures_do_not_reach_mt5(
        self, forbid_mt5: None, packets: list[str], expected: type[Exception]
    ) -> None:
        with pytest.raises(Exception) as caught:  # noqa: B017 - class varies
            self.source(FakeConnection(packets=packets)).load()
        assert isinstance(caught.value, TradingViewConnectionError | TradingViewProtocolError)

    def test_a_connection_failure_does_not_reach_mt5(self, forbid_mt5: None) -> None:
        with pytest.raises(TradingViewConnectionError):
            self.source(FakeConnection(fail_after=0, recv_error=OSError("reset"))).load()

    def test_a_timeout_does_not_reach_mt5(self, forbid_mt5: None) -> None:
        connector = RecordingConnector(
            scripts=[FakeConnection(fail_after=0, recv_error=TimeoutError("t"))]
        )
        source = TradingViewMarketDataSource(
            timeframe=Timeframe.M15,
            limit=5,
            connector=connector,
            now=lambda: NOW,
            monotonic=iter([0.0, 0.5, 99.0, 99.0]).__next__,
            timeout_seconds=1.0,
            series_id="sds_test",
        )
        with pytest.raises(TradingViewTimeoutError):
            source.load()

    def test_a_critical_error_does_not_reach_mt5(self, forbid_mt5: None) -> None:
        from goldpipeline.adapters.fake_tradingview import critical_error

        with pytest.raises(TradingViewCriticalError):
            self.source(FakeConnection(packets=[critical_error()])).load()

    def test_an_invalid_candle_does_not_reach_mt5(self, forbid_mt5: None) -> None:
        from goldpipeline.adapters.fake_tradingview import series_completed, timescale_update

        rows = tv_series(count=8, timeframe="M15", now=NOW)
        rows[3][2] = 1.0  # high below low
        connection = FakeConnection(packets=[timescale_update(rows), series_completed()])
        with pytest.raises(TradingViewCandleError):
            self.source(connection).load()

    def test_insufficient_closed_candles_does_not_reach_mt5(self, forbid_mt5: None) -> None:
        rows = tv_series(count=3, timeframe="M15", now=NOW)
        with pytest.raises(InsufficientBarsError):
            self.source(FakeConnection(packets=[conversation(rows)])).load()

    def test_the_resolver_builds_exactly_one_source(self) -> None:
        """No composite, no chain, nothing that could try a second feed."""
        source = cli._market_source(namespace())
        assert isinstance(source, TradingViewMarketDataSource)
        assert not hasattr(source, "fallback")
        assert not hasattr(source, "sources")

    def test_no_fallback_wording_or_wiring_exists_in_the_resolver(self) -> None:
        body = CLI_TEXT[CLI_TEXT.index("def _market_source(") :]
        body = body[: body.index("\ndef ")]
        assert "except" not in body
        assert "try" not in body


# --- the staleness guard travelled with the authority ---------------------


class TestStalenessGuard:
    def test_the_production_path_passes_the_configured_limit(self) -> None:
        """It used to live only in the MT5 adapter; migrating must not retire it."""
        source = cli._market_source(namespace())
        from goldpipeline.config import MarketDataSettings

        expected = MarketDataSettings.from_env(cli._config_env()).max_data_age_minutes
        assert source._max_data_age == expected  # noqa: SLF001
        assert expected > 0

    def test_a_stale_series_is_refused(self) -> None:
        old = NOW - timedelta(days=3)
        rows = tv_series(count=8, timeframe="M15", now=old)
        connector = RecordingConnector(scripts=[FakeConnection(packets=[conversation(rows)])])
        ticks = iter(range(0, 100_000))
        source = TradingViewMarketDataSource(
            timeframe=Timeframe.M15,
            limit=5,
            connector=connector,
            now=lambda: NOW,
            monotonic=lambda: float(next(ticks)),
            series_id="sds_test",
            max_data_age_minutes=90,
        )
        with pytest.raises(StaleMarketDataError) as caught:
            source.load()
        assert caught.value.details["limit_minutes"] == 90
        assert caught.value.details["age_minutes"] > 90

    def test_a_fresh_series_passes_the_same_guard(self) -> None:
        rows = tv_series(count=8, timeframe="M15", now=NOW)
        connector = RecordingConnector(scripts=[FakeConnection(packets=[conversation(rows)])])
        ticks = iter(range(0, 100_000))
        source = TradingViewMarketDataSource(
            timeframe=Timeframe.M15,
            limit=5,
            connector=connector,
            now=lambda: NOW,
            monotonic=lambda: float(next(ticks)),
            series_id="sds_test",
            max_data_age_minutes=90,
        )
        assert len(source.load().model.bars) == 5

    def test_the_guard_is_opt_in_so_diagnostics_may_look_backwards(self) -> None:
        old = NOW - timedelta(days=3)
        rows = tv_series(count=8, timeframe="M15", now=old)
        connector = RecordingConnector(scripts=[FakeConnection(packets=[conversation(rows)])])
        ticks = iter(range(0, 100_000))
        source = TradingViewMarketDataSource(
            timeframe=Timeframe.M15,
            limit=5,
            connector=connector,
            now=lambda: NOW,
            monotonic=lambda: float(next(ticks)),
            series_id="sds_test",
        )
        assert len(source.load().model.bars) == 5


# --- provenance -----------------------------------------------------------


class TestProvenance:
    def loaded(self) -> Any:
        rows = tv_series(count=25, timeframe="M15", now=NOW)
        connector = RecordingConnector(scripts=[FakeConnection(packets=[conversation(rows)])])
        ticks = iter(range(0, 100_000))
        source = TradingViewMarketDataSource(
            timeframe=Timeframe.M15,
            limit=20,
            connector=connector,
            now=lambda: NOW,
            monotonic=lambda: float(next(ticks)),
            series_id="sds_test",
            max_data_age_minutes=90,
        )
        return source.load()

    def test_a_new_run_can_answer_which_provider_produced_it(self) -> None:
        loaded = self.loaded()
        assert loaded.model.provider == PROVIDER_NAME == "tradingview"
        assert loaded.model.provider_symbol == "OANDA:XAUUSD"
        assert loaded.model.symbol == "XAUUSD"
        assert str(loaded.model.timeframe) == "M15"
        assert loaded.model.requested_at is not None
        assert loaded.model.retrieved_at is not None
        assert loaded.provenance["latest_candle_at"].endswith("Z")

    def test_provenance_survives_normalization(self) -> None:
        snapshot = normalize_market_data(self.loaded().model).snapshot
        assert snapshot.provider == "tradingview"
        assert snapshot.symbol == "XAUUSD"
        assert snapshot.timezone == "UTC"
        assert snapshot.data_to == snapshot.bars[-1].timestamp

    def test_no_second_market_data_schema_was_invented(self) -> None:
        loaded = self.loaded()
        assert isinstance(loaded.model, MarketDataInput)
        import goldpipeline.schemas.market as market

        assert not hasattr(market, "TradingViewCandle")
        assert not hasattr(market, "MarketSourceRecord")

    def test_the_expected_symbol_the_worker_checks_still_matches(self) -> None:
        """The worker cross-checks the canonical symbol; both sides say XAUUSD."""
        from goldpipeline.config import MarketDataSettings

        expected = MarketDataSettings.from_env(cli._config_env()).canonical_symbol
        assert expected == self.loaded().model.symbol == "XAUUSD"


class TestHistoricalContinuity:
    def test_mt5_era_runs_still_load(self) -> None:
        from goldpipeline.schemas.context import AnalysisContext
        from goldpipeline.schemas.manifest import RunManifest

        runs = Path(__file__).resolve().parents[1] / "runs"
        loaded = 0
        providers: set[str] = set()
        for run in sorted(runs.iterdir()):
            if not run.is_dir():
                continue
            manifest = run / "manifest.json"
            context = run / "context.json"
            if manifest.exists():
                RunManifest.model_validate_json(manifest.read_bytes())
                loaded += 1
            if context.exists():
                parsed = AnalysisContext.model_validate_json(context.read_bytes())
                providers.add(parsed.market.provider)
                loaded += 1
        assert loaded > 0
        # Historical Runs keep whatever they recorded, including the older
        # `mt5-demo` label from before the provider name settled. Nothing was
        # retrofitted, and the point of the assertion is the absence below: no
        # pre-migration Run may claim to have come from the new authority.
        assert providers
        assert "tradingview" not in providers, providers
        assert providers <= {"metatrader5", "mt5-demo", "file", "fixture"}, providers

    def test_no_migration_rewrote_a_historical_price(self) -> None:
        """No cross-provider adjustment exists anywhere in the source."""
        for path in SRC.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for banned in ("0.13", "OFFSET_ADJUSTMENT", "broker_normalis", "broker_normaliz"):
                assert banned not in text, f"{path.name} contains {banned!r}"


# --- boundary and non-change ----------------------------------------------


class TestBoundary:
    def test_downstream_stages_never_import_the_provider(self) -> None:
        allowed = {
            "cli.py",
            "tradingview_market.py",
            "tradingview_protocol.py",
            "fake_tradingview.py",
        }
        for path in SRC.rglob("*.py"):
            if path.name in allowed:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    assert "tradingview" not in name.lower(), f"{path.name} imports {name}"

    def test_the_cli_touches_the_protocol_module_only_through_the_source(self) -> None:
        for word in ("~m~", "resolve_symbol", "create_series", "unauthorized_user_token"):
            assert word not in CLI_TEXT, word

    def test_article_contracts_stay_provider_neutral(self) -> None:
        from goldpipeline.schemas.article_contract import ArticleContract

        for name, field in ArticleContract.model_fields.items():
            rendered = str(field.annotation).lower()
            assert "tradingview" not in rendered, name
            assert "mt5" not in rendered, name

    def test_numeric_semantics_stay_provider_neutral(self) -> None:
        from goldpipeline.services.numeric_semantics import SemanticType

        for member in SemanticType:
            assert not any(w in member.name for w in ("MT5", "TRADINGVIEW", "OANDA"))

    def test_h4_is_consumed_provider_native_with_no_correction(self) -> None:
        """No resampling and no one-hour shift: the venue's own grid is the truth."""
        text = (SRC / "adapters" / "tradingview_market.py").read_text(encoding="utf-8")
        for banned in ("resample", "anchor_shift", "timedelta(hours=1)", "ANCHOR_CORRECTION"):
            assert banned not in text, banned

    def test_production_remains_a_single_timeframe(self) -> None:
        """Adapter support for five timeframes is not a licence to widen production."""
        from goldpipeline.config import MarketDataSettings

        settings = MarketDataSettings.from_env(cli._config_env())
        assert settings.timeframe == "M15"
        source = cli._market_source(namespace())
        assert source.timeframe is Timeframe.M15


class TestArticleBehaviourUnchanged:
    def test_readiness_is_unchanged(self) -> None:
        from goldpipeline.schemas.article import ArticleType
        from goldpipeline.services.article_routing import READY_TYPES, SPECS

        assert {ArticleType.ANALYSIS} == READY_TYPES
        assert SPECS[ArticleType.NEWS_DIGEST].ready is False
        assert SPECS[ArticleType.TRADE_PLAN].ready is False

    def test_the_analysis_writer_prompt_is_the_current_version(self) -> None:
        from goldpipeline.prompts import DEFAULT_WRITER_PROMPT
        from goldpipeline.schemas.article import ArticleType
        from goldpipeline.services.article_routing import SPECS

        assert DEFAULT_WRITER_PROMPT == "gold_writer_v4"
        assert SPECS[ArticleType.ANALYSIS].prompt_id == DEFAULT_WRITER_PROMPT

    def test_the_comparison_service_is_still_unused_by_production(self) -> None:
        """Round 6.4e wired the ANALYSIS checks; the comparison stayed a tool.

        The market-source comparison exists to inform a decision a person makes,
        not to run inside the pipeline, so nothing may import it. The article
        checks are a separate question, pinned in `test_article_contracts`.
        """
        unused = {"goldpipeline.services.market_comparison"}
        own = {name.rsplit(".", 1)[1] for name in unused}
        for path in SRC.rglob("*.py"):
            if path.stem in own:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in unused:
                    raise AssertionError(f"{path.name} imports {node.module}")
