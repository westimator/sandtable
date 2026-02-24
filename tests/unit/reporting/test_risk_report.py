"""
tests/unit/reporting/test_risk_report.py

Tests for standalone PDF risk report generation.
"""

import matplotlib

matplotlib.use("Agg")

from datetime import datetime

import pytest

from sandtable.api import run_backtest
from sandtable.core.events import Direction, MarketDataEvent, RiskAction, RiskBreachEvent, SignalEvent
from sandtable.reporting.risk_report import generate_risk_report
from sandtable.strategy.abstract_strategy import AbstractStrategy
from tests.conftest import make_data_handler


class _SimpleStrategy(AbstractStrategy):
    """Minimal strategy for testing reports."""

    def __init__(self, *, lookback: int = 10, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lookback = lookback

    def generate_signal(self, bar: MarketDataEvent) -> SignalEvent | None:
        if self.symbol_bar_count(bar.symbol) < self.lookback:
            return None
        closes = self.get_historical_closes(self.lookback, bar.symbol)
        if closes[-1] > sum(closes) / len(closes):
            return SignalEvent(
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                direction=Direction.LONG,
                strength=1.0,
            )
        return None


@pytest.fixture
def sample_result():
    data = make_data_handler(["SPY"])
    return run_backtest(strategy=_SimpleStrategy(lookback=10), data=data)


@pytest.fixture
def sample_breaches():
    return [
        RiskBreachEvent(
            timestamp=datetime(2024, 1, 15),
            rule_name="MaxPositionSizeRule",
            symbol="SPY",
            proposed_qty=500,
            action=RiskAction.RESIZED,
            breach_value=0.35,
            threshold=0.25,
            final_qty=250,
        ),
        RiskBreachEvent(
            timestamp=datetime(2024, 1, 16),
            rule_name="MaxDrawdownRule",
            symbol="AAPL",
            proposed_qty=100,
            action=RiskAction.REJECTED,
            breach_value=0.22,
            threshold=0.20,
        ),
    ]


class TestGenerateRiskReport:
    def test_generates_pdf_file(self, tmp_path, sample_result):
        out = str(tmp_path / "risk.pdf")
        result_path = generate_risk_report(sample_result, output_path=out)
        assert result_path == out
        assert (tmp_path / "risk.pdf").exists()
        assert (tmp_path / "risk.pdf").stat().st_size > 0

    def test_pdf_starts_with_magic_bytes(self, tmp_path, sample_result):
        out = str(tmp_path / "risk.pdf")
        generate_risk_report(sample_result, output_path=out)
        with open(out, "rb") as f:
            assert f.read(5) == b"%PDF-"

    def test_with_breach_log_events(self, tmp_path, sample_result, sample_breaches):
        out = str(tmp_path / "risk_breaches.pdf")
        generate_risk_report(sample_result, breach_log=sample_breaches, output_path=out)
        assert (tmp_path / "risk_breaches.pdf").exists()

    def test_with_breach_log_dicts(self, tmp_path, sample_result):
        breaches = [
            {
                "timestamp": datetime(2024, 1, 15),
                "rule_name": "MaxPositionSizeRule",
                "symbol": "SPY",
                "action": RiskAction.REJECTED,
                "breach_value": 0.35,
                "threshold": 0.25,
            },
        ]
        out = str(tmp_path / "risk_dict.pdf")
        generate_risk_report(sample_result, breach_log=breaches, output_path=out)
        assert (tmp_path / "risk_dict.pdf").exists()

    def test_with_many_breaches_truncation(self, tmp_path, sample_result):
        """More than 30 breaches triggers truncation message."""
        breaches = [
            RiskBreachEvent(
                timestamp=datetime(2024, 1, i % 28 + 1),
                rule_name="MaxPositionSizeRule",
                symbol="SPY",
                proposed_qty=100,
                action=RiskAction.REJECTED,
                breach_value=0.35,
                threshold=0.25,
            )
            for i in range(35)
        ]
        out = str(tmp_path / "risk_many.pdf")
        generate_risk_report(sample_result, breach_log=breaches, output_path=out)
        assert (tmp_path / "risk_many.pdf").exists()

    def test_default_output_path(self, sample_result, monkeypatch, tmp_path):
        from sandtable.config import Settings
        from sandtable.reporting import risk_report as rr_mod
        custom = Settings(BACKTESTER_OUTPUT_DIR=tmp_path)
        monkeypatch.setattr(rr_mod, "settings", custom)
        path = generate_risk_report(sample_result)
        assert tmp_path.name in path
        assert path.endswith(".pdf")

    def test_no_breaches_no_equity(self, tmp_path):
        """Report still generates with minimal result."""
        data = make_data_handler(["SPY"])
        result = run_backtest(strategy=_SimpleStrategy(lookback=10), data=data)
        out = str(tmp_path / "risk_min.pdf")
        generate_risk_report(result, output_path=out)
        assert (tmp_path / "risk_min.pdf").exists()
