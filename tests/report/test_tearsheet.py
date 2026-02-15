"""
tests/report/test_tearsheet.py

Tests for tearsheet and comparison report generation.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest

from sandtable.api import run_backtest
from sandtable.core.events import Direction, MarketDataEvent, SignalEvent
from sandtable.data_handlers.csv_data_handler import CSVDataHandler
from sandtable.strategy.abstract_strategy import AbstractStrategy

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_ohlcv.csv"

plotly = pytest.importorskip("plotly", reason="plotly not installed")


@dataclass
class SimpleTestStrategy(AbstractStrategy):
    fast_period: int = 10
    slow_period: int = 30

    def generate_signal(self, bar: MarketDataEvent) -> SignalEvent | None:
        closes = self.get_historical_closes(self.slow_period, bar.symbol)
        if len(closes) < self.slow_period:
            return None
        fast_ma = sum(closes[-self.fast_period :]) / self.fast_period
        slow_ma = sum(closes) / len(closes)
        if fast_ma > slow_ma:
            return SignalEvent(
                timestamp=bar.timestamp, symbol=bar.symbol,
                direction=Direction.LONG, strength=1.0,
            )
        elif fast_ma < slow_ma:
            return SignalEvent(
                timestamp=bar.timestamp, symbol=bar.symbol,
                direction=Direction.SHORT, strength=1.0,
            )
        return None


@pytest.fixture
def sample_result():
    data = CSVDataHandler(DATA_PATH, "SPY")
    return run_backtest(
        strategy=SimpleTestStrategy(),
        data=data,
    )


class TestTearsheet:
    def test_generate_tearsheet_returns_html(self, sample_result):
        from sandtable.report.tearsheet import generate_tearsheet
        html = generate_tearsheet(sample_result)
        assert isinstance(html, str)
        assert "<html>" in html.lower() or "<!doctype" in html.lower()
        assert "plotly" in html.lower()

    def test_tearsheet_writes_file(self, sample_result, tmp_path):
        from sandtable.report.tearsheet import generate_tearsheet
        output = tmp_path / "tearsheet.html"
        html = generate_tearsheet(sample_result, output_path=str(output))
        assert output.exists()
        assert output.read_text() == html

    def test_tearsheet_via_result_method(self, sample_result, tmp_path):
        output = tmp_path / "tearsheet.html"
        html = sample_result.tearsheet(output_path=str(output))
        assert output.exists()
        assert "plotly" in html.lower()

    def test_tearsheet_contains_metrics(self, sample_result):
        from sandtable.report.tearsheet import generate_tearsheet
        html = generate_tearsheet(sample_result)
        assert "Sharpe" in html
        assert "SPY" in html


class TestComparison:
    def test_compare_strategies(self, sample_result):
        from sandtable.report.comparison import compare_strategies

        data = CSVDataHandler(DATA_PATH, "SPY")
        result2 = run_backtest(
            strategy=SimpleTestStrategy(fast_period=5, slow_period=20),
            data=data,
        )
        html = compare_strategies({"MA(10,30)": sample_result, "MA(5,20)": result2})
        assert isinstance(html, str)
        assert "MA(10,30)" in html
        assert "MA(5,20)" in html

    def test_compare_writes_file(self, sample_result, tmp_path):
        from sandtable.report.comparison import compare_strategies

        output = tmp_path / "comparison.html"
        compare_strategies({"Strategy A": sample_result}, output_path=str(output))
        assert output.exists()
