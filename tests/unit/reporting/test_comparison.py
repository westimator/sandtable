"""
tests/unit/reporting/test_comparison.py

Tests for PDF strategy comparison report generation.
"""

import matplotlib

matplotlib.use("Agg")

import pytest

from sandtable.api import run_backtest
from sandtable.core.events import Direction, MarketDataEvent, SignalEvent
from sandtable.reporting.comparison import generate_comparison_report
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
def two_results():
    data_a = make_data_handler(["SPY"])
    result_a = run_backtest(strategy=_SimpleStrategy(lookback=10), data=data_a)
    data_b = make_data_handler(["SPY"])
    result_b = run_backtest(strategy=_SimpleStrategy(lookback=20), data=data_b)
    return {"Strategy A": result_a, "Strategy B": result_b}


class TestGenerateComparisonReport:
    def test_generates_pdf_file(self, tmp_path, two_results):
        out = str(tmp_path / "comparison.pdf")
        result_path = generate_comparison_report(two_results, output_path=out)
        assert result_path == out
        assert (tmp_path / "comparison.pdf").exists()
        assert (tmp_path / "comparison.pdf").stat().st_size > 0

    def test_pdf_starts_with_magic_bytes(self, tmp_path, two_results):
        out = str(tmp_path / "cmp.pdf")
        generate_comparison_report(two_results, output_path=out)
        with open(out, "rb") as f:
            assert f.read(5) == b"%PDF-"

    def test_with_correlation_matrix(self, tmp_path, two_results):
        import pandas as pd
        corr = pd.DataFrame(
            [[1.0, 0.8], [0.8, 1.0]],
            index=["Strategy A", "Strategy B"],
            columns=["Strategy A", "Strategy B"],
        )
        out = str(tmp_path / "cmp_corr.pdf")
        generate_comparison_report(two_results, correlation_matrix=corr, output_path=out)
        assert (tmp_path / "cmp_corr.pdf").exists()

    def test_default_output_path(self, two_results, monkeypatch, tmp_path):
        from sandtable.config import Settings
        from sandtable.reporting import comparison as cmp_mod
        custom = Settings(BACKTESTER_OUTPUT_DIR=tmp_path)
        monkeypatch.setattr(cmp_mod, "settings", custom)
        path = generate_comparison_report(two_results)
        assert tmp_path.name in path
        assert path.endswith(".pdf")

    def test_single_strategy(self, tmp_path):
        data = make_data_handler(["SPY"])
        result = run_backtest(strategy=_SimpleStrategy(lookback=10), data=data)
        out = str(tmp_path / "single.pdf")
        generate_comparison_report({"Only": result}, output_path=out)
        assert (tmp_path / "single.pdf").exists()
