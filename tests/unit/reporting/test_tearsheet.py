"""
tests/unit/reporting/test_tearsheet.py

Tests for PDF tearsheet generation.
"""

import matplotlib

matplotlib.use("Agg")

import pytest

from sandtable.api import run_backtest
from sandtable.core.events import Direction, MarketDataEvent, SignalEvent
from sandtable.reporting.tearsheet import generate_pdf_tearsheet
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


class TestPdfTearsheet:
    def test_generates_pdf_file(self, tmp_path, sample_result):
        """generate_pdf_tearsheet produces a PDF file."""
        out = str(tmp_path / "test_tearsheet.pdf")
        result_path = generate_pdf_tearsheet(sample_result, output_path=out)

        assert result_path == out
        assert (tmp_path / "test_tearsheet.pdf").exists()
        assert (tmp_path / "test_tearsheet.pdf").stat().st_size > 0

    def test_pdf_is_valid(self, tmp_path, sample_result):
        """Generated file starts with PDF magic bytes."""
        out = str(tmp_path / "test.pdf")
        generate_pdf_tearsheet(sample_result, output_path=out)

        with open(out, "rb") as f:
            header = f.read(5)
        assert header == b"%PDF-"

    def test_include_stats(self, tmp_path, sample_result):
        """include_stats=True doesn't error."""
        out = str(tmp_path / "stats_tearsheet.pdf")
        generate_pdf_tearsheet(
            sample_result, output_path=out, include_stats=True,
            n_simulations=50, random_seed=42,
        )
        assert (tmp_path / "stats_tearsheet.pdf").exists()

    def test_default_output_path(self, sample_result, monkeypatch, tmp_path):
        """Default path uses output dir from settings."""
        from sandtable.config import Settings
        from sandtable.reporting import tearsheet as ts_mod
        custom = Settings(BACKTESTER_OUTPUT_DIR=tmp_path)
        monkeypatch.setattr(ts_mod, "settings", custom)
        path = generate_pdf_tearsheet(sample_result)
        assert tmp_path.name in path
        assert path.endswith(".pdf")

    def test_returns_path_string(self, tmp_path, sample_result):
        """Return value is a string path."""
        out = str(tmp_path / "out.pdf")
        result = generate_pdf_tearsheet(sample_result, output_path=out)
        assert isinstance(result, str)
