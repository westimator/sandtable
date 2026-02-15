"""
tests/data_handlers/test_yfinance_handler.py

Tests for YFinanceDataHandler.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from sandtable.data_handlers.abstract_data_handler import AbstractDataHandler
from sandtable.data_handlers.yfinance_handler import YFinanceDataHandler


def _mock_download_df() -> pd.DataFrame:
    """Create a mock DataFrame matching yfinance download output."""
    dates = pd.date_range("2023-01-03", periods=5, freq="B")
    df = pd.DataFrame(
        {
            ("Open", "SPY"): [380.0, 382.0, 381.0, 385.0, 383.0],
            ("High", "SPY"): [385.0, 386.0, 384.0, 388.0, 387.0],
            ("Low", "SPY"): [378.0, 380.0, 379.0, 383.0, 381.0],
            ("Close", "SPY"): [383.0, 384.0, 382.0, 387.0, 385.0],
            ("Volume", "SPY"): [1e6, 1.1e6, 9e5, 1.2e6, 1e6],
        },
        index=dates,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


@pytest.fixture(autouse=True)
def mock_yf(monkeypatch):
    """Patch yf on the already-imported handler module."""
    mock_module = MagicMock()
    mock_module.download.return_value = _mock_download_df()
    monkeypatch.setattr("sandtable.data_handlers.yfinance_handler.yf", mock_module)
    yield mock_module


@pytest.fixture
def make_handler():
    return YFinanceDataHandler


class TestYFinanceProtocol:
    def test_satisfies_datahandler_protocol(self, make_handler):
        handler = make_handler("SPY", start="2023-01-01", end="2023-01-10")
        assert isinstance(handler, AbstractDataHandler)

    def test_symbol_attribute(self, make_handler):
        handler = make_handler("AAPL", start="2023-01-01", end="2023-01-10")
        assert handler.symbol == "AAPL"


class TestYFinanceIteration:
    def test_bar_iteration(self, make_handler):
        handler = make_handler("SPY", start="2023-01-01", end="2023-01-10")
        bars = []
        while (bar := handler.get_next_bar()) is not None:
            bars.append(bar)
        assert len(bars) == 5

    def test_bar_values(self, make_handler):
        handler = make_handler("SPY", start="2023-01-01", end="2023-01-10")
        bar = handler.get_next_bar()
        assert bar.symbol == "SPY"
        assert bar.open == 380.0
        assert bar.close == 383.0

    def test_len(self, make_handler):
        handler = make_handler("SPY", start="2023-01-01", end="2023-01-10")
        assert len(handler) == 5


class TestYFinanceHistorical:
    def test_historical_bars(self, make_handler):
        handler = make_handler("SPY", start="2023-01-01", end="2023-01-10")
        handler.get_next_bar()
        handler.get_next_bar()
        handler.get_next_bar()

        hist = handler.get_historical_bars(2)
        assert len(hist) == 2
        assert hist[0].close == 384.0
        assert hist[1].close == 382.0

    def test_get_latest_bar(self, make_handler):
        handler = make_handler("SPY", start="2023-01-01", end="2023-01-10")
        assert handler.get_latest_bar() is None
        handler.get_next_bar()
        assert handler.get_latest_bar().close == 383.0


class TestYFinanceReset:
    def test_reset(self, make_handler):
        handler = make_handler("SPY", start="2023-01-01", end="2023-01-10")
        handler.get_next_bar()
        handler.get_next_bar()
        assert handler.current_index == 2

        handler.reset()
        assert handler.current_index == 0
        bar = handler.get_next_bar()
        assert bar.open == 380.0


class TestYFinanceCaching:
    def test_caching_writes_file(self, make_handler, mock_yf, tmp_path):
        handler = make_handler("SPY", start="2023-01-01", end="2023-01-10", cache_dir=tmp_path)
        assert len(handler) == 5
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == 1

    def test_caching_reads_from_cache(self, make_handler, mock_yf, tmp_path):
        # First call downloads + caches
        make_handler("SPY", start="2023-01-01", end="2023-01-10", cache_dir=tmp_path)
        assert mock_yf.download.call_count == 1

        # Second call reads from cache (no additional download)
        handler2 = make_handler("SPY", start="2023-01-01", end="2023-01-10", cache_dir=tmp_path)
        assert mock_yf.download.call_count == 1
        assert len(handler2) == 5


class TestYFinanceErrors:
    def test_empty_download_raises(self, make_handler, mock_yf):
        mock_yf.download.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="No data returned"):
            make_handler("FAKE", start="2023-01-01", end="2023-01-10")
