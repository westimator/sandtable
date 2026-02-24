"""
Tests for DataProvider implementations (CSVProvider, YFinanceProvider).
"""

from pathlib import Path

import pandas as pd
import pytest

from sandtable.data_engine.data_providers import CSVProvider, YFinanceProvider, _enforce_ohlcv_contract
from sandtable.utils.exceptions import DataFetchError

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "data" / "fixtures"


class TestValidateDataframe:
    def test_valid_dataframe(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0, 4.0, 5.0],
                "high": [1.5, 2.5, 3.5, 4.5, 5.5],
                "low": [0.5, 1.5, 2.5, 3.5, 4.5],
                "close": [1.2, 2.2, 3.2, 4.2, 5.2],
                "volume": [100, 200, 300, 400, 500],
            },
            index=dates,
        )
        result = _enforce_ohlcv_contract(df, "TEST")
        assert result.index.name == "date"
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    def test_missing_columns(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=dates)
        with pytest.raises(DataFetchError, match="Missing required columns"):
            _enforce_ohlcv_contract(df, "TEST")

    def test_non_datetime_index(self):
        df = pd.DataFrame(
            {
                "open": [1.0],
                "high": [1.5],
                "low": [0.5],
                "close": [1.2],
                "volume": [100],
            },
            index=[0],
        )
        with pytest.raises(DataFetchError, match="DatetimeIndex"):
            _enforce_ohlcv_contract(df, "TEST")

    def test_nan_in_close(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        df = pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [1.5, 2.5, 3.5],
                "low": [0.5, 1.5, 2.5],
                "close": [1.2, float("nan"), 3.2],
                "volume": [100, 200, 300],
            },
            index=dates,
        )
        with pytest.raises(DataFetchError, match="NaN"):
            _enforce_ohlcv_contract(df, "TEST")


class TestCSVProvider:
    @pytest.fixture
    def provider(self):
        return CSVProvider(FIXTURES_DIR)

    def test_fetch_loads_fixture_with_correct_columns(self, provider):
        df = provider.fetch("SPY", "2018-01-01", "2023-12-31")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "date"
        assert len(df) > 0

    def test_fetch_loads_all_four_fixtures(self, provider):
        for symbol in ["SPY", "QQQ", "IWM", "AAPL", "MSFT"]:
            df = provider.fetch(symbol, "2018-01-01", "2023-12-31")
            assert len(df) > 1000, f"{symbol} should have >1000 bars"
            assert not df["close"].isna().any(), f"{symbol} has NaN in close"

    def test_fetch_filters_to_date_range(self, provider):
        df = provider.fetch("SPY", "2020-01-01", "2020-12-31")
        assert df.index.min() >= pd.Timestamp("2020-01-01")
        assert df.index.max() <= pd.Timestamp("2020-12-31")
        # Should be roughly 252 trading days in a year
        assert 200 < len(df) < 260

    def test_fetch_missing_symbol_raises(self, provider):
        with pytest.raises(FileNotFoundError, match="NOSYMBOL"):
            provider.fetch("NOSYMBOL", "2020-01-01", "2020-12-31")

    def test_fetch_data_is_sorted(self, provider):
        df = provider.fetch("SPY", "2018-01-01", "2023-12-31")
        assert df.index.is_monotonic_increasing

    def test_fetch_fallback_file(self, tmp_path):
        """Test fallback to {symbol}.csv when exact match not found."""
        csv_content = "date,open,high,low,close,volume\n2020-01-02,100,105,99,103,1000\n2020-01-03,103,107,101,106,1200\n"
        (tmp_path / "TEST.csv").write_text(csv_content)
        provider = CSVProvider(tmp_path)
        df = provider.fetch("TEST", "2020-01-01", "2020-12-31")
        assert len(df) == 2
        assert df.iloc[0]["close"] == 103.0


class TestCSVProviderEdgeCases:
    def test_no_date_column_raises(self, tmp_path):
        """CSV with no date/datetime/timestamp column raises DataFetchError."""
        csv_content = "price,open,high,low,close,volume\n100,100,105,99,103,1000\n"
        (tmp_path / "NODATE.csv").write_text(csv_content)
        provider = CSVProvider(tmp_path)
        with pytest.raises(DataFetchError, match="No date column"):
            provider.fetch("NODATE", "2020-01-01", "2020-12-31")

    def test_glob_fallback(self, tmp_path):
        """Falls back to {symbol}_*.csv pattern match."""
        csv_content = "date,open,high,low,close,volume\n2020-01-02,100,105,99,103,1000\n"
        (tmp_path / "GLOB_2020.csv").write_text(csv_content)
        provider = CSVProvider(tmp_path)
        df = provider.fetch("GLOB", "2020-01-01", "2020-12-31")
        assert len(df) == 1


class TestYFinanceProvider:
    """Integration tests for YFinanceProvider. May be skipped in CI."""

    @pytest.mark.skipif(True, reason="Requires network access - run manually")
    def test_fetch_returns_valid_dataframe(self):
        provider = YFinanceProvider()
        df = provider.fetch("SPY", "2023-01-01", "2023-03-31")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "date"
        assert len(df) > 50


class TestYFinanceProviderMocked:
    """Unit tests for YFinanceProvider with mocked yfinance."""

    def test_fetch_normalizes_columns(self, monkeypatch):
        """Multi-level columns from yfinance are flattened."""
        dates = pd.date_range("2023-01-03", periods=3, freq="B")
        multi_cols = pd.MultiIndex.from_tuples(
            [("Open", "SPY"), ("High", "SPY"), ("Low", "SPY"),
             ("Close", "SPY"), ("Volume", "SPY")]
        )
        data = pd.DataFrame(
            [[100, 105, 99, 103, 1000],
             [103, 107, 101, 106, 1200],
             [106, 110, 104, 109, 1100]],
            index=dates,
            columns=multi_cols,
        )

        monkeypatch.setattr("sandtable.data_engine.data_providers.yf.download", lambda *a, **kw: data)
        provider = YFinanceProvider()
        df = provider.fetch("SPY", "2023-01-03", "2023-01-06")
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 3

    def test_fetch_empty_raises(self, monkeypatch):
        """Empty response from yfinance raises DataFetchError."""
        monkeypatch.setattr(
            "sandtable.data_engine.data_providers.yf.download",
            lambda *a, **kw: pd.DataFrame(),
        )
        provider = YFinanceProvider()
        with pytest.raises(DataFetchError, match="No data returned"):
            provider.fetch("FAKE", "2023-01-01", "2023-03-31")

    def test_fetch_exception_raises(self, monkeypatch):
        """Exception from yfinance is wrapped in DataFetchError."""
        def _boom(*a, **kw):
            raise ConnectionError("network error")
        monkeypatch.setattr("sandtable.data_engine.data_providers.yf.download", _boom)
        provider = YFinanceProvider()
        with pytest.raises(DataFetchError, match="yfinance download failed"):
            provider.fetch("SPY", "2023-01-01", "2023-03-31")

    def test_fetch_strips_timezone(self, monkeypatch):
        """Timezone-aware index is localized to None."""
        dates = pd.date_range("2023-01-03", periods=3, freq="B", tz="US/Eastern")
        data = pd.DataFrame(
            {"Open": [100, 103, 106], "High": [105, 107, 110],
             "Low": [99, 101, 104], "Close": [103, 106, 109],
             "Volume": [1000, 1200, 1100]},
            index=dates,
        )
        monkeypatch.setattr("sandtable.data_engine.data_providers.yf.download", lambda *a, **kw: data)
        provider = YFinanceProvider()
        df = provider.fetch("SPY", "2023-01-03", "2023-01-06")
        assert df.index.tz is None
