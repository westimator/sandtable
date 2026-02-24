"""
Tests for CachingProvider.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from sandtable.data_engine.cache import CachingProvider
from sandtable.data_engine.data_providers import AbstractDataProvider


def _make_sample_df():
    """Create a sample DataFrame matching the provider contract."""
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [95.0, 96.0, 97.0, 98.0, 99.0],
            "close": [103.0, 104.0, 105.0, 106.0, 107.0],
            "volume": [1000, 2000, 3000, 4000, 5000],
        },
        index=dates,
    )


@pytest.fixture
def mock_provider():
    provider = MagicMock(spec=AbstractDataProvider)
    provider.fetch.return_value = _make_sample_df()
    return provider


class TestCachingProvider:
    def test_writes_cache_on_first_fetch(self, mock_provider, tmp_path):
        caching = CachingProvider(mock_provider, cache_dir=tmp_path, cache_format="parquet")
        df = caching.fetch("SPY", "2020-01-01", "2020-01-10")

        assert len(df) == 5
        cache_file = tmp_path / "SPY_2020-01-01_2020-01-10.parquet"
        assert cache_file.exists()
        mock_provider.fetch.assert_called_once()

    def test_reads_from_cache_on_second_fetch(self, mock_provider, tmp_path):
        caching = CachingProvider(mock_provider, cache_dir=tmp_path, cache_format="parquet")

        # First fetch - hits provider
        df1 = caching.fetch("SPY", "2020-01-01", "2020-01-10")
        assert mock_provider.fetch.call_count == 1

        # Second fetch - reads from cache, no second provider call
        df2 = caching.fetch("SPY", "2020-01-01", "2020-01-10")
        assert mock_provider.fetch.call_count == 1  # Still 1 - no second call
        pd.testing.assert_frame_equal(df1, df2, check_freq=False)

    def test_cache_key_format(self, mock_provider, tmp_path):
        caching = CachingProvider(mock_provider, cache_dir=tmp_path, cache_format="parquet")
        expected_path = tmp_path / "AAPL_2021-01-01_2021-12-31.parquet"

        caching.fetch("AAPL", "2021-01-01", "2021-12-31")
        assert expected_path.exists()

    def test_parquet_roundtrip_preserves_dtypes(self, mock_provider, tmp_path):
        caching = CachingProvider(mock_provider, cache_dir=tmp_path, cache_format="parquet")

        original = caching.fetch("SPY", "2020-01-01", "2020-01-10")
        cached = caching.fetch("SPY", "2020-01-01", "2020-01-10")

        assert isinstance(cached.index, pd.DatetimeIndex)
        assert cached.index.name == original.index.name
        for col in ["open", "high", "low", "close", "volume"]:
            assert original[col].dtype == cached[col].dtype

    def test_csv_fallback(self, mock_provider, tmp_path):
        caching = CachingProvider(mock_provider, cache_dir=tmp_path, cache_format="csv")

        df = caching.fetch("SPY", "2020-01-01", "2020-01-10")
        cache_file = tmp_path / "SPY_2020-01-01_2020-01-10.csv"
        assert cache_file.exists()
        assert len(df) == 5

        # Second fetch reads CSV cache
        df2 = caching.fetch("SPY", "2020-01-01", "2020-01-10")
        assert mock_provider.fetch.call_count == 1
        pd.testing.assert_frame_equal(df, df2, check_freq=False)

    def test_different_symbols_cached_separately(self, mock_provider, tmp_path):
        caching = CachingProvider(mock_provider, cache_dir=tmp_path, cache_format="parquet")

        caching.fetch("SPY", "2020-01-01", "2020-01-10")
        caching.fetch("AAPL", "2020-01-01", "2020-01-10")

        assert (tmp_path / "SPY_2020-01-01_2020-01-10.parquet").exists()
        assert (tmp_path / "AAPL_2020-01-01_2020-01-10.parquet").exists()
        assert mock_provider.fetch.call_count == 2

    def test_cache_dir_created_automatically(self, mock_provider, tmp_path):
        cache_dir = tmp_path / "nested" / "cache"
        caching = CachingProvider(mock_provider, cache_dir=cache_dir, cache_format="parquet")
        caching.fetch("SPY", "2020-01-01", "2020-01-10")
        assert cache_dir.exists()
