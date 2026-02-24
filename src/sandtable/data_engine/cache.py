"""
src/sandtable/data_engine/cache.py

CachingProvider wraps any DataProvider with transparent local file caching.

Cache key: {symbol}_{start_date}_{end_date}.parquet (or .csv).
Cache is append-only with no invalidation, appropriate for historical backtest data.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import pandas as pd

from sandtable.data_engine.data_providers import AbstractDataProvider
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)


class CacheFormat(StrEnum):
    """
    Supported file formats for cached data.
    """
    CSV = "csv"
    PARQUET = "parquet"


class CachingProvider(AbstractDataProvider):
    """
    Wraps any DataProvider with local file caching.

    On fetch: if cache file exists, read and return. Otherwise, delegate
    to the inner provider, write the cache file, and return.

    Parquet is the default format (fast, type-preserving). CSV is available
    as an alternative via cache_format='csv'.
    """

    def __init__(
        self,
        provider: AbstractDataProvider,
        cache_dir: str | Path = "data/cache",
        cache_format: CacheFormat | str = CacheFormat.PARQUET,
    ) -> None:
        """
        Args:
            provider: The inner DataProvider to cache.
            cache_dir: Directory for cache files. Created on first write.
            cache_format: 'parquet' or 'csv'.
        """
        self._provider = provider
        self._cache_dir = Path(cache_dir)
        self._cache_format = CacheFormat(cache_format)

    def _cache_path(self, symbol: str, start_date: str, end_date: str) -> Path:
        """Build the cache file path for a given symbol and date range."""
        return self._cache_dir / f"{symbol}_{start_date}_{end_date}.{self._cache_format}"

    def _read_cache(self, path: Path) -> pd.DataFrame:
        """Read a cached DataFrame from disk (parquet or CSV)."""
        logger.debug("Reading cache from %s", path)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df

    def _write_cache(self, df: pd.DataFrame, path: Path) -> None:
        """Write a DataFrame to disk as a cache file, creating directories as needed."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Writing cache to %s", path)
        if path.suffix == ".parquet":
            df.to_parquet(path)
        else:
            df.to_csv(path)

    def fetch(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data, returning from cache if available or delegating to the inner provider."""
        cache_path = self._cache_path(symbol, start_date, end_date)

        if cache_path.exists():
            return self._read_cache(cache_path)

        df = self._provider.fetch(symbol, start_date, end_date)
        self._write_cache(df, cache_path)
        return df
