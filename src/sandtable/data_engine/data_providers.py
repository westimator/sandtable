"""
src/sandtable/data_engine/data_providers.py

AbstractDataProvider ABC and concrete implementations (YFinanceProvider, CSVProvider).

All providers return DataFrames with DatetimeIndex named 'date' and columns:
[open, high, low, close, volume]. Data is adjusted for splits/dividends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import yfinance as yf

from sandtable.utils.exceptions import DataFetchError
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)


def _enforce_ohlcv_contract(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Validate and normalize a provider DataFrame to the standard contract.

    Ensures:
    - Columns: open, high, low, close, volume (lowercase)
    - Index: DatetimeIndex named 'date'
    - No NaN values in the close column

    Returns the validated DataFrame.
    """
    required = {"open", "high", "low", "close", "volume"}

    missing = required - set(df.columns)
    if missing:
        raise DataFetchError(f"[{symbol}] Missing required columns: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataFetchError(f"[{symbol}] Index must be a DatetimeIndex, got {type(df.index).__name__}")

    if df["close"].isna().any():
        nan_count = df["close"].isna().sum()
        raise DataFetchError(f"[{symbol}] Found {nan_count} NaN values in 'close' column")

    df.index.name = "date"
    df = df.sort_index()

    # keep only the required columns
    return df[["open", "high", "low", "close", "volume"]]


class AbstractDataProvider(ABC):
    """
    Abstract base class for all data providers.

    Providers fetch OHLCV data for a single symbol over a date range.
    """

    @abstractmethod
    def fetch(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single symbol.

        Args:
            symbol: Ticker symbol (e.g. 'SPY')
            start_date: Start date as 'YYYY-MM-DD'
            end_date: End date as 'YYYY-MM-DD'

        Returns:
            DataFrame with DatetimeIndex named 'date' and columns:
            [open, high, low, close, volume].
            Adjusted for splits and dividends.
        """


class CSVProvider(AbstractDataProvider):
    """
    Loads OHLCV data from local CSV files.

    File resolution order:
    1. {data_dir}/{symbol}_{start_date}_{end_date}.csv (exact match)
    2. {data_dir}/{symbol}.csv (fallback, filtered to date range)
    3. {data_dir}/{symbol}_*.csv (any file starting with symbol, filtered to date range)

    Expected CSV columns: date, open, high, low, close, volume
    """

    def __init__(self, data_dir: str | Path) -> None:
        """
        Args:
            data_dir: Directory containing CSV data files.
        """
        self._data_dir = Path(data_dir)

    def _resolve_path(self, symbol: str, start_date: str, end_date: str) -> Path:
        """Resolve the CSV file path for a symbol, trying exact match then fallbacks."""
        exact_path = self._data_dir / f"{symbol}_{start_date}_{end_date}.csv"
        fallback_path = self._data_dir / f"{symbol}.csv"

        if exact_path.exists():
            return exact_path
        if fallback_path.exists():
            return fallback_path

        # try any file matching {symbol}_*.csv
        candidates = sorted(self._data_dir.glob(f"{symbol}_*.csv"))
        if candidates:
            return candidates[0]

        raise FileNotFoundError(
            f"No data file found for {symbol}. "
            f"Looked for: {exact_path} and {fallback_path}"
        )

    def fetch(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load OHLCV data from a CSV file, filtering to the requested date range."""
        path = self._resolve_path(symbol, start_date, end_date)

        logger.debug("Loading %s from %s", symbol, path)
        df = pd.read_csv(path)

        # normalize column names
        df.columns = df.columns.str.strip().str.lower()

        # handle 'date' or 'datetime' column
        date_col = None
        for col in ("date", "datetime", "timestamp"):
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            raise DataFetchError(f"[{symbol}] No date column found in {path}. Expected 'date', 'datetime', or 'timestamp'.")

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

        # filter to requested date range
        df = df.loc[start_date:end_date]

        df = _enforce_ohlcv_contract(df, symbol)
        logger.debug("Loaded %d bars for %s from %s", len(df), symbol, path.name)
        return df


class YFinanceProvider(AbstractDataProvider):
    """
    Fetches OHLCV data from Yahoo Finance via yfinance.

    Free, no API key required. Known to be flaky and rate-limited -
    use CachingProvider wrapper for reliability.
    """

    def fetch(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download OHLCV data from Yahoo Finance for the given symbol and date range."""
        logger.debug("Downloading %s from yfinance (%s to %s)", symbol, start_date, end_date)
        try:
            ticker = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            raise DataFetchError(f"[{symbol}] yfinance download failed: {e}") from e

        if ticker.empty:
            raise DataFetchError(f"[{symbol}] No data returned from yfinance for {start_date} to {end_date}")

        # handle multi-level columns from yfinance (group_by='ticker')
        if isinstance(ticker.columns, pd.MultiIndex):
            ticker.columns = ticker.columns.droplevel(1)

        # normalize column names to lowercase
        ticker.columns = ticker.columns.str.lower()

        # ensure DatetimeIndex is timezone-naive
        if ticker.index.tz is not None:
            ticker.index = ticker.index.tz_localize(None)

        df = _enforce_ohlcv_contract(ticker, symbol)
        logger.debug("Fetched %d bars for %s from yfinance", len(df), symbol)
        return df
