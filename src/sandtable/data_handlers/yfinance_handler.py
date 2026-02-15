"""
src/sandtable/data_handlers/yfinance_handler.py

Data handler that downloads OHLCV data from Yahoo Finance via yfinance.

Supports optional file caching to avoid re-downloading.
"""

import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from sandtable.config import settings
from sandtable.data_handlers.single_symbol_handler import SingleSymbolDataHandler
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)


class YFinanceDataHandler(SingleSymbolDataHandler):
    """
    Downloads and iterates over OHLCV data from Yahoo Finance.

    Attributes:
        symbol: Ticker symbol to download
    """

    def __init__(
        self,
        symbol: str,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.symbol = symbol
        self._start = start
        self._end = end
        self._cache_dir = Path(cache_dir) if cache_dir else settings.BACKTESTER_CACHE_DIR
        self._current_index = 0

        self._data = self._load_data()
        logger.debug(
            "Loaded %d bars for %s from yfinance",
            len(self._data),
            self.symbol,
        )

    def __repr__(self) -> str:
        return (
            f"YFinanceDataHandler(symbol={self.symbol!r}, "
            f"bars={len(self._data)}, "
            f"current_index={self._current_index})"
        )

    def _cache_path(self) -> Path | None:
        if self._cache_dir is None:
            return None
        key = f"{self.symbol}_{self._start}_{self._end}"
        h = hashlib.md5(string=key.encode()).hexdigest()[:12]
        return self._cache_dir / f"{self.symbol}_{h}.csv"

    def _load_data(self) -> pd.DataFrame:
        cache = self._cache_path()
        if cache and cache.exists():
            logger.debug("Loading cached data from %s", cache)
            df = pd.read_csv(cache)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df

        logger.debug("Downloading %s from yfinance (start=%s, end=%s)", self.symbol, self._start, self._end)
        ticker = yf.download(
            self.symbol,
            start=self._start,
            end=self._end,
            auto_adjust=True,
            progress=False,
        )

        if ticker.empty:
            raise ValueError(f"No data returned for {self.symbol}")

        # Handle both single-level and multi-level column indices
        if isinstance(ticker.columns, pd.MultiIndex):
            ticker.columns = ticker.columns.droplevel(1)

        df = pd.DataFrame(
            {
                "timestamp": ticker.index,
                "open": ticker["Open"].values,
                "high": ticker["High"].values,
                "low": ticker["Low"].values,
                "close": ticker["Close"].values,
                "volume": ticker["Volume"].values,
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        if cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache, index=False)
            logger.debug("Cached data to %s", cache)

        return df
