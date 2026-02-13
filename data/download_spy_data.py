"""
data/download_spy_data.py

Download SPY data from Yahoo Finance and save it to a CSV file.

Usage:
    python data/download_spy_data.py

Output:
    data/spy.csv
"""

import logging

import yfinance as yf

logger = logging.getLogger(__name__)


def main():
    logger.info("Downloading SPY data from Yahoo Finance...")
    spy = yf.download(
        tickers="SPY",
        start="2022-01-01",
        end="2024-01-01",
    )
    spy.to_csv(
        path_or_buf="data/sample_ohlcv.csv",
        index=False,
    )
    logger.info("Downloaded SPY data and saved to data/sample_ohlcv.csv")


if __name__ == "__main__":
    main()
