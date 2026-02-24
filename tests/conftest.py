"""
Shared fixtures for the test suite.
"""

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_CSV = DATA_DIR / "sample_ohlcv.csv"


@pytest.fixture
def sample_csv_path():
    """Path to the shared sample OHLCV CSV fixture."""
    return SAMPLE_CSV


def make_data_handler(symbols: list[str] | None = None):
    """
    Build a DataHandler backed by CSVProvider pointing at tests/data/.

    Loads all data (no date filtering needed for test fixtures).
    """
    from sandtable.data.universe import Universe
    from sandtable.data_engine import CSVProvider, DataHandler

    if symbols is None:
        symbols = ["SPY"]

    provider = CSVProvider(DATA_DIR)
    universe = Universe.from_symbols(symbols)
    handler = DataHandler(provider, universe)
    handler.load("2000-01-01", "2099-12-31")
    return handler
