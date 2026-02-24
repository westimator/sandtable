"""
Tests for DataHandler and Universe integration.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from sandtable.core.events import MarketDataEvent
from sandtable.data.universe import Universe
from sandtable.data_engine.data_providers import AbstractDataProvider
from sandtable.data_engine.handler import DataHandler


def _make_df(dates, close_values, symbol="TEST"):
    """Create a sample DataFrame with given dates and close values."""
    n = len(dates)
    return pd.DataFrame(
        {
            "open": [c - 1.0 for c in close_values],
            "high": [c + 1.0 for c in close_values],
            "low": [c - 2.0 for c in close_values],
            "close": close_values,
            "volume": [1000] * n,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


@pytest.fixture
def mock_provider():
    return MagicMock(spec=AbstractDataProvider)


class TestUniverse:
    def test_symbols(self):
        u = Universe.from_symbols(["SPY", "AAPL"])
        assert u.symbols == ["SPY", "AAPL"]

    def test_contains(self):
        u = Universe.from_symbols(["SPY", "AAPL"])
        assert "SPY" in u
        assert "MSFT" not in u

    def test_len(self):
        u = Universe.from_symbols(["SPY", "AAPL", "MSFT"])
        assert len(u) == 3

    def test_empty(self):
        u = Universe.from_symbols([])
        assert len(u) == 0
        assert u.symbols == []


class TestDataHandler:
    def test_iter_events_chronological_order(self, mock_provider):
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        closes = [100.0, 101.0, 102.0, 103.0, 104.0]
        mock_provider.fetch.return_value = _make_df(dates, closes)

        universe = Universe.from_symbols(["SPY"])
        handler = DataHandler(mock_provider, universe)
        handler.load("2020-01-01", "2020-01-10")

        events = list(handler.iter_events())
        assert len(events) == 5

        # All events should be in chronological order
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps)

    def test_iter_events_yields_market_data_events(self, mock_provider):
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        closes = [100.0, 101.0, 102.0]
        mock_provider.fetch.return_value = _make_df(dates, closes)

        universe = Universe.from_symbols(["SPY"])
        handler = DataHandler(mock_provider, universe)
        handler.load("2020-01-01", "2020-01-10")

        events = list(handler.iter_events())
        for event in events:
            assert isinstance(event, MarketDataEvent)
            assert event.symbol == "SPY"

    def test_multi_symbol_interleave_by_date(self, mock_provider):
        """Multi-symbol events should interleave correctly by date."""
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        spy_df = _make_df(dates, [300.0, 301.0, 302.0])
        aapl_df = _make_df(dates, [150.0, 151.0, 152.0])

        def side_effect(symbol, start_date, end_date):
            if symbol == "SPY":
                return spy_df
            return aapl_df

        mock_provider.fetch.side_effect = side_effect

        universe = Universe.from_symbols(["SPY", "AAPL"])
        handler = DataHandler(mock_provider, universe)
        handler.load("2020-01-01", "2020-01-10")

        events = list(handler.iter_events())
        # 3 dates x 2 symbols = 6 events
        assert len(events) == 6

        # Should be globally sorted by timestamp
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps)

        # For each date, SPY should come before AAPL (universe order)
        for i in range(0, 6, 2):
            assert events[i].symbol == "SPY"
            assert events[i + 1].symbol == "AAPL"
            assert events[i].timestamp == events[i + 1].timestamp

    def test_empty_universe_produces_no_events(self, mock_provider):
        universe = Universe.from_symbols([])
        handler = DataHandler(mock_provider, universe)
        handler.load("2020-01-01", "2020-12-31")

        events = list(handler.iter_events())
        assert events == []

    def test_single_symbol_one_event_per_bar(self, mock_provider):
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        closes = [100.0 + i for i in range(10)]
        mock_provider.fetch.return_value = _make_df(dates, closes)

        universe = Universe.from_symbols(["SPY"])
        handler = DataHandler(mock_provider, universe)
        handler.load("2020-01-01", "2020-01-20")

        events = list(handler.iter_events())
        assert len(events) == 10

    def test_symbols_with_different_calendars(self, mock_provider):
        """Symbols with gaps should still produce correct interleaved events."""
        dates_spy = pd.date_range("2020-01-01", periods=5, freq="B")  # Mon-Fri
        dates_other = pd.date_range("2020-01-02", periods=3, freq="B")  # Tue-Thu

        spy_df = _make_df(dates_spy, [300.0, 301.0, 302.0, 303.0, 304.0])
        other_df = _make_df(dates_other, [50.0, 51.0, 52.0])

        def side_effect(symbol, start_date, end_date):
            if symbol == "SPY":
                return spy_df
            return other_df

        mock_provider.fetch.side_effect = side_effect

        universe = Universe.from_symbols(["SPY", "OTHER"])
        handler = DataHandler(mock_provider, universe)
        handler.load("2020-01-01", "2020-01-10")

        events = list(handler.iter_events())
        # SPY has 5 bars, OTHER has 3 bars = 8 events total
        assert len(events) == 8

        # All should be in chronological order
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps)

    def test_event_fields_match_dataframe(self, mock_provider):
        dates = [datetime(2020, 1, 2)]
        df = _make_df(dates, [105.0])
        mock_provider.fetch.return_value = df

        universe = Universe.from_symbols(["SPY"])
        handler = DataHandler(mock_provider, universe)
        handler.load("2020-01-01", "2020-01-05")

        events = list(handler.iter_events())
        assert len(events) == 1
        event = events[0]
        assert event.symbol == "SPY"
        assert event.open == 104.0
        assert event.high == 106.0
        assert event.low == 103.0
        assert event.close == 105.0
        assert event.volume == 1000.0

    def test_data_property(self, mock_provider):
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        df = _make_df(dates, [100.0, 101.0, 102.0])
        mock_provider.fetch.return_value = df

        universe = Universe.from_symbols(["SPY"])
        handler = DataHandler(mock_provider, universe)
        handler.load("2020-01-01", "2020-01-10")

        data = handler.data
        assert "SPY" in data
        assert len(data["SPY"]) == 3

    def test_load_clears_previous_data(self, mock_provider):
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        df = _make_df(dates, [100.0, 101.0, 102.0])
        mock_provider.fetch.return_value = df

        universe = Universe.from_symbols(["SPY"])
        handler = DataHandler(mock_provider, universe)

        handler.load("2020-01-01", "2020-01-10")
        assert len(handler.data) == 1

        handler.load("2020-01-01", "2020-01-10")
        assert len(handler.data) == 1  # Still 1, not accumulated

    def test_accepts_list_of_strings(self, mock_provider):
        """DataHandler should accept a plain list[str] for backward compatibility."""
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        mock_provider.fetch.return_value = _make_df(dates, [100.0, 101.0, 102.0])

        handler = DataHandler(mock_provider, ["SPY"])
        handler.load("2020-01-01", "2020-01-10")

        events = list(handler.iter_events())
        assert len(events) == 3
        assert isinstance(handler.universe, Universe)
