"""
tests/data_handlers/test_multi_handler.py

Tests for MultiDataHandler.
"""

from datetime import datetime, timedelta

import pytest

from sandtable.data_handlers.multi_handler import MultiDataHandler


class FakeHandler:
    """Minimal DataHandler-compatible fake for testing."""

    def __init__(self, symbol: str, bars: list[dict]):
        self.symbol = symbol
        self._bars = bars
        self._current_index = 0

    @property
    def data(self):
        import pandas as pd
        return pd.DataFrame(self._bars)

    @property
    def current_index(self):
        return self._current_index

    def __len__(self):
        return len(self._bars)

    def get_next_bar(self):
        if self._current_index >= len(self._bars):
            return None
        from sandtable.core.events import MarketDataEvent
        b = self._bars[self._current_index]
        self._current_index += 1
        return MarketDataEvent(**b)

    def get_historical_bars(self, n):
        from sandtable.core.events import MarketDataEvent
        end = self._current_index
        start = max(0, end - n)
        return [MarketDataEvent(**self._bars[i]) for i in range(start, end)]

    def get_latest_bar(self):
        if self._current_index == 0:
            return None
        from sandtable.core.events import MarketDataEvent
        return MarketDataEvent(**self._bars[self._current_index - 1])

    def reset(self):
        self._current_index = 0


def _make_bars(symbol, start, count, offset_hours=0):
    """Generate fake bar dicts."""
    base = datetime(2023, 1, 3, 9, 30) + timedelta(hours=offset_hours)
    bars = []
    for i in range(count):
        ts = base + timedelta(days=i)
        bars.append({
            "timestamp": ts,
            "symbol": symbol,
            "open": 100.0 + i,
            "high": 105.0 + i,
            "low": 95.0 + i,
            "close": 102.0 + i,
            "volume": 1e6,
        })
    return bars


class TestMultiHandlerInterleaving:
    def test_bars_in_timestamp_order(self):
        h1 = FakeHandler("SPY", _make_bars("SPY", "2023-01-03", 3))
        h2 = FakeHandler("AAPL", _make_bars("AAPL", "2023-01-03", 3))
        multi = MultiDataHandler({"SPY": h1, "AAPL": h2})

        bars = []
        while (bar := multi.get_next_bar()) is not None:
            bars.append(bar)

        assert len(bars) == 6
        timestamps = [b.timestamp for b in bars]
        assert timestamps == sorted(timestamps)

    def test_alternating_symbols_same_timestamp(self):
        h1 = FakeHandler("SPY", _make_bars("SPY", "2023-01-03", 3))
        h2 = FakeHandler("AAPL", _make_bars("AAPL", "2023-01-03", 3))
        multi = MultiDataHandler({"SPY": h1, "AAPL": h2})

        bars = []
        while (bar := multi.get_next_bar()) is not None:
            bars.append(bar)

        # Both symbols should appear
        symbols = {b.symbol for b in bars}
        assert symbols == {"SPY", "AAPL"}

    def test_different_lengths(self):
        h1 = FakeHandler("SPY", _make_bars("SPY", "2023-01-03", 5))
        h2 = FakeHandler("AAPL", _make_bars("AAPL", "2023-01-03", 2))
        multi = MultiDataHandler({"SPY": h1, "AAPL": h2})

        bars = []
        while (bar := multi.get_next_bar()) is not None:
            bars.append(bar)

        assert len(bars) == 7

    def test_offset_timestamps(self):
        """Handlers with offset timestamps should still interleave correctly."""
        h1 = FakeHandler("SPY", _make_bars("SPY", "2023-01-03", 3, offset_hours=0))
        h2 = FakeHandler("AAPL", _make_bars("AAPL", "2023-01-03", 3, offset_hours=1))
        multi = MultiDataHandler({"SPY": h1, "AAPL": h2})

        bars = []
        while (bar := multi.get_next_bar()) is not None:
            bars.append(bar)

        timestamps = [b.timestamp for b in bars]
        assert timestamps == sorted(timestamps)


class TestMultiHandlerProperties:
    def test_len(self):
        h1 = FakeHandler("SPY", _make_bars("SPY", "2023-01-03", 3))
        h2 = FakeHandler("AAPL", _make_bars("AAPL", "2023-01-03", 5))
        multi = MultiDataHandler({"SPY": h1, "AAPL": h2})
        assert len(multi) == 8

    def test_current_index(self):
        h1 = FakeHandler("SPY", _make_bars("SPY", "2023-01-03", 3))
        multi = MultiDataHandler({"SPY": h1})
        assert multi.current_index == 0
        multi.get_next_bar()
        assert multi.current_index == 1

    def test_symbol(self):
        h1 = FakeHandler("SPY", _make_bars("SPY", "2023-01-03", 1))
        h2 = FakeHandler("AAPL", _make_bars("AAPL", "2023-01-03", 1))
        multi = MultiDataHandler({"SPY": h1, "AAPL": h2})
        assert multi.symbol == "AAPL,SPY"


class TestMultiHandlerReset:
    def test_reset(self):
        h1 = FakeHandler("SPY", _make_bars("SPY", "2023-01-03", 3))
        h2 = FakeHandler("AAPL", _make_bars("AAPL", "2023-01-03", 3))
        multi = MultiDataHandler({"SPY": h1, "AAPL": h2})

        # Consume some bars
        multi.get_next_bar()
        multi.get_next_bar()
        multi.get_next_bar()
        assert multi.current_index == 3

        multi.reset()
        assert multi.current_index == 0

        # Should be able to iterate again from the start
        bars = []
        while (bar := multi.get_next_bar()) is not None:
            bars.append(bar)
        assert len(bars) == 6


class TestMultiHandlerDataProperty:
    def test_data_returns_all_bars_sorted(self):
        h1 = FakeHandler("SPY", _make_bars("SPY", "2023-01-03", 2))
        h2 = FakeHandler("AAPL", _make_bars("AAPL", "2023-01-03", 2))
        multi = MultiDataHandler({"SPY": h1, "AAPL": h2})

        df = multi.data
        assert len(df) == 4
        assert "symbol" in df.columns
        assert set(df["symbol"]) == {"SPY", "AAPL"}
        timestamps = df["timestamp"].tolist()
        assert timestamps == sorted(timestamps)


class TestMultiHandlerHistoricalBars:
    def test_historical_bars_after_reads(self):
        h1 = FakeHandler("SPY", _make_bars("SPY", "2023-01-03", 3))
        h2 = FakeHandler("AAPL", _make_bars("AAPL", "2023-01-03", 3, offset_hours=1))
        multi = MultiDataHandler({"SPY": h1, "AAPL": h2})

        # Read 4 bars
        for _ in range(4):
            multi.get_next_bar()

        hist = multi.get_historical_bars(3)
        assert len(hist) == 3
        timestamps = [b.timestamp for b in hist]
        assert timestamps == sorted(timestamps)

    def test_historical_bars_limits_to_n(self):
        h1 = FakeHandler("SPY", _make_bars("SPY", "2023-01-03", 3))
        h2 = FakeHandler("AAPL", _make_bars("AAPL", "2023-01-03", 3))
        multi = MultiDataHandler({"SPY": h1, "AAPL": h2})

        # Read all 6 bars
        for _ in range(6):
            multi.get_next_bar()

        hist = multi.get_historical_bars(2)
        assert len(hist) == 2

    def test_historical_bars_before_get_next(self):
        """Seeding the heap pre-reads one bar per handler."""
        h1 = FakeHandler("SPY", _make_bars("SPY", "2023-01-03", 3))
        multi = MultiDataHandler({"SPY": h1})

        # __init__ seeds the heap by calling get_next_bar on the sub-handler,
        # so the sub-handler already has 1 bar read
        hist = multi.get_historical_bars(5)
        assert len(hist) == 1


class TestMultiHandlerLatestBar:
    def test_latest_bar_before_get_next(self):
        """Seeding pre-reads one bar, so latest_bar is not None."""
        h1 = FakeHandler("SPY", _make_bars("SPY", "2023-01-03", 3))
        multi = MultiDataHandler({"SPY": h1})
        # Sub-handler already advanced by heap seeding
        assert multi.get_latest_bar() is not None

    def test_latest_bar_returns_most_recent(self):
        h1 = FakeHandler("SPY", _make_bars("SPY", "2023-01-03", 3))
        h2 = FakeHandler("AAPL", _make_bars("AAPL", "2023-01-03", 3, offset_hours=1))
        multi = MultiDataHandler({"SPY": h1, "AAPL": h2})

        # Read 2 bars (SPY day1, AAPL day1)
        multi.get_next_bar()
        multi.get_next_bar()

        latest = multi.get_latest_bar()
        assert latest is not None
        # AAPL has the later timestamp (offset by 1 hour)
        assert latest.symbol == "AAPL"


class TestMultiHandlerRepr:
    def test_repr(self):
        h1 = FakeHandler("SPY", _make_bars("SPY", "2023-01-03", 3))
        multi = MultiDataHandler({"SPY": h1})
        r = repr(multi)
        assert "MultiDataHandler" in r
        assert "SPY" in r
        assert "total_bars=3" in r


class TestMultiHandlerEdgeCases:
    def test_single_handler(self):
        h1 = FakeHandler("SPY", _make_bars("SPY", "2023-01-03", 3))
        multi = MultiDataHandler({"SPY": h1})

        bars = []
        while (bar := multi.get_next_bar()) is not None:
            bars.append(bar)

        assert len(bars) == 3
        assert all(b.symbol == "SPY" for b in bars)

    def test_empty_handler_raises(self):
        with pytest.raises(ValueError, match="At least one handler"):
            MultiDataHandler({})

    def test_one_handler_empty(self):
        h1 = FakeHandler("SPY", _make_bars("SPY", "2023-01-03", 3))
        h2 = FakeHandler("AAPL", [])
        multi = MultiDataHandler({"SPY": h1, "AAPL": h2})

        bars = []
        while (bar := multi.get_next_bar()) is not None:
            bars.append(bar)

        assert len(bars) == 3
        assert all(b.symbol == "SPY" for b in bars)
