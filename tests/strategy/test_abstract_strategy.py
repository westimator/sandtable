"""
tests/strategy/test_abstract_strategy.py

Tests for AbstractStrategy base class, focusing on per-symbol
bar history filtering, warmup counting, and reset behavior.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from sandtable.core.events import Direction, MarketDataEvent, SignalEvent
from sandtable.strategy.abstract_strategy import AbstractStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(symbol: str, close: float, day_offset: int = 0) -> MarketDataEvent:
    """Create a minimal MarketDataEvent for testing."""
    return MarketDataEvent(
        timestamp=datetime(2024, 1, 1) + timedelta(days=day_offset),
        symbol=symbol,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000.0,
    )


@dataclass
class AlwaysLongStrategy(AbstractStrategy):
    """Trivial strategy that always goes LONG — used to test base class."""

    def generate_signal(self, bar: MarketDataEvent) -> SignalEvent | None:
        return SignalEvent(
            timestamp=bar.timestamp,
            symbol=bar.symbol,
            direction=Direction.LONG,
            strength=1.0,
        )


# ---------------------------------------------------------------------------
# bar_count / symbol_bar_count
# ---------------------------------------------------------------------------

class TestBarCount:
    def test_bar_count_starts_at_zero(self):
        s = AlwaysLongStrategy()
        assert s.bar_count == 0

    def test_bar_count_increments(self):
        s = AlwaysLongStrategy()
        s.on_bar(_bar("SPY", 100.0))
        assert s.bar_count == 1
        s.on_bar(_bar("SPY", 101.0, day_offset=1))
        assert s.bar_count == 2

    def test_symbol_bar_count_single_symbol(self):
        s = AlwaysLongStrategy()
        for i in range(5):
            s.on_bar(_bar("SPY", 100.0 + i, day_offset=i))
        assert s.symbol_bar_count("SPY") == 5

    def test_symbol_bar_count_multi_symbol(self):
        s = AlwaysLongStrategy()
        s.on_bar(_bar("SPY", 100.0, day_offset=0))
        s.on_bar(_bar("AAPL", 150.0, day_offset=0))
        s.on_bar(_bar("SPY", 101.0, day_offset=1))
        s.on_bar(_bar("AAPL", 151.0, day_offset=1))
        s.on_bar(_bar("SPY", 102.0, day_offset=2))

        assert s.bar_count == 5  # total
        assert s.symbol_bar_count("SPY") == 3
        assert s.symbol_bar_count("AAPL") == 2
        assert s.symbol_bar_count("MSFT") == 0  # never seen


# ---------------------------------------------------------------------------
# get_historical_closes — per-symbol filtering
# ---------------------------------------------------------------------------

class TestGetHistoricalCloses:
    def test_returns_all_when_fewer_than_n(self):
        s = AlwaysLongStrategy()
        s.on_bar(_bar("SPY", 100.0))
        s.on_bar(_bar("SPY", 101.0, day_offset=1))
        closes = s.get_historical_closes(10)
        assert closes == [100.0, 101.0]

    def test_returns_last_n(self):
        s = AlwaysLongStrategy()
        for i in range(10):
            s.on_bar(_bar("SPY", 100.0 + i, day_offset=i))
        closes = s.get_historical_closes(3)
        assert closes == [107.0, 108.0, 109.0]

    def test_without_symbol_returns_mixed(self):
        """Without symbol filter, closes from all symbols are interleaved."""
        s = AlwaysLongStrategy()
        s.on_bar(_bar("SPY", 100.0, day_offset=0))
        s.on_bar(_bar("AAPL", 200.0, day_offset=0))
        s.on_bar(_bar("SPY", 101.0, day_offset=1))
        s.on_bar(_bar("AAPL", 201.0, day_offset=1))

        closes = s.get_historical_closes(10)
        assert closes == [100.0, 200.0, 101.0, 201.0]

    def test_with_symbol_filters_correctly(self):
        """With symbol filter, only that symbol's closes are returned."""
        s = AlwaysLongStrategy()
        s.on_bar(_bar("SPY", 100.0, day_offset=0))
        s.on_bar(_bar("AAPL", 200.0, day_offset=0))
        s.on_bar(_bar("SPY", 101.0, day_offset=1))
        s.on_bar(_bar("AAPL", 201.0, day_offset=1))
        s.on_bar(_bar("SPY", 102.0, day_offset=2))

        spy_closes = s.get_historical_closes(10, symbol="SPY")
        assert spy_closes == [100.0, 101.0, 102.0]

        aapl_closes = s.get_historical_closes(10, symbol="AAPL")
        assert aapl_closes == [200.0, 201.0]

    def test_with_symbol_limits_to_n(self):
        s = AlwaysLongStrategy()
        for i in range(10):
            s.on_bar(_bar("SPY", 100.0 + i, day_offset=i))
            s.on_bar(_bar("AAPL", 200.0 + i, day_offset=i))

        spy_closes = s.get_historical_closes(3, symbol="SPY")
        assert spy_closes == [107.0, 108.0, 109.0]

        aapl_closes = s.get_historical_closes(2, symbol="AAPL")
        assert aapl_closes == [208.0, 209.0]

    def test_unknown_symbol_returns_empty(self):
        s = AlwaysLongStrategy()
        s.on_bar(_bar("SPY", 100.0))
        assert s.get_historical_closes(5, symbol="MSFT") == []

    def test_three_symbols_interleaved(self):
        """Ensure filtering works correctly with 3+ symbols."""
        s = AlwaysLongStrategy()
        for i in range(5):
            s.on_bar(_bar("A", 10.0 + i, day_offset=i))
            s.on_bar(_bar("B", 20.0 + i, day_offset=i))
            s.on_bar(_bar("C", 30.0 + i, day_offset=i))

        assert s.bar_count == 15
        assert s.get_historical_closes(5, symbol="A") == [10.0, 11.0, 12.0, 13.0, 14.0]
        assert s.get_historical_closes(5, symbol="B") == [20.0, 21.0, 22.0, 23.0, 24.0]
        assert s.get_historical_closes(3, symbol="C") == [32.0, 33.0, 34.0]


# ---------------------------------------------------------------------------
# get_historical_bars — per-symbol filtering
# ---------------------------------------------------------------------------

class TestGetHistoricalBars:
    def test_without_symbol_returns_all(self):
        s = AlwaysLongStrategy()
        s.on_bar(_bar("SPY", 100.0, day_offset=0))
        s.on_bar(_bar("AAPL", 200.0, day_offset=0))
        bars = s.get_historical_bars(10)
        assert len(bars) == 2
        assert bars[0].symbol == "SPY"
        assert bars[1].symbol == "AAPL"

    def test_with_symbol_filters(self):
        s = AlwaysLongStrategy()
        s.on_bar(_bar("SPY", 100.0, day_offset=0))
        s.on_bar(_bar("AAPL", 200.0, day_offset=0))
        s.on_bar(_bar("SPY", 101.0, day_offset=1))

        spy_bars = s.get_historical_bars(10, symbol="SPY")
        assert len(spy_bars) == 2
        assert all(b.symbol == "SPY" for b in spy_bars)

        aapl_bars = s.get_historical_bars(10, symbol="AAPL")
        assert len(aapl_bars) == 1
        assert aapl_bars[0].symbol == "AAPL"

    def test_with_symbol_limits_to_n(self):
        s = AlwaysLongStrategy()
        for i in range(10):
            s.on_bar(_bar("SPY", 100.0 + i, day_offset=i))

        bars = s.get_historical_bars(3, symbol="SPY")
        assert len(bars) == 3
        assert [b.close for b in bars] == [107.0, 108.0, 109.0]


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_bar_history(self):
        s = AlwaysLongStrategy()
        s.on_bar(_bar("SPY", 100.0))
        s.on_bar(_bar("AAPL", 200.0))
        assert s.bar_count == 2

        s.reset()
        assert s.bar_count == 0
        assert s.symbol_bar_count("SPY") == 0
        assert s.symbol_bar_count("AAPL") == 0
        assert s.get_historical_closes(10) == []

    def test_reset_allows_fresh_bars(self):
        s = AlwaysLongStrategy()
        s.on_bar(_bar("SPY", 100.0))
        s.reset()
        s.on_bar(_bar("SPY", 999.0))
        assert s.get_historical_closes(10) == [999.0]


# ---------------------------------------------------------------------------
# max_history limit
# ---------------------------------------------------------------------------

class TestMaxHistory:
    def test_max_history_evicts_old_bars(self):
        s = AlwaysLongStrategy(max_history=5)
        for i in range(10):
            s.on_bar(_bar("SPY", float(i), day_offset=i))
        assert s.bar_count == 5
        assert s.get_historical_closes(10) == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_max_history_per_symbol(self):
        """With max_history=6, interleaving 2 symbols keeps ~3 each."""
        s = AlwaysLongStrategy(max_history=6)
        for i in range(5):
            s.on_bar(_bar("SPY", 100.0 + i, day_offset=i))
            s.on_bar(_bar("AAPL", 200.0 + i, day_offset=i))

        # deque has last 6 bars total
        assert s.bar_count == 6
        spy_closes = s.get_historical_closes(10, symbol="SPY")
        aapl_closes = s.get_historical_closes(10, symbol="AAPL")
        # Exact split depends on interleaving; important thing is both
        # symbols still have bars and old ones were evicted
        assert len(spy_closes) + len(aapl_closes) == 6
        assert len(spy_closes) >= 2
        assert len(aapl_closes) >= 2
