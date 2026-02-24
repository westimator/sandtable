"""
tests/strategy/test_buy_and_hold.py

Tests for BuyAndHoldStrategy - 100% coverage target.
"""

from datetime import datetime, timedelta

from sandtable.core.events import Direction, MarketDataEvent
from sandtable.strategy.buy_and_hold_strategy import BuyAndHoldStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(symbol: str, close: float, day_offset: int = 0) -> MarketDataEvent:
    return MarketDataEvent(
        timestamp=datetime(2024, 1, 1) + timedelta(days=day_offset),
        symbol=symbol,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=1000.0,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_construction(self):
        s = BuyAndHoldStrategy()
        assert s.max_history == 500

    def test_custom_max_history(self):
        s = BuyAndHoldStrategy(max_history=100)
        assert s.max_history == 100

    def test_initial_state_empty(self):
        s = BuyAndHoldStrategy()
        assert s.bar_count == 0
        assert s._entered == set()


# ---------------------------------------------------------------------------
# Signal generation - single symbol
# ---------------------------------------------------------------------------

class TestSingleSymbol:
    def test_first_bar_emits_long_signal(self):
        s = BuyAndHoldStrategy()
        sig = s.on_bar(_bar("SPY", 100.0))
        assert sig is not None
        assert sig.direction == Direction.LONG
        assert sig.symbol == "SPY"
        assert sig.strength == 1.0

    def test_signal_timestamp_matches_bar(self):
        s = BuyAndHoldStrategy()
        bar = _bar("SPY", 100.0, day_offset=5)
        sig = s.on_bar(bar)
        assert sig.timestamp == bar.timestamp

    def test_second_bar_no_signal(self):
        s = BuyAndHoldStrategy()
        s.on_bar(_bar("SPY", 100.0, day_offset=0))
        sig = s.on_bar(_bar("SPY", 105.0, day_offset=1))
        assert sig is None

    def test_many_bars_only_one_signal(self):
        s = BuyAndHoldStrategy()
        signals = []
        for i in range(50):
            sig = s.on_bar(_bar("SPY", 100.0 + i, day_offset=i))
            if sig is not None:
                signals.append(sig)
        assert len(signals) == 1
        assert signals[0].direction == Direction.LONG

    def test_bars_are_stored_in_history(self):
        s = BuyAndHoldStrategy()
        for i in range(5):
            s.on_bar(_bar("SPY", 100.0 + i, day_offset=i))
        assert s.bar_count == 5
        assert s.symbol_bar_count("SPY") == 5


# ---------------------------------------------------------------------------
# Multi-symbol
# ---------------------------------------------------------------------------

class TestMultiSymbol:
    def test_each_symbol_gets_one_signal(self):
        s = BuyAndHoldStrategy()
        sig_spy = s.on_bar(_bar("SPY", 100.0, day_offset=0))
        sig_aapl = s.on_bar(_bar("AAPL", 200.0, day_offset=0))
        sig_msft = s.on_bar(_bar("MSFT", 300.0, day_offset=0))

        assert sig_spy is not None and sig_spy.symbol == "SPY"
        assert sig_aapl is not None and sig_aapl.symbol == "AAPL"
        assert sig_msft is not None and sig_msft.symbol == "MSFT"

    def test_second_bar_per_symbol_no_signal(self):
        s = BuyAndHoldStrategy()
        s.on_bar(_bar("SPY", 100.0, day_offset=0))
        s.on_bar(_bar("AAPL", 200.0, day_offset=0))

        assert s.on_bar(_bar("SPY", 105.0, day_offset=1)) is None
        assert s.on_bar(_bar("AAPL", 210.0, day_offset=1)) is None

    def test_interleaved_symbols(self):
        s = BuyAndHoldStrategy()
        signals = []
        for i in range(10):
            sig = s.on_bar(_bar("SPY", 100.0 + i, day_offset=i))
            if sig:
                signals.append(sig)
            sig = s.on_bar(_bar("AAPL", 200.0 + i, day_offset=i))
            if sig:
                signals.append(sig)

        assert len(signals) == 2
        symbols = {sig.symbol for sig in signals}
        assert symbols == {"SPY", "AAPL"}

    def test_new_symbol_late_entry(self):
        """A symbol appearing mid-stream still gets its signal."""
        s = BuyAndHoldStrategy()
        for i in range(10):
            s.on_bar(_bar("SPY", 100.0, day_offset=i))

        # QQQ appears on bar 10
        sig = s.on_bar(_bar("QQQ", 400.0, day_offset=10))
        assert sig is not None
        assert sig.symbol == "QQQ"
        assert sig.direction == Direction.LONG

    def test_signal_for_one_symbol_doesnt_affect_other(self):
        s = BuyAndHoldStrategy()
        s.on_bar(_bar("SPY", 100.0, day_offset=0))  # triggers SPY signal

        # AAPL should still get its signal
        sig = s.on_bar(_bar("AAPL", 200.0, day_offset=1))
        assert sig is not None and sig.symbol == "AAPL"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_entered_set(self):
        s = BuyAndHoldStrategy()
        s.on_bar(_bar("SPY", 100.0))
        assert "SPY" in s._entered

        s.reset()
        assert s._entered == set()

    def test_reset_clears_bar_history(self):
        s = BuyAndHoldStrategy()
        s.on_bar(_bar("SPY", 100.0))
        assert s.bar_count == 1

        s.reset()
        assert s.bar_count == 0

    def test_signal_fires_again_after_reset(self):
        s = BuyAndHoldStrategy()
        sig1 = s.on_bar(_bar("SPY", 100.0, day_offset=0))
        assert sig1 is not None

        s.reset()

        sig2 = s.on_bar(_bar("SPY", 110.0, day_offset=1))
        assert sig2 is not None
        assert sig2.direction == Direction.LONG

    def test_multi_symbol_reset(self):
        s = BuyAndHoldStrategy()
        s.on_bar(_bar("SPY", 100.0))
        s.on_bar(_bar("AAPL", 200.0))

        s.reset()

        sig_spy = s.on_bar(_bar("SPY", 105.0, day_offset=1))
        sig_aapl = s.on_bar(_bar("AAPL", 205.0, day_offset=1))
        assert sig_spy is not None
        assert sig_aapl is not None


# ---------------------------------------------------------------------------
# Integration with base class helpers
# ---------------------------------------------------------------------------

class TestBaseClassHelpers:
    def test_get_historical_closes(self):
        s = BuyAndHoldStrategy()
        for i in range(5):
            s.on_bar(_bar("SPY", 100.0 + i, day_offset=i))
        closes = s.get_historical_closes(3, symbol="SPY")
        assert closes == [102.0, 103.0, 104.0]

    def test_get_historical_bars(self):
        s = BuyAndHoldStrategy()
        for i in range(5):
            s.on_bar(_bar("SPY", 100.0 + i, day_offset=i))
        bars = s.get_historical_bars(2, symbol="SPY")
        assert len(bars) == 2
        assert bars[0].close == 103.0
        assert bars[1].close == 104.0

    def test_bars_property(self):
        s = BuyAndHoldStrategy()
        s.on_bar(_bar("SPY", 100.0, day_offset=0))
        s.on_bar(_bar("SPY", 101.0, day_offset=1))
        assert len(s.bars) == 2
        assert s.bars[0].close == 100.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_same_symbol_different_prices(self):
        """Even if price changes drastically, only one signal per symbol."""
        s = BuyAndHoldStrategy()
        sig1 = s.on_bar(_bar("SPY", 100.0, day_offset=0))
        sig2 = s.on_bar(_bar("SPY", 0.01, day_offset=1))
        sig3 = s.on_bar(_bar("SPY", 99999.0, day_offset=2))
        assert sig1 is not None
        assert sig2 is None
        assert sig3 is None

    def test_zero_price(self):
        s = BuyAndHoldStrategy()
        sig = s.on_bar(_bar("PENNY", 0.0))
        assert sig is not None
        assert sig.direction == Direction.LONG

    def test_generate_signal_called_via_on_bar(self):
        """generate_signal is called by on_bar, not directly in normal use,
        but calling it directly should also work."""
        s = BuyAndHoldStrategy()
        bar = _bar("SPY", 100.0)
        # call generate_signal directly (bar not in history yet)
        sig = s.generate_signal(bar)
        assert sig is not None
        assert sig.direction == Direction.LONG

        # calling again with same symbol returns None
        sig2 = s.generate_signal(bar)
        assert sig2 is None
