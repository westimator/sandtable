"""
tests/strategy/test_ma_crossover.py

Tests for MACrossoverStrategy, with emphasis on multi-symbol correctness.

The key invariant: each symbol's MAs and crossover state must be
completely independent. A signal for SPY must never depend on AAPL's prices.
"""

from datetime import datetime, timedelta

import pytest

from sandtable.core.events import Direction, MarketDataEvent, SignalEvent
from sandtable.strategy.ma_crossover import MACrossoverStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(symbol: str, close: float, day_offset: int = 0) -> MarketDataEvent:
    return MarketDataEvent(
        timestamp=datetime(2024, 1, 1) + timedelta(days=day_offset),
        symbol=symbol,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000.0,
    )


def _feed_constant(strategy: MACrossoverStrategy, symbol: str, close: float, n: int, day_start: int = 0):
    """Feed n bars of constant price for a symbol, returning all signals."""
    signals = []
    for i in range(n):
        sig = strategy.on_bar(_bar(symbol, close, day_offset=day_start + i))
        if sig is not None:
            signals.append(sig)
    return signals


def _feed_series(strategy: MACrossoverStrategy, symbol: str, closes: list[float], day_start: int = 0):
    """Feed a series of closes for a symbol, returning all signals."""
    signals = []
    for i, close in enumerate(closes):
        sig = strategy.on_bar(_bar(symbol, close, day_offset=day_start + i))
        if sig is not None:
            signals.append(sig)
    return signals


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_fast_must_be_less_than_slow(self):
        with pytest.raises(ValueError, match="fast_period.*must be less than"):
            MACrossoverStrategy(fast_period=30, slow_period=10)

    def test_fast_equals_slow_raises(self):
        with pytest.raises(ValueError):
            MACrossoverStrategy(fast_period=10, slow_period=10)


# ---------------------------------------------------------------------------
# Single-symbol warmup
# ---------------------------------------------------------------------------

class TestWarmup:
    def test_no_signals_during_warmup(self):
        s = MACrossoverStrategy(fast_period=2, slow_period=5)
        signals = _feed_constant(s, "SPY", 100.0, n=4)
        assert signals == []

    def test_first_signal_possible_after_warmup(self):
        """After slow_period bars, the strategy has enough data.
        No crossover yet (constant price), but it shouldn't return None
        due to warmup anymore."""
        s = MACrossoverStrategy(fast_period=2, slow_period=5)
        # Feed 5 bars at 100, then ramp up to force a crossover
        _feed_constant(s, "SPY", 100.0, n=5)
        # Now fast_ma == slow_ma == 100; add rising prices
        signals = _feed_series(s, "SPY", [105.0, 110.0, 115.0], day_start=5)
        # Should get a LONG signal when fast MA crosses above slow MA
        assert any(sig.direction == Direction.LONG for sig in signals)


# ---------------------------------------------------------------------------
# Single-symbol crossover logic
# ---------------------------------------------------------------------------

class TestSingleSymbolCrossover:
    def test_bullish_crossover(self):
        """Price rising after a dip should trigger LONG."""
        s = MACrossoverStrategy(fast_period=2, slow_period=5)
        # 5 bars at 100 (warmup), then prices rise sharply
        _feed_constant(s, "SPY", 100.0, n=5)
        signals = _feed_series(s, "SPY", [110.0, 120.0, 130.0], day_start=5)
        long_signals = [sig for sig in signals if sig.direction == Direction.LONG]
        assert len(long_signals) >= 1
        assert all(sig.symbol == "SPY" for sig in long_signals)

    def test_bearish_crossover(self):
        """Price dropping after rise should trigger SHORT."""
        s = MACrossoverStrategy(fast_period=2, slow_period=5)
        # Rising trend establishes fast MA > slow MA
        _feed_series(s, "SPY", [95.0, 100.0, 105.0, 110.0, 115.0], day_start=0)
        # Sharp reversal — fast MA crosses below slow MA
        signals = _feed_series(s, "SPY", [90.0, 80.0, 70.0], day_start=5)
        short_signals = [sig for sig in signals if sig.direction == Direction.SHORT]
        assert len(short_signals) >= 1
        assert all(sig.symbol == "SPY" for sig in short_signals)

    def test_no_signal_on_flat_prices(self):
        """Constant price should never produce a crossover."""
        s = MACrossoverStrategy(fast_period=2, slow_period=5)
        signals = _feed_constant(s, "SPY", 100.0, n=20)
        assert signals == []

    def test_signal_strength_is_one(self):
        s = MACrossoverStrategy(fast_period=2, slow_period=5)
        _feed_constant(s, "SPY", 100.0, n=5)
        signals = _feed_series(s, "SPY", [110.0, 120.0], day_start=5)
        for sig in signals:
            assert sig.strength == 1.0


# ---------------------------------------------------------------------------
# Multi-symbol: THE CRITICAL TESTS
#
# These tests would have caught the original bug where:
# 1. get_historical_closes() returned mixed-symbol prices
# 2. _prev_fast_ma / _prev_slow_ma were shared scalars, not per-symbol dicts
# ---------------------------------------------------------------------------

class TestMultiSymbolIndependence:
    """Each symbol's warmup, MA calculation, and crossover state must
    be completely independent."""

    def test_per_symbol_warmup(self):
        """Symbol A should not warm up Symbol B."""
        s = MACrossoverStrategy(fast_period=2, slow_period=5)
        # Feed 10 bars for SPY — it's warmed up
        _feed_constant(s, "SPY", 100.0, n=10)
        assert s.symbol_bar_count("SPY") == 10
        assert s.symbol_bar_count("AAPL") == 0

        # Feed only 3 bars for AAPL — not enough for slow_period=5
        signals = _feed_constant(s, "AAPL", 200.0, n=3, day_start=10)
        # AAPL should produce NO signals (still in warmup)
        aapl_signals = [sig for sig in signals if sig.symbol == "AAPL"]
        assert aapl_signals == []

    def test_per_symbol_ma_calculation(self):
        """MAs for SPY must use only SPY prices, not AAPL's.

        This is the exact bug that was caught: with interleaved bars,
        get_historical_closes(5) without a symbol filter would return
        mixed prices from both symbols."""
        s = MACrossoverStrategy(fast_period=2, slow_period=5)

        # Interleave SPY at 100 and AAPL at 500 — very different prices
        for i in range(6):
            s.on_bar(_bar("SPY", 100.0, day_offset=i))
            s.on_bar(_bar("AAPL", 500.0, day_offset=i))

        # SPY's slow MA should be ~100, not ~300 (average of mixed prices)
        spy_closes = s.get_historical_closes(5, symbol="SPY")
        assert all(abs(c - 100.0) < 0.01 for c in spy_closes), (
            f"SPY closes should be ~100 but got {spy_closes}"
        )

        aapl_closes = s.get_historical_closes(5, symbol="AAPL")
        assert all(abs(c - 500.0) < 0.01 for c in aapl_closes), (
            f"AAPL closes should be ~500 but got {aapl_closes}"
        )

    def test_crossover_independent_per_symbol(self):
        """A crossover in SPY should not affect AAPL signals."""
        s = MACrossoverStrategy(fast_period=2, slow_period=5)

        # Both symbols warm up with constant prices
        for i in range(5):
            s.on_bar(_bar("SPY", 100.0, day_offset=i))
            s.on_bar(_bar("AAPL", 100.0, day_offset=i))

        # SPY prices rise (should trigger LONG crossover)
        # AAPL stays flat (should trigger nothing)
        spy_signals = []
        aapl_signals = []
        for i in range(3):
            day = 5 + i
            sig = s.on_bar(_bar("SPY", 110.0 + i * 10, day_offset=day))
            if sig:
                spy_signals.append(sig)
            sig = s.on_bar(_bar("AAPL", 100.0, day_offset=day))
            if sig:
                aapl_signals.append(sig)

        # SPY should have a LONG signal
        assert len(spy_signals) >= 1
        assert all(sig.symbol == "SPY" for sig in spy_signals)
        assert any(sig.direction == Direction.LONG for sig in spy_signals)

        # AAPL should have NO signals (flat price, no crossover)
        assert aapl_signals == []

    def test_opposite_signals_for_diverging_symbols(self):
        """SPY going up and AAPL going down should produce
        LONG for SPY and SHORT for AAPL."""
        s = MACrossoverStrategy(fast_period=2, slow_period=5)

        # Warmup with same prices
        for i in range(5):
            s.on_bar(_bar("SPY", 100.0, day_offset=i))
            s.on_bar(_bar("AAPL", 100.0, day_offset=i))

        # Now diverge: SPY up, AAPL down
        spy_signals = []
        aapl_signals = []
        for i in range(3):
            day = 5 + i
            sig = s.on_bar(_bar("SPY", 110.0 + i * 10, day_offset=day))
            if sig:
                spy_signals.append(sig)
            sig = s.on_bar(_bar("AAPL", 90.0 - i * 10, day_offset=day))
            if sig:
                aapl_signals.append(sig)

        spy_directions = {sig.direction for sig in spy_signals}
        aapl_directions = {sig.direction for sig in aapl_signals}

        assert Direction.LONG in spy_directions, f"Expected SPY LONG, got {spy_directions}"
        assert Direction.SHORT in aapl_directions, f"Expected AAPL SHORT, got {aapl_directions}"

    def test_three_symbols_all_independent(self):
        """Three symbols, each with different behavior."""
        s = MACrossoverStrategy(fast_period=2, slow_period=5)

        # Warmup all three
        for i in range(5):
            s.on_bar(_bar("A", 100.0, day_offset=i))
            s.on_bar(_bar("B", 100.0, day_offset=i))
            s.on_bar(_bar("C", 100.0, day_offset=i))

        signals_by_symbol: dict[str, list[SignalEvent]] = {"A": [], "B": [], "C": []}
        for i in range(3):
            day = 5 + i
            # A: rising → expect LONG
            sig = s.on_bar(_bar("A", 120.0 + i * 10, day_offset=day))
            if sig:
                signals_by_symbol["A"].append(sig)
            # B: flat → expect nothing
            sig = s.on_bar(_bar("B", 100.0, day_offset=day))
            if sig:
                signals_by_symbol["B"].append(sig)
            # C: falling → expect SHORT
            sig = s.on_bar(_bar("C", 80.0 - i * 10, day_offset=day))
            if sig:
                signals_by_symbol["C"].append(sig)

        assert any(sig.direction == Direction.LONG for sig in signals_by_symbol["A"])
        assert signals_by_symbol["B"] == []
        assert any(sig.direction == Direction.SHORT for sig in signals_by_symbol["C"])

    def test_late_arriving_symbol(self):
        """A symbol that starts appearing after others are already warmed up
        must still go through its own warmup period."""
        s = MACrossoverStrategy(fast_period=2, slow_period=5)

        # SPY feeds for 10 bars
        _feed_constant(s, "SPY", 100.0, n=10)

        # AAPL starts late — needs its own 5-bar warmup
        late_signals = []
        for i in range(4):
            sig = s.on_bar(_bar("AAPL", 200.0, day_offset=10 + i))
            if sig:
                late_signals.append(sig)

        # AAPL has only 4 bars, still in warmup
        assert late_signals == []
        assert s.symbol_bar_count("AAPL") == 4


# ---------------------------------------------------------------------------
# Reset with multi-symbol state
# ---------------------------------------------------------------------------

class TestMultiSymbolReset:
    def test_reset_clears_per_symbol_ma_state(self):
        s = MACrossoverStrategy(fast_period=2, slow_period=5)
        _feed_constant(s, "SPY", 100.0, n=6)
        _feed_constant(s, "AAPL", 200.0, n=6, day_start=0)

        # MA state should exist for both symbols
        assert "SPY" in s._prev_fast_ma
        assert "AAPL" in s._prev_fast_ma

        s.reset()

        assert s._prev_fast_ma == {}
        assert s._prev_slow_ma == {}
        assert s.bar_count == 0

    def test_strategy_works_correctly_after_reset(self):
        s = MACrossoverStrategy(fast_period=2, slow_period=5)
        _feed_constant(s, "SPY", 100.0, n=6)
        s.reset()

        # After reset, SPY needs full warmup again
        signals = _feed_constant(s, "SPY", 100.0, n=4)
        assert signals == []  # still in warmup
