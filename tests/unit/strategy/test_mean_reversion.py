"""
tests/strategy/test_mean_reversion.py

Tests for MeanReversionStrategy, including multi-symbol correctness.
"""

from datetime import datetime, timedelta

import pytest

from sandtable.core.events import Direction, MarketDataEvent
from sandtable.strategy.mean_reversion_strategy import MeanReversionStrategy

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


def _feed_series(strategy, symbol: str, closes: list[float], day_start: int = 0):
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
    def test_lookback_must_be_at_least_2(self):
        with pytest.raises(ValueError, match="lookback.*must be >= 2"):
            MeanReversionStrategy(lookback=1)

    def test_threshold_must_be_positive(self):
        with pytest.raises(ValueError, match="threshold.*must be > 0"):
            MeanReversionStrategy(threshold=0)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            MeanReversionStrategy(threshold=-1.0)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

class TestWarmup:
    def test_no_signals_during_warmup(self):
        s = MeanReversionStrategy(lookback=10, threshold=2.0)
        signals = _feed_series(s, "SPY", [100.0 + i for i in range(9)])
        assert signals == []

    def test_signal_possible_after_warmup(self):
        s = MeanReversionStrategy(lookback=5, threshold=1.0)
        # 5 bars at 100 (establishes mean), then price drops sharply
        closes = [100.0] * 5 + [80.0]
        signals = _feed_series(s, "SPY", closes)
        # z-score of 80 relative to mean=100 should trigger LONG
        assert len(signals) >= 1


# ---------------------------------------------------------------------------
# Signal direction
# ---------------------------------------------------------------------------

class TestSignalDirection:
    def test_price_below_mean_triggers_long(self):
        """Price far below moving average should trigger LONG (buy the dip)."""
        s = MeanReversionStrategy(lookback=5, threshold=1.0)
        closes = [100.0] * 5 + [70.0]  # sharp drop
        signals = _feed_series(s, "SPY", closes)
        long_signals = [sig for sig in signals if sig.direction == Direction.LONG]
        assert len(long_signals) >= 1

    def test_price_above_mean_triggers_short(self):
        """Price far above moving average should trigger SHORT."""
        s = MeanReversionStrategy(lookback=5, threshold=1.0)
        closes = [100.0] * 5 + [130.0]  # sharp rise
        signals = _feed_series(s, "SPY", closes)
        short_signals = [sig for sig in signals if sig.direction == Direction.SHORT]
        assert len(short_signals) >= 1

    def test_price_at_mean_no_signal(self):
        """Price near the mean should not trigger a signal."""
        s = MeanReversionStrategy(lookback=5, threshold=2.0)
        closes = [100.0] * 10  # constant price
        signals = _feed_series(s, "SPY", closes)
        assert signals == []

    def test_zero_std_no_signal(self):
        """When all prices are identical, std=0 and no signal should fire."""
        s = MeanReversionStrategy(lookback=5, threshold=0.5)
        closes = [50.0] * 10
        signals = _feed_series(s, "SPY", closes)
        assert signals == []

    def test_uses_sample_std_not_population(self):
        """
        Regression: strategy must use sample std (N-1), not population std (N).

        With lookback=2 and prices [100, 104], the bar.close is the last price
        fed. The lookback window after feeding 2 bars is [100, 104].
        mean = 102, pop_std = 2.0, sample_std = 2.828...
        With threshold=1.0 and bar.close=104:
          pop z-score = (104-102)/2.0 = 1.0 (exactly at threshold, no signal)
          sample z-score = (104-102)/2.828 = 0.707 (below threshold, no signal)
        With threshold=0.5:
          pop z-score = 1.0 / 0.5 ... but that's still > threshold -> signal
          sample z-score = 0.707 -> exceeds 0.5 -> signal with strength (0.707-0.5)/0.5=0.414

        The key: with population std and threshold=1.0, z=1.0 is NOT > threshold,
        so no signal. With sample std and threshold=0.5, z=0.707 IS > threshold.
        We test that the z-score matches the sample std calculation.
        """
        s = MeanReversionStrategy(lookback=2, threshold=0.5)
        closes = [100.0, 104.0]
        signals = _feed_series(s, "SPY", closes)
        # sample z-score = (104-102)/sqrt(8) = 0.707, threshold=0.5
        # strength = (0.707 - 0.5) / 0.5 = 0.414
        assert len(signals) == 1
        assert signals[0].strength == pytest.approx(0.414, abs=0.02)


# ---------------------------------------------------------------------------
# Signal strength
# ---------------------------------------------------------------------------

class TestSignalStrength:
    def test_strength_capped_at_1(self):
        s = MeanReversionStrategy(lookback=5, threshold=1.0)
        closes = [100.0] * 5 + [50.0]  # extreme deviation
        signals = _feed_series(s, "SPY", closes)
        assert len(signals) >= 1
        assert all(sig.strength <= 1.0 for sig in signals)

    def test_strength_starts_near_zero_at_threshold(self):
        """Signal just barely past threshold should have strength near 0, not 1."""
        # use a series with some spread
        closes = [100.0 + (i % 3 - 1) * 0.5 for i in range(20)]
        s2 = MeanReversionStrategy(lookback=20, threshold=0.5)
        # feed warmup
        for i, c in enumerate(closes):
            s2.on_bar(_bar("SPY", c, day_offset=i))
        # now send a bar just past threshold
        mean_val = sum(closes) / len(closes)
        std_val = (sum((c - mean_val) ** 2 for c in closes) / (len(closes) - 1)) ** 0.5
        # z-score of ~0.6 with threshold=0.5 -> excess=0.1 -> strength=0.1/0.5=0.2
        target_z = 0.6
        target_price = mean_val + target_z * std_val
        sig = s2.on_bar(_bar("SPY", target_price, day_offset=20))
        assert sig is not None
        assert sig.strength < 0.5, f"strength={sig.strength} should be well below 1.0 near threshold"

    def test_strength_is_not_always_one(self):
        """Regression: old formula always produced strength=1.0 because
        abs(z_score)/threshold > 1 whenever a signal fires. The fix uses
        (abs(z_score) - threshold) / threshold so strength varies."""
        s = MeanReversionStrategy(lookback=5, threshold=2.0)
        # 5 bars at 100, then small deviation
        closes = [100.0] * 5 + [90.0]
        signals = _feed_series(s, "SPY", closes)
        if signals:
            assert signals[0].strength < 1.0, (
                "Signal strength should be < 1.0 for moderate deviations"
            )

    def test_strength_scales_with_deviation(self):
        """Larger deviations should produce strictly stronger signals."""
        # use lookback=50 with alternating prices so there's a non-trivial
        # std dev, and a single test bar barely shifts the window stats.
        warmup = [99.0, 101.0] * 25  # 50 bars, mean~100, std~1

        s1 = MeanReversionStrategy(lookback=50, threshold=2.0)
        _feed_series(s1, "SPY", warmup)
        sig_mild = s1.on_bar(_bar("SPY", 96.0, day_offset=50))

        s2 = MeanReversionStrategy(lookback=50, threshold=2.0)
        _feed_series(s2, "SPY", warmup)
        sig_extreme = s2.on_bar(_bar("SPY", 90.0, day_offset=50))

        assert sig_mild is not None, "mild deviation should produce a signal"
        assert sig_extreme is not None, "extreme deviation should produce a signal"
        assert sig_mild.strength < 1.0, "mild signal should not be capped"
        assert sig_extreme.strength > sig_mild.strength


# ---------------------------------------------------------------------------
# Multi-symbol: per-symbol independence
# ---------------------------------------------------------------------------

class TestMultiSymbolIndependence:
    def test_per_symbol_warmup(self):
        """Each symbol needs its own lookback bars before signals fire."""
        s = MeanReversionStrategy(lookback=5, threshold=1.0)

        # SPY gets 10 bars, AAPL only 3
        for i in range(10):
            s.on_bar(_bar("SPY", 100.0, day_offset=i))
        for i in range(3):
            s.on_bar(_bar("AAPL", 200.0, day_offset=10 + i))

        assert s.symbol_bar_count("SPY") == 10
        assert s.symbol_bar_count("AAPL") == 3

        # AAPL extreme price should NOT trigger signal (still in warmup)
        sig = s.on_bar(_bar("AAPL", 50.0, day_offset=13))
        assert sig is None  # only 4 bars for AAPL, need 5

    def test_per_symbol_ma_calculation(self):
        """SPY's mean should be ~100, AAPL's mean should be ~500,
        even when bars are interleaved."""
        s = MeanReversionStrategy(lookback=5, threshold=1.0)

        for i in range(5):
            s.on_bar(_bar("SPY", 100.0, day_offset=i))
            s.on_bar(_bar("AAPL", 500.0, day_offset=i))

        # SPY at 80 is far below SPY's mean of 100 → LONG
        sig_spy = s.on_bar(_bar("SPY", 80.0, day_offset=5))
        assert sig_spy is not None
        assert sig_spy.symbol == "SPY"
        assert sig_spy.direction == Direction.LONG

        # AAPL at 80 is far below AAPL's mean of 500 → LONG
        # but more importantly, the deviation is relative to AAPL's mean, not SPY's
        sig_aapl = s.on_bar(_bar("AAPL", 300.0, day_offset=5))
        assert sig_aapl is not None
        assert sig_aapl.symbol == "AAPL"
        assert sig_aapl.direction == Direction.LONG

    def test_signal_for_one_symbol_doesnt_affect_other(self):
        """SPY triggering LONG should not influence AAPL's state."""
        s = MeanReversionStrategy(lookback=5, threshold=1.0)

        for i in range(5):
            s.on_bar(_bar("SPY", 100.0, day_offset=i))
            s.on_bar(_bar("AAPL", 100.0, day_offset=i))

        # SPY drops → triggers LONG
        sig_spy = s.on_bar(_bar("SPY", 70.0, day_offset=5))
        assert sig_spy is not None and sig_spy.direction == Direction.LONG

        # AAPL stays at mean → no signal
        sig_aapl = s.on_bar(_bar("AAPL", 100.0, day_offset=5))
        assert sig_aapl is None

    def test_opposite_signals_per_symbol(self):
        """SPY dropping (LONG) while AAPL rising (SHORT)."""
        s = MeanReversionStrategy(lookback=5, threshold=1.0)

        for i in range(5):
            s.on_bar(_bar("SPY", 100.0, day_offset=i))
            s.on_bar(_bar("AAPL", 100.0, day_offset=i))

        sig_spy = s.on_bar(_bar("SPY", 70.0, day_offset=5))
        sig_aapl = s.on_bar(_bar("AAPL", 130.0, day_offset=5))

        assert sig_spy is not None and sig_spy.direction == Direction.LONG
        assert sig_aapl is not None and sig_aapl.direction == Direction.SHORT
