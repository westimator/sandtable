"""
Tests for historical Value-at-Risk computation.
"""

import pytest

from sandtable.risk.var import compute_var


class TestComputeVar:
    def test_empty_returns(self):
        """Empty input returns 0."""
        assert compute_var([]) == 0.0

    def test_single_return(self):
        """Single return is insufficient data, returns 0."""
        assert compute_var([0.01]) == 0.0

    def test_all_positive_returns(self):
        """All positive returns - VaR should be the smallest positive return."""
        returns = [0.01, 0.02, 0.03, 0.04, 0.05] * 20  # 100 returns
        var = compute_var(returns, lookback=100, confidence=0.95)
        # 5th percentile of all-positive returns is 0.01
        assert var == pytest.approx(0.01)

    def test_negative_returns_give_negative_var(self):
        """Distribution with losses gives negative VaR."""
        # 90 positive days, 10 bad days
        returns = [0.01] * 90 + [-0.05] * 10
        var = compute_var(returns, lookback=100, confidence=0.95)
        assert var < 0

    def test_confidence_level_affects_var(self):
        """Higher confidence produces more extreme (lower) VaR."""
        returns = [0.01] * 80 + [-0.02] * 10 + [-0.05] * 10
        var_95 = compute_var(returns, lookback=100, confidence=0.95)
        var_99 = compute_var(returns, lookback=100, confidence=0.99)
        assert var_99 <= var_95

    def test_lookback_limits_window(self):
        """Lookback parameter limits how many returns are used."""
        old_returns = [-0.10] * 50  # old bad data
        recent_returns = [0.01] * 50  # recent good data
        all_returns = old_returns + recent_returns

        var_full = compute_var(all_returns, lookback=100, confidence=0.95)
        var_recent = compute_var(all_returns, lookback=50, confidence=0.95)

        # recent-only should have better (higher) VaR
        assert var_recent > var_full

    def test_known_distribution(self):
        """Test with a known uniform distribution."""
        # 100 returns from -0.05 to 0.05
        returns = [i * 0.001 - 0.05 for i in range(101)]
        var = compute_var(returns, lookback=101, confidence=0.95)
        # 5th percentile of [-0.05, ..., 0.05]: index 5 = -0.045
        assert var == pytest.approx(-0.045, abs=0.002)
