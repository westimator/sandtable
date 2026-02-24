"""
tests/unit/stats/test_significance.py

Tests for statistical significance testing module.
"""

import math
import random

import pytest

from sandtable.stats.significance import (
    TRADING_DAYS_PER_YEAR,
    SignificanceResult,
    _compute_sharpe,
    bootstrap_sharpe,
    permutation_test_sharpe,
    run_significance_tests,
    ttest_mean_return,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _constant_returns(r: float, n: int) -> list[float]:
    """Returns [r] * n."""
    return [r] * n


def _zero_returns(n: int) -> list[float]:
    """Returns [0.0] * n."""
    return [0.0] * n


def _trending_returns(n: int, daily_r: float = 0.001, seed: int = 42) -> list[float]:
    """Returns with positive drift + small noise (should be significant)."""
    rng = random.Random(seed)
    return [daily_r + rng.gauss(0, daily_r * 0.5) for _ in range(n)]


def _random_returns(n: int, seed: int = 42) -> list[float]:
    """Zero-mean Gaussian noise (should NOT be significant).

    Returns are centered to have exactly zero sample mean so any
    particular seed doesn't accidentally produce a significant drift.
    """
    rng = random.Random(seed)
    raw = [rng.gauss(0, 0.01) for _ in range(n)]
    mean_r = sum(raw) / len(raw)
    return [r - mean_r for r in raw]


# ---------------------------------------------------------------------------
# TestComputeSharpe
# ---------------------------------------------------------------------------


class TestComputeSharpe:
    def test_zero_returns(self):
        """Sharpe is 0.0 for zero returns."""
        assert _compute_sharpe(_zero_returns(100)) == 0.0

    def test_constant_positive_returns(self):
        """Std is 0, returns 0.0 (edge case)."""
        assert _compute_sharpe(_constant_returns(0.01, 100)) == 0.0

    def test_positive_drift_returns(self):
        """Sharpe > 0 for trending returns."""
        returns = _trending_returns(252, daily_r=0.001)
        sharpe = _compute_sharpe(returns)
        assert sharpe > 0

    def test_empty_returns(self):
        """Returns 0.0 for empty list."""
        assert _compute_sharpe([]) == 0.0

    def test_single_return(self):
        """Returns 0.0 for single return (not enough data)."""
        assert _compute_sharpe([0.01]) == 0.0


# ---------------------------------------------------------------------------
# TestPermutationTestSharpe
# ---------------------------------------------------------------------------


class TestPermutationTestSharpe:
    def test_strong_signal_is_significant(self):
        """Trending returns yield p_value < 0.05."""
        returns = _trending_returns(252, daily_r=0.002, seed=42)
        result = permutation_test_sharpe(returns, n_simulations=500, random_seed=42)

        assert result.p_value < 0.05
        assert result.is_significant

    def test_random_returns_not_significant(self):
        """Zero-mean noise yields p_value >= 0.05."""
        returns = _random_returns(252, seed=42)
        result = permutation_test_sharpe(returns, n_simulations=500, random_seed=42)

        # Random returns should generally not be significant
        # (the shuffled Sharpes will be similar to the observed)
        assert result.p_value >= 0.05

    def test_seed_reproducibility(self):
        """Same seed produces same simulated_statistics."""
        returns = _trending_returns(100)
        r1 = permutation_test_sharpe(returns, n_simulations=100, random_seed=123)
        r2 = permutation_test_sharpe(returns, n_simulations=100, random_seed=123)

        assert r1.simulated_statistics == r2.simulated_statistics
        assert r1.p_value == r2.p_value

    def test_n_simulations_count(self):
        """len(result.simulated_statistics) == n_simulations."""
        returns = _trending_returns(50)
        result = permutation_test_sharpe(returns, n_simulations=200, random_seed=42)

        assert len(result.simulated_statistics) == 200

    def test_p_value_in_valid_range(self):
        """0.0 <= p_value <= 1.0."""
        returns = _trending_returns(100)
        result = permutation_test_sharpe(returns, n_simulations=100, random_seed=42)

        assert 0.0 <= result.p_value <= 1.0

    def test_result_fields_populated(self):
        """All fields of SignificanceResult are set."""
        returns = _trending_returns(100)
        result = permutation_test_sharpe(returns, n_simulations=50, random_seed=42)

        assert result.test_name == "Permutation Test (Sharpe)"
        assert isinstance(result.observed_statistic, float)
        assert isinstance(result.p_value, float)
        assert result.n_simulations == 50
        assert len(result.simulated_statistics) == 50
        assert isinstance(result.is_significant, bool)
        assert result.significance_level == 0.05
        assert isinstance(result.null_mean, float)
        assert isinstance(result.null_std, float)


# ---------------------------------------------------------------------------
# TestTtestMeanReturn
# ---------------------------------------------------------------------------


class TestTtestMeanReturn:
    def test_positive_mean_significant(self):
        """Large positive drift yields p_value < 0.05."""
        returns = _trending_returns(252, daily_r=0.002, seed=42)
        result = ttest_mean_return(returns)

        assert result.p_value < 0.05
        assert result.is_significant

    def test_zero_mean_not_significant(self):
        """Zero returns yield high p_value."""
        returns = _zero_returns(100)
        result = ttest_mean_return(returns)

        assert result.p_value >= 0.05

    def test_negative_mean_not_significant(self):
        """Negative drift, one-tailed test (testing for positive mean)."""
        returns = _trending_returns(252, daily_r=-0.002, seed=42)
        result = ttest_mean_return(returns)

        assert result.p_value > 0.5  # one-tailed, negative mean -> large p

    def test_p_value_in_range(self):
        """0.0 <= p_value <= 1.0."""
        returns = _random_returns(100)
        result = ttest_mean_return(returns)

        assert 0.0 <= result.p_value <= 1.0

    def test_single_return(self):
        """Edge case - should not crash, returns not significant."""
        result = ttest_mean_return([0.01])

        assert result.p_value == 1.0
        assert not result.is_significant

    def test_two_returns(self):
        """Minimal degrees of freedom (df=1)."""
        result = ttest_mean_return([0.01, 0.02])

        assert 0.0 <= result.p_value <= 1.0
        assert result.test_name == "t-test (Mean Return)"


# ---------------------------------------------------------------------------
# TestBootstrapSharpe
# ---------------------------------------------------------------------------


class TestBootstrapSharpe:
    def test_strong_signal_is_significant(self):
        """Trending returns yield p_value < 0.05."""
        returns = _trending_returns(252, daily_r=0.002, seed=42)
        result = bootstrap_sharpe(returns, n_simulations=500, random_seed=42)

        assert result.p_value < 0.05
        assert result.is_significant

    def test_random_returns_not_significant(self):
        """Zero-mean noise should generally not be significant."""
        returns = _random_returns(252, seed=42)
        result = bootstrap_sharpe(returns, n_simulations=500, random_seed=42)

        # p-value should be large (many bootstrap Sharpes near 0 or negative)
        assert result.p_value >= 0.05

    def test_seed_reproducibility(self):
        """Same seed produces same results."""
        returns = _trending_returns(100)
        r1 = bootstrap_sharpe(returns, n_simulations=100, random_seed=123)
        r2 = bootstrap_sharpe(returns, n_simulations=100, random_seed=123)

        assert r1.simulated_statistics == r2.simulated_statistics
        assert r1.p_value == r2.p_value

    def test_block_size_larger_than_series(self):
        """Graceful handling when block_size > n."""
        returns = _trending_returns(10)
        result = bootstrap_sharpe(returns, n_simulations=50, block_size=100, random_seed=42)

        assert isinstance(result, SignificanceResult)
        assert len(result.simulated_statistics) == 50

    def test_block_size_one_degenerates_to_iid(self):
        """block_size=1 is equivalent to i.i.d. bootstrap."""
        returns = _trending_returns(100)
        result = bootstrap_sharpe(returns, n_simulations=100, block_size=1, random_seed=42)

        assert isinstance(result, SignificanceResult)
        assert len(result.simulated_statistics) == 100


# ---------------------------------------------------------------------------
# TestRunSignificanceTests
# ---------------------------------------------------------------------------


class TestRunSignificanceTests:
    def test_returns_all_three_keys(self):
        """Dict has 'permutation', 't_test', 'bootstrap'."""
        returns = _trending_returns(100)
        results = run_significance_tests(returns, n_simulations=50, random_seed=42)

        assert set(results.keys()) == {"permutation", "t_test", "bootstrap"}

    def test_accepts_list_of_floats(self):
        """Pass raw list, no error."""
        returns = [0.01, 0.02, -0.005, 0.003, 0.01] * 20
        results = run_significance_tests(returns, n_simulations=50, random_seed=42)

        assert "permutation" in results

    def test_all_results_are_significance_result(self):
        """All values are SignificanceResult."""
        returns = _trending_returns(100)
        results = run_significance_tests(returns, n_simulations=50, random_seed=42)

        for v in results.values():
            assert isinstance(v, SignificanceResult)


# ---------------------------------------------------------------------------
# TestSignificanceResult
# ---------------------------------------------------------------------------


class TestSignificanceResult:
    def test_z_score_computation(self):
        """Verify z_score = (observed - null_mean) / null_std."""
        result = SignificanceResult(
            test_name="test",
            observed_statistic=2.0,
            p_value=0.05,
            n_simulations=100,
            simulated_statistics=[],
            is_significant=True,
            significance_level=0.05,
            null_mean=1.0,
            null_std=0.5,
        )

        assert result.z_score == pytest.approx(2.0)  # (2.0 - 1.0) / 0.5

    def test_z_score_zero_std(self):
        """Returns 0.0 when null_std is 0."""
        result = SignificanceResult(
            test_name="test",
            observed_statistic=2.0,
            p_value=0.05,
            n_simulations=100,
            simulated_statistics=[],
            is_significant=True,
            significance_level=0.05,
            null_mean=1.0,
            null_std=0.0,
        )

        assert result.z_score == 0.0

    def test_is_significant_matches_p_value(self):
        """Flag agrees with threshold comparison."""
        r1 = SignificanceResult(
            test_name="test", observed_statistic=0, p_value=0.03,
            n_simulations=0, simulated_statistics=[], is_significant=True,
            significance_level=0.05, null_mean=0, null_std=0,
        )
        r2 = SignificanceResult(
            test_name="test", observed_statistic=0, p_value=0.10,
            n_simulations=0, simulated_statistics=[], is_significant=False,
            significance_level=0.05, null_mean=0, null_std=0,
        )

        assert r1.is_significant is True
        assert r1.p_value < r1.significance_level
        assert r2.is_significant is False
        assert r2.p_value >= r2.significance_level

    def test_frozen(self):
        """Cannot mutate fields."""
        result = SignificanceResult(
            test_name="test", observed_statistic=0, p_value=0.5,
            n_simulations=0, simulated_statistics=[], is_significant=False,
            significance_level=0.05, null_mean=0, null_std=0,
        )

        with pytest.raises(AttributeError):
            result.p_value = 0.99


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_identical_returns(self):
        """Zero variance, no crash."""
        returns = [0.01] * 100
        results = run_significance_tests(returns, n_simulations=50, random_seed=42)

        for v in results.values():
            assert isinstance(v, SignificanceResult)

    def test_empty_returns(self):
        """Empty list, graceful result."""
        results = run_significance_tests([], n_simulations=50, random_seed=42)

        for v in results.values():
            assert isinstance(v, SignificanceResult)
            assert v.p_value == 1.0

    def test_very_short_series(self):
        """3 returns, runs without error."""
        returns = [0.01, -0.005, 0.003]
        results = run_significance_tests(returns, n_simulations=50, random_seed=42)

        for v in results.values():
            assert isinstance(v, SignificanceResult)


class TestSharpeUsesConfiguredTradingDays:
    """Regression: _compute_sharpe must use TRADING_DAYS_PER_YEAR, not hardcoded 252."""

    def test_sharpe_matches_metrics_module(self):
        """
        _compute_sharpe in stats should produce the same result as
        calculate_sharpe_ratio in metrics for the same inputs.
        """
        from sandtable.metrics.performance import calculate_sharpe_ratio

        returns = _trending_returns(100, daily_r=0.001, seed=99)
        stats_sharpe = _compute_sharpe(returns, 0.0)
        metrics_sharpe = calculate_sharpe_ratio(returns, 0.0)

        assert stats_sharpe == pytest.approx(metrics_sharpe, rel=1e-6)

    def test_sharpe_annualization_uses_trading_days(self):
        """
        Hand-compute Sharpe and verify it uses TRADING_DAYS_PER_YEAR.
        """
        returns = [0.001, 0.002, 0.003, -0.001, 0.001]
        n = len(returns)
        mean_r = sum(returns) / n
        var = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
        std = math.sqrt(var)
        expected = (mean_r / std) * math.sqrt(TRADING_DAYS_PER_YEAR)

        assert _compute_sharpe(returns, 0.0) == pytest.approx(expected, rel=1e-6)
