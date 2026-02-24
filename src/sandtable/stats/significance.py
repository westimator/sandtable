"""
src/sandtable/stats/significance.py

Statistical significance tests for backtest results.

Three complementary tests:
1. Permutation test on Sharpe - shuffle daily returns, recompute Sharpe, derive p-value.
2. One-sample t-test on mean return - test H0: mean daily return = 0.
3. Circular block bootstrap on Sharpe - resample blocks preserving autocorrelation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scipy import stats as scipy_stats

from sandtable.config import settings

if TYPE_CHECKING:
    from sandtable.core.result import BacktestResult

TRADING_DAYS_PER_YEAR = settings.BACKTESTER_TRADING_DAYS


@dataclass(frozen=True)
class SignificanceResult:
    """
    Result of a statistical significance test.

    Attributes:
        test_name: Name of the test
        observed_statistic: Test statistic from actual returns
        p_value: Probability of observing this result by chance
        n_simulations: Number of simulations performed
        simulated_statistics: Distribution of simulated stats
        is_significant: True if p_value < significance_level
        significance_level: Alpha threshold (default 0.05)
        null_mean: Mean of simulated distribution
        null_std: Std dev of simulated distribution
    """
    test_name: str
    observed_statistic: float
    p_value: float
    n_simulations: int
    simulated_statistics: list[float]
    is_significant: bool
    significance_level: float
    null_mean: float
    null_std: float

    @property
    def z_score(self) -> float:
        """
        Z-score of the observed statistic relative to the null distribution.

        Returns 0.0 if null_std is 0.
        """
        if self.null_std == 0:
            return 0.0
        return (self.observed_statistic - self.null_mean) / self.null_std


## Internal helpers

def _compute_sharpe(daily_returns: list[float], risk_free_rate: float = 0.0) -> float:
    """
    Compute annualized Sharpe ratio from daily returns.

    Annualized Sharpe = mean(excess_daily) / std(daily) * sqrt(252).
    Returns 0.0 if std is 0 or returns list has fewer than 2 elements.
    """
    if len(daily_returns) < 2:
        return 0.0

    n = len(daily_returns)
    mean_r = sum(daily_returns) / n
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_mean = mean_r - daily_rf

    variance = sum((r - mean_r) ** 2 for r in daily_returns) / (n - 1)
    std = math.sqrt(variance)

    if std == 0:
        return 0.0

    return (excess_mean / std) * math.sqrt(TRADING_DAYS_PER_YEAR)


def _extract_daily_returns(result: BacktestResult) -> list[float]:
    """
    Extract daily returns from a BacktestResult's equity curve.
    """
    eq_df = result.equity_dataframe()
    if eq_df.empty or len(eq_df) < 2:
        return []
    returns = eq_df["equity"].pct_change().dropna().tolist()
    return returns


## t-distribution CDF via scipy

def _t_cdf(t_stat: float, df: float) -> float:
    """
    CDF of Student's t-distribution with df degrees of freedom.
    """
    if df <= 0:
        return 0.5
    return float(scipy_stats.t.cdf(t_stat, df))


# Public API


def permutation_test_sharpe(
    daily_returns: list[float],
    n_simulations: int = 1000,
    significance_level: float = 0.05,
    risk_free_rate: float = 0.0,
    random_seed: int | None = None,
) -> SignificanceResult:
    """
    Sign-randomization permutation test on Sharpe ratio.

    For each simulation, randomly flips the sign of each daily return
    and recomputes the Sharpe ratio. This tests whether the observed
    positive Sharpe is significantly different from zero under the null
    hypothesis that returns are symmetric around zero.

    Args:
        daily_returns: List of daily return values
        n_simulations: Number of permutations to run
        significance_level: Alpha threshold for significance
        risk_free_rate: Annualized risk-free rate
        random_seed: Optional seed for reproducibility

    Returns:
        SignificanceResult with p-value and null distribution
    """
    if len(daily_returns) < 2:
        return SignificanceResult(
            test_name="Permutation Test (Sharpe)",
            observed_statistic=0.0,
            p_value=1.0,
            n_simulations=n_simulations,
            simulated_statistics=[],
            is_significant=False,
            significance_level=significance_level,
            null_mean=0.0,
            null_std=0.0,
        )

    observed = _compute_sharpe(daily_returns, risk_free_rate)

    rng = random.Random(random_seed)
    simulated: list[float] = []

    for _ in range(n_simulations):
        flipped = [r * rng.choice((-1, 1)) for r in daily_returns]
        sim_sharpe = _compute_sharpe(flipped, risk_free_rate)
        simulated.append(sim_sharpe)

    count = sum(1 for s in simulated if s >= observed)
    p_value = count / n_simulations

    null_mean = sum(simulated) / len(simulated)
    null_std = math.sqrt(sum((s - null_mean) ** 2 for s in simulated) / len(simulated))

    return SignificanceResult(
        test_name="Permutation Test (Sharpe)",
        observed_statistic=observed,
        p_value=p_value,
        n_simulations=n_simulations,
        simulated_statistics=simulated,
        is_significant=p_value < significance_level,
        significance_level=significance_level,
        null_mean=null_mean,
        null_std=null_std,
    )


def ttest_mean_return(
    daily_returns: list[float],
    significance_level: float = 0.05,
) -> SignificanceResult:
    """
    One-sample t-test on mean daily return.

    Tests H0: mean daily return = 0 (one-tailed, testing for positive mean).
    Uses scipy's Student's t-distribution CDF.

    Args:
        daily_returns: List of daily return values
        significance_level: Alpha threshold for significance

    Returns:
        SignificanceResult with t-statistic and p-value
    """
    n = len(daily_returns)

    if n < 2:
        return SignificanceResult(
            test_name="t-test (Mean Return)",
            observed_statistic=0.0,
            p_value=1.0,
            n_simulations=0,
            simulated_statistics=[],
            is_significant=False,
            significance_level=significance_level,
            null_mean=0.0,
            null_std=0.0,
        )

    mean_r = sum(daily_returns) / n
    variance = sum((r - mean_r) ** 2 for r in daily_returns) / (n - 1)
    std = math.sqrt(variance)

    if std == 0:
        t_stat = 0.0
    else:
        t_stat = mean_r / (std / math.sqrt(n))

    # One-tailed p-value (testing H1: mean > 0)
    p_value = 1.0 - _t_cdf(t_stat, n - 1)

    return SignificanceResult(
        test_name="t-test (Mean Return)",
        observed_statistic=t_stat,
        p_value=p_value,
        n_simulations=0,
        simulated_statistics=[],
        is_significant=p_value < significance_level,
        significance_level=significance_level,
        null_mean=0.0,
        null_std=0.0,
    )


def bootstrap_sharpe(
    daily_returns: list[float],
    n_simulations: int = 1000,
    block_size: int = 20,
    significance_level: float = 0.05,
    risk_free_rate: float = 0.0,
    random_seed: int | None = None,
) -> SignificanceResult:
    """
    Circular block bootstrap on Sharpe ratio.

    Resamples blocks of returns (preserving autocorrelation) and builds
    a confidence interval. Block size of 20 (approx. 1 trading month)
    is the standard default.

    Args:
        daily_returns: List of daily return values
        n_simulations: Number of bootstrap iterations
        block_size: Size of contiguous blocks to resample
        significance_level: Alpha threshold for significance
        risk_free_rate: Annualized risk-free rate
        random_seed: Optional seed for reproducibility

    Returns:
        SignificanceResult with bootstrap distribution
    """
    n = len(daily_returns)

    if n < 2:
        return SignificanceResult(
            test_name="Block Bootstrap (Sharpe)",
            observed_statistic=0.0,
            p_value=1.0,
            n_simulations=n_simulations,
            simulated_statistics=[],
            is_significant=False,
            significance_level=significance_level,
            null_mean=0.0,
            null_std=0.0,
        )

    observed = _compute_sharpe(daily_returns, risk_free_rate)

    rng = random.Random(random_seed)
    effective_block_size = min(block_size, n)
    n_blocks = math.ceil(n / effective_block_size)
    simulated: list[float] = []

    for _ in range(n_simulations):
        resampled: list[float] = []
        for _ in range(n_blocks):
            start = rng.randint(0, n - 1)
            for j in range(effective_block_size):
                idx = (start + j) % n  # circular wrap
                resampled.append(daily_returns[idx])
        resampled = resampled[:n]  # truncate to original length

        sim_sharpe = _compute_sharpe(resampled, risk_free_rate)
        simulated.append(sim_sharpe)

    # p-value: fraction of bootstrap Sharpes <= 0 (tests if Sharpe is significantly positive)
    count = sum(1 for s in simulated if s <= 0)
    p_value = count / n_simulations

    null_mean = sum(simulated) / len(simulated)
    null_std = math.sqrt(sum((s - null_mean) ** 2 for s in simulated) / len(simulated))

    return SignificanceResult(
        test_name="Block Bootstrap (Sharpe)",
        observed_statistic=observed,
        p_value=p_value,
        n_simulations=n_simulations,
        simulated_statistics=simulated,
        is_significant=p_value < significance_level,
        significance_level=significance_level,
        null_mean=null_mean,
        null_std=null_std,
    )


## Convenience wrapper

def run_significance_tests(
    result_or_returns: BacktestResult | list[float],
    n_simulations: int = 1000,
    significance_level: float = 0.05,
    risk_free_rate: float = 0.0,
    random_seed: int | None = None,
) -> dict[str, SignificanceResult]:
    """
    Run all three significance tests on backtest results.

    Accepts either a BacktestResult or a raw list of daily returns.

    Args:
        result_or_returns: BacktestResult object or list of daily returns
        n_simulations: Number of simulations for permutation/bootstrap tests
        significance_level: Alpha threshold for all tests
        risk_free_rate: Annualized risk-free rate
        random_seed: Optional seed for reproducibility

    Returns:
        Dict with keys "permutation", "t_test", "bootstrap" mapping to
        SignificanceResult objects.
    """
    if isinstance(result_or_returns, list):
        daily_returns = result_or_returns
    else:
        daily_returns = _extract_daily_returns(result_or_returns)

    return {
        "permutation": permutation_test_sharpe(
            daily_returns,
            n_simulations=n_simulations,
            significance_level=significance_level,
            risk_free_rate=risk_free_rate,
            random_seed=random_seed,
        ),
        "t_test": ttest_mean_return(
            daily_returns,
            significance_level=significance_level,
        ),
        "bootstrap": bootstrap_sharpe(
            daily_returns,
            n_simulations=n_simulations,
            significance_level=significance_level,
            risk_free_rate=risk_free_rate,
            random_seed=random_seed,
        ),
    }
