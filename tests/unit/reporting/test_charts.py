"""
tests/unit/reporting/test_charts.py

Tests for shared matplotlib chart generators.
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for tests

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from sandtable.reporting.charts import (
    plot_correlation_heatmap,
    plot_cost_decomposition,
    plot_cumulative_pnl,
    plot_equity_curve,
    plot_leverage_timeseries,
    plot_monthly_returns_heatmap,
    plot_null_distribution,
    plot_pnl_distribution,
    plot_rolling_sharpe,
    plot_var_timeseries,
)


@pytest.fixture
def equity_series():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    values = 100000 + np.cumsum(np.random.randn(100) * 100)
    return pd.Series(values, index=dates)


@pytest.fixture
def daily_returns():
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    return pd.Series(np.random.randn(300) * 0.01, index=dates)


class TestPlotEquityCurve:
    def test_returns_figure(self, equity_series):
        fig = plot_equity_curve(equity_series)
        assert isinstance(fig, Figure)

    def test_with_drawdown(self, equity_series):
        running_max = equity_series.cummax()
        dd = (equity_series - running_max) / running_max
        fig = plot_equity_curve(equity_series, dd)
        assert isinstance(fig, Figure)


class TestPlotMonthlyReturns:
    def test_returns_figure(self):
        data = np.random.randn(3, 12) * 0.02
        df = pd.DataFrame(data, index=[2020, 2021, 2022], columns=range(1, 13))
        fig = plot_monthly_returns_heatmap(df)
        assert isinstance(fig, Figure)


class TestPlotRollingSharpe:
    def test_returns_figure(self, daily_returns):
        fig = plot_rolling_sharpe(daily_returns)
        assert isinstance(fig, Figure)

    def test_short_series(self):
        short = pd.Series([0.01, -0.01, 0.02], index=pd.date_range("2020-01-01", periods=3))
        fig = plot_rolling_sharpe(short)
        assert isinstance(fig, Figure)


class TestPlotPnlDistribution:
    def test_returns_figure(self):
        fig = plot_pnl_distribution([10, -5, 20, -15, 8, -3])
        assert isinstance(fig, Figure)

    def test_empty(self):
        fig = plot_pnl_distribution([])
        assert isinstance(fig, Figure)


class TestPlotCumulativePnl:
    def test_returns_figure(self):
        fig = plot_cumulative_pnl([10, -5, 20, -15, 8])
        assert isinstance(fig, Figure)

    def test_empty(self):
        fig = plot_cumulative_pnl([])
        assert isinstance(fig, Figure)


class TestPlotCostDecomposition:
    def test_returns_figure(self):
        fig = plot_cost_decomposition({"slippage": 100, "impact": 50, "commission": 200})
        assert isinstance(fig, Figure)


class TestPlotVarTimeseries:
    def test_returns_figure(self):
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        var = pd.Series(np.random.rand(50) * -0.02, index=dates)
        fig = plot_var_timeseries(var)
        assert isinstance(fig, Figure)


class TestPlotLeverageTimeseries:
    def test_returns_figure(self):
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        lev = pd.Series(np.random.rand(50) * 0.5 + 0.8, index=dates)
        fig = plot_leverage_timeseries(lev)
        assert isinstance(fig, Figure)


class TestPlotNullDistribution:
    def test_returns_figure(self):
        fig = plot_null_distribution([0.1, 0.2, -0.1, 0.3, 0.05], 0.5, "Test")
        assert isinstance(fig, Figure)

    def test_empty_simulated(self):
        fig = plot_null_distribution([], 0.5, "Test")
        assert isinstance(fig, Figure)


class TestPlotCorrelationHeatmap:
    def test_returns_figure(self):
        corr = pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        fig = plot_correlation_heatmap(corr)
        assert isinstance(fig, Figure)
