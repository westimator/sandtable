"""
tests/report/test_tearsheet.py

Tests for tearsheet and comparison report generation.
"""

import pytest

from sandtable.api import run_backtest
from sandtable.core.events import Direction, MarketDataEvent, SignalEvent
from sandtable.strategy.abstract_strategy import AbstractStrategy
from tests.conftest import make_data_handler


class SimpleTestStrategy(AbstractStrategy):

    def __init__(self, *, fast_period: int = 10, slow_period: int = 30, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signal(self, bar: MarketDataEvent) -> SignalEvent | None:
        closes = self.get_historical_closes(self.slow_period, bar.symbol)
        if len(closes) < self.slow_period:
            return None
        fast_ma = sum(closes[-self.fast_period :]) / self.fast_period
        slow_ma = sum(closes) / len(closes)
        if fast_ma > slow_ma:
            return SignalEvent(
                timestamp=bar.timestamp, symbol=bar.symbol,
                direction=Direction.LONG, strength=1.0,
            )
        elif fast_ma < slow_ma:
            return SignalEvent(
                timestamp=bar.timestamp, symbol=bar.symbol,
                direction=Direction.SHORT, strength=1.0,
            )
        return None


@pytest.fixture
def sample_result():
    data = make_data_handler(["SPY"])
    return run_backtest(
        strategy=SimpleTestStrategy(),
        data=data,
    )


class TestTearsheet:
    def test_generate_tearsheet_returns_html(self, sample_result):
        from sandtable.report.tearsheet import generate_tearsheet
        html = generate_tearsheet(sample_result)
        assert isinstance(html, str)
        assert "<html>" in html.lower() or "<!doctype" in html.lower()
        assert "data:image/png;base64," in html

    def test_tearsheet_writes_file(self, sample_result, tmp_path):
        from sandtable.report.tearsheet import generate_tearsheet
        output = tmp_path / "tearsheet.html"
        html = generate_tearsheet(sample_result, output_path=str(output))
        assert output.exists()
        assert output.read_text() == html

    def test_tearsheet_via_result_method(self, sample_result, tmp_path):
        output = tmp_path / "tearsheet.html"
        html = sample_result.tearsheet(output_path=str(output))
        assert output.exists()
        assert "data:image/png;base64," in html

    def test_tearsheet_contains_metrics(self, sample_result):
        from sandtable.report.tearsheet import generate_tearsheet
        html = generate_tearsheet(sample_result)
        assert "Sharpe" in html
        assert "SPY" in html


class TestComparison:
    def test_compare_strategies(self, sample_result):
        from sandtable.report.comparison import compare_strategies

        data = make_data_handler(["SPY"])
        result2 = run_backtest(
            strategy=SimpleTestStrategy(fast_period=5, slow_period=20),
            data=data,
        )
        html = compare_strategies({"MA(10,30)": sample_result, "MA(5,20)": result2})
        assert isinstance(html, str)
        assert "MA(10,30)" in html
        assert "MA(5,20)" in html

    def test_compare_writes_file(self, sample_result, tmp_path):
        from sandtable.report.comparison import compare_strategies

        output = tmp_path / "comparison.html"
        compare_strategies({"Strategy A": sample_result}, output_path=str(output))
        assert output.exists()

    def test_drawdown_not_shown_positive(self, sample_result):
        """Regression: max_drawdown must not render with a + sign."""
        from sandtable.report.comparison import compare_strategies

        html = compare_strategies({"Strategy A": sample_result})
        # drawdown is stored as a positive float (e.g. 0.05 for 5%)
        # it should render as negative (e.g. -5.00%) not +5.00%
        dd = sample_result.metrics.max_drawdown
        bad_str = f"+{dd:.2%}"
        assert bad_str not in html, f"Drawdown should not appear as {bad_str}"
        # it should appear as a negative percentage
        expected_str = f"-{dd:.2%}"
        assert expected_str in html


class TestTearsheetIncludeStats:
    def test_include_stats_adds_significance_section(self, sample_result):
        """include_stats=True adds significance table."""
        from sandtable.report.tearsheet import generate_tearsheet
        html = generate_tearsheet(
            sample_result,
            include_stats=True,
            n_simulations=50,
            random_seed=42,
        )
        assert "Statistical Significance" in html

    def test_tearsheet_pdf_output(self, sample_result, tmp_path):
        """Generating tearsheet to a .pdf path converts via weasyprint."""
        from sandtable.report.tearsheet import generate_tearsheet
        out = str(tmp_path / "tearsheet.pdf")
        try:
            generate_tearsheet(sample_result, output_path=out)
            assert (tmp_path / "tearsheet.pdf").exists()
        except ImportError:
            pytest.skip("weasyprint not installed")
