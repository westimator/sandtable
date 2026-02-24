"""
tests/unit/reporting/test_tca.py

Tests for transaction cost analysis aggregation.
"""

from datetime import datetime

from sandtable.core.events import Direction, FillEvent
from sandtable.reporting.tca import TCAReport, compute_tca


def _make_fill(symbol: str = "SPY", commission: float = 1.0,
               slippage: float = 0.5, impact: float = 0.3) -> FillEvent:
    return FillEvent(
        timestamp=datetime(2023, 1, 1),
        symbol=symbol,
        direction=Direction.LONG,
        quantity=100,
        fill_price=150.0,
        commission=commission,
        slippage=slippage,
        market_impact=impact,
    )


class TestComputeTCA:
    def test_basic_aggregation(self):
        """Correctly sums slippage, impact, commission costs."""
        fills = [_make_fill(), _make_fill()]
        tca = compute_tca(fills, gross_pnl=1000.0)

        assert tca.total_slippage == 1.0
        assert tca.total_impact == 0.6
        assert tca.total_commission == 2.0
        assert abs(tca.total_cost - 3.6) < 1e-10

    def test_net_pnl(self):
        """Net PnL = gross - total cost."""
        fills = [_make_fill(commission=5.0, slippage=2.0, impact=1.0)]
        tca = compute_tca(fills, gross_pnl=100.0)

        assert abs(tca.net_pnl - 92.0) < 1e-10

    def test_cost_per_trade(self):
        """Average cost per fill."""
        fills = [_make_fill(), _make_fill(), _make_fill()]
        tca = compute_tca(fills, gross_pnl=1000.0)

        assert abs(tca.cost_per_trade - tca.total_cost / 3) < 1e-10

    def test_cost_as_pct_of_gross(self):
        """Cost percentage of absolute gross PnL."""
        fills = [_make_fill(commission=10.0, slippage=0.0, impact=0.0)]
        tca = compute_tca(fills, gross_pnl=100.0)

        assert abs(tca.cost_as_pct_of_gross - 10.0) < 1e-10

    def test_per_symbol_breakdown(self):
        """Per-symbol costs sum to totals."""
        fills = [
            _make_fill("SPY", commission=1.0, slippage=0.5, impact=0.3),
            _make_fill("QQQ", commission=2.0, slippage=1.0, impact=0.5),
        ]
        tca = compute_tca(fills, gross_pnl=1000.0)

        assert "SPY" in tca.cost_by_symbol
        assert "QQQ" in tca.cost_by_symbol
        assert abs(tca.cost_by_symbol["SPY"]["commission"] - 1.0) < 1e-10
        assert abs(tca.cost_by_symbol["QQQ"]["commission"] - 2.0) < 1e-10

        # per-symbol totals should sum to overall
        sym_total = sum(s["total"] for s in tca.cost_by_symbol.values())
        assert abs(sym_total - tca.total_cost) < 1e-10

    def test_cost_by_component(self):
        """Component dict has expected keys."""
        fills = [_make_fill()]
        tca = compute_tca(fills, gross_pnl=1000.0)

        assert "slippage" in tca.cost_by_component
        assert "impact" in tca.cost_by_component
        assert "commission" in tca.cost_by_component

    def test_empty_fills(self):
        """Empty fills list returns zero costs."""
        tca = compute_tca([], gross_pnl=1000.0)

        assert tca.total_cost == 0.0
        assert tca.cost_per_trade == 0.0
        assert tca.net_pnl == 1000.0
        assert tca.cost_by_symbol == {}

    def test_zero_gross_pnl(self):
        """Zero gross PnL doesn't cause division error."""
        fills = [_make_fill()]
        tca = compute_tca(fills, gross_pnl=0.0)

        assert tca.cost_as_pct_of_gross == 0.0

    def test_returns_tca_report(self):
        """Returns a TCAReport instance."""
        tca = compute_tca([], gross_pnl=0.0)
        assert isinstance(tca, TCAReport)
