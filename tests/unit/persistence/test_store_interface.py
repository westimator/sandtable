"""
tests/unit/persistence/test_store_interface.py

Tests for the ResultStore ABC and RunSummary dataclass.
"""

import pytest

from sandtable.persistence.abstract_store import AbstractResultStore, RunSummary


class TestAbstractResultStore:
    def test_cannot_instantiate(self):
        """AbstractResultStore is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractResultStore()  # type: ignore[abstract]

    def test_has_required_methods(self):
        """ABC declares the four required methods."""
        assert hasattr(AbstractResultStore, "save_run")
        assert hasattr(AbstractResultStore, "load_run")
        assert hasattr(AbstractResultStore, "list_runs")
        assert hasattr(AbstractResultStore, "delete_run")


class TestRunSummary:
    def test_fields(self):
        """RunSummary has all expected fields."""
        s = RunSummary(
            run_id="abc-123",
            strategy_name="MA",
            universe=["SPY"],
            start_date="2020-01-01",
            end_date="2023-12-31",
            sharpe_ratio=1.5,
            cagr=0.12,
            max_drawdown=-0.15,
            total_trades=42,
            created_at="2024-01-01T00:00:00",
            tags={"version": "v1"},
        )
        assert s.run_id == "abc-123"
        assert s.strategy_name == "MA"
        assert s.universe == ["SPY"]
        assert s.sharpe_ratio == 1.5
        assert s.tags == {"version": "v1"}

    def test_tags_default_empty(self):
        """Tags default to empty dict."""
        s = RunSummary(
            run_id="x",
            strategy_name="X",
            universe=[],
            start_date="",
            end_date="",
            sharpe_ratio=0.0,
            cagr=0.0,
            max_drawdown=0.0,
            total_trades=0,
            created_at="",
        )
        assert s.tags == {}
