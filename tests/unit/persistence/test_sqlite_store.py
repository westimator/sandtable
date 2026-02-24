"""
tests/unit/persistence/test_sqlite_store.py

Tests for SQLiteResultStore CRUD operations.
"""

import pytest

from sandtable.api import run_backtest
from sandtable.config import BacktestConfig
from sandtable.core.events import Direction, MarketDataEvent, SignalEvent
from sandtable.persistence.sqlite_store import SQLiteResultStore
from sandtable.strategy.abstract_strategy import AbstractStrategy
from tests.conftest import make_data_handler


class _DummyStrategy(AbstractStrategy):
    """Minimal strategy that always goes long."""

    def __init__(self, *, lookback: int = 5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lookback = lookback

    def generate_signal(self, bar: MarketDataEvent) -> SignalEvent | None:
        if self.symbol_bar_count(bar.symbol) < self.lookback:
            return None
        return SignalEvent(
            timestamp=bar.timestamp,
            symbol=bar.symbol,
            direction=Direction.LONG,
            strength=1.0,
        )


@pytest.fixture
def store(tmp_path):
    """Create a SQLiteResultStore in a temp directory."""
    db_path = str(tmp_path / "test.db")
    return SQLiteResultStore(db_path)


@pytest.fixture
def backtest_result():
    """Run a quick backtest and return (config, result)."""
    data = make_data_handler(["SPY"])
    strategy = _DummyStrategy(lookback=5)
    result = run_backtest(strategy=strategy, data=data)
    config = BacktestConfig(
        strategy_cls=_DummyStrategy,
        strategy_params={"lookback": 5},
        universe=["SPY"],
    )
    return config, result


class TestSaveAndLoad:
    def test_round_trip(self, store, backtest_result):
        """save_run followed by load_run round-trips correctly."""
        config, result = backtest_result
        run_id = store.save_run(config, result)

        loaded_config, loaded_result = store.load_run(run_id)

        # config fields match
        assert loaded_config.strategy_cls is _DummyStrategy
        assert loaded_config.strategy_params == {"lookback": 5}
        assert loaded_config.universe == ["SPY"]

        # result metrics close
        assert abs(loaded_result.metrics.sharpe_ratio - result.metrics.sharpe_ratio) < 1e-6
        assert abs(loaded_result.metrics.total_return - result.metrics.total_return) < 1e-6
        assert loaded_result.metrics.num_trades == result.metrics.num_trades

    def test_equity_curve_round_trips(self, store, backtest_result):
        """Equity curve timestamps and values preserved."""
        config, result = backtest_result
        run_id = store.save_run(config, result)
        _, loaded = store.load_run(run_id)

        assert len(loaded.equity_curve) == len(result.equity_curve)
        for orig, loaded_ep in zip(result.equity_curve, loaded.equity_curve):
            assert orig.timestamp == loaded_ep.timestamp
            assert abs(orig.equity - loaded_ep.equity) < 1e-6
            assert abs(orig.cash - loaded_ep.cash) < 1e-6

    def test_fills_round_trip(self, store, backtest_result):
        """Fill data round-trips with all cost fields intact."""
        config, result = backtest_result
        run_id = store.save_run(config, result)
        _, loaded = store.load_run(run_id)

        assert len(loaded.trades) == len(result.trades)
        for orig, loaded_fill in zip(result.trades, loaded.trades):
            assert orig.symbol == loaded_fill.symbol
            assert orig.direction == loaded_fill.direction
            assert orig.quantity == loaded_fill.quantity
            assert abs(orig.fill_price - loaded_fill.fill_price) < 1e-6
            assert abs(orig.commission - loaded_fill.commission) < 1e-6
            assert abs(orig.slippage - loaded_fill.slippage) < 1e-6
            assert abs(orig.market_impact - loaded_fill.market_impact) < 1e-6

    def test_tags_round_trip(self, store, backtest_result):
        """Tags are stored and retrieved correctly."""
        config, result = backtest_result
        tags = {"version": "v3", "experiment": "phase6"}
        store.save_run(config, result, tags=tags)

        summaries = store.list_runs()
        assert len(summaries) == 1
        assert summaries[0].tags == tags

    def test_load_nonexistent_raises(self, store):
        """Loading a non-existent run_id raises KeyError."""
        with pytest.raises(KeyError):
            store.load_run("nonexistent-id")


class TestListRuns:
    def test_list_all(self, store, backtest_result):
        """list_runs with no filters returns all runs."""
        config, result = backtest_result
        store.save_run(config, result)
        store.save_run(config, result)

        summaries = store.list_runs()
        assert len(summaries) == 2

    def test_filter_by_strategy(self, store, backtest_result):
        """list_runs with strategy filter returns only matching."""
        config, result = backtest_result
        store.save_run(config, result)

        # should find it
        found = store.list_runs(strategy="_DummyStrategy")
        assert len(found) == 1

        # should not find it
        found = store.list_runs(strategy="NonExistent")
        assert len(found) == 0

    def test_filter_by_min_sharpe(self, store, backtest_result):
        config, result = backtest_result
        store.save_run(config, result)
        sharpe = result.metrics.sharpe_ratio

        # threshold below actual sharpe - should find it
        found = store.list_runs(min_sharpe=sharpe - 10.0)
        assert len(found) == 1

        # threshold above actual sharpe - should not find it
        found = store.list_runs(min_sharpe=sharpe + 10.0)
        assert len(found) == 0

    def test_filter_by_date_range(self, store, backtest_result):
        config, result = backtest_result
        store.save_run(config, result)

        # after far in the past - should find it
        found = store.list_runs(after="2000-01-01")
        assert len(found) == 1

        # after far in the future - should not find it
        found = store.list_runs(after="2099-01-01")
        assert len(found) == 0

    def test_limit(self, store, backtest_result):
        """limit parameter caps results."""
        config, result = backtest_result
        for _ in range(5):
            store.save_run(config, result)

        found = store.list_runs(limit=3)
        assert len(found) == 3


class TestDeleteRun:
    def test_delete_existing(self, store, backtest_result):
        """delete_run removes run and all related data."""
        config, result = backtest_result
        run_id = store.save_run(config, result, tags={"k": "v"})

        assert store.delete_run(run_id) is True

        # verify it's gone
        with pytest.raises(KeyError):
            store.load_run(run_id)
        assert store.list_runs() == []

    def test_delete_nonexistent(self, store):
        """delete_run returns False for non-existent run_id."""
        assert store.delete_run("nonexistent") is False

    def test_cascade_deletes_child_tables(self, store, backtest_result):
        """Deleting a run cascades to equity curves, fills, and tags."""
        import sqlite3

        config, result = backtest_result
        run_id = store.save_run(config, result, tags={"k": "v"})

        store.delete_run(run_id)

        # verify child tables are empty
        conn = sqlite3.connect(store._db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        for table in ["equity_curves", "fills", "tags"]:
            count = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            assert count == 0, f"{table} should be empty after cascade delete"
        conn.close()


class TestSchemaIdempotent:
    def test_double_init(self, tmp_path):
        """Calling _ensure_schema twice doesn't error."""
        db_path = str(tmp_path / "test.db")
        SQLiteResultStore(db_path)
        store2 = SQLiteResultStore(db_path)  # same path, re-creates schema
        # just verify no error and both work
        assert store2.list_runs() == []


class TestWALMode:
    def test_wal_enabled(self, store):
        """WAL journal mode should be active."""
        import sqlite3

        conn = sqlite3.connect(store._db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"
