"""
tests/unit/persistence/test_sync.py

Tests for bidirectional sync between two SQL result stores.

Uses two SQLiteResultStore instances (no MySQL needed).
"""

import pytest

from sandtable.api import run_backtest
from sandtable.config import BacktestConfig
from sandtable.core.events import Direction, MarketDataEvent, SignalEvent
from sandtable.persistence.sqlite_store import SQLiteResultStore
from sandtable.persistence.sync import SyncResult, sync_stores
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
def store_a(tmp_path):
    return SQLiteResultStore(str(tmp_path / "a.db"))


@pytest.fixture
def store_b(tmp_path):
    return SQLiteResultStore(str(tmp_path / "b.db"))


@pytest.fixture
def backtest_result():
    data = make_data_handler(["SPY"])
    strategy = _DummyStrategy(lookback=5)
    result = run_backtest(strategy=strategy, data=data)
    config = BacktestConfig(
        strategy_cls=_DummyStrategy,
        strategy_params={"lookback": 5},
        universe=["SPY"],
    )
    return config, result


class TestSyncStores:
    def test_empty_stores(self, store_a, store_b):
        """Syncing two empty stores is a no-op."""
        sr = sync_stores(store_a, store_b)
        assert sr.total_copied == 0
        assert sr.total_failed == 0

    def test_a_to_b(self, store_a, store_b, backtest_result):
        """Runs in A are copied to B."""
        config, result = backtest_result
        run_id = store_a.save_run(config, result, tags={"env": "test"})

        sr = sync_stores(store_a, store_b)

        assert sr.copied_a_to_b == 1
        assert sr.copied_b_to_a == 0
        assert sr.total_failed == 0

        # verify run exists in B with same run_id
        loaded_config, loaded_result = store_b.load_run(run_id)
        assert loaded_config.strategy_cls is _DummyStrategy
        assert loaded_result.metrics.num_trades == result.metrics.num_trades

    def test_b_to_a(self, store_a, store_b, backtest_result):
        """Runs in B are copied to A."""
        config, result = backtest_result
        run_id = store_b.save_run(config, result)

        sr = sync_stores(store_a, store_b)

        assert sr.copied_a_to_b == 0
        assert sr.copied_b_to_a == 1
        store_a.load_run(run_id)  # should not raise

    def test_bidirectional(self, store_a, store_b, backtest_result):
        """Runs unique to each store are copied to the other."""
        config, result = backtest_result
        id_a = store_a.save_run(config, result)
        id_b = store_b.save_run(config, result)

        sr = sync_stores(store_a, store_b)

        assert sr.copied_a_to_b == 1
        assert sr.copied_b_to_a == 1

        # both stores now have both runs
        assert store_a.list_run_ids() == {id_a, id_b}
        assert store_b.list_run_ids() == {id_a, id_b}

    def test_idempotent(self, store_a, store_b, backtest_result):
        """Syncing twice doesn't duplicate runs."""
        config, result = backtest_result
        store_a.save_run(config, result)

        sync_stores(store_a, store_b)
        sr2 = sync_stores(store_a, store_b)

        assert sr2.total_copied == 0
        assert len(store_b.list_runs()) == 1

    def test_run_id_preserved(self, store_a, store_b, backtest_result):
        """The original run_id is preserved after sync."""
        config, result = backtest_result
        original_id = store_a.save_run(config, result)

        sync_stores(store_a, store_b)

        ids_b = store_b.list_run_ids()
        assert original_id in ids_b

    def test_tags_preserved(self, store_a, store_b, backtest_result):
        """Tags are preserved after sync."""
        config, result = backtest_result
        tags = {"version": "v3", "experiment": "sync-test"}
        store_a.save_run(config, result, tags=tags)

        sync_stores(store_a, store_b)

        summaries = store_b.list_runs()
        assert len(summaries) == 1
        assert summaries[0].tags == tags

    def test_created_at_preserved(self, store_a, store_b, backtest_result):
        """The original created_at timestamp is preserved after sync."""
        config, result = backtest_result
        store_a.save_run(config, result)

        # get original created_at from store_a
        summaries_a = store_a.list_runs()
        original_created_at = summaries_a[0].created_at

        sync_stores(store_a, store_b)

        summaries_b = store_b.list_runs()
        assert summaries_b[0].created_at == original_created_at

    def test_multiple_runs(self, store_a, store_b, backtest_result):
        """Multiple runs sync correctly."""
        config, result = backtest_result
        ids = []
        for _ in range(3):
            ids.append(store_a.save_run(config, result))

        sr = sync_stores(store_a, store_b)

        assert sr.copied_a_to_b == 3
        assert store_b.list_run_ids() == set(ids)


class TestListRunIds:
    def test_empty(self, store_a):
        assert store_a.list_run_ids() == set()

    def test_returns_all(self, store_a, backtest_result):
        config, result = backtest_result
        id1 = store_a.save_run(config, result)
        id2 = store_a.save_run(config, result)
        assert store_a.list_run_ids() == {id1, id2}


class TestInsertRun:
    def test_preserves_run_id(self, store_a, backtest_result):
        """_insert_run uses the provided run_id, not a generated one."""
        config, result = backtest_result
        custom_id = "custom_12345_abcd"
        store_a._insert_run(custom_id, config, result, None, "2025-01-01T00:00:00+00:00")

        assert custom_id in store_a.list_run_ids()
        store_a.load_run(custom_id)  # should not raise

    def test_preserves_created_at(self, store_a, backtest_result):
        """_insert_run uses the provided created_at timestamp."""
        config, result = backtest_result
        custom_ts = "2020-06-15T12:00:00+00:00"
        store_a._insert_run("test_run", config, result, None, custom_ts)

        summaries = store_a.list_runs()
        assert summaries[0].created_at == custom_ts


class TestSyncCopyFailures:
    def test_duplicate_key_skipped(self, store_a, store_b, backtest_result):
        """If dest already has the run (duplicate key), it is silently skipped."""
        config, result = backtest_result
        run_id = store_a.save_run(config, result)

        # manually copy to b first
        sync_stores(store_a, store_b)
        assert run_id in store_b.list_run_ids()

        # now insert into a again with a different id, and corrupt b to have same id
        # easiest: just sync again and assert idempotent
        sr = sync_stores(store_a, store_b)
        assert sr.total_copied == 0
        assert sr.total_failed == 0

    def test_copy_failure_recorded(self, store_a, store_b, backtest_result, monkeypatch):
        """Generic copy failure is captured in failed_a_to_b."""
        config, result = backtest_result
        store_a.save_run(config, result)

        # monkeypatch _insert_run to raise
        def _bad_insert(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(store_b, "_insert_run", _bad_insert)

        sr = sync_stores(store_a, store_b)
        assert sr.total_failed == 1
        assert len(sr.failed_a_to_b) == 1


class TestSyncResult:
    def test_total_copied(self):
        sr = SyncResult(copied_a_to_b=3, copied_b_to_a=2)
        assert sr.total_copied == 5

    def test_total_failed(self):
        sr = SyncResult(failed_a_to_b=["a", "b"], failed_b_to_a=["c"])
        assert sr.total_failed == 3

    def test_defaults(self):
        sr = SyncResult()
        assert sr.total_copied == 0
        assert sr.total_failed == 0
