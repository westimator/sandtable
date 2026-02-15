"""
tests/data_handlers/test_single_symbol_handler.py

Tests for SingleSymbolDataHandler iteration logic.

Uses a minimal concrete subclass to test the shared behaviour
independently of any file-loading or network concerns.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from sandtable.core.events import MarketDataEvent
from sandtable.data_handlers.single_symbol_handler import SingleSymbolDataHandler


# ── Minimal concrete subclass ────────────────────────────────────────


class StubHandler(SingleSymbolDataHandler):
    """In-memory handler for testing SingleSymbolDataHandler logic."""

    def __init__(self, symbol: str, bars: int = 5, start: datetime | None = None) -> None:
        self.symbol = symbol
        self._current_index = 0
        self._data = self._generate(bars, start or datetime(2024, 1, 1))

    @staticmethod
    def _generate(n: int, start: datetime) -> pd.DataFrame:
        rows = []
        for i in range(n):
            rows.append(
                {
                    "timestamp": start + timedelta(days=i),
                    "open": 100.0 + i,
                    "high": 105.0 + i,
                    "low": 95.0 + i,
                    "close": 101.0 + i,
                    "volume": 1_000_000.0 + i * 100_000,
                }
            )
        return pd.DataFrame(rows)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def handler() -> StubHandler:
    return StubHandler("TEST", bars=5)


# ── Tests ────────────────────────────────────────────────────────────


class TestLen:
    def test_len(self, handler: StubHandler) -> None:
        assert len(handler) == 5

    def test_len_zero(self) -> None:
        h = StubHandler("EMPTY", bars=0)
        assert len(h) == 0


class TestData:
    def test_returns_copy(self, handler: StubHandler) -> None:
        df = handler.data
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        # Mutation of the copy should not affect the handler
        df.drop(df.index, inplace=True)
        assert len(handler.data) == 5

    def test_columns(self, handler: StubHandler) -> None:
        cols = set(handler.data.columns)
        assert {"timestamp", "open", "high", "low", "close", "volume"} <= cols


class TestGetNextBar:
    def test_returns_market_data_event(self, handler: StubHandler) -> None:
        bar = handler.get_next_bar()
        assert isinstance(bar, MarketDataEvent)
        assert bar.symbol == "TEST"
        assert bar.close == 101.0

    def test_advances_index(self, handler: StubHandler) -> None:
        assert handler.current_index == 0
        handler.get_next_bar()
        assert handler.current_index == 1
        handler.get_next_bar()
        assert handler.current_index == 2

    def test_returns_none_when_exhausted(self, handler: StubHandler) -> None:
        for _ in range(5):
            assert handler.get_next_bar() is not None
        assert handler.get_next_bar() is None

    def test_consecutive_none(self, handler: StubHandler) -> None:
        for _ in range(5):
            handler.get_next_bar()
        assert handler.get_next_bar() is None
        assert handler.get_next_bar() is None

    def test_empty_handler(self) -> None:
        h = StubHandler("EMPTY", bars=0)
        assert h.get_next_bar() is None


class TestGetHistoricalBars:
    def test_empty_before_any_read(self, handler: StubHandler) -> None:
        assert handler.get_historical_bars(10) == []

    def test_returns_read_bars(self, handler: StubHandler) -> None:
        handler.get_next_bar()
        handler.get_next_bar()
        hist = handler.get_historical_bars(10)
        assert len(hist) == 2
        assert hist[0].close == 101.0
        assert hist[1].close == 102.0

    def test_respects_n_limit(self, handler: StubHandler) -> None:
        for _ in range(5):
            handler.get_next_bar()
        hist = handler.get_historical_bars(3)
        assert len(hist) == 3
        # Should be the 3 most recent
        assert [b.close for b in hist] == [103.0, 104.0, 105.0]

    def test_oldest_first(self, handler: StubHandler) -> None:
        for _ in range(4):
            handler.get_next_bar()
        hist = handler.get_historical_bars(4)
        timestamps = [b.timestamp for b in hist]
        assert timestamps == sorted(timestamps)

    def test_no_future_data(self, handler: StubHandler) -> None:
        """Historical bars must never include unread bars."""
        handler.get_next_bar()  # read bar 0
        hist = handler.get_historical_bars(100)
        assert len(hist) == 1
        assert hist[0].close == 101.0  # bar 0
        # bar 1 (close=102.0) hasn't been read yet
        closes = {b.close for b in hist}
        assert 102.0 not in closes


class TestGetLatestBar:
    def test_none_at_start(self, handler: StubHandler) -> None:
        assert handler.get_latest_bar() is None

    def test_tracks_most_recent(self, handler: StubHandler) -> None:
        handler.get_next_bar()
        assert handler.get_latest_bar().close == 101.0
        handler.get_next_bar()
        assert handler.get_latest_bar().close == 102.0


class TestReset:
    def test_resets_index(self, handler: StubHandler) -> None:
        handler.get_next_bar()
        handler.get_next_bar()
        assert handler.current_index == 2
        handler.reset()
        assert handler.current_index == 0

    def test_replays_from_start(self, handler: StubHandler) -> None:
        first = handler.get_next_bar()
        handler.get_next_bar()
        handler.reset()
        replayed = handler.get_next_bar()
        assert replayed.close == first.close
        assert replayed.timestamp == first.timestamp

    def test_historical_empty_after_reset(self, handler: StubHandler) -> None:
        for _ in range(3):
            handler.get_next_bar()
        handler.reset()
        assert handler.get_historical_bars(10) == []

    def test_latest_bar_none_after_reset(self, handler: StubHandler) -> None:
        handler.get_next_bar()
        handler.reset()
        assert handler.get_latest_bar() is None


class TestGetPriceData:
    def test_returns_dict_keyed_by_symbol(self, handler: StubHandler) -> None:
        result = handler.get_price_data()
        assert "TEST" in result
        assert isinstance(result["TEST"], pd.DataFrame)
        assert len(result["TEST"]) == 5


class TestRowToEvent:
    def test_field_mapping(self, handler: StubHandler) -> None:
        bar = handler.get_next_bar()
        assert bar.open == 100.0
        assert bar.high == 105.0
        assert bar.low == 95.0
        assert bar.close == 101.0
        assert bar.volume == 1_000_000.0
        assert bar.timestamp == datetime(2024, 1, 1)
