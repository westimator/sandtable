"""tests/data/test_no_lookahead.py

Tests for data handler lookahead prevention.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from backtester.data.data_handler import CSVDataHandler


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """
    Create a sample CSV file for testing.
    """
    csv_content = """Date,Open,High,Low,Close,Volume
2024-01-01,100.0,102.0,99.0,101.0,1000000
2024-01-02,101.0,103.0,100.0,102.0,1100000
2024-01-03,102.0,104.0,101.0,103.0,1200000
2024-01-04,103.0,105.0,102.0,104.0,1300000
2024-01-05,104.0,106.0,103.0,105.0,1400000
2024-01-08,105.0,107.0,104.0,106.0,1500000
2024-01-09,106.0,108.0,105.0,107.0,1600000
2024-01-10,107.0,109.0,106.0,108.0,1700000
"""
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text(csv_content)
    return csv_file


class TestDataHandlerBasics:
    """
    Test basic data handler functionality.
    """

    def test_load_csv(self, sample_csv: Path):
        """
        Should load CSV file successfully.
        """
        handler = CSVDataHandler(sample_csv, "TEST")
        assert len(handler) == 8
        assert handler.symbol == "TEST"

    def test_file_not_found(self, tmp_path: Path):
        """
        Should raise error for missing file.
        """
        with pytest.raises(FileNotFoundError):
            CSVDataHandler(tmp_path / "nonexistent.csv", "TEST")

    def test_get_next_bar_returns_event(self, sample_csv: Path):
        """
        get_next_bar should return MarketDataEvent.
        """
        handler = CSVDataHandler(sample_csv, "TEST")
        bar = handler.get_next_bar()

        assert bar is not None
        assert bar.symbol == "TEST"
        assert bar.close == 101.0  # First bar close
        assert bar.timestamp == datetime(2024, 1, 1)

    def test_get_next_bar_advances_index(self, sample_csv: Path):
        """
        get_next_bar should advance current_index.
        """
        handler = CSVDataHandler(sample_csv, "TEST")

        assert handler.current_index == 0
        handler.get_next_bar()
        assert handler.current_index == 1
        handler.get_next_bar()
        assert handler.current_index == 2

    def test_get_next_bar_returns_none_at_end(self, sample_csv: Path):
        """
        get_next_bar should return None when data exhausted.
        """
        handler = CSVDataHandler(sample_csv, "TEST")

        # Consume all bars
        for _ in range(8):
            assert handler.get_next_bar() is not None

        # Next call should return None
        assert handler.get_next_bar() is None


class TestLookaheadPrevention:
    """
    Test that historical bars don't include future data.
    """

    def test_historical_bars_empty_at_start(self, sample_csv: Path):
        """
        Before any bars read, historical should be empty.
        """
        handler = CSVDataHandler(sample_csv, "TEST")
        historical = handler.get_historical_bars(5)
        assert len(historical) == 0

    def test_historical_bars_after_one_read(self, sample_csv: Path):
        """
        After reading one bar, historical should have one bar.
        """
        handler = CSVDataHandler(sample_csv, "TEST")
        handler.get_next_bar()  # Read first bar

        historical = handler.get_historical_bars(5)
        assert len(historical) == 1
        assert historical[0].close == 101.0  # First bar

    def test_historical_bars_never_include_current(self, sample_csv: Path):
        """
        Historical bars should not include the 'current' unread bar.
        """
        handler = CSVDataHandler(sample_csv, "TEST")

        # Read 3 bars (closes: 101, 102, 103)
        handler.get_next_bar()
        handler.get_next_bar()
        handler.get_next_bar()

        # Historical should have bars 1-3 (closes 101, 102, 103)
        # NOT bar 4 (close 104) which hasn't been read yet
        historical = handler.get_historical_bars(10)
        assert len(historical) == 3

        closes = [bar.close for bar in historical]
        assert closes == [101.0, 102.0, 103.0]
        assert 104.0 not in closes  # Next bar NOT included

    def test_historical_bars_respects_n_limit(self, sample_csv: Path):
        """
        get_historical_bars(n) should return at most n bars.
        """
        handler = CSVDataHandler(sample_csv, "TEST")

        # Read 5 bars
        for _ in range(5):
            handler.get_next_bar()

        # Request only 3
        historical = handler.get_historical_bars(3)
        assert len(historical) == 3

        # Should be the MOST RECENT 3 bars (closes: 103, 104, 105)
        closes = [bar.close for bar in historical]
        assert closes == [103.0, 104.0, 105.0]

    def test_historical_bars_oldest_first(self, sample_csv: Path):
        """
        Historical bars should be ordered oldest first.
        """
        handler = CSVDataHandler(sample_csv, "TEST")

        for _ in range(4):
            handler.get_next_bar()

        historical = handler.get_historical_bars(4)
        timestamps = [bar.timestamp for bar in historical]

        # Should be in chronological order
        assert timestamps == sorted(timestamps)

    def test_no_lookahead_during_iteration(self, sample_csv: Path):
        """
        During iteration, only past data should be accessible.
        """
        handler = CSVDataHandler(sample_csv, "TEST")

        all_closes = [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0]

        for i in range(8):
            current_bar = handler.get_next_bar()
            historical = handler.get_historical_bars(100)

            # Historical should only have bars up to and including current
            historical_closes = [bar.close for bar in historical]
            expected_closes = all_closes[: i + 1]

            assert historical_closes == expected_closes

            # Future bars should NOT be accessible
            for future_close in all_closes[i + 1 :]:
                assert future_close not in historical_closes


class TestDataHandlerReset:
    """
    Test reset functionality.
    """

    def test_reset_returns_to_start(self, sample_csv: Path):
        """
        Reset should return to beginning of data.
        """
        handler = CSVDataHandler(sample_csv, "TEST")

        # Read some bars
        handler.get_next_bar()
        handler.get_next_bar()
        handler.get_next_bar()
        assert handler.current_index == 3

        # Reset
        handler.reset()
        assert handler.current_index == 0

        # Should read first bar again
        bar = handler.get_next_bar()
        assert bar.close == 101.0

    def test_reset_clears_historical(self, sample_csv: Path):
        """
        After reset, historical bars should be empty.
        """
        handler = CSVDataHandler(sample_csv, "TEST")

        for _ in range(5):
            handler.get_next_bar()

        assert len(handler.get_historical_bars(10)) == 5

        handler.reset()
        assert len(handler.get_historical_bars(10)) == 0


class TestGetLatestBar:
    """
    Test get_latest_bar functionality.
    """

    def test_latest_bar_none_at_start(self, sample_csv: Path):
        """
        Before reading any bars, latest should be None.
        """
        handler = CSVDataHandler(sample_csv, "TEST")
        assert handler.get_latest_bar() is None

    def test_latest_bar_after_read(self, sample_csv: Path):
        """
        Latest bar should be the most recently read bar.
        """
        handler = CSVDataHandler(sample_csv, "TEST")

        handler.get_next_bar()
        latest = handler.get_latest_bar()
        assert latest.close == 101.0

        handler.get_next_bar()
        latest = handler.get_latest_bar()
        assert latest.close == 102.0
