"""
tests/unit/research/test_config.py

Tests for BacktestConfig serialization and round-tripping.
"""

from sandtable.config import BacktestConfig
from sandtable.core.events import MarketDataEvent, SignalEvent
from sandtable.strategy.abstract_strategy import AbstractStrategy


class _DummyStrategy(AbstractStrategy):
    """Minimal strategy for config tests."""

    def __init__(self, *, lookback: int = 20, threshold: float = 1.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lookback = lookback
        self.threshold = threshold

    def generate_signal(self, bar: MarketDataEvent) -> SignalEvent | None:
        return None


class TestBacktestConfig:
    def test_to_dict_json_serializable(self):
        config = BacktestConfig(
            strategy_cls=_DummyStrategy,
            strategy_params={"lookback": 30, "threshold": 2.0},
            universe=["SPY", "AAPL"],
        )
        d = config.to_dict()
        assert isinstance(d, dict)
        # all values should be JSON-serializable primitives
        assert isinstance(d["strategy_cls"], str)
        assert isinstance(d["strategy_params"], dict)
        assert isinstance(d["universe"], list)

    def test_strategy_cls_serialized_as_string(self):
        config = BacktestConfig(strategy_cls=_DummyStrategy)
        d = config.to_dict()
        assert "." in d["strategy_cls"]
        assert d["strategy_cls"].endswith("_DummyStrategy")

    def test_round_trip(self):
        config = BacktestConfig(
            strategy_cls=_DummyStrategy,
            strategy_params={"lookback": 30, "threshold": 2.0},
            universe=["SPY", "QQQ"],
            start_date="2020-01-01",
            end_date="2022-12-31",
            initial_capital=50_000.0,
            position_size_pct=0.05,
            data_source="yfinance",
        )
        d = config.to_dict()
        restored = BacktestConfig.from_dict(d)

        assert restored.strategy_cls is _DummyStrategy
        assert restored.strategy_params == {"lookback": 30, "threshold": 2.0}
        assert restored.universe == ["SPY", "QQQ"]
        assert restored.start_date == "2020-01-01"
        assert restored.end_date == "2022-12-31"
        assert restored.initial_capital == 50_000.0
        assert restored.position_size_pct == 0.05
        assert restored.data_source == "yfinance"

    def test_defaults(self):
        config = BacktestConfig(strategy_cls=_DummyStrategy)
        assert config.universe == ["SPY"]
        assert config.initial_capital == 100_000.0
        assert config.position_size_pct == 0.10

    def test_frozen(self):
        config = BacktestConfig(strategy_cls=_DummyStrategy)
        import pytest
        with pytest.raises(AttributeError):
            config.initial_capital = 50_000.0  # type: ignore[misc]

    def test_from_dict_with_real_strategy(self):
        """Round-trip with a strategy from the actual sandtable package."""
        from sandtable.strategy.ma_crossover_strategy import MACrossoverStrategy

        config = BacktestConfig(
            strategy_cls=MACrossoverStrategy,
            strategy_params={"fast_period": 5, "slow_period": 20},
        )
        d = config.to_dict()
        restored = BacktestConfig.from_dict(d)
        assert restored.strategy_cls is MACrossoverStrategy
