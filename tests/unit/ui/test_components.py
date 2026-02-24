"""
tests/unit/ui/test_components.py

Tests for shared UI component functions.
These test the non-Streamlit logic (strategy registry, data handler builder, etc.)
without requiring a running Streamlit server.
"""

from __future__ import annotations

import pandas as pd

from sandtable.strategy.ma_crossover_strategy import MACrossoverStrategy
from sandtable.strategy.mean_reversion_strategy import MeanReversionStrategy
from sandtable.ui.components import (
    CSV_FIXTURE_SYMBOLS,
    STRATEGY_REGISTRY,
    build_data_handler,
)


class TestStrategyRegistry:
    def test_registry_has_expected_strategies(self):
        assert "MA Crossover" in STRATEGY_REGISTRY
        assert "Mean Reversion" in STRATEGY_REGISTRY

    def test_registry_values_are_strategy_classes(self):
        for name, cls in STRATEGY_REGISTRY.items():
            assert hasattr(cls, "generate_signal"), f"{name} missing generate_signal"

    def test_ma_crossover_maps_correctly(self):
        assert STRATEGY_REGISTRY["MA Crossover"] is MACrossoverStrategy

    def test_mean_reversion_maps_correctly(self):
        assert STRATEGY_REGISTRY["Mean Reversion"] is MeanReversionStrategy


class TestAvailableSymbols:
    def test_has_default_symbols(self):
        assert "SPY" in CSV_FIXTURE_SYMBOLS
        assert "QQQ" in CSV_FIXTURE_SYMBOLS
        assert "AAPL" in CSV_FIXTURE_SYMBOLS
        assert "MSFT" in CSV_FIXTURE_SYMBOLS


class TestBuildDataHandler:
    def test_builds_handler_with_valid_symbols(self):
        handler = build_data_handler(["SPY"], "2020-01-01", "2020-12-31")
        assert "SPY" in handler.universe
        assert len(handler.data) == 1

    def test_builds_handler_with_multiple_symbols(self):
        handler = build_data_handler(["SPY", "QQQ"], "2020-01-01", "2020-12-31")
        assert len(handler.data) == 2

    def test_handler_has_price_data(self):
        handler = build_data_handler(["SPY"], "2020-01-01", "2020-12-31")
        price_data = handler.get_price_data()
        assert "SPY" in price_data
        assert isinstance(price_data["SPY"], pd.DataFrame)
        assert len(price_data["SPY"]) > 0


class TestImports:
    """Verify all page modules can be imported without errors."""

    def test_import_components(self):
        import sandtable.ui.components  # noqa: F401

    def test_import_ui_package(self):
        import sandtable.ui  # noqa: F401
