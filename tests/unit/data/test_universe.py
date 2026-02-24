"""
tests/unit/data/test_universe.py

Tests for Universe class.
"""

from datetime import datetime

import pytest

from sandtable.core.events import Direction, OrderEvent, OrderType
from sandtable.data.instrument import Equity, InstrumentType
from sandtable.data.universe import Universe


def _make_order(symbol: str = "SPY") -> OrderEvent:
    return OrderEvent(
        timestamp=datetime(2023, 1, 1),
        symbol=symbol,
        direction=Direction.LONG,
        quantity=100,
        order_type=OrderType.MARKET,
    )


class TestUniverse:
    def test_symbols(self):
        u = Universe.from_symbols(["SPY", "QQQ"])
        assert u.symbols == ["SPY", "QQQ"]

    def test_contains(self):
        u = Universe.from_symbols(["SPY", "QQQ"])
        assert "SPY" in u
        assert "AAPL" not in u

    def test_len(self):
        u = Universe.from_symbols(["SPY", "QQQ", "AAPL"])
        assert len(u) == 3

    def test_get_instrument(self):
        u = Universe.from_symbols(["SPY"])
        inst = u.get_instrument("SPY")
        assert inst.symbol == "SPY"
        assert inst.instrument_type == InstrumentType.EQUITY

    def test_get_instrument_missing(self):
        u = Universe.from_symbols(["SPY"])
        with pytest.raises(KeyError):
            u.get_instrument("QQQ")

    def test_validate_order_in_universe(self):
        u = Universe.from_symbols(["SPY"])
        assert u.validate_order(_make_order("SPY")) is True

    def test_validate_order_not_in_universe(self):
        u = Universe.from_symbols(["SPY"])
        assert u.validate_order(_make_order("QQQ")) is False

    def test_from_symbols_creates_equities(self):
        u = Universe.from_symbols(["SPY", "AAPL"])
        for symbol in ["SPY", "AAPL"]:
            inst = u.get_instrument(symbol)
            assert inst.instrument_type == InstrumentType.EQUITY
            assert inst.tick_size == 0.01

    def test_from_custom_instruments(self):
        """Can build a universe from custom instrument objects."""
        instruments = {
            "SPY": Equity("SPY", spread_pct=0.02),
            "QQQ": Equity("QQQ", spread_pct=0.03),
        }
        u = Universe(instruments=instruments)
        assert u.get_instrument("SPY").spread_estimate_pct == 0.02
        assert u.get_instrument("QQQ").spread_estimate_pct == 0.03

    def test_empty_universe(self):
        u = Universe.from_symbols([])
        assert len(u) == 0
        assert u.symbols == []

    def test_repr(self):
        u = Universe.from_symbols(["SPY"])
        assert "SPY" in repr(u)
