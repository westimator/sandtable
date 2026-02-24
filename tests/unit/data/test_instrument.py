"""
tests/unit/data/test_instrument.py

Tests for Instrument, Equity, Future, InstrumentType, TradingHours.
"""

from datetime import time
from zoneinfo import ZoneInfo

import pytest

from sandtable.data.instrument import Equity, Future, Instrument, InstrumentType, TradingHours


class TestInstrument:
    def test_frozen(self):
        """Instrument is immutable."""
        inst = Equity("SPY")
        with pytest.raises(AttributeError):
            inst.symbol = "QQQ"

    def test_defaults(self):
        inst = Instrument(
            symbol="TEST",
            instrument_type=InstrumentType.EQUITY,
            tick_size=0.01,
        )
        assert inst.lot_size == 1
        assert inst.currency == "USD"
        assert inst.margin_requirement == 1.0
        assert inst.contract_multiplier == 1.0
        assert inst.spread_estimate_pct == 0.01
        assert inst.max_participation_rate == 0.02


class TestEquity:
    def test_defaults(self):
        inst = Equity("SPY")
        assert inst.symbol == "SPY"
        assert inst.instrument_type == InstrumentType.EQUITY
        assert inst.tick_size == 0.01
        assert inst.lot_size == 1
        assert inst.currency == "USD"
        assert inst.contract_multiplier == 1.0
        assert inst.margin_requirement == 1.0

    def test_custom_spread(self):
        inst = Equity("AAPL", spread_pct=0.05)
        assert inst.spread_estimate_pct == 0.05


class TestFuture:
    def test_es_defaults(self):
        inst = Future("ES", multiplier=50.0, tick_size=0.25)
        assert inst.symbol == "ES"
        assert inst.instrument_type == InstrumentType.FUTURE
        assert inst.tick_size == 0.25
        assert inst.contract_multiplier == 50.0
        assert inst.margin_requirement == 0.1

    def test_custom_margin(self):
        inst = Future("NQ", multiplier=20.0, tick_size=0.25, margin=0.05)
        assert inst.margin_requirement == 0.05


class TestInstrumentType:
    def test_values(self):
        assert InstrumentType.EQUITY == "EQUITY"
        assert InstrumentType.FUTURE == "FUTURE"
        assert InstrumentType.FX == "FX"


class TestTradingHours:
    def test_defaults(self):
        hours = TradingHours()
        assert hours.open == time(9, 30)
        assert hours.close == time(16, 0)
        assert hours.tz == ZoneInfo("US/Eastern")

    def test_frozen(self):
        hours = TradingHours()
        with pytest.raises(AttributeError):
            hours.open = "10:00"
