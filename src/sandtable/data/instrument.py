"""
src/sandtable/data/instrument.py

Instrument abstraction carrying all properties needed by execution,
portfolio, and risk components.

For v3, only equities are implemented end-to-end. Futures and FX
constructors are stubbed for future extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo

from sandtable.data_types.currency import Currency
from sandtable.data_types.instrument_type import InstrumentType


@dataclass(frozen=True)
class TradingHours:
    """
    Trading session hours for an instrument.

    Attributes:
        open: Session open time
        close: Session close time
        tz: Timezone (validated against the system tz database)
    """
    open: time = time(9, 30)  # 9:30am ET
    close: time = time(16, 0)  # 4:00pm ET
    tz: ZoneInfo = ZoneInfo("US/Eastern")


@dataclass(frozen=True)
class Instrument:
    """
    A tradeable instrument with all properties needed by execution,
    portfolio, and risk components.

    Attributes:
        symbol: Ticker or contract identifier
        instrument_type: EQUITY, FUTURE, or FX
        tick_size: Minimum price increment
        lot_size: Minimum order quantity increment
        currency: Settlement currency (Currency enum)
        trading_hours: Session hours and timezone
        margin_requirement: Fraction of notional required as margin (1.0 = fully funded)
        contract_multiplier: Multiplier for notional value (1.0 for equities, 50.0 for ES)
        spread_estimate_pct: Default bid-ask spread estimate as percentage
        max_participation_rate: Maximum fraction of ADV for a single order
    """
    symbol: str
    instrument_type: InstrumentType
    tick_size: float
    lot_size: int = 1
    currency: Currency = Currency.USD
    trading_hours: TradingHours = TradingHours()
    margin_requirement: float = 1.0
    contract_multiplier: float = 1.0
    spread_estimate_pct: float = 0.01
    max_participation_rate: float = 0.02


def Equity(symbol: str, *, spread_pct: float = 0.01) -> Instrument:
    """
    Create an equity instrument with sensible defaults.

    Args:
        symbol: Ticker symbol (e.g. 'SPY')
        spread_pct: Estimated bid-ask spread as percentage
    """
    return Instrument(
        symbol=symbol,
        instrument_type=InstrumentType.EQUITY,
        tick_size=0.01,
        spread_estimate_pct=spread_pct,
    )


def Future(
    symbol: str,
    *,
    multiplier: float,
    tick_size: float,
    margin: float = 0.1,
) -> Instrument:
    """
    Create a futures instrument.

    Args:
        symbol: Contract identifier (e.g. 'ES')
        multiplier: Contract multiplier (e.g. 50.0 for ES)
        tick_size: Minimum price increment (e.g. 0.25 for ES)
        margin: Margin requirement as fraction of notional
    """
    return Instrument(
        symbol=symbol,
        instrument_type=InstrumentType.FUTURE,
        tick_size=tick_size,
        contract_multiplier=multiplier,
        margin_requirement=margin,
    )
