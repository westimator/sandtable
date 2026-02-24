"""
src/sandtable/data/universe.py

Universe, a collection of instruments defining the tradeable universe.

Provides instrument lookup, order validation, and a backward-compatible
factory (from_symbols) that auto-constructs default equities from a symbol list.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sandtable.core.events import OrderEvent
from sandtable.data.instrument import Equity, Instrument


@dataclass
class Universe:
    """
    A collection of instruments defining the tradeable universe.

    Attributes:
        instruments: Mapping from symbol string to Instrument.
    """

    instruments: dict[str, Instrument] = field(default_factory=dict)

    @property
    def symbols(self) -> list[str]:
        """Return list of all symbol strings in the universe."""
        return list(self.instruments.keys())

    def __contains__(self, symbol: str) -> bool:
        """Check whether a symbol is in this universe."""
        return symbol in self.instruments

    def __len__(self) -> int:
        """Return the number of instruments in the universe."""
        return len(self.instruments)

    def __repr__(self) -> str:
        return f"Universe(symbols={self.symbols})"

    def validate_order(self, order: OrderEvent) -> bool:
        """Reject orders for symbols not in the universe."""
        return order.symbol in self.instruments

    def get_instrument(self, symbol: str) -> Instrument:
        """
        Look up an instrument by symbol.

        Raises:
            KeyError: If symbol is not in the universe.
        """
        return self.instruments[symbol]

    @classmethod
    def from_symbols(cls, symbols: list[str]) -> Universe:
        """
        Create a universe of default Equity instruments from a symbol list.

        This is the backward-compatibility bridge: when users pass a list of
        strings, the engine auto-constructs a Universe of default equities.
        """
        return cls(
            instruments={s: Equity(s) for s in symbols},
        )
