"""
src/sandtable/portfolio/portfolio.py

Portfolio management for tracking positions, cash, and equity.

Handles position sizing, fill processing, and equity curve recording.
Supports both long and short positions with correct P&L accounting.
"""

from dataclasses import dataclass, field
from datetime import datetime

from sandtable.core.events import (
    Direction,
    FillEvent,
    MarketDataEvent,
    OrderEvent,
    OrderType,
    SignalEvent,
)
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    """
    Represents a position in a single symbol.

    Attributes:
        symbol: Ticker symbol
        quantity: Number of shares (positive for long, negative for short)
        avg_cost: Average cost basis per share
        realized_pnl: Cumulative realized P&L from closed trades
    """

    symbol: str
    quantity: int = 0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0

    @property
    def is_long(self) -> bool:
        """
        True if position is long (positive shares).
        """
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """
        True if position is short (negative shares).
        """
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        """
        True if no position.
        """
        return self.quantity == 0

    def market_value(self, current_price: float) -> float:
        """
        Calculate current market value of the position.

        For long positions: positive value
        For short positions: negative value (liability)
        """
        return self.quantity * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L at current price.
        I.e., what the position is worth if it were closed today at the current price.

        For long: (current_price - avg_cost) * quantity
        For short: (avg_cost - current_price) * abs(quantity)
        """
        if self.is_flat:
            return 0.0
        return (current_price - self.avg_cost) * self.quantity


@dataclass
class EquityPoint:
    """
    A point in the equity curve.

    Attributes:
        timestamp: Time of the equity snapshot
        equity: Total portfolio equity (cash + positions market value)
        cash: Cash balance
        positions_value: Total market value of all positions
    """

    timestamp: datetime
    equity: float
    cash: float
    positions_value: float


@dataclass
class Portfolio:
    """
    Manages portfolio state including positions, cash, and equity tracking.

    Attributes:
        initial_capital: Starting cash amount
        position_size_pct: Fraction of equity to allocate per trade (default 10%)
    """

    initial_capital: float = 100_000.0
    position_size_pct: float = 0.10

    _cash: float = field(init=False)
    _positions: dict[str, Position] = field(init=False, default_factory=dict)
    _current_prices: dict[str, float] = field(init=False, default_factory=dict)
    _equity_curve: list[EquityPoint] = field(init=False, default_factory=list)
    _trades: list[FillEvent] = field(init=False, default_factory=list)

    ## Magic methods

    def __post_init__(self) -> None:
        """
        Initialize portfolio state.
        """
        self._cash = self.initial_capital
        self._positions = {}
        self._current_prices = {}
        self._equity_curve = []
        self._trades = []
        logger.debug("Portfolio initialized with $%.2f capital", self.initial_capital)

    def __repr__(self) -> str:
        return (
            f"Portfolio(equity=${self.equity():.2f}, "
            f"cash=${self._cash:.2f}, "
            f"positions={len(self._positions)})"
        )

    ## Properties

    @property
    def cash(self) -> float:
        """
        Current cash balance.
        """
        return self._cash

    @property
    def positions(self) -> dict[str, Position]:
        """
        Current positions (read-only copy).
        """
        return dict(self._positions)

    @property
    def equity_curve(self) -> list[EquityPoint]:
        """
        Equity curve history.
        """
        return list(self._equity_curve)

    @property
    def trades(self) -> list[FillEvent]:
        """
        List of all executed trades.
        """
        return list(self._trades)

    ## Private methods

    def _process_buy(self, pos: Position, fill: FillEvent, trade_value: float) -> None:
        """
        Process a buy order (LONG direction).
        """
        # Cash decreases
        self._cash -= trade_value + fill.commission

        if pos.quantity >= 0:
            # Adding to long or opening new long
            new_quantity = pos.quantity + fill.quantity
            if new_quantity > 0:
                # Update average cost
                old_value = pos.quantity * pos.avg_cost
                new_value = old_value + trade_value
                pos.avg_cost = new_value / new_quantity
            pos.quantity = new_quantity
        else:
            # Covering short position
            shares_to_cover = min(fill.quantity, abs(pos.quantity))

            # Realize P&L on covered shares
            realized = (pos.avg_cost - fill.fill_price) * shares_to_cover
            pos.realized_pnl += realized

            logger.debug(
                "Covering %d short shares, realized P&L: $%.2f",
                shares_to_cover,
                realized,
            )

            pos.quantity += fill.quantity

            if pos.quantity > 0:
                # Flipped to long
                pos.avg_cost = fill.fill_price
            elif pos.quantity == 0:
                pos.avg_cost = 0.0

    def _process_sell(self, pos: Position, fill: FillEvent, trade_value: float) -> None:
        """
        Process a sell order (SHORT direction).
        """
        # Cash increases
        self._cash += trade_value - fill.commission

        if pos.quantity <= 0:
            # Adding to short or opening new short
            new_quantity = pos.quantity - fill.quantity
            if new_quantity < 0:
                # Update average cost for short
                old_value = abs(pos.quantity) * pos.avg_cost
                new_value = old_value + trade_value
                pos.avg_cost = new_value / abs(new_quantity)
            pos.quantity = new_quantity
        else:
            # Closing long position
            shares_to_close = min(fill.quantity, pos.quantity)

            # Realize P&L on closed shares
            realized = (fill.fill_price - pos.avg_cost) * shares_to_close
            pos.realized_pnl += realized

            logger.debug(
                "Closing %d long shares, realized P&L: $%.2f",
                shares_to_close,
                realized,
            )

            pos.quantity -= fill.quantity

            if pos.quantity < 0:
                # Flipped to short
                pos.avg_cost = fill.fill_price
            elif pos.quantity == 0:
                pos.avg_cost = 0.0

    ## Public methods

    def get_position(self, symbol: str) -> Position:
        """
        Get position for a symbol, creating if needed.
        """
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)
            logger.debug("New position created for %s", symbol)
        return self._positions[symbol]

    def positions_value(self) -> float:
        """
        Calculate total market value of all positions.
        """
        total = 0.0
        for symbol, pos in self._positions.items():
            if symbol in self._current_prices:
                total += pos.market_value(self._current_prices[symbol])
        return total

    def equity(self) -> float:
        """
        Calculate total portfolio equity (cash + positions).
        """
        return self._cash + self.positions_value()

    def update_price(self, symbol: str, price: float) -> None:
        """
        Update the current price for a symbol.
        """
        self._current_prices[symbol] = price
        logger.debug("Price update: %s = $%.2f", symbol, price)

    def on_market_data(self, event: MarketDataEvent) -> None:
        """
        Process market data event to update prices.
        """
        self.update_price(event.symbol, event.close)

    def on_fill(self, fill: FillEvent) -> None:
        """
        Process a fill event to update positions and cash.

        For buys (LONG direction):
            - Cash decreases by (quantity * fill_price + commission)
            - Position increases

        For sells (SHORT direction):
            - Cash increases by (quantity * fill_price - commission)
            - Position decreases

        Handles position flipping (long to short or vice versa).
        """
        self._trades.append(fill)
        pos = self.get_position(fill.symbol)
        trade_value = fill.quantity * fill.fill_price

        if fill.direction == Direction.LONG:
            # Buying shares
            self._process_buy(pos, fill, trade_value)
        else:
            # Selling shares
            self._process_sell(pos, fill, trade_value)

        logger.debug(
            "Position update: %s qty=%d avg_cost=$%.2f cash=$%.2f equity=$%.2f",
            fill.symbol,
            pos.quantity,
            pos.avg_cost,
            self._cash,
            self.equity(),
        )

    def signal_to_order(
        self,
        signal: SignalEvent,
        current_price: float,
    ) -> OrderEvent | None:
        """
        Convert a signal to an order with position sizing.

        Position sizing: allocate position_size_pct of equity per trade.
        Adjusts for signal strength.

        Args:
            signal: The signal to convert
            current_price: Current price for the symbol

        Returns:
            OrderEvent if an order should be placed, None otherwise
        """
        pos = self.get_position(signal.symbol)
        current_equity = self.equity()

        # Calculate target position value
        target_value = current_equity * self.position_size_pct * signal.strength

        # Calculate target quantity
        if current_price <= 0:
            logger.warning("Invalid price $%.2f for %s", current_price, signal.symbol)
            return None

        target_quantity = int(target_value / current_price)

        if target_quantity == 0:
            logger.debug("Target quantity is 0, skipping order")
            return None

        # Determine order direction and quantity based on signal and current position
        if signal.direction == Direction.LONG:
            if pos.quantity < 0:
                # Close short first, then go long
                order_quantity = abs(pos.quantity) + target_quantity
            elif pos.quantity >= target_quantity:
                # Already have enough long exposure
                logger.debug("Already have sufficient long position")
                return None
            else:
                order_quantity = target_quantity - pos.quantity
            order_direction = Direction.LONG
        else:  # SHORT signal
            if pos.quantity > 0:
                # Close long first, then go short
                order_quantity = pos.quantity + target_quantity
            elif abs(pos.quantity) >= target_quantity:
                # Already have enough short exposure
                logger.debug("Already have sufficient short position")
                return None
            else:
                order_quantity = target_quantity - abs(pos.quantity)
            order_direction = Direction.SHORT

        if order_quantity <= 0:
            return None

        order = OrderEvent(
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            direction=order_direction,
            quantity=order_quantity,
            order_type=OrderType.MARKET,
        )

        logger.debug(
            "Order generated: %s %d %s (signal strength=%.2f, equity=$%.2f)",
            order_direction.name,
            order_quantity,
            signal.symbol,
            signal.strength,
            current_equity,
        )

        return order

    def record_equity(self, timestamp: datetime) -> None:
        """
        Record current equity to the equity curve.
        """
        point = EquityPoint(
            timestamp=timestamp,
            equity=self.equity(),
            cash=self._cash,
            positions_value=self.positions_value(),
        )
        self._equity_curve.append(point)
        logger.debug(
            "Equity recorded: $%.2f (cash=$%.2f, positions=$%.2f)",
            point.equity,
            point.cash,
            point.positions_value,
        )

    def reset(self) -> None:
        """
        Reset portfolio to initial state.
        """
        self._cash = self.initial_capital
        self._positions.clear()
        self._current_prices.clear()
        self._equity_curve.clear()
        self._trades.clear()
        logger.debug("Portfolio reset to initial state")

    def total_realized_pnl(self) -> float:
        """
        Calculate total realized P&L across all positions.
        """
        return sum(pos.realized_pnl for pos in self._positions.values())

    def total_unrealized_pnl(self) -> float:
        """
        Calculate total unrealized P&L across all positions.
        """
        total = 0.0
        for symbol, pos in self._positions.items():
            if symbol in self._current_prices:
                total += pos.unrealized_pnl(self._current_prices[symbol])
        return total
