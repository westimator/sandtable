"""
src/backtester/core/backtest.py

Main backtest engine that orchestrates the event-driven simulation.

The Backtest class coordinates data handling, strategy execution,
order processing, and portfolio management through a central event queue.
"""

import logging
from dataclasses import dataclass, field

from backtester.core.event_queue import EventQueue
from backtester.core.events import (
    EventType,
    FillEvent,
    MarketDataEvent,
    OrderEvent,
    SignalEvent,
)
from backtester.data.data_handler import CSVDataHandler
from backtester.execution.simulator import ExecutionSimulator
from backtester.metrics.performance import PerformanceMetrics, calculate_metrics
from backtester.portfolio.portfolio import Portfolio
from backtester.strategy.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class Backtest:
    """
    Event-driven backtesting engine.

    Orchestrates the backtest by processing events through a central queue.
    Events flow in this order:
        MARKET_DATA → SIGNAL → ORDER → FILL

    Attributes:
        data_handler: Provides market data bars
        strategy: Generates trading signals
        portfolio: Manages positions and cash
        executor: Simulates order execution

    Example:
        >>> data = CSVDataHandler("data/spy.csv", "SPY")
        >>> strategy = MACrossoverStrategy(fast_period=10, slow_period=30)
        >>> portfolio = Portfolio(initial_capital=100_000)
        >>> executor = ExecutionSimulator()
        >>> backtest = Backtest(data, strategy, portfolio, executor)
        >>> metrics = backtest.run()
        >>> print(metrics)
    """

    ## Properties

    data_handler: CSVDataHandler
    strategy: Strategy
    portfolio: Portfolio
    executor: ExecutionSimulator

    _event_queue: EventQueue = field(init=False)
    _current_bar: MarketDataEvent | None = field(init=False, default=None)

    ## Magic methods

    def __post_init__(self) -> None:
        """
        Initialize the event queue.
        """
        self._event_queue = EventQueue()
        self._current_bar = None

    ## Private methods

    def _run_event_loop(self) -> None:
        """
        Main event loop that processes all events.

        When the queue is empty, fetches the next bar from data handler.
        When a bar is available, pushes it to the queue.
        When no more bars, exits the loop.
        """
        running = True
        while running:
            if self._event_queue:
                # process next event from queue
                event = self._event_queue.pop()
                self._process_event(event)  # routes to appropriate handler based on type
            else:
                # queue empty, get next bar
                bar = self.data_handler.get_next_bar()
                if bar is None:
                    # no more data
                    running = False
                else:
                    self._event_queue.push(bar)

    def _process_event(self, event: MarketDataEvent | SignalEvent | OrderEvent | FillEvent) -> None:
        """
        Route event to appropriate handler based on type.

        Args:
            event: The event to process
        """
        if event.event_type == EventType.MARKET_DATA:
            self._handle_market_data(event)
        elif event.event_type == EventType.SIGNAL:
            self._handle_signal(event)
        elif event.event_type == EventType.ORDER:
            self._handle_order(event)
        elif event.event_type == EventType.FILL:
            self._handle_fill(event)
        else:
            logger.warning("Unknown event type: %s", event.event_type)

    def _handle_market_data(self, event: MarketDataEvent) -> None:
        """
        Handle market data event.

        1. Store current bar for order execution
        2. Update portfolio with new price
        3. Pass to strategy for signal generation
        4. Record equity snapshot

        Args:
            event: Market data event with OHLCV data
        """
        self._current_bar = event

        logger.info(
            "Bar: %s %s O=%.2f H=%.2f L=%.2f C=%.2f V=%.0f",
            event.symbol,
            event.timestamp.date(),
            event.open,
            event.high,
            event.low,
            event.close,
            event.volume,
        )

        # update portfolio with current price
        self.portfolio.on_market_data(event)

        # pass to strategy, may generate signal
        signal = self.strategy.on_bar(event)
        if signal is not None:
            self._event_queue.push(signal)

        # record equity at end of bar
        self.portfolio.record_equity(event.timestamp)

    def _handle_signal(self, event: SignalEvent) -> None:
        """
        Handle signal event.

        Converts signal to order via portfolio position sizing.

        Args:
            event: Signal event from strategy
        """
        logger.info(
            "Signal: %s %s strength=%.2f",
            event.direction.name,
            event.symbol,
            event.strength,
        )

        # get current price for position sizing
        if self._current_bar is None:
            logger.warning("No current bar for signal processing")
            return

        current_price = self._current_bar.close

        # convert signal to order
        order = self.portfolio.signal_to_order(event, current_price)
        if order is not None:
            self._event_queue.push(order)

    def _handle_order(self, event: OrderEvent) -> None:
        """
        Handle order event.

        Sends order to execution simulator for fill.

        Args:
            event: Order event to execute
        """
        logger.info(
            "Order: %s %d %s @ %s",
            event.direction.name,
            event.quantity,
            event.symbol,
            event.order_type.name,
        )

        if self._current_bar is None:
            logger.warning("No current bar for order execution")
            return

        # execute order and get fill
        fill = self.executor.process_order(event, self._current_bar)
        self._event_queue.push(fill)

    def _handle_fill(self, event: FillEvent) -> None:
        """
        Handle fill event.

        Updates portfolio with executed trade.

        Args:
            event: Fill event with execution details
        """
        logger.info(
            "Fill: %s %d %s @ $%.2f (comm=$%.2f)",
            event.direction.name,
            event.quantity,
            event.symbol,
            event.fill_price,
            event.commission,
        )

        # update portfolio
        self.portfolio.on_fill(event)

    ## Public methods

    def run(self) -> PerformanceMetrics:
        """
        Run the backtest and return performance metrics.

        Processes all market data through the event loop, then
        calculates and returns performance metrics.

        Returns:
            PerformanceMetrics with backtest results
        """
        logger.info(
            "Starting backtest: %s with %d bars",
            self.data_handler.symbol,
            len(self.data_handler),
        )

        self._run_event_loop()

        logger.info(
            "Backtest complete: processed %d bars",
            self.data_handler.current_index,
        )

        # calculate and return metrics
        metrics = calculate_metrics(
            equity_curve=self.portfolio.equity_curve,
            trades=self.portfolio.trades,
        )

        return metrics

    def reset(self) -> None:
        """
        Reset all components for a new backtest run.
        """
        self.data_handler.reset()
        self.strategy.reset()
        self.portfolio.reset()
        self._event_queue = EventQueue()
        self._current_bar = None
        logger.info("Backtest reset")
