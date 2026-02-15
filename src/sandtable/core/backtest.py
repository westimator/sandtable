"""
src/sandtable/core/backtest.py

Main backtest engine that orchestrates the event-driven simulation.

The Backtest class coordinates data handling, strategy execution,
order processing, and portfolio management through a central event queue.
"""

from dataclasses import dataclass, field

from sandtable.core.event_queue import EventQueue
from sandtable.core.events import (
    EventType,
    FillEvent,
    MarketDataEvent,
    OrderEvent,
    SignalEvent,
)
from sandtable.data_handlers.abstract_data_handler import AbstractDataHandler
from sandtable.execution.simulator import ExecutionSimulator
from sandtable.metrics.performance import PerformanceMetrics, calculate_metrics
from sandtable.portfolio.portfolio import Portfolio
from sandtable.strategy.abstract_strategy import AbstractStrategy
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)


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

    data_handler: AbstractDataHandler
    strategy: AbstractStrategy
    portfolio: Portfolio
    executor: ExecutionSimulator

    _event_queue: EventQueue = field(init=False)
    _current_bars: dict[str, MarketDataEvent] = field(init=False, default_factory=dict)

    ## Magic methods

    def __post_init__(self) -> None:
        """
        Initialize the event queue.
        """
        self._event_queue = EventQueue()
        self._current_bars = {}

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
        logger.debug("Processing event: %s", event.event_type)
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
        logger.debug(
            "Handling arrival of bar: %s %s O=%.2f H=%.2f L=%.2f C=%.2f V=%.0f",
            event.symbol,
            event.timestamp.date(),
            event.open,
            event.high,
            event.low,
            event.close,
            event.volume,
        )

        self._current_bars[event.symbol] = event

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
        logger.debug(
            "Handling signal: %s %s strength=%.2f",
            event.direction.name,
            event.symbol,
            event.strength,
        )

        # get current price for position sizing
        current_bar = self._current_bars.get(event.symbol)
        if current_bar is None:
            logger.warning("No current bar for signal processing")
            return

        current_price = current_bar.close

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
        logger.debug(
            "Handling order: %s %d %s @ %s",
            event.direction.name,
            event.quantity,
            event.symbol,
            event.order_type.name,
        )

        current_bar = self._current_bars.get(event.symbol)
        if current_bar is None:
            logger.warning("No current bar for order execution")
            return

        # execute order and get fill
        fill = self.executor.process_order(event, current_bar)
        self._event_queue.push(fill)

    def _handle_fill(self, event: FillEvent) -> None:
        """
        Handle fill event.

        Updates portfolio with executed trade.

        Args:
            event: Fill event with execution details
        """
        logger.debug(
            "Handling fill: %s %d %s @ $%.2f (comm=$%.2f)",
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

        logger.debug("Event loop finished, calculating metrics")

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
        logger.debug("Resetting backtest...")
        self.data_handler.reset()
        self.strategy.reset()
        self.portfolio.reset()
        self._event_queue = EventQueue()
        self._current_bars = {}
        logger.debug("Backtest successfully reset")
