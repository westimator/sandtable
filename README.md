# Event-driven backtester

A Python backtesting framework where all components communicate exclusively through a central event queue. This design enforces temporal causality and prevents look-ahead bias by construction.

## Why event-driven?

Traditional backtesting frameworks often allow direct access to future data, making it easy to accidentally introduce look-ahead bias. This framework prevents that by design:

1. <ins>Temporal causality:</ins> Events are processed in strict timestamp order via a priority queue
2. <ins>No future data access:</ins> The `DataHandler` only exposes historical data up to the current bar
3. <ins>Realistic execution:</ins> Orders are filled with configurable slippage, market impact, and commissions
4. <ins>Clear data flow:</ins> Events flow in one direction: `MARKET_DATA → SIGNAL → ORDER → FILL`

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ DataHandler │────▶│  Strategy   │────▶│  Portfolio  │────▶│  Executor   │
│ (bars)      │     │  (signals)  │     │  (orders)   │     │  (fills)    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │
       └───────────────────┴───────────────────┴───────────────────┘
                                    │
                           ┌────────▼────────┐
                           │   Event Queue   │
                           │  (priority by   │
                           │   timestamp)    │
                           └─────────────────┘
```

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
# install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# create venv and install dependencies
uv sync

# or with dev dependencies (pytest, ruff)
uv sync --dev

# or with everything
uv sync --all-extras

# install package in editable mode (so project imports work)
uv pip install -e .
```

## Quick start

Run the example MA crossover backtest:

```bash
uv run python examples/run_ma_crossover.py
```

Example output:
```
Loading data from: data/sample_ohlcv.csv
Running backtest on 501 bars of SPY...
Strategy: MA Crossover (fast=10, slow=30)
Initial capital: $100,000.00

============================================================
BACKTEST RESULTS
============================================================
PerformanceMetrics(
  Returns:
    total_return:  -0.06%
    cagr:          -0.03%
  Risk:
    sharpe_ratio:  -0.01
    sortino_ratio: -0.01
    max_drawdown:  2.54%
  Trades:
    num_trades:    16
    win_rate:      37.5%
    profit_factor: 0.65
    avg_trade_pnl: $-63.22
  ...
)
```

## Usage

### Basic backtest

```python
from backtester.core.backtest import Backtest
from backtester.data.data_handler import CSVDataHandler
from backtester.strategy.ma_crossover import MACrossoverStrategy
from backtester.execution.simulator import ExecutionSimulator, ExecutionConfig
from backtester.execution.slippage import FixedSlippage
from backtester.portfolio.portfolio import Portfolio

# set up components
data = CSVDataHandler("data/sample_ohlcv.csv", "SPY")
strategy = MACrossoverStrategy(fast_period=10, slow_period=30)
portfolio = Portfolio(initial_capital=100_000)
executor = ExecutionSimulator(
    config=ExecutionConfig(commission_per_share=0.005),
    slippage_model=FixedSlippage(bps=5),
)

# run backtest
backtest = Backtest(data, strategy, portfolio, executor)
metrics = backtest.run()
print(metrics)
```

### Custom strategy

```python
from backtester.strategy.base import Strategy
from backtester.core.events import MarketDataEvent, SignalEvent, Direction

class MyStrategy(Strategy):
    def generate_signal(self, bar: MarketDataEvent) -> SignalEvent | None:
        closes = self.get_historical_closes(20)
        if len(closes) < 20:
            return None  # warmup period

        # [your logic here]
        if closes[-1] > sum(closes) / len(closes):
            return SignalEvent(
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                direction=Direction.LONG,
                strength=1.0,
            )
        return None
```

### Execution models

```python
from backtester.execution.slippage import ZeroSlippage, FixedSlippage, SpreadSlippage
from backtester.execution.impact import NoMarketImpact, SquareRootImpactModel

# no transaction costs (unrealistic baseline)
executor = ExecutionSimulator(
    slippage_model=ZeroSlippage(),
    impact_model=NoMarketImpact(),
)

# realistic costs
executor = ExecutionSimulator(
    config=ExecutionConfig(
        commission_per_share=0.005,
        commission_minimum=1.0,
    ),
    slippage_model=FixedSlippage(bps=5),
    impact_model=SquareRootImpactModel(eta=0.1),
)
```

## Project structure

```
event-backtester/
├── src/backtester/
│   ├── core/           # Events, queue, backtest engine
│   ├── data/           # Data loading and handling
│   ├── strategy/       # Strategy base class and implementations
│   ├── execution/      # Slippage, impact, and fill simulation
│   ├── portfolio/      # Position and cash management
│   └── metrics/        # Performance calculation
├── tests/              # Unit tests (92 tests)
├── data/               # Sample OHLCV data
├── examples/           # Example scripts
└── pyproject.toml
```

## Running tests

```bash
# run all tests
uv run pytest

# run with verbose output
uv run pytest -v

# run specific test file
uv run pytest tests/core/test_event_queue.py
```

## Design decisions

1. <ins>Lookahead Prevention:</ins> `DataHandler.get_historical_bars(n)` only returns data before the current index
2. <ins>Event Ordering:</ins> Priority queue with `(timestamp, counter)` ensures correct ordering and FIFO for same-timestamp events
3. <ins>Fill Price Bounds:</ins> Fill prices are clamped to the bar's `[low, high]` range
4. <ins>Short Positions:</ins> Cash increases on short sale, decreases on cover, with correct P&L tracking
5. <ins>Warmup Period:</ins> Strategies return `None` until they have enough data for their indicators

## Performance metrics

The `PerformanceMetrics` dataclass includes:

| Category | Metrics |
|----------|---------|
| Returns | `total_return`, `cagr` |
| Risk | `sharpe_ratio`, `sortino_ratio`, `max_drawdown` |
| Trades | `num_trades`, `win_rate`, `profit_factor`, `avg_trade_pnl` |

## License

See [LICENSE](LICENSE) file.
