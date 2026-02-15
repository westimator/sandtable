# sandtable

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

## Installation

```bash
pip install sandtable
```

See [sandtable on PyPI](https://pypi.org/project/sandtable/) for available versions.

### Development setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
# install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# create venv and install dependencies
uv sync

# install with all extras (yfinance, matplotlib, plotly)
uv pip install -e ".[all]"

# or with dev dependencies (pytest, ruff, viz, reports)
uv pip install -e ".[dev]"
```

## Quick start

### One-liner API

```python
from sandtable import run_backtest, AbstractStrategy, SignalEvent, MarketDataEvent, Direction, FixedSlippage

class MeanReversion(AbstractStrategy):
    lookback: int = 20
    threshold: float = 2.0

    def generate_signal(self, bar: MarketDataEvent) -> SignalEvent | None:
        closes = self.get_historical_closes(self.lookback)
        if len(closes) < self.lookback:
            return None
        mean = sum(closes) / len(closes)
        std = (sum((c - mean) ** 2 for c in closes) / len(closes)) ** 0.5
        if std == 0:
            return None
        z_score = (bar.close - mean) / std
        if z_score < -self.threshold:
            return SignalEvent(
                timestamp=bar.timestamp, symbol=bar.symbol,
                direction=Direction.LONG, strength=1.0,
            )
        return None

result = run_backtest(
    strategy=MeanReversion(),
    symbols="SPY",
    start="2022-01-01", end="2023-12-31",
    slippage=FixedSlippage(bps=5),
    commission=0.005,
)
print(result.metrics)
result.tearsheet("tearsheet.html")
```

### Parameter sweep

```python
from sandtable import Metric, run_parameter_sweep

sweep = run_parameter_sweep(
    strategy_class=MeanReversion,
    param_grid={"lookback": [10, 20, 30], "threshold": [1.5, 2.0, 2.5]},
    symbols="SPY",
    start="2022-01-01", end="2023-12-31",
    metric=Metric.SHARPE_RATIO,
)
print(sweep.best_params)
print(sweep.to_dataframe())
```

### Run the example

```bash
uv run python examples/quick_start.py
```

## Usage

### Basic backtest (manual wiring)

```python
from sandtable import CSVDataHandler, MACrossoverStrategy
from sandtable.core import Backtest
from sandtable.execution import ExecutionConfig, ExecutionSimulator, FixedSlippage
from sandtable.portfolio import Portfolio

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
from sandtable import AbstractStrategy, MarketDataEvent, SignalEvent, Direction

class MyStrategy(AbstractStrategy):
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

### Multi-symbol backtest

```python
from sandtable import run_backtest

result = run_backtest(
    strategy=MyStrategy(),
    symbols=["SPY", "QQQ", "IWM"],
    start="2022-01-01", end="2023-12-31",
)
```

### Tearsheet and comparison

```python
# Single strategy tearsheet
result.tearsheet("tearsheet.html")

# Compare multiple strategies
from sandtable import compare_strategies

compare_strategies(
    {"Strategy A": result_a, "Strategy B": result_b},
    output_path="comparison.html",
)
```

### Execution models

```python
from sandtable.execution import (
    ExecutionConfig, ExecutionSimulator,
    ZeroSlippage, FixedSlippage, SpreadSlippage,
    NoMarketImpact, SquareRootImpactModel,
)

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
src/sandtable/
├── __init__.py        # Public API exports
├── api.py             # run_backtest(), run_parameter_sweep()
├── config.py          # Configuration dataclasses
├── core/              # Events, queue, backtest engine, result
├── data_handlers/     # DataHandler protocol, CSV, yfinance, multi-symbol
├── strategy/          # Strategy base class and implementations
├── execution/         # Slippage, impact, and fill simulation
├── portfolio/         # Position and cash management
├── metrics/           # Performance calculation
├── report/            # HTML tearsheet and strategy comparison
├── utils/             # Shared utilities
└── viz/               # matplotlib charts and animation
```

## Running tests

```bash
# run all tests
uv run python -m pytest

# run with verbose output
uv run python -m pytest -v

# run specific test file
uv run python -m pytest tests/core/test_event_queue.py

# run with coverage
uv run python -m coverage run --include="src/sandtable/*" -m pytest tests/
uv run python -m coverage report --show-missing
```

## Design decisions

1. <ins>Lookahead Prevention:</ins> `DataHandler.get_historical_bars(n)` only returns data before the current index
2. <ins>Event Ordering:</ins> Priority queue with `(timestamp, counter)` ensures correct ordering and FIFO for same-timestamp events
3. <ins>Fill Price Bounds:</ins> Fill prices are clamped to the bar's `[low, high]` range
4. <ins>Short Positions:</ins> Cash increases on short sale, decreases on cover, with correct P&L tracking
5. <ins>Warmup Period:</ins> Strategies return `None` until they have enough data for their indicators
6. <ins>Multi-symbol:</ins> `MultiDataHandler` merges bars from multiple sources via min-heap for correct temporal ordering

## Performance metrics

The `PerformanceMetrics` dataclass includes:

| Category | Metrics |
|----------|---------|
| Returns | `total_return`, `cagr` |
| Risk | `sharpe_ratio`, `sortino_ratio`, `max_drawdown` |
| Trades | `num_trades`, `win_rate`, `profit_factor`, `avg_trade_pnl` |

## Further reading

Related concepts:

- [Backtesting](https://en.wikipedia.org/wiki/Backtesting)
- [Event-driven architecture](https://en.wikipedia.org/wiki/Event-driven_architecture)
- [Moving average crossover](https://en.wikipedia.org/wiki/Moving_average_crossover)
- [Mean reversion](https://en.wikipedia.org/wiki/Mean_reversion_(finance))
- [Sharpe ratio](https://en.wikipedia.org/wiki/Sharpe_ratio)
- [Sortino ratio](https://en.wikipedia.org/wiki/Sortino_ratio)
- [Maximum drawdown](https://en.wikipedia.org/wiki/Drawdown_(economics))
- [CAGR](https://en.wikipedia.org/wiki/Compound_annual_growth_rate)
- [Slippage](https://en.wikipedia.org/wiki/Slippage_(finance))
- [Market impact](https://en.wikipedia.org/wiki/Market_impact)

## License

See [LICENSE](LICENSE) file.
