# Convexity

Convexity combines historical data, an event-driven strategy runtime, simulated
stock execution, IBKR paper trading, and a quantitative math library.
Python 3.14 or newer is required. Run the setup commands from this checkout.

| Task | Start here |
| --- | --- |
| Query market history into pandas | [Historical data](docs/data.md) |
| Run a strategy against historical events | [Research and backtesting](docs/backtesting.md) |
| Define strategy contracts and callbacks | [Building strategies](docs/strategies.md) |
| Submit, amend, or cancel orders; manage subscriptions | [Order lifecycle](docs/order-lifecycle.md) |
| Use models, estimators, and optimizers | [Convexity math](docs/math/README.md) |
| Configure paper execution and recovery | [IBKR guide](docs/ibkr.md) and [operations handoff](docs/handoff.md) |

## Strategy research

```sh
python -m pip install -e '.[ibkr,calendars]'
```

Strategies declare their identity, trading instruments, subscriptions, and warmup.
`Research` takes an inclusive range of session dates; run options configure
capital, execution assumptions, data acquisition, and exports.

```python
import convexity as cx

class BuyAndHold(cx.Strategy):
    metadata = cx.StrategyMetadata("buy-hold")
    contract = cx.StrategyContract(instruments=("SPY",))

    def on_start(self, context):
        return cx.SubmitOrder.market("SPY", "BUY", 10, client_order_id="entry")

result = cx.Research("2025-01-02", "2025-02-07").run(BuyAndHold)
print(result.metrics, result.positions, result.open_orders)
```

The default provider acquires Alpaca SIP history, with read-only IBKR fallback.
Run `python scripts/configure_alpaca.py` to save paper API keys outside the repo.
Validated observations persist in SQLite; covered requests replay offline and
expanded requests acquire missing sessions. Research needs no paper-account
configuration. Orders enter the simulated exchange queue; callbacks report
acceptance, executions, and failures.

In notebooks, use `await research.run_async(...)`. See the
[research guide](docs/backtesting.md) for connection settings, execution presets,
cache behavior, and result interpretation. The
[guided notebook](examples/convexity_demo.ipynb) includes charts and experiments;
[the strategy workflow](examples/strategy_workflow.py) demonstrates the same
strategy in research and paper execution.

## Historical DataFrames

```sh
python -m pip install -e '.[data,yahoo]'
```

```python
from convexity import data

df = data.history(["SPY", "QQQ"], provider="yahoo",
                  start="2024-01-01", end="2025-01-01")
closes = df.pivot(columns="symbol", values="close")
```

The standalone data API supports Yahoo Finance, Binance, Deribit, and IBKR.
It uses an exclusive end boundary and has no persistent cache. See
[historical data](docs/data.md) for provider dependencies, timestamps, and units.

## Paper execution

`PaperTrading` manages data polling, broker callbacks, schedules, checkpoints,
and reconnects. Strategies use the same command/event interface; supported
feeds and execution semantics depend on the backend. See
[capabilities and limits](docs/order-lifecycle.md#capabilities) before comparing
research and paper runs.

The [operations handoff](docs/handoff.md) contains explicit paper-session and
recovery commands. [Paper execution evidence](docs/paper-evidence.md) records
previous acceptance runs.
