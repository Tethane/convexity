# Convexity

Convexity is a Python toolkit for market data, quantitative models, and
strategy research. Strategies run through an event-driven stock simulator or
an IBKR paper-trading runtime. The math library provides models and numerical
tools independently of either runtime.

Requires Python 3.14+. From this checkout:

```sh
python -m pip install -e '.[dev,data,yahoo,calendars,neural,notebooks]'
```

## Examples

- [Basic Convexity](examples/basic_convexity.ipynb): buy-and-hold, overnight
  exposure, and intraday trading through the strategy API.
- [Predictive trading](examples/predictive_trading.ipynb): real prices, feature
  analysis, regression and neural models, and a held-out trading comparison.
- [Portfolio rebalancing](examples/portfolio_rebalancing.ipynb): robust mean-CVaR,
  Markowitz, and equal-weight portfolios with rolling estimates and trading costs.

The notebooks state their data sources, timing assumptions, and limitations.
The basic notebook uses Alpaca history; configure credentials with
`python scripts/configure_alpaca.py`. For IBKR data or paper execution, install
`.[ibkr]` and follow the [connection guide](docs/ibkr.md).

## Quick start

```python
import convexity as cx

prices = cx.data.history(
    ["SPY", "QQQ"], provider="yahoo",
    start="2020-01-01", end="2025-01-01",
)
closes = prices.pivot(columns="symbol", values="close")
returns = closes.pct_change(fill_method=None)
```

`data.history` returns a pandas DataFrame and uses an exclusive end date.
`Research(start, end)` replays strategies over inclusive session dates and
caches validated history. In notebooks, call `await research.run_async(...)`.

| Guide | Contents |
| --- | --- |
| [Data](docs/data.md) | Providers, timestamps, units, and missing observations |
| [Backtesting](docs/backtesting.md) | Research setup, execution assumptions, results |
| [Strategies](docs/strategies.md) | Callbacks, schedules, sizing, and paper recovery |
| [Orders](docs/order-lifecycle.md) | Amendments, stops, brackets, and subscriptions |
| [Portfolios](docs/portfolio.md) | Target weights and historical universe strategies |
| [Universes](docs/universes.md) / [prefetch](docs/data-prefetch.md) | Membership, identity, and daily coverage |
| [Corporate actions](docs/corporate-actions.md) | Explicit split, dividend, and merger accounting |
| [Math](docs/math/README.md) | Module map, examples, and numerical conventions |
| [IBKR](docs/ibkr.md) | Paper setup, execution, and reconciliation |

## Development

```sh
python -m pytest -q
python -m pyright
python -m ruff format --check convexity tests examples scripts docs/math/examples
```

Optional integrations need their corresponding extras; inspect pytest's skip
summary when checking one. Model tests establish numerical behavior on tested
cases, not predictive skill. Historical simulation uses OHLCV bars, modeled
costs, and limited liquidity assumptions. Paper execution is supported;
live-money deployment and options backtesting are not.
