# Backtesting

`Research` acquires history and replays a strategy through the stock simulator.
Session dates are inclusive. `Backtest` accepts caller-supplied data directly.
For exploratory DataFrames, use [the data API](data.md).

## Setup and first run

```sh
python -m pip install -e '.[ibkr,calendars]'
python scripts/configure_alpaca.py
```

The configuration script saves Alpaca keys outside the repository in
`~/.config/convexity/alpaca.json` (or under `XDG_CONFIG_HOME`), with owner-only
permissions. Both `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` in the environment
override that file. Research calls market-data APIs only.

```python
import convexity as cx

class BuyAndHold(cx.Strategy):
    metadata = cx.StrategyMetadata("buy-hold")
    contract = cx.StrategyContract(instruments=("SPY",))

    def on_start(self, context):
        return cx.SubmitOrder.market(
            "SPY", "BUY", 10, client_order_id="entry",
        )

research = cx.Research("2025-01-02", "2025-02-07")
result = research.run(BuyAndHold, initial_cash=100_000, output="out/buy-hold")
```

In a notebook, use `await research.run_async(...)`. Each run creates fresh
strategy instances; pass a sequence of factories for several strategies sharing
an account. Names must be unique. Use separate Research instances for concurrent
runs. A prior-session observation supports startup sizing without exposing the
first execution bar. Returning an order does not mean it filled.

## Data and cache

The default provider requests Alpaca SIP, split-adjusted minute history and
aggregates regular-session bars. If acquisition fails, it warns and retries the
whole requested panel through read-only IBKR. `provider=cx.AlpacaHistoricalData()`
or `provider=cx.IBKRHistoricalData(...)` requires a specific source. Inspect
`result.config["data_source"]` and research metadata for actual feed and fallback.

IBKR discovery tries Gateway/TWS ports 4002, 7497, 4001, 7496 on localhost and,
under WSL, its Windows host. `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID` (default
72), and `IBKR_TIMEOUT` (default 15 seconds) override discovery. Connection
refusal means the socket API is unavailable; changing strategy code will not
fix it. A paper-account configuration is unnecessary for historical reads.

Validated observations persist in `.convexity/history.sqlite3`. Covered ranges
replay offline; expanded requests fetch missing sessions. Prior-session and
execution feeds must also be cached. `cache=...` changes the path, `cache=None`
disables the outer SQLite cache, and `refresh=True` reacquires requested history.
An explicit `FallbackHistoricalProvider` owns its inner caches; configure it
there. Provider identity, calendar, resolution, and price basis separate feeds.

Missing intervals fail rather than being filled with invented observations.
Minute replay needs dense history and can be slow even when cached. A 15:50
timer requires minute execution data throughout the period. Native daily bars
have different availability and volume semantics; see [daily prefetch](data-prefetch.md).
Standalone `data.history` has no persistent cache and uses an exclusive end date.

## Replaying a DataFrame

Use `DataFrameHistoricalProvider` to analyze and backtest the same daily snapshot:

```python
from convexity.backtest import DataFrameHistoricalProvider

prices = cx.data.history(
    "SPY", provider="yahoo", auto_adjust=False,
    start="2024-12-01", end="2025-02-08", interval="1d",
)
provider = DataFrameHistoricalProvider(
    prices, price_basis="split_adjusted_price_return",
)
result = await cx.Research("2025-01-02", "2025-02-07").run_async(
    BuyAndHold, provider=provider, cache=None,
)
```

Include prior-session data and the strategy's full warmup in the frame. The
provider copies the snapshot, retains provenance, and validates daily coverage
against its declared calendar (XNYS by default). It does not download missing
rows or refresh itself; acquire a new frame to update the snapshot.

Native daily bars are observed at the following session's open. Static Research
supports these delayed observations: an order decided from yesterday's bar can
execute at today's open under the zero-latency model. This does not make today's
close available at the decision. Fill callbacks and account views also withhold
unpublished closing marks; reported equity still uses actual closing valuations.
Daily OHLC execution remains an approximation;
it cannot resolve intraday price order or regular-session liquidity from a native
bar's full-day volume. `auto_adjust=False` keeps the Yahoo split-adjusted OHLC
basis explicit; these results omit dividends and are not total returns.

## Execution assumptions

Pass `execution="standard"`, `"frictionless"`, `"conservative"`, or an
`ExecutionModel`. Set account constraints with `RiskLimits`.

| Preset | Spread / slippage | Participation | Other costs and timing |
| --- | --- | --- | --- |
| standard | 2 / 1 bp | 10% | $0.005/share, $1/order minimum; 3% annual short borrow |
| conservative | 5 / 3 bp | 1% | Same commission; 5% borrow; one-interval submit/cancel delays; strict limit penetration |

These assumptions are not calibrated market-impact estimates. OHLCV replay
cannot reconstruct an intrabar tape or queue position. Orders cannot fill from
the bar that generated them. Eligible opening fills become visible at the bar's
end; limit crossings use the end timestamp. Fills update accounts before
callbacks; bars precede equal-time timers. FIFO orders share available volume.
There is no automatic liquidation at the end of a run.

Named order constructors default to DAY. GTC orders may remain open at the end.
IOC fills what it can at the first eligible opening opportunity and cancels the
rest; FOK requires the whole quantity there. Stops and groups require DAY/GTC.
See [order lifecycle](order-lifecycle.md) for stop, bracket, and amendment rules.

Default history is split-adjusted price data: it omits dividend cash flows and
uses adjusted share units. It is not a total-return benchmark. Explicit
[corporate actions](corporate-actions.md) require unadjusted input and sourced
events. Refresh the full study history after adjustment revisions to avoid
mixing price scales across acquisition dates.

The simulator supports stocks, one exchange calendar per run, and one resolution
per symbol per strategy. Different strategies may use different resolutions.
Historical [universes and portfolios](portfolio.md) support daily data and
session-boundary schedules. Options simulation and L2/L3 feeds are unsupported.

## Currency ledgers and FX

`initial_cash` accepts a scalar in `base_currency` (default USD), or a currency
mapping. Supply `fx_rates=(...)` for foreign balances. Each
`FXRate(currency, base_currency, rate, time)` means one foreign unit is worth
`rate` base units, known at the timezone-aware timestamp. Initial nonzero foreign
balances need a rate at or before startup; future rates cannot fill that gap.
Research does not acquire FX history automatically.

Cash remains in native currency. Funding a USD purchase with foreign collateral
can create a USD loan; no FX trade, deposit interest, or currency-loan interest
is simulated. Short-stock borrow is modeled. Supplied Backtest instruments can
have other currencies; automatic Research acquisition remains US USD stocks.
Equity includes FX revaluation; position costs retain instrument currency.

## Results

```python
import pandas as pd

equity = pd.DataFrame(result.equity).set_index("time")
equity.index = pd.to_datetime(equity.index, utc=True)
equity["equity"].astype(float).plot(ylabel="USD")

print(result.metrics)
print(result.positions)
print(result.open_orders)
result.export("out/review")
```

Metrics use Decimal values; returns and drawdowns are fractions. Convert to
float for plots while keeping the original accounting records. Exports include
fills, orders, cash flows, events, assumptions, provenance, and replay inputs.
`BacktestFailure.result` retains partial results. A successful run can still
contain rejections, partial fills, or open positions. Inspect those before
interpreting its return.

See [basic usage](../examples/basic_convexity.ipynb),
[predictive trading](../examples/predictive_trading.ipynb), and
[portfolio rebalancing](../examples/portfolio_rebalancing.ipynb) for comparisons.
