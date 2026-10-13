# Historical DataFrames

`convexity.data` queries market history directly into pandas. It does not import
the event runtime, backtesting, trading, or broker-account modules. You do not
need a strategy, exchange calendar, paper account configuration, or backtest.

For strategy replay, use the [event-driven research guide](backtesting.md).
The two interfaces have different data contracts:

| | `data.history(...)` | `Research(start, end).run(...)` |
| --- | --- | --- |
| Result | pandas DataFrame (or a tuple with `split=True`) | Strategy orders, fills, positions, events, and metrics |
| End boundary | Exclusive | Inclusive exchange-session date |
| History cache | None | Persistent SQLite cache by default |
| Coverage | Provider observations; gaps remain visible | Validated complete sessions for replay |
| Data source | A `DataProvider` | A `HistoricalProvider` returning a validated dataset |

Install from this checkout (Python 3.14+):

```sh
python -m pip install -e '.[data,yahoo,ibkr]'
```

`data` installs pandas; Binance and Deribit use public HTTPS endpoints without
API keys. `yahoo` adds yfinance. `ibkr` adds ib_async; TWS or IB Gateway must be
running with socket API access and the appropriate market-data permissions.

## Quick start

```python
from convexity import data

df = data.history(
    ["SPY", "QQQ"], provider="yahoo",
    start="2024-01-01", end="2025-01-01", interval="1d",
)
closes = df.pivot(columns="symbol", values="close")
returns = closes.pct_change(fill_method=None)
print(returns.corr())
df.to_csv("prices.csv")
```

By default, a query returns a **plain pandas DataFrame** with a UTC `DatetimeIndex`
named `timestamp`, a `symbol` column, and float64 `open`, `high`, `low`, `close`,
`volume` columns. Multiple symbols use the same long format, sorted by timestamp
then symbol. Additional provider fields are lowercase numeric columns. One
symbol needs no special unpacking: `df["close"]` is its price series.

### Separate DataFrames per symbol

Use `split=True` to unpack frames directly, in the same order as your symbols:

```python
btc_df, eth_df = data.history(
    ["BTCUSDT", "ETHUSDT"], provider="binance",
    start="2024-01-01", end="2025-01-01", interval="1d",
    split=True,
)
print(btc_df["close"])
print(eth_df["close"])
```

This returns a tuple of independent DataFrames, each retaining its `symbol`
column, UTC index, numeric fields, and its own source metadata. The order is
the request order, even when symbols sort differently alphabetically. With
`allow_empty=True`, an empty symbol still occupies its position. A single symbol
with `split=True` returns a one-element tuple. `split=True` also works on
`DataClient.history`, `data.ahistory`, and `DataClient.ahistory`.

If you already downloaded a combined `df`, separate it without another request:

```python
tables = data.split_history(df)
btc_df = tables["BTCUSDT"]
eth_df = tables["ETHUSDT"]
```

`split_history` returns a dictionary keyed by symbol. Without `split=True`,
`history` continues to return the combined DataFrame.

`start` is inclusive and `end` exclusive. ISO strings, Python dates/datetimes,
and pandas timestamps are accepted; naive boundaries mean UTC. Intraday
timestamps identify bar starts. Yahoo calendar bars and IBKR daily bars use
the provider's session date labeled at midnight UTC, **not a close timestamp**;
their boundaries must be midnight UTC dates. Daily labels across different
markets do not imply simultaneous observations.

Rows are clipped to the requested range. Exact duplicate bars are removed;
conflicting duplicates raise `DataError`. Missing dates and numeric NaNs remain
missing: there is no interpolation, forward filling, or synthetic fallback.
This API does not certify complete market coverage. Requests through the present
can include an unfinished candle. Prices and volume units remain provider native;
the API does not impose a common currency, adjustment, or contract multiplier.

`df.attrs` records the provider, symbols, interval, boundaries, acquisition time,
and per-symbol source information (adjustments, volume units, contract identity
where available). Save this separately if your export format discards attrs.

## Providers

```python
# Binance spot, native concatenated symbol; paginates past 1,000 bars.
btc = data.history("BTCUSDT", provider="binance", interval="1h",
                   start="2024-01-01", end="2025-01-01")

# Deribit: an instrument name, including perpetuals or dated contracts.
perp = data.history("BTC-PERPETUAL", provider="deribit", interval="1h",
                    start="2024-01-01", end="2025-01-01")

# IBKR: stock/ETF symbols resolve on SMART in USD by default.
spy = data.history("SPY", provider="ibkr", interval="1d",
                   start="2024-01-01", end="2025-01-01",
                   host="127.0.0.1", port=7497, client_id=71)
```

Symbols are provider-specific. Switching just `provider="binance"` to
`provider="deribit"` while retaining `BTCUSDT`/`ETHUSDT` fails: those are Binance
symbols. To request Deribit perpetual futures explicitly:

```python
btc_df, eth_df = data.history(
    ["BTC-PERPETUAL", "ETH-PERPETUAL"], provider="deribit",
    start="2024-01-01", end="2025-01-01", interval="1d", split=True,
)
```

These are perpetual futures, not the Binance spot instruments. Convexity does
not silently substitute them. Binance-style BTC/ETH symbols produce an error
explaining the difference and the appropriate native instrument names.

| Provider | Scope and options |
| --- | --- |
| `yahoo` | yfinance history; `auto_adjust=False`, `actions=True`, `prepost=False`, `timeout=30`. Default OHLC uses Yahoo's split-adjusted prices; `adj_close` is retained when supplied. `auto_adjust=True` also adjusts OHLC for dividends. Actions include `dividends` and `stock_splits` when supplied. |
| `binance` | Spot candles at `https://data-api.binance.vision`; `volume` is base-asset volume. Also returns `quote_volume`, `trade_count`, `taker_buy_volume`, `taker_buy_quote_volume`. This adapter does not query Binance futures. |
| `deribit` | Chart OHLCV at `https://www.deribit.com`, using bounded time windows. Also returns `cost_volume`; volume/cost units and chart behavior depend on the instrument and provider. It does not reconstruct historical option chains or Greeks. |
| `ibkr` | Fresh read-only TWS/Gateway connection per symbol. Options: `host`, `port`, `client_id`, `timeout=60`, `request_interval=0.35`, `exchange="SMART"`, `currency="USD"`, `use_rth=True`, `what_to_show="TRADES"`, `contracts`. No account-number or paper-account restriction. |

IBKR `TRADES` stock prices are split-adjusted, not dividend-adjusted. Other
`what_to_show` values retain IBKR semantics. Negative sentinel volume becomes
NaN (for example, instruments with no reported trade volume). Subminute requests
within a symbol query are spaced by at least 10.1 seconds (20.1 for `BID_ASK`,
which counts twice); other broker pacing and retention limits
still apply. Each HTTP request has a timeout; crypto adapters retry transient
network failures, HTTP 429 and 5xx at most twice by default. These limits are per
request, not a deadline for the entire dataset.

### IBKR bulk-history performance

For minute-or-larger bars, the adapter sizes each request for at most 2,000
potential bars, assuming a full 24-hour market, and caps the duration at 365
days. This applies equally to stocks, extended hours, FX and explicit contracts.
Requests remain sequential with the configured pause between them; the speedup
comes from larger batches. Examples of the maximum batch durations:

| Bar interval | Days per request |
| --- | ---: |
| 1 minute | 1 |
| 5 minutes | 6 |
| 15 minutes | 20 |
| 30 minutes | 41 |
| 1 hour | 83 |
| 1 day | 365 |

Four years of hourly SPY and QQQ history require approximately **36 requests**;
exact counts depend on the dates. The start/end boundaries remain
inclusive/exclusive, and rounded-duration overfetch is clipped and deduplicated.

This uses IBKR's current
[maximum duration documentation](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/historical-bars/max-duration-per-bar-size),
rather than the smaller batches in its deprecated documentation. Subminute
requests retain their earlier duration limits and stricter pacing.
IBKR can still throttle or reject requests, especially if other notebooks or
clients share the connection/account limits; this adapter does not coordinate
their traffic or automatically retry broker pacing errors. Data is not cached.

To see acquisition progress, enable the adapter's logger:

```python
import logging

logger = logging.getLogger("convexity.data.ibkr")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())  # Run this setup once per notebook.
```

After a query, `df.attrs["sources"]["SPY"]["request_count"]` and
`df.attrs["sources"]["SPY"]["elapsed_seconds"]` report its request count and
acquisition time (including connection setup). These fields are also preserved
on frames returned with `split=True`.

Live verification on the configured Gateway (2026-10-09):

- Hourly SPY/QQQ, 2024-01-01 through 2024-04-01 exclusive: 130.8 seconds with
  seven-day batches versus 41.7 seconds with the new batches, with identical
  timestamps and OHLCV across all 854 rows.
- Hourly SPY/QQQ, 2022-01-01 through 2026-01-01 exclusive: 183.4 seconds and
  36 requests, returning 6,994 bars per symbol. All timestamps matched the
  expected regular-session hourly starts from the XNYS calendar, including
  early closes and daylight-saving changes, with no missing/extra/duplicate bars.

These are individual local measurements, not latency guarantees; Gateway and
server load, upstream caches, subscriptions and other clients affect timings.

Crypto providers accept `base_url`, `timeout=30`, `retries=2`, and
`request_interval=0.2`. Use a provider endpoint available to your account and
region. Provider errors, entitlement failures and regional restrictions surface
as errors, rather than empty success responses.

The common intervals are `1m`, `5m`, `15m`, `30m`, `1h`, `1d`.
Each provider exposes its additional supported intervals via `.intervals`:

```python
print(data.Binance.intervals)
print(data.IBKR.intervals)
```

Use `1w` and `1mo` for Yahoo/Binance weekly and monthly candles. No implicit
resampling is performed. Yahoo imposes intraday lookback limits; IBKR history
depends on bar size, instrument, subscriptions, and retention. An adapter cannot
recover history the upstream provider does not retain.

## Reusable configuration and other IBKR instruments

```python
from convexity.data import DataClient, IBKR
from ib_async import Forex

client = DataClient(IBKR(
    port=4002, client_id=72, what_to_show="MIDPOINT", use_rth=False,
    contracts={"EURUSD": Forex("EURUSD")},
))
eurusd = client.history("EURUSD", start="2025-01-02", end="2025-01-04", interval="1h")
```

The contract mapping also accepts explicit stock, future, option, index, or other
ib_async contracts where IBKR supplies history. Keys become the output symbol
labels. Ambiguous contracts fail rather than selecting an arbitrary match. Use
distinct client IDs for concurrent IBKR queries. Connections always disconnect
on success or failure; this data API has no order methods.

## Notebooks, errors, and extensions

Synchronous `history()` works in scripts and notebooks. To avoid blocking a
notebook's event loop, use the equivalent asynchronous API:

```python
df = await data.ahistory("SPY", provider="yahoo",
                         start="2025-01-01", end="2025-02-01")
# Also available: await client.ahistory(...)
```

The asynchronous wrapper uses a worker thread. Cancelling the await does not
interrupt an in-flight blocking request; configured request timeouts still apply.

A successfully returned empty response raises `NoDataError` by default. Set
`allow_empty=True` to permit empty frames, including individual empty symbols
within a panel. Provider failures still raise `DataError`; a failure for any
symbol fails the whole query. Invalid arguments raise `ValueError`/`TypeError`, and missing
optional packages raise `ImportError`. Some providers report unavailable history
as an upstream error rather than an empty response. There is no hidden disk cache.

To add a source, implement the `DataProvider` protocol: a `name` string and
`read(symbol, start, end, interval)` returning a DataFrame with a DatetimeIndex,
lowercase numeric OHLCV columns, and optional metadata in `attrs`. Pass that
instance to `DataClient` or `history(provider=...)`; normalization, panel assembly,
and async access are shared.

Provider references:
[yfinance history](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html),
[Binance spot market data](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints),
[Deribit chart history](https://docs.deribit.com/api-reference/market-data/public-get_tradingview_chart_data),
[ib_async API](https://ib-api-reloaded.github.io/ib_async/api.html).
