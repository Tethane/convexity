# Historical data

`convexity.data` returns pandas DataFrames without a strategy or broker account.
Install `.[data,yahoo]` for Yahoo or `.[data,ibkr]` for TWS/Gateway. Binance and
Deribit use public HTTP endpoints and need only the data extra.

```python
from convexity import data

prices = data.history(
    ["SPY", "QQQ"], provider="yahoo",
    start="2020-01-01", end="2025-01-01", interval="1d",
)
closes = prices.pivot(columns="symbol", values="close")
returns = closes.pct_change(fill_method=None)
```

The frame has a UTC `timestamp` index, a `symbol` column, and float64 OHLCV
columns. Provider fields such as `adj_close` remain available. `split=True`
returns a tuple of frames in requested symbol order; `data.split_history(frame)`
splits an existing panel into a symbol-keyed dictionary.

## Time and missing data

`start` is inclusive; `end` is exclusive. Naive boundaries mean UTC. Intraday
labels are bar starts. Yahoo calendar bars and IBKR daily bars use session dates
at midnight UTC, not publication timestamps; daily boundaries must be midnight
dates. Labels across markets do not establish simultaneous availability.

Exact duplicates are removed; conflicting duplicates raise `DataError`. Gaps
and NaNs remain missing. There is no persistent cache, forward fill, synthetic
fallback, or completeness guarantee. Requests reaching the present can include
unfinished candles. Save `frame.attrs` separately if an export discards them:
they contain source, adjustment, units, request bounds, and acquisition metadata.

[Research](backtesting.md) instead uses inclusive session dates, validates replay
coverage, and caches data in SQLite. A `DataProvider` returning DataFrames is a
different interface from a replay `HistoricalProvider`.

## Providers

| Provider | Example symbol | Semantics |
| --- | --- | --- |
| Yahoo | `SPY` | Defaults to `auto_adjust=False`, `actions=True`, regular sessions. OHLC is split adjusted; `adj_close` also reflects dividends when supplied. `auto_adjust=True` dividend-adjusts OHLC. |
| Binance | `BTCUSDT` | Spot candles; volume is in base-asset units, with quote volume and trade counts retained. |
| Deribit | `BTC-PERPETUAL` | Derivative chart candles, with instrument-specific volume/cost units. No historical chain or Greek reconstruction. |
| IBKR | `SPY` | Read-only connection; defaults to SMART/USD, `use_rth=True`, `what_to_show="TRADES"`. Stock TRADES is split adjusted, not dividend adjusted. |

Symbols, currencies, contract multipliers, and adjustments remain provider
specific. A Deribit perpetual is a different instrument from Binance spot.
Common intervals are `1m`, `5m`, `15m`, `30m`, `1h`, and `1d`; inspect each
adapter's `.intervals` for others. No implicit resampling occurs. Upstream
entitlements, retention, and regional restrictions still apply.

IBKR accepts `host`, `port`, `client_id`, `timeout`, `request_interval`, exchange,
currency, and explicit native `contracts`. It serializes history requests and
paces subminute bars more strictly; it does not coordinate other API clients.
Enable the `convexity.data.ibkr` logger for progress. Per-symbol source metadata
includes request count and acquisition time. HTTP providers bound individual
requests and retries, not the total duration of a large download.

## Async calls and reusable clients

```python
from convexity.data import DataClient, YahooFinance

client = DataClient(YahooFinance())
prices = await client.ahistory(
    "SPY", start="2020-01-01", end="2025-01-01", interval="1d",
)
```

`await data.ahistory(...)` is also available. Use async access inside a running
event loop. Errors name the provider and failed operation; empty results fail
unless `allow_empty=True`. A custom `DataProvider` supplies `name` and
`read(symbol, start, end, interval)`; DataClient handles normalization and panel
assembly. Return a DatetimeIndex frame with lowercase OHLCV columns and source
metadata in `attrs`.

For bulk daily research and constituent histories, see [prefetch](data-prefetch.md)
and [historical universes](universes.md).
