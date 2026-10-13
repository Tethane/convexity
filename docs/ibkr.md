# IBKR paper account APIs

Convexity provides read-only account/quote APIs and explicit persistent paper
execution adapters. `read_paper_account` only reads and disconnects; `PaperBroker`,
`VerticalBroker`, `PaperRuntime`, and the managed `PaperTrading` runner can submit
orders when explicitly invoked. For custom strategies, start with
[the managed Python workflow](strategies.md#one-strategy-research-or-paper).

## Setup

Use the project's `quant` conda environment (Python 3.14 or newer) and install the
optional broker integration:

```sh
conda activate quant
python -m pip install -e '.[ibkr]'
```

Log into **paper trading** in TWS or IB Gateway. In its API settings, enable socket
clients and enable downloading open orders on connection. The probe works with
**Read-Only API** either enabled or disabled; it sends only read requests.
Use the configured socket port and an unused nonzero client ID. If the broker runs
on another machine, configure the host and trusted API access appropriately.
No IBKR username or password belongs in Convexity's configuration.

Set `IBKR_ACCOUNT`, `IBKR_HOST`, `IBKR_PORT`, and `IBKR_CLIENT_ID` in the
deployment environment, and `IBKR_PAPER_CONFIRMED=true` after checking the paper
login. Alternatively construct `PaperAccountConfig(...)` directly in Python. This initial
integration accepts individual paper account IDs with a `DU` prefix, optional letters, and digits (for example `DUQ123456`).
That naming check is a guard, not independent proof of session type; the caller's
paper-login confirmation and exact match against broker-managed accounts are both
required. The port alone never establishes paper mode.

The backend is pinned to `ib_async==2.1.0`. Its `readonly` connection flag controls
startup behavior, not server permissions. The probe itself exposes no trading actions.
See [IBKR's API documentation](https://www.interactivebrokers.com/docs/tws-api/doc/introduction)
and [the client's API documentation](https://ib-api-reloaded.github.io/ib_async/api.html).

## CLI

From the repository root, with the configured Python environment:

```sh
python scripts/ibkr_probe.py
python scripts/ibkr_probe.py --redact
```

Without activating a shell, use `conda run -n quant python` in place of `python`.
Run the focused checks with `conda run -n quant python -m pytest tests/brokers -q`.

Success prints JSON containing the matched account ID, collection timestamps,
account summary, positions, and open orders, followed by connection status in the
same JSON object. Empty positions/orders appear as `[]`. Decimal values are JSON
strings. The process disconnects before printing success. Failures exit nonzero
and name the failed stage; no partial snapshot is reported as successful.

`--redact` masks the account ID. It does **not** hide balances, holdings, or broker
order IDs. Review output before sharing. To retain local evidence:

```sh
mkdir -p out
python scripts/ibkr_probe.py --redact > out/paper-account.json
```

Only a successful real broker response completes task 01. An error or a mock test
is not paper-account evidence.

## Python API

```python
import asyncio
from convexity.brokers.ibkr import PaperAccountConfig, read_paper_account

config = PaperAccountConfig.from_env()
snapshot = asyncio.run(read_paper_account(config))

print(snapshot.account)
for position in snapshot.positions:
    print(position.contract.contract_id, position.contract.symbol, position.quantity)
```

In a notebook or existing async application, use:

```python
snapshot = await read_paper_account(config)
```

You can also construct `PaperAccountConfig` directly; TOML and the CLI are optional.
The public result types are frozen Convexity dataclasses: `PaperAccountSnapshot`,
`AccountValue`, `Contract`, `Position`, and `OpenOrder`. No client-library objects
escape through this API. The broker extra is loaded only when connecting; math
imports and configuration/result types do not require it.

`PaperAccountError` reports connection, account-verification, and read failures.
Configuration errors raise `ValueError`; file errors raise `OSError`. Cancellation
propagates after disconnect. `timeout` bounds the entire connect-and-read operation.

## Snapshot limits

- The reads happen sequentially over the recorded interval, not atomically.
- Account-summary values retain broker strings and currency labels; callers choose
  which tags to interpret as amounts.
- Position cost is the broker's average cost, not a normalized underlying price.
  Conversion to `Decimal` preserves the values received from the client; it cannot
  recover precision lost in broker/client floating-point fields.
- Open orders are the cross-client snapshot visible to this API session, filtered
  to the expected account. This is not a historical order query or ongoing stream.
  Visibility depends on broker/session permissions and settings.
- This one-shot reader is not runtime reconciliation or a persistent connection.

Keep future broker operations in the package with typed results and documented
examples; scripts should remain thin wrappers around the public Convexity API.

## Verified local setup

Task 01 passed on 2026-10-09 in conda `quant` against the running Windows IB
Gateway from WSL. The Gateway was reachable at `172.31.96.1:4002`; WSL's
`127.0.0.1` did not reach it. Connection and expected-account settings now come from environment variables
or a directly constructed Python configuration. The Gateway's read-only setting was disabled;
this probe still performed only read operations.

The [local redacted snapshot](../out/paper-account.json) contains 21 summary fields,
7 positions, and 0 open orders. Connection and all reads completed successfully,
followed by clean disconnect. The output is gitignored and retains financial values.

If connectivity stops after restarting Windows/WSL, check the Windows host address
shown by `ip route` and update `IBKR_HOST` if necessary. No automatic host or
account discovery occurs in the public snapshot API.

## Resolve and quote a stock

The next public API resolves a unique stock/ETF contract and reads one ordinary
market-data snapshot. It uses the same verified paper connection and always
disconnects. It does not submit orders or request paid regulatory snapshots.

```sh
conda run -n quant python scripts/ibkr_quote.py --symbol SPY
```

From your own Python script:

```python
import asyncio
from convexity.brokers.ibkr import (
    PaperAccountConfig,
    StockQuoteRequest,
    read_stock_quote,
)

config = PaperAccountConfig.from_env()
quote = asyncio.run(read_stock_quote(config, StockQuoteRequest("SPY")))
print(quote.contract.contract_id, quote.bid, quote.ask, quote.status)
```

In a notebook, use `await read_stock_quote(config, StockQuoteRequest("SPY"))`.
The result is a frozen Convexity `StockQuote`, with no client-library objects.
`StockQuoteRequest` accepts exchange (default SMART), currency (default USD), an
optional primary exchange to disambiguate listings, requested market-data type,
and maximum local bid/ask receipt age (default 30 seconds). Unknown or ambiguous
contracts fail explicitly; the API never chooses the first of multiple matches.

The output includes resolved contract ID, primary exchange, bid/ask prices and
sizes, separate bid/ask receipt timestamps, completion time, requested and actual
feed types, snapshot completion, status, and broker diagnostics. Prices/sizes are
Decimal values in Python and strings in JSON. Missing values are `None`/`null`.

| Status | Meaning |
| --- | --- |
| `received` | Completed snapshot; positive two-sided prices/sizes recently received on a live feed |
| `missing` | At least one price or size is unavailable; no substitution with last/close |
| `stale` | Bid or ask receipt is older than max_age, or the local clock moved backward |
| `non_live` | Valid prices arrived on a delayed, frozen, or delayed-frozen feed |
| `invalid` | Crossed bid/ask |
| `unknown` | Feed type or a bid/ask receipt timestamp could not be established |
| `incomplete` | Request failed or timed out; any partial prices remain diagnostic only |

Receipt timestamps are **local receipt times, not exchange quote timestamps**.
A recent last trade does not refresh the recorded bid/ask times. `received` is a
point-in-time observation, not a guarantee of current executable liquidity; prices
can change after return. Execution adapters recheck receipt age before submission;
`PaperRuntime` and durable schedules provide strategy delivery.

There is no automatic fallback from live to delayed/frozen data. For an explicit
diagnostic request when subscriptions/session conditions require it:

```sh
conda run -n quant python scripts/ibkr_quote.py --symbol SPY --data-type delayed_frozen
```

The actual feed may differ from the requested type; results report both. See
[IBKR's market-data modes](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-delayed/introduction).
CLI exit codes: 0 for `received`, 2 for a diagnostic quote result, 1 for a
configuration/connection/resolution failure, and 130 for interruption. Diagnostic
results are still printed as JSON. `config.timeout` covers connection, resolution,
and the entire snapshot; ordinary snapshots can take roughly 11 seconds to finish.

Task 02 passed on 2026-10-09 against the actual paper Gateway in conda `quant`:
SPY resolved to contract `756733`, primary exchange ARCA, routing SMART. The
[local evidence](../out/spy-quote.json) records live feed status, bid 776.30 and ask
776.36 USD, with both sides received at 01:24:09 America/Toronto. These are the
captured values, not a current quote. The snapshot completed and disconnected.

## Submit, observe, and cancel paper stock orders

`PaperBroker` is a persistent async stock execution connection. The order types
and journal live in `convexity.trading`. `PaperRuntime` uses this adapter with
explicit strategy budgets, quote gates, and pending-order reservations. Reopening
a broker session reconciles saved intents against available broker replay; the
application controls reconnects and keeps the same journal.

```python
from decimal import Decimal
from convexity.brokers.ibkr import PaperAccountConfig
from convexity.brokers.ibkr_trading import PaperBroker
from convexity.trading import OrderJournal, StockOrder

async def submit_example(symbol: str, quantity: int, limit_price: Decimal):
    config = PaperAccountConfig.from_env()
    with OrderJournal("out/orders.sqlite") as journal:
        async with PaperBroker(config, journal, on_update=print) as broker:
            request = StockOrder(
                client_order_id="my-first-order",  # Persist/reuse for this intent.
                strategy_id="manual-test",
                symbol=symbol,
                side="BUY",
                quantity=quantity,
                limit_price=limit_price,
                session="extended",
            )
            await broker.submit(request)
            result = await broker.observe(request.client_order_id, seconds=15)
            if not result.terminal:
                await broker.cancel(request.client_order_id)
                result = await broker.observe(request.client_order_id, seconds=15)
            return result
```

The example places a paper order when invoked. Supply an explicit price and size.
It does not automatically reprice or replace an order. Cancellation may race with
fills; check the returned state and executions rather than assuming cancellation
removed exposure. Closing the session only disconnects.

CLI (explicit limit supplied by you; this example requires shell variables):

```sh
conda run -n quant python scripts/ibkr_order.py --wait 15 submit \
  --id my-first-order --strategy manual-test --symbol SPY --side BUY \
  --quantity 1 --limit "$LIMIT_PRICE" --session extended --cancel-after-wait

conda run -n quant python scripts/ibkr_order.py observe --id my-first-order
conda run -n quant python scripts/ibkr_order.py cancel --id my-first-order
```

The default journal is `out/orders.sqlite`; pass `--journal` before the subcommand
to select another file. `--wait` bounds observation, not connection setup, and must
be between 0 and 300 seconds. Without `--cancel-after-wait`, a submit command can
finish with a working order. With it, the command requests cancellation after the
observation window and waits again; a pending or uncertain cancellation exits 2.
JSON lines show status changes, acknowledgments/rejections, broker/permanent IDs,
executions, remaining quantity, and errors. Final output also shows broker positions
and the position attributed to that strategy's recorded executions. Account IDs
are masked in CLI output; the local journal retains them for correct correlation.
A broker position can include your existing/manual holdings.

### Explicit sessions

| session | Route and flags |
| --- | --- |
| `regular` | SMART, DAY limit order, outsideRth false |
| `extended` | SMART, DAY limit order, outsideRth true (regular plus pre/post-market) |
| `overnight` | OVERNIGHT, DAY limit order, outsideRth false; the dedicated route defines the overnight session |

Extended hours do not imply overnight routing. No automatic route fallback is
performed. The adapter checks the requested route against broker contract metadata.
Gateway may queue an order for a future eligible session or reject it. Acknowledged
or PreSubmitted does not mean filled. The manual API does not schedule orders;
use `SessionCalendar` and `DurableSchedule` with the runtime for timed rules.

For the verified SPY contract on 2026-10-09, Gateway reported SMART hours starting
at 04:00 Eastern, and an OVERNIGHT session ending at 03:50 Eastern. These were
broker-provided dates/hours, not hardcoded trading-calendar rules. The first
OVERNIGHT submission was rejected with error 10329 because the paper Gateway's
API precautions blocked direct routing. The narrow Gateway setting is “Bypass
Redirect Order warning for Stock API Orders” under API → Precautions. Convexity
does not change or bypass Gateway settings automatically. See
[IBKR overnight routing](https://www.interactivebrokers.com/campus/?p=201561) and
[API precautions](https://www.interactivebrokers.com/docs/tws-api/protobuf/api-precautions-config).

### Journal and lifecycle guarantees

- `StockOrder` is immutable, requires an explicit ID and strategy tag, and currently
  supports positive whole-share USD stock limit DAY/GTC orders (DAY by default;
  overnight routing supports DAY only). The broker minimum
  price increment is checked; broker validation still applies to price-tier rules.
- Intent, resolved contract ID, and assigned broker/client IDs commit to SQLite
  **before** the transport sends the order. The broker `orderRef` is the local ID.
- Reusing the same ID/account/intent returns the saved record without resubmission.
  Reusing an ID with different parameters is rejected. Never invent a new ID to
  retry an unknown outcome. Keep using the same journal between invocations.
- A timeout/disconnect is not a rejection. Unacknowledged submissions remain
  uncertain; affected new submissions stop. Opening a session reads owned open/
  completed orders and available executions, but missing results do not prove an
  order never existed. IBKR query retention/visibility can limit recovery.
- SQLite deduplicates fills by account and execution ID. Replayed fills are applied
  once. Filled status quantities and individual executions are not added together.
  Attribution derives from executions, not assumed fills or broker position changes.
- Partial fills and cancel-pending are separate. A late fill is still recorded after
  cancellation. Only broker status/error confirmation counts as cancellation;
  synthetic client-library cancellation callbacks are ignored.
- Fees are recorded when their reports arrive. Missing fees remain unknown.
  Conflicting quantities/prices cause a fault. A timestamp-only broker replay
  correction is retained as an audited journal event. Broker balances remain
  authoritative; recovery never invents missing execution details.
- `journal.get(id)`, `journal.orders()`, `journal.events(id)`, and
  `journal.attributed_position(account, strategy_id, contract_id)` expose durable
  records after the process ends. `PaperRuntime` enforces strategy limits and
  reservations; the explicit stock broker API does not impose portfolio budgets.

Run focused checks in the project's environment:

```sh
conda run -n quant python -m pytest tests/brokers tests/trading -q
```

### Paper verification on 2026-10-09

- `smoke-20261009-resting`: one-share SPY BUY limit 775.60, OVERNIGHT.
  Rejected (10329/201) by Gateway's direct-routing precaution; no execution.
- `smoke-20261009-smart-resting`: one-share BUY limit 775.66, SMART extended
  hours. Acknowledged, held for 04:00 Eastern, then explicitly canceled with
  broker confirmation. Reusing the same intent ID did not submit another order.
- `smoke-20261009-smart-fill`: one-share BUY limit 776.63, SMART extended hours.
  This fixed limit was the fresh ask plus 0.02. It remained queued for 04:00,
  then was explicitly canceled; no repricing or fill was assumed.

The [resting-order log](../out/paper-smart-resting.jsonl),
[marketable-limit attempt](../out/paper-smart-fill.jsonl),
[ID replay](../out/paper-idempotent-replay.jsonl), and
[post-test account snapshot](../out/paper-account-after-orders.json) are local
evidence of those overnight attempts. Later regular-session paper tests produced
actual stock and vertical executions. These historical snapshots do not establish
the current account state or completion of a combined strategy session.

### Option discovery and quotes

`convexity.brokers.ibkr_options.discover_options(config, symbol)` returns typed
chain metadata. Chain expiry/strike combinations are candidates; each exact
contract must still qualify. `read_option_quote(config, OptionQuoteRequest(...))`
preserves expiry, strike, right, multiplier, trading class and broker identity.
It returns the same typed `Quote` quality/receipt fields as stock quotes. This
MVP reader supports standard USD options with multiplier 100.

```sh
conda run -n quant python scripts/ibkr_options.py --symbol SPY chain
conda run -n quant python scripts/ibkr_options.py --symbol SPY quote \
  --expiry 2026-10-09 --strike 777 --right C --trading-class SPY
```

The dated command is a recorded example, not a reusable current contract.
On 2026-10-09 actual same-day SPY 777/778 calls and QQQ 751 calls returned live
two-sided quotes; local evidence is `out/spy-option-quote.json`,
`out/spy-option-wing-quote.json`, and `out/qqq-option-quote.json`. These reads
submitted no orders. Combo execution is provided by `VerticalBroker` below.
The implementation uses the client's documented
[option-chain and quote APIs](https://ib-api-reloaded.github.io/ib_async/api.html).

### Restart with incomplete execution history

If the broker reports cumulative fills exceeding the execution details saved
locally, `PaperBroker` blocks new intents even when the order is terminal.
Broker replay retention can make this happen after downtime. Recover the actual
missing executions; do not manufacture prices or infer strategy ownership from
an account position. Duplicate intent IDs still return their existing records.
`ibkr_status.py` shows attribution gaps and unassigned broker positions explicitly.


## Shared paper runtime and vertical execution

`PaperRuntime` uses the public `Strategy` callback contract for stocks and options;
see [strategy rules and scheduling](strategies.md). It requires explicit budgets
and order limits, uses real broker balances and fresh quotes, and keeps manual
account holdings distinct from execution-derived ownership. Its stock simulator
is not a model of the broker's mixed-asset margin account. Actual submissions are
only enabled by explicitly constructing and running the paper adapter.

`VerticalBroker(config, journal, strategy_budgets=..., max_contracts=...)` accepts
`VerticalOrder` with two exact `OptionIdentity` legs. It submits a single SMART
BAG DAY limit, never two independent legs. Entry reserves conservative spread
width risk against the owner's dollar budget and debit commitments against
broker AvailableFunds. Disjoint pairs may share an owner within its budget;
working overlapping legs block another intent. Normal submissions require fresh
live two-sided leg quotes. Closing requires balanced owned legs and exact broker
position reconciliation. No manual option holdings are adopted.

Option ownership is derived from actual leg execution IDs. BAG summary fills do
not double-count legs. Missing execution details block new submissions, including
a completed Filled status whose replayed executions have not arrived. Reconnect
recovers available open/completed orders and executions without resending intent
IDs. Broker replay limits can still require operator reconciliation.

`PaperBroker.quote` preserves qualified identity and returns a diagnostic
incomplete quote for market-data request errors/timeouts, including competing
session error 10197. It does not treat diagnostic prices as executable quotes.
Identity/qualification or connection failures remain errors. There is no silent
fallback to delayed data.

### Explicit close during a quote outage

`await broker.close_at_limit(sell_vertical)` is an explicit operator action that
accepts a caller-supplied minimum credit without requiring market data. It only
reduces balanced journal-owned legs and retains account, identity, quantity,
working-order, and unknown-outcome checks. Ordinary `submit` still requires
quotes. The supplied credit is a limit, not an assertion of fair value or a fill.

```sh
python scripts/ibkr_vertical_close.py --journal out/orders.sqlite \
  --entry-id EXISTING_BUY_ID --id NEW_CLOSE_ID --quantity 1 --minimum-credit 0.25
```

The CLI observes, requests cancellation if still working, and reports both broker
and journal positions. It does not automatically replace a rejected/canceled
order. A broker price precaution can reject even a risk-reducing limit. Success
requires terminal known state, complete execution details, no remaining owned
legs, and broker/journal reconciliation. Reusing an existing close ID never sends
another order. A new replacement requires an explicitly chosen new intent after
the previous outcome is known.

The local paper tests exercised this outage-close path on 2026-10-09: a 0.01
minimum-credit request was rejected/canceled by a broker price precaution; an
explicit replacement at 0.25 filled at 0.37, with both legs reconciled flat.
This is individual execution evidence, not proof of a completed combined session.
