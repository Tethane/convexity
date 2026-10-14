# IBKR paper trading

## Setup

Install `.[ibkr,calendars]` with Python 3.14+. Log into paper TWS or IB Gateway,
enable API socket clients and downloading open orders, then configure:

```sh
export IBKR_HOST=127.0.0.1
export IBKR_PORT=4002
export IBKR_CLIENT_ID=72
export IBKR_ACCOUNT=YOUR_PAPER_ACCOUNT
export IBKR_PAPER_CONFIRMED=true

python scripts/ibkr_probe.py --redact
python scripts/ibkr_status.py
```

Use the actual account and socket settings. Paper mode requires explicit
confirmation and a match with broker-managed accounts; a port or DU account
prefix alone does not establish it. WSL may need the Windows host from `ip route`
instead of localhost. Credentials stay in TWS/Gateway.

The probe reads account state and disconnects. `--redact` masks the account ID,
not balances or holdings. Python callers use
`await read_paper_account(PaperAccountConfig.from_env())` from
`convexity.brokers.ibkr`. Reads are sequential snapshots, not atomic account
state or ongoing reconciliation. Broker strings/currencies remain intact;
position cost is broker average cost, not normalized underlying price.

## Quotes and discovery

```sh
python scripts/ibkr_quote.py --symbol SPY
python scripts/ibkr_options.py --symbol SPY chain
```

Stock quotes retain qualified identity, bid/ask sizes, feed type, and local
receipt times. `received` means a completed live two-sided snapshot; missing,
stale, crossed, delayed, or incomplete quotes remain diagnostic. Receipt time is
not exchange time. There is no silent delayed-data fallback. Quote CLI exit codes
are 0 for received, 2 for diagnostic results, 1 for errors, and 130 for interruption.

`read_option_quote(config, OptionQuoteRequest(...))` preserves exact expiry,
strike, right, multiplier, trading class, and contract ID. Chain combinations are
candidates and must still qualify. Supported options are standard USD contracts
with multiplier 100. Competing session error 10197 requires restoring market-data
access before fresh-quote execution can proceed.

## Execution and ownership

[PaperTrading](strategies.md#one-strategy-research-or-paper) manages strategy
execution. Lower-level `PaperBroker` and `VerticalBroker` use a persistent
`OrderJournal`; read-only account/quote calls do not submit orders.

```sh
python scripts/ibkr_order.py --help
python scripts/ibkr_order.py observe --id EXISTING_ORDER_ID
python scripts/ibkr_order.py cancel --id EXISTING_ORDER_ID
```

The default journal is `out/orders.sqlite`; pass `--journal` before the
subcommand to change it. Manual stock submissions require explicit ID, owner,
quantity, limit, and session. Managed market commands become protective limits
and may not fill. Cancellation can race with executions; inspect the terminal
state rather than treating a cancellation request as a flat position.

| Stock session | Route |
| --- | --- |
| regular | SMART, outsideRth false |
| extended | SMART, outsideRth true |
| overnight | OVERNIGHT route, DAY only |

Extended hours do not imply overnight routing. Contract metadata and broker
precautions still apply; Convexity does not bypass them. DAY/GTC whole-share USD
stock orders are supported. See [order lifecycle](order-lifecycle.md) for stops,
amendments, and groups.

Intents and transport IDs commit before transmission. Reusing the same
ID/account/intent returns the saved record; changed intent rejects. A timeout
is an unknown outcome, not evidence of rejection. Never assign a new ID merely
to retry an uncertain submission.

Executions determine strategy ownership. Fill IDs deduplicate replay; cumulative
status and individual fills are not added together. Fees remain unknown until
commission reports arrive. Manual positions stay unassigned. The broker is
authoritative for balances; the stock simulator is not a model of its mixed-asset
margin system.

## Recovery

Keep the same deployment directory and journal when reconnecting. The broker
adapter reads available open/completed orders and executions without resending
saved intents. Broker replay retention is finite: missing records do not prove
an order never existed. Cumulative fills exceeding known execution details block
new submissions, even for terminal orders. Recover actual executions; do not
manufacture fill prices or infer strategy ownership from the account position.

Inspect `ibkr_status.py`, journal orders/events, and
[strategy callback and timer state](strategies.md#interrupted-callbacks) before
explicitly resolving unknown effects. Disconnecting or stopping the runner does
not flatten positions. Back up the deployment directory.

## Vertical spreads

`VerticalBroker` submits one SMART BAG DAY limit with two exact legs. Entry
reserves spread-width risk against the strategy budget and debit commitments
against AvailableFunds. Working overlapping legs block new intent. Normal
submissions require fresh live two-sided quotes. Closing requires balanced
journal-owned legs and broker-position reconciliation. BAG summary fills never
double-count leg executions; unassigned option holdings are not adopted.

For a quote outage, an operator can explicitly choose a closing minimum credit:

```sh
python scripts/ibkr_vertical_close.py --journal out/orders.sqlite \
  --entry-id EXISTING_BUY_ID --id NEW_CLOSE_ID --quantity 1 --minimum-credit 0.25
```

The supplied credit is a limit, not a valuation or promised fill. This path only
reduces verified balanced legs and retains ownership, quantity, working-order,
and uncertainty checks. It observes, requests cancellation if still working,
and reports broker/journal positions. It does not automatically replace rejected
orders. Reusing the same close ID does not send a second order.
