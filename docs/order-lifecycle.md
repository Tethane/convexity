# Orders and subscription lifecycle

Strategies submit intent. A command returning from a callback does not mean the
broker has accepted it, an order has filled, or a cancellation has succeeded.
`OrderUpdate`, `FillEvent`, `Rejection`, and `SubscriptionUpdate` report outcomes.
Positions change on executions, not on command submission. Every command is
attributed to the calling strategy; order IDs do not confer ownership.

## Amend and cancel

```python
return cx.ReplaceOrder(
    "alpha:entry", quantity=12, limit_price="499.50",
    client_command_id="entry-reprice-1",
)
```

Quantity means **total quantity including fills**, not remaining quantity. If
five shares have filled, replacing with twelve requests seven more. Quantity
must exceed executed quantity; use `CancelOrder` to cancel the remainder.
Omitted fields retain their values. Symbol,
side, order type, and time in force stay fixed. Use a new order for a new intent.

The public order ID and prior executions survive replacement. Accepted changes
increment `Order.revision`; paper orders expose `replace_pending` while broker
confirmation is outstanding. The broker modifies the existing native order ID.
A local status echo is not sufficient to acknowledge new terms. Reusing the same
`client_command_id` with the same intent is idempotent; conflicting reuse fails.
Fills can arrive during an amendment. Uncertain transport outcomes require
reconciliation; the runtime does not blindly cancel and recreate the order.
Research amendments lose FIFO priority and cannot use an already elapsed bar.
Bracket members support price amendments, but quantity amendments are rejected
because independently resizing an entry or exit could leave inconsistent protection.

```python
return cx.CancelOrder("alpha:entry")
return cx.CancelOrders(symbol="SPY", side="BUY")
return cx.CancelOrders(group_id="alpha:position-1")
return cx.CancelOrders()  # This strategy's working orders only.
```

Bulk cancellation snapshots matching working orders at command processing time.
Filters combine with AND. It never cancels another strategy's or a manual order.
Each selected order has its own outcome. Cancellation failures leave the order
working or unresolved, with rejection/diagnostic events. `cancel_pending` is not
confirmation, and fills racing cancellation still count. Orders submitted after
the snapshot are not included. Replacing an order with cancellation outstanding
is rejected.

## Stop orders

```python
cx.SubmitOrder.stop("SPY", "SELL", 10, stop_price="490", time_in_force="GTC")
cx.SubmitOrder.stop_limit("SPY", "SELL", 10, stop_price="490",
                         limit_price="489", time_in_force="GTC")
```

A stop becomes a market order on triggering; its execution price is not bounded
by the stop price. A stop-limit becomes a limit order and may never fill after a
gap. Stops require DAY or GTC. Paper stops currently require regular sessions.
Paper uses broker-native STP / STP LMT orders;
triggering does not depend on the application remaining connected.

Research uses completed OHLC observations. An opening gap triggers at the opening
price; an intrabar stop trigger uses the trigger price with modeled friction.
When an intrabar stop-limit trigger and limit touch have ambiguous ordering, the
limit waits for a subsequent execution interval. Triggered orders retain their
state across partial executions. These rules do not invent a price path inside
a bar.

## Brackets and OCO

```python
return cx.BracketOrder(
    entry=cx.SubmitOrder.limit("SPY", "BUY", 10, "500", time_in_force="DAY"),
    take_profit="510", stop_loss="490", client_order_id="position-1",
)

return cx.OCOOrders(
    orders=(
        cx.SubmitOrder.limit("SPY", "SELL", 10, "510", time_in_force="GTC"),
        cx.SubmitOrder.stop("SPY", "SELL", 10, "490", time_in_force="GTC"),
    ),
    client_order_id="exits-1",
)
```

Groups create ordinary inspectable orders with `group_id`, `parent_order_id`,
and `active` fields. A bracket named `position-1` owned by `alpha` uses IDs
`alpha:position-1:entry`, `alpha:position-1:stop_loss`, and
`alpha:position-1:take_profit`. OCO members use `alpha:exits-1:0`, `:1`, etc.
Child IDs are derived deterministically; conflicting custom child IDs fail.
Reuse the same group ID only for the same intent.

Bracket exits have GTC lifetimes and the opposite side/quantity to the entry.
An optional `stop_limit_price` makes the protective exit a stop-limit.
Exits are held until the entry fills **completely**. This matches native parent/
child activation: a partially filled entry is not yet protected by its held
exits. Canceling or expiring a partially filled parent cancels its children and
leaves the executed position visible; the strategy must explicitly manage it.
`context.pending(symbol)` excludes held children.

The first OCO execution, including a partial execution, cancels the other members.
The winning order can remain partially filled; it does not automatically acquire
new protection. In research, cancellation is immediate within the group. Native
paper OCA cancellation is broker-managed and its confirmations remain asynchronous.
A bracket's stop is evaluated before its profit exit when the same research bar
could hit both, and children never execute inside the interval filling the parent.

Paper groups use native parent IDs, OCA groups, and staged transmission. All local
intents are persisted before transmission begins. A partially transmitted or
uncertain group is reconciled instead of resubmitted. Local cancellation or
amendment success is never substituted for a broker acknowledgment. Native OCA
semantics are documented by [IBKR](https://www.interactivebrokers.com/docs/tws-api/ref/order).

## Dynamic subscriptions

The contract declares initial subscriptions and an optional pool that may be
activated later. This keeps data requirements discoverable before research and
prevents a strategy from expanding its permissions during execution.

```python
BARS = cx.BarSubscription(("SPY",), bar_size="1 min")

class Selective(cx.Strategy):
    metadata = cx.StrategyMetadata("selective")
    contract = cx.StrategyContract(
        instruments=("SPY",),
        subscriptions=(cx.ClockSubscription("stop-observing", at="15:50"),),
        available_subscriptions=(BARS,),
    )

    def on_start(self, ctx):
        ctx.subscribe(BARS, subscription_id="prices")

    def on_timer(self, event, ctx):
        ctx.unsubscribe("prices")

    def on_subscription(self, event, ctx):
        # status: active, inactive, or rejected; reason explains failure.
        pass
```

`ctx.subscribe()` and `ctx.unsubscribe()` enqueue commands for callback completion.
Returning the same command is also supported and does not submit it twice.
A callback that raises or returns an invalid command batch publishes none of its
queued requests. Explicit `Subscribe` and `Unsubscribe` commands are equivalent.
The context snapshot does not change during a callback; inspect later
`SubscriptionUpdate` events and `ctx.subscriptions` for active identities.

Without an explicit ID, a stable ID is derived from the declaration. IDs are
scoped to their strategy and cannot be rebound to a different declaration.
Acknowledgments are delivered to the owner even without subscribing to lifecycle
events. Subscription changes do not grant trading permission or cancel orders.

Deactivation stops future delivery, including subsequent queued events. Reactivation
does not replay missed clocks, expose an inactive interval as newly observed
history, or retroactively invoke decisions. Each owner has independent delivery
and history. Research prefetches the declared pool and dynamically filters delivery;
shared paper feeds remain available while any owner needs them. Subscription state
is durable across paper reconnection, alongside existing callback/checkpoint state.

## Capabilities

Inspect `cx.RESEARCH_CAPABILITIES` and `cx.PAPER_CAPABILITIES` for supported command,
event, order, lifetime, and subscription types plus restrictions. Capabilities are
also available on runtime instances. Unsupported requests fail explicitly.

Both runtimes support stock replace/cancel groups, stops, bracket/OCO commands,
and dynamic subscriptions. Research supports IOC/FOK for standalone market/limit
orders; paper supports DAY/GTC. Paper supports option verticals; research does not.
Managed paper bar feeds currently support one-minute observations. Research uses
one exchange calendar and one bar resolution per symbol per strategy. Dynamic
subscriptions are limited to the declared pool. This release does not add L2/L3
feeds, option simulation, or live-money deployment.
