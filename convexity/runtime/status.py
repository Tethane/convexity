"""Detached runtime status suitable for terminals and notebooks."""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, cast

from convexity.trading import OrderRecord, VerticalOrder
from .account import AccountState
from .types import Fill, Order, Position

if TYPE_CHECKING:
    from .engine import Runtime
    from .paper import PaperRuntime


@dataclass(frozen=True)
class RuntimeStatus:
    time: datetime | None
    cash: Decimal
    equity: Decimal
    strategy_states: dict[str, str]
    positions: dict[str, dict[str, Position]]
    outstanding_orders: tuple[Order, ...]
    recent_fills: tuple[Fill, ...]
    errors: tuple[str, ...]
    broker_orders: tuple[OrderRecord, ...] = ()
    account: AccountState | None = None

    def format(self) -> str:
        lines = [f"Runtime {self.time} | cash {self.cash:.2f} | equity {self.equity:.2f}"]
        if self.account is not None:
            lines[0] += f" {self.account.base_currency}"
        for owner, state in self.strategy_states.items():
            positions = ", ".join(f"{s} {p.quantity:+d}" for s, p in self.positions[owner].items() if p.quantity) or "stocks flat"
            lines.append(f"{owner}: {state}; {positions}")
        legs: dict[tuple[str, int], Decimal] = {}
        for record in self.broker_orders:
            if isinstance(record.request, VerticalOrder):
                for execution in record.executions:
                    key = (record.request.strategy_id, execution.contract_id)
                    legs[key] = legs.get(key, Decimal(0)) + execution.quantity * (1 if execution.side == "BUY" else -1)
                if not record.terminal:
                    lines.append(f"COMBO {record.request.client_order_id} {record.request.side} "
                                 f"{record.remaining_quantity} {record.request.symbol}: {record.status}")
        lines.extend(f"OPTION {owner} contract {contract}: {quantity:+}" for (owner, contract), quantity in legs.items() if quantity)
        lines.extend(f"ORDER {o.order_id} {o.request.side} {o.remaining} {o.request.symbol}: {o.status}"
                     for o in self.outstanding_orders)
        lines.extend(f"FILL {f.execution_id} {f.strategy_id} {f.side} {f.quantity} {f.symbol} @ {f.price}"
                     for f in self.recent_fills)
        lines.extend(f"ERROR {error}" for error in self.errors)
        return "\n".join(lines)


def runtime_status(runtime: "Runtime | PaperRuntime", *, recent: int = 10) -> RuntimeStatus:
    if type(recent) is not int or recent < 0:
        raise ValueError("recent must be a nonnegative integer.")
    from .paper import PaperRuntime
    if isinstance(runtime, PaperRuntime):
        contexts = {owner: runtime.context(owner) for owner in runtime.strategies}
        first = next(iter(contexts.values()))
        # Quote/account retention must not erase the latest executions from the
        # status view. Read the durable journal instead of the observation ring.
        executions = sorted(((execution.occurred_at, execution.execution_id, record, execution)
            for context in contexts.values() for record in context.broker_orders
            if not isinstance(record.request, VerticalOrder) for execution in record.executions
            if execution.commission is not None and execution.commission_currency == "USD"), key=lambda row: row[:2])
        fills = tuple(Fill(execution.execution_id, runtime._order(record).order_id,
            record.request.strategy_id, record.request.symbol, execution.side, int(execution.quantity),
            execution.price, cast(Decimal, execution.commission), datetime.fromisoformat(execution.occurred_at),
            max(runtime.time, datetime.fromisoformat(execution.occurred_at)))
            for _, _, record, execution in (executions[-recent:] if recent else ()))
        return RuntimeStatus(runtime.time, first.cash, first.equity,
            {owner: "paused" if owner in runtime.paused else "active" for owner in runtime.strategies},
            {owner: dict(context.positions) for owner, context in contexts.items()},
            tuple(order for context in contexts.values() for order in context.orders if not order.terminal),
            fills, tuple(runtime.errors),
            tuple(record for context in contexts.values() for record in context.broker_orders), account=first.account)
    return RuntimeStatus(runtime.time, runtime.cash, runtime.equity,
        {owner: "paused" if owner in runtime.paused else "active" for owner in runtime.strategies},
        {owner: {s: p for (o, s), p in runtime.positions.items() if o == owner} for owner in runtime.strategies},
        tuple(o for o in runtime.orders.values() if not o.terminal),
        tuple(runtime.fills[-recent:]) if recent else (), tuple(runtime.diagnostics),
        account=runtime.account_state() if runtime.time is not None else None)
