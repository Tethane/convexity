"""Read-only broker status with execution-derived strategy attribution."""

from dataclasses import dataclass
from decimal import Decimal

from convexity.trading import OrderJournal, VerticalOrder
from convexity.trading.journal import OrderRecord
from .ibkr import PaperAccountConfig, PaperAccountSnapshot, read_paper_account


@dataclass(frozen=True)
class PaperStatus:
    snapshot: PaperAccountSnapshot
    orders: tuple[OrderRecord, ...]
    owned_positions: dict[str, dict[int, Decimal]]
    unassigned_positions: dict[int, Decimal]
    faults: tuple[str, ...]

    def format(self) -> str:
        lines = [
            f"Paper account DU[REDACTED] verified at {self.snapshot.completed_at}; disconnected cleanly"
        ]
        lines.extend(
            f"ACCOUNT {v.tag}: {v.value} {v.currency}"
            for v in self.snapshot.summary
            if v.tag in ("NetLiquidation", "AvailableFunds", "TotalCashValue")
        )
        for owner, positions in self.owned_positions.items():
            lines.append(
                f"STRATEGY {owner}: "
                + (
                    ", ".join(f"contract {c} {q:+}" for c, q in positions.items() if q)
                    or "flat"
                )
            )
        lines.extend(
            f"UNASSIGNED contract {c}: {q:+}"
            for c, q in self.unassigned_positions.items()
            if q
        )
        lines.extend(
            f"ORDER {o.request.client_order_id} {o.request.strategy_id}: {o.status}; filled {o.filled_quantity}/{o.request.quantity}; uncertain {o.uncertain}"
            for o in self.orders
            if not o.terminal or o.uncertain
        )
        fills = sorted(
            (
                (e.occurred_at, o.request.strategy_id, e)
                for o in self.orders
                for e in o.executions
            ),
            key=lambda row: (row[0], row[2].execution_id),
        )
        lines.extend(
            f"FILL {e.execution_id} {owner} {e.side} {e.quantity} @ {e.price}; commission {e.commission}"
            for _, owner, e in fills[-10:]
        )
        lines.extend(f"FAULT {fault}" for fault in self.faults)
        return "\n".join(lines)


async def read_paper_status(
    config: PaperAccountConfig, journal: OrderJournal
) -> PaperStatus:
    """Read broker state without sending orders; manual differences stay unassigned.

    This reports differences, not automatic ownership reconciliation. The broker
    snapshot spans a collection interval and can race incoming executions.
    """
    snapshot = await read_paper_account(config)
    orders = tuple(o for o in journal.orders() if o.account == config.account)
    owned: dict[str, dict[int, Decimal]] = {}
    faults = []
    for order in orders:
        positions = owned.setdefault(order.request.strategy_id, {})
        contracts = (
            (order.request.long_leg.contract_id, order.request.short_leg.contract_id)
            if isinstance(order.request, VerticalOrder)
            else (order.contract_id,)
        )
        for contract_id in contracts:
            positions.setdefault(contract_id, Decimal(0))
        for execution in order.executions:
            contract_id = execution.contract_id or order.contract_id
            positions[contract_id] = positions.get(contract_id, Decimal(0)) + (
                execution.quantity if execution.side == "BUY" else -execution.quantity
            )
        if order.uncertain:
            faults.append(
                f"{order.request.client_order_id}: unresolved broker outcome; reconcile before new orders."
            )
        if order.filled_quantity > order.executed_quantity:
            faults.append(
                f"{order.request.client_order_id}: missing execution details; strategy attribution incomplete."
            )
    pairs = {
        (
            order.request.strategy_id,
            order.request.long_leg.contract_id,
            order.request.short_leg.contract_id,
        )
        for order in orders
        if isinstance(order.request, VerticalOrder)
    }
    for owner, long_id, short_id in sorted(pairs):
        long_quantity = owned[owner].get(long_id, Decimal(0))
        short_quantity = owned[owner].get(short_id, Decimal(0))
        if long_quantity != -short_quantity or long_quantity < 0:
            faults.append(
                f"{owner}: unbalanced vertical legs {long_id}/{short_id}; reconcile before new orders."
            )
    differences = {p.contract.contract_id: p.quantity for p in snapshot.positions}
    for positions in owned.values():
        for contract, quantity in positions.items():
            differences[contract] = differences.get(contract, Decimal(0)) - quantity
    return PaperStatus(
        snapshot,
        orders,
        owned,
        {c: q for c, q in differences.items() if q},
        tuple(faults),
    )
