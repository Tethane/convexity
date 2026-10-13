"""Small shared transformations for stock order intents."""
from dataclasses import replace

from .types import BracketOrder, OCOOrders, ReplaceOrder, SubmitOrder


def expand_orders(command: BracketOrder | OCOOrders) -> tuple[SubmitOrder, ...]:
    """Stable child identities; bracket stops precede profit exits in simulation.

    Protective exits are GTC even when the entry is DAY. They remain held until
    the entry fills completely. Individual child IDs cannot override group IDs.
    """
    group = command.client_order_id
    if isinstance(command, OCOOrders):
        children = []
        for index, order in enumerate(command.orders):
            identity = f"{group}:{index}"
            if order.client_order_id not in (None, identity):
                raise ValueError("OCO child identities are derived from the group identity.")
            children.append(replace(order, client_order_id=identity))
        return tuple(children)
    if command.entry.client_order_id not in (None, f"{group}:entry"):
        raise ValueError("Bracket entry identity is derived from the group identity.")
    entry = replace(command.entry, client_order_id=f"{group}:entry")
    side = "SELL" if entry.side == "BUY" else "BUY"
    stop = (SubmitOrder.stop(entry.symbol, side, entry.quantity, command.stop_loss,
                             time_in_force="GTC", client_order_id=f"{group}:stop_loss")
            if command.stop_limit_price is None else
            SubmitOrder.stop_limit(entry.symbol, side, entry.quantity, command.stop_loss,
                                  command.stop_limit_price, time_in_force="GTC", client_order_id=f"{group}:stop_loss"))
    profit = SubmitOrder.limit(entry.symbol, side, entry.quantity, command.take_profit,
                               time_in_force="GTC", client_order_id=f"{group}:take_profit")
    return entry, stop, profit


def amended_request(request: SubmitOrder, command: ReplaceOrder) -> SubmitOrder:
    changes = {name: getattr(command, name) for name in ("quantity", "limit_price", "stop_price")
               if getattr(command, name) is not None}
    return replace(request, **changes)
