"""Target weights translated into ordinary runtime orders and transparent plans."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Iterable
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from types import MappingProxyType

from convexity.runtime import (
    Bars,
    Context,
    OrderUpdate,
    Rejection,
    Strategy,
    SubmitOrder,
    Timer,
)
from convexity.runtime.types import decimal

ZERO = Decimal(0)
WEIGHT_TOLERANCE = Decimal("1e-8")
Numeric = Decimal | int | float | str


def equal_weights(
    symbols: Iterable[str], *, invested: Numeric = 1
) -> dict[str, Decimal]:
    symbols = tuple(dict.fromkeys(symbols))
    invested = decimal(invested)
    if not 0 <= invested <= 1:
        raise ValueError("invested must be between zero and one.")
    return {symbol: invested / len(symbols) for symbol in symbols}


@dataclass(frozen=True)
class RebalancePlan:
    orders: tuple[SubmitOrder, ...]
    targets: Mapping[str, int]
    weights: Mapping[str, Decimal]
    allocations: Mapping[str, Decimal]
    turnover: Decimal
    estimated_cost: Decimal
    diagnostics: tuple[str, ...]

    def __post_init__(self):
        for name in ("targets", "weights", "allocations"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


def target_weight_orders(
    context: Context,
    weights: Mapping[str, Numeric],
    *,
    max_price_age: timedelta = timedelta(days=4),
    cash_buffer: Numeric = Decimal("0.01"),
    cost_bps: Numeric = 0,
    commission_per_order: Numeric = 0,
    fractional: bool = False,
    _targets: Mapping[str, int] | None = None,
) -> RebalancePlan:
    """Plan long-only whole-share targets using observed marks and available cash.

    Weights sum to at most one. Solver residue up to 1e-8 is clipped at zero
    and a budget overshoot within that tolerance is normalized back to one.
    Missing/stale marks are reported and left alone.
    Exits precede entries; anticipated proceeds never finance unfilled buys.
    Costs are conservative planning estimates; actual fees/slippage remain the
    execution model's responsibility. Orders may still be rejected after gaps.
    """
    if fractional:
        raise NotImplementedError(
            "Current runtime order and fill contracts support whole shares only."
        )
    weights = {symbol: decimal(weight) for symbol, weight in weights.items()}
    cash_buffer, cost_bps, commission_per_order = map(
        decimal, (cash_buffer, cost_bps, commission_per_order)
    )
    if any(
        not symbol or weight < -WEIGHT_TOLERANCE for symbol, weight in weights.items()
    ):
        raise ValueError("Target weights must be nonnegative and sum to at most one.")

    weights = {symbol: max(ZERO, weight) for symbol, weight in weights.items()}
    total_weight = sum(weights.values(), ZERO)
    if total_weight > 1 + WEIGHT_TOLERANCE:
        raise ValueError("Target weights must be nonnegative and sum to at most one.")
    if total_weight > 1:
        weights = {symbol: weight / total_weight for symbol, weight in weights.items()}

    if not 0 <= cash_buffer < 1 or cost_bps < 0 or commission_per_order < 0:
        raise ValueError("Require 0 <= cash_buffer < 1 and nonnegative cost estimates.")
    if context.equity <= 0:
        raise ValueError("Target allocation requires positive account equity.")

    diagnostics, targets, prices, allocations = [], {}, {}, {}
    eligible = {
        member.security_id
        for snapshot in context.universes.values()
        for member in snapshot.members
    }
    symbols = set(weights) | {s for s, p in context.positions.items() if p.quantity}
    for symbol in sorted(symbols):
        if (
            _targets is not None
            and symbol not in _targets
            and weights.get(symbol, ZERO)
        ):
            diagnostics.append(
                f"{symbol}: no target could be sized at the decision timestamp; wait for rebalance"
            )
            continue
        if context.universes and weights.get(symbol, ZERO) and symbol not in eligible:
            diagnostics.append(f"{symbol}: outside the decision-time eligible universe")
            continue
        try:
            observed = context.price_info(symbol, max_age=max_price_age)
            price = (
                context.account.convert(observed.value, observed.currency)
                if context.account
                else observed.value
            )
        except ValueError as exc:
            diagnostics.append(f"{symbol}: {exc}")
            continue
        prices[symbol] = price
        allocations[symbol] = context.position(symbol) * price / context.equity
        targets[symbol] = (
            int(context.equity * (1 - cash_buffer) * weights.get(symbol, ZERO) / price)
            if _targets is None
            else _targets.get(symbol, 0)
        )

    reductions, additions = [], []
    for symbol, target in targets.items():
        if any(
            not order.terminal and order.request.symbol == symbol
            for order in context.orders
        ):
            diagnostics.append(f"{symbol}: waiting for a working order")
            continue
        current = context.position(symbol)
        delta = target - current
        if not delta:
            continue
        if current < 0:
            delta = -current  # flatten a legacy short before any long allocation
        order = SubmitOrder(symbol, "BUY" if delta > 0 else "SELL", abs(delta))
        (reductions if abs(current + delta) < abs(current) else additions).append(order)

    # Capacity only becomes available after actual exits. Active orders also own
    # their reservations, so a second callback cannot spend the same cash twice.
    orders = list(reductions)
    cost = ZERO
    if not reductions and not any(not o.terminal for o in context.orders):
        available = max(ZERO, context.cash - context.equity * cash_buffer)
        if context.account:
            available = min(
                [
                    available,
                    *(
                        max(ZERO, value)
                        for value in (
                            context.account.available_funds,
                            context.account.sizing_power,
                        )
                        if value is not None
                    ),
                ]
            )
        for order in additions:
            price = prices[order.symbol]
            per_share = price * (1 + cost_bps / 10_000)
            quantity = min(
                order.quantity,
                max(0, int((available - commission_per_order) / per_share)),
            )
            if quantity < order.quantity:
                diagnostics.append(
                    f"{order.symbol}: cash constraint permits {quantity} of {order.quantity} shares"
                )
            if quantity:
                orders.append(SubmitOrder(order.symbol, order.side, quantity))
                estimate = quantity * price * cost_bps / 10_000 + commission_per_order
                cost += estimate
                available -= quantity * price + estimate
    else:
        cost = sum(
            (
                o.quantity * prices[o.symbol] * cost_bps / 10_000 + commission_per_order
                for o in orders
            ),
            ZERO,
        )

    turnover = (
        sum((o.quantity * prices[o.symbol] for o in orders), ZERO) / context.equity
    )
    return RebalancePlan(
        tuple(orders), targets, weights, allocations, turnover, cost, tuple(diagnostics)
    )


class TargetWeightStrategy(Strategy):
    """Freeze targets on a scheduled decision and pursue them on later bars.

    Subclasses provide a contract and metadata. ``target(context)`` returns
    security-ID weights for dynamic universes or symbol weights for fixed ones.
    """

    def __init__(
        self,
        target: Callable[[Context], Mapping[str, Decimal | float] | None],
        *,
        timer: str = "rebalance",
        **planning_options,
    ):
        if not callable(target) or not timer:
            raise ValueError("Supply a target function and timer name.")
        self.target = target
        self.timer = timer
        self.planning_options = planning_options
        self.weights = None
        self.targets = None
        self.plans: list[RebalancePlan] = []
        self.rejections: list[str] = []
        self._last_timer = None

    def on_event(self, event, context):
        if isinstance(event, Rejection):
            self.rejections.append(event.reason)
            return
        if isinstance(event, OrderUpdate) and event.order.status == "rejected":
            self.rejections.append(event.order.reason or "Order rejected")
            return

        if isinstance(event, Timer) and event.name == self.timer:
            if event.time == self._last_timer:
                return
            self._last_timer = event.time
            result = self.target(context)
            self.weights = None if result is None else dict(result)
            self.targets = None
        elif not isinstance(event, Bars):
            return
        if self.weights is None:
            return

        # Membership can change between rebalances: removed holdings target zero.
        if context.universes:
            eligible = {
                member.security_id
                for snapshot in context.universes.values()
                for member in snapshot.members
            }
            self.weights = {s: w for s, w in self.weights.items() if s in eligible}
            if self.targets is not None:
                self.targets = {
                    s: q if s in eligible else 0 for s, q in self.targets.items()
                }
        plan = target_weight_orders(
            context, self.weights, _targets=self.targets, **self.planning_options
        )
        self.targets = dict(plan.targets)
        if plan.orders or plan.diagnostics or isinstance(event, Timer):
            self.plans.append(plan)
        return plan.orders
