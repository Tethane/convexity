"""Deterministic stock exchange driven by completed, aligned OHLCV bars."""
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Any, Mapping, Literal

from convexity.runtime import Bars, Fill, FillEvent, Instrument, Order, OrderUpdate
from convexity.runtime.types import ZERO, decimal
from convexity.runtime.capabilities import RESEARCH_CAPABILITIES


@dataclass(frozen=True)
class ExecutionModel:
    spread_bps: Decimal = ZERO
    slippage_bps: Decimal = ZERO
    commission_per_share: Decimal = ZERO
    minimum_commission: Decimal = ZERO
    volume_participation: Decimal = Decimal(1)
    submission_delay_bars: int = 0
    cancel_delay_bars: int = 0
    limit_rule: Literal["touch", "penetration"] = "touch"
    allow_short: bool = True
    annual_borrow_rate: Decimal = ZERO

    @classmethod
    def preset(cls, name: str = "standard") -> "ExecutionModel":
        """Named, deterministic execution assumptions (not calibrated venue models).

        All presets queue orders for a subsequent eligible market observation.
        Volume is shared FIFO across strategies; partial executions are events.
        """
        if name == "frictionless":
            return cls()
        if name == "standard":
            return cls(spread_bps=Decimal(2), slippage_bps=Decimal(1), commission_per_share=Decimal("0.005"),
                       minimum_commission=Decimal(1), volume_participation=Decimal("0.1"),
                       annual_borrow_rate=Decimal("0.03"))
        if name == "conservative":
            return cls(spread_bps=Decimal(5), slippage_bps=Decimal(3), commission_per_share=Decimal("0.005"),
                       minimum_commission=Decimal(1), volume_participation=Decimal("0.01"),
                       submission_delay_bars=1, cancel_delay_bars=1,
                       limit_rule="penetration", annual_borrow_rate=Decimal("0.05"))
        raise ValueError("Execution preset must be standard, frictionless, or conservative.")

    def __post_init__(self):
        for name in ("spread_bps", "slippage_bps", "commission_per_share", "minimum_commission", "volume_participation", "annual_borrow_rate"):
            object.__setattr__(self, name, decimal(getattr(self, name)))
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative.")
        if not 0 < self.volume_participation <= 1:
            raise ValueError("volume_participation must be in (0, 1].")
        if self.spread_bps / 2 + self.slippage_bps >= 10_000:
            raise ValueError("Execution friction must be less than 100% of price.")
        if self.limit_rule not in ("touch", "penetration"):
            raise ValueError("limit_rule must be touch or penetration.")
        if any(type(v) is not int or v < 0 for v in (self.submission_delay_bars, self.cancel_delay_bars)):
            raise ValueError("Delays must be nonnegative whole bars.")
        if type(self.allow_short) is not bool:
            raise ValueError("allow_short must be a bool.")


@dataclass
class _Working:
    order: Order
    delay: int
    cancel_delay: int | None = None
    triggered: bool = False
    eligible_at: datetime | None = None


class SimulatedExchange:
    capabilities = RESEARCH_CAPABILITIES

    def __init__(self, instruments: Mapping[str, Instrument], model: ExecutionModel | None = None):
        self.instruments = dict(instruments)
        self.model = model or ExecutionModel()
        self.allow_short = self.model.allow_short
        self.annual_borrow_rate = self.model.annual_borrow_rate
        self._working: dict[str, _Working] = {}
        self._seen_ids: set[str] = set()
        self._execution_sequence = 0

    def fee(self, quantity: int, already_filled: int = 0) -> Decimal:
        def total(q):
            return max(self.model.minimum_commission, q * self.model.commission_per_share) if q else ZERO
        return total(quantity + already_filled) - total(already_filled)

    def submit(self, order: Order) -> list[OrderUpdate]:
        if order.order_id in self._seen_ids:
            raise ValueError("Exchange order identity already exists.")
        self._seen_ids.add(order.order_id)
        self._working[order.order_id] = _Working(order, self.model.submission_delay_bars)
        return [OrderUpdate(order.accepted_at, order)]

    def replace(self, order: Order, time: datetime) -> list[OrderUpdate]:
        working = self._working.get(order.order_id)
        if working is None or working.order.terminal or working.cancel_delay is not None:
            raise ValueError("Replace requires a working order without a pending cancel.")
        working.order = order
        working.delay = self.model.submission_delay_bars
        working.eligible_at = time
        if order.terminal:
            self._working.pop(order.order_id, None)
        return [OrderUpdate(time, order)]

    def _cancel_children(self, parent_id: str, time: datetime) -> list[OrderUpdate]:
        events = []
        for child in tuple(self._working.values()):
            if child.order.parent_order_id == parent_id:
                child.order = replace(child.order, status="canceled", cancel_pending=False,
                                      reason="Parent terminated before full execution; partial exposure is not protected.")
                self._working.pop(child.order.order_id, None)
                events.append(OrderUpdate(time, child.order))
        return events

    def cancel(self, order_id: str, time: datetime) -> list[OrderUpdate]:
        working = self._working[order_id]
        if working.cancel_delay is not None:
            return []
        if self.model.cancel_delay_bars == 0:
            working.order = replace(working.order, status="canceled", cancel_pending=False)
            del self._working[order_id]
            return [OrderUpdate(time, working.order), *self._cancel_children(order_id, time)]
        working.cancel_delay = self.model.cancel_delay_bars
        working.order = replace(working.order, cancel_pending=True)
        return [OrderUpdate(time, working.order)]

    def _price(self, raw: Decimal, side: str, tick: Decimal) -> Decimal:
        adjustment = (self.model.spread_bps / 2 + self.model.slippage_bps) / 10_000
        value = raw * (1 + adjustment if side == "BUY" else 1 - adjustment)
        return (value / tick).to_integral_value(rounding=ROUND_CEILING if side == "BUY" else ROUND_FLOOR) * tick

    def process(self, bars: Bars, runtime: Any) -> None:
        # OHLC does not reveal which prices occurred before an intrabar expiry.
        # Reject that ambiguous input before executing any order in this panel.
        for working in self._working.values():
            order = working.order
            bar = bars.bars.get(order.request.symbol)
            if (not order.terminal and bar is not None and order.expires_at is not None
                    and bar.start < order.expires_at < bar.end):
                raise ValueError("A bar crosses order expiry; supply bars ending at the session close.")
        available = {s: int(bar.volume * self.model.volume_participation) for s, bar in bars.bars.items()}
        deferred_brackets: set[str] = set()
        bracket_priority = {}
        for working in self._working.values():
            order = working.order
            if order.parent_order_id is not None:
                bracket_priority[order.parent_order_id] = min(bracket_priority.get(order.parent_order_id, order.sequence), order.sequence)
        def priority(working):
            order = working.order
            if order.parent_order_id is not None:
                return (bracket_priority[order.parent_order_id], 0 if order.request.order_type in ("stop", "stop_limit") else 1, order.sequence)
            return (order.sequence, 0, order.sequence)
        for working in sorted(self._working.values(), key=priority):
            order = working.order
            if order.terminal or order.parent_order_id in deferred_brackets:
                continue
            bar = bars.bars.get(order.request.symbol)
            if bar is None or bar.start < order.accepted_at or (working.eligible_at is not None and bar.start < working.eligible_at):
                continue
            if order.expires_at is not None and bar.start >= order.expires_at:
                continue
            if working.cancel_delay is not None:
                if working.cancel_delay == 0:
                    working.order = replace(order, status="canceled", cancel_pending=False)
                    del self._working[order.order_id]
                    runtime.apply(OrderUpdate(bars.time, working.order))
                    for child_event in self._cancel_children(order.order_id, bars.time):
                        runtime.apply(child_event)
                    continue
                working.cancel_delay -= 1
            if not order.active:
                continue
            if working.delay:
                working.delay -= 1
                continue
            request = order.request
            immediate = request.time_in_force in ("IOC", "FOK")
            tick = self.instruments[request.symbol].tick_size
            price = self._price(bar.open, request.side, tick)
            occurred_at = bar.start
            if request.order_type in ("stop", "stop_limit") and not working.triggered:
                stop = request.stop_price
                assert stop is not None
                opening_trigger = bar.open >= stop if request.side == "BUY" else bar.open <= stop
                crossed = bar.high >= stop if request.side == "BUY" else bar.low <= stop
                if not crossed:
                    continue
                working.triggered = True
                if not opening_trigger:
                    # OHLC cannot establish whether the limit was executable after
                    # an intrabar trigger. Defer stop-limit execution to a later bar.
                    if request.order_type == "stop_limit":
                        working.eligible_at = bar.end
                        if order.parent_order_id is not None:
                            deferred_brackets.add(order.parent_order_id)
                        continue
                    price = self._price(stop, request.side, tick)
                    occurred_at = bar.end
            if request.order_type in ("limit", "stop_limit"):

                limit = request.limit_price
                assert limit is not None
                extreme = self._price(bar.low if request.side == "BUY" else bar.high, request.side, tick)
                touched = extreme <= limit if request.side == "BUY" else extreme >= limit
                penetrated = extreme < limit if request.side == "BUY" else extreme > limit
                favorable = price <= limit if request.side == "BUY" else price >= limit
                # Immediate orders have one opportunity, at the opening quote
                # proxy. A later intrabar crossing cannot rescue an IOC/FOK.
                if immediate and not favorable:
                    self._cancel_remainder(working, bars.time, runtime)
                    continue
                if not (touched if self.model.limit_rule == "touch" else penetrated):
                    if immediate:
                        self._cancel_remainder(working, bars.time, runtime)
                    continue
                # A favorable opening gap gets the friction-adjusted opening price;
                # an intrabar crossing gets the limit and an interval-end timestamp.
                if not favorable:
                    price = limit
                    occurred_at = bar.end
            quantity = min(order.remaining, available[request.symbol])
            if request.time_in_force == "FOK" and quantity < order.remaining:
                self._cancel_remainder(working, bars.time, runtime)
                continue
            if not quantity:
                if immediate:
                    self._cancel_remainder(working, bars.time, runtime)
                continue
            fee = self.fee(quantity, order.filled_quantity)
            if price <= 0 or not runtime.fill_allowed(order, quantity, price, fee, occurred_at):
                working.order = replace(order, status="rejected", cancel_pending=False,
                                        reason="Execution price/gap exceeds available capacity.")
                del self._working[order.order_id]
                runtime.apply(OrderUpdate(bars.time, working.order))
                for child_event in self._cancel_children(order.order_id, bars.time):
                    runtime.apply(child_event)
                continue
            available[request.symbol] -= quantity
            self._execution_sequence += 1
            fill = Fill(f"execution-{self._execution_sequence:06d}", order.order_id,
                        order.strategy_id, request.symbol, request.side, quantity,
                        price, fee, occurred_at, bar.end, self.instruments[request.symbol].currency)
            runtime.apply(FillEvent(bar.end, fill))
            filled = order.filled_quantity + quantity
            working.order = replace(order, filled_quantity=filled,
                                    status="filled" if filled == request.quantity else "partially_filled",
                                    cancel_pending=order.cancel_pending and filled < request.quantity)
            if working.order.terminal:
                del self._working[order.order_id]
            runtime.apply(OrderUpdate(bar.end, working.order))
            # OCO cancels on the first execution, even a partial execution. The
            # winning residual remains live; peer cancels ignore cancel latency.
            if order.group_id is not None:
                for peer in tuple(self._working.values()):
                    if (peer.order.order_id != order.order_id and peer.order.group_id == order.group_id
                            and peer.order.parent_order_id == order.parent_order_id):
                        peer.order = replace(peer.order, status="canceled", cancel_pending=False,
                                             reason="OCO peer executed.")
                        self._working.pop(peer.order.order_id, None)
                        runtime.apply(OrderUpdate(bar.end, peer.order))
            if working.order.status == "filled":
                for child in tuple(self._working.values()):
                    if child.order.parent_order_id == order.order_id:
                        child.order = replace(child.order, active=True)
                        child.eligible_at = bar.end
                        runtime.apply(OrderUpdate(bar.end, child.order))
            if immediate and not working.order.terminal:
                self._cancel_remainder(working, bars.time, runtime)

    def _cancel_remainder(self, working: _Working, time: datetime, runtime: Any) -> None:
        """IOC/FOK cancellation is immediate, independent of cancel latency."""
        working.order = replace(working.order, status="canceled", cancel_pending=False,
                                reason=f"{working.order.request.time_in_force} remainder canceled.")
        self._working.pop(working.order.order_id, None)
        runtime.apply(OrderUpdate(time, working.order))
        for child_event in self._cancel_children(working.order.order_id, time):
            runtime.apply(child_event)

    def expire(self, time: datetime) -> list[OrderUpdate]:
        events = []
        for working in tuple(self._working.values()):
            order = working.order
            if not order.terminal and order.expires_at is not None and time >= order.expires_at:
                working.order = replace(order, status="expired", cancel_pending=False)
                del self._working[order.order_id]
                events.append(OrderUpdate(time, working.order))
                events.extend(self._cancel_children(order.order_id, time))
        return events
