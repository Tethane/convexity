"""Serial strategy dispatch and stock accounting, independent of historical loading."""
from bisect import bisect_right
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from typing import Mapping, cast

from .account import AccountState, FXRate, currency
from .types import PriceObservation

from .contracts import (ContractViolation, assert_contract_unchanged, dispatch_event, filter_context, select_event,
                        strategy_contract, validate_commands)
from .subscriptions import SubscriptionBook
from .capabilities import RESEARCH_CAPABILITIES
from .types import (
    ZERO, Bars, Bar, CancelOrder, Command, Context, Event, Exchange, Fill, FillEvent,
    Instrument, Order, OrderUpdate, Position, Rejection, RiskLimits, Strategy,
    SubmitOrder, Timer, VerticalOrder, aware, decimal, COMMAND_TYPES, ReplaceOrder,
    CancelOrders, BracketOrder, OCOOrders, Subscribe, Unsubscribe, collect_commands,
)


class Runtime:
    capabilities = RESEARCH_CAPABILITIES

    def __init__(self, strategies: Mapping[str, Strategy], exchange: Exchange,
                 instruments: Mapping[str, Instrument], initial_cash: Decimal | int | str | Mapping[str, Decimal] = 100_000,
                 limits: RiskLimits | None = None, session_closes: tuple[datetime, ...] = (), *,
                 base_currency: str = "USD", fx_rates: tuple[FXRate, ...] = ()):
        if not strategies or any(not k for k in strategies):
            raise ValueError("Register at least one named strategy.")
        self.strategies = dict(strategies)
        self._registered_contracts = {owner: strategy_contract(strategy) for owner, strategy in self.strategies.items()}
        self._subscriptions = {owner: SubscriptionBook(contract, owner) for owner, contract in self._registered_contracts.items()}
        self._warmup_history = {}
        self._owner_history = {}
        self._started_at = None
        self._replacement_intents: dict[str, Order] = {}
        self._command_ids = {}
        self._group_intents = {}
        self.exchange = exchange
        self.instruments = dict(instruments)
        if any(symbol != instrument.symbol for symbol, instrument in instruments.items()):
            raise ValueError("Instruments must be a symbol-keyed mapping.")
        self.base_currency = currency(base_currency)
        self.cash_balances = ({currency(c): decimal(v) for c, v in initial_cash.items()}
                              if isinstance(initial_cash, Mapping) else {base_currency: decimal(initial_cash)})
        self.fx_history = tuple(sorted(fx_rates, key=lambda r: r.time))
        if any(r.base_currency != base_currency for r in self.fx_history):
            raise ValueError("FX rates must use the runtime base currency.")
        identities = {}
        for rate in self.fx_history:
            key = rate.currency, rate.time
            if key in identities and identities[key] != rate.rate:
                raise ValueError("Conflicting FX observations.")
            identities[key] = rate.rate
        self._fx_series: dict[str, list[FXRate]] = {}
        for rate in self.fx_history:
            self._fx_series.setdefault(rate.currency, []).append(rate)
        self._fx_times = {code: [rate.time for rate in values] for code, values in self._fx_series.items()}
        self.fx_rates: dict[str, FXRate] = {}
        self.initial_cash = ZERO
        self.limits = limits or RiskLimits()
        for close in session_closes:
            aware(close)
        self.session_closes = tuple(sorted(session_closes))
        self.time: datetime | None = None
        self.orders: dict[str, Order] = {}
        self._active_orders: dict[str, Order] = {}
        self._orders_by_owner: dict[str, dict[str, Order]] = {owner: {} for owner in strategies}
        self.positions: dict[tuple[str, str], Position] = {}
        self.marks: dict[str, Decimal] = {}
        self.mark_times: dict[str, datetime] = {}
        self.history: dict[str, list[Bar]] = {s: [] for s in instruments}
        self.fills: list[Fill] = []
        self._executions: dict[str, Fill] = {}
        self.events: list[Event] = []
        self.diagnostics: list[str] = []
        self.paused: set[str] = set()
        self.fees = ZERO
        self.borrow_cost = ZERO
        self.cash_flows: list[dict] = []
        self.equity_history: list[dict] = []
        self._queue: deque[tuple[Event, tuple[str, ...] | None]] = deque()
        self._sequence = 0
        self._used = False

    def _update_fx(self, time: datetime) -> None:
        for code, values in self._fx_series.items():
            index = bisect_right(self._fx_times[code], time)
            if index:
                self.fx_rates[code] = values[index - 1]
            else:
                self.fx_rates.pop(code, None)

    def _fx(self, code: str) -> Decimal:
        if code == self.base_currency:
            return Decimal(1)
        if code not in self.fx_rates:
            raise ValueError(f"No observed FX rate for {code}/{self.base_currency}.")
        return self.fx_rates[code].rate

    def _value(self, symbol: str, amount) -> Decimal:
        return decimal(amount) * self._fx(self.instruments[symbol].currency)

    def _cash_change(self, code: str, amount: Decimal) -> None:
        self.cash_balances[code] = self.cash_balances.get(code, ZERO) + amount

    @property
    def cash(self) -> Decimal:
        return sum((value * self._fx(code) for code, value in self.cash_balances.items() if value), ZERO)

    def account_state(self) -> AccountState:
        assert self.time is not None
        gross = sum((self._value(s, abs(p.quantity) * self.marks.get(s, ZERO))
                     for (_, s), p in self.positions.items() if p.quantity), ZERO)
        # Reserve conservative incremental gross exposure, grouping OCO peers.
        reserved = ZERO
        for owner in self.strategies:
            for symbol in self.instruments:
                q = self.positions.get((owner, symbol), Position()).quantity
                pending = [o for o in self._active_orders.values()
                           if o.strategy_id == owner and o.request.symbol == symbol and o.active and not o.terminal]
                if not pending:
                    continue  # With no pending orders, incremental exposure and fees are zero.
                def quantity(side):
                    groups = {}
                    for order in pending:
                        if order.request.side == side:
                            key = order.group_id or order.order_id
                            groups[key] = max(groups.get(key, 0), order.remaining)
                    return sum(groups.values())
                mark = self.marks.get(symbol, ZERO)
                price = max([mark] + [max(o.request.limit_price or mark, o.request.stop_price or mark) for o in pending])
                exposure = max(abs(q - quantity("SELL")), abs(q + quantity("BUY")))
                projected = self._value(symbol, exposure * price) if exposure else ZERO
                current = self._value(symbol, abs(q) * mark) if q else ZERO
                fees = sum((self._value(symbol, self.exchange.fee(o.remaining, o.filled_quantity)) for o in pending), ZERO)
                reserved += max(ZERO, projected - current) + fees
        leverage = self.limits.gross_leverage
        initial = gross / leverage
        maintenance = gross * (self.limits.maintenance_margin_rate
                               if self.limits.maintenance_margin_rate is not None else 1 / leverage)
        cash, equity = self.cash, self.equity
        buying_power = max(ZERO, equity * leverage - gross)
        return AccountState(self.time, self.base_currency, self.cash_balances, self.fx_rates,
                            cash, equity, "simulated_gross_leverage",
                            initial, maintenance, equity - initial, equity - maintenance,
                            buying_power, reserved, max(ZERO, buying_power - reserved))

    @property
    def account_positions(self) -> dict[str, int]:
        return {s: sum(p.quantity for (owner, sym), p in self.positions.items() if sym == s)
                for s in self.instruments}

    @property
    def equity(self) -> Decimal:
        return self.cash + sum((self._value(s, q * self.marks.get(s, ZERO)) for s, q in self.account_positions.items() if q), ZERO)

    def context(self, strategy_id: str) -> Context:
        assert self.time is not None
        history = {s: tuple(bars) for s, bars in self.history.items()}
        contract = self._registered_contracts[strategy_id]
        if contract is not None:
            if strategy_id in self._owner_history:
                history = {symbol: tuple(self._owner_history[strategy_id].get(symbol, ()))
                           for symbol in contract.available_observed_symbols}
            history = {s: tuple(b for b in bars if self._started_at is not None and b.end > self._started_at)
                       for s, bars in history.items()}
            for bar in self._warmup_history.get(strategy_id, ()):
                history[bar.symbol] = (*history.get(bar.symbol, ()), bar)
            history = {s: tuple(sorted(bars, key=lambda b: b.end)) for s, bars in history.items()}
        account_state = self.account_state()
        return self._subscriptions[strategy_id].filter_context(Context(self.time, strategy_id, self.cash, self.equity,
                       {s: p for (owner, s), p in self.positions.items() if owner == strategy_id},
                       self.account_positions, self.marks, self.mark_times,
                       tuple(self._orders_by_owner[strategy_id].values()),
                       history, account=account_state,
                       instrument_currencies={s: i.currency for s, i in self.instruments.items()},
                       prices={s: PriceObservation(p, self.mark_times[s], self.instruments[s].currency, "bar_close")
                               for s, p in self.marks.items() if s in self.mark_times}))

    def warmup_history(self, owner: str, bars) -> None:
        """Seed only this owner's declared prestart observations, without callbacks."""
        if self._used:
            raise RuntimeError("Warmup history must be set before runtime startup.")
        if owner not in self.strategies:
            raise ValueError("Warmup history requires a registered owner.")
        self._warmup_history[owner] = tuple(bars)

    def start(self, time: datetime) -> None:
        aware(time)
        if self._used:
            raise RuntimeError("Runtime is single-use; create fresh state for each run.")
        self._used = True
        self.time = time.astimezone(timezone.utc)
        self._started_at = self.time
        self._update_fx(self.time)
        self.initial_cash = self.cash
        if self.initial_cash <= 0:
            raise ValueError("Initial cash value must be positive.")
        self.record_equity()
        for owner in self.strategies:
            self._invoke(owner, None)
        self._drain()

    def stop(self) -> None:
        for owner, strategy in self.strategies.items():
            try:
                strategy.on_stop(self.context(owner))
            except Exception as exc:
                self._fail(owner, exc)

    def _fail(self, owner: str, exc: Exception) -> None:
        self.paused.add(owner)
        self.diagnostics.append(f"Strategy {owner} paused: {type(exc).__name__}: {exc}")

    def _invoke(self, owner: str, event: Event | None) -> None:
        if owner in self.paused:
            return
        try:
            strategy = self.strategies[owner]
            assert_contract_unchanged(strategy, self._registered_contracts[owner])
            if event is not None:
                event = self._subscriptions[owner].select(event)
                if event is None:
                    return
            context = self.context(owner)
            output = strategy.on_start(context) if event is None else dispatch_event(strategy, event, context)
            assert_contract_unchanged(strategy, self._registered_contracts[owner])
            # Queue and return forms are collected once and validated as one batch.
            commands = collect_commands(context, output)
            assert_contract_unchanged(strategy, self._registered_contracts[owner])
            validate_commands(strategy, commands)
            for command in commands:
                self.command(owner, command)
        except Exception as exc:
            self._fail(owner, exc)
            if isinstance(exc, ContractViolation):
                raise

    def _drain(self) -> None:
        count = 0
        while self._queue:
            count += 1
            if count > 10_000:
                raise RuntimeError("Strategy event feedback exceeded 10000 events in one step.")
            event, scoped_owners = self._queue.popleft()
            if isinstance(event, OrderUpdate):
                owners = (event.order.strategy_id,)
            elif isinstance(event, FillEvent):
                owners = (event.fill.strategy_id,)
            elif isinstance(event, Rejection):
                owners = (event.strategy_id,)
            else:
                owners = tuple(self.strategies)
            if scoped_owners is not None:
                owners = scoped_owners
            for owner in owners:
                self._invoke(owner, event)

    def emit(self, event: Event, owners: tuple[str, ...] | None = None) -> None:
        self.events.append(event)
        self._queue.append((event, owners))

    def apply(self, event: OrderUpdate | FillEvent) -> None:
        if not isinstance(event, (OrderUpdate, FillEvent)):
            raise TypeError("Execution events must be OrderUpdate or FillEvent.")
        aware(event.time)
        if self.time is None or event.time > self.time:
            raise ValueError("Execution event is ahead of the runtime clock.")
        current = None
        if isinstance(event, FillEvent):
            fill = event.fill
            previous = self._executions.get(fill.execution_id)
            if previous is not None:
                if previous != fill:
                    raise ValueError("Conflicting duplicate execution.")
                return
            order = self.orders[fill.order_id]
            if fill.available_at != event.time or fill.occurred_at < order.accepted_at:
                raise ValueError("Execution timestamps do not match event/order timing.")
            if (fill.strategy_id, fill.symbol, fill.side) != (order.strategy_id, order.request.symbol, order.request.side):
                raise ValueError("Execution attribution does not match order.")
            if fill.currency != self.instruments[fill.symbol].currency:
                raise ValueError("Execution currency does not match the instrument.")
            if fill.quantity <= 0 or fill.quantity > order.remaining or fill.price <= 0 or fill.fee < 0:
                raise ValueError("Invalid execution quantity/price/fee.")
            key = (fill.strategy_id, fill.symbol)
            p = self.positions.get(key, Position())
            delta = fill.quantity * (1 if fill.side == "BUY" else -1)
            quantity = p.quantity + delta
            realized = p.realized_pnl
            if p.quantity == 0 or (p.quantity > 0) == (delta > 0):
                cost = (abs(p.quantity) * p.average_cost + abs(delta) * fill.price) / abs(quantity)
            else:
                realized += min(abs(p.quantity), abs(delta)) * (fill.price - p.average_cost) * (1 if p.quantity > 0 else -1)
                cost = ZERO if quantity == 0 else (p.average_cost if (quantity > 0) == (p.quantity > 0) else fill.price)
            self.positions[key] = Position(quantity, cost, realized, self.instruments[fill.symbol].currency)
            code = self.instruments[fill.symbol].currency
            self._cash_change(code, -delta * fill.price - fill.fee)
            self.fees += self._value(fill.symbol, fill.fee)
            # The bar is published at its end, but market/gap fills occur at its
            # open. Correct borrow charged on the old position for that interval.
            short_delta = max(0, -quantity) - max(0, -p.quantity)
            duration = Decimal(str((fill.available_at - fill.occurred_at).total_seconds())) / Decimal(365 * 86400)
            reference = fill.price if short_delta > 0 else self.marks.get(fill.symbol, fill.price)
            correction = short_delta * reference * self.exchange.annual_borrow_rate * duration
            if correction:
                self._cash_change(code, -correction)
                self.borrow_cost += self._value(fill.symbol, correction)
                self.cash_flows.append(dict(time=fill.available_at, kind="borrow_fill_timing",
                                           amount=-correction, currency=code))
            self._executions[fill.execution_id] = fill
            self.fills.append(fill)
            filled = order.filled_quantity + fill.quantity
            self.orders[fill.order_id] = replace(order, filled_quantity=filled,
                status="filled" if filled == order.request.quantity else "partially_filled",
                cancel_pending=order.cancel_pending and filled != order.request.quantity)
        else:
            current = self.orders.get(event.order.order_id)
            incoming = event.order
            if incoming.strategy_id not in self.strategies or incoming.request.symbol not in self.instruments and incoming.status != "rejected":
                raise ValueError("Order update has an unknown strategy or instrument.")
            if incoming.accepted_at > event.time:
                raise ValueError("Order update precedes acceptance.")
            if current is not None:
                identity = ("strategy_id", "request", "accepted_at", "sequence", "expires_at", "revision", "group_id", "parent_order_id")
                amended = self._replacement_intents.get(incoming.order_id) == incoming
                if any(getattr(incoming, key) != getattr(current, key) for key in identity) and not amended:
                    raise ValueError("Order update cannot change the accepted intent without an authorized replacement.")
                if amended:
                    self._replacement_intents.pop(incoming.order_id, None)
                if incoming.filled_quantity != current.filled_quantity:
                    raise ValueError("Filled quantity changes require execution events.")
                if current.terminal and incoming.status != current.status:
                    raise ValueError("Order update cannot reopen or change a terminal order.")
            elif incoming.filled_quantity:
                raise ValueError("Unknown orders cannot have unaccounted executions.")
            self.orders[incoming.order_id] = incoming
        updated = self.orders[event.fill.order_id if isinstance(event, FillEvent) else event.order.order_id]
        self._orders_by_owner[updated.strategy_id][updated.order_id] = updated
        if updated.terminal:
            self._active_orders.pop(updated.order_id, None)
        else:
            self._active_orders[updated.order_id] = updated
        self.emit(event)
        if isinstance(event, OrderUpdate) and event.order.status == "rejected" and (current is None or current.status != "rejected"):
            self.emit(Rejection(event.time, event.order.strategy_id, event.order.reason or "Exchange rejected the order.", event.order.request))

    def _capacity(self, orders: list[Order], positions: dict[tuple[str, str], int],
                  cash: Decimal, marks: dict[str, Decimal]) -> str | None:
        orders = [order for order in orders if order.active]
        gross_by_owner: dict[str, Decimal] = {}
        fees = ZERO
        keys = set(positions) | {(o.strategy_id, o.request.symbol) for o in orders if not o.terminal}
        for owner, symbol in sorted(keys):
            q = positions.get((owner, symbol), 0)
            pending = [o for o in orders if not o.terminal and o.strategy_id == owner and o.request.symbol == symbol]
            def reserved(side):
                groups = {}
                for order in pending:
                    if order.request.side == side:
                        key = order.group_id or order.order_id
                        groups[key] = max(groups.get(key, 0), order.remaining)
                return sum(groups.values())
            low = q - reserved("SELL")
            high = q + reserved("BUY")
            if low < 0 and (not self.exchange.allow_short or not self.instruments[symbol].shortable):
                return f"Shorting unavailable for {symbol}."
            mark = marks.get(symbol, ZERO)
            price = max([mark] + [max(o.request.limit_price or mark, o.request.stop_price or mark)
                                 for o in pending])
            gross_by_owner[owner] = gross_by_owner.get(owner, ZERO) + (self._value(symbol, max(abs(low), abs(high)) * price) if (low or high) else ZERO)
            fees += sum((self._value(symbol, self.exchange.fee(o.remaining, o.filled_quantity)) for o in pending), ZERO)
        equity = cash + sum((self._value(s, q * marks.get(s, ZERO)) for (_, s), q in positions.items() if q), ZERO)
        for owner, cap in self.limits.strategy_gross_limits.items():
            if gross_by_owner.get(owner, ZERO) > cap:
                return f"Strategy {owner} exceeds its gross notional limit."
        if sum(gross_by_owner.values(), ZERO) + fees > max(ZERO, equity) * self.limits.gross_leverage:
            return "Account gross leverage/capacity exceeded (including pending orders and fees)."
        return None

    def fill_allowed(self, order: Order, quantity: int, price: Decimal, fee: Decimal,
                     occurred_at: datetime | None = None) -> bool:
        # Opening fills cannot borrow a later FX observation from the bar close.
        if occurred_at is None:
            return self._fill_allowed(order, quantity, price, fee)
        rates = self.fx_rates
        try:
            self.fx_rates = {}
            self._update_fx(occurred_at)
            return self._fill_allowed(order, quantity, price, fee)
        finally:
            self.fx_rates = rates

    def _fill_allowed(self, order: Order, quantity: int, price: Decimal, fee: Decimal) -> bool:
        positions = {k: p.quantity for k, p in self.positions.items()}
        key = (order.strategy_id, order.request.symbol)
        before = positions.get(key, 0)
        delta = quantity * (1 if order.request.side == "BUY" else -1)
        positions[key] = before + delta
        # Always permit strict reductions of an existing position at execution.
        if abs(before + delta) < abs(before) and (before + delta == 0 or (before > 0) == (before + delta > 0)):
            return True
        if self.limits.max_order_notional is not None and self._value(order.request.symbol, order.request.quantity * price) > self.limits.max_order_notional:
            return False
        changed = replace(order, filled_quantity=order.filled_quantity + quantity,
                          status="filled" if quantity == order.remaining else "partially_filled")
        # The exchange cancels OCO peers atomically on the first execution.
        # Project that same state here so a peer's reservation cannot reject a
        # valid winner (or force a later, more favorable peer to win instead).
        orders = [changed if o.order_id == order.order_id else o
                  for o in self._active_orders.values()
                  if o.order_id == order.order_id or not (
                      order.group_id is not None and o.group_id == order.group_id
                      and o.parent_order_id == order.parent_order_id)]
        marks = dict(self.marks, **{order.request.symbol: price})
        return self._capacity(orders, positions, self.cash - self._value(order.request.symbol, delta * price + fee), marks) is None

    def _candidate_reason(self, order: Order, orders: list[Order]) -> str | None:
        request = order.request
        instrument = self.instruments.get(request.symbol)
        if instrument is None:
            return "Unknown instrument."
        if request.symbol not in self.marks:
            return "No observed price yet; orders cannot use future bars."
        if any(price is not None and price % instrument.tick_size for price in (request.limit_price, request.stop_price)):
            return "Order price violates the instrument tick size."
        if request.time_in_force == "DAY" and order.expires_at is None:
            return "DAY orders require a future session close in the calendar."
        price = max(self.marks[request.symbol], request.limit_price or ZERO, request.stop_price or ZERO)
        if self.limits.max_order_notional is not None and self._value(request.symbol, request.quantity * price) > self.limits.max_order_notional:
            return "Order notional limit exceeded."
        return self._capacity(orders, {key: position.quantity for key, position in self.positions.items()}, self.cash, self.marks)

    def _replace_order(self, owner: str, command: ReplaceOrder) -> None:
        assert self.time is not None
        now = self.time
        def reject(reason):
            self.emit(Rejection(now, owner, reason, command))
        key = (owner, command.client_command_id)
        if command.client_command_id is not None and key in self._command_ids:
            if self._command_ids[key] != command:
                reject("Command ID already belongs to another replacement intent.")
            return
        order = self.orders.get(command.order_id)
        if order is None or order.strategy_id != owner or order.terminal:
            reject("Replace requires an owned working order.")
            return
        if order.cancel_pending or order.replace_pending:
            reject("Replace cannot race a pending cancel or replacement.")
            return
        quantity = command.quantity if command.quantity is not None else order.request.quantity
        if quantity <= order.filled_quantity:
            reject("Replacement total quantity must exceed executed quantity; cancel the remainder instead.")
            return
        bracket_member = order.parent_order_id is not None or any(item.parent_order_id == order.order_id for item in self.orders.values())
        if bracket_member and quantity != order.request.quantity:
            reject("Bracket quantity amendments require canceling and recreating the group; dependent exits cannot be resized independently.")
            return
        try:
            request = replace(order.request, quantity=quantity,
                              limit_price=command.limit_price if command.limit_price is not None else order.request.limit_price,
                              stop_price=command.stop_price if command.stop_price is not None else order.request.stop_price)
        except (TypeError, ValueError) as exc:
            reject(str(exc))
            return
        if request == order.request:
            if command.client_command_id is not None:
                self._command_ids[key] = command
            return
        self._sequence += 1
        updated = replace(order, request=request, accepted_at=self.time, sequence=self._sequence,
                          revision=order.revision + 1, replace_pending=False,
                          status="filled" if quantity == order.filled_quantity else ("partially_filled" if order.filled_quantity else "working"))
        candidates = [updated if item.order_id == order.order_id else item for item in self._active_orders.values()]
        reason = self._candidate_reason(updated, candidates)
        if reason:
            reject(reason)
            return
        amend = getattr(self.exchange, "replace", None)
        if not callable(amend):
            reject("Exchange does not support replacement.")
            return
        self._replacement_intents[order.order_id] = updated
        try:
            events = cast(list[OrderUpdate], amend(updated, self.time))
        except (ValueError, KeyError) as exc:
            self._replacement_intents.pop(order.order_id, None)
            reject(str(exc))
            return
        for event in events:
            self.apply(event)
        if command.client_command_id is not None:
            self._command_ids[key] = command

    def _submit_group(self, owner: str, command: BracketOrder | OCOOrders) -> None:
        from .orders import expand_orders
        assert self.time is not None
        group_id = f"{owner}:{command.client_order_id}"
        if group_id in self._group_intents:
            if self._group_intents[group_id] != command:
                self.emit(Rejection(self.time, owner, "Group ID already belongs to another intent.", command))
            return
        requests = expand_orders(command)
        orders = []
        parent_id = f"{owner}:{requests[0].client_order_id}" if isinstance(command, BracketOrder) else None
        for index, request in enumerate(requests):
            self._sequence += 1
            close_index = bisect_right(self.session_closes, self.time)
            expires = self.session_closes[close_index] if request.time_in_force == "DAY" and close_index < len(self.session_closes) else None
            orders.append(Order(f"{owner}:{request.client_order_id}", owner, request, self.time, self._sequence,
                                expires_at=expires, group_id=group_id,
                                parent_order_id=parent_id if parent_id is not None and index else None,
                                active=not (parent_id is not None and index)))
        if len({order.order_id for order in orders}) != len(orders) or any(order.order_id in self.orders for order in orders):
            self.emit(Rejection(self.time, owner, "Group member order IDs must be new and unique.", command))
            return
        candidates = list(self._active_orders.values()) + orders
        for order in orders:
            reason = self._candidate_reason(order, candidates)
            if reason:
                self.emit(Rejection(self.time, owner, f"Group rejected atomically: {reason}", command))
                return
        self._group_intents[group_id] = command
        # All identities, tick constraints and risk reservations validate before
        # the first child reaches the deterministic exchange.
        for order in orders:
            for event in self.exchange.submit(order):
                self.apply(event)

    def command(self, owner: str, command: Command) -> None:
        if self.time is None:
            raise RuntimeError("Start the runtime before issuing commands.")
        if owner not in self.strategies:
            raise ValueError("Commands require a registered strategy owner.")
        if not isinstance(command, COMMAND_TYPES):
            raise TypeError("Unsupported strategy command.")
        assert_contract_unchanged(self.strategies[owner], self._registered_contracts[owner])
        validate_commands(self.strategies[owner], (command,))
        if isinstance(command, (Subscribe, Unsubscribe)):
            self.emit(self._subscriptions[owner].apply(command, self.time), (owner,))
            return
        if isinstance(command, VerticalOrder):
            self.emit(Rejection(self.time, owner, "Option simulation is unsupported; use PaperRuntime with VerticalBroker.", command))
            return
        if owner in self.paused and isinstance(command, (SubmitOrder, ReplaceOrder, BracketOrder, OCOOrders)):
            self.emit(Rejection(self.time, owner, "Strategy is paused.", command))
            return
        if isinstance(command, CancelOrders):
            selected = tuple(order for order in self._orders_by_owner[owner].values()
                if not order.terminal and (command.symbol is None or order.request.symbol == command.symbol)
                and (command.side is None or order.request.side == command.side)
                and (command.group_id is None or order.group_id in (command.group_id, f"{owner}:{command.group_id}")))
            for order in selected:
                # An earlier parent cancellation may already have canceled this child.
                if not self.orders[order.order_id].terminal:
                    self.command(owner, CancelOrder(order.order_id))
            return
        if isinstance(command, ReplaceOrder):
            self._replace_order(owner, command)
            return
        if isinstance(command, (BracketOrder, OCOOrders)):
            self._submit_group(owner, command)
            return
        if isinstance(command, CancelOrder):
            order = self.orders.get(command.order_id)
            if order is None or order.strategy_id != owner or order.terminal:
                self.emit(Rejection(self.time, owner, "Cancel requires an owned working order.", command))
                return
            try:
                updates = self.exchange.cancel(command.order_id, self.time)
            except (KeyError, ValueError) as exc:
                self.emit(Rejection(self.time, owner, f"Cancellation rejected: {exc}", command))
                return
            for event in updates:
                self.apply(event)
            return
        if not isinstance(command, SubmitOrder):
            raise TypeError("This command requires the subscription runtime.")
        self._sequence += 1
        order_id = f"{owner}:{command.client_order_id}" if command.client_order_id else f"order-{self._sequence:06d}"
        if order_id in self.orders:
            if self.orders[order_id].request != command:
                self.emit(Rejection(self.time, owner, "Order ID already belongs to another intent.", command))
            return
        close_index = bisect_right(self.session_closes, self.time)
        expires = self.session_closes[close_index] if command.time_in_force == "DAY" and close_index < len(self.session_closes) else None
        order = Order(order_id, owner, command, self.time, self._sequence, expires_at=expires)
        reason = None
        instrument = self.instruments.get(command.symbol)
        if instrument is None:
            reason = "Unknown instrument."
        elif command.limit_price is not None and command.limit_price % instrument.tick_size:
            reason = "Limit price violates the instrument tick size."
        elif command.stop_price is not None and command.stop_price % instrument.tick_size:
            reason = "Stop price violates the instrument tick size."
        elif command.symbol not in self.marks:
            reason = "No observed price yet; orders cannot use future bars."
        elif command.time_in_force == "DAY" and expires is None:
            reason = "DAY orders require a future session close in the calendar."
        elif self.limits.max_order_notional is not None and self._value(command.symbol, command.quantity * max(self.marks[command.symbol], command.limit_price or ZERO, command.stop_price or ZERO)) > self.limits.max_order_notional:
            reason = "Order notional limit exceeded."
        else:
            reason = self._capacity(list(self._active_orders.values()) + [order],
                                    {k: p.quantity for k, p in self.positions.items()}, self.cash, self.marks)
            current = self.positions.get((owner, command.symbol), Position()).quantity
            target = current + command.quantity * (1 if command.side == "BUY" else -1)
            pending = any(not o.terminal and o.strategy_id == owner and o.request.symbol == command.symbol for o in self._active_orders.values())
            # Permit an unambiguous reduction even after adverse marks breach a
            # gross limit; reserve competing closing orders normally.
            if not pending and abs(target) < abs(current) and (target == 0 or (target > 0) == (current > 0)):
                reason = None
        if reason:
            self.apply(OrderUpdate(self.time, replace(order, status="rejected", reason=reason)))
        else:
            self.orders[order_id] = order
            self._active_orders[order_id] = order
            self._orders_by_owner[owner][order_id] = order
            for event in self.exchange.submit(order):
                self.apply(event)

    def advance(self, time: datetime, bars: Bars | None = None, timers: tuple[Timer, ...] = (),
                *, publish_bars: bool = True, observation: Bars | None = None,
                observations_by_owner: Mapping[str, Bars] | None = None,
                timer_owners: Mapping[Timer, tuple[str, ...]] | None = None) -> None:
        assert self.time is not None
        aware(time)
        time = time.astimezone(timezone.utc)
        if time < self.time:
            raise ValueError("Runtime clock cannot move backward.")
        if bars is not None:
            if bars.time != time:
                raise ValueError("Bar event time must match the runtime clock.")
            for symbol, bar in bars.bars.items():
                if symbol not in self.instruments:
                    raise ValueError("Bar contains an unknown instrument.")
                close_index = bisect_right(self.session_closes, bar.start)
                if close_index < len(self.session_closes) and self.session_closes[close_index] < bar.end:
                    raise ValueError("A bar crosses session close/order expiry; use finer bars.")
                previous = self.history[symbol]
                if bar.start < self.time or (previous and bar.start < previous[-1].end):
                    raise ValueError("Bars cannot overlap published history or elapsed runtime time.")
        if any(not isinstance(timer, Timer) or timer.time != time for timer in timers):
            raise ValueError("Timer event time must match the runtime clock.")
        if observation is not None:
            if publish_bars or bars is None or observation.time != time or set(observation.bars) != set(bars.bars):
                raise ValueError("Separate observations require matching execution bars and publish_bars=False.")
            for symbol, bar in observation.bars.items():
                previous = self.history[symbol]
                if (bar.close != bars.bars[symbol].close or bar.start > bars.bars[symbol].start
                        or (previous and bar.start < previous[-1].end)):
                    raise ValueError("Observation must end at the execution close without overlapping published history.")
        if observations_by_owner is not None:
            if publish_bars or observation is not None:
                raise ValueError("Owner observations require publish_bars=False and no shared observation.")
            for owner, panel in observations_by_owner.items():
                if owner not in self.strategies or panel.time != time:
                    raise ValueError("Owner observations require registered owners and the current event time.")
                for symbol, bar in panel.bars.items():
                    if symbol not in self.instruments:
                        raise ValueError("Observation contains an unknown instrument.")
                    previous = self._owner_history.get(owner, {}).get(symbol, ())
                    if previous and bar.start < previous[-1].end:
                        raise ValueError("Owner observations cannot overlap previously published history.")
        if timer_owners is not None and any(owner not in self.strategies for owners in timer_owners.values() for owner in owners):
            raise ValueError("Timer recipients must be registered strategy owners.")
        elapsed = Decimal(str((time - self.time).total_seconds())) / Decimal(365 * 86400)
        for (_, symbol), position in self.positions.items():
            if position.quantity < 0:
                borrow = -position.quantity * self.marks.get(symbol, ZERO) * self.exchange.annual_borrow_rate * elapsed
                if borrow:
                    code = self.instruments[symbol].currency
                    self._cash_change(code, -borrow)
                    self.borrow_cost += self._value(symbol, borrow)
                    self.cash_flows.append(dict(time=time, kind="borrow", amount=-borrow, currency=code))
        self.time = time
        self._update_fx(time)
        if bars is not None:
            self.exchange.process(bars, self)
        for event in self.exchange.expire(time):
            self.apply(event)
        # Fills update accounting before any strategy sees the completed bars.
        if bars is not None:
            for symbol, bar in bars.bars.items():
                self.marks[symbol] = bar.close
                self.mark_times[symbol] = time
                if publish_bars:
                    self.history[symbol].append(bar)
            if publish_bars:
                self.emit(bars)
        if observation is not None:
            for symbol, bar in observation.bars.items():
                self.history[symbol].append(bar)
            self.emit(observation)
        for owner, panel in (observations_by_owner or {}).items():
            history = self._owner_history.setdefault(owner, {})
            for symbol, bar in panel.bars.items():
                history.setdefault(symbol, []).append(bar)
            self.emit(panel, (owner,))
        for timer in timers:
            self.emit(timer, timer_owners.get(timer, ()) if timer_owners is not None else None)
        self._drain()
        self.record_equity()

    def record_equity(self) -> None:
        state = self.account_state()
        self.equity_history.append(dict(time=self.time, cash=state.cash, equity=state.equity,
                                       fees=self.fees, borrow_cost=self.borrow_cost,
                                       base_currency=self.base_currency, cash_balances=dict(self.cash_balances),
                                       initial_margin=state.initial_margin, maintenance_margin=state.maintenance_margin,
                                       available_funds=state.available_funds, excess_liquidity=state.excess_liquidity,
                                       buying_power=state.buying_power, reserved=state.reserved,
                                       sizing_power=state.sizing_power))
