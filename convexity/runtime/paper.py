"""Serial strategy bridge to an explicitly constructed stock/vertical paper broker.

Broker base-currency balances and currency ledgers are authoritative. This adapter never applies the simulated
cash, leverage or borrow ledger to an account containing other asset classes.
Market commands use an explicitly configured price collar and become limit
orders with the requested time in force; they can remain unfilled.
Call refresh() to consume broker updates.
"""
import asyncio
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
import math
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from convexity.brokers.ibkr import PaperAccountError, PaperAccountSnapshot, Quote, StockQuoteRequest
from convexity.brokers.ibkr_trading import PaperBroker
from convexity.brokers.ibkr_verticals import VerticalBroker
from convexity.trading import OrderRecord, StockOrder, VerticalOrder
from .account import broker_account
from .types import PriceObservation
from .contracts import assert_contract_unchanged, dispatch_event, filter_context, select_event, strategy_contract, validate_commands
from .subscriptions import SubscriptionBook
from .state import DeliveryError, DeliveryResult, StrategyState
from .capabilities import PAPER_CAPABILITIES
from .orders import expand_orders, amended_request
from .types import BracketOrder, OCOOrders, ReplaceOrder, CancelOrders, Command
from .types import (COMMAND_TYPES, Subscribe, SubscriptionUpdate, Unsubscribe, collect_commands, AccountUpdate, Bars, BrokerOrderUpdate, CancelOrder, ConnectionHealth, Context, Event, Fill, FillEvent, Instrument, OptionQuotes, Order, StockQuotes,
                    OrderUpdate, Position, Rejection, Strategy, SubmitOrder, Timer, ZERO, decimal)


class PaperRuntime:
    """Run shared Strategy callbacks against broker-authoritative state.

    Stock and vertical strategies use separate owners and dollar budgets.
    Vertical commands require VerticalBroker and publish leg-level journal updates.
    Every strategy requires a gross dollar budget. AvailableFunds is a second,
    conservative gate, not a reproduction of IB margin. Configured symbols must
    have no unassigned/manual positions. Other account assets remain unmanaged.
    Unknown outcomes, missing executions, stale quotes and attribution differences
    block submissions. Cancellation remains explicit; close never liquidates.

    Broker sessions are single-use. Existing strategy checkpoints receive unseen
    reconciled fills on restart. A first deployment baselines earlier executions
    without historical callbacks. Callback order IDs are deterministic;
    pass durable timer identities, and persist strategy decisions when resuming
    unfinished multi-step strategies is required.
    """
    capabilities = PAPER_CAPABILITIES

    def __init__(self, broker: PaperBroker, strategies: Mapping[str, Strategy],
                 instruments: Mapping[str, Instrument], *, budgets: Mapping[str, Decimal],
                 max_order_notional: Decimal, market_collar_bps: Decimal,
                 quote_max_age: float = 30, allow_short: bool = False,
                 acknowledgment_timeout: float = 15, defer_start: bool = False,
                 bar_max_age: float = 120, state_path: str | Path | None = None,
                 decision_guard: Callable[[], None] | None = None):
        if not strategies or any(not name for name in strategies):
            raise ValueError("Register named strategies.")
        if any(instrument.currency != "USD" for instrument in instruments.values()):
            raise ValueError("Paper order routing currently supports USD instruments only.")
        if any(name != instrument.symbol for name, instrument in instruments.items()):
            raise ValueError("Supply symbol-keyed stock instruments.")
        self.budgets = {name: decimal(value) for name, value in budgets.items()}
        self.max_order_notional = decimal(max_order_notional)
        self.market_collar_bps = decimal(market_collar_bps)
        if type(quote_max_age) not in (int, float) or not math.isfinite(quote_max_age) or quote_max_age <= 0:
            raise ValueError("quote_max_age must be finite and positive.")
        if (set(self.budgets) != set(strategies) or any(value <= 0 for value in self.budgets.values())
                or self.max_order_notional <= 0 or not 0 <= self.market_collar_bps < 1000
                or type(allow_short) is not bool):
            raise ValueError("Require positive budgets/order cap, a collar below 1000 bps, and bool short permission.")
        if (type(acknowledgment_timeout) not in (int, float) or not math.isfinite(acknowledgment_timeout)
                or not 0 < acknowledgment_timeout <= 300):
            raise ValueError("acknowledgment_timeout must be in (0, 300] seconds.")
        self.acknowledgment_timeout = acknowledgment_timeout
        if type(defer_start) is not bool or type(bar_max_age) not in (int, float) or not math.isfinite(bar_max_age) or bar_max_age <= 0:
            raise ValueError("defer_start must be bool and bar_max_age finite and positive.")
        self.defer_start, self.bar_max_age = defer_start, bar_max_age
        if decision_guard is not None and not callable(decision_guard):
            raise ValueError("decision_guard must be callable.")
        self.decision_guard = decision_guard
        self.state = StrategyState(state_path if state_path is not None else broker.journal.path, broker.config.account)
        self._connected = False
        self.broker, self.strategies, self.instruments = broker, dict(strategies), dict(instruments)
        self._registered_contracts = {owner: strategy_contract(strategy) for owner, strategy in self.strategies.items()}
        self._subscriptions = {owner: SubscriptionBook(contract, owner, self.state.load_subscriptions(owner))
                               for owner, contract in self._registered_contracts.items()}
        self.quote_max_age, self.allow_short = quote_max_age, allow_short
        self.quotes: dict[str, Quote] = {}
        self.history = {symbol: [] for symbol in instruments}
        self.snapshot: PaperAccountSnapshot | None = None
        self.time = datetime.now(timezone.utc)
        self.paused: set[str] = set()
        self.errors: list[str] = []
        self.events: list[Event] = []
        self._queue: deque[OrderRecord] = deque()
        self._rejections: deque[Rejection] = deque()
        self._subscription_updates: deque[tuple[SubscriptionUpdate, str]] = deque()
        self._seen_orders: dict[str, Order] = {}
        self._seen_broker_orders: dict[str, OrderRecord] = {}
        self._started = False
        self._started_at = None
        self._warmup_history = {}
        self._seen_executions: set[str] = set()
        self._lock = asyncio.Lock()
        self._active = False
        self._used = False
        self._previous_handler = broker.on_update

    async def __aenter__(self):
        if self._used:
            raise RuntimeError("PaperRuntime is single-use.")
        self._used = True
        def changed(record):
            self._queue.append(record)
            if self._previous_handler is not None:
                self._previous_handler(record)
        self.broker.on_update = changed
        try:
            await self.broker.__aenter__()
            self._active = True
            self._connected = True
            await self.refresh()
            self._queue.clear()
            for owner, strategy in self.strategies.items():
                try:
                    restored = self.state.has_checkpoint(owner)
                    strategy.load_state(self.state.load(owner, self._strategy_kind(owner)))
                    if self.state.unresolved(owner):
                        raise DeliveryError("Interrupted strategy callback requires reconciliation before restarting.")
                    records = tuple(record for record in self._broker_records() if record.request.strategy_id == owner)
                    if restored:
                        # Reconciliation can discover fills while disconnected or
                        # finish commissions without producing a new on_update.
                        # Callback claims suppress already delivered executions.
                        for record in records:
                            consumed = self.state.consumed(owner, self._record_identity(record))
                            if consumed:
                                if isinstance(record.request, StockOrder):
                                    order = self._order(record)
                                    self._seen_orders[order.order_id] = order
                                else:
                                    self._seen_broker_orders[record.request.client_order_id] = record
                            unseen_fill = False
                            for fill in record.executions:
                                if self.state.consumed(owner, f"fill:{fill.execution_id}"):
                                    self._seen_executions.add(fill.execution_id)
                                elif fill.commission is not None and isinstance(record.request, StockOrder):
                                    unseen_fill = True
                            if not consumed or unseen_fill:
                                self._queue.append(record)
                    else:
                        fills = tuple(fill.execution_id for record in records for fill in record.executions)
                        self.state.baseline(owner, (f"fill:{identity}" for identity in fills))
                        self.state.baseline(owner, (self._record_identity(record) for record in records))
                        self._seen_executions.update(fills)
                        self._seen_orders.update((self._order(record).order_id, self._order(record))
                                                 for record in records if isinstance(record.request, StockOrder))
                        self._seen_broker_orders.update((record.request.client_order_id, record) for record in records)
                except Exception as error:
                    self._fail(owner, error)
            if not self.defer_start:
                await self.start()
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, *exc):
        if self._active and self._started:
            for owner, strategy in self.strategies.items():
                try:
                    strategy.on_stop(self.context(owner))
                    if owner not in self.paused and not self.state.unresolved(owner):
                        self.state.save(owner, self._strategy_kind(owner), strategy.dump_state())
                except Exception as error:
                    self._fail(owner, error)
        self._active = False
        self._connected = False
        self.broker.on_update = self._previous_handler
        try:
            await self.broker.__aexit__(*exc)
        finally:
            self.state.close()

    def _strategy_kind(self, owner: str) -> str:
        strategy = self.strategies[owner]
        return type(strategy).__module__ + "." + type(strategy).__qualname__

    def delivery_result(self, owners: Iterable[str] | None = None) -> DeliveryResult:
        """Current callback health, including failures retained across restart."""
        selected = self._recipients(owners)
        prefixes = tuple(prefix for owner in selected for prefix in (f"Strategy {owner} paused:", f"{owner}:"))
        return DeliveryResult(tuple(sorted(self.paused.intersection(selected))),
                              tuple(error for error in self.errors if error.startswith(prefixes)))

    def _recipients(self, owners: Iterable[str] | None) -> tuple[str, ...]:
        if isinstance(owners, str):
            raise ValueError("Supply a nonempty iterable of registered strategy owners.")
        selected = tuple(self.strategies) if owners is None else tuple(owners)
        if not selected or len(selected) != len(set(selected)) or any(owner not in self.strategies for owner in selected):
            raise ValueError("Supply distinct registered strategy owners.")
        return selected

    async def start(self) -> DeliveryResult:
        """Enable strategy callbacks after optional command-free history warmup."""
        async with self._lock:
            if not self._active:
                raise RuntimeError("Enter PaperRuntime before starting it.")
            if self._started:
                return self.delivery_result()
            self._started = True
            self.time = max(self.time, datetime.now(timezone.utc))
            self._started_at = self.time
            for owner in self.strategies:
                await self._invoke(owner, None, "startup")
            await self._drain()
            return self.delivery_result()

    def warmup_history(self, owner: str, bars) -> None:
        """Seed a strategy's exact declared observations before its startup."""
        if self._started:
            raise RuntimeError("Warmup history must precede startup.")
        if owner not in self.strategies:
            raise ValueError("Warmup history requires a registered owner.")
        self._warmup_history[owner] = tuple(bars)

    async def warmup(self, events) -> None:
        """Load historical bars before start without invoking any strategy callbacks."""
        if self._started:
            raise RuntimeError("Warmup must precede start; use ingest_history for validated backfills.")
        await self.ingest_history(events)

    async def ingest_history(self, events) -> None:
        """Append observed history without callbacks; never enable trading from a backfill.

        Sources/runners validate session continuity before passing backfills.
        This method validates identity, observation time and overlap atomically.
        """
        async with self._lock:
            if not self._active:
                raise RuntimeError("Enter PaperRuntime before loading history.")
            staged = {symbol: list(bars) for symbol, bars in self.history.items()}
            for event in events:
                if not isinstance(event, Bars) or event.time > datetime.now(timezone.utc):
                    raise ValueError("History must contain observed Bars.")
                for symbol, bar in event.bars.items():
                    if symbol not in staged or (staged[symbol] and bar.start < staged[symbol][-1].end):
                        raise ValueError("History must use configured symbols without overlapping bars.")
                    staged[symbol].append(bar)
            self.history = staged

    def _broker_records(self) -> tuple[OrderRecord, ...]:
        return tuple(record for record in self.broker.journal.orders()
                     if record.account == self.broker.config.account and record.request.strategy_id in self.strategies)

    def _record_identity(self, record: OrderRecord) -> str:
        if isinstance(record.request, StockOrder):
            order = self._order(record)
            return f"order:{order.order_id}:{order.status}:{order.filled_quantity}:{order.cancel_pending}:{order.revision}:{order.replace_pending}:{order.active}"
        # Include commission/cancellation changes, which can arrive separately
        # from leg executions and must reach options strategies after reconnect.
        payload = json.dumps(asdict(record), default=str, sort_keys=True)
        return f"broker:{record.request.client_order_id}:" + hashlib.sha256(payload.encode()).hexdigest()

    def _records(self) -> tuple[OrderRecord, ...]:
        return tuple(record for record in self.broker.journal.orders()
                     if record.account == self.broker.config.account and isinstance(record.request, StockOrder)
                     and record.request.strategy_id in self.strategies and record.request.symbol in self.instruments)

    def _balance(self, tag: str) -> Decimal:
        assert self.snapshot is not None
        try:
            account = broker_account(self.snapshot)
            fields = {"TotalCashValue": account.cash, "NetLiquidation": account.equity,
                      "AvailableFunds": account.available_funds}
            value = fields[tag]
            if value is None:
                raise ValueError(f"Broker {tag} is unavailable.")
            # Existing order budgets and USD order gates remain denominated in USD.
            return account.convert(value, account.base_currency, "USD")
        except ValueError as exc:
            raise PaperAccountError(str(exc)) from exc

    def _reserved_notional(self, records=None) -> Decimal:
        reserves = {}
        for record in (self.broker.journal.orders() if records is None else records):
            if record.account != self.broker.config.account or record.terminal:
                continue
            if isinstance(record.request, StockOrder) and not self._order(record).active:
                continue
            price = record.request.limit_price if isinstance(record.request, VerticalOrder) else record.request.risk_price
            amount = record.remaining_quantity * price * (100 if isinstance(record.request, VerticalOrder) else 1)
            key = record.group_id or record.request.client_order_id
            reserves[key] = max(reserves.get(key, ZERO), amount)
        return sum(reserves.values(), ZERO)

    def _order(self, record: OrderRecord) -> Order:
        request = record.request
        assert isinstance(request, StockOrder)
        if record.filled_quantity != record.filled_quantity.to_integral_value():
            raise PaperAccountError("Fractional executions cannot enter the whole-share runtime.")
        first = self.broker.journal.events(request.client_order_id)[0]["recorded_at"]
        accepted = datetime.fromisoformat(first)
        state = {"Filled": "filled", "Cancelled": "canceled", "ApiCancelled": "canceled", "Rejected": "rejected"}.get(record.status)
        state = state or ("partially_filled" if record.filled_quantity else "working")
        metadata = self.broker.journal.runtime_intent(request.client_order_id)
        original = (SubmitOrder(**json.loads(metadata["command"])) if metadata else
                    SubmitOrder(request.symbol, request.side, request.quantity, request.order_type, request.limit_price,
                                request.time_in_force, request.client_order_id, request.stop_price))
        if record.revision:
            original = replace(original, quantity=request.quantity, limit_price=request.limit_price if original.order_type != 'market' else None,
                               stop_price=request.stop_price)
        parent = self.broker.journal.get(record.parent_order_id) if record.parent_order_id else None
        parent_metadata = self.broker.journal.runtime_intent(record.parent_order_id) if record.parent_order_id else None
        group = f"{request.strategy_id}:{original.client_order_id.rsplit(':', 1)[0]}" if record.group_id and original.client_order_id else record.group_id
        return Order(metadata["public_id"] if metadata else request.client_order_id, request.strategy_id,
                     original,
                     accepted, record.broker_order_id, state, int(record.filled_quantity), record.cancel_pending,
                     revision=record.revision, replace_pending=record.replace_pending, group_id=group,
                     parent_order_id=parent_metadata['public_id'] if parent_metadata else record.parent_order_id,
                     active=parent is None or parent.filled_quantity == parent.request.quantity)

    def _positions(self) -> dict[tuple[str, str], Position]:
        positions: dict[tuple[str, str], Position] = {}
        executions = sorted(((fill.occurred_at, fill.execution_id, record.request, fill)
                             for record in self._records() for fill in record.executions), key=lambda row: row[:2])
        for _, _, request, fill in executions:
            if fill.quantity != fill.quantity.to_integral_value():
                raise PaperAccountError("Fractional executions cannot enter the whole-share runtime.")
            key = request.strategy_id, request.symbol
            old = positions.get(key, Position())
            delta = int(fill.quantity) * (1 if fill.side == "BUY" else -1)
            quantity = old.quantity + delta
            realized = old.realized_pnl
            if not old.quantity or (old.quantity > 0) == (delta > 0):
                average = (abs(old.quantity) * old.average_cost + abs(delta) * fill.price) / abs(quantity)
            else:
                realized += min(abs(old.quantity), abs(delta)) * (fill.price - old.average_cost) * (1 if old.quantity > 0 else -1)
                average = ZERO if not quantity else old.average_cost if (quantity > 0) == (old.quantity > 0) else fill.price
            positions[key] = Position(quantity, average, realized)
        return positions

    def context(self, owner: str) -> Context:
        if owner not in self.strategies or self.snapshot is None:
            raise ValueError("A started runtime and registered strategy are required.")
        positions = self._positions()
        account = {}
        for symbol, quote in self.quotes.items():
            quantity = sum((p.quantity for p in self.snapshot.positions if p.contract.contract_id == quote.contract.contract_id), ZERO)
            if quantity != quantity.to_integral_value():
                raise PaperAccountError("Configured stocks require whole-share account positions.")
            account[symbol] = int(quantity)
        marks = {symbol: (quote.bid + quote.ask) / 2 for symbol, quote in self.quotes.items()
                 if quote.bid is not None and quote.ask is not None}
        history = {symbol: tuple(bars) for symbol, bars in self.history.items()}
        contract = strategy_contract(self.strategies[owner])
        if contract is not None and (owner in self._warmup_history or contract.warmup_sessions == 0):
            history = {symbol: tuple(bar for bar in bars if self._started_at is not None and bar.end > self._started_at)
                       for symbol, bars in history.items()}
            for bar in self._warmup_history.get(owner, ()):
                history[bar.symbol] = (*history.get(bar.symbol, ()), bar)
            history = {symbol: tuple(sorted(bars, key=lambda bar: bar.end)) for symbol, bars in history.items()}
        state = broker_account(self.snapshot)
        reserved = state.convert(self._reserved_notional(), "USD")
        limits = [v for v in (state.available_funds, state.buying_power) if v is not None]
        state = replace(state, reserved=reserved,
                        sizing_power=max(ZERO, min(limits) - reserved) if limits else None)
        def observed(symbol, value, stamp):
            quote = self.quotes[symbol]
            return PriceObservation(value, stamp or quote.completed_at, quote.contract.currency,
                                    "broker_quote", self._quote_valid(quote))
        prices = {s: observed(s, p, min(q.bid_received_at, q.ask_received_at)
                             if q.bid_received_at and q.ask_received_at else None)
                  for s, p in marks.items() for q in (self.quotes[s],)}
        asks = {s: observed(s, q.ask, q.ask_received_at) for s, q in self.quotes.items() if q.ask is not None and q.ask > 0}
        bids = {s: observed(s, q.bid, q.bid_received_at) for s, q in self.quotes.items() if q.bid is not None and q.bid > 0}
        return self._subscriptions[owner].filter_context(Context(self.time, owner, state.cash, state.equity,
            {symbol: position for (strategy, symbol), position in positions.items() if strategy == owner}, account,
            marks, {symbol: quote.completed_at for symbol, quote in self.quotes.items()},
            tuple(self._order(record) for record in self._records() if record.request.strategy_id == owner),
            history,
            tuple(record for record in self._broker_records() if record.request.strategy_id == owner),
            account=state, instrument_currencies={s: i.currency for s, i in self.instruments.items()},
            prices=prices, ask_prices=asks, bid_prices=bids))

    def _quote_valid(self, quote: Quote) -> bool:
        now = datetime.now(timezone.utc)
        return (quote.status == "received" and quote.bid is not None and quote.ask is not None
                and quote.bid > 0 and quote.ask >= quote.bid
                and all(receipt is not None and 0 <= (now - receipt).total_seconds() <= self.quote_max_age
                        for receipt in (quote.bid_received_at, quote.ask_received_at)))

    async def _refresh(self):
        if not self._active:
            raise RuntimeError("Enter PaperRuntime before using it.")
        for symbol in self.instruments:
            cached = self.quotes.get(symbol)
            if cached is not None and self._quote_valid(cached):
                continue
            quote = await self.broker.quote(StockQuoteRequest(symbol, max_age=self.quote_max_age))
            if quote.contract.symbol != symbol or quote.contract.security_type != "STK" or quote.contract.currency != "USD":
                raise PaperAccountError("Quote identity does not match the configured USD stock.")
            self.quotes[symbol] = quote
        self.snapshot = await self.broker.snapshot()
        self.time = datetime.now(timezone.utc)
        for tag in ("TotalCashValue", "NetLiquidation", "AvailableFunds"):
            self._balance(tag)

    async def refresh(self):
        """Refresh broker balances/quotes and deliver queued order/fill events."""
        async with self._lock:
            await self._refresh()
            if self._started:
                await self._drain()
                assert self.snapshot is not None
                for event in (StockQuotes(self.time, self.quotes), AccountUpdate(self.time, self.snapshot)):
                    self.events.append(event)
                    for owner in self.strategies:
                        await self._invoke(owner, event, f"{type(event).__name__}:{self.time.isoformat()}")
                await self._drain()
            return self.delivery_result()

    async def step(self, event: Bars | Timer | OptionQuotes | StockQuotes | AccountUpdate | ConnectionHealth,
                   *, owners: Iterable[str] | None = None):
        """Deliver observations or a due timer to explicit owners (default all).

        Delivery health and timer failure reporting cover the selected owners.
        Broker tracking and global observations remain account-wide, so paused
        owners never prevent healthy strategies from receiving scoped timers.
        """
        selected = self._recipients(owners)
        async with self._lock:
            if not self._started:
                raise RuntimeError("Call start before delivering strategy events.")
            if not isinstance(event, (Bars, Timer, OptionQuotes, StockQuotes, AccountUpdate, ConnectionHealth)) or event.time > datetime.now(timezone.utc):
                raise ValueError("Supply observed bars or a due timer.")
            if isinstance(event, Bars):
                if (datetime.now(timezone.utc) - event.time).total_seconds() > self.bar_max_age:
                    raise ValueError("Stale Bars cannot drive paper decisions; load historical data through warmup/ingest_history.")
                for symbol, bar in event.bars.items():
                    if symbol not in self.history or (self.history[symbol] and bar.start < self.history[symbol][-1].end):
                        raise ValueError("Bars must belong to configured stocks without overlapping history.")
            if isinstance(event, ConnectionHealth):
                self._connected = event.connected
            if self._connected:
                await self._refresh()
                await self._drain()
            if isinstance(event, Bars):
                for symbol, bar in event.bars.items():
                    self.history[symbol].append(bar)
            self.events.append(event)
            identity = f"{type(event).__name__}:{event.time.isoformat()}:{event.name if isinstance(event, Timer) else ''}"
            for owner in selected:
                await self._invoke(owner, event, identity)
            await self._drain()
            result = self.delivery_result(selected)
            # Raising is essential: a schedule adapter may discard our return value.
            if isinstance(event, Timer):
                result.raise_for_failure()
            return result

    def _fail(self, owner, error):
        self.paused.add(owner)
        self.errors.append(f"Strategy {owner} paused: {type(error).__name__}: {error}")

    async def _invoke(self, owner, event, identity):
        if owner in self.paused:
            return
        assert_contract_unchanged(self.strategies[owner], self._registered_contracts[owner])
        if event is not None:
            event = self._subscriptions[owner].select(event)
            if event is None:
                return
        if self.decision_guard is not None and not isinstance(event, SubscriptionUpdate) and not (isinstance(event, ConnectionHealth) and not event.connected):
            self.decision_guard()
        try:
            if not self.state.claim(owner, identity):
                return
            strategy = self.strategies[owner]
            context = self.context(owner)
            output = strategy.on_start(context) if event is None else dispatch_event(strategy, event, context)
            commands = collect_commands(context, output)
            assert_contract_unchanged(strategy, self._registered_contracts[owner])
            validate_commands(strategy, commands)
            self.state.save(owner, self._strategy_kind(owner), strategy.dump_state())
            if commands and not self._connected:
                raise DeliveryError("Trading commands are disabled while disconnected.")
            for index, command in enumerate(commands):
                await self._command(owner, command, identity, index)
                if any(record.replace_pending or (record.uncertain and not record.terminal) or record.filled_quantity > record.executed_quantity
                       for record in self._broker_records() if record.request.strategy_id == owner):
                    raise DeliveryError("A submitted command has an unresolved broker outcome; reconcile before continuing.")
            self.state.complete(owner, identity)
        except Exception as error:
            self.state.fail(owner, identity, f"{type(error).__name__}: {error}")
            self._fail(owner, error)

    async def command(self, owner: str, command: Command) -> tuple[Event, ...]:
        """Issue an explicit operator command, including exits for paused owners.

        Submissions require stable client IDs. Paused owners may cancel or reduce
        reconciled exposure but cannot open/increase positions. This does not
        resume the strategy or resolve an interrupted callback.
        """
        if owner not in self.strategies:
            raise ValueError("Commands require a registered strategy owner.")
        if not isinstance(command, COMMAND_TYPES):
            raise TypeError("Supply a submit/cancel command.")
        assert_contract_unchanged(self.strategies[owner], self._registered_contracts[owner])
        validate_commands(self.strategies[owner], (command,))
        if isinstance(command, SubmitOrder) and not command.client_order_id:
            raise ValueError("Operator submissions require a stable client_order_id.")
        if isinstance(command, ReplaceOrder) and not command.client_command_id:
            command = replace(command, client_command_id='operator-' + uuid4().hex)
        async with self._lock:
            if not self._active or not self._connected:
                raise DeliveryError("Operator commands require a connected paper runtime.")
            if not isinstance(command, (CancelOrder, CancelOrders, Subscribe, Unsubscribe)):
                await self._refresh()
            first = len(self.events)
            if owner in self.paused and not isinstance(command, (CancelOrder, CancelOrders, Subscribe, Unsubscribe)):
                context = self.context(owner)
                if isinstance(command, SubmitOrder):
                    current = context.position(command.symbol)
                    target = current + (command.quantity if command.side == "BUY" else -command.quantity)
                    reduction = (abs(target) < abs(current) and (target == 0 or (target > 0) == (current > 0))
                                 and not any(not order.terminal and order.request.symbol == command.symbol for order in context.orders))
                elif isinstance(command, VerticalOrder):
                    positions = [self.broker.journal.attributed_position(self.broker.config.account, owner, leg.contract_id)
                                 for leg in (command.long_leg, command.short_leg)]
                    reduction = (command.side == "SELL" and positions[0] >= command.quantity and positions[1] <= -command.quantity
                                 and not any(not order.terminal for order in context.broker_orders))
                else:
                    reduction = False
                if not reduction:
                    event = Rejection(self.time, owner, "Paused strategies permit only cancellation or unambiguous position reduction.", command)
                    self.events.append(event)
                    return (event,)
            await self._command(owner, command, "operator", 0)
            await self._drain()
            return tuple(self.events[first:])

    def _check_decision_data(self, owner, command, identity):
        if self.decision_guard is None:
            return
        if identity == "operator":
            if isinstance(command, VerticalOrder) and command.side == "SELL":
                # The broker validates balanced ownership and closing quantity.
                return
            if isinstance(command, SubmitOrder):
                context = self.context(owner)
                current = context.position(command.symbol)
                delta = command.quantity if command.side == "BUY" else -command.quantity
                if (current * delta < 0 and abs(delta) <= abs(current)
                        and not any(not order.terminal and order.request.symbol == command.symbol for order in context.orders)):
                    return
        self.decision_guard()

    async def _command(self, owner, command, identity, index):
        assert_contract_unchanged(self.strategies[owner], self._registered_contracts[owner])
        validate_commands(self.strategies[owner], (command,))
        if isinstance(command, (Subscribe, Unsubscribe)):
            self.time = max(self.time, datetime.now(timezone.utc))
            event = self._subscriptions[owner].apply(command, self.time)
            self.state.save_subscriptions(owner, self._subscriptions[owner].dump())
            self.events.append(event)
            self._subscription_updates.append((event, f"subscription:{identity}:{index}:{event.time.isoformat()}"))
            return
        if isinstance(command, CancelOrders):
            selected = [order for order in self.context(owner).orders if not order.terminal
                        and (command.symbol is None or order.request.symbol == command.symbol)
                        and (command.side is None or order.request.side == command.side)
                        and (command.group_id is None or order.group_id in (command.group_id, f'{owner}:{command.group_id}'))]
            for offset, order in enumerate(selected):
                await self._command(owner, CancelOrder(order.order_id), identity, index + offset)
            return
        if isinstance(command, (ReplaceOrder, BracketOrder, OCOOrders)):
            try:
                if isinstance(command, ReplaceOrder):
                    await self._replace_stock(owner, command, identity, index)
                else:
                    await self._group_stock(owner, command, identity)
            except (KeyError, PaperAccountError, ValueError) as error:
                event = Rejection(self.time, owner, str(error), command)
                self.events.append(event)
                self._rejections.append(event)
                self.errors.append(f'{owner}: {error}')
            return
        if isinstance(command, SubmitOrder) and command.time_in_force in ("IOC", "FOK"):
            event = Rejection(self.time, owner, "Paper broker adapter supports DAY/GTC only; IOC/FOK cannot be routed.", command)
            self.events.append(event)
            self._rejections.append(event)
            return
        if isinstance(command, CancelOrder):
            local_id = self.broker.journal.broker_order_id(self.broker.config.account, command.order_id)
            try:
                record = self.broker.journal.get(local_id)
                if record.request.strategy_id != owner or record.account != self.broker.config.account or record.terminal:
                    raise ValueError("Cancel requires an owned working order.")
                await self.broker.cancel(local_id)
            except (KeyError, ValueError, PaperAccountError) as error:
                reason = "Cancel requires an owned working order." if isinstance(error, KeyError) else str(error)
                event = Rejection(self.time, owner, reason, command)
                self.events.append(event)
                self._rejections.append(event)
            return
        if isinstance(command, VerticalOrder):
            try:
                if command.strategy_id != owner:
                    raise PaperAccountError("Vertical strategy identity must match its callback owner.")
                if not isinstance(self.broker, VerticalBroker):
                    raise PaperAccountError("Vertical commands require a VerticalBroker.")
                budget = self.broker.strategy_budgets.get(owner)
                if budget is None or budget > self.budgets[owner]:
                    raise PaperAccountError("VerticalBroker requires a strategy budget within the runtime allocation.")
                context = self.context(owner)
                if command.side == "BUY" and (any(position.quantity for position in context.positions.values())
                        or any(not order.terminal for order in context.orders)):
                    raise PaperAccountError("Use separate strategy budgets for concurrent stock and option exposures.")
                if command.quantity * command.limit_price * 100 > self.max_order_notional:
                    raise PaperAccountError("Vertical order dollar limit exceeded.")
                self._check_decision_data(owner, command, identity)
                record = await self.broker.submit(command,
                    submission_guard=lambda: self._check_decision_data(owner, command, identity))
                await self._await_submission(record)
            except (PaperAccountError, ValueError) as error:
                event = Rejection(datetime.now(timezone.utc), owner, str(error), command)
                self.events.append(event)
                self._rejections.append(event)
                self.errors.append(f"{owner}: {error}")
            return
        token = (owner, "client", command.client_order_id) if command.client_order_id else (owner, identity, index)
        local_id = "rt-" + hashlib.sha256(json.dumps(token).encode()).hexdigest()[:40]
        public_id = f"{owner}:{command.client_order_id}" if command.client_order_id else local_id
        payload = self._command_payload(command)
        try:
            try:
                existing = self.broker.journal.get(local_id)
            except KeyError:
                existing = None
            metadata = self.broker.journal.runtime_intent(local_id)
            if metadata is not None and (metadata["command"] != payload or metadata["account"] != self.broker.config.account):
                raise ValueError("Durable order ID already belongs to a different intent/account.")
            if existing is not None:
                if metadata is None:
                    raise ValueError("Legacy order lacks original command metadata; reconcile before reusing its ID.")
                if existing.account != self.broker.config.account:
                    raise ValueError("Durable order ID already belongs to a different account.")
                await self._await_submission(existing)
                return
            if command.symbol not in self.instruments:
                raise PaperAccountError("Unknown configured stock.")
            quote = await self.broker.quote(StockQuoteRequest(command.symbol, max_age=self.quote_max_age))
            expected = self.quotes.get(command.symbol)
            if (quote.contract.symbol != command.symbol or quote.contract.security_type != "STK"
                    or quote.contract.currency != "USD" or expected is None
                    or quote.contract.contract_id != expected.contract.contract_id):
                raise PaperAccountError("Quote identity changed; reconcile before submitting.")
            self.quotes[command.symbol] = quote
            self.snapshot = await self.broker.snapshot()
            self.time = datetime.now(timezone.utc)
            if not self._quote_valid(quote):
                raise PaperAccountError("A fresh, complete live stock quote is required.")
            instrument = self.instruments[command.symbol]
            limit = command.limit_price
            if limit is None and command.order_type == 'market':
                assert quote.ask is not None and quote.bid is not None
                raw = quote.ask * (1 + self.market_collar_bps / 10000) if command.side == "BUY" else quote.bid * (1 - self.market_collar_bps / 10000)
                limit = (raw / instrument.tick_size).to_integral_value(rounding=ROUND_FLOOR if command.side == "BUY" else ROUND_CEILING) * instrument.tick_size
            risk_price = max(limit or ZERO, command.stop_price or ZERO)
            if risk_price <= 0 or any(price is not None and price % instrument.tick_size for price in (limit, command.stop_price)):
                raise PaperAccountError("Order limit must follow the configured stock tick.")
            self._risk(owner, command, risk_price)
            self._check_decision_data(owner, command, identity)
            self.broker.journal.record_runtime_intent(local_id, self.broker.config.account, public_id, payload)
            def transmit_guard():
                self._check_decision_data(owner, command, identity)
                if not self._quote_valid(quote):
                    raise PaperAccountError("Execution quote expired during broker preflight; no order sent.")
                self._risk(owner, command, risk_price)
            record = await self.broker.submit(StockOrder(local_id, owner, command.symbol, command.side, command.quantity,
                                                       limit, time_in_force=cast(Literal["DAY", "GTC"], command.time_in_force),
                                                       order_type=cast(Literal['limit', 'stop', 'stop_limit'], 'limit' if command.order_type == 'market' else command.order_type),
                                                       stop_price=command.stop_price),
                                              submission_guard=transmit_guard)
            await self._await_submission(record)
        except (PaperAccountError, ValueError) as error:
            event = Rejection(self.time, owner, str(error), command)
            self.events.append(event)
            self._rejections.append(event)
            self.errors.append(f"{owner}: {error}")

    async def _replace_stock(self, owner, command, identity, index):
        local_id = self.broker.journal.broker_order_id(self.broker.config.account, command.order_id)
        record = self.broker.journal.get(local_id)
        if record.account != self.broker.config.account or record.request.strategy_id != owner or not isinstance(record.request, StockOrder):
            raise ValueError('Replacement requires an owned stock order.')
        command_id = command.client_command_id or hashlib.sha256(json.dumps((owner, identity, index)).encode()).hexdigest()
        previous = self.broker.journal.replacement_command(local_id, command_id)
        if previous is not None:
            desired = json.loads(previous['request'])
            for field in ('quantity', 'limit_price', 'stop_price'):
                supplied = getattr(command, field)
                if supplied is not None and (desired[field] is None or Decimal(str(supplied)) != Decimal(str(desired[field]))):
                    raise ValueError('Replacement command ID already belongs to different terms.')
            await self._await_submission(record)
            return
        order = self._order(record)
        amended = amended_request(order.request, command)
        request = replace(record.request, quantity=amended.quantity,
                          limit_price=amended.limit_price if amended.order_type != 'market' else record.request.limit_price,
                          stop_price=amended.stop_price)
        def guard():
            self._check_decision_data(owner, amended, identity)
            if amended.quantity <= record.filled_quantity:
                raise ValueError('Replacement quantity must exceed cumulative fills; cancel the remainder instead.')
            if record.parent_order_id:
                if amended.quantity * request.risk_price > self.max_order_notional:
                    raise PaperAccountError('Order dollar limit exceeded.')
            else:
                self._risk(owner, replace(amended, quantity=amended.quantity - int(record.filled_quantity)),
                           request.risk_price, excluding_id=local_id, excluding_group_id=record.group_id)
                if record.group_kind == 'oco':
                    for peer in self.broker.journal.orders():
                        if peer.group_id == record.group_id and peer.request.client_order_id != local_id and not peer.terminal and isinstance(peer.request, StockOrder):
                            self._risk(owner, replace(self._order(peer).request, quantity=int(peer.remaining_quantity)),
                                       peer.request.risk_price, excluding_id=local_id, excluding_group_id=record.group_id)
        guard()
        updated = await self.broker.replace(local_id, request, command_id=command_id, submission_guard=guard)
        await self._await_submission(updated)
        accepted = self.broker.journal.get(local_id)
        if accepted.request != request:
            raise PaperAccountError('Broker did not accept replacement terms; inspect the order diagnostics and reconcile.')

    async def _group_stock(self, owner, command, identity):
        children = expand_orders(command)
        group_id = 'rt-group-' + hashlib.sha256(json.dumps((self.broker.config.account, owner, command.client_order_id)).encode()).hexdigest()[:40]
        requests = []
        guards = []
        for index, child in enumerate(children):
            if child.time_in_force not in ('DAY', 'GTC') or child.symbol not in self.instruments:
                raise ValueError('Groups require configured stocks and DAY/GTC lifetimes.')
            local_id = 'rt-' + hashlib.sha256(json.dumps((owner, 'client', child.client_order_id)).encode()).hexdigest()[:40]
            payload = self._command_payload(child)
            metadata = self.broker.journal.runtime_intent(local_id)
            if metadata is not None and (metadata['command'] != payload or metadata['account'] != self.broker.config.account):
                raise ValueError('Group child ID already belongs to different intent/account.')
            if self.broker.journal.group(group_id):
                stored_group = self.broker.journal.group(group_id)
                assert stored_group is not None
                original = json.loads(stored_group['intent'])[index]
                for field in ('limit_price', 'stop_price'):
                    if original.get(field) is not None:
                        original[field] = Decimal(original[field])
                requests.append(StockOrder(**original))
                continue
            quote = await self.broker.quote(StockQuoteRequest(child.symbol, max_age=self.quote_max_age))
            expected = self.quotes.get(child.symbol)
            if (not self._quote_valid(quote) or expected is None or quote.contract.contract_id != expected.contract.contract_id
                    or quote.contract.symbol != child.symbol or quote.contract.security_type != 'STK' or quote.contract.currency != 'USD'):
                raise PaperAccountError('Fresh, identity-consistent live quotes are required for group members.')
            self.quotes[child.symbol] = quote
            instrument = self.instruments[child.symbol]
            limit = child.limit_price
            if child.order_type == 'market':
                assert quote.ask is not None and quote.bid is not None
                raw = quote.ask * (1 + self.market_collar_bps / 10000) if child.side == 'BUY' else quote.bid * (1 - self.market_collar_bps / 10000)
                limit = (raw / instrument.tick_size).to_integral_value(rounding=ROUND_FLOOR if child.side == 'BUY' else ROUND_CEILING) * instrument.tick_size
            if any(price is not None and price % instrument.tick_size for price in (limit, child.stop_price)):
                raise ValueError('Group member prices must follow the configured tick.')
            risk_price = max(limit or ZERO, child.stop_price or ZERO)
            if child.quantity * risk_price > self.max_order_notional:
                raise PaperAccountError('Group member order dollar limit exceeded.')
            def guard(child=child, risk_price=risk_price, quote=quote, index=index):
                self._check_decision_data(owner, child, identity)
                if not self._quote_valid(quote):
                    raise PaperAccountError('Group quote expired during preflight.')
                if not isinstance(command, BracketOrder) or index == 0:
                    self._risk(owner, child, risk_price)
            guards.append(guard)
            requests.append(StockOrder(local_id, owner, child.symbol, child.side, child.quantity, limit,
                                       time_in_force=cast(Literal['DAY', 'GTC'], child.time_in_force),
                                       order_type=cast(Literal['limit', 'stop', 'stop_limit'], 'limit' if child.order_type == 'market' else child.order_type),
                                       stop_price=child.stop_price))
        if not self.broker.journal.group(group_id):
            self.snapshot = await self.broker.snapshot()
            for guard in guards:
                guard()
            for child, request in zip(children, requests, strict=True):
                self.broker.journal.record_runtime_intent(request.client_order_id, self.broker.config.account,
                                                         f'{owner}:{child.client_order_id}', self._command_payload(child))
        def transmit_guard():
            for guard in guards:
                guard()
        records = await self.broker.submit_group(group_id, 'bracket' if isinstance(command, BracketOrder) else 'oco',
                                                 tuple(requests), submission_guard=transmit_guard)
        for record in records:
            await self._await_submission(record)

    @staticmethod
    def _command_payload(command):
        data = asdict(command)
        for name in ('limit_price', 'stop_price'):
            if data.get(name) is not None:
                data[name] = str(data[name].normalize())
        return json.dumps(data, default=str, sort_keys=True)

    async def _await_submission(self, record: OrderRecord) -> None:
        """Serialize new intents through acknowledgment and reported-fill replay.

        A completed combo status can arrive ahead of its leg executions. Yield to
        broker callbacks until the durable journal catches up, without inventing
        fills or resending the intent. Timeouts keep subsequent entries gated.
        """
        identity = record.request.client_order_id
        try:
            async with asyncio.timeout(self.acknowledgment_timeout):
                if not record.acknowledged and not record.terminal:
                    record = await self.broker.observe(identity, seconds=self.acknowledgment_timeout, until="acknowledged")
                if not record.acknowledged and not record.terminal:
                    raise DeliveryError("Order acknowledgment remains unresolved; no dependent intent is safe to submit.")
                while record.replace_pending or record.filled_quantity > record.executed_quantity:
                    await asyncio.sleep(.05)
                    record = self.broker.journal.get(identity)
        except TimeoutError as error:
            self.broker.journal.uncertain(identity,
                "Runtime acknowledgment/execution replay timed out; reconcile before dependent orders.")
            raise DeliveryError("Order acknowledgment or execution details remain unresolved after timeout.") from error

    def _risk(self, owner, command, limit, *, excluding_id=None, excluding_group_id=None):
        all_records = self.broker.journal.orders()
        records = tuple(record for record in all_records if record.request.client_order_id != excluding_id
                        and (excluding_group_id is None or record.group_id != excluding_group_id))
        for record in all_records:
            if record.request.client_order_id == excluding_id:
                continue
            if record.account == self.broker.config.account and (record.replace_pending or (record.uncertain and not record.terminal) or record.filled_quantity > record.executed_quantity):
                raise PaperAccountError("Unresolved order/fill state blocks new submissions.")
        context = self.context(owner)
        if excluding_id is not None:
            excluded = {self._order(record).order_id for record in all_records if isinstance(record.request, StockOrder)
                        and (record.request.client_order_id == excluding_id or (excluding_group_id and record.group_id == excluding_group_id))}
            context = replace(context, orders=tuple(order for order in context.orders if order.order_id not in excluded))
        all_positions = self._positions()
        for symbol in self.instruments:
            owned = sum(p.quantity for (strategy, name), p in all_positions.items() if name == symbol)
            if context.account_positions.get(symbol, 0) != owned:
                raise PaperAccountError(f"Unassigned or mismatched broker position for {symbol}; reconcile before trading.")
        if command.quantity * limit > self.max_order_notional:
            raise PaperAccountError("Order dollar limit exceeded.")
        current = context.position(command.symbol)
        target = current + command.quantity * (1 if command.side == "BUY" else -1)
        reduction = (abs(target) < abs(current) and (target == 0 or (target > 0) == (current > 0))
                     and not any(not order.terminal and order.request.symbol == command.symbol for order in context.orders))
        # An unambiguous exit can release exposure even after marks/funds breach
        # entry limits. Broker acceptance and the per-order price cap still apply.
        if reduction:
            return
        for record in context.broker_orders:
            if isinstance(record.request, VerticalOrder) and (not record.terminal or any(
                    self.broker.journal.attributed_position(self.broker.config.account, owner, leg.contract_id)
                    for leg in (record.request.long_leg, record.request.short_leg))):
                raise PaperAccountError("Use separate strategy budgets for concurrent stock and option exposures.")
        gross = ZERO
        for symbol in self.instruments:
            orders = [order for order in context.orders if order.request.symbol == symbol and not order.terminal and order.active]
            def pending(side):
                groups = {}
                for order in orders:
                    if order.request.side == side:
                        key = order.group_id or order.order_id
                        groups[key] = max(groups.get(key, 0), order.remaining)
                return sum(groups.values())
            buys, sells = pending('BUY'), pending('SELL')
            if symbol == command.symbol:
                buys += command.quantity if command.side == "BUY" else 0
                sells += command.quantity if command.side == "SELL" else 0
            current = context.position(symbol)
            if current - sells < 0 and (not self.allow_short or not self.instruments[symbol].shortable):
                raise PaperAccountError(f"Shorting is not enabled for {symbol}.")
            if current or buys or sells:
                quote = self.quotes[symbol]
                if not self._quote_valid(quote):
                    raise PaperAccountError(f"Fresh marks are required for held/pending {symbol}.")
                assert quote.ask is not None
                prices = [quote.ask] + [record.request.risk_price for record in context.broker_orders
                                       if isinstance(record.request, StockOrder) and record.request.symbol == symbol and not record.terminal]
                if symbol == command.symbol:
                    prices.append(limit)
                gross += max(abs(current - sells), abs(current + buys)) * max(prices)
        if gross > self.budgets[owner]:
            raise PaperAccountError("Strategy gross dollar budget exceeded, including pending orders.")
        reserved = self._reserved_notional(records)
        if reserved + command.quantity * limit > max(ZERO, self._balance("AvailableFunds")):
            raise PaperAccountError("Broker AvailableFunds gate exceeded, including pending reservations.")

    async def _drain(self):
        count = 0
        # Reconciliation may legitimately contain years of unseen broker state.
        # Bound newly generated feedback, not the finite backlog already queued.
        feedback_limit = len(self._queue) + len(self._rejections) + len(self._subscription_updates) + 1000
        while self._queue or self._rejections or self._subscription_updates:
            count += 1
            if count > feedback_limit:
                raise PaperAccountError("Strategy feedback exceeded 1000 broker updates.")
            if not self._queue and self._subscription_updates:
                update, identity = self._subscription_updates.popleft()
                await self._invoke(update.strategy_id, update, identity)
                continue
            if not self._queue:
                rejection = self._rejections.popleft()
                await self._invoke(rejection.strategy_id, rejection, f"rejection:{rejection.time.isoformat()}:{count}")
                continue
            record = self._queue.popleft()
            if record not in self._records():
                # A newer journal record may supersede a queued snapshot.
                try:
                    record = self.broker.journal.get(record.request.client_order_id)
                except KeyError:
                    continue
            if record.account != self.broker.config.account or record.request.strategy_id not in self.strategies:
                continue
            if isinstance(record.request, VerticalOrder):
                if self._seen_broker_orders.get(record.request.client_order_id) != record:
                    self._seen_broker_orders[record.request.client_order_id] = record
                    event = BrokerOrderUpdate(datetime.now(timezone.utc), record)
                    self.events.append(event)
                    await self._invoke(record.request.strategy_id, event, self._record_identity(record))
                continue
            if record.request.symbol not in self.instruments:
                continue
            owner = record.request.strategy_id
            order = self._order(record)
            previous_record = self._seen_broker_orders.get(record.request.client_order_id)
            previous_diagnostics = previous_record.diagnostics if previous_record else ()
            self._seen_broker_orders[record.request.client_order_id] = record
            for diagnostic in record.diagnostics[len(previous_diagnostics):]:
                rejected_command = None
                if diagnostic.startswith(('IBKR 10147:', 'IBKR 10148:')):
                    rejected_command = CancelOrder(order.order_id)
                elif diagnostic.startswith(('IBKR 201:', 'IBKR 10329:', 'IBKR 104:', 'IBKR 105:', 'IBKR 329:')):
                    revision = self.broker.journal.latest_replacement(record.request.client_order_id)
                    if revision and revision['status'] == 'rejected':
                        desired = json.loads(revision['request'])
                        rejected_command = ReplaceOrder(order.order_id, quantity=desired['quantity'],
                                                        limit_price=desired['limit_price'], stop_price=desired.get('stop_price'),
                                                        client_command_id=revision['command_id'])
                if rejected_command is not None:
                    rejection = Rejection(datetime.now(timezone.utc), owner, diagnostic, rejected_command)
                    self.events.append(rejection)
                    rejection_id = hashlib.sha256(diagnostic.encode()).hexdigest()
                    await self._invoke(owner, rejection, f'mutation-rejected:{order.order_id}:{rejection_id}')
            previous = self._seen_orders.get(order.order_id)
            if previous != order:
                self._seen_orders[order.order_id] = order
                event = OrderUpdate(datetime.now(timezone.utc), order)
                self.events.append(event)
                await self._invoke(owner, event, self._record_identity(record))
                if order.status == "rejected" and (previous is None or previous.status != "rejected"):
                    rejection = Rejection(event.time, owner, "; ".join(record.diagnostics) or "Broker rejected the order.", order.request)
                    self.events.append(rejection)
                    self._rejections.append(rejection)
            for execution in record.executions:
                if execution.execution_id in self._seen_executions or execution.commission is None:
                    continue
                if execution.commission_currency != "USD":
                    self._fail(owner, ValueError("Only USD execution commissions are supported."))
                    continue
                self._seen_executions.add(execution.execution_id)
                now = datetime.now(timezone.utc)
                fill = Fill(execution.execution_id, order.order_id, owner, record.request.symbol, execution.side,
                            int(execution.quantity), execution.price, execution.commission,
                            datetime.fromisoformat(execution.occurred_at), now)
                event = FillEvent(now, fill)
                self.events.append(event)
                await self._invoke(owner, event, f"fill:{execution.execution_id}")
