"""Immutable strategy/runtime contracts. Monetary values use Decimal."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Mapping, Literal, Protocol, Iterable, TYPE_CHECKING

from .account import AccountState, currency

from convexity.brokers.ibkr import PaperAccountSnapshot, Quote
from convexity.trading import OrderRecord, VerticalOrder

if TYPE_CHECKING:
    from .contracts import StrategyContract, StrategyMetadata

ZERO = Decimal(0)


def decimal(value) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("Numeric values must be finite.")
    return result


def aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamps must be timezone-aware.")


@dataclass(frozen=True)
class Instrument:
    symbol: str
    tick_size: Decimal = Decimal("0.01")
    shortable: bool = True
    currency: str = "USD"

    def __post_init__(self):
        object.__setattr__(self, "tick_size", decimal(self.tick_size))
        currency(self.currency)
        if not self.symbol or self.tick_size <= 0:
            raise ValueError("Supply a stock identity with a positive tick size.")


@dataclass(frozen=True)
class Bar:
    symbol: str
    start: datetime
    end: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int

    def __post_init__(self):
        aware(self.start)
        aware(self.end)
        object.__setattr__(self, "start", self.start.astimezone(timezone.utc))
        object.__setattr__(self, "end", self.end.astimezone(timezone.utc))
        for name in ("open", "high", "low", "close"):
            object.__setattr__(self, name, decimal(getattr(self, name)))
        if not self.symbol or self.start >= self.end:
            raise ValueError("Bars need a symbol and start < end.")
        if (
            not 0
            < self.low
            <= min(self.open, self.close)
            <= max(self.open, self.close)
            <= self.high
        ):
            raise ValueError("Invalid OHLC prices.")
        if type(self.volume) is not int or self.volume < 0:
            raise ValueError("Bar volume must be a nonnegative whole-share count.")


@dataclass(frozen=True)
class Bars:
    time: datetime
    bars: Mapping[str, Bar]
    delayed: bool = False

    def __post_init__(self):
        aware(self.time)
        object.__setattr__(self, "time", self.time.astimezone(timezone.utc))
        if type(self.delayed) is not bool:
            raise ValueError("delayed must be boolean.")
        if not self.bars or any(
            symbol != bar.symbol
            or (bar.end > self.time if self.delayed else bar.end != self.time)
            for symbol, bar in self.bars.items()
        ):
            raise ValueError(
                "Bars must end at event time; delayed observations may end earlier."
            )
        object.__setattr__(self, "bars", MappingProxyType(dict(self.bars)))


@dataclass(frozen=True)
class Timer:
    time: datetime
    name: str

    def __post_init__(self):
        aware(self.time)
        object.__setattr__(self, "time", self.time.astimezone(timezone.utc))
        if not self.name:
            raise ValueError("Timer name is required.")


@dataclass(frozen=True)
class SubmitOrder:
    """One order intent. An explicit client_order_id is an idempotency key.

    Reuse it only to retry the same order, including after it becomes terminal.
    Recurring decisions must omit it or supply a unique key per decision/event.
    """

    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    order_type: Literal["market", "limit", "stop", "stop_limit"] = "market"
    limit_price: Decimal | None = None
    time_in_force: Literal["GTC", "DAY", "IOC", "FOK"] = "GTC"
    client_order_id: str | None = None
    stop_price: Decimal | None = None

    @classmethod
    def market(
        cls,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        *,
        time_in_force: Literal["GTC", "DAY", "IOC", "FOK"] = "DAY",
        client_order_id: str | None = None,
    ):
        """Submit a market intent; acceptance and fills arrive as later events."""
        return cls(
            symbol,
            side,
            quantity,
            "market",
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )

    @classmethod
    def limit(
        cls,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        limit_price,
        *,
        time_in_force: Literal["GTC", "DAY", "IOC", "FOK"] = "DAY",
        client_order_id: str | None = None,
    ):
        """Submit a limit intent at a specified price; execution is not guaranteed."""
        return cls(
            symbol, side, quantity, "limit", limit_price, time_in_force, client_order_id
        )

    @classmethod
    def stop(
        cls,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        stop_price,
        *,
        time_in_force: Literal["GTC", "DAY"] = "DAY",
        client_order_id: str | None = None,
    ):
        """A stop becomes a market order after triggering; its price is not guaranteed."""
        return cls(
            symbol,
            side,
            quantity,
            "stop",
            time_in_force=time_in_force,
            client_order_id=client_order_id,
            stop_price=stop_price,
        )

    @classmethod
    def stop_limit(
        cls,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        stop_price,
        limit_price,
        *,
        time_in_force: Literal["GTC", "DAY"] = "DAY",
        client_order_id: str | None = None,
    ):
        """A stop becomes a limit order after triggering; it may remain unfilled."""
        return cls(
            symbol,
            side,
            quantity,
            "stop_limit",
            limit_price,
            time_in_force,
            client_order_id,
            stop_price,
        )

    def __post_init__(self):
        if (
            not self.symbol
            or self.side not in ("BUY", "SELL")
            or type(self.quantity) is not int
            or self.quantity <= 0
        ):
            raise ValueError(
                "Orders require a symbol, BUY/SELL, and positive whole-share quantity."
            )
        if self.order_type not in (
            "market",
            "limit",
            "stop",
            "stop_limit",
        ) or self.time_in_force not in ("GTC", "DAY", "IOC", "FOK"):
            raise ValueError(
                "Only market/limit/stop/stop_limit and GTC/DAY/IOC/FOK orders are supported."
            )
        if self.order_type in ("stop", "stop_limit") and self.time_in_force in (
            "IOC",
            "FOK",
        ):
            raise ValueError("Stop orders support DAY/GTC lifetimes.")
        if self.limit_price is not None:
            object.__setattr__(self, "limit_price", decimal(self.limit_price))
        if self.order_type in ("limit", "stop_limit") and (
            self.limit_price is None or self.limit_price <= 0
        ):
            raise ValueError("Limit orders require a positive limit price.")
        if self.order_type in ("market", "stop") and self.limit_price is not None:
            raise ValueError("Market/stop orders cannot specify a limit price.")
        if self.stop_price is not None:
            object.__setattr__(self, "stop_price", decimal(self.stop_price))
        if self.order_type in ("stop", "stop_limit"):
            if self.stop_price is None or self.stop_price <= 0:
                raise ValueError("Stop orders require a positive stop price.")
        elif self.stop_price is not None:
            raise ValueError("Only stop orders may specify a stop price.")
        if self.client_order_id is not None and not self.client_order_id:
            raise ValueError("Explicit order IDs must be nonempty.")


@dataclass(frozen=True)
class CancelOrder:
    order_id: str

    def __post_init__(self):
        if not self.order_id:
            raise ValueError("Cancel requires an order identity.")


@dataclass(frozen=True)
class ReplaceOrder:
    """Amend an owned order in place. Quantity is TOTAL, including executions.

    Omitted fields retain their accepted values; symbol, side, type and lifetime
    cannot change. The stable order ID is preserved across acknowledged revisions.
    """

    order_id: str
    quantity: int | None = None
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    client_command_id: str | None = None

    def __post_init__(self):
        if not self.order_id or (
            self.client_command_id is not None and not self.client_command_id
        ):
            raise ValueError(
                "Replacement requires an order identity and nonempty command identity if supplied."
            )
        if (
            self.quantity is None
            and self.limit_price is None
            and self.stop_price is None
        ):
            raise ValueError("ReplaceOrder requires at least one changed field.")
        if self.quantity is not None and (
            type(self.quantity) is not int or self.quantity <= 0
        ):
            raise ValueError(
                "Replacement quantity must be a positive total whole-share quantity."
            )
        for name in ("limit_price", "stop_price"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, decimal(value))
                if getattr(self, name) <= 0:
                    raise ValueError("Replacement prices must be positive.")


@dataclass(frozen=True)
class CancelOrders:
    """Cancel a snapshot of this strategy's working orders matching all filters."""

    symbol: str | None = None
    side: Literal["BUY", "SELL"] | None = None
    group_id: str | None = None

    def __post_init__(self):
        if (
            self.symbol == ""
            or self.group_id == ""
            or self.side not in (None, "BUY", "SELL")
        ):
            raise ValueError(
                "Cancellation filters require nonempty identities and BUY/SELL side."
            )


@dataclass(frozen=True)
class BracketOrder:
    """An entry with held profit/stop exits; exits activate after full entry fill."""

    entry: SubmitOrder
    take_profit: Decimal
    stop_loss: Decimal
    client_order_id: str
    stop_limit_price: Decimal | None = None

    def __post_init__(self):
        if not isinstance(self.entry, SubmitOrder) or not self.client_order_id:
            raise ValueError("Bracket requires an entry and stable group identity.")
        if self.entry.time_in_force not in ("DAY", "GTC"):
            raise ValueError("Brackets support DAY/GTC entries.")
        for name in ("take_profit", "stop_loss", "stop_limit_price"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, decimal(value))
                if getattr(self, name) <= 0:
                    raise ValueError("Bracket prices must be positive.")
        if (self.entry.side == "BUY" and self.take_profit <= self.stop_loss) or (
            self.entry.side == "SELL" and self.take_profit >= self.stop_loss
        ):
            raise ValueError(
                "Take-profit must lie beyond stop-loss in the entry direction."
            )


@dataclass(frozen=True)
class OCOOrders:
    """Related orders: the first execution requests cancellation of peers."""

    orders: tuple[SubmitOrder, ...]
    client_order_id: str

    def __post_init__(self):
        object.__setattr__(self, "orders", tuple(self.orders))
        if (
            not self.client_order_id
            or len(self.orders) < 2
            or any(not isinstance(order, SubmitOrder) for order in self.orders)
        ):
            raise ValueError(
                "OCO requires a stable group identity and at least two orders."
            )
        if any(order.time_in_force not in ("DAY", "GTC") for order in self.orders):
            raise ValueError("OCO orders support DAY/GTC lifetimes.")


@dataclass(frozen=True)
class Subscribe:
    subscription: Any
    subscription_id: str

    def __post_init__(self):
        if not self.subscription_id:
            raise ValueError("Subscription identity must be nonempty.")


@dataclass(frozen=True)
class Unsubscribe:
    subscription_id: str

    def __post_init__(self):
        if not self.subscription_id:
            raise ValueError("Subscription identity must be nonempty.")


@dataclass(frozen=True)
class SubscriptionUpdate:
    time: datetime
    strategy_id: str
    subscription_id: str
    subscription: Any
    status: Literal["active", "inactive", "rejected"]
    reason: str | None = None

    def __post_init__(self):
        aware(self.time)
        if (
            not self.strategy_id
            or not self.subscription_id
            or self.status not in ("active", "inactive", "rejected")
        ):
            raise ValueError("Invalid subscription lifecycle event.")


@dataclass(frozen=True)
class Order:
    order_id: str
    strategy_id: str
    request: SubmitOrder
    accepted_at: datetime
    sequence: int
    status: str = "working"
    filled_quantity: int = 0
    cancel_pending: bool = False
    expires_at: datetime | None = None
    reason: str | None = None
    revision: int = 0
    replace_pending: bool = False
    group_id: str | None = None
    parent_order_id: str | None = None
    active: bool = True

    def __post_init__(self):
        aware(self.accepted_at)
        if self.expires_at is not None:
            aware(self.expires_at)
            if self.expires_at <= self.accepted_at:
                raise ValueError("Order expiry must follow acceptance.")
        if not self.order_id or not self.strategy_id:
            raise ValueError("Order and strategy identities are required.")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("Order revision must be a nonnegative integer.")
        if self.status not in (
            "working",
            "partially_filled",
            "filled",
            "canceled",
            "rejected",
            "expired",
        ):
            raise ValueError("Unknown order status.")
        if (
            type(self.filled_quantity) is not int
            or not 0 <= self.filled_quantity <= self.request.quantity
        ):
            raise ValueError(
                "Filled quantity must be within the requested whole-share quantity."
            )
        if (self.status == "filled") != (self.filled_quantity == self.request.quantity):
            raise ValueError("Filled status must agree with the completed quantity.")

    @property
    def remaining(self) -> int:
        return self.request.quantity - self.filled_quantity

    @property
    def terminal(self) -> bool:
        return self.status in ("filled", "canceled", "rejected", "expired")


@dataclass(frozen=True)
class Fill:
    execution_id: str
    order_id: str
    strategy_id: str
    symbol: str
    side: str
    quantity: int
    price: Decimal
    fee: Decimal
    occurred_at: datetime
    available_at: datetime
    currency: str = "USD"

    def __post_init__(self):
        currency(self.currency)
        aware(self.occurred_at)
        aware(self.available_at)
        if self.occurred_at > self.available_at:
            raise ValueError("Execution cannot become available before it occurs.")
        if not all(
            (self.execution_id, self.order_id, self.strategy_id, self.symbol)
        ) or self.side not in ("BUY", "SELL"):
            raise ValueError("Execution identities and BUY/SELL side are required.")
        if type(self.quantity) is not int or self.quantity <= 0:
            raise ValueError("Execution quantity must be a positive whole-share count.")
        object.__setattr__(self, "price", decimal(self.price))
        object.__setattr__(self, "fee", decimal(self.fee))
        if self.price <= 0 or self.fee < 0:
            raise ValueError("Execution price must be positive and fee nonnegative.")


@dataclass(frozen=True)
class OrderUpdate:
    time: datetime
    order: Order


@dataclass(frozen=True)
class FillEvent:
    time: datetime
    fill: Fill


@dataclass(frozen=True)
class BrokerOrderUpdate:
    """Durable broker order state, including option leg executions."""

    time: datetime
    record: OrderRecord

    def __post_init__(self):
        aware(self.time)


@dataclass(frozen=True)
class OptionQuotes:
    """Observed option snapshots; their individual quality status is preserved."""

    time: datetime
    quotes: tuple[Quote, ...]

    def __post_init__(self):
        aware(self.time)
        object.__setattr__(self, "time", self.time.astimezone(timezone.utc))
        object.__setattr__(self, "quotes", tuple(self.quotes))
        if any(
            quote.contract.security_type != "OPT"
            or quote.contract.contract_id <= 0
            or quote.completed_at > self.time
            for quote in self.quotes
        ):
            raise ValueError(
                "Option events require resolved option quotes observed by the event time."
            )
        if len({quote.contract.contract_id for quote in self.quotes}) != len(
            self.quotes
        ):
            raise ValueError("Option quote events cannot repeat contract identities.")


@dataclass(frozen=True)
class StockQuotes:
    """Observed stock quotes, including explicit unavailable/stale outcomes."""

    time: datetime
    quotes: Mapping[str, Quote]

    def __post_init__(self):
        aware(self.time)
        object.__setattr__(self, "time", self.time.astimezone(timezone.utc))
        if any(
            symbol != quote.contract.symbol
            or quote.contract.security_type != "STK"
            or quote.contract.contract_id <= 0
            or quote.completed_at > self.time
            for symbol, quote in self.quotes.items()
        ):
            raise ValueError(
                "Stock events require resolved symbol-keyed quotes observed by the event time."
            )
        object.__setattr__(self, "quotes", MappingProxyType(dict(self.quotes)))


@dataclass(frozen=True)
class AccountUpdate:
    """A broker-authoritative account snapshot available at this event time."""

    time: datetime
    snapshot: PaperAccountSnapshot

    def __post_init__(self):
        aware(self.time)
        aware(self.snapshot.started_at)
        aware(self.snapshot.completed_at)
        object.__setattr__(self, "time", self.time.astimezone(timezone.utc))
        if not self.snapshot.started_at <= self.snapshot.completed_at <= self.time:
            raise ValueError("Account snapshot must have completed by its event time.")


@dataclass(frozen=True)
class ConnectionHealth:
    """Connection/data readiness; disconnected events cannot authorize entries."""

    time: datetime
    connected: bool
    reason: str = ""

    def __post_init__(self):
        aware(self.time)
        object.__setattr__(self, "time", self.time.astimezone(timezone.utc))
        if type(self.connected) is not bool:
            raise ValueError("Connection health requires a boolean connected flag.")


@dataclass(frozen=True)
class Rejection:
    time: datetime
    strategy_id: str
    reason: str
    command: Command


@dataclass(frozen=True)
class Position:
    quantity: int = 0
    average_cost: Decimal = ZERO
    realized_pnl: Decimal = ZERO
    currency: str = "USD"


@dataclass(frozen=True)
class PriceObservation:
    """Price known at time, in the instrument's currency; not a fill guarantee."""

    value: Decimal
    time: datetime
    currency: str
    source: str
    usable: bool = True

    def __post_init__(self):
        aware(self.time)
        currency(self.currency)
        object.__setattr__(self, "value", decimal(self.value))
        if self.value <= 0:
            raise ValueError("Observed prices must be positive.")


@dataclass(frozen=True)
class Context:
    time: datetime
    strategy_id: str
    cash: Decimal
    equity: Decimal
    positions: Mapping[str, Position]
    account_positions: Mapping[str, int]
    marks: Mapping[str, Decimal]
    mark_times: Mapping[str, datetime]
    orders: tuple[Order, ...]
    history: Mapping[str, tuple[Bar, ...]]
    broker_orders: tuple[OrderRecord, ...] = ()
    subscriptions: Mapping[str, Any] = field(default_factory=dict)
    _commands: list[Any] = field(default_factory=list, repr=False, compare=False)

    account: AccountState | None = None
    instrument_currencies: Mapping[str, str] = field(default_factory=dict)
    prices: Mapping[str, PriceObservation] = field(default_factory=dict)
    ask_prices: Mapping[str, PriceObservation] = field(default_factory=dict)
    bid_prices: Mapping[str, PriceObservation] = field(default_factory=dict)
    universes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in (
            "universes",
            "instrument_currencies",
            "prices",
            "ask_prices",
            "bid_prices",
            "positions",
            "account_positions",
            "marks",
            "mark_times",
            "history",
            "subscriptions",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(self, "broker_orders", tuple(self.broker_orders))

    def universe(self, name: str):
        """Membership known at this callback's decision timestamp."""
        if name not in self.universes:
            raise ValueError(
                f"Universe {name!r} is not declared in this strategy contract."
            )
        return self.universes[name]

    def price_info(
        self,
        symbol: str,
        *,
        side: Literal["BUY", "SELL", "MARK"] = "MARK",
        max_age: timedelta | None = None,
    ) -> PriceObservation:
        """Use ask/bid for buys/sells when supplied, otherwise the observed mark."""
        if side not in ("BUY", "SELL", "MARK"):
            raise ValueError("side must be BUY, SELL or MARK.")
        if max_age is not None and max_age < timedelta(0):
            raise ValueError("max_age must be nonnegative.")
        values = (
            self.ask_prices
            if side == "BUY"
            else self.bid_prices
            if side == "SELL"
            else self.prices
        )
        price = values.get(symbol) or self.prices.get(symbol)
        if price is None and symbol in self.marks and symbol in self.mark_times:
            price = PriceObservation(
                self.marks[symbol],
                self.mark_times[symbol],
                self.instrument_currencies.get(symbol, "USD"),
                "mark",
            )
        if price is None:
            raise ValueError(f"No observed price for {symbol}.")
        if not price.usable:
            raise ValueError(f"Price for {symbol} is not a usable live observation.")
        if price.time > self.time:
            raise ValueError(
                f"Future price for {symbol} is not available at context.time."
            )
        if max_age is not None and self.time - price.time > max_age:
            raise ValueError(f"Stale price for {symbol}.")
        return price

    def price(
        self,
        symbol: str,
        *,
        side: Literal["BUY", "SELL", "MARK"] = "MARK",
        max_age: timedelta | None = None,
    ) -> Decimal:
        return self.price_info(symbol, side=side, max_age=max_age).value

    def position(self, symbol: str) -> int:
        return self.positions.get(symbol, Position()).quantity

    def subscribe(
        self, subscription, *, subscription_id: str | None = None
    ) -> Subscribe:
        """Request activation at callback completion; observe SubscriptionUpdate."""
        from .subscriptions import subscription_id as identity

        command = Subscribe(
            subscription,
            subscription_id if subscription_id is not None else identity(subscription),
        )
        self._commands.append(command)
        return command

    def unsubscribe(self, subscription_id: str) -> Unsubscribe:
        """Request removal at callback completion; already delivered events stand."""
        command = Unsubscribe(subscription_id)
        self._commands.append(command)
        return command

    def pending(self, symbol: str) -> int:
        return sum(
            o.remaining * (1 if o.request.side == "BUY" else -1)
            for o in self.orders
            if o.active and not o.terminal and o.request.symbol == symbol
        )


Event = (
    Bars
    | Timer
    | OrderUpdate
    | FillEvent
    | Rejection
    | BrokerOrderUpdate
    | OptionQuotes
    | StockQuotes
    | AccountUpdate
    | ConnectionHealth
    | SubscriptionUpdate
)
Command = (
    SubmitOrder
    | CancelOrder
    | ReplaceOrder
    | CancelOrders
    | BracketOrder
    | OCOOrders
    | Subscribe
    | Unsubscribe
    | VerticalOrder
)
COMMAND_TYPES = (
    SubmitOrder,
    CancelOrder,
    ReplaceOrder,
    CancelOrders,
    BracketOrder,
    OCOOrders,
    Subscribe,
    Unsubscribe,
    VerticalOrder,
)


def collect_commands(context: Context, output) -> tuple[Command, ...]:
    """Materialize a whole callback before routing; returned queued intents count once."""
    returned = (
        ()
        if output is None
        else ((output,) if isinstance(output, COMMAND_TYPES) else tuple(output))
    )
    queued = tuple(context._commands)
    return queued + tuple(
        command for command in returned if not any(command is item for item in queued)
    )


class Strategy:
    """Override callbacks; return one command, an iterable, or None."""

    # Declarations belong to the strategy and are identical in every runtime.
    metadata: StrategyMetadata | None = None
    contract: StrategyContract | None = None

    def on_start(self, context: Context) -> Command | Iterable[Command] | None:
        return None

    def on_event(
        self, event: Event, context: Context
    ) -> Command | Iterable[Command] | None:
        return None

    def on_stop(self, context: Context) -> None:
        pass

    def dump_state(self):
        """Return JSON-serializable durable decision state, or None if stateless.

        PaperRuntime checkpoints this after callbacks, before routing commands.
        Do not include resources, market history, or broker-derived positions.
        """
        return None

    def load_state(self, state) -> None:
        """Restore dump_state output before on_start; override for stateful strategies."""
        if state is not None:
            raise ValueError(
                "Override load_state to restore a nonempty strategy checkpoint."
            )


@dataclass(frozen=True)
class RiskLimits:
    gross_leverage: Decimal = Decimal(1)
    max_order_notional: Decimal | None = None
    strategy_gross_limits: Mapping[str, Decimal] = field(default_factory=dict)
    maintenance_margin_rate: Decimal | None = None

    def __post_init__(self):
        object.__setattr__(self, "gross_leverage", decimal(self.gross_leverage))
        if self.gross_leverage <= 0:
            raise ValueError("gross_leverage must be positive.")
        if self.maintenance_margin_rate is not None:
            rate = decimal(self.maintenance_margin_rate)
            if not 0 <= rate <= 1 / self.gross_leverage:
                raise ValueError(
                    "Maintenance margin rate must be between zero and the initial margin rate."
                )
            object.__setattr__(self, "maintenance_margin_rate", rate)
        if self.max_order_notional is not None:
            object.__setattr__(
                self, "max_order_notional", decimal(self.max_order_notional)
            )
            if self.max_order_notional <= 0:
                raise ValueError("max_order_notional must be positive.")
        limits = {k: decimal(v) for k, v in self.strategy_gross_limits.items()}
        if any(v <= 0 for v in limits.values()):
            raise ValueError("Strategy gross limits must be positive dollar amounts.")
        object.__setattr__(self, "strategy_gross_limits", MappingProxyType(limits))


class Exchange(Protocol):
    """Execution boundary; the Runtime owns accounting and strategy invocation."""

    allow_short: bool
    annual_borrow_rate: Decimal

    def fee(self, quantity: int, already_filled: int = 0) -> Decimal: ...
    def submit(self, order: Order) -> list[OrderUpdate]: ...
    def cancel(self, order_id: str, time: datetime) -> list[OrderUpdate]: ...
    def replace(self, order: Order, time: datetime) -> list[OrderUpdate]: ...
    def process(self, bars: Bars, runtime: object) -> None: ...
    def expire(self, time: datetime) -> list[OrderUpdate]: ...
