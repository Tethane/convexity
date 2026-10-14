"""Inspectable backend capabilities; unsupported intents fail explicitly."""

from dataclasses import dataclass

from .contracts import BarSubscription, ClockSubscription, EventSubscription
from .calendar_schedule import SessionSchedule
from .types import (
    AccountUpdate,
    Bars,
    BracketOrder,
    BrokerOrderUpdate,
    CancelOrder,
    CancelOrders,
    ConnectionHealth,
    FillEvent,
    OCOOrders,
    OptionQuotes,
    OrderUpdate,
    Rejection,
    ReplaceOrder,
    StockQuotes,
    SubmitOrder,
    Subscribe,
    SubscriptionUpdate,
    Timer,
    Unsubscribe,
    VerticalOrder,
)


@dataclass(frozen=True)
class RuntimeCapabilities:
    commands: tuple[type, ...]
    events: tuple[type, ...]
    order_types: tuple[str, ...]
    time_in_force: tuple[str, ...]
    subscription_types: tuple[type, ...]
    native_order_groups: bool
    notes: tuple[str, ...]


_SHARED_COMMANDS = (
    SubmitOrder,
    CancelOrder,
    ReplaceOrder,
    CancelOrders,
    BracketOrder,
    OCOOrders,
    Subscribe,
    Unsubscribe,
)
_SHARED_EVENTS = (Bars, Timer, OrderUpdate, FillEvent, Rejection, SubscriptionUpdate)
_SUBSCRIPTIONS = (
    BarSubscription,
    ClockSubscription,
    EventSubscription,
    SessionSchedule,
)

RESEARCH_CAPABILITIES = RuntimeCapabilities(
    _SHARED_COMMANDS,
    _SHARED_EVENTS,
    ("market", "limit", "stop", "stop_limit"),
    ("DAY", "GTC", "IOC", "FOK"),
    _SUBSCRIPTIONS,
    False,
    (
        "Stop and grouped orders require DAY/GTC.",
        "OHLC observations do not reveal intrabar ordering; protective stops take priority in ambiguous brackets.",
        "Bracket exits activate only after the entry is completely filled.",
        "Bracket members support price amendments, not independent quantity amendments.",
        "Dynamic subscriptions must belong to the strategy's declared available pool.",
        "One exchange calendar per research run; one bar resolution per symbol per strategy.",
    ),
)

PAPER_CAPABILITIES = RuntimeCapabilities(
    (*_SHARED_COMMANDS, VerticalOrder),
    (
        *_SHARED_EVENTS,
        BrokerOrderUpdate,
        StockQuotes,
        OptionQuotes,
        AccountUpdate,
        ConnectionHealth,
    ),
    ("market", "limit", "stop", "stop_limit"),
    ("DAY", "GTC"),
    _SUBSCRIPTIONS,
    True,
    (
        "Market intents use the configured protective limit collar; stops use native broker stop types.",
        "Native bracket exits are held until complete entry execution; partial entries remain exposed.",
        "Bracket members support price amendments, not independent quantity amendments.",
        "Native stops currently require regular sessions.",
        "ReplaceOrder amends stock orders; option vertical amendments are unsupported.",
        "Managed paper bar feeds support one-minute observations.",
        "Dynamic subscriptions must belong to the strategy's declared available pool.",
    ),
)
