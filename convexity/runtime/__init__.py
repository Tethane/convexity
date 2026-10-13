"""Public strategy interface, event contracts, and serial trading Runtime."""
from .types import (Bar, Bars, CancelOrder, Command, Context, Event, Exchange, Fill, FillEvent, Instrument,
                    Order, OrderUpdate, Position, Rejection, RiskLimits, Strategy, SubmitOrder, Timer)
from .engine import Runtime

__all__ = ["Bar", "Bars", "CancelOrder", "Command", "Context", "Event", "Exchange", "Fill", "FillEvent",
           "Instrument", "Order", "OrderUpdate", "Position", "Rejection", "RiskLimits",
           "Runtime", "Strategy", "SubmitOrder", "Timer"]

from .schedules import DurableSchedule, ScheduledRun

__all__ += ["DurableSchedule", "ScheduledRun"]

from .status import RuntimeStatus, runtime_status

__all__ += ["RuntimeStatus", "runtime_status"]

from .types import AccountUpdate, BrokerOrderUpdate, ConnectionHealth, OptionQuotes, StockQuotes

__all__ += ["AccountUpdate", "BrokerOrderUpdate", "ConnectionHealth", "OptionQuotes", "StockQuotes"]

from .paper import PaperRuntime

__all__ += ["PaperRuntime"]

from .state import DeliveryError, DeliveryResult, StrategyState
from .calendar_schedule import SessionSchedule

__all__ += ["DeliveryError", "DeliveryResult", "StrategyState", "SessionSchedule"]

# Keep the managed runner lazy: historical data imports runtime contracts too.
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import DataFeedError, IBKRDataSource, PaperDataSource, PaperTrading, ScheduleSpec, StockSubscription

_RUNNER_EXPORTS = ("DataFeedError", "IBKRDataSource", "PaperDataSource", "PaperTrading", "ScheduleSpec", "StockSubscription")
__all__ += ["DataFeedError", "IBKRDataSource", "PaperDataSource", "PaperTrading", "ScheduleSpec", "StockSubscription"]


def __getattr__(name):
    if name not in _RUNNER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".runner", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

from .contracts import (BarSubscription, ClockSubscription, ContractViolation, EventSubscription,
                        StrategyContract, StrategyMetadata)

__all__ += ["BarSubscription", "ClockSubscription", "ContractViolation", "EventSubscription",
            "StrategyContract", "StrategyMetadata"]

from .types import (ReplaceOrder, CancelOrders, BracketOrder, OCOOrders, Subscribe, Unsubscribe, SubscriptionUpdate)
from .capabilities import RuntimeCapabilities, RESEARCH_CAPABILITIES, PAPER_CAPABILITIES

__all__ += ["ReplaceOrder", "CancelOrders", "BracketOrder", "OCOOrders", "Subscribe", "Unsubscribe",
            "SubscriptionUpdate", "RuntimeCapabilities", "RESEARCH_CAPABILITIES", "PAPER_CAPABILITIES"]

from .account import AccountState, FXRate
from .types import PriceObservation

__all__ += ["AccountState", "FXRate", "PriceObservation"]
