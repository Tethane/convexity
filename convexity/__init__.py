"""Quantitative math, strategy runtime, stock backtesting and broker integrations."""

from importlib import import_module
from typing import TYPE_CHECKING
from . import math

if TYPE_CHECKING:
    from .runtime import AccountState, FXRate, PriceObservation
    from . import backtest, brokers, runtime, trading, strategies, data
    from .backtest import (
        Backtest,
        BacktestFailure,
        BacktestResult,
        ExecutionModel,
        HistoricalData,
        HistoricalDataset,
        HistoryError,
        HistoryRequest,
        IBKRHistoricalData,
        Research,
        SessionCalendar,
        DataFrameHistoricalProvider,
    )
    from .brokers.ibkr import PaperAccountConfig
    from .runtime import (
        AccountUpdate,
        Bar,
        Bars,
        CancelOrder,
        ConnectionHealth,
        Context,
        DeliveryError,
        FillEvent,
        Instrument,
        OptionQuotes,
        OrderUpdate,
        PaperTrading,
        Rejection,
        RiskLimits,
        ScheduleSpec,
        SessionSchedule,
        StockQuotes,
        StockSubscription,
        Strategy,
        SubmitOrder,
        Timer,
    )
    from .runtime import (
        StrategyMetadata,
        StrategyContract,
        BarSubscription,
        ClockSchedule,
        ClockSubscription,
        EventSubscription,
        ContractViolation,
    )
    from .backtest import (
        CachedHistoricalProvider,
        HistoricalConnectionConfig,
        AlpacaCredentials,
        AlpacaHistoricalData,
        FallbackHistoricalProvider,
    )
    from .runtime import (
        ReplaceOrder,
        CancelOrders,
        BracketOrder,
        OCOOrders,
        Subscribe,
        Unsubscribe,
        SubscriptionUpdate,
        RuntimeCapabilities,
        RESEARCH_CAPABILITIES,
        PAPER_CAPABILITIES,
    )
    from .strategies import (
        TargetStrategy,
        TargetWeightStrategy,
        RebalancePlan,
        equal_weights,
        target_weight_orders,
    )
    from .runtime import (
        UniverseSubscription,
        CorporateAction,
        UnsupportedCorporateAction,
    )

_MODULES = ("math", "runtime", "backtest", "trading", "brokers", "strategies", "data")
_EXPORT_GROUPS = {
    "runtime": (
        "AccountState",
        "FXRate",
        "PriceObservation",
        "AccountUpdate",
        "Bar",
        "Bars",
        "CancelOrder",
        "ConnectionHealth",
        "Context",
        "FillEvent",
        "Instrument",
        "OptionQuotes",
        "OrderUpdate",
        "Rejection",
        "RiskLimits",
        "StockQuotes",
        "Strategy",
        "SubmitOrder",
        "Timer",
        "PaperTrading",
        "ScheduleSpec",
        "SessionSchedule",
        "StockSubscription",
        "DeliveryError",
        "StrategyMetadata",
        "StrategyContract",
        "BarSubscription",
        "ClockSchedule",
        "ClockSubscription",
        "EventSubscription",
        "ContractViolation",
        "ReplaceOrder",
        "CancelOrders",
        "BracketOrder",
        "OCOOrders",
        "Subscribe",
        "Unsubscribe",
        "SubscriptionUpdate",
        "RuntimeCapabilities",
        "RESEARCH_CAPABILITIES",
        "PAPER_CAPABILITIES",
        "UniverseSubscription",
        "CorporateAction",
        "UnsupportedCorporateAction",
    ),
    "backtest": (
        "Backtest",
        "BacktestFailure",
        "BacktestResult",
        "ExecutionModel",
        "HistoricalData",
        "HistoricalDataset",
        "HistoryError",
        "HistoryRequest",
        "IBKRHistoricalData",
        "Research",
        "SessionCalendar",
        "CachedHistoricalProvider",
        "HistoricalConnectionConfig",
        "AlpacaCredentials",
        "AlpacaHistoricalData",
        "FallbackHistoricalProvider",
        "DataFrameHistoricalProvider",
    ),
    "brokers.ibkr": ("PaperAccountConfig",),
    "strategies": (
        "TargetStrategy",
        "TargetWeightStrategy",
        "target_weight_orders",
        "equal_weights",
        "RebalancePlan",
    ),
}
_API_EXPORTS = {
    name: module for module, names in _EXPORT_GROUPS.items() for name in names
}
__all__ = [
    "math",
    "runtime",
    "backtest",
    "trading",
    "brokers",
    "strategies",
    "data",
    "AccountState",
    "FXRate",
    "PriceObservation",
    "AccountUpdate",
    "Bar",
    "Bars",
    "CancelOrder",
    "ConnectionHealth",
    "Context",
    "FillEvent",
    "Instrument",
    "OptionQuotes",
    "OrderUpdate",
    "Rejection",
    "RiskLimits",
    "StockQuotes",
    "Strategy",
    "SubmitOrder",
    "Timer",
    "PaperTrading",
    "ScheduleSpec",
    "SessionSchedule",
    "StockSubscription",
    "DeliveryError",
    "StrategyMetadata",
    "StrategyContract",
    "BarSubscription",
    "ClockSchedule",
    "ClockSubscription",
    "EventSubscription",
    "ContractViolation",
    "ReplaceOrder",
    "CancelOrders",
    "BracketOrder",
    "OCOOrders",
    "Subscribe",
    "Unsubscribe",
    "SubscriptionUpdate",
    "RuntimeCapabilities",
    "RESEARCH_CAPABILITIES",
    "PAPER_CAPABILITIES",
    "UniverseSubscription",
    "CorporateAction",
    "UnsupportedCorporateAction",
    "Backtest",
    "BacktestFailure",
    "BacktestResult",
    "ExecutionModel",
    "HistoricalData",
    "HistoricalDataset",
    "HistoryError",
    "HistoryRequest",
    "IBKRHistoricalData",
    "Research",
    "SessionCalendar",
    "CachedHistoricalProvider",
    "HistoricalConnectionConfig",
    "AlpacaCredentials",
    "AlpacaHistoricalData",
    "FallbackHistoricalProvider",
    "DataFrameHistoricalProvider",
    "PaperAccountConfig",
    "TargetStrategy",
    "TargetWeightStrategy",
    "target_weight_orders",
    "equal_weights",
    "RebalancePlan",
]


def __getattr__(name):
    if name in _API_EXPORTS:
        value = getattr(import_module(f".{_API_EXPORTS[name]}", __name__), name)
    elif name in _MODULES:
        value = import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
