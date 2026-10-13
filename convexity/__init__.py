"""Quantitative math, strategy runtime, stock backtesting and broker integrations."""

from importlib import import_module
from typing import TYPE_CHECKING
from . import math

if TYPE_CHECKING:
    from .runtime import AccountState, FXRate, PriceObservation
    from . import backtest, brokers, runtime, trading, strategies, data
    from .backtest import (Backtest, BacktestFailure, BacktestResult, ExecutionModel, HistoricalData,
                           HistoricalDataset, HistoryError, HistoryRequest, IBKRHistoricalData, Research, SessionCalendar)
    from .brokers.ibkr import PaperAccountConfig
    from .runtime import (AccountUpdate, Bar, Bars, CancelOrder, ConnectionHealth, Context,
                          DeliveryError, FillEvent, Instrument, OptionQuotes, OrderUpdate,
                          PaperTrading, Rejection, RiskLimits, ScheduleSpec, SessionSchedule, StockQuotes,
                          StockSubscription, Strategy, SubmitOrder, Timer)
    from .runtime import (StrategyMetadata, StrategyContract, BarSubscription, ClockSchedule, ClockSubscription, EventSubscription, ContractViolation)
    from .backtest import (CachedHistoricalProvider, HistoricalConnectionConfig,
                           AlpacaCredentials, AlpacaHistoricalData, FallbackHistoricalProvider)
    from .runtime import (ReplaceOrder, CancelOrders, BracketOrder, OCOOrders, Subscribe, Unsubscribe, SubscriptionUpdate, RuntimeCapabilities, RESEARCH_CAPABILITIES, PAPER_CAPABILITIES)
    from .strategies import TargetStrategy

_MODULES = ("math", "runtime", "backtest", "trading", "brokers", "strategies", "data")
_API_EXPORTS = {
    **dict.fromkeys(("AccountState", "FXRate", "PriceObservation"), "runtime"),
    **dict.fromkeys(("Backtest", "BacktestFailure", "BacktestResult", "ExecutionModel", "HistoricalData",
                    "HistoricalDataset", "HistoryError", "HistoryRequest", "IBKRHistoricalData", "Research", "SessionCalendar"), "backtest"),
    **dict.fromkeys(("AccountUpdate", "Bar", "Bars", "CancelOrder", "ConnectionHealth", "Context", "FillEvent",
                    "Instrument", "OptionQuotes", "OrderUpdate", "Rejection", "RiskLimits", "StockQuotes", "Strategy",
                    "SubmitOrder", "Timer", "PaperTrading", "ScheduleSpec", "SessionSchedule", "StockSubscription", "DeliveryError"), "runtime"),
    **dict.fromkeys(('StrategyMetadata', 'StrategyContract', 'BarSubscription', 'ClockSchedule', 'ClockSubscription', 'EventSubscription', 'ContractViolation'), "runtime"),
    **dict.fromkeys(('CachedHistoricalProvider', 'HistoricalConnectionConfig', 'AlpacaCredentials',
                    'AlpacaHistoricalData', 'FallbackHistoricalProvider'), "backtest"),
    **dict.fromkeys(('ReplaceOrder', 'CancelOrders', 'BracketOrder', 'OCOOrders', 'Subscribe', 'Unsubscribe', 'SubscriptionUpdate', 'RuntimeCapabilities', 'RESEARCH_CAPABILITIES', 'PAPER_CAPABILITIES'), "runtime"),
    "PaperAccountConfig": "brokers.ibkr",
    "TargetStrategy": "strategies",
}
__all__ = ["math", "runtime", "backtest", "trading", "brokers", "strategies", "data",
           "Backtest", "BacktestResult", "ExecutionModel", "HistoricalData", "SessionCalendar",
           "BacktestFailure", "HistoricalDataset", "HistoryError", "HistoryRequest", "IBKRHistoricalData", "Research",
           "AccountUpdate", "Bar", "Bars", "CancelOrder", "ConnectionHealth", "Context", "FillEvent",
           "Instrument", "OptionQuotes", "OrderUpdate", "Rejection", "RiskLimits", "StockQuotes", "Strategy",
           "SubmitOrder", "Timer", "PaperTrading", "ScheduleSpec", "SessionSchedule", "StockSubscription",
           "DeliveryError", "PaperAccountConfig", "TargetStrategy"]

__all__ += ['StrategyMetadata', 'StrategyContract', 'BarSubscription', 'ClockSchedule', 'ClockSubscription', 'EventSubscription', 'ContractViolation', 'CachedHistoricalProvider', 'HistoricalConnectionConfig']
__all__ += ['AlpacaCredentials', 'AlpacaHistoricalData', 'FallbackHistoricalProvider']
__all__ += ['ReplaceOrder', 'CancelOrders', 'BracketOrder', 'OCOOrders', 'Subscribe', 'Unsubscribe', 'SubscriptionUpdate', 'RuntimeCapabilities', 'RESEARCH_CAPABILITIES', 'PAPER_CAPABILITIES']

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

__all__ += ["AccountState", "FXRate", "PriceObservation"]
