"""Stock backtesting through the Convexity Runtime and configurable exchange."""
from .data import HistoricalData, Session, SessionCalendar
from .exchange import ExecutionModel, SimulatedExchange
from .runner import Backtest, BacktestFailure, BacktestResult
from .history import HistoricalConnectionConfig, HistoricalDataset, HistoryError, HistoryRequest, IBKRHistoricalData
from .cache import CachedHistoricalProvider
from .alpaca import AlpacaCredentials, AlpacaHistoricalData
from .providers import FallbackHistoricalProvider
from .research import HistoricalProvider, Research

__all__ = ["Backtest", "BacktestResult", "ExecutionModel", "HistoricalData", "Session",
           "SessionCalendar", "SimulatedExchange", "BacktestFailure", "HistoricalDataset",
           "HistoryError", "HistoryRequest", "IBKRHistoricalData", "HistoricalProvider", "Research",
           "CachedHistoricalProvider", "HistoricalConnectionConfig", "AlpacaCredentials",
           "AlpacaHistoricalData", "FallbackHistoricalProvider"]
