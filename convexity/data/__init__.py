"""Standalone historical market data as pandas DataFrames.

No strategy, event runtime, backtest, or brokerage account model is required.
"""

from .core import (
    DataClient,
    DataError,
    DataProvider,
    NoDataError,
    ahistory,
    history,
    split_history,
)
from .crypto import Binance, Deribit
from .ibkr import IBKR
from .yahoo import YahooFinance

__all__ = [
    "IBKR", "Binance", "DataClient", "DataError", "DataProvider", "Deribit",
    "NoDataError", "YahooFinance", "ahistory", "history", "split_history",
]
