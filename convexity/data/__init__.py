"""Standalone historical market data as pandas DataFrames.

No strategy, event runtime, backtest, or brokerage account model is required.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .coverage import build_coverage, enforce_coverage
    from .prefetch import (
        DailyPrefetcher,
        PermanentHistoryError,
        PrefetchConfig,
        daily_provider,
    )

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
    "IBKR",
    "Binance",
    "DataClient",
    "DataError",
    "DataProvider",
    "Deribit",
    "NoDataError",
    "YahooFinance",
    "ahistory",
    "history",
    "split_history",
]

from .security import (
    KnowledgeUnavailableError,
    Security,
    SecurityMaster,
    SymbolMapping,
    UniverseError,
)
from .universe import (
    HistoricalMember,
    UniverseMember,
    UniverseSnapshot,
    UniverseStore,
    universe,
)
from .universe_sources import fetch_candidate, import_snapshot_csv

__all__ += [
    "HistoricalMember",
    "KnowledgeUnavailableError",
    "Security",
    "SecurityMaster",
    "SymbolMapping",
    "UniverseError",
    "UniverseMember",
    "UniverseSnapshot",
    "UniverseStore",
    "fetch_candidate",
    "import_snapshot_csv",
    "universe",
]

__all__ += [
    "DailyPrefetcher",
    "PermanentHistoryError",
    "PrefetchConfig",
    "build_coverage",
    "enforce_coverage",
    "daily_provider",
]


def __getattr__(name):
    # The standalone DataFrame API must not import the event runtime/brokers.
    from importlib import import_module

    if name in (
        "DailyPrefetcher",
        "PermanentHistoryError",
        "PrefetchConfig",
        "daily_provider",
    ):
        module = import_module(".prefetch", __name__)
    elif name in ("build_coverage", "enforce_coverage"):
        module = import_module(".coverage", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value
