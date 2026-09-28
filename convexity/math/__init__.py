"""Standalone mathematical tools, organized by mathematical family."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import dependence, distributions, estimation, gaussian_processes, learning, numerical, transforms

__all__ = ["numerical", "distributions", "estimation", "transforms", "dependence", "learning",
           "gaussian_processes"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(__all__))
