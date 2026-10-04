"""Standalone mathematical tools, organized by mathematical family."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import types
    from . import risk, utility, optimization, robust, control, policies, execution, wealth, diagnostics, reinforcement, measures, contracts, curves, pricing, implied_volatility, calibration, stopping, sensitivities, fixed_income, rates, credit, bayesian, dependence, distributions, estimation, forecasting, gaussian_processes, learning, numerical, processes, regimes, rough, simulation, state_space, transforms, volatility

__all__ = ["numerical", "distributions", "estimation", "transforms", "dependence", "learning",
           "gaussian_processes", "forecasting", "volatility", "state_space", "bayesian", "regimes", "processes", "rough", "simulation", 'measures', 'contracts', 'curves', 'pricing', 'implied_volatility', 'calibration', 'stopping', 'sensitivities', 'fixed_income', 'rates', 'credit', 'risk', 'utility', 'optimization', 'robust', 'control', 'policies', 'execution', 'wealth', 'diagnostics', 'reinforcement']


def __getattr__(name):
    if name not in __all__ and name != "types":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(__all__) | {"types"})
