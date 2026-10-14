"""Concrete specifications and convenience functions for simulation.

Ordinary workflows may import these names directly from convexity.math.simulation.
"""

from .drivers import gaussian_innovations, brownian_increments
from .paths import simulate, sample_paths, integrate, geometric_brownian_paths
from .gaussian import gaussian_path_distribution
from .estimation import estimate, estimate_scenarios, control_variates, qmc_expectation
from .observations import SimulatedObservations, simulate_observations

__all__ = [
    "gaussian_innovations",
    "brownian_increments",
    "simulate",
    "sample_paths",
    "integrate",
    "geometric_brownian_paths",
    "gaussian_path_distribution",
    "estimate",
    "estimate_scenarios",
    "control_variates",
    "qmc_expectation",
    "SimulatedObservations",
    "simulate_observations",
]
