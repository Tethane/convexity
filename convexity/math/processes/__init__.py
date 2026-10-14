"""Stochastic processes and differential equations.

Explicit Ito coefficients, generators, exact laws, and labeled numerical
transitions. NumPy generators are required; optional backends load on use.
"""

from . import builtins, interfaces
from .builtins import (
    BrownianMotion,
    ArithmeticBrownianMotion,
    OrnsteinUhlenbeck,
    LinearDiffusion,
    VectorOrnsteinUhlenbeck,
    GeometricBrownianMotion,
    CIR,
    Heston,
    ItoDiffusion,
    AdditiveJumpDiffusion,
    MertonJumpDiffusion,
)
from .interfaces import (
    StochasticProcess,
    ItoProcess,
    Transition,
    ProcessPaths,
    transition,
    diffusion_generator,
    sample_paths,
)

__all__ = [
    "builtins",
    "interfaces",
    "BrownianMotion",
    "ArithmeticBrownianMotion",
    "OrnsteinUhlenbeck",
    "LinearDiffusion",
    "VectorOrnsteinUhlenbeck",
    "GeometricBrownianMotion",
    "CIR",
    "Heston",
    "ItoDiffusion",
    "AdditiveJumpDiffusion",
    "MertonJumpDiffusion",
    "StochasticProcess",
    "ItoProcess",
    "Transition",
    "ProcessPaths",
    "transition",
    "diffusion_generator",
    "sample_paths",
]
