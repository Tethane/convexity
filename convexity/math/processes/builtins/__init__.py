"""Concrete specifications and convenience functions for processes.

Ordinary workflows may import these names directly from convexity.math.processes.
See docs/math/reference/processes.md for public signatures and mathematical conventions.
"""
from .gaussian import BrownianMotion, ArithmeticBrownianMotion, OrnsteinUhlenbeck, LinearDiffusion, VectorOrnsteinUhlenbeck
from .positive import GeometricBrownianMotion, CIR
from .heston import Heston
from .custom import ItoDiffusion
from .jumps import AdditiveJumpDiffusion, MertonJumpDiffusion

__all__ = ['BrownianMotion', 'ArithmeticBrownianMotion', 'OrnsteinUhlenbeck',
           'LinearDiffusion', 'VectorOrnsteinUhlenbeck', 'GeometricBrownianMotion',
           'CIR', 'Heston', 'ItoDiffusion', 'AdditiveJumpDiffusion', 'MertonJumpDiffusion']
