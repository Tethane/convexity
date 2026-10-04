"""Owned problem/result records and structural extension protocols for processes.

Ordinary workflows may import these names directly from convexity.math.processes.
See docs/reference/processes.md for public signatures and mathematical conventions.
"""
from .base import StochasticProcess, ItoProcess, transition, diffusion_generator, sample_paths
from .results import Transition, ProcessPaths

__all__ = ['StochasticProcess', 'ItoProcess', 'transition', 'diffusion_generator',
           'sample_paths', 'Transition', 'ProcessPaths']
