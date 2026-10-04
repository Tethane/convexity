"""Owned problem/result records and structural extension protocols for state_space.

Ordinary workflows may import these names directly from convexity.math.state_space.
See docs/reference/state_space.md for public signatures and mathematical conventions.
"""
from .models import ObservationModel, StateEstimator, StateSpaceModel, TimeStep, TransitionDensityModel, TransitionModel, filter_states
from .results import FilterResult, GaussianState, SimulationResult, StateSequence

__all__ = ['ObservationModel', 'StateEstimator', 'StateSpaceModel', 'TimeStep', 'TransitionDensityModel', 'TransitionModel',
           'filter_states', 'FilterResult', 'GaussianState', 'SimulationResult', 'StateSequence']
