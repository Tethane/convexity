"""Concrete specifications and convenience functions for state_space.

Ordinary workflows may import these names directly from convexity.math.state_space.
See docs/reference/state_space.md for public signatures and mathematical conventions.
"""
from .dynamics import continuous_linear_transition, local_level, local_linear_trend
from .filters import ExtendedKalmanFilter, KalmanFilter, ParticleFilter, UnscentedKalmanFilter
from .models import (DistributionObservation, DistributionTransition, GaussianObservation, GaussianTransition,
                     LinearGaussianObservation, LinearGaussianTransition, linear_gaussian)
from .simulation import simulate

__all__ = ['continuous_linear_transition', 'local_level', 'local_linear_trend', 'ExtendedKalmanFilter',
           'KalmanFilter', 'ParticleFilter', 'UnscentedKalmanFilter', 'DistributionObservation',
           'DistributionTransition', 'GaussianObservation', 'GaussianTransition', 'LinearGaussianObservation',
           'LinearGaussianTransition', 'linear_gaussian', 'simulate']
