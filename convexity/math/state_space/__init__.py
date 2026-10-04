"""State-space models, filtering and smoothing (family 10).

TimeSeries inputs and Distribution outputs compose with existing families.
statsmodels, FilterPy, particles and SciPy are loaded only when needed.
"""
from ..forecasting.interfaces import TimeSeries
from . import builtins, interfaces
from .builtins import (DistributionObservation, DistributionTransition, ExtendedKalmanFilter,
                       GaussianObservation, GaussianTransition, KalmanFilter, LinearGaussianObservation,
                       LinearGaussianTransition, ParticleFilter, UnscentedKalmanFilter,
                       continuous_linear_transition, linear_gaussian, local_level, local_linear_trend, simulate)
from .interfaces import (FilterResult, GaussianState, ObservationModel, SimulationResult, StateEstimator,
                         StateSequence, StateSpaceModel, TimeStep, TransitionDensityModel, TransitionModel, filter_states)
from ._backends.particle import ParticleDegeneracyError

__all__ = ['builtins', 'interfaces', 'TimeSeries', 'DistributionObservation', 'DistributionTransition',
           'ExtendedKalmanFilter', 'GaussianObservation', 'GaussianTransition', 'KalmanFilter',
           'LinearGaussianObservation', 'LinearGaussianTransition', 'ParticleFilter', 'UnscentedKalmanFilter',
           'continuous_linear_transition', 'linear_gaussian', 'local_level', 'local_linear_trend', 'simulate',
           'FilterResult', 'GaussianState', 'ObservationModel', 'SimulationResult', 'StateEstimator',
           'StateSequence', 'StateSpaceModel', 'TimeStep', 'TransitionDensityModel', 'TransitionModel', 'filter_states', 'ParticleDegeneracyError']
