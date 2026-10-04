"""Regimes, catalysts and event models (family 12).

Distribution and TimeSeries composition, statsmodels switching fits, explicit
random generators, and numeric event times in a caller-selected common unit.
"""
from . import builtins, interfaces
from .builtins import (MarkovChain, HiddenMarkovModel, SwitchingDynamics, SwitchingRegression,
                       SwitchingAutoregression, FittedSwitching, ConstantIntensity,
                       PiecewiseConstantIntensity, PoissonProcess, ExponentialHawkes,
                       ScheduledEvent, EventSchedule, MarkedPointProcess)
from .interfaces import (EventProcess, Intensity, RegimeInference, infer_regimes, sample_events,
                         MarkedEvents, RegimePaths, RegimeResult)

__all__ = ['builtins', 'interfaces', 'MarkovChain', 'HiddenMarkovModel', 'SwitchingDynamics',
           'SwitchingRegression', 'SwitchingAutoregression', 'FittedSwitching', 'ConstantIntensity',
           'PiecewiseConstantIntensity', 'PoissonProcess', 'ExponentialHawkes', 'ScheduledEvent',
           'EventSchedule', 'MarkedPointProcess', 'EventProcess', 'Intensity', 'RegimeInference',
           'infer_regimes', 'sample_events', 'MarkedEvents', 'RegimePaths', 'RegimeResult']
