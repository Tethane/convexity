"""State-space shapes, covariance checks, and explicit elapsed-time semantics."""
import numpy as np

from .._validation import array, boolean, choice, generator, integer, scalar
from ..forecasting.interfaces.data import TimeSeries, reject_covariates


def shaped(value, shape, name):
    out = array(value, name)
    if out.shape != shape:
        raise ValueError(f'{name} must have shape {shape}; got {out.shape}')
    return out


def covariance(value, dimension, name='covariance', *, positive=False):
    from .._covariance import psd
    return psd(shaped(value, (dimension, dimension), name), name, definite=positive)


def states(value, dimension):
    out = array(value, 'states', ndim=2)
    if out.shape[1] != dimension:
        raise ValueError('states must have shape (particles,state_dimension)')
    return out


def data(value, dimension):
    series = value if isinstance(value, TimeSeries) else TimeSeries(value)
    reject_covariates(series)
    if series.dimension != dimension:
        raise ValueError('observations must match observation dimension')
    return series


def elapsed(times, mode, unit):
    choice(mode, ('discrete', 'elapsed'), 'time_mode')
    if mode == 'discrete':
        if unit is not None:
            raise ValueError('time_unit applies only to elapsed models')
        if len(times) > 2:
            differences = np.diff(times)
            regular = (np.all(differences == differences[0]) if times.dtype.kind == 'M'
                       else np.allclose(differences, differences[0], rtol=1e-10, atol=0))
            if not regular:
                raise ValueError('irregular timestamps require time_mode="elapsed"')
        return np.r_[0., np.ones(len(times) - 1)]
    if times.dtype.kind == 'M':
        if not isinstance(unit, np.timedelta64) or np.isnat(unit):
            raise ValueError('datetime elapsed models require a positive numpy.timedelta64 time_unit')
        try:
            if unit <= np.timedelta64(0, 'ns'):
                raise ValueError('datetime elapsed models require a positive time_unit')
            result = np.diff(times) / unit
        except (TypeError, OverflowError) as exc:
            raise ValueError('time_unit must be a fixed elapsed duration') from exc
    else:
        if unit is not None:
            raise ValueError('numeric timestamps already specify the elapsed time unit')
        # Subtract integers before conversion to preserve large timestamp gaps.
        result = np.diff(times).astype(float)
    if np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise ValueError('elapsed times must be positive and representable')
    return np.r_[0., result]


def parameter(value, shape, name, *, cov=False):
    """Copy constants now; validate time-dependent callbacks when evaluated."""
    if callable(value):
        return value
    return covariance(value, shape[0], name) if cov else shaped(value, shape, name)


def evaluate(value, step, shape, name, *, cov=False):
    result = value(step) if callable(value) else value
    return covariance(result, shape[0], name) if cov else shaped(result, shape, name)
