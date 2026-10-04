"""statsmodels linear filtering and fixed-interval smoothing adapter."""
import numpy as np
from typing import Any

from .. import _validation as v
from ..interfaces.results import FilterResult, GaussianState, StateSequence


def run(model, observations, *, smooth):
    from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother
    if not isinstance(model.initial, GaussianState):
        raise TypeError('Gaussian filters require an explicit GaussianState initial law')
    if not all(callable(getattr(kernel, 'coefficients', None)) for kernel in (model.transition, model.observation)):
        raise TypeError('Kalman inference requires linear Gaussian kernels with coefficients')
    series = v.data(observations, model.observation_dimension)
    steps = model.steps(series)
    n, d, k = len(series), model.state_dimension, model.observation_dimension
    transitions, offsets, noises = [], [], []
    designs, intercepts, obs_noises = [], [], []
    for i, step in enumerate(steps):
        # statsmodels stores the outgoing transition at row i, whereas our
        # kernel receives the destination step. The final transition is unused.
        if i + 1 < n:
            matrix, offset = model.transition.coefficients(steps[i + 1])
            noise = model.transition.covariance(steps[i + 1])
        else:
            matrix, offset, noise = np.eye(d), np.zeros(d), np.zeros((d, d))
        transitions.append(v.shaped(matrix, (d, d), 'transition'))
        offsets.append(v.shaped(offset, (d,), 'transition offset'))
        noises.append(v.covariance(noise, d, 'transition covariance'))
        design, intercept = model.observation.coefficients(step)
        designs.append(v.shaped(design, (k, d), 'observation matrix'))
        intercepts.append(v.shaped(intercept, (k,), 'observation offset'))
        obs_noises.append(v.covariance(model.observation.covariance(step), k, 'observation noise'))
    backend = KalmanSmoother(k_endog=k, k_states=d, k_posdef=d)
    backend.bind(np.ascontiguousarray(series.values.copy()))
    backend.initialize_known(model.initial.mean.copy(), model.initial.covariance.copy())
    backend['transition'] = np.moveaxis(np.array(transitions), 0, -1)
    backend['state_intercept'] = np.array(offsets).T
    backend['state_cov'] = np.moveaxis(np.array(noises), 0, -1)
    backend['selection'] = np.eye(d)
    backend['design'] = np.moveaxis(np.array(designs), 0, -1)
    backend['obs_intercept'] = np.array(intercepts).T
    backend['obs_cov'] = np.moveaxis(np.array(obs_noises), 0, -1)
    # Disable steady-state truncation so exact identities retain full precision.
    backend.tolerance = 0.
    result: Any = backend.smooth() if smooth else backend.filter()
    predicted_mean = result.predicted_state[:, :n].T
    predicted_cov = np.moveaxis(result.predicted_state_cov[:, :, :n], -1, 0)
    obs_mean = np.einsum('tkd,td->tk', designs, predicted_mean) + intercepts
    obs_cov = np.array([h @ p @ h.T + r for h, p, r in zip(designs, predicted_cov, obs_noises)])
    for y, cov in zip(series.values, obs_cov):
        keep = np.flatnonzero(~np.isnan(y))
        if len(keep):
            v.covariance(cov[np.ix_(keep, keep)], len(keep), 'innovation covariance', positive=True)
    return FilterResult(series,
        StateSequence(result.filtered_state.T, np.moveaxis(result.filtered_state_cov, -1, 0)),
        result.llf_obs, 'statsmodels', 'kalman',
        predicted=StateSequence(predicted_mean, predicted_cov),
        smoothed=StateSequence(result.smoothed_state.T, np.moveaxis(result.smoothed_state_cov, -1, 0)) if smooth else None,
        observation_mean=obs_mean, observation_covariance=obs_cov, state_names=model.state_names)
