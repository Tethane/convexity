"""Finite difference algebra and statsmodels deterministic feature adapters."""

import numpy as np

from ..interfaces import DifferenceResult, FeatureMatrix


def difference(values, *, lag, order):
    first, last = [], []
    current = values
    for _ in range(order):
        first.append(current[:lag])
        last.append(current[-lag:])
        current = current[lag:] - current[:-lag]
    return DifferenceResult(current, first, last, lag)


def integrate(values, boundaries, *, continuation):
    current = values
    for seed in reversed(boundaries):
        lag = len(seed)
        restored = np.concatenate((seed, current)).copy()
        # Each congruence class modulo lag is an independent cumulative sum.
        for offset in range(lag):
            restored[offset::lag] = np.cumsum(restored[offset::lag], axis=0)
        current = restored[lag:] if continuation else restored
    return current


def lags(values, lags):
    x = values[:, None] if values.ndim == 1 else values
    rows = np.arange(max(lags), len(x))
    result = np.concatenate([x[rows - lag] for lag in lags], axis=1)
    columns = tuple(f"x{j}_lag{lag}" for lag in lags for j in range(x.shape[1]))
    return FeatureMatrix(result, rows, columns)


def fourier(n, *, start, period, order):
    from statsmodels.tsa.deterministic import Fourier
    values = Fourier(period, order).in_sample(np.arange(n)).to_numpy(copy=True)
    # Shift the phase without constructing or fitting any preceding rows.
    for harmonic in range(1, order + 1):
        angle = 2 * np.pi * harmonic * (start % period) / period
        sine, cosine = values[:, 2 * harmonic - 2:2 * harmonic].T.copy()
        values[:, 2 * harmonic - 2] = sine * np.cos(angle) + cosine * np.sin(angle)
        values[:, 2 * harmonic - 1] = cosine * np.cos(angle) - sine * np.sin(angle)
    columns = tuple(f"{kind}_{j}" for j in range(1, order + 1) for kind in ("sin", "cos"))
    return FeatureMatrix(values, np.arange(start, start + n), columns)


def seasonal(n, *, start, period, drop_first):
    from statsmodels.tsa.deterministic import Seasonality
    values = Seasonality(period, initial_period=start % period + 1).in_sample(np.arange(n)).to_numpy()
    first = 1 if drop_first else 0
    return FeatureMatrix(values[:, first:], np.arange(start, start + n),
                         tuple(f"season_{j}" for j in range(first, period)))
