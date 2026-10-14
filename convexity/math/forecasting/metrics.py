"""Point, marginal-probability, and joint-path scores with explicit reductions."""

import numpy as np

from . import _validation as v


def _pair(actual, prediction):
    a, b = v.array(actual, "actual"), v.array(prediction, "prediction")
    if a.shape != b.shape:
        raise ValueError("actual and prediction must have identical shapes")
    return a, b


def mae(actual, prediction, *, axis=None):
    a, b = _pair(actual, prediction)
    return np.mean(abs(a - b), axis=axis)


def rmse(actual, prediction, *, axis=None):
    a, b = _pair(actual, prediction)
    return np.sqrt(np.mean((a - b) ** 2, axis=axis))


def smape(actual, prediction, *, axis=None):
    """Symmetric percentage error on [0,200]; both-zero coordinates score zero."""
    a, b = _pair(actual, prediction)
    denominator = abs(a) + abs(b)
    errors = np.divide(
        200 * abs(a - b), denominator, out=np.zeros_like(a), where=denominator > 0
    )
    return np.mean(errors, axis=axis)


def mase(actual, prediction, training, *, seasonality=1, axis=None):
    a, b = _pair(actual, prediction)
    history = v.matrix(training, "training")
    period = v.integer(seasonality, "seasonality", 1)
    if len(history) <= period:
        raise ValueError("MASE requires more training rows than the seasonality")
    scale = abs(history[period:] - history[:-period]).mean(axis=0)
    if np.any(scale == 0):
        raise ValueError("MASE is undefined for a zero training naive-error scale")
    if a.shape[-1:] != scale.shape:
        raise ValueError("MASE target axes must match training columns")
    return np.mean(abs(a - b) / scale, axis=axis)


def pinball_loss(actual, prediction, probability, *, axis=None):
    a, b = _pair(actual, prediction)
    p = v.probability(probability)
    error = a - b
    return np.mean(np.maximum(p * error, (p - 1) * error), axis=axis)


def _actual(forecast, actual):
    actual = v.matrix(actual, "actual")
    if actual.shape != forecast.shape:
        raise ValueError("actual values must align with forecast (horizon,target)")
    return actual


def crps(forecast, actual, *, axis=None):
    """Closed-form Gaussian or exact finite-ensemble marginal CRPS."""
    actual = _actual(forecast, actual)
    if forecast.paths is not None:
        x = np.sort(forecast.paths, axis=0) - actual
        n = len(x)
        ranks = (2 * np.arange(1, n + 1) - n - 1)[:, None, None]
        scores = abs(x).mean(axis=0) - (ranks * x).sum(axis=0) / n**2
    elif forecast.gaussian:
        from scipy.special import ndtr

        sigma = np.sqrt(forecast.variance())
        diff = actual - forecast.mean()
        z = np.divide(diff, sigma, out=np.zeros_like(diff), where=sigma > 0)
        scores = sigma * (
            z * (2 * ndtr(z) - 1)
            + 2 * np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)
            - 1 / np.sqrt(np.pi)
        )
        scores = np.where(sigma > 0, scores, abs(diff))
    else:
        raise NotImplementedError("CRPS requires Gaussian marginals or empirical paths")
    return np.mean(scores, axis=axis)


def log_score(forecast, actual, *, axis=None):
    """Negative marginal log density; not a joint path log likelihood."""
    actual = _actual(forecast, actual)
    scores = np.empty_like(actual)
    for h, k in np.ndindex(actual.shape):
        marginal = forecast.marginal(h, k)
        value = actual[h, k] if not marginal.event_shape else [actual[h, k]]
        scores[h, k] = -float(marginal.log_density(value))
    return np.mean(scores, axis=axis)


def interval_coverage(forecast, actual, probability=0.9, *, axis=None):
    actual = _actual(forecast, actual)
    bounds = forecast.interval(probability)
    return np.mean((actual >= bounds[..., 0]) & (actual <= bounds[..., 1]), axis=axis)


def sharpness(forecast, probability=0.9, *, axis=None):
    bounds = forecast.interval(probability)
    return np.mean(bounds[..., 1] - bounds[..., 0], axis=axis)


def energy_score(forecast, actual):
    """Exact energy score of the empirical joint path law; O(draws**2)."""
    from scipy.spatial.distance import pdist

    actual = _actual(forecast, actual)
    if forecast.paths is None:
        raise NotImplementedError("energy_score requires explicit sampled paths")
    paths = forecast.paths.reshape(len(forecast.paths), -1)
    return float(
        np.linalg.norm(paths - actual.ravel(), axis=1).mean()
        - pdist(paths).sum() / len(paths) ** 2
    )


def calibration(forecast, actual, *, levels=(0.1, 0.5, 0.9)):
    """Nominal quantiles and observed fractions at/below them (pooled coordinates)."""
    actual = _actual(forecast, actual)
    levels = v.levels(levels)
    return v.array(
        np.column_stack(
            (levels, [np.mean(actual <= forecast.quantile(p)) for p in levels])
        ),
        "calibration",
    )
