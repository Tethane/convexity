"""statsmodels estimation and exact Gaussian-log HAR forecast propagation."""

from dataclasses import dataclass
from typing import Any, Literal
import warnings

import numpy as np

from ...forecasting import _validation as v
from ...forecasting.interfaces.data import prepare, reject_covariates
from ..interfaces import (
    Diagnostics,
    FittedVolatilityForecaster,
    VolatilityForecastResult,
)


@dataclass(frozen=True)
class LognormalPathAdapter:
    """Exact finite-dimensional exp(Gaussian) law in horizon order."""

    location: Any
    log_covariance: Any
    measure: Literal["continuous"] = "continuous"
    capabilities = frozenset(
        {"mean", "variance", "covariance", "sample", "log_density", "marginal"}
    )

    @property
    def event_shape(self):
        return self.location.shape

    def evaluate(self, operation, *args, **kwargs):
        from ...distributions import Distribution

        mean = np.exp(self.location + 0.5 * np.diag(self.log_covariance))
        if operation == "mean":
            return mean
        if operation == "variance":
            return np.expm1(np.diag(self.log_covariance)) * mean**2
        if operation == "covariance":
            return np.expm1(self.log_covariance) * np.outer(mean, mean)
        if operation == "sample":
            from ..._backends.gaussian import sample

            return np.exp(
                sample(self.location, self.log_covariance, args[0], kwargs["rng"])
            )
        if operation == "marginal":
            ix = args[0]
            return Distribution(
                LognormalPathAdapter(
                    self.location[ix], self.log_covariance[np.ix_(ix, ix)]
                )
            )
        if operation == "log_density":
            from ..._backends.gaussian import logpdf

            x = args[0]
            flat = x.reshape(-1, len(mean))
            inside = np.all((flat > 0) & np.isfinite(flat), axis=1)
            result = np.full(len(flat), -np.inf)
            if np.any(inside):
                logs = np.log(flat[inside])
                result[inside] = logpdf(
                    logs, self.location, self.log_covariance
                ) - logs.sum(axis=1)
            return result.reshape(x.shape[:-1])
        raise NotImplementedError(operation)


@dataclass(frozen=True)
class HARAdapter:
    spec: Any
    history: Any
    coefficients: Any
    innovation_variance: float

    def forecast(self, steps, covariates, *, rng=None):
        spec = self.spec
        values = list(
            np.log(self.history.values[:, 0])
            if spec.transform == "log"
            else self.history.values[:, 0]
        )
        mean = []
        for h in range(steps):
            features = [1.0, *[np.mean(values[-lag:]) for lag in spec.lags]]
            if covariates is not None:
                features.extend(covariates[h])
            value = float(np.dot(features, self.coefficients))
            values.append(value)
            mean.append(value)
        mean = np.asarray(mean)
        metadata = {
            "method": spec.method,
            "transform": spec.transform,
            "uncertainty": "conditional on fitted coefficients and supplied future covariates",
            "variance_units": "squared input-return units per observation",
        }
        if spec.transform == "level":
            if np.any(mean < 0):
                raise ValueError(
                    "level HAR predicted negative realized variance; use a log HAR model"
                )
            return VolatilityForecastResult(
                mean[:, None],
                conditional_variance=mean[:, None],
                target_kind="realized_variance",
                metadata=metadata,
            )
        # Expand window averages into ordinary AR coefficients, then obtain
        # impulse responses. Cross-horizon covariance matters before exp().
        ar = np.zeros(max(spec.lags))
        for beta, lag in zip(self.coefficients[1:], spec.lags):
            ar[:lag] += beta / lag
        impulse = np.zeros(steps)
        impulse[0] = 1.0
        for h in range(1, steps):
            impulse[h] = sum(
                ar[j - 1] * impulse[h - j] for j in range(1, min(h, len(ar)) + 1)
            )
        propagation = np.zeros((steps, steps))
        for h in range(steps):
            propagation[h, : h + 1] = impulse[: h + 1][::-1]
        covariance = self.innovation_variance * propagation @ propagation.T
        variance = np.diag(covariance)
        point = np.exp(mean + 0.5 * variance)[:, None]
        metadata["uncertainty"] += "; iid Gaussian log-variance innovations"
        if spec.method == "simulation":
            rng = v.generator(rng)
            log_paths = mean + rng.normal(
                size=(spec.simulations, steps)
            ) @ propagation.T * np.sqrt(self.innovation_variance)
            paths = np.exp(log_paths)[..., None]
            return VolatilityForecastResult(
                point,
                paths=paths,
                conditional_variance=point,
                conditional_variance_paths=paths,
                target_kind="realized_variance",
                metadata=metadata,
            )
        from ...distributions import Distribution, empirical, lognormal

        if self.innovation_variance == 0:
            law = empirical(point.ravel()[None, :])
            marginals = tuple(empirical([value]) for value in point.ravel())
        else:
            law = Distribution(
                LognormalPathAdapter(
                    v.array(mean, "log mean"), v.array(covariance, "log covariance")
                )
            )
            marginals = tuple(
                lognormal(mu, np.sqrt(var)) for mu, var in zip(mean, variance)
            )
        return VolatilityForecastResult(
            point,
            law=law,
            marginal_laws=marginals,
            conditional_variance=point,
            target_kind="realized_variance",
            metadata=metadata,
        )


def fit_har(spec, data, covariates):
    from statsmodels.regression.linear_model import OLS

    history = prepare(data, covariates, missing="raise")
    reject_covariates(history, known=True)
    if history.dimension != 1:
        raise ValueError("HAR requires one realized-variance target")
    y = history.values[:, 0]
    if np.any(y < 0) or (spec.transform == "log" and np.any(y == 0)):
        raise ValueError(
            "realized variance must be nonnegative, and strictly positive for log HAR"
        )
    y = np.log(y) if spec.transform == "log" else y
    start = max(spec.lags)
    x = None if history.covariates is None else history.covariates.known_future
    width = 1 + len(spec.lags) + (0 if x is None else x.shape[1])
    if len(y) - start <= width:
        raise ValueError(
            "HAR needs more training rows after the largest lag than coefficients"
        )
    design = np.column_stack(
        (
            np.ones(len(y) - start),
            *[
                np.array([np.mean(y[t - lag : t]) for t in range(start, len(y))])
                for lag in spec.lags
            ],
        )
    )
    if x is not None:
        design = np.column_stack((design, x[start:]))
    if np.linalg.matrix_rank(design) < width:
        raise ValueError("HAR regression design is rank deficient")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = OLS(y[start:], design).fit(cov_type=spec.covariance)
    residuals = np.full(len(y), np.nan)
    residuals[start:] = result.resid
    diagnostics = Diagnostics(
        "statsmodels.har",
        warnings=tuple(str(w.message) for w in caught),
        residuals=residuals,
        message="OLS; residuals are in fitted transform units",
    )
    coefficients = v.array(result.params, "coefficients")
    innovation_variance = float(np.dot(result.resid, result.resid) / result.df_resid)
    parameters = {
        "coefficients": coefficients,
        "parameter_covariance": np.asarray(result.cov_params()),
        "innovation_variance": innovation_variance,
        "transform": spec.transform,
        "effective_nobs": int(result.nobs),
        "residual_df": int(result.df_resid),
        "feature_names": (
            "constant",
            *[f"lag_mean_{lag}" for lag in spec.lags],
            *[f"x{i}" for i in range(0 if x is None else x.shape[1])],
        ),
    }
    return FittedVolatilityForecaster(
        HARAdapter(spec, history, coefficients, innovation_variance),
        history,
        spec,
        diagnostics,
        parameters,
        "statsmodels.har." + spec.transform,
    )
