"""Exact elapsed-time Ornstein-Uhlenbeck forecasting on irregular observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .. import Covariates, TimeSeries

from dataclasses import dataclass
from typing import Any

from .. import _validation as v
from ..interfaces.base import ConvergenceError, Diagnostics, FittedForecaster
from ..interfaces.data import prepare, reject_covariates
from ..interfaces.result import ForecastResult


def elapsed(times, origin, unit):
    values = np.asarray(times)
    return (
        ((values - origin) / np.timedelta64(1, unit)).astype(float)
        if values.dtype.kind == "M"
        else (values - origin).astype(float)
    )


@dataclass(frozen=True)
class OUForecaster:
    """Independent scalar OU components; parameters use the selected time unit."""

    time_unit: str = "D"
    maxiter: int = 1000

    def __post_init__(self) -> None:
        v.choice(self.time_unit, ("s", "m", "h", "D"), "time_unit")
        v.integer(self.maxiter, "maxiter", 1)

    def fit(
        self, data: TimeSeries | ArrayLike, *, covariates: Covariates | None = None
    ):
        from scipy.optimize import minimize

        history = prepare(data, covariates)
        reject_covariates(history)
        if len(history) < 4:
            raise ValueError("OU fitting requires at least four observations")
        dt = np.diff(elapsed(history.timestamps, history.timestamps[0], self.time_unit))
        parameters, iterations, converged = [], 0, True
        for k in range(history.dimension):
            y = history.values[:, k]
            if np.std(y) == 0:
                raise ValueError(
                    "OU innovation variance is unidentified for constant history"
                )

            def objective(p):
                theta, sigma = np.exp(p[1:])
                phi = np.exp(-theta * dt)
                mean = p[0] + phi * (y[:-1] - p[0])
                variance = sigma**2 * -np.expm1(-2 * theta * dt) / (2 * theta)
                return 0.5 * np.sum(
                    np.log(2 * np.pi * variance) + (y[1:] - mean) ** 2 / variance
                )

            result = minimize(
                objective,
                [
                    y.mean(),
                    np.log(1 / dt.mean()),
                    np.log(np.std(y) / np.sqrt(dt.mean())),
                ],
                method="L-BFGS-B",
                bounds=[(None, None), (-15, 15), (-20, 20)],
                options={"maxiter": self.maxiter},
            )
            parameters.append([result.x[0], np.exp(result.x[1]), np.exp(result.x[2])])
            iterations += int(result.nit)
            converged &= bool(result.success)
        fitted = FittedForecaster(
            _OUAdapter(history, v.array(parameters, "parameters"), self),
            history,
            self,
            Diagnostics(
                "scipy.OU",
                converged,
                "Exact irregular transition likelihood",
                iterations,
            ),
            {"mu_theta_sigma": np.array(parameters), "time_unit": self.time_unit},
            "OU",
        )
        if not converged:
            raise ConvergenceError(fitted)
        return fitted


@dataclass(frozen=True)
class _OUAdapter:
    history: Any
    parameters: Any
    spec: Any
    elapsed_time = True

    def forecast(self, steps, covariates, *, rng: np.random.Generator | None = None):
        raise ValueError("OU forecasts require aligned future timestamps")

    def forecast_at(
        self, horizon, covariates, *, rng: np.random.Generator | None = None
    ):
        times = elapsed(horizon.timestamps, self.history.cutoff, self.spec.time_unit)
        mu, theta, sigma = self.parameters.T
        mean = mu + np.exp(-times[:, None] * theta) * (self.history.values[-1] - mu)
        covariance = np.zeros((mean.size, mean.size))
        for k in range(self.history.dimension):
            a, b = times[:, None], times[None, :]
            block = (
                sigma[k] ** 2
                / (2 * theta[k])
                * (np.exp(-theta[k] * abs(a - b)) - np.exp(-theta[k] * (a + b)))
            )
            coords = np.arange(len(times)) * self.history.dimension + k
            covariance[np.ix_(coords, coords)] = block
        return ForecastResult(
            mean,
            horizon=horizon,
            mean_values=mean,
            joint_covariance=covariance,
            gaussian=True,
        )
