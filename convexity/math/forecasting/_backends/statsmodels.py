"""Classical fitting and honest forecast uncertainty from statsmodels."""

from dataclasses import dataclass, replace
from copy import deepcopy
from typing import Any
import warnings

import numpy as np

from ..interfaces.base import ConvergenceError, Diagnostics, FittedForecaster
from ..interfaces.data import prepare, reject_covariates
from ..interfaces.result import ForecastResult


@dataclass(frozen=True)
class ClassicalAdapter:
    model: Any
    kind: str
    history: Any
    spec: Any

    def forecast(self, steps, covariates, *, rng=None):
        model, kind = self.model, self.kind
        if kind == "var":
            mean = model.forecast(
                self.history[-max(1, model.k_ar) :], steps, exog_future=covariates
            )
            # Moving-average impulse responses define the full conditional path law.
            ma = model.ma_rep(maxn=steps - 1)
            covariance = _path_covariance(ma, model.sigma_u, steps)
            return ForecastResult(
                mean, mean_values=mean, joint_covariance=covariance, gaussian=True
            )
        if kind == "vecm":
            mean = model.predict(steps=steps, exog_fc=covariates)
            covariance = _path_covariance(
                model.ma_rep(maxn=steps - 1), model.sigma_u, steps
            )
            return ForecastResult(
                mean, mean_values=mean, joint_covariance=covariance, gaussian=True
            )
        if kind in ("exponential_smoothing", "ets"):
            mean = np.asarray(model.forecast(steps)).reshape(steps, 1)
            generator = np.random.default_rng(self.spec.seed) if rng is None else rng
            # Explicit innovations work with both older random_state and newer
            # rng statsmodels APIs and never mutate a process-global generator.
            residuals = np.asarray(model.resid)
            residuals = residuals[np.isfinite(residuals)]
            errors = generator.normal(
                0, np.sqrt(np.mean(residuals**2)), size=(steps, self.spec.samples)
            )
            simulated = model.simulate(
                steps, anchor="end", repetitions=self.spec.samples, random_errors=errors
            )
            paths = np.asarray(simulated).reshape(steps, self.spec.samples).T[..., None]
            return ForecastResult(
                mean,
                paths=paths,
                metadata={"uncertainty": "fitted ETS innovation simulation"},
            )
        pred = model.get_forecast(
            steps=steps, **({} if covariates is None else {"exog": covariates})
        )
        mean = np.asarray(pred.predicted_mean).reshape(steps, -1)
        variance = np.asarray(pred.var_pred_mean)
        if variance.ndim == 3:
            variance = np.diagonal(variance, axis1=1, axis2=2)
        variance = variance.reshape(mean.shape)
        pr = pred.prediction_results
        matrices = pr.oos_results if pr.oos_results is not None else pr.results
        state_cov = pr.predicted_state_cov
        k = mean.shape[1]
        covariance = np.zeros((steps * k, steps * k))

        def at(matrix, t):
            return matrix[..., 0 if matrix.shape[-1] == 1 else t]

        for i in range(steps):
            for j in range(i, steps):
                if i == j:
                    block = np.asarray(pred.var_pred_mean)[i]
                    block = np.asarray(block).reshape(k, k)
                else:
                    cross = np.array(state_cov[..., i])
                    for t in range(i, j):
                        cross = cross @ at(matrices.transition, t).T
                    block = at(matrices.design, i) @ cross @ at(matrices.design, j).T
                covariance[i * k : (i + 1) * k, j * k : (j + 1) * k] = block
                covariance[j * k : (j + 1) * k, i * k : (i + 1) * k] = block.T
        return ForecastResult(
            mean,
            mean_values=mean,
            joint_covariance=covariance,
            gaussian=True,
            metadata={
                "uncertainty": "joint Gaussian law conditional on fitted parameters"
            },
        )

    def update(self, history, *, mode):
        if mode != "incremental" or not hasattr(self.model, "append"):
            raise ValueError(
                "incremental filtering requires a state-space results.append engine"
            )
        n = len(self.history)
        y = history.values[n:]
        if history.dimension == 1:
            y = y[:, 0]
        cov = history.covariates
        x = None if cov is None or cov.known_future is None else cov.known_future[n:]
        model = self.model.append(y, exog=x, refit=False)
        return FittedForecaster(
            ClassicalAdapter(model, self.kind, history.values, self.spec),
            history,
            self.spec,
            Diagnostics(
                "statsmodels." + self.kind,
                message="Incremental filtering; fixed parameters",
            ),
            {"params": np.asarray(model.params)},
            "statsmodels." + self.kind,
        )


def _path_covariance(ma, innovation, steps):
    k = innovation.shape[0]
    propagation = np.zeros((steps * k, steps * k))
    for h in range(steps):
        for j in range(h + 1):
            propagation[h * k : (h + 1) * k, j * k : (j + 1) * k] = ma[h - j]
    return propagation @ np.kron(np.eye(steps), innovation) @ propagation.T


def fit_classical(spec, data, covariates):
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.statespace.varmax import VARMAX
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.tsa.vector_ar.vecm import VECM

    history = prepare(data, covariates, missing=spec.missing)
    kind = spec.model
    reject_covariates(history, known=kind not in ("exponential_smoothing", "ets"))
    multi = kind in ("var", "varmax", "vecm", "dynamic_factor")
    if (multi and history.dimension < 2) or (not multi and history.dimension != 1):
        raise ValueError(
            "this model requires " + ("multiple targets" if multi else "one target")
        )
    y = np.array(history.values if multi else history.values[:, 0])
    x = None if history.covariates is None else history.covariates.known_future
    factories = dict(
        arima=ARIMA,
        sarimax=SARIMAX,
        exponential_smoothing=ExponentialSmoothing,
        ets=ETSModel,
        var=VAR,
        varmax=VARMAX,
        vecm=VECM,
        structural=UnobservedComponents,
        dynamic_factor=DynamicFactor,
    )
    options = deepcopy(spec.model_options)
    if x is not None:
        options["exog"] = np.array(x)
    fit_options = deepcopy(spec.fit_options)
    if kind in ("sarimax", "varmax", "structural", "dynamic_factor", "ets"):
        fit_options.setdefault("disp", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        factory: Any = factories[kind]
        model: Any = factory(y, **options).fit(**fit_options)
    details = getattr(model, "mle_retvals", {}) or {}
    converged = bool(details.get("converged", details.get("success", True)))
    residuals = np.asarray(model.resid).reshape(-1, history.dimension)
    diagnostics = Diagnostics(
        "statsmodels." + kind,
        converged,
        "Backend fit converged" if converged else "Backend optimizer did not converge",
        details.get("iterations", details.get("nit")),
        tuple(str(w.message) for w in caught),
        residuals,
    )
    parameters = {"model_options": spec.model_options, "fit_options": spec.fit_options}
    for name in (
        "params",
        "alpha",
        "beta",
        "gamma",
        "sigma_u",
        "coefs",
        "k_ar",
        "aic",
        "bic",
    ):
        if hasattr(model, name):
            parameters[name] = getattr(model, name)
    fitted = FittedForecaster(
        ClassicalAdapter(model, kind, history.values, spec),
        history,
        spec,
        diagnostics,
        parameters,
        "statsmodels." + kind,
    )
    if spec.check_convergence and not converged:
        raise ConvergenceError(fitted)
    return fitted
