"""ARCH estimation and forecasts, retaining return and innovation variance units."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np

from ...distributions import Distribution, empirical
from ...forecasting import _validation as v
from ...forecasting.interfaces.data import prepare, reject_covariates
from ..interfaces import (
    ConvergenceError,
    Diagnostics,
    FittedVolatilityForecaster,
    VolatilityForecastResult,
)
from .innovations import backend_distribution, InnovationAdapter


def make_model(spec, history):
    from arch import arch_model
    from arch.univariate import ConstantVariance, EWMAVariance

    cov = history.covariates
    x = None if cov is None else cov.known_future
    mean: Any = {"zero": "Zero", "constant": "Constant", "ar": "AR"}[spec.mean]
    if x is not None:
        mean = "ARX" if spec.mean == "ar" else "LS"
    model = arch_model(
        np.array(history.values[:, 0]),
        x=None if x is None else np.array(x),
        mean=mean,
        lags=spec.lags,
        vol="EGARCH" if spec.model == "egarch" else "GARCH",
        p=max(1, spec.p),
        o=spec.o,
        q=spec.q,
        rescale=spec.rescale,
    )
    model.distribution = backend_distribution(spec.innovation)
    if spec.model == "constant":
        model.volatility = ConstantVariance()
    elif spec.model == "ewma":
        model.volatility = EWMAVariance(spec.decay)
    return model


@dataclass(frozen=True)
class ARCHAdapter:
    model: Any
    spec: Any

    def forecast(self, steps, covariates, *, rng=None):
        spec, model = self.spec, self.model
        unsupported_analytic = steps > 1 and (
            spec.model == "egarch"
            or (spec.o > 0 and spec.innovation == "skew_student_t")
        )
        method = spec.method
        if method == "auto":
            method = "simulation" if unsupported_analytic else "analytic"
        if method == "analytic" and unsupported_analytic:
            raise NotImplementedError(
                "multi-step EGARCH and skewed asymmetric GARCH require simulation or bootstrap"
            )
        options = {}
        shape_parameters = (
            np.asarray(model.params)[-model.model.distribution.num_params :]
            if model.model.distribution.num_params
            else np.empty(0)
        )
        if method != "analytic":
            rng = v.generator(rng)
            if method == "simulation":
                options["rng"] = backend_distribution(
                    spec.innovation, rng=rng
                ).simulate(shape_parameters)
            else:
                # arch's bootstrap API requires a legacy RandomState. It is local
                # and seeded exclusively from the caller's Generator.
                options["random_state"] = np.random.RandomState(
                    rng.integers(0, 2**32, dtype=np.uint32)
                )
        if covariates is not None:
            options["x"] = np.asarray(covariates).T[:, None, :]
        result = model.forecast(
            horizon=steps,
            method=method,
            simulations=spec.simulations,
            reindex=False,
            **options,
        )
        scale = float(model.model.scale)
        mean = np.asarray(result.mean)[-1, :, None] / scale
        variance = np.asarray(result.variance)[-1, :, None] / scale**2
        residual_variance = np.asarray(result.residual_variance)[-1, :, None] / scale**2
        metadata = {
            "method": method,
            "innovation": spec.innovation,
            "input_scale": scale,
            "uncertainty": "conditional on fitted parameters and supplied future covariates",
            "variance_units": "squared input-return units per observation",
        }
        if method != "analytic":
            paths = np.asarray(result.simulations.values)[-1, :, :, None] / scale
            variance_paths = (
                np.asarray(result.simulations.residual_variances)[-1, :, :, None]
                / scale**2
            )
            metadata["probability_representation"] = "empirical joint paths"
            metadata["moment_interpretation"] = (
                "finite simulation averages; population moments are not guaranteed"
            )
            return VolatilityForecastResult(
                mean,
                paths=paths,
                conditional_variance=residual_variance,
                conditional_variance_paths=variance_paths,
                metadata=metadata,
            )
        if steps == 1:
            law = (
                Distribution(
                    InnovationAdapter(
                        spec.innovation,
                        shape_parameters,
                        float(mean[0, 0]),
                        float(np.sqrt(variance[0, 0])),
                    )
                )
                if variance[0, 0] > 0
                else empirical([float(mean[0, 0])])
            )
            return VolatilityForecastResult(
                mean,
                marginal_laws=(law,),
                conditional_variance=residual_variance,
                metadata=metadata,
            )
        # Even normal innovations do not make future stochastic-volatility
        # mixtures Gaussian. These multi-step analytic outputs supply moments.
        return VolatilityForecastResult(
            mean,
            mean_values=mean,
            variance_values=variance,
            conditional_variance=residual_variance,
            metadata=metadata,
        )


def fit_arch(spec, data, covariates):
    history = prepare(data, covariates, missing="raise")
    reject_covariates(history, known=spec.mean != "zero")
    if history.dimension != 1:
        raise ValueError("ARCH models require one return target")
    if len(history) < max(10, spec.lags + max(spec.p, spec.o, spec.q) + 3):
        raise ValueError("insufficient observations for the mean/variance lags")
    if np.var(history.values) == 0:
        raise ValueError("volatility fitting requires nonconstant returns")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = make_model(spec, history).fit(
            disp="off", update_freq=0, **deepcopy(spec.fit_options)
        )
    scale = float(result.model.scale)
    optimizer = result.optimization_result
    converged = result.convergence_flag == 0
    diagnostics = Diagnostics(
        "arch." + spec.model,
        converged,
        str(optimizer.get("message", "Backend fit converged")),
        optimizer.get("nit"),
        tuple(str(w.message) for w in caught),
        np.asarray(result.resid) / scale,
    )
    parameters = {
        "coefficients": result.params.to_dict(),
        "scale": scale,
        "coefficient_units": "arch internal scaled-return units",
        "conditional_volatility": np.asarray(result.conditional_volatility) / scale,
        "standardized_residuals": np.asarray(result.std_resid),
        "innovation_parameters": {
            name: float(np.asarray(result.params[name]).item())
            for name in result.model.distribution.parameter_names()
        },
        "loglikelihood": float(result.loglikelihood + result.nobs * np.log(scale)),
        "effective_nobs": result.nobs,
    }
    fitted = FittedVolatilityForecaster(
        ARCHAdapter(result, spec),
        history,
        spec,
        diagnostics,
        parameters,
        "arch." + spec.model,
    )
    if spec.check_convergence and not converged:
        raise ConvergenceError(fitted)
    return fitted
