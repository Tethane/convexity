"""statsmodels performs Hamilton/Kim inference and maximum likelihood fitting."""

from typing import Any
import numpy as np
from .. import _validation as v


def fit(spec, data, *, exog=None, initial=None, start_params=None):
    from ...forecasting.interfaces.data import prepare, reject_covariates
    from ..builtins.switching import (
        FittedSwitching,
        SwitchingAutoregression,
        design_matrix,
    )
    from ..builtins.markov import MarkovChain
    from ..interfaces import RegimeResult
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    from statsmodels.tsa.regime_switching.markov_autoregression import (
        MarkovAutoregression,
    )

    series = prepare(data)
    reject_covariates(series)
    if series.dimension != 1:
        raise ValueError("switching regression requires one target")
    order = spec.order if isinstance(spec, SwitchingAutoregression) else 0
    if len(series) <= order + 1:
        raise ValueError("insufficient observations for autoregressive order")
    if exog is not None:
        exog = v.array(exog, "exog", ndim=2)
        if len(exog) != len(series):
            raise ValueError("exog must align with observations")
    options: dict[str, Any] = dict(
        k_regimes=spec.regimes,
        trend=spec.trend,
        exog=exog,
        switching_trend=spec.switching_trend,
        switching_exog=spec.switching_exog,
        switching_variance=spec.switching_variance,
    )
    if order:
        model = MarkovAutoregression(
            series.values[:, 0], order=order, switching_ar=spec.switching_ar, **options
        )
    else:
        model = MarkovRegression(series.values[:, 0], **options)
    if initial is not None:
        initial = v.probabilities(initial, "initial")
        if initial.shape != (spec.regimes,):
            raise ValueError("initial must match regime count")
        model.initialize_known(initial)
    if start_params is not None:
        start_params = v.array(start_params, "start_params", ndim=1)
        if start_params.shape != (model.k_params,):
            raise ValueError(
                "start_params must match statsmodels transformed parameter order"
            )
    result: Any = model.fit(
        start_params=start_params,
        maxiter=spec.maxiter,
        em_iter=spec.em_iter,
        disp=False,
        search_reps=0,
    )
    params = np.asarray(result.params)
    coefficients = np.array(
        [params[model.parameters[j, "exog"]] for j in range(spec.regimes)]
    )
    variances = np.array(
        [params[model.parameters[j, "variance"]][0] for j in range(spec.regimes)]
    )
    ar = (
        np.array(
            [params[model.parameters[j, "autoregressive"]] for j in range(spec.regimes)]
        )
        if order
        else np.empty((spec.regimes, 0))
    )
    inference = RegimeResult(
        result.predicted_marginal_probabilities,
        result.filtered_marginal_probabilities,
        result.llf_obs,
        result.smoothed_marginal_probabilities,
        timestamps=series.timestamps[order:],
    )
    # statsmodels is destination-by-source; the public chain is source-by-destination.
    chain = MarkovChain(np.asarray(result.regime_transition)[..., 0].T)
    return FittedSwitching(
        spec,
        series,
        chain,
        inference,
        params,
        tuple(model.param_names),
        coefficients,
        variances,
        ar,
        bool(result.mle_retvals.get("converged", False)),
        float(result.aic),
        float(result.bic),
        design_matrix(len(series), spec.trend, exog),
        np.asarray(result.filtered_joint_probabilities)[..., -1],
        0 if exog is None else exog.shape[1],
    )
