"""Gaussian switching regression/AR specifications and fitted predictions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import (
    FloatArray,
)
from ...forecasting import TimeSeries
from .. import _validation as v
from ..interfaces import RegimePaths, RegimeResult
from .markov import MarkovChain


def design_matrix(n: int, trend, exog=None, *, offset: int = 0):
    columns = []
    if trend in ("c", "ct"):
        columns.append(np.ones((n, 1)))
    if trend in ("t", "ct"):
        columns.append(np.arange(offset + 1, offset + n + 1)[:, None])
    if exog is not None:
        columns.append(exog)
    return np.column_stack(columns) if columns else np.empty((n, 0))


@dataclass(frozen=True)
class SwitchingRegression:
    """Regression whose parameters depend on a latent Markov regime, with explicit state order."""

    regimes: int = 2
    trend: str = "c"
    switching_trend: bool = True
    switching_exog: bool = True
    switching_variance: bool = False
    maxiter: int = 200
    em_iter: int = 10

    def __post_init__(self) -> None:
        v.integer(self.regimes, "regimes", 2)
        v.choice(self.trend, ("n", "c", "t", "ct"), "trend")
        for name in ("switching_trend", "switching_exog", "switching_variance"):
            v.boolean(getattr(self, name), name)
        v.integer(self.maxiter, "maxiter", 1)
        v.integer(self.em_iter, "em_iter")

    def fit(
        self,
        data: TimeSeries | ArrayLike,
        *,
        exog=None,
        initial: ArrayLike | None = None,
        start_params=None,
    ):
        """Fit deterministic statsmodels MLE; no random optimizer restarts.

        initial is passed to statsmodels initialize_known with its native
        presample convention; inference.predicted[0] reports the first scored
        prior. The default uses stationary initialization.
        """
        from .._backends.switching import fit

        return fit(self, data, exog=exog, initial=initial, start_params=start_params)


@dataclass(frozen=True)
class SwitchingAutoregression(SwitchingRegression):
    """statsmodels centered AR: y[t]-x[t]β[S[t]] = Σφ[j,S[t]](y[t-j]-x[t-j]β[S[t-j]])+ε[t]."""

    order: int = 1
    switching_ar: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        v.integer(self.order, "order", 1)
        v.boolean(self.switching_ar, "switching_ar")


@dataclass(frozen=True)
class FittedSwitching:
    """Owned fit snapshot. Parameter uncertainty is not included in predictions."""

    specification: SwitchingRegression
    history: TimeSeries
    chain: MarkovChain
    inference: RegimeResult
    parameters: FloatArray
    parameter_names: tuple[str, ...]
    coefficients: FloatArray
    variances: FloatArray
    autoregressive: FloatArray
    converged: bool
    aic: float
    bic: float
    _design: FloatArray = field(repr=False)
    _joint: FloatArray = field(repr=False)
    exog_columns: int = 0

    def __post_init__(self) -> None:
        k = self.chain.size
        for name in (
            "parameters",
            "coefficients",
            "variances",
            "autoregressive",
            "_design",
            "_joint",
        ):
            object.__setattr__(
                self, name, v.array(getattr(self, name), name, empty=True)
            )
        if (
            self.coefficients.shape != (k, self._design.shape[1])
            or self.variances.shape != (k,)
            or np.any(self.variances <= 0)
            or self.autoregressive.shape != (k, self.order)
            or len(self.parameters) != len(self.parameter_names)
            or self._design.shape[0] != len(self.history)
            or self.inference.filtered.shape != (len(self.history) - self.order, k)
        ):
            raise ValueError("invalid fitted switching parameter shapes or variances")
        if (
            self._joint.shape != (k,) * (max(self.order, 1) + 1)
            or np.any(self._joint < 0)
            or not np.isclose(self._joint.sum(), 1)
        ):
            raise ValueError("invalid terminal joint regime probabilities")
        object.__setattr__(
            self,
            "_joint",
            v.array(self._joint / self._joint.sum(), "joint probabilities"),
        )

    @property
    def order(self):
        return (
            self.specification.order
            if isinstance(self.specification, SwitchingAutoregression)
            else 0
        )

    def _future_design(self, steps, exog):
        steps = v.integer(steps, "steps", 1)
        if self.exog_columns:
            if exog is None:
                raise ValueError("future exog is required")
            exog = v.array(exog, "future exog", ndim=2)
            if exog.shape != (steps, self.exog_columns):
                raise ValueError(
                    "future exog must cover every future row and match training columns"
                )
        elif exog is not None:
            raise ValueError("model was fitted without exog")
        return design_matrix(
            steps, self.specification.trend, exog, offset=len(self.history)
        )

    def marginal(self, steps: int = 1, *, exog=None):
        """Exact regression marginal at the requested step; AR uses joint sampling."""
        from ...distributions import mixture, normal

        if self.order:
            raise NotImplementedError(
                "autoregressive prediction requires sample or predict"
            )
        design = self._future_design(steps, exog)
        means = self.coefficients @ design[-1]
        probabilities = self.chain.propagate(self.inference.filtered[-1], steps)
        return mixture(
            [
                normal(mean, np.sqrt(variance))
                for mean, variance in zip(means, self.variances)
            ],
            probabilities,
        )

    def sample(
        self, steps, *, draws: int = 1000, exog=None, rng: np.random.Generator
    ) -> RegimePaths:
        """Joint future draws conditional on observed data and fitted parameters."""
        v.generator(rng)
        draws = v.integer(draws, "draws", 1)
        design = self._future_design(steps, exog)
        p, k = self.order, self.chain.size
        flat = rng.choice(self._joint.size, draws, p=self._joint.ravel())
        history_regimes = np.array(np.unravel_index(flat, self._joint.shape)).T
        current = history_regimes[:, 0]
        residuals = np.empty((draws, p))
        for lag in range(p):
            residuals[:, lag] = (
                self.history.values[-lag - 1, 0]
                - self.coefficients[history_regimes[:, lag]] @ self._design[-lag - 1]
            )
        regimes, values = np.empty((draws, steps), dtype=int), np.empty((draws, steps))
        for t in range(steps):
            cumulative = self.chain.transition[current].cumsum(axis=1)
            cumulative[:, -1] = 1
            current = (rng.random(draws)[:, None] >= cumulative).sum(axis=1)
            baseline = self.coefficients[current] @ design[t]
            residual = np.sum(self.autoregressive[current] * residuals, axis=1)
            residual += rng.normal(size=draws) * np.sqrt(self.variances[current])
            values[:, t], regimes[:, t] = baseline + residual, current
            if p:
                residuals[:, 1:] = residuals[:, :-1]
                residuals[:, 0] = residual
        return RegimePaths(regimes, values)

    def predict(
        self, horizon, *, draws: int = 1000, exog=None, rng: np.random.Generator
    ):
        """Existing ForecastResult with empirical joint paths; sparse horizons allowed."""
        from ...forecasting import ForecastHorizon, ForecastResult

        h = (
            horizon
            if isinstance(horizon, ForecastHorizon)
            else ForecastHorizon(horizon)
        )
        h = h.align(self.history)
        result = self.sample(h.maximum, draws=draws, exog=exog, rng=rng)
        return ForecastResult.from_paths(
            result.values[:, h.steps - 1, None],
            horizon=h,
            model_identity=type(self.specification).__name__,
            cutoff=self.history.cutoff,
            metadata={
                "backend": "statsmodels",
                "parameter_uncertainty": False,
                "probability_representation": "empirical joint paths",
            },
        )

    def __init__(
        self,
        specification: SwitchingRegression,
        history: TimeSeries,
        chain: MarkovChain,
        inference: RegimeResult,
        parameters: ArrayLike,
        parameter_names: tuple[str, ...],
        coefficients: ArrayLike,
        variances: ArrayLike,
        autoregressive: ArrayLike,
        converged: bool,
        aic: float,
        bic: float,
        _design: ArrayLike,
        _joint: ArrayLike,
        exog_columns: int = 0,
    ) -> None:
        object.__setattr__(self, "specification", specification)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "chain", chain)
        object.__setattr__(self, "inference", inference)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "parameter_names", parameter_names)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "variances", variances)
        object.__setattr__(self, "autoregressive", autoregressive)
        object.__setattr__(self, "converged", converged)
        object.__setattr__(self, "aic", aic)
        object.__setattr__(self, "bic", bic)
        object.__setattr__(self, "_design", _design)
        object.__setattr__(self, "_joint", _joint)
        object.__setattr__(self, "exog_columns", exog_columns)
        self.__post_init__()
