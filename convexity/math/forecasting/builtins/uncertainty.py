"""Parameter uncertainty via an explicit parametric-bootstrap fitting distribution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .. import Covariates

from dataclasses import dataclass
from typing import Any

from .. import _validation as v
from ..interfaces.base import Diagnostics, FittedForecaster, Forecaster, fit
from ..interfaces.data import TimeSeries, prepare
from ..interfaces.result import ForecastResult


@dataclass(frozen=True)
class BootstrapForecaster:
    """Parametric bootstrap of fitted state-space histories, followed by refits.

    Simulated histories preserve the fitted model's transition dynamics and
    covariates. Parameter draws are re-filtered against the actual history before
    forecasting. This approximates sampling uncertainty, not a Bayesian posterior.
    """

    forecaster: Forecaster
    replications: int = 30
    samples: int = 1000
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("replications", "samples"):
            v.integer(getattr(self, name), name, 1)
        v.seed(self.seed)

    def fit(
        self, data: TimeSeries | ArrayLike, *, covariates: Covariates | None = None
    ) -> FittedForecaster:
        history = prepare(data, covariates)
        central = fit(history, self.forecaster)
        adapter: Any = central.adapter
        if not hasattr(adapter, "model") or not hasattr(
            adapter.model, "filter_results"
        ):
            raise ValueError(
                "parameter bootstrap currently requires a linear state-space forecaster"
            )
        burn = max(1, int(getattr(adapter.model, "loglikelihood_burn", 0)))
        if len(history) <= burn + 2:
            raise ValueError("bootstrap needs enough post-initialization observations")
        from inspect import signature

        rng_key = (
            "rng"
            if "rng" in signature(adapter.model.simulate).parameters
            else "random_state"
        )
        rng, members, params = np.random.default_rng(self.seed), [], []
        for _ in range(self.replications):
            # Simulate recursively through the fitted transition system. Adding
            # noise to fixed fitted means would break lag dependence and bias
            # bootstrap autoregressive parameter estimates toward zero.
            simulated = adapter.model.simulate(
                len(history) - burn, anchor=burn, **{rng_key: rng}
            )
            values = np.concatenate(
                (
                    history.values[:burn],
                    np.asarray(simulated).reshape(-1, history.dimension),
                )
            )
            fitted = fit(
                TimeSeries(
                    values, history.timestamps, history.covariates, history.names
                ),
                self.forecaster,
            )
            # Re-filter actual history with bootstrapped parameters, keeping the
            # actual cutoff state instead of a fabricated bootstrap endpoint.
            conditioned = adapter.model.model.filter(
                getattr(fitted.adapter, "model").params
            )
            from dataclasses import replace

            members.append(replace(adapter, model=conditioned))
            params.append(np.asarray(getattr(fitted.adapter, "model").params))
        return FittedForecaster(
            _BootstrapAdapter(tuple(members), self.samples),
            history,
            self,
            Diagnostics(
                "parametric-bootstrap", message="Approximate fitted-parameter mixture"
            ),
            {"parameter_draws": np.array(params), "bootstrap": "parametric"},
            "bootstrap",
            self.seed,
        )


@dataclass(frozen=True)
class _BootstrapAdapter:
    members: tuple
    samples: int

    def forecast(self, steps, covariates, *, rng: Any = None):
        v.generator(rng)
        labels = rng.integers(len(self.members), size=self.samples)
        paths = None
        for i, member in enumerate(self.members):
            n = int(np.count_nonzero(labels == i))
            if n:
                result = member.forecast(steps, covariates, rng=rng)
                if paths is None:
                    paths = np.empty((self.samples,) + result.shape)
                paths[labels == i] = result.sample(n, rng=rng)
        assert paths is not None
        return ForecastResult.from_paths(
            paths,
            metadata={
                "uncertainty": "innovations plus parametric-bootstrap parameters"
            },
        )
