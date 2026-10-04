"""Immutable PyMC algorithm specifications with explicit random generators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import _validation as v


@dataclass(frozen=True)
class MCMC:
    """PyMC auto steps (usually NUTS), explicit NUTS, or random-walk Metropolis.

    A numerical BayesianProblem uses Metropolis and needs a finite initial point.
    PyMCModel uses native symbolic gradients, constraints and compound steps.
    """

    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    cores: int = 1
    method: str = "auto"
    target_accept: float = 0.9
    progressbar: bool = False

    def __post_init__(self) -> None:
        for name in ("draws", "chains", "cores"):
            v.integer(getattr(self, name), name, 1)
        v.integer(self.tune, "tune")
        v.choice(self.method, ("auto", "nuts", "metropolis"), "method")
        v.probability(self.target_accept, "target_accept")
        v.boolean(self.progressbar, "progressbar")

    def fit(self, model, *, rng: np.random.Generator):
        from .._backends.pymc import sample

        return sample(model, self, rng=rng)


@dataclass(frozen=True)
class VariationalInference:
    """PyMC mean-field or full-rank ADVI; loss history is not a convergence proof."""

    iterations: int = 20000
    draws: int = 2000
    method: str = "advi"
    learning_rate: float = 0.01
    progressbar: bool = False

    def __post_init__(self) -> None:
        v.integer(self.iterations, "iterations", 1)
        v.integer(self.draws, "draws", 1)
        v.choice(self.method, ("advi", "fullrank_advi"), "method")
        v.positive(self.learning_rate, "learning_rate")
        v.boolean(self.progressbar, "progressbar")

    def fit(self, model, *, rng: np.random.Generator):
        from .._backends.pymc import variational

        return variational(model, self, rng=rng)
