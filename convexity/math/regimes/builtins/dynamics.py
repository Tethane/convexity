"""User-defined dynamics conditional on the entering regime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from ...distributions import Distribution
from .. import _validation as v
from ..interfaces import RegimePaths
from .markov import MarkovChain


@dataclass(frozen=True)
class SwitchingDynamics:
    """First draw S[t], then X[t] ~ law(S[t], X[t-1], t), for t=1,... .

    The callback returns an existing Distribution, allowing non-Gaussian and
    nonlinear dynamics. Initial state and regime are retained in sampled paths.
    """

    chain: MarkovChain
    law: Callable[[int, FloatArray, int], Distribution]

    def __post_init__(self) -> None:
        if not isinstance(self.chain, MarkovChain) or not callable(self.law):
            raise TypeError("chain must be MarkovChain and law must be callable")

    def distribution(self, state: ArrayLike, regime, *, step: int = 1):
        from ...distributions import mixture

        regime, step = v.state(regime, self.chain.size), v.integer(step, "step", 1)
        state = v.array(state, "state")
        if state.ndim > 1:
            raise ValueError("state must be scalar or vector")
        laws, weights = [], []
        for j, weight in enumerate(self.chain.transition[regime]):
            if weight > 0:
                laws.append(v.law(self.law(j, state, step), state.shape))
                weights.append(weight)
        return mixture(laws, weights)

    def sample(
        self,
        initial_state: ArrayLike,
        initial_regime,
        steps,
        *,
        draws: int = 1,
        rng: np.random.Generator,
    ) -> RegimePaths:
        state = v.array(initial_state, "initial state")
        if state.ndim > 1:
            raise ValueError("initial state must be scalar or vector")
        regime = v.state(initial_regime, self.chain.size)
        states = self.chain.sample(
            np.eye(self.chain.size)[regime], steps, draws=draws, rng=rng
        )
        values = np.empty(states.shape + state.shape)
        values[:, 0] = state
        for t in range(1, states.shape[1]):
            for i in range(len(states)):
                previous = v.array(values[i, t - 1], "previous state")
                law = v.law(self.law(int(states[i, t]), previous, t), state.shape)
                values[i, t] = law.sample(rng=rng)
        return RegimePaths(states, values)
