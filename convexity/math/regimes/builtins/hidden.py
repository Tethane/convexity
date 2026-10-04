"""Exact inference for supplied scalar/vector emission distributions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import (
    FloatArray,
)

if TYPE_CHECKING:
    from ...distributions import Distribution
from .. import _validation as v
from ..interfaces import RegimePaths, RegimeResult
from .markov import MarkovChain


@dataclass(frozen=True)
class HiddenMarkovModel:
    """Discrete hidden-state model with explicit transition probabilities and state-dependent observation laws."""

    chain: MarkovChain
    emissions: Sequence[Distribution]
    initial: FloatArray

    def __post_init__(self) -> None:
        if not isinstance(self.chain, MarkovChain):
            raise TypeError("chain must be a MarkovChain")
        laws = tuple(v.law(x) for x in self.emissions)
        if (
            len(laws) != self.chain.size
            or len({(x.event_shape, x.measure) for x in laws}) != 1
        ):
            raise ValueError("one compatible emission law is required per regime")
        initial = v.probabilities(self.initial, "initial")
        if initial.shape != (self.chain.size,):
            raise ValueError("initial must match regime count")
        object.__setattr__(self, "emissions", laws)
        object.__setattr__(self, "initial", initial)

    def _forward(self, data):
        from ...forecasting.interfaces import TimeSeries
        from ...forecasting.interfaces.data import reject_covariates
        from .._backends.hidden import forward

        series = data if isinstance(data, TimeSeries) else TimeSeries(data)
        reject_covariates(series)
        shape = self.emissions[0].event_shape
        if series.dimension != (shape[0] if shape else 1):
            raise ValueError("observations must match emission dimension")
        logs = np.zeros((len(series), self.chain.size))
        for t, row in enumerate(series.values):
            keep = np.flatnonzero(~np.isnan(row))
            if not len(keep):
                continue
            for j, law in enumerate(self.emissions):
                if not shape:
                    value = law.log_density(row[0])
                else:
                    marginal = law if len(keep) == len(row) else law.marginal(keep)
                    value = marginal.log_density(row[keep])
                logs[t, j] = float(value)
        if np.any(np.isposinf(logs)):
            raise ValueError("emission log densities must not be +inf")
        predicted, filtered, ll = forward(self.initial, self.chain.transition, logs)
        return series, logs, predicted, filtered, ll

    def filter(self, data) -> RegimeResult:
        series, _, predicted, filtered, ll = self._forward(data)
        return RegimeResult(
            np.exp(predicted), np.exp(filtered), ll, timestamps=series.timestamps
        )

    def smooth(self, data) -> RegimeResult:
        from .._backends.hidden import backward

        series, logs, predicted, filtered, ll = self._forward(data)
        smoothed, pairs = backward(self.chain.transition, logs, filtered, ll)
        return RegimeResult(
            np.exp(predicted), np.exp(filtered), ll, smoothed, pairs, series.timestamps
        )

    def posterior_paths(self, data, *, draws: int = 1, rng: np.random.Generator):
        """Exact forward-filter/backward draws of S[0:n] given all observations."""
        from scipy.special import logsumexp

        v.generator(rng)
        draws = v.integer(draws, "draws", 1)
        _, _, _, filtered, _ = self._forward(data)
        n, k = filtered.shape
        out = np.empty((draws, n), dtype=np.int64)
        out[:, -1] = rng.choice(k, draws, p=np.exp(filtered[-1]))
        with np.errstate(divide="ignore"):
            loga = np.log(self.chain.transition)
        for t in range(n - 2, -1, -1):
            for state in range(k):
                rows = np.flatnonzero(out[:, t + 1] == state)
                if len(rows):
                    logits = filtered[t] + loga[:, state]
                    out[rows, t] = rng.choice(
                        k, len(rows), p=np.exp(logits - logsumexp(logits))
                    )
        out.setflags(write=False)
        return out

    def decode(self, data):
        """Viterbi joint maximum-probability path, not marginal argmax labels."""
        _, emissions, _, _, _ = self._forward(data)
        n, k = emissions.shape
        with np.errstate(divide="ignore"):
            score, loga = (
                np.log(self.initial) + emissions[0],
                np.log(self.chain.transition),
            )
        parents = np.zeros((n, k), dtype=int)
        for t in range(1, n):
            joint = score[:, None] + loga
            parents[t] = joint.argmax(axis=0)
            score = joint.max(axis=0) + emissions[t]
        out = np.empty(n, dtype=int)
        out[-1] = score.argmax()
        for t in range(n - 2, -1, -1):
            out[t] = parents[t + 1, out[t + 1]]
        out.setflags(write=False)
        return out

    def predictive(self, probabilities, *, steps: int = 1):
        from ...distributions import mixture

        return mixture(self.emissions, self.chain.propagate(probabilities, steps))

    def sample(
        self, observations: int, *, draws: int = 1, rng: np.random.Generator
    ) -> RegimePaths:
        n = v.integer(observations, "observations", 1)
        states = self.chain.sample(self.initial, n - 1, draws=draws, rng=rng)
        values = np.empty(states.shape + self.emissions[0].event_shape)
        for j, law in enumerate(self.emissions):
            mask = states == j
            values[mask] = law.sample(int(mask.sum()), rng=rng)
        return RegimePaths(states, values)

    def __init__(
        self,
        chain: MarkovChain,
        emissions: Sequence[Distribution],
        initial: ArrayLike,
    ) -> None:
        object.__setattr__(self, "chain", chain)
        object.__setattr__(self, "emissions", emissions)
        object.__setattr__(self, "initial", initial)
        self.__post_init__()
