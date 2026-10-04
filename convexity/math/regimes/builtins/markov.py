"""Finite, row-stochastic, discrete-time Markov chains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import (
    FloatArray,
)
from .. import _validation as v


@dataclass(frozen=True)
class MarkovChain:
    """transition[i,j] = P(S[t+1]=j | S[t]=i); one step is one row."""

    transition: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "transition", v.transition(self.transition))

    @property
    def size(self):
        return len(self.transition)

    def matrix(self, steps: int = 1) -> FloatArray:
        return v.array(
            np.linalg.matrix_power(self.transition, v.integer(steps, "steps")),
            "transition power",
        )

    def propagate(self, probabilities, steps: int = 1):
        p = v.probabilities(probabilities)
        if p.shape != (self.size,):
            raise ValueError("probabilities must match chain size")
        return v.probabilities(p @ self.matrix(steps))

    def stationary(self):
        """Unique invariant law, including periodic chains; reject nonuniqueness."""
        a = np.vstack([self.transition.T - np.eye(self.size), np.ones(self.size)])
        p, _, rank, _ = np.linalg.lstsq(a, np.r_[np.zeros(self.size), 1.0], rcond=None)
        if rank != self.size:
            raise ValueError("chain does not have a unique stationary distribution")
        return v.probabilities(np.maximum(p, 0) / np.maximum(p, 0).sum())

    def durations(self) -> FloatArray:
        """Expected consecutive observations in each regime, including entry."""
        with np.errstate(divide="ignore"):
            return v.array(
                1 / (1 - np.diag(self.transition)), "durations", finite=False
            )

    def sample(
        self,
        initial: ArrayLike,
        steps: int,
        *,
        draws: int = 1,
        rng: np.random.Generator,
    ):
        """Return (draw,steps+1), including the initial state at column zero."""
        v.generator(rng)
        steps, draws = v.integer(steps, "steps"), v.integer(draws, "draws", 1)
        p = v.probabilities(initial)
        if p.shape != (self.size,):
            raise ValueError("initial probabilities must match chain size")
        out = np.empty((draws, steps + 1), dtype=np.int64)
        out[:, 0] = rng.choice(self.size, draws, p=p)
        for t in range(1, steps + 1):
            u = rng.random(draws)
            cumulative = self.transition[out[:, t - 1]].cumsum(axis=1)
            cumulative[:, -1] = 1
            out[:, t] = (u[:, None] >= cumulative).sum(axis=1)
        out.setflags(write=False)
        return out

    @classmethod
    def fit(
        cls, states: ArrayLike, *, size: int | None = None, pseudocount: float = 0.0
    ):
        """Observed-state MLE; optional symmetric transition pseudocounts."""
        x = v.array(states, "states", ndim=1)
        if np.any(x < 0) or np.any(x != np.floor(x)):
            raise ValueError("states must be nonnegative integers")
        k = int(x.max()) + 1 if size is None else v.integer(size, "size", 1)
        if np.any(x >= k):
            raise ValueError("states exceed declared size")
        counts = np.full((k, k), v.scalar(pseudocount, "pseudocount", minimum=0))
        np.add.at(counts, (x[:-1].astype(int), x[1:].astype(int)), 1)
        if np.any(counts.sum(axis=1) == 0):
            raise ValueError(
                "unobserved outgoing transitions require positive pseudocounts"
            )
        return cls(counts / counts.sum(axis=1, keepdims=True))

    def __init__(
        self,
        transition: ArrayLike,
    ) -> None:
        object.__setattr__(self, "transition", transition)
        self.__post_init__()
