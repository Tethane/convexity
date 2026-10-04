"""Optional RL complements exact control references and supplied action masks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from ..._provider_types import NativeTensor

if TYPE_CHECKING:
    from .. import TrainedAgent
if TYPE_CHECKING:
    from ...control import FiniteMDP

if TYPE_CHECKING:
    from .. import QResult


if TYPE_CHECKING:
    from ..._provider_types import NativeRLModel
from dataclasses import dataclass, field

import numpy as np

from ... import _decision as v
from .._backends import tabular


def _model(model):
    from ...control import FiniteMDP

    if not isinstance(model, FiniteMDP) or model.discount >= 1 or model.terminal.all():
        raise ValueError(
            "RL requires a FiniteMDP with discount<1 and at least one nonterminal state"
        )


def q_learning(
    model: FiniteMDP,
    *,
    steps: int = 100000,
    epsilon: float = 0.2,
    learning_rate: float | None = None,
    rng: np.random.Generator,
) -> QResult:
    """Sampled tabular Q-learning; residual is diagnostic, never a success claim."""
    _model(model)
    if learning_rate is not None:
        v.probability(learning_rate, "learning rate", open_interval=True)
    return tabular.q_learning(
        model,
        v.integer(steps, "steps", 1),
        v.probability(epsilon, "epsilon"),
        learning_rate,
        v.generator(rng),
    )


def fitted_q(
    model: FiniteMDP,
    features: ArrayLike,
    *,
    ridge: float = 0.0,
    tolerance: float = 1e-9,
    maxiter: int = 10000,
) -> QResult:
    """Model-based fitted Q with linear state/action features and least squares.

    One-hot features recover tabular iteration. Projected iteration can diverge;
    convergence of coefficients need not mean the Bellman residual is small.
    """
    _model(model)
    x = v.array(features, "features", ndim=3)
    if x.shape[:2] != model.rewards.shape:
        raise ValueError("features must be (state,action,feature)")
    return tabular.fitted_q(
        model,
        x,
        v.scalar(ridge, "ridge", minimum=0),
        v.scalar(tolerance, "tolerance", minimum=0, strict=True),
        v.integer(maxiter, "maxiter", 1),
    )


def bellman_targets(
    rewards: ArrayLike,
    next_q: ArrayLike,
    terminal: ArrayLike,
    *,
    discount: float = 0.99,
    feasible: ArrayLike | None = None,
) -> FloatArray:
    """Batch masked max targets; terminal rows never bootstrap."""
    r, q = v.array(rewards, "rewards", ndim=1), v.array(next_q, "next q", ndim=2)
    t = v.mask(terminal, r.shape, "terminal")
    f = (
        np.ones(q.shape, dtype=bool)
        if feasible is None
        else v.mask(feasible, q.shape, "feasible")
    )
    if len(q) != len(r) or np.any(~f.any(1) & ~t):
        raise ValueError("nonterminal rows require at least one feasible action")
    future = np.zeros_like(r)
    future[~t] = np.where(f[~t], q[~t], -np.inf).max(1)
    return v.array(r + v.probability(discount, "discount") * future, "targets")


def bellman_loss(
    predictions: ArrayLike, targets: ArrayLike, *, weights: ArrayLike | None = None
) -> float:
    """Return weighted mean squared error between matching prediction and Bellman-target arrays."""
    p, t = (
        v.array(predictions, "predictions", ndim=1),
        v.array(targets, "targets", ndim=1),
    )
    if p.shape != t.shape:
        raise ValueError("predictions and targets must align")
    return float(v.weights(weights, len(p)) @ (p - t) ** 2)


def actor_critic_loss(
    logits: NativeTensor,
    actions: NativeTensor,
    values: NativeTensor,
    returns: NativeTensor,
    feasible: NativeTensor,
    *,
    value_weight: float = 1.0,
    entropy_weight: float = 0.0,
) -> NativeTensor:
    """Tensor scalar loss with masked categorical actions; gradients stay intact."""
    try:
        from .._backends.torch import actor_critic_loss as run
    except ImportError as exc:
        raise ImportError("Install convexity[reinforcement] for tensor losses") from exc
    return run(
        logits,
        actions,
        values,
        returns,
        feasible,
        v.scalar(value_weight, "value weight", minimum=0),
        v.scalar(entropy_weight, "entropy weight", minimum=0),
    )


@dataclass(frozen=True)
class TrainedAgent:
    """Trained policy wrapper with observation/action conventions and native training state."""

    _native: NativeRLModel = field(repr=False)
    states: int
    algorithm: str

    def action(self, state: int):
        from .._backends.sb3 import action

        state = v.integer(state, "state")
        if state >= self.states:
            raise ValueError("state out of range")
        return action(self._native, state)


def train_agent(
    model: FiniteMDP,
    *,
    algorithm: str = "a2c",
    steps: int = 10000,
    seed: int = 0,
    max_episode_steps: int = 100,
    learning_rate: float = 0.001,
    n_steps: int = 32,
) -> TrainedAgent:
    """A2C/PPO finite-state integration; all actions must be admissible.

    SB3 vanilla categorical policies do not implement state-dependent masking.
    Use tabular/fitted-Q or the masked tensor loss when such constraints apply.
    """
    from .._backends.sb3 import train

    _model(model)
    if not model.feasible.all():
        raise ValueError(
            "SB3 adapter requires all actions admissible; use masked RL methods otherwise"
        )
    return train(
        model,
        v.choice(algorithm, ("a2c", "ppo"), "algorithm"),
        v.integer(steps, "steps", 1),
        v.integer(seed, "seed"),
        v.integer(max_episode_steps, "max episode steps", 1),
        v.scalar(learning_rate, "learning rate", minimum=0, strict=True),
        v.integer(n_steps, "n steps", 2),
    )
