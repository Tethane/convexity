"""Finite control models with explicit information and action axes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import ArrayLike

from ... import _decision as v
from ..._typing import BoolArray, FloatArray, IntArray


@dataclass(frozen=True)
class FiniteMDP:
    """Transitions (state,action,next_state), expected rewards (state,action).

    Rewards are paid on leaving a state; terminal states pay zero thereafter.
    Feasible actions are explicit. Ties resolve to the smallest action index.
    """

    transitions: FloatArray
    rewards: FloatArray
    discount: float = 0.99
    feasible: BoolArray = cast("BoolArray", None)
    terminal: BoolArray = cast("BoolArray", None)

    def __post_init__(self) -> None:
        p, r = (
            v.array(self.transitions, "transitions", ndim=3),
            v.array(self.rewards, "rewards", ndim=2),
        )
        s, a, k = p.shape
        if (
            k != s
            or r.shape != (s, a)
            or np.any(p < 0)
            or not np.allclose(p.sum(-1), 1.0, atol=1e-12, rtol=1e-10)
        ):
            raise ValueError(
                "transitions must be stochastic (state,action,state) with matching rewards"
            )
        feasible = v.mask(
            np.ones((s, a), dtype=bool) if self.feasible is None else self.feasible,
            (s, a),
            "feasible",
        )
        terminal = v.mask(
            np.zeros(s, dtype=bool) if self.terminal is None else self.terminal,
            (s,),
            "terminal",
        )
        if not feasible.any(axis=1).all():
            raise ValueError(
                "every state, including terminal states, requires an admissible action"
            )
        v.probability(self.discount, "discount")
        for name, x in [
            ("transitions", p),
            ("rewards", r),
            ("feasible", feasible),
            ("terminal", terminal),
        ]:
            object.__setattr__(self, name, x)

    def __init__(
        self,
        transitions: ArrayLike,
        rewards: ArrayLike,
        discount: float = 0.99,
        feasible: ArrayLike | None = None,
        terminal: ArrayLike | None = None,
    ) -> None:
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "discount", discount)
        object.__setattr__(self, "feasible", feasible)
        object.__setattr__(self, "terminal", terminal)
        self.__post_init__()


@dataclass(frozen=True)
class ControlResult:
    """Value function and integer action policy on (state,) or (time,state) axes, with Bellman residual and iteration count."""

    values: FloatArray
    policy: IntArray
    residual: float = 0.0
    iterations: int = 0

    def __post_init__(self) -> None:
        x = v.array(self.values, "values")
        p = np.asarray(self.policy)
        if x.ndim not in (1, 2) or p.ndim != x.ndim or p.shape[-1] != x.shape[-1]:
            raise ValueError("values/policy must have (state,) or (time,state) axes")
        object.__setattr__(self, "values", x)
        object.__setattr__(self, "policy", v.indices(p, p.shape, "policy"))
        v.scalar(self.residual, "residual", minimum=0)
        v.integer(self.iterations, "iterations")

    def __init__(
        self,
        values: ArrayLike,
        policy: ArrayLike,
        residual: float = 0.0,
        iterations: int = 0,
    ) -> None:
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "iterations", iterations)
        self.__post_init__()


@dataclass(frozen=True)
class BeliefResult:
    """Action selected for an input belief, its value, action-value vector and number of evaluated search nodes."""

    value: float
    action: int
    action_values: FloatArray
    nodes: int

    def __post_init__(self) -> None:
        v.scalar(self.value, "value")
        object.__setattr__(
            self, "action_values", v.array(self.action_values, "action values", ndim=1)
        )
        v.integer(self.action, "action")
        v.integer(self.nodes, "nodes", 1)

    def __init__(
        self,
        value: float,
        action: int,
        action_values: ArrayLike,
        nodes: int,
    ) -> None:
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "action_values", action_values)
        object.__setattr__(self, "nodes", nodes)
        self.__post_init__()


@dataclass(frozen=True)
class LQRResult:
    """Linear-quadratic regulator gain/value matrix and diagnostics for the declared time convention."""

    gain: FloatArray
    riccati: FloatArray
    closed_loop: FloatArray
    residual: float

    def __post_init__(self) -> None:
        for name in ("gain", "riccati", "closed_loop"):
            object.__setattr__(self, name, v.array(getattr(self, name), name, ndim=2))
        v.scalar(self.residual, "residual", minimum=0)

    def __init__(
        self,
        gain: ArrayLike,
        riccati: ArrayLike,
        closed_loop: ArrayLike,
        residual: float,
    ) -> None:
        object.__setattr__(self, "gain", gain)
        object.__setattr__(self, "riccati", riccati)
        object.__setattr__(self, "closed_loop", closed_loop)
        object.__setattr__(self, "residual", residual)
        self.__post_init__()


@dataclass(frozen=True)
class ScenarioTree:
    """Root 0; parent precedes child. Conditional branch masses sum to one.

    Disturbance (node,state) is realized on arrival, root disturbance is zero.
    A node represents an observed history; hidden shocks must not split nodes
    before they become observable. One action per node enforces nonanticipation.
    """

    parents: IntArray
    probabilities: FloatArray
    disturbances: FloatArray

    def __post_init__(self) -> None:
        w = v.array(self.probabilities, "probabilities", ndim=1)
        parents = v.indices(self.parents, (len(w),), "parents")
        d = v.array(self.disturbances, "disturbances", ndim=2)
        if (
            len(w) < 2
            or parents[0] != -1
            or w[0] != 1
            or np.any(w < 0)
            or d.shape[0] != len(w)
            or np.any(d[0] != 0)
        ):
            raise ValueError(
                "tree requires root parent=-1, probability=1, zero disturbance and aligned nodes"
            )
        if np.any(parents[1:] < 0) or np.any(parents[1:] >= np.arange(1, len(w))):
            raise ValueError("each parent must precede its child")
        for i in range(len(w)):
            child = parents == i
            if child.any() and not np.isclose(
                w[child].sum(), 1.0, atol=1e-12, rtol=1e-10
            ):
                raise ValueError("conditional child probabilities must sum to one")
        for name, x in [
            ("parents", parents),
            ("probabilities", w),
            ("disturbances", d),
        ]:
            object.__setattr__(self, name, x)

    def masses(self) -> FloatArray:
        mass = np.ones(len(self.parents))
        for i in range(1, len(mass)):
            mass[i] = mass[self.parents[i]] * self.probabilities[i]
        return v.array(mass, "node masses")

    def __init__(
        self,
        parents: ArrayLike,
        probabilities: ArrayLike,
        disturbances: ArrayLike,
    ) -> None:
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "disturbances", disturbances)
        self.__post_init__()


@dataclass(frozen=True)
class TreeControlResult:
    """Scenario-tree control solution with node-aligned states/actions and solver diagnostics."""

    states: FloatArray
    actions: FloatArray
    objective: float
    max_violation: float
    solver: str

    def __post_init__(self) -> None:
        for name in ("states", "actions"):
            object.__setattr__(self, name, v.array(getattr(self, name), name, ndim=2))
        v.scalar(self.objective, "objective")
        v.scalar(self.max_violation, "max_violation", minimum=0)

    def __init__(
        self,
        states: ArrayLike,
        actions: ArrayLike,
        objective: float,
        max_violation: float,
        solver: str,
    ) -> None:
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "max_violation", max_violation)
        object.__setattr__(self, "solver", solver)
        self.__post_init__()


class ControlSolver(Protocol):
    """Solve a control problem and return owned state/action/value information with diagnostics."""

    def solve(self, model: FiniteMDP) -> ControlResult: ...
