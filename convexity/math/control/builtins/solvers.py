"""Finite-state control, belief control and linear-quadratic MPC."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from .. import ControlResult


if TYPE_CHECKING:
    from .. import LQRResult


if TYPE_CHECKING:
    from .. import BeliefResult


import numpy as np

if TYPE_CHECKING:
    from .. import TreeControlResult


from ... import _decision as v
from .._backends import dynamic
from ..interfaces import FiniteMDP, ScenarioTree


def finite_horizon(
    model: FiniteMDP, horizon: int, *, terminal_values: ArrayLike | None = None
) -> ControlResult:
    """Backward induction for a FiniteMDP over horizon integer decisions.

    Return values (horizon+1,state) and policies (horizon,state). Terminal values
    are in reward units; discounting follows the MDP. Terminal actions use -1."""
    h = v.integer(horizon, "horizon", 1)
    t = v.vector(
        0.0 if terminal_values is None else terminal_values,
        len(model.transitions),
        "terminal values",
    )
    return dynamic.finite_horizon(model, h, t)


def value_iteration(
    model: FiniteMDP, *, tolerance: float = 1e-10, maxiter: int = 100000
) -> ControlResult:
    """Solve a discounted finite MDP to the specified Bellman tolerance.

    Return state values and action indices; failure to converge raises explicitly."""
    if model.discount >= 1:
        raise ValueError("stationary value iteration requires discount < 1")
    return dynamic.value_iteration(
        model,
        v.scalar(tolerance, "tolerance", minimum=0, strict=True),
        v.integer(maxiter, "maxiter", 1),
    )


def belief_update(
    belief: ArrayLike, transition: ArrayLike, likelihood: ArrayLike
) -> FloatArray:
    """Bayes-update a probability vector after one transition and observation.

    transition[i,j] = P(next=j|current=i); likelihood[j] weights the next state.
    Return normalized next-state probabilities; zero evidence is an error."""
    b = v.array(belief, "belief", ndim=1)
    if np.any(b < 0) or not np.isclose(b.sum(), 1):
        raise ValueError("belief must sum to one")
    p = v.array(transition, "transition", ndim=2)
    likelihood = v.array(likelihood, "likelihood", ndim=1)
    if (
        p.shape != (len(b), len(b))
        or likelihood.shape != b.shape
        or np.any(p < 0)
        or not np.allclose(p.sum(1), 1)
        or np.any(likelihood < 0)
    ):
        raise ValueError("invalid stochastic transition or likelihood")
    post = (b @ p) * likelihood
    if post.sum() <= 0:
        raise ValueError("observation has zero probability")
    return v.array(post / post.sum(), "posterior")


def belief_plan(
    model: FiniteMDP,
    observations: ArrayLike,
    belief: ArrayLike,
    horizon: int,
    *,
    terminal_values: ArrayLike | None = None,
    max_nodes: int = 100000,
) -> BeliefResult:
    """Exact finite POMDP: observations (action,next_state,observation).

    All actions must be available in every hidden state; terminal state flags
    are unsupported (encode absorbing zero-reward states explicitly instead).
    """
    p = model.transitions
    o = v.array(observations, "observations", ndim=3)
    b = v.array(belief, "belief", ndim=1)
    if (
        o.shape[:2] != (p.shape[1], p.shape[0])
        or np.any(o < 0)
        or not np.allclose(o.sum(-1), 1)
    ):
        raise ValueError("observation rows must be probability laws")
    if b.shape != (len(p),) or np.any(b < 0) or not np.isclose(b.sum(), 1):
        raise ValueError("belief must be a state probability vector")
    if not model.feasible.all() or model.terminal.any():
        raise ValueError(
            "belief planning requires common actions and explicitly absorbing terminal dynamics"
        )
    t = v.vector(
        0.0 if terminal_values is None else terminal_values, len(p), "terminal values"
    )
    return dynamic.belief_plan(
        p,
        model.rewards,
        o,
        b,
        v.integer(horizon, "horizon", 1),
        model.discount,
        t,
        v.integer(max_nodes, "max_nodes", 1),
    )


def _linear(a, b, q, r):
    a, b, q, r = (
        v.array(a, "A", ndim=2),
        v.array(b, "B", ndim=2),
        v.psd(q, "Q"),
        v.psd(r, "R", definite=True),
    )
    n, m = b.shape
    if a.shape != (n, n) or q.shape != (n, n) or r.shape != (m, m):
        raise ValueError("A,B,Q,R dimensions must align")
    return a, b, q, r


def lqr(a: ArrayLike, b: ArrayLike, q: ArrayLike, r: ArrayLike) -> LQRResult:
    """Discrete infinite-horizon LQR, u=-gain@x; costs x'Qx+u'Ru."""
    from .._backends.lqr import solve

    return solve(*_linear(a, b, q, r))


def scenario_mpc(
    tree: ScenarioTree,
    a: ArrayLike,
    b: ArrayLike,
    q: ArrayLike,
    r: ArrayLike,
    initial: ArrayLike,
    *,
    terminal_cost: ArrayLike | None = None,
    lower_actions: ArrayLike = -np.inf,
    upper_actions: ArrayLike = np.inf,
    lower_states: ArrayLike = -np.inf,
    upper_states: ArrayLike = np.inf,
    solver: str | None = None,
) -> TreeControlResult:
    """Solve quadratic control on a ScenarioTree with node probabilities.

    Dynamics are x_child=A*x_parent+B*u_parent+disturbance_child. Matrices Q,R
    weight state/action costs. Return node states/actions and feasibility diagnostics."""
    from .._backends.tree import solve

    a, b, q, r = _linear(a, b, q, r)
    n, m = b.shape
    x = v.array(initial, "initial", ndim=1)
    t = q if terminal_cost is None else v.psd(terminal_cost, "terminal cost")
    if x.shape != (n,) or tree.disturbances.shape[1] != n or t.shape != (n, n):
        raise ValueError("tree, initial state and terminal cost must align")
    lo, hi = (
        v.vector(lower_actions, m, "lower actions", finite=False),
        v.vector(upper_actions, m, "upper actions", finite=False),
    )
    sl, sh = (
        v.vector(lower_states, n, "lower states", finite=False),
        v.vector(upper_states, n, "upper states", finite=False),
    )
    if (
        np.any(lo > hi)
        or np.any(sl > sh)
        or np.any(lo == np.inf)
        or np.any(hi == -np.inf)
        or np.any(sl == np.inf)
        or np.any(sh == -np.inf)
    ):
        raise ValueError("bounds must admit finite points")
    return solve(tree, a, b, q, r, t, x, lo, hi, sl, sh, solver)


def generator_hjb(
    generators: ArrayLike,
    rewards: ArrayLike,
    discount_rate: float,
    *,
    tolerance: float = 1e-10,
    maxiter: int = 100000,
) -> ControlResult:
    """Solve rho V = max_a(r_a + G_a V) by uniformization.

    G axes (state,action,next_state); off-diagonal entries nonnegative and
    rows sum to zero. Caller supplies a monotone spatial discretization.
    """
    g = v.array(generators, "generators", ndim=3)
    s, _, k = g.shape
    if s != k or np.any(abs(g.sum(-1)) > 1e-12 * np.max(abs(g), axis=-1)):
        raise ValueError("generator rows must sum to zero")
    off = g.copy()
    off[np.arange(s), :, np.arange(s)] = 0
    if np.any(off < 0):
        raise ValueError("generator off-diagonal entries must be nonnegative")
    rate = float(np.max(-g[np.arange(s), :, np.arange(s)]))
    rho = v.scalar(discount_rate, "discount rate", minimum=0, strict=True)
    p = (g / rate if rate > 0 else np.zeros_like(g)) + np.eye(s)[:, None, :]
    model = FiniteMDP(
        p, v.array(rewards, "rewards", ndim=2) / (rho + rate), rate / (rho + rate)
    )
    result = value_iteration(model, tolerance=tolerance / (rho + rate), maxiter=maxiter)
    from ..interfaces import ControlResult

    return ControlResult(
        result.values, result.policy, result.residual * (rho + rate), result.iterations
    )


def impulse_control(
    model: FiniteMDP,
    destinations: ArrayLike,
    costs: ArrayLike,
    *,
    tolerance: float = 1e-10,
    maxiter: int = 100000,
) -> ControlResult:
    """Add one-period intervention actions: pay cost now, jump next period.

    This is a discrete impulse MDP, not an instantaneous continuous-time QVI.
    destinations/costs have (state,intervention) axes; costs are nonnegative.
    """
    c = v.array(costs, "costs", ndim=2)
    d = v.indices(destinations, c.shape, "destinations")
    s = len(model.transitions)
    if c.shape[0] != s or np.any(c < 0) or np.any(d < 0) or np.any(d >= s):
        raise ValueError("intervention destinations and costs are invalid")
    p = np.eye(s)[d]
    augmented = FiniteMDP(
        np.concatenate((model.transitions, p), axis=1),
        np.concatenate((model.rewards, -c), axis=1),
        model.discount,
        np.concatenate((model.feasible, np.ones(c.shape, dtype=bool)), axis=1),
        model.terminal,
    )
    return value_iteration(augmented, tolerance=tolerance, maxiter=maxiter)
