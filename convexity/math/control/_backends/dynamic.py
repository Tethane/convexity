"""Finite Bellman operators and exact belief branching."""

import numpy as np
from ..interfaces import ControlResult, BeliefResult


def action_values(model, values):
    q = model.rewards + model.discount * np.einsum(
        "sak,k->sa", model.transitions, values
    )
    q[model.terminal] = 0.0
    return np.where(model.feasible, q, -np.inf)


def finite_horizon(model, horizon, terminal):
    s = len(model.transitions)
    values, policy = np.zeros((horizon + 1, s)), np.zeros((horizon, s), dtype=np.int64)
    values[-1] = terminal
    values[-1, model.terminal] = 0.0
    for t in range(horizon - 1, -1, -1):
        q = action_values(model, values[t + 1])
        policy[t], values[t] = np.argmax(q, axis=1), np.max(q, axis=1)
    return ControlResult(values, policy, iterations=horizon)


def value_iteration(model, tolerance, maxiter):
    values = np.zeros(len(model.transitions))
    for iteration in range(1, maxiter + 1):
        values = action_values(model, values).max(axis=1)
        q = action_values(model, values)
        residual = float(np.max(abs(q.max(axis=1) - values)))
        if residual <= tolerance:
            return ControlResult(values, q.argmax(axis=1), residual, iteration)
    raise RuntimeError(
        f"Bellman iteration failed after {maxiter} steps; residual={residual}"
    )


def belief_plan(p, r, obs, belief, horizon, discount, terminal, max_nodes):
    nodes = 0

    def recurse(b, h):
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            raise ValueError("belief tree exceeds max_nodes")
        if h == 0:
            return float(b @ terminal), np.zeros(r.shape[1])
        q = b @ r
        for a in range(r.shape[1]):
            predicted = b @ p[:, a, :]
            for o in range(obs.shape[2]):
                joint = predicted * obs[a, :, o]
                mass = joint.sum()
                if mass > 0:
                    q[a] += discount * mass * recurse(joint / mass, h - 1)[0]
        return float(q.max()), q

    value, q = recurse(belief, horizon)
    return BeliefResult(value, int(q.argmax()), q, nodes)
