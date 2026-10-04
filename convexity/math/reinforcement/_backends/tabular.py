import numpy as np
from ..interfaces import QResult


def targets(model, q):
    value = np.where(model.feasible, q, -np.inf).max(axis=1)
    value[model.terminal] = 0.0
    result = model.rewards + model.discount * np.einsum(
        "sak,k->sa", model.transitions, value
    )
    result[model.terminal] = 0.0
    return result


def result(model, q, iterations, coefficients=None):
    residual = float(np.max(abs(targets(model, q) - q)[model.feasible]))
    return QResult(
        q,
        np.where(model.feasible, q, -np.inf).argmax(1),
        residual,
        iterations,
        coefficients,
    )


def q_learning(model, steps, epsilon, rate, rng):
    q, visits = np.zeros_like(model.rewards), np.zeros_like(model.rewards)
    live = np.flatnonzero(~model.terminal)
    state = int(rng.choice(live))
    for _ in range(steps):
        actions = np.flatnonzero(model.feasible[state])
        action = (
            int(rng.choice(actions))
            if rng.random() < epsilon
            else int(actions[np.argmax(q[state, actions])])
        )
        next_state = int(rng.choice(len(q), p=model.transitions[state, action]))
        future = (
            0.0
            if model.terminal[next_state]
            else float(np.max(q[next_state, model.feasible[next_state]]))
        )
        visits[state, action] += 1
        alpha = rate if rate is not None else 1 / visits[state, action] ** 0.6
        q[state, action] += alpha * (
            model.rewards[state, action] + model.discount * future - q[state, action]
        )
        state = int(rng.choice(live)) if model.terminal[next_state] else next_state
    return result(model, q, steps)


def fitted_q(model, features, ridge, tolerance, maxiter):
    x = features[model.feasible]
    q = np.zeros_like(model.rewards)
    beta = np.zeros(features.shape[-1])
    for iteration in range(1, maxiter + 1):
        y = targets(model, q)[model.feasible]
        if ridge:
            beta = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)
        else:
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
        updated = features @ beta
        updated[model.terminal] = 0.0
        delta = float(np.max(abs(updated - q)[model.feasible]))
        q = updated
        if not np.isfinite(q).all():
            raise RuntimeError("fitted Q iteration diverged")
        if delta <= tolerance:
            return result(model, q, iteration, beta)
    raise RuntimeError(
        "fitted Q iteration did not converge; linear projection need not be a contraction"
    )
