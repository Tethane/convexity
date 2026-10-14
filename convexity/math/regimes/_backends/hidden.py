"""Log-space finite-state forward/backward recursions."""

import numpy as np


def forward(initial, transition, emissions):
    from scipy.special import logsumexp

    n, k = emissions.shape
    with np.errstate(divide="ignore"):
        logp, loga = np.log(initial), np.log(transition)
    predicted, filtered = np.empty((n, k)), np.empty((n, k))
    ll = np.empty(n)
    for t in range(n):
        predicted[t] = (
            logp if t == 0 else logsumexp(filtered[t - 1, :, None] + loga, axis=0)
        )
        joint = predicted[t] + emissions[t]
        ll[t] = logsumexp(joint)
        if not np.isfinite(ll[t]):
            raise ValueError(
                f"observation at row {t} has zero probability under every reachable regime"
            )
        filtered[t] = joint - ll[t]
    return predicted, filtered, ll


def backward(transition, emissions, filtered, ll):
    from scipy.special import logsumexp

    n, k = emissions.shape
    with np.errstate(divide="ignore"):
        loga = np.log(transition)
    beta = np.zeros((n, k))
    for t in range(n - 2, -1, -1):
        beta[t] = logsumexp(loga + emissions[t + 1] + beta[t + 1], axis=1) - ll[t + 1]
    smoothed = filtered + beta
    smoothed -= logsumexp(smoothed, axis=1, keepdims=True)
    pairs = np.empty((n - 1, k, k))
    for t in range(n - 1):
        joint = filtered[t, :, None] + loga + emissions[t + 1] + beta[t + 1]
        pairs[t] = np.exp(joint - logsumexp(joint))
    return np.exp(smoothed), pairs
