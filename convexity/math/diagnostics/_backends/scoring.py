import numpy as np
from scipy.spatial.distance import cdist


def crps(x, actual, w):
    order = np.argsort(x, kind="stable")
    x, w = x[order] - actual, w[order]
    before = np.cumsum(w) - w
    half_pair = np.sum(w * x * (2 * before + w - 1))
    return float(w @ abs(x) - half_pair)


def energy(x, actual, w, beta):
    return float(
        w @ np.linalg.norm(x - actual, axis=1) ** beta
        - 0.5 * w @ (cdist(x, x) ** beta) @ w
    )
