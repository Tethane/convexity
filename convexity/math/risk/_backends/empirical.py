"""Exact functionals of a finite probability law, including atoms at VaR."""

import numpy as np


def quantile(x, w, level):
    keep = w > 0
    x, w = x[keep], w[keep]
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    cdf = np.cumsum(w)
    cdf[-1] = 1.0
    return float(x[min(np.searchsorted(cdf, level, side="left"), len(x) - 1)])


def tail_weights(x, w, level):
    q = quantile(x, w, level)
    above, tied = x > q, x == q
    remaining = max(0.0, (1 - level) - w[above].sum())
    result = np.where(above, w, 0.0)
    result[tied] = w[tied] * remaining / w[tied].sum()
    return result / (1 - level)
