"""Stable scalar utility calculations."""

import numpy as np
from typing import cast
from scipy.special import logsumexp


def exponential_ce(x, w, aversion):
    keep = w > 0
    return -float(cast(float, logsumexp(-aversion * x[keep], b=w[keep]))) / aversion


def smooth_inverse(y, scale, aversion):
    return (y + aversion * np.hypot(y, np.sqrt(1 - aversion**2) * scale)) / (
        1 - aversion**2
    )
