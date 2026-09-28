"""Log prior density evaluation through SciPy distributions."""

import numpy as np
from scipy import stats


def independent_log_density(family, theta, loc, scale):
    law = stats.norm if family == "normal" else stats.laplace
    return float(np.sum(law.logpdf(theta, loc=loc, scale=scale)))
