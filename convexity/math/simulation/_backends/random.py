"""NumPy MC and SciPy scrambled nets with explicit local random state."""
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc


def normals(draws, dimension, method, rng):
    if dimension == 0:
        return np.empty((draws, 0))
    if method == 'mc':
        return rng.standard_normal((draws, dimension))
    # A fresh seed is drawn from the supplied generator on every call. Passing
    # the Generator directly to SciPy may spawn from its SeedSequence without
    # advancing its bit-generator state, breaking saved-state replay semantics.
    seed = int(rng.integers(0, 2**63))
    if method == 'sobol':
        points = qmc.Sobol(dimension, scramble=True, rng=seed).random_base2(draws.bit_length()-1)
    else:
        points = qmc.Halton(dimension, scramble=True, rng=seed).random(draws)
    # Guard inverse-normal endpoints at floating precision. No points are
    # dropped, thinned or padded (in particular the Sobol balance is retained).
    return ndtri(np.clip(points, np.finfo(float).eps/2, 1-np.finfo(float).eps/2))
