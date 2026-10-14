"""SciPy owns subspace extraction, alignment and positive-definite solves."""

import numpy as np

from ..interfaces import Alignment, Subspace


def orthogonal_basis(matrix, rcond):
    from scipy.linalg import orth

    return Subspace(orth(matrix, rcond=rcond))


def principal_angles(source, target):
    from scipy.linalg import subspace_angles

    if min(source.dimension, target.dimension) == 0:
        return np.empty(0)
    return subspace_angles(source.basis, target.basis)


def align(source, target):
    from scipy.linalg import orthogonal_procrustes

    if source.dimension == 0:
        rotation = np.empty((0, 0))
    else:
        rotation, _ = orthogonal_procrustes(source.basis, target.basis)
    aligned = source.basis @ rotation
    return Alignment(
        rotation,
        Subspace(aligned),
        float(np.linalg.norm(aligned - target.basis)),
        principal_angles(source, target),
    )


def factor_posterior(model):
    from scipy.linalg import solve

    weighted = model.loadings / model.noise_variance[:, None]
    precision = np.eye(model.output_dimension) + model.loadings.T @ weighted
    covariance = solve(precision, np.eye(model.output_dimension), assume_a="pos")
    return weighted @ covariance, covariance
