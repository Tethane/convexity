"""Rank-aware subspaces, principal angles and basis alignment."""

from .. import _validation as v
from .._backends import scipy
from ..interfaces import Subspace


def subspace(spanning_columns, *, rcond=None):
    """SVD range basis. Retain singular values > rcond * largest singular value."""
    a = v.array(spanning_columns, "spanning_columns", ndim=2)
    if rcond is not None:
        threshold = v.scalar(rcond, "rcond")
        if not 0 <= threshold < 1:
            raise ValueError("rcond must lie in [0,1)")
    return scipy.orthogonal_basis(a, rcond)


def _compatible(source, target):
    if not isinstance(source, Subspace) or not isinstance(target, Subspace):
        raise TypeError("source and target must be Subspace objects")
    if source.ambient_dimension != target.ambient_dimension:
        raise ValueError("subspaces must have the same ambient dimension")


def principal_angles(source, target):
    """min(rank(source),rank(target)) angles, descending, in radians."""
    _compatible(source, target)
    return v.array(scipy.principal_angles(source, target), "angles", empty=True)


def align_subspaces(source, target):
    """Solve min_R ||Q_source R - Q_target||_F with R.T R = I.

    Reflections are allowed. Aligning the basis rotates scores with the same R
    so that scores @ basis.T remains unchanged. Spanned spaces never change.
    """
    _compatible(source, target)
    if source.dimension != target.dimension:
        raise ValueError("alignment requires equal subspace dimensions")
    return scipy.align(source, target)
