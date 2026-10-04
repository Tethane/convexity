"""Euclidean subspaces and orthogonal changes of factor basis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray
from .. import _validation as v


@dataclass(frozen=True, init=False)
class Subspace:
    """An orthonormal column basis Q; ambient row vectors project as x @ Q @ Q.T."""

    basis: FloatArray

    def __init__(self, basis: ArrayLike) -> None:
        q = v.array(basis, "basis", ndim=2, empty=True)
        if q.shape[0] < 1 or q.shape[1] > q.shape[0]:
            raise ValueError("basis must have shape (d,k), d>=1 and 0<=k<=d")
        if not np.allclose(q.T @ q, np.eye(q.shape[1]), rtol=1e-8, atol=1e-10):
            raise ValueError(
                "basis columns must be orthonormal; use subspace for arbitrary spanning columns"
            )
        object.__setattr__(self, "basis", q)

    @property
    def dimension(self):
        return self.basis.shape[1]

    @property
    def ambient_dimension(self):
        return self.basis.shape[0]

    @property
    def projector(self) -> FloatArray:
        return v.array(self.basis @ self.basis.T, "projector")

    def coordinates(self, values: ArrayLike) -> FloatArray:
        x = v.coordinates(values, self.ambient_dimension, "values")
        return v.array(x @ self.basis, "subspace coordinates", empty=True)

    def reconstruct(self, coordinates: ArrayLike) -> FloatArray:
        z = v.coordinates(coordinates, self.dimension)
        return v.array(z @ self.basis.T, "reconstruction")

    def project(self, values: ArrayLike):
        return self.reconstruct(self.coordinates(values))

    def residuals(self, values: ArrayLike) -> FloatArray:
        x = v.coordinates(values, self.ambient_dimension, "values")
        return v.array(x - self.project(x), "residuals")


@dataclass(frozen=True, init=False)
class Alignment:
    """source.basis @ rotation approximates target.basis, without translation/scaling."""

    rotation: FloatArray
    aligned: Subspace
    error: float
    angles: FloatArray

    def __init__(
        self, rotation: ArrayLike, aligned: Subspace, error: float, angles: ArrayLike
    ) -> None:
        if not isinstance(aligned, Subspace):
            raise TypeError("aligned must be a Subspace")
        k = aligned.dimension
        r = v.array(rotation, "rotation", ndim=2, empty=True)
        a = v.array(angles, "angles", ndim=1, empty=True)
        if r.shape != (k, k) or not np.allclose(
            r.T @ r, np.eye(k), rtol=1e-8, atol=1e-10
        ):
            raise ValueError("rotation must be orthogonal with shape (k,k)")
        if a.shape != (k,) or np.any(a < 0) or np.any(a > np.pi / 2 + 1e-10):
            raise ValueError("angles must have shape (k,) and lie in [0,pi/2]")
        err = v.scalar(error, "error")
        if err < 0:
            raise ValueError("error must be nonnegative")
        for name, value in dict(
            rotation=r, aligned=aligned, error=err, angles=a
        ).items():
            object.__setattr__(self, name, value)

    def transform(self, coordinates: ArrayLike) -> FloatArray:
        z = v.coordinates(coordinates, self.aligned.dimension)
        return v.array(z @ self.rotation, "aligned coordinates", empty=True)

    def inverse_transform(self, coordinates: ArrayLike) -> FloatArray:
        z = v.coordinates(coordinates, self.aligned.dimension)
        return v.array(z @ self.rotation.T, "original coordinates", empty=True)
