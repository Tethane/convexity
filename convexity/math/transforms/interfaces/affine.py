"""Affine coordinates with explicit reconstruction and exact-inverse capability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray
from .. import _validation as v


@dataclass(frozen=True, init=False)
class AffineTransform:
    """z = (x - location) @ encoder; reconstruction = z @ decoder + location.

    encoder @ decoder need not be the identity. Only a square inverse pair
    supports inverse_transform; dimensional reduction uses reconstruct.
    """

    location: FloatArray
    encoder: FloatArray
    decoder: FloatArray

    def __init__(
        self, location: ArrayLike, encoder: ArrayLike, decoder: ArrayLike
    ) -> None:
        loc = v.array(location, "location", ndim=1)
        enc = v.array(encoder, "encoder", ndim=2)
        dec = v.array(decoder, "decoder", ndim=2)
        if enc.shape[0] != len(loc) or dec.shape != (enc.shape[1], len(loc)):
            raise ValueError("encoder must be (d,k), decoder (k,d), location (d,)")
        for name, value in dict(location=loc, encoder=enc, decoder=dec).items():
            object.__setattr__(self, name, value)

    @property
    def input_dimension(self):
        return len(self.location)

    @property
    def output_dimension(self):
        return self.encoder.shape[1]

    @property
    def is_invertible(self):
        return (
            self.input_dimension == self.output_dimension
            and np.allclose(
                self.encoder @ self.decoder,
                np.eye(self.input_dimension),
                rtol=1e-8,
                atol=1e-10,
            )
            and np.allclose(
                self.decoder @ self.encoder,
                np.eye(self.input_dimension),
                rtol=1e-8,
                atol=1e-10,
            )
        )

    def transform(self, values: ArrayLike) -> FloatArray:
        x = v.coordinates(values, self.input_dimension, "values")
        return v.array((x - self.location) @ self.encoder, "transformed values")

    def reconstruct(self, coordinates: ArrayLike) -> FloatArray:
        z = v.coordinates(coordinates, self.output_dimension)
        return v.array(z @ self.decoder + self.location, "reconstruction")

    def inverse_transform(self, coordinates):
        if not self.is_invertible:
            raise ValueError(
                "map has no exact inverse; use reconstruct for a decoded approximation"
            )
        return self.reconstruct(coordinates)

    def residuals(self, values: ArrayLike) -> FloatArray:
        x = v.coordinates(values, self.input_dimension, "values")
        return v.array(x - self.reconstruct(self.transform(x)), "residuals")
