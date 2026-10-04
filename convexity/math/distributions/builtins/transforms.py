"""Continuous scalar transformations; empirical maps use law.map directly."""

from __future__ import annotations

import numpy as np

from .. import _validation as v
from .._backends.transforms import TransformedAdapter
from ..interfaces import Distribution, MonotoneTransform


def _validate(base, transform):
    if (
        not isinstance(base, Distribution)
        or base.event_shape
        or base.measure != "continuous"
    ):
        raise ValueError("transformed requires a continuous scalar law")
    if not isinstance(transform, MonotoneTransform):
        raise TypeError("transform must be MonotoneTransform")


def transformed(base: Distribution, transform: MonotoneTransform) -> Distribution:
    """Push a scalar base law through a monotone invertible transform.

    Density uses the inverse Jacobian; decreasing maps reverse probability order.
    Only capabilities justified by the base law and transform are exposed."""
    _validate(base, transform)
    return Distribution(TransformedAdapter(base, transform))


def affine(base: Distribution, shift: float = 0.0, scale: float = 1.0) -> Distribution:
    """Law of shift + scale*X for a continuous scalar X; negative scales allowed."""
    shift, scale = v.scalar(shift, "shift"), v.scalar(scale, "scale")
    if scale == 0:
        raise ValueError("scale must be nonzero for an invertible transformation")
    transform = MonotoneTransform(
        lambda x: shift + scale * x,
        lambda y: (y - shift) / scale,
        lambda x: np.full_like(x, np.log(abs(scale))),
        increasing=scale > 0,
    )
    _validate(base, transform)
    return Distribution(
        TransformedAdapter(base, transform, affine_parameters=(shift, scale))
    )


def exponential(base: Distribution) -> Distribution:
    """Law of exp(X). Nonpositive density queries return zero, not log-domain NaNs."""
    return transformed(
        base, MonotoneTransform(np.exp, np.log, lambda x: x, image=(0.0, np.inf))
    )
