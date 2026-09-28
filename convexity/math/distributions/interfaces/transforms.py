"""Caller-supplied scalar bijections; no symbolic algebra or inferred inverses."""

from dataclasses import dataclass
from typing import Callable

from .. import _validation as v


@dataclass(frozen=True)
class MonotoneTransform:
    """Differentiable bijection from R onto the open interval image.

    All callbacks accept/return same-shaped NumPy arrays. log_abs_derivative is
    log(|d forward(x)/dx|), evaluated at the BASE point x, not at forward(x).
    Callers own monotonicity, inverse correctness, and the nonzero Jacobian.
    """

    forward: Callable
    inverse: Callable
    log_abs_derivative: Callable
    increasing: bool = True
    image: tuple[float, float] = (-float("inf"), float("inf"))

    def __post_init__(self):
        for name in ("forward", "inverse", "log_abs_derivative"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        if not isinstance(self.increasing, bool):
            raise TypeError("increasing must be boolean")
        bounds = v.array(self.image, "image", finite=False, ndim=1)
        if bounds.shape != (2,) or bounds[0] >= bounds[1]:
            raise ValueError("image must contain two increasing endpoints")
        object.__setattr__(self, "image", tuple(float(x) for x in bounds))
