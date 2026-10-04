"""Immutable kernel facade; scikit-learn supplies the covariance algebra."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from sklearn.gaussian_process.kernels import Kernel as SklearnKernel


from typing import cast

import numpy as np

if TYPE_CHECKING:
    from sklearn.gaussian_process.kernels import Kernel as NativeKernel

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, overload

from numpy.typing import NDArray

from .. import _validation as v


@dataclass(frozen=True, repr=False)
class Kernel:
    """A copied scikit-learn kernel with checked arrays and immutable updates.

    Use the builtin factories or from_sklearn for custom sklearn kernels.
    theta and bounds use log coordinates, excluding fixed hyperparameters.
    """

    _kernel: NativeKernel = field(repr=False)

    def __post_init__(self) -> None:
        from sklearn.gaussian_process.kernels import Kernel as SKKernel

        if not isinstance(self._kernel, SKKernel):
            raise TypeError("kernel must implement scikit-learn's Kernel interface")
        object.__setattr__(self, "_kernel", deepcopy(self._kernel))

    def __repr__(self):
        return repr(self._kernel)

    @property
    def theta(self) -> FloatArray:
        return v.array(self._kernel.theta, "theta", ndim=1, empty=True)

    @property
    def bounds(self) -> FloatArray:
        """Finite log bounds, shape (number of free parameters, 2)."""
        return v.array(
            np.asarray(self._kernel.bounds).reshape(-1, 2), "bounds", empty=True
        )

    @property
    def parameter_names(self):
        names = []
        for p in self._kernel.hyperparameters:
            if not p.fixed:
                names.extend(
                    [p.name]
                    if p.n_elements == 1
                    else [f"{p.name}[{i}]" for i in range(p.n_elements)]
                )
        return tuple(names)

    def with_theta(self, theta: ArrayLike) -> Kernel:
        """Clone at explicit log parameters; never mutate this kernel.

        Bounds constrain fitting, not evaluation at an explicit theta.
        """
        theta = v.array(theta, "theta", ndim=1, empty=True)
        if theta.shape != self.theta.shape:
            raise ValueError("theta must match the number of free parameters")
        with np.errstate(over="ignore", under="ignore"):
            physical = np.exp(theta)
        if np.any(~np.isfinite(physical)) or np.any(physical == 0):
            raise ValueError("exp(theta) must be positive and finite")
        return Kernel(cast("NativeKernel", self._kernel.clone_with_theta(theta)))

    @overload
    def __call__(
        self,
        x: ArrayLike,
        y: ArrayLike | None = None,
        *,
        eval_gradient: Literal[False] = False,
    ) -> NDArray[np.float64]: ...

    @overload
    def __call__(
        self, x: ArrayLike, y: ArrayLike | None = None, *, eval_gradient: Literal[True]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    @overload
    def __call__(
        self, x: ArrayLike, y: ArrayLike | None = None, *, eval_gradient: bool
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    def __call__(self, x, y=None, *, eval_gradient: bool = False):
        """Gram/cross matrix, optionally with derivatives with respect to theta."""
        x = v.features(x)
        y = None if y is None else v.features(y, x.shape[1])
        v.boolean(eval_gradient, "eval_gradient")
        if eval_gradient and y is not None:
            raise ValueError("kernel gradients require y=None")
        result = cast(
            "ArrayLike | tuple[ArrayLike, ArrayLike]",
            self._kernel(x, y, eval_gradient=eval_gradient),
        )
        if eval_gradient:
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("kernel gradient must return (matrix, gradient)")
            matrix, gradient = result
        else:
            matrix, gradient = result, None
        matrix = v.array(matrix, "kernel matrix", ndim=2)
        expected = (len(x), len(x) if y is None else len(y))
        if matrix.shape != expected:
            raise ValueError("kernel matrix must preserve input rows")
        if not eval_gradient:
            return matrix
        gradient = v.array(gradient, "kernel gradient", ndim=3, empty=True)
        if gradient.shape != expected + self.theta.shape:
            raise ValueError(
                "kernel gradient must have shape (n,n,number_of_parameters)"
            )
        return matrix, gradient

    def diag(self, x):
        x = v.features(x)
        result = v.array(self._kernel.diag(x), "kernel diagonal", ndim=1)
        if result.shape != (len(x),) or np.any(result < 0):
            raise ValueError("kernel diagonal must be nonnegative and preserve rows")
        return result

    def __add__(self, other):
        return Kernel(self._kernel + _operand(other)._kernel)

    def __radd__(self, other):
        return _operand(other) + self

    def __mul__(self, other):
        return Kernel(self._kernel * _operand(other)._kernel)

    def __rmul__(self, other):
        return _operand(other) * self

    def __pow__(self, exponent):
        # Integer entrywise powers preserve positive semidefiniteness.
        exponent = v.integer(exponent, "exponent", 1)
        return Kernel(self._kernel**exponent)


def _operand(value):
    if isinstance(value, Kernel):
        return value
    from ..builtins.kernels import Constant

    return Constant(value, bounds="fixed")


def from_sklearn(kernel: SklearnKernel) -> Kernel:
    """Copy a custom sklearn kernel; its implementation owns the PSD claim."""
    return Kernel(kernel)
