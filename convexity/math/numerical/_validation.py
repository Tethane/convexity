"""Input checks shared by the numerical contracts; no numerical solvers."""

from numbers import Integral
from typing import Any

from .._validation import boolean, choice

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator


def array(
    value: ArrayLike,
    name: str,
    *,
    ndim: int | None = None,
    real: bool = False,
    finite: bool = True,
    allow_empty: bool = False,
) -> NDArray[Any]:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iufc":
        raise ValueError(f"{name} must contain numeric values")
    if real and np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    result = np.array(raw, dtype=np.complex128 if np.iscomplexobj(raw) else np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; got {result.shape}")
    if (not allow_empty and not result.size) or (
        finite and not np.all(np.isfinite(result))
    ):
        raise ValueError(f"{name} must be nonempty and finite")
    result.setflags(write=False)
    return result


def square(value: ArrayLike, name: str) -> NDArray[Any]:
    result = array(value, name, ndim=2)
    if result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be square")
    return result


def hermitian(value: NDArray[Any], name: str) -> None:
    scale = float(np.max(np.abs(value)))
    if scale and not np.allclose(
        value / scale, value.conj().T / scale, rtol=1e-12, atol=1e-14
    ):
        raise ValueError(f"{name} must be Hermitian (symmetric for real inputs)")


def matrix(value: Any, name: str) -> Any:
    """Copy dense or sparse matrix data without densifying sparse inputs."""
    if isinstance(value, LinearOperator):
        if value.shape[0] != value.shape[1] or not value.shape[0]:
            raise ValueError(f"{name} must be a nonempty square operator")
        if value.dtype is None or np.dtype(str(value.dtype)).kind not in "fc":
            raise ValueError(f"{name} must have floating-point or complex dtype")
        return value
    if not issparse(value):
        return square(value, name)
    result = value.tocsr(copy=True)
    if (
        len(result.shape) != 2
        or result.shape[0] != result.shape[1]
        or not result.shape[0]
    ):
        raise ValueError(f"{name} must be a nonempty square matrix")
    if result.dtype.kind not in "iufc" or not np.all(np.isfinite(result.data)):
        raise ValueError(f"{name} must contain finite numeric values")
    return result.astype(np.complex128 if np.iscomplexobj(result.data) else np.float64)


def scalar(value: Any, name: str, *, finite: bool = True) -> float:
    result = array(value, name, ndim=0, real=True, finite=finite)
    if np.isnan(result):
        raise ValueError(f"{name} must not be NaN")
    return float(result)


def positive(value: Any, name: str, *, zero: bool = False) -> float:
    result = scalar(value, name)
    if result < 0 or (not zero and result == 0):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")
    return result


def integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def function(value: Any, name: str = "function") -> None:
    if not callable(value):
        raise TypeError(f"{name} must be callable")


def tolerances(atol: float, rtol: float) -> None:
    positive(atol, "atol", zero=True)
    positive(rtol, "rtol", zero=True)
    if atol == rtol == 0:
        raise ValueError("at least one tolerance must be positive")
