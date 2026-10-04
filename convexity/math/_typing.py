"""Shared vocabulary for checked mathematical inputs and owned results.

ArrayLike describes accepted constructor arguments, not stored result arrays.
Shapes, units and coordinate order belong in each public object's docstring;
NumPy's dtype typing does not statically enforce those mathematical constraints.
"""

from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from typing import TypeAlias, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
ComplexArray: TypeAlias = NDArray[np.complex128]
NumericArray: TypeAlias = NDArray[np.float64 | np.complex128]
IndexArray: TypeAlias = NDArray[np.integer | np.floating | np.datetime64]
SampleSize: TypeAlias = int | tuple[int, ...]
DateLike: TypeAlias = str | date | datetime
ScalarLike: TypeAlias = (
    float | np.integer | np.floating | NDArray[np.integer | np.floating]
)
ScalarFunction: TypeAlias = Callable[[FloatArray], float]
ArrayFunction: TypeAlias = Callable[[FloatArray], ArrayLike]
ScalarOrArray: TypeAlias = float | FloatArray
MetadataValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | np.generic
    | FloatArray
    | IntArray
    | Mapping[str, "MetadataValue"]
    | Sequence["MetadataValue"]
)


class _FactoryDefault:
    def __repr__(self) -> str:
        return "<factory>"


# Typed constructors use the same per-call default-factory semantics as
# dataclass-generated constructors. This sentinel is never stored in a result.
_FACTORY = _FactoryDefault()

# SciPy is a core dependency. These aliases preserve sparse/operator inputs.
if TYPE_CHECKING:
    from scipy.sparse import spmatrix, sparray, csr_matrix, csr_array
    from scipy.sparse.linalg import LinearOperator
type MatrixLike = ArrayLike | spmatrix | sparray | LinearOperator
type Matrix = NumericArray | csr_matrix | csr_array | LinearOperator
