"""Array and callback vocabulary shared by the 36 mathematical families.

Constructors accept NumPy ArrayLike inputs (including Python lists). Validated
real-valued outputs use FloatArray; NumericArray additionally permits complex
values. Dtype annotations do not encode rank, shape, units or finiteness: those
constraints are documented and checked by the receiving mathematical contract.

ScalarLike includes NumPy scalar values and zero-dimensional real arrays;
non-scalar arrays are rejected where a scalar is required. IndexArray preserves
numeric or datetime coordinates rather than coercing calendar dates to floats.
"""

from numpy.typing import ArrayLike, DTypeLike

from ._typing import (
    ArrayFunction,
    BoolArray,
    ComplexArray,
    DateLike,
    FloatArray,
    IndexArray,
    IntArray,
    MetadataValue,
    NumericArray,
    SampleSize,
    ScalarFunction,
    ScalarLike,
)

__all__ = [
    "ArrayLike",
    "DTypeLike",
    "ArrayFunction",
    "BoolArray",
    "ComplexArray",
    "DateLike",
    "FloatArray",
    "IndexArray",
    "IntArray",
    "MetadataValue",
    "NumericArray",
    "SampleSize",
    "ScalarFunction",
    "ScalarLike",
]
