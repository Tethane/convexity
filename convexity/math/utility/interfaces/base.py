"""Increasing utility functions, with optional inverse for certainty equivalents."""

from __future__ import annotations

from typing import Protocol

from numpy.typing import ArrayLike

from ..._typing import FloatArray


class Utility(Protocol):
    """Wealth utility extension with explicit domain and preference parameters."""

    def __call__(self, wealth: ArrayLike) -> FloatArray: ...
    def inverse(self, utility: ArrayLike) -> FloatArray: ...
