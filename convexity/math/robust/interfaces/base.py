"""Uncertainty-set extension contract."""

from __future__ import annotations

from typing import Protocol

from numpy.typing import ArrayLike


class UncertaintySet(Protocol):
    """Set of admissible uncertain coordinates; its support operation uses the supplied direction and units."""
    def support(self, direction: ArrayLike) -> float: ...
