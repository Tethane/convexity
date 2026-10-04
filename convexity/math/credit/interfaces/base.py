"""Default-law protocols separate measure labels from curve mechanics."""

from __future__ import annotations

from typing import Protocol


class DefaultLaw(Protocol):
    """Default-time probability law with survival/default semantics declared by its operations."""
    measure: str

    def survival(self, maturity: float) -> float: ...
    def hazard(self, maturity: float) -> float: ...
    def default_probability(self, start, end) -> float: ...
