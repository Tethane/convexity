"""Risk-axis by instrument-axis exposure geometry."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike

from ..._finance import array
from ..._typing import FloatArray


@dataclass(frozen=True)
class HedgeResult:
    """Hedge positions and remaining exposure in the declared instrument and risk-factor order."""

    holdings: FloatArray
    residual_exposure: FloatArray
    nullspace: FloatArray
    success: bool
    message: str

    def __post_init__(self) -> None:
        for n in ("holdings", "residual_exposure", "nullspace"):
            object.__setattr__(self, n, array(getattr(self, n), n, empty=True))
        if (
            self.holdings.ndim != 1
            or self.residual_exposure.ndim != 1
            or self.nullspace.ndim != 2
            or self.nullspace.shape[0] != len(self.holdings)
        ):
            raise ValueError("hedge result axes are inconsistent")

    def __init__(
        self,
        holdings: ArrayLike,
        residual_exposure: ArrayLike,
        nullspace: ArrayLike,
        success: bool,
        message: str,
    ) -> None:
        object.__setattr__(self, "holdings", holdings)
        object.__setattr__(self, "residual_exposure", residual_exposure)
        object.__setattr__(self, "nullspace", nullspace)
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "message", message)
        self.__post_init__()
