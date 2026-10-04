"""Native calibration diagnostics remain distinct from statistical uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike

from ..._finance import ResultRecord, array, integer
from ..._typing import FloatArray
from ..._validation import boolean


@dataclass(frozen=True)
class NativeCalibrationResult(ResultRecord):
    """Native model calibration output and diagnostics; native handles are advanced interoperability objects."""

    parameters: FloatArray
    residuals: FloatArray
    end_criteria: int
    success: bool

    def __post_init__(self) -> None:
        for name in ("parameters", "residuals"):
            object.__setattr__(self, name, array(getattr(self, name), name, ndim=1))
        integer(self.end_criteria, "end criteria")
        boolean(self.success, "success")

    def __init__(
        self,
        parameters: ArrayLike,
        residuals: ArrayLike,
        end_criteria: int,
        success: bool,
    ) -> None:
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "end_criteria", end_criteria)
        object.__setattr__(self, "success", success)
        self.__post_init__()
