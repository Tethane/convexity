"""Owned problem/result records and structural extension protocols for calibration.

Ordinary workflows may import these names directly from convexity.math.calibration.
See docs/math/reference/calibration.md for public signatures and mathematical conventions.
"""
from .base import CalibrationProblem, CalibrationResult, Calibrator

from .results import NativeCalibrationResult

__all__ = [
    "NativeCalibrationResult",
    "CalibrationProblem",
    "CalibrationResult",
    "Calibrator",
]
