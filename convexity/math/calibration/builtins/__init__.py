"""Concrete specifications and convenience functions for calibration.

Ordinary workflows may import these names directly from convexity.math.calibration.
See docs/reference/calibration.md for public signatures and mathematical conventions.
"""
from .solvers import LeastSquaresCalibration, calibrate, quantlib_calibrate

__all__ = ["LeastSquaresCalibration", "calibrate", "quantlib_calibrate"]
