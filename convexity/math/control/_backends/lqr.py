"""Discrete algebraic LQR via python-control's SciPy implementation."""

import numpy as np
import control
from ..interfaces import LQRResult


def solve(a, b, q, r):
    gain, s, _ = control.dlqr(a, b, q, r, method="scipy")
    closed = a - b @ gain
    residual = s - (q + a.T @ s @ a - a.T @ s @ b @ gain)
    if np.max(abs(np.linalg.eigvals(closed))) >= 1:
        raise RuntimeError("LQR backend did not produce a stabilizing solution")
    return LQRResult(gain, s, closed, float(np.linalg.norm(residual, np.inf)))
