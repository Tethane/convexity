"""Public accuracy namespace; implementation adapters remain private."""
from .builtins.accuracy import (
    FloatingComparison,
    Residual,
    isclose,
    allclose,
    machine_precision,
    require_finite,
    residual,
)

__all__ = ['FloatingComparison', 'Residual', 'isclose', 'allclose', 'machine_precision', 'require_finite', 'residual']
