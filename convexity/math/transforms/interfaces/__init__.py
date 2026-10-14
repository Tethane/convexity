"""Public mathematical objects and extension protocols."""

from .affine import AffineTransform
from .base import (
    ConvergenceError,
    Diagnostics,
    Transform,
    TransformFitter,
    checked_transform,
    fit,
)
from .decomposition import (
    FactorAnalysisResult,
    FittedScaling,
    GaussianFactorModel,
    PCAResult,
)
from .geometry import Alignment, Subspace
from .sequence import DifferenceResult, FeatureMatrix

__all__ = [
    "AffineTransform",
    "Alignment",
    "ConvergenceError",
    "Diagnostics",
    "DifferenceResult",
    "FactorAnalysisResult",
    "FeatureMatrix",
    "FittedScaling",
    "GaussianFactorModel",
    "PCAResult",
    "Subspace",
    "Transform",
    "TransformFitter",
    "checked_transform",
    "fit",
]
