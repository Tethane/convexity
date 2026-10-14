"""Mathematical constructors and reusable fitting tools."""

from .composition import FittedPipeline, Pipeline
from .decomposition import (
    FactorAnalysis,
    PCA,
    Whitening,
    factor_analysis,
    pca,
    whitening,
)
from .geometry import align_subspaces, principal_angles, subspace
from .scaling import (
    MinMaxScale,
    RobustScale,
    Standardize,
    minmax_scale,
    robust_scale,
    standardize,
)
from .sequence import (
    Difference,
    FourierFeatures,
    LagFeatures,
    SeasonalFeatures,
    difference,
    fourier_features,
    lag_features,
    seasonal_features,
)

__all__ = [
    "Difference",
    "FactorAnalysis",
    "FittedPipeline",
    "FourierFeatures",
    "LagFeatures",
    "MinMaxScale",
    "PCA",
    "Pipeline",
    "RobustScale",
    "SeasonalFeatures",
    "Standardize",
    "Whitening",
    "align_subspaces",
    "difference",
    "factor_analysis",
    "fourier_features",
    "lag_features",
    "minmax_scale",
    "pca",
    "principal_angles",
    "robust_scale",
    "seasonal_features",
    "standardize",
    "subspace",
    "whitening",
]
