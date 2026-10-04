"""Principal coordinates, PCA/ZCA whitening and Gaussian factor analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .. import AffineTransform


if TYPE_CHECKING:
    from .. import PCAResult


if TYPE_CHECKING:
    from .. import FactorAnalysisResult


from dataclasses import dataclass

import numpy as np

from .. import _validation as v
from .._backends import sklearn
from ..interfaces import AffineTransform, ConvergenceError


@dataclass(frozen=True)
class PCA:
    """Fit principal component axes from observation rows, with optional whitening and explicit covariance ddof."""

    n_components: int | None = None
    whiten: bool = False
    ddof: int = 1

    def __post_init__(self) -> None:
        if self.n_components is not None:
            v.integer(self.n_components, "n_components", 1)
        v.boolean(self.whiten, "whiten")
        v.integer(self.ddof, "ddof")
        v.choice(self.ddof, (0, 1), "ddof")

    def fit(self, samples: ArrayLike) -> PCAResult:
        x = v.samples(samples, minimum=2)
        k = min(x.shape) if self.n_components is None else self.n_components
        if k > min(x.shape):
            raise ValueError("n_components cannot exceed min(n,d)")
        return sklearn.pca(x, n_components=k, whiten=self.whiten, ddof=self.ddof)


@dataclass(frozen=True)
class Whitening:
    """PCA whitening may reduce rank; ZCA requires the full nonsingular basis."""

    method: str = "zca"
    n_components: int | None = None
    ddof: int = 1

    def __post_init__(self) -> None:
        v.choice(self.method, ("pca", "zca"), "method")
        PCA(self.n_components, True, self.ddof)
        if self.method == "zca" and self.n_components is not None:
            raise ValueError(
                "ZCA uses all coordinates; use method='pca' for reduced whitening"
            )

    def fit(self, samples: ArrayLike) -> AffineTransform | PCAResult:
        result = PCA(self.n_components, whiten=True, ddof=self.ddof).fit(samples)
        if self.method == "pca":
            return result
        if result.input_dimension != result.output_dimension:
            raise ValueError("ZCA requires full column rank and n > d")
        basis = result.components
        return AffineTransform(
            result.location, result.encoder @ basis.T, basis @ result.decoder
        )


@dataclass(frozen=True)
class FactorAnalysis:
    """Fit latent factors with diagonal residual variance; n_components sets latent dimension and rotation is explicit."""

    n_components: int
    rotation: str | None = None
    tolerance: float = 1e-6
    maxiter: int = 1000
    check_convergence: bool = True

    def __post_init__(self) -> None:
        v.integer(self.n_components, "n_components", 1)
        v.choice(self.rotation, (None, "varimax", "quartimax"), "rotation")
        v.scalar(self.tolerance, "tolerance", positive=True)
        v.integer(self.maxiter, "maxiter", 1)
        v.boolean(self.check_convergence, "check_convergence")

    def fit(self, samples: ArrayLike) -> FactorAnalysisResult:
        x = v.samples(samples, minimum=2)
        if self.n_components > min(x.shape[0] - 1, x.shape[1]):
            raise ValueError("n_components cannot exceed min(n-1,d)")
        if np.any(np.var(x, axis=0) == 0):
            raise ValueError("factor analysis requires nonconstant training columns")
        result = sklearn.factor_analysis(
            x,
            n_components=self.n_components,
            rotation=self.rotation,
            tolerance=self.tolerance,
            maxiter=self.maxiter,
        )
        if self.check_convergence and not result.diagnostics.converged:
            raise ConvergenceError(result)
        return result


def pca(
    samples: ArrayLike,
    n_components: int | None = None,
    *,
    whiten: bool = False,
    ddof: int = 1,
) -> PCAResult:
    """Fit principal components to samples (observation,feature).

    Columns are original coordinates; n_components selects retained dimension.
    Whitening uses the supplied ddof; discarded components cannot be reconstructed."""
    return PCA(n_components, whiten, ddof).fit(samples)


def whitening(
    samples: ArrayLike,
    *,
    method: str = "zca",
    n_components: int | None = None,
    ddof: int = 1,
) -> AffineTransform:
    """Fit a PCA or ZCA whitening map to samples (observation,feature).

    ddof fixes the covariance normalization; rank-deficient directions require
    explicit dimensional reduction rather than infinite inverse variance."""
    return Whitening(method, n_components, ddof).fit(samples)


def factor_analysis(
    samples: ArrayLike,
    n_components: int,
    *,
    rotation: str | None = None,
    tolerance: float = 1e-6,
    maxiter: int = 1000,
    check_convergence: bool = True,
) -> FactorAnalysisResult:
    """Fit a latent Gaussian factor model to samples (observation,feature).

    Return fitted loadings, noise and transform; n_components selects latent dimension.
    Rotation changes coordinates, not the represented covariance."""
    return FactorAnalysis(
        n_components, rotation, tolerance, maxiter, check_convergence
    ).fit(samples)
