"""Fitted GP queries and optimization diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from sklearn.gaussian_process.kernels import Kernel as NativeKernel

if TYPE_CHECKING:
    from sklearn.gaussian_process import GaussianProcessRegressor

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, Protocol, overload

from numpy.typing import NDArray

from .. import _validation as v
from .kernels import Kernel
from .prediction import GaussianPrediction


class Regressor(Protocol):
    """Fit a Gaussian-process prediction contract from supplied feature rows and responses."""
    def fit(
        self, features: ArrayLike, response: ArrayLike
    ) -> "FittedGaussianProcess": ...


@dataclass(frozen=True)
class Diagnostics:
    """Backend termination diagnostics. Inspect converged and message before using a fitted approximation; iterations may be unavailable."""

    converged: bool = True
    optimized: bool = False
    message: str = ""
    iterations: int = 0
    optimizer_runs: int = 0
    warnings: tuple[str, ...] = ()
    backend: str = "sklearn.GaussianProcessRegressor"


class ConvergenceError(RuntimeError):
    """Raised when a checked numerical method does not converge; result retains the attempted solution and diagnostics."""
    def __init__(self, result: FittedGaussianProcess) -> None:
        self.result = result
        super().__init__(
            result.diagnostics.message or "GP hyperparameter fit did not converge"
        )


@dataclass(frozen=True)
class FittedGaussianProcess:
    """A scalar-response GP with private copied backend state.

    All queries take nonempty (m,d) features. alpha affects conditioning only;
    a WhiteNoise component remains part of the predicted kernel's variance.
    """

    _model: GaussianProcessRegressor = field(repr=False)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_model", deepcopy(self._model))

    @property
    def input_dimension(self):
        return self.training_features.shape[1]

    @property
    def nobs(self):
        return len(self._model.X_train_)

    @property
    def training_features(self) -> FloatArray:
        return v.array(self._model.X_train_, "training_features")

    @property
    def kernel(self) -> Kernel:
        return Kernel(cast("NativeKernel", self._model.kernel_))

    def predict(self, features: ArrayLike) -> FloatArray:
        """Posterior mean (m,), without constructing an (m,m) covariance."""
        x = v.features(features, self.input_dimension)
        return v.array(self._model.predict(x), "posterior mean", ndim=1)

    def predict_std(
        self, features: ArrayLike, *, noise_variance: ArrayLike = 0.0
    ) -> FloatArray:
        """Marginal standard deviation (m,), without a full query covariance."""
        x = v.features(features, self.input_dimension)
        noise = v.variance(noise_variance, "noise_variance", len(x))
        _, raw_std = cast(
            "tuple[ArrayLike, ArrayLike]", self._model.predict(x, return_std=True)
        )
        std = v.array(raw_std, "posterior std", ndim=1)
        return v.array(np.sqrt(std**2 + noise), "posterior std", ndim=1)

    def posterior(
        self, features: ArrayLike, *, noise_variance: ArrayLike = 0.0
    ) -> GaussianPrediction:
        """Joint prediction, optionally adding independent future observation noise.

        noise_variance is a scalar or (m,) vector in original target units
        squared. It is separate from training alpha and any WhiteNoise kernel.
        """
        x = v.features(features, self.input_dimension)
        noise = v.variance(noise_variance, "noise_variance", len(x))
        mean, raw_cov = cast(
            "tuple[ArrayLike, ArrayLike]", self._model.predict(x, return_cov=True)
        )
        cov = np.array(raw_cov, dtype=float, copy=True)
        cov[np.diag_indices_from(cov)] += noise
        return GaussianPrediction(mean, cov)

    @overload
    def log_marginal_likelihood(
        self, theta: ArrayLike | None = None, *, eval_gradient: Literal[False] = False
    ) -> float: ...

    @overload
    def log_marginal_likelihood(
        self, theta: ArrayLike | None = None, *, eval_gradient: Literal[True]
    ) -> tuple[float, NDArray[np.float64]]: ...

    @overload
    def log_marginal_likelihood(
        self, theta: ArrayLike | None = None, *, eval_gradient: bool
    ) -> float | tuple[float, NDArray[np.float64]]: ...

    def log_marginal_likelihood(
        self, theta: ArrayLike | None = None, *, eval_gradient: bool = False
    ):
        """Training log marginal likelihood and optional gradient in log coordinates.

        Evaluation at another theta never changes the fitted predictions.
        """
        v.boolean(eval_gradient, "eval_gradient")
        if theta is None and not eval_gradient:
            return float(cast("float", self._model.log_marginal_likelihood()))
        kernel = self.kernel if theta is None else self.kernel.with_theta(theta)
        result = self._model.log_marginal_likelihood(
            kernel.theta, eval_gradient=eval_gradient, clone_kernel=True
        )
        if not eval_gradient:
            return float(cast("float", result))
        value, gradient = cast("tuple[float, ArrayLike]", result)
        return float(value), v.array(
            gradient, "likelihood gradient", ndim=1, empty=True
        )
