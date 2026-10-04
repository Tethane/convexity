"""Optional differentiable laws with native PyTorch batch/event conventions.

Tensor laws are separate from NumPy laws: dtype, device, parameter batches and
autograd graphs are preserved. sample/rsample use the native PyTorch RNG;
torch.manual_seed and torch.random.fork_rng control reproducibility. Discrete
laws have no pathwise rsample. No automatic numerical or Monte Carlo fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from torch.distributions import Distribution as TorchDistribution
from dataclasses import dataclass

from ..._typing import SampleSize
from .. import _validation as v
from .._backends.torch import make, torch_module


@dataclass(frozen=True)
class TensorDistribution:
    """An explicit adapter around a native torch.distributions.Distribution.

    The caller retains ownership of tensor parameters. Rebuild the law after
    parameter updates; do not mutate parameters while a graph is in use.
    Native support validation and NotImplementedError behavior propagate.
    """

    native: TorchDistribution
    _cdf: bool = False
    _quantile: bool = False

    def __post_init__(self) -> None:
        torch = torch_module()
        if not isinstance(self.native, torch.distributions.Distribution):
            raise TypeError("native must be a torch.distributions.Distribution")
        if not isinstance(self._cdf, bool) or not isinstance(self._quantile, bool):
            raise TypeError("CDF and quantile capability flags must be boolean")

    @property
    def event_shape(self):
        return tuple(self.native.event_shape)

    @property
    def batch_shape(self):
        return tuple(self.native.batch_shape)

    @property
    def measure(self):
        return (
            "discrete"
            if getattr(self.native.support, "is_discrete", False)
            else "continuous"
        )

    @property
    def capabilities(self):
        capabilities = {"density", "log_density", "sample"}
        if self.native.has_rsample:
            capabilities.add("rsample")
        if self._cdf:
            capabilities.add("cdf")
        if self._quantile:
            capabilities.add("quantile")
        # Native properties may explicitly raise NotImplementedError for custom
        # or composite torch distributions, so these are not globally advertised.
        torch = torch_module()
        if isinstance(
            self.native,
            (
                torch.distributions.Normal,
                torch.distributions.StudentT,
                torch.distributions.LogNormal,
                torch.distributions.Bernoulli,
                torch.distributions.Binomial,
                torch.distributions.Poisson,
                torch.distributions.MultivariateNormal,
            ),
        ):
            capabilities |= {"mean", "variance", "std"}
            if not self.event_shape:
                capabilities.add("covariance")
        if isinstance(self.native, torch.distributions.MultivariateNormal):
            capabilities.add("covariance")
        return frozenset(capabilities)

    def log_density(self, value):
        return self.native.log_prob(value)

    def density(self, value):
        return self.log_density(value).exp()

    def cdf(self, value):
        if not self._cdf:
            raise NotImplementedError("CDF is not declared for this tensor law")
        return self.native.cdf(value)

    def quantile(self, probability: ArrayLike):
        if not self._quantile:
            raise NotImplementedError("quantile is not declared for this tensor law")
        torch = torch_module()
        if (
            not isinstance(probability, torch.Tensor)
            or not probability.is_floating_point()
        ):
            raise TypeError("probability must be a real floating-point tensor")
        if not bool(
            (
                torch.isfinite(probability) & (probability >= 0) & (probability <= 1)
            ).all()
        ):
            raise ValueError("probability must lie in [0, 1]")
        return self.native.icdf(probability)

    def sample(self, size: SampleSize = ()):
        return self.native.sample(v.sample_shape(size))

    def rsample(self, size: SampleSize = ()):
        """Pathwise differentiable samples, only when the backend supports them."""
        if not self.native.has_rsample:
            raise NotImplementedError("this tensor law has no reparameterized sampler")
        return self.native.rsample(v.sample_shape(size))

    def mean(self):
        return self.native.mean

    def variance(self):
        return self.native.variance

    def std(self):
        return self.native.stddev

    def covariance(self):
        if not self.event_shape:
            return self.variance()
        if not hasattr(self.native, "covariance_matrix"):
            raise NotImplementedError(
                "native tensor law does not expose covariance_matrix"
            )
        return getattr(self.native, "covariance_matrix")


def from_torch(law, *, cdf: bool = False, quantile: bool = False) -> TensorDistribution:
    """Wrap a native law; explicitly declare any supported CDF/ICDF operations."""
    return TensorDistribution(law, cdf, quantile)


def normal(loc: float = 0.0, scale: float = 1.0) -> TensorDistribution:
    return TensorDistribution(make("Normal", loc, scale), True, True)


def student_t(df: float, loc: float = 0.0, scale: float = 1.0) -> TensorDistribution:
    return TensorDistribution(make("StudentT", df, loc, scale))


def lognormal(log_loc: float = 0.0, log_scale: float = 1.0) -> TensorDistribution:
    return TensorDistribution(make("LogNormal", log_loc, log_scale), True, True)


def bernoulli(p: float) -> TensorDistribution:
    return TensorDistribution(make("Bernoulli", probs=p))


def binomial(n: int, p: float) -> TensorDistribution:
    return TensorDistribution(make("Binomial", total_count=n, probs=p))


def poisson(rate: float) -> TensorDistribution:
    return TensorDistribution(make("Poisson", rate))


def multivariate_normal(mean: ArrayLike, covariance: ArrayLike) -> TensorDistribution:
    return TensorDistribution(
        make("MultivariateNormal", mean, covariance_matrix=covariance)
    )


def independent(
    base: TensorDistribution, reinterpreted_batch_ndims: int = 1
) -> TensorDistribution:
    """Reinterpret trailing parameter batch axes as independent event axes."""
    n = v.integer(reinterpreted_batch_ndims, "reinterpreted_batch_ndims", minimum=1)
    return TensorDistribution(make("Independent", base.native, n))


def mixture(weights: ArrayLike, components: TensorDistribution) -> TensorDistribution:
    """Same-family mixture; the last component batch axis indexes mixture members."""
    selector = make("Categorical", probs=weights)
    return TensorDistribution(make("MixtureSameFamily", selector, components.native))


def transformed(base: TensorDistribution, transforms) -> TensorDistribution:
    """Compose native torch Transform objects, retaining their Jacobians/graphs."""
    return TensorDistribution(make("TransformedDistribution", base.native, transforms))


__all__ = [
    "TensorDistribution",
    "from_torch",
    "normal",
    "student_t",
    "lognormal",
    "bernoulli",
    "binomial",
    "poisson",
    "multivariate_normal",
    "independent",
    "mixture",
    "transformed",
]
