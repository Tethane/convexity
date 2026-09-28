from dataclasses import dataclass
from .. import _validation as v
from .._backends import integration as backend
from ..interfaces.base import run
from ..interfaces.calculus import QuadratureProblem
from ..interfaces.integration import BoxIntegral, QuadratureRuleProblem
from .calculus import AdaptiveQuadrature, quad


@dataclass(frozen=True)
class MultidimensionalQuadrature:
    atol: float = 1e-8
    rtol: float = 1e-8
    limit: int = 100

    def __post_init__(self):
        AdaptiveQuadrature(self.atol, self.rtol, self.limit)

    def __call__(self, problem: BoxIntegral):
        return backend.box_integral(problem, atol=self.atol, rtol=self.rtol, limit=self.limit)


@dataclass(frozen=True)
class GaussianQuadrature:
    n: int = 16

    def __post_init__(self):
        v.integer(self.n, "n")

    def __call__(self, problem: QuadratureProblem):
        return backend.gaussian(problem, n=self.n)


@dataclass(frozen=True)
class GaussianRule:
    def __call__(self, problem: QuadratureRuleProblem):
        return backend.rule(problem)


def nquad(function, bounds, *, tool=None, check_convergence=True):
    return run(BoxIntegral(function, bounds), tool if tool is not None else MultidimensionalQuadrature(),
               check_convergence=check_convergence)


def gauss_quad(function, a, b, *, n=16, tool=None):
    return run(QuadratureProblem(function, a, b), tool if tool is not None else GaussianQuadrature(n))


def quadrature_rule(n, *, family="legendre", a=-1., b=1., tool=None):
    return run(QuadratureRuleProblem(n, family, a, b), tool if tool is not None else GaussianRule())


def integrate_density(function, density, a, b, *, tool=None, check_convergence=True):
    """Integral f(x)*density(x) dx. Density is supplied; no normalization is assumed or fitted."""
    v.function(function)
    v.function(density, "density")

    def weighted(x):
        weight = v.scalar(density(x), "density output")
        if weight < 0:
            raise ValueError("density must be nonnegative at evaluated points")
        return v.scalar(function(x), "function output") * weight

    return quad(weighted, a, b, tool=tool, check_convergence=check_convergence)


__all__ = ["MultidimensionalQuadrature", "GaussianQuadrature", "GaussianRule", "nquad", "gauss_quad",
           "quadrature_rule", "integrate_density"]
