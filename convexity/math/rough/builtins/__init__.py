"""Concrete specifications and convenience functions for rough.

Ordinary workflows may import these names directly from convexity.math.rough.
See docs/math/reference/rough.md for public signatures and mathematical conventions.
"""
from .kernels import FractionalKernel, ExponentialKernel, CallableKernel
from .drivers import FractionalBrownianMotion, VolterraGaussian
from .approximation import approximate_fractional, kernel_error
from .variance import ForwardVarianceCurve, ForwardVarianceState, LognormalVolterraVariance
from .bergomi import RoughBergomi, VolterraBergomi, BergomiPaths

__all__ = ['FractionalKernel', 'ExponentialKernel', 'CallableKernel',
           'FractionalBrownianMotion', 'VolterraGaussian', 'approximate_fractional',
           'kernel_error', 'ForwardVarianceCurve', 'ForwardVarianceState',
           'LognormalVolterraVariance', 'RoughBergomi', 'VolterraBergomi', 'BergomiPaths']
