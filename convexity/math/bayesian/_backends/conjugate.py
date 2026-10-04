"""Joint normal/inverse-gamma density and conditional simulation."""
from dataclasses import dataclass
from typing import Any, Literal
import numpy as np
from scipy import stats


@dataclass(frozen=True)
class NormalInverseGammaAdapter:
    parameters: Any
    event_shape: tuple[int, ...] = (2,)
    measure: Literal['continuous'] = 'continuous'
    capabilities = frozenset({'sample', 'log_density', 'mean', 'variance', 'covariance'})

    def evaluate(self, operation, *args, **kwargs):
        p = self.parameters
        if operation == 'sample':
            variance = stats.invgamma.rvs(p.shape, scale=p.scale, size=args[0], random_state=kwargs['rng'])
            mean = kwargs['rng'].normal(p.mean, np.sqrt(variance/p.precision))
            return np.stack([mean, variance], axis=-1)
        if operation == 'log_density':
            x = args[0]
            valid = (x[..., 1] > 0) & np.isfinite(x).all(axis=-1)
            variance = np.where(valid, x[..., 1], 1.)
            result = stats.invgamma.logpdf(variance, p.shape, scale=p.scale)+stats.norm.logpdf(
                x[..., 0], p.mean, np.sqrt(variance/p.precision))
            return np.where(valid, result, -np.inf)
        if operation == 'mean':
            return np.array([p.mean if p.shape > .5 else np.nan,
                             p.scale/(p.shape-1) if p.shape > 1 else np.inf])
        var_mu = p.scale/((p.shape-1)*p.precision) if p.shape > 1 else (np.inf if p.shape > .5 else np.nan)
        var_sigma = p.scale**2/((p.shape-1)**2*(p.shape-2)) if p.shape > 2 else np.inf
        if operation == 'variance':
            return np.array([var_mu, var_sigma])
        # The cross moment exists only for shape > 3/2.
        cross = 0. if p.shape > 1.5 else np.nan
        return np.array([[var_mu, cross], [cross, var_sigma]])
