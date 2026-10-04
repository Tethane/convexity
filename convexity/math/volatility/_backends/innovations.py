"""Standardized arch innovation laws through Convexity's Distribution adapter."""
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


def backend_distribution(name, *, rng=None):
    from arch.univariate import Normal, StudentsT, SkewStudent, GeneralizedError
    return {'normal': Normal, 'student_t': StudentsT, 'skew_student_t': SkewStudent,
            'ged': GeneralizedError}[name](seed=rng)


@dataclass(frozen=True)
class InnovationAdapter:
    name: str
    parameters: Any
    location: float = 0.
    scale: float = 1.
    event_shape: tuple = ()
    measure: Literal['continuous'] = 'continuous'
    capabilities = frozenset({'log_density', 'sample', 'mean', 'variance', 'cdf', 'quantile', 'support'})

    def evaluate(self, operation, *args, **kwargs):
        backend = backend_distribution(self.name, rng=kwargs.get('rng'))
        if operation == 'sample':
            draws = backend.simulate(self.parameters)(int(np.prod(args[0], dtype=int)))
            return np.asarray(self.location + self.scale * draws).reshape(args[0])
        if operation == 'mean':
            return np.asarray(self.location)
        if operation == 'variance':
            return np.asarray(self.scale**2)
        if operation == 'support':
            return np.array([-np.inf, np.inf])
        if operation == 'quantile':
            return np.asarray(self.location + self.scale * backend.ppf(args[0].ravel(), self.parameters)).reshape(args[0].shape)
        x = (args[0] - self.location) / self.scale
        if operation == 'cdf':
            return np.asarray(backend.cdf(x.ravel(), self.parameters)).reshape(x.shape)
        if operation == 'log_density':
            return np.asarray(backend.loglikelihood(self.parameters, x.ravel(), np.ones(x.size), individual=True)).reshape(x.shape) - np.log(self.scale)
        raise NotImplementedError(operation)
