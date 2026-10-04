"""Shared simulation checks and explicit sampling-design labels."""
from ..processes._validation import (array, boolean, choice, covariance, generator,
                                     integer, label, sample_shape, scalar, shape,
                                     shaped, times)
from ..distributions._validation import weights


def sampling(value):
    return choice(value, ('iid', 'antithetic', 'sobol', 'halton', 'unspecified'), 'sampling')
