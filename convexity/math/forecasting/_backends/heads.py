"""Trainable probability heads with exact scalar laws in original units."""
import numpy as np
import torch
from torch import nn


def channels(spec):
    return {'point': 1, 'gaussian': 2, 'student_t': 3, 'lognormal': 2,
            'mixture': 3 * spec.mixture_components, 'categorical': len(spec.category_values),
            'quantile': len(spec.quantiles), 'custom': spec.output_channels}[spec.head]


def loss(spec, raw, target):
    scale = lambda x: nn.functional.softplus(x) + 1e-6
    if spec.head == 'student_t':
        return -torch.distributions.StudentT(2 + scale(raw[..., 2]), raw[..., 0], scale(raw[..., 1])).log_prob(target)
    if spec.head == 'lognormal':
        # Training values have already been mapped to log units.
        return -torch.distributions.Normal(raw[..., 0], scale(raw[..., 1])).log_prob(target)
    if spec.head == 'mixture':
        k = spec.mixture_components
        law = torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=raw[..., :k]),
                    torch.distributions.Normal(raw[..., k:2*k], scale(raw[..., 2*k:])))
        return -law.log_prob(target)
    values = torch.tensor(spec.category_values, device=target.device, dtype=target.dtype)
    labels = torch.argmin(abs(target[..., None] - values), dim=-1)
    return -torch.distributions.Categorical(logits=raw).log_prob(labels)


def decode(spec, raw, location, scale):
    from ... import distributions as d
    from ..interfaces.result import ForecastResult
    a = raw.detach().cpu().numpy()
    positive = lambda x: np.logaddexp(0, x) + 1e-6
    laws = []
    for h, k in np.ndindex(a.shape[:2]):
        row = a[h, k]
        if spec.head == 'student_t':
            law = d.student_t(2 + positive(row[2]), row[0] * scale[k] + location[k], positive(row[1]) * scale[k])
        elif spec.head == 'lognormal':
            law = d.lognormal(row[0] * scale[k] + location[k], positive(row[1]) * scale[k])
        elif spec.head == 'mixture':
            m = spec.mixture_components
            weights = np.exp(row[:m] - row[:m].max())
            laws_k = [d.normal(row[m+j] * scale[k] + location[k], positive(row[2*m+j]) * scale[k]) for j in range(m)]
            law = d.mixture(laws_k, weights)
        else:
            weights = np.exp(row - row.max())
            law = d.empirical(spec.category_values, weights=weights)
        laws.append(law)
    return ForecastResult.from_independent(laws, a.shape[:2])
