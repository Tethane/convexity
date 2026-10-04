"""Independent products and finite mixtures of compatible public laws."""

from __future__ import annotations

from collections.abc import Sequence

from numpy.typing import ArrayLike

from .. import _validation as v
from .._backends.composition import IndependentAdapter, MixtureAdapter
from ..interfaces import Distribution


def _components(components):
    components = tuple(components)
    if not components or any(not isinstance(c, Distribution) for c in components):
        raise TypeError(
            "components must be a nonempty sequence of Distribution objects"
        )
    if len({c.measure for c in components}) != 1:
        raise ValueError(
            "components must use the same measure (continuous or discrete)"
        )
    return components


def independent(components: Sequence[Distribution]) -> Distribution:
    """Vector law of independent scalar components, all discrete or all continuous."""
    components = _components(components)
    if any(c.event_shape for c in components):
        raise ValueError("independent components must be scalar laws")
    return Distribution(IndependentAdapter(components))


def mixture(
    components: Sequence[Distribution], weights: ArrayLike | None = None
) -> Distribution:
    """Finite mixture; component shapes and reference measures must agree.

    Relative weights are normalized and zero-weight components are discarded.
    Mixing a continuous density with atom masses is intentionally unsupported.
    """
    components = _components(components)
    if len({c.event_shape for c in components}) != 1:
        raise ValueError("mixture components must have the same event shape")
    mass = v.weights(weights, len(components))
    positive = mass > 0
    components = tuple(c for c, keep in zip(components, positive) if keep)
    normalized = mass[positive] / mass[positive].sum()
    normalized.setflags(write=False)
    return Distribution(MixtureAdapter(components, normalized))
