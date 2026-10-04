"""Unit-variance innovation distributions, including skewed and heavy-tailed laws."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from ...distributions import Distribution


import numpy as np

from ...forecasting import _validation as v


def innovation_distribution(
    name: str = "normal", parameters: ArrayLike = ()
) -> Distribution:
    """Normal: (); Student-t: (nu,); skew Student-t: (eta,lambda); GED: (nu,).

    Every law has mean zero and variance one, unlike an unscaled Student-t.
    Shape-parameter constraints are those of the arch distribution backend.
    """
    from ...distributions import Distribution
    from .._backends.innovations import InnovationAdapter, backend_distribution

    v.choice(name, ("normal", "student_t", "skew_student_t", "ged"), "name")
    parameters = v.array(parameters, "innovation parameters", ndim=1, empty=True)
    backend = backend_distribution(name)
    if len(parameters) != backend.num_params:
        raise ValueError(f"{name} requires parameters {backend.parameter_names()}")
    a, b = backend.constraints()
    if np.any(a @ parameters < b) or (
        name == "skew_student_t" and abs(parameters[1]) >= 1
    ):
        raise ValueError("innovation shape parameters violate backend constraints")
    return Distribution(InnovationAdapter(name, parameters))
