"""Conditional simulated observations on supplied joint latent paths."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..._typing import FloatArray

if TYPE_CHECKING:
    from .. import PathScenarios
from dataclasses import dataclass

import numpy as np

from ...distributions import Distribution
from ...processes import Transition
from .. import _validation as v
from ..interfaces import PathScenarios


@dataclass(frozen=True)
class SimulatedObservations:
    """Observed values generated from a latent simulation, preserving observation-time alignment."""

    latent: PathScenarios
    observations: PathScenarios

    def __post_init__(self) -> None:
        if not isinstance(self.latent, PathScenarios) or not isinstance(
            self.observations, PathScenarios
        ):
            raise TypeError("latent and observations must be PathScenarios")
        if (
            not np.array_equal(self.latent.times, self.observations.times)
            or self.latent.weights.shape != self.observations.weights.shape
            or not np.allclose(
                self.latent.weights, self.observations.weights, rtol=1e-14, atol=0
            )
        ):
            raise ValueError(
                "latent and observations must align by time and scenario weight"
            )

    def joint(self) -> PathScenarios:
        """One vector state per time, latent coordinates then observations."""
        n, t = self.latent.values.shape[:2]
        values = np.concatenate(
            (
                self.latent.values.reshape(n, t, -1),
                self.observations.values.reshape(n, t, -1),
            ),
            axis=2,
        )
        return PathScenarios(
            self.latent.times,
            values,
            self.latent.weights,
            sampling=self.observations.sampling,
            method="latent-and-observation",
        )


def simulate_observations(
    latent: PathScenarios,
    observation: Callable[[float, FloatArray], Distribution | Transition],
    *,
    rng: np.random.Generator,
) -> SimulatedObservations:
    """observation(time,state) returns Distribution or Transition.

    Draws are conditionally independent across grid times and rows given the
    entire latent path. Correlated observation noise must instead be included
    in an augmented latent process. The first grid point is observed as well.
    """
    if not isinstance(latent, PathScenarios):
        raise TypeError("latent must be PathScenarios")
    if not callable(observation):
        raise TypeError("observation must be callable")
    v.generator(rng)
    shape = None
    rows = []
    for path in latent.values:
        row = []
        for time, state in zip(latent.times, path):
            kernel = observation(time, state)
            if not isinstance(kernel, (Distribution, Transition)):
                raise TypeError("observation must return a Distribution or Transition")
            if shape is None:
                shape = kernel.event_shape
            if kernel.event_shape != shape:
                raise ValueError("observation event_shape must remain constant")
            row.append(kernel.sample(rng=rng))
        rows.append(row)
    # Independent observation noise does not itself preserve a QMC net.
    sampling = "iid" if latent.sampling == "iid" else "unspecified"
    observed = PathScenarios(
        latent.times, rows, latent.weights, sampling, "conditional-observation"
    )
    return SimulatedObservations(latent, observed)
