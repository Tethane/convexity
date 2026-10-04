"""Thresholds and no-action bands; preferences never alter wealth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ... import _decision as v
from .._backends import transitions
from ..interfaces import ActionDecision, PolicyState


@dataclass(frozen=True)
class Hysteresis:
    """Policy switching with separate entry/exit thresholds to avoid boundary chattering."""

    entry: float = 0.9
    exit: float = 0.6
    convergence_gap: float = 0.0

    def __post_init__(self) -> None:
        v.probability(self.entry, "entry")
        v.probability(self.exit, "exit")
        if self.entry <= self.exit:
            raise ValueError("entry threshold must exceed exit threshold")
        v.scalar(self.convergence_gap, "convergence gap", minimum=0)

    def update(
        self,
        state: PolicyState,
        probability: float,
        time: float,
        *,
        gap: float | None = None,
        adverse: bool = False,
        mandatory: bool = False,
        scheduled: bool = True,
    ):
        p, t = v.probability(probability), v.scalar(time, "time")
        if state.last_time is not None and t < state.last_time:
            raise ValueError("policy time cannot move backward")
        convergence = gap is not None and v.scalar(gap, "gap") <= self.convergence_gap
        return transitions.update(
            state,
            p,
            t,
            self.entry,
            self.exit,
            convergence,
            v.boolean(adverse, "adverse"),
            v.boolean(mandatory, "mandatory"),
            v.boolean(scheduled, "scheduled"),
        )


def no_action(
    values: ArrayLike,
    *,
    hold: int = 0,
    feasible: ArrayLike | None = None,
    switching: float = 0.0,
    mandatory: int | None = None,
) -> ActionDecision:
    """Maximize action values less switching cost. Hold wins exact ties.

    `mandatory` is an explicit feasible action index and overrides the band.
    If hold is infeasible, choose the first maximizing feasible alternative.
    """
    x = v.array(values, "values", ndim=1)
    hold = v.integer(hold, "hold")
    if hold >= len(x):
        raise ValueError("hold index out of range")
    f = (
        np.ones(len(x), dtype=bool)
        if feasible is None
        else v.mask(feasible, x.shape, "feasible")
    )
    if not f.any():
        raise ValueError("no feasible action")
    penalty = v.scalar(switching, "switching", minimum=0)
    adjusted = x - penalty * (np.arange(len(x)) != hold)
    best = int(np.argmax(np.where(f, adjusted, -np.inf)))
    if f[hold] and adjusted[best] <= x[hold]:
        best = hold
    reason = "hold" if best == hold else "switch" if f[hold] else "infeasible_hold"
    if mandatory is not None:
        best = v.integer(mandatory, "mandatory")
        if best >= len(x) or not f[best]:
            raise ValueError("mandatory action must be feasible")
        reason = "mandatory"
    holding = float(x[hold]) if f[hold] else None
    return ActionDecision(
        best,
        float(adjusted[best]),
        holding,
        float(adjusted[best] - x[hold]) if f[hold] else None,
        reason,
    )
