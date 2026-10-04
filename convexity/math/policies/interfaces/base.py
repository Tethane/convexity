"""Immutable lifecycle state and deterministic policy decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ... import _decision as v


@dataclass(frozen=True)
class PolicyState:
    """Policy lifecycle state: deadline, mode, last update time and reason. It is distinct from market/model state."""

    deadline: float
    mode: str = "waiting"
    last_time: float | None = None
    reason: str = "initialized"

    def __post_init__(self) -> None:
        v.scalar(self.deadline, "deadline")
        v.choice(self.mode, ("waiting", "active", "closed"), "mode")
        if self.last_time is not None:
            v.scalar(self.last_time, "last time")


@dataclass(frozen=True)
class ActionDecision:
    """Policy action with accompanying decision metadata; it does not execute an external trade."""

    action: int
    value: float
    holding_value: float | None
    advantage: float | None
    reason: str


class Policy(Protocol):
    """Map supplied state and context to an ActionDecision without executing external actions."""
    def update(
        self, state: PolicyState, probability: float, time: float, **evidence
    ) -> PolicyState: ...
