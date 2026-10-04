"""Reinforcement mathematics on supplied arrays and models."""

from . import interfaces, builtins
from .interfaces import QResult, Agent
from .builtins import (
    q_learning,
    fitted_q,
    bellman_targets,
    bellman_loss,
    actor_critic_loss,
    TrainedAgent,
    train_agent,
)

__all__ = [
    "interfaces",
    "builtins",
    "QResult",
    "Agent",
    "q_learning",
    "fitted_q",
    "bellman_targets",
    "bellman_loss",
    "actor_critic_loss",
    "TrainedAgent",
    "train_agent",
]
