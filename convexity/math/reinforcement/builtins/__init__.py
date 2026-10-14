"""Concrete specifications and convenience functions for reinforcement.

Ordinary workflows may import these names directly from convexity.math.reinforcement.
"""

from .agents import (
    q_learning,
    fitted_q,
    bellman_targets,
    bellman_loss,
    actor_critic_loss,
    TrainedAgent,
    train_agent,
)

__all__ = [
    "q_learning",
    "fitted_q",
    "bellman_targets",
    "bellman_loss",
    "actor_critic_loss",
    "TrainedAgent",
    "train_agent",
]
