"""exact control versus sampled and approximated value learning.

The base example needs no optional tensor libraries. Install [reinforcement]
and pass --neural to train the Stable-Baselines3 actor-critic backend too.
"""

import sys
import numpy as np
from convexity.math import control, reinforcement

mdp = control.FiniteMDP([[[1.0], [1.0]]], [[0.0, 1.0]], 0.5)
exact = control.value_iteration(mdp)
sampled = reinforcement.q_learning(
    mdp, steps=30000, epsilon=0.4, rng=np.random.default_rng(3600)
)
fitted = reinforcement.fitted_q(mdp, np.eye(2).reshape(1, 2, 2))
print(
    "Exact value:",
    exact.values,
    "sampled Q:",
    sampled.q_values,
    "fitted Q:",
    fitted.q_values,
)
print("Bellman residuals:", sampled.residual, fitted.residual)
if "--neural" in sys.argv:
    agent = reinforcement.train_agent(mdp, steps=512, seed=11, learning_rate=0.01)
    print("Actor-critic action:", agent.action(0))
