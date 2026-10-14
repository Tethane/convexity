"""Optional Stable-Baselines3 integration on a supplied finite MDP."""

import numpy as np
from ..builtins.agents import TrainedAgent


def train(model, algorithm, steps, seed, max_episode_steps, learning_rate, n_steps):
    try:
        import gymnasium as gym
        from stable_baselines3 import A2C, PPO
    except ImportError as exc:
        raise ImportError(
            "Install convexity[reinforcement] for Stable-Baselines3 agents"
        ) from exc

    class Environment(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Discrete(len(model.transitions))
            self.action_space = gym.spaces.Discrete(model.transitions.shape[1])
            self.state = 0
            self.elapsed = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.state = int(self.np_random.choice(np.flatnonzero(~model.terminal)))
            self.elapsed = 0
            return self.state, {}

        def step(self, action):
            reward = float(model.rewards[self.state, action])
            self.state = int(
                self.np_random.choice(
                    len(model.transitions), p=model.transitions[self.state, action]
                )
            )
            self.elapsed += 1
            terminated = bool(model.terminal[self.state])
            truncated = self.elapsed >= max_episode_steps and not terminated
            return self.state, reward, terminated, truncated, {}

    cls = A2C if algorithm == "a2c" else PPO
    kwargs = {"n_steps": n_steps}
    if algorithm == "ppo":
        kwargs["batch_size"] = min(64, n_steps)
    import random
    import torch
    from typing import Any, cast

    python_state = random.getstate()
    numpy_state = cast(tuple[Any, ...], np.random.get_state(legacy=True))
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            agent = cls(
                "MlpPolicy",
                Environment(),
                gamma=model.discount,
                learning_rate=learning_rate,
                seed=seed,
                device="cpu",
                verbose=0,
                **kwargs,
            )
            agent.learn(total_timesteps=steps)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    return TrainedAgent(agent, len(model.transitions), algorithm)


def action(agent, state):
    return int(np.asarray(agent.predict(state, deterministic=True)[0]).item())
