# `convexity.math.reinforcement`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Reinforcement mathematics on supplied arrays and models.

## `QResult`

```python
QResult(q_values: 'ArrayLike', policy: 'ArrayLike', residual: 'float', iterations: 'int', coefficients: 'ArrayLike | None' = None) -> 'None'
```

Learned state-action values and training information in the declared finite state/action order.

## `Agent`

Map a state to an action; generic parameters retain the user-defined state and action types.

### `Agent.action`

```python
action(self, state: 'State') -> 'Action'
```

## `q_learning`

```python
q_learning(model: 'FiniteMDP', *, steps: 'int' = 100000, epsilon: 'float' = 0.2, learning_rate: 'float | None' = None, rng: 'np.random.Generator') -> 'QResult'
```

Sampled tabular Q-learning; residual is diagnostic, never a success claim.

## `fitted_q`

```python
fitted_q(model: 'FiniteMDP', features: 'ArrayLike', *, ridge: 'float' = 0.0, tolerance: 'float' = 1e-09, maxiter: 'int' = 10000) -> 'QResult'
```

Model-based fitted Q with linear state/action features and least squares.

One-hot features recover tabular iteration. Projected iteration can diverge;
convergence of coefficients need not mean the Bellman residual is small.

## `bellman_targets`

```python
bellman_targets(rewards: 'ArrayLike', next_q: 'ArrayLike', terminal: 'ArrayLike', *, discount: 'float' = 0.99, feasible: 'ArrayLike | None' = None) -> 'FloatArray'
```

Batch masked max targets; terminal rows never bootstrap.

## `bellman_loss`

```python
bellman_loss(predictions: 'ArrayLike', targets: 'ArrayLike', *, weights: 'ArrayLike | None' = None) -> 'float'
```

Return weighted mean squared error between matching prediction and Bellman-target arrays.

## `actor_critic_loss`

```python
actor_critic_loss(logits: 'NativeTensor', actions: 'NativeTensor', values: 'NativeTensor', returns: 'NativeTensor', feasible: 'NativeTensor', *, value_weight: 'float' = 1.0, entropy_weight: 'float' = 0.0) -> 'NativeTensor'
```

Tensor scalar loss with masked categorical actions; gradients stay intact.

## `TrainedAgent`

```python
TrainedAgent(_native: 'NativeRLModel', states: 'int', algorithm: 'str') -> None
```

Trained policy wrapper with observation/action conventions and native training state.

### `TrainedAgent.action`

```python
action(self, state: 'int')
```

## `train_agent`

```python
train_agent(model: 'FiniteMDP', *, algorithm: 'str' = 'a2c', steps: 'int' = 10000, seed: 'int' = 0, max_episode_steps: 'int' = 100, learning_rate: 'float' = 0.001, n_steps: 'int' = 32) -> 'TrainedAgent'
```

A2C/PPO finite-state integration; all actions must be admissible.

SB3 vanilla categorical policies do not implement state-dependent masking.
Use tabular/fitted-Q or the masked tensor loss when such constraints apply.
