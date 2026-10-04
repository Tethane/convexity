# `convexity.math.control`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Control mathematics on supplied arrays and models.

## `FiniteMDP`

```python
FiniteMDP(transitions: 'ArrayLike', rewards: 'ArrayLike', discount: 'float' = 0.99, feasible: 'ArrayLike | None' = None, terminal: 'ArrayLike | None' = None) -> 'None'
```

Transitions (state,action,next_state), expected rewards (state,action).

Rewards are paid on leaving a state; terminal states pay zero thereafter.
Feasible actions are explicit. Ties resolve to the smallest action index.

## `ControlResult`

```python
ControlResult(values: 'ArrayLike', policy: 'ArrayLike', residual: 'float' = 0.0, iterations: 'int' = 0) -> 'None'
```

Value function and integer action policy on (state,) or (time,state) axes, with Bellman residual and iteration count.

## `BeliefResult`

```python
BeliefResult(value: 'float', action: 'int', action_values: 'ArrayLike', nodes: 'int') -> 'None'
```

Action selected for an input belief, its value, action-value vector and number of evaluated search nodes.

## `LQRResult`

```python
LQRResult(gain: 'ArrayLike', riccati: 'ArrayLike', closed_loop: 'ArrayLike', residual: 'float') -> 'None'
```

Linear-quadratic regulator gain/value matrix and diagnostics for the declared time convention.

## `ScenarioTree`

```python
ScenarioTree(parents: 'ArrayLike', probabilities: 'ArrayLike', disturbances: 'ArrayLike') -> 'None'
```

Root 0; parent precedes child. Conditional branch masses sum to one.

Disturbance (node,state) is realized on arrival, root disturbance is zero.
A node represents an observed history; hidden shocks must not split nodes
before they become observable. One action per node enforces nonanticipation.

### `ScenarioTree.masses`

```python
masses(self) -> 'FloatArray'
```

## `TreeControlResult`

```python
TreeControlResult(states: 'ArrayLike', actions: 'ArrayLike', objective: 'float', max_violation: 'float', solver: 'str') -> 'None'
```

Scenario-tree control solution with node-aligned states/actions and solver diagnostics.

## `ControlSolver`

Solve a control problem and return owned state/action/value information with diagnostics.

### `ControlSolver.solve`

```python
solve(self, model: 'FiniteMDP') -> 'ControlResult'
```

## `finite_horizon`

```python
finite_horizon(model: 'FiniteMDP', horizon: 'int', *, terminal_values: 'ArrayLike | None' = None) -> 'ControlResult'
```

Backward induction for a FiniteMDP over horizon integer decisions.

Return values (horizon+1,state) and policies (horizon,state). Terminal values
are in reward units; discounting follows the MDP. Terminal actions use -1.

## `value_iteration`

```python
value_iteration(model: 'FiniteMDP', *, tolerance: 'float' = 1e-10, maxiter: 'int' = 100000) -> 'ControlResult'
```

Solve a discounted finite MDP to the specified Bellman tolerance.

Return state values and action indices; failure to converge raises explicitly.

## `belief_update`

```python
belief_update(belief: 'ArrayLike', transition: 'ArrayLike', likelihood: 'ArrayLike') -> 'FloatArray'
```

Bayes-update a probability vector after one transition and observation.

transition[i,j] = P(next=j|current=i); likelihood[j] weights the next state.
Return normalized next-state probabilities; zero evidence is an error.

## `belief_plan`

```python
belief_plan(model: 'FiniteMDP', observations: 'ArrayLike', belief: 'ArrayLike', horizon: 'int', *, terminal_values: 'ArrayLike | None' = None, max_nodes: 'int' = 100000) -> 'BeliefResult'
```

Exact finite POMDP: observations (action,next_state,observation).

All actions must be available in every hidden state; terminal state flags
are unsupported (encode absorbing zero-reward states explicitly instead).

## `lqr`

```python
lqr(a: 'ArrayLike', b: 'ArrayLike', q: 'ArrayLike', r: 'ArrayLike') -> 'LQRResult'
```

Discrete infinite-horizon LQR, u=-gain@x; costs x'Qx+u'Ru.

## `scenario_mpc`

```python
scenario_mpc(tree: 'ScenarioTree', a: 'ArrayLike', b: 'ArrayLike', q: 'ArrayLike', r: 'ArrayLike', initial: 'ArrayLike', *, terminal_cost: 'ArrayLike | None' = None, lower_actions: 'ArrayLike' = -inf, upper_actions: 'ArrayLike' = inf, lower_states: 'ArrayLike' = -inf, upper_states: 'ArrayLike' = inf, solver: 'str | None' = None) -> 'TreeControlResult'
```

Solve quadratic control on a ScenarioTree with node probabilities.

Dynamics are x_child=A*x_parent+B*u_parent+disturbance_child. Matrices Q,R
weight state/action costs. Return node states/actions and feasibility diagnostics.

## `generator_hjb`

```python
generator_hjb(generators: 'ArrayLike', rewards: 'ArrayLike', discount_rate: 'float', *, tolerance: 'float' = 1e-10, maxiter: 'int' = 100000) -> 'ControlResult'
```

Solve rho V = max_a(r_a + G_a V) by uniformization.

G axes (state,action,next_state); off-diagonal entries nonnegative and
rows sum to zero. Caller supplies a monotone spatial discretization.

## `impulse_control`

```python
impulse_control(model: 'FiniteMDP', destinations: 'ArrayLike', costs: 'ArrayLike', *, tolerance: 'float' = 1e-10, maxiter: 'int' = 100000) -> 'ControlResult'
```

Add one-period intervention actions: pay cost now, jump next period.

This is a discrete impulse MDP, not an instantaneous continuous-time QVI.
destinations/costs have (state,intervention) axes; costs are nonnegative.
