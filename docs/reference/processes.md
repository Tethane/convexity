# `convexity.math.processes`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Stochastic processes and differential equations (family 13).

Explicit Ito coefficients, generators, exact laws, and labeled numerical
transitions. NumPy generators are required; optional backends load on use.

## `BrownianMotion`

```python
BrownianMotion() -> None
```

Standard scalar Wiener process; the initial level is supplied separately.

### `BrownianMotion.diffusion`

```python
diffusion(self, time: 'float', state) -> 'FloatArray'
```

### `BrownianMotion.drift`

```python
drift(self, time: 'float', state) -> 'FloatArray'
```

### `BrownianMotion.generator`

```python
generator(self, time: 'float', state, gradient: 'ArrayLike | Callable[[FloatArray], ArrayLike]', hessian: 'ArrayLike | Callable[[FloatArray], ArrayLike]')
```

### `BrownianMotion.sample`

```python
sample(self, initial: 'ArrayLike', times, *, draws: 'int' = 1, rng: 'np.random.Generator', method: 'str' = 'exact', substeps: 'int' = 1)
```

### `BrownianMotion.transition`

```python
transition(self, state, dt: 'float', *, time: 'float' = 0.0, method: 'str' = 'exact', substeps: 'int' = 1)
```

## `ArithmeticBrownianMotion`

```python
ArithmeticBrownianMotion(mu: 'float' = 0.0, sigma: 'float' = 1.0) -> None
```

dX = mu dt + sigma dW. sigma is standard deviation per sqrt(time).

### `ArithmeticBrownianMotion.diffusion`

```python
diffusion(self, time: 'float', state) -> 'FloatArray'
```

### `ArithmeticBrownianMotion.drift`

```python
drift(self, time: 'float', state) -> 'FloatArray'
```

### `ArithmeticBrownianMotion.generator`

```python
generator(self, time: 'float', state, gradient: 'ArrayLike | Callable[[FloatArray], ArrayLike]', hessian: 'ArrayLike | Callable[[FloatArray], ArrayLike]')
```

### `ArithmeticBrownianMotion.sample`

```python
sample(self, initial: 'ArrayLike', times, *, draws: 'int' = 1, rng: 'np.random.Generator', method: 'str' = 'exact', substeps: 'int' = 1)
```

### `ArithmeticBrownianMotion.transition`

```python
transition(self, state, dt: 'float', *, time: 'float' = 0.0, method: 'str' = 'exact', substeps: 'int' = 1)
```

## `OrnsteinUhlenbeck`

```python
OrnsteinUhlenbeck(kappa: 'float', theta: 'float' = 0.0, sigma: 'float' = 1.0) -> None
```

dX = kappa(theta-X) dt + sigma dW; kappa=0 is driftless Brownian motion.

### `OrnsteinUhlenbeck.diffusion`

```python
diffusion(self, time: 'float', state) -> 'FloatArray'
```

### `OrnsteinUhlenbeck.drift`

```python
drift(self, time: 'float', state) -> 'FloatArray'
```

### `OrnsteinUhlenbeck.generator`

```python
generator(self, time: 'float', state, gradient: 'ArrayLike | Callable[[FloatArray], ArrayLike]', hessian: 'ArrayLike | Callable[[FloatArray], ArrayLike]')
```

### `OrnsteinUhlenbeck.sample`

```python
sample(self, initial: 'ArrayLike', times, *, draws: 'int' = 1, rng: 'np.random.Generator', method: 'str' = 'exact', substeps: 'int' = 1)
```

### `OrnsteinUhlenbeck.stationary`

```python
stationary(self)
```

### `OrnsteinUhlenbeck.transition`

```python
transition(self, state, dt: 'float', *, time: 'float' = 0.0, method: 'str' = 'exact', substeps: 'int' = 1)
```

## `LinearDiffusion`

```python
LinearDiffusion(matrix: 'ArrayLike', loading: 'ArrayLike', forcing: 'ArrayLike | None' = None) -> 'None'
```

dX=(A X+b)dt+B dW with vector state, including singular A and noise.

The exact semigroup reuses the family-10 SciPy matrix-exponential adapter.

### `LinearDiffusion.coefficients`

```python
coefficients(self, dt: 'float')
```

Return (F, offset, Q) for X[t+dt]=F X[t]+offset+N(0,Q).

### `LinearDiffusion.diffusion`

```python
diffusion(self, time: 'float', state)
```

### `LinearDiffusion.drift`

```python
drift(self, time: 'float', state) -> 'FloatArray'
```

### `LinearDiffusion.generator`

```python
generator(self, time: 'float', state, gradient: 'ArrayLike | Callable[[FloatArray], ArrayLike]', hessian: 'ArrayLike | Callable[[FloatArray], ArrayLike]')
```

### `LinearDiffusion.sample`

```python
sample(self, initial: 'ArrayLike', times, *, draws: 'int' = 1, rng: 'np.random.Generator', method: 'str' = 'exact', substeps: 'int' = 1)
```

### `LinearDiffusion.stationary`

```python
stationary(self)
```

### `LinearDiffusion.transition`

```python
transition(self, state, dt: 'float', *, time: 'float' = 0.0, method: 'str' = 'exact', substeps: 'int' = 1)
```

## `VectorOrnsteinUhlenbeck`

```python
VectorOrnsteinUhlenbeck(kappa: 'ArrayLike', theta: 'ArrayLike', loading: 'ArrayLike') -> 'None'
```

dX=K(theta-X)dt+B dW; stability requires eigenvalues(K).real > 0.

### `VectorOrnsteinUhlenbeck.coefficients`

```python
coefficients(self, dt: 'float')
```

Return (F, offset, Q) for X[t+dt]=F X[t]+offset+N(0,Q).

### `VectorOrnsteinUhlenbeck.diffusion`

```python
diffusion(self, time: 'float', state)
```

### `VectorOrnsteinUhlenbeck.drift`

```python
drift(self, time: 'float', state) -> 'FloatArray'
```

### `VectorOrnsteinUhlenbeck.generator`

```python
generator(self, time: 'float', state, gradient: 'ArrayLike | Callable[[FloatArray], ArrayLike]', hessian: 'ArrayLike | Callable[[FloatArray], ArrayLike]')
```

### `VectorOrnsteinUhlenbeck.sample`

```python
sample(self, initial: 'ArrayLike', times, *, draws: 'int' = 1, rng: 'np.random.Generator', method: 'str' = 'exact', substeps: 'int' = 1)
```

### `VectorOrnsteinUhlenbeck.stationary`

```python
stationary(self)
```

### `VectorOrnsteinUhlenbeck.transition`

```python
transition(self, state, dt: 'float', *, time: 'float' = 0.0, method: 'str' = 'exact', substeps: 'int' = 1)
```

## `GeometricBrownianMotion`

```python
GeometricBrownianMotion(mu: 'float' = 0.0, sigma: 'float' = 1.0) -> None
```

dS=mu S dt+sigma S dW. mu is the expected proportional drift.

### `GeometricBrownianMotion.diffusion`

```python
diffusion(self, time: 'float', state) -> 'FloatArray'
```

### `GeometricBrownianMotion.drift`

```python
drift(self, time: 'float', state) -> 'FloatArray'
```

### `GeometricBrownianMotion.generator`

```python
generator(self, time: 'float', state, gradient: 'ArrayLike | Callable[[FloatArray], ArrayLike]', hessian: 'ArrayLike | Callable[[FloatArray], ArrayLike]')
```

### `GeometricBrownianMotion.sample`

```python
sample(self, initial: 'ArrayLike', times, *, draws: 'int' = 1, rng: 'np.random.Generator', method: 'str' = 'exact', substeps: 'int' = 1)
```

### `GeometricBrownianMotion.transition`

```python
transition(self, state, dt: 'float', *, time: 'float' = 0.0, method: 'str' = 'exact', substeps: 'int' = 1)
```

## `CIR`

```python
CIR(kappa: 'float', theta: 'float', sigma: 'float') -> None
```

dX=kappa(theta-X)dt+sigma sqrt(X)dW; exact noncentral chi-square law.

kappa,theta > 0; sigma >= 0. Feller violations are allowed. theta=0's
mixed atom/continuous transition is outside this constructor's scope.

### `CIR.diffusion`

```python
diffusion(self, time: 'float', state) -> 'FloatArray'
```

### `CIR.drift`

```python
drift(self, time: 'float', state) -> 'FloatArray'
```

### `CIR.generator`

```python
generator(self, time: 'float', state, gradient: 'ArrayLike | Callable[[FloatArray], ArrayLike]', hessian: 'ArrayLike | Callable[[FloatArray], ArrayLike]')
```

### `CIR.sample`

```python
sample(self, initial: 'ArrayLike', times, *, draws: 'int' = 1, rng: 'np.random.Generator', method: 'str' = 'exact', substeps: 'int' = 1)
```

### `CIR.stationary`

```python
stationary(self)
```

### `CIR.transition`

```python
transition(self, state, dt: 'float', *, time: 'float' = 0.0, method: 'str' = 'exact', substeps: 'int' = 1)
```

## `Heston`

```python
Heston(mu: 'float', kappa: 'float', theta: 'float', sigma: 'float', rho: 'float') -> None
```

dS=mu S dt+sqrt(V) S dW1; dV=kappa(theta-V)dt+sigma sqrt(V)dW2.

State is (S,V), corr(dW1,dW2)=rho. mu is supplied by the caller; this
object assigns no pricing/physical measure. QE is not an exact joint law.

### `Heston.diffusion`

```python
diffusion(self, time: 'float', state) -> 'FloatArray'
```

### `Heston.drift`

```python
drift(self, time: 'float', state) -> 'FloatArray'
```

### `Heston.generator`

```python
generator(self, time: 'float', state, gradient: 'ArrayLike | Callable[[FloatArray], ArrayLike]', hessian: 'ArrayLike | Callable[[FloatArray], ArrayLike]')
```

### `Heston.sample`

```python
sample(self, initial: 'ArrayLike', times, *, draws: 'int' = 1, rng: 'np.random.Generator', method: 'str' = 'qe', substeps: 'int' = 1)
```

### `Heston.transition`

```python
transition(self, state, dt: 'float', *, time: 'float' = 0.0, method: 'str' = 'qe', substeps: 'int' = 1)
```

## `ItoDiffusion`

```python
ItoDiffusion(drift_function: 'Callable[[float, FloatArray], ArrayLike]', diffusion_function: 'Callable[[float, FloatArray], ArrayLike]', event_shape: 'tuple[int, ...]' = (), noise_dimension: 'int' = 1) -> None
```

Callbacks drift(time,state), diffusion(time,state); independent Wiener drivers.

For scalar state (), diffusion returns (m,). For vector (d,), it returns
(d,m). Correlation belongs in this loading, not in an implicit driver law.
Coefficients are Ito, not Stratonovich. Callbacks receive owned read-only states.

### `ItoDiffusion.diffusion`

```python
diffusion(self, time: 'float', state) -> 'FloatArray'
```

### `ItoDiffusion.drift`

```python
drift(self, time: 'float', state) -> 'FloatArray'
```

### `ItoDiffusion.generator`

```python
generator(self, time: 'float', state, gradient: 'ArrayLike | Callable[[FloatArray], ArrayLike]', hessian: 'ArrayLike | Callable[[FloatArray], ArrayLike]')
```

### `ItoDiffusion.sample`

```python
sample(self, initial: 'ArrayLike', times, *, draws: 'int' = 1, rng: 'np.random.Generator', method: 'str' = 'exact', substeps: 'int' = 1)
```

### `ItoDiffusion.transition`

```python
transition(self, state, dt: 'float', *, time: 'float' = 0.0, method: 'str' = 'exact', substeps: 'int' = 1)
```

## `AdditiveJumpDiffusion`

```python
AdditiveJumpDiffusion(mu: 'ArrayLike', loading: 'ArrayLike', intensity: 'float', jumps: 'Distribution') -> 'None'
```

dX=mu dt+B dW+J dN, N has constant intensity; marks are IID.

mu is the uncompensated continuous drift. Scalar loading has shape (m,),
vector loading (d,m). Existing Distribution marks supply size/shape laws.
No common-reference density is asserted for this potentially mixed kernel.

### `AdditiveJumpDiffusion.diffusion`

```python
diffusion(self, time: 'float', state)
```

### `AdditiveJumpDiffusion.drift`

```python
drift(self, time: 'float', state)
```

### `AdditiveJumpDiffusion.generator`

```python
generator(self, time: 'float', state, gradient: 'ArrayLike | Callable[[FloatArray], ArrayLike]', hessian: 'ArrayLike | Callable[[FloatArray], ArrayLike]', *, function, jump_expectation=None)
```

Lf=continuous part+lambda E[f(x+J)-f(x)].

Supply jump_expectation explicitly if the mark law cannot integrate f.
It must be the expectation of the difference, without the intensity.

### `AdditiveJumpDiffusion.sample`

```python
sample(self, initial: 'ArrayLike', times, *, draws: 'int' = 1, rng: 'np.random.Generator', method: 'str' = 'exact', substeps: 'int' = 1)
```

### `AdditiveJumpDiffusion.transition`

```python
transition(self, state, dt: 'float', *, time: 'float' = 0.0, method: 'str' = 'exact', substeps: 'int' = 1)
```

## `MertonJumpDiffusion`

```python
MertonJumpDiffusion(mu: 'float', sigma: 'float', intensity: 'float', jump_mean: 'float', jump_sigma: 'float') -> None
```

dS/S-=(mu-lambda E[e^J-1])dt+sigma dW+(e^J-1)dN.

J~Normal(jump_mean,jump_sigma). mu is total expected proportional drift.
Exact terminal sampling conditions on Poisson counts; no mixture truncation.

### `MertonJumpDiffusion.diffusion`

```python
diffusion(self, time: 'float', state) -> 'FloatArray'
```

### `MertonJumpDiffusion.drift`

```python
drift(self, time: 'float', state) -> 'FloatArray'
```

### `MertonJumpDiffusion.generator`

```python
generator(self, time: 'float', state, gradient: 'ArrayLike | Callable[[FloatArray], ArrayLike]', hessian: 'ArrayLike | Callable[[FloatArray], ArrayLike]', *, function, jump_expectation=None)
```

### `MertonJumpDiffusion.sample`

```python
sample(self, initial: 'ArrayLike', times, *, draws: 'int' = 1, rng: 'np.random.Generator', method: 'str' = 'exact', substeps: 'int' = 1)
```

### `MertonJumpDiffusion.transition`

```python
transition(self, state, dt: 'float', *, time: 'float' = 0.0, method: 'str' = 'exact', substeps: 'int' = 1)
```

## `StochasticProcess`

Conditional-transition interface with declared event_shape; the returned Transition states available sampling and law capabilities.

### `StochasticProcess.transition`

```python
transition(self, state: 'ArrayLike', dt: 'float', *, time: 'float' = 0.0, method: 'str' = 'exact', substeps: 'int' = 1) -> 'Transition'
```

## `ItoProcess`

Ito drift/diffusion extension; state, time and driving-noise coordinates follow the process contract.

### `ItoProcess.diffusion`

```python
diffusion(self, time: 'float', state: 'ArrayLike') -> 'FloatArray'
```

### `ItoProcess.drift`

```python
drift(self, time: 'float', state: 'ArrayLike') -> 'FloatArray'
```

### `ItoProcess.transition`

```python
transition(self, state: 'ArrayLike', dt: 'float', *, time: 'float' = 0.0, method: 'str' = 'exact', substeps: 'int' = 1) -> 'Transition'
```

## `Transition`

```python
Transition(event_shape: 'tuple[int, ...]', method: 'str', backend: 'str', exact: 'bool', _sampler: 'Callable[[tuple[int, ...], np.random.Generator], ArrayLike]', _law: 'Distribution | None' = None, _mean: 'ArrayLike | None' = None, _covariance: 'ArrayLike | None' = None) -> 'None'
```

A conditional kernel for one fixed initial state and elapsed interval.

sampler(size, rng) returns size + event_shape. A Distribution is optional:
numerical kernels and mixed jump laws need not pretend to have a density.
mean/covariance are analytic moments when supplied, never Monte Carlo fits.

### `Transition.covariance`

```python
covariance(self)
```

### `Transition.distribution`

```python
distribution(self)
```

### `Transition.from_distribution`

```python
from_distribution(law, *, method: 'str' = 'exact', backend: 'str' = 'scipy', exact: 'bool' = True)
```

### `Transition.identity`

```python
identity(state: 'ArrayLike')
```

Zero-duration kernel; sampling consumes no random numbers.

### `Transition.mean`

```python
mean(self)
```

### `Transition.sample`

```python
sample(self, size: 'SampleSize' = (), *, rng: 'np.random.Generator')
```

## `ProcessPaths`

```python
ProcessPaths(times: 'ArrayLike', values: 'ArrayLike', method: 'str', backend: 'str', exact: 'bool', methods: 'tuple[str, ...]' = (), backends: 'tuple[str, ...]' = ()) -> 'None'
```

Joint draws (draw,time,*event_shape), including the supplied initial state.

### `ProcessPaths.marginal`

```python
marginal(self, index: 'int')
```

An explicitly empirical marginal, even when the transition is exact.

## `transition`

```python
transition(process: 'StochasticProcess', state: 'ArrayLike', dt: 'float', *, time: 'float' = 0.0, method: 'str' = 'exact', substeps: 'int' = 1) -> 'Transition'
```

Construct a conditional process transition from state over nonnegative dt.

time and dt share process units. exact requires an analytic supported transition;
Euler substeps are approximate. Sample-only kernels never fabricate a density.

## `diffusion_generator`

```python
diffusion_generator(process: 'ItoProcess', time: 'float', state: 'ArrayLike', gradient: 'ArrayLike | Callable[[FloatArray], ArrayLike]', hessian: 'ArrayLike | Callable[[FloatArray], ArrayLike]') -> 'float'
```

Spatial Ito generator b·grad(f) + 1/2 tr(B B.T Hess(f)).

Derivatives are supplied explicitly. This is the continuous part only for
a jump process; the model's generator adds its jump expectation separately.

## `sample_paths`

```python
sample_paths(process: 'StochasticProcess', initial: 'ArrayLike', times: 'ArrayLike', *, draws: 'int' = 1, rng: 'np.random.Generator', method: 'str' = 'exact', substeps: 'int' = 1) -> 'ProcessPaths'
```

Minimal sequential kernel composition; times use caller-selected units.
