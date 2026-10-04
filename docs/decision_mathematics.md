# Risk and decision mathematics: families 27–36

This is the finite acceptance catalog for the final ten mathematical families.
All ordinary inputs are arrays, numbers and callbacks. Public `interfaces/`
own contracts and results, `builtins/` own model configuration, and `_backends/`
own numerical execution. Arrays in result records are owned and read-only.
Losses are positive for adverse outcomes; rewards and wealth are maximized.
Weights describe a finite probability law, not independent sample counts.

| Family | Module | Acceptance catalog |
|---|---|---|
| 27 | `risk` | Weighted variance, downside moments, VaR and fractional-atom ES; Euler volatility/ES contributions; absolute/relative drawdown; grid breach probabilities; affine and callback stresses; scenario margin. |
| 28 | `utility` | Linear, log, exponential, CRRA and options-sheet smooth utility; weighted expectation and certainty equivalents; mean–variance/mean–CVaR; scaled turnover and switching preferences. |
| 29 | `optimization` | Array LP/QP, second-order cones, integer/binary domains via CVXPY; callback nonlinear problems via SciPy; bounded enumeration; feasibility, duals, solver status and available gaps. |
| 30 | `robust` | Box/ellipsoid support functions; ellipsoidal robust mean allocation; affine minimax; weighted scenario CVaR; Gaussian affine chance constraints; finite-support KL-distributional ambiguity. |
| 31 | `control` | Finite-horizon and discounted stationary MDPs; exact finite-horizon belief-state POMDP recursion; discrete LQR; nonanticipative linear-quadratic scenario-tree MPC; monotone finite-state generator HJB and discrete-time impulse control. |
| 32 | `policies` | Probability hysteresis, immutable lifecycle state, absorbing closure, fixed deadlines, convergence/invalidation exits and explicit mandatory overrides; no-action bands with fixed ties. |
| 33 | `execution` | Linear/quadratic costs; independent binomial fills and assignment; bounded binomial logistic likelihood fitting; constrained multi-asset liquidation with inventory penalties. |
| 34 | `wealth` | Self-financing trades; lending/borrowing accrual; collateral transfers/accrual; coupons/dividends; signed option exercise and pending delivery; futures variation margin; financed gains; exact and Taylor P&L attribution. |
| 35 | `diagnostics` | Ljung–Box, Jarque–Bera, ADF/KPSS, Engle–Granger; likelihood/pinball/weighted CRPS/energy/variogram scores; randomized PIT and interval coverage; circular block bootstrap; HAC mean covariance/model comparison; Holm/BH corrections. |
| 36 | `reinforcement` | Tabular Q learning and exact MDP references; linear fitted Q iteration; masked Bellman targets and PyTorch actor–critic losses; optional Stable-Baselines3 A2C/PPO on supplied finite MDPs. |

The options-sheet utility is `u(w) = w - lambda*sqrt(a²+w²)`, with `a>0`
and `0<lambda<1` (complete SPY options sheet, O5). Turnover and switching
are preferences; actual costs enter wealth exactly once. Family 36 remains
optional and never imports PyTorch or Stable-Baselines3 at package import.

The catalog is bounded: conic support means second-order cones; integer cone
problems require a compatible installed solver. Nonlinear solvers report local
solutions. Scenario chance/CVaR calculations describe the supplied law.
Belief recursion is exact only for the supplied finite POMDP and horizon.
HJB support is a supplied monotone finite-state generator discretization, not
an automatic continuous-state PDE discretizer. RL does not establish forecast
quality or enforce constraints omitted from the declared action set.

## Installation and public workflows

CVXPY and python-control are principal dependencies. `pip install -e '.[dev]'`
installs the decision core. Install `.[reinforcement]` for PyTorch losses and
Stable-Baselines3 A2C/PPO, and `.[notebooks]` for the executed investigations.
Importing `reinforcement` does not load either optional neural provider.

```python
from convexity.math import risk, utility, optimization

losses = [-3., 1., 7., 9.]
masses = [.3, .4, .2, .1]
print(risk.value_at_risk(losses, .5, weights=masses))       # 1
print(risk.expected_shortfall(losses, .5, weights=masses)) # 5

problem = optimization.QuadraticProgram(
    linear=[-3., -2.], inequality=[[1., 1.], [1., 0.]],
    upper=[4., 2.], lower_bounds=0.,
)
result = optimization.solve(problem)
print(result.solution, result.objective, result.duals)    # [2, 2], -10
print(utility.certainty_equivalent(
    [-5., 2., 10.], utility.SmoothUtility(aversion=.6, scale=3.)
))
```

The [combined decision workflow](../examples/decision_workflows.py) starts with
an explicit P/Q GBM pair from family 16. Q determines today's option premium;
P supplies terminal spot outcomes. Trades and funding generate wealth under
each P scenario. ES restricts admissible contract quantities, finite enumeration
maximizes expected utility, and a switching preference determines whether to
move away from holding. Physical scenario weights are never relabeled Q.

## 27. Risk (`risk`)

`LossScenarios(losses, weights)` owns one scalar loss per scenario. Positive
loss means adverse; a negative VaR or ES is meaningful and is not clipped.
Masses are normalized robustly, may contain zeros, and need not sum to one on
input. Zero-mass outcomes cannot determine an endpoint quantile.

`variance` is a population functional of the supplied law, without a degrees
of freedom correction. `downside_moment` uses returns and computes the
**unconditional** lower partial moment around a target. `value_at_risk` uses
`inf{x:F(x)>=level}`; the zero-level convention is the smallest supported loss.
`expected_shortfall` integrates the worst `1-level` probability mass. It accepts
`0<=level<1`, equals the mean at zero, and splits an atom at VaR rather than
averaging every observation tied at the threshold.

Volatility and ES contributions use holdings `(instrument,)`, covariance
`(instrument,instrument)` or unit losses `(scenario,instrument)`. Components
sum to total risk by Euler's identity. At an ES tie the marginal vector is a
valid proportional-tail subgradient, not a unique derivative. Zero-volatility
portfolios have no unique volatility gradient and raise.

`drawdown` accepts `(time,)` or `(scenario,time)` wealth, including initial
wealth. Relative drawdown divides by the running peak, which must be positive;
absolute drawdown remains in wealth units. `breach_probability` accepts scalar
`PathScenarios` directly, preserves their weights, and includes equality at the
barrier. It says nothing about crossing between grid points. Stress maps do not
infer probabilities. `scenario_margin` is a transparent worst-stress-loss
function with supplied floor, multiplier and add-on, not a broker margin rule.

## 28. Utility (`utility`)

Utilities operate elementwise on wealth. Linear, exponential and `SmoothUtility`
accept real wealth; logarithmic and CRRA utilities require positive wealth.
`PowerUtility(gamma)` is `(w**(1-gamma)-1)/(1-gamma)` with its logarithmic limit.
`ExponentialUtility(a)` is `-exp(-a*w)`; its certainty equivalent uses log-sum-exp
to remain finite when direct exponential utility evaluation would overflow.
Zero-mass scenarios are excluded before evaluating a utility's domain.

`expected_utility` accepts a compatible elementwise callable;
`certainty_equivalent` additionally needs `inverse`, as described by the
`Utility` protocol. Mean–variance uses `mean - aversion*variance/2`;
mean–CVaR uses `mean(gain) - aversion*ES(-gain)`. `preference_penalty` computes
scaled L1 turnover plus a fixed cost for any nonzero trade. It is not a cash
cost and must not be debited again from wealth.

## 29. Optimization (`optimization`)

`QuadraticProgram` minimizes `.5*x.T@Q@x + c@x`. Inequalities are `A@x<=b`,
equalities are `E@x=f`; each matrix has `(constraint,variable)` axes. Bounds
broadcast over the variable vector and default to **free** variables. Domains
are `continuous`, `integer` or `binary`. A `SecondOrderCone` represents
`||M@x+offset|| <= linear@x+bound`. Q must be symmetric positive semidefinite;
nonconvex quadratic problems belong in `NonlinearProblem` instead.

`ConvexSolver` uses CVXPY, with Clarabel as the continuous default. The bundled
SciPy/HiGHS route handles mixed-integer **linear** problems; pass
`ConvexSolver('SCIPY')` to select it. Integer quadratic or conic formulations
need a compatible installed solver. Unsupported solver/problem combinations
raise instead of relaxing discrete variables. Scalar domain attributes avoid
version-dependent CVXPY conventions for indexing multiple integer coordinates.

`NonlinearProblem` accepts scalar objective and optional gradient callbacks;
inequalities return values `<=0`, equalities return zero. SLSQP reports a local
solution and verifies returned constraint residuals. No global gap is invented.
`enumerate_minimum` checks the Cartesian search size against a bound before
iteration, is exact on those finite domains, and resolves ties by input order.

`SolveResult` contains the candidate, objective, status, maximum primal
violation, available named duals, solver, iterations, and any reported MIP gap.
Absent gaps and duals remain absent. Bounds' dual arrays cover their finite
coordinates in original order. Integrality and binary bounds contribute to the
reported violation. The feasibility tolerance is not a bound on distance to the
true optimizer. `optimal_inaccurate` is not silently accepted. `solve` raises
`OptimizationError` on unsuccessful results unless `check_success=False`;
the exception retains the result. A custom object implementing `solve(problem)`
can return the same result type without registration or inheritance.

## 30. Robust and stochastic optimization (`robust`)

`Box` and `Ellipsoid` provide analytic support functions. The ellipsoid is
`center + shape@u`, `||u||<=radius`, allowing rectangular or singular shapes.
`robust_allocation` minimizes worst-case negative mean plus a variance penalty,
with supplied bounds and `sum(x)=budget`. The ambiguity center must equal the
stated mean. `minimax` minimizes the maximum supplied affine scenario loss.
`mean_cvar` implements the weighted Rockafellar–Uryasev hinge formulation with
explicit tail confidence and probability masses. Its result objective is a
minimized loss criterion, not expected portfolio return.

`gaussian_chance_constraint` constructs an exact SOC for a scalar Gaussian
linear exposure with required probability in `[.5,1)`. Its covariance is that
of the random coefficient vector, not an estimated parameter covariance unless
that is the caller's intended law. `chance_probability` instead measures the
fraction of supplied outcomes satisfying a bound; it is not an out-of-sample
coverage certificate. `kl_worst_case` maximizes expected loss over finite masses
satisfying `KL(q||p)<=radius`; it reports the maximizing q and positive worst
expected loss. Zero radius returns p exactly. Unsupported outcomes stay excluded.

## 31. Control (`control`)

`FiniteMDP` takes transition probabilities `(state,action,next_state)` and
expected one-step rewards `(state,action)`. A boolean feasible mask determines
legal actions. Terminal states have zero continuation and no further reward;
put their payoff on the transition into them. Every state requires an available
action, including a dummy action for terminal states. `finite_horizon` accepts a
terminal value vector for nonterminal states; its values have `(horizon+1,state)`
axes and policy `(horizon,state)`. Stationary value iteration requires discount
below one, reports a Bellman residual, and raises on exhaustion. Its residual
implies a value sup-norm bound of `residual/(1-discount)`.

`belief_update` performs predict-then-condition Bayes updating. `belief_plan`
branches exactly over actions and observations, updating beliefs before future
decisions. Observation probabilities are `(action,next_state,observation)`.
All actions must be common to hidden states; terminal behavior is represented
by explicit absorbing zero-reward states. A node budget limits exponential
branching; exceeding it raises rather than pruning without disclosure.

`lqr` uses python-control's discrete Riccati solver and returns gain K,
Riccati matrix, `A-BK` and equation residual. The convention is `u=-Kx` and
stage cost `x'Qx+u'Ru`, with Q semidefinite and R positive definite. A returned
solution must stabilize the supplied dynamics.

`ScenarioTree` stores parents, conditional branch probabilities and realized
arrival disturbances `(node,state)`. Root probability is one and its disturbance
zero. Parent indices precede children and sibling probabilities sum to one.
`scenario_mpc` minimizes probability-weighted quadratic state/action costs plus
leaf terminal costs. States and actions are node-indexed; leaf actions are zero.
It enforces dynamics and supplied state/action bounds in every branch, including
zero-mass stresses. Execute only `actions[0]` before replanning. A caller must
construct nodes from **observable histories**, not hidden states: sharing a
root action prevents one kind of foresight, but cannot repair a tree that leaks
latent information at later nodes.

`generator_hjb` solves `rho*V=max_a(r_a+G_a@V)` by uniformization of a supplied
conservative Markov generator. This verifies finite-state control equations;
it does not certify a continuum mesh. `impulse_control` augments an MDP with
paid jumps arriving next period. There is at most one action per time step;
it is not an instantaneous continuous-time intervention obstacle solver.

## 32. Policies (`policies`)

`PolicyState` carries the fixed deadline and last observed time. Hysteresis
activates at `p>=entry`, retains the active mode between thresholds, and closes
at `p<=exit`. Closure priority is existing closed state, mandatory override,
deadline, adverse evidence, convergence, then active-state invalidation. Deadline,
adverse evidence and convergence can close a waiting record too. `scheduled=False`
suppresses new activation; it does not suppress exit evidence. Calls cannot move
time backward. Closed states are absorbing until the caller creates a new record.

`no_action` compares values after a supplied switching preference, retains holding
on a tie, and uses the first maximizing alternative otherwise. It handles an
infeasible hold explicitly and rejects an empty feasible set. A supplied mandatory
index must still be feasible. A lifecycle's active mode does not require a
nonzero position; exposure selection is a separate mathematical decision.

## 33. Execution (`execution`)

`QuadraticCost` computes `spread@abs(trade) + trade'impact trade` in cash units.
`BinomialFills` and `assignment` return `(draw,leg)` integer counts from supplied
nonnegative quantities and probabilities with an explicit NumPy generator.
They assume conditional independence across legs and impose no time dependence.
Alternative fill laws can implement the `FillLaw` protocol.

`fit_assignment` fits a binomial logistic likelihood with finite coefficient
bounds using SciPy. Include the intercept column explicitly. Trial counts,
features and assignments share the observation axis. Covariance is local inverse
information only when full rank and away from bounds; it is absent at a boundary.
The returned log likelihood includes binomial combinatorial constants.

`liquidate` takes signed initial inventory, positive time intervals, spread,
impact, price-increment covariance per time, inventory aversion and optional
maximum rates. Trades reduce inventory, their sum equals initial holdings, and
terminal inventory is zero. By default each leg trades only toward zero.
Temporary impact is `trade'impact trade/dt`; the risk preference is
`aversion*dt*end_inventory'covariance end_inventory`. Interval choice therefore
matters. Cash cost and risk preference are reported separately. This is a
quadratic scheduling model, not a fill guarantee or permanent-impact model.

## 34. Wealth (`wealth`)

`Account` owns settled holdings/cash, posted collateral and pending delivery
holdings/cash in one valuation currency. Wealth sums their marked economic
values; collateral reduces available cash without destroying wealth.
`trade` exchanges cash for marked assets and debits an actual cash cost once.
`accrue` applies separate positive lending, borrowing and collateral growth
factors; distributions are per settled unit and can represent dividends/coupons.
Pending claims do not accrue automatically. `cash_flow` accepts independently
calculated entitlements, including results composed from family 17.

`exercise_delivery` removes signed integral exercised/assigned contracts within
the existing position and records supplied per-contract shares and cash as pending.
A contract's public `delivery(spot)` can supply these terms. At intrinsic value,
exercise conserves marked wealth; before expiry it may surrender time value.
`settle` transfers pending claims and clears them, making repetition idempotent.
The caller controls event ordering, entitlements and delivery dates.

Futures quantities are **outside** the account's marked asset vector because
variation margin is credited directly as cash by `futures_variation`.
`financed_gain` is `W1 - funding_factor*W0 - terminal_external_flows`, where the
last term values external contributions at the terminal date. It is distinct
from raw wealth change. `pnl_attribution` reports exact repricing, first-order,
second-order and remainder terms; it does not label Taylor P&L as exact.

## 35. Diagnostics (`diagnostics`)

Statistical tests return statistics, p-values, method and null hypothesis.
Ljung–Box tests residual autocorrelation, Jarque–Bera tests Gaussianity, ADF's
null is a unit root, KPSS's null is stationarity, and Engle–Granger's null is no
cointegration. Trend choices and lag counts are explicit. Provider warnings
about tabulated p-value limits remain visible. Rejection is not a trading rule.

Scores use smaller-is-better orientation. Log score consumes supplied log
densities; CRPS is exact for a weighted scalar finite law; energy score handles
joint vectors or whole `PathScenarios`; variogram score sums unordered coordinate
pairs. The latter two require an explicit event scale when units differ. CRPS
uses sorting, while energy is quadratic in scenario count. These are scores of
the empirical law, not unbiased estimates of an underlying continuous-law score.
Randomized PIT accepts both F(y-) and F(y) plus an explicit generator for atoms.
Coverage includes interval endpoints and does not imply simultaneous path coverage.

`block_bootstrap` resamples circular contiguous blocks of paired observation
rows and returns percentile intervals. A block length of one gives IID
resampling; choose a longer length to reflect dependence and overlapping
horizons. `hac_mean_covariance` estimates covariance of the **sample mean** with
Bartlett weights and no degrees-of-freedom correction. `compare_scores` applies
an asymptotic two-sided normal test to paired score differences; it is not an
exact finite-sample Diebold–Mariano correction. Holm controls familywise error
under arbitrary dependence; BH's FDR guarantee needs its usual independence or
positive-dependence conditions. Caller-supplied tests and resampling assumptions
remain essential.

## 36. Optional reinforcement learning (`reinforcement`)

Finite MDPs from family 31 are the shared reference. `q_learning` samples the
supplied dynamics with explicit RNG and epsilon-greedy feasible actions; its
Bellman residual is a diagnostic, not a promise of convergence. Exploration
starts uniformly among nonterminal states and resets on termination. Unreachable
state/action pairs may remain unlearned. `fitted_q` performs model-based linear
least-squares Bellman updates. One-hot features recover the tabular reference;
projection and ridge can leave a nonzero residual, and projected iteration may
diverge. Exhausting the iteration budget raises.

`bellman_targets` masks unavailable future actions and never bootstraps terminal
rows, even if those rows have no feasible actions. `bellman_loss` is weighted
squared error. The tensor `actor_critic_loss` combines a masked categorical
policy-gradient term, half-squared value error and entropy preference. Return
targets and the policy advantage are detached; gradients flow to logits and
value estimates. The mask excludes illegal actions from probability mass.

`train_agent` runs actual Stable-Baselines3 A2C/PPO through a Gymnasium finite-MDP
environment, preserving episode termination versus time-limit truncation. The
adapter requires every action to be admissible because vanilla SB3 policies do
not implement dynamic action masks. Use the masked tabular/fitted/tensor tools
for state-dependent feasibility. The returned agent exposes deterministic
`action(state)`; its trained provider remains a mutable backend object. Neural
training is a numerical implementation, not proof of a useful financial policy.

## Verification and references

One script and one executed notebook per family are linked from the main
[README](../README.md) and [notebook guide](../notebooks/README.md). The test suite
checks analytic identities, optimal solutions, information constraints, generated
parameter recovery, input failures, backend boundaries and optional real training.
There are 240 exhaustive lifecycle combinations, in addition to numerical tests.

```bash
python -m pytest -q tests/math/decisions
python examples/decision_workflows.py
python examples/reinforcement.py --neural
python scripts/build_decision_notebooks.py --execute
```

Backend conventions were checked against the primary documentation:
[CVXPY solver capabilities](https://www.cvxpy.org/tutorial/solvers/index.html),
[python-control discrete LQR](https://python-control.readthedocs.io/en/latest/generated/control.dlqr.html),
and [Stable-Baselines3 A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html).
The utility and ledger equations follow the local
[complete SPY options sheet](mathematical_trading_systems/complete_spy_options_system_sheet.pdf),
sections O5–O8; execution and finite control also follow the
[complete futures sheet](mathematical_trading_systems/complete_futures_system_sheet.pdf).

### Completion record — 2026-09-30

Verified in the repository's `quant` interpreter (Python 3.14):

| Check | Result |
|---|---|
| Full repository suite after the solver regression fix | **1,870 passed**, no skips; 70 existing provider warnings |
| Families 27–36 within that suite | **304 passed**, including 240 exhaustive lifecycle combinations |
| Focused Pyright: ten packages, shared validation, tests, eleven scripts and notebook builder | **0 errors, 0 warnings** |
| Ruff E4/E7/E9/F on the same new code | Passed |
| Standalone scripts | **11 of 11 executed** |
| New notebooks | **10 of 10 executed**, retained figures, no error outputs |
| Editable installation and dependency consistency | Passed; `pip check` clean |
| Documentation links and whitespace | Local links resolve; `git diff --check` clean |

The backend runs used CVXPY 1.9.2, Clarabel 0.11.1, SciPy 1.18.1,
python-control 0.10.2, statsmodels 0.15.0, PyTorch 2.14.0,
Gymnasium 1.3.0 and Stable-Baselines3 2.9.0. The full dependency range is not
claimed to have been exhaustively tested on every Python/solver combination.

Verification found and corrected two integration defects beyond the initial
mathematical checks: CVXPY's indexed integer-variable convention failed for
multiple discrete coordinates (now covered by four regression cases), and a
standalone `control.py` example shadowed the python-control provider (renamed
to `optimal_control.py` and rerun). The initial default QP solver also used a
looser accuracy setting than the analytic reference check; the continuous
default is now Clarabel. Existing user work in earlier families was preserved.
