# Numerical mathematics — completed family 1 catalog

`convexity.math.numerical` implements the ten numerical-mathematics sections
below. It operates on supplied arrays, functions, grids, and differential
equations. The module does not import other Convexity mathematical families.

It excludes PCA, covariance estimation, regression models, distributions,
Monte Carlo, stochastic processes/SDEs, statistical estimation, optimization,
time-series forecasting, financial pricing, and application-specific PDEs.
Least squares here is an algebraic system solve; polynomial approximation fits
a supplied numerical function/sample set, with no statistical inference.

## Architecture and installation

For a worked walkthrough, open the [numerical mathematics problem book](../notebooks/numerical_mathematics.ipynb).
Its 13 worked sections progress from linear systems and quadrature to stiff ODEs,
boundary-value problems, and mesh refinement, with plots, reference checks, and a
custom-tool example. Install `python -m pip install -e '.[notebooks,autodiff]'`,
select that environment as the notebook kernel, and run all cells. PyTorch is
optional; without the `autodiff` extra, its demonstration is explicitly skipped.

1. `_backends/`: private NumPy/SciPy/PyTorch/findiff adapters. Libraries supply
   the algorithms. Do not import these modules in application code.
2. `interfaces/`: validated problem/result schemas, `NumericalTool` callable
   protocol, `run(problem, tool)`, and explicit `Backend` capability routing.
3. `builtins/`: concrete tools and convenience functions. Each numerical
   operation accepts a compatible `tool=` override. Public namespaces organize
   these functions; there is no mutable global backend selection.

```bash
python -m pip install -e '.[dev]'
python examples/numerical_mathematics.py

# Optional PyTorch automatic differentiation
python -m pip install -e '.[autodiff]'
python examples/numerical_mathematics.py --autodiff
```

NumPy >= 1.26 and SciPy >= 1.15 supply the main algorithms. `findiff >= 0.12.1`
supplies differentiation matrices, including nonuniform grids and boundary
stencils. PyTorch >= 2.2 is optional and imported only by automatic
 differentiation calls. JAX can be integrated through a custom tool; no built-in
JAX adapter is claimed.

```python
from convexity.math import numerical as cx

x = cx.linalg.solve([[3., 1.], [1., 2.]], [9., 8.]).value
q = cx.linalg.qr([[1., 2.], [3., 4.]])
g = cx.calculus.gradient(lambda x: x @ x, [1., 2.]).value
r = cx.roots.solve(lambda x: x*x-2., [1.]).value
integral = cx.integrate.quad(lambda x: x*x, 0., 1.).value
curve = cx.interpolation.spline([0, 1, 2, 3], [0, 1, 4, 9])
trajectory = cx.ode.solve(lambda t, y: -y, [1.], [0., .5, 1.])
```

Existing flat calls (`cx.solve`, `cx.gradient`, `cx.interpolate`, etc.) remain
available. Namespaces are `linalg`, `roots`, `calculus`, `integrate`,
`interpolation`, `ode`, `grids`, `operators`, and `accuracy`.

## Finite acceptance catalog

| Section | Implemented operations | Built-in choices |
|---|---|---|
| 1. Dense linear algebra | `solve`, `lstsq`, `eig`, `eigh`, `generalized_eig`, `svd`, `qr`, `lu`, `cholesky`, `schur`, `matrix_rank`, `condition_number`, `norm`, `pinv`, `det`, `slogdet`, `expm`, `logm`, `sqrtm`, `solve_lyapunov`, `solve_riccati` | General square solve; SVD minimum-norm least squares; general/Hermitian eigenpairs; reduced/full SVD; reduced QR with optional column pivoting; pivoted LU; lower Cholesky; real/complex Schur; continuous/discrete matrix equations |
| 2. Sparse and matrix-free algebra | `sparse_matrix`, `linear_operator`, `solve`, `sparse_eig`, `sparse_svd` | Native CSR/CSC containers; direct solve; CG/GMRES/BiCGSTAB with optional preconditioner; ARPACK eigen/SVD; native `LinearOperator` interoperability |
| 3. Roots and fixed points | `root_scalar`, `root` / `roots.solve`, `fixed_point` | Brent, bisection, TOMS748, Newton, secant; Powell hybrid, Broyden 1/2, Anderson, Krylov; fixed-point iteration or Steffensen acceleration |
| 4. Differentiation | `derivative`, `gradient`, `jacobian`, `hessian`, `jvp`, `vjp`, `check_derivative` | SciPy adaptive finite differences; PyTorch automatic derivatives and products; componentwise derivative checks against finite differences |
| 5. Integration | `quad`, `nquad`, `gauss_quad`, `integrate_density`, `quadrature_rule` | Adaptive 1-D and rectangular multidimensional quadrature, including improper bounds; fixed-order Gaussian quadrature; Legendre/Hermite/Laguerre nodes and weights |
| 6. Interpolation and approximation | `interpolate`, `spline`, `regular_grid`, `scattered`, `polynomial_approximation` | 1-D linear/cubic/PCHIP; degree 1/3/5 splines; cubic natural/clamped boundaries; regular-grid linear/nearest/cubic; scattered linear/nearest and 2-D cubic; scaled polynomial/Chebyshev approximation |
| 7. Initial-value ODEs | `solve_ivp`, `ode.solve` | Real ODE systems; RK23/RK45/DOP853/Radau/BDF/LSODA; adaptive steps, optional Jacobian, directional/terminal events, dense solutions, forward/backward integration |
| 8. Boundary-value and PDE primitives | `solve_bvp`, `differentiation_matrix`, `partial_derivative_matrix`, `laplacian_matrix` | Real first-order BVP systems with endpoint boundary residuals; sparse uniform/nonuniform derivative and mixed-partial matrices; uniform isotropic Laplacian with homogeneous Dirichlet/Neumann/periodic boundaries |
| 9. Numerical construction | `linspace`, `logspace`, `geomspace`, `arange`, `meshgrid`, `cartesian_product`, `quadrature_rule` | NumPy grid/mesh construction; `ij` indexing and C flattening; explicit Gaussian weight functions |
| 10. Accuracy and stability | `Tolerances`, `isclose`, `allclose`, `machine_precision`, `require_finite`, `residual`, numerical diagnostics | Explicit absolute/relative tolerances; machine limits; shape-safe comparison; NaN/Inf policies; conditioning, residuals, convergence reports, and library error estimates where available |

The catalog supplies these concrete methods rather than every method in each
backend library. There is no PDE evolution engine, finite-element solver,
application-specific boundary condition, or Black–Scholes/HJB/Fokker–Planck
formulation. BVP support is for real systems without unknown model parameters
or singular endpoint terms. Multidimensional integration uses rectangular
bounds; multidimensional interpolation uses real sample values. There is no
implicit batch dimension or autograd graph spanning Convexity calls.

## Shared conventions

Dense numerical input is copied to float64 or complex128, validated, and stored
read-only. Numeric strings, booleans as data, empty inputs, ragged arrays, and
nonfinite inputs are rejected unless a specific contract explicitly permits
them. Sparse inputs are copied without dense conversion. Operators hold supplied
callbacks and are not materialized. Callbacks must be deterministic and must
not mutate their arguments or a constructed problem during execution.

There are no implicit financial units. Callers supply consistent units.
Derivative units are output/input; integral units are integrand times input.
`ArrayResult.value` is a NumPy array, including a zero-dimensional array for
scalar results. Use `float(result.value)` for a Python scalar. Factorizations,
eigenpairs, interpolants, trajectories, and scalar measures with legitimate
infinities use their documented result schemas.

### Linear algebra shapes and identities

- `solve`: `A: (n,n)`, `b: (n,)` or `(n,k)`; solution preserves the RHS shape.
- `lstsq`: `A: (m,n)`, `b: (m,)` or `(m,k)`; solution has shape `(n,)` or `(n,k)`.
  `LeastSquaresResult` includes rank, singular values, and residual norm. Rank
  deficiency is supported and returns a minimum-norm solution.
- `eig`: right eigenvectors are columns, `A @ vectors == vectors * values`.
  Hermitian eigenvalues are ascending and real. General eigenpairs are unsorted.
- `generalized_eig`: returns `alpha`, `beta`, and `vectors`, satisfying
  `beta * (A @ vectors) == alpha * (B @ vectors)`. `.values` forms `alpha/beta`;
  zero `beta` denotes an infinite eigenvalue, while `alpha=beta=0` is
  indeterminate (`NaN`). Hermitian generalized problems require positive-definite
  `B`; SciPy checks that requirement.
- `svd`: `A == (u[:, :k] * s) @ vh[:k, :]`, `k=min(m,n)`; singular values descend.
- Factorization results expose a read-only `factors` mapping. QR has `q`, `r`
  and a separate integer `permutation`: `A[:, permutation] == q @ r`. LU has
  `p`, `l`, `u`: `A == p @ l @ u`. Cholesky has `l`: `A == l @ l.conj().T`.
  Schur has `t`, `z`: `A == z @ t @ z.conj().T`.
- `norm` accepts vectors or matrices. Rank tolerance is an absolute singular-value
  cutoff; pseudoinverse tolerance is a relative cutoff. Defaults use the library
  conventions. `condition_number` uses NumPy's norm-order conventions and may
  return infinity for a singular matrix.
- `slogdet` returns `sign` (complex phase for complex matrices) and `logabsdet`.
  A singular determinant is represented by zero sign and negative infinity.
  `logm` rejects singular matrices. `sqrtm` and `logm` may return complex results
  for real inputs; no imaginary component is silently discarded.

Sparse iterative solves require one RHS vector. A `LinearOperator` selects GMRES
by default; an explicit sparse matrix selects the direct solver. CG requires a
Hermitian positive-definite operator. Explicit sparse symmetry is checked, but
positive definiteness and matrix-free symmetry remain caller assumptions.
`preconditioner` represents the action of an approximate inverse. GMRES
`maxiter` counts inner iterations. For matrix-free SVD, supply `rmatvec` as the
**conjugate transpose** action. Real operators can act on complex vectors.

ARPACK requires `k<n` for real Hermitian eigenproblems, `k<n-1` for general or
complex eigenproblems, and `k<min(m,n)` for sparse SVD. Selection uses `which`;
sparse SVD singular values are returned descending. Library nonconvergence
exceptions, including ARPACK partial eigenpair diagnostics, propagate.

The matrix equation conventions, with `H` denoting conjugate transpose, are:

```text
Continuous Lyapunov: A X + X A^H + Q = 0
Discrete Lyapunov:   A X A^H - X + Q = 0
Continuous Riccati:  A^H X + X A - X B R^-1 B^H X + Q = 0
Discrete Riccati:    A^H X A - X
                    - A^H X B (R + B^H X B)^-1 B^H X A + Q = 0
```

`time="continuous"` is the default. `Q` and `R` must be Hermitian; shapes are
`A,Q: (n,n)`, `B: (n,m)`, `R: (m,m)`. Inverse notation is implemented using
library solves. Existence/stabilizability is a solver concern; no control policy
is introduced. The private Lyapunov adapter handles SciPy's different continuous
sign convention.

### Roots, derivatives, and accuracy

Scalar bracketing methods require finite increasing endpoints, a sign change,
and continuity. Endpoint roots are accepted. Newton requires a derivative and
`x0`; secant requires distinct `x0,x1`. Unbracketed methods can find any root in
their basin of attraction. Scalar convergence follows backend step/bracket
criteria; inspect the residual to assess function error.

System roots require `f: R^n -> R^n`. An explicit Jacobian `(n,n)` applies to the
hybrid solver; quasi-Newton methods build their own approximations. Hybrid
`max_evaluations` and quasi-Newton `maxiter` are distinct limits. Both backend
success and the requested residual norm are required. Fixed-point iteration
accepts a scalar or vector and checks `||f(x)-x||`; SciPy raises if iteration
fails. No universal contraction/convergence claim is made.

Differentiation inputs are scalar for `derivative`, `(n,)` otherwise. A gradient
or Hessian callback returns a scalar; a Jacobian callback returns `(m,)` and its
result is `(m,n)`. Hessians are `(n,n)`. JVP consumes an input-space vector `(n,)`
and returns `(m,)`; VJP consumes an output-space vector `(m,)` and returns `(n,)`.

`FiniteDifference` calls ordinary one-point Python/NumPy callbacks. SciPy owns
sampling, weights, refinement, and error estimates. `initial_step` uses the
input units; callbacks must be defined around the evaluation point. One-sided
scalar derivatives use `direction=-1` or `1`. Finite-difference products
materialize a Jacobian; `AutoDiffProduct` delegates products directly to PyTorch.

`AutoDiff` and `AutoDiffProduct` pass float64 PyTorch tensors to the supplied
function, with optional `device="cuda"`. Functions must use differentiable
PyTorch operations and return real floating-point tensors of the expected
shape. Results are detached NumPy arrays. Constant/disconnected outputs have
zero derivatives (`strict=False`), so accidentally detaching an output also
removes its derivatives. No finite-difference error estimate is fabricated.

`check_derivative` compares a supplied derivative array or `ArrayResult` against
finite differences of a NumPy callback. The reference must converge; a failed
reference raises. A wrong supplied derivative returns `passed=False` and
componentwise errors. For PyTorch checks, supply the NumPy-equivalent function
for the reference.

`Tolerances` uses `|actual-reference| <= atol + rtol*|reference|` and permits
exact comparison with both zero. Comparison does not broadcast mismatched
shapes. Same-sign infinities compare equal; NaNs compare unequal unless
`equal_nan=True`. `require_finite` rejects NaN/Inf and never silently replaces
them. Relative residual is `||Ax-b|| / ||b||`; with a zero RHS it is zero for an
exact solution and infinity otherwise. It is not a forward-error bound.

### Integration and construction

`quad` integrates a scalar callback on finite/infinite bounds, including reversed
bounds. `nquad` consumes `f(x: (d,))` and increasing bounds `(d,2)`, which may be
infinite. Its reported error is SciPy's maximum nested absolute error estimate,
not a rigorously accumulated multidimensional bound. Cost grows with dimension.
Gaussian quadrature uses a fixed number of nodes and reports no adaptive error
estimate. `integrate_density` computes `integral(f(x)*density(x))`; sampled
density values must be nonnegative. It does not construct or normalize a
probability distribution. For multidimensional weights, supply the product
integrand to `nquad`.

`quadrature_rule` supports Legendre weight `1` on a supplied finite interval,
Hermite weight `exp(-x**2)` on the real line, and Laguerre weight `exp(-x)` on
`[0,infinity)`. Do not multiply those weight functions into the integrand again.

Grids and meshes return NumPy arrays directly. `logspace` endpoints are
exponents; `geomspace` endpoints are values. `arange` follows NumPy's exclusive
stop/floating-step conventions. Mesh indexing is `ij`; Cartesian products and
operator sample vectors use C order. Grid construction does not impose a
resource limit on the requested number of points.

Differentiation matrices delegate stencil construction to findiff. Specify an
order per axis for mixed derivatives, with positive even accuracy. Nonperiodic
boundaries use one-sided stencils, **not** imposed PDE boundary conditions.
Periodic axes must be uniform and omit the duplicate endpoint; their period is
`number_of_points * spacing`. The Laplacian primitive uses SciPy's homogeneous
boundary conventions and one shared positive spacing on all axes. Its sign is
that of the continuous Laplacian (negative semidefinite), not its negation.

### Interpolation and differential equations

One-dimensional knots increase strictly, with sample values `(N,...)` on axis
zero. Query shape `S` returns `S + sample_channel_shape`. The default rejects
extrapolation; `extrapolate=True` explicitly enables the selected method's
continuation. Cubic interpolation uses not-a-knot boundaries. `Spline` supports
not-a-knot degrees 1/3/5 and cubic natural or zero-slope clamped boundaries.
Polynomial results expose monomial coefficients in increasing powers of `x`,
while evaluation retains the scaled polynomial/Chebyshev fit.

Regular-grid values have shape `(N1,...,Nd,...)`; scattered points have shape
`(N,d)` with `d>=2` and values `(N,...)`. Queries `(...,d)` produce
`(...,sample_channel_shape)`. `extrapolation` is `raise`, `nan`, or `extrapolate`.
Scattered domains are convex hulls, not bounding boxes; only nearest scattered
interpolation supports extrapolation. Cubic scattered interpolation is 2-D.
Degenerate point sets propagate SciPy/Qhull errors.

IVP callbacks are `f(t,y: (n,)) -> (n,)`. `t_span` can run in either direction;
`t_eval` must follow that direction within the span. Result `y` has shape
`(n,len(t))`. Dense scalar queries return `(n,)`; vector queries return `(n,k)`.
Dense output is restricted to the interval actually solved, including early
termination. Terminal events are successful completion, separately marked by
`terminated_by_event`. An event before the first requested output can leave
`t` empty and `y` of shape `(n,0)`; event states remain available. Event arrays
have shape `(events,n)`. Events use sign changes over steps, so multiple crossings
inside one step can be missed; use `max_step` appropriate to the application.

Stiff IVP methods accept an optional Jacobian callback. BVP callbacks consume
`x: (mesh_size,)`, `y: (n,mesh_size)` and preserve the state shape; endpoint
boundary residuals have shape `(n,)`. BVPs adapt the supplied initial mesh/guess
and return a checked continuous solution and per-interval RMS residuals.

## Custom tools and explicit backend abstraction

```python
import numpy as np
from convexity.math import numerical as cx

class NumpySolve:
    def __call__(self, problem: cx.LinearSystem) -> cx.ArrayResult:
        if not isinstance(problem.a, np.ndarray):
            raise ValueError("this custom solver accepts dense matrices only")
        b = np.asarray(problem.b)
        x = np.linalg.solve(problem.a, b)
        return cx.ArrayResult(x, cx.Diagnostics(
            residual_norm=float(np.linalg.norm(problem.a @ x-b))))

problem = cx.LinearSystem([[2., 0.], [0., 4.]], [6., 8.])
answer = cx.run(problem, NumpySolve())
same_answer = cx.linalg.solve(problem.a, problem.b, tool=NumpySolve())

backend = cx.scipy_backend().with_tools(solve=NumpySolve())
answer = backend.run("solve", problem)
same_answer = cx.linalg.solve(problem.a, problem.b, tool=backend.tool("solve"))

# PyTorch advertises only its differentiation capabilities; there is no silent fallback.
torch_tools = cx.torch_backend()
gradient = cx.calculus.gradient(lambda x: (x*x).sum(), [1., 2.],
                                tool=torch_tools.tool("gradient"))
```

`Backend.tools` is immutable. `with_tools` creates another bundle, leaving the
original unchanged. Unsupported capabilities raise `NotImplementedError`.
Creating a bundle does not import PyTorch. User tools need only the relevant
problem/result contract; no subclassing or whole-backend implementation is
required. `run` validates output schema/shape and convergence. User tools remain
responsible for mathematical correctness and truthful diagnostics; Convexity
does not independently re-solve their problems.

## Diagnostics, verification, and references

`Diagnostics` contains convergence, message, and optional iterations,
evaluations, residual norm, and error estimate. Unknown quantities are `None`.
Residuals use Euclidean/Frobenius norms. A small residual does not guarantee
forward accuracy for an ill-conditioned problem. Backend warnings remain visible;
dense solve ill-conditioning warnings are promoted to errors.

Reported nonconvergence raises `ConvergenceError`, retaining `.result`. Calls
with convergence-reporting results accept `check_convergence=False` to inspect
partial results. Invalid contracts raise `ValueError`/`TypeError`; callback and
library exceptions propagate. Direct factorizations, fixed-point failures, and
ARPACK failures may raise library exceptions without a Convexity partial result.

```bash
python -m pytest -q tests/math/numerical
python -m pyright convexity/math examples --pythonpath "$(command -v python)"
```

Verification includes known solutions, complex algebraic identities, rank
 deficiency, matrix-free operations, Gaussian polynomial exactness, numerical
refinement, JVP/VJP adjoint identities, stiff/non-stiff ODEs, event termination,
BVP analytic solutions, convex-hull bounds, finite-difference convergence,
backend replacement, and failure/shape checks. Minimum dependency versions are
checked separately, with optional PyTorch tests skipped when absent.

Library references: [SciPy linear algebra](https://docs.scipy.org/doc/scipy/reference/linalg.html),
[sparse linear algebra](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html),
[differentiation](https://docs.scipy.org/doc/scipy/reference/differentiate.html),
[IVPs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html),
[BVPs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html),
[findiff differentiation operators](https://findiff.readthedocs.io/en/latest/source/getstarted.html),
[PyTorch autograd](https://docs.pytorch.org/docs/stable/autograd.html).
