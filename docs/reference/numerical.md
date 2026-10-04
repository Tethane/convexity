# `convexity.math.numerical`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Numerical mathematics: checked contracts, replaceable tools, concrete built-ins.

    from convexity.math import numerical as nm
    solution = nm.solve([[2., 0.], [0., 4.]], [6., 8.])
    print(solution.value)

Use interfaces.run(problem, callable) for custom tools. Private _backends are
implementation details and are not part of the supported public API.

## `AdaptiveQuadrature`

```python
AdaptiveQuadrature(atol: 'float' = 1e-10, rtol: 'float' = 1e-10, limit: 'int' = 100) -> None
```

QUADPACK adaptive real scalar integration, including infinite bounds.

## `AutoDiff`

```python
AutoDiff(device: 'str' = 'cpu') -> None
```

PyTorch derivatives of tensor callbacks, evaluated in float64.

Inputs/results use the ordinary NumPy contract; only the callback uses
tensors. Results are detached. Constant/disconnected outputs have zero
derivatives, following PyTorch's strict=False convention.

## `DenseSolve`

```python
DenseSolve() -> None
```

General dense square solve; singular/ill-conditioned systems raise.

## `Eigendecomposition`

```python
Eigendecomposition() -> None
```

Compute eigenvalues and column eigenvectors; hermitian mode requires a Hermitian input matrix.

## `FiniteDifference`

```python
FiniteDifference(atol: 'float' = 1e-08, rtol: 'float' = 1e-06, order: 'int' = 8, maxiter: 'int' = 10, initial_step: 'float' = 0.5, direction: 'int' = 0) -> None
```

SciPy adaptive finite differences of one-point Python/NumPy callbacks.

direction=-1/+1 requests one-sided scalar first derivatives. Other
operations use central differences. Error estimates are componentwise.

## `Interpolation`

```python
Interpolation(method: Literal['linear', 'cubic', 'pchip'] = 'linear') -> None
```

Cubic uses not-a-knot boundaries; PCHIP preserves monotone sample shape.

## `LyapunovSolver`

```python
LyapunovSolver() -> None
```

Solve the continuous or discrete matrix equation declared by LyapunovEquation.

## `MatrixExponentialTool`

```python
MatrixExponentialTool() -> None
```

Compute the matrix exponential exp(A), distinct from elementwise exponentiation.

## `RiccatiSolver`

```python
RiccatiSolver() -> None
```

Compute the stabilizing algebraic Riccati solution for the declared continuous or discrete system.

## `ScalarRootSolver`

```python
ScalarRootSolver(method: Literal['brentq', 'bisect', 'toms748', 'newton', 'secant'] = 'brentq', xtol: float = 1e-12, rtol: float = 1e-10, maxiter: int = 100) -> None
```

Find a scalar zero using a bracket or initial values, with convergence diagnostics.

## `SingularValueDecomposition`

```python
SingularValueDecomposition() -> None
```

Factor a matrix into U, singular values and conjugate-transposed right vectors.

## `SparseSolve`

```python
SparseSolve(method: "Literal['direct', 'cg', 'gmres', 'bicgstab']" = 'direct', atol: 'float' = 0.0, rtol: 'float' = 1e-10, maxiter: 'int' = 1000, preconditioner: 'MatrixLike | None' = None) -> None
```

Sparse direct, CG (requires positive definite A), or GMRES solve.

Iterative methods accept one RHS. maxiter counts inner iterations for
GMRES. No dense conversion or positive-definiteness factorization occurs.

## `SystemRootSolver`

```python
SystemRootSolver(xtol: float = 1e-10, residual_tol: float = 1e-08, max_evaluations: int = 1000, method: str = 'hybr', maxiter: int = 100) -> None
```

Powell hybrid solver; require both solver success and a small residual.

## `derivative`

```python
derivative(function: 'Callable[[NativeTensor], NativeTensor]', x: 'float', *, tool: 'AutoDiff', check_convergence: 'bool' = True) -> 'ArrayResult'
derivative(function: 'Callable[[float], ArrayLike]', x: 'float', *, tool: 'NumericalTool[DifferentiationProblem, ArrayResult] | None' = None, check_convergence: 'bool' = True) -> 'ArrayResult'
```

First derivative of a scalar function at a scalar point.

## `eig`

```python
eig(a: 'ArrayLike', *, hermitian: 'bool' = False, tool: 'NumericalTool[Eigenproblem, EigenResult] | None' = None) -> 'EigenResult'
```

Right eigenpairs; Hermitian eigenvalues are ascending and real.

## `eigh`

```python
eigh(a: 'ArrayLike', b: 'None' = None, *, tool: 'NumericalTool[Eigenproblem, EigenResult] | None' = None) -> 'EigenResult'
eigh(a: 'ArrayLike', b: 'ArrayLike', *, tool: 'NumericalTool[GeneralizedEigenproblem, GeneralizedEigenResult] | None' = None) -> 'GeneralizedEigenResult'
```

Hermitian eigenpairs; supplying B solves A v = lambda B v with B positive definite.

Columns are eigenvectors, ordered by increasing eigenvalue. Ordinary tools
receive Eigenproblem; generalized tools receive GeneralizedEigenproblem.
The selected problem checks the corresponding result contract.

## `expm`

```python
expm(a: 'ArrayLike', *, tool: 'NumericalTool[MatrixExponential, ArrayResult] | None' = None) -> 'ArrayResult'
```

Dense matrix exponential, not elementwise exponentiation.

## `gradient`

```python
gradient(function: 'Callable[[NativeTensor], NativeTensor]', x: 'ArrayLike', *, tool: 'AutoDiff', check_convergence: 'bool' = True) -> 'ArrayResult'
gradient(function: 'Callable[[FloatArray], ArrayLike]', x: 'ArrayLike', *, tool: 'NumericalTool[DifferentiationProblem, ArrayResult] | None' = None, check_convergence: 'bool' = True) -> 'ArrayResult'
```

Gradient of a scalar-valued function; x and result have shape (n,).

## `hessian`

```python
hessian(function: 'Callable[[NativeTensor], NativeTensor]', x: 'ArrayLike', *, tool: 'AutoDiff', check_convergence: 'bool' = True) -> 'ArrayResult'
hessian(function: 'Callable[[FloatArray], ArrayLike]', x: 'ArrayLike', *, tool: 'NumericalTool[DifferentiationProblem, ArrayResult] | None' = None, check_convergence: 'bool' = True) -> 'ArrayResult'
```

Hessian of a scalar-valued function; result has shape (n, n).

## `interpolate`

```python
interpolate(x: ArrayLike, y: ArrayLike, *, extrapolate: bool = False, tool: convexity.math.numerical.interfaces.base.NumericalTool[convexity.math.numerical.interfaces.interpolation.InterpolationProblem, convexity.math.numerical.interfaces.interpolation.Interpolant] | None = None) -> convexity.math.numerical.interfaces.interpolation.Interpolant
```

Fit a callable interpolant; out-of-domain queries raise unless opted in.

## `jacobian`

```python
jacobian(function: 'Callable[[NativeTensor], NativeTensor]', x: 'ArrayLike', *, tool: 'AutoDiff', check_convergence: 'bool' = True) -> 'ArrayResult'
jacobian(function: 'Callable[[FloatArray], ArrayLike]', x: 'ArrayLike', *, tool: 'NumericalTool[DifferentiationProblem, ArrayResult] | None' = None, check_convergence: 'bool' = True) -> 'ArrayResult'
```

Jacobian of f: R^n -> R^m; result has shape (m, n).

## `quad`

```python
quad(function: 'Callable[[float], float]', a: 'float', b: 'float', *, tool: 'NumericalTool[QuadratureProblem, ArrayResult] | None' = None, check_convergence: 'bool' = True) -> 'ArrayResult'
```

Integrate a real scalar function; result includes an absolute error estimate.

## `root`

```python
root(function: Callable[..., ArrayLike], x0: ArrayLike, *, jacobian: Callable[..., ArrayLike] | None = None, tool: convexity.math.numerical.interfaces.base.NumericalTool[convexity.math.numerical.interfaces.roots.RootSystem, convexity.math.numerical.interfaces.base.ArrayResult] | None = None, check_convergence: bool = True) -> convexity.math.numerical.interfaces.base.ArrayResult
```

Find a zero of a square nonlinear system; inspect residual_norm in diagnostics.

## `root_scalar`

```python
root_scalar(function: Callable[[float], float], *, bracket: tuple[float, float] | None = None, x0: float | None = None, x1: float | None = None, derivative: Callable[[float], float] | None = None, tool: convexity.math.numerical.interfaces.base.NumericalTool[convexity.math.numerical.interfaces.roots.ScalarRootProblem, convexity.math.numerical.interfaces.base.ArrayResult] | None = None, check_convergence: bool = True) -> convexity.math.numerical.interfaces.base.ArrayResult
```

Find a scalar zero; the default Brent solver requires a sign-changing bracket.

## `solve`

```python
solve(a: 'MatrixLike', b: 'ArrayLike', *, tool: 'NumericalTool[LinearSystem, ArrayResult] | None' = None, check_convergence: 'bool' = True) -> 'ArrayResult'
```

Solve A x = b, selecting a dense or sparse direct solver by default.

## `solve_lyapunov`

```python
solve_lyapunov(a: ArrayLike, q: ArrayLike, *, time: Literal['continuous', 'discrete'] = 'continuous', tool: convexity.math.numerical.interfaces.base.NumericalTool[convexity.math.numerical.interfaces.matrix_equations.LyapunovEquation, convexity.math.numerical.interfaces.base.ArrayResult] | None = None, check_convergence: bool = True) -> convexity.math.numerical.interfaces.base.ArrayResult
```

Solve AX+XA^H+Q=0 (continuous) or AXA^H-X+Q=0 (discrete).

## `solve_riccati`

```python
solve_riccati(a: ArrayLike, b: ArrayLike, q: ArrayLike, r: ArrayLike, *, time: Literal['continuous', 'discrete'] = 'continuous', tool: convexity.math.numerical.interfaces.base.NumericalTool[convexity.math.numerical.interfaces.matrix_equations.RiccatiEquation, convexity.math.numerical.interfaces.base.ArrayResult] | None = None, check_convergence: bool = True) -> convexity.math.numerical.interfaces.base.ArrayResult
```

Solve the standard CARE or DARE; return X and its equation residual norm.

## `sparse_matrix`

```python
sparse_matrix(data: 'MatrixLike', *, format: "Literal['csr', 'csc']" = 'csr', shape: 'tuple[int, int] | None' = None) -> 'csr_array | csc_array'
```

Create a native SciPy sparse array from dense, sparse, or COO-tuple data.

COO form: (values, (rows, columns)), with shape=(m, n). Sparse arrays retain
standard @, slicing, arithmetic, .toarray(), and .asformat() operations.

## `svd`

```python
svd(a: 'ArrayLike', *, full_matrices: 'bool' = False, tool: 'NumericalTool[SVDProblem, SVDResult] | None' = None) -> 'SVDResult'
```

Reduced SVD by default; singular values are descending.

## `ArrayResult`

```python
ArrayResult(value: 'ArrayLike', diagnostics: 'Diagnostics | None' = None) -> 'None'
```

A scalar (0-D), vector, or matrix value with numerical diagnostics.

## `ConvergenceError`

```python
ConvergenceError(result: 'object') -> 'None'
```

A tool reported failure. The unmodified result is available as .result.

## `Diagnostics`

```python
Diagnostics(converged: 'bool' = True, message: 'str' = '', iterations: 'int | None' = None, evaluations: 'int | None' = None, residual_norm: 'float | None' = None, error_estimate: 'ArrayLike | None' = None) -> 'None'
```

Convergence report; unavailable quantities are None, never fabricated.

## `DifferentiationKind`

```python
DifferentiationKind = Literal['derivative', 'gradient', 'jacobian', 'hessian']
```

Permitted values for this public type alias.

## `DifferentiationProblem`

```python
DifferentiationProblem(function: 'Callable[[FloatArray], ArrayLike] | Callable[[float], ArrayLike] | Callable[[NativeTensor], NativeTensor]', x: 'ArrayLike', kind: 'DifferentiationKind' = 'gradient') -> None
```

One evaluation point; scalar for derivative, (n,) for other operations.

NumPy callbacks consume one point at a time. Autodiff callbacks consume a
PyTorch tensor of the same shape and must return a tensor. A Jacobian's
axes are (output, input); Hessians and gradients require scalar output.

### `DifferentiationProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `Eigenproblem`

```python
Eigenproblem(a: 'ArrayLike', hermitian: 'bool' = False) -> None
```

Square-matrix spectral problem. Eigenvectors are columns; hermitian=True asserts conjugate symmetry.

### `Eigenproblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `EigenResult`

```python
EigenResult(values: 'ArrayLike', vectors: 'ArrayLike') -> 'None'
```

Right eigenvectors are columns: A @ vectors = vectors * values.

## `Interpolant`

```python
Interpolant(evaluate: 'Callable[[NumericArray], ArrayLike]', domain: 'tuple[float, float]', output_shape: 'tuple[int, ...]' = (), extrapolate: 'bool' = False) -> 'None'
```

Checked callable. Query shape S returns shape S + output_shape.

## `InterpolationProblem`

```python
InterpolationProblem(x: 'ArrayLike', y: 'ArrayLike', extrapolate: 'bool' = False) -> None
```

Strictly increasing x: (n,), y: (n, ...); n >= 2, sample axis zero.

### `InterpolationProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `LinearSystem`

```python
LinearSystem(a: 'MatrixLike', b: 'ArrayLike') -> 'None'
```

A x = b: square dense/sparse A; b has shape (n,) or (n, k).

### `LinearSystem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `LyapunovEquation`

```python
LyapunovEquation(a: ArrayLike, q: ArrayLike, time: Literal['continuous', 'discrete'] = 'continuous') -> None
```

Continuous: AX + XA^H + Q = 0. Discrete: AXA^H - X + Q = 0.

### `LyapunovEquation.validate_result`

```python
validate_result(self, result: object) -> None
```

## `MatrixExponential`

```python
MatrixExponential(a: 'ArrayLike') -> None
```

Square matrix whose exponential is requested; accepts real or complex entries.

### `MatrixExponential.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `NumericalTool`

Implement __call__, or supply an ordinary function with this signature.

## `Problem`

Structural numerical problem with validate_result(result); custom tools must satisfy the declared result contract.

### `Problem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `QuadratureProblem`

```python
QuadratureProblem(function: 'Callable[[float], float]', a: 'float', b: 'float') -> None
```

Integral of a real scalar function over [a, b]; infinities are allowed.

### `QuadratureProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `RiccatiEquation`

```python
RiccatiEquation(a: ArrayLike, b: ArrayLike, q: ArrayLike, r: ArrayLike, time: Literal['continuous', 'discrete'] = 'continuous') -> None
```

CARE: A^H X + XA - XBR^-1 B^H X + Q = 0.

DARE: A^H XA - X - A^H XB(R+B^H XB)^-1 B^H XA + Q = 0.
Q and R are Hermitian. Existence/stabilizability is checked by the solver.

### `RiccatiEquation.validate_result`

```python
validate_result(self, result: object) -> None
```

## `RootSystem`

```python
RootSystem(function: Callable[..., ArrayLike], x0: ArrayLike, jacobian: Callable[..., ArrayLike] | None = None) -> None
```

f: R^n -> R^n, starting at x0; optional Jacobian has shape (n, n).

### `RootSystem.validate_result`

```python
validate_result(self, result: object) -> None
```

## `ScalarRootProblem`

```python
ScalarRootProblem(function: Callable[[float], float], bracket: tuple[float, float] | None = None, x0: float | None = None, x1: float | None = None, derivative: Callable[[float], float] | None = None) -> None
```

Find f(x)=0 using a bracket or initial guesses (depending on the tool).

### `ScalarRootProblem.validate_result`

```python
validate_result(self, result: object) -> None
```

## `SVDProblem`

```python
SVDProblem(a: 'ArrayLike', full_matrices: 'bool' = False) -> None
```

Rectangular matrix decomposition request; full_matrices controls full versus reduced bases.

### `SVDProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `SVDResult`

```python
SVDResult(u: 'ArrayLike', s: 'ArrayLike', vh: 'ArrayLike') -> 'None'
```

A = U[:, :k] @ diag(s) @ Vh[:k, :], k = min(m, n).

## `TimeDomain`

```python
TimeDomain = Literal['continuous', 'discrete']
```

Permitted values for this public type alias.

## `run`

```python
run(problem: 'P', tool: 'NumericalTool[P, R]', *, check_convergence: 'bool' = True) -> 'R'
```

Execute a compatible tool and check its output contract and convergence.

Problem constructors validate and copy inputs before a tool is called.
User tools own their algorithms, convergence claims, and diagnostics.

## `LeastSquares`

```python
LeastSquares(rcond: 'float | None' = None) -> None
```

Solve linear least squares using a chosen numerical decomposition and rank threshold.

## `GeneralizedEigen`

```python
GeneralizedEigen() -> None
```

Solve A v = lambda B v, retaining homogeneous eigenvalue coordinates for singular B.

## `Factorization`

```python
Factorization() -> None
```

Construct the requested QR, LU or Cholesky factors; inspect named factors and any permutation.

## `MatrixProperties`

```python
MatrixProperties() -> None
```

Evaluate a named matrix property such as rank, norm, condition or determinant.

## `lstsq`

```python
lstsq(a: 'ArrayLike', b: 'ArrayLike', *, tool: 'NumericalTool[LeastSquaresProblem, LeastSquaresResult] | None' = None) -> 'LeastSquaresResult'
```

Minimize ||A @ x - b||_2 for A (m,n), b (m,) or (m,k).

Return the minimum-norm solution, numerical rank and singular values; no
statistical covariance is inferred. A user tool receives LeastSquaresProblem.

## `generalized_eig`

```python
generalized_eig(a: 'ArrayLike', b: 'ArrayLike', *, hermitian: 'bool' = False, tool: 'NumericalTool[GeneralizedEigenproblem, GeneralizedEigenResult] | None' = None) -> 'GeneralizedEigenResult'
```

Solve A v = lambda B v for square matrices of equal shape.

Return homogeneous alpha, beta and column eigenvectors: beta*A*v = alpha*B*v.
Zero beta represents infinite eigenvalues; alpha=beta=0 is indeterminate.

## `qr`

```python
qr(a: 'ArrayLike', *, pivoting: 'bool' = False, tool: 'NumericalTool[FactorizationProblem, FactorizationResult] | None' = None) -> 'FactorizationResult'
```

Factor A (m,n) into reduced Q and R.

Return named factors and a column permutation p with A[:,p] = Q @ R.
With pivoting=False, p is the identity.

## `lu`

```python
lu(a: 'ArrayLike', *, tool: 'NumericalTool[FactorizationProblem, FactorizationResult] | None' = None) -> 'FactorizationResult'
```

Factor A (m,n) with row pivoting; return p, l, u satisfying A = p @ l @ u.

## `cholesky`

```python
cholesky(a: 'ArrayLike', *, tool: 'NumericalTool[FactorizationProblem, FactorizationResult] | None' = None) -> 'FactorizationResult'
```

Factor a Hermitian positive-definite matrix A; return l with A = l @ l.conj().T.

## `schur`

```python
schur(a: 'ArrayLike', *, tool: 'NumericalTool[FactorizationProblem, FactorizationResult] | None' = None) -> 'FactorizationResult'
```

Return Schur factors t, z satisfying A = z @ t @ z.conj().T.

## `matrix_rank`

```python
matrix_rank(a: 'ArrayLike', *, tolerance: 'float | None' = None, tool: 'NumericalTool[MatrixPropertyProblem, ArrayResult] | None' = None) -> 'ArrayResult'
```

Return the numerical rank of A (m,n), using an optional absolute singular-value tolerance.

## `condition_number`

```python
condition_number(a: 'ArrayLike', *, order: 'float | str | None' = None, tool: 'NumericalTool[MatrixPropertyProblem, ConditionResult] | None' = None) -> 'ConditionResult'
```

Return ||A|| * ||A^-1|| in the selected norm; singular matrices may return infinity.

## `norm`

```python
norm(a: 'ArrayLike', *, order: 'float | str | None' = None, tool: 'NumericalTool[MatrixPropertyProblem, ArrayResult] | None' = None) -> 'ArrayResult'
```

Return a scalar vector/matrix norm; order follows NumPy norm conventions.

## `pinv`

```python
pinv(a: 'ArrayLike', *, tolerance: 'float | None' = None, tool: 'NumericalTool[MatrixPropertyProblem, ArrayResult] | None' = None) -> 'ArrayResult'
```

Return the Moore-Penrose pseudoinverse of A (m,n), shaped (n,m).

Tolerance controls numerical rank, not an elementwise approximation error.

## `det`

```python
det(a: 'ArrayLike', *, tool: 'NumericalTool[MatrixPropertyProblem, ArrayResult] | None' = None) -> 'ArrayResult'
```

Return the scalar determinant of a square real or complex matrix.

## `slogdet`

```python
slogdet(a: 'ArrayLike', *, tool: 'NumericalTool[MatrixPropertyProblem, LogDetResult] | None' = None) -> 'LogDetResult'
```

Return determinant phase/sign and log absolute magnitude.

A singular determinant has sign=0 and logabsdet=-inf; complex phase is preserved.

## `logm`

```python
logm(a: 'ArrayLike', *, tool: 'NumericalTool[MatrixPropertyProblem, ArrayResult] | None' = None) -> 'ArrayResult'
```

Return the principal matrix logarithm, which may be complex for real input.

## `sqrtm`

```python
sqrtm(a: 'ArrayLike', *, tool: 'NumericalTool[MatrixPropertyProblem, ArrayResult] | None' = None) -> 'ArrayResult'
```

Return a principal matrix square root, which may be complex for real input.

## `MatrixFreeOperator`

```python
MatrixFreeOperator() -> None
```

Wrap a shape and matrix-vector action without materializing a dense matrix.

## `SparseSpectral`

```python
SparseSpectral(tolerance: 'float' = 1e-10, maxiter: 'int' = 1000) -> None
```

Compute selected sparse eigenpairs with explicit selection and convergence settings.

## `linear_operator`

```python
linear_operator(shape: 'tuple[int, int]', matvec: 'Callable[[NumericArray], ArrayLike]', *, rmatvec: 'Callable[[NumericArray], ArrayLike] | None' = None, dtype: 'DTypeLike' = <class 'numpy.float64'>, tool: 'NumericalTool[OperatorProblem, LinearOperator] | None' = None) -> 'LinearOperator'
```

rmatvec applies the conjugate transpose; required by sparse SVD.

## `sparse_eig`

```python
sparse_eig(a: 'MatrixLike', k: 'int' = 1, *, hermitian: 'bool' = True, which: 'str' = 'LM', tool: 'NumericalTool[SparseSpectrumProblem, EigenResult] | None' = None) -> 'EigenResult'
```

Compute k eigenpairs of a sparse matrix or linear operator.

Eigenvectors are columns. which selects spectral location; hermitian=True
requires a self-adjoint operator. Convergence failures are reported explicitly.

## `sparse_svd`

```python
sparse_svd(a: 'MatrixLike', k: 'int' = 1, *, which: 'str' = 'LM', tool: 'NumericalTool[SparseSpectrumProblem, SVDResult] | None' = None) -> 'SVDResult'
```

Compute k singular triplets of a sparse matrix or linear operator.

Return u, nonnegative s, vh; truncation is an approximation, not full factorization.

## `FixedPointSolver`

```python
FixedPointSolver(method: 'str' = 'del2', xtol: 'float' = 1e-10, residual_tol: 'float' = 1e-08, maxiter: 'int' = 500) -> None
```

Iterate a vector map toward x = f(x); termination depends on the configured tolerance.

## `fixed_point`

```python
fixed_point(function: 'Callable[[FloatArray], ArrayLike]', x0: 'ArrayLike', *, tool: 'NumericalTool[FixedPointProblem, ArrayResult] | None' = None, check_convergence: 'bool' = True) -> 'ArrayResult'
```

Solve x = function(x) from scalar or vector x0.

The callback must preserve shape. Return ArrayResult with convergence diagnostics;
check_convergence=True raises ConvergenceError on reported failure.

## `FiniteDifferenceProduct`

```python
FiniteDifferenceProduct(differentiation: 'FiniteDifference' = FiniteDifference(atol=1e-08, rtol=1e-06, order=8, maxiter=10, initial_step=0.5, direction=0)) -> None
```

Approximate Jacobian/Hessian products numerically using the configured finite-difference step.

## `AutoDiffProduct`

```python
AutoDiffProduct(device: 'str' = 'cpu') -> None
```

Compute derivative products through a differentiable tensor callback; NumPy conversion breaks the graph.

## `jvp`

```python
jvp(function: 'Callable[[FloatArray], ArrayLike]', x: 'ArrayLike', vector: 'ArrayLike', *, tool: 'NumericalTool[DerivativeProductProblem, ArrayResult] | None' = None, check_convergence: 'bool' = True) -> 'ArrayResult'
```

Evaluate Jacobian(function)(x) @ vector for vector x and vector output.

vector has input shape; the result has output shape and numerical diagnostics.

## `vjp`

```python
vjp(function: 'Callable[[FloatArray], ArrayLike]', x: 'ArrayLike', vector: 'ArrayLike', *, tool: 'NumericalTool[DerivativeProductProblem, ArrayResult] | None' = None, check_convergence: 'bool' = True) -> 'ArrayResult'
```

Evaluate vector @ Jacobian(function)(x) for vector x and vector output.

vector has output shape; the result has input shape and numerical diagnostics.

## `check_derivative`

```python
check_derivative(function: 'Callable[[FloatArray], ArrayLike]', x: 'ArrayLike', actual: 'ArrayLike | ArrayResult', *, kind: 'DifferentiationKind' = 'jacobian', atol: 'float' = 1e-06, rtol: 'float' = 1e-05, reference_tool: 'NumericalTool[DifferentiationProblem, ArrayResult] | None' = None) -> 'DerivativeCheck'
```

Compare a supplied derivative array/result against numerical differences.

Pass a NumPy callback as function even when actual was computed with PyTorch.
Failed reference differentiation raises rather than producing a passing check.

## `MultidimensionalQuadrature`

```python
MultidimensionalQuadrature(atol: 'float' = 1e-08, rtol: 'float' = 1e-08, limit: 'int' = 100) -> None
```

Integrate a scalar callback over a rectangular domain with numerical error controls.

## `GaussianQuadrature`

```python
GaussianQuadrature(n: 'int' = 16) -> None
```

Integrate by a finite Gaussian quadrature rule; polynomial exactness depends on the chosen rule.

## `GaussianRule`

```python
GaussianRule() -> None
```

Construct weighted Gaussian quadrature nodes, with the declared weight function and domain.

## `nquad`

```python
nquad(function: 'Callable[[FloatArray], float]', bounds: 'ArrayLike | str', *, tool: 'NumericalTool[BoxIntegral, ArrayResult] | None' = None, check_convergence: 'bool' = True) -> 'ArrayResult'
```

Integrate a scalar function(x) over bounds (dimension,2).

The callback receives one coordinate vector; bounds may be infinite.
Return the integral with a numerical error estimate, not a probabilistic interval.

## `gauss_quad`

```python
gauss_quad(function: 'Callable[[float], float]', a: 'float', b: 'float', *, n: 'int' = 16, tool: 'NumericalTool[QuadratureProblem, ArrayResult] | None' = None) -> 'ArrayResult'
```

Approximate the scalar integral from a to b using n Gaussian nodes.

The fixed-order rule does not supply an adaptive error guarantee.

## `quadrature_rule`

```python
quadrature_rule(n: 'int', *, family: 'str' = 'legendre', a: 'float' = -1.0, b: 'float' = 1.0, tool: 'NumericalTool[QuadratureRuleProblem, QuadratureRule] | None' = None) -> 'QuadratureRule'
```

Return n increasing nodes and positive weights.

Legendre integrates unit weight on [a,b]; Hermite uses exp(-x**2) on the real
line and Laguerre uses exp(-x) on [0,infinity). Weights are not probabilities.

## `integrate_density`

```python
integrate_density(function: 'Callable[[float], float]', density: 'Callable[[float], float]', a: 'float', b: 'float', *, tool: 'NumericalTool[QuadratureProblem, ArrayResult] | None' = None, check_convergence: 'bool' = True) -> 'ArrayResult'
```

Integral f(x)*density(x) dx. Density is supplied; no normalization is assumed or fitted.

## `Spline`

```python
Spline(degree: 'int' = 3, boundary: 'str' = 'not-a-knot') -> None
```

Construct a one-dimensional spline with explicit boundary and extrapolation conventions.

## `MultivariateInterpolation`

```python
MultivariateInterpolation(method: 'str' = 'linear') -> None
```

Interpolate regular-grid or scattered observations with explicit extrapolation behavior.

## `PolynomialApproximation`

```python
PolynomialApproximation(basis: 'str' = 'chebyshev') -> None
```

Fit a polynomial of the requested degree to supplied observations.

## `spline`

```python
spline(x: 'ArrayLike', y: 'ArrayLike', *, extrapolate: 'bool' = False, tool: 'NumericalTool[InterpolationProblem, Interpolant] | None' = None) -> 'Interpolant'
```

Interpolate values y with leading axis len(x) on a strictly increasing x grid.

Return a callable interpolant; queries outside the domain require extrapolate=True.

## `regular_grid`

```python
regular_grid(axes: 'Sequence[ArrayLike]', values: 'ArrayLike', *, extrapolation: 'str' = 'raise', tool: 'NumericalTool[MultivariateInterpolationProblem, MultivariateInterpolant] | None' = None) -> 'MultivariateInterpolant'
```

Interpolate a tensor grid with one increasing axis per coordinate.

Leading values dimensions match axis lengths; query shape is (...,dimension).
Trailing value axes survive. Extrapolation must be selected explicitly.

## `scattered`

```python
scattered(points: 'ArrayLike', values: 'ArrayLike', *, extrapolation: 'str' = 'raise', tool: 'NumericalTool[MultivariateInterpolationProblem, MultivariateInterpolant] | None' = None) -> 'MultivariateInterpolant'
```

Interpolate points (n,dimension) and values (n,...), dimension>=2.

Queries have shape (...,dimension). Outside-hull behavior follows extrapolation.

## `polynomial_approximation`

```python
polynomial_approximation(x: 'ArrayLike', y: 'ArrayLike', degree: 'int', *, extrapolate: 'bool' = False, tool: 'NumericalTool[PolynomialProblem, PolynomialResult] | None' = None, check_convergence: 'bool' = True) -> 'PolynomialResult'
```

Fit degree+1 coefficients to scalar y on increasing sample coordinates x.

Return a callable approximation and coefficients in increasing ordinary powers
of x. Rank loss is reflected in convergence diagnostics.

## `IVPSolver`

```python
IVPSolver(method: 'str' = 'RK45', atol: 'float' = 1e-09, rtol: 'float' = 1e-07, max_step: 'float' = inf) -> None
```

Integrate dy/dt = f(t,y) with adaptive numerical tolerances and optional event termination.

## `BVPSolver`

```python
BVPSolver(tolerance: 'float' = 1e-06, boundary_tolerance: 'float' = 1e-08, max_nodes: 'int' = 1000) -> None
```

Solve a vector boundary-value problem on an initial mesh; inspect residuals and convergence.

## `solve_ivp`

```python
solve_ivp(function: 'Callable[[float, FloatArray], ArrayLike]', y0: 'ArrayLike', t_span: 'tuple[float, float]', *, t_eval: 'ArrayLike | None' = None, events: 'Sequence[ODEEvent]' = (), jacobian: 'Callable[[float, FloatArray], ArrayLike] | None' = None, tool: 'NumericalTool[InitialValueProblem, ODEResult] | None' = None, check_convergence: 'bool' = True) -> 'ODEResult'
```

Integrate y'=function(t,y), where y is a state vector.

Return times (t,), states (state,t), dense solution and event records.
Event states use (event,state). t_eval follows the integration direction;
terminal events may stop before t_span ends. Callback units must be consistent.

## `solve_ode`

```python
solve_ode(function: 'Callable[[float, FloatArray], ArrayLike]', y0: 'ArrayLike', t: 'ArrayLike', *, tool: 'NumericalTool[InitialValueProblem, ODEResult] | None' = None, events: 'Sequence[ODEEvent]' = (), jacobian: 'Callable[[float, FloatArray], ArrayLike] | None' = None, check_convergence: 'bool' = True) -> 'ODEResult'
```

Convenience IVP on a supplied increasing or decreasing time grid.

## `solve_bvp`

```python
solve_bvp(function: 'Callable[[FloatArray, FloatArray], ArrayLike]', boundary: 'Callable[[FloatArray, FloatArray], ArrayLike]', x: 'ArrayLike', y: 'ArrayLike', *, tool: 'NumericalTool[BoundaryValueProblem, BoundaryValueResult] | None' = None, check_convergence: 'bool' = True) -> 'BoundaryValueResult'
```

Solve y'=function(x,y), boundary(y_left,y_right)=0 on an increasing mesh.

Callbacks use y (state,mesh), boundary returns (state,). Return an adaptive
mesh, solution, residuals and convergence diagnostics.

## `GridConstruction`

```python
GridConstruction() -> None
```

Build a linear, logarithmic, geometric or arithmetic coordinate grid.

## `MeshConstruction`

```python
MeshConstruction() -> None
```

Combine coordinate axes with explicit mesh indexing or a Cartesian-product layout.

## `linspace`

```python
linspace(start: 'float', stop: 'float', num: 'int' = 50, *, endpoint: 'bool' = True, tool: 'NumericalTool[GridProblem, FloatArray] | None' = None) -> 'FloatArray'
```

Return num evenly spaced scalar coordinates from start to stop.

endpoint controls inclusion of stop; returns a read-only float64 array.

## `logspace`

```python
logspace(start: 'float', stop: 'float', num: 'int' = 50, *, base: 'float' = 10.0, endpoint: 'bool' = True, tool: 'NumericalTool[GridProblem, FloatArray] | None' = None) -> 'FloatArray'
```

Uniformly spaced exponents, as in NumPy logspace.

## `geomspace`

```python
geomspace(start: 'float', stop: 'float', num: 'int' = 50, *, endpoint: 'bool' = True, tool: 'NumericalTool[GridProblem, FloatArray] | None' = None) -> 'FloatArray'
```

Return num geometrically spaced coordinates with the selected endpoint convention.

## `arange`

```python
arange(start: 'float', stop: 'float', step: 'float' = 1.0, *, tool: 'NumericalTool[GridProblem, FloatArray] | None' = None) -> 'FloatArray'
```

Return a half-open arithmetic grid [start,stop) with nonzero step.

## `meshgrid`

```python
meshgrid(*axes: 'ArrayLike', sparse: 'bool' = False, tool: 'NumericalTool[MeshProblem, tuple[FloatArray, ...]] | None' = None) -> 'tuple[FloatArray, ...]'
```

Tensor mesh with indexing='ij'. Flattening uses C order throughout numerics.

## `cartesian_product`

```python
cartesian_product(*axes: 'ArrayLike', tool: 'NumericalTool[MeshProblem, FloatArray] | None' = None) -> 'FloatArray'
```

Return all combinations of one-dimensional axes as rows, in input-axis order.

## `DifferenceOperator`

```python
DifferenceOperator() -> None
```

Build a finite-difference operator on the supplied coordinate axes and derivative orders.

## `LaplacianOperator`

```python
LaplacianOperator() -> None
```

Build a discrete Laplacian using the supplied grid shape, spacing and boundary convention.

## `differentiation_matrix`

```python
differentiation_matrix(x: 'ArrayLike', *, order: 'int' = 1, accuracy: 'int' = 2, periodic: 'bool' = False, tool: 'NumericalTool[DifferenceOperatorProblem, csr_matrix | csr_array] | None' = None) -> 'csr_matrix | csr_array'
```

Return a finite-difference matrix on increasing coordinates x.

Multiplication by sampled values approximates the requested derivative order;
accuracy controls the stencil. Periodic grids require the periodic convention.

## `partial_derivative_matrix`

```python
partial_derivative_matrix(axes: 'Sequence[ArrayLike]', orders: 'Sequence[int]', *, accuracy: 'int' = 2, periodic: 'bool' = False, tool: 'NumericalTool[DifferenceOperatorProblem, csr_matrix | csr_array] | None' = None) -> 'csr_matrix | csr_array'
```

Mixed partial derivative on a tensor grid; multiply by C-flattened samples.

Nonperiodic boundaries use one-sided stencils, not imposed PDE conditions.
Periodic axes omit the duplicate endpoint.

## `laplacian_matrix`

```python
laplacian_matrix(shape: 'tuple[int, ...]', *, spacing: 'float' = 1.0, boundary: 'str' = 'dirichlet', tool: 'NumericalTool[LaplacianProblem, csr_matrix | csr_array] | None' = None) -> 'csr_matrix | csr_array'
```

Uniform isotropic grid Laplacian with homogeneous boundary conventions.

## `FloatingComparison`

```python
FloatingComparison() -> None
```

Compare arrays using explicit absolute/relative tolerances and optional equal-NaN semantics.

## `Residual`

```python
Residual() -> None
```

Evaluate a linear-system residual and its absolute and relative norms.

## `isclose`

```python
isclose(actual: 'ArrayLike', reference: 'ArrayLike', *, tolerances: 'Tolerances' = Tolerances(atol=1e-08, rtol=1e-05), equal_nan: 'bool' = False, tool: 'NumericalTool[ComparisonProblem, BoolArray] | None' = None) -> 'BoolArray'
```

Compare equal-shaped arrays using |actual-reference| <= atol+rtol*|reference|.

Return a Boolean array; broadcasting is deliberately disallowed. NaNs match
only when equal_nan=True.

## `allclose`

```python
allclose(actual: 'ArrayLike', reference: 'ArrayLike', *, tolerances: 'Tolerances' = Tolerances(atol=1e-08, rtol=1e-05), equal_nan: 'bool' = False, tool: 'NumericalTool[ComparisonProblem, BoolArray] | None' = None) -> 'bool'
```

Return whether every coordinate passes isclose with the supplied Tolerances.

## `machine_precision`

```python
machine_precision(dtype: 'DTypeLike' = <class 'numpy.float64'>) -> 'MachinePrecision'
```

Return floating-point limits for dtype, including epsilon and normal/subnormal minima.

## `require_finite`

```python
require_finite(values: 'ArrayLike') -> 'NumericArray'
```

Validate and snapshot numeric data, rejecting NaN and Inf rather than replacing them.

## `residual`

```python
residual(a: 'MatrixLike', x: 'ArrayLike', b: 'ArrayLike', *, tool: 'NumericalTool[ResidualProblem, ResidualResult] | None' = None) -> 'ResidualResult'
```

Absolute Euclidean/Frobenius residual and residual relative to ||b||.

## `scipy_backend`

```python
scipy_backend() -> 'Backend'
```

NumPy/SciPy numerical tools, with findiff for differentiation matrices.

## `torch_backend`

```python
torch_backend(*, device: 'str' = 'cpu') -> 'Backend'
```

Autodiff capabilities only; unsupported operations fail explicitly.

## `LeastSquaresProblem`

```python
LeastSquaresProblem(a: 'ArrayLike', b: 'ArrayLike') -> 'None'
```

Minimize the Euclidean residual norm of A x - b; b may hold one or multiple right-hand sides.

### `LeastSquaresProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `LeastSquaresResult`

```python
LeastSquaresResult(value: 'ArrayLike', rank: 'int', singular_values: 'ArrayLike', diagnostics: 'Diagnostics | None' = None) -> 'None'
```

Least-squares solution with numerical rank, singular values and termination diagnostics.

## `GeneralizedEigenproblem`

```python
GeneralizedEigenproblem(a: 'ArrayLike', b: 'ArrayLike', hermitian: 'bool' = False) -> 'None'
```

Square matrices A and B defining A v = lambda B v; Hermitian mode also restricts B.

### `GeneralizedEigenproblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `GeneralizedEigenResult`

```python
GeneralizedEigenResult(alpha: 'ArrayLike', beta: 'ArrayLike', vectors: 'ArrayLike') -> 'None'
```

beta * (A @ vectors) = alpha * (B @ vectors), including infinite eigenvalues.

## `FactorizationProblem`

```python
FactorizationProblem(a: 'ArrayLike', kind: 'str' = 'qr', pivoting: 'bool' = False) -> 'None'
```

Matrix and factorization kind, with pivoting requested explicitly where supported.

### `FactorizationProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `FactorizationResult`

```python
FactorizationResult(factors: 'Mapping[str, NumericArray]', permutation: 'IntArray | None' = None) -> None
```

Named matrix factors plus an optional integer permutation; conventions follow the requested factorization.

## `MatrixPropertyProblem`

```python
MatrixPropertyProblem(a: 'ArrayLike', operation: 'str', order: 'float | str | None' = None, tolerance: 'float | None' = None) -> 'None'
```

Matrix-property request with optional norm order or numerical rank tolerance.

### `MatrixPropertyProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `ConditionResult`

```python
ConditionResult(value: 'float') -> None
```

Infinity is a legitimate condition number for a singular matrix.

## `LogDetResult`

```python
LogDetResult(sign: 'float | complex', logabsdet: 'float') -> None
```

Determinant represented by sign/complex phase and log absolute magnitude; singular matrices have zero sign.

## `OperatorProblem`

```python
OperatorProblem(shape: 'tuple[int, int]', matvec: 'Callable[[NumericArray], ArrayLike]', rmatvec: 'Callable[[NumericArray], ArrayLike] | None' = None, dtype: 'DTypeLike' = <class 'numpy.float64'>) -> None
```

Shape and matrix-vector callbacks defining a linear operator; adjoint action is optional.

### `OperatorProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `SparseSpectrumProblem`

```python
SparseSpectrumProblem(a: 'MatrixLike', k: 'int', kind: 'str' = 'eigen', hermitian: 'bool' = True, which: 'str' = 'LM') -> 'None'
```

Selected eigenpairs of a sparse matrix or linear operator, with Hermitian structure declared explicitly.

### `SparseSpectrumProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `FixedPointProblem`

```python
FixedPointProblem(function: 'Callable[[FloatArray], ArrayLike]', x0: 'ArrayLike') -> 'None'
```

Vector map f and initial x0 for solving f(x) = x.

### `FixedPointProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `DerivativeProductProblem`

```python
DerivativeProductProblem(function: 'Callable[[FloatArray], ArrayLike] | Callable[[NativeTensor], NativeTensor]', x: 'ArrayLike', vector: 'ArrayLike', kind: 'str' = 'jvp') -> 'None'
```

Callback, evaluation point and direction for a JVP or VJP.

### `DerivativeProductProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `DerivativeCheck`

```python
DerivativeCheck(passed: 'bool', actual: 'ArrayLike', reference: 'ArrayLike', absolute_error: 'ArrayLike', scaled_error: 'ArrayLike') -> 'None'
```

Comparison to a converged finite-difference reference, component by component.

## `BoxIntegral`

```python
BoxIntegral(function: 'Callable[[FloatArray], float]', bounds: 'ArrayLike') -> None
```

Integral of f(x: (d,)) over rectangular, possibly infinite bounds.

### `BoxIntegral.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `QuadratureRuleProblem`

```python
QuadratureRuleProblem(n: 'int', family: 'str' = 'legendre', a: 'float' = -1.0, b: 'float' = 1.0) -> None
```

Number of nodes, polynomial family and integration endpoints for a weighted Gaussian rule.

### `QuadratureRuleProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `QuadratureRule`

```python
QuadratureRule(nodes: 'ArrayLike', weights: 'ArrayLike', weight_function: 'str' = '1') -> 'None'
```

Nodes and positive weights. Weight function is explicit, never a distribution.

## `ODEEvent`

```python
ODEEvent(function: 'Callable[[float, FloatArray], float]', terminal: 'bool' = False, direction: 'int' = 0) -> None
```

Scalar event function of (time,state); direction selects crossing sign and terminal requests stopping.

## `InitialValueProblem`

```python
InitialValueProblem(function: 'Callable[[float, FloatArray], ArrayLike]', y0: 'ArrayLike', t_span: 'tuple[float, float]', t_eval: 'ArrayLike | None' = None, events: 'tuple[ODEEvent, ...]' = (), jacobian: 'Callable[[float, FloatArray], ArrayLike] | None' = None) -> 'None'
```

dy/dt = f(t,y) with one-dimensional initial state, time interval, optional evaluation times and events.

### `InitialValueProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `DenseSolution`

```python
DenseSolution(_evaluate: 'Callable[[FloatArray], ArrayLike]', domain: 'tuple[float, float]', state_size: 'int') -> None
```

Callable dense ODE solution on a declared domain, returning (state,) for a scalar time or (state,time) for a time vector.

## `EventResult`

```python
EventResult(t: 'ArrayLike', y: 'ArrayLike') -> 'None'
```

Detected event times and corresponding states, with one state row per event.

## `ODEResult`

```python
ODEResult(t: 'ArrayLike', y: 'ArrayLike', solution: 'DenseSolution', diagnostics: 'Diagnostics' = Diagnostics(converged=True, message='', iterations=None, evaluations=None, residual_norm=None, error_estimate=None), events: 'tuple[EventResult, ...]' = (), terminated_by_event: 'bool' = False) -> 'None'
```

Numerical trajectory with t=(time,), y=(state,time), dense solution and per-event records.

## `BoundaryValueProblem`

```python
BoundaryValueProblem(function: 'Callable[[FloatArray, FloatArray], ArrayLike]', boundary: 'Callable[[FloatArray, FloatArray], ArrayLike]', x: 'ArrayLike', y: 'ArrayLike') -> 'None'
```

y'=f(x,y), bc(y(a),y(b))=0. Callback states have shape (n, mesh_size).

### `BoundaryValueProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `BoundaryValueResult`

```python
BoundaryValueResult(x: 'ArrayLike', y: 'ArrayLike', solution: 'DenseSolution', rms_residuals: 'ArrayLike', diagnostics: 'Diagnostics') -> 'None'
```

Boundary-value trajectory and dense solution, with mesh residuals and convergence diagnostics.

## `GridProblem`

```python
GridProblem(kind: 'str', start: 'float', stop: 'float', num: 'int' = 50, step: 'float' = 1.0, endpoint: 'bool' = True, base: 'float' = 10.0) -> None
```

Coordinate-grid specification; kind determines whether num, step or base controls construction.

### `GridProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `MeshProblem`

```python
MeshProblem(axes: 'tuple[ArrayLike, ...]', cartesian: 'bool' = False, sparse: 'bool' = False) -> 'None'
```

Coordinate axes and indexing convention for a tensor mesh or Cartesian product.

### `MeshProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `DifferenceOperatorProblem`

```python
DifferenceOperatorProblem(axes: 'tuple[ArrayLike, ...]', orders: 'tuple[int, ...]', accuracy: 'int' = 2, periodic: 'bool' = False) -> 'None'
```

Derivative orders aligned with coordinate axes, plus stencil accuracy and periodicity.

### `DifferenceOperatorProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `LaplacianProblem`

```python
LaplacianProblem(shape: 'tuple[int, ...]', spacing: 'float' = 1.0, boundary: 'str' = 'dirichlet') -> None
```

Grid shape, uniform spacing and boundary condition for a discrete Laplacian.

### `LaplacianProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `MultivariateInterpolationProblem`

```python
MultivariateInterpolationProblem(points: 'ArrayLike | tuple[ArrayLike, ...]', values: 'ArrayLike', layout: 'str' = 'regular', extrapolation: 'str' = 'raise') -> 'None'
```

Regular-grid axes or scattered point rows and aligned values; extrapolation is explicit.

### `MultivariateInterpolationProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `MultivariateInterpolant`

```python
MultivariateInterpolant(_evaluate: 'Callable[[FloatArray], ArrayLike]', dimension: 'int', output_shape: 'tuple[int, ...]' = (), extrapolation: 'str' = 'raise') -> None
```

Callable interpolant accepting points with a final coordinate axis and preserving leading batch axes.

## `PolynomialProblem`

```python
PolynomialProblem(x: 'ArrayLike', y: 'ArrayLike', extrapolate: 'bool' = False, degree: 'int' = 3) -> None
```

One-dimensional sample coordinates and values with an explicit polynomial degree.

### `PolynomialProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `PolynomialResult`

```python
PolynomialResult(interpolant: 'Interpolant', coefficients: 'ArrayLike', rank: 'int', diagnostics: 'Diagnostics') -> 'None'
```

Coefficients are in increasing ordinary powers of x, regardless of fit basis.

## `Tolerances`

```python
Tolerances(atol: 'float' = 1e-08, rtol: 'float' = 1e-05) -> None
```

Componentwise |actual-reference| <= atol + rtol*|reference|.

## `ComparisonProblem`

```python
ComparisonProblem(actual: 'ArrayLike', reference: 'ArrayLike', tolerances: 'Tolerances' = Tolerances(atol=1e-08, rtol=1e-05), equal_nan: 'bool' = False) -> 'None'
```

Actual and reference arrays with tolerance and NaN-equality conventions; shapes are validated.

### `ComparisonProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `MachinePrecision`

```python
MachinePrecision(dtype: 'str', epsilon: 'float | np.floating', smallest_normal: 'float | np.floating', smallest_subnormal: 'float | np.floating', largest: 'float | np.floating') -> None
```

Floating dtype limits: machine epsilon, smallest normal/subnormal and largest finite magnitude.

## `ResidualProblem`

```python
ResidualProblem(a: 'MatrixLike', x: 'ArrayLike', b: 'ArrayLike') -> 'None'
```

Linear system A x = b whose residual is to be measured.

### `ResidualProblem.validate_result`

```python
validate_result(self, result: 'object') -> 'None'
```

## `ResidualResult`

```python
ResidualResult(value: 'ArrayLike', absolute: 'float', relative: 'float') -> 'None'
```

Residual array with absolute and scaled relative norms.

## `Backend`

```python
Backend(name: 'str', tools: 'Mapping[str, Callable[..., object]]') -> None
```

Immutable named capability-to-tool mapping. Known routes retain typed results; custom routes require caller narrowing.

### `Backend.run`

```python
run(self, capability: "Literal['lstsq']", problem: 'LeastSquaresProblem', *, check_convergence: 'bool' = True) -> 'LeastSquaresResult'
run(self, capability: "Literal['generalized_eig']", problem: 'GeneralizedEigenproblem', *, check_convergence: 'bool' = True) -> 'GeneralizedEigenResult'
run(self, capability: "Literal['qr']", problem: 'FactorizationProblem', *, check_convergence: 'bool' = True) -> 'FactorizationResult'
run(self, capability: "Literal['lu']", problem: 'FactorizationProblem', *, check_convergence: 'bool' = True) -> 'FactorizationResult'
run(self, capability: "Literal['cholesky']", problem: 'FactorizationProblem', *, check_convergence: 'bool' = True) -> 'FactorizationResult'
run(self, capability: "Literal['schur']", problem: 'FactorizationProblem', *, check_convergence: 'bool' = True) -> 'FactorizationResult'
run(self, capability: "Literal['matrix_rank']", problem: 'MatrixPropertyProblem', *, check_convergence: 'bool' = True) -> 'ArrayResult'
run(self, capability: "Literal['norm']", problem: 'MatrixPropertyProblem', *, check_convergence: 'bool' = True) -> 'ArrayResult'
run(self, capability: "Literal['pinv']", problem: 'MatrixPropertyProblem', *, check_convergence: 'bool' = True) -> 'ArrayResult'
run(self, capability: "Literal['det']", problem: 'MatrixPropertyProblem', *, check_convergence: 'bool' = True) -> 'ArrayResult'
run(self, capability: "Literal['logm']", problem: 'MatrixPropertyProblem', *, check_convergence: 'bool' = True) -> 'ArrayResult'
run(self, capability: "Literal['sqrtm']", problem: 'MatrixPropertyProblem', *, check_convergence: 'bool' = True) -> 'ArrayResult'
run(self, capability: "Literal['condition_number']", problem: 'MatrixPropertyProblem', *, check_convergence: 'bool' = True) -> 'ConditionResult'
run(self, capability: "Literal['slogdet']", problem: 'MatrixPropertyProblem', *, check_convergence: 'bool' = True) -> 'LogDetResult'
run(self, capability: "Literal['sparse_eig']", problem: 'SparseSpectrumProblem', *, check_convergence: 'bool' = True) -> 'EigenResult'
run(self, capability: "Literal['sparse_svd']", problem: 'SparseSpectrumProblem', *, check_convergence: 'bool' = True) -> 'SVDResult'
run(self, capability: "Literal['linear_operator']", problem: 'OperatorProblem', *, check_convergence: 'bool' = True) -> 'LinearOperator'
run(self, capability: "Literal['fixed_point']", problem: 'FixedPointProblem', *, check_convergence: 'bool' = True) -> 'ArrayResult'
run(self, capability: "Literal['jvp']", problem: 'DerivativeProductProblem', *, check_convergence: 'bool' = True) -> 'ArrayResult'
run(self, capability: "Literal['vjp']", problem: 'DerivativeProductProblem', *, check_convergence: 'bool' = True) -> 'ArrayResult'
run(self, capability: "Literal['nquad']", problem: 'BoxIntegral', *, check_convergence: 'bool' = True) -> 'ArrayResult'
run(self, capability: "Literal['gauss_quad']", problem: 'QuadratureProblem', *, check_convergence: 'bool' = True) -> 'ArrayResult'
run(self, capability: "Literal['quadrature_rule']", problem: 'QuadratureRuleProblem', *, check_convergence: 'bool' = True) -> 'QuadratureRule'
run(self, capability: "Literal['spline']", problem: 'InterpolationProblem', *, check_convergence: 'bool' = True) -> 'Interpolant'
run(self, capability: "Literal['regular_grid']", problem: 'MultivariateInterpolationProblem', *, check_convergence: 'bool' = True) -> 'MultivariateInterpolant'
run(self, capability: "Literal['scattered']", problem: 'MultivariateInterpolationProblem', *, check_convergence: 'bool' = True) -> 'MultivariateInterpolant'
run(self, capability: "Literal['polynomial_approximation']", problem: 'PolynomialProblem', *, check_convergence: 'bool' = True) -> 'PolynomialResult'
run(self, capability: "Literal['solve_ivp']", problem: 'InitialValueProblem', *, check_convergence: 'bool' = True) -> 'ODEResult'
run(self, capability: "Literal['solve_ode']", problem: 'InitialValueProblem', *, check_convergence: 'bool' = True) -> 'ODEResult'
run(self, capability: "Literal['solve_bvp']", problem: 'BoundaryValueProblem', *, check_convergence: 'bool' = True) -> 'BoundaryValueResult'
run(self, capability: "Literal['linspace']", problem: 'GridProblem', *, check_convergence: 'bool' = True) -> 'FloatArray'
run(self, capability: "Literal['logspace']", problem: 'GridProblem', *, check_convergence: 'bool' = True) -> 'FloatArray'
run(self, capability: "Literal['geomspace']", problem: 'GridProblem', *, check_convergence: 'bool' = True) -> 'FloatArray'
run(self, capability: "Literal['arange']", problem: 'GridProblem', *, check_convergence: 'bool' = True) -> 'FloatArray'
run(self, capability: "Literal['meshgrid']", problem: 'MeshProblem', *, check_convergence: 'bool' = True) -> 'tuple[FloatArray, ...]'
run(self, capability: "Literal['cartesian_product']", problem: 'MeshProblem', *, check_convergence: 'bool' = True) -> 'FloatArray'
run(self, capability: "Literal['differentiation_matrix']", problem: 'DifferenceOperatorProblem', *, check_convergence: 'bool' = True) -> 'csr_matrix | csr_array'
run(self, capability: "Literal['partial_derivative_matrix']", problem: 'DifferenceOperatorProblem', *, check_convergence: 'bool' = True) -> 'csr_matrix | csr_array'
run(self, capability: "Literal['laplacian_matrix']", problem: 'LaplacianProblem', *, check_convergence: 'bool' = True) -> 'csr_matrix | csr_array'
run(self, capability: "Literal['isclose']", problem: 'ComparisonProblem', *, check_convergence: 'bool' = True) -> 'BoolArray'
run(self, capability: "Literal['residual']", problem: 'ResidualProblem', *, check_convergence: 'bool' = True) -> 'ResidualResult'
run(self, capability: "Literal['solve']", problem: 'LinearSystem', *, check_convergence: 'bool' = True) -> 'ArrayResult'
run(self, capability: "Literal['eig']", problem: 'Eigenproblem', *, check_convergence: 'bool' = True) -> 'EigenResult'
run(self, capability: "Literal['svd']", problem: 'SVDProblem', *, check_convergence: 'bool' = True) -> 'SVDResult'
run(self, capability: "Literal['expm']", problem: 'MatrixExponential', *, check_convergence: 'bool' = True) -> 'ArrayResult'
run(self, capability: "Literal['quad']", problem: 'QuadratureProblem', *, check_convergence: 'bool' = True) -> 'ArrayResult'
run(self, capability: "Literal['root']", problem: 'RootSystem', *, check_convergence: 'bool' = True) -> 'ArrayResult'
run(self, capability: "Literal['derivative']", problem: 'DifferentiationProblem', *, check_convergence: 'bool' = True) -> 'ArrayResult'
run(self, capability: "Literal['gradient']", problem: 'DifferentiationProblem', *, check_convergence: 'bool' = True) -> 'ArrayResult'
run(self, capability: "Literal['jacobian']", problem: 'DifferentiationProblem', *, check_convergence: 'bool' = True) -> 'ArrayResult'
run(self, capability: "Literal['hessian']", problem: 'DifferentiationProblem', *, check_convergence: 'bool' = True) -> 'ArrayResult'
run(self, capability: "Literal['interpolate']", problem: 'InterpolationProblem', *, check_convergence: 'bool' = True) -> 'Interpolant'
run(self, capability: "Literal['solve_lyapunov']", problem: 'LyapunovEquation', *, check_convergence: 'bool' = True) -> 'ArrayResult'
run(self, capability: "Literal['solve_riccati']", problem: 'RiccatiEquation', *, check_convergence: 'bool' = True) -> 'ArrayResult'
run(self, capability: 'str', problem: 'Problem', *, check_convergence: 'bool' = True) -> 'object'
```

### `Backend.tool`

```python
tool(self, capability: "Literal['lstsq']") -> 'NumericalTool[LeastSquaresProblem, LeastSquaresResult]'
tool(self, capability: "Literal['generalized_eig']") -> 'NumericalTool[GeneralizedEigenproblem, GeneralizedEigenResult]'
tool(self, capability: "Literal['qr']") -> 'NumericalTool[FactorizationProblem, FactorizationResult]'
tool(self, capability: "Literal['lu']") -> 'NumericalTool[FactorizationProblem, FactorizationResult]'
tool(self, capability: "Literal['cholesky']") -> 'NumericalTool[FactorizationProblem, FactorizationResult]'
tool(self, capability: "Literal['schur']") -> 'NumericalTool[FactorizationProblem, FactorizationResult]'
tool(self, capability: "Literal['matrix_rank']") -> 'NumericalTool[MatrixPropertyProblem, ArrayResult]'
tool(self, capability: "Literal['norm']") -> 'NumericalTool[MatrixPropertyProblem, ArrayResult]'
tool(self, capability: "Literal['pinv']") -> 'NumericalTool[MatrixPropertyProblem, ArrayResult]'
tool(self, capability: "Literal['det']") -> 'NumericalTool[MatrixPropertyProblem, ArrayResult]'
tool(self, capability: "Literal['logm']") -> 'NumericalTool[MatrixPropertyProblem, ArrayResult]'
tool(self, capability: "Literal['sqrtm']") -> 'NumericalTool[MatrixPropertyProblem, ArrayResult]'
tool(self, capability: "Literal['condition_number']") -> 'NumericalTool[MatrixPropertyProblem, ConditionResult]'
tool(self, capability: "Literal['slogdet']") -> 'NumericalTool[MatrixPropertyProblem, LogDetResult]'
tool(self, capability: "Literal['sparse_eig']") -> 'NumericalTool[SparseSpectrumProblem, EigenResult]'
tool(self, capability: "Literal['sparse_svd']") -> 'NumericalTool[SparseSpectrumProblem, SVDResult]'
tool(self, capability: "Literal['linear_operator']") -> 'NumericalTool[OperatorProblem, LinearOperator]'
tool(self, capability: "Literal['fixed_point']") -> 'NumericalTool[FixedPointProblem, ArrayResult]'
tool(self, capability: "Literal['jvp']") -> 'NumericalTool[DerivativeProductProblem, ArrayResult]'
tool(self, capability: "Literal['vjp']") -> 'NumericalTool[DerivativeProductProblem, ArrayResult]'
tool(self, capability: "Literal['nquad']") -> 'NumericalTool[BoxIntegral, ArrayResult]'
tool(self, capability: "Literal['gauss_quad']") -> 'NumericalTool[QuadratureProblem, ArrayResult]'
tool(self, capability: "Literal['quadrature_rule']") -> 'NumericalTool[QuadratureRuleProblem, QuadratureRule]'
tool(self, capability: "Literal['spline']") -> 'NumericalTool[InterpolationProblem, Interpolant]'
tool(self, capability: "Literal['regular_grid']") -> 'NumericalTool[MultivariateInterpolationProblem, MultivariateInterpolant]'
tool(self, capability: "Literal['scattered']") -> 'NumericalTool[MultivariateInterpolationProblem, MultivariateInterpolant]'
tool(self, capability: "Literal['polynomial_approximation']") -> 'NumericalTool[PolynomialProblem, PolynomialResult]'
tool(self, capability: "Literal['solve_ivp']") -> 'NumericalTool[InitialValueProblem, ODEResult]'
tool(self, capability: "Literal['solve_ode']") -> 'NumericalTool[InitialValueProblem, ODEResult]'
tool(self, capability: "Literal['solve_bvp']") -> 'NumericalTool[BoundaryValueProblem, BoundaryValueResult]'
tool(self, capability: "Literal['linspace']") -> 'NumericalTool[GridProblem, FloatArray]'
tool(self, capability: "Literal['logspace']") -> 'NumericalTool[GridProblem, FloatArray]'
tool(self, capability: "Literal['geomspace']") -> 'NumericalTool[GridProblem, FloatArray]'
tool(self, capability: "Literal['arange']") -> 'NumericalTool[GridProblem, FloatArray]'
tool(self, capability: "Literal['meshgrid']") -> 'NumericalTool[MeshProblem, tuple[FloatArray, ...]]'
tool(self, capability: "Literal['cartesian_product']") -> 'NumericalTool[MeshProblem, FloatArray]'
tool(self, capability: "Literal['differentiation_matrix']") -> 'NumericalTool[DifferenceOperatorProblem, csr_matrix | csr_array]'
tool(self, capability: "Literal['partial_derivative_matrix']") -> 'NumericalTool[DifferenceOperatorProblem, csr_matrix | csr_array]'
tool(self, capability: "Literal['laplacian_matrix']") -> 'NumericalTool[LaplacianProblem, csr_matrix | csr_array]'
tool(self, capability: "Literal['isclose']") -> 'NumericalTool[ComparisonProblem, BoolArray]'
tool(self, capability: "Literal['residual']") -> 'NumericalTool[ResidualProblem, ResidualResult]'
tool(self, capability: "Literal['solve']") -> 'NumericalTool[LinearSystem, ArrayResult]'
tool(self, capability: "Literal['eig']") -> 'NumericalTool[Eigenproblem, EigenResult]'
tool(self, capability: "Literal['svd']") -> 'NumericalTool[SVDProblem, SVDResult]'
tool(self, capability: "Literal['expm']") -> 'NumericalTool[MatrixExponential, ArrayResult]'
tool(self, capability: "Literal['quad']") -> 'NumericalTool[QuadratureProblem, ArrayResult]'
tool(self, capability: "Literal['root']") -> 'NumericalTool[RootSystem, ArrayResult]'
tool(self, capability: "Literal['derivative']") -> 'NumericalTool[DifferentiationProblem, ArrayResult]'
tool(self, capability: "Literal['gradient']") -> 'NumericalTool[DifferentiationProblem, ArrayResult]'
tool(self, capability: "Literal['jacobian']") -> 'NumericalTool[DifferentiationProblem, ArrayResult]'
tool(self, capability: "Literal['hessian']") -> 'NumericalTool[DifferentiationProblem, ArrayResult]'
tool(self, capability: "Literal['interpolate']") -> 'NumericalTool[InterpolationProblem, Interpolant]'
tool(self, capability: "Literal['solve_lyapunov']") -> 'NumericalTool[LyapunovEquation, ArrayResult]'
tool(self, capability: "Literal['solve_riccati']") -> 'NumericalTool[RiccatiEquation, ArrayResult]'
tool(self, capability: 'str') -> 'Callable[..., object]'
```

### `Backend.with_tools`

```python
with_tools(self, **replacements) -> 'Backend'
```
