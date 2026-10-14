"""Public schemas and callable protocol for user-written numerical tools."""

from .base import (
    ArrayResult,
    ConvergenceError,
    Diagnostics,
    NumericalTool,
    Problem,
    run,
)
from .calculus import DifferentiationKind, DifferentiationProblem, QuadratureProblem
from .interpolation import Interpolant, InterpolationProblem
from .linalg import (
    Eigenproblem,
    EigenResult,
    LinearSystem,
    MatrixExponential,
    SVDProblem,
    SVDResult,
)
from .matrix_equations import LyapunovEquation, RiccatiEquation, TimeDomain
from .roots import RootSystem, ScalarRootProblem


from .linear_algebra import (
    LeastSquaresProblem,
    LeastSquaresResult,
    GeneralizedEigenproblem,
    GeneralizedEigenResult,
    FactorizationProblem,
    FactorizationResult,
    MatrixPropertyProblem,
    ConditionResult,
    LogDetResult,
)

from .sparse import (
    OperatorProblem,
    SparseSpectrumProblem,
)

from .fixed_point import (
    FixedPointProblem,
)

from .products import (
    DerivativeProductProblem,
    DerivativeCheck,
)

from .integration import (
    BoxIntegral,
    QuadratureRuleProblem,
    QuadratureRule,
)

from .ode import (
    ODEEvent,
    InitialValueProblem,
    DenseSolution,
    EventResult,
    ODEResult,
    BoundaryValueProblem,
    BoundaryValueResult,
)

from .construction import (
    GridProblem,
    MeshProblem,
    DifferenceOperatorProblem,
    LaplacianProblem,
)

from .approximation import (
    MultivariateInterpolationProblem,
    MultivariateInterpolant,
    PolynomialProblem,
    PolynomialResult,
)

from .accuracy import (
    Tolerances,
    ComparisonProblem,
    MachinePrecision,
    ResidualProblem,
    ResidualResult,
)

from .backend import (
    Backend,
)

__all__ = [
    "ArrayResult",
    "ConvergenceError",
    "Diagnostics",
    "NumericalTool",
    "Problem",
    "run",
    "DifferentiationKind",
    "DifferentiationProblem",
    "QuadratureProblem",
    "Interpolant",
    "InterpolationProblem",
    "Eigenproblem",
    "EigenResult",
    "LinearSystem",
    "MatrixExponential",
    "SVDProblem",
    "SVDResult",
    "LyapunovEquation",
    "RiccatiEquation",
    "TimeDomain",
    "RootSystem",
    "ScalarRootProblem",
    "LeastSquaresProblem",
    "LeastSquaresResult",
    "GeneralizedEigenproblem",
    "GeneralizedEigenResult",
    "FactorizationProblem",
    "FactorizationResult",
    "MatrixPropertyProblem",
    "ConditionResult",
    "LogDetResult",
    "OperatorProblem",
    "SparseSpectrumProblem",
    "FixedPointProblem",
    "DerivativeProductProblem",
    "DerivativeCheck",
    "BoxIntegral",
    "QuadratureRuleProblem",
    "QuadratureRule",
    "ODEEvent",
    "InitialValueProblem",
    "DenseSolution",
    "EventResult",
    "ODEResult",
    "BoundaryValueProblem",
    "BoundaryValueResult",
    "GridProblem",
    "MeshProblem",
    "DifferenceOperatorProblem",
    "LaplacianProblem",
    "MultivariateInterpolationProblem",
    "MultivariateInterpolant",
    "PolynomialProblem",
    "PolynomialResult",
    "Tolerances",
    "ComparisonProblem",
    "MachinePrecision",
    "ResidualProblem",
    "ResidualResult",
    "Backend",
]
