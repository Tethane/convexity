"""Explicit, immutable capability routing; no global backend state."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, overload

from ..._typing import BoolArray

if TYPE_CHECKING:
    from scipy.sparse import csr_array, csr_matrix
    from scipy.sparse.linalg import LinearOperator

    from .. import (
        ArrayResult,
        BoundaryValueProblem,
        BoundaryValueResult,
        BoxIntegral,
        ComparisonProblem,
        ConditionResult,
        DerivativeProductProblem,
        DifferenceOperatorProblem,
        DifferentiationProblem,
        Eigenproblem,
        EigenResult,
        FactorizationProblem,
        FactorizationResult,
        FixedPointProblem,
        GeneralizedEigenproblem,
        GeneralizedEigenResult,
        GridProblem,
        InitialValueProblem,
        Interpolant,
        InterpolationProblem,
        LaplacianProblem,
        LeastSquaresProblem,
        LeastSquaresResult,
        LinearSystem,
        LogDetResult,
        LyapunovEquation,
        MatrixExponential,
        MatrixPropertyProblem,
        MeshProblem,
        MultivariateInterpolant,
        MultivariateInterpolationProblem,
        ODEResult,
        OperatorProblem,
        PolynomialProblem,
        PolynomialResult,
        Problem,
        QuadratureProblem,
        QuadratureRule,
        QuadratureRuleProblem,
        ResidualProblem,
        ResidualResult,
        RiccatiEquation,
        RootSystem,
        SparseSpectrumProblem,
        SVDProblem,
        SVDResult,
    )
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..._typing import FloatArray
from .. import _validation as v
from .base import NumericalTool, run


@dataclass(frozen=True)
class Backend:
    """Immutable named capability-to-tool mapping. Known routes retain typed results; custom routes require caller narrowing."""

    name: str
    tools: Mapping[str, Callable[..., object]]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("backend name must be nonempty")
        for key, tool in self.tools.items():
            if not isinstance(key, str) or not key:
                raise ValueError("capability names must be nonempty strings")
            v.function(tool, key)
        object.__setattr__(self, "tools", MappingProxyType(dict(self.tools)))

    @overload
    def tool(
        self, capability: Literal["lstsq"]
    ) -> NumericalTool[LeastSquaresProblem, LeastSquaresResult]: ...

    @overload
    def tool(
        self, capability: Literal["generalized_eig"]
    ) -> NumericalTool[GeneralizedEigenproblem, GeneralizedEigenResult]: ...

    @overload
    def tool(
        self, capability: Literal["qr"]
    ) -> NumericalTool[FactorizationProblem, FactorizationResult]: ...

    @overload
    def tool(
        self, capability: Literal["lu"]
    ) -> NumericalTool[FactorizationProblem, FactorizationResult]: ...

    @overload
    def tool(
        self, capability: Literal["cholesky"]
    ) -> NumericalTool[FactorizationProblem, FactorizationResult]: ...

    @overload
    def tool(
        self, capability: Literal["schur"]
    ) -> NumericalTool[FactorizationProblem, FactorizationResult]: ...

    @overload
    def tool(
        self, capability: Literal["matrix_rank"]
    ) -> NumericalTool[MatrixPropertyProblem, ArrayResult]: ...

    @overload
    def tool(
        self, capability: Literal["norm"]
    ) -> NumericalTool[MatrixPropertyProblem, ArrayResult]: ...

    @overload
    def tool(
        self, capability: Literal["pinv"]
    ) -> NumericalTool[MatrixPropertyProblem, ArrayResult]: ...

    @overload
    def tool(
        self, capability: Literal["det"]
    ) -> NumericalTool[MatrixPropertyProblem, ArrayResult]: ...

    @overload
    def tool(
        self, capability: Literal["logm"]
    ) -> NumericalTool[MatrixPropertyProblem, ArrayResult]: ...

    @overload
    def tool(
        self, capability: Literal["sqrtm"]
    ) -> NumericalTool[MatrixPropertyProblem, ArrayResult]: ...

    @overload
    def tool(
        self, capability: Literal["condition_number"]
    ) -> NumericalTool[MatrixPropertyProblem, ConditionResult]: ...

    @overload
    def tool(
        self, capability: Literal["slogdet"]
    ) -> NumericalTool[MatrixPropertyProblem, LogDetResult]: ...

    @overload
    def tool(
        self, capability: Literal["sparse_eig"]
    ) -> NumericalTool[SparseSpectrumProblem, EigenResult]: ...

    @overload
    def tool(
        self, capability: Literal["sparse_svd"]
    ) -> NumericalTool[SparseSpectrumProblem, SVDResult]: ...

    @overload
    def tool(
        self, capability: Literal["linear_operator"]
    ) -> NumericalTool[OperatorProblem, LinearOperator]: ...

    @overload
    def tool(
        self, capability: Literal["fixed_point"]
    ) -> NumericalTool[FixedPointProblem, ArrayResult]: ...

    @overload
    def tool(
        self, capability: Literal["jvp"]
    ) -> NumericalTool[DerivativeProductProblem, ArrayResult]: ...

    @overload
    def tool(
        self, capability: Literal["vjp"]
    ) -> NumericalTool[DerivativeProductProblem, ArrayResult]: ...

    @overload
    def tool(
        self, capability: Literal["nquad"]
    ) -> NumericalTool[BoxIntegral, ArrayResult]: ...

    @overload
    def tool(
        self, capability: Literal["gauss_quad"]
    ) -> NumericalTool[QuadratureProblem, ArrayResult]: ...

    @overload
    def tool(
        self, capability: Literal["quadrature_rule"]
    ) -> NumericalTool[QuadratureRuleProblem, QuadratureRule]: ...

    @overload
    def tool(
        self, capability: Literal["spline"]
    ) -> NumericalTool[InterpolationProblem, Interpolant]: ...

    @overload
    def tool(
        self, capability: Literal["regular_grid"]
    ) -> NumericalTool[MultivariateInterpolationProblem, MultivariateInterpolant]: ...

    @overload
    def tool(
        self, capability: Literal["scattered"]
    ) -> NumericalTool[MultivariateInterpolationProblem, MultivariateInterpolant]: ...

    @overload
    def tool(
        self, capability: Literal["polynomial_approximation"]
    ) -> NumericalTool[PolynomialProblem, PolynomialResult]: ...

    @overload
    def tool(
        self, capability: Literal["solve_ivp"]
    ) -> NumericalTool[InitialValueProblem, ODEResult]: ...

    @overload
    def tool(
        self, capability: Literal["solve_ode"]
    ) -> NumericalTool[InitialValueProblem, ODEResult]: ...

    @overload
    def tool(
        self, capability: Literal["solve_bvp"]
    ) -> NumericalTool[BoundaryValueProblem, BoundaryValueResult]: ...

    @overload
    def tool(
        self, capability: Literal["linspace"]
    ) -> NumericalTool[GridProblem, FloatArray]: ...

    @overload
    def tool(
        self, capability: Literal["logspace"]
    ) -> NumericalTool[GridProblem, FloatArray]: ...

    @overload
    def tool(
        self, capability: Literal["geomspace"]
    ) -> NumericalTool[GridProblem, FloatArray]: ...

    @overload
    def tool(
        self, capability: Literal["arange"]
    ) -> NumericalTool[GridProblem, FloatArray]: ...

    @overload
    def tool(
        self, capability: Literal["meshgrid"]
    ) -> NumericalTool[MeshProblem, tuple[FloatArray, ...]]: ...

    @overload
    def tool(
        self, capability: Literal["cartesian_product"]
    ) -> NumericalTool[MeshProblem, FloatArray]: ...

    @overload
    def tool(
        self, capability: Literal["differentiation_matrix"]
    ) -> NumericalTool[DifferenceOperatorProblem, csr_matrix | csr_array]: ...

    @overload
    def tool(
        self, capability: Literal["partial_derivative_matrix"]
    ) -> NumericalTool[DifferenceOperatorProblem, csr_matrix | csr_array]: ...

    @overload
    def tool(
        self, capability: Literal["laplacian_matrix"]
    ) -> NumericalTool[LaplacianProblem, csr_matrix | csr_array]: ...

    @overload
    def tool(
        self, capability: Literal["isclose"]
    ) -> NumericalTool[ComparisonProblem, BoolArray]: ...

    @overload
    def tool(
        self, capability: Literal["residual"]
    ) -> NumericalTool[ResidualProblem, ResidualResult]: ...

    @overload
    def tool(
        self, capability: Literal["solve"]
    ) -> NumericalTool[LinearSystem, ArrayResult]: ...

    @overload
    def tool(
        self, capability: Literal["eig"]
    ) -> NumericalTool[Eigenproblem, EigenResult]: ...

    @overload
    def tool(
        self, capability: Literal["svd"]
    ) -> NumericalTool[SVDProblem, SVDResult]: ...

    @overload
    def tool(
        self, capability: Literal["expm"]
    ) -> NumericalTool[MatrixExponential, ArrayResult]: ...

    @overload
    def tool(
        self, capability: Literal["quad"]
    ) -> NumericalTool[QuadratureProblem, ArrayResult]: ...

    @overload
    def tool(
        self, capability: Literal["root"]
    ) -> NumericalTool[RootSystem, ArrayResult]: ...

    @overload
    def tool(
        self, capability: Literal["derivative"]
    ) -> NumericalTool[DifferentiationProblem, ArrayResult]: ...

    @overload
    def tool(
        self, capability: Literal["gradient"]
    ) -> NumericalTool[DifferentiationProblem, ArrayResult]: ...

    @overload
    def tool(
        self, capability: Literal["jacobian"]
    ) -> NumericalTool[DifferentiationProblem, ArrayResult]: ...

    @overload
    def tool(
        self, capability: Literal["hessian"]
    ) -> NumericalTool[DifferentiationProblem, ArrayResult]: ...

    @overload
    def tool(
        self, capability: Literal["interpolate"]
    ) -> NumericalTool[InterpolationProblem, Interpolant]: ...

    @overload
    def tool(
        self, capability: Literal["solve_lyapunov"]
    ) -> NumericalTool[LyapunovEquation, ArrayResult]: ...

    @overload
    def tool(
        self, capability: Literal["solve_riccati"]
    ) -> NumericalTool[RiccatiEquation, ArrayResult]: ...

    @overload
    def tool(self, capability: str) -> Callable[..., object]: ...

    def tool(self, capability: str) -> Callable[..., object]:
        if capability not in self.tools:
            raise NotImplementedError(
                f"backend {self.name!r} does not support {capability!r}"
            )
        return self.tools[capability]

    @overload
    def run(
        self,
        capability: Literal["lstsq"],
        problem: LeastSquaresProblem,
        *,
        check_convergence: bool = True,
    ) -> LeastSquaresResult: ...

    @overload
    def run(
        self,
        capability: Literal["generalized_eig"],
        problem: GeneralizedEigenproblem,
        *,
        check_convergence: bool = True,
    ) -> GeneralizedEigenResult: ...

    @overload
    def run(
        self,
        capability: Literal["qr"],
        problem: FactorizationProblem,
        *,
        check_convergence: bool = True,
    ) -> FactorizationResult: ...

    @overload
    def run(
        self,
        capability: Literal["lu"],
        problem: FactorizationProblem,
        *,
        check_convergence: bool = True,
    ) -> FactorizationResult: ...

    @overload
    def run(
        self,
        capability: Literal["cholesky"],
        problem: FactorizationProblem,
        *,
        check_convergence: bool = True,
    ) -> FactorizationResult: ...

    @overload
    def run(
        self,
        capability: Literal["schur"],
        problem: FactorizationProblem,
        *,
        check_convergence: bool = True,
    ) -> FactorizationResult: ...

    @overload
    def run(
        self,
        capability: Literal["matrix_rank"],
        problem: MatrixPropertyProblem,
        *,
        check_convergence: bool = True,
    ) -> ArrayResult: ...

    @overload
    def run(
        self,
        capability: Literal["norm"],
        problem: MatrixPropertyProblem,
        *,
        check_convergence: bool = True,
    ) -> ArrayResult: ...

    @overload
    def run(
        self,
        capability: Literal["pinv"],
        problem: MatrixPropertyProblem,
        *,
        check_convergence: bool = True,
    ) -> ArrayResult: ...

    @overload
    def run(
        self,
        capability: Literal["det"],
        problem: MatrixPropertyProblem,
        *,
        check_convergence: bool = True,
    ) -> ArrayResult: ...

    @overload
    def run(
        self,
        capability: Literal["logm"],
        problem: MatrixPropertyProblem,
        *,
        check_convergence: bool = True,
    ) -> ArrayResult: ...

    @overload
    def run(
        self,
        capability: Literal["sqrtm"],
        problem: MatrixPropertyProblem,
        *,
        check_convergence: bool = True,
    ) -> ArrayResult: ...

    @overload
    def run(
        self,
        capability: Literal["condition_number"],
        problem: MatrixPropertyProblem,
        *,
        check_convergence: bool = True,
    ) -> ConditionResult: ...

    @overload
    def run(
        self,
        capability: Literal["slogdet"],
        problem: MatrixPropertyProblem,
        *,
        check_convergence: bool = True,
    ) -> LogDetResult: ...

    @overload
    def run(
        self,
        capability: Literal["sparse_eig"],
        problem: SparseSpectrumProblem,
        *,
        check_convergence: bool = True,
    ) -> EigenResult: ...

    @overload
    def run(
        self,
        capability: Literal["sparse_svd"],
        problem: SparseSpectrumProblem,
        *,
        check_convergence: bool = True,
    ) -> SVDResult: ...

    @overload
    def run(
        self,
        capability: Literal["linear_operator"],
        problem: OperatorProblem,
        *,
        check_convergence: bool = True,
    ) -> LinearOperator: ...

    @overload
    def run(
        self,
        capability: Literal["fixed_point"],
        problem: FixedPointProblem,
        *,
        check_convergence: bool = True,
    ) -> ArrayResult: ...

    @overload
    def run(
        self,
        capability: Literal["jvp"],
        problem: DerivativeProductProblem,
        *,
        check_convergence: bool = True,
    ) -> ArrayResult: ...

    @overload
    def run(
        self,
        capability: Literal["vjp"],
        problem: DerivativeProductProblem,
        *,
        check_convergence: bool = True,
    ) -> ArrayResult: ...

    @overload
    def run(
        self,
        capability: Literal["nquad"],
        problem: BoxIntegral,
        *,
        check_convergence: bool = True,
    ) -> ArrayResult: ...

    @overload
    def run(
        self,
        capability: Literal["gauss_quad"],
        problem: QuadratureProblem,
        *,
        check_convergence: bool = True,
    ) -> ArrayResult: ...

    @overload
    def run(
        self,
        capability: Literal["quadrature_rule"],
        problem: QuadratureRuleProblem,
        *,
        check_convergence: bool = True,
    ) -> QuadratureRule: ...

    @overload
    def run(
        self,
        capability: Literal["spline"],
        problem: InterpolationProblem,
        *,
        check_convergence: bool = True,
    ) -> Interpolant: ...

    @overload
    def run(
        self,
        capability: Literal["regular_grid"],
        problem: MultivariateInterpolationProblem,
        *,
        check_convergence: bool = True,
    ) -> MultivariateInterpolant: ...

    @overload
    def run(
        self,
        capability: Literal["scattered"],
        problem: MultivariateInterpolationProblem,
        *,
        check_convergence: bool = True,
    ) -> MultivariateInterpolant: ...

    @overload
    def run(
        self,
        capability: Literal["polynomial_approximation"],
        problem: PolynomialProblem,
        *,
        check_convergence: bool = True,
    ) -> PolynomialResult: ...

    @overload
    def run(
        self,
        capability: Literal["solve_ivp"],
        problem: InitialValueProblem,
        *,
        check_convergence: bool = True,
    ) -> ODEResult: ...

    @overload
    def run(
        self,
        capability: Literal["solve_ode"],
        problem: InitialValueProblem,
        *,
        check_convergence: bool = True,
    ) -> ODEResult: ...

    @overload
    def run(
        self,
        capability: Literal["solve_bvp"],
        problem: BoundaryValueProblem,
        *,
        check_convergence: bool = True,
    ) -> BoundaryValueResult: ...

    @overload
    def run(
        self,
        capability: Literal["linspace"],
        problem: GridProblem,
        *,
        check_convergence: bool = True,
    ) -> FloatArray: ...

    @overload
    def run(
        self,
        capability: Literal["logspace"],
        problem: GridProblem,
        *,
        check_convergence: bool = True,
    ) -> FloatArray: ...

    @overload
    def run(
        self,
        capability: Literal["geomspace"],
        problem: GridProblem,
        *,
        check_convergence: bool = True,
    ) -> FloatArray: ...

    @overload
    def run(
        self,
        capability: Literal["arange"],
        problem: GridProblem,
        *,
        check_convergence: bool = True,
    ) -> FloatArray: ...

    @overload
    def run(
        self,
        capability: Literal["meshgrid"],
        problem: MeshProblem,
        *,
        check_convergence: bool = True,
    ) -> tuple[FloatArray, ...]: ...

    @overload
    def run(
        self,
        capability: Literal["cartesian_product"],
        problem: MeshProblem,
        *,
        check_convergence: bool = True,
    ) -> FloatArray: ...

    @overload
    def run(
        self,
        capability: Literal["differentiation_matrix"],
        problem: DifferenceOperatorProblem,
        *,
        check_convergence: bool = True,
    ) -> csr_matrix | csr_array: ...

    @overload
    def run(
        self,
        capability: Literal["partial_derivative_matrix"],
        problem: DifferenceOperatorProblem,
        *,
        check_convergence: bool = True,
    ) -> csr_matrix | csr_array: ...

    @overload
    def run(
        self,
        capability: Literal["laplacian_matrix"],
        problem: LaplacianProblem,
        *,
        check_convergence: bool = True,
    ) -> csr_matrix | csr_array: ...

    @overload
    def run(
        self,
        capability: Literal["isclose"],
        problem: ComparisonProblem,
        *,
        check_convergence: bool = True,
    ) -> BoolArray: ...

    @overload
    def run(
        self,
        capability: Literal["residual"],
        problem: ResidualProblem,
        *,
        check_convergence: bool = True,
    ) -> ResidualResult: ...

    @overload
    def run(
        self,
        capability: Literal["solve"],
        problem: LinearSystem,
        *,
        check_convergence: bool = True,
    ) -> ArrayResult: ...

    @overload
    def run(
        self,
        capability: Literal["eig"],
        problem: Eigenproblem,
        *,
        check_convergence: bool = True,
    ) -> EigenResult: ...

    @overload
    def run(
        self,
        capability: Literal["svd"],
        problem: SVDProblem,
        *,
        check_convergence: bool = True,
    ) -> SVDResult: ...

    @overload
    def run(
        self,
        capability: Literal["expm"],
        problem: MatrixExponential,
        *,
        check_convergence: bool = True,
    ) -> ArrayResult: ...

    @overload
    def run(
        self,
        capability: Literal["quad"],
        problem: QuadratureProblem,
        *,
        check_convergence: bool = True,
    ) -> ArrayResult: ...

    @overload
    def run(
        self,
        capability: Literal["root"],
        problem: RootSystem,
        *,
        check_convergence: bool = True,
    ) -> ArrayResult: ...

    @overload
    def run(
        self,
        capability: Literal["derivative"],
        problem: DifferentiationProblem,
        *,
        check_convergence: bool = True,
    ) -> ArrayResult: ...

    @overload
    def run(
        self,
        capability: Literal["gradient"],
        problem: DifferentiationProblem,
        *,
        check_convergence: bool = True,
    ) -> ArrayResult: ...

    @overload
    def run(
        self,
        capability: Literal["jacobian"],
        problem: DifferentiationProblem,
        *,
        check_convergence: bool = True,
    ) -> ArrayResult: ...

    @overload
    def run(
        self,
        capability: Literal["hessian"],
        problem: DifferentiationProblem,
        *,
        check_convergence: bool = True,
    ) -> ArrayResult: ...

    @overload
    def run(
        self,
        capability: Literal["interpolate"],
        problem: InterpolationProblem,
        *,
        check_convergence: bool = True,
    ) -> Interpolant: ...

    @overload
    def run(
        self,
        capability: Literal["solve_lyapunov"],
        problem: LyapunovEquation,
        *,
        check_convergence: bool = True,
    ) -> ArrayResult: ...

    @overload
    def run(
        self,
        capability: Literal["solve_riccati"],
        problem: RiccatiEquation,
        *,
        check_convergence: bool = True,
    ) -> ArrayResult: ...

    @overload
    def run(
        self, capability: str, problem: Problem, *, check_convergence: bool = True
    ) -> object: ...

    def run(
        self, capability: str, problem: Problem, *, check_convergence: bool = True
    ) -> object:
        return run(problem, self.tool(capability), check_convergence=check_convergence)

    def with_tools(self, **replacements) -> Backend:
        return Backend(self.name, {**self.tools, **replacements})
