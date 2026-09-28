import numpy as np
from ..interfaces.base import ArrayResult, Diagnostics
from ..interfaces.calculus import DifferentiationProblem
from ..interfaces.products import DerivativeCheck
from . import calculus


def finite_product(problem, *, options):
    result = calculus.finite_difference(DifferentiationProblem(problem.function, problem.x, "jacobian"), **options)
    jac = result.value
    vector = np.asarray(problem.vector)
    if problem.kind == "vjp" and vector.shape != (jac.shape[0],):
        raise ValueError("VJP vector must have the output shape")
    value = jac @ vector if problem.kind == "jvp" else vector @ jac
    # Propagate componentwise absolute estimates through the linear product.
    error = np.asarray(result.diagnostics.error_estimate)
    error = error @ abs(vector) if problem.kind == "jvp" else abs(vector) @ error
    report = result.diagnostics
    return ArrayResult(value, Diagnostics(report.converged, report.message, report.iterations,
                                          report.evaluations, error_estimate=error))


def autodiff_product(problem, *, device):
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise ImportError("PyTorch autodiff requires: pip install 'convexity[autodiff]'") from exc
    x = torch.tensor(np.array(problem.x), dtype=torch.float64, device=device)
    vector = torch.tensor(np.array(problem.vector), dtype=torch.float64, device=device)
    evaluations = 0

    def function(point):
        nonlocal evaluations
        evaluations += 1
        output = problem.function(point)
        if not isinstance(output, torch.Tensor) or not output.is_floating_point() or output.ndim != 1 or not output.numel():
            raise ValueError("product callback must return a real vector torch.Tensor")
        if not bool(torch.isfinite(output).all()):
            raise ValueError("product callback output must be finite")
        if problem.kind == "vjp" and output.shape != vector.shape:
            raise ValueError("VJP vector must have the output shape")
        return output

    operation = torch.autograd.functional.jvp if problem.kind == "jvp" else torch.autograd.functional.vjp
    _, value = operation(function, x, v=vector, strict=False, create_graph=False)
    assert isinstance(value, torch.Tensor)
    return ArrayResult(value.detach().cpu().numpy(), Diagnostics(evaluations=evaluations))


def compare(actual, reference, *, atol, rtol):
    error = np.abs(actual-reference)
    scale = atol + rtol*np.abs(reference)
    scaled = np.divide(error, scale, out=np.zeros_like(error), where=scale != 0)
    # A nonzero error with zero permitted error cannot pass; use a finite marker.
    scaled = np.where((scale == 0) & (error != 0), np.finfo(float).max, scaled)
    return DerivativeCheck(bool(np.all(error <= scale)), actual, reference, error, scaled)
