"""Optional PyTorch integration. Importing numerical never imports torch."""

import numpy as np

from ..interfaces import ArrayResult, Diagnostics, DifferentiationProblem


def differentiate(problem: DifferentiationProblem, *, device: str) -> ArrayResult:
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise ImportError("PyTorch autodiff requires: pip install 'convexity[autodiff]'") from exc

    point = torch.tensor(np.array(problem.x), dtype=torch.float64, device=device)
    evaluations = 0
    output_shape = None

    def evaluate(x):
        nonlocal evaluations, output_shape
        evaluations += 1
        output = problem.function(x)
        if not isinstance(output, torch.Tensor) or not output.is_floating_point():
            raise ValueError("autodiff function must return a real floating-point torch.Tensor")
        expected_ndim = 1 if problem.kind == "jacobian" else 0
        if output.ndim != expected_ndim or output.numel() == 0:
            raise ValueError(f"autodiff function must return a tensor with {expected_ndim} dimensions")
        if not bool(torch.isfinite(output).all()):
            raise ValueError("autodiff function output must be finite")
        if output_shape is not None and output.shape != output_shape:
            raise ValueError("function output shape changed during differentiation")
        output_shape = output.shape
        return output

    operation = torch.autograd.functional.hessian if problem.kind == "hessian" else torch.autograd.functional.jacobian
    value = operation(evaluate, point, create_graph=False, strict=False)
    assert isinstance(value, torch.Tensor)
    return ArrayResult(value.detach().cpu().numpy(), Diagnostics(evaluations=evaluations))
