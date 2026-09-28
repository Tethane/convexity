"""Optional CPU float64 neural learning and isolated module evaluation."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..interfaces import Diagnostics, FittedModel


def torch_module():
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise ImportError("Neural learning requires: pip install 'convexity[neural]'") from exc
    return torch


@dataclass(frozen=True)
class TorchAdapter:
    _module: Any = field(repr=False)
    input_dimension: int
    output_shape: tuple[int, ...]
    capabilities: frozenset[str]
    n_classes: int | None = None

    def __post_init__(self):
        torch = torch_module()
        if not isinstance(self._module, torch.nn.Module):
            raise TypeError("module must be a torch.nn.Module")
        model = deepcopy(self._module).to(device="cpu", dtype=torch.float64).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        object.__setattr__(self, "_module", model)

    def evaluate(self, operation, features, *, offset=None):
        torch = torch_module()
        with torch.no_grad():
            values = self._module(torch.tensor(np.array(features), dtype=torch.float64, device="cpu"))
            width = self.n_classes if operation == "probability" else (self.output_shape[0] if self.output_shape else 1)
            if not isinstance(values, torch.Tensor) or tuple(values.shape) != (len(features), width):
                raise ValueError("neural module must return (n, outputs), including (n,1) for scalar regression")
            if not bool(torch.isfinite(values).all()):
                raise ValueError("neural module output must be finite")
            if operation == "probability":
                values = torch.softmax(values, dim=-1)
            else:
                values = values.reshape((len(features),) + self.output_shape)
            return values.detach().cpu().numpy()


def neural(spec, x, y, w):
    torch = torch_module()
    classification = spec.task == "classification"
    classes, encoded = np.unique(y, return_inverse=True) if classification else (None, None)
    outputs = len(classes) if classes is not None else (y.shape[1] if y.ndim == 2 else 1)
    widths = (x.shape[1],) + spec.hidden_layers + (outputs,)
    activation = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU, "sigmoid": torch.nn.Sigmoid}[spec.activation]
    # Initialize only the CPU generator and restore it, without touching GPU RNGs.
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device="cpu").manual_seed(spec.seed)
        torch.set_rng_state(generator.get_state())
        layers = []
        for i, (a, b) in enumerate(zip(widths, widths[1:])):
            layers.append(torch.nn.Linear(a, b, dtype=torch.float64, device="cpu"))
            if i < len(widths) - 2:
                layers.append(activation())
        model = torch.nn.Sequential(*layers)
    inputs = torch.tensor(np.array(x), dtype=torch.float64, device="cpu")
    target = (torch.tensor(encoded, dtype=torch.long, device="cpu") if classification else
              torch.tensor(np.array(y).reshape(len(x), outputs), dtype=torch.float64, device="cpu"))
    weight = torch.tensor(np.ones(len(x)) if w is None else np.array(w), dtype=torch.float64, device="cpu")
    weight = weight / weight.sum()
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.learning_rate)
    history, converged = [], False
    iteration = 0
    for iteration in range(spec.maxiter + 1):
        optimizer.zero_grad()
        prediction = model(inputs)
        losses = (torch.nn.functional.cross_entropy(prediction, target, reduction="none") if classification else
                  ((prediction - target) ** 2).mean(dim=-1) / 2)
        loss = torch.sum(weight * losses)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("neural training produced a nonfinite loss")
        loss.backward()
        norms = []
        for parameter in model.parameters():
            if parameter.grad is None:
                raise RuntimeError("neural parameter has no gradient")
            norms.append(float(parameter.grad.detach().abs().max()))
        if not np.isfinite(norms).all():
            raise FloatingPointError("neural training produced nonfinite gradients")
        gradient = max(norms)
        history.append(float(loss.detach()))
        if gradient <= spec.tolerance:
            converged = True
            break
        if iteration < spec.maxiter:
            optimizer.step()
    message = (f"gradient infinity norm {gradient:.6g} <= {spec.tolerance:g}" if converged else
               f"Adam iteration limit reached; gradient infinity norm {gradient:.6g} > {spec.tolerance:g}")
    caps = frozenset({"probability" if classification else "mean"})
    adapter = TorchAdapter(model, x.shape[1], y.shape[1:], caps, None if classes is None else len(classes))
    return FittedModel(adapter, Diagnostics("torch.Adam", converged, message, iteration), len(x),
                       classes=classes, loss_history=history)
