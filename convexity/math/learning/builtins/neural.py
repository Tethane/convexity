"""Dense neural mean/probability learners and user-supplied neural functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from torch.nn import Module


from dataclasses import dataclass, field

from .. import _validation as v
from .._backends import torch
from ..interfaces import Diagnostics, FittedModel
from ..interfaces.base import checked_convergence


@dataclass(frozen=True)
class NeuralNetwork:
    """Full-batch CPU float64 Adam; convergence checks the gradient infinity norm."""

    task: str = "regression"
    hidden_layers: tuple[int, ...] = (32, 32)
    activation: str = "tanh"
    learning_rate: float = 0.01
    maxiter: int = 2000
    tolerance: float = 1e-5
    seed: int = 0
    check_convergence: bool = True

    def __post_init__(self) -> None:
        v.choice(self.task, ("regression", "classification"), "task")
        widths = tuple(v.integer(n, "hidden width", 1) for n in self.hidden_layers)
        object.__setattr__(self, "hidden_layers", widths)
        v.choice(self.activation, ("tanh", "relu", "sigmoid"), "activation")
        object.__setattr__(
            self,
            "learning_rate",
            v.scalar(self.learning_rate, "learning_rate", minimum=0, strict=True),
        )
        object.__setattr__(
            self,
            "tolerance",
            v.scalar(self.tolerance, "tolerance", minimum=0, strict=True),
        )
        v.integer(self.maxiter, "maxiter", 1)
        v.seed(self.seed)
        v.boolean(self.check_convergence, "check_convergence")

    def fit(
        self,
        features: ArrayLike,
        response: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
    ) -> FittedModel:
        x, y, w = v.training(
            features,
            response,
            sample_weight,
            classification=self.task == "classification",
        )
        return checked_convergence(torch.neural(self, x, y, w), self.check_convergence)


@dataclass(frozen=True)
class NeuralRegressor(NeuralNetwork):
    """Feed-forward mean regressor with explicit architecture, training controls and random seed."""

    task: str = field(default="regression", init=False)


@dataclass(frozen=True)
class NeuralClassifier(NeuralNetwork):
    """Feed-forward classifier with explicit architecture, training controls and random seed."""

    task: str = field(default="classification", init=False)


def neural_function(
    module: Module,
    input_dimension: int,
    *,
    output_shape: tuple[int, ...] = (),
    classes: ArrayLike | None = None,
) -> FittedModel:
    """Snapshot a torch module as a deterministic CPU float64 predictor.

    Regression modules return (n,m), m=1 for scalar outputs. Classification
    modules return (n,c) logits in the explicitly supplied class order.
    """
    dimension = v.integer(input_dimension, "input_dimension", 1)
    labels = None if classes is None else v.labels(classes)
    shape = tuple(output_shape)
    if len(shape) > 1 or (labels is not None and shape):
        raise ValueError("output_shape must be () or (m,); classifiers require ()")
    for size in shape:
        v.integer(size, "output dimension", 1)
    caps = frozenset({"mean" if labels is None else "probability"})
    adapter = torch.TorchAdapter(
        module, dimension, shape, caps, None if labels is None else len(labels)
    )
    return FittedModel(
        adapter,
        Diagnostics("torch.nn.Module", message="Supplied function; no fit performed"),
        classes=labels,
    )
