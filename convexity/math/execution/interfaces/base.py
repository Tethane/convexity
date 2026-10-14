from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from ... import _decision as v
from ..._typing import FloatArray


@dataclass(frozen=True)
class LiquidationResult:
    """Inventory/trade schedule and modeled liquidation objective under supplied impact/risk assumptions."""

    trades: FloatArray
    inventory: FloatArray
    expected_cost: float
    inventory_penalty: float
    objective: float
    solver: str
    max_violation: float

    def __post_init__(self) -> None:
        for name in ("trades", "inventory"):
            object.__setattr__(self, name, v.array(getattr(self, name), name, ndim=2))
        if self.inventory.shape != (self.trades.shape[0] + 1, self.trades.shape[1]):
            raise ValueError("inventory must include initial and terminal states")
        for name in (
            "expected_cost",
            "inventory_penalty",
            "objective",
            "max_violation",
        ):
            v.scalar(getattr(self, name), name)

    def __init__(
        self,
        trades: ArrayLike,
        inventory: ArrayLike,
        expected_cost: float,
        inventory_penalty: float,
        objective: float,
        solver: str,
        max_violation: float,
    ) -> None:
        object.__setattr__(self, "trades", trades)
        object.__setattr__(self, "inventory", inventory)
        object.__setattr__(self, "expected_cost", expected_cost)
        object.__setattr__(self, "inventory_penalty", inventory_penalty)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "max_violation", max_violation)
        self.__post_init__()


@dataclass(frozen=True)
class LogisticFit:
    """Fitted logistic response parameters and diagnostics for the declared explanatory coordinates."""

    coefficients: FloatArray
    covariance: FloatArray | None
    log_likelihood: float
    iterations: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "coefficients", v.array(self.coefficients, "coefficients", ndim=1)
        )
        if self.covariance is not None:
            object.__setattr__(self, "covariance", v.psd(self.covariance, "covariance"))
        v.scalar(self.log_likelihood, "log likelihood")

    def probability(self, features: ArrayLike) -> FloatArray:
        from .._backends.statistical import predict

        x = v.array(features, "features", ndim=2)
        if x.shape[1] != len(self.coefficients):
            raise ValueError(
                "features must align with coefficients, including any intercept"
            )
        return v.array(predict(x, self.coefficients), "probability")

    def __init__(
        self,
        coefficients: ArrayLike,
        covariance: ArrayLike | None,
        log_likelihood: float,
        iterations: int,
    ) -> None:
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "log_likelihood", log_likelihood)
        object.__setattr__(self, "iterations", iterations)
        self.__post_init__()


class CostModel(Protocol):
    """Evaluate modeled execution costs for the supplied trade/state coordinates."""

    def __call__(self, trade: ArrayLike) -> float: ...


class FillLaw(Protocol):
    """Conditional execution-fill law; distinguish fill probability from price and cost expectations."""

    def sample(
        self, quantities: ArrayLike, *, draws: int, rng: np.random.Generator
    ) -> FloatArray: ...
