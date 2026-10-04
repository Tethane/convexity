"""Execution mathematics on supplied arrays and models."""

from . import interfaces, builtins
from .interfaces import LiquidationResult, LogisticFit, CostModel, FillLaw
from .builtins import (
    QuadraticCost,
    BinomialFills,
    assignment,
    fit_assignment,
    liquidate,
)

__all__ = [
    "interfaces",
    "builtins",
    "LiquidationResult",
    "LogisticFit",
    "CostModel",
    "FillLaw",
    "QuadraticCost",
    "BinomialFills",
    "assignment",
    "fit_assignment",
    "liquidate",
]
