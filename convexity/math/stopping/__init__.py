"""Financial mathematics: stopping."""

from . import interfaces, builtins
from .interfaces import StoppingProblem, StoppingResult, StoppingSolver
from .builtins import BackwardInduction, ExercisePolicy, least_squares_policy

from .interfaces import PolicyEvaluation

__all__ = [
    "PolicyEvaluation",
    "interfaces",
    "builtins",
    "StoppingProblem",
    "StoppingResult",
    "StoppingSolver",
    "BackwardInduction",
    "ExercisePolicy",
    "least_squares_policy",
]

from .builtins.solvers import ContinuationRegressor

__all__.append("ContinuationRegressor")
