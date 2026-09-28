"""Bounded tree and ensemble catalog for means and class probabilities."""

from dataclasses import dataclass, field

from .. import _validation as v
from .._backends import sklearn


@dataclass(frozen=True)
class Tree:
    task: str = "regression"
    kind: str = "decision_tree"
    max_depth: int | None = None
    min_samples_leaf: int = 1
    n_estimators: int = 100
    learning_rate: float = .1
    seed: int = 0

    def __post_init__(self):
        v.choice(self.task, ("regression", "classification"), "task")
        v.choice(self.kind, ("decision_tree", "random_forest", "extra_trees", "gradient_boosting"), "kind")
        if self.max_depth is not None:
            v.integer(self.max_depth, "max_depth", 1)
        v.integer(self.min_samples_leaf, "min_samples_leaf", 1)
        v.integer(self.n_estimators, "n_estimators", 1)
        object.__setattr__(self, "learning_rate", v.scalar(self.learning_rate, "learning_rate", minimum=0, strict=True))
        v.seed(self.seed)

    def fit(self, features, response, *, sample_weight=None):
        x, y, w = v.training(features, response, sample_weight,
                              classification=self.task == "classification", multi=self.kind != "gradient_boosting")
        return sklearn.trees(self, x, y, w)


@dataclass(frozen=True)
class DecisionTree(Tree):
    kind: str = field(default="decision_tree", init=False)


@dataclass(frozen=True)
class RandomForest(Tree):
    kind: str = field(default="random_forest", init=False)


@dataclass(frozen=True)
class ExtraTrees(Tree):
    kind: str = field(default="extra_trees", init=False)


@dataclass(frozen=True)
class GradientBoosting(Tree):
    kind: str = field(default="gradient_boosting", init=False)
    max_depth: int | None = 3
