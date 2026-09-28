"""Train-only composition with the row-preserving transforms family."""

from dataclasses import dataclass, field
from typing import Any

from .. import _validation as v
from ..interfaces import FittedModel, fit


@dataclass(frozen=True)
class PipelineAdapter:
    transform: Any = field(repr=False)
    model: FittedModel

    @property
    def input_dimension(self):
        return self.transform.input_dimension

    @property
    def output_shape(self):
        return self.model.output_shape

    @property
    def capabilities(self):
        return self.model.capabilities

    @property
    def supports_offset(self):
        return getattr(self.model.adapter, "supports_offset", False)

    def evaluate(self, operation, features, *, offset=None):
        from ...transforms import checked_transform
        return self.model._evaluate(operation, checked_transform(self.transform, features), offset)


@dataclass(frozen=True)
class Pipeline:
    """Fit transformations and a learner on exactly the supplied training rows.

    Weights apply to the learner; preprocessing uses unweighted training rows.
    Callers needing weighted preprocessing should supply a custom transform.
    """

    transforms: tuple[Any, ...]
    learner: Any

    def __post_init__(self):
        steps = tuple(self.transforms)
        if not steps or any(not callable(getattr(s, "fit", None)) for s in steps):
            raise TypeError("transforms must contain one or more transform fitters")
        if not callable(getattr(self.learner, "fit", None)):
            raise TypeError("learner must implement fit")
        object.__setattr__(self, "transforms", steps)

    def fit(self, features, response, **kwargs):
        from ...transforms import Pipeline as TransformPipeline

        x = v.features(features)
        transform = TransformPipeline(self.transforms).fit(x)
        model = fit(transform.transform(x), response, self.learner, **kwargs)
        return FittedModel(PipelineAdapter(transform, model), model.diagnostics, model.nobs,
                           classes=model.classes, quantiles=model.quantiles, loss_history=model.loss_history,
                           requires_offset=model.requires_offset)
