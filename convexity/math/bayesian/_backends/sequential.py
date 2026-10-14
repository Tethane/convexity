"""Compose the particles backend with conditional state-space models."""

import numpy as np
from .. import _validation as v
from ..interfaces.sequential import JointFilterResult, StateParameterModel


class _Initial:
    def __init__(self, model):
        self.model = model

    def sample(self, size, *, rng):
        theta = self.model.prior.sample(size, rng=rng)
        states = [
            self.model.conditional_model(t).initial_distribution().sample(rng=rng)
            for t in theta
        ]
        return np.column_stack([theta, states])


class _Transition:
    def __init__(self, model):
        self.model = model

    def sample(self, particles, step, *, rng):
        p = self.model.prior.event_shape[0]
        out = np.array(particles, copy=True)
        for i, row in enumerate(particles):
            model = self.model.conditional_model(row[:p])
            values = v.array(
                model.transition.sample(row[None, p:], step, rng=rng),
                "transition samples",
                ndim=2,
            )
            if values.shape != (1, model.state_dimension):
                raise ValueError(
                    "transition samples must have shape (1,state_dimension)"
                )
            out[i, p:] = values[0]
        return out


class _Observation:
    def __init__(self, model):
        self.model = model

    def log_density(self, observation, particles, step):
        p = self.model.prior.event_shape[0]
        result = []
        for row in particles:
            model = self.model.conditional_model(row[:p])
            logp = model.observation.log_density(observation, row[None, p:], step)
            result.append(float(v.logs(logp, "observation log density", (1,))[0]))
        return np.asarray(result)


class _Augmented:
    """Internal sampling model; no false density claim for mixed parameter/state laws."""

    def __init__(self, model):
        self.model = model
        self.transition = _Transition(model)
        self.observation = _Observation(model)
        self.state_dimension = (
            model.prior.event_shape[0] + model._prototype.state_dimension
        )
        self.observation_dimension = model._prototype.observation_dimension
        self.state_names = tuple(
            f"parameter:{n}" for n in model.parameter_names
        ) + tuple(f"state:{n}" for n in model._prototype.state_names)

    def initial_distribution(self):
        return _Initial(self.model)

    def steps(self, series):
        return self.model._prototype.steps(series)


def run(model, observations, specification, *, rng):
    from ...state_space._backends.particle import run as particle_run

    if not isinstance(model, StateParameterModel):
        raise TypeError("model must be StateParameterModel")
    assert model.parameter_names is not None
    result = particle_run(_Augmented(model), observations, specification, rng=rng)
    return JointFilterResult(result, model.parameter_names)
