"""PyMC inference and ArviZ chain diagnostics; imports remain optional."""

import numpy as np
from typing import Any
from .. import _validation as v
from ..interfaces import (
    BayesianProblem,
    InferenceDiagnostics,
    PosteriorSamples,
    PyMCModel,
)


def _backend():
    try:
        import pymc
    except ImportError as exc:
        raise ImportError("PyMC inference requires convexity[bayesian]") from exc
    return pymc


def _seed(rng):
    return int(v.generator(rng).integers(0, 2**30))


def _build(specification):
    pm = _backend()
    if isinstance(specification, PyMCModel):
        model = specification.builder()
        if not isinstance(model, pm.Model):
            raise TypeError("PyMCModel.builder must return pymc.Model")
        if not model.free_RVs:
            raise ValueError("model must contain free random variables")
        return model
    if not isinstance(specification, BayesianProblem):
        raise TypeError("model must be PyMCModel or BayesianProblem")
    if specification.initial is None:
        raise ValueError(
            "numerical BayesianProblem requires an explicit initial point for MCMC"
        )
    from pytensor.tensor.type import dvector, dscalar
    from pytensor.graph.op import Op

    class NumericalPosterior(Op):
        itypes = [dvector]
        otypes = [dscalar]

        def perform(self, node, inputs, output_storage):
            output_storage[0][0] = np.asarray(
                specification.log_posterior(inputs[0]), dtype="float64"
            )

    with pm.Model() as model:
        # Flat supplies no extra density. All normalized prior and likelihood
        # terms come from the user's complete numerical posterior callback.
        theta = pm.Flat(
            "_convexity_theta",
            shape=specification.prior.event_shape,
            initval=np.asarray(specification.initial),
            dtype="float64",
        )
        potential: Any = NumericalPosterior()(theta)
        pm.Potential("_convexity_logp", potential)
    return model


def _samples(trace, specification=None):
    if isinstance(specification, BayesianProblem):
        assert specification.names is not None
        theta = np.asarray(trace.posterior["_convexity_theta"])
        return {name: theta[..., i] for i, name in enumerate(specification.names)}
    return {
        name: np.asarray(value) for name, value in trace.posterior.data_vars.items()
    }


def _diagnostics(trace, samples, method):
    import arviz as az

    data = az.from_dict(posterior=samples)
    shape = next(iter(samples.values())).shape

    def metric(function, **kwargs):
        return {n: np.asarray(a) for n, a in function(data, **kwargs).data_vars.items()}

    # Rank diagnostics require enough draws; R-hat also requires multiple chains.
    rhat = metric(az.rhat) if shape[0] >= 2 and shape[1] >= 4 else None
    bulk = metric(az.ess, method="bulk") if shape[1] >= 4 else None
    tail = metric(az.ess, method="tail") if shape[1] >= 4 else None
    stats = getattr(trace, "sample_stats", {})
    divergences = (
        int(np.asarray(stats["diverging"]).sum()) if "diverging" in stats else None
    )
    return InferenceDiagnostics(
        method,
        "pymc",
        rhat,
        bulk,
        tail,
        divergences,
        note="Inspect R-hat, ESS, and divergences; completion alone does not establish convergence.",
    )


def sample(specification, options, *, rng):
    v.generator(rng)
    if isinstance(specification, BayesianProblem) and options.method == "nuts":
        raise ValueError(
            "NUTS requires a symbolic PyMCModel; numerical callbacks use Metropolis"
        )
    if (
        isinstance(specification, BayesianProblem)
        and specification.prior.measure != "continuous"
    ):
        raise ValueError(
            "numerical Metropolis requires a continuous prior; use PyMCModel for discrete parameters"
        )
    pm = _backend()
    model = _build(specification)
    method = (
        "metropolis" if isinstance(specification, BayesianProblem) else options.method
    )
    with model:
        if method == "metropolis":
            step = pm.Metropolis()
        elif method == "nuts":
            step = pm.NUTS(target_accept=options.target_accept)
        else:
            step = None
        kwargs = (
            {"target_accept": options.target_accept}
            if method == "auto" and model.continuous_value_vars
            else {}
        )
        trace = pm.sample(
            draws=options.draws,
            tune=options.tune,
            chains=options.chains,
            cores=options.cores,
            step=step,
            random_seed=_seed(rng),
            progressbar=options.progressbar,
            return_inferencedata=True,
            discard_tuned_samples=True,
            compute_convergence_checks=False,
            **kwargs,
        )
    samples = _samples(trace, specification)
    diagnostics = _diagnostics(trace, samples, method)
    # A numerical Potential has no forward observation graph. Generic conditional
    # law prediction is available separately; don't expose misleading PyMC draws.
    native = isinstance(specification, PyMCModel)
    return PosteriorSamples(
        samples, diagnostics, model if native else None, trace if native else None
    )


def variational(specification, options, *, rng):
    v.generator(rng)
    if not isinstance(specification, PyMCModel):
        raise TypeError("variational inference requires a differentiable PyMCModel")
    pm = _backend()
    model = _build(specification)
    if model.discrete_value_vars:
        raise ValueError("ADVI requires continuous latent variables")
    with model:
        approximation = pm.fit(
            n=options.iterations,
            method=options.method,
            random_seed=_seed(rng),
            obj_optimizer=pm.adam(learning_rate=options.learning_rate),
            progressbar=options.progressbar,
        )
        trace = approximation.sample(
            draws=options.draws, random_seed=_seed(rng), return_inferencedata=True
        )
    diagnostics = InferenceDiagnostics(
        options.method,
        "pymc",
        loss=approximation.hist,
        note="Variational approximation; loss is negative ELBO. No MCMC R-hat or ESS is implied.",
    )
    return PosteriorSamples(_samples(trace), diagnostics, model, trace)


def predict(result, *, names, model, rng):
    v.generator(rng)
    if result._trace is None or result._model is None:
        raise ValueError("PyMC prediction requires a native PyMCModel inference result")
    pm = _backend()
    target = result._model if model is None else _build(model)
    selected = (
        tuple(rv.name for rv in target.observed_RVs) if names is None else tuple(names)
    )
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(n not in target.named_vars for n in selected)
    ):
        raise ValueError(
            "names must select distinct model variables; default requires observed variables"
        )
    # Requesting a fitted free RV in forward sampling would resample its prior.
    free_names = {rv.name for rv in target.free_RVs}
    if free_names.intersection(selected).intersection(result.samples):
        raise ValueError("use posterior samples directly for fitted free variables")
    values = pm.sample_posterior_predictive(
        result._trace,
        model=target,
        var_names=list(selected),
        random_seed=_seed(rng),
        progressbar=False,
        return_inferencedata=False,
    )
    return PosteriorSamples(
        values,
        InferenceDiagnostics(
            "posterior_predictive",
            "pymc",
            note="Forward observations conditional on joint posterior draws, including observation noise.",
        ),
    )
