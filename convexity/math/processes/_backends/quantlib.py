"""QuantLib Heston QE in (spot,variance) coordinates, with local flat curves."""
import numpy as np


def heston_kernel(model, state):
    try:
        import QuantLib as ql
    except ImportError as exc:
        raise ImportError('Heston QE requires convexity[processes] (QuantLib)') from exc
    # Fixed reference date; never mutate QuantLib's global evaluation date.
    reference = ql.Date(1, 1, 2000)
    day_count = ql.Actual365Fixed()
    drift = ql.YieldTermStructureHandle(ql.FlatForward(reference, model.mu, day_count))
    zero = ql.YieldTermStructureHandle(ql.FlatForward(reference, 0., day_count))
    return ql.HestonProcess(drift, zero, ql.QuoteHandle(ql.SimpleQuote(float(state[0]))),
                           float(state[1]), model.kappa, model.theta, model.sigma, model.rho,
                           ql.HestonProcess.QuadraticExponential)


def heston_sample(process, state, dt, substeps, size, rng):
    result = np.empty((int(np.prod(size)), 2))
    h = dt/substeps
    for i in range(len(result)):
        x = state
        for j in range(substeps):
            x = np.asarray(process.evolve(j*h, x.tolist(), h, rng.standard_normal(2).tolist()))
        result[i] = x
    return result.reshape(size + (2,))
