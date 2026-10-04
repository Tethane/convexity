"""FilterPy EKF/UKF with checked missing-data updates and nonlinear RTS passes."""
import numpy as np

from .. import _validation as v
from ..builtins.models import _log_gaussian
from ..interfaces.results import FilterResult, GaussianState, StateSequence


def _sqrt_psd(matrix):
    """A row-oriented sigma root, permitting exactly deterministic coordinates."""
    matrix = v.covariance(matrix, len(matrix))
    from ..._backends.gaussian import square_root
    return square_root(matrix).T


def run(model, observations, *, method, smooth, alpha=.5, beta=2., kappa=0.):
    try:
        from filterpy.kalman import ExtendedKalmanFilter, MerweScaledSigmaPoints, UnscentedKalmanFilter, unscented_transform
    except ImportError as exc:
        raise ImportError('nonlinear Gaussian inference requires convexity[state-space] (FilterPy)') from exc
    from ..._backends.gaussian import conditional_gain
    if not isinstance(model.initial, GaussianState):
        raise TypeError('Gaussian filters require an explicit GaussianState initial law')
    for kernel in (model.transition, model.observation):
        if not all(callable(getattr(kernel, name, None)) for name in ('mean', 'covariance')):
            raise TypeError('Gaussian inference requires additive Gaussian kernels')
    series = v.data(observations, model.observation_dimension)
    steps = model.steps(series)
    d, k, n = model.state_dimension, model.observation_dimension, len(series)
    points = None
    if method == 'unscented':
        if d + kappa <= 0 or not np.isfinite(alpha**2 * (d + kappa)) or alpha**2 * (d + kappa) == 0:
            raise ValueError('unscented scaling requires finite positive alpha^2*(dimension+kappa)')
        points = MerweScaledSigmaPoints(d, alpha, beta, kappa, sqrt_method=_sqrt_psd)
    mean, cov = model.initial.mean.copy(), model.initial.covariance.copy()

    class NonlinearExtended(ExtendedKalmanFilter):
        def predict_x(self, u=0):
            self.x = v.shaped(model.transition.mean(self.x, step), (d,), 'transition mean').copy()

    pm, pc, fm, fc, cross, om, oc, ll = [], [], [], [], [], [], [], []
    for step, y in zip(steps, series.values):
        if step.index:
            q = v.covariance(model.transition.covariance(step), d, 'process covariance')
            if method == 'extended':
                f = v.shaped(model.transition.jacobian(mean, step), (d, d), 'transition Jacobian')
                cross.append(cov @ f.T)
                engine = NonlinearExtended(d, k)
                engine.x = mean.copy()
                engine.P, engine.F, engine.Q = cov.copy(), f, q
                engine.predict()
                mean, cov = engine.x, engine.P
            else:
                assert points is not None
                sigmas = points.sigma_points(mean, cov)
                propagated = np.array([v.shaped(model.transition.mean(x, step), (d,), 'transition mean') for x in sigmas])
                old_mean = mean.copy()
                engine = UnscentedKalmanFilter(d, k, step.dt,
                    hx=lambda x: model.observation.mean(x, step),
                    fx=lambda x, dt: model.transition.mean(x, step), points=points)
                engine.x, engine.P, engine.Q = mean.copy(), cov.copy(), q
                engine.predict(dt=step.dt)
                mean, cov = engine.x, engine.P
                cross.append(np.einsum('s,si,sj->ij', points.Wc, sigmas-old_mean, propagated-mean))
        cov = v.covariance(cov, d, 'predicted covariance')
        pm.append(mean.copy())
        pc.append(cov.copy())
        r = v.covariance(model.observation.covariance(step), k, 'observation covariance')
        if method == 'extended':
            h = v.shaped(model.observation.jacobian(mean, step), (k, d), 'observation Jacobian')
            observation_mean = v.shaped(model.observation.mean(mean, step), (k,), 'observation mean')
            observation_cov = h @ cov @ h.T + r
        else:
            assert points is not None
            # Regenerate points from the full predicted covariance, including Q.
            # Reusing FilterPy's propagated points would omit Q in the update.
            sigmas = points.sigma_points(mean, cov)
            observation_sigmas = np.array([v.shaped(model.observation.mean(x, step), (k,), 'observation mean') for x in sigmas])
            observation_mean, observation_cov = unscented_transform(observation_sigmas, points.Wm, points.Wc, r)
        om.append(observation_mean)
        oc.append(observation_cov)
        keep = np.flatnonzero(~np.isnan(y))
        if len(keep):
            innovation_cov = observation_cov[np.ix_(keep, keep)]
            ll.append(float(_log_gaussian((y[keep]-observation_mean[keep])[None, :], innovation_cov)[0]))
            partial_r = r[np.ix_(keep, keep)]
            if method == 'extended':
                engine = ExtendedKalmanFilter(d, len(keep))
                engine.x, engine.P, engine.R = mean.copy(), cov.copy(), partial_r
                engine.update(y[keep], HJacobian=lambda x: h[keep],
                              Hx=lambda x: model.observation.mean(x, step)[keep])
            else:
                engine = UnscentedKalmanFilter(d, len(keep), 1.,
                    hx=lambda x: model.observation.mean(x, step)[keep], fx=lambda x, dt: x, points=points)
                engine.x, engine.P, engine.R = mean.copy(), cov.copy(), partial_r
                engine.sigmas_f = sigmas
                engine.update(y[keep])
            mean, cov = engine.x, engine.P
        else:
            ll.append(0.)
        cov = v.covariance(cov, d, 'filtered covariance')
        fm.append(mean.copy())
        fc.append(cov.copy())
    smoothed = None
    if smooth:
        means, covariances = np.array(fm), np.array(fc)
        # The nonlinear RTS identity uses the stored predicted means and cross
        # covariances; FilterPy's linear RTS helper assumes F @ x with no offset.
        for t in range(n-2, -1, -1):
            gain = conditional_gain(cross[t], pc[t+1])
            means[t] += gain @ (means[t+1] - pm[t+1])
            covariances[t] += gain @ (covariances[t+1] - pc[t+1]) @ gain.T
        smoothed = StateSequence(means, covariances)
    return FilterResult(series, StateSequence(fm, fc), ll, 'filterpy', method,
        predicted=StateSequence(pm, pc), smoothed=smoothed, observation_mean=om,
        observation_covariance=oc, state_names=model.state_names,
        approximation='first-order Gaussian' if method == 'extended' else 'sigma-point Gaussian')
