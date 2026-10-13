"""HAR models from generated intraperiod returns, with known-future covariates."""
import numpy as np

from convexity.math import volatility as vol


def generated_realized_variance(n=700, seed=7):
    rng = np.random.default_rng(seed)
    log_variance = np.zeros(n)
    for t in range(22, n):
        log_variance[t] = (-.08 + .4*log_variance[t-1] + .25*log_variance[t-5:t].mean()
                           + .15*log_variance[t-22:t].mean() + rng.normal(scale=.12))
    intraperiod_returns = rng.normal(size=(n, 100)) * np.sqrt(np.exp(log_variance[:, None])/100)
    return vol.realized_variance(intraperiod_returns)[100:]


def main():
    realized = generated_realized_variance()
    # A supplied deterministic calendar feature is known beyond the cutoff.
    calendar = np.sin(np.arange(len(realized)+5)*2*np.pi/5)[:, None]
    data = vol.TimeSeries(realized, covariates=vol.Covariates(known_future=calendar[:-5]), names=('RV',))
    for transform in ('level', 'log'):
        model = vol.HARForecaster(transform=transform)
        out = model.fit(data).forecast(5, covariates=calendar[-5:])
        print(transform, 'HAR realized variance forecast:', out.point.ravel())
        scores = vol.variance_backtest(model, data, horizon=[1, 5], initial_window=400, step=50)
        print(transform, 'HAR QLIKE:', scores.score(), 'RMSE:', scores.score('rmse'))

    fitted = vol.HARForecaster(method='simulation', simulations=5000).fit(data)
    out = fitted.forecast([1, 3, 5], covariates=calendar[-5:], rng=np.random.default_rng(21))
    print('90% realized variance intervals:', out.interval(.9)[:, 0])
    assert out.paths is not None
    print('Variance paths:', out.paths.shape)
    print('Coefficients:', dict(zip(np.asarray(fitted.parameters['feature_names'], dtype=str), np.asarray(fitted.parameters['coefficients'], dtype=float))))
    print('Volatility, sqrt(E[RV]):', out.volatility().ravel())


if __name__ == '__main__':
    main()
