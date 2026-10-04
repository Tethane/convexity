"""Generated forecasting examples: classical, regression, joint paths, evaluation."""
import numpy as np

from convexity.math import forecasting as ts


def main():
    rng = np.random.default_rng(42)
    n = 220
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = .75 * y[t - 1] + rng.normal(scale=.3)
    ar = ts.ARForecaster(trend='n')
    fitted = ar.fit(y)
    forecast = fitted.forecast([1, 3, 5])
    print('AR conditional means:', forecast.mean().ravel())
    print('90% marginal intervals:', forecast.interval(.9)[:, 0])
    assert forecast.horizon is not None
    print('Cutoff and forecast times:', forecast.cutoff, forecast.horizon.timestamps)

    results = ts.compare({'AR': ar, 'ridge': ts.RegressionForecaster(lags=(1, 2, 3))},
                         y, horizon=5, initial_window=140, step=20)
    print('Rolling-origin RMSE:', {name: round(float(result.score('rmse')), 4)
                                   for name, result in results.items()})
    print('AR CRPS by horizon:', results['AR'].score('crps', by_horizon=True).ravel())

    # Two dependent target coordinates and a genuinely joint future law.
    vector = np.column_stack((y, .5 * y + rng.normal(size=n) * .2))
    joint = ts.VARForecaster(lags=1).fit(vector).forecast(5)
    draws = joint.sample(2000, rng=rng)
    paths = ts.ForecastResult.from_paths(draws)
    assert paths.paths is not None
    print('Path shape:', paths.paths.shape)
    print('Empirical probability both last targets > 0:',
          paths.probability(lambda path: bool(np.all(path[-1] > 0))))

    x = rng.normal(size=(n, 1))
    regression = ts.RegressionForecaster(strategy='direct', max_horizon=3)
    conditioned = regression.fit(y + 2 * x[:, 0], covariates=ts.Covariates(known_future=x))
    print('Known-covariate forecast:', conditioned.forecast(3, covariates=[[0], [1], [2]]).point.ravel())

    selection = ts.select_model({'last': ts.NaiveForecaster(), 'AR': ar}, y,
                                 horizon=3, initial_window=140, step=20)
    print('Temporal validation winner:', selection.best_name)
    print('Artifact:', fitted.model_identity, fitted.training_digest[:12])


if __name__ == '__main__':
    main()
