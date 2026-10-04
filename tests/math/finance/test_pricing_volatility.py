import numpy as np
import pytest
import QuantLib as ql
from scipy.stats import norm
from convexity.math.contracts import Exercise, VanillaContract, CashFlow
from convexity.math.pricing import *
from convexity.math.implied_volatility import *


def problem(bsm, valuation, kind="call", style="european"):
    dates = {
        "european": ("2027-01-01",),
        "american": (valuation, "2027-01-01"),
        "bermudan": ("2026-07-01", "2027-01-01"),
    }[style]
    return PricingProblem(
        VanillaContract(100, Exercise(style, dates), kind), bsm, valuation
    )


def test_bsm_parity_and_greeks(bsm, valuation):
    call = price(problem(bsm, valuation), VanillaEngine())
    put = price(problem(bsm, valuation, "put"), VanillaEngine())
    assert call.value == pytest.approx(9.227005508154, abs=1e-10)
    assert call.value - put.value == pytest.approx(
        100 * np.exp(-0.02) - 100 * np.exp(-0.05)
    )
    assert call.greeks["delta"] == pytest.approx(np.exp(-0.02) * norm.cdf(0.25))
    assert call.greeks["gamma"] > 0 and call.greeks["vega"] > 0
    assert call.error_estimate is None
    with pytest.raises(TypeError):
        call.greeks["delta"] = 0  # pyright: ignore[reportIndexIssue]  # Exercise runtime rejection.


@pytest.mark.parametrize(
    "method,tolerance", [("tree", 0.025), ("fd", 0.01), ("mc", 0.20)]
)
def test_engine_convergence(bsm, valuation, method, tolerance):
    result = price(
        problem(bsm, valuation), VanillaEngine(method, steps=200, samples=60000)
    )
    assert result.value == pytest.approx(9.227005508, abs=tolerance)
    if method == "mc":
        assert result.error_estimate is not None
        assert result.error_estimate > 0


@pytest.mark.parametrize("method", ["tree", "fd", "lsmc"])
def test_american_put_dominates_european(bsm, valuation, method):
    european = price(problem(bsm, valuation, "put"), VanillaEngine()).value
    result = price(
        problem(bsm, valuation, "put", "american"),
        VanillaEngine(method, steps=100, samples=20000),
    )
    assert result.value > european + 0.2


def test_bermudan_dividend_and_rejection(bsm, valuation):
    bermudan = price(
        problem(bsm, valuation, "put", "bermudan"), VanillaEngine("tree")
    ).value
    american = price(
        problem(bsm, valuation, "put", "american"), VanillaEngine("tree")
    ).value
    assert bermudan <= american
    base = price(problem(bsm, valuation), VanillaEngine("fd")).value
    div = price(
        problem(bsm, valuation),
        VanillaEngine("fd", dividends=(CashFlow("2026-07-01", 3),)),
    ).value
    assert div < base
    with pytest.raises(ValueError):
        price(problem(bsm, valuation, "put", "american"), VanillaEngine())
    with pytest.raises(ValueError):
        price(
            problem(bsm, valuation),
            VanillaEngine("tree", dividends=(CashFlow("2026-07-01", 3),)),
        )


def test_heston_engines(bsm, valuation):
    from dataclasses import replace

    model = HestonModel(100, bsm.discount, bsm.dividend, 0.04, 2.0, 0.04, 0.3, -0.7)
    p = replace(problem(bsm, valuation), model=model)
    analytical = price(p, VanillaEngine("heston")).value
    grid = price(p, VanillaEngine("heston_fd", steps=100, grid=100)).value
    assert analytical == pytest.approx(grid, abs=0.03)
    near_bsm = replace(model, kappa=10.0, sigma=0.001, rho=0.0)
    assert price(
        replace(p, model=near_bsm), VanillaEngine("heston")
    ).value == pytest.approx(9.227, abs=0.002)


def test_path_and_multiasset_capabilities(bsm, valuation):
    dates = ["2026-04-01", "2026-07-01", "2026-10-01", "2027-01-01"]
    geometric = asian_price(bsm, valuation, 100, dates, average="geometric")
    arithmetic = asian_price(bsm, valuation, 100, dates, samples=40000)
    assert arithmetic.value > geometric.value - 0.05
    one = basket_price([bsm], [[1]], [1], valuation, dates[-1], 100, samples=60000)
    assert one.error_estimate is not None
    assert one.value == pytest.approx(9.227, abs=3 * one.error_estimate)
    vanilla = price(problem(bsm, valuation), VanillaEngine()).value
    barrier_in = barrier_price(
        bsm, valuation, dates[-1], 100, 80, barrier_type="down_in"
    ).value
    barrier_out = barrier_price(
        bsm, valuation, dates[-1], 100, 80, barrier_type="down_out"
    ).value
    assert barrier_in + barrier_out == pytest.approx(vanilla)


@pytest.mark.parametrize(
    "model,forward,strike,vol",
    [("black76", 103, 100, 0.2), ("bachelier", -0.02, -0.01, 0.015)],
)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_forward_prices_inversion(model, forward, strike, vol, kind):
    p = forward_option(forward, strike, vol, 2.0, discount=0.95, model=model, kind=kind)
    actual = implied_volatility(
        p, forward, strike, 2.0, discount=0.95, model=model, kind=kind
    )
    assert actual == pytest.approx(vol, rel=1e-9)
    intrinsic = forward_option(
        forward, strike, 0.0, 2.0, discount=0.95, model=model, kind=kind
    )
    assert (
        implied_volatility(
            intrinsic, forward, strike, 2.0, discount=0.95, model=model, kind=kind
        )
        == 0
    )
    with pytest.raises(ValueError):
        implied_volatility(-1, forward, strike, 2, model=model)


def test_implied_bounds_shift():
    with pytest.raises(ValueError):
        implied_volatility(100, 100, 100, 1)
    with pytest.raises(ValueError):
        implied_volatility(0, 110, 100, 1)
    p = forward_option(-0.01, 0.01, 0.2, 1, displacement=0.05)
    assert implied_volatility(p, -0.01, 0.01, 1, displacement=0.05) == pytest.approx(
        0.2
    )


def test_svi_ssvi_fit_constraints():
    k = np.linspace(-1, 1, 31)
    smile = SVI(0.02, 0.07, -0.3, 0.01, 0.2)
    fitted = SVI.fit(k, smile(k))
    np.testing.assert_allclose(fitted(k), smile(k), atol=2e-6)
    assert np.min(smile.butterfly(k)) > 0
    surface = SmileSurface([1, 2], (smile, SVI(0.04, 0.09, -0.3, 0.01, 0.2)))
    assert surface.diagnose(k).passed
    assert surface.volatility(1, k) == pytest.approx(np.sqrt(smile(k)))
    with pytest.raises(ValueError):
        surface.total_variance(0.5, k)
    bad = SmileSurface([1, 2], (SVI(0.1, 0.07, -0.3, 0, 0.2), smile))
    assert not bad.diagnose(k).passed
    ssvi = SSVI(-0.5, 0.4)
    assert ssvi.validate_domain(2)
    assert ssvi(0.04, 0) == pytest.approx(0.04)
    with pytest.raises(ValueError):
        SSVI(-0.5, 10).validate_domain(2)
    with pytest.raises(ValueError):
        SVI(-1, 0.1, 0, 0, 0.1)


def test_sabr_surface_density_localvol(bsm, valuation):
    assert sabr_volatility(100, 100, 1, 2.0, 0.5, 0.3, -0.2) > 0
    assert sabr_volatility(0.03, 0.03, 1, 0.01, 0.5, 0.3, -0.2, model="bachelier") > 0
    surface = black_variance_surface(
        valuation, ["2027-01-01", "2028-01-01"], [50, 100, 150], np.full((3, 2), 0.2)
    )
    local = dupire_local_volatility(surface, bsm.discount, bsm.dividend, 100, 1.2, 100)
    assert local == pytest.approx(0.2, abs=1e-6)
    f = 100 * np.exp(0.03)
    d = np.exp(-0.05)

    def call(k):
        return forward_option(f, k, 0.2, 1, discount=d)

    k = np.linspace(60, 160, 41)
    density = risk_neutral_density(call, k, d)
    exact = norm.pdf((np.log(k / f) + 0.5 * 0.2**2) / 0.2) / (0.2 * k)
    np.testing.assert_allclose(density, exact, rtol=2e-6, atol=1e-9)


def test_scenario_and_extension_engines(bsm, valuation):
    from convexity.math.measures import gbm_pair
    from convexity.math.simulation import PathScenarios

    pair = gbm_pair(0.1, 0.05, 0.02, 0.2)
    rng = np.random.default_rng(27)
    terminal = 100 * np.exp(0.03 - 0.5 * 0.2**2 + 0.2 * rng.normal(size=100000))
    scenarios = PathScenarios(
        [0, 1], np.c_[np.full(len(terminal), 100), terminal], sampling="iid"
    )
    result = scenario_price(
        scenarios, lambda path: max(path[-1] - 100, 0), pair.pricing
    )
    assert result.error_estimate is not None
    assert result.value == pytest.approx(9.227, abs=5 * result.error_estimate)
    with pytest.raises(ValueError):
        scenario_price(scenarios, lambda p: p[-1], pair.physical)
    extension = QuantLibEngine(
        lambda model: ql.AnalyticEuropeanEngine(model.to_quantlib())
    )
    assert price(problem(bsm, valuation), extension).value == pytest.approx(9.227005508)
    with pytest.raises(ValueError):
        PricingProblem(None, object(), valuation)  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.
