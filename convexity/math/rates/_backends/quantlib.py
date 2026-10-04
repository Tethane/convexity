"""Private quantlib execution for rates."""

from ..interfaces import RatePaths
from ..._backends.quantlib import (
    fixing_context,
    set_fixings,
    frequency as frequency_value,
)
import numpy as np
import QuantLib as ql
from ..._finance import positive, scalar, choice, array, qdate, integer, day_count
from ...processes import OrnsteinUhlenbeck, CIR
from ..._finance import evaluation_date
from ...pricing._backends.quantlib_engines import _result


def rate_model(self):
    p = self.parameters
    if self.kind == "vasicek":
        return ql.Vasicek(p["r0"], p["kappa"], p["theta"], p["sigma"], 0.0)
    if self.kind == "cir":
        if p["sigma"] ** 2 >= 2 * p["kappa"] * p["theta"]:
            raise ValueError("QuantLib CIR valuation requires sigma² < 2*kappa*theta")
        return ql.CoxIngersollRoss(p["r0"], p["theta"], p["kappa"], p["sigma"])
    if self.kind == "hull_white":
        return ql.HullWhite(self.curve.handle, p["kappa"], p["sigma"])
    return ql.G2(self.curve.handle, p["a"], p["sigma"], p["b"], p["eta"], p["rho"])


def rate_process(self):
    """Existing exact transition interface for time-homogeneous P or Q laws."""
    p = self.parameters
    if self.kind == "vasicek":
        return OrnsteinUhlenbeck(p["kappa"], p["theta"], p["sigma"])
    if self.kind == "cir":
        return CIR(p["kappa"], p["theta"], p["sigma"])
    if self.kind == "hull_white":
        from ..builtins.processes import HullWhiteProcess

        return HullWhiteProcess(self)
    from ...processes import LinearDiffusion

    return LinearDiffusion(
        [[-p["a"], 0.0], [0.0, -p["b"]]],
        [
            [p["sigma"], 0.0],
            [p["rho"] * p["eta"], p["eta"] * np.sqrt(1 - p["rho"] ** 2)],
        ],
    )


def gaussian_hjm(curve, step_dates, volatilities, reversions, *, terminal_horizon=60.0):
    """QuantLib GSR: one-factor Gaussian HJM with piecewise volatility/reversion.

    Native GSR state uses its terminal forward measure; do not label simulated
    native states as bank-account-measure short rates.
    """
    dates = [qdate(d) for d in step_dates]
    v = array(volatilities, "volatilities", ndim=1)
    a = array(reversions, "reversions", ndim=1)
    if (
        len(v) != len(dates) + 1
        or len(a) not in (1, len(v))
        or np.any(v <= 0)
        or np.any(a <= 0)
        or any((b <= c for c, b in zip(dates, dates[1:])))
        or (dates and dates[0] <= curve.reference_date)
    ):
        raise ValueError(
            "GSR requires future ordered steps, one volatility per interval, and positive reversions"
        )
    return ql.Gsr(
        curve.handle,
        dates,
        [ql.QuoteHandle(ql.SimpleQuote(x)) for x in v],
        [ql.QuoteHandle(ql.SimpleQuote(x)) for x in a],
        positive(terminal_horizon, "terminal horizon"),
    )


def market_model_paths(model, *, draws, seed=42):
    """QuantLib predictor-corrector forward-rate paths under terminal measure.

    model is a native QuantLib MarketModel (e.g. AbcdVol); output axes are
    (draw,evolution step,forward rate). Numeraire weights are returned explicitly.
    """
    draws = integer(draws, "draws", 1)
    evolution = model.evolution()
    steps = evolution.numberOfSteps()
    rates = model.numberOfRates()
    numeraires = [rates] * steps
    evolver = ql.LogNormalFwdRateIpc(
        model, ql.MTBrownianGeneratorFactory(integer(seed, "seed", 1)), numeraires
    )
    out = np.empty((draws, steps, rates))
    weights = np.empty(draws)
    for i in range(draws):
        weight = evolver.startNewPath()
        for j in range(steps):
            weight *= evolver.advanceStep()
            out[i, j] = list(evolver.currentState().forwardRates())
        weights[i] = weight
    return RatePaths(
        values=array(out, "forward paths"),
        weights=array(weights, "weights"),
        times=array(list(evolution.evolutionTimes()), "times"),
        measure="terminal_forward",
    )


def inflation_curve(
    reference_date,
    dates,
    rates,
    *,
    frequency=None,
    convention="actual365",
    year_on_year=False,
):
    frequency = frequency_value(frequency, default=12)
    ds = [qdate(d) for d in dates]
    r = array(rates, "inflation rates", ndim=1)
    if (
        len(ds) != len(r)
        or len(ds) < 2
        or any((b <= a for a, b in zip(ds, ds[1:])))
        or np.any(r <= -1)
    ):
        raise ValueError("ordered dates and matching inflation rates > -1 required")
    constructor = ql.YoYInflationCurve if year_on_year else ql.ZeroInflationCurve
    return constructor(
        qdate(reference_date), ds, r.tolist(), frequency, day_count(convention)
    )


def inflation_index(curve, *, year_on_year=False, fixings=None):
    index = (
        ql.YYUSCPI(ql.YoYInflationTermStructureHandle(curve))
        if year_on_year
        else ql.USCPI(ql.ZeroInflationTermStructureHandle(curve))
    )

    return set_fixings(index, fixings)


def rate_option_price(
    instrument,
    curve,
    *,
    kind=None,
    method="black",
    volatility=None,
    model=None,
    steps=300,
    displacement=0.0,
):
    expected = (
        "cap_floor"
        if isinstance(instrument, ql.CapFloor)
        else "swaption"
        if isinstance(instrument, ql.Swaption)
        else None
    )
    kind = expected if kind is None else kind
    choice(kind, ("cap_floor", "swaption"), "instrument kind")
    if kind != expected:
        raise ValueError("rate option kind does not match the instrument")
    displacement = scalar(displacement, "displacement", minimum=0)
    if method == "normal" and displacement:
        raise ValueError("displacement applies only to Black pricing")
    if isinstance(instrument, ql.Swaption):
        if instrument.exercise().lastDate() <= curve.reference_date:
            raise ValueError("swaption must expire after valuation")
        if (
            method in ("black", "normal", "analytic", "g2")
            and instrument.exercise().type() != ql.Exercise.European
        ):
            raise ValueError(
                "this swaption engine requires European exercise; use tree"
            )
    choice(method, ("black", "normal", "analytic", "tree", "g2"), "rate engine")
    if method in ("black", "normal"):
        v = ql.QuoteHandle(ql.SimpleQuote(scalar(volatility, "volatility", minimum=0)))
        if kind == "cap_floor":
            engine = (
                ql.BlackCapFloorEngine(
                    curve.handle, v, ql.Actual365Fixed(), displacement
                )
                if method == "black"
                else ql.BachelierCapFloorEngine(curve.handle, v)
            )
        else:
            engine = (
                ql.BlackSwaptionEngine(
                    curve.handle, v, ql.Actual365Fixed(), displacement
                )
                if method == "black"
                else ql.BachelierSwaptionEngine(curve.handle, v)
            )
    else:
        if model is None or model.measure != "Q":
            raise ValueError("rate valuation requires a Q model")
        if model.curve is not None and model.curve.structure is not curve.structure:
            raise ValueError(
                "curve-fitted rate model must use the supplied pricing curve"
            )
        native = model.to_quantlib()
        if method == "g2":
            if kind != "swaption" or model.kind != "g2":
                raise ValueError("G2 analytic engine requires a G2 swaption")
            engine = ql.G2SwaptionEngine(native, 6.0, 64)
        elif method == "tree":
            engine = (
                ql.TreeCapFloorEngine if kind == "cap_floor" else ql.TreeSwaptionEngine
            )(native, integer(steps, "steps", 2), curve.handle)
        else:
            engine = (
                ql.AnalyticCapFloorEngine
                if kind == "cap_floor"
                else ql.JamshidianSwaptionEngine
            )(native, curve.handle)
    with evaluation_date(curve.reference_date), fixing_context(instrument):
        instrument.setPricingEngine(engine)
        return _result(
            instrument,
            f"{kind}-{method}",
            currency=getattr(instrument, "_convexity_currency", "USD"),
        )


def inflation_swap_price(instrument, discount):
    with evaluation_date(discount.reference_date), fixing_context(instrument):
        instrument.setPricingEngine(ql.DiscountingSwapEngine(discount.handle))
        return _result(
            instrument,
            "inflation-swap",
            currency=getattr(instrument, "_convexity_currency", "USD"),
        )


def hull_white_coefficient(model, time, state, kind):
    with evaluation_date(model.curve.reference_date):
        process = ql.HullWhiteProcess(
            model.curve.handle, model.parameters["kappa"], model.parameters["sigma"]
        )
        value = getattr(process, kind)(scalar(time, "time", minimum=0), float(state))
        return array([value] if kind == "diffusion" else value, kind)


def hull_white_transition(model, state, dt, time):
    from ...processes.interfaces import Transition
    from ...processes._backends.laws import gaussian

    with evaluation_date(model.curve.reference_date):
        process = ql.HullWhiteProcess(
            model.curve.handle, model.parameters["kappa"], model.parameters["sigma"]
        )
        mean = process.expectation(time, float(state), dt)
        variance = process.variance(time, float(state), dt)
    return Transition.from_distribution(gaussian(mean, variance), backend="quantlib")
