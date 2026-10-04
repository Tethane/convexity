"""Private quantlib_engines execution for pricing."""

import numpy as np
import QuantLib as ql
from ..._finance import (
    evaluation_date,
    choice,
    scalar,
    positive,
    integer,
    option_type,
    qdate,
    array,
)
from ...contracts import VanillaContract
from ..interfaces import PricingResult
from ..builtins.models import BlackScholesModel, HestonModel
from .quantlib_models import (
    constant_volatility,
    require_constant_coefficients,
    same_numeraire,
    zero_volatility,
)


def _result(instrument, name, multiplier=1.0, currency="USD", *, collect_greeks=True):
    value = instrument.NPV() * multiplier
    greeks = {}
    for key in ("delta", "gamma", "vega", "theta", "rho", "dividendRho"):
        method = getattr(instrument, key, None) if collect_greeks else None
        if method is not None:
            try:
                greeks[key] = method() * multiplier
            except RuntimeError:
                pass
    try:
        error = instrument.errorEstimate() * multiplier
    except RuntimeError:
        error = None
    return PricingResult(value, greeks, error, name, currency)


def price_vanilla(self, problem):
    contract, model = (problem.contract, problem.model)
    if (
        not isinstance(contract, VanillaContract)
        or contract.exercise.style not in self.capabilities
    ):
        raise ValueError("contract/exercise is unsupported by this engine")
    is_heston = self.method.startswith("heston")
    if not isinstance(model, HestonModel if is_heston else BlackScholesModel):
        raise TypeError("engine and pricing model are incompatible")
    if model.discount.reference_date != problem.valuation_date:
        raise ValueError("valuation date must equal discount curve reference date")
    if contract.exercise.dates[-1] <= problem.valuation_date:
        raise ValueError("option expiry must follow valuation date")
    if any((cf.currency != contract.currency for cf in self.dividends)):
        raise ValueError("dividend currency must match the contract")
    if self.dividends and self.method != "fd":
        raise ValueError("discrete dividends require the BSM finite-difference engine")
    if self.method == "tree":
        require_constant_coefficients(model, "binomial tree")
    zero_vol = isinstance(model, BlackScholesModel) and zero_volatility(model)
    if zero_vol and self.method in ("tree", "fd", "lsmc"):
        raise ValueError("this engine requires positive volatility")
    with evaluation_date(problem.valuation_date):
        native = model.to_quantlib()
        if self.method == "analytic":
            engine = ql.AnalyticEuropeanEngine(native)
        elif self.method == "tree":
            engine = ql.BinomialVanillaEngine(native, "crr", self.steps)
        elif self.method == "fd":
            divs = [
                ql.FixedDividend(scalar(cf.amount, "dividend", minimum=0), cf.date)
                for cf in self.dividends
            ]
            if any(
                (
                    cf.date <= problem.valuation_date
                    or cf.date > contract.exercise.dates[-1]
                    for cf in self.dividends
                )
            ):
                raise ValueError("dividends must lie in (valuation,expiry]")
            engine = ql.FdBlackScholesVanillaEngine(
                native,
                divs,
                self.steps,
                self.grid,
                0,
                ql.FdmSchemeDesc.Douglas(),
                not constant_volatility(model),
            )
        elif self.method == "heston":
            engine = ql.AnalyticHestonEngine(native)
        elif self.method == "heston_fd":
            engine = ql.FdHestonVanillaEngine(
                native, self.steps, self.grid, max(20, self.grid // 2)
            )
        elif self.method == "lsmc":
            engine = ql.MCAmericanEngine(
                native,
                "pseudorandom",
                timeSteps=self.steps,
                requiredSamples=self.samples,
                seed=self.seed,
                nCalibrationSamples=self.samples,
                seedCalibration=self.seed + 1,
            )
        else:
            engine = ql.MCEuropeanEngine(
                native,
                "pseudorandom",
                timeSteps=self.steps,
                requiredSamples=self.samples,
                seed=self.seed,
            )
        instrument = contract.to_quantlib()
        instrument.setPricingEngine(engine)
        return _result(
            instrument,
            self.method,
            contract.multiplier,
            contract.currency,
            collect_greeks=not zero_vol,
        )


def price_native(self, problem):
    with evaluation_date(problem.valuation_date):
        instrument = (
            problem.contract.to_quantlib()
            if hasattr(problem.contract, "to_quantlib")
            else problem.contract
        )
        instrument.setPricingEngine(self.factory(problem.model))
        return _result(
            instrument,
            self.name,
            getattr(problem.contract, "multiplier", 1.0),
            getattr(problem.contract, "currency", "USD"),
        )


def forward_option(
    forward,
    strike,
    volatility,
    maturity,
    *,
    discount=1.0,
    kind="call",
    model="black76",
    displacement=0.0,
):
    """Black-76/normal prices; normal volatility has price/sqrt(year) units."""
    f, k = (scalar(forward, "forward"), scalar(strike, "strike"))
    sd = scalar(volatility, "volatility", minimum=0) * np.sqrt(
        scalar(maturity, "maturity", minimum=0)
    )
    d, sign = (positive(discount, "discount"), option_type(kind))
    choice(model, ("black76", "bachelier"), "model")
    if model == "bachelier":
        if displacement != 0:
            raise ValueError("displacement is only supported for Black")
        return ql.bachelierBlackFormula(sign, k, f, sd, d)
    shift = scalar(displacement, "displacement", minimum=0)
    if f + shift <= 0 or k + shift < 0:
        raise ValueError(
            "shifted forward must be positive and shifted strike nonnegative"
        )
    return ql.blackFormula(sign, k, f, sd, d, shift)


def asian_price(
    model,
    valuation_date,
    strike,
    fixing_dates,
    *,
    kind="call",
    average="arithmetic",
    samples=32768,
    seed=42,
    past_fixings=0,
    running_accumulator=None,
):
    """Discrete European average-price option; running sum/product is explicit."""
    choice(average, ("arithmetic", "geometric"), "average")
    if not isinstance(model, BlackScholesModel):
        raise TypeError("Asian engines require a BlackScholesModel")
    if average == "geometric":
        require_constant_coefficients(model, "analytic geometric Asian")
    elif not constant_volatility(model):
        raise ValueError("fixing-grid Asian Monte Carlo requires constant volatility")
    ds = [qdate(d) for d in fixing_dates]
    vd = qdate(valuation_date)
    if not ds or any((b <= a for a, b in zip(ds, ds[1:]))) or ds[0] <= vd:
        raise ValueError("future fixing dates must be ordered and after valuation")
    if vd != model.discount.reference_date:
        raise ValueError("valuation and curve reference must agree")
    past = integer(past_fixings, "past fixings")
    if past and running_accumulator is None:
        raise ValueError("past fixings require running sum/product")
    acc = (
        (0.0 if average == "arithmetic" else 1.0)
        if running_accumulator is None
        else scalar(running_accumulator, "accumulator")
    )
    neutral = 0.0 if average == "arithmetic" else 1.0
    if past == 0 and acc != neutral or (past > 0 and acc <= 0):
        raise ValueError("running accumulator must match positive past fixings")
    with evaluation_date(vd):
        process = model.to_quantlib()
        instrument = ql.DiscreteAveragingAsianOption(
            ql.Average.Arithmetic if average == "arithmetic" else ql.Average.Geometric,
            acc,
            past,
            ds,
            ql.PlainVanillaPayoff(option_type(kind), positive(strike, "strike")),
            ql.EuropeanExercise(ds[-1]),
        )
        engine = (
            ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(process)
            if average == "geometric"
            else ql.MCDiscreteArithmeticAPEngine(
                process,
                "pseudorandom",
                requiredSamples=integer(samples, "samples", 2),
                seed=integer(seed, "seed", 1),
            )
        )
        instrument.setPricingEngine(engine)
        return _result(instrument, f"{average}-asian")


def basket_price(
    models,
    correlation,
    weights,
    valuation_date,
    expiry,
    strike,
    *,
    kind="call",
    samples=32768,
    seed=42,
):
    """European weighted arithmetic basket; correlated Q GBM processes."""
    models = tuple(models)
    w, corr = (
        array(weights, "weights", ndim=1),
        array(correlation, "correlation", ndim=2),
    )
    n = len(models)
    if (
        not n
        or w.shape != (n,)
        or corr.shape != (n, n)
        or (not np.allclose(corr, corr.T))
        or (not np.allclose(np.diag(corr), 1))
        or (np.linalg.eigvalsh(corr).min() < -1e-12)
    ):
        raise ValueError("basket weights and PSD correlation must match model count")
    if any(
        (
            not isinstance(m, BlackScholesModel) or not constant_volatility(m)
            for m in models
        )
    ):
        raise ValueError(
            "one-step basket engine requires constant-volatility BSM models"
        )
    vd, end = (qdate(valuation_date), qdate(expiry))
    if end <= vd or any((m.discount.reference_date != vd for m in models)):
        raise ValueError("expiry must follow the shared valuation date")
    if any(not same_numeraire(m.discount, models[0].discount) for m in models):
        raise ValueError("basket assets must use the same discount numeraire")
    with evaluation_date(vd):
        process = ql.StochasticProcessArray(
            [m.to_quantlib() for m in models], ql.Matrix(corr.tolist())
        )
        payoff = ql.AverageBasketPayoff(
            ql.PlainVanillaPayoff(option_type(kind), scalar(strike, "strike")),
            w.tolist(),
        )
        instrument = ql.BasketOption(payoff, ql.EuropeanExercise(end))
        instrument.setPricingEngine(
            ql.MCEuropeanBasketEngine(
                process,
                "pseudorandom",
                timeSteps=1,
                requiredSamples=integer(samples, "samples", 2),
                seed=integer(seed, "seed", 1),
            )
        )
        return _result(instrument, "basket-mc")


def barrier_price(
    model,
    valuation_date,
    expiry,
    strike,
    barrier,
    *,
    barrier_type="down_out",
    rebate=0.0,
    kind="call",
):
    if not isinstance(model, BlackScholesModel):
        raise TypeError("barrier engine requires a BlackScholesModel")
    require_constant_coefficients(model, "analytic barrier")
    types = {
        "down_out": ql.Barrier.DownOut,
        "up_out": ql.Barrier.UpOut,
        "down_in": ql.Barrier.DownIn,
        "up_in": ql.Barrier.UpIn,
    }
    vd, end = (qdate(valuation_date), qdate(expiry))
    if end <= vd or vd != model.discount.reference_date:
        raise ValueError("expiry must follow shared valuation date")
    with evaluation_date(vd):
        instrument = ql.BarrierOption(
            types[choice(barrier_type, tuple(types), "barrier type")],
            positive(barrier, "barrier"),
            scalar(rebate, "rebate", minimum=0),
            ql.PlainVanillaPayoff(option_type(kind), positive(strike, "strike")),
            ql.EuropeanExercise(end),
        )
        instrument.setPricingEngine(ql.AnalyticBarrierEngine(model.to_quantlib()))
        return _result(instrument, "analytic-barrier")
