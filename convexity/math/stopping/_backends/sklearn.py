"""Private sklearn execution for stopping."""

from ..interfaces import PolicyEvaluation
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from ..._finance import array, integer


def _inputs(states, rewards, discounts, exercise):
    x = array(states, "states")
    r = array(rewards, "rewards", ndim=2)
    if np.any(r < 0):
        raise ValueError("LSMC requires nonnegative exercise rewards")
    if x.ndim == 2:
        x = x[:, :, None]
    if x.ndim != 3 or x.shape[:2] != r.shape or len(r) < 2 or (r.shape[1] < 2):
        raise ValueError(
            "states (draw,time[,state]) and rewards (draw,time) must align"
        )
    d = np.broadcast_to(array(discounts, "discounts"), (len(r), r.shape[1] - 1))
    if np.any(d <= 0):
        raise ValueError("discount factors must be positive")
    e = np.ones(r.shape[1], dtype=bool) if exercise is None else np.asarray(exercise)
    if e.dtype.kind != "b" or e.shape != (r.shape[1],) or (not e[-1]):
        raise ValueError("boolean exercise mask must include maturity")
    return (x, r, d, e)


def evaluate_policy(self, states, rewards, discounts):
    x, r, d, e = _inputs(states, rewards, discounts, np.array(self.exercise))
    if len(self.regressions) != r.shape[1] - 1 or x.shape[2] != self.state_dimension:
        raise ValueError("validation paths must match policy times and states")
    tau = np.full(len(r), r.shape[1] - 1, dtype=int)
    alive = np.ones(len(r), dtype=bool)
    for t, regression in enumerate(self.regressions):
        if e[t] and regression is not None:
            stop = alive & (r[:, t] > 0) & (r[:, t] >= regression.predict(x[:, t]))
            tau[stop] = t
            alive[stop] = False
    discount = np.c_[np.ones(len(r)), np.cumprod(d, axis=1)]
    pay = r[np.arange(len(r)), tau] * discount[np.arange(len(r)), tau]
    tau.setflags(write=False)
    return PolicyEvaluation(
        value=float(pay.mean()),
        standard_error=float(pay.std(ddof=1) / np.sqrt(len(pay))),
        exercise_indices=tau,
        discounted_payoffs=array(pay, "payoffs"),
    )


def least_squares_policy(states, rewards, discounts, *, degree=2, exercise=None):
    """Longstaff–Schwartz fit on Q paths; use independent paths for valuation.

    Nonnegative exercise rewards are required. State includes all information
    needed to make the contract Markov (e.g. running average for Asian claims).
    """
    from ..builtins.solvers import ExercisePolicy

    x, r, d, e = _inputs(states, rewards, discounts, exercise)
    if np.any(r < 0):
        raise ValueError("LSMC requires nonnegative exercise rewards")
    degree = integer(degree, "degree", 1)
    cash = r[:, -1].copy()
    regressions = [None] * (r.shape[1] - 1)
    for t in range(r.shape[1] - 2, -1, -1):
        cash *= d[:, t]
        itm = r[:, t] > 0
        if e[t] and np.any(itm):
            regression = make_pipeline(
                StandardScaler(),
                PolynomialFeatures(degree, include_bias=False),
                LinearRegression(),
            )
            regression.fit(x[itm, t], cash[itm])
            regressions[t] = regression
            stop = itm & (r[:, t] >= regression.predict(x[:, t]))
            cash[stop] = r[stop, t]
    return ExercisePolicy(tuple(regressions), tuple(e.tolist()), x.shape[2])
