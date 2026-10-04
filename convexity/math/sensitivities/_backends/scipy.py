from numpy.typing import ArrayLike
"""Private scipy execution for sensitivities."""

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import lsq_linear, minimize
from ..._finance import array, scalar
from ..interfaces import HedgeResult


def hedge(
    exposures,
    target,
    *,
    weights=None,
    lower: ArrayLike=-np.inf,
    upper: ArrayLike=np.inf,
    equality=None,
    equality_target=None,
    ridge=0.0,
):
    """Choose h to minimize ||sqrt(W)(A h + target)||² + ridge ||h||².

    A has (risk,instrument) axes; equality is E h = b. Nullspace is the
    unconstrained instrument-space exposure nullspace, independent of bounds.
    """
    a = array(exposures, "exposures", ndim=2)
    b = array(target, "target", ndim=1)
    if a.shape[0] != len(b):
        raise ValueError("target must match exposure risk axis")
    w = np.ones(len(b)) if weights is None else array(weights, "weights", ndim=1)
    if w.shape != b.shape or np.any(w < 0) or (not np.any(w > 0)):
        raise ValueError("risk weights must align and have positive mass")
    n = a.shape[1]
    lo = np.broadcast_to(array(lower, "lower", finite=False), (n,))
    hi = np.broadcast_to(array(upper, "upper", finite=False), (n,))
    if np.any(lo >= hi):
        raise ValueError("lower bounds must be below upper bounds")
    r = scalar(ridge, "ridge", minimum=0)
    aw = np.sqrt(w)[:, None] * a
    bw = np.sqrt(w) * b
    aa = np.vstack([aw, np.sqrt(r) * np.eye(n)])
    bb = np.r_[-bw, np.zeros(n)]
    if equality is None:
        if equality_target is not None:
            raise ValueError("equality target needs equality matrix")
        fit = lsq_linear(aa, bb, bounds=(lo, hi), tol=1e-12)
    else:
        e = array(equality, "equality", ndim=2)
        f = array(equality_target, "equality target", ndim=1)
        if e.shape != (len(f), n):
            raise ValueError("equality matrix must have (constraint,instrument) axes")
        start = np.clip(np.linalg.lstsq(aa, bb, rcond=None)[0], lo, hi)
        fit = minimize(
            lambda h: 0.5 * np.sum((aa @ h - bb) ** 2),
            start,
            jac=lambda h: aa.T @ (aa @ h - bb),
            method="SLSQP",
            bounds=list(zip(lo.tolist(), hi.tolist())),
            constraints=[
                {"type": "eq", "fun": lambda h: e @ h - f, "jac": lambda h: e}
            ],
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if fit.success and (not np.allclose(e @ fit.x, f, rtol=1e-08, atol=1e-08)):
            raise RuntimeError("hedge solver returned infeasible equality constraints")
    if not fit.success:
        raise RuntimeError(f"hedge optimization failed: {fit.message}")
    return HedgeResult(
        fit.x, a @ fit.x + b, null_space(a), bool(fit.success), str(fit.message)
    )
