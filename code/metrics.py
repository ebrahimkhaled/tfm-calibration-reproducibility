"""Calibration estimands (REBUILD_PLAN 4.3, pre-registration section 4) and Monte Carlo SEs.

Oracle estimands use the true risk p* of each evaluation row:
    true ICI, E50, E90, Emax   |p_hat - c(p_hat)|, c = loess of p* on logit(p_hat)
                               (degree 2, span 0.75, gaussian family, no robustness iterations)
    b*, a*                     slope of a quasi-binomial fit of p* on logit(p_hat), and the intercept
                               of the offset model (slope fixed at 1)
    OCE                        mean |p_hat - p*|
Empirical estimands use the observed outcome Y, as a validation study would (Austin-Steyerberg ICI,
calibration slope and intercept, Brier, log-loss, AUC, 10-bin ECE).

Every computation clips p_hat to [EPS, 1 - EPS] (EPS = 1e-6, pre-registration section 4) and
reports how many values were clipped.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

EPS = 1e-6

# hierarchy thresholds (pre-registration 4.5); "scale" multiplies all four for the sensitivity runs
HIERARCHY = dict(mean=0.01, weak_a=0.1, weak_b=0.1, moderate=0.015, strong_e90=0.03)


def clip(p, eps: float = EPS):
    p = np.asarray(p, dtype=float)
    return np.clip(p, eps, 1 - eps), int(np.sum((p < eps) | (p > 1 - eps)))


def logit(p):
    return special.logit(p)


# ----------------------------------------------------------------------------- loess

def loess(x, y, x_new=None, span: float = 0.75, degree: int = 2, exact: bool | None = None,
          n_anchor: int = 400):
    """Local polynomial regression in one dimension (R's loess with family = "gaussian").

    The fit at a point is R's surface = "direct" computation: q = floor(n * span) nearest
    neighbours, tricube weights on distance / (distance of the q-th neighbour), weighted least
    squares of degree 1 or 2. With exact = True it is computed at every point of x_new. Otherwise
    (the default above 1,200 points; 8 times faster at 3,334) it is computed exactly at about
    n_anchor anchor points (quantiles of x_new plus an even grid, both ends included) and joined by
    a cubic spline. The fit has small kinks where the neighbourhood changes, so single points can
    be off by up to about 1e-4, but the ICI moves by less than 1e-6 (tests/test_metrics.py), far
    below its Monte Carlo SE. R's default surface = "interpolate" uses far fewer vertices and
    differs from the direct fit by up to 1e-2 at single points and about 1e-3 in the ICI."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = x if x_new is None else np.asarray(x_new, dtype=float)
    if exact is None:
        exact = len(x_new) <= 1200
    if not exact:
        u = np.unique(x_new)
        if len(u) > n_anchor:
            from scipy.interpolate import CubicSpline
            half = n_anchor // 2
            anchors = np.unique(np.r_[np.quantile(u, np.linspace(0, 1, half)),
                                      np.linspace(u[0], u[-1], half)])
            fit = _loess_direct(x, y, anchors, span, degree)
            return CubicSpline(anchors, fit)(x_new)
    # the fit at a point depends only on its value, so tied points are fitted once (isotonic
    # ladder columns have a handful of distinct values among 3,334 rows; same numbers, far faster)
    u, inv = np.unique(x_new, return_inverse=True)
    if len(u) < len(x_new):
        return _loess_direct(x, y, u, span, degree)[inv.ravel()]
    return _loess_direct(x, y, x_new, span, degree)


_SMOOTHER_CACHE: dict = {}


def _loess_direct(x, y, x_new, span: float, degree: int, chunk: int = 256):
    """The direct fit as a smoother matrix L (rows = x_new) times y. The three loess calls of one
    prediction column (p* and logit p* in the oracle metrics, Y in the empirical ones) share x and
    x_new, so L is built once and kept for the next call. Same numbers as
    _loess_direct_reference up to rounding (about 1e-14; tests/test_metrics.py), 3 to 10 times
    faster."""
    key = (float(span), int(degree))
    c = _SMOOTHER_CACHE
    if not (c.get("key") == key and np.array_equal(c.get("x"), x) and np.array_equal(c.get("x_new"), x_new)):
        c.clear()
        c.update(key=key, x=np.array(x, copy=True), x_new=np.array(x_new, copy=True),
                 L=_smoother_matrix(x, x_new, span, degree, chunk))
    return c["L"] @ np.asarray(y, dtype=float)


def _smoother_matrix(x, x_new, span: float, degree: int, chunk: int = 256):
    n = len(x)
    q = int(math.floor(n * span))
    if q < degree + 1:
        raise ValueError("span too small for this many points")
    q = min(q, n)
    k = degree + 1
    L = np.empty((len(x_new), n))
    for s in range(0, len(x_new), chunk):
        t = x_new[s:s + chunk]
        z = x[None, :] - t[:, None]                       # signed distances, scaled below
        u = np.abs(z)
        h = np.partition(u, q - 1, axis=1)[:, q - 1]
        if span > 1:
            h = h * span                                   # R: enlarge the window beyond all points
        h = np.where(h > 0, h, 1.0)
        z /= h[:, None]
        u /= h[:, None]
        w = u * u
        w *= u
        np.subtract(1.0, w, out=w)
        np.maximum(w, 0.0, out=w)                         # tricube: zero weight for u >= 1
        wz = w * w
        wz *= w                                           # wz holds w z^j as j runs
        mom = np.empty((len(t), 2 * degree + 1))
        rows = []
        for j in range(2 * degree + 1):
            mom[:, j] = wz.sum(axis=1)
            if j <= degree:
                rows.append(wz.copy())
            if j < 2 * degree:
                wz *= z
        A = np.empty((len(t), k, k))
        for i in range(k):
            for j in range(k):
                A[:, i, j] = mom[:, i + j]
        # pseudo-inverse: tied x values can leave fewer than k distinct points in a window
        P0 = np.linalg.pinv(A, rcond=1e-10)[:, 0, :]
        Lc = P0[:, [0]] * rows[0]
        for j in range(1, k):
            Lc += P0[:, [j]] * rows[j]
        L[s:s + chunk] = Lc
    return L


def _loess_direct_reference(x, y, x_new, span: float, degree: int, chunk: int = 256):
    """The original implementation, kept for the equivalence test."""
    n = len(x)
    q = int(math.floor(n * span))
    if q < degree + 1:
        raise ValueError("span too small for this many points")
    q = min(q, n)
    out = np.empty(len(x_new))
    k = degree + 1
    for s in range(0, len(x_new), chunk):
        t = x_new[s:s + chunk]
        d = x[None, :] - t[:, None]                       # signed distances
        ad = np.abs(d)
        h = np.partition(ad, q - 1, axis=1)[:, q - 1]
        if span > 1:
            h = h * span                                   # R: enlarge the window beyond all points
        h = np.where(h > 0, h, 1.0)
        u = ad / h[:, None]
        w = np.where(u < 1, (1 - u ** 3) ** 3, 0.0)
        z = d / h[:, None]                                # centred and scaled design
        # moments sum w z^j for j = 0 .. 2*degree, and sum w y z^j for j = 0 .. degree
        zp = np.ones_like(z)
        mom = []
        for _ in range(2 * degree + 1):
            mom.append(np.sum(w * zp, axis=1))
            zp = zp * z
        A = np.empty((len(t), k, k))
        for i in range(k):
            for j in range(k):
                A[:, i, j] = mom[i + j]
        zp = np.ones_like(z)
        b = np.empty((len(t), k))
        for i in range(k):
            b[:, i] = np.sum(w * y[None, :] * zp, axis=1)
            zp = zp * z
        # pseudo-inverse: tied x values can leave fewer than k distinct points in a window
        coef = np.einsum("tij,tj->ti", np.linalg.pinv(A, rcond=1e-10), b)
        out[s:s + chunk] = coef[:, 0]
    return out


def _abs_summary(e: np.ndarray, prefix: str) -> dict:
    return {f"{prefix}ici": float(np.mean(e)), f"{prefix}e50": float(np.median(e)),
            f"{prefix}e90": float(np.quantile(e, 0.9)), f"{prefix}emax": float(np.max(e))}


# ----------------------------------------------------------------------------- GLM helpers

def fit_logistic(x, t, offset=None, slope: bool = True, max_iter: int = 100, tol: float = 1e-10):
    """Binomial ML (or quasi-binomial point estimate when t is a probability) of t on
    intercept (+ slope * x) + offset. Returns (intercept, slope or 1.0, converged)."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    off = np.zeros_like(x) if offset is None else np.asarray(offset, dtype=float)
    Z = np.column_stack([np.ones_like(x), x]) if slope else np.ones((len(x), 1))
    tbar = t.mean()
    if tbar <= 0 or tbar >= 1:
        return float("nan"), float("nan"), False
    beta = np.zeros(Z.shape[1])
    beta[0] = special.logit(tbar) - off.mean()
    dev_old = np.inf
    converged = False
    for _ in range(max_iter):
        eta = Z @ beta + off
        mu = special.expit(eta)
        W = np.maximum(mu * (1 - mu), 1e-300)
        grad = Z.T @ (t - mu)
        H = (Z * W[:, None]).T @ Z
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        # step halving on the (fractional) deviance
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for _h in range(30):
                nb = beta + step
                eta_n = Z @ nb + off
                # binomial deviance written with log-expit, finite even when mu reaches 0 or 1
                dev = -2 * np.sum(t * special.log_expit(eta_n) + (1 - t) * special.log_expit(-eta_n))
                if (np.isfinite(dev) and dev <= dev_old + 1e-12) or not np.isfinite(dev_old):
                    break
                step = step / 2
        beta = nb
        if np.isfinite(dev) and abs(dev_old - dev) < tol * (abs(dev) + 0.1) and np.max(np.abs(step)) < 1e-8:
            converged = True
            break
        dev_old = dev
    if not np.all(np.isfinite(beta)) or np.max(np.abs(beta)) > 1e6:
        converged = False
    b = float(beta[1]) if slope else 1.0
    return float(beta[0]), b, converged


def offset_intercept(lp, t) -> float:
    """Intercept a of t ~ a + offset(lp): the root of sum(t - expit(a + lp)) = 0."""
    a, _, ok = fit_logistic(np.zeros(len(lp)), t, offset=lp, slope=False)
    return a if ok else float("nan")


# ----------------------------------------------------------------------------- AUC

def expected_auc(score, p) -> float:
    """AUC of `score` when row i is an event with probability p_i; with p = Y it is the usual AUC."""
    score = np.asarray(score, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(score, kind="mergesort")
    s, w1 = score[order], p[order]
    w0 = 1.0 - w1
    new = np.r_[True, s[1:] != s[:-1]]
    starts = np.flatnonzero(new)
    grp = np.cumsum(new) - 1
    neg_in = np.add.reduceat(w0, starts)
    neg_before = np.r_[0.0, np.cumsum(neg_in)[:-1]]
    num = np.sum(w1 * (neg_before[grp] + 0.5 * (neg_in[grp] - w0)))
    den = w1.sum() * w0.sum() - np.sum(w1 * w0)
    return float(num / den) if den > 0 else float("nan")


# ----------------------------------------------------------------------------- estimands

def oracle_metrics(p_hat, p_star, span: float = 0.75) -> dict:
    """Primary and secondary oracle estimands of one prediction vector on one evaluation set."""
    ph, n_clip = clip(p_hat)
    ps = np.asarray(p_star, dtype=float)
    lp = logit(ph)
    out = {"n": len(ph), "n_clipped": n_clip}
    if np.ptp(lp) == 0:
        c = np.full_like(ph, ps.mean())                   # constant predictions: c is the mean risk
        c_ll = c
    else:
        c = np.clip(loess(lp, ps, span=span), 0, 1)
        # same smoother with logit p* as the response: zero for a perfect model and for any
        # p_hat whose logit is linear in logit p* (see README, "loess floor")
        lps = special.logit(np.clip(ps, 1e-15, 1 - 1e-15))
        c_ll = special.expit(loess(lp, lps, span=span))
    out.update(_abs_summary(np.abs(ph - c), "true_"))
    out.update(_abs_summary(np.abs(ph - c_ll), "true_ll_"))
    a2, b, ok2 = fit_logistic(lp, ps)
    out["true_slope"] = b if ok2 else float("nan")       # b*
    out["true_intercept_2p"] = a2 if ok2 else float("nan")
    out["true_intercept"] = offset_intercept(lp, ps)     # a* (slope fixed at 1)
    out["true_citl_prob"] = float(ph.mean() - ps.mean())  # mean p_hat - mean p*
    out["oce"] = float(np.mean(np.abs(ph - ps)))
    out["true_auc"] = expected_auc(ph, ps)                # AUC of p_hat when outcomes follow p*
    out["oracle_auc"] = expected_auc(ps, ps)
    out["expected_brier"] = float(np.mean((ph - ps) ** 2 + ps * (1 - ps)))
    out["expected_logloss"] = float(-np.mean(ps * np.log(ph) + (1 - ps) * np.log(1 - ph)))
    return out


def empirical_metrics(y, p_hat, span: float = 0.75, scale: str = "logit") -> dict:
    """What a validation study sees: estimands from the observed outcomes Y.

    scale = "logit" smooths Y on logit(p_hat) (pre-registration 4.3); "prob" on p_hat itself
    (the form in Austin and Steyerberg's 2019 code)."""
    y = np.asarray(y, dtype=float)
    ph, n_clip = clip(p_hat)
    lp = logit(ph)
    x = lp if scale == "logit" else ph
    out = {"n": len(ph), "n_clipped": n_clip, "events": int(y.sum())}
    if np.ptp(x) == 0:
        c = np.full_like(ph, y.mean())
    else:
        c = np.clip(loess(x, y, span=span), 0, 1)
    out.update(_abs_summary(np.abs(ph - c), "emp_"))
    a2, b, ok = fit_logistic(lp, y)
    out["emp_slope"] = b if ok else float("nan")
    out["emp_intercept_2p"] = a2 if ok else float("nan")
    out["emp_intercept"] = offset_intercept(lp, y)
    out["emp_citl_prob"] = float(ph.mean() - y.mean())
    out["brier"] = float(np.mean((ph - y) ** 2))
    out["logloss"] = float(-np.mean(y * np.log(ph) + (1 - y) * np.log(1 - ph)))
    out["auc"] = expected_auc(ph, y)
    out["ece10"] = ece(y, ph, bins=10)
    return out


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins (machine-learning convention)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(tot / len(p))


# Thresholds on the probability scale (mean, moderate, strong) are relative at pi = 3 %
# (pre-registration 4.5): absolute values of 0.01, 0.015 and 0.03 are easy to meet when the mean
# risk is 0.03, so there they are multiplied by pi / 0.10 = 0.3. Logit-scale rules (weak) are not.
RELATIVE_THRESHOLD_PI = 0.03
REFERENCE_PI = 0.10


def probability_scale(pi: float) -> float:
    """Factor for the probability-scale thresholds at event rate pi (1 except at pi = 3 %)."""
    return pi / REFERENCE_PI if abs(pi - RELATIVE_THRESHOLD_PI) < 1e-9 else 1.0


def hierarchy_class(m: dict, scale: float = 1.0, prefix: str = "true_", prob_scale: float = 1.0) -> dict:
    """Calibration classes of one (arm, replicate), from its oracle metrics (pre-registration 4.5).

    cls_mean .. cls_strong are the four rules on their own. cls_highest is nested, as in Van
    Calster et al. (2016): the highest class whose rule holds together with every lower rule. (The
    absolute thresholds make the upper rules easy to meet at a 3 % event rate, so an arm can meet
    "strong" while failing "weak"; the nested label does not count that as strong.)"""
    t = {k: v * scale * (prob_scale if k in ("mean", "moderate", "strong_e90") else 1.0)
         for k, v in HIERARCHY.items()}
    is_mean = abs(m["true_citl_prob"]) <= t["mean"]
    is_weak = (abs(m["true_intercept"]) <= t["weak_a"]) and (abs(m["true_slope"] - 1) <= t["weak_b"])
    is_mod = m[f"{prefix}ici"] <= t["moderate"]
    is_strong = m[f"{prefix}e90"] <= t["strong_e90"]
    label = "none"
    for name, ok in (("mean", is_mean), ("weak", is_weak), ("moderate", is_mod), ("strong", is_strong)):
        if not ok:
            break
        label = name
    return {"cls_mean": bool(is_mean), "cls_weak": bool(is_weak), "cls_moderate": bool(is_mod),
            "cls_strong": bool(is_strong), "cls_highest": label}


# ----------------------------------------------------------------------------- Monte Carlo SE

def mcse_mean(x) -> float:
    """MCSE of a mean over replicates: SD / sqrt(R)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def mcse_proportion(k, r) -> float:
    """MCSE of a rejection rate or class proportion k / r."""
    p = k / r
    return float(math.sqrt(p * (1 - p) / r))


def mcse_paired_diff(a, b) -> tuple[float, float]:
    """Mean paired difference a - b over the replicates where both are finite, and its MCSE."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    return float(d.mean()), mcse_mean(d)


def mcse_sd(x) -> float:
    """MCSE of an empirical SD (Morris et al. 2019, table 6: SD / sqrt(2 (R - 1)))."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1) / math.sqrt(2 * (len(x) - 1)))


def median_ci(x, level: float = 0.95) -> tuple[float, float, float]:
    """Median with a distribution-free order-statistic confidence interval."""
    from scipy import stats
    x = np.sort(np.asarray(x, dtype=float)[np.isfinite(x)])
    r = len(x)
    lo = int(stats.binom.ppf((1 - level) / 2, r, 0.5))
    hi = int(stats.binom.isf((1 - level) / 2, r, 0.5))
    lo = max(lo - 1, 0)
    hi = min(hi, r - 1)
    return float(np.median(x)), float(x[lo]), float(x[hi])


def mcse_bootstrap(x, stat=np.median, b: int = 2000, seed: int = 1) -> float:
    """MCSE of any summary over replicates by bootstrap (for medians, quantiles, ...)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    vals = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(b)]
    return float(np.std(vals, ddof=1))
