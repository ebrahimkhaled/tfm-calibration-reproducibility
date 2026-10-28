"""Recalibration ladder (REBUILD_PLAN 4.6, pre-registration 5.5).

A local sample of m rows (the first m rows of the replicate's recalibration set, m in {100, 500})
updates an arm's frozen predictions; the update is applied to the evaluation rows.

    R0  none
    R1  intercept only:     glm(y ~ 1, offset = logit(p_hat))
    R2  logistic (a, b):    glm(y ~ logit(p_hat))
    R3  isotonic (PAV) of y on p_hat, clipped to [EPS, 1 - EPS]

Predictions are clipped to [EPS, 1 - EPS] before the logit, as everywhere else.
"""

from __future__ import annotations

import numpy as np
from scipy import special

from metrics import EPS, clip, fit_logistic

LEVELS = ("R1", "R2", "R3")
M_SIZES = (100, 500)


def column_name(arm: str, level: str, m: int) -> str:
    """Column of the ladder output in the prediction files, e.g. p_flic__R2_m500."""
    return f"p_{arm}__{level}_m{m}"


def recalibrate(p_local, y_local, p_target, level: str) -> tuple[np.ndarray | None, dict]:
    """Fit one ladder level on (p_local, y_local) and apply it to p_target."""
    y = np.asarray(y_local, dtype=float)
    pl, _ = clip(p_local)
    pt, _ = clip(p_target)
    info = {"m": len(y), "events": int(y.sum())}
    if y.sum() == 0 or y.sum() == len(y):
        info["error"] = "local sample has a single class"
        return None, info
    if level == "R1":
        a, _, ok = fit_logistic(np.zeros(len(pl)), y, offset=special.logit(pl), slope=False)
        info.update(a=a, converged=ok)
        if not ok:
            info["error"] = "intercept update did not converge"
            return None, info
        return special.expit(a + special.logit(pt)), info
    if level == "R2":
        a, b, ok = fit_logistic(special.logit(pl), y)
        info.update(a=a, b=b, converged=ok)
        if not ok:
            info["error"] = "logistic recalibration did not converge (separation?)"
            return None, info
        return special.expit(a + b * special.logit(pt)), info
    if level == "R3":
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        iso.fit(pl, y)
        out = np.clip(iso.predict(pt), EPS, 1 - EPS)
        info["n_steps"] = int(len(np.unique(iso.predict(pl))))
        return out, info
    raise ValueError(f"unknown ladder level {level}")


def ladder(p_recal, y_recal, p_eval, m_sizes=M_SIZES, levels=LEVELS) -> tuple[dict, dict]:
    """All levels and local sizes for one arm: ({(level, m): p_eval_updated}, {(level, m): info})."""
    preds, infos = {}, {}
    for m in m_sizes:
        for lev in levels:
            p, info = recalibrate(p_recal[:m], y_recal[:m], p_eval, lev)
            preds[(lev, m)] = p
            infos[(lev, m)] = info
    return preds, infos
