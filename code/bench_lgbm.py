"""LightGBM arm as fixed in the plan (section 3): small fixed grid, 5-fold CV with
early stopping on log-loss, refit on all rows at the CV-chosen number of trees,
plus Platt scaling fitted on the out-of-fold predictions of the chosen config.

Returns both the raw and the Platt-scaled probabilities.
"""
import itertools

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

GRID = {
    "num_leaves": [4, 8, 16],
    "min_child_samples": [5, 20],
    "learning_rate": [0.05],
    "feature_fraction": [0.8],
}
MAX_ROUNDS = 2000
EARLY_STOP = 50
N_FOLDS = 5


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_predict_lgbm(x_tr, y_tr, x_ev, seed=0, n_threads=4):
    base = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1,
            "num_threads": n_threads, "seed": seed, "bagging_seed": seed,
            "feature_fraction_seed": seed, "deterministic": True}
    # with very few events the inner folds need at least one event each
    n_folds = int(min(N_FOLDS, max(2, y_tr.sum()), max(2, (1 - y_tr).sum())))
    folds = list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(x_tr, y_tr))

    best = None
    for values in itertools.product(*GRID.values()):
        params = dict(base, **dict(zip(GRID.keys(), values)))
        oof = np.zeros(len(y_tr))
        iters, losses = [], []
        for tr_idx, va_idx in folds:
            dtr = lgb.Dataset(x_tr[tr_idx], y_tr[tr_idx])
            dva = lgb.Dataset(x_tr[va_idx], y_tr[va_idx], reference=dtr)
            m = lgb.train(params, dtr, MAX_ROUNDS, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
            oof[va_idx] = m.predict(x_tr[va_idx], num_iteration=m.best_iteration)
            iters.append(m.best_iteration)
            losses.append(m.best_score["valid_0"]["binary_logloss"] * len(va_idx))
        loss = sum(losses) / len(y_tr)
        if best is None or loss < best["loss"]:
            best = {"loss": loss, "params": params, "n_iter": max(1, int(np.mean(iters))), "oof": oof}

    final = lgb.train(best["params"], lgb.Dataset(x_tr, y_tr), best["n_iter"])
    p_raw = final.predict(x_ev)

    # Platt on out-of-fold scores (no data reused for both fit and calibration)
    platt = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    platt.fit(_logit(best["oof"]).reshape(-1, 1), y_tr)
    p_platt = platt.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]
    info = {k: best["params"][k] for k in GRID} | {"n_iter": best["n_iter"], "n_folds": n_folds}
    return p_raw, p_platt, info
