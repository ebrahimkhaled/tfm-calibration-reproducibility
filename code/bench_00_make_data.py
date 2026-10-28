"""Timing benchmark, step 0: write the datasets that every arm (Python and R) reads.

One development set per (p, n_train, rep) and one 10,000-row evaluation set
per (p, rep). Seeds come from SeedSequence((p, n_train, rep)) so nothing
depends on run order. Each file gets a SHA-256 and a simple weighted sum
(`checksum`) in results/bench/data/manifest.json; both the Python and the R
runner recompute the checksum and stop if it does not match.

Data: X ~ N(0, Sigma) with exchangeable correlation 0.3, logistic outcome,
slopes 1/sqrt(j) rescaled so the linear predictor has SD 1.2, intercept set
for a 10 % event rate. This is only for timing; the study DGPs live elsewhere.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "bench" / "data"
OUT.mkdir(parents=True, exist_ok=True)

P_GRID = [5, 20]
N_GRID = [100, 500, 2000]
REPS = [1, 2, 3]
N_EVAL = 10_000
PREV = 0.10
RHO = 0.3
LP_SD = 1.2


def design(p):
    """Slopes and intercept for a given p (fixed, not random)."""
    beta = 1.0 / np.sqrt(np.arange(1, p + 1))
    sigma = np.full((p, p), RHO) + np.eye(p) * (1 - RHO)
    beta *= LP_SD / np.sqrt(beta @ sigma @ beta)
    # intercept by the exact marginal prevalence on a large fixed draw
    z = np.random.default_rng(12345).multivariate_normal(np.zeros(p), sigma, 200_000) @ beta
    alpha = brentq(lambda a: expit(a + z).mean() - PREV, -10, 5)
    return alpha, beta, sigma


def draw(seed_key, n, p, alpha, beta, sigma):
    rng = np.random.default_rng(np.random.SeedSequence(seed_key))
    x = rng.multivariate_normal(np.zeros(p), sigma, n)
    pt = expit(alpha + x @ beta)
    y = rng.binomial(1, pt)
    df = pd.DataFrame(x, columns=[f"x{j}" for j in range(1, p + 1)])
    df["y"] = y.astype(np.int32)
    df["p_true"] = pt
    return df


def checksum(df, p):
    """Weighted sum both runners can recompute (R reads the same parquet)."""
    x = df[[f"x{j}" for j in range(1, p + 1)]].to_numpy()
    w_row = np.arange(1, len(df) + 1) / len(df)
    w_col = np.arange(1, p + 1)
    return float(w_row @ x @ w_col + (w_row * df["y"].to_numpy()).sum())


def main():
    manifest = {}
    for p in P_GRID:
        alpha, beta, sigma = design(p)
        manifest[f"design_p{p}"] = {"alpha": alpha, "beta": beta.tolist(), "rho": RHO}
        for rep in REPS:
            sets = {f"eval_p{p}_r{rep}": draw((p, 0, rep), N_EVAL, p, alpha, beta, sigma)}
            for n in N_GRID:
                sets[f"train_p{p}_n{n}_r{rep}"] = draw((p, n, rep), n, p, alpha, beta, sigma)
            for name, df in sets.items():
                f = OUT / f"{name}.parquet"
                df.to_parquet(f, index=False)
                manifest[name] = {
                    "file": f.name,
                    "rows": len(df),
                    "events": int(df["y"].sum()),
                    "checksum": checksum(df, p),
                    "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(manifest)} entries to {OUT}")
    for k, v in manifest.items():
        if k.startswith("train"):
            print(k, v["rows"], "rows,", v["events"], "events")


if __name__ == "__main__":
    main()
