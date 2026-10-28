"""GPU memory and time of the three TFMs at (n, p, predicted rows) cells of the random-design tier.

Sets the device rule of pre-registration 3.9: which designs run TabPFN-3.5 on the GPU, and where
Mitra-v2 is recorded as "does not fit in 6 GB GPU memory". Data come from design R0001's setting
with n and p replaced (all continuous predictors, smoke-type seeds); no calibration summary is computed.

    .venv-gpu/Scripts/python.exe code/probe_random_memory.py [arm ...]  -> results/random_memory_probe.csv
"""

import csv
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import numpy as np  # noqa: E402

import dgp_random  # noqa: E402
import models  # noqa: E402

models.configure_environment()
OUT = CODE.parent / "results" / "random_memory_probe.csv"

CELLS = {
    "tabpfn": [(800, 30, 2000), (1500, 20, 5000), (4000, 8, 2000), (1300, 30, 2000), (2000, 20, 2000),
               (4000, 12, 5000), (2500, 30, 2000)],
    "mitra": [(1000, 30, 2000), (2500, 15, 2000), (3500, 8, 2000), (1500, 30, 2000), (2000, 20, 2000),
              (4000, 6, 5000), (4000, 10, 2000)],
    "tabicl": [(4000, 30, 5000), (2000, 30, 2000)],
}


def context(n, p, n_pred, seed=1):
    s = dgp_random.draw_setting(1)
    s.update(p=p, kinds=["cont"] * p, columns=[f"x{j}" for j in range(1, p + 1)],
             beta={f"x{j}": 0.5 for j in range(1, p + 1)}, cat_levels={}, cat_effects={}, bin_prev={})
    rng = np.random.default_rng(np.random.SeedSequence([dgp_random.RANDOM_STREAM, 9, n, p, seed]))
    X = dgp_random.draw_covariates(s, n + n_pred, rng)
    eta = -2.0 + X[:, :4].sum(axis=1) * 0.5
    y = (rng.random(n + n_pred) < 1 / (1 + np.exp(-eta))).astype(np.int8)
    return models.FitContext(X_dev=X[:n], y_dev=y[:n], X_pred=X[n:], p_star_pred=np.zeros(n_pred),
                             columns=s["columns"], cont=s["columns"], bin=[], cat=[], cat_levels={},
                             seeds={a: 11 for a in models.ARM_CODES}, device="cuda", threads=6)


def main():
    arms = sys.argv[1:] or list(CELLS)
    fitters = {"tabpfn": models.fit_tabpfn, "mitra": models.fit_mitra, "tabicl": models.fit_tabicl}
    rows = []
    for arm in arms:
        for n, p, npred in CELLS[arm]:
            ctx = context(n, p, npred)
            t0 = time.perf_counter()
            res = fitters[arm](ctx)[0]
            err = str(res.info.get("error", ""))
            rows.append(dict(arm=arm, n=n, p=p, n_pred=npred, n_times_p=n * p, status=res.status,
                             seconds=round(time.perf_counter() - t0, 1), peak_gpu_mb=res.info.get("peak_gpu_mb"),
                             oom="out of memory" in err.lower() or "cut its context" in err, error=err[:160]))
            print(rows[-1], flush=True)
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
    new = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
