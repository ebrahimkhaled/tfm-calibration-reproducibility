"""Time TabPFN-3.5 standard against TabPFN-3.5-Fast on the GPU (pre-registration 5.2: which
checkpoint the main tabpfn arm uses). Smoke seed stream, the run's stacked prediction rows
(evaluation + recalibration + probe). Time and memory only; no calibration summary is computed.

    .venv-gpu/Scripts/python.exe code/time_tabpfn35_fast.py  -> results/tabpfn35_fast_timing.csv
"""

import csv
import os
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import numpy as np  # noqa: E402

import dgp  # noqa: E402
import models  # noqa: E402

models.configure_environment()
CELLS = ["S1_pi03_n200", "S2_pi10_n500", "S3_pi30_n1000", "S4_pi03_n3000"]
REPS = [1, 2]
out = CODE.parent / "results" / "tabpfn35_fast_timing.csv"
rows = []
for version in ("V3_5", "V3_5_FAST"):
    os.environ["TFMCAL_TABPFN_VERSION"] = version
    for name in CELLS:
        probe = dgp.make_probe(name)
        for rep in REPS:
            r = dgp.make_replicate(name, rep, smoke=True)
            sc = r.scenario
            ctx = models.FitContext(
                X_dev=r.dev.X, y_dev=r.dev.y, X_pred=np.vstack([r.eval.X, r.recal.X, probe.X]),
                p_star_pred=np.zeros(1), columns=sc["columns"], cont=sc["cont_cols"], bin=sc["bin_cols"],
                cat=sc["cat_cols"], cat_levels=sc["cat_levels"],
                seeds={a: r.child_seed(4, c) for a, c in models.ARM_CODES.items()}, device="cuda", threads=6)
            t0 = time.perf_counter()
            res = models.fit_tabpfn(ctx)[0]
            rows.append(dict(version=version, scenario=name, rep=rep, status=res.status,
                             seconds=round(time.perf_counter() - t0, 2), rows=len(ctx.X_pred),
                             checkpoint=res.info.get("checkpoint"), sha256=res.info.get("checkpoint_sha256"),
                             temperature=res.info.get("softmax_temperature"),
                             peak_gpu_mb=res.info.get("peak_gpu_mb"), error=res.info.get("error", "")[:200]))
            print(rows[-1], flush=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote", out)
