"""Pilot extra: what would Mitra cost on the CPU in the cell where the GPU runs out of memory?

    .venv-gpu/Scripts/python.exe code/pilot_mitra_cpu.py

Fits Mitra through the engine's own wrapper (models.fit_mitra) on the CPU, predicting only the
first 100 evaluation rows, for T2_pi10_n3000 rep 1 (fails on the GPU) and S2_pi10_n500 rep 1
(works on the GPU, so CPU and GPU predictions can be compared). Prints seconds and differences.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REBUILD = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import models  # noqa: E402

models.configure_environment(threads=6)

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import dgp  # noqa: E402

ROWS = 100


def ctx_for(name: str, rep: int, rows: int) -> models.FitContext:
    sc = dgp.get_scenario(name)
    r = dgp.make_replicate(sc["scenario_id"], rep, smoke=True)
    return models.FitContext(
        X_dev=r.dev.X, y_dev=r.dev.y, X_pred=r.eval.X[:rows], p_star_pred=r.eval.p_star[:rows],
        columns=sc["columns"], cont=sc["cont_cols"], bin=sc["bin_cols"], cat=sc["cat_cols"],
        cat_levels=sc["cat_levels"], seeds={a: r.child_seed(4, models.ARM_CODES[a]) for a in models.ARM_CODES},
        device="cpu", threads=6)


def main() -> int:
    for name in ("S2_pi10_n500", "T2_pi10_n3000"):
        ctx = ctx_for(name, 1, ROWS)
        t0 = time.perf_counter()
        res = models.fit_mitra(ctx)[0]
        secs = time.perf_counter() - t0
        line = f"{name} rep 1 cpu: status {res.status}, {secs:.1f} s for {ROWS} rows " \
               f"(fit {res.info.get('fit_seconds', float('nan')):.1f} s, predict {res.info.get('predict_seconds', float('nan')):.1f} s)"
        if res.status == "ok":
            stored = pq.read_table(REBUILD / "results" / "pilot" / "preds" / name / "0001.parquet").to_pandas()
            if "p_mitra" in stored:
                g = stored.loc[stored["set"] == "eval", "p_mitra"].to_numpy(float)[:ROWS]
                d = np.abs(res.p - g)
                line += f"; |cpu - gpu| max {d.max():.2e}, mean {d.mean():.2e}"
        else:
            line += f"; error {res.info.get('error', '')[:150]}"
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
