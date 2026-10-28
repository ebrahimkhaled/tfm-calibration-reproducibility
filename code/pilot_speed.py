"""Pilot side experiment: can TabPFN run faster on this GPU without changing its predictions?

    .venv-gpu/Scripts/python.exe code/pilot_speed.py

The engine calls TabPFN with the package defaults (memory_saving_mode "auto", one preprocessing
job, fit_mode "fit_preprocessors"). On a 6 GB card the GPU sat well below full use in the review,
so part of the time may be CPU-side preprocessing or memory-saving batching. Each variant below is
fitted on pilot replicates (smoke stream) and compared with the stored engine prediction. A variant
is only useful if it is faster AND gives the same predictions to float32 precision.

Writes results/pilot/tables/tabpfn_speed.csv. Nothing in results/pilot/preds is changed.
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
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import dgp  # noqa: E402

CASES = [("S1_pi03_n200", 1), ("S1_pi03_n200", 2), ("S2_pi10_n500", 1), ("S2_pi10_n500", 2),
         ("S4_pi03_n3000", 1), ("S4_pi03_n3000", 2), ("T2_pi10_n3000", 1)]
VARIANTS = {
    "default": {},
    "no_memory_saving": dict(memory_saving_mode=False),
    "prep_jobs_6": dict(n_preprocessing_jobs=6),
    "fit_with_cache": dict(fit_mode="fit_with_cache"),
    "no_mem_save+prep6": dict(memory_saving_mode=False, n_preprocessing_jobs=6),
}


def main() -> int:
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(6)
    ver, path = models.resolve_tabpfn("tabpfn")
    rows = []
    warm = True
    for name, rep in CASES:
        sc = dgp.get_scenario(name)
        r = dgp.make_replicate(sc["scenario_id"], rep, smoke=True)
        probe = dgp.make_probe(sc["scenario_id"])
        X_pred = np.vstack([r.eval.X, r.recal.X, probe.X])
        cols = sc["columns"]
        cat_idx = [cols.index(c) for c in sc["bin_cols"] + sc["cat_cols"]]
        seed = r.child_seed(4, models.ARM_CODES["tabpfn"])
        stored = pq.read_table(REBUILD / "results" / "pilot" / "preds" / name / f"{rep:04d}.parquet").to_pandas()
        ref = stored["p_tabpfn"].to_numpy(float)
        if warm:   # one throwaway fit so that CUDA start-up is not charged to the first variant
            TabPFNClassifier.create_default_for_version(
                getattr(ModelVersion, ver), model_path=str(path), device="cuda", random_state=seed,
                categorical_features_indices=cat_idx or None).fit(r.dev.X, r.dev.y.astype(int)).predict_proba(X_pred[:50])
            warm = False
        for vname, extra in VARIANTS.items():
            models._gpu_peak_reset("cuda")
            t0 = time.perf_counter()
            try:
                clf = TabPFNClassifier.create_default_for_version(
                    getattr(ModelVersion, ver), model_path=str(path), device="cuda", random_state=seed,
                    categorical_features_indices=cat_idx or None, **extra)
                clf.fit(r.dev.X, r.dev.y.astype(int))
                t_fit = time.perf_counter() - t0
                p = models._positive_column(clf.predict_proba(X_pred), clf.classes_)
                secs = time.perf_counter() - t0
                d = np.abs(p.astype(np.float32).astype(float) - ref)
                rows.append(dict(scenario=name, rep=rep, variant=vname, seconds=round(secs, 2),
                                 fit_seconds=round(t_fit, 2), peak_mb=round(models._gpu_peak_mb("cuda"), 0),
                                 max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()), error=""))
            except Exception as e:  # noqa: BLE001
                rows.append(dict(scenario=name, rep=rep, variant=vname, error=f"{type(e).__name__}: {e}"[:200]))
            print(rows[-1], flush=True)
            torch.cuda.empty_cache()
    out = REBUILD / "results" / "pilot" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "tabpfn_speed.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
