"""Oracle and empirical estimands of every stored prediction column (pre-registration section 4).

    python score.py --scenarios all [--smoke] [--n-workers 4]

Reads results/preds/<scenario>/**/<rep>.parquet and writes one row per (replicate, column) to
results/metrics/<scenario>.parquet: the arm (or arm__ladder level), oracle metrics, empirical
metrics on Y and the calibration classes. A scenario is rescored only when one of its prediction
files is newer than its metrics file. Every replicate's stored outcomes are checked against the
regenerated data (fingerprint) before scoring.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# one BLAS/OpenMP thread per worker: with the default (one per core) every worker reserves its own
# thread buffers and 20 workers ran out of memory on 2026-10-14; one thread holds about 0.13 GB
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import dgp  # noqa: E402
import metrics  # noqa: E402
from models import USABLE_STATUSES as USABLE  # noqa: E402
from run import META_KEY, parse_scenarios  # noqa: E402

REBUILD = CODE_DIR.parent


def score_file(f: Path, check: bool = True) -> list[dict]:
    t = pq.read_table(f)
    meta = json.loads(t.schema.metadata[META_KEY])
    df = t.to_pandas()
    ev = df[df["set"] == "eval"]
    y = ev["y"].to_numpy(dtype=float)
    ps = ev["p_star"].to_numpy(dtype=float)
    if check:
        r = dgp.make_replicate(meta["scenario_id"], meta["rep"], smoke=meta["smoke"])
        if not (np.array_equal(r.eval.y, y.astype(np.int8)) and np.array_equal(r.eval.p_star, ps)):
            raise AssertionError(f"{f}: stored evaluation data differ from the regenerated replicate")
    # a file in a tag sub-folder carries the same p_star as the main file: score the oracle once
    tag = "" if f.parent.name == meta["scenario"] else f.parent.name
    cols = {} if tag and (f.parent.parent / f.name).exists() else {"oracle": ps}
    for c in ev.columns:
        if c.startswith("p_") and c != "p_star":
            cols[c[2:]] = ev[c].to_numpy(dtype=float)
    pi = float(dgp.get_scenario(meta["scenario"])["pi"])
    rows = []
    # failed arms and failed ladder levels have no column; they still get a row, so that
    # degenerate-fit counts per scenario come out of the same table (never silently dropped)
    for arm, info in meta["arms"].items():
        if info.get("status") not in USABLE and f"p_{arm}" not in ev.columns:
            rows.append(dict(scenario=meta["scenario"], rep=meta["rep"], tag=tag, column=arm, arm=arm,
                             ladder="R0", status=info.get("status"), error=str(info.get("error", ""))[:300]))
    for arm, levels in meta.get("ladder", {}).items():
        for lev_m, info in levels.items():
            if info.get("status") != "ok":
                rows.append(dict(scenario=meta["scenario"], rep=meta["rep"], tag=tag,
                                 column=f"{arm}__{lev_m}", arm=arm, ladder=lev_m, status="failed",
                                 error=str(info.get("error", ""))[:300]))
    for name, p in cols.items():
        arm, _, level = name.partition("__")
        ai = meta["arms"].get(arm, {})
        row = dict(scenario=meta["scenario"], rep=meta["rep"], tag=tag, column=name, arm=arm,
                   ladder=level or "R0", status=ai.get("status", "ok"),
                   # degenerate fits (pre-registration 8.2) are scored like any other, and counted;
                   # a ladder column inherits the label of the arm it recalibrates
                   degenerate=ai.get("status") == "degenerate",
                   degenerate_reason=ai.get("degenerate_reason"),
                   platt_fallback=ai.get("platt_fallback"),
                   lambda_at_edge=ai.get("lambda_at_edge"),
                   # which weights produced the column, so a mix of versions under one arm shows
                   model=ai.get("model_version") or ai.get("checkpoint") or ai.get("revision"))
        if np.any(~np.isfinite(p)):
            row["status"] = "missing"
            rows.append(row)
            continue
        om = metrics.oracle_metrics(p, ps)
        em = metrics.empirical_metrics(y, p)
        row.update({f"o_{k}": v for k, v in om.items()})
        row.update({f"e_{k}": v for k, v in em.items()})
        ps_ = metrics.probability_scale(pi)
        row.update(metrics.hierarchy_class(om, prob_scale=ps_))
        row.update({f"{k}_ll": v for k, v in metrics.hierarchy_class(om, prefix="true_ll_", prob_scale=ps_).items()})
        # the same classes at half and double every threshold (pre-registered sensitivity)
        for tag_, sc_ in (("half", 0.5), ("double", 2.0)):
            row.update({f"{k}_{tag_}": v for k, v in metrics.hierarchy_class(om, scale=sc_, prob_scale=ps_).items()})
        rows.append(row)
    return rows


def score_scenario(args) -> str:
    scenario, pred_dir, met_dir, force = args
    files = sorted((pred_dir / scenario).rglob("*.parquet"))
    if not files:
        return f"{scenario}: no prediction files"
    out = met_dir / f"{scenario}.parquet"
    if out.exists() and not force and out.stat().st_mtime > max(f.stat().st_mtime for f in files):
        return f"{scenario}: up to date"
    rows = []
    for f in files:
        rows += score_file(f)
    met_dir.mkdir(parents=True, exist_ok=True)
    tab = pd.DataFrame(rows)
    tab.to_parquet(out, index=False)
    r0 = tab[tab["ladder"] == "R0"] if "ladder" in tab else tab
    deg = int(r0["status"].eq("degenerate").sum()) if "status" in r0 else 0
    fail = int(r0["status"].eq("failed").sum()) if "status" in r0 else 0
    return f"{scenario}: {len(files)} file(s), {len(rows)} rows ({deg} degenerate, {fail} failed fits) -> {out}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--preds", default=None, help="prediction folder (default results[/smoke]/preds)")
    ap.add_argument("--out", default=None, help="metrics folder (default next to --preds, 'metrics')")
    a = ap.parse_args(argv)
    base = REBUILD / "results" / ("smoke" if a.smoke else "")
    pred_dir = Path(a.preds) if a.preds else base / "preds"
    met_dir = Path(a.out) if a.out else pred_dir.parent / "metrics"
    jobs = [(s, pred_dir, met_dir, a.force) for s in parse_scenarios(a.scenarios)
            if (pred_dir / s).exists()]
    if a.n_workers > 1:
        with ProcessPoolExecutor(a.n_workers) as ex:
            for msg in ex.map(score_scenario, jobs):
                print(msg, flush=True)
    else:
        for j in jobs:
            print(score_scenario(j), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
