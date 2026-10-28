"""Computing cost of the study, read from the run's own records (nothing is estimated or typed).

    python analysis/compute_cost.py [--no-gof]   -> analysis/tables/compute/*.csv

Sources
  fits.parquet (analysis/collect.py)   seconds of every fit + prediction in the main run and the
                                       random-design tier, as stored in each prediction file's
                                       metadata (code/run.py; R arms timed inside R, without the
                                       Rscript start-up; LightGBM and LightGBM-raw share one timing).
  results/gof*/**/*.csv.gz             'secs' of every call of the test layer (R/gof_layer.R): the
  results/realdata/<key>/gof/          battery (run.all.external), edge.gof, localize.external and
                                       deepgof1.external; one value per call.
  logs/realdata/fit_*.jsonl            per-replicate arm seconds of the real-data run.
  logs/main/*.out, logs/realdata/*.out start and end times of the streams (wall clock).

Outputs (analysis/tables/compute/)
  fit_seconds_by_n.csv   median, quartiles and 90th percentile of seconds per fit, by tier, arm,
                         device and n (main run and random tier)
  fit_hours.csv          total fit hours by run, arm and device
  gof_hours.csv          total hours and median seconds per call of the test layer, by run and block
  wall_clock.csv         first and last time stamp of each run
Runs at below-normal priority with one thread (analysis/_common.py).
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import _common as C  # noqa: F401  (priority and thread limits)

import numpy as np
import pandas as pd

ANALYSIS = Path(__file__).resolve().parent
REB = ANALYSIS.parent
OUT = ANALYSIS / "tables" / "compute"
OUT.mkdir(parents=True, exist_ok=True)

MAIN_ARMS = ["lr", "flic", "ridge", "gam", "lgbm", "tabpfn", "tabicl", "mitra",
             "tabpfn_t1", "tabicl_t1", "tabpfn_legacy"]


def q(s, p):
    return float(np.quantile(s, p)) if len(s) else np.nan


def fit_tables():
    f = pd.read_parquet(REB / "analysis" / "tables" / "fits.parquet",
                        columns=["scenario", "rep", "tag", "arm", "status", "device", "seconds", "n", "pi", "tier"])
    f = f[f["arm"].isin(MAIN_ARMS) & f["status"].isin(["ok", "degenerate", "failed"])]
    f["device"] = f["device"].fillna("cpu")
    f["stress"] = f["scenario"].str.startswith("T")
    rows = []
    for (tier, arm, dev, n), g in f[f["tier"] != "random"].groupby(["tier", "arm", "device", "n"]):
        s = g["seconds"].dropna().to_numpy()
        rows.append(dict(tier=tier, arm=arm, device=dev, n=int(n), fits=len(s), median=q(s, 0.5),
                         q25=q(s, 0.25), q75=q(s, 0.75), p90=q(s, 0.9), hours=s.sum() / 3600))
    # pooled over n for the random tier (n is continuous there) and for T2 at n = 3000
    for (tier, arm, dev), g in f[f["tier"] == "random"].groupby(["tier", "arm", "device"]):
        s = g["seconds"].dropna().to_numpy()
        rows.append(dict(tier="random_all", arm=arm, device=dev, n=-1, fits=len(s), median=q(s, 0.5),
                         q25=q(s, 0.25), q75=q(s, 0.75), p90=q(s, 0.9), hours=s.sum() / 3600))
    for sc in ["T2_pi10_n3000", "T2_pi10_n500"]:
        for (arm, dev), g in f[f["scenario"] == sc].groupby(["arm", "device"]):
            s = g["seconds"].dropna().to_numpy()
            rows.append(dict(tier=sc, arm=arm, device=dev, n=int(g["n"].iloc[0]), fits=len(s), median=q(s, 0.5),
                             q25=q(s, 0.25), q75=q(s, 0.75), p90=q(s, 0.9), hours=s.sum() / 3600))
    pd.DataFrame(rows).to_csv(OUT / "fit_seconds_by_n.csv", index=False)
    f["run"] = np.where(f["tier"] == "random", "random", "main")
    h = (f.groupby(["run", "arm", "device"])["seconds"].agg(fits="size", hours=lambda s: s.sum() / 3600)
         .reset_index())
    return h


def realdata_fit_hours():
    rows = []
    for p in sorted((REB / "logs" / "realdata").glob("fit_*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") != "written":
                continue
            for arm, s in (r.get("arm_seconds") or {}).items():
                if s is None or arm == "lgbm_raw" or arm == "oracle":
                    continue
                if arm not in r.get("arms_run", [arm]):
                    continue
                dev = "cuda" if "_gpu_" in p.name else "cpu"
                rows.append(dict(setting=r.get("setting"), rep=r.get("rep"), arm=arm, device=dev, seconds=float(s)))
    d = pd.DataFrame(rows)
    d = d[d["seconds"] > 0]          # 0 s = recorded as not run
    # hours: every attempt that wrote a record (refits included); fits: distinct (setting, replicate, arm)
    h = d.groupby(["arm", "device"]).agg(fits=("seconds", "size"), hours=("seconds", lambda s: s.sum() / 3600),
                                         distinct=("setting", lambda s: len(set(zip(s, d.loc[s.index, "rep"])))))
    h = h.reset_index()
    h.insert(0, "run", "realdata")
    return h


def block_of(test: str, family: str) -> str:
    if family == "Localize":
        return "localize.external"
    if family == "DeepGOF":
        return "deepgof1.external"
    if test.startswith("edge.gof"):
        return "edge.gof"
    return "run.all.external"


def read_gof_dir(job):
    """All timed calls in one folder of test-layer files (one scenario or one real-data setting)."""
    C.lower_priority()
    run, folder = job
    parts = []
    for p in Path(folder).rglob("*.csv.gz"):
        try:
            d = pd.read_csv(p, usecols=lambda c: c in ("scenario", "n_eval", "test", "family", "secs", "arm"))
        except Exception:  # noqa: BLE001
            continue
        d = d[d["secs"].notna()]
        if not len(d):
            continue
        d = d.assign(run=run, block=[block_of(str(t), str(f)) for t, f in zip(d["test"], d["family"])])
        if run == "main" and "scenario" in d:
            d["run"] = np.where(d["scenario"].astype(str).str.match(r"^R\d+$"), "random", "main")
        parts.append(d[["run", "block", "n_eval", "secs"]])
    return pd.concat(parts, ignore_index=True) if parts else None


def gof_tables(workers=4):
    from concurrent.futures import ProcessPoolExecutor
    roots = [("main", REB / "results" / "gof"), ("main_subsamples", REB / "results" / "gof_e50"),
             ("main_subsamples", REB / "results" / "gof_e200")]
    roots += [("realdata", p / "gof") for p in sorted((REB / "results" / "realdata").iterdir())
              if (p / "gof").is_dir()]
    jobs = [(run, str(d)) for run, root in roots for d in sorted(root.iterdir()) if d.is_dir()]
    parts = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, d in enumerate(ex.map(read_gof_dir, jobs, chunksize=4), 1):
            if d is not None:
                parts.append(d)
            if i % 100 == 0:
                print(f"gof folders read: {i} of {len(jobs)}", flush=True)
    g = pd.concat(parts, ignore_index=True)
    out = (g.groupby(["run", "block"])["secs"]
           .agg(calls="size", hours=lambda s: s.sum() / 3600, median="median",
                q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)).reset_index())
    out.to_csv(OUT / "gof_hours.csv", index=False)
    print(f"gof: {len(jobs)} folders, {len(g)} timed calls", flush=True)


STAMP = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")


def wall_clock():
    rows = []
    for run, folder in (("main", REB / "logs" / "main"), ("realdata", REB / "logs" / "realdata")):
        stamps = []
        for p in folder.glob("*.out"):
            if p.stem == "digest":      # the results digest is analysis, not part of the run
                continue
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                m = STAMP.match(line)
                if m:
                    stamps.append(pd.Timestamp(m.group(1)))
        if run == "realdata":   # the real-data .out files start at the stream launch; the first fit log is earlier
            for p in folder.glob("fit_*.jsonl"):
                m = re.search(r"(\d{8}_\d{6})", p.name)
                if m:
                    stamps.append(pd.to_datetime(m.group(1), format="%Y%m%d_%H%M%S"))
        rows.append(dict(run=run, start=min(stamps), end=max(stamps),
                         hours=(max(stamps) - min(stamps)).total_seconds() / 3600))
    pd.DataFrame(rows).to_csv(OUT / "wall_clock.csv", index=False)


def main():
    h = fit_tables()
    h = pd.concat([h, realdata_fit_hours()], ignore_index=True)
    h.to_csv(OUT / "fit_hours.csv", index=False)
    wall_clock()
    if "--no-gof" not in sys.argv:      # the test-layer pass reads about 45,000 files
        gof_tables()
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
