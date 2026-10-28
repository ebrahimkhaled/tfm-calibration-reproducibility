"""Prediction instability on the probe set (pre-registration 4.4, exploratory) and the oracle
reference of the localization summary (11, "Localization").

    python analysis/instability_probe.py [--workers 4]

1. Instability. Every scenario has a fixed probe set of 500 rows, the same in every replicate.
   For each arm and probe row, the spread of the predicted risk across replicates is summarised by
   its standard deviation and by the width of its 2.5-97.5 % range (Riley and Collins 2023). Every
   arm uses replicates 1 to R of the TFM schedule (200, or 150 at n = 3000), so all arms are
   compared on the same development sets. A failed fit (no prediction) is left out of that row's
   spread. Per scenario and arm the file reports the mean, median and 90th percentile over the 500
   rows of the per-row standard deviation, and the mean range width.
   -> tables/summary/instability.csv

2. Localization reference (S3 and T3 only). For each replicate and arm, on the evaluation rows:
   the share of the variance of logit(p*) that a smooth function of logit(p-hat) does not explain.
   The smooth is a cubic regression spline (truncated-power basis, interior knots at the quartiles
   of logit p-hat) fitted by least squares. A large share means that the arm's error depends on
   the covariates and not only on its own prediction, so a covariate-level (COV) error exists.
   -> tables/summary/localization_reference.csv (mean over replicates, with MCSE)

Reads results/preds only; writes only under analysis/.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import _common as C

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

ARMS = ["lr", "flic", "ridge", "gam", "lgbm", "tabpfn", "tabicl", "mitra"]
REF_FAMILIES = ("S3", "T3")


def _logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def unexplained_share(lp_hat: np.ndarray, lp_star: np.ndarray) -> float:
    ok = np.isfinite(lp_hat) & np.isfinite(lp_star)
    x, y = lp_hat[ok], lp_star[ok]
    if len(x) < 50 or np.var(y) <= 0:
        return float("nan")
    s = (x - x.mean()) / (x.std() or 1.0)
    knots = np.quantile(s, [0.25, 0.5, 0.75])
    X = np.column_stack([np.ones_like(s), s, s ** 2, s ** 3] + [np.clip(s - k, 0, None) ** 3 for k in knots])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return float(np.var(res) / np.var(y))


CACHE = C.TABLES / "cache" / "instability"


def one_scenario(scen: str) -> tuple[list[dict], list[dict]]:
    """Cached per scenario, so an interrupted pass resumes where it stopped."""
    import json
    cf = CACHE / f"{scen}.json"
    if cf.exists():
        d = json.loads(cf.read_text())
        return d["inst"], d["ref"]
    inst, ref = _one_scenario(scen)
    CACHE.mkdir(parents=True, exist_ok=True)
    C.assert_writable(cf)
    cf.write_text(json.dumps({"inst": inst, "ref": ref}))
    return inst, ref


def _one_scenario(scen: str) -> tuple[list[dict], list[dict]]:
    info = C.parse_scenario(scen)
    R = C.r_tfm(info["n"])
    folder = C.PREDS / scen
    probe = {a: [] for a in ARMS}
    ref = {a: [] for a in ARMS}
    want_ref = info["family"] in REF_FAMILIES
    n_files = 0
    for r in range(1, R + 1):
        # an arm run in a separate stream (e.g. TabPFN-3.5 on CPU workers) sits in a subfolder
        # with the same replicate file name; merge every part on (set, row)
        files = sorted(folder.rglob(f"{r:04d}.parquet"))
        if not files:
            continue
        d = None
        for f in files:
            names = set(pq.read_schema(f).names)
            want = [f"p_{a}" for a in ARMS if f"p_{a}" in names and (d is None or f"p_{a}" not in d.columns)]
            if d is None:
                d = pq.read_table(f, columns=["set", "row", "p_star"] + want).to_pandas()
            elif want:
                d = d.merge(pq.read_table(f, columns=["set", "row"] + want).to_pandas(), on=["set", "row"], how="left")
        n_files += 1
        pr = d[d["set"] == "probe"].sort_values("row")
        ev = d[d["set"] == "eval"]
        for a in ARMS:
            c = f"p_{a}"
            if c not in d.columns:
                probe[a].append(np.full(len(pr), np.nan))
                continue
            probe[a].append(pr[c].to_numpy(dtype=float))
            if want_ref:
                v = ev[c].to_numpy(dtype=float)
                if np.isfinite(v).all():
                    ref[a].append(unexplained_share(_logit(v), _logit(ev["p_star"].to_numpy(dtype=float))))
    inst_rows, ref_rows = [], []
    for a in ARMS:
        if not probe[a]:
            continue
        M = np.vstack(probe[a])                      # replicates x probe rows
        usable = np.isfinite(M).all(axis=1) if M.size else np.array([], bool)
        k = int(np.isfinite(M).any(axis=1).sum())
        if k < 10:
            continue
        sd = np.nanstd(M, axis=0, ddof=1)
        lo, hi = np.nanpercentile(M, 2.5, axis=0), np.nanpercentile(M, 97.5, axis=0)
        inst_rows.append(dict(scenario=scen, family=info["family"], pi=info["pi"], n=info["n"], arm=a,
                              R=k, R_complete=int(usable.sum()), mean_sd=float(np.mean(sd)),
                              median_sd=float(np.median(sd)), p90_sd=float(np.percentile(sd, 90)),
                              mean_range=float(np.mean(hi - lo)), n_probe=M.shape[1]))
        if want_ref and ref[a]:
            v = np.array([x for x in ref[a] if np.isfinite(x)])
            if len(v):
                ref_rows.append(dict(scenario=scen, family=info["family"], pi=info["pi"], n=info["n"], arm=a,
                                     R=len(v), mean_share=float(v.mean()),
                                     mcse=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")))
    C.progress(f"{scen}: {n_files} files")
    return inst_rows, ref_rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    scens = C.all_scenarios()
    inst, ref = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in ex.map(one_scenario, scens):
            inst += i
            ref += r
    out = C.TABLES / "summary"
    C.assert_writable(out)
    pd.DataFrame(inst).to_csv(out / "instability.csv", index=False)
    pd.DataFrame(ref).to_csv(out / "localization_reference.csv", index=False)
    C.progress("written: tables/summary/instability.csv, tables/summary/localization_reference.csv")


if __name__ == "__main__":
    main()
