"""Calibration audit of any model's predictions in one call, from Python (pre-registration 14, b).

Takes validation outcomes y and predicted risks p from any model (a tabular foundation model, a
logistic regression, a gradient-boosted model ...), optionally with the validation covariates, and
runs the study's recipe in R: ebrahim.gof::run.all.external() and localize.external(), through
R/audit_calibration.R. No rpy2 is needed: the data go to R through a temporary csv file.

    python code/audit_calibration.py data.csv [--out audit.csv] [--seed 1]
        data.csv has columns y and p, and optionally covariates named x...
    python code/audit_calibration.py --y y.npy --p p.npy [--x X.npy] [--out audit.csv]

From Python code:
    from audit_calibration import audit
    table = audit(y, p, X=None)          # a pandas DataFrame, one row per test

Needs R (Rscript on the PATH, or RSCRIPT set) with ebrahim.gof >= 2.10.0 installed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

R_RECIPE = Path(__file__).resolve().parents[1] / "R" / "audit_calibration.R"


def _rscript() -> str:
    r = os.environ.get("RSCRIPT") or shutil.which("Rscript")
    if not r:
        raise RuntimeError("Rscript not found: install R and ebrahim.gof (>= 2.10.0), or set RSCRIPT")
    return r


def audit(y, p, X=None, seed: int = 1, M: int = 999, G: int = 10) -> pd.DataFrame:
    """Run the battery and the localization on (y, p[, X]); returns one row per test."""
    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y has {y.size} values and p has {p.size}")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must be 0/1")
    d = pd.DataFrame({"y": y, "p": p})
    if X is not None:
        Xd = pd.DataFrame(np.asarray(X, dtype=float))
        Xd.columns = [f"x{j + 1}" for j in range(Xd.shape[1])]
        d = pd.concat([d, Xd], axis=1)
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "in.csv", Path(tmp) / "out.csv"
        d.to_csv(src, index=False, float_format="%.17g")
        r = subprocess.run([_rscript(), str(R_RECIPE), str(src), str(out), "--seed", str(seed),
                            "--M", str(M), "--G", str(G)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"the R recipe failed:\n{r.stderr[-2000:]}")
        return pd.read_csv(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", help="csv with columns y, p and optional x...")
    ap.add_argument("--y")
    ap.add_argument("--p")
    ap.add_argument("--x")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--M", type=int, default=999)
    ap.add_argument("--G", type=int, default=10)
    a = ap.parse_args(argv)
    if a.data:
        d = pd.read_csv(a.data)
        d.columns = [c.lower() for c in d.columns]
        xc = [c for c in d.columns if c.startswith("x")]
        y, p, X = d["y"], d["p"], (d[xc] if xc else None)
    elif a.y and a.p:
        y, p = np.load(a.y), np.load(a.p)
        X = np.load(a.x) if a.x else None
    else:
        ap.error("give a csv file, or --y and --p")
    t = audit(y, p, X, seed=a.seed, M=a.M, G=a.G)
    if a.out:
        t.to_csv(a.out, index=False)
    with pd.option_context("display.width", 160, "display.max_colwidth", 60):
        print(t[["test", "statistic", "df", "p_value", "note"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
