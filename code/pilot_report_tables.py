"""Condensed tables for PILOT_REPORT.md, built from the pilot outputs (no hand-typed numbers).

    python pilot_report_tables.py     (after pilot_analyse.py and score.py on results/pilot_tabpfn35)

Writes results/pilot/tables/report_*.md.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
import metrics  # noqa: E402
import pilot_analyse as pa  # noqa: E402

RES = pa.RES
TAB = pa.OUT
SHOW = ["oracle", "lr", "flic", "ridge", "gam", "lgbm", "lgbm_raw", "tabpfn", "tabpfn_legacy", "tabpfn_t1",
        "tabicl", "mitra", "tabpfn35"]
LABEL = {"tabpfn": "TabPFN V2 (T 0.9)", "tabpfn_legacy": "TabPFN V2 legacy seed", "tabpfn_t1": "TabPFN V2, T 1.0",
         "tabpfn35": "TabPFN-3.5 (T 1.0)", "lgbm": "LightGBM + Platt", "lgbm_raw": "LightGBM raw",
         "tabicl": "TabICLv2 (T 0.9)", "mitra": "Mitra-v2", "flic": "FLIC", "lr": "LR-MLE", "gam": "GAM",
         "ridge": "ridge-CV", "oracle": "oracle"}


def ms(x, d=4):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return "-"
    return f"{x.mean():.{d}f} ({x.std(ddof=1) / math.sqrt(len(x)):.{d}f})"


def med(x, d=2):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return "-"
    return f"{np.median(x):.{d}f} ({metrics.mcse_bootstrap(x):.{d}f})"


def load_all_metrics() -> pd.DataFrame:
    m = pa.load_metrics()
    p35 = RES / "pilot_tabpfn35" / "metrics"
    if p35.exists():
        m35 = pd.concat([pd.read_parquet(f) for f in sorted(p35.glob("*.parquet"))], ignore_index=True)
        m35 = m35[m35.arm == "tabpfn"].copy()
        m35["column"] = m35.column.str.replace("tabpfn", "tabpfn35", regex=False)
        m35["arm"] = "tabpfn35"
        m = pd.concat([m, m35], ignore_index=True)
    return m


def oracle_table(m: pd.DataFrame) -> str:
    r0 = m[m.ladder == "R0"]
    out = []
    for sc in pa.SCEN:
        g = r0[r0.scenario == sc]
        for arm in [a for a in SHOW if a in set(g.arm)]:
            a = g[g.arm == arm]
            ok = a[(a.status == "ok") & a.o_true_ici.notna()]
            k = int(ok.cls_weak.sum())
            out.append({"scenario": sc, "arm": LABEL[arm], "R ok/failed": f"{len(ok)}/{len(a) - len(ok)}",
                        "true ICI": ms(ok.o_true_ici), "OCE": ms(ok.o_oce),
                        "b* median": med(ok.o_true_slope), "a* mean": ms(ok.o_true_intercept, 2),
                        "mean p - mean p*": ms(ok.o_true_citl_prob, 4), "AUC (vs p*)": ms(ok.o_true_auc, 3),
                        "weak": (f"{k}/{len(ok)}" if len(ok) else "-")})
    return pa.md_table(pd.DataFrame(out))


def gof_real_table() -> str:
    g = pa.load_gof()
    tests = ["EDGE (G=10)", "Cox recalibration", "Calibration-in-the-large", "Hosmer-Lemeshow (external)",
             "Stukel (offset)", "le Cessie (external)", "Localize (any part named)", "DeepGOF-1 external (combined)"]
    g35 = None
    p = RES / "pilot_tabpfn35" / "gof"
    if p.exists() and any(p.glob("*/*.csv")):
        g35 = pd.concat([pd.read_csv(x, dtype={"rep": str}, keep_default_na=False, na_values=[""])
                         for x in sorted(p.glob("*/*.csv"))], ignore_index=True)
        g35["p_value"] = pd.to_numeric(g35.p_value, errors="coerce")
        g35 = g35[g35.arm == "tabpfn"].assign(arm="tabpfn35")
        g = pd.concat([g, g35], ignore_index=True)
    g = g[(g.ytype == "real") & ~g.arm.str.contains("__")]
    out = []
    for sc in pa.SCEN:
        s = g[g.scenario == sc]
        for arm in [a for a in SHOW if a in set(s.arm)]:
            row = {"scenario": sc, "arm": LABEL[arm]}
            for t in tests:
                k, r, nr = pa.rate(s[(s.arm == arm) & (s.test == t)].p_value)
                row[pa.SHORT[t]] = (f"{k}/{r}" if r else "-") + (f" ({nr} NR)" if nr else "")
            out.append(row)
    return pa.md_table(pd.DataFrame(out))


def fit_table() -> str:
    f = pd.read_csv(TAB / "fits.csv")
    f = f[f.folder.isin(["pilot", "tfm35"])].copy()
    f.loc[f.folder == "tfm35", "arm"] = "tabpfn35"
    out = []
    for sc in pa.SCEN:
        for arm in [a for a in SHOW if a != "oracle"]:
            g = f[(f.scenario == sc) & (f.arm == arm)]
            if g.empty:
                continue
            ok = g[g.status == "ok"]
            nb = int((ok.n_boundary.fillna(0) > 100).sum())
            if int((g.status != "ok").sum()) == 0 and nb == 0:
                continue                       # the report lists only fits with a problem
            out.append({"scenario": sc, "arm": LABEL[arm], "fits": len(g), "failed": int((g.status != "ok").sum()),
                        "degenerate (>100 preds at 0/1)": nb,
                        "median s": f"{ok.seconds.median():.1f}" if len(ok) else "-",
                        "max s": f"{ok.seconds.max():.1f}" if len(ok) else "-",
                        "peak GPU MB": f"{ok.peak_gpu_mb.max():.0f}" if ok.peak_gpu_mb.notna().any() else "-"})
    return pa.md_table(pd.DataFrame(out))


def main() -> int:
    m = load_all_metrics()
    pa.write("report_oracle.md", oracle_table(m))
    pa.write("report_gof_real.md", gof_real_table())
    pa.write("report_fits.md", fit_table())
    print("report tables written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
