"""Results digest of the real-data stream (pre-registration Section 10): one Markdown file with every
number a writer needs.

    python analysis/realdata_digest.py            -> analysis/tables/realdata/REALDATA_DIGEST.md

Inputs: the per-replicate metrics in results/realdata/<key>/metrics/ (code/run_realdata.py score),
the summary tables in analysis/tables/realdata/ (code/run_realdata.py summarise), the clean-table
meta files in data/clean/, and, for the comparison with the simulation, analysis/tables/summary/
arm_summaries.parquet. Real-data results are descriptive and supportive (10.5); intervals are
95 % Monte Carlo intervals over the B (or R) development subsamples, so they describe the
variability of model development on one external sample, not sampling uncertainty of that sample.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS = Path(__file__).resolve().parent
REBUILD = ANALYSIS.parent
sys.path.insert(0, str(REBUILD / "code"))
import run_realdata as R  # noqa: E402

TABLES = R.TABLES
OUT = TABLES / "REALDATA_DIGEST.md"
KEYS = R.KEYS
LABEL = {"ist_death14": "IST, death at 14 days", "ist_dd6": "IST, dead or dependent at 6 months",
         "crash2_death28": "CRASH-2, death at 28 days", "sepsis_death": "Sepsis (UCI 827), death",
         "physionet_ihd": "PhysioNet 2012, in-hospital death"}
ARMS = ["lr", "flic", "ridge", "gam", "lgbm", "lgbm_raw", "tabpfn", "tabicl", "mitra"]
TFMS = ["tabpfn", "tabicl", "mitra"]
ARM_LABEL = {"oracle": "Oracle", "lr": "LR-MLE", "flic": "FLIC", "ridge": "Ridge", "gam": "GAM",
             "lgbm": "LightGBM+Platt", "lgbm_raw": "LightGBM raw", "tabpfn": "TabPFN-3.5",
             "tabicl": "TabICLv2", "mitra": "Mitra-v2"}
NS = [200, 500, 1000, 3000]
NL2 = chr(10) * 2
TESTS = [("EDGE (G=10)", "EDGE10"), ("Cox recalibration", "Cox"), ("Calibration-in-the-large", "CITL"),
         ("Spiegelhalter-z", "Spieg"), ("Hosmer-Lemeshow (external)", "HL"), ("Stukel (offset)", "Stukel"),
         ("le Cessie (external)", "leCessie"), ("DeepGOF-1 external", "DeepGOF")]


# ----------------------------------------------------------------------------- helpers

def f(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:.{d}f}"


def mean_ci(x, d=3):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if len(x) < 2:
        return f(x.mean() if len(x) else np.nan, d)
    m, h = x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    return f"{m:.{d}f} [{m - h:.{d}f}, {m + h:.{d}f}]"


def med_iqr(x, d=2):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if not len(x):
        return "-"
    q = np.quantile(x, [0.25, 0.5, 0.75])
    return f"{q[1]:.{d}f} ({q[0]:.{d}f}, {q[2]:.{d}f})"


def table(rows: list[list], head: list[str]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def load_metrics() -> pd.DataFrame:
    parts = []
    for key in KEYS:
        for p in sorted((R.RESULTS / key / "metrics").glob("*.parquet")):
            parts.append(pd.read_parquet(p))
    if not parts:
        return pd.DataFrame()
    m = pd.concat(parts, ignore_index=True)
    m = m[~m["arm"].isin(R.DROPPED_ARMS)]
    m["usable"] = m["status"].isin(["ok", "degenerate"])
    return m


def read_table(name):
    p = TABLES / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


# ----------------------------------------------------------------------------- sections

def sec_flow() -> str:
    rows = []
    for key in KEYS:
        meta = json.loads((R.rd.CLEAN / f"{key}.meta.json").read_text(encoding="utf8"))
        sp = meta["splits"]["split"]
        fl = meta["flow"]
        drop = fl.get("dropped_missing_predictor", 0)
        rows.append([LABEL[key], f"{meta['n']:,}", f"{meta['events']:,} ({meta['event_rate']:.1%})", f"{drop:,}",
                     f"{sp['dev_pool']['n']:,} ({sp['dev_pool']['events']:,}; {sp['dev_pool']['rate']:.1%})",
                     f"{sp['ext_eval']['n']:,} ({sp['ext_eval']['events']:,}; {sp['ext_eval']['rate']:.1%})",
                     f"{sp['ext_recal']['n']} ({sp['ext_recal']['events']})"])
    txt = table(rows, ["Outcome", "Complete cases", "Events", "Excluded (missing predictor)",
                       "Development pool (events; rate)", "External evaluation (events; rate)",
                       "Local recalibration sample (events)"])
    rows2 = []
    for key in R.TEMPORAL_KEYS:
        meta = json.loads((R.rd.CLEAN / f"{key}.meta.json").read_text(encoding="utf8"))
        sp = meta["splits"]["split_temporal"]
        rows2.append([LABEL[key], f"{sp['dev_pool']['n']:,} ({sp['dev_pool']['events']:,}; {sp['dev_pool']['rate']:.1%})",
                      f"{sp['ext_eval']['n']:,} ({sp['ext_eval']['events']:,}; {sp['ext_eval']['rate']:.1%})",
                      f"{sp['ext_recal']['n']} ({sp['ext_recal']['events']})"])
    flows = []
    for key in KEYS:
        meta = json.loads((R.rd.CLEAN / f"{key}.meta.json").read_text(encoding="utf8"))
        flows.append(f"- {LABEL[key]}: " + "; ".join(f"{k.replace('_', ' ')} {v:,}" for k, v in meta["flow"].items()))
    splits = ("Splits (10.1): IST, UK develops and the other 35 countries validate; CRASH-2, 74 of 270 hospitals held "
              "out (D-01); Sepsis, non-Sepsis-3 admissions develop and the Sepsis-3 study cohort validates; "
              "PhysioNet, sets A+B develop and set C validates.")
    return ("## 1. Data flow\n\n" + splits + "\n\n" + txt + "\n\nPatient flow as recorded by the loaders:\n\n" +
            "\n".join(flows) + "\n\nIST temporal sensitivity split (randomised before 1994 develops; D-05):\n\n" +
            table(rows2, ["Outcome", "Development pool", "External evaluation", "Local recalibration sample"]))


def sec_learning(m: pd.DataFrame, part="lc", title="## 2. Learning curves (10.2): external calibration by n") -> str:
    d = m[(m["part"] == part) & (m["ladder"] == "R0")]
    if d.empty:
        return title + "\n\nNo results yet.\n"
    out = [title, "",
           "Per data set, n and arm: usable fits / attempted; empirical ICI, mean [95 % interval]; calibration "
           "slope, median (IQR); calibration-in-the-large (logit, slope fixed at 1), median (IQR); AUC and Brier, "
           "mean; share of subsamples meeting the empirical weak class (|CITL| <= 0.1 and |slope - 1| <= 0.1)."]
    for key in [k for k in KEYS if k in set(d["key"])]:
        rows = []
        for n in NS:
            for arm in ARMS:
                g = d[(d["key"] == key) & (d["n"] == n) & (d["arm"] == arm)]
                if g.empty:
                    continue
                u = g[g["usable"]]
                st = g["status"].value_counts().to_dict()
                tag = "" if not st.get("not_run") else " (not run, T2 rule)"
                weak = u["emp_cls_weak"].astype(float).mean() if "emp_cls_weak" in u and len(u) else np.nan
                rows.append([n, ARM_LABEL[arm] + tag, f"{len(u)}/{len(g)}", mean_ci(u.get("e_emp_ici", [])),
                             med_iqr(u.get("e_emp_slope", [])), med_iqr(u.get("e_emp_intercept", [])),
                             f(u["e_auc"].mean() if len(u) else np.nan), f(u["e_brier"].mean() if len(u) else np.nan, 4),
                             f(weak, 2)])
        out += ["", f"### {LABEL[key]}", "",
                table(rows, ["n", "Arm", "Usable", "Empirical ICI", "Slope", "CITL", "AUC", "Brier", "Weak (emp.)"])]
    return "\n".join(out) + "\n"


def paired(m: pd.DataFrame, part: str, est_ici: str, est_slope: str, est_citl: str, reps_max=None):
    d = m[(m["part"] == part) & (m["ladder"] == "R0") & m["usable"]]
    if reps_max:
        d = d[d["rep"] <= reps_max]
    keycols = ["key", "prevalence", "n", "rep"]
    d = d.assign(prevalence=d["prevalence"].fillna(""))
    fl = d[d["arm"] == "flic"][keycols + [est_ici, est_slope, est_citl]]
    rows = []
    for tfm in TFMS:
        t = d[d["arm"] == tfm][keycols + [est_ici, est_slope, est_citl]]
        j = t.merge(fl, on=keycols, suffixes=("_t", "_f"))
        for (key, prev, n), g in j.groupby(["key", "prevalence", "n"]):
            dici = g[est_ici + "_t"] - g[est_ici + "_f"]
            dls = np.abs(np.log(np.clip(g[est_slope + "_t"], 1e-6, None))) - np.abs(np.log(np.clip(g[est_slope + "_f"], 1e-6, None)))
            dcitl = g[est_citl + "_t"].abs() - g[est_citl + "_f"].abs()
            rows.append(dict(key=key, prevalence=prev, n=n, arm=tfm, pairs=len(g), d_ici=mean_ci(dici, 4),
                             share_lower=f((dici < 0).mean(), 2), d_abslogb=mean_ci(dls, 3), d_abscitl=mean_ci(dcitl, 3)))
    return pd.DataFrame(rows)


def sec_paired(m: pd.DataFrame, part="lc", title="## 3. TFM minus FLIC, paired on the same subsample") -> str:
    p = paired(m, part, "e_emp_ici", "e_emp_slope", "e_emp_intercept")
    if p.empty:
        return title + "\n\nNo results yet.\n"
    rows = [[LABEL[r.key], r.n, ARM_LABEL[r.arm], r.pairs, r.d_ici, r.share_lower, r.d_abslogb, r.d_abscitl]
            for r in p.sort_values(["key", "arm", "n"]).itertuples()]
    return (title + "\n\nDifferences TFM - FLIC per subsample, mean [95 % interval]: empirical ICI; share of "
            "subsamples where the TFM's ICI is lower; |log slope| (distance of the slope from 1); |CITL|. Negative "
            "values favour the TFM.\n\n" +
            table(rows, ["Outcome", "n", "TFM", "Pairs", "Delta ICI", "Share TFM lower", "Delta |log slope|",
                         "Delta |CITL|"]) + "\n")


def sec_ladder(m: pd.DataFrame) -> str:
    d = m[(m["part"] == "lc") & m["usable"] & m["arm"].isin(["flic"] + TFMS)]
    if d.empty:
        return "## 4. Repair ladder (10.3)\n\nNo results yet.\n"
    levels = ["R0", "R1_m500", "R2_m500", "R3_m500", "R1_m100", "R2_m100", "R3_m100"]
    rows = []
    for key in [k for k in KEYS if k in set(d["key"])]:
        for n in NS:
            for arm in ["flic"] + TFMS:
                g = d[(d["key"] == key) & (d["n"] == n) & (d["arm"] == arm)]
                if g.empty:
                    continue
                ici = [f(g[g["ladder"] == lv]["e_emp_ici"].mean(), 4) for lv in levels]
                sl = [f(g[g["ladder"] == lv]["e_emp_slope"].median(), 2) for lv in ("R0", "R2_m500")]
                rows.append([LABEL[key], n, ARM_LABEL[arm]] + ici + sl)
    return ("## 4. Repair ladder (10.3)\n\nFitted on the 500-row local recalibration sample (m = 500) or its first "
            "100 rows (m = 100), evaluated on the external evaluation set. Mean empirical ICI per level; median "
            "slope before and after logistic recalibration (R2, m = 500). R1 intercept only, R2 logistic, R3 "
            "isotonic.\n\n" +
            table(rows, ["Outcome", "n", "Arm"] + [f"ICI {lv}" for lv in levels] + ["Slope R0", "Slope R2_m500"]) + "\n")


def sec_rule() -> str:
    t = read_table("test_guided_rule")
    if t.empty:
        return "## 5. Test-guided rule (5.5)\n\nNo results yet.\n"
    t = t[t["part"] == "lc"] if "part" in t else t
    rows = [[LABEL[r.key], r.n, ARM_LABEL.get(r.base, r.base), int(r.r), f(r.share_recalibrated, 2), f(r.share_insufficient, 2)]
            for r in t.sort_values(["key", "base", "n"]).itertuples()]
    return ("## 5. Test-guided rule (5.5)\n\nShare of subsamples in which the Cox recalibration test rejects at 0.05 "
            "on the R0 predictions (so R2 with m = 500 is applied), and share in which EDGE (G = 10) still rejects "
            "after R2 (flag 'recalibration insufficient'), on the external evaluation set.\n\n" +
            table(rows, ["Outcome", "n", "Arm", "Subsamples", "Recalibrate (Cox rejects)", "Insufficient (EDGE rejects after R2)"]) + "\n")


def sec_localization() -> str:
    t = read_table("localization")
    if t.empty:
        return "## 6. Localization (10.3)\n\nNo results yet.\n"
    t = t[(t["part"] == "lc") & ~t["arm"].str.contains("__")]
    w = t.pivot_table(index=["key", "n", "arm"], columns="group", values="share_named").reset_index()
    rr = t.groupby(["key", "n", "arm"])["r"].max().reset_index()
    w = w.merge(rr, on=["key", "n", "arm"])
    groups = [g for g in ["INTERCEPT", "SLOPE", "LINK", "COV"] if g in w.columns]
    rows = [[LABEL[r["key"]], r["n"], ARM_LABEL.get(r["arm"], r["arm"]), int(r["r"])] + [f(r[g], 2) for g in groups]
            for _, r in w.sort_values(["key", "arm", "n"]).iterrows()]
    return ("## 6. Localization (10.3)\n\nlocalize.external (M = 999, familywise error 0.05) on the external "
            "evaluation set with the covariates of D-04 item 5 (age band, sex, region for IST, severity score). "
            "Share of subsamples in which each part is named.\n\n" +
            table(rows, ["Outcome", "n", "Arm", "Subsamples"] + groups) + "\n")


def sec_rejection(part="lc", title="## 7. Goodness-of-fit battery on the external evaluation set (real Y)") -> str:
    t = read_table("gof_rejection")
    if t.empty:
        return title + "\n\nNo results yet.\n"
    t = t[(t["part"] == part) & (t["ytype"] == "real") & ~t["arm"].str.contains("__")]
    t = t.copy()
    t["short"] = None
    for full, short in TESTS:
        t.loc[t["test"].str.startswith(full), "short"] = short
    t = t[t["short"].notna() & ~t["test"].str.contains(r"\(axes\)$|\(allpairs\)$", regex=True)]
    w = t.pivot_table(index=["key", "n", "arm"], columns="short", values="rejection", aggfunc="first").reset_index()
    cols = [s for _, s in TESTS if s in w.columns]
    rows = [[LABEL[r["key"]], r["n"], ARM_LABEL.get(r["arm"], r["arm"])] + [f(r[c], 2) for c in cols]
            for _, r in w.sort_values(["key", "arm", "n"]).iterrows()]
    size = read_table("gof_rejection")
    size = size[(size["part"] == part) & (size["ytype"] == "ystar")]
    srows = []
    for full, short in TESTS:
        s = size[size["test"].str.startswith(full)]
        if len(s):
            srows.append([short, f(np.average(s["rejection"], weights=s["r"]), 3), int(s["r"].sum())])
    return (title + "\n\nRejection rate at 0.05 over subsamples. DeepGOF-1 uses the combined reading (CRASH-2, "
            "Sepsis) or the axes reading (IST, PhysioNet; more than 12 predictors).\n\n" +
            table(rows, ["Outcome", "n", "Arm"] + cols) +
            "\n\nSize on each arm's exact null Y* ~ Bernoulli(p-hat), pooled over data sets, n and arms "
            "(should be near 0.05):\n\n" + table(srows, ["Test", "Rejection under Y*", "Calls"]) + "\n")


def sec_plasmode(m: pd.DataFrame) -> str:
    d = m[(m["part"] == "pl") & (m["ladder"] == "R0")]
    if d.empty:
        return "## 8. Plasmode tier (10.4)\n\nNo results yet.\n"
    out = ["## 8. Plasmode tier (10.4): oracle estimands against the frozen true risk", "",
           "Real covariates, Y* drawn from the frozen truth (main effects plus a U-shaped term 0.4 (z^2 - 1) and an "
           "interaction 0.6 z_a x_b) at mean p* = 10 % and 30 %. The foundation models run at R = 50 (replicates "
           "1-50, D-05); the classical arms and the oracle at R = 200, summarised here over the same replicates "
           "1-50 (paired), with the all-200 mean ICI beside them. True ICI, mean [95 % interval]; b* and a*, median "
           "(IQR); OCE, mean; share meeting the weak and moderate classes (4.5)."]
    for key in [k for k in R.PLASMODE_KEYS if k in set(d["key"])]:
        for prev in R.PREVS:
            rows = []
            for n in NS:
                for arm in ["oracle"] + ARMS:
                    g = d[(d["key"] == key) & (d["prevalence"] == prev) & (d["n"] == n) & (d["arm"] == arm)]
                    if g.empty:
                        continue
                    g50 = g[g["rep"] <= 50]
                    u = g50[g50["usable"]]
                    u200 = g[g["usable"]]
                    rows.append([n, ARM_LABEL[arm], f"{len(u)}/{len(g50)}", mean_ci(u.get("o_true_ici", []), 4),
                                 med_iqr(u.get("o_true_slope", [])), med_iqr(u.get("o_true_intercept", [])),
                                 f(u["o_oce"].mean() if len(u) else np.nan, 4),
                                 f(u["cls_weak"].astype(float).mean() if len(u) else np.nan, 2),
                                 f(u["cls_moderate"].astype(float).mean() if len(u) else np.nan, 2),
                                 f(u200["o_true_ici"].mean(), 4) + f" (R={len(u200)})" if arm not in TFMS else "-"])
            out += ["", f"### {LABEL[key]}, mean p* = {int(prev)} %", "",
                    table(rows, ["n", "Arm", "Usable (reps 1-50)", "True ICI", "b*", "a*", "OCE", "Weak", "Moderate",
                                 "True ICI, all reps"])]
    p = paired(m, "pl", "o_true_ici", "o_true_slope", "o_true_intercept", reps_max=50)
    if len(p):
        rows = [[LABEL[r.key], f"{r.prevalence} %", r.n, ARM_LABEL[r.arm], r.pairs, r.d_ici, r.share_lower, r.d_abslogb, r.d_abscitl]
                for r in p.sort_values(["key", "prevalence", "arm", "n"]).itertuples()]
        out += ["", "### Plasmode: TFM minus FLIC on the oracle estimands (paired, replicates 1-50)", "",
                table(rows, ["Outcome", "Mean p*", "n", "TFM", "Pairs", "Delta true ICI", "Share TFM lower",
                             "Delta |log b*|", "Delta |a*|"])]
    t = read_table("gof_rejection")
    if len(t):
        s = t[(t["part"] == "pl") & (((t["arm"] == "oracle") & (t["ytype"] == "real")) | (t["ytype"] == "ystar"))]
        srows = []
        for full, short in TESTS:
            o = s[s["test"].str.startswith(full) & (s["arm"] == "oracle") & (s["ytype"] == "real")]
            a = s[s["test"].str.startswith(full) & (s["ytype"] == "ystar") & (s["arm"] != "oracle")]
            if len(o) or len(a):
                srows.append([short, f(np.average(o["rejection"], weights=o["r"]), 3) if len(o) else "-",
                              int(o["r"].sum()) if len(o) else 0,
                              f(np.average(a["rejection"], weights=a["r"]), 3) if len(a) else "-", int(a["r"].sum()) if len(a) else 0])
        out += ["", "### Plasmode: size of the battery (exact nulls)", "",
                "The oracle's outcome Y* ~ Bernoulli(p*) is an exact null for the oracle; each arm's own Y* ~ "
                "Bernoulli(p-hat) is an exact null for that arm. Pooled over data sets, prevalences and n.", "",
                table(srows, ["Test", "Oracle on Y*", "Calls", "Arms on own Y*", "Calls"])]
    return "\n".join(out) + "\n"


def sec_simulation(m: pd.DataFrame) -> str:
    f_sim = REBUILD / "analysis" / "tables" / "summary" / "arm_summaries.parquet"
    d = m[(m["part"] == "pl") & (m["ladder"] == "R0") & m["usable"] & (m["rep"] <= 50)]
    if not f_sim.exists() or d.empty:
        return "## 9. Comparison with the simulation\n\nNot available yet.\n"
    s = pd.read_parquet(f_sim)
    s = s[(s["subset"] == "all") & (s["ladder"] == "R0") & s["scenario"].str.match(r"^S[1-4]_pi(10|30)_n\d+$")]
    s = s.assign(pi=s["scenario"].str.extract(r"_pi(\d+)_")[0], n=s["scenario"].str.extract(r"_n(\d+)$")[0].astype(int))
    rows = []
    for arm in ["flic"] + TFMS:
        for n in NS:
            for pi in ["10", "30"]:
                ss = s[(s["arm"] == arm) & (s["n"] == n) & (s["pi"] == pi)]
                sim_b = ss[ss["estimand"] == "o_true_slope"]["median"].mean()
                sim_i = ss[ss["estimand"] == "o_true_ici"]["mean"].mean()
                pl = d[(d["arm"] == arm) & (d["n"] == n) & (d["prevalence"] == pi)]
                per = "; ".join(f"{k.split('_')[0]} {f(g['o_true_slope'].median(), 2)} / {f(g['o_true_ici'].mean(), 4)}"
                                for k, g in pl.groupby("key"))
                rows.append([ARM_LABEL[arm], n, f"{pi} %", f(sim_b, 2), f(sim_i, 4), per or "-"])
    return ("## 9. Comparison with the simulation (secondary check, 10.4)\n\nSimulation: core shapes S1-S4 at "
            "pi = 10 % and 30 %, mean over the four shapes of the median b* and of the mean true ICI (complete "
            "run, analysis/tables/summary/arm_summaries). Plasmode: median b* / mean true ICI per data set at the "
            "same mean risk (replicates 1-50). The simulation's labelled conclusions are in "
            "analysis/tables/summary/summary_of_findings.csv; the real-data results enter no label (10.5).\n\n" +
            table(rows, ["Arm", "n", "pi", "Sim median b*", "Sim mean true ICI", "Plasmode b* / true ICI by data set"]) + "\n")


def sec_failures(m: pd.DataFrame) -> str:
    d = m[m["ladder"] == "R0"]
    if d.empty:
        return ""
    c = d.groupby(["part", "key", "arm", "status"]).size().unstack("status", fill_value=0).reset_index()
    rows = []
    for _, r in c.iterrows():
        bad = {k: int(r[k]) for k in ("failed", "degenerate", "not_run", "missing") if k in r and r[k]}
        if bad:
            rows.append([r["part"], LABEL[r["key"]], ARM_LABEL.get(r["arm"], r["arm"]), int(r.get("ok", 0)),
                         ", ".join(f"{k} {v}" for k, v in bad.items())])
    return ("## 10. Failed, degenerate and not-run fits\n\nCounted, never dropped silently (8.2). Degenerate fits "
            "(more than 100 evaluation predictions within 1e-6 of 0 or 1, or LR-MLE separation) are scored like any "
            "other. Not run: Mitra-v2 at n = 3000 for IST and PhysioNet (D-03). GAM on Sepsis fails when a "
            "subsample has fewer than five distinct episode values (registered k = 5; D-05).\n\n" +
            table(rows, ["Part", "Outcome", "Arm", "OK", "Other statuses"]) + "\n")


def sec_coverage(m: pd.DataFrame) -> str:
    d = m[(m["ladder"] == "R0") & m["usable"]]
    if d.empty:
        return ""
    d = d.assign(prevalence=d["prevalence"].fillna(""))
    c = d.groupby(["part", "key", "arm"]).size().unstack("arm", fill_value=0)
    cols = [a for a in ["oracle"] + ARMS if a in c.columns]
    rows = [[p_, LABEL[k]] + [int(c.loc[(p_, k), a]) for a in cols] for (p_, k) in c.index]
    intro = ("## 0. Coverage" + NL2 + "Usable fits (ok or degenerate) per part and arm, summed over n (and prevalence). "
             "Full coverage: learning curves (lc) 400 per arm (4 n x B = 100; Mitra-v2 300 where n = 3000 is not run); "
             "plasmode (pl) 1,600 per classical arm (2 prevalences x 4 n x 200) and 400 per foundation model "
             "(R = 50; 300 for Mitra-v2 on IST and PhysioNet); IST temporal split (lct) 400 per arm." + NL2)
    return intro + table(rows, ["Part", "Outcome"] + [ARM_LABEL[a] for a in cols]) + chr(10)


def main() -> int:
    m = load_metrics()
    complete = (TABLES / "COMPLETE").exists()
    head = [
        "# Real-data results digest",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by analysis/realdata_digest.py from results/realdata and "
        f"analysis/tables/realdata. Data state: **{'complete' if complete else 'PARTIAL - the run is still going; numbers will change'}**.",
        "",
        "Protocol: pre-registration Section 10, with DEVIATIONS D-01 (CRASH-2 hospital split), D-02 (CRASH-2 terms; "
        "permission to publish aggregate results granted 2026-10-23), D-03 (Mitra-v2 not run and TabPFN-3.5 on CPU "
        "where the 6 GB card cannot hold them), D-04 (implementation details) and D-05 (side arms dropped; plasmode "
        "foundation models at R = 50; IST temporal split). Real-data results are descriptive and supportive (10.5). "
        "Arms: LR-MLE, FLIC, ridge, GAM, LightGBM + Platt (and raw), TabPFN-3.5, TabICLv2, Mitra-v2. Learning "
        "curves: B = 100 subsamples per n. CRASH-2: aggregate results only; no per-hospital result exists.",
        "",
    ]
    secs = [sec_coverage(m), sec_flow(), sec_learning(m), sec_paired(m), sec_ladder(m), sec_rule(), sec_localization(),
            sec_rejection(), sec_plasmode(m), sec_simulation(m), sec_failures(m),
            sec_learning(m, "lct", "## 11. IST temporal sensitivity split (D-05): external calibration by n"),
            sec_paired(m, "lct", "## 12. IST temporal split: TFM minus FLIC, paired"),
            sec_rejection("lct", "## 13. IST temporal split: goodness-of-fit battery (real Y)")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(head) + "\n" + "\n\n".join(secs) + "\n", encoding="utf8")
    print(f"digest written: {OUT.relative_to(REBUILD)} ({len(m)} metric rows; {'complete' if complete else 'partial'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
