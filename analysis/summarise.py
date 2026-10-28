"""Every pre-specified summary and label, from the tidy tables of collect.py.

    python analysis/summarise.py [--skip-detection] [--detection-B 500] [--meta-dir DIR]

Writes analysis/tables/summary/*.csv (and .parquet for the large ones):

  arm_summaries        6.1: per scenario x column (arm or ladder level) x estimand: mean (MCSE,
                       95 % interval), median (order-statistic 95 % interval, bootstrap MCSE), IQR,
                       10th and 90th percentiles. Subset "all" uses every replicate of the arm; the
                       subset "no_degenerate" repeats LR-MLE without its degenerate fits (8.2).
  paired_vs_flic       6.1 / 7.0: each arm minus FLIC on the replicates where both are usable:
                       mean paired difference (MCSE, 95 % interval) for every metric; HL estimate
                       with its signed-rank interval (and bootstrap MCSE for the TFM, side and
                       ladder columns) for |log b*| and |a*|; the labels of 7.0.
  classes              4.5 / 6.1: share of replicates in each nested class, and each rule alone,
                       with MCSE; main thresholds, half, double, and true_ll_ici in place of true ICI.
  min_n                4.5 / 11 (primary output 4): minimum n for every arm, shape, event rate,
                       class and variant, the "> 3000" label, the pooled row per event rate, beside
                       Riley's minimum n.
  rejection_rates      6.1: every test x arm x scenario x outcome type x evaluation set, with MCSE.
  q1_slope .. q6_transfer, q4a_size, q4b_detection, q5a_*, q5b_rule
                       the research questions with the fixed reporting rules of Section 7.
  fit_counts           6.1 / 8.2: failed, degenerate and not-run fits and every flag, with the
                       5 % failure flag.
  summary_of_findings  one row per question with its label counts (7.0 "Reporting").
  sanity_checks        the classical-arm checks (oracle ICI floor, oracle size).

Data state: when tables/manifest.json says the run is not complete, every file carries
data_state = "PARTIAL - NOT FOR INTERPRETATION" and the console prints no TFM label.
"""

from __future__ import annotations

import argparse
import warnings
import json
import subprocess
import sys
import time

import _common as C

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rules as Rr  # noqa: E402

OUT = C.TABLES / "summary"
warnings.simplefilter("ignore", FutureWarning)
PARTIAL = "PARTIAL - NOT FOR INTERPRETATION"

ESTIMANDS_ORACLE = ["o_true_ici", "o_true_e50", "o_true_e90", "o_true_emax", "o_true_ll_ici",
                    "o_true_ll_e50", "o_true_ll_e90", "o_true_ll_emax", "o_true_slope",
                    "o_true_intercept", "o_true_citl_prob", "o_oce", "o_true_auc", "o_oracle_auc",
                    "abs_logb", "abs_a", "o_n_clipped"]
ESTIMANDS_EMP = ["e_emp_ici", "e_emp_e50", "e_emp_e90", "e_emp_emax", "e_emp_slope",
                 "e_emp_intercept", "e_emp_citl_prob", "e_brier", "e_logloss", "e_auc", "e_ece10",
                 "ici_gap"]
PAIR_MEAN = ["o_true_ici", "o_true_ll_ici", "o_oce", "o_true_e90", "o_true_slope",
             "o_true_intercept", "o_true_citl_prob", "abs_logb", "abs_a", "e_emp_ici"]
PAIR_HL = ["abs_logb", "abs_a"]
CLASSES = ["mean", "weak", "moderate", "strong"]
VARIANTS = {"main": "cls_highest", "half": "cls_highest_half", "double": "cls_highest_double",
            "ll": "cls_highest_ll"}


def state() -> str:
    m = C.TABLES / "manifest.json"
    if m.exists() and json.loads(m.read_text()).get("complete"):
        return "complete"
    return PARTIAL


def write(df: pd.DataFrame, name: str, data_state: str, parquet: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    C.assert_writable(OUT)
    df = df.copy()
    df.insert(0, "data_state", data_state)
    df.to_csv(OUT / f"{name}.csv", index=False)
    if parquet:
        df.to_parquet(OUT / f"{name}.parquet", index=False)


def tfm_main_cell(row) -> bool:
    return row["family"] in C.SHAPES and abs(row["pi"] - 0.10) < 1e-9 and row["n"] in (200, 500)


# ----------------------------------------------------------------------------- 6.1 arm summaries

def arm_summaries(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scen, col), g in s.groupby(["scenario", "column"], sort=True):
        ladder = g["ladder"].iloc[0]
        ests = ESTIMANDS_ORACLE + (ESTIMANDS_EMP if ladder == "R0" else [])
        subsets = [("all", g)]
        if g["arm"].iloc[0] == "lr" and ladder == "R0":
            subsets.append(("no_degenerate", g[g["status"] != "degenerate"]))
        for sub, gg in subsets:
            u = gg[gg["usable"]]
            for e in ests:
                if e not in u:
                    continue
                r = Rr.summary_row(u[e].to_numpy(float))
                r.update(scenario=scen, column=col, arm=g["arm"].iloc[0], ladder=ladder, subset=sub,
                         estimand=e, n_reps_attempted=len(gg), n_usable=len(u))
                rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 6.1 / 7.0 paired

def is_labelled(column: str) -> bool:
    arm = column.split("__")[0]
    return arm in C.TFMS or arm in C.SIDE


def paired_summaries(p: pd.DataFrame, fail_flags: set, oracle_ici: dict) -> pd.DataFrame:
    rows = []
    for (kind, scen, col, ref), g in p.groupby(["kind", "scenario", "column", "ref"], sort=True):
        pi = float(g["pi"].iloc[0])
        base = dict(kind=kind, scenario=scen, column=col, ref=ref, arm=g["arm"].iloc[0],
                    ladder=g["ladder"].iloc[0], family=g["family"].iloc[0], pi=pi, n=int(g["n"].iloc[0]),
                    R_pairs=len(g), fail_flag=(scen, col) in fail_flags or (scen, ref) in fail_flags)
        for m in PAIR_MEAN:
            d = Rr.finite(g[f"d_{m}"])
            if len(d) < 2:
                continue
            mean, se = float(d.mean()), Rr.mcse_mean(d)
            r = dict(base, metric=m, summary="mean", estimate=mean, mcse=se,
                     lo=mean - Rr.Z95 * se, hi=mean + Rr.Z95 * se, n_excluded_nonfinite=len(g) - len(d))
            if m == "o_true_ici":
                r["label"] = Rr.label_ici(r["lo"], r["hi"], pi)
                r["margin"] = C.ici_margin(pi)
                # floor: both arms' lower limits at or below the oracle's upper limit (6.3)
                x, y = Rr.finite(g["x_o_true_ici"]), Rr.finite(g["r_o_true_ici"])
                ohi = oracle_ici.get(scen, np.nan)
                lo_x = x.mean() - Rr.Z95 * Rr.mcse_mean(x)
                lo_y = y.mean() - Rr.Z95 * Rr.mcse_mean(y)
                r["bound"] = "floor" if np.isfinite(ohi) and Rr.ici_at_floor(lo_x, lo_y, ohi) else None
            rows.append(r)
        for m in PAIR_HL:
            d = Rr.finite(g[f"d_{m}"])
            if len(d) < 2:
                continue
            est, lo, hi = Rr.hodges_lehmann(d)
            mc = Rr.mcse_hl_boot(d) if is_labelled(col) else np.nan
            delta = C.MARGIN_LOGB if m == "abs_logb" else C.MARGIN_ABS_A
            rows.append(dict(base, metric=m, summary="HL", estimate=est, mcse=mc, lo=lo, hi=hi,
                             label=Rr.label_vs_margin(lo, hi, delta), margin=delta,
                             n_excluded_nonfinite=len(g) - len(d)))
        if kind == "temperature":
            # the signed paired effect of the temperature on a* and log b* (descriptive, 7.3)
            for m, f in (("signed_a", g["d_o_true_intercept"]),
                         ("signed_log_b", np.log(g["x_o_true_slope"].where(g["x_o_true_slope"] > 0))
                          - np.log(g["r_o_true_slope"].where(g["r_o_true_slope"] > 0)))):
                d = Rr.finite(f)
                if len(d) >= 2:
                    est, lo, hi = Rr.hodges_lehmann(d)
                    rows.append(dict(base, metric=m, summary="HL", estimate=est, mcse=Rr.mcse_hl_boot(d),
                                     lo=lo, hi=hi, n_excluded_nonfinite=len(g) - len(d)))
    out = pd.DataFrame(rows)
    if len(out):
        if "bound" not in out:
            out["bound"] = None
        if "label" not in out:
            out["label"] = None
        labelled = out["label"].notna()
        out.loc[labelled & out["bound"].notna(), "label"] = "uninformative (floor)"
        # the 5 % flag wins: such cells are reported but kept out of every label (8.2)
        out.loc[labelled & out["fail_flag"], "label"] = "flagged: > 5 % failed fits (8.2)"
    return out


# ----------------------------------------------------------------------------- 4.5 classes, min n

def class_shares(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scen, col), g in s.groupby(["scenario", "column"], sort=True):
        r_all = len(g)                         # failed fits count as not meeting (they are kept)
        for var, ccol in VARIANTS.items():
            if ccol not in g:
                continue
            hi = g[ccol].where(g["usable"], "none").fillna("none")
            for cls in CLASSES:
                k = int(Rr.meets_class(hi, cls).sum())
                suffix = {"main": "", "half": "_half", "double": "_double", "ll": "_ll"}[var]
                alone = g.get(f"cls_{cls}{suffix}")
                k_alone = int((alone.fillna(False).astype(bool) & g["usable"]).sum()) if alone is not None else np.nan
                rows.append(dict(scenario=scen, column=col, arm=g["arm"].iloc[0], ladder=g["ladder"].iloc[0],
                                 variant=var, cls=cls, R=r_all, k=k, share=k / r_all,
                                 mcse=Rr.mcse_prop(k, r_all), k_rule_alone=k_alone,
                                 share_rule_alone=k_alone / r_all if r_all else np.nan,
                                 bound=Rr.proportion_bound(k, r_all)))
    return pd.DataFrame(rows)


def min_n_table(cs: pd.DataFrame, sc: pd.DataFrame) -> pd.DataFrame:
    c = cs[cs["ladder"] == "R0"].merge(sc[["scenario", "family", "pi", "n", "riley_n_min_pmsampsize", "epv"]],
                                      on="scenario")
    c = c[c["family"].isin(C.SHAPES)]
    rows = []
    for (arm, fam, pi, cls, var), g in c.groupby(["arm", "family", "pi", "cls", "variant"]):
        shares = dict(zip(g["n"], g["share"]))
        counts = dict(zip(g["n"], g["R"]))
        main = Rr.min_n(shares)
        low = Rr.min_n(shares, counts, lower_limit=True)
        rows.append(dict(arm=arm, family=fam, pi=pi, cls=cls, variant=var, min_n=str(main),
                         min_n_lower_limit=str(low), riley_n_min=int(g["riley_n_min_pmsampsize"].iloc[0]),
                         n_present=",".join(str(int(x)) for x in sorted(g["n"])),
                         **{f"share_n{int(n)}": shares.get(n) for n in C.NS},
                         **{f"R_n{int(n)}": counts.get(n) for n in C.NS}))
    t = pd.DataFrame(rows)
    pooled = []
    for (arm, pi, cls, var), g in t.groupby(["arm", "pi", "cls", "variant"]):
        worst = max(g["min_n"], key=Rr.min_n_rank)
        worst_l = max(g["min_n_lower_limit"], key=Rr.min_n_rank)
        pooled.append(dict(arm=arm, family="pooled (largest of the four shapes)", pi=pi, cls=cls, variant=var,
                           min_n=worst, min_n_lower_limit=worst_l, riley_n_min=int(g["riley_n_min"].max()),
                           n_present=f"{len(g)} shapes"))
    return pd.concat([t, pd.DataFrame(pooled)], ignore_index=True)


def q2(s: pd.DataFrame, p: pd.DataFrame, fail_flags: set) -> pd.DataFrame:
    """TFM vs FLIC minimum n for weak calibration on the common replicates (7.2)."""
    rows = []
    vs = p[(p["kind"] == "vs_flic") & p["arm"].isin(C.TFMS) & p["family"].isin(C.SHAPES)]
    key = s.set_index(["scenario", "rep", "column"])
    for (arm, fam, pi), g in vs.groupby(["arm", "family", "pi"]):
        for var, ccol in VARIANTS.items():
            if var == "ll":
                continue
            sh = {"tfm": {}, "flic": {}}
            cnt = {"tfm": {}, "flic": {}}
            flagged = False
            for n, gn in g.groupby("n"):
                idx_t = list(zip(gn["scenario"], gn["rep"], gn["column"]))
                idx_f = list(zip(gn["scenario"], gn["rep"], gn["ref"]))
                ht = key.loc[idx_t, ccol].fillna("none")
                hf = key.loc[idx_f, ccol].fillna("none")
                r = len(gn)
                sh["tfm"][int(n)] = Rr.meets_class(ht, "weak").mean()
                sh["flic"][int(n)] = Rr.meets_class(hf, "weak").mean()
                cnt["tfm"][int(n)] = cnt["flic"][int(n)] = r
                flagged |= ((gn["scenario"].iloc[0], arm) in fail_flags
                            or (gn["scenario"].iloc[0], C.COMPARATOR) in fail_flags)
            mt, mf = Rr.min_n(sh["tfm"]), Rr.min_n(sh["flic"])
            lt, lf = Rr.min_n(sh["tfm"], cnt["tfm"], True), Rr.min_n(sh["flic"], cnt["flic"], True)
            main, low = Rr.compare_min_n(mt, mf), Rr.compare_min_n(lt, lf)
            rows.append(dict(arm=arm, family=fam, pi=pi, thresholds=var, tfm_min_n=str(mt), flic_min_n=str(mf),
                             tfm_min_n_lower=str(lt), flic_min_n_lower=str(lf), ordering=main,
                             ordering_lower_limit=low,
                             clarity=Rr.q2_clarity(main, low) if var == "main" else "",
                             n_present=",".join(str(k) for k in sorted(sh["tfm"])),
                             fail_flag=flagged))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- rejection rates, Q4a

def rejection_rates(g: pd.DataFrame) -> pd.DataFrame:
    t = g[g["reject"].notna()]
    agg = t.groupby(["eval_set", "scenario", "column", "arm_base", "ladder", "ytype", "test"])["reject"].agg(["sum", "count"])
    agg = agg.reset_index().rename(columns={"sum": "k", "count": "R"})
    agg["rate"] = agg["k"] / agg["R"]
    agg["mcse"] = [Rr.mcse_prop(k, r) for k, r in zip(agg["k"], agg["R"])]
    agg["bound"] = [Rr.proportion_bound(int(k), int(r)) for k, r in zip(agg["k"], agg["R"])]
    na = g[g["reject"].isna()].groupby(["eval_set", "scenario", "column", "ytype", "test"]).size().rename("n_not_run")
    agg = agg.merge(na.reset_index(), how="left", on=["eval_set", "scenario", "column", "ytype", "test"])
    agg["n_not_run"] = agg["n_not_run"].fillna(0).astype(int)
    return agg


def size_rows(g: pd.DataFrame) -> pd.DataFrame:
    """The null rows of Q4a: Y* for every arm; for the oracle also its real Y (5.3)."""
    t = g[(g["eval_set"] == "eval") & (g["ladder"] == "R0") & g["reject"].notna()]
    null = t[(t["ytype"] == "ystar") | ((t["arm_base"] == C.ORACLE) & (t["ytype"] == "real"))]
    # the battery and the familywise localization row; the "(named)" parts are closed-testing
    # components, not level-alpha tests, and the edge.gof rows repeat the battery's EDGE rows
    return null[null["test"].isin(C.BATTERY + C.EXTRA_TESTS)]


def q4a(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    null = size_rows(g)
    per = null.groupby(["test", "column", "scenario"])["reject"].agg(["sum", "count"]).reset_index()
    per = per.rename(columns={"sum": "k", "count": "R"})
    per["rate"] = per["k"] / per["R"]
    per["mcse"] = [Rr.size_mcse(r) for r in per["R"]]
    per["in_band"] = [Rr.in_size_band(rt, r) for rt, r in zip(per["rate"], per["R"])]
    rows = []
    for (test, col), gg in per.groupby(["test", "column"]):
        k, r = int(gg["k"].sum()), int(gg["R"].sum())
        pooled = k / r
        share = float(gg["in_band"].mean())
        rows.append(dict(test=test, column=col, n_scenarios=len(gg), k=k, R=r, pooled_size=pooled,
                         pooled_mcse=Rr.mcse_prop(k, r), share_scenarios_in_band=share,
                         label=Rr.size_label(pooled, share),
                         outside_band=";".join(gg.loc[~gg["in_band"], "scenario"])))
    return pd.DataFrame(rows), per


# ----------------------------------------------------------------------------- Q1, Q3

def q1(paired: pd.DataFrame, arm_sum: pd.DataFrame) -> pd.DataFrame:
    hl = paired[(paired["kind"] == "vs_flic") & (paired["metric"] == "abs_logb") & (paired["summary"] == "HL") & paired["arm"].isin(C.TFMS)
                & paired["family"].isin(C.SHAPES)].copy()
    med = arm_sum[(arm_sum["estimand"] == "o_true_slope") & (arm_sum["subset"] == "all") & (arm_sum["ladder"] == "R0")]
    med = med[["scenario", "arm", "median", "median_lo", "median_hi", "median_mcse", "R"]]
    med["direction"] = [Rr.direction_label(lo, hi) for lo, hi in zip(med["median_lo"], med["median_hi"])]
    hl = hl.merge(med.add_prefix("tfm_").rename(columns={"tfm_scenario": "scenario", "tfm_arm": "arm"}),
                  on=["scenario", "arm"], how="left")
    fl = med[med["arm"] == C.COMPARATOR].drop(columns="arm").add_prefix("flic_").rename(columns={"flic_scenario": "scenario"})
    hl = hl.merge(fl, on="scenario", how="left")
    hl["main_cell"] = hl.apply(tfm_main_cell, axis=1)
    return hl


def q3(paired: pd.DataFrame, arm_sum: pd.DataFrame) -> pd.DataFrame:
    hl = paired[(paired["kind"] == "vs_flic") & (paired["metric"] == "abs_a") & (paired["summary"] == "HL") & paired["arm"].isin(C.TFMS)
                & paired["family"].isin(C.SHAPES)].copy()
    med = arm_sum[(arm_sum["estimand"] == "o_true_intercept") & (arm_sum["subset"] == "all") & (arm_sum["ladder"] == "R0")]
    med = med[["scenario", "arm", "median", "median_lo", "median_hi"]]
    hl = hl.merge(med.add_prefix("tfm_").rename(columns={"tfm_scenario": "scenario", "tfm_arm": "arm"}),
                  on=["scenario", "arm"], how="left")
    fl = med[med["arm"] == C.COMPARATOR].drop(columns="arm").add_prefix("flic_").rename(columns={"flic_scenario": "scenario"})
    hl = hl.merge(fl, on="scenario", how="left")
    cit = paired[(paired["kind"] == "vs_flic") & (paired["metric"] == "o_true_citl_prob")][["scenario", "column", "estimate", "lo", "hi"]]
    hl = hl.merge(cit.rename(columns={"estimate": "d_mean_risk", "lo": "d_mean_risk_lo", "hi": "d_mean_risk_hi"}),
                  on=["scenario", "column"], how="left")
    hl["shipped_temperature"] = hl["arm"].map(C.SHIPPED_TEMPERATURE)
    hl["primary_cells"] = np.isclose(hl["pi"], 0.03)
    hl["shown_first"] = hl["primary_cells"] & (hl["n"] >= 1000)
    return hl.sort_values(["arm", "primary_cells", "shown_first", "family", "n"], ascending=[True, False, False, True, True])


# ----------------------------------------------------------------------------- Q5

def q5a(arm_sum: pd.DataFrame, s: pd.DataFrame, paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lad = arm_sum[(arm_sum["estimand"] == "o_true_ici") & arm_sum["arm"].isin(C.LADDER_ARMS) & (arm_sum["subset"] == "all")]
    lad = lad[["scenario", "column", "arm", "ladder", "R", "mean", "mean_mcse", "mean_lo", "mean_hi"]]
    vsf = paired[(paired["kind"] == "ladder_vs_flic") & (paired["ladder"] == "R2_m500") & (paired["metric"] == "o_true_ici")]
    # best arm: smallest mean true ICI among FLIC and the TFMs after R2 (m = 500) and LR, Ridge, GAM,
    # LightGBM (Platt) as fitted; each TFM against it on their common replicates, with paired MCSE
    cand_cols = [f"{a}__R2_m500" for a in C.LADDER_ARMS] + ["lr", "ridge", "gam", "lgbm"]
    rows = []
    core = s[s["family"].isin(C.SHAPES)]
    piv = core[core["column"].isin(cand_cols) & core["usable"]].pivot_table(
        index=["scenario", "rep"], columns="column", values="o_true_ici")
    for scen, g in piv.groupby(level=0):
        pi = C.parse_scenario(scen)["pi"]
        for tfm in C.TFMS:
            tc = f"{tfm}__R2_m500"
            if tc not in g or g[tc].notna().sum() < 2:
                continue
            gt = g[g[tc].notna()]
            means = {c: gt[c].mean() for c in cand_cols if c in gt and gt[c].notna().sum() >= 2}
            best = min(means, key=means.get)
            d = Rr.finite((gt[tc] - gt[best]).to_numpy())
            est = float(d.mean()) if len(d) else np.nan
            se = Rr.mcse_mean(d) if best != tc else 0.0
            lo, hi = est - Rr.Z95 * se, est + Rr.Z95 * se
            rows.append(dict(scenario=scen, pi=pi, arm=tfm, best_column=best, best_mean=means[best],
                             tfm_mean=means.get(tc), R_pairs=len(d), estimate=est, mcse=se, lo=lo, hi=hi,
                             label=Rr.best_arm_label(lo, hi)))
    return lad, vsf, pd.DataFrame(rows)


def cluster_boot_ratio(num: np.ndarray, den: np.ndarray, clusters: np.ndarray, b: int = 1000, seed: int = 20261008):
    """Percentile interval of sum(num)/sum(den) under a cluster bootstrap."""
    u, inv = np.unique(clusters, return_inverse=True)
    sn = np.bincount(inv, weights=num, minlength=len(u))
    sd = np.bincount(inv, weights=den, minlength=len(u))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
        dd = (w * sd).sum()
        vals.append((w * sn).sum() / dd if dd > 0 else np.nan)
    vals = Rr.finite(vals)
    if not len(vals):
        return np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def q5b(s: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """The test-guided rule (5.5, 7.5), pooled over the core scenarios."""
    real = g[(g["eval_set"] == "eval") & (g["ytype"] == "real") & g["reject"].notna()]
    cox = real[(real["test"] == "Cox recalibration") & (real["ladder"] == "R0")][["scenario", "rep", "arm_base", "reject"]]
    edge = real[(real["test"] == "EDGE (G=10)") & (real["ladder"] == "R2_m500")][["scenario", "rep", "arm_base", "reject"]]
    tru = s[(s["ladder"] == "R2_m500") & s["usable"] & s["family"].isin(C.SHAPES)][["scenario", "rep", "arm", "pi", "o_true_ici"]]
    tru = tru.rename(columns={"arm": "arm_base"})
    tru["truly_insufficient"] = tru["o_true_ici"] > 0.015 * np.array([C.prob_scale(x) for x in tru["pi"]])
    m = cox.rename(columns={"reject": "cox"}).merge(edge.rename(columns={"reject": "edge_r2"}),
                                                     on=["scenario", "rep", "arm_base"], how="left")
    m = m.merge(tru, on=["scenario", "rep", "arm_base"], how="inner")
    rows = []
    for arm in C.LADDER_ARMS:
        a = m[m["arm_base"] == arm]
        if not len(a):
            continue
        flagged = (a["cox"] == 1) & (a["edge_r2"] == 1)
        notflag = (a["cox"] == 1) & (a["edge_r2"] == 0)
        cl = (a["scenario"] + "|" + a["rep"].astype(str)).to_numpy()
        ti = a["truly_insufficient"].to_numpy(float)
        nf = int(flagged.sum())
        ppv = float(ti[flagged].mean()) if nf else np.nan
        plo, phi = cluster_boot_ratio(ti * flagged, flagged.to_numpy(float), cl)
        nn = int(notflag.sum())
        npv = float((1 - ti)[notflag].mean()) if nn else np.nan
        nlo, nhi = cluster_boot_ratio((1 - ti) * notflag, notflag.to_numpy(float), cl)
        rows.append(dict(arm=arm, n_replicates=len(a), n_cox_rejected=int((a["cox"] == 1).sum()),
                         n_flagged=nf, ppv=ppv, ppv_lo=plo, ppv_hi=phi, ppv_label=Rr.ppv_label(plo, phi, nf),
                         n_not_flagged=nn, npv=npv, npv_lo=nlo, npv_hi=nhi))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Q6

def q6(paired: pd.DataFrame, meta_dir) -> pd.DataFrame:
    f = meta_dir / "transfer_predictions.csv" if meta_dir else None
    if not f or not f.exists():
        return pd.DataFrame()
    pred = pd.read_csv(f)
    core = paired[(paired["kind"] == "vs_flic") & (paired["metric"] == "o_true_ici") & paired["arm"].isin(C.TFMS)
                  & paired["family"].isin(C.SHAPES)][["scenario", "arm", "estimate", "mcse"]]
    m = core.merge(pred, left_on=["scenario", "arm"], right_on=["scenario", "model"], how="inner")
    if "in_region" in m:
        m = m[m["in_region"].astype(bool)]
    m["agrees"] = [Rr.transfer_agrees(a, b, c, d) for a, b, c, d in zip(m["estimate"], m["mcse"], m["pred"], m["se"])]
    return m


# ----------------------------------------------------------------------------- fit counts

def fit_counts(fits: pd.DataFrame) -> pd.DataFrame:
    f = fits[~fits["arm"].str.contains("__")].copy()
    for c in ("platt_fallback", "lambda_at_edge", "small_n_flag"):
        f[c] = f[c].fillna(False).astype(bool) if c in f else False
    f["boundary"] = pd.to_numeric(f.get("n_boundary_eval"), errors="coerce").fillna(0) > 0
    g = f.groupby(["scenario", "arm"])
    t = pd.DataFrame(dict(
        R=g["rep"].nunique(), ok=g["status"].apply(lambda x: int((x == "ok").sum())),
        degenerate=g["status"].apply(lambda x: int((x == "degenerate").sum())),
        failed=g["status"].apply(lambda x: int((x == "failed").sum())),
        not_run=g["status"].apply(lambda x: int((x == "not_run").sum())),
        platt_fallback=g["platt_fallback"].sum(), lambda_at_edge=g["lambda_at_edge"].sum(),
        small_n_flag=g["small_n_flag"].sum(), any_boundary_prediction=g["boundary"].sum())).reset_index()
    t["fail_flag"] = [Rr.fail_flag(a, b) for a, b in zip(t["failed"], t["R"] - t["not_run"])]
    lad = fits[fits["arm"].str.contains("__")].copy()
    if len(lad):
        lg = lad.groupby(["scenario", "arm"]).agg(R=("rep", "nunique"),
                                                  failed=("status", lambda x: int((x != "ok").sum()))).reset_index()
        lg["fail_flag"] = [Rr.fail_flag(a, b) for a, b in zip(lg["failed"], lg["R"])]
        t = pd.concat([t, lg], ignore_index=True)
    return t


# ----------------------------------------------------------------------------- summary of findings

def summary_of_findings(q1t, q2t, q3t, q4at, q4bt, q5at, q5best, q5bt, q6t) -> pd.DataFrame:
    rows = []

    def counts(df, col, by="arm", where=None):
        if df is None or not len(df):
            return {}
        d = df if where is None else df[where(df)]
        return {a: dict(d[d[by] == a][col].value_counts()) for a in sorted(d[by].unique())}

    rows.append(dict(question="Q1 closeness of b* to 1 (8 main cells)",
                     counts=counts(q1t, "label", where=lambda d: d["main_cell"])))
    rows.append(dict(question="Q1 direction of the TFM b* (8 main cells)",
                     counts=counts(q1t, "tfm_direction", where=lambda d: d["main_cell"])))
    rows.append(dict(question="Q1 closeness of b* to 1 (all 48 core scenarios)", counts=counts(q1t, "label")))
    rows.append(dict(question="Q2 minimum n for weak calibration vs FLIC",
                     counts=counts(q2t, "ordering", where=lambda d: d["thresholds"] == "main")))
    rows.append(dict(question="Q2 clarity of the ordering",
                     counts=counts(q2t, "clarity", where=lambda d: d["thresholds"] == "main")))
    rows.append(dict(question="Q3 |a*| at pi = 3 % (16 cells)",
                     counts=counts(q3t, "label", where=lambda d: d["primary_cells"])))
    rows.append(dict(question="Q4a size (every test x arm)", counts=counts(q4at, "label", by="column")))
    rows.append(dict(question="Q4b EDGE minus HL at the four ICI points",
                     counts={"EDGE vs HL": dict(q4bt["label"].value_counts())} if q4bt is not None and len(q4bt) else {}))
    rows.append(dict(question="Q5a TFM after R2 (m = 500) vs FLIC after R2", counts=counts(q5at, "label")))
    rows.append(dict(question="Q5a TFM after R2 vs the best arm (48 core scenarios)", counts=counts(q5best, "label")))
    rows.append(dict(question="Q5b PPV of the test-guided rule", counts=counts(q5bt, "ppv_label")))
    if q6t is not None and len(q6t):
        agree = {a: int(g["agrees"].sum()) for a, g in q6t.groupby("arm")}
        rows.append(dict(question="Q6 transfer of the grid", counts={
            a: {Rr.transfer_label(agree[a], int((q6t["arm"] == a).sum())): f"{agree[a]} of {int((q6t['arm'] == a).sum())} agree"}
            for a in agree}))
    else:
        rows.append(dict(question="Q6 transfer of the grid", counts="random-design tier not run yet"))
    out = pd.DataFrame(rows)
    out["counts"] = out["counts"].map(lambda x: json.dumps(x, default=lambda o: int(o) if hasattr(o, "__int__") else str(o)))
    return out


# ----------------------------------------------------------------------------- sanity checks

def sanity(arm_sum: pd.DataFrame, q4a_per: pd.DataFrame, q4a_pooled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    o = arm_sum[(arm_sum["column"] == C.ORACLE) & (arm_sum["subset"] == "all")]
    for e in ("o_true_ici", "o_true_ll_ici", "o_oce", "o_true_slope", "o_true_intercept"):
        x = o[o["estimand"] == e]
        if len(x):
            rows.append(dict(check=f"oracle {e} across scenarios", n_scenarios=len(x), min=x["mean"].min(),
                             median=x["mean"].median(), max=x["mean"].max(), max_mcse=x["mean_mcse"].max(),
                             reference=("pilot floor 0.0008-0.0029 (4.1)" if e == "o_true_ici" else
                                        "exactly 0 / 1 / 0 for the oracle" if e != "o_true_ll_ici" else "exactly 0")))
    q = q4a_pooled[q4a_pooled["column"] == C.ORACLE] if len(q4a_pooled) else q4a_pooled
    for _, r in q.iterrows():
        rows.append(dict(check=f"oracle size: {r['test']}", n_scenarios=r["n_scenarios"], R=r["R"],
                         pooled=r["pooled_size"], pooled_mcse=r["pooled_mcse"],
                         share_in_band=r["share_scenarios_in_band"], reference="0.05"))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- random tier input

def random_tier_inputs(s: pd.DataFrame, p: pd.DataFrame, g: pd.DataFrame) -> None:
    """Meta-model input tables (6.4) once the random designs exist; nothing to do before that."""
    rd = p[(p["kind"] == "vs_flic") & (p["tier"] == "random") & p["arm"].isin(C.TFMS)] if len(p) else p
    if not len(rd):
        return
    sc = s[s["tier"] == "random"].drop_duplicates("scenario").set_index("scenario")
    d = pd.DataFrame(dict(design=rd["scenario"].values, model=rd["arm"].values,
                          D_ICI=rd["d_o_true_ici"].values,
                          D_logb=(np.minimum(rd["x_abs_logb"], C.LOGB_CAP_META) -
                                  np.minimum(rd["r_abs_logb"], C.LOGB_CAP_META)).values))
    for c_out, c_in in (("n", "n"), ("pi", "pi"), ("p", "p"), ("auc_target", "auc_target"), ("w_nl", "w_nl"),
                        ("w_int", "w_int"), ("w_link", "w_link"), ("rho", "rho"), ("frac_disc", "frac_disc")):
        d[c_out] = sc.loc[d["design"], c_in].values
    d.to_parquet(C.assert_writable(C.TABLES / "random_tier.parquet"), index=False)
    if len(g):
        r = g[(g["tier"] == "random") & (g["eval_set"] == "eval") & (g["ladder"] == "R0") & g["reject"].notna()]
        r = r[["scenario", "column", "ytype", "test", "reject"]].rename(columns={"scenario": "design", "column": "arm"})
        for c in ("n", "pi", "p", "auc_target", "w_nl", "w_int", "w_link", "rho", "frac_disc"):
            r[c] = sc.loc[r["design"], c].values
        r.to_parquet(C.TABLES / "random_tier_gof.parquet", index=False)


# ----------------------------------------------------------------------------- detection (R)

def run_detection(b: int, workers: int) -> int:
    cmd = [C.rscript(), str(C.ANALYSIS / "detection_curves.R"), "--B", str(b), "--workers", str(workers)]
    C.progress("detection curves: " + " ".join(cmd[1:]))
    return subprocess.run(cmd, cwd=str(C.REBUILD)).returncode


def q4b_labels() -> pd.DataFrame:
    f = OUT / "detection_edge_vs_hl.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f)
    d["bound"] = [Rr.pair_at_bound(a, b, c, e, C.DETECTION_MARGIN)
                  for a, b, c, e in zip(d["edge_lo"], d["edge_hi"], d["hl_lo"], d["hl_hi"])]
    d["label"] = [("uninformative (" + bd + ")") if isinstance(bd, str) else Rr.detection_label(lo, hi)
                  for bd, lo, hi in zip(d["bound"], d["diff_lo"], d["diff_hi"])]
    return d


# ----------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-detection", action="store_true")
    ap.add_argument("--detection-B", type=int, default=500)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--meta-dir", default=None, help="meta-model output folder (default tables/meta)")
    a = ap.parse_args(argv)
    t0 = time.time()
    st = state()
    s = pd.read_parquet(C.TABLES / "scores.parquet")
    p = pd.read_parquet(C.TABLES / "pairs.parquet")
    fits = pd.read_parquet(C.TABLES / "fits.parquet")
    # 13.5 million rows: text columns stay arrow strings (as python objects they do not fit in memory)
    import pyarrow as pa
    import pyarrow.parquet as pq
    g = pq.read_table(C.TABLES / "gof.parquet").to_pandas(
        ignore_metadata=True, split_blocks=True, self_destruct=True,
        types_mapper=lambda t: pd.StringDtype("pyarrow") if (pa.types.is_string(t) or pa.types.is_large_string(t)) else None)
    from collect import scenario_table
    sc = scenario_table()
    core_stress = s[s["tier"] != "random"]

    fc = fit_counts(fits)
    write(fc, "fit_counts", st)
    fail_flags = set(zip(fc.loc[fc["fail_flag"].fillna(False).astype(bool), "scenario"],
                         fc.loc[fc["fail_flag"].fillna(False).astype(bool), "arm"]))
    C.progress(f"[{time.time() - t0:6.0f} s] fit counts")

    asum = arm_summaries(core_stress)
    write(asum, "arm_summaries", st, parquet=True)
    C.progress(f"[{time.time() - t0:6.0f} s] arm summaries: {len(asum)} rows")
    o = asum[(asum["column"] == C.ORACLE) & (asum["estimand"] == "o_true_ici") & (asum["subset"] == "all")]
    oracle_hi = dict(zip(o["scenario"], o["mean_hi"]))

    pr = p[p["tier"] != "random"] if len(p) else p
    paired = paired_summaries(pr, fail_flags, oracle_hi)
    write(paired, "paired_vs_flic", st, parquet=True)
    C.progress(f"[{time.time() - t0:6.0f} s] paired summaries: {len(paired)} rows")

    cs = class_shares(core_stress)
    write(cs, "classes", st, parquet=True)
    mn = min_n_table(cs, sc)
    write(mn, "min_n", st)
    q2t = q2(core_stress, pr, fail_flags)
    write(q2t, "q2_min_n_vs_flic", st)
    C.progress(f"[{time.time() - t0:6.0f} s] classes and minimum n")

    gg = g[g["tier"] != "random"] if len(g) and "tier" in g else g.iloc[0:0]
    if len(gg):
        rr = rejection_rates(gg)
        write(rr, "rejection_rates", st, parquet=True)
        q4a_pooled, q4a_per = q4a(gg)
        write(q4a_pooled, "q4a_size", st)
        write(q4a_per, "q4a_size_by_scenario", st)
    else:
        q4a_pooled = q4a_per = pd.DataFrame()
    C.progress(f"[{time.time() - t0:6.0f} s] rejection rates and size")

    q1t = q1(paired, asum)
    write(q1t, "q1_slope", st)
    q3t = q3(paired, asum)
    write(q3t, "q3_intercept", st)
    tmp = paired[paired["kind"] == "temperature"]
    write(tmp, "q3_temperature", st)
    lad, q5at, q5best = q5a(asum, core_stress, paired)
    write(lad, "q5a_ladder_means", st)
    write(q5at, "q5a_vs_flic", st)
    write(q5best, "q5a_vs_best_arm", st)
    q5bt = q5b(core_stress, gg) if len(gg) else pd.DataFrame()
    write(q5bt, "q5b_rule", st)
    C.progress(f"[{time.time() - t0:6.0f} s] Q1, Q3, Q5")

    if not a.skip_detection and len(gg):
        rc = run_detection(a.detection_B, a.workers)
        C.progress(f"[{time.time() - t0:6.0f} s] detection curves (R exit {rc})")
    q4bt = q4b_labels()
    if len(q4bt):
        write(q4bt, "q4b_detection", st)

    from pathlib import Path
    meta_dir = Path(a.meta_dir) if a.meta_dir else C.TABLES / "meta"
    random_tier_inputs(s, p, g if "tier" in g else g.iloc[0:0])
    q6t = q6(paired, meta_dir)
    if len(q6t):
        write(q6t, "q6_transfer", st)

    sof = summary_of_findings(q1t, q2t, q3t, q4a_pooled, q4bt, q5at, q5best, q5bt, q6t)
    write(sof, "summary_of_findings", st)
    san = sanity(asum, q4a_per, q4a_pooled)
    write(san, "sanity_checks", st)
    C.progress(f"summarise done in {time.time() - t0:.0f} s; data state: {st}")
    if st != "complete":
        C.progress("partial data: no TFM label is printed; the tables carry the PARTIAL banner")
    return 0


if __name__ == "__main__":
    sys.exit(main())
