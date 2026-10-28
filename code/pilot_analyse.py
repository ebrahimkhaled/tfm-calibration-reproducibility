"""Summaries of the Phase-1 pilot (see pilot_run.sh). Reads only; writes results/pilot/tables/.

    python pilot_analyse.py

Inputs
    results/pilot/preds        prediction files (6 scenarios x 20 reps, every arm)
    results/pilot/metrics      score.py output for those files
    results/pilot/gof          GOF layer output
    results/pilot_timing       extra TFM fits for the n x event-rate cells the pilot misses
    results/pilot_concurrency  the same fits made by two GPU processes at once
    logs/pilot_*               step times, GPU utilisation, CPU load

Every mean comes with its Monte Carlo SE; nothing here draws a conclusion.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CODE_DIR = Path(__file__).resolve().parent
REBUILD = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))
import metrics  # noqa: E402

RES = REBUILD / "results"
OUT = RES / "pilot" / "tables"
SCEN = ["S1_pi03_n200", "S2_pi10_n500", "S3_pi30_n1000", "S4_pi03_n3000", "T2_pi10_n3000", "T3_pi10_n500"]
ARM_ORDER = ["oracle", "lr", "flic", "ridge", "gam", "lgbm", "lgbm_raw", "tabpfn", "tabpfn_legacy",
             "tabpfn_t1", "tabicl", "mitra"]
TFMS = ["tabpfn", "tabpfn_legacy", "tabpfn_t1", "tabicl", "mitra"]
MAIN_TESTS = ["EDGE (G=10)", "EDGE (G=auto)", "Cox recalibration", "Calibration-in-the-large",
              "Spiegelhalter-z", "GiViTI", "Hosmer-Lemeshow (external)", "Stukel (offset)",
              "le Cessie (external)", "Localize (any part named)", "DeepGOF-1 external (combined)"]
SHORT = {"EDGE (G=10)": "EDGE10", "EDGE (G=auto)": "EDGEa", "Cox recalibration": "Cox",
         "Calibration-in-the-large": "CITL", "Spiegelhalter-z": "Spieg", "GiViTI": "GiViTI",
         "Hosmer-Lemeshow (external)": "HL", "Stukel (offset)": "Stukel", "le Cessie (external)": "leCes",
         "Localize (any part named)": "Loc", "DeepGOF-1 external (combined)": "DG1"}
# first row of each call type in a GOF file (the row that carries the call's wall time)
CALL_START = {"EDGE (G=10)": "battery", "edge.gof default (G = auto)": "edge.gof",
              "Localize (any part named)": "localize", "DeepGOF-1 external (combined)": "deepgof"}


def arm_key(a: str) -> int:
    return ARM_ORDER.index(a) if a in ARM_ORDER else 99


def f(x, d=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.{d}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write(name: str, text: str):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------------- fits

def read_fits(folder: Path, label: str) -> pd.DataFrame:
    rows = []
    for p in sorted(folder.rglob("*.parquet")):
        meta = json.loads(pq.read_schema(p).metadata[b"tfmcal"])
        hist = meta.get("history", [])
        for arm, i in meta["arms"].items():
            # the write that fitted this arm: its pid tells which process (and warm-up) it was
            h = next((h for h in hist if arm in h.get("arms", [])), {})
            rows.append(dict(folder=label, scenario=meta["scenario"], rep=meta["rep"], arm=arm,
                             status=i.get("status"), error=str(i.get("error", ""))[:200],
                             seconds=i.get("seconds"), fit_seconds=i.get("fit_seconds"),
                             predict_seconds=i.get("predict_seconds"),
                             r_call_seconds=i.get("r_call_seconds"), peak_gpu_mb=i.get("peak_gpu_mb"),
                             n_boundary=i.get("n_boundary"), model=i.get("model_version") or i.get("checkpoint")
                             or i.get("revision"), checkpoint=i.get("checkpoint"),
                             n_estimators=i.get("n_estimators"), temperature=i.get("softmax_temperature"),
                             context_rows=i.get("context_rows"), events_dev=meta["events"]["dev"],
                             n_dev=meta["sizes"]["dev"], n_pred=meta["sizes"]["eval"] + meta["sizes"]["recal"]
                             + meta["sizes"]["probe"], pid=h.get("pid"), time=h.get("time"),
                             separation=i.get("separation_flag"), lambda_edge=i.get("lambda_at_edge")))
    return pd.DataFrame(rows)


def fit_summary(fits: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (fo, sc, arm), g in fits.groupby(["folder", "scenario", "arm"]):
        ok = g[g.status == "ok"]
        s = ok.seconds.dropna()
        out.append(dict(folder=fo, scenario=sc, arm=arm, R=len(g), ok=len(ok), failed=int((g.status != "ok").sum()),
                        med_s=f(s.median(), 1) if len(s) else "-", p90_s=f(s.quantile(.9), 1) if len(s) else "-",
                        max_s=f(s.max(), 1) if len(s) else "-",
                        peak_mb=f(ok.peak_gpu_mb.max(), 0) if ok.peak_gpu_mb.notna().any() else "-",
                        errors="; ".join(sorted({e[:90] for e in g[g.status != "ok"].error}))[:240]))
    d = pd.DataFrame(out)
    d["k"] = d.arm.map(arm_key)
    return d.sort_values(["folder", "scenario", "k"]).drop(columns="k")


# ----------------------------------------------------------------------------- oracle metrics

def load_metrics() -> pd.DataFrame:
    fs = sorted((RES / "pilot" / "metrics").glob("*.parquet"))
    return pd.concat([pd.read_parquet(x) for x in fs], ignore_index=True) if fs else pd.DataFrame()


def msd(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))), float(x.std(ddof=1))


def oracle_tables(m: pd.DataFrame):
    r0 = m[(m.ladder == "R0")]
    rows, req, pair = [], [], []
    for sc in SCEN:
        g = r0[r0.scenario == sc]
        fl = g[(g.arm == "flic") & (g.status == "ok")].set_index("rep")
        for arm in sorted(g.arm.unique(), key=arm_key):
            a = g[g.arm == arm]
            ok = a[(a.status == "ok") & a.o_true_ici.notna()]
            nfail = int(len(a) - len(ok))
            ici, ici_se, ici_sd = msd(ok.o_true_ici)
            lli, lli_se, _ = msd(ok.o_true_ll_ici)
            oce, oce_se, _ = msd(ok.o_oce)
            e90, e90_se, _ = msd(ok.o_true_e90)
            b_mean, b_se, b_sd = msd(ok.o_true_slope)
            b = ok.o_true_slope.dropna().to_numpy()
            b_med = float(np.median(b)) if len(b) else float("nan")
            b_med_se = metrics.mcse_bootstrap(b) if len(b) > 2 else float("nan")
            a_mean, a_se, _ = msd(ok.o_true_intercept)
            auc, auc_se, _ = msd(ok.o_true_auc)
            eici, eici_se, _ = msd(ok.e_emp_ici)
            k = int(ok.cls_weak.sum()) if len(ok) else 0
            wr = k / len(ok) if len(ok) else float("nan")
            rows.append(dict(scenario=sc, arm=arm, ok=len(ok), failed=nfail,
                             true_ICI=f"{f(ici)} ({f(ici_se)})", true_ll_ICI=f"{f(lli)} ({f(lli_se)})",
                             OCE=f"{f(oce)} ({f(oce_se)})", E90=f"{f(e90)} ({f(e90_se)})",
                             b_med=f"{f(b_med, 2)} ({f(b_med_se, 2)})", b_mean=f"{f(b_mean, 2)} ({f(b_se, 2)})",
                             a_mean=f"{f(a_mean, 2)} ({f(a_se, 2)})", AUC=f"{f(auc, 3)} ({f(auc_se, 3)})",
                             emp_ICI=f"{f(eici)} ({f(eici_se)})",
                             weak=f"{f(wr, 2)} ({f(metrics.mcse_proportion(k, len(ok)) if len(ok) else float('nan'), 2)})"))
            # replicates needed for the plan's MCSE targets (section 4.7)
            target = 0.002 if arm in TFMS else 0.001
            req.append(dict(scenario=sc, arm=arm, R=len(ok), sd_true_ici=f(ici_sd),
                            target=target, R_needed=int(math.ceil((ici_sd / target) ** 2)) if math.isfinite(ici_sd) else "-",
                            sd_b=f(b_sd, 3), R_for_b_se_0p02=int(math.ceil((b_sd / 0.02) ** 2)) if math.isfinite(b_sd) else "-"))
            if arm != "flic" and len(fl):
                j = ok.set_index("rep").join(fl[["o_true_ici", "o_true_slope"]], rsuffix="_flic", how="inner")
                if len(j) > 2:
                    d_ici = (j.o_true_ici - j.o_true_ici_flic).to_numpy()
                    d_b = (j.o_true_slope - j.o_true_slope_flic).to_numpy()
                    di, di_se, di_sd = msd(d_ici)
                    db, db_se, db_sd = msd(d_b)
                    pair.append(dict(scenario=sc, arm=arm, R_pairs=len(j),
                                     d_true_ICI=f"{f(di)} ({f(di_se)})", sd_d=f(di_sd),
                                     R_for_d_se_0p001=int(math.ceil((di_sd / 0.001) ** 2)),
                                     d_b=f"{f(db, 3)} ({f(db_se, 3)})"))
    return pd.DataFrame(rows), pd.DataFrame(req), pd.DataFrame(pair)


def ladder_table(m: pd.DataFrame) -> pd.DataFrame:
    lad = m[m.arm.isin(["flic", "tabpfn", "tabicl", "mitra"])]
    out = []
    for sc in SCEN:
        for arm in ["flic", "tabpfn", "tabicl", "mitra"]:
            row = dict(scenario=sc, arm=arm)
            for lev in ["R0", "R1_m100", "R2_m100", "R3_m100", "R1_m500", "R2_m500", "R3_m500"]:
                g = lad[(lad.scenario == sc) & (lad.arm == arm) & (lad.ladder == lev)]
                ok = g[(g.status == "ok") & g.o_true_ici.notna()]
                mu, se, _ = msd(ok.o_true_ici)
                nf = len(g) - len(ok)
                row[lev] = f"{f(mu, 4)} ({f(se, 4)})" + (f" [{nf}F]" if nf else "")
            out.append(row)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- GOF layer

def load_gof() -> pd.DataFrame:
    fs = sorted((RES / "pilot" / "gof").glob("*/*.csv"))
    if not fs:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(x, dtype={"rep": str}, keep_default_na=False, na_values=[""]) for x in fs],
                  ignore_index=True)
    d["p_value"] = pd.to_numeric(d.p_value, errors="coerce")
    d["secs"] = pd.to_numeric(d.secs, errors="coerce")
    return d


def rate(p):
    p = np.asarray(p, dtype=float)
    ran = np.isfinite(p)
    k, r = int((p[ran] <= 0.05).sum()), int(ran.sum())
    return k, r, int((~ran).sum())


def fmt_rate(k, r):
    if r == 0:
        return "-"
    return f"{k / r:.3f} ({metrics.mcse_proportion(k, r):.3f})"


def gof_tables(g: pd.DataFrame):
    base = g[~g.arm.str.contains("__")]                    # R0 columns only
    size_rows, real_rows = [], []
    for sc in SCEN + ["pooled"]:
        s = base if sc == "pooled" else base[base.scenario == sc]
        row = dict(scenario=sc)
        for t in MAIN_TESTS:
            k, r, nr = rate(s[(s.ytype == "ystar") & (s.test == t)].p_value)
            row[SHORT[t]] = fmt_rate(k, r) + (f" [{nr} NR]" if nr else "")
        size_rows.append(row)
    # size by arm, pooled over the scenarios (exact null Y* ~ Bern(p_hat)) plus the oracle on real Y
    by_arm = []
    for arm in sorted(base.arm.unique(), key=arm_key):
        s = base[(base.arm == arm) & (base.ytype == "ystar")]
        row = dict(arm=arm)
        for t in MAIN_TESTS:
            k, r, nr = rate(s[s.test == t].p_value)
            row[SHORT[t]] = fmt_rate(k, r)
        by_arm.append(row)
    s = base[(base.arm == "oracle") & (base.ytype == "real")]
    row = dict(arm="oracle, real Y")
    for t in MAIN_TESTS:
        k, r, nr = rate(s[s.test == t].p_value)
        row[SHORT[t]] = fmt_rate(k, r)
    by_arm.append(row)
    for sc in SCEN:
        s = base[(base.scenario == sc) & (base.ytype == "real")]
        for arm in sorted(s.arm.unique(), key=arm_key):
            row = dict(scenario=sc, arm=arm)
            for t in MAIN_TESTS:
                k, r, nr = rate(s[(s.arm == arm) & (s.test == t)].p_value)
                row[SHORT[t]] = (f"{k}/{r}" if r else "-") + (f" [{nr} NR]" if nr else "")
            real_rows.append(row)
    return pd.DataFrame(size_rows), pd.DataFrame(by_arm), pd.DataFrame(real_rows)


def gof_not_run(g: pd.DataFrame) -> pd.DataFrame:
    s = g[g.test.isin(MAIN_TESTS) & g.p_value.isna()]
    if s.empty:
        return pd.DataFrame()
    s = s.assign(reason=s.note.astype(str).str.slice(0, 70))
    return (s.groupby(["scenario", "arm", "test", "reason"]).size().reset_index(name="count")
            .sort_values(["scenario", "test", "arm"]))


def gof_timing(g: pd.DataFrame):
    calls = g[g.test.isin(CALL_START) & g.secs.notna()].copy()
    calls["call"] = calls.test.map(CALL_START)
    calls["ladder"] = calls.arm.str.contains("__")
    t1 = (calls.groupby(["scenario", "call", "ladder"]).secs
          .agg(["count", "median", "mean", "max"]).round(2).reset_index())
    per_rep = g.groupby(["scenario", "rep"]).secs.sum().groupby("scenario").agg(["count", "mean", "median", "max"]).round(0)
    share = calls.groupby(["scenario", "call"]).secs.sum().unstack(fill_value=0)
    share = (share.div(share.sum(axis=1), axis=0) * 100).round(1)
    return t1, per_rep.reset_index(), share.reset_index()


# ----------------------------------------------------------------------------- concurrency and grid

def compare_preds(dir_a: Path, dir_b: Path, arms: list[str]) -> pd.DataFrame:
    out = []
    for fb in sorted(dir_b.rglob("*.parquet")):
        fa = dir_a / fb.relative_to(dir_b)
        if not fa.exists():
            continue
        a, b = pq.read_table(fa).to_pandas(), pq.read_table(fb).to_pandas()
        for arm in arms:
            c = f"p_{arm}"
            if c in a and c in b:
                d = np.abs(a[c].to_numpy(float) - b[c].to_numpy(float))
                out.append(dict(file=str(fb.relative_to(dir_b)), arm=arm, max_abs_diff=float(np.nanmax(d))))
    return pd.DataFrame(out)


def step_windows() -> pd.DataFrame:
    p = REBUILD / "logs" / "pilot_steps.log"
    rows = []
    for line in p.read_text().splitlines():
        if line.startswith("["):
            ts, msg = line[1:20], line[22:]
            rows.append(dict(time=pd.Timestamp(ts), msg=msg))
    return pd.DataFrame(rows)


def util_by_step() -> pd.DataFrame:
    w = step_windows()
    gpu = pd.read_csv(REBUILD / "logs" / "pilot_gpu_util.csv", header=None, names=["time", "util", "mem"])
    gpu["time"] = pd.to_datetime(gpu.time.str.strip(), format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    gpu["util"] = gpu.util.str.replace("%", "").astype(float)
    gpu["mem"] = gpu.mem.str.replace("MiB", "").astype(float)
    cpu = pd.read_csv(REBUILD / "logs" / "pilot_cpu_load.csv", header=None, names=["time", "load"],
                      skiprows=1, on_bad_lines="skip")
    cpu["time"] = pd.to_datetime(cpu.time, format="%m/%d/%Y %H:%M:%S.%f", errors="coerce")
    cpu["load"] = pd.to_numeric(cpu.load, errors="coerce")
    starts = w[w.msg.str.match(r"step \d:")].reset_index(drop=True)
    out = []
    for i, r in starts.iterrows():
        end = starts.time[i + 1] if i + 1 < len(starts) else w.time.max()
        gg = gpu[(gpu.time >= r.time) & (gpu.time < end)]
        cc = cpu[(cpu.time >= r.time) & (cpu.time < end)]
        out.append(dict(step=r.msg[:60], start=str(r.time), minutes=round((end - r.time).total_seconds() / 60, 1),
                        gpu_util_mean=round(gg.util.mean(), 0) if len(gg) else "-",
                        gpu_util_p90=round(gg.util.quantile(.9), 0) if len(gg) else "-",
                        gpu_mem_max_mb=round(gg.mem.max(), 0) if len(gg) else "-",
                        cpu_load_mean=round(cc.load.mean(), 0) if len(cc) else "-"))
    return pd.DataFrame(out)


def main() -> int:
    fits = pd.concat([read_fits(RES / "pilot" / "preds", "pilot"),
                      read_fits(RES / "pilot_timing" / "preds", "timing"),
                      read_fits(RES / "pilot_concurrency" / "preds", "concurrent"),
                      # TabPFN-3.5 (arrived 2026-10-07 08:15, after the main pilot): GPU and CPU
                      read_fits(RES / "pilot_tabpfn35" / "preds", "tfm35"),
                      read_fits(RES / "pilot_tabpfn35_cpu" / "preds", "tfm35_cpu")], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fits.to_csv(OUT / "fits.csv", index=False)
    write("fit_summary.md", md_table(fit_summary(fits)))

    m = load_metrics()
    if len(m):
        o, req, pair = oracle_tables(m)
        write("oracle.md", md_table(o))
        write("mcse_requirements.md", md_table(req))
        write("paired_vs_flic.md", md_table(pair))
        write("ladder.md", md_table(ladder_table(m)))
        m.to_csv(OUT / "metrics_all.csv", index=False)

    g = load_gof()
    if len(g):
        size, by_arm, real = gof_tables(g)
        write("gof_size_by_scenario.md", md_table(size))
        write("gof_size_by_arm.md", md_table(by_arm))
        write("gof_real.md", md_table(real))
        nr = gof_not_run(g)
        write("gof_not_run.md", md_table(nr) if len(nr) else "none")
        t1, per_rep, share = gof_timing(g)
        write("gof_timing_calls.md", md_table(t1))
        write("gof_timing_per_rep.md", md_table(per_rep))
        write("gof_timing_share.md", md_table(share))

    conc = compare_preds(RES / "pilot" / "preds", RES / "pilot_concurrency" / "preds", ["tabpfn", "tabicl", "mitra"])
    grid = compare_preds(RES / "pilot" / "preds", RES / "pilot_timing" / "preds", ["tabpfn", "tabicl", "mitra"])
    write("repro_concurrent_vs_solo.md", md_table(conc.groupby("arm").max_abs_diff.agg(["count", "max"]).reset_index())
          if len(conc) else "none")
    write("repro_loaded_vs_solo.md", md_table(grid.groupby("arm").max_abs_diff.agg(["count", "max"]).reset_index())
          if len(grid) else "none")
    try:
        write("util_by_step.md", md_table(util_by_step()))
    except Exception as e:  # noqa: BLE001
        write("util_by_step.md", f"could not parse: {e}")
    print("tables in", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
