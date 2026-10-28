"""Compute budget of the full design (58 scenarios) from the pilot's measured times.

    python pilot_budget.py

Reads results/pilot/tables/fits.csv (pilot_analyse.py) and results/pilot/gof/*/*.csv.
Writes results/pilot/tables/budget_*.md and budget.json.

Model fit times: the median seconds per fit of each arm in the measured cell with the same
development n, the same number of predicted rows (4,334 at pi = 3 %, 2,000 otherwise) and the
same covariate type (p = 8; T2 with p = 30; T3 with the 6-level categorical). Cells that were not
measured borrow from the nearest measured cell and are marked.

GOF times: core-seconds per replicate, split into the parts that scale with different replicate
counts (classical arms, TFM arms, side arms, and DeepGOF-1 separately so that its options can be
priced). A scenario borrows the measured pilot scenario with the same evaluation size and
covariate type (the GOF cost depends on N_eval and the covariates, not on n).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
REBUILD = CODE_DIR.parent
TAB = REBUILD / "results" / "pilot" / "tables"
sys.path.insert(0, str(CODE_DIR))
import dgp  # noqa: E402

CLASSICAL = ["oracle", "lr", "flic", "ridge", "gam", "lgbm", "lgbm_raw"]
TFM_MAIN = ["tabpfn", "tabicl", "mitra"]
SIDE = ["tabpfn_legacy", "tabpfn_t1"]


def ctype(sc) -> str:
    return {"T2": "p30", "T3": "cat"}.get(sc["family"], "p8")


def n_pred(sc) -> int:
    return max(1000, math.ceil(100 / sc["pi"] - 1e-9)) + 1000


def fit_times() -> pd.DataFrame:
    """Median seconds per fit. TabPFN-3.5 fits (folder tfm35, GPU) are filed as arm 'tabpfn35';
    the 'tabpfn' rows of the main pilot are the V2 checkpoint."""
    f = pd.read_csv(TAB / "fits.csv")
    f = f[(f.status == "ok") & f.folder.isin(["pilot", "timing", "tfm35"])].copy()
    f.loc[f.folder == "tfm35", "arm"] = "tabpfn35"
    tab = dgp.load_table().set_index("scenario")
    f["ctype"] = f.scenario.map(lambda s: ctype(tab.loc[s]))
    return f.groupby(["arm", "ctype", "n_dev", "n_pred"]).seconds.median().reset_index()


def lookup(ft: pd.DataFrame, arm: str, ct: str, n: int, npred: int) -> tuple[float, str]:
    g = ft[(ft.arm == arm)]
    exact = g[(g.ctype == ct) & (g.n_dev == n) & (g.n_pred == npred)]
    if len(exact):
        return float(exact.seconds.iloc[0]), ""
    # same type and n, other row count: scale the p = 8 ratio between row counts at this n
    p8 = g[g.ctype == "p8"]
    base = p8[(p8.n_dev == n) & (p8.n_pred == npred)]
    same_ct = g[(g.ctype == ct)]
    if len(base) and len(same_ct):
        # ratio of this type to p = 8 measured at some n, applied to the p = 8 cell
        r = []
        for _, x in same_ct.iterrows():
            b = p8[(p8.n_dev == x.n_dev) & (p8.n_pred == x.n_pred)]
            if len(b):
                r.append(x.seconds / float(b.seconds.iloc[0]))
        if r:
            return float(base.seconds.iloc[0]) * float(np.median(r)), "scaled"
    if len(base):
        return float(base.seconds.iloc[0]), "p8 cell"
    near = g[g.ctype == ct] if len(g[g.ctype == ct]) else p8
    near = near.iloc[(near.n_dev - n).abs().argsort()]
    return (float(near.seconds.iloc[0]), "nearest") if len(near) else (float("nan"), "none")



# ----------------------------------------------------------------------------- GOF cost model

def gof_parts() -> pd.DataFrame:
    """Mean core-seconds per replicate of each GOF component, per pilot scenario."""
    fs = sorted((REBUILD / "results" / "pilot" / "gof").glob("*/*.csv"))
    g = pd.concat([pd.read_csv(x, dtype={"rep": str}, keep_default_na=False, na_values=[""]) for x in fs])
    g["secs"] = pd.to_numeric(g.secs, errors="coerce").fillna(0)
    g["base"] = g.arm.str.split("__").str[0]
    lad = g.arm.str.contains("__")
    dg = g.test.str.startswith("DeepGOF")
    part = np.select(
        [dg & (g.base == "flic"), dg, g.base == "oracle", lad & (g.base == "flic"), lad,
         g.base.isin(CLASSICAL), g.base.isin(TFM_MAIN)],
        ["dg_flic", "dg_tfm", "oracle", "flic_ladder", "tfm_ladder", "classical", "tfm"], "side")
    g["part"] = part
    s = g.groupby(["scenario", "rep", "part"]).secs.sum().unstack(fill_value=0)
    return s.groupby("scenario").mean()


# measured on this machine (see PILOT_REPORT.md section 4)
CLASSICAL_FIT_S = {("p8", 200): 6.6, ("p8", 500): 3.7, ("p8", 1000): 4.7, ("p8", 3000): 8.1,
                   ("p30", 500): 10.0, ("p30", 3000): 37.9, ("cat", 500): 3.9, ("cat", 3000): 8.1}
SCORE_S_FULL = {1000: 12.5, 3334: 13.5}      # score.py core-seconds per file, every arm and ladder
SCORE_S_CLASSICAL = {1000: 5.0, 3334: 5.5}   # classical arms + FLIC ladder only (reps > R_tfm)



# TabPFN-3.5 on the CPU (8 threads), measured for the one cell that runs out of GPU memory
TABPFN35_CPU_T2_3000 = 134.0
CORES_FOR_GOF = 14          # GOF + scoring workers during the run (24 cores; GPU process, one CPU
                            # TabPFN worker with 8 threads and the author's jobs need the rest)
GOF_CONTENTION = 1.25       # pilot GOF times were measured with 7 workers; allow for 14


def tabpfn35_seconds(ft: pd.DataFrame, b: pd.DataFrame, tab: pd.DataFrame) -> tuple[list, list]:
    """Measured TabPFN-3.5 seconds where the cell was run; elsewhere the V2 time times the
    3.5 / V2 ratio measured at the same n (p = 8)."""
    t35 = ft[ft.arm == "tabpfn35"]
    v2 = ft[ft.arm == "tabpfn"]
    ratio = {}
    for _, x in t35[t35.ctype == "p8"].iterrows():
        m = v2[(v2.ctype == "p8") & (v2.n_dev == x.n_dev) & (v2.n_pred == x.n_pred)]
        if len(m):
            ratio.setdefault(int(x.n_dev), []).append(x.seconds / float(m.seconds.iloc[0]))
    ratio = {k: float(np.median(v)) for k, v in ratio.items()}
    out, how = [], []
    for _, r in b.iterrows():
        sc = tab[tab.scenario == r.scenario].iloc[0]
        if r.scenario == "T2_pi10_n3000":
            out.append(float("nan"))
            how.append("GPU OOM; CPU stream")
            continue
        e = t35[(t35.ctype == ctype(sc)) & (t35.n_dev == r.n) & (t35.n_pred == r.n_eval + 1000)]
        if len(e):
            out.append(float(e.seconds.iloc[0]))
            how.append("measured")
        else:
            out.append(r.s_tabpfn * ratio[int(r.n)])
            how.append(f"V2 x {ratio[int(r.n)]:.2f}")
    return out, how


def main() -> int:
    ft = fit_times()
    tab = dgp.load_table()
    gp = gof_parts()
    gp.round(2).to_csv(TAB / "gof_parts_per_rep.csv")

    def gof_src(sc):
        if sc["family"] == "T2":
            return ["T2_pi10_n3000"]
        if sc["family"] == "T3":
            return ["T3_pi10_n500"]
        return ["S1_pi03_n200", "S4_pi03_n3000"] if sc["pi"] < 0.05 else ["S2_pi10_n500", "S3_pi30_n1000"]

    rows = []
    for _, sc in tab.iterrows():
        ct, n, npred = ctype(sc), int(sc["n"]), n_pred(sc)
        row = dict(scenario=sc["scenario"], family=sc["family"], pi=sc["pi"], n=n, n_eval=npred - 1000)
        for arm in TFM_MAIN:
            row[f"s_{arm}"], _ = lookup(ft, arm, ct, n, npred)
        row["legacy_here"] = sc["kind"] == "core" and abs(sc["pi"] - 0.10) < 1e-9
        row["t1_here"] = row["legacy_here"] and n in (200, 3000)
        row["cls_fit"] = CLASSICAL_FIT_S.get((ct, n), CLASSICAL_FIT_S[("p8", n)])
        row["score_full"] = SCORE_S_FULL[npred - 1000]
        row["score_cls"] = SCORE_S_CLASSICAL[npred - 1000]
        src = gp.loc[[x for x in gof_src(sc) if x in gp.index]]
        for c in gp.columns:
            row[f"g_{c}"] = float(src[c].mean())
        rows.append(row)
    b = pd.DataFrame(rows)
    b["s_tabpfn35"], b["how35"] = tabpfn35_seconds(ft, b, tab)
    b.to_csv(TAB / "budget_per_scenario.csv", index=False)
    t2 = (b.family == "T2").to_numpy()
    t2_3000 = (b.scenario == "T2_pi10_n3000").to_numpy()
    n3000 = (b.n == 3000).to_numpy()

    def plan(R=200, R3000=200, R_legacy=200, R_t=0, dg_p8=1.0, dg_t2=0.5, late="oracle", R_cls=500):
        Rv = np.where(n3000, R3000, R)
        gpu = dict(tabpfn35=float(np.nansum(b.s_tabpfn35 * Rv) / 3600),
                   tabicl=float((b.s_tabicl * Rv).sum() / 3600),
                   mitra=float((b.s_mitra.where(~t2_3000, 0) * Rv).sum() / 3600),
                   legacy_v2=float(b.s_tabpfn[b.legacy_here].sum() * R_legacy / 3600),
                   temp_sens=float(b.s_tabpfn35[b.t1_here].sum() * R_t / 3600))
        gpu["total"] = sum(gpu.values())
        dgs = np.where(t2, dg_t2, dg_p8)
        early = ((b.g_oracle + b.g_classical + b.g_flic_ladder + b.g_tfm + b.g_tfm_ladder) * Rv
                 + b.g_side * b.legacy_here * min(R_legacy, R))
        dg = (b.g_dg_flic + b.g_dg_tfm) * dgs * Rv
        later_unit = b.g_oracle if late == "oracle" else b.g_oracle + b.g_classical + b.g_flic_ladder
        later = later_unit * (R_cls - Rv)
        cpu = dict(classical_fits=float(b.cls_fit.sum() * R_cls / 3600),
                   tabpfn35_cpu_T2=float(TABPFN35_CPU_T2_3000 * Rv[t2_3000].sum() * 8 / 3600),
                   gof_battery=float((early.sum() + later.sum()) * GOF_CONTENTION / 3600),
                   gof_deepgof=float(dg.sum() * GOF_CONTENTION / 3600),
                   scoring=float((b.score_full * Rv + b.score_cls * (R_cls - Rv)).sum() / 3600))
        cpu["total"] = sum(cpu.values())
        # wall clock: GPU queue alone; GOF + scoring on CORES_FOR_GOF; the CPU TabPFN worker runs
        # beside them (7.4 h at R = 200); classical fits go first on the same cores
        wall = dict(gpu_h=gpu["total"],
                    cpu_h=(cpu["classical_fits"] + cpu["gof_battery"] + cpu["gof_deepgof"] + cpu["scoring"])
                    / CORES_FOR_GOF)
        wall["days"] = max(wall["gpu_h"], wall["cpu_h"]) / 24
        return dict(gpu=gpu, cpu=cpu, wall=wall)

    plans = {
        "P1 plan as written: R 200, legacy 200, T-sensitivity 200 (8 sc), DeepGOF B 199, GOF on every classical rep":
            plan(R_legacy=200, R_t=200, dg_t2=1.0, late="all"),
        "P2 R 200; legacy 100; no T-sensitivity; DeepGOF B 199 (T2: 99); reps 201-500 GOF on the oracle only":
            plan(R_legacy=100),
        "P3 as P2 with DeepGOF B 99 everywhere":
            plan(R_legacy=100, dg_p8=0.5),
        "P4 (recommended) as P3 with R 150 at n = 3000":
            plan(R3000=150, R_legacy=100, dg_p8=0.5),
        "P5 lean: R 150 everywhere; legacy 100; DeepGOF B 99":
            plan(R=150, R3000=150, R_legacy=100, dg_p8=0.5),
    }

    # Balance: move part of the TabPFN-3.5 queue to CPU workers (whole scenarios, so each scenario
    # stays on one device) until GPU and CPU finish together. f = CPU time / GPU time per fit with
    # 8 threads: 2.4 measured (S2_pi10_n500, TabPFN-3.5, idle machine); 3.0 allows for load.
    for v in plans.values():
        g, c = v["gpu"], v["cpu"]
        v["balanced"] = {}
        for cores in (16, 20):
            for f in (2.4, 3.0):
                k = 8 * f
                x = (cores * g["total"] - c["total"]) / (cores + k)
                x = max(0.0, min(x, g["tabpfn35"]))
                v["balanced"][f"{cores} cores, f {f}"] = dict(moved_gpu_h=x, wall_h=g["total"] - x,
                                                             days=(g["total"] - x) / 24)
    (TAB / "budget.json").write_text(json.dumps(plans, indent=1))
    bl = ["| plan | 16 cores, f 2.4 | 16 cores, f 3.0 | 20 cores, f 2.4 | 20 cores, f 3.0 |", "|---|---|---|---|---|"]
    for name, v in plans.items():
        bl.append(f"| {name} | " + " | ".join(f"{d['days']:.1f} d (move {d['moved_gpu_h']:.0f} GPU-h)"
                                              for d in v["balanced"].values()) + " |")
    (TAB / "budget_balanced.md").write_text("\n".join(bl), encoding="utf-8")
    print("\n".join(bl))
    lines = ["| plan | GPU h: TabPFN-3.5 / TabICL / Mitra / legacy V2 / T-sens | GPU h total | "
             "CPU core-h: classical fits / TabPFN-3.5 T2 on CPU / GOF battery / DeepGOF / scoring | "
             "wall h: GPU / CPU (14 cores) | days |", "|---|---|---|---|---|---|"]
    for name, v in plans.items():
        g, c, w = v["gpu"], v["cpu"], v["wall"]
        lines.append(f"| {name} | {g['tabpfn35']:.0f} / {g['tabicl']:.0f} / {g['mitra']:.0f} / {g['legacy_v2']:.0f} / "
                     f"{g['temp_sens']:.0f} | {g['total']:.0f} | {c['classical_fits']:.0f} / {c['tabpfn35_cpu_T2']:.0f} / "
                     f"{c['gof_battery']:.0f} / {c['gof_deepgof']:.0f} / {c['scoring']:.0f} | "
                     f"{w['gpu_h']:.0f} / {w['cpu_h']:.0f} | {w['days']:.1f} |")
    (TAB / "budget.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    cols = ["scenario", "s_tabpfn35", "how35", "s_tabpfn", "s_tabicl", "s_mitra", "cls_fit"]
    (TAB / "budget_per_scenario.txt").write_text(b[cols].round(1).to_string())
    print(b[cols].round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
