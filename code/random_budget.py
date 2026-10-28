"""Compute budget of the random-design tier (pre-registration 3.9 and 9.3): choose D.

Per-fit costs, all measured on this machine:
  TFMs on the GPU   pilot medians (results/pilot/tables/fits.csv: the TabPFN-3.5, TabICLv2 and
                    Mitra-v2 cells) and the memory probe (results/random_memory_probe.csv, first fit of
                    each arm left out: it carries the warm-up), fitted as log(seconds) on log n,
                    log(predicted rows) and log p, one model per arm
  TabPFN-3.5 on CPU 3.0 x the GPU time with 6 threads (pilot: 2.4 with 8 threads on a quiet machine;
                    the 2026-10-07 smoke: 2.7-3.1 with 6 threads)
  classical arms    results/random_cost_probe (50 random designs, smoke stream): wall seconds per
                    replicate, 1 thread
  test layer        the same 50 designs: core-seconds per design for the classical arms, plus three TFM
                    arms at the measured per-call costs (battery and edge.gof on Y and Y*, localization on
                    Y and Y*, DeepGOF-1 at B = 99 on Y)
  scoring           2 s per design

    python code/random_budget.py [--D 600,800,1000]    -> results/random_budget.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CODE = Path(__file__).resolve().parent
REB = CODE.parent
sys.path.insert(0, str(CODE))
import dgp_random  # noqa: E402

CPU_FACTOR = 3.0
SCORE_S = 2.0
# main-run clock (pre-registration 9.3): hours from the start at which each stream ends (mid values)
T_CPU_B, T_CPU_A, T_GPU, T_END = 64.0, 70.0, 79.0, 82.0
GOF_WORKERS_DURING, GOF_WORKERS_AFTER = 6, 12
CLASSICAL_WORKERS = 6


def tfm_time_models() -> dict:
    f = pd.read_csv(REB / "results/pilot/tables/fits.csv")
    f["p"] = np.where(f.scenario.str.startswith("T2"), 30, 8)
    keep = ((f.arm == "tabpfn") & (f.model == "V3_5") & (f.folder == "tfm35")) | \
           (f.arm.isin(["tabicl", "mitra"]) & f.folder.isin(["pilot", "timing"]))
    g = f[keep & (f.status == "ok")].groupby(["arm", "n_dev", "n_pred", "p"]).seconds.median().reset_index()
    g.columns = ["arm", "n", "n_pred", "p", "seconds"]
    pr = pd.read_csv(REB / "results/random_memory_probe.csv")
    first = pr.groupby("arm").head(1).index          # the first fit of each arm carries the warm-up
    pr = pr[(pr.status == "ok") & ~pr.index.isin(first)][["arm", "n", "n_pred", "p", "seconds"]]
    d = pd.concat([g, pr], ignore_index=True)
    out = {}
    for arm, s in d.groupby("arm"):
        A = np.column_stack([np.ones(len(s)), np.log(s.n), np.log(s.n_pred), np.log(s.p)])
        coef, *_ = np.linalg.lstsq(A, np.log(s.seconds), rcond=None)
        resid = np.log(s.seconds) - A @ coef
        out[arm] = dict(coef=coef.tolist(), points=len(s), rmse_log=float(np.sqrt(np.mean(resid ** 2))),
                        max_ratio=float(np.exp(np.abs(resid).max())))
    return out, d


def predict(m: dict, n, n_pred, p):
    c = np.asarray(m["coef"])
    return np.exp(c[0] + c[1] * np.log(n) + c[2] * np.log(n_pred) + c[3] * np.log(p))


def cpu_side_costs() -> dict:
    """Classical-arm wall seconds and test-layer core-seconds per design, from the cost sample."""
    pred = REB / "results/random_cost_probe/preds"
    rows = []
    for f in sorted(pred.rglob("*.parquet")):
        m = json.loads(pq.read_schema(f).metadata[b"tfmcal"])
        rows.append(dict(scenario=m["scenario"], classical_s=sum(h["seconds"] for h in m["history"])))
    c = pd.DataFrame(rows)
    gofs = []
    for f in sorted((REB / "results/random_cost_probe/gof").rglob("*.csv.gz")):
        g = pd.read_csv(f)
        fam = np.select([g.test.str.startswith("edge.gof"), g.family == "Localize", g.family == "DeepGOF"],
                        ["edge", "localize", "deepgof"], "battery")
        g["kind"] = fam
        s = g.groupby(["arm", "ytype", "kind"]).secs.sum().reset_index()
        s["scenario"] = g.scenario.iloc[0]
        gofs.append(s)
    G = pd.concat(gofs, ignore_index=True)
    per_design = G.groupby("scenario").secs.sum().rename("gof_classical_s").reset_index()
    bat = G[G.kind.isin(["battery", "edge"])].groupby(["scenario", "arm", "ytype"]).secs.sum()
    loc = G[G.kind == "localize"].groupby(["scenario", "arm", "ytype"]).secs.sum()
    dg = G[G.kind == "deepgof"].groupby(["scenario", "arm"]).secs.sum()
    calls = dict(battery_s=float(bat.mean()), localize_s=float(loc.mean()), deepgof_s=float(dg.mean()))
    t = c.merge(per_design, on="scenario")
    # three TFM arms: battery + edge on Y and Y*, localization on Y and Y*, DeepGOF-1 on Y, at the
    # design's own per-call costs (FLIC's DeepGOF-1 call stands for a TFM's)
    per = t.set_index("scenario")
    b_s = bat.groupby("scenario").mean()
    l_s = loc.groupby("scenario").mean()
    d_s = dg.groupby("scenario").mean()
    per["gof_tfm_s"] = 3 * (2 * b_s + 2 * l_s + d_s)
    per["gof_total_s"] = per.gof_classical_s + per.gof_tfm_s
    return dict(per_design=per.reset_index(), calls=calls)


def budget(D: int, tm: dict, cpu: dict, scale: float = 1.0) -> dict:
    """scale multiplies every per-fit and per-call cost (2.0 = the pessimistic case)."""
    t = dgp_random.load_table().head(D)
    n, ne, p = t.n.to_numpy(float), t.n_eval.to_numpy(float), t.p.to_numpy(float)
    on_gpu = ((t.n + t.n_eval) * t.p <= dgp_random.TABPFN_GPU_MAX_CELLS).to_numpy()
    mitra_ok = (t.n.astype(float) ** 2 * t.p <= dgp_random.MITRA_MAX_N2P).to_numpy()
    tab = scale * predict(tm["tabpfn"], n, ne, p)
    other = predict(tm["tabicl"], n, ne, p).sum() + (predict(tm["mitra"], n, ne, p) * mitra_ok).sum()
    gpu_h = (scale * other + (tab * on_gpu).sum()) / 3600
    cpu_tabpfn_h = (CPU_FACTOR * tab * ~on_gpu).sum() / 3600          # wall hours on one 6-thread worker
    pd_ = cpu["per_design"]
    classical_core_h = scale * D * pd_.classical_s.mean() / 3600
    gof_core_h = scale * D * (pd_.gof_total_s.mean() + SCORE_S) / 3600
    # schedule: R classical arms on 6 workers when CPU worker B is done, then TabPFN-3.5 on two CPU
    # workers (the slots of A and B, half of the CPU designs each); TFMs on the GPU when the GPU
    # stream is done; the test layer and scoring from the end of the main run's last test pass
    t_class_end = T_CPU_B + classical_core_h / CLASSICAL_WORKERS
    t_cpu_end = max(T_CPU_A + cpu_tabpfn_h / 2, t_class_end + cpu_tabpfn_h / 2)
    t_gpu_end = T_GPU + gpu_h
    t_fits_end = max(t_cpu_end, t_gpu_end, t_class_end)
    # test layer: 6 workers while fits run (from T_END), 12 after; a last pass of about 1 hour
    gof_done_during = max(0.0, t_fits_end - T_END) * GOF_WORKERS_DURING
    rest = max(0.0, gof_core_h - gof_done_during)
    t_end = max(t_fits_end, T_END) + rest / GOF_WORKERS_AFTER + 0.5
    return dict(D=D, scale=scale, gpu_designs=int(on_gpu.sum()), cpu_designs=int((~on_gpu).sum()),
                mitra_not_run=int((~mitra_ok).sum()), gpu_h=gpu_h, cpu_tabpfn_h=cpu_tabpfn_h,
                classical_core_h=classical_core_h, gof_core_h=gof_core_h,
                t_gpu_end=t_gpu_end, t_cpu_end=t_cpu_end, t_end=t_end, added_h=t_end - T_END)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    Ds = [int(x) for x in (argv[argv.index("--D") + 1] if "--D" in argv else "600,800,1000").split(",")]
    tm, pts = tfm_time_models()
    cpu = cpu_side_costs()
    lines = ["# Random-design tier: compute budget", "",
             "Built by `code/random_budget.py` from measured per-fit costs (see its docstring).", "",
             "## Per-fit time models (GPU seconds = exp(a + b log n + c log rows + d log p))", "",
             "| arm | points | a | b (n) | c (rows) | d (p) | RMSE of log | worst ratio |", "|---|---|---|---|---|---|---|---|"]
    for arm, m in tm.items():
        a, b, c, d = m["coef"]
        lines.append(f"| {arm} | {m['points']} | {a:.2f} | {b:.3f} | {c:.3f} | {d:.3f} | {m['rmse_log']:.2f} | {m['max_ratio']:.2f} |")
    pdd = cpu["per_design"]
    lines += ["", "## CPU side, from the 50-design cost sample (R0006-R0055, smoke stream, a loaded machine)", "",
              f"- classical arms: {pdd.classical_s.mean():.1f} s per design (median {pdd.classical_s.median():.1f}, max {pdd.classical_s.max():.1f})",
              f"- test layer, classical arms: {pdd.gof_classical_s.mean():.1f} core-s per design; per call: battery + edge.gof "
              f"{cpu['calls']['battery_s']:.2f} s, localization {cpu['calls']['localize_s']:.2f} s, DeepGOF-1 (B = 99) {cpu['calls']['deepgof_s']:.1f} s",
              f"- test layer with the three TFM arms added: {pdd.gof_total_s.mean():.1f} core-s per design", "",
              "## Totals and schedule (hours from the start of the main run)", "",
              f"Main-run clock assumed: CPU worker B done at {T_CPU_B:.0f} h, A at {T_CPU_A:.0f} h, GPU at {T_GPU:.0f} h, "
              f"last test pass at {T_END:.0f} h.", "",
              "| D | costs | TabPFN on GPU / CPU | Mitra not run | GPU h | CPU TabPFN h (one 6-thread worker) | classical core-h | test layer + scoring core-h | GPU done | CPU done | tier done | added wall h |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res = [budget(D, tm, cpu) for D in Ds] + [budget(max(Ds), tm, cpu, scale=2.0)]
    for r in res:
        lines.append(f"| {r['D']} | {'measured' if r['scale'] == 1 else 'x' + format(r['scale'], 'g')} | {r['gpu_designs']} / {r['cpu_designs']} | {r['mitra_not_run']} | {r['gpu_h']:.1f} | "
                     f"{r['cpu_tabpfn_h']:.1f} | {r['classical_core_h']:.1f} | {r['gof_core_h']:.1f} | {r['t_gpu_end']:.1f} | "
                     f"{r['t_cpu_end']:.1f} | {r['t_end']:.1f} | **{r['added_h']:.1f}** |")
    out = REB / "results" / "random_budget.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
