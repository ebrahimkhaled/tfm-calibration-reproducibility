"""Markdown tables for ENV_REPORT.md from rebuild/timings.csv and the GPU probes.

Cells are the median over the 3 repeats of fit + predict seconds for a
10,000-row evaluation set; "s/1k" is the median predict time per 1,000
predicted rows (useful to rescale to the study's evaluation sizes).
"""
from pathlib import Path

import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "timings.csv")
ok = d[d.status == "ok"].copy()

LABEL = {"tabpfn_v2": "TabPFN v2 (Real-TabPFN-2, 8 est.)", "tabicl_v2": "TabICLv2 (8 est.)",
         "mitra_v2": "Mitra-v2 (ICL, 1 est.)", "lgbm": "LightGBM CV grid + Platt",
         "LR": "LR-MLE (glm)", "FLIC": "FLIC (logistf)", "RIDGE": "Ridge (cv.glmnet)", "GAM": "GAM (mgcv REML)"}
ORDER = list(LABEL)


def fmt(x):
    if pd.isna(x):
        return "–"
    return f"{x:.3f}" if x < 1 else (f"{x:.1f}" if x < 100 else f"{x:.0f}")


def grid_table(sub, title):
    cols = [(p, n) for p in (5, 20) for n in (100, 500, 2000)]
    lines = [f"**{title}**", "",
             "| Arm | device | threads | " + " | ".join(f"p={p}, n={n}" for p, n in cols) + " |",
             "|---|---|---|" + "---|" * len(cols)]
    med = sub.groupby(["model", "device", "threads", "p", "n_train"]).total_s.median()
    for (model, device, threads), _ in sub.groupby(["model", "device", "threads"]):
        cells = [fmt(med.get((model, device, threads, p, n))) for p, n in cols]
        lines.append(f"| {LABEL.get(model, model)} | {device} | {threads} | " + " | ".join(cells) + " |")
    lines.sort(key=lambda s: next((i for i, k in enumerate(ORDER) if s.startswith(f"| {LABEL[k]} ")), -1))
    return "\n".join(lines)


ok["key"] = ok.model.map({k: i for i, k in enumerate(ORDER)})
ok = ok.sort_values("key")
print(grid_table(ok[ok.device == "cuda"], "GPU (GTX 1660 Ti), seconds per fit + predict, median of 3"))
print()
print(grid_table(ok[(ok.device == "cpu") & (ok.impl != "R")], "CPU, Python arms, seconds per fit + predict, median of 3"))
print()
print(grid_table(ok[ok.impl == "R"], "CPU, R arms (single thread), seconds per fit + predict, median of 3"))
print()

# predict cost per 1,000 rows and peak memory for the GPU arms
g = ok[ok.device == "cuda"].groupby(["model", "p", "n_train"]).agg(
    fit=("fit_s", "median"), pred=("predict_s", "median"), peak=("peak_gpu_mb", "max")).reset_index()
print("| Arm | p | n | fit s | predict s per 1k rows | peak GPU MB |")
print("|---|---|---|---|---|---|")
for _, r in g.iterrows():
    print(f"| {LABEL[r.model]} | {r.p} | {r.n_train} | {fmt(r.fit)} | {fmt(r.pred / 10)} | {r.peak:.0f} |")
print()

skipped = d[d.status != "ok"].groupby(["model", "device", "threads", "status"]).size()
print("Non-ok rows:")
print(skipped.to_string() if len(skipped) else "none")
