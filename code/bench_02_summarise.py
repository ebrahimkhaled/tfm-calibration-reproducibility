"""Merge the Python and R timing files into rebuild/timings.csv and print a
median table (seconds per fit+predict, 10,000 evaluation rows) for the report.

Run with any Python that has pandas:  python code/bench_02_summarise.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "results" / "bench"

parts = [pd.read_csv(f) for f in (B / "timings_py.csv", B / "timings_r.csv") if f.exists()]
d = pd.concat(parts, ignore_index=True)
d = d.sort_values(["model", "device", "p", "n_train", "rep"])
d.to_csv(ROOT / "timings.csv", index=False)

ok = d[d.status == "ok"]
tab = (ok.groupby(["model", "device", "threads", "p", "n_train"])
         .agg(median_s=("total_s", "median"), max_s=("total_s", "max"), reps=("rep", "count"),
              peak_gpu_mb=("peak_gpu_mb", "max"), auc=("auc", "mean"))
         .reset_index())
wide = tab.pivot_table(index=["model", "device", "threads"], columns=["p", "n_train"], values="median_s")
pd.set_option("display.width", 200)
print(wide.round(2).to_string())
print()
print(tab.groupby(["model", "device"]).peak_gpu_mb.max().dropna().round(0).to_string())
print()
print(d.groupby(["model", "device", "status"]).size().to_string())
tab.to_csv(B / "timings_summary.csv", index=False)
