"""Figure 8. Real-data learning curves (pre-registration 10.2): calibration on the external
evaluation set against development size n, per data set, for FLIC, LightGBM + Platt and the three
TFMs. Two-column width (183 mm).

    row a  empirical ICI, mean +/- 1.96 MCSE over the B = 100 subsamples
    row b  empirical calibration slope, median and inter-quartile range (6.1: medians for slopes,
           because single separated fits give extreme values); reference line at 1
    row c  empirical calibration-in-the-large (logit scale, slope fixed at 1), median and IQR;
           reference line at 0

Input: analysis/tables/realdata/learning_curves.parquet (code/run_realdata.py summarise; columns
dataset, n, b, arm, ladder, emp_ici, emp_slope, emp_intercept, ...). CRASH-2 is included: its Chief
Investigators allowed the publication of aggregate results on 2026-10-23 (DEVIATIONS D-02);
--without-crash2 draws the version without it.

    python analysis/figures/fig8_realdata.py [--without-crash2] [--arms flic,lgbm,tabpfn,tabicl,mitra]
"""

import argparse
import sys

import numpy as np
import pandas as pd

import _style as S
from _style import C, plt

DATASETS = ["ist_death14", "ist_dd6", "crash2_death28", "sepsis_death", "physionet_ihd"]
DATASET_LABEL = {"ist_death14": "IST, death at 14 days", "ist_dd6": "IST, dead or dependent at 6 months",
                 "crash2_death28": "CRASH-2, death at 28 days", "sepsis_death": "Sepsis, death",
                 "physionet_ihd": "PhysioNet 2012, in-hospital death"}
ROWS = [("emp_ici", "Empirical ICI", None, "mean"),
        ("emp_slope", "Calibration slope", 1.0, "median"),
        ("emp_intercept", "Calibration-in-the-large", 0.0, "median")]


def load(without_crash2: bool) -> pd.DataFrame:
    f = C.TABLES / "realdata" / "learning_curves.parquet"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_parquet(f)
    if without_crash2:
        d = d[d["dataset"] != "crash2_death28"]
    return d[d["ladder"] == "R0"]


def wrap(s: str, width: int = 22) -> str:
    words, lines, cur = s.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return "\n".join(lines + [cur])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--without-crash2", action="store_true")
    ap.add_argument("--arms", default="flic,lgbm," + ",".join(C.TFMS))
    a = ap.parse_args(argv)
    d = load(a.without_crash2)
    if d.empty:
        C.progress("fig8_realdata: no real-data learning-curve table yet; nothing drawn")
        return 0
    arms = a.arms.split(",")
    sets = [s for s in DATASETS if s in set(d["dataset"])]
    S.setup()
    fig, axes = plt.subplots(len(ROWS), len(sets), figsize=(S.TWO_COL, 112 * S.MM), squeeze=False, sharex=True)
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for i, (col, ylab, ref, how) in enumerate(ROWS):
        for j, ds in enumerate(sets):
            ax = axes[i, j]
            dd = d[d["dataset"] == ds]
            for arm in arms:
                g = dd[dd["arm"] == arm].groupby("n")[col]
                if not len(g.size()):
                    continue
                st = S.style_of(arm)
                if how == "mean":
                    m = g.mean()
                    half = 1.96 * g.std(ddof=1) / np.sqrt(g.size())
                    lo, hi = m - half, m + half
                else:
                    m, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
                ax.fill_between(m.index, lo.values, hi.values, color=st["color"], alpha=0.15, linewidth=0)
                ax.plot(m.index, m.values, label=S.label(arm), **st)
            if ref is not None:
                ax.axhline(ref, color="#7F7F7F", linewidth=S.LW, linestyle=(0, (4, 2)), zorder=0)
            if col == "emp_ici":
                ax.set_ylim(bottom=0)
            S.log_n_axis(ax)
            if i == 0:
                ax.text(0.5, 1.06, wrap(DATASET_LABEL.get(ds, ds)), transform=ax.transAxes, ha="center", va="bottom")
            if j == 0:
                ax.set_ylabel(ylab)
            if i == len(ROWS) - 1:
                ax.set_xlabel("Development size n")
            S.panel_label(ax, next(letters), x=-0.30 if j == 0 else -0.18)
    h, l_ = S.legend_handles(arms)
    fig.legend(h, l_, loc="lower center", ncol=len(arms), bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.12, wspace=0.38, hspace=0.25)
    # partial until the real-data run writes analysis/tables/realdata/COMPLETE
    if not (C.TABLES / "realdata" / "COMPLETE").exists():
        S.watermark(fig, S.WATERMARK_PARTIAL)
    S.save(fig, "fig8_realdata_without_crash2" if a.without_crash2 else "fig8_realdata")
    return 0


if __name__ == "__main__":
    sys.exit(main())
