"""Supplementary figure. Prediction instability on the probe set (pre-registration 4.4; exploratory).

For each probe patient, the standard deviation of the predicted risk across development sets
(replicates 1 to R of the TFM schedule, the same data sets for every arm); the figure shows the
mean of this standard deviation over the 500 probe patients, against n, by event rate (rows) and
shape (columns). Input: tables/summary/instability.csv (analysis/instability_probe.py).
Two-column width.

    python analysis/figures/figS_instability.py
"""

import pandas as pd

import _style as S
from _style import C, plt

ARMS = ["lr", "flic", "ridge", "gam", "lgbm"] + C.TFMS


def main():
    S.setup()
    t = pd.read_csv(C.TABLES / "summary" / "instability.csv")
    t = t[t["family"].isin(C.SHAPES)]
    fig, axes = plt.subplots(3, 4, figsize=(S.TWO_COL, 118 * S.MM), sharex=True)
    letters = iter("abcdefghijkl")
    for i, pi in enumerate(C.PIS):
        for j, shape in enumerate(C.SHAPES):
            ax = axes[i, j]
            d = t[(t["family"] == shape) & (abs(t["pi"] - pi) < 1e-9)]
            for arm in ARMS:
                g = d[d["arm"] == arm].sort_values("n")
                if len(g):
                    ax.plot(g["n"], g["mean_sd"], **S.style_of(arm))
            S.log_n_axis(ax)
            ax.set_yscale("log")
            lo, hi = ax.get_ylim()
            ticks = [t for t in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2) if lo <= t <= hi]
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t:g}" for t in ticks])
            ax.yaxis.set_minor_formatter(plt.NullFormatter())
            if i == 0:
                ax.text(0.5, 1.04, shape, transform=ax.transAxes, ha="center", va="bottom")
            if j == 3:
                ax.text(1.04, 0.5, S.pi_label(pi), transform=ax.transAxes, rotation=270, ha="left", va="center")
            if j == 0:
                ax.set_ylabel("Mean SD of predicted risk")
            if i == 2:
                ax.set_xlabel("Development size n")
            S.panel_label(ax, next(letters), x=-0.3 if j == 0 else -0.16)
    h, l_ = S.legend_handles(ARMS)
    fig.legend(h, l_, loc="lower center", ncol=len(ARMS), bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.08, right=0.955, top=0.94, bottom=0.14, wspace=0.34, hspace=0.32)
    S.watermark(fig)
    S.save(fig, "figS_instability")


if __name__ == "__main__":
    main()
