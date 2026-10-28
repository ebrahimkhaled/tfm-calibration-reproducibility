"""Supplementary figure. Detection maps of the random-design tier (pre-registration 6.4).

For FLIC and the three TFMs (columns) and two tests (rows: EDGE with 10 groups, Hosmer-Lemeshow),
the binomial meta-model's predicted probability that the test rejects on the observed outcomes,
over development size n and event rate pi, at the median of the other design factors.
Input: tables/meta/detection_map_lognlogpi.csv (meta_model.R). Two-column width.

    python analysis/figures/figS_detection_map.py
"""

import numpy as np
import pandas as pd

import _style as S
from _style import C, plt

ARMS = ["flic"] + C.TFMS
TESTS = ["EDGE (G=10)", "Hosmer-Lemeshow (external)"]


def main():
    S.setup()
    d = pd.read_csv(C.TABLES / "meta" / "detection_map_lognlogpi.csv")
    fig, axes = plt.subplots(len(TESTS), len(ARMS), figsize=(S.TWO_COL, 92 * S.MM), sharex=True, sharey=True)
    letters = iter("abcdefgh")
    levels = np.linspace(0, 1, 11)
    im = None
    for i, t in enumerate(TESTS):
        for j, arm in enumerate(ARMS):
            ax = axes[i, j]
            g = d[(d["arm"] == arm) & (d["test"] == t)]
            if len(g):
                piv = g.pivot_table(index="log_pi", columns="log_n", values="fit")
                X, Y = np.exp(piv.columns.values), np.exp(piv.index.values)
                im = ax.contourf(X, Y, piv.values, levels=levels, cmap="Greys", vmin=0, vmax=1.15)
                cs = ax.contour(X, Y, piv.values, levels=[0.2, 0.4, 0.6, 0.8], colors="black", linewidths=S.LW)
                ax.clabel(cs, fmt="%.1f", fontsize=S.FONT, inline_spacing=2)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([200, 500, 1000, 3000])
            ax.set_xticklabels(["200", "500", "1000", "3000"])
            ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.4])
            ax.set_yticklabels(["0.02", "0.05", "0.1", "0.2", "0.4"])
            ax.minorticks_off()
            if i == 0:
                ax.text(0.5, 1.04, S.label(arm), transform=ax.transAxes, ha="center", va="bottom")
            if j == len(ARMS) - 1:
                ax.text(1.04, 0.5, C.TEST_LABEL.get(t, t), transform=ax.transAxes, rotation=270, ha="left", va="center")
            if j == 0:
                ax.set_ylabel("Event rate π")
            if i == len(TESTS) - 1:
                ax.set_xlabel("Development size n")
            S.panel_label(ax, next(letters), x=-0.3 if j == 0 else -0.16, y=1.03)
    if im is not None:
        cax = fig.add_axes([0.3, 0.115, 0.4, 0.025])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cb.set_label("Predicted probability of rejection (observed outcomes)")
        cb.outline.set_linewidth(S.LW)
    fig.subplots_adjust(left=0.08, right=0.94, top=0.9, bottom=0.28, wspace=0.2, hspace=0.2)
    S.watermark(fig)
    S.save(fig, "figS_detection_map")


if __name__ == "__main__":
    main()
