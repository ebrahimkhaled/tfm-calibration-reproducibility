"""Figure 2. Distribution of the true calibration slope b* in the eight main cells of Q1
(pi = 10 %, n in {200, 500}, four shapes).

Box: median and inter-quartile range; whiskers: 10th and 90th percentiles (6.1). The grey band is
the slope tolerance [0.9, 1.1] of 7.1. Every arm is drawn on the same replicates (1-R of the TFM
schedule), so the boxes describe the same data sets. b* <= 0 or no finite b* is counted above the box.
Two-column width.
"""

import matplotlib.colors
import numpy as np
import pandas as pd

import _style as S
from _style import C, plt

ARMS = ["flic", "gam", "lgbm"] + C.TFMS


def main():
    S.setup()
    s = pd.read_parquet(C.TABLES / "scores.parquet",
                        columns=["scenario", "rep", "column", "arm", "ladder", "usable", "o_true_slope", "family", "pi", "n"])
    s = s[(s["ladder"] == "R0") & s["family"].isin(C.SHAPES) & np.isclose(s["pi"], 0.10) & s["n"].isin([200, 500])]
    s = s[s["rep"] <= C.R_TFM_SMALL]
    fig, axes = plt.subplots(2, 4, figsize=(S.TWO_COL, 92 * S.MM), sharey=True)
    letters = iter("abcdefgh")
    for i, n in enumerate([200, 500]):
        for j, shape in enumerate(C.SHAPES):
            ax = axes[i, j]
            ax.axhspan(0.9, 1.1, color="#E6E6E6", zorder=0, linewidth=0)
            ax.axhline(1.0, color="#7F7F7F", linewidth=S.LW, zorder=1)
            d = s[(s["family"] == shape) & (s["n"] == n) & s["usable"]]
            for k, arm in enumerate(ARMS):
                b = d.loc[d["arm"] == arm, "o_true_slope"].to_numpy(float)
                if not len(b):
                    continue
                bad = int(np.sum(~np.isfinite(b) | (b <= 0)))
                b = b[np.isfinite(b) & (b > 0)]
                q10, q25, q50, q75, q90 = np.quantile(b, [0.1, 0.25, 0.5, 0.75, 0.9])
                col = S.COLOR[arm]
                edge = S.EDGE.get(arm, col)
                ax.add_patch(plt.Rectangle((k - 0.32, q25), 0.64, q75 - q25, facecolor=matplotlib.colors.to_rgba(col, 0.35),
                                           edgecolor=edge, linewidth=S.LW, zorder=2))
                ax.plot([k - 0.32, k + 0.32], [q50, q50], color=edge, linewidth=S.LW * 1.5, zorder=3)
                ax.plot([k, k], [q10, q25], color=edge, linewidth=S.LW, zorder=2)
                ax.plot([k, k], [q75, q90], color=edge, linewidth=S.LW, zorder=2)
                if bad:
                    ax.text(k, 3.6, str(bad), ha="center", va="center")
            ax.set_yscale("log")
            ax.set_ylim(0.3, 4.2)
            ax.set_yticks([0.5, 0.9, 1.1, 2])
            ax.set_yticklabels(["0.5", "0.9", "1.1", "2"])
            ax.minorticks_off()
            ax.set_xlim(-0.6, len(ARMS) - 0.4)
            ax.set_xticks(range(len(ARMS)))
            ax.set_xticklabels([S.label(a).replace(" + Platt", "\n+ Platt") for a in ARMS], rotation=90)
            if i == 0:
                ax.text(0.5, 1.04, shape, transform=ax.transAxes, ha="center", va="bottom")
                ax.set_xticklabels([])
            if j == 3:
                ax.text(1.04, 0.5, f"n = {n}", transform=ax.transAxes, rotation=270, ha="left", va="center")
            if j == 0:
                ax.set_ylabel("True calibration slope b*")
            S.panel_label(ax, next(letters), x=-0.3 if j == 0 else -0.1)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.93, bottom=0.22, wspace=0.12, hspace=0.12)
    S.watermark(fig)
    S.save(fig, "fig2_slope_distributions")


if __name__ == "__main__":
    main()
