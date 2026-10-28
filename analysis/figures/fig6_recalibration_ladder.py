"""Figure 6. The recalibration ladder (Q5a): mean true ICI after each level, R0 (none),
R1 (intercept), R2 (logistic) and R3 (isotonic), with a local sample of m = 500 (solid) and
m = 100 (dashed), for FLIC and the three TFMs; rows are event rates, columns shapes, at one n.
Points +/- 1.96 MCSE (6.1). Two-column width.

    python analysis/figures/fig6_recalibration_ladder.py [--n 500]     (other n: Supplementary)
"""

import argparse

import numpy as np

import _style as S
from _style import C, plt

LEVELS = ["R0", "R1", "R2", "R3"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args(argv)
    S.setup()
    t = S.summary_table("q5a_ladder_means")
    info = t["scenario"].map(C.parse_scenario)
    t["family"], t["pi"], t["n"] = [i["family"] for i in info], [i["pi"] for i in info], [i["n"] for i in info]
    t = t[t["n"] == a.n]
    fig, axes = plt.subplots(3, 4, figsize=(S.TWO_COL, 118 * S.MM), sharex=True)
    letters = iter("abcdefghijkl")
    for i, pi in enumerate(C.PIS):
        for j, shape in enumerate(C.SHAPES):
            ax = axes[i, j]
            d = t[(t["family"] == shape) & np.isclose(t["pi"], pi)]
            for k, arm in enumerate(C.LADDER_ARMS):
                st = S.style_of(arm)
                for m, ls in ((500, "-"), (100, (0, (3, 1.5)))):
                    cols = [arm if lev == "R0" else f"{arm}__{lev}_m{m}" for lev in LEVELS]
                    g = d.set_index("column").reindex(cols)
                    x = np.arange(4) + (k - 1.5) * 0.06
                    ax.errorbar(x, g["mean"], yerr=1.96 * g["mean_mcse"], color=st["color"], marker=st["marker"],
                                markersize=3.0, markeredgecolor=st["markeredgecolor"], linestyle=ls,
                                linewidth=S.LW, elinewidth=S.LW, label=S.label(arm) if m == 500 else None)
            ax.set_xticks(range(4))
            ax.set_xticklabels(LEVELS)
            ax.set_xlim(-0.3, 3.3)
            ax.set_ylim(bottom=0)
            if i == 0:
                ax.text(0.5, 1.04, shape, transform=ax.transAxes, ha="center", va="bottom")
            if j == 3:
                ax.text(1.04, 0.5, S.pi_label(pi), transform=ax.transAxes, rotation=270, ha="left", va="center")
            if j == 0:
                ax.set_ylabel("Mean true ICI")
            if i == 2:
                ax.set_xlabel("Recalibration level")
            S.panel_label(ax, next(letters), x=-0.26 if j == 0 else -0.14)
    h, l_ = S.legend_handles(C.LADDER_ARMS)
    h += [plt.Line2D([], [], color="black", linewidth=S.LW), plt.Line2D([], [], color="black", linewidth=S.LW, linestyle=(0, (3, 1.5)))]
    l_ += ["m = 500", "m = 100"]
    fig.legend(h, l_, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.07, right=0.955, top=0.94, bottom=0.14, wspace=0.32, hspace=0.32)
    S.watermark(fig)
    S.save(fig, "fig6_recalibration_ladder" + ("" if a.n == 500 else f"_n{a.n}"))


if __name__ == "__main__":
    main()
