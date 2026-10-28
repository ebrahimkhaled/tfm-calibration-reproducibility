"""Figure 3b. The share of replicates that reach weak (a-c) or moderate (d-f) calibration, against n.

Companion of the minimum-n table (fig3_min_n_table.py, 4.5). The table reports only the smallest n
at which an arm reaches a class in at least 80 % of replicates, so a model at 0.78 and a model at
0.10 both read "> 3000". This figure shows the shares themselves. Lines are the mean of the four
shapes, the band their range; the dashed line marks the 80 % rule of the table. Each share has
an MCSE of at most 0.022 (R = 500). Two-column width.

    python analysis/figures/fig3b_share_reaching_class.py [--variant main|half|double|ll]
"""

import argparse

import numpy as np

import _style as S
from _style import C, plt

ARMS = ["flic", "gam", "lgbm"] + C.TFMS


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main")
    a = ap.parse_args(argv)
    S.setup()
    t = S.summary_table("min_n")
    t = t[(t["variant"] == a.variant) & t["family"].isin(C.SHAPES)]
    share_cols = [f"share_n{n}" for n in C.NS]
    fig, axes = plt.subplots(2, 3, figsize=(S.TWO_COL, 100 * S.MM), sharex=True, sharey=True)
    letters = iter("abcdef")
    for i, cls in enumerate(["weak", "moderate"]):
        for j, pi in enumerate(C.PIS):
            ax = axes[i, j]
            d = t[(t["cls"] == cls) & np.isclose(t["pi"], pi)]
            for arm in ARMS:
                g = d[d["arm"] == arm]
                if not len(g):
                    continue
                v = g[share_cols].to_numpy(dtype=float)
                st = S.style_of(arm)
                ax.fill_between(C.NS, np.nanmin(v, axis=0), np.nanmax(v, axis=0), color=st["color"],
                                alpha=0.12, linewidth=0)
                ax.plot(C.NS, np.nanmean(v, axis=0), **st)
            ax.axhline(C.MIN_N_SHARE, color="black", linestyle=(0, (3, 2)), linewidth=S.LW)
            S.log_n_axis(ax)
            ax.set_ylim(0, 1)
            if i == 0:
                ax.text(0.5, 1.04, S.pi_label(pi), transform=ax.transAxes, ha="center", va="bottom")
            if j == 0:
                ax.set_ylabel(f"Share reaching {cls}\ncalibration")
            if i == 1:
                ax.set_xlabel("Development size n")
            S.panel_label(ax, next(letters), x=-0.2, y=1.04)
    hs, ls = S.legend_handles(ARMS)
    hs.append(plt.Line2D([], [], color="black", linestyle=(0, (3, 2)), linewidth=S.LW))
    ls.append("80 % rule of the table")
    fig.legend(hs, ls, loc="lower center", ncol=len(ls), bbox_to_anchor=(0.5, 0.0), handlelength=2.0,
               columnspacing=0.9)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.17, hspace=0.28, wspace=0.12)
    S.watermark(fig)
    S.save(fig, "fig3b_share_reaching_class" + ("" if a.variant == "main" else f"_{a.variant}"))


if __name__ == "__main__":
    main()
