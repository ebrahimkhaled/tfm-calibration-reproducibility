"""Figure 1. Mean true ICI against development size n, by event rate (rows) and shape (columns).

Primary estimand 4.1 per arm, mean +/- 1.96 MCSE over replicates (6.1); the oracle row is the
loess floor of the estimator (4.1). Two-column width.

    python analysis/figures/fig1_true_ici_vs_n.py [--all-arms]      (--all-arms: Supplementary Fig. S1)
"""

import argparse

import _style as S
from _style import C, plt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-arms", action="store_true")
    a = ap.parse_args(argv)
    S.setup()
    t = S.summary_table("arm_summaries")
    t = t[(t["estimand"] == "o_true_ici") & (t["subset"] == "all") & (t["ladder"] == "R0")].copy()
    info = t["scenario"].map(C.parse_scenario)
    t["family"], t["pi"], t["n"] = [i["family"] for i in info], [i["pi"] for i in info], [i["n"] for i in info]
    arms = (["oracle", "lr", "flic", "ridge", "gam", "lgbm", "lgbm_raw"] + C.TFMS if a.all_arms
            else ["oracle", "flic", "gam", "lgbm"] + C.TFMS)
    fig, axes = plt.subplots(3, 4, figsize=(S.TWO_COL, 118 * S.MM), sharex=True)
    letters = iter("abcdefghijkl")
    for i, pi in enumerate(C.PIS):
        for j, shape in enumerate(C.SHAPES):
            ax = axes[i, j]
            d = t[(t["family"] == shape) & (abs(t["pi"] - pi) < 1e-9)]
            for arm in arms:
                g = d[d["arm"] == arm].sort_values("n")
                if not len(g):
                    continue
                st = S.style_of(arm)
                ax.fill_between(g["n"], g["mean_lo"], g["mean_hi"], color=st["color"], alpha=0.15, linewidth=0)
                ax.plot(g["n"], g["mean"], label=S.label(arm), **st)
            S.log_n_axis(ax)
            ax.set_ylim(bottom=0)
            if i == 0:
                ax.text(0.5, 1.04, shape, transform=ax.transAxes, ha="center", va="bottom")
            if j == 3:
                ax.text(1.04, 0.5, S.pi_label(pi), transform=ax.transAxes, rotation=270, ha="left", va="center")
            if j == 0:
                ax.set_ylabel("Mean true ICI")
            if i == 2:
                ax.set_xlabel("Development size n")
            S.panel_label(ax, next(letters), x=-0.26 if j == 0 else -0.14)
    h, l_ = S.legend_handles(arms)
    fig.legend(h, l_, loc="lower center", ncol=len(arms), bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.07, right=0.955, top=0.94, bottom=0.14, wspace=0.32, hspace=0.32)
    S.watermark(fig)
    S.save(fig, "figS1_true_ici_vs_n_all_arms" if a.all_arms else "fig1_true_ici_vs_n")


if __name__ == "__main__":
    main()
