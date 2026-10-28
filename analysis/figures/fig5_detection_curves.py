"""Figure 5. Detection curves of the goodness-of-fit tests, with their size (Q4, 6.2).

Each panel: P(reject at 0.05 | true ICI) on the observed Y from the logistic GAM, with the 95 %
cluster-bootstrap band (detection_curves.R). Inset: the pooled size of the test under the exact
null Y* for every arm (oracle: its real Y and Y*), +/- 1.96 MCSE, against the band [0.04, 0.06]
of Q4a, so that no test's detection is shown without its size (7.4). Two-column width.
"""

import numpy as np

import _style as S
from _style import C, plt

TESTS = C.BATTERY + ["DeepGOF-1 external (combined)", "Localize (any part named)"]
INSET_ARMS = ["oracle", "flic", "gam", "lgbm"] + C.TFMS


def main():
    S.setup()
    cur = S.summary_table("detection_curves")
    try:
        size = S.summary_table("q4a_size")
    except FileNotFoundError:
        size = None
    fig, axes = plt.subplots(3, 4, figsize=(S.TWO_COL, 128 * S.MM), sharex=True, sharey=True)
    letters = iter("abcdefghijkl")
    xmax = float(cur["ici"].max()) if len(cur) else 0.1
    for idx, (ax, test) in enumerate(zip(axes.flat, TESTS)):
        g = cur[cur["test"] == test].sort_values("ici")
        if len(g):
            ax.fill_between(g["ici"], g["lo"], g["hi"], color="#BFBFBF", linewidth=0)
            ax.plot(g["ici"], g["fit"], color="black", linewidth=S.LW)
        for x in C.DETECTION_POINTS:
            ax.axvline(x, color="#D9D9D9", linewidth=S.LW, zorder=0)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, xmax)
        ax.text(0.03, 0.97, C.TEST_LABEL.get(test, test), transform=ax.transAxes, ha="left", va="top")
        S.panel_label(ax, next(letters), x=-0.2 if idx % 4 == 0 else -0.07)
        if size is not None and len(size):
            ins = ax.inset_axes([0.55, 0.1, 0.42, 0.36])
            ins.axhspan(0.04, 0.06, color="#E6E6E6", linewidth=0)
            ss = size[size["test"] == test].set_index("column")
            for k, arm in enumerate(INSET_ARMS):
                if arm in ss.index:
                    r = ss.loc[arm]
                    half = 1.96 * r["pooled_mcse"]
                    st = S.style_of(arm)
                    ins.errorbar([k], [r["pooled_size"]], yerr=[[half], [half]], fmt=st["marker"] or "o",
                                 color=st["color"], markeredgecolor=st["markeredgecolor"], markersize=2.6,
                                 elinewidth=S.LW)
            if test not in set(size["test"]):
                ins.text(0.5, 0.62, "\n".join(["size: auxiliary", "study (5.3)"]), transform=ins.transAxes,
                         ha="center", va="center")
            ins.set_xlim(-0.7, len(INSET_ARMS) - 0.3)
            ins.set_ylim(0, 0.12)
            ins.set_xticks([])
            ins.set_yticks([0.05, 0.1])
            ins.set_yticklabels(["0.05", "0.1"])
            ins.tick_params(length=2, pad=1)
            ins.set_facecolor("white")
    for ax in axes[-1, :]:
        ax.set_xlabel("True ICI")
    for ax in axes[:, 0]:
        ax.set_ylabel("P(reject at 0.05)")
    for ax in axes.flat[len(TESTS):]:
        ax.set_visible(False)
    # legend of the inset markers
    hs = [plt.Line2D([], [], linestyle="none", marker=S.style_of(a)["marker"] or "o", color=S.COLOR[a],
                     markeredgecolor=S.EDGE.get(a, S.COLOR[a]), markersize=3.2) for a in INSET_ARMS]
    fig.legend(hs, [S.label(a) for a in INSET_ARMS], loc="lower center", ncol=len(INSET_ARMS),
               title=None, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.12, wspace=0.12, hspace=0.18)
    S.watermark(fig)
    S.save(fig, "fig5_detection_curves")


if __name__ == "__main__":
    main()
