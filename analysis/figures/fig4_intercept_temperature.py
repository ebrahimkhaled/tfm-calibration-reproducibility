"""Figure 4. Calibration-in-the-large at pi = 3 % (Q3) and the effect of the softmax temperature.

a, Hodges-Lehmann estimate of the paired difference |a*|(TFM) - |a*|(FLIC) with its 95 %
signed-rank interval in the 16 cells at pi = 3 %, the cells at n >= 1000 first (7.3); dashed lines
at the margin +/- 0.05. The shipped temperature of each TFM is in its legend entry.
b-d, the paired temperature effect (temperature arm minus its main arm, 8 scenarios at pi = 10 %):
b, a* (HL); c, log b* (HL); d, true ICI (mean, +/- 1.96 MCSE). Two-column width.
"""

import numpy as np

import _style as S
from _style import C, plt


def cell_order():
    big = [(sh, n) for sh in C.SHAPES for n in (1000, 3000)]
    small = [(sh, n) for sh in C.SHAPES for n in (200, 500)]
    return big + small


def main():
    S.setup()
    p = S.summary_table("paired_vs_flic")
    fig = plt.figure(figsize=(S.TWO_COL, 112 * S.MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.55, wspace=0.32,
                          left=0.1, right=0.99, top=0.95, bottom=0.17)
    ax = fig.add_subplot(gs[0, :])
    cells = cell_order()
    q = p[(p["kind"] == "vs_flic") & (p["metric"] == "abs_a") & (p["summary"] == "HL") & np.isclose(p["pi"], 0.03) & p["arm"].isin(C.TFMS)]
    for k, arm in enumerate(C.TFMS):
        g = q[q["arm"] == arm].set_index(["family", "n"])
        xs, ys, lo, hi = [], [], [], []
        for i, c in enumerate(cells):
            if c in g.index:
                r = g.loc[c]
                xs.append(i + (k - 1) * 0.22); ys.append(r["estimate"]); lo.append(r["lo"]); hi.append(r["hi"])
        if not xs:
            continue
        st = S.style_of(arm)
        temp = C.SHIPPED_TEMPERATURE.get(arm)
        lab = f"{S.label(arm)} (T = {temp:.1f})" if temp else f"{S.label(arm)} (no temperature)"
        ax.errorbar(xs, ys, yerr=[np.subtract(ys, lo), np.subtract(hi, ys)], fmt=st["marker"], color=st["color"],
                    markeredgecolor=st["markeredgecolor"], markersize=3.2, elinewidth=S.LW, label=lab)
    for m in (-C.MARGIN_ABS_A, C.MARGIN_ABS_A):
        ax.axhline(m, color="#7F7F7F", linestyle=(0, (4, 2)), linewidth=S.LW)
    ax.axhline(0, color="black", linewidth=S.LW)
    ax.axvline(7.5, color="#BFBFBF", linewidth=S.LW)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"{sh}\n{n}" for sh, n in cells])
    ax.set_xlim(-0.6, len(cells) - 0.4)
    ax.set_ylabel("Paired difference in |a*|\n(TFM minus FLIC)")
    hs, _ = S.legend_handles(C.TFMS)
    labs = [f"{S.label(a)} (T = {C.SHIPPED_TEMPERATURE[a]:.1f})" if C.SHIPPED_TEMPERATURE.get(a)
            else f"{S.label(a)} (no temperature)" for a in C.TFMS]
    for hh in hs:
        hh.set_linestyle("none")
    ax.legend(hs, labs, loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.13))
    S.panel_label(ax, "a", x=-0.09, y=1.02)

    t = p[p["kind"] == "temperature"]
    h, l_ = [], []
    scen = [f"{sh}_pi10_n{n}" for sh in C.SHAPES for n in (200, 3000)]
    for j, (metric, ylab, letter) in enumerate([("signed_a", "Effect on a*", "b"), ("signed_log_b", "Effect on log b*", "c"),
                                                 ("o_true_ici", "Effect on true ICI", "d")]):
        ax = fig.add_subplot(gs[1, j])
        drawn = 0
        for k, t1 in enumerate(C.TEMPERATURE_PAIRS):
            g = t[(t["column"] == t1) & (t["metric"] == metric)].set_index("scenario")
            xs, ys, lo, hi = [], [], [], []
            for i, sname in enumerate(scen):
                if sname in g.index:
                    r = g.loc[sname]
                    xs.append(i + (k - 0.5) * 0.3); ys.append(r["estimate"]); lo.append(r["lo"]); hi.append(r["hi"])
            if not xs:
                continue
            drawn += 1
            st = S.style_of(t1)
            ax.errorbar(xs, ys, yerr=[np.subtract(ys, lo), np.subtract(hi, ys)], fmt=st["marker"], color=st["color"],
                        markersize=3.2, elinewidth=S.LW, label=f"{S.label(t1)} minus {S.label(C.TEMPERATURE_PAIRS[t1])}")
        ax.axhline(0, color="black", linewidth=S.LW)
        ax.set_xticks(range(len(scen)))
        ax.set_xticklabels([s.split("_")[0] + "\n" + s.split("_n")[1] for s in scen])
        ax.set_xlim(-0.6, len(scen) - 0.4)
        ax.set_ylabel(ylab)
        S.panel_label(ax, letter, x=-0.27, y=1.02)
        if j == 0:
            h, l_ = ax.get_legend_handles_labels()
        if not drawn:
            ax.text(0.5, 0.5, "temperature arms not run yet", transform=ax.transAxes, ha="center", va="center")
    fig.legend(h, l_, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    S.watermark(fig)
    S.save(fig, "fig4_intercept_temperature")


if __name__ == "__main__":
    main()
