"""Figure 3 of the manuscript: the minimum-n table and the shares behind it, in one figure.

(a, b) The pre-registered minimum-n heat-map table (4.5) for weak and moderate calibration, as in
fig3_min_n_table.py. (c-h) The share of replicates reaching weak (c-e) or moderate (f-h)
calibration against n (fig3b_share_reaching_class.py): mean of the four shapes, band = their
range, dashed line = the 80 % rule of the table. The two parts are drawn by the same code as the
separate figures, so the numbers agree. Two-column width.

    python analysis/figures/fig3c_min_n_and_shares.py [--variant main|half|double|ll]
"""

import argparse

import numpy as np

import _style as S
from _style import C, plt
from fig3_min_n_table import ARMS as TABLE_ARMS, LEVELS, SEQ
from fig3b_share_reaching_class import ARMS as CURVE_ARMS


def draw_table(ax, d, cls_label):
    cols = [(pi, sh) for pi in C.PIS for sh in C.SHAPES + ["pooled"]]
    rows = TABLE_ARMS + ["riley"]
    for i, arm in enumerate(rows):
        for j, (pi, sh) in enumerate(cols):
            fam = sh if sh != "pooled" else "pooled (largest of the four shapes)"
            g = d[(d["arm"] == (arm if arm != "riley" else "flic")) & (d["family"] == fam) & np.isclose(d["pi"], pi)]
            if not len(g):
                continue
            if arm == "riley":
                txt, face, fc = str(int(g["riley_n_min"].iloc[0])), "white", "black"
            else:
                txt = str(g["min_n"].iloc[0])
                k = LEVELS.index(txt) if txt in LEVELS else 0
                face, fc = SEQ[k], ("white" if k >= 3 else "black")
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=face, edgecolor="white", linewidth=S.LW))
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", color=fc, fontsize=S.FONT)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(rows), 0)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([S.label(r) if r != "riley" else "Riley minimum n" for r in rows])
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels([sh if sh != "pooled" else "all" for _, sh in cols])
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    for k, pi in enumerate(C.PIS):
        ax.text(5 * k + 2.5, -0.3, S.pi_label(pi), ha="center", va="bottom")
        if k:
            ax.axvline(5 * k, color="black", linewidth=S.LW)
    ax.axhline(len(TABLE_ARMS), color="black", linewidth=S.LW)
    ax.text(-0.01, 1.11, f"Minimum n for {cls_label} calibration", transform=ax.transAxes, ha="left",
            va="bottom")


def draw_curves(ax, d):
    share_cols = [f"share_n{n}" for n in C.NS]
    for arm in CURVE_ARMS:
        g = d[d["arm"] == arm]
        if not len(g):
            continue
        v = g[share_cols].to_numpy(dtype=float)
        st = S.style_of(arm)
        ax.fill_between(C.NS, np.nanmin(v, axis=0), np.nanmax(v, axis=0), color=st["color"], alpha=0.12,
                        linewidth=0)
        ax.plot(C.NS, np.nanmean(v, axis=0), **st)
    ax.axhline(C.MIN_N_SHARE, color="black", linestyle=(0, (3, 2)), linewidth=S.LW)
    S.log_n_axis(ax)
    ax.set_ylim(0, 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main")
    a = ap.parse_args(argv)
    S.setup()
    t = S.summary_table("min_n")
    t = t[t["variant"] == a.variant]
    fig = plt.figure(figsize=(S.TWO_COL, 205 * S.MM))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 0.78, 0.78], left=0.15, right=0.99, top=0.955,
                          bottom=0.085, hspace=0.42, wspace=0.12)
    for r, (cls, letter) in enumerate([("weak", "a"), ("moderate", "b")]):
        ax = fig.add_subplot(gs[r, :])
        draw_table(ax, t[t["cls"] == cls], cls)
        S.panel_label(ax, letter, x=-0.16, y=1.08)
    letters = iter("cdefgh")
    curves = t[t["family"].isin(C.SHAPES)]
    first = None
    for i, cls in enumerate(["weak", "moderate"]):
        for j, pi in enumerate(C.PIS):
            ax = fig.add_subplot(gs[2 + i, j], sharex=first, sharey=first)
            first = first or ax
            draw_curves(ax, curves[(curves["cls"] == cls) & np.isclose(curves["pi"], pi)])
            if i == 0:
                ax.text(0.5, 1.05, S.pi_label(pi), transform=ax.transAxes, ha="center", va="bottom")
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Development size n")
            if j == 0:
                ax.set_ylabel(f"Share reaching\n{cls} calibration")
            else:
                ax.tick_params(labelleft=False)
            S.panel_label(ax, next(letters), x=-0.13 if j else -0.3, y=1.03)
    hs, ls = S.legend_handles(CURVE_ARMS)
    hs.append(plt.Line2D([], [], color="black", linestyle=(0, (3, 2)), linewidth=S.LW))
    ls.append("80 % rule of the table")
    fig.legend(hs, ls, loc="lower center", ncol=len(ls), bbox_to_anchor=(0.57, 0.0), handlelength=2.0,
               columnspacing=0.9)
    S.watermark(fig)
    S.save(fig, "fig3c_min_n_and_shares" + ("" if a.variant == "main" else f"_{a.variant}"))


if __name__ == "__main__":
    main()
