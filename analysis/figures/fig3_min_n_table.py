"""Figure 3. The minimum-n-for-calibration table as a heat-map (4.5; primary output 4 of Section 11).

Each cell is the smallest development n at which the arm meets the class in at least 80 % of
replicates ("> 3000" when no n qualifies), for weak (a) and moderate (b) calibration, per shape
and event rate, with the pooled column of each event rate (the largest of the four shapes) and
Riley's minimum n beside it. A single-hue sequential scale codes the n; the text is the value.
Two-column width.

    python analysis/figures/fig3_min_n_table.py [--variant main|half|double|ll] [--lower-limit]
"""

import argparse

import numpy as np

import _style as S
from _style import C, plt

ARMS = ["lr", "flic", "ridge", "gam", "lgbm", "lgbm_raw"] + C.TFMS
LEVELS = ["200", "500", "1000", "3000", "> 3000"]
SEQ = ["#EFF3FF", "#BDD7E7", "#6BAED6", "#2171B5", "#08306B"]      # ColorBrewer Blues (colour-blind safe)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main")
    ap.add_argument("--lower-limit", action="store_true")
    a = ap.parse_args(argv)
    S.setup()
    t = S.summary_table("min_n")
    t = t[t["variant"] == a.variant]
    col = "min_n_lower_limit" if a.lower_limit else "min_n"
    cols = [(pi, sh) for pi in C.PIS for sh in C.SHAPES + ["pooled"]]
    fig, axes = plt.subplots(2, 1, figsize=(S.TWO_COL, 112 * S.MM))
    for ax, cls, letter in zip(axes, ["weak", "moderate"], "ab"):
        d = t[t["cls"] == cls]
        rows = ARMS + ["riley"]
        for i, arm in enumerate(rows):
            for j, (pi, sh) in enumerate(cols):
                fam = sh if sh != "pooled" else "pooled (largest of the four shapes)"
                g = d[(d["arm"] == (arm if arm != "riley" else "flic")) & (d["family"] == fam) & np.isclose(d["pi"], pi)]
                if not len(g):
                    continue
                if arm == "riley":
                    txt, face, fc = str(int(g["riley_n_min"].iloc[0])), "white", "black"
                else:
                    txt = str(g[col].iloc[0])
                    k = LEVELS.index(txt) if txt in LEVELS else 0
                    face, fc = SEQ[k], ("white" if k >= 3 else "black")
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=face, edgecolor="white", linewidth=S.LW))
                ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", color=fc)
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
            ax.text(5 * k + 2.5, -0.35, S.pi_label(pi), ha="center", va="bottom")
            if k:
                ax.axvline(5 * k, color="black", linewidth=S.LW)
        ax.axhline(len(ARMS), color="black", linewidth=S.LW)
        S.panel_label(ax, letter, x=-0.16, y=1.08)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.93, bottom=0.04, hspace=0.32)
    S.watermark(fig)
    suffix = "" if (a.variant == "main" and not a.lower_limit) else f"_{a.variant}{'_lower' if a.lower_limit else ''}"
    S.save(fig, "fig3_min_n_table" + suffix)


if __name__ == "__main__":
    main()
