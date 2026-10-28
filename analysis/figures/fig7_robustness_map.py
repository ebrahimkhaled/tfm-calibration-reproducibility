"""Figure 7. The robustness map of the random-design tier (6.4, Q6; primary output 5).

For each TFM, the meta-model's prediction of D_ICI = true ICI(TFM) - true ICI(FLIC) on
(log n, log pi), at the median of the other factors: dark grey where the 95 % simultaneous band
lies below -0.01 ("TFM better"), hatched light grey where it lies above +0.01 ("TFM worse"), white
otherwise; contours of the predicted D_ICI. Cross-hatched: Mitra-v2 not run (does not fit in
6 GB GPU memory, 3.9). Two-column width.

    python analysis/figures/fig7_robustness_map.py [--response D_logb]   (D_logb: Supplementary)
Until the random tier has run, it draws the meta-model fitted to SYNTHETIC test data, watermarked.
"""

import argparse

import numpy as np
import pandas as pd

import _style as S
from _style import C, plt

CODES = {"TFM better": 0, "within": 1, "TFM worse": 2}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--response", default="D_ICI")
    a = ap.parse_args(argv)
    S.setup()
    real = C.TABLES / "meta" / f"map_{a.response}_lognlogpi.csv"
    synth = C.TABLES / "meta_synthetic" / f"map_{a.response}_lognlogpi.csv"
    f, mark = (real, None) if real.exists() else (synth, S.WATERMARK_SYNTH)
    m = pd.read_csv(f)
    margin = 0.01 if a.response == "D_ICI" else float(np.log(1.1))
    fig, axes = plt.subplots(1, 3, figsize=(S.TWO_COL, 66 * S.MM), sharey=True)
    for ax, tfm, letter in zip(axes, C.TFMS, "abc"):
        d = m[m["model"] == tfm]
        if not len(d):
            ax.set_visible(False)
            continue
        xs, ys = np.sort(d["log_n"].unique()), np.sort(d["log_pi"].unique())
        piv = lambda col: d.pivot_table(index="log_pi", columns="log_n", values=col).reindex(index=ys, columns=xs).to_numpy()  # noqa: E731
        code = d["class"].map(CODES)
        d = d.assign(code=code)
        Z = piv("code")
        X, Y = np.meshgrid(np.exp(xs), np.exp(ys))
        ax.contourf(X, Y, Z, levels=[-0.5, 0.5, 1.5, 2.5], colors=[S.REGION[k] for k in CODES],
                    hatches=[S.REGION_HATCH[k] for k in CODES])
        cs = ax.contour(X, Y, piv("fit"), levels=np.round(np.arange(-4, 5) * margin, 6), colors="black",
                        linewidths=S.LW)
        ax.clabel(cs, fmt=lambda v: f"{v:+.2f}".replace("+0.00", "0"), fontsize=S.FONT, inline_spacing=2)
        if "mitra_not_run" in d and d["mitra_not_run"].astype(bool).any():
            nr = piv("mitra_not_run").astype(float)
            ax.contourf(X, Y, nr, levels=[0.5, 1.5], colors="none", hatches=["xxxx"])
            ax.contour(X, Y, nr, levels=[0.5], colors="black", linewidths=S.LW)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([200, 500, 1000, 3000])
        ax.set_xticklabels(["200", "500", "1000", "3000"])
        ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.4])
        ax.set_yticklabels(["2 %", "5 %", "10 %", "20 %", "40 %"])
        ax.minorticks_off()
        ax.set_xlabel("Development size n")
        if letter == "a":
            ax.set_ylabel("Event rate π")
        ax.text(0.5, 1.03, S.label(tfm), transform=ax.transAxes, ha="center", va="bottom")
        S.panel_label(ax, letter, x=-0.2 if letter == "a" else -0.08)
    hs = [plt.Rectangle((0, 0), 1, 1, facecolor=S.REGION[k], edgecolor="black", linewidth=S.LW, hatch=S.REGION_HATCH[k])
          for k in CODES]
    labs = [f"TFM better (band below -{margin:.2g})", f"within {margin:.2g}", f"TFM worse (band above +{margin:.2g})"]
    hs.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", linewidth=S.LW, hatch="xxxx"))
    labs.append("Mitra-v2 not run")
    fig.legend(hs, labs, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.27, wspace=0.12)
    S.watermark(fig, mark)
    S.save(fig, "fig7_robustness_map" + ("" if a.response == "D_ICI" else f"_{a.response}"))


if __name__ == "__main__":
    main()
