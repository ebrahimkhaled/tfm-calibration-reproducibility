"""Supplementary figure. Partial effects of the random-tier meta-model (6.4), per factor (columns)
and TFM (colour), with 95 % simultaneous bands from 10,000 posterior draws.

    python analysis/figures/figS_meta_partial_effects.py [--response D_ICI|D_logb]
Until the random tier has run, it draws the meta-model fitted to SYNTHETIC test data, watermarked.
"""

import argparse

import numpy as np
import pandas as pd

import _style as S
from _style import C, plt

TERMS = [("log_n", "log n"), ("log_pi", "log π"), ("p", "Predictors p"), ("auc_target", "Oracle AUC"),
         ("w_nl", "Non-linearity severity"), ("w_int", "Interaction severity"), ("w_link", "Link asymmetry severity"),
         ("rho", "Correlation ρ"), ("frac_disc", "Discrete fraction")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--response", default="D_ICI")
    a = ap.parse_args(argv)
    S.setup()
    real = C.TABLES / "meta" / f"partial_effects_{a.response}.csv"
    synth = C.TABLES / "meta_synthetic" / f"partial_effects_{a.response}.csv"
    f, mark = (real, None) if real.exists() else (synth, S.WATERMARK_SYNTH)
    pe = pd.read_csv(f)
    fig, axes = plt.subplots(3, 3, figsize=(S.TWO_COL, 140 * S.MM))
    for ax, (term, lab), letter in zip(axes.flat, TERMS, "abcdefghi"):
        for tfm in C.TFMS:
            g = pe[(pe["model"] == tfm) & (pe["term"] == term)].sort_values("x")
            if not len(g):
                continue
            st = S.style_of(tfm)
            ax.fill_between(g["x"], g["lo"], g["hi"], color=st["color"], alpha=0.18, linewidth=0)
            ax.plot(g["x"], g["fit"], color=st["color"], linewidth=S.LW, label=S.label(tfm))
        ax.axhline(0, color="#7F7F7F", linewidth=S.LW)
        ax.set_xlabel(lab)
        if letter in "adg":
            ax.set_ylabel({"D_ICI": "Effect on ICI difference",
                           "D_logb": "Effect on |log b*| difference"}.get(a.response, a.response.replace('_', ' ')))
        S.panel_label(ax, letter, x=-0.22)
    h, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=3)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.1, wspace=0.3, hspace=0.45)
    S.watermark(fig, mark)
    S.save(fig, f"figS_meta_partial_effects_{a.response}")


if __name__ == "__main__":
    main()
