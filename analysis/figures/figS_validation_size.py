"""Supplementary figure. Size and rejection of the tests against the size of the validation set
(pre-registration 11: "the validation-size subsamples").

Every replicate was also tested on two subsamples of its evaluation set: about 50 expected events
(ceil(50/pi) rows, every scenario) and about 200 expected events (ceil(200/pi) rows, only at
pi = 30 %, where the evaluation set is large enough). The x axis is the expected number of events
in the validation set: 50, the full evaluation set (100 at pi = 3 % and 10 %, 300 at pi = 30 %),
and 200.

    row a-c  pooled size: rejection rate on each arm's exact null outcomes Y* ~ Bernoulli(p-hat)
    row d-f  rejection rate on the observed outcomes

Pooled over FLIC and the three TFMs and over every scenario at that event rate (core and stress),
as the total of rejections over the total of tested sets; error bars +/- 1.96 binomial MCSE.
Two-column width.

    python analysis/figures/figS_validation_size.py
"""

import numpy as np
import pandas as pd

import _style as S
from _style import C, plt

ARMS = ["flic"] + C.TFMS
TESTS = ["EDGE (G=10)", "Cox recalibration", "Hosmer-Lemeshow (external)", "Calibration-in-the-large",
         "Spiegelhalter-z", "Osius-Rojek (external)", "Stukel (offset)", "le Cessie (external)"]
TSTYLE = {"EDGE (G=10)": (S.OI["black"], "o", "-"), "Cox recalibration": (S.OI["vermillion"], "s", "-"),
          "Hosmer-Lemeshow (external)": (S.OI["blue"], "^", "-"),
          "Calibration-in-the-large": (S.OI["orange"], "v", "-"), "Spiegelhalter-z": (S.OI["sky"], "D", "-"),
          "Osius-Rojek (external)": (S.OI["green"], "P", "-"), "Stukel (offset)": (S.OI["purple"], "X", "-"),
          "le Cessie (external)": (S.OI["grey"], "h", (0, (4, 2)))}
EVAL_EVENTS = {0.03: 100, 0.10: 100, 0.30: 300}


def main():
    S.setup()
    r = S.summary_table("rejection_rates")
    r = r[(r["ladder"] == "R0") & r["column"].isin(ARMS) & r["test"].isin(TESTS)].copy()
    info = r["scenario"].map(C.parse_scenario)
    r["pi"] = [i["pi"] for i in info]
    xs = {"e50": lambda pi: 50, "eval": lambda pi: EVAL_EVENTS[pi], "e200": lambda pi: 200}
    rows = []
    for (pi, es, yt, t), g in r.groupby(["pi", "eval_set", "ytype", "test"]):
        k, n = g["k"].sum(), g["R"].sum()
        if n <= 0:
            continue
        p = k / n
        rows.append(dict(pi=pi, eval_set=es, ytype=yt, test=t, x=xs[es](round(pi, 2)), rate=p,
                         mcse=np.sqrt(p * (1 - p) / n), n=n))
    d = pd.DataFrame(rows)
    C.progress(d.groupby(["eval_set", "ytype"])["n"].agg(["min", "max"]).to_string())
    fig, axes = plt.subplots(2, 3, figsize=(S.TWO_COL, 100 * S.MM))
    letters = iter("abcdef")
    for i, yt in enumerate(["ystar", "real"]):
        for j, pi in enumerate(C.PIS):
            ax = axes[i, j]
            if yt == "ystar":
                ax.axhspan(*C.SIZE_BAND, color="#E6E6E6", linewidth=0, zorder=0)
            for t in TESTS:
                g = d[(np.isclose(d["pi"], pi)) & (d["ytype"] == yt) & (d["test"] == t)].sort_values("x")
                if not len(g):
                    continue
                col, mk, ls = TSTYLE[t]
                ax.errorbar(g["x"], g["rate"], yerr=1.96 * g["mcse"], color=col, marker=mk, linestyle=ls,
                            markersize=3.2, linewidth=S.LW, elinewidth=S.LW, markeredgecolor=col)
            ax.set_xscale("log")
            ticks = sorted(set([50, EVAL_EVENTS[pi]] + ([200] if pi == 0.30 else [])))
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(x) for x in ticks])
            ax.minorticks_off()
            ax.set_xlim(40, 380)
            if yt == "ystar":
                ax.set_ylim(0.0, 0.10)
            else:
                ax.set_ylim(0, 1.0)
            if i == 0:
                ax.text(0.5, 1.04, S.pi_label(pi), transform=ax.transAxes, ha="center", va="bottom")
            if j == 0:
                ax.set_ylabel("Size (null outcomes)" if yt == "ystar" else "Rejection rate (observed outcomes)")
            if i == 1:
                ax.set_xlabel("Expected events in the validation set")
            S.panel_label(ax, next(letters), x=-0.22 if j == 0 else -0.14)
    hs = [plt.Line2D([], [], color=TSTYLE[t][0], marker=TSTYLE[t][1], linestyle=TSTYLE[t][2], markersize=3.2,
                     linewidth=S.LW) for t in TESTS]
    fig.legend(hs, [C.TEST_LABEL.get(t, t) for t in TESTS], loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.2, wspace=0.28, hspace=0.32)
    S.watermark(fig)
    S.save(fig, "figS_validation_size")


if __name__ == "__main__":
    main()
