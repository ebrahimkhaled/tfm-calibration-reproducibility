"""House style of every figure: one look, one font size, one colour per model.

- Arial (Helvetica-compatible) at 7 pt for all text, bold lower-case panel labels at 8 pt;
- white background, no 3-D, no in-image titles, every line at least 1 pt;
- the Okabe-Ito colour-blind-safe palette, with a fixed colour, marker and line style per model;
- vector PDF at the final width: one column 89 mm, two columns 183 mm, TrueType fonts embedded.

While the run is not complete, every figure carries the watermark
"PARTIAL DATA - NOT FOR INTERPRETATION" (synthetic inputs carry their own watermark).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import _common as C  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

MM = 1 / 25.4
ONE_COL, TWO_COL = 89 * MM, 183 * MM
FONT = 7
PANEL_FONT = 8
LW = 1.0
OUT = C.FIG_OUT

# Okabe-Ito (2008): black, orange, sky blue, bluish green, yellow, blue, vermillion, reddish purple,
# and its grey. Each model keeps its colour in every figure; side arms share their family's colour
# with another line style.
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", vermillion="#D55E00", purple="#CC79A7", grey="#999999")
COLOR = {"oracle": OI["grey"], "flic": OI["black"], "lr": OI["sky"], "ridge": OI["purple"],
         "gam": OI["orange"], "lgbm": OI["yellow"], "lgbm_raw": OI["yellow"],
         "tabpfn": OI["vermillion"], "tabicl": OI["blue"], "mitra": OI["green"],
         "tabpfn_t1": OI["vermillion"], "tabicl_t1": OI["blue"], "tabpfn_legacy": OI["vermillion"]}
MARKER = {"oracle": "", "flic": "s", "lr": "v", "ridge": "D", "gam": "^", "lgbm": "P", "lgbm_raw": "X",
          "tabpfn": "o", "tabicl": "h", "mitra": "d", "tabpfn_t1": "o", "tabicl_t1": "h", "tabpfn_legacy": "*"}
LINESTYLE = {"oracle": (0, (4, 2)), "lgbm_raw": (0, (1, 1.5)), "tabpfn_t1": (0, (4, 1.5)),
             "tabicl_t1": (0, (4, 1.5)), "tabpfn_legacy": (0, (1, 1.5))}
# a yellow line needs a dark edge on its markers to be seen on white
EDGE = {"lgbm": "#555555", "lgbm_raw": "#555555"}
# region classes of the robustness map are not models, so they use greys and hatching only
REGION = {"TFM better": "#4D4D4D", "within": "#F2F2F2", "TFM worse": "#BDBDBD"}
REGION_HATCH = {"TFM better": "", "within": "", "TFM worse": "////"}

WATERMARK_PARTIAL = "PARTIAL DATA - NOT FOR INTERPRETATION"
WATERMARK_SYNTH = "SYNTHETIC DATA - PIPELINE TEST ONLY"


def setup() -> None:
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT, "xtick.labelsize": FONT,
        "ytick.labelsize": FONT, "legend.fontsize": FONT, "figure.titlesize": FONT,
        "axes.linewidth": LW, "lines.linewidth": LW, "patch.linewidth": LW, "hatch.linewidth": LW,
        "xtick.major.width": LW, "ytick.major.width": LW, "xtick.minor.width": LW, "ytick.minor.width": LW,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.direction": "out", "ytick.direction": "out", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": False, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "boxplot.boxprops.linewidth": LW,
        "boxplot.whiskerprops.linewidth": LW, "boxplot.capprops.linewidth": LW,
        "boxplot.medianprops.linewidth": LW, "errorbar.capsize": 0, "axes.unicode_minus": True,
        "mathtext.default": "regular", "axes.labelpad": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "legend.handlelength": 2.2, "legend.columnspacing": 1.2, "legend.handletextpad": 0.5,
    })


def style_of(arm: str) -> dict:
    return dict(color=COLOR.get(arm, "#000000"), marker=MARKER.get(arm, "o"),
                linestyle=LINESTYLE.get(arm, "-"), markeredgecolor=EDGE.get(arm, COLOR.get(arm, "#000000")),
                markersize=3.2, linewidth=LW, markeredgewidth=LW)


def label(arm: str) -> str:
    return C.ARM_LABEL.get(arm, arm)


def panel_label(ax, letter: str, x: float = -0.16, y: float = 1.06) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=PANEL_FONT, fontweight="bold", va="bottom", ha="left")


def data_state() -> str:
    m = C.TABLES / "manifest.json"
    if m.exists() and json.loads(m.read_text()).get("complete"):
        return "complete"
    return "partial"


def watermark(fig, text: str | None = None) -> None:
    """Diagonal watermark over the whole figure; on by default until the run is complete."""
    if text is None:
        if data_state() == "complete":
            return
        text = WATERMARK_PARTIAL
    fig.text(0.5, 0.5, text, rotation=18, ha="center", va="center", fontsize=14, fontweight="bold",
             color="#D55E00", alpha=0.32, zorder=1000)


def save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = C.assert_writable(OUT / f"{name}.pdf")
    fig.savefig(path, format="pdf", metadata={"Creator": None, "Producer": None})
    import os
    if os.environ.get("FIG_PREVIEW"):        # raster copies for checking the layout by eye
        (OUT / "preview").mkdir(exist_ok=True)
        fig.savefig(OUT / "preview" / f"{name}.png", dpi=170)
    plt.close(fig)
    C.progress(f"figure written: {path.relative_to(C.REBUILD)}")
    return path


def log_n_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xticks(C.NS)
    ax.set_xticklabels([str(n) for n in C.NS])
    ax.minorticks_off()
    ax.set_xlim(160, 3800)


def summary_table(name: str):
    import pandas as pd
    f = C.TABLES / "summary" / f"{name}.parquet"
    return pd.read_parquet(f) if f.exists() else pd.read_csv(C.TABLES / "summary" / f"{name}.csv")


def pi_label(pi: float) -> str:
    return f"π = {int(round(pi * 100))} %"


def legend_handles(arms: list[str]):
    """Legend entries for a fixed list of arms, whether or not a given panel holds data for them."""
    hs = []
    for a in arms:
        st = style_of(a)
        hs.append(plt.Line2D([], [], color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                             markeredgecolor=st["markeredgecolor"], markersize=st["markersize"], linewidth=LW))
    return hs, [label(a) for a in arms]
