"""Check meta_model.R on the synthetic random tier, whose true D_ICI and D_logb are known.

    Rscript analysis/make_synthetic_random.R && Rscript analysis/meta_model.R --synthetic
    python analysis/tests/check_meta_synthetic.py

Passes when, for every TFM and both responses: the predictions at the designs' own settings track
the truth (correlation > 0.9 with >= 800 designs, > 0.8 on a subset), the 95 % simultaneous bands cover the truth at >= 90 % of designs
(they are simultaneous, so nearly all), the critical value exceeds the pointwise 1.96, every
partial effect has its grid, the region shares sum to 1, Mitra-v2 is absent where it does not
run, and the transfer table has the 48 core scenarios for each TFM.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> int:
    syn = pd.read_parquet(C.TABLES / "synthetic" / "random_tier.parquet")
    md = Path(sys.argv[1]) if len(sys.argv) > 1 else C.TABLES / "meta_synthetic"
    bad = []
    for resp in ("D_ICI", "D_logb"):
        reg = pd.read_csv(md / f"regions_{resp}.csv")
        m = reg.merge(syn[["design", "model", f"true_{resp}"]], on=["design", "model"])
        for tfm, g in m.groupby("model"):
            r = np.corrcoef(g["fit"], g[f"true_{resp}"])[0, 1]
            cover = float(((g["lo"] <= g[f"true_{resp}"]) & (g[f"true_{resp}"] <= g["hi"])).mean())
            crit = float(g["crit"].iloc[0])
            print(f"{resp:7s} {tfm:7s} designs {len(g):4d}  corr {r:.3f}  band coverage {cover:.3f}  crit {crit:.2f}")
            # a 300-design subset (the exact-gam test) carries less information than the full tier
            if r <= (0.9 if len(g) >= 800 else 0.8):
                bad.append(f"{resp} {tfm}: correlation {r:.3f}")
            if cover < 0.9:
                bad.append(f"{resp} {tfm}: coverage {cover:.3f}")
            if not crit > 1.96:
                bad.append(f"{resp} {tfm}: critical value {crit}")
        mitra = reg[reg["model"] == "mitra"]
        if (np.exp(mitra["log_n"]) ** 2 * mitra["p"] > 8.0e7 * (1 + 1e-9)).any():
            bad.append("Mitra-v2 predicted where it does not run")
        sh = pd.read_csv(md / f"region_shares_{resp}.csv")
        if not np.allclose(sh.groupby("model")["share"].sum(), 1):
            bad.append(f"{resp}: region shares do not sum to 1")
        pe = pd.read_csv(md / f"partial_effects_{resp}.csv")
        if not (pe.groupby(["model", "term"]).size() == 100).all() or pe["term"].nunique() != 9:
            bad.append(f"{resp}: partial-effect grids incomplete")
    tp = pd.read_csv(md / "transfer_predictions.csv")
    if not (tp.groupby("model")["scenario"].nunique() == 48).all():
        bad.append("transfer predictions: not 48 core scenarios per TFM")
    if not tp["data_source"].str.startswith("SYNTHETIC TEST DATA").all():
        bad.append("synthetic outputs are not labelled as synthetic")
    for b in bad:
        print("FAIL", b)
    print("meta-model synthetic check:", "passed" if not bad else f"{len(bad)} failure(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
