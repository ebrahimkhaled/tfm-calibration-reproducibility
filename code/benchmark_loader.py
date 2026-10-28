"""Loader of the released benchmark: every replicate's predictions and goodness-of-fit results
(pre-registration section 14, deliverable a).

The deposit holds, per scenario (or random design) and replicate, the evaluation rows with the true
risk p_star, the outcome y, the exact-null outcome of each arm, and the frozen prediction of every arm,
so that a new calibration method or test can be run on exactly the same predictions:

    preds/<scenario>/[<tag>/]<rep>.parquet     predictions (one file per replicate; see the dictionary)
    gof/<scenario>/<rep>.csv.gz                the study's own test results, one row per arm x outcome x test
    scenarios.csv, random_designs.csv          the design of every scenario and random design

    from benchmark_loader import Benchmark
    b = Benchmark("path/to/deposit")
    b.scenarios()                               # names
    ev = b.predictions("S3_pi10_n500", 7)        # wide: row, y, p_star, x..., p_<arm>, ystar_<arm>
    long = b.long("S3_pi10_n500", 7)            # long: row, y, p_star, arm, p_hat, y_star
    for name, rep, ev in b.replicates("R"):      # iterate over a family ("S1".."T4b", "R") or "all"
        ...
    g = b.gof("S3_pi10_n500", 7)                # the study's GOF rows for that replicate

    python code/benchmark_loader.py <deposit> [scenario rep]     prints a summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

META_KEY = b"tfmcal"


class Benchmark:
    def __init__(self, root):
        self.root = Path(root)
        self.pred_dir = self.root / "preds" if (self.root / "preds").exists() else self.root
        self.gof_dir = self.root / "gof"

    # ------------------------------------------------------------------ listing
    def scenarios(self, family: str = "all") -> list[str]:
        names = sorted(p.name for p in self.pred_dir.iterdir() if p.is_dir() and not p.name.startswith("_"))
        if family == "all":
            return names
        return [s for s in names if s.split("_")[0] == family or (family == "R" and s.startswith("R"))]

    def reps(self, scenario: str) -> list[int]:
        return sorted({int(f.stem) for f in (self.pred_dir / scenario).rglob("*.parquet")})

    # ------------------------------------------------------------------ one replicate
    def files(self, scenario: str, rep: int) -> list[Path]:
        return sorted((self.pred_dir / scenario).rglob(f"{int(rep):04d}.parquet"))

    def meta(self, scenario: str, rep: int) -> dict:
        """Seeds, fingerprints, and the status, timing and flags of every arm (merged over the files)."""
        out: dict = {}
        for f in self.files(scenario, rep):
            m = json.loads(pq.read_schema(f).metadata[META_KEY])
            if not out:
                out = m
            else:
                out["arms"].update(m.get("arms", {}))
        return out

    def predictions(self, scenario: str, rep: int, set_: str = "eval") -> pd.DataFrame:
        """The evaluation rows of one replicate, every arm's column merged (TFMs fitted on a CPU
        worker sit in a tag sub-folder of the same replicate)."""
        base = None
        for f in self.files(scenario, rep):
            d = pq.read_table(f).to_pandas()
            d = d[d["set"] == set_].drop(columns="set").reset_index(drop=True)
            if base is None:
                base = d
                continue
            if not (d["row"].equals(base["row"]) and d["y"].equals(base["y"])):
                raise ValueError(f"{f}: rows or outcomes differ from the other file of this replicate")
            new = [c for c in d.columns if c.startswith(("p_", "ystar_")) and c not in base.columns]
            base = pd.concat([base, d[new]], axis=1)
        if base is None:
            raise FileNotFoundError(f"no prediction file for {scenario} replicate {rep}")
        return base

    def long(self, scenario: str, rep: int) -> pd.DataFrame:
        """One row per (evaluation record, arm): row, y, p_star, arm, p_hat, y_star. The oracle arm's
        p_hat is p_star. Recalibration-ladder columns are arms named <arm>__R<k>_m<m>."""
        w = self.predictions(scenario, rep)
        arms = ["oracle"] + [c[2:] for c in w.columns if c.startswith("p_") and c != "p_star"]
        parts = []
        for a in arms:
            ph = w["p_star"] if a == "oracle" else w[f"p_{a}"]
            ys = (w[f"ystar_{a}"].astype("Int8") if f"ystar_{a}" in w
                  else pd.Series(pd.array([pd.NA] * len(w), dtype="Int8"), index=w.index))   # ladder columns
            parts.append(pd.DataFrame({"row": w["row"], "y": w["y"], "p_star": w["p_star"], "arm": a,
                                       "p_hat": ph.astype(float), "y_star": ys}))
        return pd.concat(parts, ignore_index=True)

    def replicates(self, family: str = "all"):
        for s in self.scenarios(family):
            for r in self.reps(s):
                yield s, r, self.predictions(s, r)

    def gof(self, scenario: str, rep: int) -> pd.DataFrame:
        f = self.gof_dir / scenario / f"{int(rep):04d}.csv.gz"
        if not f.exists():
            f = f.with_suffix("")             # an older uncompressed .csv
        return pd.read_csv(f, dtype={"scenario": str, "rep": str, "fingerprint": str, "pred_hash": str})


def _main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    b = Benchmark(argv[1])
    sc = b.scenarios()
    print(f"{len(sc)} scenarios: {', '.join(sc[:8])}{' ...' if len(sc) > 8 else ''}")
    s, r = (argv[2], int(argv[3])) if len(argv) > 3 else (sc[0], b.reps(sc[0])[0])
    w = b.predictions(s, r)
    print(f"{s} replicate {r}: {len(w)} evaluation rows; arms "
          f"{', '.join(c[2:] for c in w.columns if c.startswith('p_') and c != 'p_star')}")
    if (b.gof_dir / s).exists():
        g = b.gof(s, r)
        print(f"GOF rows: {len(g)} ({g['gof_ref'].iloc[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
