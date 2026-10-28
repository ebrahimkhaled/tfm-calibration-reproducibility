"""Summary of the pre-freeze smoke (code/smoke_freeze.sh): fits, flags, scoring and GOF rows.
Errors and coverage only; the smoke stream's calibration numbers are not inspected as results.

    python code/smoke_freeze_summary.py [results/smoke_freeze_b]   -> <folder>/SUMMARY.md
    (default folder results/smoke_freeze, the first smoke of 2026-10-07)
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REB = Path(__file__).resolve().parents[1]


def md(df: pd.DataFrame, index: bool = True) -> str:
    """A markdown table without the optional tabulate package."""
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep] + body)


OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REB / "results" / "smoke_freeze"
lines = ["# Pre-freeze smoke summary", ""]

fits = []
for f in sorted((OUT / "preds").rglob("*.parquet")):
    m = json.loads(pq.read_schema(f).metadata[b"tfmcal"])
    for arm, i in m["arms"].items():
        fits.append(dict(scenario=m["scenario"], rep=m["rep"], file=f.relative_to(OUT / "preds").as_posix(), arm=arm,
                         status=i.get("status"), device=i.get("device"), seconds=round(i.get("seconds") or 0, 1),
                         model=i.get("model_version") or i.get("checkpoint") or i.get("revision") or "",
                         sha=(i.get("checkpoint_sha256") or "")[:12], T=i.get("softmax_temperature"),
                         n_boundary_eval=i.get("n_boundary_eval"), degenerate=i.get("degenerate_reason") or "",
                         platt_b=i.get("platt_b"), platt_fallback=i.get("platt_fallback"),
                         ridge_lambda_idx=i.get("lambda_min_index"), ridge_edge=i.get("lambda_at_edge"),
                         error=(i.get("error") or "")[:80]))
fits = pd.DataFrame(fits)
lines += ["## Fits (every arm, both replicates)", "", md(fits, index=False), ""]

met = pd.concat([pd.read_parquet(f) for f in sorted((OUT / "metrics").glob("*.parquet"))])
r0 = met[met.ladder == "R0"]
cnt = r0.groupby(["scenario", "status"]).size().unstack(fill_value=0)
lines += ["## Scoring: R0 rows by status", "", md(cnt), "",
          f"Rows scored: {len(met)}; non-finite true ICI among usable rows: "
          f"{int(met[met.status.isin(['ok', 'degenerate'])].o_true_ici.isna().sum())}", ""]

g = pd.concat([pd.read_csv(f, dtype={"rep": str}) for f in sorted((OUT / "gof").rglob("*.csv.gz"))])
plain = list((OUT / "gof").rglob("*.csv"))
lines += ["## GOF layer", "",
          f"Files: {len(list((OUT / 'gof').rglob('*.csv.gz')))} .csv.gz, {len(plain)} uncompressed .csv; "
          f"ebrahim.gof reference {sorted(g.gof_ref.unique())}", ""]
base = g[~g.arm.str.contains("__")]
pairs = base.groupby(["scenario", "arm"]).ytype.apply(lambda s: ",".join(sorted(s.unique()))).unstack(0)
lines += ["Arms tested, by y type:", "", md(pairs), ""]
orr = base[base.test == "Osius-Rojek (external)"]
lines += [f"Osius-Rojek (external) rows: {len(orr)} (finite p-value: {int(orr.p_value.notna().sum())}); "
          f"arm x y-type pairs without the row: "
          f"{len(base.groupby(['scenario', 'rep', 'arm', 'ytype']).size()) - len(orr)}", ""]
st = base[base.test == "Stukel (offset)"]
lines += [f"Stukel (offset) rows: {len(st)}; Note says \"score test\": {int(st.note.astype(str).str.contains('score test').sum())}; "
          f"finite p-value: {int(st.p_value.notna().sum())}; df: {sorted(st.df.dropna().unique().tolist())}", ""]
errs = list((OUT / "gof").rglob("*.error.txt"))
lines += [f"Open GOF error files: {len(errs)}", ""]
dg = g[g.family == "DeepGOF"]
lines += ["DeepGOF-1 rows (first row of each call):", "",
          dg[dg.secs.notna()][["scenario", "rep", "arm", "test", "p_value", "note", "secs"]].pipe(md, index=False), ""]
notrun = g[g.note.astype(str).str.contains("Not run|Failed", na=False)]
lines += [f"Rows not run or failed: {len(notrun)}", ""]
if len(notrun):
    lines += [notrun[["scenario", "rep", "arm", "ytype", "test", "note"]].pipe(md, index=False), ""]
(OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
sys.exit(0)
