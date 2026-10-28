"""Collect the run's outputs into tidy tables under analysis/tables/ (read only on the run's folders).

    python analysis/collect.py [--scenarios all|S1_pi10_n500,...] [--reps 1-500] [--workers 2]
                               [--no-score] [--min-age 120]

Tables written (parquet):
  fits.parquet     one row per (scenario, rep, file tag, arm) from each prediction file's metadata:
                   status (ok / degenerate / failed / not_run), degenerate reason, the flags of 8.2
                   (platt_fallback, lambda_at_edge, small_n_flag, n_boundary_eval), device,
                   temperature, checkpoint.
  scores.parquet   one row per (scenario, rep, column) with every estimand of Section 4, exactly as
                   code/score.py computes it. The run's own results/metrics/<scenario>.parquet is used
                   for every file older than it; any newer file is scored in-process with
                   score.score_file() (same code, same fingerprint check), and cached.
  gof.parquet      one row per (scenario, rep, eval_set, arm, ytype, test) from results/gof*/.
  pairs.parquet    paired rows for every comparison of Section 7 (8.2: both arms usable, degenerate
                   fits kept): every arm against FLIC, each ladder level against FLIC's same level,
                   and each temperature arm against its main arm.
  manifest.json    what was read, what is still missing, and whether the data are complete.

Incremental and idempotent: per-scenario caches in tables/cache/ are keyed by each input file's
size and modification time, so a re-run reads only what changed. Files modified in the last
--min-age seconds, or with a live .lock beside them, are left for the next run (the run may be
writing them).
"""

from __future__ import annotations

import argparse
import os
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _common as C

sys.path.insert(0, str(C.CODE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import rules  # noqa: E402

META_KEY = b"tfmcal"
FIT_FLAGS = ("status", "degenerate_reason", "platt_fallback", "lambda_at_edge", "small_n_flag",
             "n_boundary", "n_boundary_eval", "device", "softmax_temperature", "n_estimators",
             "model_version", "checkpoint", "sha256", "seconds", "error")
GOF_KEEP = ["scenario", "rep", "eval_set", "n_eval", "events", "arm", "ytype", "ysource", "test",
            "statistic", "df", "p_value", "note", "gof_ref", "fingerprint"]
PAIR_METRICS = ["o_true_ici", "o_true_ll_ici", "o_oce", "o_true_e90", "o_true_slope",
                "o_true_intercept", "o_true_citl_prob", "abs_logb", "abs_a", "e_emp_ici"]


# ----------------------------------------------------------------------------- scenario info

def scenario_table() -> pd.DataFrame:
    s = pd.read_csv(C.CODE / "scenarios.csv")
    s = s[["scenario_id", "scenario", "family", "pi", "n", "p", "n_eval", "oracle_auc", "epv",
           "riley_n_min_pmsampsize"]].copy()
    s["tier"] = "core"
    s.loc[~s["family"].isin(C.SHAPES), "tier"] = "stress"
    rd = C.CODE / "random_designs.csv"
    if rd.exists():
        r = pd.read_csv(rd)
        r = r.rename(columns={"riley_n_min_py": "riley_n_min_pmsampsize"})
        r["tier"] = "random"
        cols = ["scenario_id", "scenario", "family", "pi", "n", "p", "n_eval", "oracle_auc", "epv",
                "riley_n_min_pmsampsize", "tier", "rho", "auc_target", "w_nl", "w_int", "w_link",
                "frac_disc"]
        s = pd.concat([s, r[cols]], ignore_index=True)
    s["pi"] = s["pi"].astype(float).round(10)
    return s


def parse_reps(text: str | None):
    if not text:
        return None
    lo, _, hi = text.partition("-")
    return int(lo), int(hi or lo)


def scenario_files(scen: str, reps, min_age: float) -> tuple[list[Path], list[str]]:
    """Prediction files of one scenario that are safe to read now, and those left for later."""
    root = C.PREDS / scen
    now = time.time()
    ok, later = [], []
    for f in sorted(root.rglob("*.parquet")):
        if any(part.startswith("_") for part in f.relative_to(root).parts):
            continue
        try:
            rep = int(f.stem)
        except ValueError:
            continue
        if reps and not (reps[0] <= rep <= reps[1]):
            continue
        st = f.stat()
        if now - st.st_mtime < min_age or f.with_name(f.name + ".lock").exists():
            later.append(str(f.relative_to(C.PREDS)))
            continue
        ok.append(f)
    return ok, later


def file_key(f: Path) -> list:
    st = f.stat()
    return [st.st_size, st.st_mtime_ns]


# ----------------------------------------------------------------------------- per-file work

def read_fits(f: Path) -> list[dict]:
    md = pq.read_schema(f).metadata
    meta = json.loads(md[META_KEY])
    tag = "" if f.parent.name == meta["scenario"] else f.parent.name
    rows = []
    for arm, info in meta["arms"].items():
        r = dict(scenario=meta["scenario"], rep=int(meta["rep"]), tag=tag, arm=arm)
        for k in FIT_FLAGS:
            v = info.get(k)
            r[k] = v if (v is None or isinstance(v, (bool, int, float, str))) else json.dumps(v)
        r["error"] = str(r.get("error") or "")[:300]
        rows.append(r)
    for arm, levels in meta.get("ladder", {}).items():
        for lev, info in levels.items():
            rows.append(dict(scenario=meta["scenario"], rep=int(meta["rep"]), tag=tag,
                             arm=f"{arm}__{lev}", status=info.get("status"),
                             error=str(info.get("error", "") or "")[:300]))
    return rows


def score_one(f: str) -> tuple[str, list[dict] | None, str | None]:
    """Worker: score.score_file on one file (imports here, so workers start light)."""
    import score  # noqa: E402  (code/score.py)
    try:
        return f, score.score_file(Path(f)), None
    except Exception as e:  # recorded in the manifest, never hidden
        return f, None, f"{type(e).__name__}: {e}"


# ----------------------------------------------------------------------------- scores

def collect_scenario_scores(scen: str, reps, min_age: float, workers: int, allow_score: bool) -> dict:
    cache_dir = C.assert_writable(C.CACHE / "scores")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath, mpath = cache_dir / f"{scen}.parquet", cache_dir / f"{scen}.files.json"
    fpath = cache_dir / f"{scen}.fits.parquet"
    cached = pd.read_parquet(cpath) if cpath.exists() else pd.DataFrame()
    cfits = pd.read_parquet(fpath) if fpath.exists() else pd.DataFrame()
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    files, later = scenario_files(scen, reps, min_age)

    run_metrics = C.METRICS / f"{scen}.parquet"
    rm = None
    rm_mtime = run_metrics.stat().st_mtime if run_metrics.exists() else -1

    keep_rows, keep_fits, todo, from_run = [], [], [], []
    new_manifest = {}
    for f in files:
        rel = str(f.relative_to(C.PREDS)).replace("\\", "/")
        key = file_key(f)
        rep = int(f.stem)
        tag = "" if f.parent.name == scen else f.parent.name
        sel = lambda d: d[(d["rep"] == rep) & (d["tag"] == tag)] if len(d) else d  # noqa: E731
        if manifest.get(rel) == key and len(cached):
            keep_rows.append(sel(cached))
            keep_fits.append(sel(cfits))
            new_manifest[rel] = key
            continue
        keep_fits.append(pd.DataFrame(read_fits(f)))
        if f.stat().st_mtime < rm_mtime:
            if rm is None:
                rm = pd.read_parquet(run_metrics)
                rm["tag"] = rm["tag"].fillna("")
            keep_rows.append(sel(rm))
            from_run.append(rel)
            new_manifest[rel] = key
        elif allow_score:
            todo.append((rel, f, key))
        else:
            later.append(rel)

    errors = {}
    if todo:
        C.progress(f"  {scen}: scoring {len(todo)} file(s) in-process ({workers} worker(s))")
        args = [str(f) for _, f, _ in todo]
        keymap = {str(f): (rel, key) for rel, f, key in todo}
        def take(results):
            for f, rows, err in results:
                rel, key = keymap[f]
                if err:
                    errors[rel] = err
                    continue
                keep_rows.append(pd.DataFrame(rows))
                new_manifest[rel] = key
        if workers <= 1:
            take(map(score_one, args))
        else:
            with ProcessPoolExecutor(workers) as ex:
                take(ex.map(score_one, args, chunksize=4))
    # entries for files outside this call's selection (another --reps range, or left for later
    # because the run is writing them) are carried over unchanged, so a filtered or interrupted
    # pass never loses work done earlier
    for rel, key in manifest.items():
        if rel in new_manifest or not (C.PREDS / rel).exists():
            continue
        p = C.PREDS / rel
        rep, tag = int(p.stem), ("" if p.parent.name == scen else p.parent.name)
        for src, dst in ((cached, keep_rows), (cfits, keep_fits)):
            if len(src):
                dst.append(src[(src["rep"] == rep) & (src["tag"] == tag)])
        new_manifest[rel] = key
    scores = pd.concat([d for d in keep_rows if len(d)], ignore_index=True) if keep_rows else pd.DataFrame()
    fits = pd.concat([d for d in keep_fits if len(d)], ignore_index=True) if keep_fits else pd.DataFrame()
    if len(scores):
        scores["tag"] = scores["tag"].fillna("")
        scores.to_parquet(cpath, index=False)
    if len(fits):
        fits.to_parquet(fpath, index=False)
    mpath.write_text(json.dumps(new_manifest))
    return dict(scenario=scen, files=len(files), scored=len(todo) - len(errors), from_run_metrics=len(from_run),
                left_for_later=len(later), errors=errors)


# ----------------------------------------------------------------------------- GOF

def normalise_arm(a: str) -> tuple[str, str, str]:
    """GOF arms are lower-cased by the R reader (flic__r2_m500): restore the score.py column name."""
    arm, _, lev = str(a).partition("__")
    lev = lev.upper().replace("_M", "_m") if lev else "R0"
    return arm, lev, (arm if lev == "R0" else f"{arm}__{lev}")


def collect_gof(scenarios: list[str], reps, min_age: float) -> dict:
    cache_dir = C.assert_writable(C.CACHE / "gof")
    cache_dir.mkdir(parents=True, exist_ok=True)
    roots = {"eval": C.GOF, "e50": C.REBUILD / "results" / "gof_e50", "e200": C.REBUILD / "results" / "gof_e200"}
    info = {}
    now = time.time()
    for es, root in roots.items():
        if not root.exists():
            continue
        for sdir in sorted(p for p in root.iterdir() if p.is_dir()):
            scen = sdir.name
            if scenarios and scen not in scenarios:
                continue
            cpath, mpath = cache_dir / f"{es}__{scen}.parquet", cache_dir / f"{es}__{scen}.files.json"
            cached = pd.read_parquet(cpath) if cpath.exists() else pd.DataFrame()
            manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
            parts, new_manifest, nread, bad = [], {}, 0, {}
            for f in sorted(sdir.iterdir()):
                name = f.name
                if not (name.endswith(".csv.gz") or name.endswith(".csv")) or "stale" in name or ".tmp" in name:
                    continue
                stem = name.split(".")[0]
                try:
                    rep = int(stem.replace("rep", ""))
                except ValueError:
                    continue
                if reps and not (reps[0] <= rep <= reps[1]):
                    continue
                if now - f.stat().st_mtime < min_age:
                    continue
                key = file_key(f)
                if manifest.get(name) == key and len(cached):
                    parts.append(cached[cached["rep"] == rep])
                    new_manifest[name] = key
                    continue
                try:
                    # text columns stay text: a fingerprint made only of digits would otherwise be
                    # read as a number (the same trap as the R checker, INCIDENTS 2026-10-14)
                    d = pd.read_csv(f, low_memory=False,
                                    dtype={"fingerprint": str, "gof_ref": str, "note": str})
                except Exception as e:  # a file caught mid-write is read next time
                    bad[name] = str(e)[:200]
                    continue
                d = d[[c for c in GOF_KEEP if c in d.columns]].copy()
                d["rep"] = rep
                d["eval_set"] = es if "eval_set" not in d else d["eval_set"].fillna(es)
                parts.append(d)
                new_manifest[name] = key
                nread += 1
            tab = pd.concat([p for p in parts if len(p)], ignore_index=True) if parts else pd.DataFrame()
            if len(tab):
                tab.to_parquet(cpath, index=False)
            mpath.write_text(json.dumps(new_manifest))
            info[f"{es}/{scen}"] = dict(files=len(new_manifest), read_now=nread, unreadable=bad)
    return info


def assemble_gof() -> pd.DataFrame:
    # about 13.5 million rows: each cache file is reduced before the concatenation. The free-text
    # columns (note, gof_ref, fingerprint) are used only to derive test_status here and stay in the
    # per-replicate files; the other text columns are stored as compact arrow strings.
    drop = ["note", "gof_ref", "fingerprint"]
    parts = []
    for p in sorted((C.CACHE / "gof").glob("*.parquet")):
        g = pd.read_parquet(p)
        note = g["note"].fillna("").astype(str) if "note" in g else pd.Series("", index=g.index)
        g["test_status"] = np.where(note.str.startswith("Not run"), "not_run",
                                    np.where(note.str.startswith("Failed"), "failed",
                                             np.where(g["p_value"].notna(), "ok", "no_p")))
        g = g.drop(columns=[c for c in drop if c in g])
        trip = {a: normalise_arm(a) for a in g["arm"].unique()}
        g["arm_base"] = g["arm"].map({a: t[0] for a, t in trip.items()})
        g["ladder"] = g["arm"].map({a: t[1] for a, t in trip.items()})
        g["column"] = g["arm"].map({a: t[2] for a, t in trip.items()})
        # rejection at 0.05; a test without a p-value counts neither way (8.2)
        g["reject"] = np.where(g["p_value"].notna(), (g["p_value"] <= C.ALPHA).astype(float), np.nan)
        for c in g.columns:
            if g[c].dtype == object:
                g[c] = g[c].astype("string[pyarrow]")
        parts.append(g)
    if not parts:
        return pd.DataFrame(columns=[c for c in GOF_KEEP if c not in drop]
                            + ["arm_base", "ladder", "column", "reject", "test_status"])
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------- pairs

def add_derived(s: pd.DataFrame) -> pd.DataFrame:
    s["abs_logb"] = rules.abs_log_b(s["o_true_slope"]) if "o_true_slope" in s else np.nan
    s["abs_logb_capped"] = rules.abs_log_b_capped(s["o_true_slope"]) if "o_true_slope" in s else np.nan
    s["abs_a"] = np.abs(s["o_true_intercept"]) if "o_true_intercept" in s else np.nan
    if "e_emp_ici" in s and "o_true_ici" in s:
        s["ici_gap"] = s["e_emp_ici"] - s["o_true_ici"]
    s["usable"] = s["status"].isin(C.USABLE)
    return s


def build_pairs(s: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "rep"]
    mets = [m for m in PAIR_METRICS if m in s.columns]
    base = s[keys + ["column", "arm", "ladder", "status"] + mets]
    out = []

    def pair(a: pd.DataFrame, b: pd.DataFrame, kind: str):
        m = a.merge(b, on=keys, suffixes=("", "_ref"))
        m = m[rules.pair_usable(m["status"], m["status_ref"])]
        if not len(m):
            return
        d = m[keys].copy()
        d["kind"], d["column"], d["arm"], d["ladder"] = kind, m["column"], m["arm"], m["ladder"]
        d["ref"] = m["column_ref"].values
        d["status"], d["status_ref"] = m["status"].values, m["status_ref"].values
        for x in mets:
            d[f"x_{x}"] = m[x].values
            d[f"r_{x}"] = m[f"{x}_ref"].values
            d[f"d_{x}"] = m[x].values - m[f"{x}_ref"].values
        out.append(d)

    flic = base[base["column"] == C.COMPARATOR]
    for col in sorted(set(base.loc[base["ladder"] == "R0", "column"]) - {C.COMPARATOR}):
        pair(base[base["column"] == col], flic, "vs_flic")
    for lev in C.LADDER_LEVELS[1:]:
        fl = base[base["column"] == f"{C.COMPARATOR}__{lev}"]
        for arm in C.TFMS:
            pair(base[base["column"] == f"{arm}__{lev}"], fl, "ladder_vs_flic")
    for t1, main in C.TEMPERATURE_PAIRS.items():
        pair(base[base["column"] == t1], base[base["column"] == main], "temperature")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ----------------------------------------------------------------------------- completeness

def expected_reps(scen: str, arm: str) -> int:
    info = C.parse_scenario(scen)
    if arm in C.CLASSICAL:
        return C.R_CLASSICAL
    if arm in C.TFMS:
        if arm == "mitra" and scen == "T2_pi10_n3000":
            return 0
        return C.r_tfm(info["n"])
    if arm == "tabpfn_legacy":
        return C.R_SIDE if info["core"] and abs(info["pi"] - 0.10) < 1e-9 else 0
    if arm in C.TEMPERATURE_PAIRS:
        return C.R_SIDE if info["core"] and abs(info["pi"] - 0.10) < 1e-9 and info["n"] in (200, 3000) else 0
    return 0


def completeness(fits: pd.DataFrame, gof: pd.DataFrame, scenarios: list[str]) -> dict:
    have = fits[~fits["arm"].str.contains("__")].groupby(["scenario", "arm"])["rep"].nunique() if len(fits) else pd.Series(dtype=int)
    rows = []
    for scen in scenarios:
        # random-design tier names (R0001 ...) have no family_pi_n form; that tier has its own check
        if "_" not in scen or C.parse_scenario(scen)["family"].startswith("R"):
            continue
        for arm in C.ALL_ARMS:
            e = expected_reps(scen, arm)
            if e:
                rows.append(dict(scenario=scen, arm=arm, expected=e, present=int(have.get((scen, arm), 0))))
    t = pd.DataFrame(rows)
    by_arm = t.groupby("arm")[["expected", "present"]].sum()
    # the GOF layer is complete when every planned scenario has its 500 oracle Y* replicates
    goft = gof[(gof["eval_set"] == "eval") & (gof["arm"] == "oracle") & (gof["ytype"] == "ystar")] if len(gof) else gof
    oracle_reps = goft.groupby("scenario")["rep"].nunique() if len(goft) else pd.Series(dtype=int)
    planned = C.all_scenarios()
    gof_done = all(int(oracle_reps.get(s, 0)) >= C.R_CLASSICAL for s in planned)
    return dict(predictions_complete=bool(len(t)) and bool((t["present"] >= t["expected"]).all()),
                by_arm={a: dict(expected=int(r.expected), present=int(r.present)) for a, r in by_arm.iterrows()},
                gof_scenarios=int(len(oracle_reps)), gof_complete=bool(gof_done))


# ----------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--reps", default=None, help="restrict to a replicate range, e.g. 1-20 (interim runs)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-score", action="store_true", help="never score in-process (use results/metrics only)")
    ap.add_argument("--min-age", type=float, default=120.0)
    ap.add_argument("--skip-gof", action="store_true")
    a = ap.parse_args(argv)
    # never more than 2 while the run is going; ANALYSIS_MAX_WORKERS may raise it once the run is over
    workers = max(1, min(a.workers, int(os.environ.get("ANALYSIS_MAX_WORKERS", "2"))))
    reps = parse_reps(a.reps)
    t0 = time.time()
    C.TABLES.mkdir(parents=True, exist_ok=True)
    C.assert_writable(C.TABLES)

    present = sorted(p.name for p in C.PREDS.iterdir() if p.is_dir())
    want = present if a.scenarios == "all" else [s for s in a.scenarios.split(",") if s in present]
    info = {}
    for scen in want:
        try:
            info[scen] = collect_scenario_scores(scen, reps, a.min_age, workers, not a.no_score)
        except Exception:
            info[scen] = dict(error=traceback.format_exc()[-800:])
        C.progress(f"{scen}: {json.dumps({k: v for k, v in info[scen].items() if k != 'errors'})}")

    sc = scenario_table()
    parts = [pd.read_parquet(p) for p in sorted((C.CACHE / "scores").glob("*.parquet"))
             if not p.name.endswith(".fits.parquet")]
    scores = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    fparts = [pd.read_parquet(p) for p in sorted((C.CACHE / "scores").glob("*.fits.parquet"))]
    fits = pd.concat(fparts, ignore_index=True) if fparts else pd.DataFrame()
    if reps and len(scores):
        scores = scores[(scores["rep"] >= reps[0]) & (scores["rep"] <= reps[1])]
        fits = fits[(fits["rep"] >= reps[0]) & (fits["rep"] <= reps[1])]
    if len(scores):
        # a replicate's arm is scored once (oracle from the main file only, TFM tags never overlap);
        # if a column ever appears twice, the main-folder copy wins
        scores = scores.sort_values(["scenario", "rep", "column", "tag"]).drop_duplicates(
            ["scenario", "rep", "column"], keep="first")
        scores = add_derived(scores).merge(sc, on="scenario", how="left")
        scores.to_parquet(C.TABLES / "scores.parquet", index=False)
        pairs = build_pairs(scores)
        pairs = pairs.merge(sc[["scenario", "family", "pi", "n", "tier"]], on="scenario", how="left") if len(pairs) else pairs
        pairs.to_parquet(C.TABLES / "pairs.parquet", index=False)
    if len(fits):
        fits = fits.merge(sc[["scenario", "family", "pi", "n", "tier"]], on="scenario", how="left")
        for c in ("degenerate_reason", "device", "model_version", "checkpoint", "sha256", "error", "status"):
            if c in fits:
                fits[c] = fits[c].astype("string")
        fits.to_parquet(C.TABLES / "fits.parquet", index=False)

    gof_info = {} if a.skip_gof else collect_gof(want, reps, a.min_age)
    gof = assemble_gof()
    if len(gof):
        gof = gof.merge(sc[["scenario", "family", "pi", "n", "tier"]], on="scenario", how="left")
    gof.to_parquet(C.TABLES / "gof.parquet", index=False)

    comp = completeness(fits, gof, want) if len(fits) else {}
    man = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 1),
               reps_filter=a.reps, n_scores=int(len(scores)), n_fits=int(len(fits)), n_gof=int(len(gof)),
               complete=bool(comp.get("predictions_complete")) and bool(comp.get("gof_complete"))
               and not a.reps,
               completeness=comp, scenarios=info, gof=gof_info)
    (C.TABLES / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    C.progress(f"collect done in {man['seconds']} s: {man['n_scores']} score rows, {man['n_gof']} GOF rows; "
               f"complete = {man['complete']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
