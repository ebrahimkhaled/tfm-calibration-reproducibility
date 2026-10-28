"""
Real-data loaders for the TFM calibration study.

Four sources, one clean analysis table per outcome:

    ist_death14      International Stroke Trial, death within 14 days
    ist_dd6          International Stroke Trial, death or dependency at 6 months
    crash2_death28   CRASH-2, death in hospital within 28 days of randomisation
    sepsis_death     Norwegian sepsis admissions (UCI 827), death at about 9 days
    physionet_ihd    PhysioNet/CinC Challenge 2012, in-hospital death

Each table is written to data/clean/<key>.parquet with a sidecar <key>.meta.json
(patient flow, predictors and their types, missing-data policy, split sizes,
event counts, SHA256 of the raw files and a fingerprint of the table itself).

Usage (from the rebuild folder):

    python code/realdata.py manifest            # hash every file in data/raw
    python code/realdata.py build               # build every table whose raw files exist
    python code/realdata.py build ist crash2    # only some sources
    python code/realdata.py plasmode ist_dd6    # freeze the plasmode truth model
    python code/realdata.py summary             # print n / events / splits

IST and CRASH-2 need the author to download the files by hand
(see data/AUTHOR_STEPS.md). Until then their builders just say so and skip.

Missing data. The policy is fixed per source and written to the meta file:
  - IST, CRASH-2: complete case on the predictors (missingness is small;
    CRASH-2 follows Perel et al. 2012, who did the same). The table with
    missing values kept is also saved as <key>_all.parquet for a sensitivity
    analysis.
  - Sepsis: nothing is missing.
  - PhysioNet 2012: complete case would drop most stays, so missing values
    stay as NaN in the table. Every arm then gets the same matrix from
    impute_simple(): medians and missing-value indicators learned on the
    development rows only.
Rows with a missing outcome are always excluded, and counted.
"""

import argparse
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
MANIFEST = RAW / "MANIFEST_SHA256.txt"

# The external region gives a fixed local sample for the recalibration ladder;
# the rest of the external region is the evaluation set.
N_RECAL = 500
SPLIT_SEED = {"ist": 20261006, "crash2": 20261007, "sepsis": 20261008, "physionet": 20261009}


# ---------------------------------------------------------------------------
# hashing, manifest and fingerprints
# ---------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest():
    out = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, rel = line.split("  ", 1)
            out[rel] = digest
    return out


def write_manifest(entries):
    lines = ["# SHA256 of every raw file (sha256sum format, paths relative to data/raw)"]
    for rel in sorted(entries):
        lines.append(f"{entries[rel]}  {rel}")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf8")


def update_manifest():
    """Hash every file under data/raw except the manifest itself.

    A file already in the manifest must still match; a changed raw file is an
    error, not something to overwrite silently.
    """
    old = read_manifest()
    new = {}
    for p in sorted(RAW.rglob("*")):
        if p.is_file() and p != MANIFEST:
            rel = p.relative_to(RAW).as_posix()
            digest = sha256_file(p)
            if rel in old and old[rel] != digest:
                raise RuntimeError(f"raw file changed since it was recorded: {rel}")
            new[rel] = digest
    write_manifest(new)
    return new


def check_raw(paths):
    """Assert raw files match the manifest; files not yet listed are added."""
    man = read_manifest()
    hashes = {}
    changed = False
    for p in paths:
        rel = Path(p).relative_to(RAW).as_posix()
        digest = sha256_file(p)
        if rel in man:
            if man[rel] != digest:
                raise RuntimeError(f"{rel}: SHA256 does not match the manifest")
        else:
            man[rel] = digest
            changed = True
        hashes[rel] = digest
    if changed:
        write_manifest(man)
    return hashes


def table_fingerprint(df):
    """Stable hash of a table's content (independent of the parquet writer)."""
    buf = io.StringIO()
    df.sort_index(axis=1).to_csv(buf, index=False, float_format="%.6g", lineterminator="\n")
    return hashlib.sha256(buf.getvalue().encode("utf8")).hexdigest()


def seed_for(*parts):
    """Per-unit seed from a readable key, e.g. seed_for('ist_dd6', 500, 17)."""
    s = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode("utf8")).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

def add_split(df, dev_mask, seed, col="split"):
    """dev_pool for the development region, ext_recal (500 rows) + ext_eval for the rest."""
    split = np.where(dev_mask, "dev_pool", "ext_eval").astype(object)
    ext_idx = np.flatnonzero(~np.asarray(dev_mask))
    rng = np.random.default_rng(seed)
    if len(ext_idx) > N_RECAL:
        pick = rng.choice(ext_idx, size=N_RECAL, replace=False)
        split[pick] = "ext_recal"
    df[col] = split
    return df


def split_counts(df, y="y", col="split"):
    out = {}
    for s, g in df.groupby(col):
        out[s] = {"n": int(len(g)), "events": int(g[y].sum()), "rate": round(float(g[y].mean()), 4)}
    return out


def subsample_dev(df, key, n, b, col="split"):
    """Row positions of learning-curve subsample b (0-based) of size n from dev_pool.

    Simple random sampling without replacement. A subsample may contain very
    few events; that is kept on purpose (degenerate fits are an outcome).
    """
    pool = np.flatnonzero(df[col].to_numpy() == "dev_pool")
    if n > len(pool):
        raise ValueError(f"{key}: n={n} exceeds the development pool ({len(pool)})")
    rng = np.random.default_rng(seed_for(key, col, n, b))
    return np.sort(rng.choice(pool, size=n, replace=False))


# ---------------------------------------------------------------------------
# missing data
# ---------------------------------------------------------------------------

def impute_simple(dev, others, predictors, indicator_min=0.05):
    """Median imputation fitted on the development rows only.

    A missing-value indicator is added for every predictor with at least
    `indicator_min` missing in the development rows. Returns new frames plus
    the list of added indicator columns. The same transform is used for every
    arm, so all models see the same matrix.
    """
    dev = dev.copy()
    others = [o.copy() for o in others]
    added = []
    for v in predictors:
        if not pd.api.types.is_numeric_dtype(dev[v]):
            if dev[v].isna().any():
                raise ValueError(f"{v}: missing values in a categorical predictor")
            continue
        miss = dev[v].isna().mean()
        med = dev[v].median()
        if np.isnan(med):
            med = 0.0
        if miss >= indicator_min:
            name = v + "_missing"
            dev[name] = dev[v].isna().astype(int)
            for o in others:
                o[name] = o[v].isna().astype(int)
            added.append(name)
        dev[v] = dev[v].fillna(med)
        for o in others:
            o[v] = o[v].fillna(med)
    return dev, others, added


def design_matrix(df, meta, onehot=True):
    """Numeric matrix for the regression arms: categorical predictors one-hot (first level dropped)."""
    cols = []
    for v, kind in meta["predictor_types"].items():
        if kind == "categorical" and onehot:
            levels = meta["levels"][v]
            for lev in levels[1:]:
                cols.append((df[v] == lev).astype(float).rename(f"{v}={lev}"))
        else:
            cols.append(df[v].astype(float))
    return pd.concat(cols, axis=1)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def save_table(key, df, meta, out_dir=None, force=False):
    out_dir = Path(out_dir or CLEAN)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = table_fingerprint(df)
    meta_path = out_dir / f"{key}.meta.json"
    if meta_path.exists() and not force:
        old = json.loads(meta_path.read_text(encoding="utf8"))
        if old.get("fingerprint") and old["fingerprint"] != fp:
            raise RuntimeError(
                f"{key}: the rebuilt table differs from the stored one "
                f"({old['fingerprint'][:12]} vs {fp[:12]}). Check the raw files; "
                f"use --force only if the change is intended and logged."
            )
    meta = dict(meta)
    meta["fingerprint"] = fp
    meta["n"] = int(len(df))
    meta["events"] = int(df["y"].sum())
    meta["event_rate"] = round(float(df["y"].mean()), 4)
    meta["splits"] = {c: split_counts(df, col=c) for c in df.columns if c.startswith("split")}
    meta["missing_by_predictor"] = {v: int(df[v].isna().sum()) for v in meta["predictors"]}
    # learning curves go up to n = 3000, so every development pool needs more than that
    for c, sc in meta["splits"].items():
        if sc.get("dev_pool", {}).get("n", 0) < 3500 and not key.endswith("_all") and key != "sepsis_korea":
            print(f"  WARNING {key}/{c}: development pool has {sc.get('dev_pool', {}).get('n', 0)} rows "
                  f"(< 3500); apply the fallback split in DATA_REPORT.md")
            meta.setdefault("warnings", []).append(f"{c}: dev_pool below 3500")
    df.to_parquet(out_dir / f"{key}.parquet", index=False)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf8")
    print(f"  {key}: n={meta['n']}, events={meta['events']} ({meta['event_rate']:.1%}), fp={fp[:12]}")
    return meta


def complete_case(df, predictors):
    keep = df[predictors].notna().all(axis=1)
    return df.loc[keep].reset_index(drop=True), int((~keep).sum())


# ---------------------------------------------------------------------------
# International Stroke Trial (IST), corrected version 2 on Edinburgh DataShare
# ---------------------------------------------------------------------------

IST_DEFICITS = [f"RDEF{i}" for i in range(1, 9)]
IST_CONT = ["age", "rdelay", "rsbp"]
IST_BIN = ["female", "rsleep", "ratrial", "rasp3", "rhep24"] + [d.lower() for d in IST_DEFICITS] + ["def_unassessable"]
IST_CAT = {"rconsc": ["F", "D", "U"], "ct": ["none", "no_infarct", "infarct"],
           "stype": ["PACS", "TACS", "LACS", "POCS", "OTH"]}
UK_CODE = 27


def find_ist_file(raw_dir):
    d = Path(raw_dir) / "ist"
    for name in ["IST_corrected.csv", "IST_corrected.txt"]:
        if (d / name).exists():
            return d / name
    return None


def _yn(s):
    s = s.astype(str).str.strip().str.upper()
    return s.map({"Y": 1.0, "N": 0.0})


# The corrected IST file writes RDATE as a Polish month abbreviation and a two-digit year
# ("sty-91" ... "gru-96"); the file is cp1250, so "paz" (October, with z-acute) reads as "pa\x9f"
# under latin-1. Only the first two letters are needed to tell the months apart, except
# "ma" (marzec / maj) and "li" (lipiec / listopad), which need three.
IST_PL_MONTHS = {"sty": 1, "lut": 2, "mar": 3, "kwi": 4, "maj": 5, "cze": 6, "lip": 7, "sie": 8,
                 "wrz": 9, "pa": 10, "lis": 11, "gru": 12}


def _pl_month(m):
    m = str(m).lower()
    if m[:3] in IST_PL_MONTHS:
        return IST_PL_MONTHS[m[:3]]
    return IST_PL_MONTHS.get(m[:2], np.nan)


def _rdate_year(s):
    s = s.astype(str).str.strip()
    out = pd.Series(np.nan, index=s.index)
    pl = s.str.extract(r"^([^\W\d_][^-]*)-(\d{2})$")
    if pl.notna().all(axis=1).any():
        mon = pl[0].map(_pl_month)
        yr = pd.to_numeric(pl[1], errors="coerce")
        yr = np.where(yr < 50, 2000 + yr, 1900 + yr)
        out = out.fillna(pd.Series(yr + (mon - 1) / 12.0, index=s.index))
    for fmt in ("%Y-%m", "%y-%b", "%b-%y", "%Y/%m", "%m/%Y"):
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        out = out.fillna(parsed.dt.year + (parsed.dt.month - 1) / 12.0)
    return out


def build_ist(raw_dir=RAW, out_dir=None, force=False):
    path = find_ist_file(raw_dir)
    if path is None:
        print("  IST: no IST_corrected.csv in data/raw/ist -- see AUTHOR_STEPS.md; skipped")
        return None
    hashes = check_raw([path]) if Path(raw_dir) == RAW else {path.name: sha256_file(path)}
    sep = "\t" if path.suffix == ".txt" else ","
    raw = pd.read_csv(path, sep=sep, low_memory=False, encoding="latin-1")
    raw.columns = [c.strip().upper() for c in raw.columns]
    if len(raw) != 19435:
        raise RuntimeError(f"IST: expected 19435 patients, found {len(raw)}")

    d = pd.DataFrame({"id": np.arange(len(raw))})
    d["age"] = pd.to_numeric(raw["AGE"], errors="coerce")
    d["female"] = raw["SEX"].astype(str).str.strip().str.upper().map({"F": 1.0, "M": 0.0})
    d["rdelay"] = pd.to_numeric(raw["RDELAY"], errors="coerce")
    d["rsbp"] = pd.to_numeric(raw["RSBP"], errors="coerce")
    d["rconsc"] = raw["RCONSC"].astype(str).str.strip().str.upper().where(lambda s: s.isin(IST_CAT["rconsc"]))
    d["rsleep"] = _yn(raw["RSLEEP"])
    d["ratrial"] = _yn(raw["RATRIAL"])          # blank for the 984 pilot patients
    d["rasp3"] = _yn(raw["RASP3"])
    d["rhep24"] = _yn(raw["RHEP24"])
    rct = _yn(raw["RCT"])
    rvis = _yn(raw["RVISINF"])
    d["ct"] = np.where(rct == 0, "none", np.where(rvis == 1, "infarct", np.where(rvis == 0, "no_infarct", None)))
    d.loc[rct.isna(), "ct"] = None
    # Deficits: Y = 1, N = 0. "C" (cannot assess, usually a drowsy or unconscious
    # patient) is coded 0 for the item and flagged once in def_unassessable.
    cant = np.zeros(len(raw), dtype=bool)
    for v in IST_DEFICITS:
        s = raw[v].astype(str).str.strip().str.upper()
        cant |= (s == "C").to_numpy()
        d[v.lower()] = s.map({"Y": 1.0, "N": 0.0, "C": 0.0})
    d["def_unassessable"] = cant.astype(float)
    d["stype"] = raw["STYPE"].astype(str).str.strip().str.upper().where(lambda s: s.isin(IST_CAT["stype"]))
    # not predictors: region, calendar time, trial arms, IST's own predicted risks
    d["country"] = pd.to_numeric(raw["CNTRYNUM"], errors="coerce")
    d["rdate"] = _rdate_year(raw["RDATE"])
    d["rxasp"] = raw["RXASP"].astype(str).str.strip().str.upper()
    d["rxhep"] = raw["RXHEP"].astype(str).str.strip().str.upper().replace({"H": "M"})
    for v in ["EXPD14", "EXPDD"]:
        if v in raw.columns:
            d[v.lower()] = pd.to_numeric(raw[v], errors="coerce")

    predictors = IST_CONT + IST_BIN + list(IST_CAT)
    types = {**{v: "continuous" for v in IST_CONT}, **{v: "binary" for v in IST_BIN},
             **{v: "categorical" for v in IST_CAT}}
    if d["rdate"].isna().all():
        raise RuntimeError("IST: RDATE could not be parsed; check its format and extend _rdate_year")

    outcomes = {}
    # 14-day death: ID14, among patients whose 14-day vital status is known (SET14D = 1)
    set14 = pd.to_numeric(raw["SET14D"], errors="coerce")
    id14 = pd.to_numeric(raw["ID14"], errors="coerce")
    outcomes["ist_death14"] = (id14.where(set14 == 1),
                               "Death from any cause within 14 days of randomisation (ID14), "
                               "patients with known 14-day status (SET14D = 1)")
    # 6-month death or dependency: OCCODE 1 (dead) or 2 (dependent) vs 3, 4; 0/9 = unknown
    occ = pd.to_numeric(raw["OCCODE"], errors="coerce")
    dd6 = pd.Series(np.where(occ.isin([1, 2]), 1.0, np.where(occ.isin([3, 4]), 0.0, np.nan)))
    outcomes["ist_dd6"] = (dd6, "Dead or dependent at 6 months (OCCODE 1 or 2 vs 3 or 4); "
                                "OCCODE 0 or 9 = outcome unknown, excluded")

    metas = {}
    for key, (y, definition) in outcomes.items():
        t = d.copy()
        t["y"] = y.to_numpy()
        flow = {"randomised": int(len(t))}
        t = t.loc[t["y"].notna()].reset_index(drop=True)
        flow["outcome_known"] = int(len(t))
        t["y"] = t["y"].astype(int)
        # Splits. Primary: geographic, UK develops, every other country validates.
        # Secondary: temporal, randomised before 1994 develops, 1994-1996 validates.
        # Split after the complete-case step so ext_recal always has 500 usable rows.
        def splits(x):
            add_split(x, (x["country"] == UK_CODE).to_numpy(), SPLIT_SEED["ist"], "split")
            add_split(x, (x["rdate"] < 1994).to_numpy(), SPLIT_SEED["ist"] + 1, "split_temporal")
            return x
        all_rows = splits(t.copy())
        t, dropped = complete_case(t, predictors)
        splits(t)
        flow["complete_case"] = int(len(t))
        flow["dropped_missing_predictor"] = dropped
        meta = {
            "key": key, "source": "International Stroke Trial database v2 (corrected)",
            "outcome": definition, "predictors": predictors, "predictor_types": types,
            "levels": IST_CAT, "flow": flow, "missing_policy": "complete case on predictors; "
            "missing-kept table in " + key + "_all.parquet",
            "splits_definition": {
                "split": "geographic: dev_pool = UK (CNTRYNUM 27); external = all other countries; "
                         f"{N_RECAL} random external rows = ext_recal",
                "split_temporal": "temporal: dev_pool = randomised before 1994; external = 1994-1996"},
            "raw_sha256": hashes,
            "notes": ["Baseline variables only (recorded before randomisation).",
                      "EXPD14/EXPDD are IST-supplied predicted risks; kept as columns, never as predictors.",
                      "Trial arms (rxasp, rxhep) kept as columns, not predictors (balanced by randomisation)."],
        }
        save_table(key + "_all", all_rows, {**meta, "key": key + "_all", "missing_policy": "missing values kept"}, out_dir, force)
        metas[key] = save_table(key, t, meta, out_dir, force)
    return metas


# ---------------------------------------------------------------------------
# CRASH-2 (freeBIRD, LSHTM)
# ---------------------------------------------------------------------------

CRASH2_CONT = ["age", "gcs", "sbp", "hr", "rr", "hours_since_injury"]
CRASH2_BIN = ["female", "penetrating"]
CRASH2_CUT = 2008.0   # randomised before 1 Jan 2008 = development, later = external


def find_crash2_file(raw_dir):
    d = Path(raw_dir) / "crash2"
    for p in sorted(d.glob("*.csv")):
        head = p.read_text(encoding="latin-1", errors="replace")[:2000].lower()
        if "ientryid" in head:
            return p
    return None


def build_crash2(raw_dir=RAW, out_dir=None, force=False):
    path = find_crash2_file(raw_dir)
    if path is None:
        print("  CRASH-2: no CSV with column ientryid in data/raw/crash2 -- see AUTHOR_STEPS.md; skipped")
        return None
    hashes = check_raw([path]) if Path(raw_dir) == RAW else {path.name: sha256_file(path)}
    raw = pd.read_csv(path, low_memory=False, encoding="latin-1")
    raw.columns = [c.strip().lower() for c in raw.columns]
    if len(raw) != 20207:
        raise RuntimeError(f"CRASH-2: expected 20207 patients, found {len(raw)}")

    num = lambda v: pd.to_numeric(raw[v], errors="coerce")
    d = pd.DataFrame({"id": num("ientryid")})
    d["age"] = num("iage")
    d["female"] = num("isex").map({1: 0.0, 2: 1.0})
    # missing-value codes from the data dictionary
    d["sbp"] = num("isbp").where(lambda s: ~s.isin([0, 999]))
    d["rr"] = num("irr").where(lambda s: s != 0)
    d["hr"] = num("ihr")
    d["hours_since_injury"] = num("ninjurytime").where(lambda s: ~s.isin([0, 999]))
    gcs = num("igcs")
    parts = num("igcseye") + num("igcsmotor") + num("igcsverbal")
    d["gcs"] = gcs.fillna(parts)
    d["penetrating"] = num("iinjurytype").map({1: 0.0, 2: 1.0, 3: 1.0})
    d["crt"] = num("icc")    # capillary refill: kept, not a predictor (dropped by Perel et al.)

    if "trandomised" not in raw.columns:
        # the freeBIRD anonymised file has no dates: deviation D-01 (see prereg/DEVIATIONS.md)
        return build_crash2_freebird(raw, d, path, hashes, out_dir, force)

    rand = pd.to_datetime(raw["trandomised"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    rand = rand.fillna(pd.to_datetime(raw["trandomised"], dayfirst=True, errors="coerce"))
    death = pd.to_datetime(raw["ddeath"], dayfirst=True, errors="coerce")
    d["rdate"] = rand.dt.year + (rand.dt.dayofyear - 1) / 365.25
    days = (death.dt.normalize() - rand.dt.normalize()).dt.days

    flow = {"randomised": int(len(d))}
    has_outcome = num("ioutcomeid").notna()
    y = pd.Series(np.where(death.notna() & (days <= 28), 1.0, 0.0))
    y[~has_outcome] = np.nan
    flow["deaths_after_day_28_coded_0"] = int((death.notna() & (days > 28)).sum())
    flow["death_date_before_randomisation"] = int((days < 0).sum())
    d["y"] = y.to_numpy()
    d = d.loc[d["y"].notna()].reset_index(drop=True)
    flow["outcome_known"] = int(len(d))
    d["y"] = d["y"].astype(int)
    if d["rdate"].isna().any():
        raise RuntimeError("CRASH-2: some randomisation dates did not parse")

    predictors = CRASH2_CONT + CRASH2_BIN
    types = {**{v: "continuous" for v in CRASH2_CONT}, **{v: "binary" for v in CRASH2_BIN}}
    # split after the complete-case step so ext_recal always has 500 usable rows;
    # the missing-kept table gets its own split with the same rule and seed
    all_rows = add_split(d.copy(), (d["rdate"] < CRASH2_CUT).to_numpy(), SPLIT_SEED["crash2"], "split")
    d, dropped = complete_case(d, predictors)
    add_split(d, (d["rdate"] < CRASH2_CUT).to_numpy(), SPLIT_SEED["crash2"], "split")
    flow["complete_case"] = int(len(d))
    flow["dropped_missing_predictor"] = dropped
    meta = {
        "key": "crash2_death28", "source": "CRASH-2 trial data, freeBIRD (doi:10.17037/DATA.10)",
        "outcome": "Death in hospital within 28 days of randomisation (ddeath recorded on the outcome form); "
                   "patients without an outcome form excluded",
        "predictors": predictors, "predictor_types": types, "levels": {}, "flow": flow,
        "missing_policy": "complete case on predictors (as Perel et al. 2012); missing-kept table in crash2_death28_all.parquet",
        "splits_definition": {"split": f"temporal: dev_pool = randomised before {int(CRASH2_CUT)}-01-01; "
                                       f"external = later; {N_RECAL} random external rows = ext_recal"},
        "raw_sha256": hashes,
        "notes": ["The public file has no country and no treatment code, so the income-region "
                  "split and the treatment adjustment of Perel et al. cannot be reproduced.",
                  "Predictors follow the Perel et al. 2012 final model (GCS, age, SBP, HR, RR, "
                  "time since injury, injury type) plus sex; capillary refill excluded as in that model.",
                  "Time since injury is kept continuous (Perel et al. dichotomised it at 3 h)."],
    }
    save_table("crash2_death28_all", all_rows, {**meta, "key": "crash2_death28_all", "missing_policy": "missing values kept"}, out_dir, force)
    return {"crash2_death28": save_table("crash2_death28", d, meta, out_dir, force)}


CRASH2_EXT_SHARE = 0.30   # share of patients in held-out hospitals (deviation D-01)


def build_crash2_freebird(raw, d, path, hashes, out_dir=None, force=False):
    """CRASH-2 from the freeBIRD anonymised file, which carries no dates.

    Outcome: death within 28 days of randomisation (daysrandtodeath <= 28); the
    outcome is known when a discharge status (istatus) or a death is recorded.
    Split: whole hospitals (iSiteID) are held out. Hospitals are put in a random
    order with the CRASH-2 seed and taken until at least 30 % of patients are in
    the held-out set; those form the external region.
    """
    num = lambda v: pd.to_numeric(raw[v], errors="coerce")
    days = num("daysrandtodeath")
    d["site"] = num("isiteid").astype("Int64")
    flow = {"randomised": int(len(d))}
    known = num("istatus").notna() | days.notna()
    y = pd.Series(np.where(days.notna() & (days <= 28), 1.0, 0.0))
    y[~known.to_numpy()] = np.nan
    flow["deaths_after_day_28_coded_0"] = int((days > 28).sum())
    flow["no_outcome_recorded"] = int((~known).sum())
    d["y"] = y.to_numpy()
    d = d.loc[d["y"].notna()].reset_index(drop=True)
    flow["outcome_known"] = int(len(d))
    d["y"] = d["y"].astype(int)

    # hospital-held-out split, fixed before any model is fitted
    rng = np.random.default_rng(SPLIT_SEED["crash2"])
    sizes = d.groupby("site").size()
    order = rng.permutation(sizes.index.to_numpy())
    held, total = [], 0
    for s in order:
        if total >= CRASH2_EXT_SHARE * len(d):
            break
        held.append(s)
        total += int(sizes[s])
    held = set(int(s) for s in held)
    is_dev = lambda df: ~df["site"].astype(int).isin(held).to_numpy()

    predictors = CRASH2_CONT + CRASH2_BIN
    types = {**{v: "continuous" for v in CRASH2_CONT}, **{v: "binary" for v in CRASH2_BIN}}
    all_rows = add_split(d.copy(), is_dev(d), SPLIT_SEED["crash2"], "split")
    d, dropped = complete_case(d, predictors)
    add_split(d, is_dev(d), SPLIT_SEED["crash2"], "split")
    flow["complete_case"] = int(len(d))
    flow["dropped_missing_predictor"] = dropped
    flow["held_out_hospitals"] = len(held)
    flow["hospitals"] = int(sizes.size)
    meta = {
        "key": "crash2_death28", "source": "CRASH-2 trial data, freeBIRD anonymised file (doi:10.17037/DATA.10)",
        "outcome": "Death within 28 days of randomisation (daysrandtodeath <= 28); patients with neither a "
                   "discharge status nor a recorded death excluded",
        "predictors": predictors, "predictor_types": types, "levels": {}, "flow": flow,
        "missing_policy": "complete case on predictors (as Perel et al. 2012); missing-kept table in crash2_death28_all.parquet",
        "splits_definition": {"split": f"hospitals held out (deviation D-01): {len(held)} of {sizes.size} hospitals, "
                                       f"taken in random order (seed {SPLIT_SEED['crash2']}) until at least "
                                       f"{int(CRASH2_EXT_SHARE * 100)} % of patients = external; "
                                       f"{N_RECAL} random external rows = ext_recal"},
        "held_out_sites": sorted(held),
        "raw_sha256": hashes,
        "notes": ["The freeBIRD file has no dates, no country and no treatment code: the pre-registered "
                  "temporal split, the income-region split and the treatment adjustment of Perel et al. "
                  "cannot be reproduced.",
                  "Predictors follow the Perel et al. 2012 final model (GCS, age, SBP, HR, RR, "
                  "time since injury, injury type) plus sex; capillary refill excluded as in that model.",
                  "Time since injury is kept continuous (Perel et al. dichotomised it at 3 h)."],
    }
    save_table("crash2_death28_all", all_rows, {**meta, "key": "crash2_death28_all", "missing_policy": "missing values kept"}, out_dir, force)
    return {"crash2_death28": save_table("crash2_death28", d, meta, out_dir, force)}


# ---------------------------------------------------------------------------
# Sepsis survival minimal clinical records (UCI 827)
# ---------------------------------------------------------------------------

SEPSIS_PREFIX = "s41598-020-73558-3_sepsis_survival_"


def _read_sepsis(raw_dir, which):
    p = Path(raw_dir) / "uci_sepsis" / f"{SEPSIS_PREFIX}{which}_cohort.csv"
    t = pd.read_csv(p)
    t.columns = ["age", "female", "episode", "alive"]
    return p, t


def build_sepsis(raw_dir=RAW, out_dir=None, force=False):
    p1 = Path(raw_dir) / "uci_sepsis" / f"{SEPSIS_PREFIX}primary_cohort.csv"
    if not p1.exists():
        print("  Sepsis: files missing in data/raw/uci_sepsis; skipped")
        return None
    p1, prim = _read_sepsis(raw_dir, "primary")
    p2, study = _read_sepsis(raw_dir, "study")
    p3, kor = _read_sepsis(raw_dir, "validation")
    hashes = check_raw([p1, p2, p3]) if Path(raw_dir) == RAW else {}
    if (len(prim), len(study), len(kor)) != (110204, 19051, 137):
        raise RuntimeError(f"Sepsis: unexpected sizes {len(prim)}, {len(study)}, {len(kor)}")

    # The study cohort (Sepsis-3 admissions) is a subset of the primary cohort,
    # and there is no row identifier. With only four columns, identical rows are
    # interchangeable, so removing the study rows pattern by pattern (multiset
    # difference) gives exactly the non-Sepsis-3 admissions.
    cols = ["age", "female", "episode", "alive"]
    cp = prim.value_counts(cols)
    cs = study.value_counts(cols).reindex(cp.index, fill_value=0)
    not_in_primary = int(study.value_counts(cols).index.difference(cp.index).size)
    if not_in_primary or (cs > cp).any():
        raise RuntimeError("Sepsis: the study cohort is not a sub-multiset of the primary cohort")
    rest = (cp - cs)
    rest = rest[rest > 0]
    rest_rows = rest.index.to_frame(index=False).loc[np.repeat(np.arange(len(rest)), rest.to_numpy())]

    def finish(t, cohort):
        t = t.reset_index(drop=True).copy()
        t["y"] = (1 - t["alive"]).astype(int)       # event = death
        t["cohort"] = cohort
        return t.drop(columns="alive")

    a = finish(rest_rows, "primary_not_sepsis3")
    b = finish(study, "study_sepsis3")
    d = pd.concat([a, b], ignore_index=True)
    d.insert(0, "id", np.arange(len(d)))
    add_split(d, (d["cohort"] == "primary_not_sepsis3").to_numpy(), SPLIT_SEED["sepsis"], "split")
    rng = np.random.default_rng(SPLIT_SEED["sepsis"] + 1)
    add_split(d, rng.random(len(d)) < 0.5, SPLIT_SEED["sepsis"] + 2, "split_random")
    k = finish(kor, "korea_validation")
    k.insert(0, "id", np.arange(len(k)))

    predictors = ["age", "female", "episode"]
    meta = {
        "key": "sepsis_death", "source": "UCI 827, Chicco & Jurman 2020 (data from Knoop et al. 2017)",
        "outcome": "Death (hospital_outcome_1alive_0dead = 0), about 9 days after admission",
        "predictors": predictors,
        "predictor_types": {"age": "continuous", "female": "binary", "episode": "continuous"},
        "levels": {}, "missing_policy": "no missing values",
        "flow": {"primary_admissions": int(len(prim)), "study_admissions_sepsis3": int(len(study)),
                 "primary_minus_study": int(len(a)), "korea_validation": int(len(kor))},
        "splits_definition": {
            "split": "case-mix (definition) shift: dev_pool = primary-cohort admissions not meeting Sepsis-3; "
                     f"external = Sepsis-3 study cohort; {N_RECAL} random external rows = ext_recal",
            "split_random": "random halves of all 110,204 admissions (sensitivity)"},
        "raw_sha256": hashes,
        "notes": ["Rows are admissions, not patients (110,204 admissions of 84,811 people); "
                  "no identifier, so repeated admissions cannot be clustered.",
                  "Only 3 predictors with few distinct patterns: observed risk per pattern in the "
                  "external set is a near-exact reference curve.",
                  "Ages 0-100 (children included, as in the source).",
                  "Korean validation cohort (n = 137) is saved as sepsis_korea but is too small for calibration."],
    }
    out = {"sepsis_death": save_table("sepsis_death", d, meta, out_dir, force)}
    k["split"] = "ext_eval"
    save_table("sepsis_korea", k, {**meta, "key": "sepsis_korea", "splits_definition": {"split": "all external"}},
               out_dir, force)
    return out


# ---------------------------------------------------------------------------
# PhysioNet/Computing in Cardiology Challenge 2012
# ---------------------------------------------------------------------------

# plausible ranges; values outside are recording errors and set to missing
PN_RANGE = {"HR": (1, 300), "SysABP": (1, 300), "NISysABP": (1, 300), "Temp": (25, 45),
            "RespRate": (1, 80), "GCS": (3, 15), "Urine": (0, 10000), "BUN": (0, 300),
            "Creatinine": (0.05, 30), "HCT": (5, 75), "WBC": (0.05, 200), "Glucose": (10, 2000),
            "K": (1, 12), "Na": (90, 190), "HCO3": (2, 60), "Bilirubin": (0, 60)}
# feature: (parameter(s), summary). SAPS-I variables (Le Gall et al. 1984) over the
# first 48 h, two-sided extremes for variables with a U-shaped risk, plus
# creatinine and bilirubin.
PN_FEATURES = {
    "hr_min": (["HR"], "min"), "hr_max": (["HR"], "max"),
    "sbp_min": (["SysABP", "NISysABP"], "min"), "sbp_max": (["SysABP", "NISysABP"], "max"),
    "temp_min": (["Temp"], "min"), "temp_max": (["Temp"], "max"),
    "rr_min": (["RespRate"], "min"), "rr_max": (["RespRate"], "max"),
    "gcs_min": (["GCS"], "min"), "urine_sum": (["Urine"], "sum"),
    "bun_max": (["BUN"], "max"), "creat_max": (["Creatinine"], "max"),
    "hct_min": (["HCT"], "min"), "wbc_min": (["WBC"], "min"), "wbc_max": (["WBC"], "max"),
    "glucose_min": (["Glucose"], "min"), "glucose_max": (["Glucose"], "max"),
    "k_min": (["K"], "min"), "k_max": (["K"], "max"),
    "na_min": (["Na"], "min"), "na_max": (["Na"], "max"),
    "hco3_min": (["HCO3"], "min"), "bili_max": (["Bilirubin"], "max"),
}
PN_ICU = {1: "CCU", 2: "CSRU", 3: "MICU", 4: "SICU"}


def _pn_record(text):
    t = pd.read_csv(io.StringIO(text))
    desc = t[t["Time"] == "00:00"].drop_duplicates("Parameter").set_index("Parameter")["Value"]
    ts = t[~t["Parameter"].isin(["RecordID", "Age", "Gender", "Height", "ICUType"])].copy()
    ts = ts[~((ts["Time"] == "00:00") & (ts["Parameter"] == "Weight"))]
    ts["Value"] = pd.to_numeric(ts["Value"], errors="coerce")
    for prm, (lo, hi) in PN_RANGE.items():
        m = ts["Parameter"] == prm
        ts.loc[m & ~ts["Value"].between(lo, hi), "Value"] = np.nan
    ts = ts.dropna(subset=["Value"])
    row = {"RecordID": int(desc["RecordID"])}
    age = float(desc.get("Age", -1))
    sex = float(desc.get("Gender", -1))
    row["age"] = age if age >= 0 else np.nan
    row["female"] = {0.0: 1.0, 1.0: 0.0}.get(sex, np.nan)   # challenge coding: 0 = female, 1 = male
    row["icu"] = PN_ICU.get(int(desc.get("ICUType", -1)), None)
    for name, (prms, how) in PN_FEATURES.items():
        v = ts.loc[ts["Parameter"].isin(prms), "Value"]
        row[name] = (getattr(v, how)() if len(v) else np.nan)
    mv = ts.loc[ts["Parameter"] == "MechVent", "Value"]
    row["mechvent"] = float((mv > 0).any())    # only recorded when ventilated
    return row


def build_physionet(raw_dir=RAW, out_dir=None, force=False):
    d0 = Path(raw_dir) / "physionet2012"
    tars = [d0 / f"set-{s}.tar.gz" for s in "abc"]
    outs = [d0 / f"Outcomes-{s}.txt" for s in "abc"]
    if not all(p.exists() for p in tars + outs):
        print("  PhysioNet 2012: files missing in data/raw/physionet2012; skipped")
        return None
    hashes = check_raw(tars + outs) if Path(raw_dir) == RAW else {}

    rows = []
    for s, tp in zip("abc", tars):
        # read straight from the archive so the result does not depend on an extracted copy
        with tarfile.open(tp, "r:gz") as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.endswith(".txt"):
                    r = _pn_record(tf.extractfile(m).read().decode("utf8"))
                    r["set"] = s
                    rows.append(r)
    feats = pd.DataFrame(rows)
    oc = pd.concat([pd.read_csv(p) for p in outs], ignore_index=True)
    if len(feats) != 12000 or len(oc) != 12000:
        raise RuntimeError(f"PhysioNet: expected 12000 records, got {len(feats)} and {len(oc)}")
    d = feats.merge(oc[["RecordID", "In-hospital_death", "SAPS-I", "SOFA"]], on="RecordID", how="inner",
                    validate="one_to_one")
    flow = {"icu_stays": int(len(d))}
    d = d.rename(columns={"In-hospital_death": "y", "SAPS-I": "saps1", "SOFA": "sofa", "RecordID": "id"})
    d = d.loc[d["y"].isin([0, 1])].reset_index(drop=True)
    d["y"] = d["y"].astype(int)
    flow["outcome_known"] = int(len(d))
    d["saps1"] = d["saps1"].where(d["saps1"] >= 0)
    d["sofa"] = d["sofa"].where(d["sofa"] >= 0)
    d = d.sort_values("id").reset_index(drop=True)

    cont = ["age"] + list(PN_FEATURES)
    predictors = cont + ["female", "mechvent", "icu"]
    types = {**{v: "continuous" for v in cont}, "female": "binary", "mechvent": "binary", "icu": "categorical"}
    add_split(d, d["set"].isin(["a", "b"]).to_numpy(), SPLIT_SEED["physionet"], "split")
    add_split(d, (d["icu"] != "MICU").to_numpy(), SPLIT_SEED["physionet"] + 1, "split_icu")
    meta = {
        "key": "physionet_ihd", "source": "PhysioNet/CinC Challenge 2012 v1.0.0 (sets A, B, C; MIMIC-II)",
        "outcome": "In-hospital death (In-hospital_death = 1)",
        "predictors": predictors, "predictor_types": types, "levels": {"icu": ["MICU", "CCU", "CSRU", "SICU"]},
        "flow": flow,
        "missing_policy": "missing kept as NaN; impute_simple() fitted on development rows "
                          "(median + indicator if >= 5 % missing), same matrix for every arm",
        "splits_definition": {
            "split": f"held-out set: dev_pool = sets A + B (8,000); external = set C; {N_RECAL} random set-C rows = ext_recal",
            "split_icu": "case-mix shift: dev_pool = CCU, CSRU, SICU; external = medical ICU (MICU)"},
        "raw_sha256": hashes,
        "notes": ["Features summarise the first 48 h of the ICU stay (SAPS-I variables, min/max where risk "
                  "is U-shaped, urine summed, creatinine and bilirubin added).",
                  "Implausible values set to missing using the ranges in PN_RANGE.",
                  "SAPS-I and SOFA from the outcome files kept as columns (reference scores), never as predictors.",
                  "Sets A, B, C are random thirds of one source, so 'split' is a held-out, not an external, validation."],
    }
    return {"physionet_ihd": save_table("physionet_ihd", d, meta, out_dir, force)}


# ---------------------------------------------------------------------------
# plasmode tier: real covariates, simulated outcomes from a frozen truth model
# ---------------------------------------------------------------------------

# The truth adds to a realistic main-effect logit (coefficients fitted once on
# the real outcome) one smooth non-linear term and one interaction, each of a
# fixed, pre-registered size on the logit scale. The intercept is solved so that
# the mean true risk hits each target prevalence.
PLASMODE_DEFAULT = {
    "ist_dd6": {"nonlinear": "rsbp", "interaction": ["age", "rconsc!=F"], "gamma_nl": 0.4, "gamma_int": 0.6},
    "ist_death14": {"nonlinear": "rsbp", "interaction": ["age", "rconsc!=F"], "gamma_nl": 0.4, "gamma_int": 0.6},
    "crash2_death28": {"nonlinear": "sbp", "interaction": ["gcs", "penetrating"], "gamma_nl": 0.4, "gamma_int": 0.6},
    "physionet_ihd": {"nonlinear": "hr_max", "interaction": ["age", "mechvent"], "gamma_nl": 0.4, "gamma_int": 0.6},
}
PLASMODE_PREVALENCE = [0.10, 0.30]


def _logit_fit(X, y, iters=50):
    """Plain IRLS logistic regression with a tiny ridge for stability."""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X1.shape[1])
    for _ in range(iters):
        eta = X1 @ beta
        p = 1 / (1 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-9, None)
        H = X1.T @ (X1 * w[:, None]) + 1e-6 * np.eye(X1.shape[1])
        step = np.linalg.solve(H, X1.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def _term(df, name):
    if "!=" in name:
        v, lev = name.split("!=")
        return (df[v] != lev).astype(float).to_numpy()
    x = df[name].astype(float).to_numpy()
    return (x - np.nanmean(x)) / np.nanstd(x)


def plasmode_truth(df, meta, spec):
    """True risk p* for every row of a complete-case table, one array per target prevalence."""
    X = design_matrix(df, meta).to_numpy()
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    beta = _logit_fit(Z, df["y"].to_numpy().astype(float))
    lp0 = Z @ beta[1:]
    z = _term(df, spec["nonlinear"])
    nl = z ** 2 - 1                     # U-shape, mean zero
    a, b = spec["interaction"]
    inter = _term(df, a) * _term(df, b)
    lp = lp0 + spec["gamma_nl"] * nl + spec["gamma_int"] * inter
    out = {}
    for prev in PLASMODE_PREVALENCE:
        lo, hi = -20.0, 20.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if np.mean(1 / (1 + np.exp(-(mid + lp)))) < prev:
                lo = mid
            else:
                hi = mid
        out[prev] = (mid, 1 / (1 + np.exp(-(mid + lp))))
    return beta, out


def build_plasmode(key, out_dir=None):
    out_dir = Path(out_dir or CLEAN)
    meta = json.loads((out_dir / f"{key}.meta.json").read_text(encoding="utf8"))
    df = pd.read_parquet(out_dir / f"{key}.parquet")
    spec = dict(PLASMODE_DEFAULT[key])
    if df[meta["predictors"]].isna().any().any():
        # PhysioNet: the truth model needs every covariate, so missing values
        # are filled with medians of the whole table. The covariates the arms
        # see are unchanged; only p* is built from the filled copy.
        df, _, _ = impute_simple(df, [], meta["predictors"], indicator_min=2.0)
        spec["covariates_for_truth"] = "median-filled copy of the table"
    beta, truth = plasmode_truth(df, meta, spec)
    res = {"key": key, "source_fingerprint": meta["fingerprint"], "spec": spec,
           "main_effect_beta_standardised": [float(b) for b in beta],
           "design_columns": list(design_matrix(df, meta).columns), "targets": {}}
    for prev, (alpha, p) in truth.items():
        tag = f"{int(round(prev * 100)):02d}"
        np.save(out_dir / f"plasmode_{key}_p{tag}.npy", p)
        res["targets"][tag] = {"intercept_shift": float(alpha), "mean_p": float(p.mean()),
                               "p_sha256": hashlib.sha256(np.round(p, 10).tobytes()).hexdigest()}
    (out_dir / f"plasmode_{key}.json").write_text(json.dumps(res, indent=2), encoding="utf8")
    print(f"  plasmode {key}: " + ", ".join(f"{t}% -> mean p* {v['mean_p']:.3f}" for t, v in res["targets"].items()))
    return res


def plasmode_outcomes(key, prevalence_tag, rep, out_dir=None):
    """Simulated outcomes for replicate `rep`; checks p* against its stored hash first."""
    out_dir = Path(out_dir or CLEAN)
    info = json.loads((out_dir / f"plasmode_{key}.json").read_text(encoding="utf8"))
    p = np.load(out_dir / f"plasmode_{key}_p{prevalence_tag}.npy")
    if hashlib.sha256(np.round(p, 10).tobytes()).hexdigest() != info["targets"][prevalence_tag]["p_sha256"]:
        raise RuntimeError(f"plasmode {key}: stored p* does not match its fingerprint")
    rng = np.random.default_rng(seed_for("plasmode", key, prevalence_tag, rep))
    return p, rng.binomial(1, p)


# ---------------------------------------------------------------------------

BUILDERS = {"ist": build_ist, "crash2": build_crash2, "sepsis": build_sepsis, "physionet": build_physionet}


def load(key, out_dir=None):
    """Read a clean table and its meta, checking the stored fingerprint."""
    out_dir = Path(out_dir or CLEAN)
    df = pd.read_parquet(out_dir / f"{key}.parquet")
    meta = json.loads((out_dir / f"{key}.meta.json").read_text(encoding="utf8"))
    if table_fingerprint(df) != meta["fingerprint"]:
        raise RuntimeError(f"{key}: table fingerprint mismatch")
    return df, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["manifest", "build", "plasmode", "summary"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--raw-dir", default=str(RAW))
    ap.add_argument("--out-dir", default=str(CLEAN))
    ap.add_argument("--force", action="store_true", help="overwrite a table whose fingerprint changed")
    a = ap.parse_args(argv)

    if a.command == "manifest":
        n = update_manifest()
        print(f"manifest: {len(n)} files")
    elif a.command == "build":
        for name in (a.names or list(BUILDERS)):
            print(f"[{name}]")
            BUILDERS[name](Path(a.raw_dir), a.out_dir, a.force)
    elif a.command == "plasmode":
        for key in a.names:
            build_plasmode(key, a.out_dir)
    elif a.command == "summary":
        for p in sorted(Path(a.out_dir).glob("*.meta.json")):
            m = json.loads(p.read_text(encoding="utf8"))
            print(m["key"], m["n"], m["events"], m["event_rate"], json.dumps(m["splits"]))


if __name__ == "__main__":
    sys.exit(main())
