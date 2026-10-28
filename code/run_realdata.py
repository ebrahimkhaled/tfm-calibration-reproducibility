"""Real-data stream (pre-registration Section 10): learning curves, repair and the plasmode tier.

    python run_realdata.py fit --part lc --keys ist_dd6 --ns 200 --reps 1-3 --arms classical --group cpu
    python run_realdata.py fit --part lc --keys all --ns all --reps 1-100 --arms tfm --group gpu --device cuda
    python run_realdata.py fit --part pl --keys all --ns all --reps 1-200 --arms classical,oracle --group cpu
    python run_realdata.py score --keys all --n-workers 8
    python run_realdata.py summarise
    python run_realdata.py status

Parts and settings
    lc  learning curves (10.2) and repair (10.3). For data set <key> and n in {200, 500, 1000, 3000},
        replicate b (file rep = b + 1) fits every requested arm on development subsample b drawn by
        realdata.subsample_dev (seed from key, split, n, b) and predicts the whole external region:
        the 500 local recalibration rows (set "recal") and the evaluation rows (set "eval"). The
        recalibration ladder of FLIC and the TFMs (recal.ladder, m = 100 and 500, the first m rows
        of the local sample) is fitted on the recal rows and applied to the eval rows.
    pl  plasmode tier (10.4). The same, with the outcome replaced by Y* ~ Bernoulli(p*) from the
        frozen truth (realdata.plasmode_outcomes; p* checked against its SHA-256), at mean
        p* = 10 % and 30 %. The development subsample has its own seed (key, prevalence, n, rep).
        The oracle arm (p* itself) is available; there is no ladder.

Files (one per replicate and arm group, so that the GPU and CPU processes never wait for each
other):

    results/realdata/<key>/preds/<setting>/<group>/<rep>.parquet
        setting = <key>__lc_n<n> or <key>__pl<10|30>_n<n>
        set, row (row of the clean table), y, [p_star], x_<predictor> (the matrix the arms saw,
        eval rows), l_<covariate> (localization covariates of 10.3, eval rows), p_<arm>,
        ystar_<arm> (exact null Y* ~ Bernoulli(p_hat) on the eval rows, own seed per arm),
        p_<arm>__<R1|R2|R3>_m<100|500> (ladder, lc only); metadata key "tfmcal" holds seeds,
        fingerprints, arm status, timings and the run history.
    results/realdata/<key>/metrics/<setting>.parquet      (score)
    results/realdata/<key>/gof/<setting>/<rep>.csv.gz      (R/realdata_gof.R)
    analysis/tables/realdata/*.parquet|csv                 (summarise; CRASH-2 apart, see below)

CRASH-2 (freeBIRD Data Sharing Agreement, DEVIATIONS D-02): per-patient predictions stay under
results/realdata/crash2_death28/ (marked DO_NOT_DEPOSIT), no per-hospital result is computed, and
its aggregate tables go to analysis/tables/realdata/crash2_hold/ until the Chief Investigators
allow publication.

Resume: a replicate whose requested arms are all recorded (ok, degenerate, failed or not run) in
any group file is skipped; a lock file per group file keeps two processes off the same file; every
seed is derived from the task itself, so results do not depend on the number of workers or on the
order of the tasks.
"""

from __future__ import annotations

import os
import sys

# one BLAS/OpenMP thread per worker unless asked otherwise (logs/main/INCIDENTS.md, 2026-10-14)
_THREADS = "1"
if "--threads" in sys.argv:
    _THREADS = sys.argv[sys.argv.index("--threads") + 1]
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = _THREADS

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

CODE_DIR = Path(__file__).resolve().parent
REBUILD = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import metrics  # noqa: E402
import models  # noqa: E402
import realdata as rd  # noqa: E402
from run import META_KEY, Lock, add_ladder, jsonable, read_meta, write_atomic  # noqa: E402

# REALDATA_SMOKE=1 sends everything to results/smoke/realdata and analysis/tables/smoke_realdata
_SMOKE = os.environ.get("REALDATA_SMOKE") == "1"
RESULTS = REBUILD / "results" / ("smoke/realdata" if _SMOKE else "realdata")
TABLES = REBUILD / "analysis" / "tables" / ("smoke_realdata" if _SMOKE else "realdata")
LOG_DIR = REBUILD / "logs" / "realdata"
SCHEMA_VERSION = 1

KEYS = ["ist_death14", "ist_dd6", "crash2_death28", "sepsis_death", "physionet_ihd"]
PLASMODE_KEYS = ["ist_dd6", "crash2_death28", "physionet_ihd"]
RESTRICTED = {"crash2_death28"}          # freeBIRD agreement: aggregate results only, held back
NS = [200, 500, 1000, 3000]
PREVS = ["10", "30"]
SPLIT_COL = "split"                      # the primary split of every data set (10.1, D-01)
# part "lct": the IST temporal sensitivity split of 10.1 (randomised before 1994 = development);
# its development pools hold about 3,240 complete cases, below the 3,500 rule (DEVIATIONS D-04, D-05)
PART_SPLIT = {"lc": "split", "pl": "split", "lct": "split_temporal"}
TEMPORAL_KEYS = ["ist_death14", "ist_dd6"]
# side arms dropped from all real data by the author on 2026-10-23 (DEVIATIONS D-05): never fitted,
# scored or tested here; records written before the decision stay in the files and are ignored
DROPPED_ARMS = ("tabpfn_t1", "tabicl_t1", "tabpfn_legacy")

# every arm of 5.2 except the oracle (10.2); the oracle exists only in the plasmode tier
LC_ARMS = ["lr", "flic", "ridge", "gam", "lgbm", "tabpfn", "tabicl", "mitra",
           "tabpfn_t1", "tabicl_t1", "tabpfn_legacy"]
ARM_SETS = {
    "classical": ["lr", "flic", "ridge", "gam", "lgbm"],
    "tfm": ["tabpfn", "tabicl", "mitra"],
    "side": ["tabpfn_t1", "tabicl_t1", "tabpfn_legacy"],
    "tfm_all": ["tabpfn", "tabicl", "mitra", "tabpfn_t1", "tabicl_t1", "tabpfn_legacy"],
    "all": LC_ARMS,
}
LADDER_ARMS = ["flic", "tabpfn", "tabicl", "mitra"]          # 5.5

# (arm, key, n) recorded as not run, without a fit. Mitra-v2 at n = 3000 on PhysioNet (31 columns)
# and IST (20 columns) is the simulation's T2 case (30 covariates, n = 3000): on the 6 GB card
# AutoGluon cuts its context to 1,024 rows even when only 500 rows are predicted at a time (smoke
# test 2026-10-16), and a cut context is never accepted (pre-registration 5.2), so the T2 rule
# applies. (Mitra-v2 fits at n = 3000 on CRASH-2 and Sepsis, with 8 and 3 columns.)
_T2 = "does not fit in 6 GB GPU memory (the T2 rule, pre-registration 5.2)"
NOT_RUN = {("mitra", "physionet_ihd", 3000): _T2, ("mitra", "ist_dd6", 3000): _T2, ("mitra", "ist_death14", 3000): _T2}
# TabPFN-3.5 (both temperatures) at PhysioNet n = 3000 does not fit the card either (out of memory
# at 500 predicted rows); like T2 at n = 3000 in the simulation it runs on CPU workers
# (code/run_realdata.sh, stream tfmcpu), and the device is recorded per replicate.
CPU_TABPFN = {("physionet_ihd", 3000)}

CODE_FILES = ["run_realdata.py", "realdata.py", "models.py", "recal.py", "metrics.py", "fit_r_arms.R"]


# ----------------------------------------------------------------------------- small helpers

def code_hash() -> str:
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update((CODE_DIR / f).read_bytes())
    return h.hexdigest()[:16]


def task_seed(*parts) -> int:
    """31-bit seed from a readable key (R's set.seed takes at most 2^31 - 1)."""
    return rd.seed_for("realdata", *parts) % (2 ** 31 - 1)


def setting_name(key: str, part: str, n: int, tag: str | None = None) -> str:
    if part in ("lc", "lct"):
        return f"{key}__{part}_n{n}"
    return f"{key}__pl{tag}_n{n}"


def parse_setting(s: str) -> dict:
    key, rest = s.split("__")
    kind, n = rest.split("_n")
    if kind in ("lc", "lct"):
        return dict(key=key, part=kind, tag=None, n=int(n))
    return dict(key=key, part="pl", tag=kind[2:], n=int(n))


def rep_path(key: str, setting: str, group: str, rep: int) -> Path:
    return RESULTS / key / "preds" / setting / group / f"{rep:04d}.parquet"


def sha(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def mark_restricted():
    """A visible marker in every CRASH-2 folder: per-patient predictions, never for deposit."""
    for key in RESTRICTED:
        d = RESULTS / key
        d.mkdir(parents=True, exist_ok=True)
        f = d / "DO_NOT_DEPOSIT.txt"
        if not f.exists():
            f.write_text(
                "CRASH-2 (freeBIRD Data Sharing Agreement; prereg/DEVIATIONS.md D-02).\n"
                "Everything under this folder is derived from CRASH-2 patient rows (per-replicate\n"
                "predictions, outcomes, covariates and test results). It must not be deposited,\n"
                "shared or transmitted. Only aggregate results may leave this machine, and only\n"
                "after the CRASH-2 Chief Investigators have given written permission.\n", encoding="utf8")
    ex = RESULTS / "DEPOSIT_EXCLUDE.txt"
    if not ex.exists():
        ex.write_text("# Paths that must never enter a deposit or any shared archive (freeBIRD agreement)\n"
                      "results/realdata/crash2_death28/\n"
                      "analysis/tables/realdata/crash2_hold/\n"
                      "data/raw/crash2/\n"
                      "data/clean/crash2_death28*\n"
                      "data/clean/plasmode_crash2_death28*\n", encoding="utf8")


# ----------------------------------------------------------------------------- data of a replicate

_TABLES: dict = {}


def load_key(key: str):
    if key not in _TABLES:
        df, meta = rd.load(key)
        _TABLES[key] = (df, meta)
    return _TABLES[key]


def localization_covariates(key: str, ext: pd.DataFrame, df_all: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Covariates for localize.external (10.3): age band, sex, region where present, and a severity
    score. Built on the external region only; none of them is a model input in this form.

      age band   quartiles of age in the external region (integer code, categorical)
      (a missing age, sex or SAPS-I takes the median of the external region)
      sex        female (0/1)
      region     IST only: four groups of the external countries (categorical)
      severity   IST conscious state (categorical F/D/U); CRASH-2 GCS; PhysioNet SAPS-I (missing
                 values filled with the external median); Sepsis has no severity score, so the
                 septic episode number stands in for it
    Returns the frame (columns l_*) and the list of its categorical columns."""
    out = pd.DataFrame(index=ext.index)
    # a missing value (PhysioNet: a few stays without age or sex) takes the external median
    med = lambda v: ext[v].astype(float).fillna(ext[v].astype(float).median()).to_numpy()
    age = med("age")
    cuts = np.quantile(age, [0.25, 0.5, 0.75])
    out["l_ageband"] = np.searchsorted(cuts, age, side="right").astype(float)
    out["l_female"] = med("female")
    cat = ["l_ageband"]
    if key.startswith("ist"):
        out["l_region"] = ext["country"].map(IST_REGION).astype(float)
        if out["l_region"].isna().any():
            raise ValueError("IST: a country without a region")
        out["l_rconsc"] = ext["rconsc"].map({"F": 0, "D": 1, "U": 2}).astype(float)
        cat += ["l_region", "l_rconsc"]
    elif key.startswith("crash2"):
        out["l_gcs"] = ext["gcs"].to_numpy(dtype=float)
    elif key.startswith("physionet"):
        s = ext["saps1"].astype(float)
        out["l_saps1"] = s.fillna(s.median()).to_numpy()
    elif key.startswith("sepsis"):
        out["l_episode"] = ext["episode"].to_numpy(dtype=float)
    return out, cat


# IST external countries in four regions (CNTRYNUM of IST_variables.csv). The UK is the development
# region. Northern Europe as in the pre-registered fallback pool (Ireland, Norway, Sweden, Denmark,
# Finland, the Netherlands); the rest of Western and Southern Europe; Central and Eastern Europe
# with Turkey; every country outside Europe.
IST_REGION = {
    9: 0, 17: 0, 24: 0, 8: 0, 10: 0, 15: 0,                          # EIRE NORW SWED DENM FINL NETH
    2: 1, 3: 1, 11: 1, 14: 1, 19: 1, 22: 1, 25: 1, 31: 1,            # AUST BELG FRAN ITAL PORT SPAI SWIT GREE
    7: 2, 18: 2, 20: 2, 33: 2, 36: 2, 44: 2, 35: 2,                  # CZEC POLA SLOV ROMA HUNG SLOK TURK
    1: 3, 5: 3, 6: 3, 13: 3, 16: 3, 21: 3, 23: 3, 28: 3, 29: 3,      # AUSL CANA CHIL ISRA NEW SOUT SRI USA ARGE
    30: 3, 34: 3, 37: 3, 38: 3, 42: 3,                               # HONG SING INDI JAPA BRAS
    27: 4,                                                           # UK: external only in the temporal split
}


def dev_rows(key: str, part: str, tag: str | None, n: int, rep: int, df: pd.DataFrame) -> np.ndarray:
    if part in ("lc", "lct"):
        return rd.subsample_dev(df, key, n, rep - 1, col=PART_SPLIT[part])   # b = rep - 1 (DATA_REPORT 5)
    pool = np.flatnonzero(df[SPLIT_COL].to_numpy() == "dev_pool")
    rng = np.random.default_rng(rd.seed_for("plasmode_dev", key, SPLIT_COL, tag, n, rep))
    return np.sort(rng.choice(pool, size=n, replace=False))


def build_replicate(key: str, part: str, tag: str | None, n: int, rep: int) -> dict:
    """Everything the arms need for one replicate, plus its fingerprints."""
    df, meta = load_key(key)
    dev_idx = dev_rows(key, part, tag, n, rep, df)
    split = df[PART_SPLIT[part]].to_numpy()
    # evaluation rows first, then the 500 local recalibration rows (table order inside each)
    ev_idx = np.flatnonzero(split == "ext_eval")
    rc_idx = np.flatnonzero(split == "ext_recal")
    ext_idx = np.r_[ev_idx, rc_idx]
    if part in ("lc", "lct"):
        y_all = df["y"].to_numpy().astype(np.int8)
        p_star = None
        p_sha = None
    else:
        p_star_all, y_star = rd.plasmode_outcomes(key, tag, rep)
        y_all = y_star.astype(np.int8)
        p_star = p_star_all
        p_sha = json.loads((rd.CLEAN / f"plasmode_{key}.json").read_text(encoding="utf8"))["targets"][tag]["p_sha256"]

    preds = list(meta["predictors"])
    types = meta["predictor_types"]
    dev = df.iloc[dev_idx][preds].copy()
    ext = df.iloc[ext_idx][preds].copy()
    added = []
    if dev.isna().any().any() or ext.isna().any().any():
        # PhysioNet (10.1): medians and missing-value indicators learned on the development rows
        dev, (ext,), added = rd.impute_simple(dev, [ext], preds)
    cont = [v for v in preds if types[v] == "continuous"]
    binv = [v for v in preds if types[v] == "binary"] + added
    cat = [v for v in preds if types[v] == "categorical"]
    levels = {v: list(meta["levels"][v]) for v in cat}
    columns = cont + binv + cat

    def matrix(t):
        cols = []
        for v in columns:
            if v in cat:
                code = t[v].map({lev: i for i, lev in enumerate(levels[v])})
                if code.isna().any():
                    raise ValueError(f"{key}: unknown level of {v}")
                cols.append(code.to_numpy(dtype=float))
            else:
                cols.append(t[v].to_numpy(dtype=float))
        return np.column_stack(cols)

    X_dev, X_ext = matrix(dev), matrix(ext)
    y_dev = y_all[dev_idx].astype(int)
    loc, loc_cat = localization_covariates(key, df.iloc[ev_idx], df)
    fp = dict(dev=sha(np.r_[dev_idx.astype(np.int64), y_dev.astype(np.int64)]),
              eval_y=sha(y_all[ev_idx]), recal_y=sha(y_all[rc_idx]),
              table=meta["fingerprint"][:16], p_star=p_sha)
    return dict(key=key, part=part, tag=tag, n=n, rep=rep, dev_idx=dev_idx, ev_idx=ev_idx,
                rc_idx=rc_idx, y_all=y_all, p_star=p_star, X_dev=X_dev, y_dev=y_dev, X_ext=X_ext,
                columns=columns, cont=cont, bin=binv, cat=cat,
                cat_levels={v: len(levels[v]) for v in cat}, levels=levels, loc=loc,
                loc_cat=loc_cat, fingerprints=fp)


def base_frame(r: dict) -> pd.DataFrame:
    ne, nr = len(r["ev_idx"]), len(r["rc_idx"])
    df = pd.DataFrame({
        "set": pd.Categorical(["eval"] * ne + ["recal"] * nr, categories=["eval", "recal"]),
        "row": np.r_[r["ev_idx"], r["rc_idx"]].astype(np.int32),
        "y": pd.array(np.r_[r["y_all"][r["ev_idx"]], r["y_all"][r["rc_idx"]]], dtype="Int8"),
    })
    if r["p_star"] is not None:
        df["p_star"] = np.r_[r["p_star"][r["ev_idx"]], r["p_star"][r["rc_idx"]]].astype(np.float64)
    for j, c in enumerate(r["columns"]):
        col = np.full(ne + nr, np.nan, dtype=np.float32)
        col[:ne] = r["X_ext"][:ne, j]
        df[f"x_{c}"] = col
    for c in r["loc"].columns:
        col = np.full(ne + nr, np.nan, dtype=np.float32)
        col[:ne] = r["loc"][c].to_numpy(dtype=np.float32)
        df[c] = col
    return df


# ----------------------------------------------------------------------------- fitting

def arm_seeds(setting: str, rep: int) -> dict:
    return {a: task_seed(setting, rep, a) for a in models.ARM_CODES}


def fit_arms(arms: list[str], r: dict, setting: str, device: str, threads: int) -> dict:
    """Fit the arms on the development rows and predict every distinct external covariate row
    once. The models are deterministic functions of a row once fitted (a test row attends only to
    the training rows in the TFMs), so predicting each distinct row once and copying the value to
    its duplicates gives the same predictions; the Sepsis region (18,551 + 500 rows, fewer than
    1,000 covariate patterns) needs it to keep the TFM fits affordable."""
    uniq, inv = np.unique(r["X_ext"], axis=0, return_inverse=True)
    inv = inv.ravel()
    ctx = models.FitContext(
        X_dev=r["X_dev"], y_dev=r["y_dev"], X_pred=uniq,
        p_star_pred=np.zeros(len(uniq)),        # unused: the oracle is taken from p* below
        columns=r["columns"], cont=r["cont"], bin=r["bin"], cat=r["cat"], cat_levels=r["cat_levels"],
        seeds=arm_seeds(setting, r["rep"]), device=device, threads=threads)
    fit_list = [a for a in arms if a != "oracle"]
    r_list = [a for a in fit_list if a in models.R_ARMS]
    tfm_list = [a for a in fit_list if a in models.GPU_ARMS]
    py_list = [a for a in fit_list if a not in models.R_ARMS and a not in models.GPU_ARMS]
    res = models.run_arms(py_list, ctx) if py_list else {}
    for a in tfm_list:
        res.update(fit_chunked(a, ctx))
    if r_list:
        ctx_r, dropped = regression_context(ctx)
        for name, ar in models.run_arms(r_list, ctx_r).items():
            ar.info = dict(ar.info or {}, dropped_constant_columns=dropped)
            res[name] = ar
    out = {}
    for name, ar in res.items():
        if ar.p is not None:
            ar.p = np.asarray(ar.p, dtype=float)[inv]
        ar.info = dict(ar.info or {}, n_pred_unique=int(len(uniq)))
        out[name] = ar
    if "oracle" in arms:
        p = np.r_[r["p_star"][r["ev_idx"]], r["p_star"][r["rc_idx"]]]
        out["oracle"] = models.ArmResult("oracle", p, "ok", {"seconds": 0.0})
    return out


TFM_CHUNK = int(os.environ.get("REALDATA_TFM_CHUNK", "1000000"))   # whole set unless it does not fit
TFM_MIN_CHUNK = 500


# Failures of the machine, not of the model (pre-registration 8.2; INCIDENTS 2026-10-16): the arm is
# left unrecorded and fitted again after a restart.
HOST_MEMORY = ("unable to allocate", "not enough memory", "cannot allocate", "could not allocate",
               "paging file is too small", "os error 1455", "dll load failed", "dynamic link library",
               "winerror 1114",
               "bad_alloc", "memoryerror", "memory allocation", "openblas error", "rscript exit", "r_calloc")
CUDA_RUNTIME = ("cuda error", "cublas_status", "cudnn_status", "invalid python storage",
                "illegal memory access", "cuda driver", "device-side assert")
GPU_FREE_MIN_MB = 4500       # the 6 GB card with only the desktop on it has about 5,400 MB free


def _err(info) -> str:
    return str((info or {}).get("error", "")).lower()


def _is_oom(info: dict) -> bool:
    """CUDA out of memory (or Mitra's context cut after one): the model may not fit the card."""
    e = _err(info)
    return "out of memory" in e or "out-of-memory" in e


def is_infra_error(info: dict) -> bool:
    """Host memory errors always; CUDA runtime errors other than out-of-memory; and anything the
    runner tagged "infra:" (for example a card already held by another process). A CUDA
    out-of-memory error on a free card is handled by chunking and, at the smallest chunk, is the
    arm's recorded failure (the model does not fit)."""
    e = _err(info)
    if e.startswith("infra:"):
        return True
    gpu_oom = _is_oom(info) and ("cuda" in e or "gpu" in e)
    if gpu_oom:
        return False
    return any(p in e for p in HOST_MEMORY) or any(p in e for p in CUDA_RUNTIME)


def gpu_free_mb(device: str):
    """Free memory of the card (whole device, all processes), or None off the GPU."""
    if not device.startswith("cuda"):
        return None
    import torch
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    return free / 2 ** 20, total / 2 ** 20


def fit_chunked(arm: str, ctx: models.FitContext) -> dict:
    """A foundation-model arm, predicting the rows in chunks of at most TFM_CHUNK.

    Each chunk refits the arm with the same seed (the same fit) and predicts its rows; a test row
    attends only to the context, so the predictions do not depend on the chunking (checked in
    tests/test_realdata.py). On a CUDA out-of-memory error the chunk is halved, down to
    TFM_MIN_CHUNK; an error at that size, or any other error, is the arm's recorded failure.
    Mitra's own rule still applies: a fit whose context AutoGluon cut is a failure."""
    import dataclasses
    mem = gpu_free_mb(ctx.device)
    if mem is not None and mem[0] < GPU_FREE_MIN_MB:
        # another process holds the card: an out-of-memory error now would say nothing about the model
        return {arm: models.ArmResult(arm, None, "failed", {
            "error": f"infra: GPU busy before the fit ({mem[0]:.0f} of {mem[1]:.0f} MB free)", "seconds": 0.0})}
    chunk = min(TFM_CHUNK, len(ctx.X_pred))
    tried = []
    while True:
        tried.append(chunk)
        ps, infos, last = [], [], None
        for s in range(0, len(ctx.X_pred), chunk):
            c = dataclasses.replace(ctx, X_pred=ctx.X_pred[s:s + chunk], p_star_pred=ctx.p_star_pred[s:s + chunk])
            res = models.run_arms([arm], c)
            last = res[arm]
            if last.status != "ok":
                break
            ps.append(np.asarray(last.p, dtype=float))
            infos.append(last.info or {})
        if last is not None and last.status == "ok":
            info = dict(infos[0])
            for k in ("seconds", "fit_seconds", "predict_seconds"):
                if all(k in i for i in infos):
                    info[k] = float(sum(i[k] for i in infos))
            pk = [i.get("peak_gpu_mb") for i in infos if i.get("peak_gpu_mb") is not None]
            info.update(peak_gpu_mb=max(pk) if pk else None, predict_chunk=chunk, n_chunks=len(infos),
                        chunks_tried=tried, gpu_free_mb_before=None if mem is None else round(mem[0]))
            return {arm: models.ArmResult(arm, np.concatenate(ps), "ok", info)}
        if _is_oom(last.info) and chunk > TFM_MIN_CHUNK:
            chunk = max(TFM_MIN_CHUNK, chunk // 2)
            continue
        info = dict(last.info or {}, predict_chunk=chunk, chunks_tried=tried,
                    gpu_free_mb_before=None if mem is None else round(mem[0]))
        if _is_oom(info) and arm != "mitra":
            # CUDA out of memory even at the smallest chunk, on a card that was free before the fit.
            # The cells where a model truly does not fit the 6 GB card are known and handled (NOT_RUN,
            # CPU_TABPFN; DEVIATIONS D-03); elsewhere the same arm fits the neighbouring replicates, so
            # this is the card's state after host-memory trouble (INCIDENTS 2026-10-27): refit later.
            info["error"] = "infra: CUDA out of memory at the smallest chunk: " + str(info.get("error", ""))[:300]
        last.info = info
        return {arm: last}


def aliased_free(X: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """Columns to keep: in order, a column is kept when it is not constant and not a linear
    combination of the intercept and the columns already kept (the columns glm would report as
    aliased, NA). Computed on centred and scaled columns, so the scale of a predictor (urine output
    in ml) does not matter."""
    keep = np.zeros(X.shape[1], dtype=bool)
    basis = np.ones((len(X), 1)) / np.sqrt(len(X))
    for j in range(X.shape[1]):
        v = X[:, j] - X[:, j].mean()
        nv = np.linalg.norm(v)
        if nv <= tol * max(1.0, np.abs(X[:, j]).max()) * np.sqrt(len(X)):
            continue                                        # constant
        v = v / nv
        resid = v - basis @ (basis.T @ v)
        nr = np.linalg.norm(resid)
        if nr > np.sqrt(tol):                               # adds a new direction
            keep[j] = True
            basis = np.column_stack([basis, resid / nr])
    return keep


def regression_context(ctx: models.FitContext) -> tuple[models.FitContext, list]:
    """The matrix of the R regression arms (LR-MLE, FLIC, ridge, GAM) on real data.

    Categorical predictors become treatment dummies (first level the reference), which is the
    design R builds from a factor, so a fit with every level present is unchanged. A design column
    that is constant in the development subsample, or a linear combination of the columns before
    it (aliased, as glm would report it), is then dropped, so its coefficient is fixed at 0:
    a level that no development row has (IST stroke subtype "other", 57 of 19,435 patients, is
    absent from about half of the n = 200 subsamples) is predicted like the reference level, and a
    binary predictor with no positive row is left out, and of two identical columns (PhysioNet's
    missing-value indicators of minimum and maximum respiratory rate) the second is left out.
    Without this, glm, logistf and mgcv stop when asked to predict a level they never saw, and
    logistf stops on a singular information matrix where glm would alias the column. TabPFN, TabICL, Mitra and LightGBM keep the
    categorical columns as they are."""
    Xd, Xp, names, binv = [ctx.X_dev[:, ctx.idx(ctx.cont)]], [ctx.X_pred[:, ctx.idx(ctx.cont)]], list(ctx.cont), []
    for c in ctx.bin:
        j = ctx.columns.index(c)
        Xd.append(ctx.X_dev[:, [j]])
        Xp.append(ctx.X_pred[:, [j]])
        names.append(c)
        binv.append(c)
    for c in ctx.cat:
        j = ctx.columns.index(c)
        for lev in range(1, ctx.cat_levels[c]):
            Xd.append((ctx.X_dev[:, [j]] == lev).astype(float))
            Xp.append((ctx.X_pred[:, [j]] == lev).astype(float))
            names.append(f"{c}_{lev}")
            binv.append(f"{c}_{lev}")
    Xd, Xp = np.hstack(Xd), np.hstack(Xp)
    keep = aliased_free(Xd)
    dropped = [nm for nm, k in zip(names, keep) if not k]
    names = [nm for nm, k in zip(names, keep) if k]
    ctx_r = models.FitContext(
        X_dev=Xd[:, keep], y_dev=ctx.y_dev, X_pred=Xp[:, keep], p_star_pred=ctx.p_star_pred,
        columns=names, cont=[c for c in ctx.cont if c in names], bin=[c for c in binv if c in names],
        cat=[], cat_levels={}, seeds=ctx.seeds, device="cpu", threads=ctx.threads)
    return ctx_r, dropped


def add_ystar(df: pd.DataFrame, setting: str, rep: int, name: str, p: np.ndarray):
    """Exact null outcome Y* ~ Bernoulli(p_hat) on the evaluation rows, own seed per arm."""
    is_eval = (df["set"] == "eval").to_numpy()
    rng = np.random.default_rng(task_seed("ystar", setting, rep, name))
    ys = (rng.random(int(is_eval.sum())) < p[is_eval]).astype(np.int8)
    col = pd.array(np.full(len(df), -1, dtype=np.int8), dtype="Int8")
    col[np.flatnonzero(is_eval)] = ys
    col[np.flatnonzero(~is_eval)] = pd.NA
    df[f"ystar_{name}"] = col


def recorded_arms(key: str, setting: str, rep: int) -> dict:
    """Status of every output stored for this replicate, over all group files."""
    seen = {}
    folder = RESULTS / key / "preds" / setting
    if not folder.exists():
        return seen
    for f in folder.glob(f"*/{rep:04d}.parquet"):
        m = read_meta(f)
        if m:
            for a, info in m.get("arms", {}).items():
                seen[a] = dict(status=info.get("status"), group=f.parent.name)
    return seen


def run_task(task: dict) -> dict:
    """Fit the missing arms of one (setting, rep, group) and write its file. Runs in a worker."""
    t0 = time.perf_counter()
    key, part, tag, n, rep, group = (task[k] for k in ("key", "part", "tag", "n", "rep", "group"))
    setting = setting_name(key, part, n, tag)
    path = rep_path(key, setting, group, rep)
    res = dict(setting=setting, rep=rep, group=group, status="skipped", arms_run=[], failures=[])
    retry = {"failed"} if task["rerun_failed"] else set()

    def needed():
        rec = recorded_arms(key, setting, rep)
        out = []
        for a in task["arms"]:
            if task["device"].startswith("cuda") and (key, n) in CPU_TABPFN and a in ("tabpfn", "tabpfn_t1"):
                continue                         # owned by the CPU stream
            outs = models.outputs_of(a)
            if any(o not in rec or (rec[o]["status"] in retry and rec[o]["group"] == group) for o in outs):
                if any(o in rec and rec[o]["group"] != group for o in outs):
                    continue                     # recorded in another group's file: never twice
                out.append(a)
        return out

    if not needed():
        return res
    lock = Lock(path)
    if not lock.acquire():
        res["status"] = "locked"
        return res
    try:
        for tmp in path.parent.glob(f"{path.name}.tmp-*"):
            tmp.unlink(missing_ok=True)          # left by a killed writer of this file
        todo = needed()
        if not todo:
            return res
        r = build_replicate(key, part, tag, n, rep)
        existing = read_meta(path) if path.exists() else None
        if existing is not None:
            if existing["fingerprints"] != jsonable(r["fingerprints"]):
                raise AssertionError(f"{setting} rep {rep}: rebuilt data differ from the stored fingerprints")
            df = pq.read_table(path).to_pandas()
            meta = existing
        else:
            df = base_frame(r)
            meta = dict(schema=SCHEMA_VERSION, key=key, setting=setting, part=part,
                        prevalence=tag, n=n, rep=rep, subsample_b=rep - 1 if part in ("lc", "lct") else None,
                        group=group, split=PART_SPLIT[part], fingerprints=r["fingerprints"],
                        restricted_freebird=key in RESTRICTED,
                        sizes=dict(dev=int(n), eval=len(r["ev_idx"]), recal=len(r["rc_idx"])),
                        events=dict(dev=int(r["y_dev"].sum()),
                                    eval=int(r["y_all"][r["ev_idx"]].sum()),
                                    recal=int(r["y_all"][r["rc_idx"]].sum())),
                        # names as the R layer sees them (x_ prefix, lower case)
                        col_types=dict(cont=[f"x_{c}" for c in r["cont"]], bin=[f"x_{c}" for c in r["bin"]],
                                       cat=[f"x_{c}" for c in r["cat"]], cat_levels=r["levels"]),
                        loc_cat=r["loc_cat"], arms={}, ladder={}, history=[])
        for a in [a for a in todo if (a, key, n) in NOT_RUN]:
            todo.remove(a)
            meta["arms"][a] = dict(status="not_run", error=NOT_RUN[(a, key, n)], seconds=0.0)
            res["arms_run"].append(a)
        fitted = fit_arms(todo, r, setting, task["device"], task["threads"]) if todo else {}
        # an infrastructure error (a CUDA runtime error, a crashed Rscript) is not a model failure
        # (pre-registration 8.2; logs/main/INCIDENTS.md): the arm is left unrecorded so that the
        # restarted process fits it again, and the process stops (a CUDA context may stay broken)
        infra = {k: ar for k, ar in fitted.items() if ar.status != "ok" and is_infra_error(ar.info)}
        for k, ar in infra.items():
            res["infra"] = res.get("infra", []) + [dict(arm=k, error=str(ar.info.get("error", ""))[:300])]
            del fitted[k]
        if infra and not fitted:
            res.update(status="infra_error", seconds=time.perf_counter() - t0)
            return res
        n_eval = len(r["ev_idx"])
        for name, ar in fitted.items():
            info = {**jsonable({k: v for k, v in (ar.info or {}).items() if k != "p"}), "status": ar.status}
            if ar.status == "ok":
                p = np.asarray(ar.p, dtype=np.float64)
                at_bound = (p <= 1e-6) | (p >= 1 - 1e-6)
                info["n_boundary"] = int(at_bound.sum())
                info["n_boundary_eval"] = int(at_bound[:n_eval].sum())
                why = models.degenerate_reason(info, info["n_boundary_eval"]) if name != "oracle" else None
                if why:
                    info.update(status="degenerate", degenerate_reason=why)
                if name != "oracle":
                    df[f"p_{name}"] = p.astype(np.float32)
                add_ystar(df, setting, rep, name, p if name == "oracle" else df[f"p_{name}"].to_numpy(dtype=float))
                for c in [c for c in df.columns if c.startswith(f"p_{name}__")]:
                    del df[c]                     # a refitted arm gets a fresh ladder
                meta["ladder"].pop(name, None)
            else:
                res["failures"].append(dict(arm=name, error=str(info.get("error", ""))[:500]))
            meta["arms"][name] = info
            res["arms_run"].append(name)
        if part in ("lc", "lct"):
            for a in LADDER_ARMS:
                if meta["arms"].get(a, {}).get("status") in models.USABLE_STATUSES and a not in meta["ladder"]:
                    meta["ladder"][a] = add_ladder(df, a)
        meta["history"].append(dict(time=datetime.now().isoformat(timespec="seconds"),
                                    host=socket.gethostname(), pid=os.getpid(), arms=res["arms_run"],
                                    code=task["code_hash"], device=task["device"],
                                    seconds=time.perf_counter() - t0))
        table = pa.Table.from_pandas(df, preserve_index=False)
        md = dict(table.schema.metadata or {})
        md[META_KEY] = json.dumps(jsonable(meta), default=str).encode()
        table = table.replace_schema_metadata(md)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(table, path)
        res.update(status="infra_error" if infra else "written", dev_fp=r["fingerprints"]["dev"],
                   seconds=time.perf_counter() - t0,
                   arm_seconds={k: meta["arms"][k].get("seconds") for k in res["arms_run"]})
        return res
    except Exception as e:  # noqa: BLE001
        res.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        return res
    finally:
        lock.release()


def _worker_init(threads: int):
    models.configure_environment(allow_download=False, threads=threads)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(threads)
    except ImportError:
        pass


def check_tabpfn(arms: list[str]):
    """The main run pins TabPFN-3.5 by version and SHA-256 (run_main.sh); so does this stream."""
    if not any(a in arms for a in ("tabpfn", "tabpfn_t1")):
        return
    ver, path = models.resolve_tabpfn("tabpfn")
    want = os.environ.get("TFMCAL_TABPFN_SHA256")
    if os.environ.get("TFMCAL_TABPFN_VERSION") != "V3_5" or not want:
        raise SystemExit("set TFMCAL_TABPFN_VERSION=V3_5 and TFMCAL_TABPFN_SHA256 (code/run_realdata.sh does)")
    got = models.file_sha256(path)
    if got != want.strip().lower():
        raise SystemExit(f"{path.name}: SHA-256 {got} differs from TFMCAL_TABPFN_SHA256")


# ----------------------------------------------------------------------------- command: fit

def parse_list(s: str, allv: list) -> list:
    if s in ("all", ""):
        return list(allv)
    return [type(allv[0])(x) for x in s.split(",") if x]


def parse_reps(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def parse_arms(s: str) -> list[str]:
    out = []
    for part in s.split(","):
        for a in ARM_SETS.get(part, [part]):
            if a not in models.ARM_CODES or a == "lgbm_raw":
                raise SystemExit(f"unknown arm {a!r}")
            if a not in out:
                out.append(a)
    return out


def cmd_fit(a) -> int:
    mark_restricted()
    keys = parse_list(a.keys, {"pl": PLASMODE_KEYS, "lct": TEMPORAL_KEYS}.get(a.part, KEYS))
    ns = parse_list(a.ns, NS)
    reps = parse_reps(a.reps)
    arms = parse_arms(a.arms)
    if a.part in ("lc", "lct") and "oracle" in arms:
        raise SystemExit("the learning curves have no oracle (10.2)")
    tags = parse_list(a.prevs, PREVS) if a.part == "pl" else [None]
    models.configure_environment(allow_download=False, threads=a.threads)
    check_tabpfn(arms)
    ch = code_hash()
    # replicate blocks first, so that every data set and n moves forward together
    tasks = []
    block = max(1, a.block)
    for b0 in range(0, len(reps), block):
        for key in keys:
            for tag in tags:
                for n in ns:
                    for rep in reps[b0:b0 + block]:
                        tasks.append(dict(key=key, part=a.part, tag=tag, n=n, rep=rep, arms=arms,
                                          group=a.group, device=a.device, threads=a.threads,
                                          rerun_failed=a.rerun_failed, code_hash=ch))
    print(f"{len(tasks)} tasks: part {a.part}, keys {keys}, n {ns}, reps {reps[0]}-{reps[-1]}, "
          f"arms {arms}, group {a.group}, device {a.device}, {a.n_workers} worker(s) x {a.threads} thread(s)",
          flush=True)
    if a.dry_run:
        return 0
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"fit_{a.part}_{a.group}_{stamp}.jsonl"
    counts: dict = {}
    seen_dev: dict = {}

    def handle(res):
        counts[res["status"]] = counts.get(res["status"], 0) + 1
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(jsonable(res), default=str) + "\n")
        if res["status"] == "written":
            # two replicates of one setting with the same development rows = a reset random stream
            k = (res["setting"], res["dev_fp"])
            if k in seen_dev and seen_dev[k] != res["rep"]:
                raise AssertionError(f"identical development rows in {res['setting']} reps {seen_dev[k]} and {res['rep']}")
            seen_dev[k] = res["rep"]
        msg = f"{res['setting']} rep {res['rep']} [{res['group']}]: {res['status']}"
        if res.get("arm_seconds"):
            msg += f" ({res['seconds']:.1f}s; " + ", ".join(
                f"{k} {v:.1f}s" for k, v in res["arm_seconds"].items() if v is not None) + ")"
        if res.get("infra"):
            msg += " INFRA (not recorded): " + "; ".join(f"{x['arm']}: {x['error'][:100]}" for x in res["infra"])
        if res["failures"]:
            msg += " FAILED: " + "; ".join(f"{x['arm']}: {x['error'][:100]}" for x in res["failures"])
        if res["status"] == "error":
            msg += " ERROR " + res["error"]
        print(msg, flush=True)

    stop = {"infra": False}

    def run_batch(batch):
        locked = []
        if a.n_workers <= 1:
            for t in batch:
                res = run_task(t)
                (locked.append(t) if res["status"] == "locked" else handle(res))
                if res["status"] == "infra_error" or (res["status"] == "error" and is_infra_error(res)):
                    stop["infra"] = True
                    break
        else:
            import multiprocessing as mp
            with ProcessPoolExecutor(a.n_workers, mp_context=mp.get_context("spawn"),
                                     initializer=_worker_init, initargs=(a.threads,)) as ex:
                futs = {ex.submit(run_task, t): t for t in batch}
                for fu in as_completed(futs):
                    res = fu.result()
                    (locked.append(futs[fu]) if res["status"] == "locked" else handle(res))
                    if (res["status"] == "infra_error" or (res["status"] == "error" and is_infra_error(res)))                             and not stop["infra"]:
                        stop["infra"] = True
                        for other in futs:
                            other.cancel()
        return locked

    t0 = time.time()
    pending = run_batch(tasks)
    while pending and not stop["infra"] and time.time() - t0 < a.lock_wait * 60 + 1:
        print(f"{len(pending)} task(s) locked by another process; retrying in 30 s", flush=True)
        time.sleep(30)
        pending = run_batch(pending)
    if pending:
        counts["locked"] = len(pending)
    print(f"done in {time.time() - t0:.0f}s: {counts}; log {log_file}", flush=True)
    if stop["infra"]:
        print("stopped after an infrastructure error (exit 75): restart to fit the unrecorded arms", flush=True)
        return 75
    return 1 if counts.get("error") else 0


# ----------------------------------------------------------------------------- command: score

def read_replicate(files: list[Path]) -> tuple[pd.DataFrame, dict]:
    """Merge the group files of one replicate (same rows and outcomes in each)."""
    base, metas = None, []
    for f in sorted(files):
        t = pq.read_table(f)
        m = json.loads(t.schema.metadata[META_KEY])
        d = t.to_pandas()
        metas.append(m)
        if base is None:
            base = d
            continue
        if not (np.array_equal(base["row"].to_numpy(), d["row"].to_numpy())
                and np.array_equal(base["y"].to_numpy(dtype=float), d["y"].to_numpy(dtype=float))):
            raise AssertionError(f"rows or outcomes differ between the group files of {f}")
        if m["fingerprints"] != metas[0]["fingerprints"]:
            raise AssertionError(f"fingerprints differ between the group files of {f}")
        new = [c for c in d.columns if (c.startswith("p_") and c != "p_star") or c.startswith("ystar_")]
        dup = [c for c in new if c in base.columns]
        if dup:
            raise AssertionError(f"{dup} in two group files of {f}")
        base = pd.concat([base, d[new]], axis=1)
    meta = dict(metas[0])
    meta["arms"] = {k: v for m in metas for k, v in m["arms"].items()}
    meta["ladder"] = {k: v for m in metas for k, v in m.get("ladder", {}).items()}
    return base, meta


def empirical_class(em: dict) -> dict:
    """The calibration classes of 4.5 computed from the observed outcomes (reported as empirical,
    never as true; 10.2). Absolute thresholds: no real-data outcome is near the 3 % event rate."""
    m = dict(true_citl_prob=em["emp_citl_prob"], true_intercept=em["emp_intercept"],
             true_slope=em["emp_slope"], true_ici=em["emp_ici"], true_e90=em["emp_e90"])
    if not np.isfinite(m["true_slope"]):
        m["true_slope"] = np.inf
    return {f"emp_{k}": v for k, v in metrics.hierarchy_class(m).items()}


def score_replicate(files: list[Path]) -> list[dict]:
    df, meta = read_replicate(files)
    ev = df[df["set"] == "eval"]
    y = ev["y"].to_numpy(dtype=float)
    ps = ev["p_star"].to_numpy(dtype=float) if "p_star" in ev.columns else None
    base = dict(key=meta["key"], setting=meta["setting"], part=meta["part"], prevalence=meta["prevalence"],
                n=meta["n"], rep=meta["rep"])
    rows = []
    meta["arms"] = {a: i for a, i in meta["arms"].items() if a not in DROPPED_ARMS}
    for arm, info in meta["arms"].items():
        if info.get("status") not in models.USABLE_STATUSES and f"p_{arm}" not in ev.columns:
            rows.append(dict(base, column=arm, arm=arm, ladder="R0", status=info.get("status"),
                             error=str(info.get("error", ""))[:300]))
    for arm, levels in meta.get("ladder", {}).items():
        for lev_m, info in levels.items():
            if info.get("status") != "ok":
                rows.append(dict(base, column=f"{arm}__{lev_m}", arm=arm, ladder=lev_m, status="failed",
                                 error=str(info.get("error", ""))[:300]))
    cols = {c[2:]: ev[c].to_numpy(dtype=float) for c in ev.columns
            if c.startswith("p_") and c != "p_star" and c[2:].partition("__")[0] not in DROPPED_ARMS}
    if ps is not None and "oracle" in meta["arms"]:
        cols = {"oracle": ps, **cols}
    for name, p in cols.items():
        arm, _, level = name.partition("__")
        ai = meta["arms"].get(arm, {})
        row = dict(base, column=name, arm=arm, ladder=level or "R0", status=ai.get("status", "ok"),
                   degenerate=ai.get("status") == "degenerate", degenerate_reason=ai.get("degenerate_reason"),
                   platt_fallback=ai.get("platt_fallback"), lambda_at_edge=ai.get("lambda_at_edge"),
                   model=ai.get("model_version") or ai.get("checkpoint") or ai.get("revision"))
        if np.any(~np.isfinite(p)):
            row["status"] = "missing"
            rows.append(row)
            continue
        em = metrics.empirical_metrics(y, p)
        row.update({f"e_{k}": v for k, v in em.items()})
        row.update(empirical_class(em))
        if ps is not None:
            om = metrics.oracle_metrics(p, ps)
            row.update({f"o_{k}": v for k, v in om.items()})
            row.update(metrics.hierarchy_class(om))
            row.update({f"{k}_ll": v for k, v in metrics.hierarchy_class(om, prefix="true_ll_").items()})
        rows.append(row)
    return rows


def score_setting(args) -> str:
    key, setting, force = args
    folder = RESULTS / key / "preds" / setting
    files = sorted(folder.glob("*/*.parquet"))
    if not files:
        return f"{setting}: no prediction files"
    out = RESULTS / key / "metrics" / f"{setting}.parquet"
    if out.exists() and not force and out.stat().st_mtime > max(f.stat().st_mtime for f in files):
        return f"{setting}: up to date"
    by_rep: dict = {}
    for f in files:
        by_rep.setdefault(f.name, []).append(f)
    rows = []
    for name in sorted(by_rep):
        rows += score_replicate(by_rep[name])
    out.parent.mkdir(parents=True, exist_ok=True)
    tab = pd.DataFrame(rows)
    tmp = out.with_name(out.name + f".tmp-{os.getpid()}")
    tab.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    return f"{setting}: {len(by_rep)} replicate(s), {len(rows)} rows -> {out.relative_to(REBUILD)}"


def settings_of(keys: list[str]) -> list[tuple[str, str]]:
    out = []
    for key in keys:
        d = RESULTS / key / "preds"
        if d.exists():
            out += [(key, s.name) for s in sorted(d.iterdir()) if s.is_dir()]
    return out


def cmd_score(a) -> int:
    keys = parse_list(a.keys, KEYS)
    jobs = [(k, s, a.force) for k, s in settings_of(keys)]
    if a.n_workers > 1:
        with ProcessPoolExecutor(a.n_workers) as ex:
            for msg in ex.map(score_setting, jobs):
                print(msg, flush=True)
    else:
        for j in jobs:
            print(score_setting(j), flush=True)
    return 0


# ----------------------------------------------------------------------------- command: summarise

SUMMARY_ESTIMANDS = ["e_emp_ici", "e_emp_e50", "e_emp_e90", "e_emp_slope", "e_emp_intercept",
                     "e_emp_citl_prob", "e_brier", "e_auc", "e_logloss",
                     "o_true_ici", "o_true_e90", "o_true_slope", "o_true_intercept", "o_oce", "o_true_ll_ici"]
CLASS_COLS = ["emp_cls_mean", "emp_cls_weak", "emp_cls_moderate", "emp_cls_strong",
              "cls_mean", "cls_weak", "cls_moderate", "cls_strong"]


def _summ(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if not len(x):
        return {}
    q = np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])
    return dict(r=len(x), mean=x.mean(), sd=x.std(ddof=1) if len(x) > 1 else np.nan,
                mcse=x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
                p10=q[0], q25=q[1], median=q[2], q75=q[3], p90=q[4])


def summarise_metrics(met: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["key", "part", "prevalence", "n", "column", "arm", "ladder"]
    met = met.copy()
    met["prevalence"] = met["prevalence"].fillna("")
    for k, g in met.groupby(keys, dropna=False):
        usable = g[g["status"].isin(models.USABLE_STATUSES)]
        base = dict(zip(keys, k), replicates=len(g), usable=len(usable),
                    failed=int(g["status"].eq("failed").sum()), degenerate=int(g["status"].eq("degenerate").sum()))
        for est in SUMMARY_ESTIMANDS:
            if est in usable.columns:
                s = _summ(usable[est])
                if s:
                    rows.append(dict(base, estimand=est, **s))
        for c in CLASS_COLS:
            if c in usable.columns and usable[c].notna().any():
                v = usable[c].dropna().astype(float)
                rows.append(dict(base, estimand=f"share_{c}", r=len(v), mean=v.mean(),
                                 mcse=np.sqrt(v.mean() * (1 - v.mean()) / len(v))))
    return pd.DataFrame(rows)


def read_gof(key: str) -> pd.DataFrame:
    import gzip
    parts = []
    for f in sorted((RESULTS / key / "gof").glob("*/*.csv.gz")):
        with gzip.open(f, "rt", encoding="utf8") as fh:
            d = pd.read_csv(fh, dtype={"scenario": str, "rep": str, "fingerprint": str, "note": str,
                                       "pred_hash": str, "engine_fingerprint": str})
        parts.append(d[["scenario", "rep", "arm", "ytype", "test", "family", "statistic", "p_value", "note"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summarise_gof(g: pd.DataFrame) -> pd.DataFrame:
    if g.empty:
        return g
    g = g.copy()
    info = g["scenario"].map(parse_setting)
    g["key"] = [i["key"] for i in info]
    g["part"] = [i["part"] for i in info]
    g["prevalence"] = [i["tag"] or "" for i in info]
    g["n"] = [i["n"] for i in info]
    g = g[g["p_value"].notna()]
    g["reject"] = g["p_value"] < 0.05
    out = (g.groupby(["key", "part", "prevalence", "n", "arm", "ytype", "test"])["reject"]
           .agg(["size", "mean"]).reset_index().rename(columns={"size": "r", "mean": "rejection"}))
    out["mcse"] = np.sqrt(out["rejection"] * (1 - out["rejection"]) / out["r"])
    return out


def test_guided_rule(g: pd.DataFrame) -> pd.DataFrame:
    """5.5 on real data: recalibrate with R2 (m = 500) when the Cox test rejects on R0; flag
    "recalibration insufficient" when EDGE (G = 10) still rejects after R2."""
    if g.empty:
        return g
    g = g[(g["ytype"] == "real") & g["test"].isin(["Cox recalibration", "EDGE (G=10)"])].copy()
    g["base"] = g["arm"].str.split("__").str[0]
    g["level"] = np.where(g["arm"].str.contains("__"), g["arm"].str.split("__").str[1].str.upper(), "R0")
    g["level"] = g["level"].str.replace("_M", "_m", regex=False)        # the R layer lower-cases names
    cox = g[(g["test"] == "Cox recalibration") & (g["level"] == "R0")][["scenario", "rep", "base", "p_value"]]
    edge = g[(g["test"] == "EDGE (G=10)") & (g["level"] == "R2_m500")][["scenario", "rep", "base", "p_value"]]
    m = cox.merge(edge, on=["scenario", "rep", "base"], how="left", suffixes=("_cox_r0", "_edge_r2"))
    m["recalibrate"] = m["p_value_cox_r0"] < 0.05
    m["insufficient"] = m["recalibrate"] & (m["p_value_edge_r2"] < 0.05)
    info = m["scenario"].map(parse_setting)
    m["key"] = [i["key"] for i in info]
    m["part"] = [i["part"] for i in info]
    m["n"] = [i["n"] for i in info]
    return (m.groupby(["key", "part", "n", "base"]).agg(r=("recalibrate", "size"), share_recalibrated=("recalibrate", "mean"),
                                                share_insufficient=("insufficient", "mean")).reset_index())


def localization_summary(g: pd.DataFrame) -> pd.DataFrame:
    if g.empty:
        return g
    loc = g[(g["family"] == "Localize") & g["test"].str.endswith("(named)") & (g["ytype"] == "real")].copy()
    if loc.empty:
        return loc
    loc["group"] = loc["test"].str.replace("Localize ", "").str.replace(" (named)", "", regex=False)
    loc["named"] = loc["note"].fillna("").str.startswith("named")
    info = loc["scenario"].map(parse_setting)
    loc["key"] = [i["key"] for i in info]
    loc["part"] = [i["part"] for i in info]
    loc["n"] = [i["n"] for i in info]
    return loc.groupby(["key", "part", "n", "arm", "group"])["named"].agg(["size", "mean"]).reset_index().rename(
        columns={"size": "r", "mean": "share_named"})


def cmd_summarise(a) -> int:
    """Summary tables of every data set. CRASH-2 is included: its Chief Investigators allowed the
    publication of aggregate results on 2026-10-23 (DEVIATIONS D-02); these tables hold aggregates
    only, and nothing per hospital is ever computed."""
    TABLES.mkdir(parents=True, exist_ok=True)
    dest = TABLES / "by_key"
    dest.mkdir(parents=True, exist_ok=True)
    for key in KEYS:
        mets = sorted((RESULTS / key / "metrics").glob("*.parquet"))
        if not mets:
            continue
        met = pd.concat([pd.read_parquet(f) for f in mets], ignore_index=True)
        met = met[~met["arm"].isin(DROPPED_ARMS)]
        g = read_gof(key)
        if len(g):
            g = g[~g["arm"].str.split("__").str[0].isin(DROPPED_ARMS)]
        out = {"metrics_summary": summarise_metrics(met), "gof_rejection": summarise_gof(g),
               "test_guided_rule": test_guided_rule(g), "localization": localization_summary(g)}
        for name, t in out.items():
            if t is not None and len(t):
                t.to_parquet(dest / f"{key}__{name}.parquet", index=False)
        print(f"{key}: {len(met)} metric rows, {len(g)} GOF rows -> {dest.relative_to(REBUILD)}", flush=True)
    for name in ["metrics_summary", "gof_rejection", "test_guided_rule", "localization"]:
        fs = sorted(dest.glob(f"*__{name}.parquet"))
        if fs:
            t = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
            t.to_parquet(TABLES / f"{name}.parquet", index=False)
            t.to_csv(TABLES / f"{name}.csv", index=False)
    lc_cols = ["e_emp_ici", "e_emp_e50", "e_emp_e90", "e_emp_slope", "e_emp_intercept", "e_brier", "e_auc"]
    for part, fname in (("lc", "learning_curves"), ("lct", "learning_curves_temporal")):
        parts = []
        for key in KEYS:
            for f in sorted((RESULTS / key / "metrics").glob(f"{key}__{part}_n*.parquet")):
                m = pd.read_parquet(f)
                m = m[m["status"].isin(models.USABLE_STATUSES) & ~m["arm"].isin(DROPPED_ARMS)]
                parts.append(m[["key", "n", "rep", "arm", "ladder", "column"] + [c for c in lc_cols if c in m]])
        if parts:
            lc = pd.concat(parts, ignore_index=True).rename(columns={"key": "dataset", "rep": "b"})
            lc.columns = [c[2:] if c.startswith("e_") else c for c in lc.columns]
            lc.to_parquet(TABLES / f"{fname}.parquet", index=False)
    return 0


# ----------------------------------------------------------------------------- command: status

def cmd_status(a) -> int:
    rows = []
    for key in KEYS:
        for k, s in settings_of([key]):
            folder = RESULTS / key / "preds" / s
            for grp in sorted(p.name for p in folder.iterdir() if p.is_dir()):
                files = list((folder / grp).glob("*.parquet"))
                gof = len(list((RESULTS / key / "gof" / s).glob("*.csv.gz")))
                rows.append(dict(setting=s, group=grp, files=len(files), gof_files=gof))
    t = pd.DataFrame(rows)
    if t.empty:
        print("no real-data results yet")
        return 0
    print(t.pivot_table(index="setting", columns="group", values="files", fill_value=0).to_string())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--part", choices=["lc", "lct", "pl"], required=True)
    f.add_argument("--keys", default="all")
    f.add_argument("--ns", default="all")
    f.add_argument("--prevs", default="all", help="plasmode prevalences: 10,30")
    f.add_argument("--reps", default="1-2")
    f.add_argument("--arms", default="classical")
    f.add_argument("--group", required=True, help="file group, e.g. cpu or gpu")
    f.add_argument("--device", default="cpu")
    f.add_argument("--n-workers", type=int, default=1)
    f.add_argument("--threads", type=int, default=1)
    f.add_argument("--block", type=int, default=25, help="replicates per block (all settings advance together)")
    f.add_argument("--rerun-failed", action="store_true")
    f.add_argument("--lock-wait", type=float, default=30.0)
    f.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--keys", default="all")
    s.add_argument("--n-workers", type=int, default=1)
    s.add_argument("--force", action="store_true")
    sub.add_parser("summarise")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    return {"fit": cmd_fit, "score": cmd_score, "summarise": cmd_summarise, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
