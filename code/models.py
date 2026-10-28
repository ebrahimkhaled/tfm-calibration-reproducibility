"""Model arms (REBUILD_PLAN section 3, pre-registration 5.2).

Every arm is fitted on the development set of a replicate and returns predicted probabilities
for one stacked matrix of rows to predict (evaluation, recalibration and probe rows). A failure is
returned as status "failed" with the error text; it is never replaced by another arm's output.

    oracle          the true risk p*
    lr, flic,       R (glm, logistf with flic = TRUE, glmnet ridge with 10-fold CV deviance,
    ridge, gam      mgcv REML with s(x, k = 5)); one Rscript call per replicate (file hand-off)
    lr_py, flic_py  Python versions of LR-MLE and FLIC, used to cross-check the R arms
    lgbm            LightGBM tuned by 5-fold stratified CV with early stopping, refitted on all
                    rows, then out-of-fold Platt scaling; also returns the raw scores as lgbm_raw.
                    A Platt slope <= 0 would turn the ranking upside down, so in that case the
                    arm falls back to the raw scores and says so (platt_fallback)
    tabpfn          the TabPFN checkpoint pinned by TFMCAL_TABPFN_VERSION (V3_5 for the main run:
                    TabPFN-3.5 standard, shipped softmax temperature 1.0); unpinned, the newest
                    checkpoint present in rebuild/models/tabpfn
    tabpfn_legacy   Real-TabPFN-2 default classifier (ModelVersion.V2)
    tabpfn_t1       temperature arm: the tabpfn checkpoint at softmax_temperature = 0.9 (the
                    name is kept from the plan; since TabPFN-3.5 ships T = 1.0, this arm now shows
                    the effect of the 0.9 that TabPFN v2 and TabICL ship with)
    tabicl          TabICLv2 (tabicl-classifier-v2-20260212), shipped softmax temperature 0.9
    tabicl_t1       temperature arm: TabICLv2 at softmax_temperature = 1.0
    mitra           Mitra-v2 (autogluon/mitra-classifier-2), in-context only, whole development
                    set as context

No model weights are downloaded unless allow_download is set, and gated TabPFN versions are never
downloaded from here: their licence must be accepted by the author (see README.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import special

from metrics import clip, fit_logistic, offset_intercept

CODE_DIR = Path(__file__).resolve().parent
REBUILD = CODE_DIR.parent
MODELS_DIR = Path(os.environ.get("TFMCAL_MODELS_DIR", REBUILD / "models"))
TMP_DIR = REBUILD / "tmp"
R_SCRIPT = CODE_DIR / "fit_r_arms.R"

# stable integer codes: used in seeds, so never renumber
ARM_CODES = {"oracle": 1, "lr": 2, "flic": 3, "ridge": 4, "gam": 5, "lgbm": 6, "lgbm_raw": 7,
             "tabpfn": 8, "tabpfn_legacy": 9, "tabicl": 10, "mitra": 11, "lr_py": 12, "flic_py": 13,
             "tabpfn_t1": 14, "tabicl_t1": 15}
R_ARMS = ("lr", "flic", "ridge", "gam")
TFM_ARMS = ("tabpfn", "tabpfn_legacy", "tabicl", "mitra")
TEMPERATURE_ARMS = ("tabpfn_t1", "tabicl_t1")
GPU_ARMS = ("tabpfn", "tabpfn_legacy", "tabpfn_t1", "tabicl", "tabicl_t1", "mitra")
# softmax temperatures of the two temperature arms (pre-registration 5.2)
TABPFN_T_ARM_TEMPERATURE = 0.9
TABICL_T_ARM_TEMPERATURE = 1.0
# a fit is "degenerate" (scored and kept, but labelled and counted) when more than this many
# evaluation predictions sit within 1e-6 of 0 or 1, or when LR-MLE shows separation
DEGENERATE_BOUNDARY_ROWS = 100
OUTPUTS = {"lgbm": ["lgbm", "lgbm_raw"]}           # arms that produce more than one column
DEFAULT_ARMS = ["oracle", "lr", "flic", "ridge", "gam", "lgbm",
                "tabpfn", "tabpfn_legacy", "tabicl", "mitra"]
CPU_SAFE_ARMS = ["oracle", "lr", "flic", "ridge", "gam", "lgbm"]

GAM_K = 5
LGBM_GRID = [dict(num_leaves=nl, min_data_in_leaf=md) for nl in (7, 15, 31) for md in (10, 30)]
LGBM_BASE = dict(objective="binary", metric="binary_logloss", learning_rate=0.05, verbose=-1,
                 deterministic=True, force_col_wise=True, feature_pre_filter=False)

# TabPFN checkpoints, newest first: (ModelVersion name, file in rebuild/models/tabpfn)
TABPFN_CANDIDATES = [
    ("V3_5", "tabpfn-v3.5-20260909.safetensors"),
    ("V3_5_FAST", "tabpfn-v3.5-fast-20260909.safetensors"),
    ("V3","tabpfn-v3-classifier-v3_default.ckpt"),
    ("V2_6", "tabpfn-v2.6-classifier-v2.6_default.ckpt"),
    ("V2_5", "tabpfn-v2.5-classifier-v2.5_default.ckpt"),
    ("V2", "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"),
]
TABPFN_LEGACY = ("V2", "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt")
MITRA_REPO = "autogluon/mitra-classifier-2"
MITRA_DIR = "mitra-classifier-2"                 # optional plain folder under rebuild/models
TABICL_REPO = "jingang/TabICL"
TABICL_CKPT = "tabicl-classifier-v2-20260212.ckpt"


def _hf_cached(repo: str, filename: str | None = None, allow_download: bool = False) -> Path:
    """Path of a Hugging Face snapshot (or one file of it) in the project cache, without network
    access unless allow_download."""
    from huggingface_hub import hf_hub_download, snapshot_download
    get = (lambda **kw: hf_hub_download(repo, filename, **kw)) if filename else \
        (lambda **kw: snapshot_download(repo, **kw))
    try:
        return Path(get(local_files_only=True))
    except Exception as e:  # noqa: BLE001
        if not allow_download:
            raise FileNotFoundError(f"{repo} {filename or ''} is not in the project cache "
                                    f"({os.environ.get('HF_HUB_CACHE')}): {type(e).__name__}") from e
        return Path(get())


def outputs_of(arm: str) -> list[str]:
    return OUTPUTS.get(arm, [arm])


def configure_environment(allow_download: bool = False, threads: int | None = None) -> None:
    """Environment for the foundation-model libraries (the same folders as rebuild/env.sh, whose
    values win when it has been sourced). Call before any of them is imported."""
    os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR / "hf" / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(MODELS_DIR / "hf" / "xet"))
    os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(MODELS_DIR / "tabpfn"))
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")   # n = 3000 on CPU workers
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")                # never open the licence page
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("AUTOGLUON_DISABLE_TELEMETRY", "1")
    if not allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if threads:
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[v] = str(threads)


# ----------------------------------------------------------------------------- context

@dataclass
class FitContext:
    X_dev: np.ndarray
    y_dev: np.ndarray
    X_pred: np.ndarray
    p_star_pred: np.ndarray
    columns: list
    cont: list
    bin: list
    cat: list
    cat_levels: dict
    seeds: dict                      # arm -> integer seed
    device: str = "cpu"
    threads: int = 1
    allow_download: bool = False

    def idx(self, cols) -> list[int]:
        return [self.columns.index(c) for c in cols]


@dataclass
class ArmResult:
    arm: str
    p: np.ndarray | None
    status: str
    info: dict = field(default_factory=dict)


def _fail(arm: str, err: str, t0: float, **info) -> ArmResult:
    return ArmResult(arm, None, "failed", dict(error=err, seconds=time.perf_counter() - t0, **info))


def _check_p(arm: str, p, n: int) -> str | None:
    p = np.asarray(p, dtype=float)
    if p.shape != (n,):
        return f"{arm}: {p.shape} predictions for {n} rows"
    if not np.all(np.isfinite(p)):
        return f"{arm}: non-finite predictions"
    if np.any((p < 0) | (p > 1)):
        return f"{arm}: predictions outside [0, 1]"
    return None


# ----------------------------------------------------------------------------- oracle

def fit_oracle(ctx: FitContext) -> list[ArmResult]:
    return [ArmResult("oracle", np.asarray(ctx.p_star_pred, dtype=float), "ok", {"seconds": 0.0})]


# ----------------------------------------------------------------------------- R arms

def _rscript() -> str:
    exe = os.environ.get("TFM_RSCRIPT") or shutil.which("Rscript")
    if exe:
        return exe
    for c in sorted(Path("C:/Program Files/R").glob("R-*/bin/Rscript.exe"), reverse=True):
        return str(c)
    raise FileNotFoundError("Rscript not found: set TFM_RSCRIPT")


def fit_r_arms(arms: list[str], ctx: FitContext, timeout: int = 1800) -> list[ArmResult]:
    """LR-MLE, FLIC, ridge-CV and GAM in one Rscript call."""
    t0 = time.perf_counter()
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    wd = Path(tempfile.mkdtemp(prefix="rarms_", dir=TMP_DIR))
    try:
        head = ",".join(["y"] + ctx.columns)
        np.savetxt(wd / "dev.csv", np.column_stack([ctx.y_dev, ctx.X_dev]), delimiter=",",
                   header=head, comments="", fmt="%.17g")
        np.savetxt(wd / "pred.csv", ctx.X_pred, delimiter=",", header=",".join(ctx.columns),
                   comments="", fmt="%.17g")
        job = dict(arms=list(arms), seeds={a: ctx.seeds[a] for a in arms}, cont=ctx.cont,
                   bin=ctx.bin, cat=ctx.cat, cat_levels=ctx.cat_levels, gam_k=GAM_K)
        (wd / "job.json").write_text(json.dumps(job))
        env = {**os.environ, "TFM_RLIB": str(REBUILD / "rlib")}
        proc = subprocess.run([_rscript(), str(R_SCRIPT), str(wd)], capture_output=True, text=True,
                              timeout=timeout, env=env)
        info_file = wd / "info.json"
        if proc.returncode != 0 or not info_file.exists():
            err = (proc.stderr or proc.stdout or "no output")[-2000:]
            return [_fail(a, f"Rscript exit {proc.returncode}: {err}", t0) for a in arms]
        meta = json.loads(info_file.read_text())
        # the R side read back the same replicate: same rows, events and covariate sum
        chk = meta["check"]
        xs = ctx.X_dev[:, [i for i, c in enumerate(ctx.columns) if c not in ctx.cat]].sum()
        if (chk["n_dev"] != len(ctx.y_dev) or chk["events_dev"] != int(ctx.y_dev.sum())
                or abs(chk["sum_x"] - xs) > 1e-6 * max(1.0, abs(xs)) or chk["n_pred"] != len(ctx.X_pred)):
            raise AssertionError(f"R read a different data set: {chk}")
        preds = {}
        if (wd / "preds.csv").exists():
            tab = np.genfromtxt(wd / "preds.csv", delimiter=",", names=True, dtype=float)
            for name in tab.dtype.names:
                preds[name] = np.atleast_1d(tab[name]).astype(float)
        out = []
        for a in arms:
            ai = dict(meta["arms"].get(a, {"status": "failed", "error": "no output from R"}))
            ai["versions"] = meta.get("versions", {})
            ai["r_call_seconds"] = time.perf_counter() - t0          # shared by the R arms
            ai["r_package_load_seconds"] = meta.get("package_load_seconds")
            if ai.get("status") == "ok" and a in preds:
                bad = _check_p(a, preds[a], len(ctx.X_pred))
                if bad:
                    out.append(ArmResult(a, None, "failed", {**ai, "error": bad}))
                else:
                    out.append(ArmResult(a, preds[a], "ok", ai))
            else:
                out.append(ArmResult(a, None, "failed", ai))
        return out
    finally:
        shutil.rmtree(wd, ignore_errors=True)


# ----------------------------------------------------------------------------- Python LR / FLIC

def design_matrix(X: np.ndarray, ctx: FitContext) -> np.ndarray:
    """Intercept, continuous and binary columns, and treatment dummies of each categorical."""
    parts = [np.ones((len(X), 1)), X[:, ctx.idx(ctx.cont + ctx.bin)]]
    for c in ctx.cat:
        codes = X[:, ctx.columns.index(c)].astype(int)
        parts.append((codes[:, None] == np.arange(1, ctx.cat_levels[c])[None, :]).astype(float))
    return np.column_stack(parts)


def _loglik(Z, y, beta):
    eta = Z @ beta
    return float(np.sum(y * special.log_expit(eta) + (1 - y) * special.log_expit(-eta)))


def logistic_newton(Z, y, firth: bool = False, max_iter: int = 200, max_step: float = 5.0,
                    tol: float = 1e-8) -> tuple[np.ndarray, dict]:
    """Logistic ML or Firth's penalized ML by Newton steps with step halving.

    Firth: the modified score X'(y - mu + h (1/2 - mu)), h the hat diagonal, maximising
    loglik + 1/2 log det(X'WX); steps capped at max_step as in logistf."""
    k = Z.shape[1]
    beta = np.zeros(k)

    def objective(b):
        val = _loglik(Z, y, b)
        if firth:
            mu = special.expit(Z @ b)
            sign, logdet = np.linalg.slogdet((Z * (mu * (1 - mu))[:, None]).T @ Z)
            val += 0.5 * logdet if sign > 0 else -np.inf
        return val

    obj = objective(beta)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        mu = special.expit(Z @ beta)
        W = mu * (1 - mu)
        info = (Z * W[:, None]).T @ Z
        try:
            cov = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            break
        resid = y - mu
        if firth:
            h = W * np.einsum("ij,jk,ik->i", Z, cov, Z)
            resid = resid + h * (0.5 - mu)
        U = Z.T @ resid
        delta = cov @ U
        m = np.max(np.abs(delta))
        if m > max_step:
            delta *= max_step / m
        for _ in range(30):
            new = objective(beta + delta)
            if new >= obj - 1e-12:
                break
            delta = delta / 2
        beta = beta + delta
        obj = new
        if np.max(np.abs(delta)) < tol and np.max(np.abs(U)) < 1e-6:
            converged = True
            break
    return beta, dict(converged=converged, iterations=it, max_abs_coef=float(np.max(np.abs(beta))))


def fit_lr_py(ctx: FitContext) -> list[ArmResult]:
    t0 = time.perf_counter()
    Z = design_matrix(ctx.X_dev, ctx)
    beta, info = logistic_newton(Z, ctx.y_dev.astype(float), firth=False, max_iter=50)
    fitted = special.expit(Z @ beta)
    info["separation_flag"] = (not info["converged"]) or info["max_abs_coef"] > 15 or bool(
        np.any((fitted < 1e-8) | (fitted > 1 - 1e-8)))
    p = special.expit(design_matrix(ctx.X_pred, ctx) @ beta)
    info["seconds"] = time.perf_counter() - t0
    return [ArmResult("lr_py", p, "ok", info)]


def fit_flic_py(ctx: FitContext) -> list[ArmResult]:
    """FLIC (Puhr et al. 2017): Firth's slopes, then the intercept re-estimated by ML with the
    Firth linear predictor (without intercept) as offset, so that mean p_hat = mean y."""
    t0 = time.perf_counter()
    Z = design_matrix(ctx.X_dev, ctx)
    y = ctx.y_dev.astype(float)
    beta, info = logistic_newton(Z, y, firth=True)
    off = Z[:, 1:] @ beta[1:]
    a = offset_intercept(off, y)
    if not np.isfinite(a):
        return [_fail("flic_py", "intercept correction failed (no events or no non-events)", t0)]
    p = special.expit(a + design_matrix(ctx.X_pred, ctx)[:, 1:] @ beta[1:])
    info.update(seconds=time.perf_counter() - t0, firth_intercept=float(beta[0]), flic_intercept=a)
    return [ArmResult("flic_py", p, "ok", info)]


# ----------------------------------------------------------------------------- LightGBM

def platt_or_raw(oof: np.ndarray, y: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """Platt scaling of the refit's scores, fitted on the out-of-fold scores.

    Platt scaling is meant to be monotone increasing. With a handful of events, early stopping can
    pick one or two boosting rounds, the out-of-fold scores are then close to noise, and the
    logistic fit can return a slope <= 0, which would reverse the ranking (pilot: 3 of 20
    replicates at n = 200, pi = 3 %). Pre-registered rule: in that case the arm keeps the raw
    LightGBM scores, and platt_fallback records it. A fit that does not converge stays a failure."""
    lo, _ = clip(oof)
    a, b, ok = fit_logistic(special.logit(lo), y)
    info = dict(platt_a=a, platt_b=b, platt_converged=ok, platt_fallback=False)
    if not ok:
        return None, info
    if not b > 0:
        info.update(platt_fallback=True, platt_fallback_reason=f"Platt slope {b:.4g} <= 0: raw scores kept")
        return np.asarray(raw, dtype=float), info
    lr_, _ = clip(raw)
    return special.expit(a + b * special.logit(lr_)), info


def fit_lgbm(ctx: FitContext) -> list[ArmResult]:
    """5-fold stratified CV over LGBM_GRID with early stopping (50 rounds, at most 2,000), refit on
    all rows at the CV-best round count, out-of-fold Platt scaling of the refit."""
    t0 = time.perf_counter()
    try:
        import lightgbm as lgb
        from sklearn.model_selection import StratifiedKFold
        X, y = ctx.X_dev, ctx.y_dev.astype(int)
        seed = ctx.seeds["lgbm"]
        if y.sum() < 2 or (len(y) - y.sum()) < 2:
            return [_fail(a, "fewer than 2 events or non-events", t0) for a in ("lgbm", "lgbm_raw")]
        with warnings.catch_warnings():
            # fewer events than folds at n = 200, pi = 3 %: expected, the event count is recorded
            warnings.simplefilter("ignore", UserWarning)
            folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))
        cat_idx = ctx.idx(ctx.cat)
        base = dict(LGBM_BASE, seed=seed, num_threads=ctx.threads)
        best = None
        for g in LGBM_GRID:
            ds = lgb.Dataset(X, y, categorical_feature=cat_idx or "auto", free_raw_data=False)
            res = lgb.cv({**base, **g}, ds, num_boost_round=2000, folds=folds, stratified=False,
                         callbacks=[lgb.early_stopping(50, verbose=False)], return_cvbooster=True)
            key = next(k for k in res if k.endswith("binary_logloss-mean"))
            loss, rounds = float(res[key][-1]), len(res[key])
            if best is None or loss < best["loss"]:
                best = dict(loss=loss, rounds=rounds, grid=g, cvb=res["cvbooster"])
        oof = np.empty(len(y))
        for (_, va), bst in zip(folds, best["cvb"].boosters):
            oof[va] = bst.predict(X[va], num_iteration=best["cvb"].best_iteration)
        final = lgb.train({**base, **best["grid"]},
                          lgb.Dataset(X, y, categorical_feature=cat_idx or "auto"),
                          num_boost_round=best["rounds"])
        raw = final.predict(ctx.X_pred)
        p_platt, pinfo = platt_or_raw(oof, y, raw)
        info = dict(seconds=time.perf_counter() - t0, grid=best["grid"], rounds=best["rounds"],
                    cv_logloss=best["loss"], events_dev=int(y.sum()), version=lgb.__version__)
        out = [ArmResult("lgbm_raw", raw, "ok", info)]
        if p_platt is not None:
            out.insert(0, ArmResult("lgbm", p_platt, "ok", {**info, **pinfo}))
        else:
            out.insert(0, ArmResult("lgbm", None, "failed",
                                    {**info, **pinfo, "error": "Platt fit did not converge"}))
        return out
    except Exception as e:  # noqa: BLE001
        return [_fail(a, f"{type(e).__name__}: {e}", t0, traceback=traceback.format_exc(limit=5))
                for a in ("lgbm", "lgbm_raw")]


# ----------------------------------------------------------------------------- foundation models

_SHA_CACHE: dict = {}


def file_sha256(path: Path) -> str:
    """SHA-256 of a checkpoint, cached in rebuild/models/_sha256.json by path, size and mtime."""
    path = Path(path)
    st = path.stat()
    key = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    if key in _SHA_CACHE:
        return _SHA_CACHE[key]
    cache_file = MODELS_DIR / "_sha256.json"
    try:
        disk = json.loads(cache_file.read_text())
    except (OSError, ValueError):
        disk = {}
    if key not in disk:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        disk[key] = h.hexdigest()
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(disk, indent=1))
            os.replace(tmp, cache_file)
        except OSError:
            pass
    _SHA_CACHE[key] = disk[key]
    return disk[key]


def resolve_tabpfn(arm: str) -> tuple[str, Path]:
    """(ModelVersion name, checkpoint path) for an arm. The newest checkpoint present wins unless
    TFMCAL_TABPFN_VERSION pins one (e.g. V3_5)."""
    folder = MODELS_DIR / "tabpfn"
    if arm == "tabpfn_legacy":
        cands = [TABPFN_LEGACY]
    else:
        pin = os.environ.get("TFMCAL_TABPFN_VERSION")
        cands = [c for c in TABPFN_CANDIDATES if pin in (None, c[0])]
    for ver, fname in cands:
        if (folder / fname).exists():
            return ver, folder / fname
    raise FileNotFoundError(f"no TabPFN checkpoint for {arm} in {folder} (looked for "
                            + ", ".join(f for _, f in cands) + ")")


def _pkg_version(name: str):
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(name)
    except PackageNotFoundError:
        return None


# Share of the 6 GB card a worker may use. Without a cap, the Windows driver quietly spills the
# overflow into system RAM (one Mitra fit took 39 minutes that way); with it, an overflow is a
# CUDA out-of-memory error that the wrappers record as a failure.
GPU_MEMORY_FRACTION = float(os.environ.get("TFMCAL_GPU_FRACTION", "0.8"))


def _gpu_peak_reset(device: str):
    if device.startswith("cuda"):
        import torch
        dev = torch.device(device)
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION,
                                                   dev.index if dev.index is not None else 0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _gpu_peak_mb(device: str):
    if device.startswith("cuda"):
        import torch
        return float(torch.cuda.max_memory_allocated() / 2 ** 20)
    return None


def _positive_column(proba: np.ndarray, classes) -> np.ndarray:
    classes = list(np.asarray(classes).tolist())
    if 1 not in classes or proba.shape[1] != len(classes):
        raise ValueError(f"model saw classes {classes}: no positive-class column")
    return proba[:, classes.index(1)]


def fit_tabpfn(ctx: FitContext, arm: str = "tabpfn", temperature: float | None = None) -> list[ArmResult]:
    t0 = time.perf_counter()
    try:
        ver, path = resolve_tabpfn("tabpfn" if arm == "tabpfn_t1" else arm)
        extra = {} if temperature is None else {"softmax_temperature": temperature}
        import tabpfn
        import torch
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        torch.set_num_threads(ctx.threads)
        _gpu_peak_reset(ctx.device)
        cat_idx = ctx.idx(ctx.bin + ctx.cat)        # binary and categorical declared categorical
        # the T = 1.0 arm reuses the main arm's seed, so the two differ only in the temperature
        seed = ctx.seeds["tabpfn" if arm == "tabpfn_t1" else arm]
        clf = TabPFNClassifier.create_default_for_version(
            getattr(ModelVersion, ver), model_path=str(path), device=ctx.device,
            random_state=seed, categorical_features_indices=cat_idx or None, **extra)
        clf.fit(ctx.X_dev, ctx.y_dev.astype(int))
        t_fit = time.perf_counter() - t0
        p = _positive_column(clf.predict_proba(ctx.X_pred), clf.classes_)
        info = dict(seconds=time.perf_counter() - t0, fit_seconds=t_fit,
                    predict_seconds=time.perf_counter() - t0 - t_fit,
                    model_version=ver, checkpoint=path.name, checkpoint_sha256=file_sha256(path),
                    tabpfn_version=tabpfn.__version__, torch_version=torch.__version__,
                    device=ctx.device, n_estimators=getattr(clf, "n_estimators_", None),
                    softmax_temperature=float(getattr(clf, "softmax_temperature_", float("nan"))),
                    peak_gpu_mb=_gpu_peak_mb(ctx.device))
        bad = _check_p(arm, p, len(ctx.X_pred))
        if bad:
            return [ArmResult(arm, None, "failed", {**info, "error": bad})]
        return [ArmResult(arm, p, "ok", info)]
    except Exception as e:  # noqa: BLE001
        return [_fail(arm, f"{type(e).__name__}: {e}", t0, traceback=traceback.format_exc(limit=5))]


def fit_tabpfn_legacy(ctx: FitContext) -> list[ArmResult]:
    return fit_tabpfn(ctx, "tabpfn_legacy")


def fit_tabpfn_t1(ctx: FitContext) -> list[ArmResult]:
    """Temperature arm: the main tabpfn checkpoint (TabPFN-3.5, shipped at T = 1.0) run at
    softmax temperature 0.9, the value TabPFN v2 and TabICL ship with. Same seed as tabpfn, so the
    two differ only in the temperature."""
    return fit_tabpfn(ctx, "tabpfn_t1", temperature=TABPFN_T_ARM_TEMPERATURE)


def _mitra_full_context(clf, n_dev: int) -> None:
    """AutoGluon answers a CUDA out-of-memory error by halving max_samples_support and trying
    again, after which every prediction uses a random subset of the development rows as context.
    That is a different model, so it counts as a failure here."""
    support = min(int(t.cfg.hyperparams["max_samples_support"]) for t in clf.trainers)
    if support < n_dev:
        raise RuntimeError(f"Mitra cut its context to {support} of {n_dev} development rows after a "
                           "GPU out-of-memory error")


def fit_mitra(ctx: FitContext) -> list[ArmResult]:
    """Mitra-v2 through autogluon's MitraClassifier: no fine-tuning, the whole development set is
    the context (no hold-out). Categorical columns keep their integer codes, which is how
    AutoGluon itself passes categoricals to Mitra (label encoding)."""
    t0 = time.perf_counter()
    try:
        local = MODELS_DIR / MITRA_DIR
        if not (local / "config.json").exists():
            local = _hf_cached(MITRA_REPO, allow_download=ctx.allow_download)   # snapshot folder
        if len(ctx.y_dev) > 5120 or ctx.X_dev.shape[1] > 50:
            raise ValueError("outside Mitra-v2 pretraining limits (5,120 rows, 50 features)")
        import autogluon.tabular as agt
        import torch
        from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier
        torch.set_num_threads(ctx.threads)
        _gpu_peak_reset(ctx.device)
        y = ctx.y_dev.astype(int)
        # the validation rows are only scored once (no fine-tuning epochs), so a small set with
        # both classes avoids predicting the whole development set twice
        rng = np.random.default_rng(ctx.seeds["mitra"])
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        if len(pos) == 0 or len(neg) == 0:
            raise ValueError("development set has a single class")
        val = np.r_[rng.choice(pos, min(2, len(pos)), replace=False),
                    rng.choice(neg, min(14, len(neg)), replace=False)]
        clf = MitraClassifier(fine_tune=False, n_estimators=1, device=ctx.device,
                              hf_model=str(local), seed=ctx.seeds["mitra"], verbose=False)
        # Mitra's preprocessor flips the sign of each feature at random (random_mirror_x, on by
        # default) with the global numpy generator, which its own seed does not touch. Unseeded,
        # a rerun of the same replicate moves predictions by about 0.003 on average and up to 0.06.
        # Seed the global generators from the replicate, and put them back afterwards.
        import random
        np_state, py_state = np.random.get_state(), random.getstate()
        np.random.seed(ctx.seeds["mitra"])
        random.seed(ctx.seeds["mitra"])
        torch.manual_seed(ctx.seeds["mitra"])
        try:
            clf.fit(ctx.X_dev, y, X_val=ctx.X_dev[val], y_val=y[val])
            t_fit = time.perf_counter() - t0
            _mitra_full_context(clf, len(y))
            mirror = [np.asarray(t.preprocessor.mirror).ravel().tolist()
                      for t in clf.trainers if hasattr(t.preprocessor, "mirror")]
            p = _positive_column(clf.predict_proba(ctx.X_pred), np.unique(y))
            _mitra_full_context(clf, len(y))
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)
        info = dict(seconds=time.perf_counter() - t0, fit_seconds=t_fit,
                    predict_seconds=time.perf_counter() - t0 - t_fit,
                    weights=MITRA_REPO, revision=local.name, context_rows=len(y), mirror_x=mirror,
                    gpu_memory_fraction=GPU_MEMORY_FRACTION if ctx.device.startswith("cuda") else None,
                    weights_sha256=file_sha256((local / "model.safetensors").resolve())
                    if (local / "model.safetensors").exists() else None,
                    autogluon_version=agt.__version__, torch_version=torch.__version__,
                    device=ctx.device, peak_gpu_mb=_gpu_peak_mb(ctx.device))
        bad = _check_p("mitra", p, len(ctx.X_pred))
        if bad:
            return [ArmResult("mitra", None, "failed", {**info, "error": bad})]
        return [ArmResult("mitra", p, "ok", info)]
    except Exception as e:  # noqa: BLE001
        return [_fail("mitra", f"{type(e).__name__}: {e}", t0, traceback=traceback.format_exc(limit=5))]


def fit_tabicl(ctx: FitContext, arm: str = "tabicl", temperature: float | None = None) -> list[ArmResult]:
    """TabICLv2 with its default settings; binary and categorical columns are passed as pandas
    categories so that the package treats them as categorical. The temperature arm tabicl_t1
    changes only softmax_temperature and reuses the tabicl seed."""
    t0 = time.perf_counter()
    try:
        import inspect

        import pandas as pd
        import torch
        from tabicl import TabICLClassifier
        torch.set_num_threads(ctx.threads)
        ckpt = MODELS_DIR / "tabicl" / TABICL_CKPT
        if not ckpt.exists():
            ckpt = _hf_cached(TABICL_REPO, TABICL_CKPT, allow_download=ctx.allow_download)
        want = dict(checkpoint_version=TABICL_CKPT, model_path=str(ckpt),
                    allow_auto_download=False, device=ctx.device,
                    random_state=ctx.seeds["tabicl"], verbose=False)
        if temperature is not None:
            want["softmax_temperature"] = temperature
        sig = inspect.signature(TabICLClassifier.__init__).parameters
        missing = [k for k in want if k not in sig]
        if missing:
            # dropping model_path or allow_auto_download would silently load another checkpoint
            raise TypeError(f"this tabicl version has no {missing} argument(s)")
        kwargs = dict(want)
        _gpu_peak_reset(ctx.device)

        def frame(X):
            d = pd.DataFrame(X, columns=ctx.columns)
            for c in ctx.bin + ctx.cat:
                d[c] = d[c].astype(int).astype("category")
            return d

        clf = TabICLClassifier(**kwargs)
        clf.fit(frame(ctx.X_dev), ctx.y_dev.astype(int))
        t_fit = time.perf_counter() - t0
        p = _positive_column(clf.predict_proba(frame(ctx.X_pred)), clf.classes_)
        info = dict(seconds=time.perf_counter() - t0, fit_seconds=t_fit,
                    predict_seconds=time.perf_counter() - t0 - t_fit, checkpoint=TABICL_CKPT,
                    checkpoint_sha256=file_sha256(ckpt.resolve()),
                    n_estimators=getattr(clf, "n_estimators", None),
                    softmax_temperature=getattr(clf, "softmax_temperature", None),
                    tabicl_version=_pkg_version("tabicl"), kwargs_used=sorted(kwargs),
                    torch_version=torch.__version__, device=ctx.device,
                    small_n_flag=len(ctx.y_dev) < 300, peak_gpu_mb=_gpu_peak_mb(ctx.device))
        bad = _check_p(arm, p, len(ctx.X_pred))
        if bad:
            return [ArmResult(arm, None, "failed", {**info, "error": bad})]
        return [ArmResult(arm, p, "ok", info)]
    except Exception as e:  # noqa: BLE001
        return [_fail(arm, f"{type(e).__name__}: {e}", t0, traceback=traceback.format_exc(limit=5))]


def fit_tabicl_t1(ctx: FitContext) -> list[ArmResult]:
    """Temperature arm: TabICLv2 at softmax temperature 1.0 instead of its shipped 0.9."""
    return fit_tabicl(ctx, "tabicl_t1", temperature=TABICL_T_ARM_TEMPERATURE)


# ----------------------------------------------------------------------------- fit status

USABLE_STATUSES = ("ok", "degenerate")     # both are scored, tested and kept in every comparison


def degenerate_reason(info: dict, n_boundary_eval: int) -> str | None:
    """Why a fit that ran should be labelled "degenerate" (pre-registration 8.2), or None.

    A degenerate fit is the arm's real output, so it is scored and never removed; the label only
    makes it countable. Two rules: more than DEGENERATE_BOUNDARY_ROWS evaluation predictions within
    1e-6 of 0 or 1 (GAM and LR-MLE at n = 200, pi = 3 % in the pilot), or LR-MLE separation."""
    why = []
    if n_boundary_eval > DEGENERATE_BOUNDARY_ROWS:
        why.append(f"{n_boundary_eval} evaluation predictions within 1e-6 of 0 or 1")
    if info.get("separation_flag"):
        why.append("separation (LR-MLE)")
    return "; ".join(why) or None


# ----------------------------------------------------------------------------- dispatch

PY_FITTERS = {"oracle": fit_oracle, "lr_py": fit_lr_py, "flic_py": fit_flic_py, "lgbm": fit_lgbm,
              "tabpfn": fit_tabpfn, "tabpfn_legacy": fit_tabpfn_legacy, "tabpfn_t1": fit_tabpfn_t1,
              "mitra": fit_mitra,
              "tabicl": fit_tabicl, "tabicl_t1": fit_tabicl_t1}


def run_arms(arms: list[str], ctx: FitContext) -> dict[str, ArmResult]:
    """Fit the requested arms; R arms share one Rscript call. Returns one result per output."""
    unknown = [a for a in arms if a not in PY_FITTERS and a not in R_ARMS]
    if unknown:
        raise ValueError(f"unknown arm(s): {unknown}")
    results: dict[str, ArmResult] = {}
    y = np.asarray(ctx.y_dev)
    if y.min() == y.max():
        # no events (about 1 replicate in 400 at n = 200, pi = 3 %): glm, logistf and mgcv would
        # return "ok" with every prediction at about 0, the other arms stop with an error. The
        # same recorded failure for every arm keeps the arms on identical replicate sets.
        for a in arms:
            if a == "oracle":
                results["oracle"] = fit_oracle(ctx)[0]
                continue
            for o in outputs_of(a):
                results[o] = ArmResult(o, None, "failed", {
                    "error": f"development set has a single class ({int(y.sum())} events)",
                    "seconds": 0.0, "degenerate": True})
        return results
    r_arms = [a for a in arms if a in R_ARMS]
    if r_arms:
        for res in fit_r_arms(r_arms, ctx):
            results[res.arm] = res
    for a in arms:
        if a in R_ARMS:
            continue
        for res in PY_FITTERS[a](ctx):
            results[res.arm] = res
    return results


def environment_info() -> dict:
    """Versions of the libraries the engine uses (heavy ones only if already imported)."""
    import platform
    import sys

    import numpy
    import pandas
    import pyarrow
    import scipy
    info = dict(python=sys.version.split()[0], platform=platform.platform(), numpy=numpy.__version__,
                scipy=scipy.__version__, pandas=pandas.__version__, pyarrow=pyarrow.__version__)
    for pkg in ("lightgbm", "scikit-learn", "torch", "tabpfn", "tabicl", "autogluon.tabular",
                "huggingface_hub"):
        info[pkg] = _pkg_version(pkg)
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return info
