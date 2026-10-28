"""Timing benchmark for the Python arms: three TFM families and LightGBM.

For every (model, device, p, n_train, rep) we fit a fresh estimator on the
development set and predict the 10,000-row evaluation set, then append one row
to results/bench/timings_py.csv straight away. Rows already marked ok are
skipped, so the script can be stopped and restarted. Per-fit predictions go to
results/bench/preds/ as float32 parquet.

A CPU budget stops hopeless cells early: if one CPU fit takes longer than
--cpu-budget seconds, the other repeats and the larger n for that model and p
are written as "skipped_budget" instead of being run.

Usage (from rebuild/, after `source env.sh`):
    .venv-gpu/Scripts/python.exe code/bench_01_python_arms.py --devices cuda cpu
    .venv-gpu/Scripts/python.exe code/bench_01_python_arms.py --models tabpfn_v35   # after the licence step
    .venv-gpu/Scripts/python.exe code/bench_01_python_arms.py --memprobe            # GPU memory at n = 3000, 5000
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "bench" / "data"
OUT_CSV = ROOT / "results" / "bench" / "timings_py.csv"
LOAD_CSV = ROOT / "results" / "bench" / "load_times.csv"
PRED_DIR = ROOT / "results" / "bench" / "preds"
PRED_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")   # TabPFN refuses > 1000 rows on CPU otherwise

import torch  # noqa: E402

FIELDS = ["model", "impl", "device", "threads", "n_train", "p", "n_eval", "rep",
          "fit_s", "predict_s", "total_s", "peak_gpu_mb", "status", "auc", "mean_phat", "note"]

MITRA_V2 = "autogluon/mitra-classifier-2"
TABICL_CKPT = "tabicl-classifier-v2-20260212.ckpt"


# ---------------------------------------------------------------- arms
def make_tabpfn(version_name):
    def fit_predict(x_tr, y_tr, x_ev, device, seed):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        clf = TabPFNClassifier.create_default_for_version(
            getattr(ModelVersion, version_name), device=device, random_state=seed)
        t0 = time.perf_counter()
        clf.fit(x_tr, y_tr)
        t1 = time.perf_counter()
        p = clf.predict_proba(x_ev)[:, list(clf.classes_).index(1)]
        t2 = time.perf_counter()
        # "auto" settings are resolved at fit time; record what was used
        note = (f"n_estimators={clf.n_estimators}->{getattr(clf, 'n_estimators_', 'NA')}; "
                f"softmax_temperature={clf.softmax_temperature}->{getattr(clf, 'softmax_temperature_', 'NA')}")
        return t1 - t0, t2 - t1, p, note
    return fit_predict


def fit_predict_tabicl(x_tr, y_tr, x_ev, device, seed):
    from tabicl import TabICLClassifier
    # n_jobs=None would let TabICL reset torch to all physical cores on CPU
    clf = TabICLClassifier(checkpoint_version=TABICL_CKPT, device=device, random_state=seed,
                           n_jobs=torch.get_num_threads(), verbose=False)
    t0 = time.perf_counter()
    clf.fit(x_tr, y_tr)
    t1 = time.perf_counter()
    p = clf.predict_proba(x_ev)[:, list(clf.classes_).index(1)]
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1, p, f"n_estimators={clf.n_estimators}; softmax_temperature={clf.softmax_temperature}"


def fit_predict_mitra(x_tr, y_tr, x_ev, device, seed):
    # In-context only (fine_tune=False). Passing X_val stops the sklearn
    # interface from splitting off 20 % of the rows: preprocessing and the
    # context then use all n rows. The validation pass is a no-op without
    # fine-tuning, so a 128-row slice keeps it cheap.
    from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier
    clf = MitraClassifier(hf_model=MITRA_V2, fine_tune=False, n_estimators=1,
                          device=device, seed=seed, verbose=False)
    k = min(128, len(y_tr))
    t0 = time.perf_counter()
    clf.fit(x_tr, y_tr, X_val=x_tr[:k], y_val=y_tr[:k])
    t1 = time.perf_counter()
    p = clf.predict_proba(x_ev)[:, 1]
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1, p, "fine_tune=False; n_estimators=1; context=all n rows"


def fit_predict_lgbm(x_tr, y_tr, x_ev, device, seed):
    from bench_lgbm import fit_predict_lgbm as f
    t0 = time.perf_counter()
    p_raw, p_platt, info = f(x_tr, y_tr, x_ev, seed=seed, n_threads=LGBM_THREADS)
    t1 = time.perf_counter()
    # tuning, refit and Platt all count as "fit"; prediction is inside it
    return t1 - t0, 0.0, p_platt, "platt; " + ";".join(f"{k}={v}" for k, v in info.items())


ARMS = {
    # name: (impl label, function, devices it can run on)
    "tabpfn_v2":  ("tabpfn ModelVersion.V2 (Real-TabPFN-2 default ckpt)", make_tabpfn("V2"), ("cuda", "cpu")),
    "tabpfn_v35": ("tabpfn ModelVersion.V3_5 (gated licence)", make_tabpfn("V3_5"), ("cuda", "cpu")),
    "tabicl_v2":  ("tabicl " + TABICL_CKPT, fit_predict_tabicl, ("cuda", "cpu")),
    "mitra_v2":   ("autogluon MitraClassifier " + MITRA_V2, fit_predict_mitra, ("cuda", "cpu")),
    "lgbm":       ("lightgbm 5-fold CV grid + OOF Platt", fit_predict_lgbm, ("cpu",)),
}
DEFAULT_MODELS = ["tabpfn_v2", "tabicl_v2", "mitra_v2", "lgbm"]
LGBM_THREADS = 1   # one thread per fit; the main run parallelises over replicates instead


# ---------------------------------------------------------------- helpers
def checksum(df, p):
    x = df[[f"x{j}" for j in range(1, p + 1)]].to_numpy()
    w_row = np.arange(1, len(df) + 1) / len(df)
    return float(w_row @ x @ np.arange(1, p + 1) + (w_row * df["y"].to_numpy()).sum())


MANIFEST = json.loads((DATA / "manifest.json").read_text()) if (DATA / "manifest.json").exists() else {}


def load(name, p):
    df = pd.read_parquet(DATA / f"{name}.parquet")
    cs = checksum(df, p)
    ref = MANIFEST[name]["checksum"]
    assert abs(cs - ref) < 1e-6 * max(1.0, abs(cs)), f"fingerprint mismatch for {name}"
    x = df[[f"x{j}" for j in range(1, p + 1)]].to_numpy(np.float32)
    return x, df["y"].to_numpy(np.int64)


def auc(y, p):
    r = pd.Series(p).rank().to_numpy()
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def done_keys(path):
    if not path.exists():
        return set()
    d = pd.read_csv(path)
    d = d[d.status.isin(["ok", "skipped_budget"])]
    return set(zip(d.model, d.device, d.n_train, d.p, d.rep, d.threads.astype(int)))


def append(path, row, fields):
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def warmup(model, device):
    """Load weights and CUDA kernels once so the timed fits do not include it."""
    impl, fn, _ = ARMS[model]
    x, y = load("train_p5_n100_r1", 5)
    xe, _ = load("eval_p5_r1", 5)
    t0 = time.perf_counter()
    try:
        fn(x, y, xe[:200], device, 0)
        status = "ok"
    except Exception as e:  # noqa: BLE001
        status = f"error: {type(e).__name__}: {str(e)[:300]}"
    append(LOAD_CSV, {"model": model, "device": device, "first_fit_s": round(time.perf_counter() - t0, 3),
                      "status": status, "time": time.strftime("%Y-%m-%d %H:%M:%S")},
           ["model", "device", "first_fit_s", "status", "time"])
    return status


def run_grid(models, devices, threads, cpu_budget, p_grid, n_grid, reps):
    torch.set_num_threads(threads)
    keys = done_keys(OUT_CSV)
    for device in devices:
        for model in models:
            impl, fn, ok_dev = ARMS[model]
            if device not in ok_dev:
                continue
            status = warmup(model, device)
            print(f"[{model}/{device}] warm-up: {status}", flush=True)
            if status != "ok":
                for p in p_grid:
                    for n in n_grid:
                        for rep in reps:
                            if (model, device, n, p, rep, threads) not in keys:
                                append(OUT_CSV, dict(model=model, impl=impl, device=device, threads=threads,
                                                     n_train=n, p=p, n_eval=10_000, rep=rep, status="error",
                                                     note="warm-up failed: " + status[:250]), FIELDS)
                continue
            n_thr = LGBM_THREADS if model == "lgbm" else (threads if device == "cpu" else 0)
            for p in p_grid:
                over_budget = False
                for n in n_grid:
                    for rep in reps:
                        if (model, device, n, p, rep, n_thr) in keys:
                            continue
                        row = dict(model=model, impl=impl, device=device, threads=n_thr,
                                   n_train=n, p=p, n_eval=10_000, rep=rep)
                        if over_budget:
                            append(OUT_CSV, row | {"status": "skipped_budget",
                                                   "note": f"a CPU fit at smaller n or earlier rep exceeded {cpu_budget}s"}, FIELDS)
                            continue
                        x, y = load(f"train_p{p}_n{n}_r{rep}", p)
                        xe, ye = load(f"eval_p{p}_r{rep}", p)
                        if device == "cuda":
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                        seed = int(np.random.SeedSequence((p, n, rep)).generate_state(1)[0] % (2**31 - 1))
                        # machine load just before the fit (other jobs share the CPU)
                        busy = psutil.cpu_percent(interval=1.0)
                        try:
                            fit_s, pred_s, phat, note = fn(x, y, xe, device, seed)
                            note = f"{note}; cpu_busy_before={busy:.0f}%"
                            peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
                            phat = np.asarray(phat, dtype=np.float64)
                            pd.DataFrame({"phat": phat.astype(np.float32)}).to_parquet(
                                PRED_DIR / f"{model}_{device}_p{p}_n{n}_r{rep}.parquet", index=False)
                            row |= dict(fit_s=round(fit_s, 4), predict_s=round(pred_s, 4),
                                        total_s=round(fit_s + pred_s, 4),
                                        peak_gpu_mb=None if peak is None else round(peak, 1),
                                        status="ok", auc=round(auc(ye, phat), 4),
                                        mean_phat=round(float(phat.mean()), 4), note=note[:250])
                            if device == "cpu" and fit_s + pred_s > cpu_budget:
                                over_budget = True
                        except Exception as e:  # noqa: BLE001
                            row |= dict(status="error", note=f"{type(e).__name__}: {str(e)[:240]}")
                            traceback.print_exc()
                            if device == "cuda" and "out of memory" in str(e).lower():
                                torch.cuda.empty_cache()
                        append(OUT_CSV, row, FIELDS)
                        print(f"{model:10s} {device:4s} p={p:2d} n={n:4d} rep={rep} {row['status']:6s} "
                              f"total={row.get('total_s')} s peak={row.get('peak_gpu_mb')} MB", flush=True)


def memprobe(models, n_list, p, mem_fraction):
    """Peak GPU memory at the plan's largest n (3000) and beyond,
    N_eval = 3,334 + 500 = 3,834 predicted rows (plan section 4.4).

    On Windows the NVIDIA driver silently spills to system RAM when the card is
    full (the first probe ran Mitra-v2 at n = 3000 for 39 minutes that way).
    With mem_fraction set, torch raises an out-of-memory error instead.
    Mitra's sklearn interface catches that error and halves its support set
    (i.e. it silently drops context rows); we capture its stdout to flag it.
    """
    import contextlib
    import io

    from bench_00_make_data import design, draw
    if mem_fraction:
        torch.cuda.set_per_process_memory_fraction(mem_fraction)
    out = ROOT / "results" / "bench" / "gpu_memprobe.csv"
    fields = ["model", "n_train", "p", "n_pred", "fit_s", "predict_s", "peak_alloc_mb", "peak_reserved_mb",
              "mem_fraction", "status", "note"]
    alpha, beta, sigma = design(p)
    for n in n_list:
        tr = draw((p, n, 99), n, p, alpha, beta, sigma)
        ev = draw((p, 1, 99), 3834, p, alpha, beta, sigma)
        x = tr.filter(like="x").to_numpy(np.float32)
        y = tr.y.to_numpy(np.int64)
        xe = ev.filter(like="x").to_numpy(np.float32)
        for model in models:
            if "cuda" not in ARMS[model][2]:
                continue
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            row = dict(model=model, n_train=n, p=p, n_pred=len(xe), mem_fraction=mem_fraction)
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    fit_s, pred_s, phat, note = ARMS[model][1](x, y, xe, "cuda", 1)
                shrunk = "Reducing max_samples" in buf.getvalue()
                row |= dict(fit_s=round(fit_s, 3), predict_s=round(pred_s, 3),
                            peak_alloc_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1),
                            peak_reserved_mb=round(torch.cuda.max_memory_reserved() / 2**20, 1),
                            status="ok_context_subsampled" if shrunk else "ok",
                            note=(note + ("; " + buf.getvalue().replace("\n", " ")[:150] if shrunk else ""))[:250])
            except Exception as e:  # noqa: BLE001
                row |= dict(status="error", note=f"{type(e).__name__}: {str(e)[:200]}")
            append(out, row, fields)
            print(row, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(ARMS))
    ap.add_argument("--devices", nargs="+", default=["cuda", "cpu"], choices=["cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads for CPU runs")
    ap.add_argument("--cpu-budget", type=float, default=240.0)
    ap.add_argument("--p", nargs="+", type=int, default=[5, 20])
    ap.add_argument("--n", nargs="+", type=int, default=[100, 500, 2000])
    ap.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--memprobe", action="store_true")
    ap.add_argument("--probe-n", nargs="+", type=int, default=[3000, 5000])
    ap.add_argument("--probe-p", type=int, default=20)
    ap.add_argument("--mem-fraction", type=float, default=0.0,
                    help="cap torch GPU memory (e.g. 0.8) so a full card raises OOM instead of spilling")
    a = ap.parse_args()
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
    if a.memprobe:
        memprobe(a.models, a.probe_n, a.probe_p, a.mem_fraction)
    else:
        run_grid(a.models, a.devices, a.threads, a.cpu_budget, a.p, a.n, a.reps)
