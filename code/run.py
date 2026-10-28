"""Simulation loop: scenario x replicate -> one prediction file per replicate.

    python run.py --scenarios S1_pi10_n200,S3_pi03_n200 --reps 1-2 --models cpu-safe --smoke
    python run.py --scenarios core --reps 1-500 --models classical --n-workers 8 --threads 1
    python run.py --scenarios core --reps 1-200 --models mitra,tabicl --device cuda --n-workers 1

Each replicate is drawn from SeedSequence([scenario_id, n_index, rep]) (smoke data add 777), every
requested arm is fitted on its development set, and one parquet file is written to
results/preds/<scenario>/<rep>.parquet (results/smoke/preds/... for smoke data):

    set = "eval"   row, y, p_star, in_sub_e50, in_sub_e200, x1.., p_<arm>, ystar_<arm>,
                   p_<arm>__<R1|R2|R3>_m<100|500> (recalibration ladder)
    set = "recal"  row, y, p_star, p_<arm>
    set = "probe"  row, p_star, p_<arm>   (the fixed probe rows of the scenario)

p_<arm> are float32 (pre-registration 3.8); p_star is float64. The oracle arm is p_star itself, so
it has no p_ column, only ystar_oracle. Seeds, fingerprints, the status, timing and flags of every
arm, the ladder fits and the run history are in the file's metadata under the key "tfmcal".

Resume: a replicate whose requested arms are all recorded (ok or failed) is skipped. Adding arms
to a finished replicate regenerates its data, asserts that the fingerprint matches the stored one,
fits only the missing arms and rewrites the file atomically. A lock file per replicate lets a
GPU process and a CPU process work on the same scenarios at the same time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REBUILD = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import dgp  # noqa: E402
import models  # noqa: E402
import recal  # noqa: E402

META_KEY = b"tfmcal"
SCHEMA_VERSION = 1
DEFAULT_LADDER_ARMS = ["flic", "tabpfn", "tabicl", "mitra"]
MODEL_SETS = {
    "all": models.DEFAULT_ARMS,
    "cpu-safe": models.CPU_SAFE_ARMS,
    "classical": ["oracle", "lr", "flic", "ridge", "gam", "lgbm"],
    "tfm": list(models.TFM_ARMS),
    "temperature": list(models.TEMPERATURE_ARMS),
}
USABLE = models.USABLE_STATUSES

# (arm, scenario) pairs that are recorded as "not_run" without a fit, with the reason
# (pre-registration 5.2). Mitra-v2 at p = 30, n = 3000 needs more than the 6 GB card: its 16-row
# validation pass alone runs out of memory, and AutoGluon's fallback would cut the context.
NOT_RUN = {("mitra", "T2_pi10_n3000"): "does not fit in 6 GB GPU memory (pre-registration 5.2)"}
LOG_DIR = REBUILD / "logs"
CODE_FILES = ["dgp.py", "dgp_random.py", "metrics.py", "models.py", "recal.py", "run.py", "fit_r_arms.R",
              "scenarios.csv", "random_designs.csv"]


def not_run_reason(arm: str, sc: dict) -> str | None:
    """Why an arm is recorded as not run in a scenario, without a fit (None: it is fitted). In the
    random designs Mitra-v2 follows the T2 rule: a design too large for the 6 GB card is recorded as
    "does not fit", and its context is never cut (pre-registration 3.9)."""
    if (arm, sc["scenario"]) in NOT_RUN:
        return NOT_RUN[(arm, sc["scenario"])]
    if arm == "mitra" and sc.get("family") == "R":
        import dgp_random
        if not dgp_random.mitra_fits(sc):
            return "does not fit in 6 GB GPU memory (pre-registration 3.9, the T2 rule)"
    return None


# ----------------------------------------------------------------------------- small helpers

def code_hash() -> str:
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update((CODE_DIR / f).read_bytes())
    return h.hexdigest()[:16]


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return o


def read_meta(path: Path) -> dict | None:
    try:
        md = pq.read_schema(path).metadata or {}
    except (OSError, pa.ArrowInvalid):
        return None
    return json.loads(md[META_KEY]) if META_KEY in md else None


def rep_file(out_dir: Path, scenario: str, rep: int, tag: str | None) -> Path:
    d = out_dir / scenario / tag if tag else out_dir / scenario
    return d / f"{rep:04d}.parquet"


def arms_recorded(out_dir: Path, scenario: str, rep: int) -> dict:
    """Status of every output already stored for this replicate, over all files (main and tags)."""
    seen = {}
    folder = out_dir / scenario
    if not folder.exists():
        return seen
    for f in folder.rglob(f"{rep:04d}.parquet"):
        m = read_meta(f)
        if m:
            for a, info in m.get("arms", {}).items():
                seen[a] = dict(status=info.get("status"), file=str(f))
    return seen


def pid_alive(pid: int) -> bool:
    """Whether a process exists. On Windows os.kill(pid, 0) would terminate it, so ask the API."""
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259          # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Lock:
    """Exclusive lock file next to a replicate file. A lock is broken when its owner process (on
    this host) is gone, or when it is older than max_age_h."""

    def __init__(self, target: Path, max_age_h: float = 6.0):
        self.path = target.with_name(target.name + ".lock")
        self.max_age = max_age_h * 3600
        self.held = False

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{socket.gethostname()} {os.getpid()} {datetime.now().isoformat()}".encode())
                os.close(fd)
                self.held = True
                return True
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                    host, pid = self.path.read_text().split()[:2]
                except (OSError, ValueError):
                    continue
                dead = host == socket.gethostname() and not pid_alive(int(pid))
                if dead or age > self.max_age:
                    try:
                        os.remove(self.path)
                    except OSError:
                        return False
                    continue
                return False
        return False

    def release(self):
        if self.held:
            try:
                os.remove(self.path)
            except OSError:
                pass
            self.held = False


def write_atomic(table: pa.Table, path: Path):
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    pq.write_table(table, tmp, compression="zstd", row_group_size=max(1, table.num_rows))
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:          # a reader (e.g. the R layer) has the file open on Windows
            time.sleep(0.5 * (attempt + 1))
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- table building

def base_frame(r: dgp.Replicate, probe: dgp.DataSet, store_x: bool) -> pd.DataFrame:
    sc = r.scenario
    ne, nr, npb = len(r.eval.y), len(r.recal.y), len(probe.p_star)
    k50 = int(np.ceil(50 / sc["pi"]))
    k200 = int(np.ceil(200 / sc["pi"]))
    rows = np.r_[np.arange(ne), np.arange(nr), np.arange(npb)]
    df = pd.DataFrame({
        "set": pd.Categorical(["eval"] * ne + ["recal"] * nr + ["probe"] * npb,
                              categories=["eval", "recal", "probe"]),
        "row": rows.astype(np.int32),
        "y": pd.array(np.r_[r.eval.y, r.recal.y, np.full(npb, -1)], dtype="Int8"),
        "p_star": np.r_[r.eval.p_star, r.recal.p_star, probe.p_star].astype(np.float64),
    })
    df.loc[df["set"] == "probe", "y"] = pd.NA
    is_eval = (df["set"] == "eval").to_numpy()
    sub50 = np.where(is_eval, rows < k50, False) if k50 <= ne else np.zeros(len(df), bool)
    sub200 = np.where(is_eval, rows < k200, False) if k200 <= ne else np.zeros(len(df), bool)
    df["in_sub_e50"] = sub50
    df["in_sub_e200"] = sub200
    if store_x:
        for j, c in enumerate(sc["columns"]):
            col = np.full(len(df), np.nan, dtype=np.float32)
            col[:ne] = r.eval.X[:, j].astype(np.float32)
            df[c] = col
    return df


def add_prediction(df: pd.DataFrame, name: str, p: np.ndarray):
    df[f"p_{name}"] = np.asarray(p, dtype=np.float32)


def add_ystar(df: pd.DataFrame, r: dgp.Replicate, name: str, p_col: str):
    """Exact null outcome Y* ~ Bern(stored p_hat) on the evaluation rows, own seed per arm."""
    is_eval = (df["set"] == "eval").to_numpy()
    p = df.loc[is_eval, p_col].to_numpy(dtype=np.float64)
    rng = r.child_rng(3, models.ARM_CODES[name])
    ys = (rng.random(len(p)) < p).astype(np.int8)
    col = pd.array(np.full(len(df), -1, dtype=np.int8), dtype="Int8")
    col[np.flatnonzero(is_eval)] = ys
    col[np.flatnonzero(~is_eval)] = pd.NA
    df[f"ystar_{name}"] = col


def add_ladder(df: pd.DataFrame, arm: str) -> dict:
    """Recalibration ladder of one arm from its stored float32 predictions."""
    ev = (df["set"] == "eval").to_numpy()
    rc = (df["set"] == "recal").to_numpy()
    p_col = f"p_{arm}"
    p_eval = df.loc[ev, p_col].to_numpy(dtype=np.float64)
    p_rec = df.loc[rc, p_col].to_numpy(dtype=np.float64)
    y_rec = df.loc[rc, "y"].to_numpy(dtype=np.float64)
    preds, infos = recal.ladder(p_rec, y_rec, p_eval)
    out = {}
    for (lev, m), p in preds.items():
        name = recal.column_name(arm, lev, m)
        col = np.full(len(df), np.nan, dtype=np.float32)
        if p is not None:
            col[ev] = p.astype(np.float32)
            df[name] = col
        out[f"{lev}_m{m}"] = dict(status="ok" if p is not None else "failed", **infos[(lev, m)])
    return out


# ----------------------------------------------------------------------------- one replicate

def run_replicate(task: dict) -> dict:
    """Fit the missing arms of one replicate and write its file. Runs inside a worker."""
    t_start = time.perf_counter()
    sc = dgp.get_scenario(task["scenario"])
    name, rep = sc["scenario"], task["rep"]
    out_dir = Path(task["out_dir"])
    path = rep_file(out_dir, name, rep, task["tag"])
    res = dict(scenario=name, rep=rep, file=str(path), status="skipped", arms_run=[], failures=[])

    retry = {"failed"} if task["rerun_failed"] else set()
    # the random designs have no recalibration set and no ladder (pre-registration 3.9)
    ladder_arms = [] if sc.get("family") == "R" else task["ladder_arms"]

    def still_needed() -> list:
        recorded = arms_recorded(out_dir, name, rep)
        return [a for a in task["arms"]
                if any(o not in recorded or recorded[o]["status"] in retry for o in models.outputs_of(a))]

    needed = still_needed()
    existing = read_meta(path) if path.exists() else None
    ladder_todo = [a for a in ladder_arms if existing and existing["arms"].get(a, {}).get("status") in USABLE
                   and a not in existing.get("ladder", {})]
    if not needed and not ladder_todo:
        return res

    lock = Lock(path)
    if not lock.acquire():
        res["status"] = "locked"
        return res
    try:
        # temp files left by a killed writer of this replicate (we hold its lock now)
        for tmp in path.parent.glob(f"{path.name}.tmp-*"):
            tmp.unlink(missing_ok=True)
        # re-read under the lock: another process may have finished the replicate meanwhile
        existing = read_meta(path) if path.exists() else None
        needed = still_needed()
        if not needed and not [a for a in ladder_arms
                               if existing and existing["arms"].get(a, {}).get("status") in USABLE
                               and a not in existing.get("ladder", {})]:
            return res
        r = dgp.make_replicate(sc["scenario_id"], rep, smoke=task["smoke"])
        probe = dgp.make_probe(sc["scenario_id"])
        if existing is not None:
            if existing["fingerprints"]["all"] != r.fingerprints["all"]:
                raise AssertionError(f"{name} rep {rep}: regenerated data do not match the stored "
                                     f"fingerprint (RNG or code change?)")
            df = pq.read_table(path).to_pandas()
            ev = (df["set"] == "eval").to_numpy()
            if not (np.array_equal(df.loc[ev, "y"].to_numpy(dtype=np.int8), r.eval.y)
                    and np.array_equal(df.loc[ev, "p_star"].to_numpy(), r.eval.p_star)):
                raise AssertionError(f"{name} rep {rep}: stored outcomes differ from regenerated ones")
            meta = existing
        else:
            df = base_frame(r, probe, task["store_x"])
            meta = dict(schema=SCHEMA_VERSION, scenario=name, scenario_id=int(sc["scenario_id"]),
                        rep=rep, smoke=task["smoke"], entropy=r.entropy, fingerprints=r.fingerprints,
                        probe_fingerprint=probe.fingerprint,
                        events=dict(dev=int(r.dev.y.sum()), eval=int(r.eval.y.sum()),
                                    recal=int(r.recal.y.sum())),
                        sizes=dict(dev=len(r.dev.y), eval=len(r.eval.y), recal=len(r.recal.y),
                                   probe=len(probe.p_star)),
                        # rows of the nested validation subsamples (first rows); None = larger
                        # than the evaluation set, so not available in this scenario
                        sub_sizes={f"e{k}": (int(np.ceil(k / sc["pi"]))
                                             if np.ceil(k / sc["pi"]) <= len(r.eval.y) else None)
                                   for k in (50, 200)},
                        col_types=dict(cont=sc["cont_cols"], bin=sc["bin_cols"], cat=sc["cat_cols"],
                                       cat_levels=sc["cat_levels"]),
                        x_stored=task["store_x"], arms={}, ladder={}, history=[])

        for a in [a for a in needed if not_run_reason(a, sc)]:
            needed.remove(a)
            meta["arms"][a] = dict(status="not_run", error=not_run_reason(a, sc), seconds=0.0)
            res["arms_run"].append(a)

        if needed:
            X_pred = np.vstack([r.eval.X, r.recal.X, probe.X])
            ctx = models.FitContext(
                X_dev=r.dev.X, y_dev=r.dev.y, X_pred=X_pred,
                p_star_pred=np.r_[r.eval.p_star, r.recal.p_star, probe.p_star],
                columns=sc["columns"], cont=sc["cont_cols"], bin=sc["bin_cols"], cat=sc["cat_cols"],
                cat_levels=sc["cat_levels"],
                seeds={a: r.child_seed(4, models.ARM_CODES[a]) for a in models.ARM_CODES},
                device=task["device"], threads=task["threads"], allow_download=task["allow_download"])
            results = models.run_arms(needed, ctx)
            n_eval = len(r.eval.y)
            for out_name, ar in results.items():
                info = {**models_info(ar), "status": ar.status}
                if ar.status == "ok":
                    p = np.asarray(ar.p, dtype=np.float64)
                    at_bound = (p <= 1e-6) | (p >= 1 - 1e-6)
                    info["n_boundary"] = int(np.sum(at_bound))                 # all predicted rows
                    info["n_boundary_eval"] = int(np.sum(at_bound[:n_eval]))   # evaluation rows
                    why = models.degenerate_reason(info, info["n_boundary_eval"]) if out_name != "oracle" else None
                    if why:
                        info.update(status="degenerate", degenerate_reason=why)
                    if out_name != "oracle":
                        add_prediction(df, out_name, p)
                    add_ystar(df, r, out_name, "p_star" if out_name == "oracle" else f"p_{out_name}")
                    for c in [c for c in df.columns if c.startswith(f"p_{out_name}__")]:
                        del df[c]                     # a re-fitted arm gets a fresh ladder
                    meta["ladder"].pop(out_name, None)
                else:
                    res["failures"].append(dict(arm=out_name, error=str(info.get("error", ""))[:500]))
                meta["arms"][out_name] = info
                res["arms_run"].append(out_name)

        for a in ladder_arms:
            if meta["arms"].get(a, {}).get("status") in USABLE and a not in meta["ladder"]:
                meta["ladder"][a] = add_ladder(df, a)

        meta["history"].append(dict(time=datetime.now().isoformat(timespec="seconds"),
                                    host=socket.gethostname(), pid=os.getpid(),
                                    arms=res["arms_run"], code=task["code_hash"],
                                    device=task["device"], seconds=time.perf_counter() - t_start))
        table = pa.Table.from_pandas(df, preserve_index=False)
        md = dict(table.schema.metadata or {})
        md[META_KEY] = json.dumps(jsonable(meta), default=str).encode()
        table = table.replace_schema_metadata(md)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(table, path)
        res.update(status="written", fingerprint=r.fingerprints["all"], dev_fp=r.fingerprints["dev"],
                   seconds=time.perf_counter() - t_start,
                   arm_seconds={k: v.get("seconds") for k, v in meta["arms"].items() if k in res["arms_run"]})
        return res
    except Exception as e:  # noqa: BLE001
        res.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        return res
    finally:
        lock.release()


def models_info(ar) -> dict:
    """Arm info without bulky fields, ready for json."""
    info = {k: v for k, v in (ar.info or {}).items() if k != "p"}
    return jsonable(info)


def _worker_init(threads: int, allow_download: bool):
    models.configure_environment(allow_download=allow_download, threads=threads)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(threads)
    except ImportError:
        pass


# ----------------------------------------------------------------------------- command line

def parse_reps(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def parse_scenarios(s: str) -> list[str]:
    import dgp_random
    t = dgp.load_table()
    out = []
    for part in s.split(","):
        part = part.strip()
        rnd = dgp_random.select(part)        # 'random', 'R0001-R0005', 'R0007' (never part of 'all')
        if rnd is not None:
            if not rnd:
                raise SystemExit(f"no random design matches {part!r}")
            out += [x for x in rnd if x not in out]
            continue
        if part in ("all", "core", "stress"):
            sel = t if part == "all" else t[t.kind == part]
        elif part in dgp.FAMILIES:
            sel = t[t.family == part]
        elif "-" in part and all(x.isdigit() for x in part.split("-")):
            a, b = map(int, part.split("-"))
            sel = t[(t.scenario_id >= a) & (t.scenario_id <= b)]
        elif part.isdigit():
            sel = t[t.scenario_id == int(part)]
        else:
            sel = t[t.scenario == part]
        if sel.empty:
            raise SystemExit(f"no scenario matches {part!r}")
        out += [x for x in sel.scenario if x not in out]
    return out


def parse_models(s: str) -> list[str]:
    out = []
    for part in s.split(","):
        for a in MODEL_SETS.get(part, [part]):
            if a not in models.ARM_CODES or a in ("lgbm_raw",):
                raise SystemExit(f"unknown model {a!r}")
            if a not in out:
                out.append(a)
    return out


def write_probe_files(scenarios: list[str], probe_dir: Path):
    probe_dir.mkdir(parents=True, exist_ok=True)
    for s in scenarios:
        f = probe_dir / f"{s}.parquet"
        pr = dgp.make_probe(s)
        if f.exists():
            md = pq.read_schema(f).metadata or {}
            if json.loads(md.get(META_KEY, b"{}")).get("fingerprint") != pr.fingerprint:
                raise AssertionError(f"probe set of {s} differs from {f}")
            continue
        sc = dgp.get_scenario(s)
        df = pd.DataFrame(pr.X.astype(np.float32), columns=sc["columns"])
        df.insert(0, "p_star", pr.p_star)
        df.insert(0, "row", np.arange(len(df), dtype=np.int32))
        t = pa.Table.from_pandas(df, preserve_index=False)
        t = t.replace_schema_metadata({META_KEY: json.dumps(dict(scenario=s, fingerprint=pr.fingerprint)).encode()})
        write_atomic(t, f)


def check_tabpfn_version(out_dir: Path, scenarios: list[str], arms: list[str], tag: str | None):
    """The tabpfn arm resolves to the newest checkpoint on disk, so placing TabPFN-3.5 in
    rebuild/models/tabpfn halfway through a run would put two models under one arm name. Refuse
    to add replicates fitted with another version than the ones already stored."""
    want = [a for a in ("tabpfn", "tabpfn_t1") if a in arms]
    if not want:
        return
    ver, path = models.resolve_tabpfn("tabpfn")
    # the main run pins the checkpoint by its SHA-256 too (RUN_MAIN.md): a half-copied or replaced
    # file must stop the run before the first replicate, not fail it replicate by replicate
    expected = os.environ.get("TFMCAL_TABPFN_SHA256")
    if expected:
        got = models.file_sha256(path)
        if got != expected.strip().lower():
            raise SystemExit(f"{path.name}: SHA-256 {got} differs from TFMCAL_TABPFN_SHA256 {expected}")
        print(f"TabPFN checkpoint {path.name} ({ver}) SHA-256 verified")
    for s in scenarios:
        folder = out_dir / s / tag if tag else out_dir / s       # the folder this run writes into
        for f in folder.glob("*.parquet") if folder.exists() else []:
            m = read_meta(f) or {}
            for a in want:
                old = m.get("arms", {}).get(a, {})
                if old.get("status") in USABLE and old.get("model_version") not in (None, ver):
                    raise SystemExit(f"{f}: {a} was fitted with TabPFN {old['model_version']}, but "
                                     f"{path.name} ({ver}) is now the newest checkpoint. Pin the version "
                                     f"(TFMCAL_TABPFN_VERSION={old['model_version']}) or write the new "
                                     f"version to another folder (--out; a --tag folder would still "
                                     f"see the stored arm as done).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default="all",
                    help="names, ids, id ranges (1-12), families (S1, T3) or core / stress / all")
    ap.add_argument("--reps", default="1-2", help="e.g. 1-200 or 1,5,9")
    ap.add_argument("--models", default="cpu-safe",
                    help="arms or sets: all, cpu-safe, classical, tfm, or names such as flic,tabpfn")
    ap.add_argument("--ladder-arms", default=",".join(DEFAULT_LADDER_ARMS))
    ap.add_argument("--device", default="cpu", help="cpu or cuda (foundation models only)")
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1, help="threads per worker")
    ap.add_argument("--smoke", action="store_true", help="smoke seed stream (777), separate output folder")
    ap.add_argument("--out", default=None, help="output folder (default results/preds or results/smoke/preds)")
    ap.add_argument("--tag", default=None, help="write into <scenario>/<tag>/ instead of <scenario>/")
    ap.add_argument("--store-x", choices=["all", "none"], default="all",
                    help="store the evaluation covariates (needed by localization and DeepGOF-1)")
    ap.add_argument("--rerun-failed", action="store_true", help="refit arms recorded as failed")
    ap.add_argument("--allow-download", action="store_true",
                    help="allow ungated weight downloads into rebuild/models (never gated TabPFN)")
    ap.add_argument("--lock-wait", type=float, default=30.0, help="minutes to wait for locked replicates")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    scenarios = parse_scenarios(a.scenarios)
    reps = parse_reps(a.reps)
    arms = parse_models(a.models)
    ladder_arms = [x for x in a.ladder_arms.split(",") if x]
    out_dir = Path(a.out) if a.out else REBUILD / "results" / ("smoke/preds" if a.smoke else "preds")
    probe_dir = out_dir.parent / "probe"
    if a.device.startswith("cuda") and a.n_workers > 1 and any(x in models.GPU_ARMS for x in arms):
        print("warning: several workers share one GPU", file=sys.stderr)

    tasks = [dict(scenario=s, rep=r, arms=arms, ladder_arms=ladder_arms, out_dir=str(out_dir),
                  tag=a.tag, smoke=a.smoke, store_x=a.store_x == "all", device=a.device,
                  threads=a.threads, rerun_failed=a.rerun_failed, allow_download=a.allow_download,
                  code_hash=code_hash())
             for s in scenarios for r in reps]
    print(f"{len(scenarios)} scenario(s) x {len(reps)} replicate(s) = {len(tasks)} tasks; arms {arms}; "
          f"device {a.device}; {a.n_workers} worker(s) x {a.threads} thread(s); out {out_dir}")
    if a.dry_run:
        return 0

    models.configure_environment(allow_download=a.allow_download, threads=a.threads)
    check_tabpfn_version(out_dir, scenarios, arms, a.tag)
    write_probe_files(scenarios, probe_dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"run_{stamp}{'_smoke' if a.smoke else ''}.jsonl"
    env_file = out_dir.parent / "environment_engine.json"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env = dict(time=stamp, code_hash=code_hash(), argv=sys.argv, host=platform.node(),
               **models.environment_info())
    env_file.write_text(json.dumps(env, indent=1, default=str))

    seen_fp: dict = {}
    counts: dict = {}

    def handle(res: dict):
        counts[res["status"]] = counts.get(res["status"], 0) + 1
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(jsonable(res), default=str) + "\n")
        if res["status"] == "written":
            # a hidden RNG reset would give two replicates the same data
            for key in ("fingerprint", "dev_fp"):
                fp = res[key]
                other = seen_fp.get((key, fp))
                if other and other != (res["scenario"], res["rep"]):
                    raise AssertionError(f"identical data in {other} and {(res['scenario'], res['rep'])}")
                seen_fp[(key, fp)] = (res["scenario"], res["rep"])
        if res["failures"]:
            with open(LOG_DIR / "failures.jsonl", "a", encoding="utf-8") as f:
                for fl in res["failures"]:
                    f.write(json.dumps(dict(time=stamp, scenario=res["scenario"], rep=res["rep"], **fl)) + "\n")
        msg = f"{res['scenario']} rep {res['rep']}: {res['status']}"
        if res["status"] == "written":
            secs = ", ".join(f"{k} {v:.1f}s" for k, v in res["arm_seconds"].items() if v is not None)
            msg += f" ({res['seconds']:.1f}s; {secs})"
        if res["failures"]:
            msg += " FAILED: " + "; ".join(f"{x['arm']}: {x['error'][:80]}" for x in res["failures"])
        if res["status"] == "error":
            msg += " ERROR " + res["error"]
        print(msg, flush=True)

    def run_batch(batch):
        locked = []
        if a.n_workers <= 1:
            for t in batch:
                res = run_replicate(t)
                (locked.append(t) if res["status"] == "locked" else handle(res))
        else:
            import multiprocessing as mp
            with ProcessPoolExecutor(a.n_workers, mp_context=mp.get_context("spawn"),
                                     initializer=_worker_init,
                                     initargs=(a.threads, a.allow_download)) as ex:
                futs = {ex.submit(run_replicate, t): t for t in batch}
                for fu in as_completed(futs):
                    res = fu.result()
                    (locked.append(futs[fu]) if res["status"] == "locked" else handle(res))
        return locked

    t0 = time.time()
    pending = run_batch(tasks)
    while pending and time.time() - t0 < a.lock_wait * 60 + 1:
        print(f"{len(pending)} replicate(s) locked by another process; retrying in 30 s", flush=True)
        time.sleep(30)
        pending = run_batch(pending)
    if pending:
        counts["locked"] = len(pending)
    print(f"done in {time.time() - t0:.0f}s: {counts}; log {log_file}")
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
