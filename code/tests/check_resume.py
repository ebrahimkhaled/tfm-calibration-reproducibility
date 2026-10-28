"""Resume and interruption checks on the smoke stream (writes only under results/smoke).

    python tests/check_resume.py

1. a replicate locked by a live process is left alone; a lock whose owner is dead is broken;
2. a run killed hard in the middle resumes: finished replicates are skipped, the rest are written;
3. a deleted replicate is regenerated with the same fingerprint and the same predictions.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import run  # noqa: E402

PY = sys.executable
OUT = CODE.parent / "results" / "smoke" / "preds"
SCEN = "S1_pi10_n200"
ARMS = ["oracle", "lr", "flic", "ridge", "gam", "lgbm"]


def task(rep):
    return dict(scenario=SCEN, rep=rep, arms=ARMS, ladder_arms=run.DEFAULT_LADDER_ARMS, out_dir=str(OUT),
                tag=None, smoke=True, store_x=True, device="cpu", threads=1, rerun_failed=False,
                allow_download=False, code_hash=run.code_hash())


def cli(reps, **kw):
    return subprocess.Popen([PY, str(CODE / "run.py"), "--smoke", "--scenarios", SCEN, "--reps", reps,
                             "--models", ",".join(ARMS), "--lock-wait", "0"],
                            cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)


def main():
    # 1. locks
    target = run.rep_file(OUT, SCEN, 9, None)
    target.unlink(missing_ok=True)
    lock = target.with_name(target.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(f"{run.socket.gethostname()} {os.getpid()} live")
    assert run.run_replicate(task(9))["status"] == "locked", "a live lock was not respected"
    lock.write_text(f"{run.socket.gethostname()} 4000000 dead")      # no such process
    res = run.run_replicate(task(9))
    assert res["status"] == "written" and not lock.exists(), res
    print("1. live lock respected, dead lock broken: ok")

    # 2. hard kill in the middle of a run, then resume
    for r in (5, 6, 7):
        run.rep_file(OUT, SCEN, r, None).unlink(missing_ok=True)
    p = cli("5-7")
    first = run.rep_file(OUT, SCEN, 5, None)
    t0 = time.time()
    while not first.exists() and time.time() - t0 < 300:
        time.sleep(0.5)
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    p.wait()
    done_before = [r for r in (5, 6, 7) if run.rep_file(OUT, SCEN, r, None).exists()]
    print(f"   killed after {time.time() - t0:.0f}s; finished replicates: {done_before}")
    out = cli("5-7").communicate()[0]
    lines = [ln for ln in out.splitlines() if ln.startswith(SCEN)]
    print("   resume:", lines)
    for r in done_before:
        assert any(ln == f"{SCEN} rep {r}: skipped" for ln in lines), f"rep {r} was refitted"
    assert all(run.rep_file(OUT, SCEN, r, None).exists() for r in (5, 6, 7))
    assert not list(OUT.joinpath(SCEN).glob("*.lock")) and not list(OUT.joinpath(SCEN).glob("*.tmp-*"))
    print("2. hard kill + resume: ok")

    # 3. delete one replicate and regenerate it
    f = run.rep_file(OUT, SCEN, 6, None)
    keep = f.with_name("_copy_0006.parquet")
    shutil.copy2(f, keep)
    f.unlink()
    cli("6").communicate()
    a, b = pq.read_table(keep), pq.read_table(f)
    ma, mb = (json.loads(t.schema.metadata[b"tfmcal"]) for t in (a, b))
    assert ma["fingerprints"] == mb["fingerprints"]
    da, db = a.to_pandas(), b.to_pandas()
    assert list(da.columns) == list(db.columns)
    for c in da.columns:
        if c.startswith(("p_", "ystar_", "x", "y")):
            assert np.array_equal(da[c].to_numpy(dtype=float, na_value=np.nan),
                                  db[c].to_numpy(dtype=float, na_value=np.nan), equal_nan=True), c
    keep.unlink()
    print("3. regenerated replicate identical (data, predictions, null outcomes, ladder): ok")


if __name__ == "__main__":
    main()
