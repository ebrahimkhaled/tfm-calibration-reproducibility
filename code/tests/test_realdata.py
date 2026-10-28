"""Real-data stream (code/run_realdata.py, code/realdata.py). Run with `python tests/run_tests.py`.

Uses the clean tables in data/clean (built by realdata.py); the tests that need a table skip
quietly when it is missing. No GPU is used; the end-to-end test needs Rscript for the R arms."""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models  # noqa: E402
import realdata as rd  # noqa: E402
import run_realdata as R  # noqa: E402

HAVE_R = shutil.which("Rscript") is not None


def _have(key):
    return (rd.CLEAN / f"{key}.parquet").exists()


# ----------------------------------------------------------------------------- loaders

def test_ist_polish_month_dates():
    s = pd.Series(["sty-91", "lut-92", "mar-93", "kwi-94", "maj-95", "cze-96", "lip-91", "sie-92",
                   "wrz-93", "pa\x9f-94", "lis-95", "gru-96"])
    y = rd._rdate_year(s)
    want = [1991 + 0 / 12, 1992 + 1 / 12, 1993 + 2 / 12, 1994 + 3 / 12, 1995 + 4 / 12, 1996 + 5 / 12,
            1991 + 6 / 12, 1992 + 7 / 12, 1993 + 8 / 12, 1994 + 9 / 12, 1995 + 10 / 12, 1996 + 11 / 12]
    assert np.allclose(y.to_numpy(), want)


def test_clean_tables_match_their_fingerprints():
    for key in R.KEYS:
        if _have(key):
            df, meta = rd.load(key)            # raises on a fingerprint mismatch
            assert set(df[R.SPLIT_COL].unique()) == {"dev_pool", "ext_eval", "ext_recal"}
            assert int((df[R.SPLIT_COL] == "ext_recal").sum()) == rd.N_RECAL


def test_ist_regions_cover_every_country():
    if not _have("ist_dd6"):
        return
    df, _ = rd.load("ist_dd6")
    assert set(df["country"].unique()) <= set(R.IST_REGION)
    ext = df[df[R.SPLIT_COL] != "dev_pool"]
    assert (ext["country"].map(R.IST_REGION) >= 0).all()          # the UK is never external


# ----------------------------------------------------------------------------- the regression design

def test_aliased_free_drops_constant_and_dependent_columns():
    rng = np.random.default_rng(1)
    a = rng.normal(size=200)
    b = rng.integers(0, 2, 200).astype(float)
    X = np.column_stack([a * 1e4, np.zeros(200), b, b, 2 * a + 3, rng.normal(size=200)])
    keep = R.aliased_free(X)
    assert keep.tolist() == [True, False, True, False, False, True]


def test_regression_context_drops_a_level_absent_from_development():
    rng = np.random.default_rng(2)
    n = 300
    X_dev = np.column_stack([rng.normal(size=n), rng.integers(0, 2, n), rng.integers(0, 2, n)])  # cat: codes 0, 1
    X_pred = np.column_stack([rng.normal(size=50), rng.integers(0, 2, 50), rng.integers(0, 3, 50)])  # code 2 unseen
    ctx = models.FitContext(X_dev=X_dev, y_dev=(rng.random(n) < 0.3).astype(int), X_pred=X_pred,
                            p_star_pred=np.zeros(50), columns=["age", "female", "stype"], cont=["age"],
                            bin=["female"], cat=["stype"], cat_levels={"stype": 3}, seeds={})
    ctx_r, dropped = R.regression_context(ctx)
    assert dropped == ["stype_2"]
    assert ctx_r.columns == ["age", "female", "stype_1"] and ctx_r.cat == []
    assert ctx_r.X_pred.shape == (50, 3)
    # a row with the unseen level is coded like the reference level
    unseen = X_pred[:, 2] == 2
    assert np.all(ctx_r.X_pred[unseen, 2] == 0)


# ----------------------------------------------------------------------------- replicates

def test_learning_curve_replicate_is_reproducible_and_uses_subsample_b():
    key = "physionet_ihd"
    if not _have(key):
        return
    r1 = R.build_replicate(key, "lc", None, 200, 3)
    r2 = R.build_replicate(key, "lc", None, 200, 3)
    r3 = R.build_replicate(key, "lc", None, 200, 4)
    df, meta = rd.load(key)
    assert np.array_equal(r1["dev_idx"], rd.subsample_dev(df, key, 200, 2))     # rep 3 = b 2
    assert r1["fingerprints"] == r2["fingerprints"] and np.array_equal(r1["X_ext"], r2["X_ext"])
    assert r1["fingerprints"]["dev"] != r3["fingerprints"]["dev"]
    assert r1["fingerprints"]["eval_y"] == r3["fingerprints"]["eval_y"]           # one external set
    assert not set(r1["dev_idx"]) & set(r1["ev_idx"]) and not set(r1["ev_idx"]) & set(r1["rc_idx"])
    assert len(r1["rc_idx"]) == rd.N_RECAL and len(r1["ev_idx"]) == meta["splits"]["split"]["ext_eval"]["n"]
    assert not np.isnan(r1["X_dev"]).any() and not np.isnan(r1["X_ext"]).any()   # imputed
    assert any(c.endswith("_missing") for c in r1["bin"])
    assert not r1["loc"].isna().any().any()


def test_plasmode_replicate_uses_the_frozen_truth():
    key = "physionet_ihd"
    if not (rd.CLEAN / f"plasmode_{key}.json").exists():
        return
    r = R.build_replicate(key, "pl", "30", 500, 7)
    p, y = rd.plasmode_outcomes(key, "30", 7)
    assert np.array_equal(r["y_all"], y.astype(np.int8)) and np.array_equal(r["p_star"], p)
    assert np.array_equal(r["y_dev"], y[r["dev_idx"]])
    lc = R.build_replicate(key, "lc", None, 500, 7)
    assert not np.array_equal(lc["dev_idx"], r["dev_idx"])                     # own subsample seed


def test_task_seed_is_stable_and_fits_in_r():
    s = R.task_seed("ist_dd6__lc_n200", 1, "flic")
    assert s == R.task_seed("ist_dd6__lc_n200", 1, "flic") and 0 <= s < 2 ** 31 - 1
    assert s != R.task_seed("ist_dd6__lc_n200", 2, "flic")


def test_setting_names_round_trip():
    for key, part, tag, n in [("ist_dd6", "lc", None, 200), ("crash2_death28", "pl", "10", 3000)]:
        s = R.setting_name(key, part, n, tag)
        assert R.parse_setting(s) == dict(key=key, part=part, tag=tag, n=n)


# ----------------------------------------------------------------------------- chunked TFM predictions

def test_fit_chunked_halves_on_out_of_memory_and_keeps_the_row_order():
    calls = []
    orig = models.run_arms

    def fake(arms, ctx):
        (a,) = arms
        calls.append(len(ctx.X_pred))
        if len(ctx.X_pred) > 1500:
            return {a: models.ArmResult(a, None, "failed", {"error": "CUDA out of memory", "seconds": 0.1})}
        return {a: models.ArmResult(a, 1 / (1 + np.exp(-ctx.X_pred[:, 0])), "ok",
                                    {"seconds": 1.0, "fit_seconds": 0.5, "predict_seconds": 0.5})}
    models.run_arms = fake
    try:
        X = np.random.default_rng(3).normal(size=(4000, 2))
        ctx = models.FitContext(X_dev=X[:10], y_dev=np.r_[np.zeros(5), np.ones(5)].astype(int), X_pred=X,
                                p_star_pred=np.zeros(4000), columns=["a", "b"], cont=["a", "b"], bin=[],
                                cat=[], cat_levels={}, seeds={})
        out = R.fit_chunked("tabpfn", ctx)["tabpfn"]
    finally:
        models.run_arms = orig
    assert out.status == "ok" and out.info["predict_chunk"] == 1000 and out.info["chunks_tried"] == [4000, 2000, 1000]
    assert np.allclose(out.p, 1 / (1 + np.exp(-X[:, 0])))
    assert out.info["seconds"] == 4.0 and out.info["n_chunks"] == 4


def test_empirical_class_mapping():
    em = dict(emp_citl_prob=0.005, emp_intercept=0.05, emp_slope=1.05, emp_ici=0.01, emp_e90=0.02)
    c = R.empirical_class(em)
    assert c["emp_cls_highest"] == "strong"
    em["emp_slope"] = float("nan")
    assert R.empirical_class(em)["emp_cls_highest"] == "mean"


# ----------------------------------------------------------------------------- end to end (CPU arms)

def test_end_to_end_classical_replicate_resume_and_score():
    key = "sepsis_death"
    if not (HAVE_R and _have(key)):
        return
    tmp = Path(tempfile.mkdtemp(prefix="rd_test_", dir=str(R.REBUILD / "tmp")))
    old = R.RESULTS
    R.RESULTS = tmp
    try:
        task = dict(key=key, part="lc", tag=None, n=200, rep=1, arms=["lr", "flic"], group="cpu",
                    device="cpu", threads=1, rerun_failed=False, code_hash="test")
        res = R.run_task(task)
        assert res["status"] == "written", res
        assert R.run_task(task)["status"] == "skipped"                         # resume
        res2 = R.run_task(dict(task, group="other"))                              # never twice
        assert res2["status"] == "skipped"
        f = R.rep_path(key, R.setting_name(key, "lc", 200), "cpu", 1)
        meta = R.read_meta(f)
        assert meta["arms"]["flic"]["status"] in models.USABLE_STATUSES
        assert set(meta["ladder"]["flic"]) == {"R1_m100", "R2_m100", "R3_m100", "R1_m500", "R2_m500", "R3_m500"}
        d = pd.read_parquet(f)
        ev = d[d["set"] == "eval"]
        assert len(ev) == 18551 and ev["x_age"].notna().all() and ev["l_ageband"].notna().all()
        assert d.loc[d["set"] == "recal", "x_age"].isna().all()
        rows = R.score_replicate([f])
        cols = {r["column"] for r in rows}
        assert {"lr", "flic", "flic__R2_m500"} <= cols
        r_flic = next(r for r in rows if r["column"] == "flic")
        assert 0 < r_flic["e_emp_ici"] < 0.5 and "o_true_ici" not in r_flic      # no truth in lc
    finally:
        R.RESULTS = old
        shutil.rmtree(tmp, ignore_errors=True)


def test_infrastructure_errors_are_told_apart_from_model_failures():
    """Host memory and CUDA runtime errors are the machine's (never stored); a CUDA out-of-memory
    error on a free card means the model does not fit (stored after the smallest chunk)."""
    infra = ["Rscript exit 1: Out of memory.\n", "'R_Calloc' could not allocate memory (9600 of 8 bytes)",
             "RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling cublasGemmEx",
             "MemoryError: Unable to allocate 2.41 MiB for an array", "infra: GPU busy before the fit (950 of 6144 MB free)",
             "RuntimeError: Attempted to access the data pointer on an invalid python storage."]
    model = ["AcceleratorError: CUDA error: out of memory", "TabPFNCUDAOutOfMemoryError: CUDA out of memory with 500 test samples",
             "RuntimeError: Mitra cut its context to 1024 of 3000 development rows after a GPU out-of-memory error",
             "glm.fit: algorithm did not converge", "development set has a single class (0 events)"]
    assert all(R.is_infra_error({"error": e}) for e in infra)
    assert not any(R.is_infra_error({"error": e}) for e in model)
