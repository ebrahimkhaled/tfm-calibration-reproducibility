"""Fixes decided by the author on 2026-10-07, before the pre-registration freeze:
LightGBM Platt slope <= 0, the ridge lambda path, the "degenerate" label and the two
temperature arms. Run with `python tests/run_tests.py`."""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import special, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dgp  # noqa: E402
import models  # noqa: E402
import run  # noqa: E402
import score  # noqa: E402

HAVE_R = shutil.which("Rscript") is not None


def _ctx(name, rep=1, smoke=True, X_pred=None, device="cpu"):
    r = dgp.make_replicate(name, rep, smoke=smoke)
    sc = r.scenario
    Xp = r.eval.X if X_pred is None else X_pred
    return models.FitContext(
        X_dev=r.dev.X, y_dev=r.dev.y, X_pred=Xp, p_star_pred=np.zeros(len(Xp)),
        columns=sc["columns"], cont=sc["cont_cols"], bin=sc["bin_cols"], cat=sc["cat_cols"],
        cat_levels=sc["cat_levels"], seeds={a: r.child_seed(4, c) for a, c in models.ARM_CODES.items()},
        device=device)


def _have(pkg):
    try:
        __import__(pkg)
        return True
    except Exception:  # noqa: BLE001
        return False


# ----------------------------------------------------------------------------- (a) Platt slope

def test_platt_slope_below_zero_keeps_the_raw_scores():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.2).astype(int)
    oof = np.where(y == 1, 0.1, 0.3) + rng.normal(0, 0.08, 400)    # scores that rank backwards
    raw = rng.uniform(0.01, 0.6, 50)
    p, info = models.platt_or_raw(oof, y, raw)
    assert info["platt_converged"] and info["platt_b"] < 0
    assert info["platt_fallback"] and "<= 0" in info["platt_fallback_reason"]
    assert np.array_equal(p, raw)                                  # ranking (and values) kept


def test_platt_with_a_positive_slope_is_monotone_increasing():
    rng = np.random.default_rng(2)
    y = (rng.random(400) < 0.2).astype(int)
    oof = np.clip(0.15 + 0.2 * y + rng.normal(0, 0.1, 400), 0.01, 0.99)
    raw = np.sort(rng.uniform(0.01, 0.6, 50))
    p, info = models.platt_or_raw(oof, y, raw)
    assert info["platt_b"] > 0 and not info["platt_fallback"]
    assert np.all(np.diff(p) > 0) and stats.spearmanr(p, raw).statistic > 1 - 1e-12


def test_lgbm_never_inverts_the_ranking_at_tiny_n():
    # n = 200 at pi = 3 % (about 6 events): the pilot's three inverted replicates came from here
    for rep in range(1, 7):
        res = models.run_arms(["lgbm"], _ctx("S1_pi03_n200", rep))
        if res["lgbm"].status != "ok":
            continue
        rho = stats.spearmanr(res["lgbm"].p, res["lgbm_raw"].p).statistic
        assert rho > 0.999 or np.allclose(res["lgbm"].p, res["lgbm_raw"].p), (rep, rho, res["lgbm"].info)
        assert res["lgbm"].info["platt_fallback"] == (not res["lgbm"].info["platt_b"] > 0)


# ----------------------------------------------------------------------------- (b) ridge path

def test_ridge_optimum_is_inside_the_extended_lambda_path():
    if not HAVE_R:
        print("  (skipped: no Rscript)")
        return
    # S3 at pi = 30 %, n = 1000: 19 of 20 pilot fits sat at the lower end of glmnet's default path
    for rep in (1, 2, 3):
        info = models.run_arms(["ridge"], _ctx("S3_pi30_n1000", rep))["ridge"].info
        assert info["status"] == "ok", info
        assert info["lambda_path_length"] == 200
        assert abs(info["lambda_path_min"] / info["lambda_path_max"] - 1e-6) < 1e-9
        assert not info["lambda_at_edge"] and 1 < info["lambda_min_index"] < 200, info


# ----------------------------------------------------------------------------- (c) degenerate

def test_degenerate_rule():
    assert models.degenerate_reason({}, 100) is None
    assert "101 evaluation predictions" in models.degenerate_reason({}, 101)
    assert "separation" in models.degenerate_reason({"separation_flag": True}, 0)
    assert models.degenerate_reason({"separation_flag": False}, 3) is None


def test_degenerate_fits_are_labelled_scored_and_counted():
    if not HAVE_R:
        print("  (skipped: no Rscript)")
        return
    out = Path(tempfile.mkdtemp(prefix="degen_", dir=run.REBUILD / "tmp"))
    try:
        seen = {"degenerate": 0, "ok": 0}
        for rep in range(1, 7):
            task = dict(scenario="S1_pi03_n200", rep=rep, arms=["oracle", "lr", "gam"], ladder_arms=[],
                        out_dir=str(out), tag=None, smoke=True, store_x=False, device="cpu", threads=1,
                        rerun_failed=False, allow_download=False, code_hash="test")
            res = run.run_replicate(task)
            assert res["status"] == "written", res
            meta = run.read_meta(Path(res["file"]))
            for arm in ("lr", "gam"):
                info = meta["arms"][arm]
                if info["status"] == "failed":
                    continue
                expect = "degenerate" if (info["n_boundary_eval"] > 100 or info.get("separation_flag")) else "ok"
                assert info["status"] == expect, (rep, arm, info)
                seen[info["status"]] += 1
            assert meta["arms"]["oracle"]["status"] == "ok"
        # the pilot saw 20 of 40 such fits at this scenario; six replicates must show at least one
        assert seen["degenerate"] >= 1, seen
        rows = [r for f in sorted((out / "S1_pi03_n200").glob("*.parquet")) for r in score.score_file(f)]
        deg = [r for r in rows if r["ladder"] == "R0" and r["status"] == "degenerate"]
        assert len(deg) == seen["degenerate"]
        assert all(r["degenerate"] and np.isfinite(r["o_true_ici"]) for r in deg)   # scored, not dropped
    finally:
        shutil.rmtree(out, ignore_errors=True)


# ----------------------------------------------------------------------------- temperature arms

def test_temperature_arm_codes_and_sets():
    assert models.ARM_CODES["tabpfn_t1"] == 14 and models.ARM_CODES["tabicl_t1"] == 15
    assert run.MODEL_SETS["temperature"] == ["tabpfn_t1", "tabicl_t1"]
    assert models.TABPFN_T_ARM_TEMPERATURE == 0.9 and models.TABICL_T_ARM_TEMPERATURE == 1.0


def _logit_slope(p_new, p_ref):
    a, b = special.logit(np.clip(p_ref, 1e-6, 1 - 1e-6)), special.logit(np.clip(p_new, 1e-6, 1 - 1e-6))
    return np.polyfit(a, b, 1)[0]


def test_tabicl_t1_differs_from_tabicl_only_in_the_temperature():
    if not _have("tabicl"):
        print("  (skipped: tabicl not installed here)")
        return
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    r = dgp.make_replicate("S2_pi10_n500", 1, smoke=True)
    ctx = _ctx("S2_pi10_n500", X_pred=r.eval.X[:300], device=dev)
    a, b = models.fit_tabicl(ctx)[0], models.fit_tabicl_t1(ctx)[0]
    assert a.status == "ok" and b.status == "ok", (a.info, b.info)
    assert a.info["softmax_temperature"] == 0.9 and b.info["softmax_temperature"] == 1.0
    assert b.arm == "tabicl_t1"
    # dividing every logit by 0.9 instead of 1.0 stretches logit p by 1/0.9
    s = _logit_slope(a.p, b.p)
    assert abs(s - 1 / 0.9) < 0.02, s


def test_tabpfn_t1_is_tabpfn35_at_temperature_09():
    if not _have("tabpfn"):
        print("  (skipped: tabpfn not installed here)")
        return
    old = os.environ.get("TFMCAL_TABPFN_VERSION")
    os.environ["TFMCAL_TABPFN_VERSION"] = "V3_5"
    try:
        try:
            models.resolve_tabpfn("tabpfn")
        except FileNotFoundError:
            print("  (skipped: TabPFN-3.5 checkpoint absent)")
            return
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        r = dgp.make_replicate("S2_pi10_n500", 1, smoke=True)
        ctx = _ctx("S2_pi10_n500", X_pred=r.eval.X[:300], device=dev)
        a, b = models.fit_tabpfn(ctx)[0], models.fit_tabpfn_t1(ctx)[0]
    finally:
        if old is None:
            os.environ.pop("TFMCAL_TABPFN_VERSION", None)
        else:
            os.environ["TFMCAL_TABPFN_VERSION"] = old
    assert a.status == "ok" and b.status == "ok", (a.info, b.info)
    assert a.info["model_version"] == b.info["model_version"] == "V3_5"
    assert a.info["checkpoint"] == b.info["checkpoint"] == "tabpfn-v3.5-20260909.safetensors"
    assert abs(a.info["softmax_temperature"] - 1.0) < 1e-9, a.info
    assert abs(b.info["softmax_temperature"] - 0.9) < 1e-9, b.info
    s = _logit_slope(b.p, a.p)
    assert abs(s - 1 / 0.9) < 0.03, s


# ----------------------------------------------------------------------------- D6 thresholds

def test_probability_thresholds_are_relative_at_three_percent():
    import metrics as M
    assert abs(M.probability_scale(0.03) - 0.3) < 1e-12
    assert M.probability_scale(0.10) == 1.0 and M.probability_scale(0.30) == 1.0
    m = dict(true_citl_prob=0.005, true_intercept=0.05, true_slope=1.05, true_ici=0.006, true_e90=0.012)
    assert M.hierarchy_class(m)["cls_highest"] == "strong"                       # absolute rules
    c = M.hierarchy_class(m, prob_scale=M.probability_scale(0.03))              # 0.003 / 0.0045 / 0.009
    assert c["cls_highest"] == "none" and c["cls_weak"] and not c["cls_mean"]
    m2 = dict(m, true_citl_prob=0.002)
    assert M.hierarchy_class(m2, prob_scale=0.3)["cls_highest"] == "weak"       # ICI 0.006 > 0.0045


# ----------------------------------------------------------------------------- Mitra at T2, n = 3000

def test_mitra_at_t2_n3000_is_recorded_as_not_run_without_a_fit():
    out = Path(tempfile.mkdtemp(prefix="notrun_", dir=run.REBUILD / "tmp"))
    try:
        task = dict(scenario="T2_pi10_n3000", rep=1, arms=["oracle", "mitra"], ladder_arms=["mitra"],
                    out_dir=str(out), tag=None, smoke=True, store_x=False, device="cpu", threads=1,
                    rerun_failed=True, allow_download=False, code_hash="test")
        res = run.run_replicate(task)
        assert res["status"] == "written", res
        meta = run.read_meta(Path(res["file"]))
        assert meta["arms"]["mitra"]["status"] == "not_run"
        assert "6 GB" in meta["arms"]["mitra"]["error"] and "mitra" not in meta["ladder"]
        assert run.run_replicate(task)["status"] == "skipped"          # never retried, even with --rerun-failed
        rows = score.score_file(Path(res["file"]))
        assert [r["status"] for r in rows if r["arm"] == "mitra"] == ["not_run"]
    finally:
        shutil.rmtree(out, ignore_errors=True)
