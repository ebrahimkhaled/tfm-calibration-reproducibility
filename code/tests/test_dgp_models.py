"""Generator, model wrappers and recalibration ladder. Run with `python tests/run_tests.py`."""

import shutil
import sys
from pathlib import Path

import numpy as np
from scipy import special

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dgp  # noqa: E402
import metrics as M  # noqa: E402
import models  # noqa: E402
import recal  # noqa: E402

HAVE_R = shutil.which("Rscript") is not None


# ----------------------------------------------------------------------------- generator

def test_scenario_table_shape():
    t = dgp.load_table()
    assert len(t) == 58 and t.scenario_id.tolist() == list(range(1, 59))
    assert (t[t.kind == "core"].shape[0], t[t.kind == "stress"].shape[0]) == (48, 10)
    assert set(t.n_eval) == {1000, 3334}
    assert np.allclose(t.realized_prevalence, t.pi, atol=1e-9)
    assert np.allclose(t[t.pi == 0.10].oracle_auc, 0.80, atol=1e-6)


def test_replicates_are_reproducible_and_distinct():
    a = dgp.make_replicate("S3_pi10_n500", 1)
    b = dgp.make_replicate("S3_pi10_n500", 1)
    c = dgp.make_replicate("S3_pi10_n500", 2)
    s = dgp.make_replicate("S3_pi10_n500", 1, smoke=True)
    assert a.fingerprints == b.fingerprints
    assert len({a.fingerprints["all"], c.fingerprints["all"], s.fingerprints["all"]}) == 3
    assert a.fingerprints["dev"] != a.fingerprints["eval"]
    assert a.child_seed(4, 3) == b.child_seed(4, 3) != a.child_seed(4, 4)


def test_prevalence_and_auc_on_fresh_draws():
    for name in ("S1_pi03_n200", "S2_pi10_n200", "S4_pi30_n200", "T1_pi10_n500", "T3_pi10_n500"):
        sc = dgp.get_scenario(name)
        X = dgp.draw_covariates(sc["covariates"], 300_000, np.random.default_rng(99))
        p = dgp.true_risk(X, sc["family"], sc["kappa"], sc["alpha"])
        assert abs(p.mean() - sc["pi"]) < 0.004 * max(1, sc["pi"] / 0.1), (name, p.mean())
        assert abs(M.expected_auc(p, p) - sc["oracle_auc"]) < 0.005, name


def test_covariate_design():
    X = dgp.draw_covariates("normal", 200_000, np.random.default_rng(5))
    assert abs(np.corrcoef(X[:, 0], X[:, 1])[0, 1] - 0.5) < 0.01
    assert abs(X[:, 6].mean() - 0.5) < 0.01 and abs(X[:, 7].mean() - 0.3) < 0.01
    T = dgp.draw_covariates("t3", 200_000, np.random.default_rng(5))
    assert abs(np.median(T[:, :6].var(axis=0)) - 1) < 0.15          # t3: heavy tails, slow variance
    C = dgp.draw_covariates("cat6", 60_000, np.random.default_rng(5))
    assert set(np.unique(C[:, 7])) == set(range(6))
    assert dgp.draw_covariates("noise22", 10, np.random.default_rng(5)).shape == (10, 30)


# ----------------------------------------------------------------------------- arms

def _ctx(name="S1_pi10_n500", rep=1, X_pred=None):
    r = dgp.make_replicate(name, rep)
    sc = r.scenario
    Xp = r.eval.X if X_pred is None else X_pred
    return models.FitContext(
        X_dev=r.dev.X, y_dev=r.dev.y, X_pred=Xp, p_star_pred=np.zeros(len(Xp)),
        columns=sc["columns"], cont=sc["cont_cols"], bin=sc["bin_cols"], cat=sc["cat_cols"],
        cat_levels=sc["cat_levels"], seeds={a: r.child_seed(4, c) for a, c in models.ARM_CODES.items()})


def test_flic_py_matches_logistf_and_lr_py_matches_glm():
    if not HAVE_R:
        print("  (skipped: no Rscript)")
        return
    for name in ("S1_pi03_n200", "T3_pi10_n500"):
        ctx = _ctx(name)
        res = models.run_arms(["lr", "flic", "lr_py", "flic_py"], ctx)
        for a in res.values():
            assert a.status == "ok", (name, a.arm, a.info)
        assert np.max(np.abs(res["flic"].p - res["flic_py"].p)) < 1e-5, name
        assert np.max(np.abs(res["lr"].p - res["lr_py"].p)) < 1e-6, name


def test_flic_mean_prediction_equals_event_rate():
    r = dgp.make_replicate("S2_pi03_n500", 4)
    ctx = _ctx("S2_pi03_n500", 4, X_pred=r.dev.X)
    p = models.fit_flic_py(ctx)[0].p
    assert abs(p.mean() - r.dev.y.mean()) < 1e-9


def test_r_arms_on_a_categorical_scenario():
    if not HAVE_R:
        return
    res = models.run_arms(["ridge", "gam"], _ctx("T3_pi10_n500"))
    assert all(a.status == "ok" for a in res.values()), {k: v.info for k, v in res.items()}
    assert res["ridge"].info["lambda_min"] > 0


def test_lgbm_arm():
    res = models.run_arms(["lgbm"], _ctx("S2_pi10_n500"))
    assert res["lgbm"].status == "ok" and res["lgbm_raw"].status == "ok", res["lgbm"].info
    assert res["lgbm"].info["grid"] in models.LGBM_GRID


def test_a_missing_checkpoint_is_a_recorded_failure():
    import os
    old = os.environ.get("TFMCAL_TABPFN_VERSION")
    os.environ["TFMCAL_TABPFN_VERSION"] = "V3"        # pinned and absent (only V2 and 3.5 are on disk)
    try:
        out = models.fit_tabpfn(_ctx())
    finally:
        if old is None:
            del os.environ["TFMCAL_TABPFN_VERSION"]
        else:
            os.environ["TFMCAL_TABPFN_VERSION"] = old
    if out[0].status == "ok":                          # the author has put TabPFN-3.5 in place
        return
    assert out[0].p is None and "checkpoint" in out[0].info["error"]


# ----------------------------------------------------------------------------- ladder

def test_ladder_repairs_a_known_miscalibration():
    rng = np.random.default_rng(8)
    sc = dgp.get_scenario("S1_pi30_n200")
    X = dgp.draw_covariates(sc["covariates"], 6000, rng)
    ps = dgp.true_risk(X, sc["family"], sc["kappa"], sc["alpha"])
    y = (rng.random(len(ps)) < ps).astype(float)
    ph = special.expit(0.5 * special.logit(ps) - 0.3)          # under-confident and shifted
    preds, infos = recal.ladder(ph[:500], y[:500], ph[1000:], m_sizes=(500,))
    before = M.oracle_metrics(ph[1000:], ps[1000:])["true_ici"]
    after = M.oracle_metrics(preds[("R2", 500)], ps[1000:])
    assert after["true_ici"] < before / 3 and abs(after["true_slope"] - 1) < 0.25
    assert abs(infos[("R2", 500)]["b"] - 2.0) < 0.5
    # R1 reproduces the local event rate on the local sample
    p1, _ = recal.recalibrate(ph[:500], y[:500], ph[:500], "R1")
    assert abs(p1.mean() - y[:500].mean()) < 1e-9
    p3, _ = recal.recalibrate(ph[:500], y[:500], ph[:500], "R3")
    assert p3.min() >= M.EPS and p3.max() <= 1 - M.EPS


def test_ladder_single_class_sample_fails_cleanly():
    p, info = recal.recalibrate(np.full(100, 0.02), np.zeros(100), np.full(5, 0.02), "R2")
    assert p is None and "single class" in info["error"]


# ----------------------------------------------------------------------------- review fixes

def test_replicate_zero_is_refused():
    # SeedSequence pads short entropy with zeros, so replicate 0 could share another key's stream
    try:
        dgp.make_replicate("S1_pi10_n200", 0)
    except ValueError:
        return
    raise AssertionError("replicate 0 was accepted")


def test_no_event_development_set_fails_every_arm_alike():
    ctx = _ctx("S1_pi03_n200")
    ctx.y_dev = np.zeros_like(ctx.y_dev)
    res = models.run_arms(["oracle", "lr", "flic", "ridge", "gam", "lgbm", "tabpfn"], ctx)
    assert res["oracle"].status == "ok"
    for k in ("lr", "flic", "ridge", "gam", "lgbm", "lgbm_raw", "tabpfn"):
        assert res[k].status == "failed" and "single class" in res[k].info["error"], (k, res[k].info)


def test_mitra_is_reproducible_from_the_replicate_seed():
    """Mitra flips feature signs with the global numpy generator; the wrapper must seed it."""
    try:
        import autogluon.tabular  # noqa: F401
        models._hf_cached(models.MITRA_REPO)
    except Exception:  # noqa: BLE001
        print("  (skipped: no autogluon or no Mitra weights in this environment)")
        return
    import torch
    r = dgp.make_replicate("S1_pi10_n200", 1)
    ctx = _ctx("S1_pi10_n200", X_pred=r.eval.X[:200])
    ctx.device = "cuda" if torch.cuda.is_available() else "cpu"
    a = models.fit_mitra(ctx)[0]
    np.random.seed(12345)
    np.random.random(777)                     # a different global state before the second fit
    state = np.random.get_state()[1].copy()
    b = models.fit_mitra(ctx)[0]
    assert a.status == "ok" and b.status == "ok", (a.info, b.info)
    assert np.array_equal(a.p, b.p) and a.info["mirror_x"] == b.info["mirror_x"]
    assert np.array_equal(np.random.get_state()[1], state)      # the caller's stream is untouched
