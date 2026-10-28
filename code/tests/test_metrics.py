"""Metrics on cases with known answers. Run with `python tests/run_tests.py` (or pytest)."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import optimize, special

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dgp  # noqa: E402
import metrics as M  # noqa: E402


def _truth(n=3334, seed=11, scen="S2_pi03_n200"):
    """p* of a real evaluation-sized draw (S2 at 3 %: skewed, non-linear)."""
    sc = dgp.get_scenario(scen)
    rng = np.random.default_rng(seed)
    X = dgp.draw_covariates(sc["covariates"], n, rng)
    return dgp.true_risk(X, sc["family"], sc["kappa"], sc["alpha"])


def test_loess_reproduces_a_quadratic_exactly():
    x = np.random.default_rng(1).normal(size=400)
    y = 1 + 2 * x - 3 * x ** 2
    assert np.max(np.abs(M.loess(x, y) - y)) < 1e-8


def test_loess_matches_r_direct():
    rs = shutil.which("Rscript")
    if rs is None:
        print("  (skipped: no Rscript)")
        return
    rng = np.random.default_rng(7)
    x = rng.normal(size=600)
    y = (rng.random(600) < special.expit(-1 + 1.5 * x)).astype(float)
    with tempfile.TemporaryDirectory() as d:
        np.savetxt(os.path.join(d, "a.csv"), np.column_stack([x, y]), delimiter=",", fmt="%.17g")
        code = ("d <- read.csv(file.path(Sys.getenv('D'), 'a.csv'), header = FALSE);"
                "f <- predict(loess(V2 ~ V1, d, degree = 2, span = 0.75, family = 'gaussian', surface = 'direct'));"
                "write.csv(f, file.path(Sys.getenv('D'), 'b.csv'), row.names = FALSE)")
        Path(d, "s.R").write_text(code)
        subprocess.run([rs, os.path.join(d, "s.R")], check=True, env={**os.environ, "D": d},
                       capture_output=True)
        r = np.genfromtxt(os.path.join(d, "b.csv"), delimiter=",", skip_header=1)
    assert np.max(np.abs(M.loess(x, y, exact=True) - r)) < 1e-10


def test_loess_anchor_spline_matches_exact_fit():
    """The fast path (anchors + cubic spline) against the exact fit at every point."""
    rng = np.random.default_rng(12)
    for n in (1000, 3334):
        ps = _truth(n=n, seed=n)
        x = special.logit(np.clip(ps, 1e-6, 1 - 1e-6))
        for y in ((rng.random(n) < ps).astype(float),                    # noisy 0/1 response
                  ps):                                                   # smooth response
            fast = np.clip(M.loess(x, y, exact=False), 0, 1)
            full = np.clip(M.loess(x, y, exact=True), 0, 1)
            assert np.max(np.abs(fast - full)) < 3e-4, (n, np.max(np.abs(fast - full)))
            ici_gap = abs(np.mean(np.abs(ps - fast)) - np.mean(np.abs(ps - full)))
            assert ici_gap < 2e-6, (n, ici_gap)


def test_smoother_matrix_matches_reference_and_cache():
    rng = np.random.default_rng(5)
    for n, deg in [(1000, 2), (3334, 2), (400, 1)]:
        x = rng.normal(size=n)
        xn = np.r_[x[:200], np.round(x[:40], 1)]
        for y in (special.expit(x + 0.3 * x ** 2), rng.binomial(1, 0.2, n).astype(float)):
            ref = M._loess_direct_reference(x, y, xn, 0.75, deg)
            assert np.max(np.abs(M._loess_direct(x, y, xn, 0.75, deg) - ref)) < 1e-12


def test_perfect_predictions():
    ps = _truth()
    om = M.oracle_metrics(ps, ps)
    assert abs(om["true_slope"] - 1) < 1e-8, om["true_slope"]
    assert abs(om["true_intercept"]) < 1e-8
    assert abs(om["true_intercept_2p"]) < 1e-8
    assert om["oce"] == 0 and om["true_citl_prob"] == 0
    assert om["true_auc"] == om["oracle_auc"]
    # logit-scale smoother: exactly zero for the oracle
    assert om["true_ll_ici"] < 1e-12 and om["true_ll_emax"] < 1e-10
    # pre-registered smoother (p* on logit p_hat, span 0.75): a floor from the curvature of the
    # logistic curve, about 0.001 here; reported as the oracle row, not an error of the code
    assert 0 < om["true_ici"] < 0.002, om["true_ici"]
    cls = M.hierarchy_class(om)
    assert cls["cls_highest"] == "strong" and all(cls[k] for k in ("cls_mean", "cls_weak", "cls_moderate"))


def test_known_scaled_logit_is_recovered():
    """p_hat = expit((logit p* - a0) / b0), so logit p* = a0 + b0 logit p_hat exactly."""
    ps = _truth()
    for a0, b0 in ((0.3, 1.5), (-0.2, 0.7), (0.0, 2.5)):
        ph = special.expit((special.logit(ps) - a0) / b0)
        om = M.oracle_metrics(ph, ps)
        # exact unless predictions were clipped at 1e-6 (then only nearly)
        tol = 1e-7 if om["n_clipped"] == 0 else 1e-4
        assert abs(om["true_slope"] - b0) < tol, (b0, om["true_slope"], om["n_clipped"])
        assert abs(om["true_intercept_2p"] - a0) < tol
        # the offset intercept is the root of sum(p* - expit(a + logit p_hat)) = 0
        lp = special.logit(np.clip(ph, M.EPS, 1 - M.EPS))
        a_ref = optimize.brentq(lambda a: np.sum(ps - special.expit(a + lp)), -10, 10, xtol=1e-14)
        assert abs(om["true_intercept"] - a_ref) < 1e-8
        # c(p_hat) = p* here, so the logit-scale true ICI equals the OCE (clipping aside), and the
        # pre-registered one equals it up to the loess floor
        assert abs(om["true_ll_ici"] - om["oce"]) < (1e-9 if om["n_clipped"] == 0 else 1e-5)
        assert abs(om["true_ici"] - om["oce"]) < 0.0025, (om["true_ici"], om["oce"])


def test_known_intercept_shift():
    ps = _truth()
    ph = special.expit(special.logit(ps) - 0.4)
    om = M.oracle_metrics(ph, ps)
    assert abs(om["true_intercept"] - 0.4) < 1e-8
    assert abs(om["true_slope"] - 1) < 1e-8
    assert not M.hierarchy_class(om)["cls_weak"]


def test_under_and_over_confidence_signs():
    ps = _truth()
    shrunk = special.expit(0.5 * special.logit(ps) + 0.5 * special.logit(ps.mean()))
    om = M.oracle_metrics(shrunk, ps)
    assert om["true_slope"] > 1.5                          # under-confident: b* > 1
    sharp = special.expit(1.6 * (special.logit(ps) - special.logit(ps.mean())) + special.logit(ps.mean()))
    assert M.oracle_metrics(sharp, ps)["true_slope"] < 0.8


def test_clipping_is_counted():
    ps = _truth(n=1000)
    ph = ps.copy()
    ph[:3] = 0.0
    ph[3:5] = 1.0
    om = M.oracle_metrics(ph, ps)
    assert om["n_clipped"] == 5 and np.isfinite(om["true_slope"])


def test_constant_predictions():
    ps = _truth(n=1000)
    om = M.oracle_metrics(np.full(1000, ps.mean()), ps)
    assert abs(om["true_citl_prob"]) < 1e-12 and abs(om["true_intercept"]) < 0.5


def test_auc_against_ranks():
    rng = np.random.default_rng(3)
    s = np.round(rng.normal(size=500), 1)                  # many ties
    y = (rng.random(500) < special.expit(s)).astype(float)
    pos, neg = s[y == 1], s[y == 0]
    ref = (np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg))
    assert abs(M.expected_auc(s, y) - ref) < 1e-12


def test_expected_auc_equals_mean_empirical_auc():
    """The oracle AUC is the expected empirical AUC over outcome draws (approximately)."""
    ps = _truth(n=2000, scen="S1_pi30_n200")
    rng = np.random.default_rng(5)
    emp = [M.expected_auc(ps, (rng.random(len(ps)) < ps).astype(float)) for _ in range(300)]
    assert abs(np.mean(emp) - M.expected_auc(ps, ps)) < 0.004


def test_empirical_metrics_large_sample():
    ps = _truth(n=20_000, scen="S1_pi30_n200")
    y = (np.random.default_rng(9).random(len(ps)) < ps).astype(float)
    em = M.empirical_metrics(y, ps)
    assert abs(em["emp_slope"] - 1) < 0.08 and abs(em["emp_intercept"]) < 0.05
    assert em["emp_ici"] < 0.01
    assert abs(em["brier"] - np.mean((ps - y) ** 2)) < 1e-15


def test_fit_logistic_matches_statsmodels():
    import statsmodels.api as sm
    rng = np.random.default_rng(4)
    x = rng.normal(size=800)
    y = (rng.random(800) < special.expit(-1 + 0.8 * x)).astype(float)
    a, b, ok = M.fit_logistic(x, y)
    ref = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial()).fit().params
    assert ok and abs(a - ref[0]) < 1e-8 and abs(b - ref[1]) < 1e-8


def test_ece_known_value():
    p = np.array([0.05, 0.05, 0.95, 0.95])
    y = np.array([0, 0, 1, 0])
    assert abs(M.ece(y, p) - (2 * 0.05 + 2 * 0.45) / 4) < 1e-12


def test_mcse_helpers():
    rng = np.random.default_rng(2)
    x = rng.normal(size=400)
    assert abs(M.mcse_mean(x) - np.std(x, ddof=1) / 20) < 1e-12
    assert abs(M.mcse_proportion(10, 200) - np.sqrt(0.05 * 0.95 / 200)) < 1e-12
    d, se = M.mcse_paired_diff(x + 1, x)
    assert abs(d - 1) < 1e-12 and se < 1e-12               # paired: the noise cancels
    med, lo, hi = M.median_ci(x)
    assert lo <= med <= hi
    assert 0 < M.mcse_bootstrap(x) < 0.2
    assert abs(M.mcse_sd(x) - np.std(x, ddof=1) / np.sqrt(2 * 399)) < 1e-12


def test_hierarchy_scaling():
    m = dict(true_citl_prob=0.008, true_intercept=0.05, true_slope=1.15, true_ici=0.012, true_e90=0.02)
    c = M.hierarchy_class(m)
    assert c["cls_mean"] and not c["cls_weak"] and c["cls_moderate"] and c["cls_strong"]
    assert c["cls_highest"] == "mean"                    # nested: a failed "weak" caps the class
    c2 = M.hierarchy_class(m, scale=2.0)
    assert c2["cls_weak"] and c2["cls_highest"] == "strong"
    c3 = M.hierarchy_class(m, scale=0.5)
    assert not c3["cls_mean"] and c3["cls_highest"] == "none"
