"""Unit tests of the analysis rules (no pytest in the project venvs: run this file directly).

    python analysis/tests/test_rules.py

Covers: the MCSE formulas, the pairing rule, the label rules at every margin boundary, the
minimum-n rule, the 3 % threshold scaling, the size rule and the reporting rule of 6.3. The
Hodges-Lehmann interval and qsignrank are checked against R itself.
"""

from __future__ import annotations

import math
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import _common as C  # noqa: E402

sys.path.insert(0, str(C.CODE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import metrics  # noqa: E402  (the engine's own helpers)
import rules as R  # noqa: E402

TESTS = []


def test(f):
    TESTS.append(f)
    return f


def close(a, b, tol=1e-12):
    assert abs(a - b) <= tol, f"{a} != {b}"


# ----------------------------------------------------------------------------- MCSE

@test
def mcse_mean_formula():
    x = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
    close(R.mcse_mean(x), np.std(x, ddof=1) / math.sqrt(5))
    close(R.mcse_mean(x), metrics.mcse_mean(x))
    close(R.mcse_mean(np.r_[x, np.nan]), R.mcse_mean(x))     # non-finite values are left out
    assert math.isnan(R.mcse_mean([3.0]))


@test
def mcse_proportion_formula():
    close(R.mcse_prop(5, 100), math.sqrt(0.05 * 0.95 / 100))
    close(R.mcse_prop(50, 1000), metrics.mcse_proportion(50, 1000))
    close(R.mcse_prop(0, 200), 0.0)
    close(R.size_mcse(1000), math.sqrt(0.05 * 0.95 / 1000))


@test
def median_interval_matches_engine():
    rng = np.random.default_rng(3)
    x = rng.normal(size=200)
    assert R.median_ci(x) == metrics.median_ci(x)
    m, lo, hi = R.median_ci(x)
    assert lo < m < hi


@test
def median_bootstrap_mcse_is_reproducible_and_plausible():
    x = np.random.default_rng(4).normal(size=400)
    a, b = R.mcse_median_boot(x), R.mcse_median_boot(x)
    assert a == b
    # asymptotic SE of a normal median: 1.2533 / sqrt(R)
    assert 0.6 * 1.2533 / 20 < a < 1.5 * 1.2533 / 20


@test
def summary_row_fields():
    r = R.summary_row(np.arange(1, 101, dtype=float))
    close(r["mean"], 50.5)
    close(r["mean_mcse"], np.std(np.arange(1, 101), ddof=1) / 10)
    close(r["q10"], np.quantile(np.arange(1, 101), 0.1))
    close(r["iqr"], np.quantile(np.arange(1, 101), 0.75) - np.quantile(np.arange(1, 101), 0.25))
    assert r["R"] == 100 and r["n_nonfinite"] == 0


def r_eval(expr: str) -> str:
    out = subprocess.run([C.rscript(), "-e", expr], capture_output=True, text=True, timeout=120)
    return out.stdout.strip()


@test
def hodges_lehmann_matches_r_wilcox():
    x = np.array([1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30])
    y = np.array([0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29])
    rng = np.random.default_rng(11)
    d30 = np.round(rng.standard_t(3, size=30), 3)
    for d in (x - y, d30):
        vec = ",".join(f"{v:.6f}" for v in d)
        s = r_eval(f"w <- wilcox.test(c({vec}), conf.int = TRUE, exact = TRUE); "
                   f"cat(sprintf('%.10f %.10f %.10f', w$estimate, w$conf.int[1], w$conf.int[2]))")
        est_r, lo_r, hi_r = map(float, s.split())
        est, lo, hi = R.hodges_lehmann(np.round(d, 6))
        close(lo, lo_r, 1e-8)
        close(hi, hi_r, 1e-8)
        close(est, est_r, 1e-8)


@test
def qsignrank_matches_r():
    s = r_eval("cat(qsignrank(0.025, c(10, 50, 200)))")
    want = list(map(int, s.split()))
    assert [R.qsignrank(0.025, n) for n in (10, 50, 200)] == want, want


@test
def hl_point_equals_full_median():
    d = np.random.default_rng(5).normal(size=41)
    close(R.hl_point(d), R.hodges_lehmann(d)[0])
    d = np.random.default_rng(6).normal(size=40)
    close(R.hl_point(d), R.hodges_lehmann(d)[0])


# ----------------------------------------------------------------------------- transforms

@test
def abs_log_b_rules():
    v = R.abs_log_b([1.0, 1.1, 1 / 1.1, 0.0, -2.0, np.nan, np.inf])
    close(v[0], 0.0)
    close(v[1], math.log(1.1))
    close(v[2], math.log(1.1))
    assert list(v[3:]) == [10.0, 10.0, 10.0, 10.0]          # above every finite value (7.1)
    c = R.abs_log_b_capped([0.0, 1e9, 2.0])
    close(c[0], math.log(10))
    close(c[1], math.log(10))                                # capped at log 10 (6.4)
    close(c[2], math.log(2))


# ----------------------------------------------------------------------------- labels at the boundaries

@test
def label_vs_margin_boundaries():
    d = 0.01
    L = R.label_vs_margin
    assert L(-0.03, -0.0100001, d) == "better"
    assert L(-0.03, -0.01, d) == "inconclusive"             # touching -delta is not "below"
    assert L(0.0100001, 0.03, d) == "worse"
    assert L(0.01, 0.03, d) == "inconclusive"
    assert L(-0.0099999, 0.0099999, d) == "equivalent"
    assert L(-0.01, 0.005, d) == "inconclusive"             # the open interval excludes -delta
    assert L(-0.005, 0.01, d) == "inconclusive"
    assert L(-0.02, 0.02, d) == "inconclusive"
    assert L(np.nan, 0.0, d) == "inconclusive"


@test
def margins_of_each_question():
    lb = math.log(1.1)
    assert R.label_vs_margin(-0.2, -lb - 1e-9, C.MARGIN_LOGB) == "better"
    assert R.label_vs_margin(-0.2, -lb, C.MARGIN_LOGB) == "inconclusive"
    assert R.label_vs_margin(0.0500001, 0.2, C.MARGIN_ABS_A) == "worse"
    assert R.label_vs_margin(0.05, 0.2, C.MARGIN_ABS_A) == "inconclusive"
    assert R.label_vs_margin(-0.0499, 0.0499, C.MARGIN_ABS_A) == "equivalent"


@test
def ici_margin_scales_at_three_percent():
    close(C.prob_scale(0.03), 0.3)
    close(C.prob_scale(0.10), 1.0)
    close(C.prob_scale(0.30), 1.0)
    close(C.ici_margin(0.03), 0.003)
    close(C.ici_margin(0.10), 0.01)
    # the same interval is "better" at 3 % but "equivalent" at 10 %
    assert R.label_ici(-0.005, -0.0031, 0.03) == "better"
    assert R.label_ici(-0.005, -0.0031, 0.10) == "equivalent"
    assert R.label_ici(-0.005, -0.003, 0.03) == "inconclusive"
    # agrees with the engine's own scaling of the class thresholds
    for pi in (0.03, 0.10, 0.30):
        close(C.prob_scale(pi), metrics.probability_scale(pi))


@test
def class_thresholds_at_three_percent():
    base = dict(true_citl_prob=0.0, true_intercept=0.0, true_slope=1.0, true_e90=0.0)
    m = dict(base, true_ici=0.0045)
    assert metrics.hierarchy_class(m, prob_scale=C.prob_scale(0.03))["cls_moderate"]
    m = dict(base, true_ici=0.0046)
    assert not metrics.hierarchy_class(m, prob_scale=C.prob_scale(0.03))["cls_moderate"]
    assert metrics.hierarchy_class(m, prob_scale=C.prob_scale(0.10))["cls_moderate"]
    m = dict(base, true_ici=0.0, true_citl_prob=0.0031)
    assert metrics.hierarchy_class(m, prob_scale=0.3)["cls_highest"] == "none"
    assert metrics.hierarchy_class(m, prob_scale=1.0)["cls_highest"] == "strong"


@test
def direction_label_boundaries():
    D = R.direction_label
    assert D(1.1000001, 1.3) == "under-confident"
    assert D(1.1, 1.3) == "unclear"
    assert D(0.5, 0.8999999) == "over-confident"
    assert D(0.5, 0.9) == "unclear"
    assert D(0.9, 1.1) == "close to 1"                        # closed [0.9, 1.1]
    assert D(0.8999, 1.05) == "unclear"


@test
def detection_and_best_arm_and_ppv_labels():
    assert R.detection_label(0.0200001, 0.1) == "detects more often"
    assert R.detection_label(0.02, 0.1) == "inconclusive"
    assert R.detection_label(-0.1, -0.0200001) == "detects less often"
    assert R.detection_label(-0.0199, 0.0199) == "equivalently"
    assert R.best_arm_label(-0.001, 0.005) == "within 0.005 of the best arm"
    assert R.best_arm_label(0.0050001, 0.01) == "more than 0.005 above it"
    assert R.best_arm_label(0.005, 0.01) == "unclear"
    assert R.ppv_label(0.9, 0.95, 49) == "not assessable"
    assert R.ppv_label(0.8, 0.95, 50) == "at least 0.8"
    assert R.ppv_label(0.6, 0.8, 50) == "unclear"
    assert R.ppv_label(0.6, 0.7999, 50) == "below 0.8"


@test
def size_rules():
    assert R.size_label(0.06, 0.95) == "holds its size"
    assert R.size_label(0.04, 1.0) == "holds its size"
    assert R.size_label(0.0600001, 1.0) == "liberal"
    assert R.size_label(0.0399999, 1.0) == "conservative"
    assert R.size_label(0.05, 0.949) == "uneven"
    half = 2.5758293035489004 * math.sqrt(0.05 * 0.95 / 1000)
    assert R.in_size_band(0.05 + half - 1e-9, 1000)
    assert not R.in_size_band(0.05 + half + 1e-9, 1000)


@test
def transfer_rule():
    assert R.transfer_label(40) == "the grid transfers"
    assert R.transfer_label(39) == "the grid transfers in part"
    assert R.transfer_label(30) == "the grid transfers in part"
    assert R.transfer_label(29) == "the grid does not transfer"
    assert R.transfer_label(35, 42) == "the grid transfers"   # 35/42 = 40/48
    assert R.transfer_agrees(0.01, 0.002, 0.012, 0.002)
    assert not R.transfer_agrees(0.01, 0.001, 0.02, 0.001)


# ----------------------------------------------------------------------------- minimum n

@test
def min_n_rule():
    assert R.min_n({200: 0.79, 500: 0.80, 1000: 0.9, 3000: 1.0}) == 500     # 80 % qualifies
    assert R.min_n({200: 0.81, 500: 0.70}) == 200
    assert R.min_n({200: 0.1, 500: 0.2, 1000: 0.5, 3000: 0.799}) == "> 3000"
    assert R.min_n({500: 0.9}) == 500                                        # missing n do not qualify
    r = {n: 200 for n in C.NS}
    assert R.min_n({200: 0.85}, r, lower_limit=True) == 200                 # .85 - .0495 = .8005
    assert R.min_n({200: 0.84}, r, lower_limit=True) == "> 3000"            # .84 - .0508 < .8
    assert R.compare_min_n("> 3000", 3000) == "needs a larger n"
    assert R.compare_min_n(500, "> 3000") == "needs a smaller n"
    assert R.compare_min_n("> 3000", "> 3000") == "the same n"
    assert R.q2_clarity("needs a smaller n", "the same n") == "sensitive to Monte Carlo error"


@test
def nested_classes():
    m = R.meets_class(["none", "mean", "weak", "moderate", "strong"], "weak")
    assert list(m) == [False, False, True, True, True]
    assert list(R.meets_class(["strong", "mean"], "mean")) == [True, True]


# ----------------------------------------------------------------------------- pairing and failures

@test
def pairing_rule():
    a = ["ok", "degenerate", "failed", "ok", "not_run", None]
    b = ["ok", "ok", "ok", "failed", "ok", "ok"]
    assert list(R.pair_usable(a, b)) == [True, True, False, False, False, False]


@test
def build_pairs_respects_the_rule():
    from collect import build_pairs
    rows = []
    for rep, (st_t, st_f) in enumerate([("ok", "ok"), ("degenerate", "ok"), ("failed", "ok"), ("ok", "failed")], 1):
        rows.append(dict(scenario="S1_pi10_n200", rep=rep, column="tabpfn", arm="tabpfn", ladder="R0", status=st_t,
                         o_true_ici=0.02 + rep / 1000, abs_logb=0.1, abs_a=0.0))
        rows.append(dict(scenario="S1_pi10_n200", rep=rep, column="flic", arm="flic", ladder="R0", status=st_f,
                         o_true_ici=0.03, abs_logb=0.2, abs_a=0.1))
    # rep 5: the TFM has not run yet (no row): never paired
    rows.append(dict(scenario="S1_pi10_n200", rep=5, column="flic", arm="flic", ladder="R0", status="ok",
                     o_true_ici=0.03, abs_logb=0.2, abs_a=0.1))
    p = build_pairs(pd.DataFrame(rows))
    assert sorted(p["rep"]) == [1, 2], p["rep"].tolist()      # degenerate kept, failures and pending out
    close(float(p.loc[p["rep"] == 2, "d_o_true_ici"].iloc[0]), 0.022 - 0.03, 1e-15)


@test
def q1_and_q3_use_the_hl_rows_only():
    import summarise
    base = dict(kind="vs_flic", scenario="S1_pi10_n200", column="tabpfn", ref="flic", arm="tabpfn", ladder="R0",
                family="S1", pi=0.10, n=200, R_pairs=200, fail_flag=False, lo=-0.01, hi=0.01, estimate=0.0)
    paired = pd.DataFrame([dict(base, metric=m, summary=s, label="x") for m in ("abs_logb", "abs_a")
                           for s in ("mean", "HL")])
    arm_sum = pd.DataFrame([dict(scenario="S1_pi10_n200", arm=a, estimand=e, subset="all", ladder="R0",
                                 median=1.0, median_lo=0.95, median_hi=1.05, median_mcse=0.01, R=200)
                            for a in ("tabpfn", "flic") for e in ("o_true_slope", "o_true_intercept")])
    assert len(summarise.q1(paired, arm_sum)) == 1
    assert (summarise.q1(paired, arm_sum)["summary"] == "HL").all()
    assert len(summarise.q3(paired, arm_sum)) == 1


@test
def failure_flag():
    assert R.fail_flag(26, 500)
    assert not R.fail_flag(25, 500)                          # exactly 5 % is not "more than 5 %"
    assert not R.fail_flag(0, 0)


# ----------------------------------------------------------------------------- reporting rule (6.3)

@test
def reporting_rule():
    assert R.proportion_bound(0, 200) == "floor"
    assert R.proportion_bound(200, 200) == "ceiling"
    assert R.proportion_bound(3, 200) is None
    assert R.pair_at_bound(0.985, 1.0, 0.99, 1.0, 0.02) == "ceiling"
    assert R.pair_at_bound(0.95, 1.0, 0.99, 1.0, 0.02) is None
    assert R.pair_at_bound(0.0, 0.01, 0.0, 0.02, 0.02) == "floor"
    try:
        R.statement("equivalent", 0.001, float("nan"), float("nan"))
        raise AssertionError("an 'equivalent' statement without an interval was allowed")
    except ValueError:
        pass
    assert R.statement("holds its size", 0.05, 0.045, 0.055).startswith("holds its size")
    assert R.statement("equivalent", 0.0, -0.001, 0.001, bound="floor").startswith("uninformative (floor)")
    assert R.ici_at_floor(0.001, 0.0012, 0.0015)
    assert not R.ici_at_floor(0.001, 0.0100, 0.0015)


# ----------------------------------------------------------------------------- run

def main() -> int:
    bad = 0
    for f in TESTS:
        try:
            f()
            print(f"ok    {f.__name__}")
        except Exception:
            bad += 1
            print(f"FAIL  {f.__name__}\n{traceback.format_exc()}")
    print(f"{len(TESTS) - bad} of {len(TESTS)} tests passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
