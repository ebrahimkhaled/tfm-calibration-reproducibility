"""The fixed summaries and reporting rules of the pre-registration, as plain functions.

Nothing in this file reads data. Every function is a rule written down before the main run was
seen (pre-registration Sections 4.5, 6.1, 6.3, 7.0-7.6, 8.2), and tests/test_rules.py checks each
one on hand-made cases, including the exact margin boundaries.

Interval conventions, taken from the wording of Section 7.0:
  better      the whole 95 % interval lies below -delta          (hi < -delta)
  worse       the whole interval lies above +delta               (lo > +delta)
  equivalent  the whole interval lies inside the open (-delta, +delta)
  inconclusive otherwise (an interval that touches a margin is never better/worse/equivalent)
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from _common import (BEST_ARM_MARGIN, DETECTION_MARGIN, FAIL_FLAG_SHARE, LOGB_CAP_META, LOGB_FAIL,
                     MIN_N_SHARE, NS, PPV_MIN_FLAGGED, PPV_TARGET, SIZE_BAND, SIZE_SCEN_SHARE,
                     SLOPE_TOL, TRANSFER_CUTS, ALPHA, prob_scale, ici_margin)  # noqa: F401

Z95 = 1.959963984540054
Z99 = 2.5758293035489004

# ----------------------------------------------------------------------------- MCSE (6.1)


def finite(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def mcse_mean(x) -> float:
    """SD / sqrt(R) over the finite values."""
    x = finite(x)
    return float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def mcse_prop(k, r) -> float:
    """sqrt(p (1 - p) / R)."""
    if r <= 0:
        return float("nan")
    p = k / r
    return float(math.sqrt(p * (1 - p) / r))


def median_ci(x, level: float = 0.95) -> tuple[float, float, float]:
    """Median and its distribution-free interval from order statistics (the binomial(R, 1/2)
    quantiles), as metrics.median_ci in the engine."""
    from scipy import stats
    x = np.sort(finite(x))
    r = len(x)
    if r == 0:
        return (float("nan"),) * 3
    lo = int(stats.binom.ppf((1 - level) / 2, r, 0.5))
    hi = int(stats.binom.isf((1 - level) / 2, r, 0.5))
    lo = max(lo - 1, 0)
    hi = min(hi, r - 1)
    return float(np.median(x)), float(x[lo]), float(x[hi])


def boot_index(r: int, b: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, r, size=(b, r))


def mcse_median_boot(x, b: int = 1000, seed: int = 20261008) -> float:
    """Bootstrap MCSE of the median over replicates (6.1)."""
    x = finite(x)
    if len(x) < 2:
        return float("nan")
    vals = np.median(x[boot_index(len(x), b, seed)], axis=1)
    return float(np.std(vals, ddof=1))


def summary_row(x, boot_b: int = 1000, seed: int = 20261008) -> dict:
    """Mean (MCSE), median (95 % order-statistic interval, bootstrap MCSE), IQR, 10/90 % (6.1)."""
    xs = finite(x)
    r = len(xs)
    out = dict(R=r, n_nonfinite=int(np.size(x) - r))
    if r == 0:
        return out
    med, mlo, mhi = median_ci(xs)
    q10, q25, q75, q90 = np.quantile(xs, [0.10, 0.25, 0.75, 0.90])
    out.update(mean=float(xs.mean()), mean_mcse=mcse_mean(xs),
               mean_lo=float(xs.mean() - Z95 * mcse_mean(xs)) if r > 1 else float("nan"),
               mean_hi=float(xs.mean() + Z95 * mcse_mean(xs)) if r > 1 else float("nan"),
               median=med, median_lo=mlo, median_hi=mhi,
               median_mcse=mcse_median_boot(xs, boot_b, seed),
               q10=float(q10), q25=float(q25), q75=float(q75), q90=float(q90), iqr=float(q75 - q25))
    return out


# ----------------------------------------------------------------------------- Hodges-Lehmann (7.0)


@lru_cache(maxsize=None)
def signrank_cdf(n: int) -> np.ndarray:
    """Exact null CDF of the Wilcoxon signed-rank statistic V for n pairs (no ties)."""
    m = n * (n + 1) // 2
    dp = np.zeros(m + 1)
    dp[0] = 1.0
    for i in range(1, n + 1):
        dp[i:] = dp[i:] + dp[:-i].copy()
        dp /= 2.0
    return np.cumsum(dp)


def qsignrank(p: float, n: int) -> int:
    """Smallest v with P(V <= v) >= p (R's qsignrank)."""
    cdf = signrank_cdf(n)
    return int(np.searchsorted(cdf, p - 1e-12, side="left"))


def walsh(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    i, j = np.triu_indices(len(d))
    return (d[i] + d[j]) / 2.0


def hodges_lehmann(d, level: float = 0.95) -> tuple[float, float, float]:
    """HL estimate of paired differences (median of the Walsh averages) and its distribution-free
    interval from the signed-rank statistic, computed as R's wilcox.test(conf.int = TRUE, exact)."""
    d = finite(d)
    n = len(d)
    if n == 0:
        return (float("nan"),) * 3
    w = np.sort(walsh(d))
    est = float(np.median(w))
    if n < 2:
        return est, float("nan"), float("nan")
    m = len(w)
    qu = qsignrank((1 - level) / 2, n)
    if qu == 0:
        qu = 1
    ql = m - qu
    return est, float(w[qu - 1]), float(w[ql])


def hl_point(d: np.ndarray) -> float:
    w = walsh(d)
    k = len(w)
    if k % 2:
        return float(np.partition(w, k // 2)[k // 2])
    part = np.partition(w, [k // 2 - 1, k // 2])
    return float((part[k // 2 - 1] + part[k // 2]) / 2)


def mcse_hl_boot(d, b: int = 500, seed: int = 20261008) -> float:
    """Bootstrap MCSE of the HL estimate (7.0)."""
    d = finite(d)
    if len(d) < 2:
        return float("nan")
    idx = boot_index(len(d), b, seed)
    vals = np.array([hl_point(d[i]) for i in idx])
    return float(np.std(vals, ddof=1))


# ----------------------------------------------------------------------------- estimand transforms


def abs_log_b(b) -> np.ndarray:
    """|log b*| for Q1 (7.1): b* <= 0 or no finite b* gets 10, above every finite value."""
    b = np.asarray(b, dtype=float)
    out = np.full(b.shape, LOGB_FAIL)
    ok = np.isfinite(b) & (b > 0)
    out[ok] = np.abs(np.log(b[ok]))
    return out


def abs_log_b_capped(b) -> np.ndarray:
    """|log b*| for the meta-model (6.4): capped at log 10, and b* <= 0 or no finite b* = log 10."""
    return np.minimum(abs_log_b(b), LOGB_CAP_META)


# ----------------------------------------------------------------------------- labels (7.0-7.6)


def label_vs_margin(lo: float, hi: float, delta: float) -> str:
    """better / worse / equivalent / inconclusive for a difference where negative favours the TFM."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "inconclusive"
    if hi < -delta:
        return "better"
    if lo > delta:
        return "worse"
    if lo > -delta and hi < delta:
        return "equivalent"
    return "inconclusive"


def label_ici(lo: float, hi: float, pi: float) -> str:
    """Q1-Q5 true-ICI label: margin 0.01, times 0.3 at pi = 3 % (7.0)."""
    return label_vs_margin(lo, hi, ici_margin(pi))


def direction_label(lo: float, hi: float) -> str:
    """Q1 direction of the median b* (7.1): interval above 1.1, below 0.9, inside [0.9, 1.1]."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "unclear"
    if lo > SLOPE_TOL[1]:
        return "under-confident"
    if hi < SLOPE_TOL[0]:
        return "over-confident"
    if lo >= SLOPE_TOL[0] and hi <= SLOPE_TOL[1]:
        return "close to 1"
    return "unclear"


def detection_label(lo: float, hi: float, margin: float = DETECTION_MARGIN) -> str:
    """Q4b EDGE minus HL at a true-ICI point (7.4)."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "inconclusive"
    if lo > margin:
        return "detects more often"
    if hi < -margin:
        return "detects less often"
    if lo > -margin and hi < margin:
        return "equivalently"
    return "inconclusive"


def best_arm_label(lo: float, hi: float, margin: float = BEST_ARM_MARGIN) -> str:
    """Q5a TFM after R2 minus the best arm (7.5)."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "unclear"
    if hi <= margin:
        return "within 0.005 of the best arm"
    if lo > margin:
        return "more than 0.005 above it"
    return "unclear"


def ppv_label(lo: float, hi: float, n_flagged: int) -> str:
    """Q5b (7.5)."""
    if n_flagged < PPV_MIN_FLAGGED:
        return "not assessable"
    if lo >= PPV_TARGET:
        return "at least 0.8"
    if hi < PPV_TARGET:
        return "below 0.8"
    return "unclear"


def transfer_label(n_agree: int, n_checked: int = 48) -> str:
    """Q6 (7.6): 40 of 48 transfers, 30-39 in part, fewer than 30 does not. Mitra-v2 is checked
    only inside the region where it ran, so the cuts are applied as the same shares (40/48, 30/48)
    of the scenarios checked; with all 48 checked this is the rule as written."""
    if n_checked <= 0:
        return "not assessable"
    share = n_agree / n_checked
    if share >= TRANSFER_CUTS[0] / 48 - 1e-12:
        return "the grid transfers"
    if share >= TRANSFER_CUTS[1] / 48 - 1e-12:
        return "the grid transfers in part"
    return "the grid does not transfer"


def transfer_agrees(core_diff: float, core_mcse: float, meta_pred: float, meta_se: float) -> bool:
    return abs(core_diff - meta_pred) < Z95 * math.sqrt(meta_se ** 2 + core_mcse ** 2)


# ----------------------------------------------------------------------------- size (Q4a)


def size_mcse(r: int, alpha: float = ALPHA) -> float:
    return math.sqrt(alpha * (1 - alpha) / r)


def in_size_band(rate: float, r: int, alpha: float = ALPHA) -> bool:
    """Inside the 99 % Monte Carlo band 0.05 +/- 2.576 MCSE, with the MCSE at the nominal 0.05."""
    return abs(rate - alpha) <= Z99 * size_mcse(r, alpha)


def size_label(pooled: float, share_in_band: float) -> str:
    """holds its size / liberal / conservative / uneven (7.4)."""
    if not np.isfinite(pooled):
        return "not assessable"
    if pooled > SIZE_BAND[1]:
        return "liberal"
    if pooled < SIZE_BAND[0]:
        return "conservative"
    return "holds its size" if share_in_band >= SIZE_SCEN_SHARE else "uneven"


# ----------------------------------------------------------------------------- minimum n (4.5, Q2)

GT3000 = "> 3000"


def meets_class(cls_highest, cls: str) -> np.ndarray:
    """The nested classes of 4.5: a replicate meets a class when its highest class is that one or above."""
    order = {"none": 0, "mean": 1, "weak": 2, "moderate": 3, "strong": 4}
    want = order[cls]
    return np.array([order.get(str(c), -1) >= want for c in cls_highest])


def min_n(shares: dict, counts: dict | None = None, lower_limit: bool = False,
          share: float = MIN_N_SHARE):
    """Smallest n in {200, 500, 1000, 3000} with share >= 0.80 (or, with lower_limit, with
    share - 1.96 MCSE >= 0.80); "> 3000" when none qualifies. Missing n do not qualify."""
    for n in NS:
        p = shares.get(n)
        if p is None or not np.isfinite(p):
            continue
        if lower_limit:
            r = counts[n]
            p = p - Z95 * mcse_prop(p * r, r)
        if p >= share - 1e-12:
            return n
    return GT3000


def min_n_rank(v) -> int:
    return 10 ** 9 if v == GT3000 else int(v)


def compare_min_n(tfm, flic) -> str:
    a, b = min_n_rank(tfm), min_n_rank(flic)
    if a < b:
        return "needs a smaller n"
    if a > b:
        return "needs a larger n"
    return "the same n"


def q2_clarity(main: str, lower: str) -> str:
    return "clear" if main == lower else "sensitive to Monte Carlo error"


# ----------------------------------------------------------------------------- failures (8.2)


def fail_flag(n_failed: int, n_total: int) -> bool:
    """More than 5 % failed replicates: reported, but kept out of the Section 7 labels."""
    return n_total > 0 and n_failed / n_total > FAIL_FLAG_SHARE


def pair_usable(status_a, status_b) -> np.ndarray:
    """The pairing rule: both arms produced usable (ok or degenerate) predictions."""
    ok = ("ok", "degenerate")
    return np.isin(np.asarray(status_a, dtype=object), ok) & np.isin(np.asarray(status_b, dtype=object), ok)


# ----------------------------------------------------------------------------- reporting rule (6.3)


def proportion_bound(k: int, r: int) -> str | None:
    """A proportion at 0 or 1 has MCSE 0 and no interval: it is a floor or a ceiling."""
    if r <= 0:
        return None
    if k == 0:
        return "floor"
    if k == r:
        return "ceiling"
    return None


def pair_at_bound(lo_a: float, hi_a: float, lo_b: float, hi_b: float, margin: float) -> str | None:
    """Two rates compared with a margin are uninformative when both intervals lie within the margin
    of 1 (ceiling) or of 0 (floor): no difference larger than the margin is then possible."""
    if lo_a >= 1 - margin and lo_b >= 1 - margin:
        return "ceiling"
    if hi_a <= margin and hi_b <= margin:
        return "floor"
    return None


def ici_at_floor(hi_tfm_lo: float, flic_lo: float, oracle_hi: float) -> bool:
    """A true-ICI comparison is at the estimator's floor when both arms' lower 95 % limits sit at or
    below the oracle's upper limit (the oracle row is the loess floor, 4.1)."""
    return hi_tfm_lo <= oracle_hi and flic_lo <= oracle_hi


SAFE_WORDS = ("equivalent", "equivalently", "holds its size", "no difference", "within 0.005 of the best arm",
              "the same n")


def statement(label: str, estimate: float, lo: float, hi: float, bound: str | None = None,
              mcse: float | None = None) -> str:
    """The text of a reported result. It refuses to write an 'equivalent'-type statement without
    an interval (6.3), and writes a floor or ceiling as uninformative whatever the label."""
    if bound:
        return f"uninformative ({bound}); estimate {estimate:.4g} [{lo:.4g}, {hi:.4g}]"
    has_ci = np.isfinite(lo) and np.isfinite(hi)
    if any(w in label for w in SAFE_WORDS) and not has_ci:
        raise ValueError(f"'{label}' cannot be reported without its interval or MCSE (6.3)")
    tail = f" (MCSE {mcse:.2g})" if mcse is not None and np.isfinite(mcse) else ""
    return f"{label}: {estimate:.4g} [{lo:.4g}, {hi:.4g}]{tail}" if has_ci else f"{label}: {estimate:.4g}{tail}"
