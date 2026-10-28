"""Random-design robustness tier, scenario family "R" (pre-registration section 3.9).

Each design d = 1..D gets one replicate, and every design draws its own setting:

    event rate pi            log-uniform on [0.02, 0.40]
    development n            log-uniform on [150, 4000], rounded to an integer
    predictors p             integer uniform on [4, 30]
    predictor correlation    uniform on [0, 0.6] (exchangeable, on the latent Gaussian scale)
    oracle AUC target        uniform on [0.65, 0.90]
    shape severities         w_nl, w_int, w_link, each uniform on [0, 1]
    discrete fraction        uniform on [0, 0.5]: round(fraction x p) predictors are binary or
                             categorical (each categorical with probability 0.25, 3-6 levels)

Covariates. A latent N(0, R) vector with exchangeable correlation rho. The first p - k columns are
continuous (the latent values); the last k are discrete: a binary is 1{z > q} with a prevalence drawn
on [0.2, 0.5], a categorical cuts z at equal-probability normal quantiles (codes 0..K-1).

Outcome. The linear term L uses coefficients drawn per design (sign random, size uniform on [0.2, 1];
each predictor after the second is pure noise with probability 0.25; categorical level effects
N(0, 0.5^2) with level 0 as reference). L is rescaled so that its SD equals the SD of the core design's
L (1.755), so the severity weights mean what gamma means in the core design. Then

    eta = alpha + kappa [L + w_nl N + w_int I]
    p*  = (1 - w_link) expit(eta) + w_link (1 - exp(-exp(eta)))

where N and I are the core design's own shape terms, taken from dgp.shape_score: N is S2's
non-linear term (gamma = 1, as T4b) and I is S3's interaction term. They are evaluated on a core-shaped
matrix whose roles are filled by the design's predictors: x1 <- first continuous, x3 <- second
continuous, x4 <- third continuous (first if there are only two), x7 <- first binary (1{x1' > 0} of the
last continuous predictor if the design has no binary). The link mixture uses dgp.inv_link.
kappa is solved so that the oracle AUC equals the design's target at its own event rate, and alpha so
that E[p*] = pi, both on 200,000 calibration draws of the design's own covariates.

Seeds (a stream of its own, never used by the core design or the pilot):
    SeedSequence([4242, 0, d])         the setting of design d
    SeedSequence([4242, 1, d])         its calibration draws
    SeedSequence([4242, 2, d, rep])    its data (rep = 1); the smoke stream adds 777

The evaluation set is sized as in the core design, max(1000, ceil(100/pi)), so it has at least 100
expected events. There are no recalibration or probe sets (one replicate per design, no ladder).

    python dgp_random.py build [--D 1000] [--workers 8]   solve every design, write random_designs.csv
    python dgp_random.py check                            re-solve and compare with the file
    python dgp_random.py show 7                           print design 7
"""

from __future__ import annotations

import json
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special

import dgp
from metrics import expected_auc

CODE_DIR = Path(__file__).resolve().parent
DESIGN_CSV = CODE_DIR / "random_designs.csv"

RANDOM_STREAM = 4242
KEY_SETTING, KEY_CALIB, KEY_DATA = 0, 1, 2
ID_OFFSET = 1000                # scenario_id = 1000 + d, apart from the core ids 1-58
N_CALIB = 200_000
FAMILY = "R"

# ranges of the setting (pre-registration 3.9)
PI_RANGE = (0.02, 0.40)
N_RANGE = (150, 4000)
P_RANGE = (4, 30)
RHO_RANGE = (0.0, 0.6)
AUC_RANGE = (0.65, 0.90)
DISC_RANGE = (0.0, 0.5)
P_CATEGORICAL = 0.25            # a discrete predictor is categorical with this probability
CAT_LEVELS = (3, 6)
BIN_PREV = (0.2, 0.5)
P_NOISE = 0.25                  # a predictor after the second carries no signal with this probability
# SD of the core design's L: x1..x6 N(0, 1) with corr(x1, x2) = 0.5, x7 ~ Bern(0.5), x8 ~ Bern(0.3)
_b = dgp.BETA
CORE_L_SD = float(np.sqrt(np.sum(_b[:6] ** 2) + 2 * 0.5 * _b[0] * _b[1]
                          + _b[6] ** 2 * 0.5 * 0.5 + _b[7] ** 2 * 0.3 * 0.7))

NAME_RE = re.compile(r"^R(\d{4})$")


def design_name(d: int) -> str:
    return f"R{d:04d}"


def is_random(key) -> bool:
    if isinstance(key, str):
        return bool(NAME_RE.match(key)) or (key.isdigit() and int(key) > ID_OFFSET)
    try:
        return int(key) > ID_OFFSET
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------------- the setting

def draw_setting(d: int) -> dict:
    """The setting of design d, from its own stream. The order of the draws is fixed."""
    rng = np.random.default_rng(np.random.SeedSequence([RANDOM_STREAM, KEY_SETTING, d]))
    lu = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))  # noqa: E731
    pi = lu(*PI_RANGE)
    n = int(round(lu(*N_RANGE)))
    p = int(rng.integers(P_RANGE[0], P_RANGE[1] + 1))
    rho = float(rng.uniform(*RHO_RANGE))
    auc = float(rng.uniform(*AUC_RANGE))
    w_nl, w_int, w_link = (float(v) for v in rng.uniform(0, 1, 3))
    frac_disc = float(rng.uniform(*DISC_RANGE))
    k = int(round(frac_disc * p))
    kinds = ["cont"] * (p - k) + ["cat" if rng.random() < P_CATEGORICAL else "bin" for _ in range(k)]
    cols = [f"x{j}" for j in range(1, p + 1)]
    beta, prev, levels, effects = {}, {}, {}, {}
    for j, (c, kind) in enumerate(zip(cols, kinds)):
        noise = j >= 2 and rng.random() < P_NOISE
        size = rng.uniform(0.2, 1.0) * (1 if rng.random() < 0.5 else -1)
        if kind == "cat":
            K = int(rng.integers(CAT_LEVELS[0], CAT_LEVELS[1] + 1))
            e = rng.normal(0, 0.5, K)
            e[0] = 0.0
            levels[c] = K
            effects[c] = [0.0] * K if noise else [float(v) for v in e]
        else:
            beta[c] = 0.0 if noise else float(size)
            if kind == "bin":
                prev[c] = float(rng.uniform(*BIN_PREV))
    return dict(design=d, pi=pi, n=n, p=p, rho=rho, auc_target=auc, w_nl=w_nl, w_int=w_int, w_link=w_link,
                frac_disc=frac_disc, n_disc=k, kinds=kinds, columns=cols, beta=beta, bin_prev=prev,
                cat_levels=levels, cat_effects=effects)


def roles(s: dict) -> dict:
    """Which of the design's columns play x1, x3, x4 and x7 of the core shape terms."""
    cont = [c for c, k in zip(s["columns"], s["kinds"]) if k == "cont"]
    binary = [c for c, k in zip(s["columns"], s["kinds"]) if k == "bin"]
    return dict(x1=cont[0], x3=cont[1], x4=cont[2] if len(cont) > 2 else cont[0],
                x7=binary[0] if binary else None, x7_from=cont[-1])


# ----------------------------------------------------------------------------- covariates and risk

def draw_covariates(s: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    p, rho = s["p"], s["rho"]
    z = math.sqrt(rho) * rng.standard_normal((n, 1)) + math.sqrt(1 - rho) * rng.standard_normal((n, p))
    X = z.copy()
    for j, (c, kind) in enumerate(zip(s["columns"], s["kinds"])):
        if kind == "bin":
            X[:, j] = (z[:, j] > special.ndtri(1 - s["bin_prev"][c])).astype(float)
        elif kind == "cat":
            K = s["cat_levels"][c]
            X[:, j] = np.searchsorted(special.ndtri(np.arange(1, K) / K), z[:, j]).astype(float)
    return X


def linear_term(X: np.ndarray, s: dict) -> np.ndarray:
    L = np.zeros(len(X))
    for j, c in enumerate(s["columns"]):
        if c in s["cat_effects"]:
            L += np.asarray(s["cat_effects"][c])[X[:, j].astype(int)]
        else:
            L += s["beta"][c] * X[:, j]
    return L


def shape_parts(X: np.ndarray, s: dict) -> tuple[np.ndarray, np.ndarray]:
    """S2's non-linear term and S3's interaction term, from dgp.shape_score on a core-shaped matrix."""
    r = roles(s)
    ix = {c: j for j, c in enumerate(s["columns"])}
    Xc = np.zeros((len(X), 8))
    Xc[:, 0] = X[:, ix[r["x1"]]]
    Xc[:, 2] = X[:, ix[r["x3"]]]
    Xc[:, 3] = X[:, ix[r["x4"]]]
    Xc[:, 6] = X[:, ix[r["x7"]]] if r["x7"] else (X[:, ix[r["x7_from"]]] > 0).astype(float)
    base = dgp.shape_score(Xc, "S1")
    nonlin = dgp.shape_score(Xc, "T4b") - base          # gamma = 1: the S2 term at full severity
    inter = dgp.shape_score(Xc, "S3") - base            # 0.5 x1 x7 + 0.5 x3 x4
    return nonlin, inter


def score_of(X: np.ndarray, s: dict) -> np.ndarray:
    """The bracket [L + w_nl N + w_int I], before kappa and alpha."""
    nonlin, inter = shape_parts(X, s)
    L = linear_term(X, s) * (CORE_L_SD / s["sd_L"])
    return L + s["w_nl"] * nonlin + s["w_int"] * inter


def inv_link(eta: np.ndarray, w_link: float) -> np.ndarray:
    return (1 - w_link) * dgp.inv_link(eta, "logit") + w_link * dgp.inv_link(eta, "cloglog")


def true_risk(X: np.ndarray, s: dict) -> np.ndarray:
    return inv_link(s["alpha"] + s["kappa"] * score_of(X, s), s["w_link"])


# ----------------------------------------------------------------------------- solving a design

def solve_design(d: int) -> dict:
    s = draw_setting(d)
    rng = np.random.default_rng(np.random.SeedSequence([RANDOM_STREAM, KEY_CALIB, d]))
    X = draw_covariates(s, N_CALIB, rng)
    L = linear_term(X, s)
    s["sd_L"] = float(L.std()) if L.std() > 0 else 1.0
    score = score_of(X, s)
    wl, pi = s["w_link"], s["pi"]

    def alpha_of(kappa):
        f = lambda a: inv_link(a + kappa * score, wl).mean() - pi  # noqa: E731
        with np.errstate(over="ignore"):
            for b in (40.0, 400.0):
                if f(-b) < 0 < f(b):
                    return optimize.brentq(f, -b, b, xtol=1e-12)
        raise ValueError(f"design {d}: no intercept gives pi = {pi} at kappa = {kappa}")

    def auc_gap(kappa):
        a = alpha_of(kappa)
        return expected_auc(score, inv_link(a + kappa * score, wl)) - s["auc_target"]

    lo, hi = 0.005, 8.0
    status = "ok"
    if auc_gap(hi) < 0:
        kappa, status = hi, "auc target not reached"
    elif auc_gap(lo) > 0:
        kappa, status = lo, "auc target below the lower bracket"
    else:
        kappa = optimize.brentq(auc_gap, lo, hi, xtol=1e-9)
    alpha = alpha_of(kappa)
    pstar = inv_link(alpha + kappa * score, wl)
    ll_true = float(np.mean(special.xlogy(pstar, pstar) + special.xlogy(1 - pstar, 1 - pstar)))
    n_params = sum(1 for k in s["kinds"] if k != "cat") + sum(K - 1 for K in s["cat_levels"].values())
    riley = dgp.riley_min_n(pi, ll_true, n_params)
    s.update(kappa=float(kappa), alpha=float(alpha), solve_status=status,
             realized_prevalence=float(pstar.mean()), oracle_auc=expected_auc(score, pstar),
             n_params=n_params, riley_n_min_py=riley["riley_n_min"], r2cs_oracle=riley["r2cs"],
             share_p_above_half=float(np.mean(pstar >= 0.5)))
    return s


def table_row(s: dict) -> dict:
    d = s["design"]
    cont = [c for c, k in zip(s["columns"], s["kinds"]) if k == "cont"]
    binary = [c for c, k in zip(s["columns"], s["kinds"]) if k == "bin"]
    cat = [c for c, k in zip(s["columns"], s["kinds"]) if k == "cat"]
    return dict(
        scenario_id=ID_OFFSET + d, scenario=design_name(d), design=d, family=FAMILY, kind="random",
        pi=s["pi"], n=s["n"], n_index=0, p=s["p"], rho=s["rho"], auc_target=s["auc_target"],
        w_nl=s["w_nl"], w_int=s["w_int"], w_link=s["w_link"], frac_disc=s["frac_disc"], n_disc=s["n_disc"],
        n_eval=dgp.n_eval_of(s["pi"]), n_recal=0, n_probe=0,
        kappa=s["kappa"], alpha=s["alpha"], sd_L=s["sd_L"], solve_status=s["solve_status"],
        realized_prevalence=s["realized_prevalence"], oracle_auc=s["oracle_auc"],
        n_params=s["n_params"], epv=s["n"] * min(s["pi"], 1 - s["pi"]) / s["n_params"],
        riley_n_min_py=s["riley_n_min_py"], r2cs_oracle=s["r2cs_oracle"],
        share_p_above_half=s["share_p_above_half"], n_times_p=s["n"] * s["p"],
        cont_cols=" ".join(cont), bin_cols=" ".join(binary), cat_cols=" ".join(cat),
        spec=json.dumps(dict(kinds=s["kinds"], beta=s["beta"], bin_prev=s["bin_prev"],
                             cat_levels=s["cat_levels"], cat_effects=s["cat_effects"]), sort_keys=True),
    )


def build_table(D: int, workers: int = 1) -> pd.DataFrame:
    ids = list(range(1, D + 1))
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            solved = list(ex.map(solve_design, ids, chunksize=8))
    else:
        solved = [solve_design(d) for d in ids]
    return pd.DataFrame([table_row(s) for s in solved])


# ----------------------------------------------------------------------------- lookup

_TABLE: pd.DataFrame | None = None


def load_table() -> pd.DataFrame:
    global _TABLE
    if _TABLE is None:
        if not DESIGN_CSV.exists():
            raise FileNotFoundError(f"{DESIGN_CSV} is missing: run `python dgp_random.py build` first")
        _TABLE = pd.read_csv(DESIGN_CSV, keep_default_na=False, float_precision="round_trip")
    return _TABLE


def get_design(key) -> dict:
    """A design as a scenario dict (the fields run.py, models.py and score.py use, plus the spec)."""
    t = load_table()
    if isinstance(key, str) and NAME_RE.match(key):
        row = t[t.scenario == key]
    else:
        row = t[t.scenario_id == int(key)]
    if len(row) != 1:
        raise KeyError(f"unknown random design {key!r}")
    d = row.iloc[0].to_dict()
    spec = json.loads(d.pop("spec"))
    for k in ("cont_cols", "bin_cols", "cat_cols"):
        d[k] = d[k].split() if d[k] else []
    d.update(spec)
    d["columns"] = [f"x{j}" for j in range(1, int(d["p"]) + 1)]
    d["cat_levels"] = {k: int(v) for k, v in spec["cat_levels"].items()}
    d["covariates"] = "random"
    return d


def select(token: str) -> list[str] | None:
    """Scenario names for a --scenarios token: 'random' (every design), 'R0001-R0005', 'R0007'.
    None when the token is not about random designs."""
    t = load_table() if DESIGN_CSV.exists() else None
    if token == "random":
        return list(t.scenario) if t is not None else []
    m = re.match(r"^R(\d{4})-R?(\d{4})$", token)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return [design_name(d) for d in range(a, b + 1) if t is not None and design_name(d) in set(t.scenario)]
    if NAME_RE.match(token):
        return [token]
    return None


# ----------------------------------------------------------------------------- device rules

# Measured on the GTX 1660 Ti with the run's 0.8 memory cap (pilot section 2 and
# results/random_memory_probe.csv, 2026-10-07). TabPFN-3.5 holds the context and the predicted rows
# together, so its size is (n + N_eval) x p: every measured cell up to 84,000 fitted (peak 4.4 GB), every
# cell from 99,000 up ran out of memory (1,500 x 20 with 5,000 predicted rows failed, 4,000 x 8 with
# 2,000 fitted). Mitra-v2 predicts in batches, and its memory grows with n^2 x p: every cell up to
# 8.0e7 fitted (2,000 x 20 at 4.1 GB; 3,000 x 8 at 3.7 GB), every cell from 9.4e7 up failed (2,500 x 15,
# 3,500 x 8, 4,000 x 6; T2 at 3,000 x 30).
TABPFN_GPU_MAX_CELLS = 80_000
MITRA_MAX_N2P = 8.0e7


def tabpfn_on_gpu(sc: dict) -> bool:
    """TabPFN-3.5 runs on the GPU when (n + N_eval) x p fits the card, else on a CPU worker."""
    return (int(sc["n"]) + int(sc["n_eval"])) * int(sc["p"]) <= TABPFN_GPU_MAX_CELLS


def mitra_fits(sc: dict) -> bool:
    """Mitra-v2 is recorded as "does not fit in 6 GB GPU memory" above this size (the T2 rule)."""
    return float(sc["n"]) ** 2 * int(sc["p"]) <= MITRA_MAX_N2P


# ----------------------------------------------------------------------------- data of a replicate

def entropy(sc: dict, rep: int, smoke: bool) -> list:
    if int(rep) < 1:
        raise ValueError(f"replicates are numbered from 1, got {rep}")
    e = [RANDOM_STREAM, KEY_DATA, int(sc["design"]), int(rep)]
    return e + [dgp.SMOKE_KEY] if smoke else e


def draw_set(sc: dict, n: int, rng: np.random.Generator) -> "dgp.DataSet":
    X = draw_covariates(sc, n, rng)
    p = true_risk(X, sc)
    y = (rng.random(n) < p).astype(np.int8)
    return dgp.DataSet(X, p, y)


# ----------------------------------------------------------------------------- command line

def _main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "check"
    opt = lambda k, v: type(v)(argv[argv.index(k) + 1]) if k in argv else v  # noqa: E731
    if cmd == "build":
        D, w = opt("--D", 1000), opt("--workers", 8)
        t = build_table(D, w)
        t.to_csv(DESIGN_CSV, index=False, float_format="%.17g")
        bad = t[t.solve_status != "ok"]
        print(t[["pi", "n", "p", "rho", "auc_target", "oracle_auc", "w_nl", "w_int", "w_link",
                 "frac_disc", "n_eval"]].describe().T.to_string())
        print(f"{len(bad)} design(s) not solved exactly: {list(bad.scenario)}")
        print(f"max |realized prevalence - pi| {np.max(np.abs(t.realized_prevalence - t.pi)):.2g}; "
              f"max |oracle AUC - target| {np.max(np.abs(t.oracle_auc - t.auc_target)):.2g}")
        print(f"wrote {DESIGN_CSV} ({len(t)} designs)")
        return 0
    if cmd == "check":
        old = pd.read_csv(DESIGN_CSV, keep_default_na=False, float_precision="round_trip")
        new = build_table(len(old), opt("--workers", 8))
        num = [c for c in new.columns if new[c].dtype.kind == "f"]
        diff = (new[num] - old[num]).abs().max().max()
        same = new.drop(columns=num).astype(str).equals(old.drop(columns=num).astype(str))
        print(f"max numeric difference {diff:.3g}; other columns identical: {same}")
        return 0 if diff < 1e-9 and same else 1
    if cmd == "show":
        print(json.dumps(get_design(design_name(int(argv[2]))), indent=1, default=str))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
