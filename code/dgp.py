"""Data-generating mechanisms of the simulation (REBUILD_PLAN 4.2, pre-registration section 3).

Covariates (core): x1..x6 N(0, 1) with corr(x1, x2) = 0.5, x7 ~ Bern(0.5), x8 ~ Bern(0.3).
Base linear term L = 1.0 x1 + 0.8 x2 + 0.6 x3 + 0.4 x4 + 0.2 x5 + 0.0 x6 + 0.5 x7 + 0.3 x8.

    S1  logit p* = alpha + kappa L
    S2  logit p* = alpha + kappa [L + gamma ((x1^2 - 1)/sqrt 2 + (x3^3 - 3 x3)/sqrt 6)], gamma = 0.5
    S3  logit p* = alpha + kappa [L + 0.5 x1 x7 + 0.5 x3 x4]      (true subgroup: x7 == 1)
    S4  cloglog: p* = 1 - exp(-exp(alpha + kappa L))

Stress families (pi = 10 %, n in {500, 3000}):
    T1  x1..x6 Student t3 (Gaussian copula keeps corr 0.5 for x1, x2), unit variance; S1 shape
    T2  the 8 core covariates plus 22 N(0, 1) noise columns x9..x30; S1 shape
    T3  x8 replaced by a 6-level categorical (equal probabilities), level effects
        (0, 0.3, -0.3, 0.6, -0.6, 0.9) inside L; S3 shape
    T4a S2 with gamma = 0.25
    T4b S2 with gamma = 1.0

kappa is solved per family so that the oracle AUC is 0.80 at pi = 10 %; alpha is then solved per
(family, pi) so that E[p*] = pi. Both are solved on 10^6 draws from a fixed calibration stream and
written to scenarios.csv, which is the single source of the numbers used by the generator.

    python dgp.py build      (re)solve kappa and alpha and write scenarios.csv
    python dgp.py check      regenerate the table and compare it with the file on disk
    python dgp.py show 12 3  print the fingerprints of scenario 12, replicate 3
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special

from metrics import expected_auc

CODE_DIR = Path(__file__).resolve().parent
SCENARIO_CSV = CODE_DIR / "scenarios.csv"

# design constants (pre-registration section 3)
BETA = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.5, 0.3])
CAT_EFFECTS = np.array([0.0, 0.3, -0.3, 0.6, -0.6, 0.9])
N_GRID = [200, 500, 1000, 3000]
PI_GRID = [0.03, 0.10, 0.30]
N_RECAL = 500
N_PROBE = 500
TARGET_AUC = 0.80
N_CALIB_DRAWS = 1_000_000
CALIB_SEED = 20261006          # seed of the calibration draws (not of any replicate)
PROBE_KEY = 99999              # SeedSequence([scenario_id, 99999]) for the probe set
SMOKE_KEY = 777                # SeedSequence([scenario_id, n_index, rep, 777]) for smoke data

# family -> (covariates, shape, gamma, link, true subgroup)
FAMILIES = {
    "S1": dict(cov="normal", shape="linear", gamma=0.0, link="logit", subgroup=""),
    "S2": dict(cov="normal", shape="nonlin", gamma=0.5, link="logit", subgroup=""),
    "S3": dict(cov="normal", shape="interact", gamma=0.0, link="logit", subgroup="x7==1"),
    "S4": dict(cov="normal", shape="linear", gamma=0.0, link="cloglog", subgroup=""),
    "T1": dict(cov="t3", shape="linear", gamma=0.0, link="logit", subgroup=""),
    "T2": dict(cov="noise22", shape="linear", gamma=0.0, link="logit", subgroup=""),
    "T3": dict(cov="cat6", shape="interact", gamma=0.0, link="logit", subgroup="x7==1"),
    "T4a": dict(cov="normal", shape="nonlin", gamma=0.25, link="logit", subgroup=""),
    "T4b": dict(cov="normal", shape="nonlin", gamma=1.0, link="logit", subgroup=""),
}
CORE_FAMILIES = ["S1", "S2", "S3", "S4"]
STRESS_FAMILIES = ["T1", "T2", "T3", "T4a", "T4b"]
STRESS_N = [500, 3000]
STRESS_PI = 0.10


# ----------------------------------------------------------------------------- covariates

def column_types(cov: str) -> dict:
    """Names of the continuous, binary and categorical columns of a covariate set."""
    cont = [f"x{j}" for j in range(1, 7)]
    if cov == "noise22":
        cont = cont + [f"x{j}" for j in range(9, 31)]
    if cov == "cat6":
        return dict(cont=cont, bin=["x7"], cat=["x8"], cat_levels={"x8": 6})
    return dict(cont=cont, bin=["x7", "x8"], cat=[], cat_levels={})


def column_names(cov: str) -> list[str]:
    p = 30 if cov == "noise22" else 8
    return [f"x{j}" for j in range(1, p + 1)]


def draw_covariates(cov: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """n rows of the covariate set. The order of the RNG calls is fixed: do not reorder."""
    z = rng.standard_normal((n, 6))
    z[:, 1] = 0.5 * z[:, 0] + math.sqrt(0.75) * z[:, 1]
    if cov == "t3":
        # Gaussian copula: same dependence as the normal design, t3 margins with unit variance
        z = special.stdtrit(3, special.ndtr(z)) / math.sqrt(3.0)
    x7 = (rng.random(n) < 0.5).astype(float)
    if cov == "cat6":
        x8 = rng.integers(0, 6, size=n).astype(float)
    else:
        x8 = (rng.random(n) < 0.3).astype(float)
    X = np.column_stack([z, x7, x8])
    if cov == "noise22":
        X = np.column_stack([X, rng.standard_normal((n, 22))])
    return X


# ----------------------------------------------------------------------------- true risk

def shape_score(X: np.ndarray, fam: str) -> np.ndarray:
    """The bracket [L + ...] of the pre-registration, before kappa and alpha."""
    spec = FAMILIES[fam]
    if spec["cov"] == "cat6":
        L = X[:, :7] @ BETA[:7] + CAT_EFFECTS[X[:, 7].astype(int)]
    else:
        L = X[:, :8] @ BETA
    if spec["shape"] == "nonlin":
        x1, x3 = X[:, 0], X[:, 2]
        return L + spec["gamma"] * ((x1 ** 2 - 1) / math.sqrt(2) + (x3 ** 3 - 3 * x3) / math.sqrt(6))
    if spec["shape"] == "interact":
        return L + 0.5 * X[:, 0] * X[:, 6] + 0.5 * X[:, 2] * X[:, 3]
    return L


def inv_link(eta: np.ndarray, link: str) -> np.ndarray:
    if link == "cloglog":
        return -np.expm1(-np.exp(eta))
    return special.expit(eta)


def true_risk(X: np.ndarray, fam: str, kappa: float, alpha: float) -> np.ndarray:
    return inv_link(alpha + kappa * shape_score(X, fam), FAMILIES[fam]["link"])


# ----------------------------------------------------------------------------- calibration

def _solve_alpha(score: np.ndarray, kappa: float, link: str, pi: float) -> float:
    f = lambda a: inv_link(a + kappa * score, link).mean() - pi
    return optimize.brentq(f, -30.0, 30.0, xtol=1e-12)


def _calibration_draws(cov: str) -> np.ndarray:
    key = {"normal": 1, "noise22": 1, "t3": 2, "cat6": 3}[cov]
    rng = np.random.default_rng(np.random.SeedSequence([CALIB_SEED, key]))
    base = "normal" if cov == "noise22" else cov     # noise columns do not enter p*
    return draw_covariates(base, N_CALIB_DRAWS, rng)


def solve_family(fam: str) -> dict:
    """kappa for AUC 0.80 at pi = 10 %, then alpha for every pi, with realized checks."""
    spec = FAMILIES[fam]
    X = _calibration_draws(spec["cov"])
    if spec["cov"] == "noise22":
        X = np.column_stack([X, np.zeros((X.shape[0], 22))])
    score = shape_score(X, fam)
    link = spec["link"]

    def auc_gap(kappa):
        a = _solve_alpha(score, kappa, link, 0.10)
        return expected_auc(score, inv_link(a + kappa * score, link)) - TARGET_AUC

    kappa = optimize.brentq(auc_gap, 0.02, 5.0, xtol=1e-10)
    out = {"kappa": kappa, "sd_score": float(score.std()), "by_pi": {}}
    for pi in PI_GRID:
        a = _solve_alpha(score, kappa, link, pi)
        p = inv_link(a + kappa * score, link)
        ll_true = np.mean(special.xlogy(p, p) + special.xlogy(1 - p, 1 - p))
        out["by_pi"][pi] = dict(alpha=a, prevalence=float(p.mean()), auc=expected_auc(score, p),
                                ll_true=float(ll_true))
    return out


def riley_min_n(pi: float, ll_true: float, n_params: int, shrink: float = 0.9,
                delta: float = 0.05, margin: float = 0.05) -> dict:
    """Riley et al. (2020) criteria for a binary outcome, with the oracle's Cox-Snell R^2.

    pmsampsize in R derives R^2 from the c-statistic instead; this exact-likelihood value is
    kept beside it as a cross-check."""
    ll_null = pi * math.log(pi) + (1 - pi) * math.log(1 - pi)
    r2 = 1 - math.exp(-2 * (ll_true - ll_null))
    r2_max = 1 - math.exp(2 * ll_null)
    n1 = n_params / ((shrink - 1) * math.log(1 - r2 / shrink))
    s2 = r2 / (r2 + delta * r2_max)
    n2 = n_params / ((s2 - 1) * math.log(1 - r2 / s2))
    n3 = (1.96 / margin) ** 2 * pi * (1 - pi)
    return dict(r2cs=r2, r2cs_max=r2_max, riley_n1=math.ceil(n1), riley_n2=math.ceil(n2),
                riley_n3=math.ceil(n3), riley_n_min=math.ceil(max(n1, n2, n3)))


def n_eval_of(pi: float) -> int:
    return max(1000, math.ceil(round(100 / pi, 9)))


def build_table() -> pd.DataFrame:
    solved = {fam: solve_family(fam) for fam in FAMILIES}
    rows = []
    sid = 0

    def add(fam, pi, n, kind):
        nonlocal sid
        sid += 1
        spec = FAMILIES[fam]
        ct = column_types(spec["cov"])
        n_params = len(ct["cont"]) + len(ct["bin"]) + sum(v - 1 for v in ct["cat_levels"].values())
        s = solved[fam]
        b = s["by_pi"][pi]
        r = riley_min_n(pi, b["ll_true"], n_params)
        rows.append(dict(
            scenario_id=sid,
            scenario=f"{fam}_pi{round(pi * 100):02d}_n{n}",
            family=fam, kind=kind, covariates=spec["cov"], shape=spec["shape"],
            gamma=spec["gamma"], link=spec["link"], pi=pi, n=n, n_index=N_GRID.index(n),
            p=len(column_names(spec["cov"])), n_params=n_params,
            n_eval=n_eval_of(pi), n_recal=N_RECAL, n_probe=N_PROBE,
            kappa=s["kappa"], alpha=b["alpha"], realized_prevalence=b["prevalence"],
            oracle_auc=b["auc"], expected_events_dev=n * pi,
            epv=n * min(pi, 1 - pi) / n_params,
            r2cs_oracle=r["r2cs"], riley_n_min_py=r["riley_n_min"],
            riley_n1=r["riley_n1"], riley_n2=r["riley_n2"], riley_n3=r["riley_n3"],
            true_subgroup=spec["subgroup"],
            cont_cols=" ".join(ct["cont"]), bin_cols=" ".join(ct["bin"]), cat_cols=" ".join(ct["cat"]),
        ))

    for fam in CORE_FAMILIES:
        for pi in PI_GRID:
            for n in N_GRID:
                add(fam, pi, n, "core")
    for fam in STRESS_FAMILIES:
        for n in STRESS_N:
            add(fam, STRESS_PI, n, "stress")
    return pd.DataFrame(rows)


def add_pmsampsize(t: pd.DataFrame) -> pd.DataFrame:
    """Riley minimum n from R pmsampsize (c-statistic route), when R is available."""
    import shutil
    import subprocess
    import tempfile
    rs = shutil.which("Rscript")
    if rs is None:
        print("Rscript not found: riley_n_min_pmsampsize left empty")
        return t.assign(riley_n_min_pmsampsize=np.nan, r2cs_pmsampsize=np.nan)
    with tempfile.TemporaryDirectory() as d:
        src, out = Path(d) / "sc.csv", Path(d) / "riley.csv"
        t.to_csv(src, index=False, float_format="%.17g")
        subprocess.run([rs, str(CODE_DIR / "riley_pmsampsize.R"), str(src), str(out)], check=True)
        r = pd.read_csv(out)
    return t.merge(r, on="scenario_id", how="left")


# ----------------------------------------------------------------------------- scenarios

_TABLE: pd.DataFrame | None = None


def load_table() -> pd.DataFrame:
    global _TABLE
    if _TABLE is None:
        if not SCENARIO_CSV.exists():
            raise FileNotFoundError(f"{SCENARIO_CSV} is missing: run `python dgp.py build` first")
        # round_trip: the default fast parser can be off in the last bit, which would change p*
        _TABLE = pd.read_csv(SCENARIO_CSV, keep_default_na=False, float_precision="round_trip")
    return _TABLE


def get_scenario(key) -> dict:
    """A scenario row as a dict, by id (int) or name (str). The random designs of family R
    (ids above 1000, names R0001...) live in dgp_random.py and its own table."""
    import dgp_random
    if dgp_random.is_random(key):
        return dgp_random.get_design(key)
    t = load_table()
    if isinstance(key, (int, np.integer)) or (isinstance(key, str) and key.isdigit()):
        row = t[t.scenario_id == int(key)]
    else:
        row = t[t.scenario == key]
    if len(row) != 1:
        raise KeyError(f"unknown scenario {key!r}")
    d = row.iloc[0].to_dict()
    for k in ("cont_cols", "bin_cols", "cat_cols"):
        d[k] = d[k].split() if d[k] else []
    d["columns"] = column_names(d["covariates"])
    d["cat_levels"] = column_types(d["covariates"])["cat_levels"]
    return d


# ----------------------------------------------------------------------------- replicates

def fingerprint(X: np.ndarray, p_star: np.ndarray, y: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(p_star, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.int8).tobytes())
    return h.hexdigest()


@dataclass
class DataSet:
    X: np.ndarray
    p_star: np.ndarray
    y: np.ndarray

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.X, self.p_star, self.y)


@dataclass
class Replicate:
    scenario: dict
    rep: int
    smoke: bool
    entropy: list
    dev: DataSet
    eval: DataSet
    recal: DataSet
    fingerprints: dict = field(default_factory=dict)

    def child_seed(self, *key: int) -> int:
        """A 31-bit integer seed for a named task of this replicate (models, null outcomes, ...)."""
        ss = np.random.SeedSequence(self.entropy, spawn_key=tuple(int(k) for k in key))
        return int(ss.generate_state(1)[0] & 0x7FFFFFFF)

    def child_rng(self, *key: int) -> np.random.Generator:
        ss = np.random.SeedSequence(self.entropy, spawn_key=tuple(int(k) for k in key))
        return np.random.default_rng(ss)


def replicate_entropy(sc: dict, rep: int, smoke: bool = False) -> list:
    # numpy pads a short entropy list with zeros (SeedSequence([a, b]) == SeedSequence([a, b, 0])),
    # so a replicate 0 could share its stream with another key; replicates are numbered from 1
    if sc.get("family") == "R":                   # the random designs have a stream of their own
        import dgp_random
        return dgp_random.entropy(sc, rep, smoke)
    if int(rep) < 1:
        raise ValueError(f"replicates are numbered from 1, got {rep}")
    e = [int(sc["scenario_id"]), int(sc["n_index"]), int(rep)]
    return e + [SMOKE_KEY] if smoke else e


def _draw_set(sc: dict, n: int, rng: np.random.Generator) -> DataSet:
    if sc.get("family") == "R":
        import dgp_random
        return dgp_random.draw_set(sc, n, rng)
    X = draw_covariates(sc["covariates"], n, rng)
    p = true_risk(X, sc["family"], sc["kappa"], sc["alpha"])
    y = (rng.random(n) < p).astype(np.int8)
    return DataSet(X, p, y)


def make_replicate(key, rep: int, smoke: bool = False) -> Replicate:
    """Development, evaluation and recalibration sets of one replicate, each from its own child
    stream of SeedSequence([scenario_id, n_index, rep] (+ [777] for smoke data))."""
    sc = get_scenario(key)
    entropy = replicate_entropy(sc, rep, smoke)
    ss_dev, ss_eval, ss_recal = np.random.SeedSequence(entropy).spawn(3)
    dev = _draw_set(sc, int(sc["n"]), np.random.default_rng(ss_dev))
    ev = _draw_set(sc, int(sc["n_eval"]), np.random.default_rng(ss_eval))
    rc = _draw_set(sc, int(sc["n_recal"]), np.random.default_rng(ss_recal))
    r = Replicate(sc, rep, smoke, entropy, dev, ev, rc)
    r.fingerprints = {"dev": dev.fingerprint, "eval": ev.fingerprint, "recal": rc.fingerprint}
    r.fingerprints["all"] = hashlib.sha256(
        "".join(r.fingerprints[k] for k in ("dev", "eval", "recal")).encode()).hexdigest()
    return r


def make_probe(key) -> DataSet:
    """The fixed 500-row probe set of a scenario (same for every replicate)."""
    sc = get_scenario(key)
    rng = np.random.default_rng(np.random.SeedSequence([int(sc["scenario_id"]), PROBE_KEY]))
    if sc.get("family") == "R":                   # no probe set in the random designs (n_probe = 0)
        import dgp_random
        d = dgp_random.draw_set(sc, int(sc["n_probe"]), rng)
        return DataSet(d.X, d.p_star, np.full(len(d.p_star), -1, dtype=np.int8))
    X = draw_covariates(sc["covariates"], int(sc["n_probe"]), rng)
    p = true_risk(X, sc["family"], sc["kappa"], sc["alpha"])
    return DataSet(X, p, np.full(len(p), -1, dtype=np.int8))


# ----------------------------------------------------------------------------- command line

def _main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "check"
    if cmd == "build":
        t = build_table()
        t = add_pmsampsize(t)
        t.to_csv(SCENARIO_CSV, index=False, float_format="%.17g")
        print(t[["scenario_id", "scenario", "kappa", "alpha", "realized_prevalence", "oracle_auc",
                 "epv", "riley_n_min_py"]].to_string(index=False))
        print(f"wrote {SCENARIO_CSV}")
        return 0
    if cmd == "check":
        new = build_table()
        old = pd.read_csv(SCENARIO_CSV, keep_default_na=False, float_precision="round_trip")
        old = old[[c for c in new.columns]]           # pmsampsize columns come from R, not checked
        num = [c for c in new.columns if new[c].dtype.kind == "f"]
        diff = (new[num] - old[num]).abs().max().max()
        same_rest = new.drop(columns=num).astype(str).equals(old.drop(columns=num).astype(str))
        print(f"max numeric difference {diff:.3g}; other columns identical: {same_rest}")
        return 0 if diff < 1e-9 and same_rest else 1
    if cmd == "show":
        r = make_replicate(int(argv[2]), int(argv[3]))
        print(json.dumps({"scenario": r.scenario["scenario"], "rep": r.rep, "entropy": r.entropy,
                          "events_dev": int(r.dev.y.sum()), "fingerprints": r.fingerprints}, indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
