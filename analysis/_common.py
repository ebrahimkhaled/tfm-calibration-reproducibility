"""Shared settings of the analysis pipeline: paths, process limits, arm lists and fixed constants.

Every script of analysis/ imports this module first. Importing it
  - caps BLAS/OpenMP at one thread (the main run must not be slowed), and
  - drops the process to below-normal priority (children such as Rscript inherit it on Windows).
Nothing here reads results; the constants below are copied from the pre-registration and are
the only place the analysis takes them from.
"""

from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"


def lower_priority() -> None:
    """Below-normal priority for this process (Windows priority class 0x4000; nice 10 elsewhere)."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.windll.kernel32
            # the pseudo-handle must be passed as a 64-bit HANDLE, not a 32-bit int
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            k32.SetPriorityClass.restype = wintypes.BOOL
            if not k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000):
                raise OSError("SetPriorityClass failed")
        else:
            os.nice(10)
    except Exception:
        pass


lower_priority()

from pathlib import Path  # noqa: E402

ANALYSIS = Path(__file__).resolve().parent
REBUILD = ANALYSIS.parent
CODE = REBUILD / "code"
PREDS = REBUILD / "results" / "preds"
GOF = REBUILD / "results" / "gof"
METRICS = REBUILD / "results" / "metrics"          # written by the run's own score.py, read only here
TABLES = ANALYSIS / "tables"
CACHE = TABLES / "cache"
FIGS = ANALYSIS / "figures"
FIG_OUT = FIGS / "out"

# the run's folders this pipeline must never write into
FORBIDDEN = [REBUILD / "results" / "preds", REBUILD / "results" / "gof", REBUILD / "logs",
             REBUILD / "results" / "gof_e50", REBUILD / "results" / "gof_e200"]


def assert_writable(path: Path) -> Path:
    """Refuse any output path outside analysis/ (the run's folders are read only)."""
    p = Path(path).resolve()
    if ANALYSIS.resolve() not in p.parents and p != ANALYSIS.resolve():
        raise PermissionError(f"refusing to write outside analysis/: {p}")
    return p


def python_cpu() -> str:
    return str(REBUILD.parent / ".venv" / "Scripts" / "python.exe")


def rscript() -> str:
    for c in (r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe", "Rscript"):
        if c == "Rscript" or Path(c).exists():
            return c
    return "Rscript"


# ----------------------------------------------------------------------------- arms

ORACLE = "oracle"
CLASSICAL = ["oracle", "lr", "flic", "ridge", "gam", "lgbm", "lgbm_raw"]
TFMS = ["tabpfn", "tabicl", "mitra"]                     # the main comparison set (7.0)
SIDE = ["tabpfn_t1", "tabicl_t1", "tabpfn_legacy"]       # reported beside it
ALL_ARMS = CLASSICAL + TFMS + SIDE
COMPARATOR = "flic"
LADDER_ARMS = ["flic"] + TFMS
LADDER_LEVELS = ["R0", "R1_m100", "R2_m100", "R3_m100", "R1_m500", "R2_m500", "R3_m500"]
TEMPERATURE_PAIRS = {"tabpfn_t1": "tabpfn", "tabicl_t1": "tabicl"}
SHIPPED_TEMPERATURE = {"tabpfn": 1.0, "tabicl": 0.9, "mitra": None, "tabpfn_t1": 0.9,
                       "tabicl_t1": 1.0, "tabpfn_legacy": 0.9}
USABLE = ("ok", "degenerate")                            # 8.2: degenerate fits are kept

ARM_LABEL = {"oracle": "Oracle", "lr": "LR-MLE", "flic": "FLIC", "ridge": "Ridge-CV", "gam": "GAM",
             "lgbm": "LightGBM + Platt", "lgbm_raw": "LightGBM (raw)", "tabpfn": "TabPFN-3.5",
             "tabicl": "TabICLv2", "mitra": "Mitra-v2", "tabpfn_t1": "TabPFN-3.5 (T = 0.9)",
             "tabicl_t1": "TabICLv2 (T = 1.0)", "tabpfn_legacy": "Real-TabPFN-2"}

# ----------------------------------------------------------------------------- design

SHAPES = ["S1", "S2", "S3", "S4"]
PIS = [0.03, 0.10, 0.30]
NS = [200, 500, 1000, 3000]
STRESS = ["T1", "T2", "T3", "T4a", "T4b"]
R_CLASSICAL = 500
R_TFM_SMALL, R_TFM_3000 = 200, 150                       # schedule P4
R_SIDE = 100


def r_tfm(n: int) -> int:
    return R_TFM_3000 if n == 3000 else R_TFM_SMALL


def parse_scenario(name: str) -> dict:
    if "_" not in name:                                  # random-design tier (R0001 ...): no fixed pi, n
        return dict(scenario=name, family="random", pi=float("nan"), n=float("nan"), core=False)
    fam, pi, n = name.split("_")
    return dict(scenario=name, family=fam, pi=int(pi[2:]) / 100, n=int(n[1:]),
                core=fam in SHAPES)


def core_scenarios() -> list[str]:
    return [f"{s}_pi{int(round(p * 100)):02d}_n{n}" for s in SHAPES for p in PIS for n in NS]


def stress_scenarios() -> list[str]:
    return [f"{t}_pi10_n{n}" for t in STRESS for n in (500, 3000)]


def all_scenarios() -> list[str]:
    return core_scenarios() + stress_scenarios()


# ----------------------------------------------------------------------------- fixed rules

ALPHA = 0.05
MARGIN_ICI = 0.01                    # 7.0; times 0.3 at pi = 3 % (4.5)
MARGIN_LOGB = __import__("math").log(1.1)   # log 1.1
MARGIN_ABS_A = 0.05
LOGB_FAIL = 10.0                     # 7.1: b* <= 0 or no finite b* -> |log b*| = 10 (above every finite value)
LOGB_CAP_META = __import__("math").log(10.0)   # 6.4: log 10 cap for the meta-model
SLOPE_TOL = (0.9, 1.1)               # 7.1 direction labels
MIN_N_SHARE = 0.80                   # 4.5
FAIL_FLAG_SHARE = 0.05               # 8.2
SIZE_BAND = (0.04, 0.06)             # 7.4
SIZE_SCEN_SHARE = 0.95
DETECTION_POINTS = [0.01, 0.02, 0.03, 0.05]
DETECTION_MARGIN = 0.02
BEST_ARM_MARGIN = 0.005              # 7.5
PPV_TARGET = 0.8
PPV_MIN_FLAGGED = 50
TRANSFER_CUTS = (40, 30)             # 7.6

BATTERY = ["EDGE (G=10)", "EDGE (G=auto)", "Cox recalibration", "Calibration-in-the-large",
           "Spiegelhalter-z", "Osius-Rojek (external)", "GiViTI", "Hosmer-Lemeshow (external)",
           "Stukel (offset)", "le Cessie (external)"]
EXTRA_TESTS = ["DeepGOF-1 external (combined)", "DeepGOF-1 external (axes reading)",
               "Localize (any part named)"]
TEST_LABEL = {"EDGE (G=10)": "EDGE (G = 10)", "EDGE (G=auto)": "EDGE (G = auto)",
              "Cox recalibration": "Cox", "Calibration-in-the-large": "CITL",
              "Spiegelhalter-z": "Spiegelhalter", "Osius-Rojek (external)": "Osius-Rojek",
              "GiViTI": "GiViTI", "Hosmer-Lemeshow (external)": "Hosmer-Lemeshow",
              "Stukel (offset)": "Stukel", "le Cessie (external)": "le Cessie",
              "DeepGOF-1 external (combined)": "DeepGOF-1", "DeepGOF-1 external (axes reading)": "DeepGOF-1 (axes)",
              "Localize (any part named)": "Localization (FWER)"}


def prob_scale(pi: float) -> float:
    """Probability-scale thresholds and the ICI margin are multiplied by 0.3 at pi = 3 % (4.5, 7.0)."""
    return pi / 0.10 if abs(pi - 0.03) < 1e-9 else 1.0


def ici_margin(pi: float) -> float:
    return MARGIN_ICI * prob_scale(pi)


def progress(msg: str) -> None:
    print(msg, flush=True)
