"""Write results/environment.json at the start of the main run (pre-registration 13) and check it
against the frozen software table: package versions of this venv, R and its packages, ebrahim.gof's
version, and the SHA-256 of every model checkpoint. Stops (exit 1) on any mismatch.

    .venv-gpu/Scripts/python.exe code/write_environment.py [--out results/environment.json]
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

CODE = Path(__file__).resolve().parent
REB = CODE.parent
sys.path.insert(0, str(CODE))
import models  # noqa: E402

# the values frozen in pre-registration section 13
FROZEN_PY = {"python": "3.11.15", "numpy": "2.4.6", "pandas": "2.3.3", "pyarrow": "24.0.0", "scipy": "1.17.1",
             "lightgbm": "4.7.0", "scikit-learn": "1.9.1", "torch": "2.13.0+cu126", "tabpfn": "9.1.0",
             "tabicl": "2.2.0", "autogluon.tabular": "1.6.3"}
FROZEN_R = {"R": "4.4.1", "ebrahim.gof": "2.10.0", "givitiR": "1.3", "callr": "3.7.6", "logistf": "1.26.1",
            "glmnet": "4.1.10", "mgcv": "1.9.4", "pmsampsize": "1.1.3", "arrow": "23.0.1.2",
            "data.table": "1.18.2.1", "digest": "0.6.39", "jsonlite": "2.0.0"}
FROZEN_SHA = {
    "tabpfn/tabpfn-v3.5-20260909.safetensors": "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
    "tabpfn/tabpfn-v3.5-fast-20260909.safetensors": "14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550",
    "tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt": "cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49",
}
FROZEN_HF = {  # (repo, file) -> SHA-256, read from the project's Hugging Face cache
    ("jingang/TabICL", "tabicl-classifier-v2-20260212.ckpt"): "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0",
    ("autogluon/mitra-classifier-2", "model.safetensors"): "5ffab0e2cf52f61c5b7c7eb1e8542996736d1023a0212190cc09abc2119a1e09",
}


def r_versions() -> dict:
    expr = ('.libPaths(c(Sys.getenv("TFM_RLIB"), .Library)); v <- c(R = paste(R.version$major, R.version$minor, sep = "."));'
            'for (p in c("ebrahim.gof","givitiR","callr","logistf","glmnet","mgcv","pmsampsize","arrow",'
            '"data.table","digest","jsonlite")) v[p] <- as.character(packageVersion(p));'
            'cat(jsonlite::toJSON(as.list(v), auto_unbox = TRUE))')
    out = subprocess.run([models._rscript(), "-e", expr], capture_output=True, text=True,
                         env={**__import__("os").environ, "TFM_RLIB": str(REB / "rlib")}, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> int:
    models.configure_environment()
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else REB / "results" / "environment.json"
    env = dict(time=datetime.now().isoformat(timespec="seconds"), **models.environment_info())
    env["r"] = r_versions()
    env["checkpoints"] = {k: models.file_sha256(models.MODELS_DIR / k) for k in FROZEN_SHA}
    env["hf_checkpoints"] = {f"{r}/{f}": models.file_sha256(models._hf_cached(r, f).resolve()) for r, f in FROZEN_HF}
    env["tabpfn_main_arm"] = dict(zip(("model_version", "path"), map(str, models.resolve_tabpfn("tabpfn"))))
    bad = []
    for k, v in FROZEN_PY.items():
        got = env.get(k)
        if got != v:
            bad.append(f"{k}: {got} (frozen {v})")
    for k, v in FROZEN_R.items():
        if env["r"].get(k) != v:
            bad.append(f"R {k}: {env['r'].get(k)} (frozen {v})")
    for k, v in FROZEN_SHA.items():
        if env["checkpoints"][k] != v:
            bad.append(f"{k}: SHA-256 differs")
    for (r, f), v in FROZEN_HF.items():
        if env["hf_checkpoints"][f"{r}/{f}"] != v:
            bad.append(f"{r}/{f}: SHA-256 differs")
    if not env["tabpfn_main_arm"]["path"].endswith("tabpfn-v3.5-20260909.safetensors"):
        bad.append(f"tabpfn arm resolves to {env['tabpfn_main_arm']} (set TFMCAL_TABPFN_VERSION=V3_5)")
    env["mismatches"] = bad
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env, indent=1, default=str))
    print(f"wrote {out}")
    for b in bad:
        print("MISMATCH", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
