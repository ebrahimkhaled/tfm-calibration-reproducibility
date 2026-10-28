"""Load every TFM once on the GPU (this also fetches the weights into rebuild/models/)
and record what was actually loaded: file names, sizes and SHA-256.

Run (from rebuild/, after `source env.sh`):  .venv-gpu/Scripts/python.exe setup/smoke_models.py
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

for var in ("HF_HUB_CACHE", "TABPFN_MODEL_CACHE_DIR", "TABPFN_NO_BROWSER"):
    assert os.environ.get(var), f"{var} not set - source env.sh first"

import torch  # noqa: E402

from bench_01_python_arms import ARMS  # noqa: E402

rng = np.random.default_rng(1)
x = rng.normal(size=(200, 5)).astype(np.float32)
y = (x[:, 0] + rng.normal(size=200) > 1.2).astype(np.int64)
xe = rng.normal(size=(500, 5)).astype(np.float32)

report = {"torch": torch.__version__, "cuda": torch.version.cuda,
          "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
          "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None,
          "arms": {}}
for name in ["tabpfn_v2", "tabicl_v2", "mitra_v2"]:
    t0 = time.perf_counter()
    try:
        fit_s, pred_s, p, note = ARMS[name][1](x, y, xe, "cuda", 0)
        report["arms"][name] = {"status": "ok", "first_call_s": round(time.perf_counter() - t0, 2),
                                "mean_p": float(np.mean(p)), "note": note}
    except Exception as e:  # noqa: BLE001
        report["arms"][name] = {"status": f"{type(e).__name__}: {e}"[:500]}
    print(name, report["arms"][name], flush=True)

# TabPFN-3.5 must not be fetched here: the licence is the author's to accept.
# Just record whether its checkpoint is already in the project cache.
report["tabpfn_v35_in_cache"] = any(Path(os.environ["TABPFN_MODEL_CACHE_DIR"]).glob("*3.5*"))

files = []
for base in (Path(os.environ["TABPFN_MODEL_CACHE_DIR"]), Path(os.environ["HF_HUB_CACHE"])):
    for f in base.rglob("*"):
        if f.is_file() and f.suffix in (".ckpt", ".safetensors", ".json", ".cpkt") and f.stat().st_size > 1000:
            files.append({"path": str(f.relative_to(ROOT)).replace("\\", "/"), "bytes": f.stat().st_size,
                          "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
report["weight_files"] = files
out = ROOT / "results" / "environment_models.json"
out.write_text(json.dumps(report, indent=1))
print(json.dumps(files, indent=1))
