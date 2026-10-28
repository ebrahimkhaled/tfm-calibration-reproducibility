"""Fetch the TabPFN-3.5 checkpoint into rebuild/models/tabpfn and record its hash.

Run this ONLY after the author has accepted the TabPFN-3.5 licence himself
(see ENV_REPORT.md, section "TabPFN-3.5"). tabpfn 9.1 checks the acceptance
against the Prior Labs account server before it downloads anything, using
either the TABPFN_TOKEN environment variable or the token it caches in
~/.cache/tabpfn/auth_token (outside the project, so the key is never archived).

    source rebuild/env.sh
    export TABPFN_TOKEN="<API key from https://ux.priorlabs.ai/account>"   # this shell only
    rebuild/.venv-gpu/Scripts/python.exe rebuild/setup/fetch_tabpfn35.py
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
cache = Path(os.environ["TABPFN_MODEL_CACHE_DIR"])

from tabpfn import TabPFNClassifier  # noqa: E402
from tabpfn.constants import ModelVersion  # noqa: E402

rng = np.random.default_rng(0)
x = rng.normal(size=(120, 4)).astype(np.float32)
y = (x[:, 0] > 0.8).astype(int)

out = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "versions": {}}
for v in (ModelVersion.V3_5, ModelVersion.V3_5_FAST):
    clf = TabPFNClassifier.create_default_for_version(v, device="cuda")
    clf.fit(x, y)
    clf.predict_proba(x[:10])
    out["versions"][v.value] = {"model_path": str(clf.model_path),
                                "n_estimators_": getattr(clf, "n_estimators_", None),
                                "softmax_temperature_": getattr(clf, "softmax_temperature_", None)}
    print(v.value, "loaded from", clf.model_path, flush=True)

out["files"] = [{"file": f.name, "bytes": f.stat().st_size,
                 "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
                for f in sorted(cache.glob("*3.5*"))]
(ROOT / "results" / "environment_tabpfn35.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
