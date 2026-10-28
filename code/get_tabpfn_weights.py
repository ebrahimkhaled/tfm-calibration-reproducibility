"""Download the TabPFN checkpoints this study used, and check them by SHA-256.

The TabPFN weights are NOT in any archive of this study. TabPFN-3.5 is licensed under the
TABPFN-3.5 License v1.0, which allows non-commercial research use only and does not allow us to
redistribute the weights. Each user gets them from Prior Labs under that licence:

  1. Read and accept the licence on the Hugging Face page of Prior-Labs/tabpfn_3_5 (and of
     Prior-Labs/TabPFN-v2-clf for the legacy arm). This script never accepts a licence for you.
  2. Log in on this machine:  hf auth login
  3. Run:  python code/get_tabpfn_weights.py [--dir <folder>] [--legacy]
     (default folder: TABPFN_MODEL_CACHE_DIR, or rebuild/models/tabpfn)

The files are checked against the SHA-256 values of pre-registration section 13; a mismatch is
reported and the file is not used by the run (run.py also checks TFMCAL_TABPFN_SHA256).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

LICENCE_NOTICE = (
    "TabPFN-3.5 (Prior Labs) is used under the TABPFN-3.5 License v1.0: non-commercial research use "
    "only; the weights may not be redistributed. See https://huggingface.co/Prior-Labs/tabpfn_3_5.")

FILES = {
    "V3_5": ("tabpfn-v3.5-20260909.safetensors",
             "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3"),
    "V2": ("tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
           "cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.environ.get("TABPFN_MODEL_CACHE_DIR",
                                                    str(Path(__file__).resolve().parents[1] / "models" / "tabpfn")))
    ap.add_argument("--legacy", action="store_true", help="also the Real-TabPFN-2 checkpoint of the legacy arm")
    a = ap.parse_args(argv)
    print(LICENCE_NOTICE)
    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import download_model
    folder = Path(a.dir)
    folder.mkdir(parents=True, exist_ok=True)
    bad = 0
    for ver in ["V3_5"] + (["V2"] if a.legacy else []):
        name, want = FILES[ver]
        path = folder / name
        if not path.exists():
            download_model(path, version=getattr(ModelVersion, ver), which="classifier")
        got = sha256(path)
        ok = got == want
        bad += not ok
        print(f"{name}: SHA-256 {got} {'matches' if ok else 'DIFFERS from ' + want}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
