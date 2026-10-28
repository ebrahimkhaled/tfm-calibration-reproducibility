"""Print the newest PyPI version (and Python / torch bounds) of the packages we need."""
import json
import sys
import urllib.request

PKGS = sys.argv[1:] or ["tabpfn", "tabpfn-extensions", "tabicl", "autogluon.tabular",
                        "lightgbm", "scikit-learn", "statsmodels", "pandas", "pyarrow",
                        "huggingface-hub", "numpy"]

for name in PKGS:
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=60) as r:
            info = json.load(r)["info"]
        reqs = [x for x in (info.get("requires_dist") or [])
                if x.split(";")[0].strip().lower().startswith(("torch", "tabpfn", "tabicl", "numpy", "scikit-learn", "pandas"))]
        print(f"{name}: {info['version']}  python {info.get('requires_python')}  | {'; '.join(reqs[:8])}")
    except Exception as e:
        print(f"{name}: ERROR {e}")
