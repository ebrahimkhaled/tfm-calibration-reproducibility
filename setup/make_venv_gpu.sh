#!/usr/bin/env bash
# Build rebuild/.venv-gpu: Python 3.11 + CUDA torch + the three TFM families.
#
# Why not --system-site-packages on the global Python 3.13?
#   The global torch is 2.6.0+cu124 and sits in the Microsoft Store *user*
#   site-packages, which a venv cannot see. Even if it could, autogluon 1.6.3
#   (Mitra) needs torch >= 2.10, and the cu124 index stops at torch 2.6.
#   So we install torch 2.13.0+cu126 (newest below autogluon's < 2.14 cap).
#
# Run:  bash rebuild/setup/make_venv_gpu.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/env.sh"

PY311="${PY311:-3.11.15}"   # a Python 3.11.15 interpreter path, or a version uv can find
VENV="$HERE/.venv-gpu"
PY="$VENV/Scripts/python.exe"

if [ ! -x "$PY" ]; then
  uv venv --python "$PY311" "$VENV"
fi

# 1) CUDA torch from the PyTorch index only
uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu126 \
  "torch==2.13.0+cu126"

# 2) everything else from PyPI; torch is pinned so the CUDA build stays
#    autogluon.tabular[mitra] caps torch < 2.14, pandas < 2.4, numpy < 2.6, pyarrow < 25, hf-hub < 2
uv pip install --python "$PY" \
  "torch==2.13.0" \
  "tabpfn==9.1.0" \
  "tabpfn-extensions==0.6.3" \
  "tabicl==2.2.0" \
  "autogluon.tabular[mitra,lightgbm]==1.6.3" \
  "lightgbm==4.7.0" \
  "scikit-learn==1.9.1" \
  "statsmodels==0.15.0" \
  "pandas<2.4" \
  "pyarrow<25" \
  "psutil" "nvidia-ml-py"

"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)"
uv pip freeze --python "$PY" > "$HERE/setup/requirements-gpu.lock.txt"
echo "venv ready: $VENV"
