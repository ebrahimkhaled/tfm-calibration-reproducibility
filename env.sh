# Source this before any Python or R work in the rebuild:   source rebuild/env.sh
# Keeps model weights, pip/uv downloads and the R library inside the project.
# The Hugging Face login token is NOT moved: HF_HOME stays at its default
# (~/.cache/huggingface), so `hf auth login` never writes a secret into the project.

REBUILD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REBUILD_W="$(cygpath -m "$REBUILD" 2>/dev/null || echo "$REBUILD")"
export REBUILD REBUILD_W

export HF_HUB_CACHE="$REBUILD_W/models/hf/hub"          # Mitra-v2, TabICL, TabPFN HF repos
export HF_XET_CACHE="$REBUILD_W/models/hf/xet"          # chunk cache used by hf downloads
export TABPFN_MODEL_CACHE_DIR="$REBUILD_W/models/tabpfn" # tabpfn's own checkpoint folder
export TORCH_HOME="$REBUILD_W/models/torch"
export UV_CACHE_DIR="$REBUILD_W/.cache/uv"
export PIP_CACHE_DIR="$REBUILD_W/.cache/pip"
export TABPFN_DISABLE_TELEMETRY=1
export TABPFN_NO_BROWSER=1            # never pop up the Prior Labs licence login from a script
export TABPFN_ALLOW_CPU_LARGE_DATASET=1 # CPU workers at n > 1000 (plan section 4.10)
export HF_HUB_DISABLE_TELEMETRY=1
export AUTOGLUON_DISABLE_TELEMETRY=1

# project R library first, then base R; the user library is left out
export R_LIBS="$REBUILD_W/rlib"
export R_LIBS_USER="$REBUILD_W/rlib"

export VENV_GPU="$REBUILD/.venv-gpu"
alias pygpu='"$VENV_GPU/Scripts/python.exe"'
