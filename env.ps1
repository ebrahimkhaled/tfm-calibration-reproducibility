# Dot-source before any Python or R work in the rebuild:   . .\rebuild\env.ps1
# Keeps model weights, pip/uv downloads and the R library inside the project.
# HF_HOME is left at its default so the Hugging Face login token stays in
# ~/.cache/huggingface and never lands in the project folder.

$REBUILD = (Resolve-Path $PSScriptRoot).Path -replace '\\', '/'
$env:REBUILD = $REBUILD

$env:HF_HUB_CACHE           = "$REBUILD/models/hf/hub"
$env:HF_XET_CACHE           = "$REBUILD/models/hf/xet"
$env:TABPFN_MODEL_CACHE_DIR = "$REBUILD/models/tabpfn"
$env:TORCH_HOME             = "$REBUILD/models/torch"
$env:UV_CACHE_DIR           = "$REBUILD/.cache/uv"
$env:PIP_CACHE_DIR          = "$REBUILD/.cache/pip"
$env:TABPFN_DISABLE_TELEMETRY    = "1"
$env:TABPFN_NO_BROWSER           = "1"   # never pop up the Prior Labs licence login from a script
$env:TABPFN_ALLOW_CPU_LARGE_DATASET = "1" # CPU workers at n > 1000 (plan section 4.10)
$env:HF_HUB_DISABLE_TELEMETRY    = "1"
$env:AUTOGLUON_DISABLE_TELEMETRY = "1"

$env:R_LIBS      = "$REBUILD/rlib"
$env:R_LIBS_USER = "$REBUILD/rlib"

$env:VENV_GPU = "$REBUILD/.venv-gpu"
function pygpu { & "$env:VENV_GPU/Scripts/python.exe" @args }
