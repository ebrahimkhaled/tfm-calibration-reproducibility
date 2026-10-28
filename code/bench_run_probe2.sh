#!/usr/bin/env bash
# GPU memory probe with a capped allocator (80 % of the card), then LightGBM
# with one thread per fit. Runs next to the CPU TFM timings.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/env.sh"
cd "$HERE"
PY=.venv-gpu/Scripts/python.exe

$PY code/bench_01_python_arms.py --memprobe --models tabpfn_v2 tabicl_v2 --probe-n 5000 --probe-p 20 --mem-fraction 0.8
$PY code/bench_01_python_arms.py --memprobe --models mitra_v2 --probe-n 2500 3000 --probe-p 20 --mem-fraction 0.8
$PY code/bench_01_python_arms.py --memprobe --models mitra_v2 --probe-n 3000 --probe-p 5 --mem-fraction 0.8
$PY code/bench_01_python_arms.py --devices cpu --models lgbm
echo "probe2 done"
